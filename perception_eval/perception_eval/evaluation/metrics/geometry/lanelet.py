# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from tier4/autoware-ml (autoware_ml/metrics/geometry/lanelet.py) at fcf86419.

"""Lanelet2 map parsing into region polygons, for the region-filter evaluation axis.

A T4 scene ships ``map/lanelet2_map.osm`` in the same local map frame the ego poses use. This
module parses it into shapely polygons grouped by lanelet2 token, either a lanelet's ``subtype``
(road, walkway, crosswalk, road_shoulder) or an area way's ``type`` (drivable_area,
crosswalk_polygon, intersection_area). A :class:`LaneletMap` then answers point-in-region and
footprint-overlap membership plus the per-lanelet speed limit. The map for a scene is parsed once
and cached process-wide by its resolved absolute path.

lanelet2 itself is not a dependency, so the OSM is parsed with the standard-library XML parser and
polygons are built with shapely through the version-neutral compatibility layer.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.geometry.shapely_compat import Prepared
from perception_eval.evaluation.metrics.geometry.shapely_compat import SpatialIndex
from perception_eval.evaluation.metrics.geometry.shapely_compat import union_all
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

# Way ``type`` values that describe a closed area usable as a region.
AREA_WAY_TYPES = frozenset(
    {
        "drivable_area",
        "crosswalk_polygon",
        "intersection_area",
        "hatched_road_markings",
        "no_obstacle_segmentation_area",
    }
)

# Region tokens a filter may request: the lanelet2 lanelet subtypes plus the area way types above.
# Requesting anything else is a configuration typo and fails loud at filter construction. A known
# token that has no polygons in a particular scene is a legitimate empty region.
KNOWN_REGION_TOKENS = (
    frozenset(
        {
            "road",
            "highway",
            "road_shoulder",
            "bicycle_lane",
            "bus_lane",
            "walkway",
            "crosswalk",
            "pedestrian_lane",
            "play_street",
            "emergency_lane",
        }
    )
    | AREA_WAY_TYPES
)

DEFAULT_MAP_FILENAME = "lanelet2_map.osm"


def _tags(element: ET.Element) -> Dict[str, str]:
    return {tag.get("k"): tag.get("v") for tag in element.findall("tag")}


def _way_coords(way: ET.Element, nodes: Dict[str, Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [nodes[nd.get("ref")] for nd in way.findall("nd") if nd.get("ref") in nodes]


def _parse_osm(osm_path: str) -> Tuple[ET.Element, Dict[str, Tuple[float, float]], Dict[str, ET.Element]]:
    """One XML pass shared by the region and speed loaders: root, nodes, ways."""
    root = ET.parse(osm_path).getroot()
    nodes: Dict[str, Tuple[float, float]] = {}
    for node in root.findall("node"):
        tags = _tags(node)
        if "local_x" in tags and "local_y" in tags:
            nodes[node.get("id")] = (float(tags["local_x"]), float(tags["local_y"]))
    ways: Dict[str, ET.Element] = {way.get("id"): way for way in root.findall("way")}
    return root, nodes, ways


def _lanelet_ring_polygon(
    relation: ET.Element,
    nodes: Dict[str, Tuple[float, float]],
    ways: Dict[str, ET.Element],
) -> Optional[Polygon]:
    """A lanelet relation's polygon: left bound + reversed right bound, repaired."""
    bounds = {
        member.get("role"): ways.get(member.get("ref"))
        for member in relation.findall("member")
        if member.get("role") in ("left", "right")
    }
    left, right = bounds.get("left"), bounds.get("right")
    if left is None or right is None:
        return None
    return _safe_polygon(_way_coords(left, nodes) + list(reversed(_way_coords(right, nodes))))


def _safe_polygon(ring: Sequence[Tuple[float, float]]) -> Optional[Polygon]:
    if len(ring) < 3:
        return None
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # repair self-touching rings
    return polygon if (not polygon.is_empty and polygon.area > 0.0) else None


def _region_polygons(
    root: ET.Element,
    nodes: Dict[str, Tuple[float, float]],
    ways: Dict[str, ET.Element],
) -> Dict[str, List[Polygon]]:
    regions: Dict[str, List[Polygon]] = defaultdict(list)
    for relation in root.findall("relation"):
        tags = _tags(relation)
        if tags.get("type") != "lanelet":
            continue
        polygon = _lanelet_ring_polygon(relation, nodes, ways)
        if polygon is not None:
            regions[tags.get("subtype", "unknown")].append(polygon)

    for way in ways.values():
        tags = _tags(way)
        if tags.get("type") not in AREA_WAY_TYPES:
            continue
        polygon = _safe_polygon(_way_coords(way, nodes))
        if polygon is not None:
            regions[tags["type"]].append(polygon)

    return dict(regions)


def _lanelet_speeds(
    root: ET.Element,
    nodes: Dict[str, Tuple[float, float]],
    ways: Dict[str, ET.Element],
) -> List[Tuple[Polygon, float]]:
    speeds: List[Tuple[Polygon, float]] = []
    for relation in root.findall("relation"):
        tags = _tags(relation)
        if tags.get("type") != "lanelet" or "speed_limit" not in tags:
            continue
        try:
            speed_mps = float(tags["speed_limit"]) / 3.6  # km/h -> m/s
        except ValueError:
            continue
        if speed_mps <= 0.0:
            continue
        polygon = _lanelet_ring_polygon(relation, nodes, ways)
        if polygon is not None:
            speeds.append((polygon, speed_mps))
    return speeds


def load_region_polygons(osm_path: str) -> Dict[str, List[Polygon]]:
    """Parse a lanelet2 OSM into ``{token: [polygon, ...]}``.

    Lanelets are keyed by their ``subtype`` (polygon = left bound + reversed right bound), area ways
    in :data:`AREA_WAY_TYPES` by their ``type``.
    """
    return _region_polygons(*_parse_osm(osm_path))


def load_lanelet_speeds(osm_path: str) -> List[Tuple[Polygon, float]]:
    """Parse lanelet relations that carry a ``speed_limit`` into ``(polygon, m/s)``.

    Lanelet ``speed_limit`` tags are in km/h. Lanelets without the tag are skipped (the caller falls
    back to a configured default).
    """
    return _lanelet_speeds(*_parse_osm(osm_path))


@lru_cache(maxsize=None)
def _load_lanelet_map(resolved_path: str) -> "LaneletMap":
    """One parsed map per resolved absolute OSM path, shared process-wide."""
    parsed = _parse_osm(resolved_path)
    return LaneletMap(_region_polygons(*parsed), _lanelet_speeds(*parsed))


def clear_lanelet_map_cache() -> None:
    """Drop every cached map (tests that rewrite a file under the same path need this)."""
    _load_lanelet_map.cache_clear()


class LaneletMap:
    """Region polygons for one scene, with region membership and per-lane speed."""

    def __init__(
        self,
        region_polygons: Dict[str, List[Polygon]],
        speed_lanelets: Optional[Sequence[Tuple[Polygon, float]]] = None,
    ) -> None:
        """Index the parsed polygons.

        Args:
            region_polygons (dict[str, list[Polygon]]): Region token to its polygons, map frame.
            speed_lanelets (list[tuple[Polygon, float]] | None): ``(polygon, speed limit in m/s)``
                pairs for lanelets that carry one.
        """
        self._region_polygons: Dict[str, List[Polygon]] = {
            token: list(polys) for token, polys in region_polygons.items()
        }
        self._speed_lanelets = list(speed_lanelets or [])
        self._speed_polys = [polygon for polygon, _ in self._speed_lanelets]
        self._speed_values = np.array([speed for _, speed in self._speed_lanelets], dtype=np.float64)
        self._speed_index = SpatialIndex(self._speed_polys) if self._speed_polys else None
        self._union_cache: Dict[Tuple[str, ...], Prepared] = {}
        self._full_surface_cache: Optional[Prepared] = None
        self._eroded_cache: Dict[float, Prepared] = {}
        self._expanded_cache: Dict[Tuple[Tuple[str, ...], float], Prepared] = {}

    @property
    def tokens(self) -> Tuple[str, ...]:
        """Region tokens present in this map."""
        return tuple(self._region_polygons.keys())

    def speed_at(self, x: float, y: float, default: float) -> float:
        """Speed limit (m/s) of the lanelet containing ``(x, y)``, else ``default``.

        On overlapping lanelets the lowest limit wins (conservative), so an object straddling a slow
        lane is not over-propagated.
        """
        if self._speed_index is None:
            return default
        point = Point(x, y)
        candidates = self._speed_index.query(point)
        speeds = [float(self._speed_values[i]) for i in candidates if self._speed_polys[i].contains(point)]
        return min(speeds) if speeds else default

    @classmethod
    def from_osm(cls, osm_path: str) -> "LaneletMap":
        """Parsed map for an OSM path, shared through the process-wide cache."""
        return _load_lanelet_map(str(Path(osm_path).resolve()))

    def _prepared_union(self, tokens: Sequence[str]) -> Prepared:
        key = tuple(tokens)
        cached = self._union_cache.get(key)
        if cached is None:
            polygons = [poly for token in key for poly in self._region_polygons.get(token, [])]
            # A known region this scene's map does not contain is a normal condition: the region is
            # empty and membership is false everywhere. Token typos are rejected at filter construction.
            cached = Prepared(union_all(polygons) if polygons else Polygon())
            self._union_cache[key] = cached
        return cached

    def region_union(self, tokens: Sequence[str]) -> BaseGeometry:
        """Shapely (multi)polygon union of the given region tokens (map frame).

        Used by the reachability collision engine to clip wheeled reachable sets to the drivable area.
        """
        return self._prepared_union(tokens).geometry

    def _full_surface(self) -> Prepared:
        """Union of every mapped region, the whole mapped surface."""
        if self._full_surface_cache is None:
            polygons = [poly for polys in self._region_polygons.values() for poly in polys]
            self._full_surface_cache = Prepared(union_all(polygons) if polygons else Polygon())
        return self._full_surface_cache

    def _eroded_full_surface(self, margin: float) -> Prepared:
        """The full mapped surface eroded inward by ``margin``.

        The erosion applies to the outer border of the whole mapped surface only: internal borders
        between adjacent regions stay intact, so no dead gap appears between regions.
        """
        cached = self._eroded_cache.get(margin)
        if cached is None:
            cached = Prepared(self._full_surface().geometry.buffer(-margin))
            self._eroded_cache[margin] = cached
        return cached

    def _expanded_region(self, tokens: Sequence[str], margin: float) -> Prepared:
        """The selected region dilated outward by ``margin``."""
        key = (tuple(tokens), margin)
        cached = self._expanded_cache.get(key)
        if cached is None:
            cached = Prepared(self._prepared_union(tokens).geometry.buffer(margin))
            self._expanded_cache[key] = cached
        return cached

    def contains(
        self,
        tokens: Sequence[str],
        xy: NDArray,
        margin: float = 0.0,
        expand: bool = False,
    ) -> NDArray[np.bool_]:
        """Boolean mask of map-frame ``xy`` points inside the union of ``tokens``.

        ``margin`` adjusts the region border by that many meters and ``expand`` picks the direction:

        * ``False`` (default) cuts inward: points within ``margin`` of the outer border of the full
          mapped surface stop counting. Internal borders between adjacent regions stay intact.
        * ``True`` grows outward: the selected region additionally claims off-map points within
          ``margin`` of it. Points belonging to another mapped region are never claimed.
        """
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        mask = self._prepared_union(tokens).contains_points(xy)
        if margin <= 0.0:
            return mask
        if expand:
            in_expanded = self._expanded_region(tokens, float(margin)).contains_points(xy)
            on_map = self._full_surface().contains_points(xy)
            return mask | (in_expanded & ~on_map)
        return mask & self._eroded_full_surface(float(margin)).contains_points(xy)

    def intersects(
        self,
        tokens: Sequence[str],
        footprints: Sequence[BaseGeometry],
        margin: float = 0.0,
        expand: bool = False,
    ) -> NDArray[np.bool_]:
        """Boolean mask of BEV footprint polygons that overlap the region.

        The box counterpart of :meth:`contains`: a detection box belongs to the region when any part
        of its footprint lies inside it, so an object overhanging the region from an off-region
        center still counts. ``margin`` / ``expand`` adjust the region border exactly as in
        :meth:`contains`.
        """
        footprints = list(footprints)
        if len(footprints) == 0:
            return np.zeros((0,), dtype=bool)
        hit = self._prepared_union(tokens).intersects_many(footprints)
        if margin <= 0.0:
            return hit
        if expand:
            in_expanded = self._expanded_region(tokens, float(margin)).intersects_many(footprints)
            on_map = self._full_surface().intersects_many(footprints)
            return hit | (in_expanded & ~on_map)
        return hit & self._eroded_full_surface(float(margin)).intersects_many(footprints)


class MapResolver:
    """Callable protocol: ``resolver(scene_id) -> Optional[Path]`` of the scene's OSM file."""

    def __call__(self, scene_id: Optional[str]) -> Optional[Path]:  # pragma: no cover - interface
        raise NotImplementedError


class T4SceneDirectoryResolver(MapResolver):
    """Resolves a T4 scene directory (or fragment under ``data_root``) to its ``lanelet2_map.osm``.

    Tried in order: ``<scene>/map/lanelet2_map.osm`` and ``<scene>/*/map/lanelet2_map.osm``
    (versioned T4 layout ``<db>/<uuid>/<version>/map``). With ``data_root`` given, ``scene_id`` is
    first taken as a fragment under ``data_root`` and then by its last path component.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        self.data_root = None if data_root is None else Path(data_root)

    def _candidates(self, scene_id: str) -> List[Path]:
        bases: List[Path] = []
        if self.data_root is not None:
            bases.append(self.data_root / scene_id)
            bases.append(self.data_root / Path(scene_id).name)
        bases.append(Path(scene_id))
        candidates: List[Path] = []
        for base in bases:
            candidates.append(base / "map" / DEFAULT_MAP_FILENAME)
            if base.is_dir():
                candidates.extend(sorted(base.glob(f"*/map/{DEFAULT_MAP_FILENAME}")))
        return candidates

    def __call__(self, scene_id: Optional[str]) -> Optional[Path]:
        if not scene_id:
            return None
        for candidate in self._candidates(str(scene_id)):
            if candidate.is_file():
                return candidate
        return None


class ExplicitMapResolver(MapResolver):
    """Resolves scene ids through an explicit ``{scene_id: osm_path}`` mapping (non-T4 data, tests)."""

    def __init__(self, mapping: Dict[str, str]) -> None:
        self.mapping = {str(key): Path(value) for key, value in dict(mapping).items()}

    def __call__(self, scene_id: Optional[str]) -> Optional[Path]:
        if scene_id is None:
            return None
        path = self.mapping.get(str(scene_id))
        return path if path is not None and path.is_file() else None


class LaneletMapProvider:
    """Loads and caches one :class:`LaneletMap` per scene.

    ``resolver(scene_id)`` maps a scene id to its OSM path (or ``None`` when the scene ships no
    map). :meth:`available` is an existence check that never parses; :meth:`get` parses (once per
    path) and lets a corrupt map raise, so a resource that should exist but is unreadable fails loud.
    """

    def __init__(self, resolver: Callable[[Optional[str]], Optional[Path]]) -> None:
        self._resolver = resolver
        self._cache: Dict[Optional[str], LaneletMap] = {}

    def resolve(self, scene_id: Optional[str]) -> Optional[Path]:
        """The scene's OSM path, or ``None`` when no map exists."""
        return self._resolver(scene_id)

    def available(self, scene_id: Optional[str]) -> bool:
        """Whether a lanelet map exists for the scene (no parse, no exception)."""
        return self.resolve(scene_id) is not None

    def get(self, scene_id: Optional[str]) -> LaneletMap:
        """The scene's parsed map, from the cache after the first request."""
        if scene_id not in self._cache:
            path = self.resolve(scene_id)
            if path is None:
                raise FileNotFoundError(
                    f"No lanelet map resolved for scene {scene_id!r}: this scene ships no map/ directory, "
                    "so map-dependent metrics cannot run on it."
                )
            self._cache[scene_id] = LaneletMap.from_osm(str(path))
        return self._cache[scene_id]


MAP_RESOLVER_TOKENS = ("t4_scene_directory", "explicit")


def build_map_provider(
    resolver_name: str,
    data_root: Optional[str] = None,
    mapping: Optional[Dict[str, str]] = None,
) -> LaneletMapProvider:
    """Build a provider from a closed set of resolver tokens (``t4_scene_directory``, ``explicit``)."""
    if resolver_name == "t4_scene_directory":
        return LaneletMapProvider(T4SceneDirectoryResolver(data_root))
    if resolver_name == "explicit":
        if not mapping:
            raise ValueError("The 'explicit' map resolver needs a non-empty scene_id -> osm_path mapping.")
        return LaneletMapProvider(ExplicitMapResolver(mapping))
    raise ValueError(f"Unknown map resolver {resolver_name!r}, expected one of {list(MAP_RESOLVER_TOKENS)}.")
