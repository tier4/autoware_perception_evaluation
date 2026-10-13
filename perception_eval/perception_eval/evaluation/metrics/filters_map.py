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
# Ported from tier4/autoware-ml (autoware_ml/metrics/filters.py) at fcf86419.

"""Map-dependent evaluation filters: lanelet regions and the ego collision area.

``RegionFilter`` keeps the elements whose base_link position (points) or footprint (boxes),
transformed to the map frame by the per-frame ego pose, falls inside a chosen set of lanelet2
regions. ``CollisionFilter`` keeps the elements inside the ego collision area clipped to the road
lanelets, the filter form of the reachability collision model. Both report ``available`` False for
frames without a pose or whose scene ships no lanelet map, so a suite excludes those frames from
their views only.
"""

from __future__ import annotations

from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.filters import box_footprints_map
from perception_eval.evaluation.metrics.filters import FrameMetricContext
from perception_eval.evaluation.metrics.filters import is_boxes
from perception_eval.evaluation.metrics.filters import MetricFilter
from perception_eval.evaluation.metrics.filters import to_map_xy
from perception_eval.evaluation.metrics.geometry.lanelet import KNOWN_REGION_TOKENS
from perception_eval.evaluation.metrics.geometry.lanelet import LaneletMap
from perception_eval.evaluation.metrics.geometry.lanelet import LaneletMapProvider
from perception_eval.evaluation.metrics.geometry.reachability import Agent
from perception_eval.evaluation.metrics.geometry.reachability import ReachabilityParams
from perception_eval.evaluation.metrics.geometry.reachability import reachable_region
from perception_eval.evaluation.metrics.geometry.reachability import WHEELED
from perception_eval.evaluation.metrics.geometry.shapely_compat import Prepared

DEFAULT_DRIVABLE_REGION: Tuple[str, ...] = ("road", "road_shoulder", "crosswalk")


def validate_region_tokens(region: Sequence[str], owner: str) -> Tuple[str, ...]:
    """Return ``region`` as a tuple of known lanelet2 tokens, raising on unknown ones."""
    if not region:
        raise ValueError(f"{owner} needs at least one lanelet2 region token.")
    tokens = tuple(str(token) for token in region)
    unknown = sorted(set(tokens) - KNOWN_REGION_TOKENS)
    if unknown:
        raise ValueError(
            f"Unknown lanelet2 region tokens {unknown}, known tokens: {sorted(KNOWN_REGION_TOKENS)}. "
            "(A known token absent from a particular scene's map is fine, that slice is simply empty.)"
        )
    return tokens


def ego_collision_agent(lanelet_map: LaneletMap, ego2map: NDArray, max_speed_mps: float, body_radius: float) -> Agent:
    """Ego as a wheeled agent of the collision model, one recipe for every consumer.

    The pose comes from ``ego2map`` and the speed is the lanelet speed limit at ego's position
    with ``max_speed_mps`` as the off-map fallback.
    """
    ego_x, ego_y, ego_heading = FrameMetricContext(ego2map=ego2map).ego_pose()
    speed = lanelet_map.speed_at(ego_x, ego_y, max_speed_mps)
    return Agent(WHEELED, ego_x, ego_y, ego_heading, speed, body_radius)


class _MapFilter(MetricFilter):
    requires_pose = True
    requires_map = True

    def __init__(self, map_provider: LaneletMapProvider) -> None:
        self.map_provider = map_provider

    def available(self, context: FrameMetricContext) -> bool:
        """False for frames without a pose or in scenes with no lanelet map."""
        return context.has_pose and self.map_provider.available(context.scene_id)


class RegionFilter(_MapFilter):
    """Keep elements inside a union of lanelet2 regions.

    Points near the outer border of the whole mapped surface are naturally noisy, so the border can
    be adjusted by a margin that either cuts inward or grows outward.
    """

    def __init__(
        self,
        region: Sequence[str],
        map_provider: LaneletMapProvider,
        margin: float = 0.0,
        expand: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Validate the region tokens and derive the display name.

        Args:
            region (Sequence[str]): Literal lanelet2 tokens (lanelet ``subtype`` or area ``type``).
            map_provider (LaneletMapProvider): Resolves a scene id to its :class:`LaneletMap`.
            margin (float): Border adjustment in meters, must be non-negative.
            expand (bool): False cuts inward (outer-border points stop counting), True grows outward
                (claims off-map points within the margin, never points of another mapped region).
            name (str | None): Display name prefixing the metric keys; derived when omitted.
        """
        super().__init__(map_provider)
        self.region = validate_region_tokens(region, "RegionFilter")
        if margin < 0.0:
            raise ValueError("margin is a distance and must be >= 0, use expand to pick the direction.")
        self.margin = float(margin)
        self.expand = bool(expand)
        if name is not None:
            if not name:
                raise ValueError("RegionFilter needs a non-empty name.")
            self.name = str(name)
        else:
            direction_token = "x" if self.expand else "m"
            margin_token = f"_{direction_token}{self.margin:g}" if self.margin else ""
            self.name = "region_" + "_".join(self.region) + margin_token

    @property
    def cache_key(self) -> str:
        direction = "out" if self.expand else "in"
        return "region:" + ",".join(sorted(self.region)) + f":{self.margin:g}:{direction}"

    def keep(self, elements: NDArray, context: FrameMetricContext) -> NDArray[np.bool_]:
        """Mask of elements in the region (footprint overlap for boxes, containment for points)."""
        elements = np.asarray(elements, dtype=np.float64)
        if elements.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        lanelet_map = self.map_provider.get(context.scene_id)
        if is_boxes(elements):
            footprints = box_footprints_map(elements, context.ego2map)
            return lanelet_map.intersects(self.region, footprints, self.margin, self.expand)
        map_xy = to_map_xy(elements, context.ego2map)
        return lanelet_map.contains(self.region, map_xy, self.margin, self.expand)


class CollisionFilter(_MapFilter):
    """Keep elements in the ego's collision area, the filter form of the collision model.

    The collision area is everything ego could collide with within the horizon at the max map-legal
    speed under bounded steering, clipped to the road lanelets so it follows the road on bends. A
    detection box is kept when its footprint meets that region, a segmentation point when it lies
    inside it. Planner-independent, and the same collision model the criticality metrics (B1, B2)
    use, so any other metric can be reported in-path too.
    """

    def __init__(
        self,
        map_provider: LaneletMapProvider,
        region: Sequence[str] = DEFAULT_DRIVABLE_REGION,
        params: Optional[ReachabilityParams] = None,
        max_speed_mps: float = 16.7,
        ego_body_m: float = 1.0,
        name: str = "collision",
    ) -> None:
        """Validate the road region tokens and the ego propagation parameters.

        Args:
            map_provider (LaneletMapProvider): Resolves a scene id to its :class:`LaneletMap`.
            region (Sequence[str]): Road lanelet tokens the collision area is clipped to.
            params (ReachabilityParams | None): Ego propagation parameters.
            max_speed_mps (float): Ego speed fallback where the map has no speed limit.
            ego_body_m (float): Assumed ego half width in meters.
            name (str): Display name prefixing the metric keys.
        """
        super().__init__(map_provider)
        self.region = validate_region_tokens(region, "CollisionFilter")
        if max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be > 0.")
        if not name:
            raise ValueError("CollisionFilter needs a non-empty name.")
        self.params = params or ReachabilityParams()
        self.max_speed_mps = float(max_speed_mps)
        self.ego_body_m = float(ego_body_m)
        self.name = str(name)
        # keep() is called for GT and predictions of the same frame in sequence; the ego collision
        # area depends only on the frame's pose, so the last region is memoized.
        self._region_memo: Optional[Tuple[tuple, Prepared]] = None

    @property
    def cache_key(self) -> str:
        region = ",".join(sorted(self.region))
        params = self.params
        return (
            f"collision-area:v{self.max_speed_mps:g}:b{self.ego_body_m:g}"
            f":h{params.horizon_s:g},{params.dt_s:g},{params.max_lateral_accel_mps2:g}"
            f",{params.min_radius_m:g},{params.arc_samples}:{region}"
        )

    def ego_region(self, context: FrameMetricContext) -> Prepared:
        """The ego collision area in the map frame for this frame (memoized)."""
        ego_x, ego_y, ego_heading = context.ego_pose()
        key = (str(context.scene_id), ego_x, ego_y, ego_heading)
        if self._region_memo is not None and self._region_memo[0] == key:
            return self._region_memo[1]
        lanelet_map = self.map_provider.get(context.scene_id)
        drivable = lanelet_map.region_union(self.region)
        ego = ego_collision_agent(lanelet_map, context.ego2map, self.max_speed_mps, self.ego_body_m)
        region = Prepared(reachable_region(ego, self.params, drivable))
        self._region_memo = (key, region)
        return region

    def keep(self, elements: NDArray, context: FrameMetricContext) -> NDArray[np.bool_]:
        """Mask of elements inside the ego collision area (footprint / point test)."""
        elements = np.asarray(elements, dtype=np.float64)
        if elements.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        region = self.ego_region(context)
        if region.is_empty:
            return np.zeros(elements.shape[0], dtype=bool)
        if is_boxes(elements):
            return region.intersects_many(box_footprints_map(elements, context.ego2map))
        return region.contains_points(to_map_xy(elements, context.ego2map))
