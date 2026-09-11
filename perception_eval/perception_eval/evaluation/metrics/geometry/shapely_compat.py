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

"""Version-neutral wrappers over the shapely calls the metric geometry needs.

``perception_eval`` supports shapely 1.8 on Python 3.10 and shapely 2.x on newer interpreters
(``pyproject.toml``). The source implementation used shapely 2 vectorized functions
(``shapely.prepare``, ``shapely.points``, ``shapely.contains``, ``shapely.intersects``,
``shapely.union_all``, ``shapely.get_parts``, integer-indexed ``STRtree.query``). Every such call
is routed through this module so metric code never depends on a shapely major version.
"""

from __future__ import annotations

from typing import Iterable
from typing import List
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import shapely
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

SHAPELY_2: bool = int(str(shapely.__version__).split(".")[0]) >= 2
"""True when the installed shapely exposes the 2.x vectorized API."""

if not SHAPELY_2:  # pragma: no cover - exercised on Python 3.10 only
    from shapely.prepared import prep as _prep


def union_all(geometries: Iterable[BaseGeometry]) -> BaseGeometry:
    """Union of many geometries (``shapely.ops.unary_union`` works on both majors)."""
    geometries = list(geometries)
    if not geometries:
        return Polygon()
    return unary_union(geometries)


def get_parts(geometry: BaseGeometry) -> List[BaseGeometry]:
    """The non-empty single parts of a (multi)geometry."""
    if geometry.is_empty:
        return []
    if hasattr(geometry, "geoms"):
        return [part for part in geometry.geoms if not part.is_empty]
    return [geometry]


def make_points(xy: NDArray) -> List[Point]:
    """``Point`` objects for ``(N, 2)`` coordinates (only needed on the 1.8 path)."""
    xy = np.asarray(xy, dtype=np.float64)
    return [Point(float(x), float(y)) for x, y in xy]


class Prepared:
    """A geometry with a prepared counterpart for repeated predicate tests.

    On shapely 2 the geometry is prepared in place and the vectorized predicates are used; on
    shapely 1.8 a ``PreparedGeometry`` is built and predicates loop in Python.
    """

    __slots__ = ("geometry", "_prepared")

    def __init__(self, geometry: BaseGeometry) -> None:
        self.geometry = geometry
        if SHAPELY_2:
            if not geometry.is_empty:
                shapely.prepare(geometry)
            self._prepared = geometry
        else:  # pragma: no cover - exercised on Python 3.10 only
            self._prepared = _prep(geometry) if not geometry.is_empty else None

    @property
    def is_empty(self) -> bool:
        return bool(self.geometry.is_empty)

    def contains_points(self, xy: NDArray) -> NDArray[np.bool_]:
        """Boolean mask of ``(N, 2)`` points contained in the geometry."""
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape[0] == 0 or self.is_empty:
            return np.zeros((xy.shape[0],), dtype=bool)
        if SHAPELY_2:
            points = shapely.points(xy[:, 0], xy[:, 1])
            return np.asarray(shapely.contains(self.geometry, points), dtype=bool)
        return np.fromiter((self._prepared.contains(point) for point in make_points(xy)), dtype=bool, count=xy.shape[0])

    def intersects_many(self, geometries: Sequence[BaseGeometry]) -> NDArray[np.bool_]:
        """Boolean mask of geometries intersecting this one."""
        if len(geometries) == 0 or self.is_empty:
            return np.zeros((len(geometries),), dtype=bool)
        if SHAPELY_2:
            return np.asarray(shapely.intersects(self.geometry, np.array(geometries, dtype=object)), dtype=bool)
        return np.fromiter((self._prepared.intersects(g) for g in geometries), dtype=bool, count=len(geometries))

    def intersects(self, geometry: BaseGeometry) -> bool:
        """Whether ``geometry`` intersects this one."""
        if self.is_empty or geometry.is_empty:
            return False
        return bool(self._prepared.intersects(geometry))

    def covers(self, geometry: BaseGeometry) -> bool:
        """Whether this geometry covers ``geometry`` (every point of it lies inside or on the border)."""
        if self.is_empty:
            return False
        if SHAPELY_2:
            return bool(shapely.covers(self.geometry, geometry))
        return bool(self._prepared.covers(geometry))

    def distance(self, geometry: BaseGeometry) -> float:
        """Distance between the wrapped geometry and ``geometry``."""
        return float(self.geometry.distance(geometry))


class SpatialIndex:
    """An STRtree over a fixed list of geometries returning integer indices."""

    def __init__(self, geometries: Sequence[BaseGeometry]) -> None:
        self.geometries = list(geometries)
        if not self.geometries:
            self._tree = None
        elif SHAPELY_2:
            self._tree = STRtree(self.geometries)
        else:  # pragma: no cover - exercised on Python 3.10 only
            self._tree = STRtree(self.geometries, items=list(range(len(self.geometries))))

    def __len__(self) -> int:
        return len(self.geometries)

    def query(self, geometry: BaseGeometry) -> NDArray[np.int64]:
        """Sorted indices of geometries whose envelope intersects ``geometry``'s envelope."""
        if self._tree is None:
            return np.zeros((0,), dtype=np.int64)
        if SHAPELY_2:
            indices = np.asarray(self._tree.query(geometry), dtype=np.int64)
        else:  # pragma: no cover
            indices = np.asarray(list(self._tree.query_items(geometry)), dtype=np.int64)
        return np.sort(indices.reshape(-1))
