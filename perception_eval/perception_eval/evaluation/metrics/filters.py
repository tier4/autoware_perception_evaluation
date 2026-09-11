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
# Ported from tier4/autoware-ml (autoware_ml/metrics/base.py, filters.py) at fcf86419.

"""Evaluation filters: the spatial selection axis of the driving-aware metrics.

A filter is an *axis*, not a metric: it decides which elements (detection boxes, segmentation
points) a metric sees, exactly as the range buckets do. ``name`` prefixes the metric keys (empty
for the identity filter). ``keep`` receives either bare base_link points ``(N, 3+)`` or full box
rows ``(N, 7+)`` and dispatches on the column count so a box's whole footprint (not just its
center) can be tested.

Filters that depend on external data (a lanelet map) report per-frame availability through
``available`` so a suite can exclude uncovered frames from that filter's view only.

:class:`IdentityFilter` and :class:`CorridorFilter` need no map. :class:`RegionFilter` and
:class:`CollisionFilter` are map-dependent and live in this module too so a config token maps to
exactly one class.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.detection.geometry import bev_corners
from perception_eval.evaluation.metrics.detection.geometry import clip_forward


@dataclass(frozen=True)
class FrameMetricContext:
    """Per-frame data the filters may need.

    Attributes:
        ego2map (NDArray | None): 4x4 homogeneous transform from base_link to the map frame, or
            ``None`` when the frame carries no pose.
        scene_id (str | None): Identifier resolved to a lanelet map by the map provider.
    """

    ego2map: Optional[NDArray] = None
    scene_id: Optional[str] = None

    @property
    def has_pose(self) -> bool:
        return self.ego2map is not None

    def ego_pose(self) -> Tuple[float, float, float]:
        """Ego map-frame ``(x, y, heading)``."""
        if self.ego2map is None:
            raise ValueError("frame context carries no ego2map transform.")
        matrix = np.asarray(self.ego2map, dtype=np.float64)
        return float(matrix[0, 3]), float(matrix[1, 3]), atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def is_boxes(elements: NDArray) -> bool:
    """Elements are detection boxes ``[cx,cy,cz,dx,dy,dz,yaw,...]`` rather than bare points."""
    return elements.ndim == 2 and elements.shape[1] >= 7


def to_map_xy(xyz: NDArray, ego2map: NDArray) -> NDArray[np.float64]:
    """Transform base_link ``xyz`` (N, 3+) to map-frame xy via the 4x4 ``ego2map``."""
    xyz = np.asarray(xyz, dtype=np.float64)
    transform = np.asarray(ego2map, dtype=np.float64)
    homogeneous = np.concatenate([xyz[:, :3], np.ones((xyz.shape[0], 1))], axis=1)
    return (homogeneous @ transform.T)[:, :2]


def box_footprints_map(boxes: NDArray, ego2map: NDArray) -> list:
    """Map-frame BEV footprint polygons for base_link box rows (shapely ``Polygon`` list)."""
    from shapely.geometry import Polygon  # geometry-only import kept local

    footprints = []
    for box in np.asarray(boxes, dtype=np.float64):
        corners = bev_corners(box)  # (4, 2) base_link
        corners3 = np.column_stack([corners, np.full(corners.shape[0], box[2])])
        footprints.append(Polygon(to_map_xy(corners3, ego2map)))
    return footprints


class MetricFilter:
    """Base class of the selection axis.

    Attributes:
        name (str): Prefix of the metric keys, empty for the identity filter.
        requires_pose (bool): Whether ``keep`` needs ``context.ego2map``.
        requires_map (bool): Whether ``available`` depends on a lanelet map for the scene.
    """

    name: str = ""
    requires_pose: bool = False
    requires_map: bool = False

    @property
    def cache_key(self) -> str:
        """Value key grouping equivalent filters; equal keys must mean equal masks."""
        return self.name

    def available(self, context: FrameMetricContext) -> bool:
        """Whether this filter can evaluate the given frame (default: always)."""
        return True

    def keep(self, elements: NDArray, context: FrameMetricContext) -> NDArray[np.bool_]:
        """Boolean mask over elements (points ``(N, 3+)`` or boxes ``(N, 7+)``) to retain."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class IdentityFilter(MetricFilter):
    """The default filter. Keeps everything, adds no key prefix."""

    name = ""

    def keep(self, elements: NDArray, context: FrameMetricContext) -> NDArray[np.bool_]:
        return np.ones((len(elements),), dtype=bool)


IDENTITY = IdentityFilter()


class CorridorFilter(MetricFilter):
    """Keep elements inside a straight corridor ahead of ego.

    The corridor is a forward strip in the ego frame: ``width_m`` across, centered on the x axis,
    with no length bound of its own because distance slicing is the range axis's job. It needs no
    map and no pose, so the slice covers every scene. A detection box is kept when its footprint
    overlaps the strip, a segmentation point when it lies inside it.
    """

    def __init__(self, width_m: float = 3.0, name: str = "corridor") -> None:
        if width_m <= 0.0:
            raise ValueError("width_m must be > 0.")
        if not name:
            raise ValueError("CorridorFilter needs a non-empty name.")
        self.width_m = float(width_m)
        self.name = str(name)

    @property
    def cache_key(self) -> str:
        return f"corridor:straight:w{self.width_m:g}"

    def keep(self, elements: NDArray, context: FrameMetricContext) -> NDArray[np.bool_]:
        elements = np.asarray(elements, dtype=np.float64)
        if elements.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        half_width = self.width_m / 2.0
        if is_boxes(elements):
            keep = np.zeros(elements.shape[0], dtype=bool)
            for index, row in enumerate(elements):
                forward = clip_forward(bev_corners(row))
                keep[index] = (
                    forward.shape[0] > 0
                    and float(forward[:, 1].min()) <= half_width
                    and float(forward[:, 1].max()) >= -half_width
                )
            return keep
        return (elements[:, 0] >= 0.0) & (np.abs(elements[:, 1]) <= half_width)


def validate_filters(filters: List[MetricFilter]) -> List[MetricFilter]:
    """Reject non-identity filters with empty or duplicate names."""
    names: List[str] = []
    for metric_filter in filters:
        if isinstance(metric_filter, IdentityFilter):
            continue
        if not metric_filter.name:
            raise ValueError(
                f"{type(metric_filter).__name__} has an empty name, it would silently be applied as the identity."
            )
        if metric_filter.name in names:
            raise ValueError(f"Filter names must be unique: {metric_filter.name!r} is used twice.")
        names.append(metric_filter.name)
    return filters


def _region_and_collision_filters() -> Tuple[Any, Any]:
    """Lazy import so the map-free filters never pull shapely-heavy modules at import time."""
    from perception_eval.evaluation.metrics.filters_map import CollisionFilter
    from perception_eval.evaluation.metrics.filters_map import RegionFilter

    return RegionFilter, CollisionFilter


def __getattr__(name: str) -> Any:  # pragma: no cover - thin re-export
    if name in ("RegionFilter", "CollisionFilter"):
        region_filter, collision_filter = _region_and_collision_filters()
        return region_filter if name == "RegionFilter" else collision_filter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
