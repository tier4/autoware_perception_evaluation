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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/collision.py) at fcf86419.

"""Detection-box to reachability time-to-collision adapter (metrics B1, B2).

Bridges the per-frame detection boxes (base_link ``[cx, cy, cz, dx, dy, dz, yaw, ...]``) to the
reachability engine: transforms ego and each box to the map frame, assigns every class its
reachable-set kind, pulls the drivable polygon and the per-lanelet speed limits from the scene's
lanelet map, and returns one TTC per box. Ego and wheeled agents move at the ``speed_limit`` of the
lanelet they are in (off-map they fall back to ``max_speed_mps``), VRUs use a per-class run speed.
Moving agents carry half their box width as the collision body radius, static agents use the full
footprint.
"""

from __future__ import annotations

from math import inf
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.filters import box_footprints_map
from perception_eval.evaluation.metrics.filters_map import DEFAULT_DRIVABLE_REGION
from perception_eval.evaluation.metrics.filters_map import ego_collision_agent
from perception_eval.evaluation.metrics.filters_map import validate_region_tokens
from perception_eval.evaluation.metrics.geometry.lanelet import LaneletMapProvider
from perception_eval.evaluation.metrics.geometry.reachability import Agent
from perception_eval.evaluation.metrics.geometry.reachability import AGENT_KINDS
from perception_eval.evaluation.metrics.geometry.reachability import EgoReachability
from perception_eval.evaluation.metrics.geometry.reachability import ReachabilityParams
from perception_eval.evaluation.metrics.geometry.reachability import STATIC
from perception_eval.evaluation.metrics.geometry.reachability import VRU
from perception_eval.evaluation.metrics.geometry.reachability import WHEELED

# AutowareLabel value -> reachable-set kind.
DEFAULT_COLLISION_KINDS: Dict[str, str] = {
    "car": WHEELED,
    "truck": WHEELED,
    "bus": WHEELED,
    "motorbike": WHEELED,
    "bicycle": VRU,
    "pedestrian": VRU,
    "animal": VRU,
    "hazard": STATIC,
    "unknown": STATIC,
}
# VRU "reasonable run" speeds (m/s). Wheeled speed comes from the lanelet map.
DEFAULT_VRU_SPEEDS: Dict[str, float] = {"pedestrian": 3.0, "animal": 4.0, "bicycle": 6.0}


class CollisionTTC:
    """Per-box reachability TTC for one detection frame.

    Args:
        class_names (Sequence[str]): Ordered class names (label index to name).
        map_provider (LaneletMapProvider): Resolves a ``scene_id`` to its lanelet map.
        kinds (dict[str, str] | None): Class name to reachable-set kind; defaults to
            :data:`DEFAULT_COLLISION_KINDS`. Every class name must be mapped.
        region (Sequence[str]): Drivable region tokens the wheeled fronts are clipped to.
        params (ReachabilityParams | None): Reachability parameters (horizon, dt, curvature bound).
        vru_speeds (dict[str, float] | None): VRU class name to run speed in m/s.
        max_speed_mps (float): Off-map fallback speed for ego and wheeled agents.
        ego_body_radius_m (float): Ego collision half-extent (ego has no detection box).
    """

    def __init__(
        self,
        class_names: Sequence[str],
        map_provider: LaneletMapProvider,
        *,
        kinds: Optional[Dict[str, str]] = None,
        region: Sequence[str] = DEFAULT_DRIVABLE_REGION,
        params: Optional[ReachabilityParams] = None,
        vru_speeds: Optional[Dict[str, float]] = None,
        max_speed_mps: float = 16.7,
        ego_body_radius_m: float = 1.0,
    ) -> None:
        self.class_names: Tuple[str, ...] = tuple(str(name) for name in class_names)
        self.map_provider = map_provider
        self.region = validate_region_tokens(region, "CollisionTTC")
        self.params = params or ReachabilityParams()
        self.kinds = {str(k): str(v) for k, v in (kinds if kinds is not None else DEFAULT_COLLISION_KINDS).items()}
        self.vru_speeds = {
            str(k): float(v) for k, v in (vru_speeds if vru_speeds is not None else DEFAULT_VRU_SPEEDS).items()
        }
        if max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be > 0.")
        self.max_speed_mps = float(max_speed_mps)
        self.ego_body_radius_m = float(ego_body_radius_m)
        unknown_kinds = sorted(set(self.kinds.values()) - AGENT_KINDS)
        if unknown_kinds:
            raise ValueError(f"unknown collision kinds {unknown_kinds}, valid: {sorted(AGENT_KINDS)}.")
        unmapped = sorted(set(self.class_names) - set(self.kinds))
        if unmapped:
            raise ValueError(f"no collision kind mapped for classes {unmapped}.")
        missing_vru = sorted(
            name for name in self.class_names if self.kinds[name] == VRU and name not in self.vru_speeds
        )
        if missing_vru:
            raise ValueError(f"no VRU run speed configured for classes {missing_vru}.")
        for name, speed in self.vru_speeds.items():
            if speed < 0.0:
                raise ValueError(f"VRU run speed for {name!r} must be >= 0.")

    def available(self, scene_id: Optional[str]) -> bool:
        """False for scenes with no lanelet map (excluded from B1/B2)."""
        return self.map_provider.available(scene_id)

    def per_box_ttc(
        self,
        boxes: NDArray,
        labels: NDArray,
        ego2map: NDArray,
        scene_id: Optional[str],
    ) -> NDArray[np.float64]:
        """TTC (seconds, ``inf`` = unreachable) for each base_link box in the frame."""
        boxes = np.asarray(boxes, dtype=np.float64)
        labels = np.asarray(labels).astype(int).reshape(-1)
        ttc = np.full(boxes.shape[0], inf, dtype=np.float64)
        if boxes.shape[0] == 0:
            return ttc
        if labels.shape[0] != boxes.shape[0]:
            raise ValueError("labels must align with boxes.")
        if labels.min() < 0 or labels.max() >= len(self.class_names):
            raise ValueError(f"labels must index class_names ({len(self.class_names)} classes).")

        lanelet_map = self.map_provider.get(scene_id)
        drivable = lanelet_map.region_union(self.region)
        ego = ego_collision_agent(lanelet_map, ego2map, self.max_speed_mps, self.ego_body_radius_m)
        frame = EgoReachability(ego, drivable, self.params)
        footprints = box_footprints_map(boxes, ego2map)
        centroids = np.array([[p.centroid.x, p.centroid.y] for p in footprints], dtype=np.float64)

        for index in range(boxes.shape[0]):
            name = self.class_names[int(labels[index])]
            kind = self.kinds[name]
            cx, cy = float(centroids[index, 0]), float(centroids[index, 1])
            body_radius = 0.5 * float(boxes[index, 4])  # half the box width
            if kind == STATIC:
                agent = Agent(STATIC, cx, cy, footprint=footprints[index])
            elif kind == WHEELED:
                speed = lanelet_map.speed_at(cx, cy, self.max_speed_mps)
                agent = Agent(WHEELED, cx, cy, float(boxes[index, 6]) + ego.heading, speed, body_radius)
            else:  # VRU: a class "reasonable run" speed, any direction.
                agent = Agent(VRU, cx, cy, 0.0, self.vru_speeds[name], body_radius)
            ttc[index] = frame.time_to_collision(agent)
        return ttc
