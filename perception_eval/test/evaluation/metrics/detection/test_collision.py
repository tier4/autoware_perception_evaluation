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

from math import inf
from math import pi
import unittest

import numpy as np
from perception_eval.evaluation.metrics.detection.collision import CollisionTTC
from perception_eval.evaluation.metrics.detection.collision import DEFAULT_COLLISION_KINDS
from perception_eval.evaluation.metrics.detection.collision import DEFAULT_VRU_SPEEDS
from perception_eval.evaluation.metrics.geometry.lanelet import LaneletMap
from perception_eval.evaluation.metrics.geometry.reachability import ReachabilityParams
from perception_eval.evaluation.metrics.geometry.reachability import STATIC
from perception_eval.evaluation.metrics.geometry.reachability import VRU
from perception_eval.evaluation.metrics.geometry.reachability import WHEELED
from shapely.geometry import Polygon

ROAD = Polygon([(-20, -10), (200, -10), (200, 10), (-20, 10)])
CLASS_NAMES = ("car", "pedestrian", "hazard")


class StubProvider:
    def __init__(self, lanelet_map: LaneletMap, no_map=()) -> None:
        self._map = lanelet_map
        self._no_map = set(no_map)

    def get(self, scene_id) -> LaneletMap:
        if scene_id in self._no_map:
            raise FileNotFoundError(scene_id)
        return self._map

    def available(self, scene_id) -> bool:
        return scene_id not in self._no_map


def _box(x: float, y: float = 0.0, yaw: float = 0.0):
    return [x, y, 0.0, 4.0, 2.0, 1.5, yaw, 0.0, 0.0]


class TestCollisionTTC(unittest.TestCase):
    def _ttc(self, **kwargs) -> CollisionTTC:
        return CollisionTTC(
            CLASS_NAMES,
            StubProvider(LaneletMap({"road": [ROAD]}), no_map=("no_map",)),
            region=("road",),
            params=ReachabilityParams(horizon_s=4.0, dt_s=0.1),
            max_speed_mps=10.0,
            **kwargs,
        )

    def test_per_box_ttc_by_kind(self):
        ttc = self._ttc()
        boxes = np.array(
            [
                _box(30.0),  # static hazard 30 m ahead -> finite
                _box(-30.0),  # static hazard behind ego -> unreachable
                _box(25.0),  # same-speed car lead -> unreachable
                _box(40.0, yaw=pi),  # oncoming car -> finite
                _box(18.0, 6.0),  # crossing pedestrian -> finite
                _box(150.0),  # static far away -> unreachable
            ]
        )
        labels = np.array([2, 2, 0, 0, 1, 2])
        result = ttc.per_box_ttc(boxes, labels, np.eye(4), "scene")
        self.assertTrue(2.4 <= result[0] <= 3.1)
        self.assertEqual(result[1], inf)
        self.assertEqual(result[2], inf)
        self.assertTrue(1.7 <= result[3] <= 2.1)
        self.assertTrue(np.isfinite(result[4]) and result[4] <= 4.0)
        self.assertEqual(result[5], inf)
        self.assertEqual(ttc.per_box_ttc(np.zeros((0, 9)), np.zeros(0), np.eye(4), "scene").shape, (0,))

    def test_ego_pose_is_applied(self):
        ttc = self._ttc()
        ego2map = np.eye(4)
        ego2map[:3, 3] = [100.0, 0.0, 0.0]
        result = ttc.per_box_ttc(np.array([_box(30.0)]), np.array([2]), ego2map, "scene")
        self.assertTrue(np.isfinite(result[0]))

    def test_availability(self):
        ttc = self._ttc()
        self.assertTrue(ttc.available("scene"))
        self.assertFalse(ttc.available("no_map"))
        with self.assertRaises(FileNotFoundError):
            ttc.per_box_ttc(np.array([_box(30.0)]), np.array([2]), np.eye(4), "no_map")

    def test_validation(self):
        provider = StubProvider(LaneletMap({"road": [ROAD]}))
        with self.assertRaisesRegex(ValueError, "no collision kind mapped"):
            CollisionTTC(("car", "spaceship"), provider)
        with self.assertRaisesRegex(ValueError, "unknown collision kinds"):
            CollisionTTC(("car",), provider, kinds={"car": "hovering"})
        with self.assertRaisesRegex(ValueError, "no VRU run speed"):
            CollisionTTC(("car", "pedestrian"), provider, vru_speeds={})
        with self.assertRaises(ValueError):
            CollisionTTC(("car",), provider, region=("nope",))
        with self.assertRaises(ValueError):
            self._ttc().per_box_ttc(np.array([_box(1.0)]), np.array([5]), np.eye(4), "scene")
        with self.assertRaises(ValueError):
            self._ttc().per_box_ttc(np.array([_box(1.0)]), np.array([0, 1]), np.eye(4), "scene")

    def test_defaults_cover_autoware_labels(self):
        expected = {"car", "truck", "bus", "motorbike", "bicycle", "pedestrian", "animal", "hazard", "unknown"}
        self.assertEqual(set(DEFAULT_COLLISION_KINDS), expected)
        self.assertEqual(DEFAULT_COLLISION_KINDS["motorbike"], WHEELED)
        self.assertEqual(DEFAULT_COLLISION_KINDS["hazard"], STATIC)
        self.assertEqual(DEFAULT_COLLISION_KINDS["bicycle"], VRU)
        self.assertEqual(set(DEFAULT_VRU_SPEEDS), {"pedestrian", "animal", "bicycle"})


if __name__ == "__main__":
    unittest.main()
