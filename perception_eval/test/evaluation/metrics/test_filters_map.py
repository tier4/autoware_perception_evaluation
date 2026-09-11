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
# Ported from tier4/autoware-ml (autoware_ml/tests/metrics/test_region_filter.py) at fcf86419.

import unittest

import numpy as np
from perception_eval.evaluation.metrics import filters
from perception_eval.evaluation.metrics.filters import FrameMetricContext
from perception_eval.evaluation.metrics.filters_map import CollisionFilter
from perception_eval.evaluation.metrics.filters_map import ego_collision_agent
from perception_eval.evaluation.metrics.filters_map import RegionFilter
from perception_eval.evaluation.metrics.geometry.lanelet import LaneletMap
from perception_eval.evaluation.metrics.geometry.reachability import ReachabilityParams
from perception_eval.evaluation.metrics.geometry.reachability import WHEELED
from shapely.geometry import Polygon


class StubProvider:
    """Returns one fixed LaneletMap for every scene id; ``no_map`` scenes are unavailable."""

    def __init__(self, lanelet_map: LaneletMap, no_map=()) -> None:
        self._map = lanelet_map
        self._no_map = set(no_map)

    def get(self, scene_id) -> LaneletMap:
        if scene_id in self._no_map:
            raise FileNotFoundError(f"map loaded for a scene declared map-less: {scene_id!r}")
        return self._map

    def available(self, scene_id) -> bool:
        return scene_id not in self._no_map


def road_map() -> LaneletMap:
    # A rectangular road region covering x in [0, 50], y in [-10, 10] (map frame).
    return LaneletMap({"road": [Polygon([(0, -10), (50, -10), (50, 10), (0, 10)])]})


def narrow_road_map() -> LaneletMap:
    return LaneletMap({"road": [Polygon([(0, -3), (50, -3), (50, 3), (0, 3)])]})


CONTEXT = FrameMetricContext(ego2map=np.eye(4), scene_id="scene")


class TestRegionFilter(unittest.TestCase):
    def test_membership(self):
        region_filter = RegionFilter(["road"], StubProvider(road_map()))
        xyz = np.array([[10.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        self.assertEqual(region_filter.keep(xyz, CONTEXT).tolist(), [True, False])
        self.assertEqual(region_filter.keep(np.zeros((0, 3)), CONTEXT).shape, (0,))
        self.assertEqual(region_filter.name, "region_road")
        self.assertEqual(RegionFilter(["road"], StubProvider(road_map()), margin=1.5).name, "region_road_m1.5")
        self.assertEqual(
            RegionFilter(["road"], StubProvider(road_map()), margin=1.0, expand=True).name, "region_road_x1"
        )
        self.assertTrue(region_filter.requires_pose and region_filter.requires_map)

    def test_pose_transform_is_applied(self):
        ego2map = np.eye(4)
        ego2map[:3, 3] = [45.0, 0.0, 0.0]
        context = FrameMetricContext(ego2map=ego2map, scene_id="scene")
        region_filter = RegionFilter(["road"], StubProvider(road_map()))
        # 3 m ahead of ego lands at map x=48 (on road), 10 m ahead at x=55 (off road).
        self.assertEqual(
            region_filter.keep(np.array([[3.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), context).tolist(), [True, False]
        )

    def test_box_footprint_any_overlap(self):
        region_filter = RegionFilter(["road"], StubProvider(narrow_road_map()))
        boxes = np.array(
            [
                [10.0, 4.0, 0.0, 4.0, 4.0, 1.5, 0.0, 0.0, 0.0],  # footprint y in [2, 6] -> overlaps road
                [10.0, 8.0, 0.0, 4.0, 4.0, 1.5, 0.0, 0.0, 0.0],  # footprint y in [6, 10] -> off road
            ]
        )
        self.assertEqual(region_filter.keep(boxes, CONTEXT).tolist(), [True, False])

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "Unknown lanelet2 region tokens"):
            RegionFilter(["sidewalk_typo"], StubProvider(road_map()))
        with self.assertRaises(ValueError):
            RegionFilter([], StubProvider(road_map()))
        with self.assertRaises(ValueError):
            RegionFilter(["road"], StubProvider(road_map()), margin=-1.0)
        with self.assertRaises(ValueError):
            RegionFilter(["road"], StubProvider(road_map()), name="")

    def test_availability(self):
        region_filter = RegionFilter(["road"], StubProvider(road_map(), no_map=("no_map",)))
        self.assertTrue(region_filter.available(CONTEXT))
        self.assertFalse(region_filter.available(FrameMetricContext(ego2map=np.eye(4), scene_id="no_map")))
        self.assertFalse(region_filter.available(FrameMetricContext(ego2map=None, scene_id="scene")))

    def test_cache_key_groups_equal_masks(self):
        a = RegionFilter(["road", "walkway"], StubProvider(road_map()), name="a")
        b = RegionFilter(["walkway", "road"], StubProvider(road_map()), name="b")
        self.assertEqual(a.cache_key, b.cache_key)
        self.assertNotEqual(a.cache_key, RegionFilter(["road"], StubProvider(road_map())).cache_key)

    def test_lazy_reexport_from_filters_module(self):
        self.assertIs(filters.RegionFilter, RegionFilter)
        self.assertIs(filters.CollisionFilter, CollisionFilter)


class TestCollisionFilter(unittest.TestCase):
    def _filter(self, **kwargs) -> CollisionFilter:
        return CollisionFilter(
            StubProvider(road_map()),
            region=["road"],
            max_speed_mps=10.0,
            params=ReachabilityParams(horizon_s=4.0, dt_s=0.1),
            **kwargs,
        )

    def test_keeps_forward_reachable_on_road(self):
        xyz = np.array(
            [
                [10.0, 0.0, 0.0],  # ahead on the centerline, reachable -> keep
                [-10.0, 0.0, 0.0],  # behind ego -> drop
                [100.0, 0.0, 0.0],  # off the road (x > 50) -> drop
            ]
        )
        self.assertEqual(self._filter().keep(xyz, CONTEXT).tolist(), [True, False, False])

    def test_curvature_bound_excludes_close_lateral(self):
        self.assertEqual(self._filter().keep(np.array([[5.0, 8.0, 0.0]]), CONTEXT).tolist(), [False])

    def test_boxes_use_footprint_overlap(self):
        boxes = np.array(
            [
                [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0],
                [-10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0],
            ]
        )
        self.assertEqual(self._filter().keep(boxes, CONTEXT).tolist(), [True, False])
        self.assertEqual(self._filter().keep(np.zeros((0, 9)), CONTEXT).shape, (0,))

    def test_validation_and_keys(self):
        with self.assertRaises(ValueError):
            CollisionFilter(StubProvider(road_map()), region=["nope"])
        with self.assertRaises(ValueError):
            CollisionFilter(StubProvider(road_map()), max_speed_mps=0.0)
        collision_filter = self._filter()
        self.assertEqual(collision_filter.name, "collision")
        self.assertTrue(collision_filter.requires_pose and collision_filter.requires_map)
        self.assertIn("collision-area:v10", collision_filter.cache_key)
        self.assertFalse(collision_filter.available(FrameMetricContext(ego2map=None, scene_id="scene")))

    def test_ego_collision_agent_uses_map_speed(self):
        lanelet_map = LaneletMap(
            {"road": [Polygon([(0, -10), (50, -10), (50, 10), (0, 10)])]},
            speed_lanelets=[(Polygon([(0, -10), (50, -10), (50, 10), (0, 10)]), 5.0)],
        )
        inside = np.eye(4)
        inside[:3, 3] = [10.0, 0.0, 0.0]
        agent = ego_collision_agent(lanelet_map, inside, max_speed_mps=16.7, body_radius=1.0)
        self.assertEqual(agent.kind, WHEELED)
        self.assertAlmostEqual(agent.x, 10.0)
        self.assertAlmostEqual(agent.speed, 5.0)
        off_map = np.eye(4)
        off_map[:3, 3] = [500.0, 0.0, 0.0]
        self.assertAlmostEqual(ego_collision_agent(lanelet_map, off_map, 16.7, 1.0).speed, 16.7)


if __name__ == "__main__":
    unittest.main()
