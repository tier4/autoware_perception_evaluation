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

import unittest

import numpy as np
from perception_eval.evaluation.metrics.filters import box_footprints_map
from perception_eval.evaluation.metrics.filters import CorridorFilter
from perception_eval.evaluation.metrics.filters import FrameMetricContext
from perception_eval.evaluation.metrics.filters import IDENTITY
from perception_eval.evaluation.metrics.filters import IdentityFilter
from perception_eval.evaluation.metrics.filters import to_map_xy
from perception_eval.evaluation.metrics.filters import validate_filters


def _box(x: float, y: float, dx: float = 4.0, dy: float = 2.0, yaw: float = 0.0):
    return [x, y, 0.0, dx, dy, 1.5, yaw, 0.0, 0.0]


class TestFrameMetricContext(unittest.TestCase):
    def test_ego_pose(self):
        context = FrameMetricContext(ego2map=None, scene_id=None)
        self.assertFalse(context.has_pose)
        with self.assertRaises(ValueError):
            context.ego_pose()
        matrix = np.eye(4)
        matrix[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]  # heading pi/2
        matrix[:3, 3] = [10.0, 20.0, 0.0]
        x, y, heading = FrameMetricContext(ego2map=matrix, scene_id="scene").ego_pose()
        self.assertEqual((x, y), (10.0, 20.0))
        self.assertAlmostEqual(heading, np.pi / 2)


class TestIdentityFilter(unittest.TestCase):
    def test_keeps_everything(self):
        context = FrameMetricContext()
        self.assertTrue(IDENTITY.available(context))
        self.assertEqual(IDENTITY.name, "")
        np.testing.assert_array_equal(IDENTITY.keep(np.zeros((3, 3)), context), [True, True, True])
        self.assertEqual(IdentityFilter().keep(np.zeros((0, 7)), context).shape, (0,))


class TestCorridorFilter(unittest.TestCase):
    def test_points(self):
        corridor = CorridorFilter(width_m=3.0)
        xyz = np.array([[10.0, 1.0, 0.0], [10.0, 1.6, 0.0], [-1.0, 0.0, 0.0], [5.0, -1.5, 0.0]])
        np.testing.assert_array_equal(corridor.keep(xyz, FrameMetricContext()), [True, False, False, True])
        self.assertEqual(corridor.keep(np.zeros((0, 3)), FrameMetricContext()).shape, (0,))

    def test_boxes_use_forward_footprint_overlap(self):
        corridor = CorridorFilter(width_m=3.0)
        boxes = np.array(
            [
                _box(10.0, 0.0),  # centered ahead -> keep
                _box(10.0, 2.0, dy=2.0),  # footprint y in [1, 3] -> overlaps |y| <= 1.5 -> keep
                _box(10.0, 3.0, dy=2.0),  # footprint y in [2, 4] -> outside -> drop
                _box(-10.0, 0.0),  # fully behind ego -> drop
                _box(-1.0, 0.0, dx=4.0),  # straddles x=0, forward part overlaps -> keep
            ]
        )
        np.testing.assert_array_equal(corridor.keep(boxes, FrameMetricContext()), [True, True, False, False, True])

    def test_validation_and_cache_key(self):
        with self.assertRaises(ValueError):
            CorridorFilter(width_m=0.0)
        with self.assertRaises(ValueError):
            CorridorFilter(width_m=1.0, name="")
        self.assertEqual(CorridorFilter(width_m=3.0).cache_key, "corridor:straight:w3")
        self.assertTrue(CorridorFilter().available(FrameMetricContext()))


class TestHelpers(unittest.TestCase):
    def test_to_map_xy_and_footprints(self):
        ego2map = np.eye(4)
        ego2map[:3, 3] = [100.0, 50.0, 0.0]
        xy = to_map_xy(np.array([[1.0, 2.0, 0.0]]), ego2map)
        np.testing.assert_allclose(xy, [[101.0, 52.0]])
        footprints = box_footprints_map(np.array([_box(0.0, 0.0, dx=4.0, dy=2.0)]), ego2map)
        self.assertEqual(len(footprints), 1)
        self.assertAlmostEqual(footprints[0].area, 8.0)
        self.assertAlmostEqual(footprints[0].centroid.x, 100.0)
        self.assertAlmostEqual(footprints[0].centroid.y, 50.0)

    def test_validate_filters(self):
        validate_filters([IDENTITY, CorridorFilter(name="a"), CorridorFilter(name="b")])
        with self.assertRaises(ValueError):
            validate_filters([CorridorFilter(name="a"), CorridorFilter(width_m=1.0, name="a")])


if __name__ == "__main__":
    unittest.main()
