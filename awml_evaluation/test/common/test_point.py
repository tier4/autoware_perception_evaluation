# Copyright 2022 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List
from typing import Tuple
import unittest

from awml_evaluation.common.point import crop_pointcloud
from awml_evaluation.common.point import distance_points
from awml_evaluation.common.point import distance_points_bev
from awml_evaluation.common.point import to_bev
import numpy as np


class TestPoint(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_distance_points(self):
        """[summary]
        Test calculating the 3d center distance between two points.

        test patterns:
            Test if the 3d distance between point_1 and point_2 is
            almost equal to ans_distance.
        """
        # patterns: (point_1, point_2, ans_distance)
        patterns: List[Tuple[np.ndarray, np.ndarray, float]] = [
            (np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)), 0.0),
            (np.array((0.0, 0.0, 0.0)), np.array((1.0, 1.0, 1.0)), math.sqrt(3 * 1.0**2)),
            (np.array((0.0, 0.0, 0.0)), np.array((1.0, -1.0, 1.0)), math.sqrt(3 * 1.0**2)),
            (np.array((0.0, 0.0, 0.0)), np.array((-1.0, -1.0, 1.0)), math.sqrt(3 * 1.0**2)),
            (np.array((0.0, 0.0, 0.0)), np.array((-1.0, -1.0, -1.0)), math.sqrt(3 * 1.0**2)),
        ]
        for point_1, point_2, ans_distance in patterns:
            with self.subTest("Test get_distance_points"):
                distance = distance_points(point_1, point_2)
                self.assertAlmostEqual(distance, ans_distance)

    def test_get_distance_points_bev(self):
        """[summary]
        Test calculating the 2d center distance between two points.

        test patterns:
            Test if the 2d distance between point_1 and point_2 is
            almost equal to ans_distance.
        """
        # patterns: (point_1, point_2, ans_distance)
        patterns: List[Tuple[np.ndarray, np.ndarray, float]] = [
            (np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 0.0)), 0.0),
            (np.array((0.0, 0.0, 0.0)), np.array((1.0, 1.0, 1.0)), math.sqrt(2 * 1.0**2)),
            (np.array((0.0, 0.0, 0.0)), np.array((1.0, -1.0, 1.0)), math.sqrt(2 * 1.0**2)),
            (np.array((0.0, 0.0, 0.0)), np.array((-1.0, -1.0, 5.0)), math.sqrt(2 * 1.0**2)),
            (np.array((0.0, 0.0, 0.0)), np.array((-1.0, -1.0, -10.0)), math.sqrt(2 * 1.0**2)),
        ]
        for point_1, point_2, ans_distance in patterns:
            with self.subTest("Test get_distance_points_bev"):
                distance = distance_points_bev(point_1, point_2)
                self.assertAlmostEqual(distance, ans_distance)

    def test_to_bev(self):
        """[summary]
        Test (x, y, z) -> (x, y)

        test patterns:
            Test if 3d point_1 is converted to 2d point_1 (=ans_point).
        """
        # patterns: (point_1,  ans_point)
        patterns: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0))),
            (np.array((1.0, 0.0, 1.0)), np.array((1.0, 0.0))),
            (np.array((0.0, -1.0, 5.0)), np.array((0.0, -1.0))),
        ]
        for point_1, ans_point in patterns:
            with self.subTest("Test to_bev"):
                point = to_bev(point_1)
                self.assertEqual(point.tolist(), ans_point.tolist())

    def test_crop_pointcloud(self):
        """[summary]
        Test crop pointcloud (N, 3) -> (M, 3)

        test parameter
            Test if 3d in_point is same as cropped ans_point
        """
        # patterns: (in_points, area, ans_points)
        patterns: List[Tuple[np.ndarray, List[Tuple[float, float]], np.ndarray]] = [
            (
                # in_points
                np.array(
                    (
                        (0.0, 0.0, 0.0),
                        (0.05, 0.05, 0.05),
                        (0.1, 0.1, 0.1),
                        (0.3, 0.3, 0.3),
                    )
                ),
                # area
                [
                    (0.0, 0.0, 0.0),
                    (0.2, 0.0, 0.0),
                    (0.2, 0.2, 0.0),
                    (0.0, 0.2, 0.0),
                    (0.0, 0.0, 0.2),
                    (0.2, 0.0, 0.2),
                    (0.2, 0.2, 0.2),
                    (0.0, 0.2, 0.2),
                ],
                # ans_points
                np.array(
                    (
                        (0.0, 0.0, 0.0),
                        (0.05, 0.05, 0.05),
                        (0.1, 0.1, 0.1),
                    )
                ),
            ),
            (
                # in_points
                np.array(
                    (
                        (0.0, 0.0, 0.0),
                        (0.1, 0.1, 0.1),
                        (0.3, 0.3, 0.3),
                    )
                ),
                # area
                [
                    (0.05, 0.05, 0.0),
                    (0.2, 0.0, 0.0),
                    (0.2, 0.2, 0.0),
                    (0.1, 0.3, 0.0),
                    (0.0, 0.2, 0.0),
                    (0.05, 0.05, 0.2),
                    (0.2, 0.0, 0.2),
                    (0.2, 0.2, 0.2),
                    (0.1, 0.3, 0.2),
                    (0.0, 0.2, 0.2),
                ],
                # ans_points
                np.array([(0.1, 0.1, 0.1)]),
            ),
        ]
        for in_points, area, ans_points in patterns:
            with self.subTest("Test crop_pointcloud"):
                out_points = crop_pointcloud(in_points, area)
                self.assertEqual(out_points.tolist(), ans_points.tolist())


if __name__ == "__main__":
    unittest.main()
