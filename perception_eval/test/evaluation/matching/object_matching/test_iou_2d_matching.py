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
from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple
import unittest

from perception_eval.evaluation.matching.object_matching import _get_area_intersection
from perception_eval.evaluation.matching.object_matching import IOU2dMatching
from perception_eval.object import DynamicObject
from perception_eval.util.debug import get_objects_with_difference


class TestIou2dMatching(unittest.TestCase):
    def setUp(self):
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()

    def test_get_area_intersection(self):
        """[summary]
        Test getting the area at intersection.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if area_intersection and ans_area_intersection are equal.
        """
        # patterns: (diff_distance, List[ans_area_intersection])
        patterns: List[Tuple[float, float]] = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        for diff_distance, ans_area_intersection in patterns:
            with self.subTest("Test get_area_intersection."):
                diff_distance_dummy_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                for estimated_object, ground_truth_object in zip(
                    diff_distance_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    area_intersection = _get_area_intersection(estimated_object, ground_truth_object)
                    self.assertAlmostEqual(area_intersection, ans_area_intersection)

    def test_get_iou_bev_matching(self):
        """[summary]
        Test calculating BEV IoU.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if iou_bev and ans_iou_bev are equal.
        """
        # patterns: (diff_distance, ans_iou_bev)
        distance_patterns: List[Tuple[float, float]] = [
            # Given no diff_distance, iou is 1.0.
            (0.0, 1.0),
            # Given diff_distance is 0.5 for one axis, iou is 0.5 / 1.5
            # since ground_truth_objects and estimated_objects are half overlapping.
            (0.5, 0.5 / 1.5),
            # Given diff_distance is 1.0 for one axis, iou is 0.0
            # since ground_truth_objects and estimated_objects are no overlapping.
            # Given no diff_yaw, iou is 1.0.
            (0.0, 1.0),
        ]
        # patterns: (diff_yaw, ans_iou_bev)
        yaw_patterns: List[Tuple[float, float]] = [
            # Given vertical diff_yaw, iou is 1.0
            # since ground_truth_objects and estimated_objects overlap exactly.
            (math.pi / 2.0, 1.0),
            (-math.pi / 2.0, 1.0),
            # Given opposite direction, iou is 1.0.
            # since ground_truth_objects and estimated_objects overlap exactly.
            (math.pi, 1.0),
            (-math.pi, 1.0),
            # Given the ground_truth_objects and estimated_objects are the same size and crossed
            # at pi/4 or 3*pi/4, intersection_area is 2*(math.sqrt(2)-1)=0.8284271247461903 and union_area
            # is 2*(2-math.sqrt(2))=1.1715728752538097 thus iou is intersection_area/union_area
            # =0.7071067811865472.
            #     /\
            #   ------
            #   |/  \|
            #  /|    |\
            #  \|    |/
            #   |\  /|
            #   ------
            #     \/
            (math.pi / 4, 0.7071067811865472),
            (-math.pi / 4, 0.7071067811865472),
            (3 * math.pi / 4, 0.7071067811865472),
            (-3 * math.pi / 4, 0.7071067811865472),
        ]

        for diff_distance, ans_iou_bev in distance_patterns:
            with self.subTest("Test diff_distance iou_bev"):
                diff_yaw_dummy_ground_truth_objects: List[DynamicObject] = []
                diff_yaw_dummy_ground_truth_objects = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0.0,
                )

                for estimated_object, ground_truth_object in zip(
                    diff_yaw_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    iou_bev = IOU2dMatching(estimated_object, ground_truth_object)
                    self.assertAlmostEqual(iou_bev.value, ans_iou_bev)

        for diff_yaw, ans_iou_bev in yaw_patterns:
            with self.subTest("Test diff_yaw iou_bev"):
                diff_yaw_dummy_ground_truth_objects: List[DynamicObject] = []
                diff_yaw_dummy_ground_truth_objects = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=diff_yaw,
                )

                for estimated_object, ground_truth_object in zip(
                    diff_yaw_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    iou_bev = IOU2dMatching(estimated_object, ground_truth_object)
                    self.assertAlmostEqual(iou_bev.value, ans_iou_bev)


if __name__ == "__main__":
    unittest.main()
