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

from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple
import unittest

from perception_eval.common.object import DynamicObject
from perception_eval.evaluation.matching.object_matching import _get_height_intersection
from perception_eval.evaluation.matching.object_matching import _get_volume_intersection
from perception_eval.evaluation.matching.object_matching import IOU3dMatching
from perception_eval.util.debug import get_objects_with_difference


class TestIou3dMatching(unittest.TestCase):
    def setUp(self):
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()

    def test_get_height_intersection(self):
        """[summary]
        Test getting the height at intersection.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if height_intersection and ans_height_intersection
            are equal.
        """
        # patterns: (diff_distance, List[ans_height_intersection])
        patterns: List[Tuple[float, float]] = [(0.0, 1.0), (-0.5, 0.5), (-1.0, 0.0)]
        for diff_distance, ans_height_intersection in patterns:
            with self.subTest("Test get_height_intersection."):
                diff_distance_dummy_ground_truth_objects: List[DynamicObject] = (
                    get_objects_with_difference(
                        ground_truth_objects=self.dummy_ground_truth_objects,
                        diff_distance=(0.0, 0.0, diff_distance),
                        diff_yaw=0,
                    )
                )
                for estimated_object, ground_truth_object in zip(
                    diff_distance_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    height_intersection = _get_height_intersection(
                        estimated_object,
                        ground_truth_object,
                    )
                    self.assertAlmostEqual(height_intersection, ans_height_intersection)

    def test_volume_intersection(self):
        """[summary]
        Test getting the volume at intersection.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if intersection and ans_intersection are equal.
        """
        # patterns: (diff_distance, List[ans_intersection])
        patterns: List[Tuple[float, float]] = [(0.0, 1.0), (0.5, 0.125), (1.5, 0.0)]
        for diff_distance, ans_intersection in patterns:
            with self.subTest("Test get_intersection."):
                diff_distance_dummy_ground_truth_objects: List[DynamicObject] = (
                    get_objects_with_difference(
                        ground_truth_objects=self.dummy_ground_truth_objects,
                        diff_distance=(diff_distance, diff_distance, diff_distance),
                        diff_yaw=0,
                    )
                )
                for estimated_object, ground_truth_object in zip(
                    diff_distance_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    intersection = _get_volume_intersection(estimated_object, ground_truth_object)
                    self.assertAlmostEqual(intersection, ans_intersection)

    def test_get_iou_3d_matching(self):
        """[summary]
        Test calculating 3d IoU.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if iou_3d and ans_iou_3d are equal.
        """
        # patterns: (diff_distance, List[ans_iou_3d])
        patterns: List[Tuple[float, float]] = [(0.0, 1.0), (-0.5, 1.0 / 3.0), (-1.0, 0.0)]
        for diff_distance, ans_iou_3d in patterns:
            with self.subTest("Test diff_x get_iou_3d."):
                diff_x_dummy_ground_truth_objects: List[DynamicObject] = (
                    get_objects_with_difference(
                        ground_truth_objects=self.dummy_ground_truth_objects,
                        diff_distance=(diff_distance, 0.0, 0.0),
                        diff_yaw=0,
                    )
                )
                for estimated_object, ground_truth_object in zip(
                    diff_x_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    iou_3d = IOU3dMatching(estimated_object, ground_truth_object)
                    self.assertAlmostEqual(iou_3d.value, ans_iou_3d)

            with self.subTest("Test diff_z get_iou_3d."):
                diff_z_dummy_ground_truth_objects: List[DynamicObject] = (
                    get_objects_with_difference(
                        ground_truth_objects=self.dummy_ground_truth_objects,
                        diff_distance=(0.0, 0.0, diff_distance),
                        diff_yaw=0,
                    )
                )
                for estimated_object, ground_truth_object in zip(
                    diff_z_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    iou_3d = IOU3dMatching(estimated_object, ground_truth_object)
                    self.assertAlmostEqual(iou_3d.value, ans_iou_3d)


if __name__ == "__main__":
    unittest.main()
