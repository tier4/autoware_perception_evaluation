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
import unittest
from test.util.dummy_object import make_dummy_data
from typing import TYPE_CHECKING, List, Tuple

from perception_eval.evaluation.matching.object_matching import PlaneDistanceMatching
from perception_eval.util.debug import get_objects_with_difference

if TYPE_CHECKING:
    from perception_eval.common.object import DynamicObject


class TestPlaneDistanceMatching(unittest.TestCase):
    def setUp(self):
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()

    def test_plane_distance_matching(self):
        """[summary]
        Test calculating plane distance for use case evaluation.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if plane_distance and ans_plane_distance are equal.
        """
        patterns: List[Tuple[float, List[float]]] = [
            (0.0, [0.0, 0.0, 0.0, 0.0]),
            (1.0, [1.0, 1.0, 1.0, 1.0]),
            (
                2.0,
                [2.0, 2.0, math.sqrt((1.0 + 5.0) / 2.0), math.sqrt((1.0 + 5.0) / 2.0)],
            ),
        ]
        fixed_dummy_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
            ground_truth_objects=self.dummy_ground_truth_objects,
            # Shift all objects a bit to exclude the effect of numerical error.
            diff_distance=(0.01, 0.0, 0.0),
            diff_yaw=0,
        )
        for diff_distance, ans_plane_distance_list in patterns:
            with self.subTest("Test get_uc_plane_distance."):
                diff_distance_dummy_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=fixed_dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                for estimated_object, ground_truth_object, ans_plane_distance in zip(
                    diff_distance_dummy_ground_truth_objects,
                    fixed_dummy_ground_truth_objects,
                    ans_plane_distance_list,
                ):
                    plane_distance = PlaneDistanceMatching(
                        estimated_object,
                        ground_truth_object,
                    )
                    self.assertAlmostEqual(
                        plane_distance.value,
                        ans_plane_distance,
                    )

    def test_dummy_objects_plane_distance_matching(self):
        """[summary]
        Test calculating plane distance for use case evaluation.

        test objects:
            dummy_estimated_objects and dummy_ground_truth_objects

        test patterns:
            Check if plane_distance and ans_plane_distance are equal.
        """
        # dummy_estimated_objects[0] (CAR) and dummy_ground_truth_objects[0] (CAR):
        #
        # dummy_estimated_objects[1] (BICYCLE) and dummy_ground_truth_objects[1] (BICYCLE):
        #
        # dummy_estimated_objects[2] (PEDESTRIAN) and dummy_ground_truth_objects[2] (CAR):

        ans_plane_distance_list = [0.3535533905932738, 0.3535533905932738, 0.0]
        for estimated_object, ground_truth_object, ans_plane_distance in zip(
            self.dummy_estimated_objects,
            self.dummy_ground_truth_objects,
            ans_plane_distance_list,
        ):
            plane_distance = PlaneDistanceMatching(
                estimated_object,
                ground_truth_object,
            )
            self.assertAlmostEqual(
                plane_distance.value,
                ans_plane_distance,
            )


if __name__ == "__main__":
    unittest.main()
