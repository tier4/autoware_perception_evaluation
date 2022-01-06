import math
from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple
import unittest

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import PlaneDistanceMatching
from awml_evaluation.util.debug import get_objects_with_difference


class TestPlaneDistanceMatching(unittest.TestCase):
    def setUp(self):
        self.dummy_predicted_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_predicted_objects, self.dummy_ground_truth_objects = make_dummy_data()

    def test_plane_distance_matching(self):
        """[summary]
        Test calculating plane distance for use case evaluation.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if plane_distance and ans_plane_distance are equal.
        """
        # patterns: (diff_distance, List[ans_plane_distance])
        patterns: List[Tuple[float, List[float]]] = [
            (0.0, [0.0, 0.0, 0.0, 0.0]),
            (1.0, [1.0, 1.0, 1.0, 1.0]),
            # (2.0, [2.0, 2.0, (1.0+math.sqrt(5.0))/2.0, (1.0+math.sqrt(5.0))/2.0]) # previous version
            (
                2.0,
                [2.0, 2.0, math.sqrt((1.0 + 5.0) / 2.0), math.sqrt((1.0 + 5.0) / 2.0)],
            ),  # new version
        ]
        fixed_dummy_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
            ground_truth_objects=self.dummy_ground_truth_objects,
            # Shift all objects a bit to exclude the effect of numerical error.
            diff_distance=(0.01, 0.0, 0.0),
            diff_yaw=0,
        )
        for diff_distance, ans_plane_distance_list in patterns:
            with self.subTest("Test get_uc_plane_distance."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=fixed_dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                for predicted_object, ground_truth_object, ans_plane_distance in zip(
                    diff_distance_dummy_ground_truth_objects,
                    fixed_dummy_ground_truth_objects,
                    ans_plane_distance_list,
                ):
                    plane_distance = PlaneDistanceMatching(
                        predicted_object,
                        ground_truth_object,
                    )
                    self.assertAlmostEqual(
                        plane_distance.value,
                        ans_plane_distance,
                    )


if __name__ == "__main__":
    unittest.main()
