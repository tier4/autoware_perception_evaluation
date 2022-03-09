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
            (
                2.0,
                [2.0, 2.0, math.sqrt((1.0 + 5.0) / 2.0),
                 math.sqrt((1.0 + 5.0) / 2.0)],
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

    def test_dummy_objects_plane_distance_matching(self):
        """[summary]
        Test calculating plane distance for use case evaluation.

        test objects:
            dummy_predicted_objects and dummy_ground_truth_objects

        test patterns:
            Check if plane_distance and ans_plane_distance are equal.
        """
        # dummy_predicted_objects[0] (CAR) and dummy_ground_truth_objects[0] (CAR):
        #   sorted pr_corner_points[:2] = [(0.25, 0.25, 1.0), (0.25, 1.75, 1.0)]
        #   sorted gt_corner_points[:2] = [(0.5, 0.5, 1.0), (1.5, 0.5, 1.0)]
        #   plane_distance = 1.2747548783981963
        #
        # dummy_predicted_objects[1] (BICYCLE) and dummy_ground_truth_objects[1] (BICYCLE):
        #   sorted pr_corner_points[:2] = [(0.75, -0.75, 1.0), (1.25, -0.75, 1.0)]
        #   sorted gt_corner_points[:2] = [(0.5, -0.5, 1.0), (1.5, -0.5, 1.0)]
        #   plane_distance = 0.3535533905932738
        #
        # dummy_predicted_objects[2] (PEDESTRIAN) and dummy_ground_truth_objects[2] (CAR):
        #   sorted pr_corner_points[:2] = [(-0.5, 0.5, 1.0), (-0.5, 1.5, 1.0)]
        #   sorted gt_corner_points[:2] = [(-0.5, 0.5, 1.0), (-0.5, 1.5, 1.0)]
        #   plane_distance = 0.0

        # patterns: List[ans_plane_distance]
        ans_plane_distance_list = [1.2747548783981963, 0.3535533905932738, 0.0]
        for predicted_object, ground_truth_object, ans_plane_distance in zip(
            self.dummy_predicted_objects,
            self.dummy_ground_truth_objects,
            ans_plane_distance_list
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
