from test.util.dummy_object import make_dummy_data
from typing import List
import unittest

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithResult
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.util.debug import get_objects_with_difference


class TestIoU3d(unittest.TestCase):
    def setUp(self):
        self.dummy_predicted_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_predicted_objects, self.dummy_ground_truth_objects = make_dummy_data()

    def test_iou3d_diff_z_distance(self):
        """[summary]
        Test whether 3d IoU is correct for different distance.

        test condition:
                the same size(1.0, 1.0, 1.0) of dummy_ground_truth_objects vs dummy_ground_truth_objects with diff_distance
        test patterns:
                Given diff_distance, check if iou_bev is correct.
        """
        # patterns: (diff_distance, ans_iou_bev)
        patterns = [
            # Given no diff_distance, iou is 1.0.
            (0.0, 1.0),
            # Given diff_distance is 0.5 for one axis, iou is 0.5 / 1.5
            # since ground_truth_objects and predicted_objects are half overlapping.
            (-0.5, 0.5 / 1.5),
            # Given diff_distance is 1.0 for one axis, iou is 0.0
            # since ground_truth_objects and predicted_objects are no overlapping.
            (-1.0, 0.0),
        ]
        for diff_distance, ans_iou_bev in patterns:
            with self.subTest("diff_yaw make map"):
                diff_yaw_dummy_ground_truth_objects = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, diff_distance),
                    diff_yaw=0.0,
                )

                object_results: List[
                    DynamicObjectWithResult
                ] = PerceptionFrameResult.get_object_results(
                    predicted_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                for object_result in object_results:
                    self.assertAlmostEqual(object_result.iou_3d.value, ans_iou_bev)


if __name__ == "__main__":
    unittest.main()
