from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple
import unittest

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.matching.objects_filter import divide_tp_fp_objects
from awml_evaluation.evaluation.matching.objects_filter import filter_ground_truth_objects
from awml_evaluation.evaluation.matching.objects_filter import filter_object_results
from awml_evaluation.evaluation.matching.objects_filter import get_fn_objects
from awml_evaluation.evaluation.result.frame_result import FrameResult
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithResult
from awml_evaluation.util.debug import get_objects_with_difference


class TestObjectsFilter(unittest.TestCase):
    def setUp(self):
        self.dummy_predicted_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_predicted_objects, self.dummy_ground_truth_objects = make_dummy_data()

        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]
        self.max_x_position_list: List[float] = [2.0, 2.0, 2.0, 2.0]
        self.max_y_position_list: List[float] = [2.0, 2.0, 2.0, 2.0]
        self.max_pos_distance_list: List[float] = [1.5, 1.5, 1.5, 1.5]
        self.min_pos_distance_list: List[float] = [0.3, 0.3, 0.3, 0.3]
        self.matching_mode: MatchingMode = MatchingMode.CENTERDISTANCE
        self.matching_threshold_list: List[float] = [2.0, 2.0, 2.0, 2.0]
        self.confidence_threshold_list: List[float] = [0.5, 0.5, 0.5, 0.5]

    def test_filter_object_results(self):
        """[summary]
        Test filtering DynamicObjectWithResult to filter ground truth objects.

        test objects:
            4 object_results made from dummy_ground_truth_objects with diff_distance

        test patterns:
            Given diff_distance, check if filtered_object_results and ans_object_results
            are the same.
        """
        # patterns: (diff_distance, List[ans_idx])
        patterns: List[Tuple[float, List[int]]] = [
            # Given no diff_distance, no object_results are filtered out.
            (0.0, [0, 1, 2, 3]),
            # Given 1.5 diff_distance for one axis, two object_results beyond max_pos_distance.
            (1.5, [2, 3]),
            # Given 2.5 diff_distance for one axis, all object_resultss beyond max_pos_distance.
            (2.5, []),
        ]
        for diff_distance, ans_idx in patterns:
            with self.subTest("Test filtered_object_results."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                object_results: List[DynamicObjectWithResult] = FrameResult.get_object_results(
                    predicted_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                filtered_object_results = filter_object_results(
                    object_results,
                    self.target_labels,
                    self.max_x_position_list,
                    self.max_y_position_list,
                    self.max_pos_distance_list,
                    self.min_pos_distance_list,
                )
                ans_object_results = [x for idx, x in enumerate(object_results) if idx in ans_idx]
                self.assertEqual(filtered_object_results, ans_object_results)

    def test_filter_ground_truth_objects(self):
        """[summary]
        Test filtering DynamicObject to filter ground truth objects.

        test objects:
            4 ground_truth_objects made from dummy_ground_truth_objects with diff_distance

        test patterns:
            Given diff_distance, check if filtered_objects and ans_objects are the same.
        """
        # patterns: (diff_distance, List[ans_idx])
        patterns: List[Tuple[float, List[int]]] = [
            # Given no diff_distance, no diff_distance_dummy_ground_truth_objects are filtered out.
            (0.0, [0, 1, 2, 3]),
            # Given 1.5 diff_distance for one axis, two objects beyond max_pos_distance.
            (1.5, [0, 1]),
            # Given 2.5 diff_distance for one axis, all objects beyond max_pos_distance.
            (2.5, []),
        ]
        for diff_distance, ans_idx in patterns:
            with self.subTest("Test filter_ground_truth_objects."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(-diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                filtered_objects = filter_ground_truth_objects(
                    diff_distance_dummy_ground_truth_objects,
                    self.target_labels,
                    self.max_x_position_list,
                    self.max_y_position_list,
                    self.max_pos_distance_list,
                    self.min_pos_distance_list,
                )
                ans_objects = [
                    x
                    for idx, x in enumerate(diff_distance_dummy_ground_truth_objects)
                    if idx in ans_idx
                ]
                self.assertEqual(filtered_objects, ans_objects)

    def test_divide_tp_fp_objects(self):
        """[summary]
        Test dividing TP (True Positive) objects and FP (False Positive) objects
        from Prediction condition positive objects.

        test objects:
            4 object_results made from dummy_ground_truth_objects with diff_distance

        test patterns:
            Given diff_distance, check if tp_results and ans_tp_results
            (fp_results and ans_fp_results) are the same.
        """
        # patterns: (diff_distance, List[ans_tp_idx])
        patterns: List[Tuple[float, List[int]]] = [
            # Given no diff_distance, all predicted_objects are tp.
            (0.0, [0, 1, 2, 3]),
            # Given 1.5 diff_distance for one axis, two predicted_objects are tp
            # and the other predicted_objects are fp since they have wrong target_labels.
            (1.5, [0, 1]),
            # Given 2.5 diff_distance for one axis, all predicted_objects are fp
            # since they are beyond matching_threshold.
            (2.5, []),
        ]
        for diff_distance, ans_tp_idx in patterns:
            with self.subTest("Test divide_tp_fp_objects."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                object_results: List[DynamicObjectWithResult] = FrameResult.get_object_results(
                    predicted_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                tp_results, fp_results = divide_tp_fp_objects(
                    object_results,
                    self.target_labels,
                    self.matching_mode,
                    self.matching_threshold_list,
                    self.confidence_threshold_list,
                )

                ans_tp_results = [x for idx, x in enumerate(object_results) if idx in ans_tp_idx]
                ans_fp_results = [
                    x for idx, x in enumerate(object_results) if idx not in ans_tp_idx
                ]
                self.assertEqual(tp_results, ans_tp_results)
                self.assertEqual(fp_results, ans_fp_results)

    def test_get_fn_objects(self):
        """[summary]
        Test getting FN (False Negative) objects from ground truth objects
        by using object result.

        test objects:
            4 ground_truth_objects with object_results made from dummy_ground_truth_objects
            with diff_distance

        test patterns:
            Given diff_distance, check if fn_objects and ans_fn_objects are the same.
        """
        # patterns: (diff_distance, List[ans_fn_idx])
        patterns: List[Tuple[float, List[int]]] = [
            # Given no diff_distance, there are no fn.
            (0.0, []),
            # Given 1.5 diff_distance for one axis, two ground_truth_objects are fn
            # since they don't match any predicted_objects in wrong target_labels.
            (1.5, [2, 3]),
            # Given 2.5 diff_distance for one axis, all ground_truth_objects are fn
            # since they don't match any predicted_objects beyond matching_threshold.
            (2.5, [0, 1, 2, 3]),
        ]

        for diff_distance, ans_fn_idx in patterns:
            with self.subTest("Test get_fn_objects."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                object_results: List[DynamicObjectWithResult] = FrameResult.get_object_results(
                    predicted_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                fn_objects = get_fn_objects(
                    self.dummy_ground_truth_objects,
                    object_results,
                    self.target_labels,
                    self.matching_mode,
                    self.matching_threshold_list,
                )

                ans_fn_objects = [
                    x for idx, x in enumerate(self.dummy_ground_truth_objects) if idx in ans_fn_idx
                ]
                self.assertEqual(fn_objects, ans_fn_objects)


if __name__ == "__main__":
    unittest.main()
