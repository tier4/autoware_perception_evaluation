from test.util.dummy_object import make_dummy_data
from typing import Dict
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
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.util.debug import get_objects_with_difference


class TestObjectsFilter(unittest.TestCase):
    def setUp(self):
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()

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
        self.min_point_numbers: List[int] = [0, 1, 10, 0]

    def test_filter_object_results(self):
        """[summary]
        Test filtering DynamicObjectWithPerceptionResult to filter ground truth objects.

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
            # Given 2.5 diff_distance for one axis, all object_results beyond max_pos_distance.
            (2.5, []),
        ]
        frame_id: str = "base_link"
        for n, (diff_distance, ans_idx) in enumerate(patterns):
            with self.subTest(f"Test filtered_object_results: {n + 1}"):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                filtered_object_results = filter_object_results(
                    frame_id,
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
            0 : AutowareLabel.CAR, 1 : AutowareLabel.BICYCLE, 2 : AutowareLabel.PEDESTRIAN, 3 : AutowareLabel.MOTORBIKE

        test patterns:
            Given diff_distance, check if filtered_objects and ans_objects are the same.
        """
        # patterns: (diff_distance, List[ans_idx], point_number_dict: Dict[str, int])
        patterns: List[Tuple[float, List[int], Dict[str, int]]] = [
            # Given no diff_distance, no diff_distance_dummy_ground_truth_objects are filtered out.
            (0.0, [0, 1, 2, 3], {}),
            # Given no diff_distance and no point clouds, 2 diff_distance_dummy_ground_truth_objects are filtered out.
            (0.0, [0, 3], {'0': 0, '1': 0, '2': 0, '3': 0}),
            # Given no diff_distance and 9 point clouds, 2 diff_distance_dummy_ground_truth_objects are filtered out.
            (0.0, [0, 1, 3], {'0': 9, '1': 9, '2': 9, '3': 9}),
            # Given 1.5 diff_distance for one axis, two objects beyond max_pos_distance.
            (1.5, [0, 1], {}),
            # Given 1.5 diff_distance and 1 point cloud, 2 diff_distance_dummy_ground_truth_objects are filtered out.
            (1.5, [0, 1], {'0': 1, '1': 1, '2': 1, '3': 1}),
            # Given 2.5 diff_distance for one axis, all objects beyond max_pos_distance.
            (2.5, [], {}),
        ]
        frame_id: str = "base_link"
        for n, (diff_distance, ans_idx, point_number_dict) in enumerate(patterns):
            with self.subTest(f"Test filter_ground_truth_objects: {n + 1}"):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(-diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )

                # make dummy data with different point cloud numbers
                for idx, pointcloud_num in point_number_dict.items():
                    diff_distance_dummy_ground_truth_objects[
                        int(idx)
                    ].pointcloud_num = pointcloud_num

                filtered_objects = filter_ground_truth_objects(
                    frame_id,
                    diff_distance_dummy_ground_truth_objects,
                    self.target_labels,
                    self.max_x_position_list,
                    self.max_y_position_list,
                    self.max_pos_distance_list,
                    self.min_pos_distance_list,
                    self.min_point_numbers,
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
        # patterns: (diff_distance, List[ans_tp_idx], label_change_dict: Dict[str, str])
        patterns: List[Tuple[float, List[int], Dict[str, str]]] = [
            # Given no diff_distance, all estimated_objects are tp.
            (0.0, [0, 1, 2, 3], {}),
            # Given no diff_distance and 2 labels changed, 2 estimated_objects are tp.
            (0.0, [1, 2], {'0': AutowareLabel.UNKNOWN, '3': AutowareLabel.ANIMAL}),
            # Given 1.5 diff_distance for one axis, all estimated_objects are tp.
            (1.5, [0, 1, 2, 3], {}),
            # Given 1.5 diff_distance for one axis and 1 labels changed, 3 estimated_objects are tp.
            (1.5, [0, 1, 3], {'2': AutowareLabel.UNKNOWN}),
            # TODO(Shin-kyoto): 以下のtestも通る必要あり．現在，ground truth一つに対しestimated objectが複数紐づくため通らなくなっている
            # Given 1.5 diff_distance for one axis and 1 labels changed, 3 estimated_objects are tp.
            # (1.5, [0, 1, 3], {'2' : AutowareLabel.CAR}),
            # Given 2.5 diff_distance for one axis, all estimated_objects are fp
            # since they are beyond matching_threshold.
            (2.5, [], {}),
        ]

        for n, (diff_distance, ans_tp_idx, label_change_dict) in enumerate(patterns):
            with self.subTest(f"Test divide_tp_fp_objects: {n + 1}"):

                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )

                # make dummy data with different label
                for idx, label in label_change_dict.items():
                    diff_distance_dummy_ground_truth_objects[int(idx)].semantic_label = label

                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
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
        Test getting FN (False Negative) objects from ground truth objects by using object result.

        test objects:
            4 ground_truth_objects with object_results made from dummy_ground_truth_objects
            with diff_distance

        test patterns:
            Given diff_distance, check if fn_objects and ans_fn_objects are the same.
        """
        # patterns: (x_diff, y_diff, List[ans_fn_idx])
        patterns: List[Tuple[float, float, List[int]]] = [
            # Given difference (0.0, 0.0), there are no fn.
            (0.0, 0.0, []),
            # Given difference (0.5, 0.0), there are no fn.
            (0.5, 0.0, []),
            # Given difference (0.0, -0.5), there are no fn.
            (0.0, -0.5, []),
            # Given difference (1.5, 0.0), there are no fn.
            (1.5, 0.0, []),
            # Given difference (2.0, 0.0), all ground_truth_objects are fn
            (2.0, 0.0, [0, 1, 2, 3]),
            # Given difference (2.5, 0.0), all ground_truth_objects are fn
            (2.5, 0.0, [0, 1, 2, 3]),
            # Given difference (2.5, 2.5), all ground_truth_objects are fn
            (2.5, 2.5, [0, 1, 2, 3]),
        ]

        for n, (x_diff, y_diff, ans_fn_idx) in enumerate(patterns):
            with self.subTest(f"Test get_fn_objects: {n + 1}"):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(x_diff, y_diff, 0.0),
                    diff_yaw=0,
                )
                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )

                tp_results, _ = divide_tp_fp_objects(
                    object_results,
                    self.target_labels,
                    self.matching_mode,
                    self.matching_threshold_list,
                    self.confidence_threshold_list,
                )

                fn_objects = get_fn_objects(
                    self.dummy_ground_truth_objects,
                    object_results,
                    tp_results,
                )

                ans_fn_objects = [
                    x for idx, x in enumerate(self.dummy_ground_truth_objects) if idx in ans_fn_idx
                ]
                self.assertEqual(fn_objects, ans_fn_objects)

    def test_get_fn_objects_for_different_label(self):
        """[summary]
        Test getting FN (False Negative) objects from ground truth objects with different label
        by using object result.

        test objects:
            4 ground_truth_objects with object_results made from dummy_ground_truth_objects
            with diff_distance

        test patterns:
            Given diff_distance, check if fn_objects and ans_fn_objects are the same.
        """

        # patterns: (x_diff, y_diff, List[ans_fn_idx], label_change_dict: Dict[str, str]))
        patterns: List[Tuple[float, float, List[int], Dict[str, str]]] = [
            # Given difference (0.0, 0.0), there are no fn.
            (0.0, 0.0, [], {}),
            # Given no diff_distance and 2 labels changed, 2 estimated_objects are fp.
            (0.0, 0.0, [0, 3], {'0': AutowareLabel.UNKNOWN, '3': AutowareLabel.ANIMAL}),
            # TODO(Shin-kyoto): 以下のtestも通る必要あり．現在，ground truth一つに対しestimated objectが複数紐づくため通らなくなっている
            # Given no diff_distance and 2 labels changed, 2 estimated_objects are fp.
            # (0.0, 0.0, [0, 3], {'0' : AutowareLabel.CAR, '3' : AutowareLabel.ANIMAL}),
            # Given difference (0.5, 0.0), there are no fn.
            (0.5, 0.0, [], {}),
            # Given difference (0.0, -0.5), there are no fn.
            (0.0, -0.5, [], {}),
            # Given difference (1.5, 0.0), there are no fn.
            (1.5, 0.0, [], {}),
            # Given difference (1.5, 0.0) and 1 labels changed, 1 estimated_objects are fp.
            (1.5, 0.0, [1], {'1': AutowareLabel.UNKNOWN}),
            # Given difference (2.5, 0.0), all ground_truth_objects are fn
            (2.5, 0.0, [0, 1, 2, 3], {}),
            # Given difference (2.5, 2.5), all ground_truth_objects are fn
            (2.5, 2.5, [0, 1, 2, 3], {}),
        ]

        for n, (x_diff, y_diff, ans_fn_idx, label_change_dict) in enumerate(patterns):
            with self.subTest(f"Test get_fn_objects: {n + 1}"):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(x_diff, y_diff, 0.0),
                    diff_yaw=0,
                )

                # make dummy data with different label
                for idx, label in label_change_dict.items():
                    diff_distance_dummy_ground_truth_objects[int(idx)].semantic_label = label

                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                tp_results, _ = divide_tp_fp_objects(
                    object_results,
                    self.target_labels,
                    self.matching_mode,
                    self.matching_threshold_list,
                    self.confidence_threshold_list,
                )
                fn_objects = get_fn_objects(
                    self.dummy_ground_truth_objects,
                    object_results,
                    tp_results,
                )

                ans_fn_objects = [
                    x for idx, x in enumerate(self.dummy_ground_truth_objects) if idx in ans_fn_idx
                ]
                self.assertEqual(fn_objects, ans_fn_objects)


if __name__ == "__main__":
    unittest.main()
