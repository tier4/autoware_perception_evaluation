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
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import unittest

import numpy as np
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import SemanticLabel
from perception_eval.matching import MatchingMode
from perception_eval.matching import MatchingPolicy
from perception_eval.matching.objects_filter import divide_objects
from perception_eval.matching.objects_filter import divide_objects_to_num
from perception_eval.matching.objects_filter import divide_tp_fp_objects
from perception_eval.matching.objects_filter import filter_object_results
from perception_eval.matching.objects_filter import filter_objects
from perception_eval.matching.objects_filter import get_fn_objects
from perception_eval.matching.objects_filter import get_negative_objects
from perception_eval.matching.objects_filter import get_positive_objects
from perception_eval.object import DynamicObject
from perception_eval.result import DynamicObjectWithPerceptionResult
from perception_eval.result import get_object_results
from perception_eval.util.debug import get_objects_with_difference


class TestObjectsFilter(unittest.TestCase):
    def setUp(self):
        _, self.dummy_ground_truth_objects = make_dummy_data()

        self.evaluation_task = EvaluationTask.DETECTION

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

            NOTE:
                - Estimated object is only matched with GT that has same label.
                - The estimations & GTs are following (number represents the index)
                    Estimation = 4
                        (0): CAR, (1): BICYCLE, (2): PEDESTRIAN, (3): MOTORBIKE
                    GT = 4
                        (0): CAR, (1): BICYCLE, (2): PEDESTRIAN, (3): MOTORBIKE
        """
        # patterns: (diff_distance, List[ans_idx])
        patterns: List[Tuple[float, np.ndarray]] = [
            # (1)
            # Pair: (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2]), (Est[3], GT[3])
            # Given no diff_distance, no object_results are filtered out.
            (0.0, np.array([[0, 0], [1, 1], [2, 2], [3, 3]])),
            # (2)
            # Pair: (Est[2], GT[2]), (Est[3], GT[3])
            # Given 1.5 diff_distance for one axis, two object_results beyond max_pos_distance.
            (1.5, np.array([[2, 2], [3, 3]])),
            # (3)
            # Pair:
            # Given 2.5 diff_distance for one axis, all object_results beyond max_pos_distance.
            (2.5, []),
        ]
        for n, (diff_distance, ans_pair_indices) in enumerate(patterns):
            with self.subTest(f"Test filtered_object_results: {n + 1}"):
                estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=estimated_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                filtered_object_results = filter_object_results(
                    object_results=object_results,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    max_distance_list=self.max_pos_distance_list,
                    min_distance_list=self.min_pos_distance_list,
                )
                self.assertEqual(len(filtered_object_results), len(ans_pair_indices))
                if len(ans_pair_indices) == 0:
                    continue
                for i, object_result_ in enumerate(filtered_object_results):
                    self.assertIn(
                        object_result_.estimated_object,
                        estimated_objects,
                        f"Unexpected estimated object at {i}",
                    )
                    est_idx: int = estimated_objects.index(object_result_.estimated_object)
                    gt_idx: int = ans_pair_indices[ans_pair_indices[:, 0] == est_idx][0, 1]
                    self.assertEqual(
                        object_result_.ground_truth_object,
                        self.dummy_ground_truth_objects[gt_idx],
                        f"Unexpected estimated object at {i}",
                    )

    def test_filter_objects(self):
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
            (0.0, [0, 3], {"0": 0, "1": 0, "2": 0, "3": 0}),
            # Given no diff_distance and 9 point clouds, 2 diff_distance_dummy_ground_truth_objects are filtered out.
            (0.0, [0, 1, 3], {"0": 9, "1": 9, "2": 9, "3": 9}),
            # Given 1.5 diff_distance for one axis, two objects beyond max_pos_distance.
            (1.5, [0, 1], {}),
            # Given 1.5 diff_distance and 1 point cloud, 2 diff_distance_dummy_ground_truth_objects are filtered out.
            (1.5, [0, 1], {"0": 1, "1": 1, "2": 1, "3": 1}),
            # Given 2.5 diff_distance for one axis, all objects beyond max_pos_distance.
            (2.5, [], {}),
        ]
        for n, (diff_distance, ans_idx, point_number_dict) in enumerate(patterns):
            with self.subTest(f"Test filter_objects: {n + 1}"):
                ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(-diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )

                # make dummy data with different point cloud numbers
                for idx, pointcloud_num in point_number_dict.items():
                    ground_truth_objects[int(idx)].pointcloud_num = pointcloud_num

                filtered_objects = filter_objects(
                    objects=ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    max_distance_list=self.max_pos_distance_list,
                    min_distance_list=self.min_pos_distance_list,
                    min_point_numbers=self.min_point_numbers,
                )
                ans_objects = [x for idx, x in enumerate(ground_truth_objects) if idx in ans_idx]
                self.assertEqual(filtered_objects, ans_objects)

    def test_get_positive_objects(self) -> None:
        """Test `get_positive_objects(...)` function."""
        # patterns: (diff_distance, List[ans_tp_idx], label_change_dict: Dict[str, str])
        patterns: List[Tuple[float, np.ndarray, np.ndarray, Dict[int, AutowareLabel]]] = [
            # (1)
            # TP: (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2]), (Est[3], GT[3])
            # FP:
            # Given no diff_distance, all estimated_objects are tp.
            (0.0, np.array([(0, 0), (1, 1), (2, 2), (3, 3)]), np.array([]), {}),
            # (2)
            # TP: (Est[1], GT[1]), (Est[2], GT[2])
            # FP: (Est[0], None), (Est[3], None)
            # Given no diff_distance and 2 labels changed, 2 estimated_objects are tp.
            (
                0.0,
                np.array([(0, 0), (1, 1), (2, 2)]),
                np.array([(3, None)]),
                {
                    0: SemanticLabel(AutowareLabel.UNKNOWN, "unknown", []),
                    3: SemanticLabel(AutowareLabel.ANIMAL, "animal", []),
                },
            ),
            # (3)
            # TP: (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2]), (Est[3], GT[3])
            # FP:
            # Given 1.5 diff_distance for one axis, all estimated_objects are tp.
            (
                1.5,
                np.array([(0, 0), (1, 1), (2, 2), (3, 3)]),
                np.array([]),
                {},
            ),
            # (4)
            # TP: (Est[0], GT[0]), (Est[1], GT[1]), (Est[3], GT[3])
            # FP: (Est[2], None)
            # Given 1.5 diff_distance for one axis and 1 labels changed, 3 estimated_objects are tp.
            (
                1.5,
                np.array([(2, 0), (1, 1), (3, 3)]),
                np.array([(0, None)]),
                {2: SemanticLabel(AutowareLabel.UNKNOWN, "unknown", [])},
            ),
            # (5)
            # TP: (Est[2], GT[0]), (Est[1], GT[1]), (Est[3], GT[3])
            # FP: (Est[0], None)
            # Given 1.5 diff_distance for one axis and 1 labels changed, 3 estimated_objects are tp.
            (
                1.5,
                np.array([(2, 0), (1, 1), (3, 3)]),
                np.array([(0, None)]),
                {2: SemanticLabel(AutowareLabel.CAR, "car", [])},
            ),
            # (6)
            # TP:
            # FP: (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2]), (Est[3], GT[3])
            # Given 2.5 diff_distance for one axis, all estimated_objects are fp
            # since they are beyond matching_threshold.
            (
                2.5,
                np.array([]),
                np.array([(0, 0), (1, 1), (2, 2), (3, 3)]),
                {},
            ),
        ]

        for n, (diff_distance, expect_tp_indices, expect_fp_indices, label_changes) in enumerate(patterns):
            with self.subTest(f"Test `get_positive_objects()`: {n + 1}"):
                estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0, 0),
                    diff_yaw=0,
                )

                # make dummy data with different label
                for idx, label in label_changes.items():
                    estimated_objects[idx].semantic_label = label

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=estimated_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    matching_policy=MatchingPolicy(label_policy="ALLOW_UNKNOWN"),
                )
                tp_results, fp_results = get_positive_objects(
                    object_results,
                    self.target_labels,
                    self.matching_mode,
                    self.matching_threshold_list,
                )
                # TP
                self.assertEqual(
                    len(tp_results),
                    len(expect_tp_indices),
                    f"Number of TP elements are not same, out: {len(tp_results)}, expect: {len(expect_tp_indices)}",
                )
                for tp_result in tp_results:
                    tp_est_idx: int = estimated_objects.index(tp_result.estimated_object)
                    tp_gt_idx: int = expect_tp_indices[expect_tp_indices[:, 0] == tp_est_idx][0, 1]
                    self.assertEqual(
                        tp_result.ground_truth_object,
                        self.dummy_ground_truth_objects[tp_gt_idx],
                    )

                # FP
                self.assertEqual(
                    len(fp_results),
                    len(expect_fp_indices),
                    f"Number of FP elements are not same, but out: {len(fp_results)}, expect: {len(expect_fp_indices)}",
                )
                for fp_result in fp_results:
                    fp_est_idx: int = estimated_objects.index(fp_result.estimated_object)
                    fp_gt_idx: Optional[int] = expect_fp_indices[expect_fp_indices[:, 0] == fp_est_idx][0, 1]
                    if fp_gt_idx is None:
                        self.assertIsNone(fp_result.ground_truth_object)
                    else:
                        self.assertEqual(
                            fp_result.ground_truth_object,
                            self.dummy_ground_truth_objects[fp_gt_idx],
                        )

    def test_get_negative_objects(self) -> None:
        """Test `get_negative_objects(...)` function."""
        # patterns: (x_diff, y_diff, List[ans_fn_idx])
        patterns: List[Tuple[float, float, List[int]]] = [
            # Given difference (0.0, 0.0), there are no fn.
            (0.0, 0.0, [], []),
            # Given difference (0.5, 0.0), there are no fn.
            (0.5, 0.0, [], []),
            # Given difference (0.0, -0.5), there are no fn.
            (0.0, -0.5, [], []),
            # Given difference (1.5, 0.0), there are no fn.
            (1.5, 0.0, [], []),
            # Given difference (2.0, 0.0), all ground_truth_objects are fn
            (2.0, 0.0, [], [0, 1, 2, 3]),
            # Given difference (2.5, 0.0), all ground_truth_objects are fn
            (2.5, 0.0, [], [0, 1, 2, 3]),
            # Given difference (2.5, 2.5), all ground_truth_objects are fn
            (2.5, 2.5, [], [0, 1, 2, 3]),
        ]

        for n, (x_diff, y_diff, expect_tn_indices, expect_fn_indices) in enumerate(patterns):
            with self.subTest(f"Test `get_negative_objects()`: {n+1}"):
                # expect TN/FN objects
                expect_tn_objects = [
                    obj for idx, obj in enumerate(self.dummy_ground_truth_objects) if idx in expect_tn_indices
                ]
                expect_fn_objects = [
                    obj for idx, obj in enumerate(self.dummy_ground_truth_objects) if idx in expect_fn_indices
                ]

                estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(x_diff, y_diff, 0.0),
                    diff_yaw=0,
                )

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=estimated_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )

                tn_objects, fn_objects = get_negative_objects(
                    self.dummy_ground_truth_objects,
                    object_results,
                    self.target_labels,
                    self.matching_mode,
                    self.matching_threshold_list,
                )
                self.assertEqual(
                    tn_objects,
                    expect_tn_objects,
                    f"[TN] out: {len(tn_objects)}, expect: {len(expect_tn_objects)}",
                )
                self.assertEqual(
                    fn_objects,
                    expect_fn_objects,
                    f"[FN] out: {len(fn_objects)}, expect: {len(expect_fn_objects)}",
                )

    def test_divide_tp_fp_objects(self):
        """[summary]
        Test dividing TP (True Positive) objects and FP (False Positive) objects
        from Prediction condition positive objects.

        test objects:
            4 object_results made from dummy_ground_truth_objects with diff_distance

        test patterns:
            Given diff_distance, check if tp_results and ans_tp_results
            (fp_results and ans_fp_results) are the same.
            NOTE:
                - Estimated object is only matched with GT that has same label.
                - The estimations & GTs are following (number represents the index)
                    Estimation = 4
                        (0): CAR, (1): BICYCLE, (2): PEDESTRIAN, (3): MOTORBIKE
                    GT = 4
                        (0): CAR, (1): BICYCLE, (2): PEDESTRIAN, (3): MOTORBIKE
        """
        # patterns: (diff_distance, List[ans_tp_idx], label_change_dict: Dict[str, str])
        patterns: List[Tuple[float, List[int], List[int], Dict[str, str]]] = [
            # (1)
            # TP: (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2]), (Est[3], GT[3])
            # FP:
            # Given no diff_distance, all estimated_objects are tp.
            (
                0.0,
                np.array([(0, 0), (1, 1), (2, 2), (3, 3)]),
                np.array([]),
                {},
            ),
            # (2)
            # TP: (Est[1], GT[1]), (Est[2], GT[2])
            # FP: (Est[0], None), (Est[3], None)
            # Given no diff_distance and 2 labels changed, 2 estimated_objects are tp.
            (
                0.0,
                np.array([(0, 0), (1, 1), (2, 2)]),
                np.array([(3, None)]),
                {
                    "0": SemanticLabel(AutowareLabel.UNKNOWN, "unknown", []),
                    "3": SemanticLabel(AutowareLabel.ANIMAL, "animal", []),
                },
            ),
            # (3)
            # TP: (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2]), (Est[3], GT[3])
            # FP:
            # Given 1.5 diff_distance for one axis, all estimated_objects are tp.
            (
                1.5,
                np.array([(0, 0), (1, 1), (2, 2), (3, 3)]),
                np.array([]),
                {},
            ),
            # (4)
            # TP: (Est[0], GT[0]), (Est[1], GT[1]), (Est[3], GT[3])
            # FP: (Est[2], None)
            # Given 1.5 diff_distance for one axis and 1 labels changed, 3 estimated_objects are tp.
            (
                1.5,
                np.array([(2, 0), (1, 1), (3, 3)]),
                np.array([(0, None)]),
                {"2": SemanticLabel(AutowareLabel.UNKNOWN, "unknown", [])},
            ),
            # (5)
            # TP: (Est[2], GT[0]), (Est[1], GT[1]), (Est[3], GT[3])
            # FP: (Est[0], None)
            # Given 1.5 diff_distance for one axis and 1 labels changed, 3 estimated_objects are tp.
            (
                1.5,
                np.array([(2, 0), (1, 1), (3, 3)]),
                np.array([(0, None)]),
                {"2": SemanticLabel(AutowareLabel.CAR, "car", [])},
            ),
            # (6)
            # TP:
            # FP: (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2]), (Est[3], GT[3])
            # Given 2.5 diff_distance for one axis, all estimated_objects are fp
            # since they are beyond matching_threshold.
            (
                2.5,
                np.array([]),
                np.array([(0, 0), (1, 1), (2, 2), (3, 3)]),
                {},
            ),
        ]

        for n, (
            diff_distance,
            ans_tp_pair_idx,
            ans_fp_pair_idx,
            label_change_dict,
        ) in enumerate(patterns):
            with self.subTest(f"Test divide_tp_fp_objects: {n + 1}"):
                estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )

                # make dummy data with different label
                for idx, label in label_change_dict.items():
                    estimated_objects[int(idx)].semantic_label = label

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=estimated_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    matching_policy=MatchingPolicy(label_policy="ALLOW_UNKNOWN"),
                )
                tp_results, fp_results = divide_tp_fp_objects(
                    object_results,
                    self.target_labels,
                    self.matching_mode,
                    self.matching_threshold_list,
                    self.confidence_threshold_list,
                )
                # TP
                self.assertEqual(
                    len(tp_results),
                    len(ans_tp_pair_idx),
                    f"[{n + 1}]: Number of elements are not same, out: {len(tp_results)}, ans: {len(ans_tp_pair_idx)}",
                )
                for i, tp_result_ in enumerate(tp_results):
                    self.assertIn(
                        tp_result_.estimated_object,
                        estimated_objects,
                        f"TP estimated objects[{i}]",
                    )
                    est_idx: int = estimated_objects.index(tp_result_.estimated_object)
                    gt_idx: int = ans_tp_pair_idx[ans_tp_pair_idx[:, 0] == est_idx][0, 1]
                    self.assertEqual(
                        tp_result_.ground_truth_object,
                        self.dummy_ground_truth_objects[gt_idx],
                    )

                # FP
                self.assertEqual(
                    len(fp_results),
                    len(ans_fp_pair_idx),
                    f"[{n + 1}]: Number of elements are not same, out: {len(fp_results)}, ans: {len(ans_fp_pair_idx)}",
                )
                for j, fp_result_ in enumerate(fp_results):
                    self.assertIn(
                        fp_result_.estimated_object,
                        estimated_objects,
                        f"FP estimated objects[{j}]",
                    )
                    est_idx: int = estimated_objects.index(fp_result_.estimated_object)
                    gt_idx: Optional[int] = ans_fp_pair_idx[ans_fp_pair_idx[:, 0] == est_idx][0, 1]
                    if gt_idx is None:
                        self.assertIsNone(
                            fp_result_.ground_truth_object,
                            "ground truth must be None",
                        )
                    else:
                        self.assertEqual(
                            fp_result_.ground_truth_object,
                            self.dummy_ground_truth_objects[gt_idx],
                        )

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
                estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(x_diff, y_diff, 0.0),
                    diff_yaw=0,
                )
                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=estimated_objects,
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

                ans_fn_objects = [x for idx, x in enumerate(self.dummy_ground_truth_objects) if idx in ans_fn_idx]
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
            (
                0.0,
                0.0,
                [3],
                {
                    "0": SemanticLabel(AutowareLabel.UNKNOWN, "unknown", []),
                    "3": SemanticLabel(AutowareLabel.ANIMAL, "animal", []),
                },
            ),
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
            (1.5, 0.0, [], {"1": SemanticLabel(AutowareLabel.UNKNOWN, "unknown", [])}),
            # Given difference (2.5, 0.0), all ground_truth_objects are fn
            (2.5, 0.0, [0, 1, 2, 3], {}),
            # Given difference (2.5, 2.5), all ground_truth_objects are fn
            (2.5, 2.5, [0, 1, 2, 3], {}),
        ]

        for n, (x_diff, y_diff, ans_fn_idx, label_change_dict) in enumerate(patterns):
            with self.subTest(f"Test get_fn_objects_for_different_label: {n + 1}"):
                estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(x_diff, y_diff, 0.0),
                    diff_yaw=0,
                )

                # make dummy data with different label
                for idx, label in label_change_dict.items():
                    estimated_objects[int(idx)].semantic_label = label

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=estimated_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    matching_policy=MatchingPolicy(label_policy="ALLOW_UNKNOWN"),
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

                ans_fn_objects = [x for idx, x in enumerate(self.dummy_ground_truth_objects) if idx in ans_fn_idx]
                self.assertEqual(fn_objects, ans_fn_objects)

    def test_filter_object_results_by_confidence(self):
        """[summary]
        Test filtering DynamicObjectWithPerceptionResult by confidence

        test objects:
            4 object_results made from dummy_ground_truth_objects with diff_distance

        test patterns:
            Given diff_distance, check if filtered_object_results and ans_object_results
            are the same.
        """
        # patterns: (confidence_threshold, confidence_change_dict: Dict[str, float], List[ans_idx])
        patterns: List[Tuple[float, Dict[str, float], List[int]]] = [
            # When confidence_threshold is 0, no object_results are filtered out.
            (0.0, {}, [0, 1, 2, 3]),
            # When confidence_threshold is 0.3 and confidences of 1 objects change to 0.1, 1 object_results are filtered out.
            (0.3, {"0": 0.1}, [1, 2, 3]),
            # When confidence_threshold is 0.3 and confidences of 2 objects change to 0.1, 0.3, 2 object_results are filtered out.
            (0.3, {"0": 0.1, "1": 0.3}, [2, 3]),
            # When confidence_threshold is 0.3 and confidences of 2 objects change to 0.1, 0.31, 1 object_results are filtered out.
            (0.3, {"0": 0.1, "1": 0.31}, [1, 2, 3]),
            # When confidence_threshold is 0.9, all object_results are filtered out.
            (0.9, {}, []),
        ]
        for n, (confidence_threshold, confidence_change_dict, ans_idx) in enumerate(patterns):
            with self.subTest(f"Test filtered_object_results by confidence: {n + 1}"):
                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=self.dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )

                # change semantic score
                for idx, confidence in confidence_change_dict.items():
                    object_results[int(idx)].estimated_object.semantic_score = confidence

                confidence_threshold_list = [confidence_threshold] * len(self.target_labels)
                filtered_object_results = filter_object_results(
                    object_results=object_results,
                    target_labels=self.target_labels,
                    confidence_threshold_list=confidence_threshold_list,
                )
                ans_object_results = [x for idx, x in enumerate(object_results) if idx in ans_idx]
                self.assertEqual(filtered_object_results, ans_object_results)

    def test_divide_objects(self):
        """[summary]
        Test divide DynamicObject or DynamicObjectWithPerceptionResult by their labels as dict.

        test objects:

        test patterns:
        """
        objects_dict: Dict[AutowareLabel, List[DynamicObject]] = divide_objects(self.dummy_ground_truth_objects)
        for label, objects in objects_dict.items():
            assert all([obj.semantic_label.label == label for obj in objects])

        objects_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=self.dummy_ground_truth_objects,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        object_results_dict: Dict[AutowareLabel, List[DynamicObjectWithPerceptionResult]] = divide_objects(
            objects_results
        )
        for label, object_results in object_results_dict.items():
            assert all([obj_result.estimated_object.semantic_label.label == label for obj_result in object_results])

    def test_divide_objects_to_num(self):
        """[summary]
        Test divide the number of DynamicObject or DynamicObjectWithPerceptionResult by their labels as dict.

        test objects:

        test patterns:
        """
        ans: Dict[AutowareLabel, int] = {
            AutowareLabel.CAR: 1,
            AutowareLabel.BICYCLE: 1,
            AutowareLabel.MOTORBIKE: 1,
            AutowareLabel.PEDESTRIAN: 1,
        }

        objects_num_dict: Dict[AutowareLabel, List[DynamicObject]] = divide_objects_to_num(
            self.dummy_ground_truth_objects
        )
        for label, num in objects_num_dict.items():
            assert ans[label] == num, f"{ans[label]}, {num}"

        objects_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=self.dummy_ground_truth_objects,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        object_results_num_dict: Dict[AutowareLabel, int] = divide_objects_to_num(objects_results)
        for label, num in object_results_num_dict.items():
            assert ans[label] == num


if __name__ == "__main__":
    unittest.main()
