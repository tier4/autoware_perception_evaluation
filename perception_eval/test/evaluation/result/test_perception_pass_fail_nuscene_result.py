# Copyright 2022 TIER IV, Inc.

from test.util.dummy_object import make_dummy_data
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import unittest

import numpy as np
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import Label
from perception_eval.common.object import DynamicObject
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.matching import MatchingLabelPolicy
from perception_eval.evaluation.matching.object_matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.object_result_matching import get_object_results
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.evaluation.result.perception_pass_fail_nuscene_result import PassFailNusceneResult
from perception_eval.util.debug import get_objects_with_difference


class TestPassFailNusceneResult(unittest.TestCase):
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
        # Minimal config for PassFailNusceneResult
        dummy_eval_cfg = PerceptionEvaluationConfig(
            dataset_paths=["dummy"],
            frame_id="base_link",
            result_root_directory="/tmp",
            evaluation_config_dict={
                "evaluation_task": "detection",
                "target_labels": [label.value for label in self.target_labels],
                "center_distance_thresholds": [self.matching_threshold_list],
                "min_point_numbers": self.min_point_numbers,
                "max_x_position": 2.0,
                "max_y_position": 2.0,
                "max_x_position_list": self.max_x_position_list,
                "max_y_position_list": self.max_y_position_list,
                "max_pos_distance_list": self.max_pos_distance_list,
                "min_pos_distance_list": self.min_pos_distance_list,
                "label_prefix": "autoware",
            },
        )
        self.critical_object_filter_config = CriticalObjectFilterConfig(
            dummy_eval_cfg,
            [label.value for label in self.target_labels],
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
        )
        self.frame_pass_fail_config = PerceptionPassFailConfig(
            dummy_eval_cfg, [label.value for label in self.target_labels], self.matching_threshold_list
        )

    def test_get_positive_objects(self):
        patterns: List[Tuple[float, np.ndarray, np.ndarray, Dict[int, Label]]] = [
            (0.0, np.array([(0, 0), (1, 1), (2, 2), (3, 3)]), np.array([]), {}),
            (
                0.0,
                np.array([(0, 0), (1, 1), (2, 2)]),
                np.array([(3, 3)]),
                {
                    0: Label(AutowareLabel.UNKNOWN, "unknown", []),
                    3: Label(AutowareLabel.ANIMAL, "animal", []),
                },
            ),
            (1.5, np.array([(0, 0), (1, 1), (2, 2), (3, 3)]), np.array([]), {}),
            (
                1.5,
                np.array([(2, 0), (1, 1), (3, 3)]),
                np.array([(0, 2)]),
                {2: Label(AutowareLabel.UNKNOWN, "unknown", [])},
            ),
            (1.5, np.array([(2, 0), (1, 1), (3, 3)]), np.array([(0, 2)]), {2: Label(AutowareLabel.CAR, "car", [])}),
            (2.5, np.array([]), np.array([(0, 0), (1, 1), (2, 2), (3, 3)]), {}),
        ]
        for n, (diff_distance, expect_tp_indices, expect_fp_indices, label_changes) in enumerate(patterns):
            with self.subTest(f"Test get_positive_objects: {n + 1}"):
                estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0, 0),
                    diff_yaw=0,
                )
                for idx, label in label_changes.items():
                    estimated_objects[idx].semantic_label = label
                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=estimated_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    matching_label_policy=MatchingLabelPolicy.ALLOW_UNKNOWN,
                )
                pf_result = PassFailNusceneResult(
                    unix_time=0,
                    frame_number=0,
                    critical_object_filter_config=self.critical_object_filter_config,
                    frame_pass_fail_config=self.frame_pass_fail_config,
                )
                tp_results, fp_results = pf_result._get_positive_objects(object_results)
                self.assertEqual(len(tp_results), len(expect_tp_indices))
                for tp_result in tp_results:
                    tp_est_idx: int = estimated_objects.index(tp_result.estimated_object)
                    tp_gt_idx: int = expect_tp_indices[expect_tp_indices[:, 0] == tp_est_idx][0, 1]
                    self.assertEqual(tp_result.ground_truth_object, self.dummy_ground_truth_objects[tp_gt_idx])
                self.assertEqual(len(fp_results), len(expect_fp_indices))
                for fp_result in fp_results:
                    fp_est_idx: int = estimated_objects.index(fp_result.estimated_object)
                    fp_gt_idx: Optional[int] = expect_fp_indices[expect_fp_indices[:, 0] == fp_est_idx][0, 1]
                    if fp_gt_idx is None:
                        self.assertIsNone(fp_result.ground_truth_object)
                    else:
                        self.assertEqual(fp_result.ground_truth_object, self.dummy_ground_truth_objects[fp_gt_idx])

    def test_get_negative_objects(self):
        patterns: List[Tuple[float, float, List[int], List[int]]] = [
            (0.0, 0.0, [], []),
            (0.5, 0.0, [], []),
            (0.0, -0.5, [], []),
            (1.5, 0.0, [], []),
            (2.0, 0.0, [], [0, 1, 2, 3]),
            (2.5, 0.0, [], [0, 1, 2, 3]),
            (2.5, 2.5, [], [0, 1, 2, 3]),
        ]
        for n, (x_diff, y_diff, expect_tn_indices, expect_fn_indices) in enumerate(patterns):
            with self.subTest(f"Test get_negative_objects: {n+1}"):
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
                pf_result = PassFailNusceneResult(
                    unix_time=0,
                    frame_number=0,
                    critical_object_filter_config=self.critical_object_filter_config,
                    frame_pass_fail_config=self.frame_pass_fail_config,
                )
                tn_objects, fn_objects = pf_result.get_negative_objects(
                    self.dummy_ground_truth_objects,
                    object_results,
                )
                self.assertEqual(tn_objects, expect_tn_objects)
                self.assertEqual(fn_objects, expect_fn_objects)


if __name__ == "__main__":
    unittest.main()
