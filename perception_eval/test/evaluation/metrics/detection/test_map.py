# Copyright 2025 TIER IV, Inc.

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

from perception_eval.common import DynamicObject
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.matching.object_matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.metrics.detection.map import Map
from perception_eval.evaluation.metrics.metrics_score_config import MetricsScoreConfig
from perception_eval.evaluation.result.object_result_matching import get_nuscene_object_results
from perception_eval.util.debug import get_objects_with_difference


class TestMap(unittest.TestCase):
    def setUp(self):
        # Create 3 predicted objects (CAR, BICYCLE, CAR) and 4 ground truths (CAR, BICYCLE, PEDESTRIAN, MOTORBIKE)
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()
        self.evaluation_task: EvaluationTask = EvaluationTask.DETECTION
        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]

    def _get_default_metrics_config(self, matching_threshold: float) -> MetricsScoreConfig:
        return MetricsScoreConfig(
            evaluation_task="detection",
            target_labels=self.target_labels,
            center_distance_thresholds=[matching_threshold],
            center_distance_bev_thresholds=[matching_threshold],
            iou_2d_thresholds=[matching_threshold],
            iou_3d_thresholds=[matching_threshold],
            plane_distance_thresholds=[matching_threshold],
        )

    def _evaluate_map(
        self,
        estimated_objects: List[DynamicObject],
        ground_truth_objects: List[DynamicObject],
        matching_mode: MatchingMode,
        matching_threshold: float,
    ) -> Map:
        metrics_config = self._get_default_metrics_config(matching_threshold)
        nuscene_object_results = get_nuscene_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=estimated_objects,
            ground_truth_objects=ground_truth_objects,
            metrics_config=metrics_config,
        )
        num_ground_truth_dict = divide_objects_to_num(ground_truth_objects, self.target_labels)
        return Map(
            object_results_dict=nuscene_object_results[matching_mode],
            num_ground_truth_dict=num_ground_truth_dict,
            target_labels=self.target_labels,
            matching_mode=matching_mode,
        )

    def _run_param_test(
        self,
        patterns: List[Tuple[float, float, float]],
        mode: MatchingMode,
        threshold: float,
        diff_type: str,
    ):
        for delta, ans_map, ans_maph in patterns:
            with self.subTest(f"{mode.name} {diff_type} delta={delta}"):
                est_objs = get_objects_with_difference(
                    self.dummy_estimated_objects,
                    diff_distance=(delta, 0, 0) if diff_type == "translation" else (0, 0, 0),
                    diff_yaw=delta if diff_type == "yaw" else 0,
                )
                map_result = self._evaluate_map(est_objs, self.dummy_ground_truth_objects, mode, threshold)
                self.assertAlmostEqual(map_result.map, ans_map, places=3)
                self.assertAlmostEqual(map_result.maph, ans_maph, places=3)

    def test_map_patterns(self):
        """Test mAP and mAPH across matching modes and difference types."""
        test_cases = [
            # (matching_mode, threshold, diff_type, [(delta, expected_map, expected_maph)])
            (
                MatchingMode.CENTERDISTANCE,
                1.0,
                "translation",
                [
                    # Values like 0.498 come from:
                    #  CAR AP = 0.9938, BICYCLE = 1.0, others = 0.0 → avg over 4 = 0.498
                    (0.0, 0.498, 0.498),
                    (0.5, 0.498, 0.498),
                    # CAR AP = 0.4444, others: 0.0
                    (2.5, 0.111, 0.111),
                ],
            ),
            (
                MatchingMode.CENTERDISTANCE,
                1.0,
                "yaw",
                [
                    (0.0, 0.498, 0.498),
                    (math.pi / 2, 0.498, 0.220),
                    (-math.pi / 2, 0.498, 0.220),
                    (math.pi, 0.498, 0.0),
                    (math.pi / 4, 0.498, 0.359),
                    (3 * math.pi / 4, 0.498, 0.08080),
                ],
            ),
            (
                MatchingMode.IOU2D,
                0.4,
                "translation",
                [
                    (0.0, 0.248, 0.248),
                    (0.3, 0.248, 0.248),
                    (0.5, 0.0, 0.0),
                ],
            ),
            (
                MatchingMode.IOU2D,
                0.4,
                "yaw",
                [
                    (0.0, 0.248, 0.248),
                    (math.pi / 2, 0.248, 0.109),
                    (math.pi, 0.248, 0.0),
                    (math.pi / 4, 0.248, 0.1787),
                    (3 * math.pi / 4, 0.248, 0.039),
                ],
            ),
            (
                MatchingMode.IOU3D,
                0.277,
                "translation",
                [
                    (0.0, 0.248, 0.248),
                    (0.3, 0.248, 0.248),
                    (0.5, 0.0, 0.0),
                ],
            ),
            (
                MatchingMode.IOU3D,
                0.29,
                "yaw",
                [
                    (0.0, 0.248, 0.248),
                    (math.pi / 2, 0.248, 0.109),
                    (math.pi, 0.248, 0.0),
                    (math.pi / 4, 0.248, 0.1787),
                    (3 * math.pi / 4, 0.248, 0.039),
                ],
            ),
            (
                MatchingMode.PLANEDISTANCE,
                1.0,
                "translation",
                [
                    (0.0, 0.498, 0.498),
                    (0.5, 0.498, 0.498),
                    (2.5, 0.111, 0.111),
                ],
            ),
            (
                MatchingMode.PLANEDISTANCE,
                1.0,
                "yaw",
                [
                    (0.0, 0.498, 0.498),
                    (math.pi / 2, 0.25, 0.111),
                    (math.pi, 0.0, 0.0),
                    (math.pi / 4, 0.498, 0.359),
                    (3 * math.pi / 4, 0.25, 0.0416),
                ],
            ),
        ]
        for mode, threshold, diff_type, patterns in test_cases:
            self._run_param_test(patterns, mode, threshold, diff_type)

    def test_map_no_prediction(self):
        """Test mAP when there are no predicted objects (all gt exist)."""
        est_objs = []  # empty prediction
        map_result = self._evaluate_map(est_objs, self.dummy_ground_truth_objects, MatchingMode.CENTERDISTANCE, 1.0)
        self.assertEqual(map_result.map, 0.0)
        self.assertEqual(map_result.maph, 0.0)

    def test_map_no_ground_truth(self):
        """Test mAP when there are no ground truth objects (but predictions exist)."""
        gt_objs = []  # empty GT
        map_result = self._evaluate_map(self.dummy_estimated_objects, gt_objs, MatchingMode.CENTERDISTANCE, 1.0)
        # mAP is defined as 0 when GT count is 0 for all labels
        self.assertEqual(map_result.map, 0.0)
        self.assertEqual(map_result.maph, 0.0)

    def test_map_no_gt_no_prediction(self):
        """Test mAP when both predictions and ground truth are empty."""
        est_objs = []
        gt_objs = []
        map_result = self._evaluate_map(est_objs, gt_objs, MatchingMode.CENTERDISTANCE, 1.0)
        self.assertEqual(map_result.map, 0.0)
        self.assertEqual(map_result.maph, 0.0)

    def test_map_str_output(self):
        """Test that the __str__ output of Map contains expected label and score info."""
        map_result = self._evaluate_map(
            self.dummy_estimated_objects,
            self.dummy_ground_truth_objects,
            MatchingMode.CENTERDISTANCE,
            1.0,
        )
        output_str = str(map_result)

        # Check label names
        for label in self.target_labels:
            self.assertIn(label.value, output_str)

        # Check table headers
        self.assertIn("Predict_num", output_str)
        self.assertIn("Groundtruth_num", output_str)
        self.assertIn("AP", output_str)

        if not map_result.is_detection_2d:
            self.assertIn("APH", output_str)
        else:
            self.assertNotIn("APH", output_str)


if __name__ == "__main__":
    unittest.main()
