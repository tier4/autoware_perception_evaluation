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

from test.util.dummy_object import make_dummy_data2d_traffic_light
from typing import List
import unittest

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import TrafficLightLabel
from perception_eval.evaluation.matching.object_matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics.classification import ClassificationMetricsScore
from perception_eval.evaluation.metrics.metrics_score_config import MetricsScoreConfig
from perception_eval.evaluation.result.object_result_matching import NuscenesObjectMatcher


class TestTrafficLightClassificationMetricsScore(unittest.TestCase):
    """The class to test ClassificationMetricsScore."""

    def setUp(self) -> None:
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data2d_traffic_light(use_roi=False)
        self.evaluation_task: EvaluationTask = EvaluationTask.CLASSIFICATION2D
        self.target_labels: List[TrafficLightLabel] = [
            TrafficLightLabel.GREEN,
            TrafficLightLabel.RED,
            TrafficLightLabel.YELLOW,
        ]
        self.matching_threshold_list: List[float] = [0.5]
        self.metric_score_config = MetricsScoreConfig(
            evaluation_task=self.evaluation_task,
            target_labels=self.target_labels,
            center_distance_thresholds=None,
            center_distance_bev_thresholds=None,
            plane_distance_thresholds=None,
            iou_2d_thresholds=None,
            iou_3d_thresholds=None,
        )

    def test_summarize(self):
        """Test ClassificationMetricsScore._summarize().

        num_est = 3
        num_gt = 4
        num_tp = 3
        num_fp = 0

        accuracy = 3 / (3 + 4 - 3) = 0.75
        precision = 3 / 3 (TP + FP) = 1.0
        recall = 3 / 4 (TP + FN) = 0.75
        f1score = 2 * 1.0 * 0.75 / (1.0 + 0.75) = 0.857...
        """
        matcher = NuscenesObjectMatcher(
            evaluation_task=self.evaluation_task,
            metrics_config=self.metric_score_config,
        )
        estimated_objects = filter_objects(
            dynamic_objects=self.dummy_estimated_objects,
            is_gt=False,
            target_labels=self.target_labels,
        )
        ground_truth_objects = filter_objects(
            dynamic_objects=self.dummy_ground_truth_objects,
            is_gt=False,
            target_labels=self.target_labels,
        )
        object_results = matcher.match(
            estimated_objects=estimated_objects,
            ground_truth_objects=ground_truth_objects,
        )
        num_ground_truth_dict = divide_objects_to_num(
            dynamic_objects=ground_truth_objects,
            target_labels=self.target_labels,
        )

        classification_score = ClassificationMetricsScore(
            nuscene_object_results=object_results,
            num_ground_truth_dict=num_ground_truth_dict,
            target_labels=self.target_labels,
        )

        summary_results = classification_score._summarize()
        for _, thresholds_to_score in summary_results.items():
            for _, scores in thresholds_to_score.items():
                print(scores)
                self.assertAlmostEqual(scores.accuracy, 0.75, delta=0.01)
                self.assertAlmostEqual(scores.precision, 1.0, delta=0.01)
                self.assertAlmostEqual(scores.recall, 0.75, delta=0.01)
                self.assertAlmostEqual(scores.f1score, 0.857, delta=0.01)
                self.assertEqual(scores.predict_num, 3)
                self.assertEqual(scores.ground_truth_num, 4)

        # String representation
        str_ = str(classification_score)
        self.assertIn(MatchingMode.TLR_CLASSIFICATION.value, str_)
        self.assertIn("Accuracy", str_)
        self.assertIn("Precision", str_)
        self.assertIn("Recall", str_)
        self.assertIn("F1score", str_)
        self.assertIn("Predict Num", str_)
        self.assertIn("Ground Truth Num", str_)
