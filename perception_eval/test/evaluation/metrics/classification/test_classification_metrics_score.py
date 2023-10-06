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

from test.util.dummy_object import make_dummy_data2d
from typing import List
import unittest

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics.classification import ClassificationMetricsScore
from perception_eval.evaluation.result.object_result import get_object_results


class TestClassificationMetricsScore(unittest.TestCase):
    """The class to test ClassificationMetricsScore."""

    def setUp(self) -> None:
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data2d(use_roi=False)
        self.evaluation_task: EvaluationTask = EvaluationTask.CLASSIFICATION2D
        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]

    def test_summarize(self):
        """Test ClassificationMetricsScore._summarize().

        num_est = 3
        num_gt = 4
        num_tp = 2
        num_fp = 1

        accuracy = 2 / (3 + 4 - 2) = 0.4
        precision = 2 / 3 = 0.66...
        recall = 2 / 4 = 0.5
        f1score = 2 * 0.5 * 0.66 / (0.5 + 0.66) = 0.57...
        """
        estimated_objects = filter_objects(
            objects=self.dummy_estimated_objects,
            is_gt=False,
            target_labels=self.target_labels,
        )
        ground_truth_objects = filter_objects(
            objects=self.dummy_ground_truth_objects,
            is_gt=False,
            target_labels=self.target_labels,
        )
        object_results = get_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=estimated_objects,
            ground_truth_objects=ground_truth_objects,
        )
        object_results_dict = divide_objects(
            objects=object_results,
            target_labels=self.target_labels,
        )
        num_ground_truth_dict = divide_objects_to_num(
            objects=ground_truth_objects,
            target_labels=self.target_labels,
        )

        classification_score = ClassificationMetricsScore(
            object_results_dict=object_results_dict,
            num_ground_truth_dict=num_ground_truth_dict,
            target_labels=self.target_labels,
        )

        accuracy, precision, recall, f1score = classification_score._summarize()
        self.assertAlmostEqual(accuracy, 0.4, delta=0.01)
        self.assertAlmostEqual(precision, 0.66, delta=0.01)
        self.assertAlmostEqual(recall, 0.5, delta=0.01)
        self.assertAlmostEqual(f1score, 0.57, delta=0.01)
