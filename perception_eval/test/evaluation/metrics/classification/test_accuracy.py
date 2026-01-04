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

from __future__ import annotations

from math import isclose
from test.util.dummy_object import make_dummy_data2d
from typing import List
import unittest

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.matching.objects_filter import filter_objects, divide_objects_to_num
from perception_eval.evaluation.metrics.classification.accuracy import ClassificationAccuracy
from perception_eval.evaluation.result.object_result_matching import get_object_results, NuscenesObjectMatcher
from perception_eval.evaluation.metrics.metrics_score_config import MetricsScoreConfig

class AnswerAccuracy:
    """Answer class for ClassificationAccuracy to compare result."""

    def __init__(
        self,
        num_ground_truth: int,
        num_tp: int,
        num_fp: int,
        accuracy: float,
        precision: float,
        recall: float,
        f1score: float,
    ) -> None:
        self.num_ground_truth: int = num_ground_truth
        self.num_tp: int = num_tp
        self.num_fp: int = num_fp
        self.accuracy: float = accuracy
        self.precision: recall = precision
        self.recall: float = recall
        self.f1score: float = f1score

    @classmethod
    def from_accuracy(cls, accuracy: ClassificationAccuracy) -> AnswerAccuracy:
        return AnswerAccuracy(
            accuracy.num_ground_truth,
            accuracy.num_tp,
            accuracy.num_fp,
            accuracy.accuracy,
            accuracy.precision,
            accuracy.recall,
            accuracy.f1score,
        )

    def __eq__(self, other: AnswerAccuracy) -> bool:
        return (
            self.num_ground_truth == other.num_ground_truth
            and self.num_tp == other.num_tp
            and self.num_fp == other.num_fp
            and isclose(self.accuracy, other.accuracy, abs_tol=0.01)
            and isclose(self.precision, other.precision, abs_tol=0.01)
            and isclose(self.recall, other.recall, abs_tol=0.01)
            and isclose(self.f1score, other.f1score, abs_tol=0.01)
        )

    def __str__(self) -> str:
        str_: str = "\n("
        str_ += f"num_ground_truth: {self.num_ground_truth}, "
        str_ += f"num_tp: {self.num_tp}, "
        str_ += f"num_fp: {self.num_fp}, "
        str_ += f"accuracy: {self.accuracy}, "
        str_ += f"precision: {self.precision}, "
        str_ += f"recall: {self.recall}, "
        str_ += f"f1score: {self.f1score}"
        str_ += ")"
        return str_


class TestClassificationAccuracy(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data2d(use_roi=False)

        self.evaluation_task: EvaluationTask = EvaluationTask.CLASSIFICATION2D
        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]
        self.matching_threshold_list: List[float] = [0.5]
        self.metric_score_config = MetricsScoreConfig(
            evaluation_task=self.evaluation_task,
            target_labels=self.target_labels,
            center_distance_thresholds=self.matching_threshold_list,
            center_distance_bev_thresholds=None,
            plane_distance_thresholds=None,
            iou_2d_thresholds=None,
            iou_3d_thresholds=None,
        )

    def test_calculate_accuracy(self):
        # patterns: List[Tuple[AutowareLabel, AnswerAccuracy]]
        patterns: List[AnswerAccuracy] = [
            (AutowareLabel.CAR, AnswerAccuracy(1, 1, 1, 0.5, 0.5, 1.0, 0.66)),
            (AutowareLabel.BICYCLE, AnswerAccuracy(1, 1, 0, 1.0, 1.0, 1.0, 1.0)),
        ]
        matcher = NuscenesObjectMatcher(
            evaluation_task=self.evaluation_task,
            metrics_config=self.metric_score_config,
        )
        for n, (target_label, answer) in enumerate(patterns):
            with self.subTest(f"Test calculate Accuracy: {n + 1}"):
                # Filter objects
                estimated_objects = filter_objects(
                    dynamic_objects=self.dummy_estimated_objects,
                    is_gt=False,
                    target_labels=[target_label],
                )
                ground_truth_objects = filter_objects(
                    dynamic_objects=self.dummy_ground_truth_objects,
                    is_gt=True,
                    target_labels=[target_label],
                )
                # Get object results
                object_results = matcher.match(
                    estimated_objects=estimated_objects,
                    ground_truth_objects=ground_truth_objects,
                )
                num_ground_truth_dict = divide_objects_to_num(ground_truth_objects, self.target_labels)
                num_ground_truth = len(ground_truth_objects)
                for _, label_object_results in object_results.items():
                    num_ground_truth = num_ground_truth_dict[target_label]
                    for _, object_results in label_object_results[target_label].items():
                        accuracy = ClassificationAccuracy(
                            object_results=object_results,
                            num_ground_truth=num_ground_truth,
                            target_labels=[target_label],
                        )
                        out_accuracy = AnswerAccuracy.from_accuracy(accuracy)
                        self.assertEqual(out_accuracy, answer, f"\nout = {str(out_accuracy)},\nanswer = {str(answer)}")
