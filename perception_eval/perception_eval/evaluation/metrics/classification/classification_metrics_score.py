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

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List

from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

from .accuracy import ClassificationAccuracy


@dataclass(frozen=True)
class ClassificationScores:
    """Dataclass for classification score.

    Attributes:
        accuracy (float): Accuracy score.
        precision (float): Precision score.
        recall (float): Recall score.
        f1score (float): F1 score.
    """

    accuracy: float
    precision: float
    recall: float
    f1score: float
    predict_num: int = 0
    ground_truth_num: int = 0


class ClassificationMetricsScore:
    """Metrics score class for classification evaluation.

    Attributes:
        self.accuracies (List[ClassificationAccuracy]): List of ClassificationAccuracy instances.

    Args:
        nuscene_object_results (Dict[MatchingMode, Dict[LabelType, Dict[matching threshold, List[DynamicObjectWithPerceptionResult]]]):
            Dict that are list of DynamicObjectWithPerceptionResult mapped by their matching mode, label and threshold.
        num_ground_truth_dict (Dict[LabelType, int]): Dict that are number of DynamicObjectWithPerceptionResult
            mapped by their labels.
        target_labels (List[LabelType]): Target labels list.
    """

    def __init__(
        self,
        nuscene_object_results: Dict[
            MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]
        ],
        num_ground_truth_dict: Dict[LabelType, int],
        target_labels: List[LabelType],
    ) -> None:
        self.target_labels = target_labels
        self.accuracies: Dict[MatchingMode, Dict[LabelType, Dict[float, ClassificationAccuracy]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(ClassificationAccuracy))
        )
        for matching_mode, label_object_results in nuscene_object_results.items():
            for target_label in target_labels:
                num_ground_truth = num_ground_truth_dict[target_label]
                selected_object_results = label_object_results.get(target_label, [])

                # Empty object results, but there are ground truths
                if not selected_object_results:
                    acc_: ClassificationAccuracy = ClassificationAccuracy(
                        object_results=[],
                        num_ground_truth=num_ground_truth,
                        target_labels=[target_label],
                    )
                    # Always only one threshold for classification, which is -1.0
                    self.accuracies[matching_mode][target_label][-1.0] = acc_
                else:
                    for threshold, object_results in label_object_results[target_label].items():
                        acc_: ClassificationAccuracy = ClassificationAccuracy(
                            object_results=object_results,
                            num_ground_truth=num_ground_truth,
                            target_labels=[target_label],
                        )
                        self.accuracies[matching_mode][target_label][threshold] = acc_

    def _summarize(self) -> Dict[MatchingMode, Dict[float, ClassificationScores]]:
        """Summarize all ClassificationAccuracy over the labels.

        Returns:
            accuracy (float): Accuracy score. When `num_est+num_gt-num_tp=0`, this is float('inf').
            precision (float): Precision score. When `num_gt+num_fp=0`, this is float('inf').
            recall (float): Recall score. When `num_gt=0`, this is float('inf').
            f1score (float): F1 score. When `precision+recall=0`, this is float('inf').
        """
        # accumulate counts for {MatchingMode: {Matching thresholds: {num_est, num_gt, num_tp, num_fp}}}
        matching_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for matching_mode, label_accuracies in self.accuracies.items():
            for _, thresholds in label_accuracies.items():
                for threshold, acc_ in thresholds.items():
                    matching_counts[matching_mode][threshold]["num_est"] += acc_.objects_results_num
                    matching_counts[matching_mode][threshold]["num_gt"] += acc_.num_ground_truth
                    matching_counts[matching_mode][threshold]["num_tp"] += acc_.num_tp
                    matching_counts[matching_mode][threshold]["num_fp"] += acc_.num_fp

        # {MatchingMode: {Matching threshold: ClassificationScores}}
        results = defaultdict(lambda: defaultdict(ClassificationScores))
        for matching_mode, thresholds in matching_counts.items():
            for threshold, counts in thresholds.items():
                num_est = counts["num_est"]
                num_gt = counts["num_gt"]
                num_tp = counts["num_tp"]
                num_fp = counts["num_fp"]

                accuracy = num_tp / (num_est + num_gt - num_tp) if (num_est + num_gt - num_tp) != 0 else float("inf")
                precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) != 0 else float("inf")
                recall = num_tp / num_gt if num_gt != 0 else float("inf")
                f1score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else float("inf")
                results[matching_mode][threshold] = ClassificationScores(
                    accuracy, precision, recall, f1score, num_est, num_gt
                )

        return results

    def __str__(self) -> str:
        """__str__ method

        Returns:
            str: Formatted string.
        """
        str_: str = "\n"

        # Summarize overall classification scores
        summarize_classification_scores = self._summarize()
        for matching_mode, thresholds in summarize_classification_scores.items():
            str_ += "---- Matching Mode: {} ----\n".format(matching_mode.value)
            str_ += "|    Threshold | Predict Num | Ground Truth Num | Accuracy | Precision | Recall | F1score |\n"
            str_ += "| :-------: | :---------: | :------: | :------: | :-------: | :----: | :-----: |\n"
            for threshold, scores in thresholds.items():
                str_ += f"| {threshold} | {scores.predict_num} | {scores.ground_truth_num} | {scores.accuracy:.4f} | {scores.precision:.4f} | {scores.recall:.4f} | {scores.f1score:.4f} |\n"

        str_ += "\n"
        # === For each label ===
        # --- Table ---
        for matching_mode, label_accuracies in self.accuracies.items():
            str_ += "---- Matching Mode: {} ----\n".format(matching_mode.value)
            str_ += (
                "|   Label | Threshold | Predict Num | Ground Truth Num | Accuracy | Precision | Recall | F1score |\n"
            )
            str_ += (
                "| :------: | :---: | :-------: | :---------: | :------: | :------: | :-------: | :----: | :-----: |\n"
            )
            for label, thresholds in label_accuracies.items():
                str_ += f"| {label} |"

                # Threshold column
                threshold_str = " / ".join([str(threshold) for threshold in thresholds.keys()])
                str_ += f" {threshold_str} |"

                # Predict Num column
                predict_num_str = " / ".join([str(v.objects_results_num) for v in thresholds.values()])
                str_ += f" {predict_num_str} |"

                # Ground truth column
                ground_truth_str = " / ".join([str(v.num_ground_truth) for v in thresholds.values()])
                str_ += f" {ground_truth_str} |"

                # Accuracy column
                accuracy_str = " / ".join([f"{v.accuracy:.4f}" for v in thresholds.values()])
                str_ += f" {accuracy_str} |"

                # Precision column
                precision_str = " / ".join([f"{v.precision:.4f}" for v in thresholds.values()])
                str_ += f" {precision_str} |"

                # Recall column
                recall_str = " / ".join([f"{v.recall:.4f}" for v in thresholds.values()])
                str_ += f" {recall_str} |"

                # F1score column
                f1score_str = " / ".join([f"{v.f1score:.4f}" for v in thresholds.values()])
                str_ += f" {f1score_str} |\n"

        str_ += "\n"

        return str_
