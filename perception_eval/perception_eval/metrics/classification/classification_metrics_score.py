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

from typing import Dict
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING

from .accuracy import ClassificationAccuracy

if TYPE_CHECKING:
    from perception_eval.common.label import LabelType
    from perception_eval.result import PerceptionObjectResult


class ClassificationMetricsScore:
    """Metrics score class for classification evaluation.

    Attributes:
        self.accuracies (List[ClassificationAccuracy]): List of ClassificationAccuracy instances.

    Args:
        object_results_dict (Dict[LabelType, List[List[PerceptionObjectResult]]]):
            Dict that are list of PerceptionObjectResult mapped by their labels.
        num_ground_truth_dict (Dict[LabelType, int]): Dict that are number of PerceptionObjectResult
            mapped by their labels.
        target_labels (List[LabelType]): Target labels list.
    """

    def __init__(
        self,
        object_results_dict: Dict[LabelType, List[List[PerceptionObjectResult]]],
        num_ground_truth_dict: Dict[LabelType, int],
        target_labels: List[LabelType],
    ) -> None:
        self.accuracies: List[ClassificationAccuracy] = []
        for target_label in target_labels:
            object_results = object_results_dict[target_label]
            num_ground_truth = num_ground_truth_dict[target_label]

            acc_ = ClassificationAccuracy(
                object_results=object_results,
                num_ground_truth=num_ground_truth,
                target_labels=[target_label],
            )
            self.accuracies.append(acc_)

    def _summarize(self) -> Tuple[float, float, float, float]:
        """Summarize all ClassificationAccuracy.

        Returns:
            accuracy (float): Accuracy score. When `num_est+num_gt-num_tp=0`, this is float('inf').
            precision (float): Precision score. When `num_gt+num_fp=0`, this is float('inf').
            recall (float): Recall score. When `num_gt=0`, this is float('inf').
            f1score (float): F1 score. When `precision+recall=0`, this is float('inf').
        """
        num_est: int = 0
        num_gt: int = 0
        num_tp: int = 0
        num_fp: int = 0
        for acc_ in self.accuracies:
            num_est += acc_.objects_results_num
            num_gt += acc_.num_ground_truth
            num_tp += acc_.num_tp
            num_fp += acc_.num_fp
        accuracy = num_tp / (num_est + num_gt - num_tp) if (num_est + num_gt - num_tp) != 0 else float("inf")
        precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) != 0 else float("inf")
        recall = num_tp / num_gt if num_gt != 0 else float("inf")
        f1score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else float("inf")
        return accuracy, precision, recall, f1score

    def __str__(self) -> str:
        """__str__ method

        Returns:
            str: Formatted string.
        """
        str_: str = "\n"
        # === Total ===
        acc, precision, recall, f1score = self._summarize()
        str_ += f"[TOTAL] Accuracy: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1score: {f1score:.3f}"
        # === For each label ===
        # --- Table ---
        str_ += "\n"
        # --- Label ----
        str_ += "|      Label |"
        # CLEAR
        for acc_ in self.accuracies:
            str_ += f" {acc_.target_labels[0]} | "
        str_ += "\n"
        str_ += "| :--------: |"
        for _ in self.accuracies:
            str_ += " :---: |"
        str_ += "\n"
        str_ += "|    predict_num |"
        for acc_ in self.accuracies:
            score: float = acc_.objects_results_num
            str_ += f" {score} |"
        str_ += "\n"
        str_ += "|    Accuracy |"
        for acc_ in self.accuracies:
            score: float = acc_.accuracy
            str_ += f" {score:.3f} |"
        str_ += "\n"
        str_ += "|   Precision |"
        for acc_ in self.accuracies:
            score: float = acc_.precision
            str_ += f" {score:.3f} |"
        str_ += "\n"
        str_ += "|   Recall |"
        for acc_ in self.accuracies:
            score: float = acc_.recall
            str_ += f" {score:.3f} |"
        str_ += "\n"
        str_ += "|   F1score |"
        for acc_ in self.accuracies:
            score: float = acc_.f1score
            str_ += f" {score:.3f} |"
        str_ += "\n"

        return str_
