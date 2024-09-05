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

from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Tuple

from perception_eval.common.label import LabelType
from perception_eval.evaluation import DynamicObjectWithPerceptionResult


class ClassificationAccuracy:
    """[summary]
    Class to calculate classification accuracy.

    Attributes:
        target_labels (List[LabelType]): Target labels list.
        num_ground_truth (int): Number of ground truths.
        objects_results_num (int): Number of object results.
        num_tp (int): Number of TP results.
        num_fp (int): Number of FP results.
        accuracy (float): Accuracy score. When `num_ground_truth+objects_results_num-num_tp=0`, this is float('inf').
        precision (float): Precision score. When `num_tp+num_fp=0`, this is float('inf').
        recall (float): Recall score. When `num_ground_truth=0`, this is float('inf').
        f1score (float): F1 score. When `precision+recall=0`, this is float('inf').
        results (Dict[str, float]): Dict that items are scores mapped by score names.

    Args:
        object_results (List[DynamicObjectWithPerceptionResult]): Object results list.
        num_ground_truth (int): Number of ground truths.
        target_labels (List[LabelType]): Target labels list.
    """

    def __init__(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        num_ground_truth: int,
        target_labels: List[LabelType],
    ) -> None:
        self.num_ground_truth: int = num_ground_truth
        self.target_labels: List[LabelType] = target_labels
        if len(object_results) == 0 or not isinstance(object_results[0], list):
            all_object_results = object_results
        else:
            all_object_results = []
            for obj_results in object_results:
                all_object_results += obj_results
        self.objects_results_num: int = len(all_object_results)
        self.num_tp, self.num_fp = self.calculate_tp_fp(all_object_results)
        self.accuracy = self.calculate_accuracy(self.num_tp)
        self.precision, self.recall = self.calculate_precision_recall(self.num_tp)
        self.f1score = self.calculate_f1score(self.precision, self.recall)

    @property
    def results(self) -> Dict[str, float]:
        return OrderedDict(
            {
                "predict_num": self.objects_results_num,
                "Accuracy": self.accuracy,
                "Precision": self.precision,
                "Recall": self.recall,
                "F1score": self.f1score,
            }
        )

    def calculate_tp_fp(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
    ) -> Tuple[int, int]:
        """Calculate accuracy score.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): Object results list.

        Returns:
            num_tp (int): Number of TP results.
            num_fp (int): Number of FP results.
        """
        num_tp: int = 0
        num_fp: int = 0
        for obj_result in object_results:
            if obj_result.is_label_correct:
                num_tp += 1
            else:
                num_fp += 1
        return num_tp, num_fp

    def calculate_accuracy(self, num_tp: int) -> float:
        """Calculate accuracy score.

        Accuracy = (TP + TN) / (TP + TN + FP + FN)

        Args:
            num_tp (int): Number of TP results.

        Returns:
            float: Accuracy score. When `objects_results_num+num_ground_truth-num_tp=0`, returns float('inf').
        """
        return (
            num_tp / (self.objects_results_num + self.num_ground_truth - num_tp)
            if (self.objects_results_num + self.num_ground_truth - num_tp) != 0
            else float("inf")
        )

    def calculate_precision_recall(self, num_tp: int) -> Tuple[float, float]:
        """Calculate precision and recall scores.

        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)

        Args:
            num_tp (int): Number of TP results.

        Returns:
            precision (float): Precision score. When `self.object_results_num=0`, returns float('inf').
            recall (float): Recall score. When `self.num_ground_truth=0`, returns float('inf').
        """
        precision = num_tp / self.objects_results_num if self.objects_results_num != 0 else float("inf")
        recall = num_tp / self.num_ground_truth if self.num_ground_truth != 0 else float("inf")
        return precision, recall

    def calculate_f1score(self, precision: float, recall: float, beta: float = 1.0) -> float:
        """Calculate F1 score.

        F score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

        Args:
            precision (float): Precision score.
            recall (float): Recall score.
            beta (float): Defaults 1.0.

        Returns:
            f1score (float): F1 score. When `precision+recall=0`, returns float('inf').
        """
        return (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            if (precision != float("inf") and recall != float("inf") and (beta**2 * precision + recall) != 0)
            else float("inf")
        )
