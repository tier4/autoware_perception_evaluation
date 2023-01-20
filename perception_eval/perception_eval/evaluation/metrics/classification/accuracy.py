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
        self.target_labels (List[LabelType])
        self.num_ground_truth (int)
        self.objects_results_num (int)
        self.num_tp (int)
        self.num_fp (int)
        self.accuracy (float)
        self.precision (float)
        self.recall (float)
        self.f1score (float)
        self.results (Dict[str, float])
    """

    def __init__(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        num_ground_truth: int,
        target_labels: List[LabelType],
    ) -> None:
        """
        Args:
            object_results (List[DynamicObjectWithPerceptionResult])
            num_ground_truth (int)
            target_labels (List[LabelType])
        """
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
    ) -> float:
        """[summary]
        Calculate accuracy score.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult])

        Returns:
            num_tp (int): Number of TP.
            num_fp (int): Number of FP.
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
        """[summary]

        Accuracy = (TP + TN) / (TP + TN + FP + FN)

        Args:
            num_tp (int): Number of TP.

        Returns:
            accuracy (float)
        """
        return (
            num_tp / (self.objects_results_num + self.num_ground_truth - num_tp)
            if (self.objects_results_num + self.num_ground_truth - num_tp) != 0
            else float("inf")
        )

    def calculate_precision_recall(self, num_tp: int) -> Tuple[float, float]:
        """[summary]

        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)

        Args:
            num_tp (int)

        Returns:
            precision (float)
            recall (float)
        """
        precision = (
            num_tp / self.objects_results_num if self.objects_results_num != 0 else float("inf")
        )
        recall = num_tp / self.num_ground_truth if self.num_ground_truth != 0 else float("inf")
        return precision, recall

    def calculate_f1score(self, precision: float, recall: float, beta: float = 1.0) -> float:
        """[summary]

        F score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

        Args:
            precision (float)
            recall (float)
            beta (float): Defaults 1.0.

        Returns:
            f1score (float)
        """
        return (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            if (
                precision != float("inf")
                and recall != float("inf")
                and (beta**2 * precision + recall) != 0
            )
            else float("inf")
        )
