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

from perception_eval.common.traffic_light import TLColor
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

from .accuracy import ClassificationAccuracy


class ClassificationMetricsScore:
    def __init__(
        self,
        object_results_dict: Dict[TLColor, List[List[DynamicObjectWithPerceptionResult]]],
        num_ground_truth_dict: Dict[TLColor, int],
        target_labels: List[TLColor],
    ) -> None:
        self.accuracies: List[ClassificationAccuracy] = []
        for color in target_labels:
            object_results = object_results_dict[color]
            num_ground_truth = num_ground_truth_dict[color]

            acc_: ClassificationAccuracy = ClassificationAccuracy(
                object_results=object_results,
                num_ground_truth=num_ground_truth,
                target_labels=[color],
            )
            self.accuracies.append(acc_)

    def _summarize(self) -> float:
        acc: float = 0.0
        tsca: float = 0.0
        num_gt: int = 0
        num_tp: int = 0
        num_id_switch: int = 0
        for acc_ in self.accuracies:
            if acc_.accuracy != float("inf"):
                acc += acc_.accuracy * acc_.num_ground_truth
            if acc_.tsca != float("inf"):
                tsca += acc_.tsca * acc_.num_ground_truth
            num_gt += acc_.num_ground_truth
            num_tp += int(acc_.tp)
            num_id_switch += acc_.id_switch
        acc: float = float("inf") if num_gt == 0 else acc / num_gt
        tsca: float = float("inf") if num_gt == 0 else tsca / num_gt
        return acc, tsca

    def __str__(self) -> str:
        """__str__ method"""
        str_: str = "\n"
        # === Total ===
        acc, tsca = self._summarize()
        str_ += f"[TOTAL] ACC: {acc:.3f}, TSCA: {tsca:.3f}"
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
        for acc_ in self.accuracies:
            score: float = acc_.accuracy
            str_ += f" {score:.3f} |"
        str_ += "\n"
        for acc_ in self.accuracies:
            score: float = acc_.tsca
            str_ += f" {score:.3f} |"
        str_ += "\n"

        return str_
