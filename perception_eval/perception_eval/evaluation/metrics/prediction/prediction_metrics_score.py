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

from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import numpy as np
from perception_eval.common.label import LabelType
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

from .path_displacement_error import PathDisplacementError


class PredictionMetricsScore:
    """Metrics score manager for motion prediction task."""

    def __init__(
        self,
        object_results_dict: Dict[LabelType, List[List[DynamicObjectWithPerceptionResult]]],
        num_ground_truth_dict: Dict[LabelType, int],
        target_labels: List[LabelType],
        top_k: int = 3,
        miss_tolerance: float = 2.0,
        kernel: Optional[Literal["min", "max", "highest"]] = None,
    ) -> None:
        """Construct a new object.

        Args:
            object_results_dict (Dict[LabelType, List[List[DynamicObjectWithPerceptionResult]]):
                object results divided by label for multi frame.
            num_ground_truth (int): The number of ground truth.
            target_labels (List[LabelType]): List of target label names.
            top_k (int, optional): The number of top K to be evaluated. Defaults to 1.
            miss_tolerance (float, optional): Threshold to determine miss. Defaults to 2.0.
            kernel (Optional[Literal["min", "max", "highest"]], optional): Kernel to evaluate displacement errors.
                "min" evaluates the minimum displacements at each time step.
                "max" evaluates the maximum at each time step.
                "highest" evaluates the highest confidence mode. Defaults to None.
        """
        self.target_labels = target_labels
        self.top_k = top_k
        self.miss_tolerance = miss_tolerance

        self.displacements: List[PathDisplacementError] = []
        for target_label in target_labels:
            object_results = object_results_dict[target_label]
            num_ground_truth = num_ground_truth_dict[target_label]
            displacement_err = PathDisplacementError(
                object_results=object_results,
                num_ground_truth=num_ground_truth,
                target_labels=[target_label],
                top_k=top_k,
                miss_tolerance=miss_tolerance,
                kernel=kernel,
            )
            self.displacements.append(displacement_err)
        self._summarize_score()

    def _summarize_score(self) -> None:
        """Summarize scores."""
        ade_list, fde_list, miss_list = [], [], []
        for err in self.displacements:
            if ~np.isnan(err.ade):
                ade_list.append(err.ade)
            if ~np.isnan(err.fde):
                fde_list.append(err.fde)
            if ~np.isnan(err.miss_rate):
                miss_list.append(err.miss_rate)

        self.ade: float = np.mean(ade_list) if 0 < len(ade_list) else np.nan
        self.fde: float = np.mean(fde_list) if 0 < len(fde_list) else np.nan
        self.miss_rate: float = np.mean(miss_list) if 0 < len(miss_list) else np.nan

    def __str__(self) -> str:
        """__str__ method"""

        str_: str = "\n"
        str_ += f"ADE: {self.ade:.3f}, FDE: {self.fde:.3f}, Miss Rate: {self.miss_rate:.3f}"
        str_ += f" (Miss Tolerance: {self.miss_tolerance}[m])"
        # Table
        str_ += "\n"
        # label
        str_ += "|      Label |"
        for err in self.displacements:
            str_ += f" {err.target_labels[0].value}(k={err.top_k}) | "
        str_ += "\n"
        str_ += "| :--------: |"
        for err in self.displacements:
            str_ += " :---: |"
        str_ += "\n"
        str_ += "| Predict_num |"
        for err in self.displacements:
            str_ += f" {err.objects_results_num} |"
        str_ += "\n"
        str_ += "|         ADE |"
        for err in self.displacements:
            str_ += f" {err.ade:.3f} | "
        str_ += "\n"
        str_ += "|         FDE |"
        for err in self.displacements:
            str_ += f" {err.fde:.3f} | "
        str_ += "\n"
        str_ += "|   Miss Rate |"
        for err in self.displacements:
            str_ += f" {err.miss_rate:.3f} | "
        str_ += "\n"

        return str_
