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

import numpy as np
from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.matching.object_matching import MatchingMode
from perception_eval.evaluation.metrics.prediction import PathDisplacementError
from perception_eval.evaluation.metrics.prediction import SoftAp
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class PredictionMetricsScore:
    """Metrics score for prediction.

    Attributes:
        self.target_labels (List[AutowareLabel]): List of target labels.
        self.matching_mode (MatchingMode): MatchingMode instance.
        self.displacements (List[PathDisplacementError]): List of PathDisplacementError instances.
        self.soft_aps (SoftAp): List of SoftAp instances.
        self.ade (float): Average Displacement Error;ADE score.
        self.fde (float): Final Displacement Error;FDE score.
        self.miss_rate (float): Miss rate.
        self.soft_map (float): Soft mAP score.
    """

    def __init__(
        self,
        object_results_dict: Dict[AutowareLabel, List[List[DynamicObjectWithPerceptionResult]]],
        num_ground_truth_dict: Dict[AutowareLabel, int],
        target_labels: List[AutowareLabel],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
    ) -> None:
        """[summary]
        Args:
            object_results_dict (Dict[AutowareLabel, List[List[DynamicObjectWithPerceptionResult]]):
                object results divided by label for multi frame.
            num_ground_truth (int): The number of ground truth.
            target_labels (List[AutowareLabel]): e.g. ["car", "pedestrian", "bus"]
            matching_mode (MatchingMode): The target matching mode.
            matching_threshold_list (List[float]): The list of matching threshold for each category. (e.g. [0.5, 0.3, 0.5])
        """
        assert len(target_labels) == len(matching_threshold_list)
        self.target_labels: List[AutowareLabel] = target_labels
        self.matching_mode: MatchingMode = matching_mode

        self.displacements: List[PathDisplacementError] = []
        self.soft_aps: List[SoftAp] = []
        for target_label, matching_threshold in zip(target_labels, matching_threshold_list):
            object_results = object_results_dict[target_label]
            num_ground_truth = num_ground_truth_dict[target_label]
            displacement_err = PathDisplacementError(
                object_results=object_results,
                num_ground_truth=num_ground_truth,
                target_labels=[target_label],
                matching_mode=self.matching_mode,
                matching_threshold_list=[matching_threshold],
            )
            soft_ap = SoftAp(
                object_results=object_results,
                num_ground_truth=num_ground_truth,
                target_labels=[target_label],
                matching_mode=self.matching_mode,
                matching_threshold_list=[matching_threshold],
            )
            self.displacements.append(displacement_err)
            self.soft_aps.append(soft_ap)
        self._summarize_score()

    def _summarize_score(self) -> None:
        """[summary]
        Summarize score
        """
        ade_list, fde_list, miss_list = [], [], []
        for err in self.displacements:
            if ~np.isnan(err.ade):
                ade_list.append(err.ade)
            if ~np.isnan(err.fde):
                fde_list.append(err.fde)
            if ~np.isnan(err.miss_rate):
                miss_list.append(err.miss_rate)
        ap_list: List[float] = [ap.ap for ap in self.soft_aps if ~np.isnan(ap.ap)]

        self.ade: float = np.mean(ade_list) if 0 < len(ade_list) else np.nan
        self.fde: float = np.mean(fde_list) if 0 < len(fde_list) else np.nan
        self.miss_rate: float = np.mean(miss_list) if 0 < len(miss_list) else np.nan
        self.soft_map = np.mean(ap_list) if 0 < len(ap_list) else np.nan

    def __str__(self) -> str:
        """__str__ method"""

        str_: str = "\n"
        str_ += f"ADE: {self.ade:.3f}, FDE: {self.fde:.3f}, Miss Rate: {self.miss_rate:.3f}, Soft mAP: {self.soft_map:.3f}"
        str_ += f"({self.matching_mode.value})\n"
        # Table
        str_ += "\n"
        # label
        str_ += "|      Label |"
        target_str: str
        for err in self.displacements:
            str_ += f" {err.target_labels[0].value}({err.matching_threshold_list[0]}) | "
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
        str_ += "|     Soft AP |"
        for ap in self.soft_aps:
            str_ += f" {ap.ap:.3f} | "
        str_ += "\n"

        return str_
