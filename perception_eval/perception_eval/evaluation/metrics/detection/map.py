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

from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.matching.object_matching import MatchingMode
from perception_eval.evaluation.metrics.detection.ap import Ap
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class Map:
    """[summary]
    mAP class

    Attributes:
        self.map_config (MapConfig): The config for mAP calculation
        self.aps (List[Ap]): The list of AP (Average Precision) for each label
        self.map (float): mAP value
    """

    def __init__(
        self,
        object_results_dict: Dict[AutowareLabel, List[DynamicObjectWithPerceptionResult]],
        num_ground_truth_dict: Dict[AutowareLabel, int],
        target_labels: List[AutowareLabel],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
        is_detection_2d: bool = False,
    ) -> None:
        """[summary]

        Args:
            object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object results
            target_labels (List[AutowareLabel]): Target labels to evaluate mAP
            matching_mode (MatchingMode): Matching mode like distance between the center of
                                           the object, 3d IoU.
            matching_threshold_list (List[float]):
                    The matching threshold to evaluate. Defaults to None.
                    For example, if matching_mode = IOU3d and matching_threshold = 0.5,
                    and IoU of the object is higher than "matching_threshold",
                    this function appends to return objects.
        """
        self.target_labels: List[AutowareLabel] = target_labels
        self.matching_mode: MatchingMode = matching_mode
        self.matching_threshold_list: List[float] = matching_threshold_list
        self.is_detection_2d: bool = is_detection_2d

        # calculate AP & APH
        self.aps: List[Ap] = []
        self.aphs: List[Ap] = []
        sum_ap: float = 0.0
        sum_aph: float = 0.0
        for target_label, matching_threshold in zip(target_labels, matching_threshold_list):
            object_results = object_results_dict[target_label]
            num_ground_truth = num_ground_truth_dict[target_label]
            ap_ = Ap(
                tp_metrics=TPMetricsAp(),
                object_results=object_results,
                num_ground_truth=num_ground_truth,
                target_labels=[target_label],
                matching_mode=matching_mode,
                matching_threshold_list=[matching_threshold],
            )
            self.aps.append(ap_)
            sum_ap += ap_.ap

            if not self.is_detection_2d:
                aph_ = Ap(
                    tp_metrics=TPMetricsAph(),
                    object_results=object_results,
                    num_ground_truth=num_ground_truth,
                    target_labels=[target_label],
                    matching_mode=matching_mode,
                    matching_threshold_list=[matching_threshold],
                )
                self.aphs.append(aph_)
                sum_aph += aph_.ap

        # calculate mAP & mAPH
        self.map: float = sum_ap / len(target_labels)
        self.maph: float = sum_aph / len(target_labels)

    def __str__(self) -> str:
        """__str__ method"""

        str_: str = "\n"
        str_ += f"mAP: {self.map:.3f}"
        str_ += ", mAPH: {self.maph:.3f} " if not self.is_detection_2d else " "
        str_ += f"({self.matching_mode.value})\n"
        # Table
        str_ += "\n"
        # label
        str_ += "|      Label |"
        target_str: str
        for ap_ in self.aps:
            # len labels and threshold_list is always 1
            str_ += f" {ap_.target_labels[0].value}({ap_.matching_threshold_list[0]}) | "
        str_ += "\n"
        str_ += "| :--------: |"
        for ap_ in self.aps:
            str_ += " :---: |"
        str_ += "\n"
        str_ += "| Predict_num |"
        for ap_ in self.aps:
            str_ += f" {ap_.objects_results_num} |"
        # Each label result
        str_ += "\n"
        str_ += "|         AP |"
        for ap_ in self.aps:
            str_ += f" {ap_.ap:.3f} | "
        str_ += "\n"
        if not self.is_detection_2d:
            str_ += "|        APH |"
            for aph_ in self.aphs:
                target_str = ""
                for target in aph_.target_labels:
                    target_str += target.value
                str_ += f" {aph_.ap:.3f} | "
            str_ += "\n"

        return str_
