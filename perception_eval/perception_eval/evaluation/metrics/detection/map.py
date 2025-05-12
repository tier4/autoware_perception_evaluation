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

from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.detection.ap import Ap
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class Map:
    """mAP metrics score class.

    Attributes:
        map_config (MapConfig): The config for mAP calculation.
        aps (List[Ap]): The list of AP (Average Precision) for each label.
        map (float): mAP value.

    Args:
        object_results_dict (Dict[LabelType, List[DynamicObjectWithPerceptionResult]]): The [label, object results] dictionary
        target_labels (List[LabelType]): Target labels to evaluate mAP
        matching_mode (MatchingMode): Matching mode like distance between the center of
            the object, 3d IoU.
        matching_threshold_list (List[float]):
            The matching threshold to evaluate. Defaults to None.
            For example, if matching_mode = IOU3d and matching_threshold = 0.5,
            and IoU of the object is higher than "matching_threshold",
            this function appends to return objects.
    """

    def __init__(
        self,
        object_results_dict: Dict[LabelType, List[DynamicObjectWithPerceptionResult]],
        num_ground_truth_dict: Dict[LabelType, int],
        target_labels: List[LabelType],
        matching_mode: MatchingMode,
        matching_threshold_list: List[List[float]],
        is_detection_2d: bool = False,
    ) -> None:
        self.num_ground_truth_dict: Dict[LabelType, int] = num_ground_truth_dict
        self.target_labels: List[LabelType] = target_labels
        self.matching_mode: MatchingMode = matching_mode
        # TODO(vividf): matching_threshold_list to matching_thresholds
        # Also thresholds should be refactor to List[float]
        self.matching_threshold_list: List[List[float]] = matching_threshold_list
        self.is_detection_2d: bool = is_detection_2d

        # calculate AP & APH
        self.aps: List[Ap] = []
        self.aphs: List[Ap] = []
        for target_label, matching_threshold in zip(target_labels, matching_threshold_list):
            if target_label not in object_results_dict:
                object_results = []
            else:
                object_results = object_results_dict[target_label]
            num_ground_truth = num_ground_truth_dict[target_label]
            ap_ = Ap(
                tp_metrics=TPMetricsAp(),
                object_results=object_results,
                num_ground_truth=num_ground_truth,
                target_labels=[target_label],
                matching_mode=matching_mode,
                matching_threshold=matching_threshold,
            )
            self.aps.append(ap_)

            if not self.is_detection_2d:
                aph_ = Ap(
                    tp_metrics=TPMetricsAph(),
                    object_results=object_results,
                    num_ground_truth=num_ground_truth,
                    target_labels=[target_label],
                    matching_mode=matching_mode,
                    matching_threshold=matching_threshold,
                )
                self.aphs.append(aph_)

        for ap in self.aps:
            print("ap: ", ap.ap)

        for aph in self.aphs:
            print("ap: ", aph.ap)

        self.map = sum(ap.ap for ap in self.aps) / len(self.aps) if self.aps else 0
        self.maph = sum(aph.ap for aph in self.aphs) / len(self.aphs) if self.aphs else 0

    def __str__(self) -> str:
        """__str__ method

        Returns:
            str: Formatted string.
        """
        str_: str = ""
        str_ += f"\nmAP: {self.map:.3f}, "
        str_ += f"mAPH: {self.maph:.3f} " if not self.is_detection_2d else ""
        str_ += f"({self.matching_mode.value})\n\n"

        # Header
        str_ += "|      Label      |"
        for ap_ in self.aps:
            label = ap_.target_labels[0].value
            threshold = ap_.matching_threshold
            str_ += f" {label}({threshold}) |"
        str_ += "\n"

        # Separator
        str_ += "|:---------------:|" + ":------------:|" * len(self.aps) + "\n"

        # Predict_num
        str_ += "|   Predict_num   |"
        for ap_ in self.aps:
            str_ += f" {ap_.objects_results_num:^12} |"
        str_ += "\n"

        # Ground Truth Num
        str_ += "| GroundTruth_num |"
        for ap_ in self.aps:
            label = ap_.target_labels[0]
            gt_num = self.num_ground_truth_dict.get(label, 0)
            str_ += f" {gt_num:^12} |"
        str_ += "\n"

        # AP
        str_ += "|       AP        |"
        for ap_ in self.aps:
            str_ += f" {ap_.ap:^12.3f} |"
        str_ += "\n"

        # APH
        if not self.is_detection_2d:
            str_ += "|      APH        |"
            for aph_ in self.aphs:
                str_ += f" {aph_.ap:^12.3f} |"
            str_ += "\n"

        return str_
