# Copyright 2025 TIER IV, Inc.

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
from logging import getLogger
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

logger = getLogger(__name__)

_UNMATCHED_LABEL = "unmatched"


@dataclass(frozen=True)
class ConfusionMatrixData:
    label: LabelType
    total_gt_nums: int
    total_tp_nums: int
    matched_boxes: Dict[str, int]  # {label: number of matched predicted boxes}

    @property
    def total_fn_nums(self) -> int:
        return self.total_gt_nums - self.total_tp_nums

    @property
    def total_prediction_nums(self) -> int:
        return sum(self.matched_boxes.values())

    @property
    def total_fp_nums(self) -> int:
        return self.total_prediction_nums - self.total_tp_nums


class ConfusionMatrix:
    """
    Class to visualize a confusion matrix across all labels.

    mAP evaluation class supporting multiple thresholds per label.

    This class calculates Average Precision (AP) and Average Precision with Heading (APH)
    for a set of perception results grouped by label and matching threshold.

    For each label:
        - It computes AP and optionally APH for all given matching thresholds.
        - It then calculates the mean AP (and APH) across thresholds for that label.

    Finally:
        - It averages the per-label mean AP (and APH) across all target labels
          to produce the final mAP and mAPH.

    This class supports both 2D and 3D detection evaluation:
        - In 2D detection, only AP is calculated (APH is skipped).
        - In 3D detection, both AP and APH are calculated.

    Attributes:
        target_labels (List[LabelType]):
            List of target labels evaluated in this instance.
        matching_mode (MatchingMode):
            The matching strategy used for TP/FP calculation (e.g., CENTERDISTANCE, IOU3D).
        is_detection_2d (bool):
            If True, only AP is computed; APH is skipped.
        label_to_aps (Dict[LabelType, List[Ap]]):
            List of AP instances (one per threshold) for each label.
        label_mean_to_ap (Dict[LabelType, float]):
            Mean AP across thresholds for each label. Can be NaN if all AP values are NaN.
        label_to_aphs (Optional[Dict[LabelType, List[Ap]]]):
            List of APH instances (one per threshold) for each label (if 3D detection).
        label_mean_to_aph (Optional[Dict[LabelType, float]]):
            Mean APH across thresholds for each label (if 3D detection). Can be NaN if all APH values are NaN.
        map (float):
            Final mean Average Precision (mAP) across all labels. Can be NaN if all label means are NaN.
        maph (Optional[float]):
            Final mean Average Precision with Heading (mAPH) across all labels,
            or None if `is_detection_2d` is True. Can be NaN if all label means are NaN.

    """

    def __init__(
        self,
        object_results_dict: Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]],
        num_ground_truth_dict: Dict[LabelType, int],
        target_labels: List[LabelType],
        matching_mode: MatchingMode,
    ) -> None:
        self.object_results_dict = object_results_dict
        self.num_ground_truth_dict = num_ground_truth_dict
        self.target_labels = target_labels
        self.matching_mode = matching_mode
        self.matching_thresholds_confusion_matrices = self.get_confusion_matrix()

    def get_confusion_matrix(self) -> Dict[float, Dict[str, Dict[str, int]]]:
        """
        Compute confusion matrix in an array of target_labels x target_labels,
        where the row is prediction and gt is ground truth.
        """
        # confusion_matrix: {matching_threshold: {predicted_label: ConfusionMatrixData}}}
        matching_threshold_confusion_matrices = defaultdict(defaultdict(ConfusionMatrixData))

        for label in self.target_labels:
            for threshold, object_results in self.object_results_dict[label].items():
                total_tp_nums = 0
                matched_boxes = defaultdict(int)
                for object_result in object_results:
                    predicted_object_label = object_result.semantic_label.name

                    if object_result.ground_truth_object is not None:
                        if object_result.is_label_correct:
                            total_tp_nums += 1
                            matched_boxes[predicted_object_label] += 1
                    else:
                        matched_boxes[_UNMATCHED_LABEL] += 1

                if _UNMATCHED_LABEL not in matched_boxes:
                    matched_boxes[_UNMATCHED_LABEL] = 0

                total_gt_nums = self.num_ground_truth_dict[label]
                matching_threshold_confusion_matrices[threshold][label] = ConfusionMatrixData(
                    label=label, total_tp_nums=total_tp_nums, total_gt_nums=total_gt_nums, matched_boxes=matched_boxes
                )

        return matching_threshold_confusion_matrices

    # def convert_to_table_str(self) -> str:
    #     """
    #     Convert confusion matrix to a string table for visualization.
    #     """
    #     table_str = ""
    #     for threshold, confusion_matrix in self.matching_thresholds_confusion_matrices.items():
    #         header = f"Confusion Matrix (Matching Threshold: {threshold})\n"
    #         table_str += header
    #         labels = list(confusion_matrix.keys())
    #         labels.append(_UNMATCHED_LABEL)
    #         header_row = "GT \\ Predictions | " + " | ".join(f"{label:15}" if label is not _UNMATCHED_LABEL else {label} for label in labels) + "\n"
    #         table_str += header_row
    #         table_str += "-" * len(header_row) + "\n"

    #         for pred_label, data in confusion_matrix.items():
    #             if pred_label == _UNMATCHED_LABEL:
    #                 pred_label += " (FP)"
    #             row = f"{pred_label:15} | "
    #             for gt_label in labels:
    #                 if gt_label == _UNMATCHED_LABEL:
    #                     count = data.matched_boxes.get(_UNMATCHED_LABEL, 0)
    #                 else:
    #                     count = data.matched_boxes.get(gt_label, 0)
    #                 row += f"{count:15} | "
    #             table_str += row + "\n"
    #         table_str += "\n"
    #     return table_str
