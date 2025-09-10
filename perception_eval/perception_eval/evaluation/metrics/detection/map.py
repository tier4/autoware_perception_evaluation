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

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.detection.ap import Ap
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class Map:
    """
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
        is_detection_2d: bool = False,
    ) -> None:
        self.object_results_dict = object_results_dict
        self.num_ground_truth_dict = num_ground_truth_dict
        self.target_labels = target_labels
        self.matching_mode = matching_mode
        self.is_detection_2d = is_detection_2d

        self.label_to_aps: Dict[LabelType, List[Ap]] = {}
        self.label_mean_to_ap: Dict[LabelType, float] = {}
        self.label_to_aphs: Dict[LabelType, List[Ap]] = {} if not self.is_detection_2d else None
        self.label_mean_to_aph: Dict[LabelType, float] = {} if not self.is_detection_2d else None

        for label in target_labels:
            ap_per_threshold = []
            aph_per_threshold = []

            for threshold, object_results in self.object_results_dict[label].items():
                num_ground_truth = num_ground_truth_dict[label]
                ap = Ap(
                    tp_metrics=TPMetricsAp(),
                    object_results=object_results,
                    num_ground_truth=num_ground_truth,
                    target_label=label,
                    matching_mode=matching_mode,
                    matching_threshold=threshold,
                )
                ap_per_threshold.append(ap)

                if not self.is_detection_2d:
                    aph = Ap(
                        tp_metrics=TPMetricsAph(),
                        object_results=object_results,
                        num_ground_truth=num_ground_truth,
                        target_label=label,
                        matching_mode=matching_mode,
                        matching_threshold=threshold,
                    )
                    aph_per_threshold.append(aph)

            self.label_to_aps[label] = ap_per_threshold
            self.label_mean_to_ap[label] = self._mean([ap.ap for ap in ap_per_threshold])

            if not self.is_detection_2d:
                self.label_to_aphs[label] = aph_per_threshold
                self.label_mean_to_aph[label] = self._mean([aph.ap for aph in aph_per_threshold])

        self.map: float = self._mean(list(self.label_mean_to_ap.values()))
        self.maph: float = self._mean(list(self.label_mean_to_aph.values())) if not self.is_detection_2d else None

    def __reduce__(self) -> Tuple[Map, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling."""
        init_args = (
            self.object_results_dict,
            self.num_ground_truth_dict,
            self.target_labels,
            self.matching_mode,
            self.is_detection_2d,
        )
        return (
            self.__class__,
            init_args,
        )

    def __str__(self) -> str:
        str_ = ""
        map_str = f"{self.map:.3f}" if not (isinstance(self.map, float) and np.isnan(self.map)) else "NaN"
        str_ += f"\nmAP: {map_str}, "
        if not self.is_detection_2d:
            maph_str = f"{self.maph:.3f}" if not (isinstance(self.maph, float) and np.isnan(self.maph)) else "NaN"
            str_ += f"mAPH: {maph_str} "
        str_ += f"({self.matching_mode.value})\n"

        # === Per-label AP Table ===
        for label in self.target_labels:
            str_ += f"\nLabel: {label.value}\n"
            str_ += "| Threshold | Predict_num | Groundtruth_num |     AP     |"
            if not self.is_detection_2d:
                str_ += "    APH    |"
            str_ += "   max_f1   |      optimal_conf      |"
            str_ += "\n"

            str_ += "|:---------:|:------------:|:----------------:|:----------:|"
            if not self.is_detection_2d:
                str_ += ":---------:|"
            str_ += ":----------:|:----------------------:|"
            str_ += "\n"

            aps = self.label_to_aps[label]
            aphs = self.label_to_aphs.get(label, []) if not self.is_detection_2d else []
            gt_num = self.num_ground_truth_dict[label]

            for ap in aps:
                threshold = ap.matching_threshold
                predict_num = ap.objects_results_num
                ap_str = f"{ap.ap:^8.3f}" if not (isinstance(ap.ap, float) and np.isnan(ap.ap)) else "   NaN   "
                str_ += f"|  {threshold:^8.2f} | {predict_num:^12} | {gt_num:^16} |  {ap_str} |"

                if not self.is_detection_2d:
                    aph = next((a for a in aphs if a.matching_threshold == threshold), None)
                    if aph:
                        aph_str = (
                            f"{aph.ap:^8.3f}" if not (isinstance(aph.ap, float) and np.isnan(aph.ap)) else "   NaN   "
                        )
                        str_ += f"  {aph_str} |"
                    else:
                        str_ += " {:^8} |".format("N/A")
                str_ += f" {ap.max_f1_score:^8.4f} | {ap.optimal_conf:^12.6f} |"
                str_ += "\n"

        # === Summary Table ===
        str_ += "\nSummary:\n"
        str_ += "|      Label      |   Thresholds   |  Mean AP   |"
        if not self.is_detection_2d:
            str_ += "  Mean APH  |"
        str_ += "\n"

        str_ += "|:---------------:|:--------------:|:-----------:|"
        if not self.is_detection_2d:
            str_ += ":-----------:|"
        str_ += "\n"

        for label in self.target_labels:
            thresholds = [f"{ap.matching_threshold:.2f}" for ap in self.label_to_aps[label]]
            mean_ap = self.label_mean_to_ap[label]
            mean_ap_str = f"{mean_ap:^9.3f}" if not (isinstance(mean_ap, float) and np.isnan(mean_ap)) else "   NaN   "
            str_ += f"| {label.value:^15} | {'/'.join(thresholds):^14} |  {mean_ap_str} |"
            if not self.is_detection_2d:
                mean_aph = self.label_mean_to_aph[label]
                mean_aph_str = (
                    f"{mean_aph:^9.3f}" if not (isinstance(mean_aph, float) and np.isnan(mean_aph)) else "   NaN   "
                )
                str_ += f"  {mean_aph_str} |"
            str_ += "\n"

        # Summary about the best
        str_ += "\n"

        return str_

    @staticmethod
    def _mean(values: List[float]) -> float:
        # Filter out NaN values
        valid_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]

        # If no valid values, return NaN
        if not valid_values:
            return np.nan

        # Return the mean of valid values
        return sum(valid_values) / len(valid_values)
