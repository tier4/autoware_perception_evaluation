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
from perception_eval.evaluation.metrics.detection.nds import NuScenesDetectionScore
from perception_eval.evaluation.metrics.detection.tp_error_metrics import TPErrorAttribute
from perception_eval.evaluation.metrics.detection.tp_error_metrics import TPErrorBEVCenterDistance
from perception_eval.evaluation.metrics.detection.tp_error_metrics import TPErrorBEVVelocity
from perception_eval.evaluation.metrics.detection.tp_error_metrics import TPErrorOrientation
from perception_eval.evaluation.metrics.detection.tp_error_metrics import TPErrorScale
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

_TP_ERROR_METRIC_TYPES = (
    TPErrorBEVCenterDistance,
    TPErrorOrientation,
    TPErrorScale,
    TPErrorBEVVelocity,
    TPErrorAttribute,
)


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
        label_mean_to_tp_error (Optional[Dict[str, Dict[LabelType, float]]]):
            Per-label mean TP error across thresholds for each metric (e.g. mATE), or None if 2D.
        mean_tp_errors (Optional[Dict[str, float]]):
            Final mean TP errors (mATE, mAOE, etc.) across all labels, or None if 2D.
        optimal_mean_tp_errors (Optional[Dict[str, float]]):
            Final mean TP errors at the F1-optimal confidence, or None if 2D.
        medium_mean_tp_errors (Optional[Dict[str, float]]):
            Final mean TP errors over the medium recall band (min_recall=0.4), or None if 2D.
        map_based_nds (Optional[NuScenesDetectionScore]):
            NuScenes Detection Score using mAP and mean TP errors, or None if 2D.
        mapH_based_nds (Optional[NuScenesDetectionScore]):
            NuScenes Detection Score using mAPH and mean TP errors, or None if 2D.
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

        self.avg_tp_error_names: Dict[str, int] = {
            metric_type.average_mode: index for index, metric_type in enumerate(_TP_ERROR_METRIC_TYPES)
        }
        self.mean_tp_error_names: Dict[str, int] = {
            metric_type.mean_average_mode: index for index, metric_type in enumerate(_TP_ERROR_METRIC_TYPES)
        }
        self.label_mean_to_tp_error: Dict[str, Dict[LabelType, float]] | None = (
            {name: {} for name in self.mean_tp_error_names} if not self.is_detection_2d else None
        )
        self.label_mean_to_medium_tp_error: Dict[str, Dict[LabelType, float]] | None = (
            {name: {} for name in self.mean_tp_error_names} if not self.is_detection_2d else None
        )
        self.label_mean_to_optimal_tp_error: Dict[str, Dict[LabelType, float]] | None = (
            {name: {} for name in self.mean_tp_error_names} if not self.is_detection_2d else None
        )
        self.mean_tp_errors: Dict[str, float] | None = None if self.is_detection_2d else {}
        self.medium_mean_tp_errors: Dict[str, float] | None = None if self.is_detection_2d else {}
        self.optimal_mean_tp_errors: Dict[str, float] | None = None if self.is_detection_2d else {}

        for label in target_labels:
            ap_per_threshold = []
            aph_per_threshold = []

            for threshold, object_results in self.object_results_dict[label].items():
                num_ground_truth = num_ground_truth_dict[label]
                tp_error_metrics = [metric_type() for metric_type in _TP_ERROR_METRIC_TYPES] if not self.is_detection_2d else None
                ap = Ap(
                    tp_metrics=TPMetricsAp(),
                    object_results=object_results,
                    num_ground_truth=num_ground_truth,
                    target_label=label,
                    matching_mode=matching_mode,
                    matching_threshold=threshold,
                    tp_error_metrics=tp_error_metrics,
                )
                ap_per_threshold.append(ap)

                if not self.is_detection_2d:
                    aph = Ap(
                        tp_metrics=TPMetricsAph(),
                        object_results=object_results,
                        num_ground_truth=num_ground_truth,
                        target_label=label,
                        matching_mode=matching_mode,
                        matching_threshold=threshold
                    )
                    aph_per_threshold.append(aph)

            self.label_to_aps[label] = ap_per_threshold
            self.label_mean_to_ap[label] = self._mean([ap.ap for ap in ap_per_threshold])

            if not self.is_detection_2d:
                self.label_to_aphs[label] = aph_per_threshold
                self.label_mean_to_aph[label] = self._mean([aph.ap for aph in aph_per_threshold])

        self.map: float = self._mean(list(self.label_mean_to_ap.values()))
        self.maph: float = self._mean(list(self.label_mean_to_aph.values())) if not self.is_detection_2d else None

        self.map_based_nds: NuScenesDetectionScore | None = None
        self.mapH_based_nds: NuScenesDetectionScore | None = None
        self.medium_map_based_nds: NuScenesDetectionScore | None = None
        self.medium_mapH_based_nds: NuScenesDetectionScore | None = None

        if not self.is_detection_2d:
            for mean_tp_error_name, tp_error_index in self.mean_tp_error_names.items():
                label_means: List[float] = []
                medium_label_means: List[float] = []
                optimal_label_means: List[float] = []
                for label in target_labels:
                    aps_for_label = self.label_to_aps[label]
                    per_threshold = [ap.tp_error_metrics[tp_error_index].avg_metric for ap in aps_for_label]
                    label_mean = self._mean(per_threshold)
                    self.label_mean_to_tp_error[mean_tp_error_name][label] = label_mean
                    label_means.append(label_mean)

                    medium_per_threshold = [
                        ap.tp_error_metrics[tp_error_index].medium_avg_metric for ap in aps_for_label
                    ]
                    medium_label_mean = self._mean(medium_per_threshold)
                    self.label_mean_to_medium_tp_error[mean_tp_error_name][label] = medium_label_mean
                    medium_label_means.append(medium_label_mean)

                    optimal_per_threshold = [
                        ap.tp_error_metrics[tp_error_index].optimal_avg_metric for ap in aps_for_label
                    ]
                    optimal_label_mean = self._mean(optimal_per_threshold)
                    self.label_mean_to_optimal_tp_error[mean_tp_error_name][label] = optimal_label_mean
                    optimal_label_means.append(optimal_label_mean)

                self.mean_tp_errors[mean_tp_error_name] = self._mean(label_means)
                self.medium_mean_tp_errors[mean_tp_error_name] = self._mean(medium_label_means)
                self.optimal_mean_tp_errors[mean_tp_error_name] = self._mean(optimal_label_means)

            self.map_based_nds = NuScenesDetectionScore(
                map=self.map,
                metric_prefix_name=f"map_based",
                mean_tp_error_metrics=self.mean_tp_errors,
            )
            self.mapH_based_nds = NuScenesDetectionScore(
                map=self.maph,
                metric_prefix_name=f"mapH_based",
                mean_tp_error_metrics=self.mean_tp_errors,
            )
            self.medium_map_based_nds = NuScenesDetectionScore(
                map=self.map,
                metric_prefix_name=f"map_based_medium",
                mean_tp_error_metrics=self.medium_mean_tp_errors,
            )
            self.medium_mapH_based_nds = NuScenesDetectionScore(
                map=self.maph,
                metric_prefix_name=f"mapH_based_medium",
                mean_tp_error_metrics=self.medium_mean_tp_errors,
            )

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

    @staticmethod
    def _format_metric(value: float) -> str:
        if isinstance(value, float) and np.isnan(value):
            return "NaN"
        return f"{value:.4f}"

    @staticmethod
    def _format_conf(value: float, width: int = 12) -> str:
        if isinstance(value, float) and np.isnan(value):
            return f"{'NaN':^{width}}"
        return f"{value:^{width}.6f}"

    @staticmethod
    def _format_conf_compact(value: float) -> str:
        if isinstance(value, float) and np.isnan(value):
            return "NaN"
        return f"{value:.4f}"

    def __str__(self) -> str:
        str_ = ""
        str_ += f"\nmAP: {self._format_metric(self.map)}, "
        if not self.is_detection_2d:
            str_ += f"mAPH: {self._format_metric(self.maph)}, "
            for mean_tp_error_name in self.mean_tp_error_names:
                mean_tp_error = self.mean_tp_errors[mean_tp_error_name]
                medium_tp_error = self.medium_mean_tp_errors[mean_tp_error_name]
                optimal_tp_error = self.optimal_mean_tp_errors[mean_tp_error_name]
                str_ += (
                    f"{mean_tp_error_name}: {self._format_metric(mean_tp_error)}, "
                    f"{mean_tp_error_name}(opt): {self._format_metric(optimal_tp_error)}, "
                    f"{mean_tp_error_name}(med): {self._format_metric(medium_tp_error)}, "
                )
            str_ += (
                f"{self.map_based_nds.metric_prefix_name} NDS: "
                f"{self._format_metric(self.map_based_nds.nds)}, "
                f"{self.mapH_based_nds.metric_prefix_name} NDS: "
                f"{self._format_metric(self.mapH_based_nds.nds)}, "
                f"{self.medium_map_based_nds.metric_prefix_name} NDS: "
                f"{self._format_metric(self.medium_map_based_nds.nds)}, "
                f"{self.medium_mapH_based_nds.metric_prefix_name} NDS: "
                f"{self._format_metric(self.medium_mapH_based_nds.nds)}, "
            )
        str_ += f"({self.matching_mode.value})\n"

        # === Per-label AP Table ===
        for label in self.target_labels:
            str_ += f"\nLabel: {label.value}\n"

            aps = self.label_to_aps[label]
            aphs = self.label_to_aphs.get(label, []) if not self.is_detection_2d else []
            gt_num = self.num_ground_truth_dict[label]

            tp_error_names: List[str] = []
            if not self.is_detection_2d:
                tp_error_names = list(self.avg_tp_error_names.keys())

            # --- Table 1: AP/APH + optimal operating point stats ---
            str_ += "| Threshold | Predict_num | Predict_match | Match@opt_conf | Groundtruth_num |     AP     |"
            if not self.is_detection_2d:
                str_ += "    APH    |"
            str_ += "   max_f1   |  optimal_recall | optimal_precision  | optimal_conf      |"
            str_ += "\n"

            str_ += "|:---------:|:-----------:|:-------------:|:--------------:|:---------------:|:----------:|"
            if not self.is_detection_2d:
                str_ += ":---------:|"
            str_ += ":----------:|:---------------:|:------------------:|:-----------------:|"
            str_ += "\n"

            for ap in aps:
                threshold = ap.matching_threshold
                predict_num = ap.objects_results_num
                predict_match = ap.num_tp
                predict_match_opt = ap.num_tp_at_optimal_conf
                ap_str = f"{ap.ap:^9.4f}" if not (isinstance(ap.ap, float) and np.isnan(ap.ap)) else "   NaN   "
                str_ += (
                    f"|  {threshold:^8.2f} | {predict_num:^11} | {predict_match:^13} | "
                    f"{predict_match_opt:^14} | {gt_num:^14}  |  {ap_str} |"
                )

                aph: Ap | None = None
                if not self.is_detection_2d:
                    aph = next((a for a in aphs if a.matching_threshold == threshold), None)
                    if aph is not None:
                        aph_str = (
                            f"{aph.ap:^9.4f}" if not (isinstance(aph.ap, float) and np.isnan(aph.ap)) else "   NaN   "
                        )
                        str_ += f"  {aph_str} |"
                    else:
                        str_ += " {:^9} |".format("N/A")
                str_ += f" {ap.max_f1_score:^9.4f} | {ap.optimal_precision:^9.4f} | {ap.optimal_recall:^9.4f}| {ap.optimal_conf:^12.6f} |"
                str_ += "\n"

            # --- Table 2: TP error metrics (moved after optimal_conf) ---
            if not self.is_detection_2d and tp_error_names:
                str_ += "\n"
                str_ += (
                    "| Threshold | min_recall_conf | medium_recall_conf |"
                )
                for mode in tp_error_names:
                    # Recall-band avg (min_recall=0.1), F1-optimal conf, medium recall band (0.4).
                    str_ += f" {mode:^8} | {mode + '@opt':^8} | {mode + '@med':^8} |"
                str_ += "\n"

                str_ += "|:---------:|:---------------:|:------------------:|"
                for _ in tp_error_names:
                    str_ += ":----------:|:----------:|:----------:|"
                str_ += "\n"

                for ap in aps:
                    threshold = ap.matching_threshold
                    str_ += f"|  {threshold:^8.2f} |"
                    if ap.tp_error_metrics is None:
                        str_ += (
                            f" {self._format_conf(np.nan, 15)} |"
                            f" {self._format_conf(np.nan, 18)} |"
                        )
                        for _ in tp_error_names:
                            str_ += " {:^8} | {:^8} | {:^8} |".format("N/A", "N/A", "N/A")
                        str_ += "\n"
                        continue

                    # Confidence at the start of each recall band (same for all TP error metrics).
                    ref_metric = ap.tp_error_metrics[0]
                    str_ += (
                        f" {self._format_conf(ref_metric.min_recall_conf, 15)} |"
                        f" {self._format_conf(ref_metric.medium_recall_conf, 18)} |"
                    )

                    for tp_error_name in tp_error_names:
                        tp_error_index = self.avg_tp_error_names[tp_error_name]
                        tp_error_metric = ap.tp_error_metrics[tp_error_index]
                        metric_val = tp_error_metric.avg_metric
                        opt_metric_val = tp_error_metric.optimal_avg_metric
                        medium_metric_val = tp_error_metric.medium_avg_metric
                        metric_str = (
                            f"{metric_val:^8.4f}"
                            if not (isinstance(metric_val, float) and np.isnan(metric_val))
                            else "  NaN  "
                        )
                        opt_metric_str = (
                            f"{opt_metric_val:^8.4f}"
                            if not (isinstance(opt_metric_val, float) and np.isnan(opt_metric_val))
                            else "  NaN  "
                        )
                        medium_metric_str = (
                            f"{medium_metric_val:^8.4f}"
                            if not (isinstance(medium_metric_val, float) and np.isnan(medium_metric_val))
                            else "  NaN  "
                        )
                        str_ += f" {metric_str} | {opt_metric_str} | {medium_metric_str} |"
                    str_ += "\n"

        # === Summary Table ===
        str_ += "\nSummary:\n"
        str_ += (
            "|      Label      |  Predict_num   | Predict_match  |"
            "   GT_nums       |  Thresholds       |  mean AP      |    APs           |"
        )
        if not self.is_detection_2d:
            str_ += "  Mean APH    |   APHs     | min_recall_conf | medium_recall_conf |"
            for mean_tp_error_name in self.mean_tp_error_names:
                str_ += (
                    f" {mean_tp_error_name:^8} | {mean_tp_error_name + '(opt)':^12} |"
                    f" {mean_tp_error_name + '(med)':^12} |"
                )
        str_ += "\n"

        str_ += (
            "|:---------------:|:--------------:|:--------------:|"
            ":---------------:|:-----------------:|:-------------:|:----------------:|"
        )
        if not self.is_detection_2d:
            str_ += ":---------------:|:---------------:|:------------------:|"
            for _ in self.mean_tp_error_names:
                str_ += ":----------:|:------------:|:------------:|"
        str_ += "\n"

        for label in self.target_labels:
            aps = self.label_to_aps[label]
            gt_num = self.num_ground_truth_dict[label]
            predict_num = aps[0].objects_results_num if len(aps) else 0
            # Per-threshold prediction match counts (raw TPs).
            predict_match_strs = [str(ap.num_tp) for ap in aps]
            thresholds = [f"{ap.matching_threshold:.2f}" for ap in aps]

            mean_ap = self.label_mean_to_ap[label]
            mean_ap_str = f"{mean_ap:^9.4f}" if not (isinstance(mean_ap, float) and np.isnan(mean_ap)) else "   NaN   "

            ap_strs = [
                f"{ap.ap:.4f}" if not (isinstance(ap.ap, float) and np.isnan(ap.ap)) else "   NaN   " for ap in aps
            ]
            str_ += (
                f"| {label.value:^15} | {predict_num:^14} | "
                f"{' / '.join(predict_match_strs):^14} | "
                f"{gt_num:^14} | {'/'.join(thresholds):^14} |  {mean_ap_str} | {' / '.join(ap_strs):^14} |"
            )
            if not self.is_detection_2d:
                mean_aph = self.label_mean_to_aph[label]
                mean_aph_str = (
                    f"{mean_aph:^9.3f}" if not (isinstance(mean_aph, float) and np.isnan(mean_aph)) else "   NaN   "
                )
                aphs = self.label_to_aphs[label]
                aph_strs = [
                    f"{aph.ap:.4f}" if not (isinstance(aph.ap, float) and np.isnan(aph.ap)) else "   NaN   "
                    for aph in aphs
                ]
                str_ += f"  {mean_aph_str} | {' / '.join(aph_strs):^14} |"
                min_recall_conf_strs = [
                    self._format_conf_compact(ap.tp_error_metrics[0].min_recall_conf)
                    if ap.tp_error_metrics is not None
                    else "NaN"
                    for ap in aps
                ]
                medium_recall_conf_strs = [
                    self._format_conf_compact(ap.tp_error_metrics[0].medium_recall_conf)
                    if ap.tp_error_metrics is not None
                    else "NaN"
                    for ap in aps
                ]
                str_ += (
                    f" {' / '.join(min_recall_conf_strs):^15} |"
                    f" {' / '.join(medium_recall_conf_strs):^18} |"
                )
                for mean_tp_error_name in self.mean_tp_error_names:
                    label_mean_tp = self.label_mean_to_tp_error[mean_tp_error_name][label]
                    label_optimal_tp = self.label_mean_to_optimal_tp_error[mean_tp_error_name][label]
                    label_medium_tp = self.label_mean_to_medium_tp_error[mean_tp_error_name][label]
                    label_mean_str = (
                        f"{label_mean_tp:^8.4f}"
                        if not (isinstance(label_mean_tp, float) and np.isnan(label_mean_tp))
                        else "  NaN  "
                    )
                    label_optimal_str = (
                        f"{label_optimal_tp:^12.4f}"
                        if not (isinstance(label_optimal_tp, float) and np.isnan(label_optimal_tp))
                        else "    NaN    "
                    )
                    label_medium_str = (
                        f"{label_medium_tp:^12.4f}"
                        if not (isinstance(label_medium_tp, float) and np.isnan(label_medium_tp))
                        else "    NaN    "
                    )
                    str_ += f" {label_mean_str} | {label_optimal_str} | {label_medium_str} |"

            str_ += "\n"

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
