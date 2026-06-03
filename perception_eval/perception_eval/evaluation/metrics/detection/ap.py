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
from logging import getLogger
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.detection.tp_error_metrics import TPErrorMetric
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

logger = getLogger(__name__)


class Ap:
    """AP class.

    Attributes:
        ap (float): AP (Average Precision) score. Can be NaN when there are no ground truths and no predictions.
        matching_average (Optional[float]): Average of matching score.
            If there are no object results, this variable is None.
        matching_mode (MatchingMode): MatchingMode instance.
        matching_threshold (float): Threshold information for the AP instance.
        matching_standard_deviation (Optional[float]): Standard deviation of matching score.
            If there are no object results, this variable is None.
        target_label (LabelType): Target label.
        tp_metrics (TPMetrics): Mode of TP metrics.
        num_ground_truth (int): Number ground truths.
        tp_list (List[float]): List of the number of TP objects ordered by their confidences.
        fp_list (List[float]): List of the number of FP objects ordered by their confidences.
        num_tp (int): Total number of prediction matches (TPs) for this matching threshold,
            considering all predictions regardless of confidence.
        num_tp_at_optimal_conf (int): Number of prediction matches (TPs) at the optimal
            confidence threshold (i.e. predictions whose confidence is greater than or equal
            to ``optimal_conf``). NaN if there are no valid predictions.
        tp_error_metrics (Optional[List[TPErrorMetric]]): List of TP error metrics.

    Args:
        tp_metrics (TPMetrics): Mode of TP (True positive) metrics.
        object_results (List[DynamicObjectWithPerceptionResult]): Object results list.
        num_ground_truth (int): Number of ground truths.
        target_label (LabelType): Target label.
        matching_mode (MatchingMode): Matching instance.
        matching_threshold (float): Threshold information for the AP instance.
    """

    # Number of recall thresholds for AP interpolation.
    # 101-point interpolation is a standard method used in object detection.
    # It uses recall levels at 0.00, 0.01, ..., 1.00 inclusive.
    NUM_RECALL_POINTS = 101

    def __init__(
        self,
        tp_metrics: Union[TPMetricsAp, TPMetricsAph],
        object_results: List[DynamicObjectWithPerceptionResult],
        num_ground_truth: int,
        target_label: LabelType,
        matching_mode: MatchingMode,
        matching_threshold: float,
        tp_error_metrics: Optional[List[TPErrorMetric]] = None,
    ) -> None:
        self.tp_metrics = tp_metrics
        self.object_results = object_results
        self.num_ground_truth = num_ground_truth

        self.target_label = target_label
        self.matching_mode = matching_mode
        self.matching_threshold = matching_threshold

        self.objects_results_num: int = len(object_results)
        self.matching_average: Optional[float] = None
        self.matching_standard_deviation: Optional[float] = None
        self.matching_average, self.matching_standard_deviation = self._calculate_average_std(
            object_results=object_results,
            matching_mode=self.matching_mode,
        )
        # True positives, false positives and their corresponding confidences
        self.tp_list, self.fp_list, self.conf_list, self.sorted_indices, self.num_tp = self._calculate_tp_fp(
            tp_metrics, object_results
        )

        self.precision_list, self.recall_list = self.get_precision_recall()
        self.precision_interp, self.recall_interp, self.conf_interp = self._interpolate_precision_recall(
            self.precision_list, self.recall_list, self.conf_list
        )
        self.f1_scores = self.compute_f1_scores(precisions=self.precision_list, recalls=self.recall_list)
        self.max_f1_score, self.max_f1_index = self.compute_max_f1_index()
        if self.max_f1_index > -1:
            self.optimal_precision = self.precision_list[self.max_f1_index]
            self.optimal_recall = self.recall_list[self.max_f1_index]
            self.optimal_conf = self.conf_list[self.max_f1_index]

            # Number of TPs at the optimal confidence threshold. Computed from the raw
            # (unweighted) TP indicators so it represents an actual prediction count for
            # both AP and APH instances.
            self.num_tp_at_optimal_conf = self._count_tp_at_index(object_results, self.conf_list, self.max_f1_index)

        else:
            self.optimal_precision = np.nan
            self.optimal_recall = np.nan
            self.optimal_conf = np.nan
            self.num_tp_at_optimal_conf = 0

        self.ap = self._calculate_ap(self.precision_interp)

        # Compute TP error metrics.
        self.tp_error_metrics = tp_error_metrics
        self.compute_tp_error_metrics_values(sorted_indices=self.sorted_indices)

        # Compute average of TP error metrics.
        self.compute_average_tp_error_metrics()

        # Compute optimal average of TP error metrics.
        self.compute_optimal_average_tp_error_metrics(optimal_conf=self.optimal_conf)
        
        # Compute recall confidence of TP error metrics.
        self.compute_recall_conf(min_recall=0.1, medium_recall=0.4, conf_interp=self.conf_interp)
        
    def __reduce__(self) -> Tuple[Ap, Tuple[Any]]:
        """Serializing and deserializing the class."""
        return (
            self.__class__,
            (
                self.tp_metrics,
                self.object_results,
                self.num_ground_truth,
                self.target_label,
                self.matching_mode,
                self.matching_threshold,
                self.tp_error_metrics,
            ),
        )

    @property
    def max_recall_ind(self):
        """
        Returns index of max recall achieved.
        Taken from nuScenes-devkit.
        https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/data_classes.py#L127
        Returns:
            int: Index of max recall achieved.
        """
        # No matches, return 0
        if len(self.conf_interp) == 0:
            return 0
        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.conf_interp)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

    def _calculate_tp_fp(
        self,
        tp_metrics: Union[TPMetricsAp, TPMetricsAph],
        object_results: List[DynamicObjectWithPerceptionResult],
    ) -> Tuple[List[float], List[float], List[float], int]:
        """
        Calculate TP/FP when object_results are stored as a dict with (matching_mode, threshold) keys.
        This assumes matching has already occurred.

        Returns:
            Tuple[List[float], List[float], List[float], List[int], int]:
                - tp_list: List of True Positive (TP) values
                - fp_list: List of False Positive (FP) values
                - conf_list: List of confidence values
                - sorted_indices: List of indices sorted by confidence in descending order
                - num_tp: Number of True Positives
        """
        tp_list: List[float] = []
        fp_list: List[float] = []
        conf_list: List[float] = []
        num_tp: int = 0

        for obj in object_results:
            is_tp = obj.ground_truth_object is not None and obj.is_label_correct
            conf_list.append(obj.estimated_object.semantic_score)
            tp_list.append(tp_metrics.get_value(obj) if is_tp else 0.0)
            fp_list.append(0.0 if is_tp else 1.0)
            num_tp += 1 if is_tp else 0

        if not conf_list:
            return [], [], [], [], 0

        # Sort by confidence
        sorted_indices = np.argsort(conf_list)[::-1]
        tp_sorted = [tp_list[i] for i in sorted_indices]
        fp_sorted = [fp_list[i] for i in sorted_indices]
        conf_sorted = [conf_list[i] for i in sorted_indices]

        tp_list = np.cumsum(tp_sorted).tolist()
        fp_list = np.cumsum(fp_sorted).tolist()

        return tp_list, fp_list, conf_sorted, sorted_indices, num_tp

    def compute_f1_scores(self, precisions: List[float], recalls: List[float]) -> List[float]:
        """
        Compute f1 scores from a list of precisions and recalls.
        """
        precisions = np.asarray(precisions)
        recalls = np.asarray(recalls)

        numerator = 2 * (precisions * recalls)
        denominator = precisions + recalls

        # Create an output array filled with np.nan
        f1_scores = np.full_like(numerator, np.nan, dtype=np.float64)

        # Create a condition: denominator is not zero and not nan
        condition = (denominator != 0) & ~np.isnan(denominator)

        np.divide(numerator, denominator, where=condition, out=f1_scores)
        return f1_scores.tolist()

    def compute_max_f1_index(self) -> Tuple[float, int]:
        """Compute max-f1 and return max-f1 and the index."""
        try:
            max_index = np.nanargmax(self.f1_scores)
        except ValueError:
            # In case, all NaN values, for example, empty detection/ground truths
            return np.nan, -1

        return self.f1_scores[max_index], max_index

    def get_precision_recall(self) -> Tuple[List[float], List[float]]:
        """
        Compute the precision and recall list.

        Returns:
            Tuple[List[float], List[float]]:
                - precision: The precision list
                - recall: The recall list
        """
        precision, recall = [], []
        for i in range(len(self.tp_list)):
            denominator = self.tp_list[i] + self.fp_list[i]
            if denominator == 0.0:
                precision.append(0.0)
            else:
                precision.append(self.tp_list[i] / denominator)
            recall.append(self.tp_list[i] / self.num_ground_truth if self.num_ground_truth > 0 else 0.0)
        return precision, recall

    def _interpolate_precision_recall(
        self, precision_list: List[float], recall_list: List[float], conf_list: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate the precision list using the recall list.
        """
        if len(precision_list) == 0:
            return np.array([]), np.array([]), np.array([])

        # Create a precision envelope: ensures non-increasing precision
        # max accumulate from right to left
        precision_envelope = np.maximum.accumulate(precision_list[::-1])[::-1]

        # Define uniformly spaced recall levels for interpolation (101 points)
        recall_interp = np.linspace(0.0, 1.0, self.NUM_RECALL_POINTS)

        # Interpolate precision at those recall levels using the envelope
        # 'right=0' means values beyond the max recall get precision=0
        precision_interp = np.interp(recall_interp, recall_list, precision_envelope, right=0)

        conf_interp = np.interp(recall_interp, recall_list, conf_list, right=0)

        return precision_interp, recall_interp, conf_interp

    def _calculate_ap(
        self,
        precisions: np.ndarray,
        min_recall: float = 0.1,
        min_precision: float = 0.1,
    ) -> float:
        """
        Calculate Average Precision (AP) using 101 uniformly spaced recall thresholds
        """

        # Special case: If ground truth is zero and prediction is zero, return NaN
        if self.num_ground_truth == 0 and self.objects_results_num == 0:
            return np.nan

        # If there are no precision values, return AP = 0.0
        if len(precisions) == 0:
            return 0.0

        # Apply a minimum recall threshold: ignore low-recall range
        first_ind = int(round(100 * min_recall)) + 1

        # Subtract the minimum precision threshold and clamp negative values to 0
        filtered_prec = precisions[first_ind:] - min_precision
        filtered_prec[filtered_prec < 0] = 0.0

        # Return the normalized mean of the filtered precision values
        # Divided by (1 - min_precision) to scale the AP
        return float(np.mean(filtered_prec)) / (1.0 - min_precision)

    @staticmethod
    def _calculate_average_std(
        object_results: List[DynamicObjectWithPerceptionResult],
        matching_mode: MatchingMode,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate the average and standard deviation of the matching scores
        for the given object results using the specified matching mode.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]):
                The list of predicted objects with associated ground truth.
            matching_mode (MatchingMode):
                The mode used to compute the matching score (e.g., center distance, IoU).

        Returns:
            Tuple[Optional[float], Optional[float]]:
                The average and standard deviation of the matching scores.
                If no valid scores are found, both values will be None.
        """

        matching_score_list: List[float] = []
        for obj in object_results:
            match = obj.get_matching(matching_mode)
            if match is not None and match.value is not None:
                matching_score_list.append(match.value)

        if not matching_score_list:
            return None, None

        mean = float(np.mean(matching_score_list))
        std = float(np.std(matching_score_list))
        return mean, std

    @staticmethod
    def _count_tp_at_index(
        object_results: List[DynamicObjectWithPerceptionResult],
        conf_list_sorted_desc: List[float],
        index: int,
    ) -> int:
        """Count the raw number of TPs among predictions whose confidence is greater than
        or equal to the confidence at ``index`` in ``conf_list_sorted_desc``.
        ``conf_list_sorted_desc`` is the per-prediction confidence list sorted by
        descending confidence. ``index`` is the cut-off (inclusive) corresponding to the
        optimal F1 operating point.
        """
        if index < 0 or index >= len(conf_list_sorted_desc):
            return 0
        threshold = conf_list_sorted_desc[index]
        num_tp = 0
        for obj in object_results:
            if obj.estimated_object.semantic_score < threshold:
                continue
            if obj.ground_truth_object is not None and obj.is_label_correct:
                num_tp += 1
        return num_tp

    def compute_tp_error_metrics_values(self, sorted_indices: List[int]) -> None:
        """Compute the values of TP error metrics."""
        if self.tp_error_metrics is None:
            return

        # Sort object results by confidence in descending order
        object_results_sorted = [self.object_results[i] for i in sorted_indices]

        for tp_error_metric in self.tp_error_metrics:
            tp_error_metric_values = []
            tp_error_metric_confidences = []
            for object_result in object_results_sorted:
                tp_error_metric_value = tp_error_metric.compute_value(object_result)
                if np.isnan(tp_error_metric_value):
                    continue
                
                tp_error_metric_values.append(tp_error_metric_value)
                tp_error_metric_confidences.append(object_result.estimated_object.semantic_score)
            tp_error_metric.values = np.array(tp_error_metric_values)
            tp_error_metric.confidences = np.array(tp_error_metric_confidences)

    def compute_average_tp_error_metrics(self, min_recall: float = 0.1, medium_recall: float = 0.4) -> None:
        """Compute the average of TP error metrics."""
        if self.tp_error_metrics is None:
            return

        max_recall_ind = self.max_recall_ind
        for tp_error_metric in self.tp_error_metrics:
            tp_error_metric.interpolated_values = tp_error_metric.interpolate_values(
                conf_interp=self.conf_interp, ascending_sorted=False
            )
            tp_error_metric.avg_metric = tp_error_metric.compute_average_value(
                min_recall=min_recall, max_recall_ind=max_recall_ind, target_label=self.target_label
            )
            tp_error_metric.medium_avg_metric = tp_error_metric.compute_average_value(
                min_recall=medium_recall, max_recall_ind=max_recall_ind, target_label=self.target_label
            )

    def compute_optimal_average_tp_error_metrics(self, optimal_conf: int) -> None:
        """Compute the optimal average of TP error metrics."""
        if self.tp_error_metrics is None:
            return

        for tp_error_metric in self.tp_error_metrics:
            tp_error_metric.optimal_avg_metric = tp_error_metric.compute_optimal_average_value(
                optimal_conf=optimal_conf
            )
    
    def compute_recall_conf(self, min_recall: float, medium_recall: float, conf_interp: np.ndarray) -> float:
        """Compute the recall confidence of TP error metrics."""
        if self.tp_error_metrics is None:
            return
        
        max_recall_ind = self.max_recall_ind
        for tp_error_metric in self.tp_error_metrics:
            if len(tp_error_metric.interpolated_values) == 0:
                continue 

            first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
            last_ind = max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
            if last_ind < first_ind:
                tp_error_metric.min_recall_conf = np.nan
            else:
                tp_error_metric.min_recall_conf = conf_interp[first_ind]


            first_ind = round(100 * medium_recall) + 1  # +1 to exclude the error at medium recall.
            if last_ind < first_ind:
                tp_error_metric.medium_recall_conf = np.nan
            else:
                tp_error_metric.medium_recall_conf = conf_interp[first_ind]
