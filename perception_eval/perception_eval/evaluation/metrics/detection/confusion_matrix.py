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
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Any
from typing import Dict

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

    label: str
    total_gt_nums: int
    total_tp_nums: int
    matched_labels: Dict[str, int]


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

    def __init__(self, 
                 object_results_dict: Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]],
                 num_ground_truth_dict: Dict[LabelType, int],
                 target_labels: List[LabelType],
                 matching_mode: MatchingMode,
                 ) -> None:
        
        self.object_results_dict = object_results_dict
        self.num_ground_truth_dict = num_ground_truth_dict
        self.target_labels = target_labels
        self.matching_mode = matching_mode
    
    def get_confusion_matrix(self) -> Dict[float, Dict[str, Dict[str, int]]]:
        """
        Compute confusion matrix in an array of target_labels x target_labels, 
        where the row is prediction and gt is ground truth.
        """
        # confusion_matrix: {matching_threshold: {predicted_label: {ground_truth: number of matched predicted boxes}}}
        matching_threshold_confusion_matrices = defaultdict(defaultdict(int))

        # total tp nums for each ground truth labels, {threshold: {ground_truth: number of tps}}
        total_tps = defaultdict(defaultdict(int))
        # 
        for label in self.target_labels:
            
            for threshold, object_results in self.object_results_dict[label].items():
                for object_result in object_results:
                    predicted_object_label = object_result.semantic_label.name

                    if object_result.ground_truth_object is not None:
                        if object_result.is_label_correct:
                            total_tps[threshold][label] += 1
                        
                        matching_threshold_confusion_matrices[threshold][predicted_object_label][predicted_object_label] += 1
                    else:
                        matching_threshold_confusion_matrices[threshold][predicted_object_label][_UNMATCHED_LABEL] += 1
                        
            
            

        

class Ap::
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
        ground_truth_objects_num (int): Number ground truths.
        tp_list (List[float]): List of the number of TP objects ordered by their confidences.
        fp_list (List[float]): List of the number of FP objects ordered by their confidences.

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
    ) -> None:
        self.tp_metrics = tp_metrics
        self.objects_results = object_results
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
        self.tp_list, self.fp_list, self.conf_list = self._calculate_tp_fp(tp_metrics, object_results)
        precision_list, recall_list = self.get_precision_recall()
        self.f1_scores = self.compute_f1_scores(precisions=precision_list, recalls=recall_list)
        self.max_f1_score, self.max_f1_index = self.compute_max_f1_index()
        self.best_conf = self.conf_list[self.max_f1_index] if self.max_f1_index > -1 else np.nan

        self.ap = self._calculate_ap(precision_list, recall_list)

    def __reduce__(self) -> Tuple[Ap, Tuple[Any]]:
        """ Serialize and deserializing the class. """
        return (self.__class__, (self.tp_metrics, self.objects_results, self.num_ground_truth, self.target_label, self.matching_mode, self.matching_threshold), )
    
    def _calculate_tp_fp(
        self,
        tp_metrics: Union[TPMetricsAp, TPMetricsAph],
        object_results: List[DynamicObjectWithPerceptionResult],
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate TP/FP when object_results are stored as a dict with (matching_mode, threshold) keys.
        This assumes matching has already occurred.
        """
        tp_list: List[float] = []
        fp_list: List[float] = []
        conf_list: List[float] = []

        for obj in object_results:
            is_tp = obj.ground_truth_object is not None and obj.is_label_correct
            conf_list.append(obj.estimated_object.semantic_score)
            tp_list.append(tp_metrics.get_value(obj) if is_tp else 0.0)
            fp_list.append(0.0 if is_tp else 1.0)

        if not conf_list:
            return [], [], []

        # Sort by confidence
        sorted_indices = np.argsort(conf_list)[::-1]
        tp_sorted = [tp_list[i] for i in sorted_indices]
        fp_sorted = [fp_list[i] for i in sorted_indices]
        conf_sorted = [conf_list[i] for i in sorted_indices]

        tp_list = np.cumsum(tp_sorted).tolist()
        fp_list = np.cumsum(fp_sorted).tolist()

        return tp_list, fp_list, conf_sorted

    def compute_f1_scores(self, precisions: List[float], recalls: List[float]) -> List[float]:
        """
        Compute f1 scores from a list of precisions and recalls.
        """
        precisions = np.asarray(precisions)
        recalls = np.asarray(recalls)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        return f1_scores.tolist()
    
    def compute_max_f1_index(self) -> Tuple[float, int]:
        """ Compute the best max-f1 and return the index. """
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

    def _calculate_ap(
        self,
        precision_list: List[float],
        recall_list: List[float],
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
        if len(precision_list) == 0:
            return 0.0

        # Create a precision envelope: ensures non-increasing precision
        # max accumulate from right to left
        precision_envelope = np.maximum.accumulate(precision_list[::-1])[::-1]

        # Define uniformly spaced recall levels for interpolation (101 points)
        recall_interp = np.linspace(0.0, 1.0, self.NUM_RECALL_POINTS)

        # Interpolate precision at those recall levels using the envelope
        # 'right=0' means values beyond the max recall get precision=0
        precision_interp = np.interp(recall_interp, recall_list, precision_envelope, right=0)

        # Apply a minimum recall threshold: ignore low-recall range
        first_ind = int(round(100 * min_recall)) + 1

        # Subtract the minimum precision threshold and clamp negative values to 0
        filtered_prec = precision_interp[first_ind:] - min_precision
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
