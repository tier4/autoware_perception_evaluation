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

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetrics
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class _TrackingMetricsBase(ABC):
    """Abstract base class for tracking metrics

    Attributes:
        self.target_labels (List[AutowareLabel]): The list of target label.
        self.matching_mode (MatchingMode): The target matching mode.
        self.metrics_field (Optional[List[str]]): The list of target metrics name. If not specified, set default supported metrics.
        self.num_ground_truth (int): The number of ground truth.
        self.support_metrics (List[str]): The list of supported metrics name.
    """

    _support_metrics: List[str] = []

    @abstractmethod
    def __init__(
        self,
        num_ground_truth: int,
        target_labels: List[AutowareLabel],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
        tp_metrics: TPMetrics,
        metrics_field: Optional[List[str]],
    ) -> None:
        """[summary]
        The abstract base class for tracking metrics.

        NOTE: objects_results, ground_truth_objects
            If evaluate 1-frame, index 0 is previous object results.
            If evaluate all frames, index 0 is empty list.

        Args:
            frame_ground_truths (List[List[DynamicObject]]): The list of ground truth objects for each frames.
            target_labels (List[AutowareLabel]): The list of target labels.
            matching_mode (MatchingMode): Matching mode class.
            metrics_field (Optional[List[str]]: The list of target sub metrics.
        """
        self._num_ground_truth: int = num_ground_truth
        self._target_labels: List[AutowareLabel] = target_labels
        self._matching_mode: MatchingMode = matching_mode
        self._matching_threshold_list = matching_threshold_list
        self._tp_metrics: TPMetrics = tp_metrics

        # Check if metrics field is supported.
        self._metrics_filed: List[str] = self._check_metrics(metrics_field)

    @abstractmethod
    def _calculate_tp_fp(
        self,
        cur_object_results: List[DynamicObjectWithPerceptionResult],
        prev_object_results: List[DynamicObjectWithPerceptionResult],
    ) -> Any:
        pass

    @property
    @abstractmethod
    def results(self) -> Dict[str, float]:
        pass

    @property
    def support_metrics(self) -> List[str]:
        return self._support_metrics

    def _check_metrics(self, metrics_field: Optional[List[str]]) -> List[str]:
        """[summary]
        Check if specified metrics is supported.

        Args:
            metrics_field (Optional[List[str]]): The list of executing metrics field.

        Returns:
            metrics_field (List[str])

        Raises:
            ValueError: If input metrics field is unsupported.
        """
        if metrics_field is None:
            return self.support_metrics

        if set(metrics_field) > set(self.support_metrics):
            raise ValueError(
                f"Unsupported metrics: {set(metrics_field) - set(self.support_metrics)}"
            )

        return metrics_field

    @property
    def num_ground_truth(self) -> int:
        return self._num_ground_truth

    @property
    def target_labels(self) -> List[AutowareLabel]:
        return self._target_labels

    @property
    def matching_mode(self) -> MatchingMode:
        return self._matching_mode

    @property
    def matching_threshold_list(self) -> List[float]:
        return self._matching_threshold_list

    @property
    def tp_metrics(self) -> TPMetrics:
        return self._tp_metrics

    @property
    def metrics_field(self) -> List[str]:
        return self._metrics_filed

    def get_precision_recall_list(
        self,
        tp_list: List[float],
        ground_truth_objects_num: int,
    ) -> Tuple[List[float], List[float]]:
        """[summary]
        Calculate precision recall.

        Args:
            tp_list (List[float])
            ground_truth_objects_num (int)

        Returns:
            Tuple[List[float], List[float]]: tp_list and fp_list
        Example:
            state
                self.tp_list = [1, 1, 2, 3]
                self.fp_list = [0, 1, 1, 1]
            return
                precision_list = [1.0, 0.5, 0.67, 0.75]
                recall_list = [0.25, 0.25, 0.5, 0.75]
        """
        precisions_list: List[float] = [0.0 for i in range(len(tp_list))]
        recalls_list: List[float] = [0.0 for i in range(len(tp_list))]

        for i in range(len(precisions_list)):
            precisions_list[i] = float(tp_list[i]) / (i + 1)
            if ground_truth_objects_num > 0:
                recalls_list[i] = float(tp_list[i]) / ground_truth_objects_num
            else:
                recalls_list[i] = 0.0

        return precisions_list, recalls_list

    def __str__(self) -> str:
        """__str__ method"""
        str_: str = "\n"
        for name, score in self.results.items():
            # Each label result
            str_ += "\n"
            str_ += f"|    {name} |"
            if isinstance(score, int):
                str_ += f"    {score:d} |"
            elif isinstance(score, float):
                str_ += f"    {score:.3f} |"
            else:
                str_ += f"    {score} |"

        return str_
