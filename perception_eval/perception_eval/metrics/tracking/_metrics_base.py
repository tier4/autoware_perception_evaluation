# Copyright 2022-2024 TIER IV, Inc.

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

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

from ..tp_metrics import TPMetricsAp

if TYPE_CHECKING:
    from perception_eval.common.label import LabelType
    from perception_eval.matching import MatchingMode
    from perception_eval.result import PerceptionObjectResult


class _TrackingMetricsBase(ABC):
    """Abstract base class for tracking metrics.

    Args:
    -----
        num_ground_truth (int): Number of ground truths.
        target_labels (List[LabelType]): Target labels list.
        matching_mode (MatchingMode): MatchingMode instance.
        matching_threshold_list (List[float]): Thresholds list for matching.
    """

    metrics: List[str]

    @abstractmethod
    def __init__(
        self,
        num_ground_truth: int,
        target_labels: List[LabelType],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
    ) -> None:
        self._num_ground_truth: int = num_ground_truth
        self._target_labels: List[LabelType] = target_labels
        self._matching_mode: MatchingMode = matching_mode
        self._matching_threshold_list = matching_threshold_list
        self._tp_metrics = TPMetricsAp()

    @abstractmethod
    def _calculate_tp_fp(
        self,
        cur_object_results: List[PerceptionObjectResult],
        prev_object_results: List[PerceptionObjectResult],
    ) -> Any:
        pass

    @property
    @abstractmethod
    def results(self) -> Dict[str, float]:
        pass

    def _check_metrics(self, metrics_field: Optional[List[str]]) -> List[str]:
        """Check if specified metrics is supported.

        If `metrics_field=None`, returns default supported metrics.

        Args:
        -----
            metrics_field (Optional[List[str]]): The list of executing metrics field.

        Returns:
        --------
            metrics_field (List[str])

        Raises:
        -------
            ValueError: If input metrics field is unsupported.
        """
        if metrics_field is None:
            return self.metrics

        if set(metrics_field) > set(self.metrics):
            raise ValueError(f"Unsupported metrics: {set(metrics_field) - set(self.metrics)}")

        return metrics_field

    @property
    def num_ground_truth(self) -> int:
        return self._num_ground_truth

    @property
    def target_labels(self) -> List[LabelType]:
        return self._target_labels

    @property
    def matching_mode(self) -> MatchingMode:
        return self._matching_mode

    @property
    def matching_threshold_list(self) -> List[float]:
        return self._matching_threshold_list

    @property
    def tp_metrics(self) -> TPMetricsAp:
        return self._tp_metrics

    def get_precision_recall_list(
        self,
        tp_list: List[float],
        ground_truth_objects_num: int,
    ) -> Tuple[List[float], List[float]]:
        """Calculate precision recall.

        Args:
        -----
            tp_list (List[float]): TP results list.
            ground_truth_objects_num (int): Number of ground truths.

        Returns:
        --------
            Tuple[List[float], List[float]]: tp_list and fp_list

        Examples:
        ---------
            >>> tp_list = [1, 1, 2, 3]
            >>> ground_truth_num = 4
            >>> precision_list, recall_list = self.get_precision_recall(tp_list, ground_truth_num)
            >>> precision_list
            [1.0, 0.5, 0.67, 0.75]
            >>> recall_list
            [0.25, 0.25, 0.5, 0.75]
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
        """__str__ method

        Returns:
            str: Formatted string.
        """
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
