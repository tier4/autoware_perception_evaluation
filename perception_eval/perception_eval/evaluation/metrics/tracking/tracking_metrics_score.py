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
from typing import Union

from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

from .clear import CLEAR


class TrackingMetricsScore:
    """Metrics score class for tracking evaluation.

    Length of input `target_labels` and `matching_threshold_list` must be same.

    Attributes:
        target_labels: (List[LabelType]): Target labels list.
        matching_mode (MatchingMode): MatchingMode instance.
        clears (List[CLEAR]): List of CLEAR instances.

    Args:
        object_results_dict (Dict[LabelType, List[List[DynamicObjectWithPerceptionResult]]):
            Dict that items are object results list mapped by their labels.
        num_ground_truth (int): Number of ground truths.
        target_labels (List[LabelType]): Target labels list.
        matching_mode (MatchingMode): MatchingMode instance.
        matching_threshold_list (List[float]): Matching thresholds list.
    """

    def __init__(
        self,
        object_results_dict: Dict[LabelType, List[List[DynamicObjectWithPerceptionResult]]],
        num_ground_truth_dict: Dict[LabelType, int],
        target_labels: List[LabelType],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
    ) -> None:
        assert len(target_labels) == len(matching_threshold_list)
        self.object_results_dict = object_results_dict
        self.num_ground_truth_dict = num_ground_truth_dict
        self.target_labels: List[LabelType] = target_labels
        self.matching_mode: MatchingMode = matching_mode
        self.matching_threshold_list: List[float] = matching_threshold_list

        # CLEAR results for each class
        self.clears: List[CLEAR] = []
        # Calculate score for each target labels
        for target_label, matching_threshold in zip(target_labels, matching_threshold_list):
            object_results = object_results_dict[target_label]
            num_ground_truth = num_ground_truth_dict[target_label]
            clear_: CLEAR = CLEAR(
                object_results=object_results,
                num_ground_truth=num_ground_truth,
                target_labels=[target_label],
                matching_mode=matching_mode,
                matching_threshold_list=[matching_threshold],
            )
            self.clears.append(clear_)

    def __reduce__(self) -> Tuple[TrackingMetricsScore, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling."""
        init_args = (
            self.object_results_dict,
            self.num_ground_truth_dict,
            self.target_labels,
            self.matching_mode,
            self.matching_threshold_list,
        )

        return (
            self.__class__,
            init_args,
        )

    def _sum_clear(self) -> Tuple[float, float, int]:
        """Summing up multi CLEAR result.

        Returns:
            mota (float): MOTA score.
            motp (float): MOTP score.
            num_id_switch (int): The number of ID switched.
        """
        mota: float = 0.0
        motp: float = 0.0
        num_gt: int = 0
        num_tp: int = 0
        num_id_switch: int = 0
        for clear in self.clears:
            if clear.mota != float("inf"):
                mota += clear.mota * clear.num_ground_truth
            if clear.motp != float("inf"):
                motp += clear.motp * clear.tp
            num_gt += clear.num_ground_truth
            num_tp += int(clear.tp)
            num_id_switch += clear.id_switch

        mota = float("inf") if num_gt == 0 else mota / num_gt
        mota = max(0.0, mota)
        # NOTE: If tp_metrics is not TPMetricsAP this calculation cause bug
        motp = float("inf") if num_tp == 0 else motp / num_tp
        return mota, motp, num_id_switch

    def __str__(self) -> str:
        """__str__ method

        Returns:
            str: Formatted string.
        """
        str_: str = "\n"
        # === Total ===
        # CLEAR
        mota, motp, id_switch = self._sum_clear()
        str_ += f"[TOTAL] MOTA: {mota:.3f}, MOTP: {motp:.3f}, ID switch: {id_switch:d} "
        str_ += f"({self.matching_mode.value})\n"
        # === For each label ===
        # --- Table ---
        str_ += "\n"
        # --- Label ----
        str_ += "|      Label |"
        # CLEAR
        for clear in self.clears:
            str_ += f" {clear.target_labels[0]}({clear.matching_threshold_list[0]}) | "
        str_ += "\n"
        str_ += "| :--------: |"
        for _ in self.clears:
            str_ += " :---: |"
        str_ += "\n"
        for field_name in self.clears[0].metrics_field:
            str_ += f"|   {field_name} |"
            for clear_ in self.clears:
                score: Union[int, float] = clear_.results[field_name]
                if isinstance(score, int):
                    str_ += f" {score:d} |"
                else:
                    str_ += f" {score:.3f} |"
            str_ += "\n"

        return str_
