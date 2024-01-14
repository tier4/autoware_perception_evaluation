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

from ctypes import Union
from typing import Any
from typing import Dict
from typing import List
from typing import TYPE_CHECKING

from .clear import CLEAR
from .clear import sum_clear

if TYPE_CHECKING:
    from perception_eval.common.label import LabelType
    from perception_eval.matching import MatchingMode
    from perception_eval.result import PerceptionObjectResult


class TrackingMetricsScore:
    """Metrics score class for tracking evaluation.

    Length of input `target_labels` and `matching_threshold_list` must be same.

    Args:
    -----
        object_results_dict (Dict[LabelType, List[List[PerceptionObjectResult]]):
            Dict that items are object results list mapped by their labels.
        num_ground_truth (int): Number of ground truths.
        target_labels (List[LabelType]): Target labels list.
        matching_mode (MatchingMode): MatchingMode instance.
        matching_threshold_list (List[float]): Matching thresholds list.
    """

    def __init__(
        self,
        object_results_dict: Dict[LabelType, List[List[PerceptionObjectResult]]],
        num_ground_truth_dict: Dict[LabelType, int],
        target_labels: List[LabelType],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
    ) -> None:
        assert len(target_labels) == len(matching_threshold_list)
        self.target_labels: List[LabelType] = target_labels
        self.matching_mode = matching_mode

        # CLEAR results for each class
        self.clears: List[CLEAR] = []
        # Calculate score for each target labels
        for target_label, matching_threshold in zip(target_labels, matching_threshold_list):
            object_results = object_results_dict[target_label]
            num_ground_truth = num_ground_truth_dict[target_label]
            clear_ = CLEAR(
                object_results=object_results,
                num_ground_truth=num_ground_truth,
                target_labels=[target_label],
                matching_mode=matching_mode,
                matching_threshold_list=[matching_threshold],
            )
            self.clears.append(clear_)

    def summarize(self) -> Dict[str, Any]:
        ret = {}
        ret.update(sum_clear(self.clears))
        return ret

    def __str__(self) -> str:
        """__str__ method

        Returns:
            str: Formatted string.
        """
        str_: str = "\n"

        # === Total ===
        summary = self.summarize()
        # CLEAR
        str_ += f"[TOTAL] MOTA: {summary['MOTA']:.3f}, MOTP: {summary['MOTP']:.3f}, IDSW: {summary['IDSW']:d} "

        str_ += f"({self.matching_mode.value})\n"
        # === For each label ===
        # --- Table ---
        str_ += "\n"
        # --- Label ----
        str_ += "|      Label |"
        for clear in self.clears:
            str_ += f" {clear.target_labels}({clear.matching_threshold_list[0]}) | "

        str_ += "\n"
        str_ += "| :--------: |"
        for _ in self.clears:
            str_ += " :---: |"
        str_ += "\n"
        for name in CLEAR.metrics:
            str_ += f"|   {name} |"
            for clear_ in self.clears:
                score: Union[int, float] = clear_.results[name]
                if isinstance(score, int):
                    str_ += f" {score:d} |"
                else:
                    str_ += f" {score:.3f} |"
            str_ += "\n"

        return str_
