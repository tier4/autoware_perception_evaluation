# Copyright 2023 TIER IV, Inc.

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

from enum import Enum
from typing import List
from typing import Tuple
from typing import Union


class MatchingStatus(Enum):
    TP = "TP"
    FP = "FP"
    FN = "FN"
    TN = "TN"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Union[MatchingStatus, str]) -> bool:
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

    def is_positive(self) -> bool:
        """Indicates whether current status is TP or FP.

        Returns:
            bool: Returns `True` if status is TP or FP.
        """
        return self in (MatchingStatus.TP, MatchingStatus.FP)

    def is_negative(self) -> bool:
        """Indicates whether current status is TN or FN.

        Returns:
            bool: Returns `True` if status is TN or FN.
        """
        return self in (MatchingStatus.TN, MatchingStatus.FN)

    def is_true(self) -> bool:
        """Indicates whether current status is TP or TN.

        Returns:
            bool: Returns `True` if status is TP or FN.
        """
        return self in (MatchingStatus.TP, MatchingStatus.TN)

    def is_false(self) -> bool:
        """Indicates whether current status is FP or FN.

        Returns:
            bool: Returns `True` if status is FP or FN.
        """
        return self in (MatchingStatus.FP, MatchingStatus.FN)


class StatusRate:
    """Class to get rate of each matching status, TP/FP/TN/FN."""

    def __init__(
        self,
        status: MatchingStatus,
        status_frame_nums: List[int],
        total_frame_nums: List[int],
    ) -> None:
        self.status = status
        self.status_frame_nums = status_frame_nums
        self.total_frame_nums = total_frame_nums

    @property
    def rate(self) -> float:
        return self.__get_rate()

    def __get_num_status_frames(self) -> int:
        return len(self.status_frame_nums)

    def __get_num_total_frames(self) -> int:
        return len(self.total_frame_nums)

    def __get_rate(self) -> float:
        num_status_frames: int = self.__get_num_status_frames()
        num_total_frames: int = self.__get_num_total_frames()
        return (
            num_status_frames / num_total_frames
            if num_status_frames != 0.0 and num_total_frames != 0.0
            else float("inf")
        )


StatusRates = Tuple[StatusRate, StatusRate, StatusRate, StatusRate]


class GroundTruthStatus:
    """Class for keeping and calculating status information of each matching status for one GT.

    Attributes:
        uuid (str): UUID of ground truth object.
        total_frame_nums (List[int]): List of frame numbers, which GT is evaluated.
        tp_frame_nums (List[int]): List of frame numbers, which GT is evaluated as TP.
        fp_frame_nums (List[int]): List of frame numbers, which GT is evaluated as FP.
        tn_frame_nums (List[int]): List of frame numbers, which GT is evaluated as TN.
        fn_frame_nums (List[int]): List of frame numbers, which GT is evaluated as FN.

    Args:
        uuid (str): object uuid
    """

    def __init__(self, uuid: str) -> None:
        self.uuid: str = uuid

        self.total_frame_nums: List[int] = []
        self.tp_frame_nums: List[int] = []
        self.fp_frame_nums: List[int] = []
        self.tn_frame_nums: List[int] = []
        self.fn_frame_nums: List[int] = []

    def add_status(self, status: MatchingStatus, frame_num: int) -> None:
        self.total_frame_nums.append(frame_num)
        if status == MatchingStatus.TP:
            self.tp_frame_nums.append(frame_num)
        elif status == MatchingStatus.FP:
            self.fp_frame_nums.append(frame_num)
        elif status == MatchingStatus.TN:
            self.tn_frame_nums.append(frame_num)
        elif status == MatchingStatus.FN:
            self.fn_frame_nums.append(frame_num)
        else:
            raise ValueError(f"Unexpected status: {status}")

    def get_status_rates(self) -> StatusRates:
        """Returns frame rates for each status.

        Returns:
            StatusRates: Rates [TP, FP, TN, FN] order.
        """
        return (
            StatusRate(MatchingStatus.TP, self.tp_frame_nums, self.total_frame_nums),
            StatusRate(MatchingStatus.FP, self.fp_frame_nums, self.total_frame_nums),
            StatusRate(MatchingStatus.TN, self.tn_frame_nums, self.total_frame_nums),
            StatusRate(MatchingStatus.FN, self.fn_frame_nums, self.total_frame_nums),
        )

    def __eq__(self, uuid: str) -> bool:
        return self.uuid == uuid


def get_scene_rates(status_list: List[GroundTruthStatus]) -> Tuple[float, float, float, float]:
    """Returns TP/FP/TN/FN rates from all `GroundTruthStatus`.

    If `status_list` is empty, returns sequence of `float("inf")`.

    Args:
        status_list (List[GroundTruthStatus]): All GT status.

    Returns:
        Tuple[float, float, float, float]: Sequence of rates, (TP, FP, TN, FN) order.
    """
    num_total_frame: int = 0
    num_tp_frame: int = 0
    num_fp_frame: int = 0
    num_tn_frame: int = 0
    num_fn_frame: int = 0
    for status in status_list:
        num_total_frame += len(status.total_frame_nums)
        num_tp_frame += len(status.tp_frame_nums)
        num_fp_frame += len(status.fp_frame_nums)
        num_tn_frame += len(status.tn_frame_nums)
        num_fn_frame += len(status.fn_frame_nums)

    if num_total_frame == 0:
        return float("inf"), float("inf"), float("inf"), float("inf")
    else:
        return (
            num_tp_frame / num_total_frame,
            num_fp_frame / num_total_frame,
            num_tn_frame / num_total_frame,
            num_fn_frame / num_total_frame,
        )
