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

from enum import Enum
from typing import Optional

from perception_eval.common.object_base import Object2DBase
from perception_eval.common.object_base import Roi
from perception_eval.common.status import Visibility


class TLColor(Enum):
    """Enum class of Traffic Light color."""

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


class TLArrow(Enum):
    """Enum class of Traffic Light arrow."""

    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    UP_LEFT = "up_left"
    UP_RIGHT = "up_eight"
    DOWN_LEFT = "down_left"
    DOWN_RIGHT = "down_right"


class TLStatus(Enum):
    """Enum class of Traffic Light status."""

    SOLID_OFF = "solid_off"
    SOLID_ON = "solid_on"
    FLASHING = "flashing"


class TrafficLight(Object2DBase):
    """[summary]
    Traffic Light class.

    Attributes:
        self.unix_time (int) : Unix time [us].
        self.semantic_label (TLColor): Color information.
        self.uuid (str): Traffic light ID.
        self.semantic_score (float): Confidence of estimation.
        self.roi (Optional[ROI]): ROI of traffic light. Defaults to None.
        self.arrow (Optional[TLArrow]): Arrow information of traffic light. Defaults to None.
        self.status (Optional[TLStatus]): Status of traffic light, especially pedestrian signal. Defaults to None.
        self.visibility (Optional[Visibility]): Visibility status of traffic light. Defaults to None.
    """

    def __init__(
        self,
        unix_time: int,
        semantic_label: TLColor,
        uuid: str,
        semantic_score: float,
        roi: Optional[Roi] = None,
        arrow: Optional[TLArrow] = None,
        status: Optional[TLStatus] = None,
        visibility: Optional[Visibility] = None,
    ) -> None:
        """
        Args:
            unix_time (int) : Unix time [us].
            semantic_label (TLColor): Color information.
            uuid (str): Traffic light ID.
            semantic_score (float): Confidence of estimation.
            roi (Optional[ROI]): ROI of traffic light. Defaults to None.
            arrow (Optional[TLArrow]): Arrow information of traffic light. Defaults to None.
            status (Optional[TLStatus]): Status of traffic light, especially pedestrian signal. Defaults to None.
            visibility (Optional[Visibility]): Visibility status of traffic light. Defaults to None.
        """
        super().__init__(
            unix_time=unix_time,
            semantic_score=semantic_score,
            roi=roi,
            uuid=uuid,
            visibility=visibility,
        )
        self.__semantic_label: TLColor = semantic_label
        self.arrow: Optional[TLArrow] = arrow
        self.status: Optional[TLStatus] = status

    @property
    def semantic_label(self) -> TLColor:
        return self.__semantic_label
