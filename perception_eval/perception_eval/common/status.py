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
import logging
from typing import Dict
from typing import Union

from perception_eval.common.evaluation_task import EvaluationTask


class FrameID(Enum):
    # 3D
    BASE_LINK = "base_link"
    MAP = "map"

    # 2D
    CAM_FRONT = "cam_front"
    CAM_FRONT_RIGHT = "cam_front_right"
    CAM_FRONT_LEFT = "cam_front_left"
    CAM_BACK = "cam_back"
    CAM_BACK_LEFT = "cam_back_left"
    CAM_BACK_RIGHT = "cam_back_right"
    CAM_TRAFFIC_LIGHT_NEAR = "cam_traffic_licht_near"
    CAM_TRAFFIC_LIGHT_FAR = "cam_traffic_light_far"

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, str):
            return self.value == __o
        return super().__eq__(__o)

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_value(cls, name: str) -> FrameID:
        """Returns FrameID from its value.

        NOTE:
            This method allow that input value is upper case.

        Args:
            name (str): Value in string.

        Returns:
            FrameID: Corresponding FrameID instance.
        """
        name = name.lower()
        for _, v in cls.__members__.items():
            if v == name:
                return v
        raise ValueError(f"Unexpected value: {name}")

    @classmethod
    def from_task(cls, task: Union[str, EvaluationTask]) -> FrameID:
        """Return FrameID from EvaluationTask.

        Args:
            task (Union[str, EvaluationTask]): Task name.

        Returns:
            FrameID: For DETECTION or SENSING, Returns BASE_LINK. For TRACKING or PREDICTION, returns MAP.

        Raises:
            ValueError: When `task` is evaluation for 2D input data.
        """
        if isinstance(task, str):
            task = EvaluationTask.from_value(task)

        if task.is_2d():
            raise ValueError(
                "For 2D task, FrameID must be initialized explicitly, or use `FrameID.from_value(name)`."
            )

        if task in (EvaluationTask.DETECTION, EvaluationTask.SENSING):
            return FrameID.BASE_LINK
        elif task in (EvaluationTask.TRACKING, EvaluationTask.PREDICTION):
            return FrameID.MAP
        else:
            raise ValueError(f"Unexpected task: {task}")


class Visibility(Enum):
    """Visibility status class.

    FULL
    MOST
    PARTIAL
    NONE
    UNAVAILABLE
    """

    FULL = "full"
    MOST = "most"
    PARTIAL = "partial"
    NONE = "none"
    UNAVAILABLE = "not available"

    @staticmethod
    def from_alias(name: str) -> Dict[str, Visibility]:
        if name == "v0-40":
            return Visibility.NONE
        elif name == "v40-60":
            return Visibility.PARTIAL
        elif name == "v60-80":
            return Visibility.MOST
        elif name == "v80-100":
            return Visibility.FULL
        else:
            logging.warning(
                f"level: {name} is not supported, Visibility.UNAVAILABLE will be assigned."
            )
            return Visibility.UNAVAILABLE

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, str):
            return self.value == __o
        return super().__eq__(__o)

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_value(cls, name: str) -> Visibility:
        """Returns Visibility instance from string.

        If `name` is not in the set of Visibility values, call self.from_alias(`name`).

        Args:
            name (str): Visibility name in string.

        Returns:
            Visibility: Visibility instance.

        Examples:
            >>> Visibility.from_value("most")
            Visibility.MOST
        """
        for k, v in cls.__members__.items():
            if v == name:
                return k
        return cls.from_alias(name)


class SensorModality(Enum):
    LIDAR = "lidar"
    CAMERA = "camera"
    RADAR = "radar"

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, str):
            return self.value == __o
        return super().__eq__(__o)

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_value(cls, name: str) -> SensorModality:
        """Returns the SensorModality instance from string.

        Args:
            name (str): Sensor name in string.

        Returns:
            SensorModality: SensorModality instance.

        Examples:
            >>> SensorModality.from_value("camera")
            SensorModality.CAMERA
        """
        for k, v in cls.__members__.items():
            if v == name:
                return k
