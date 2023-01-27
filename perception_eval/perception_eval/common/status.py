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
