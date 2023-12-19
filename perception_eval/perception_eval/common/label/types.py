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
from typing import List
from typing import Union


class AutowareLabel(Enum):
    """[summary]
    Autoware label enum.
    See https://github.com/tier4/autoware_iv_msgs/blob/main/autoware_perception_msgs/msg/object_recognition/Semantic.msg
    """

    UNKNOWN = "unknown"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    BICYCLE = "bicycle"
    MOTORBIKE = "motorbike"
    PEDESTRIAN = "pedestrian"
    ANIMAL = "animal"

    # for FP validation
    FP = "false_positive"

    def __str__(self) -> str:
        return self.value


class TrafficLightLabel(Enum):
    # except of classification
    TRAFFIC_LIGHT = "traffic_light"

    # classification
    GREEN = "green"
    RED = "red"
    YELLOW = "yellow"
    RED_STRAIGHT = "red_straight"
    RED_LEFT = "red_left"
    RED_LEFT_STRAIGHT = "red_left_straight"
    RED_LEFT_DIAGONAL = "red_left_diagonal"
    RED_RIGHT = "red_right"
    RED_RIGHT_STRAIGHT = "red_right_straight"
    RED_RIGHT_DIAGONAL = "red_right_diagonal"
    YELLOW_RIGHT = "yellow_right"

    # unknown is used in both detection and classification
    UNKNOWN = "unknown"

    # for FP validation
    FP = "false_positive"

    def __str__(self) -> str:
        return self.value


class CommonLabel(Enum):
    UNKNOWN = (AutowareLabel.UNKNOWN, TrafficLightLabel.UNKNOWN)
    FP = (AutowareLabel.FP, TrafficLightLabel.FP)

    def __eq__(self, label: LabelType) -> bool:
        return label in self.value

    def __str__(self) -> str:
        if self == CommonLabel.UNKNOWN:
            return "unknown"
        elif self == CommonLabel.FP:
            return "false_positive"
        else:
            raise ValueError(f"Unexpected element: {self}")


LabelType = Union[AutowareLabel, TrafficLightLabel]


class SemanticLabel:
    """
    Attributes:
        label (LabelType): Corresponding label.
        name (str): Label before converted.
        attributes (List[str]): List of attributes. Defaults to [].

    Args:
        label (LabelType): LabelType instance.
        name (str): Original label name.
        attributes (List[str]): List of attributes. Defaults to [].
    """

    def __init__(self, label: LabelType, name: str, attributes: List[str] = []) -> None:
        self.label = label
        self.name = name
        self.attributes = attributes

    def contains(self, key: str) -> bool:
        """Check whether self.name contains input attribute.

        Args:
            key (str): Target name or attribute.

        Returns:
            bool: Indicates whether input self.name contains input attribute.
        """
        assert isinstance(key, str), f"Expected type is str, but got {type(key)}"
        return key in self.name or key in self.attributes

    def contains_any(self, keys: List[str]) -> bool:
        assert isinstance(keys, (list, tuple)), f"Expected type is sequence, but got {type(keys)}"
        return any([self.contains(key) for key in keys])

    def is_fp(self) -> bool:
        """Returns `True`, if myself is `false_positive` label.

        Returns:
            bool: Whether myself is `false_positive`.
        """
        return self.label == CommonLabel.FP

    def is_unknown(self) -> bool:
        """Returns `True`, if myself is `unknown` label.

        Returns:
            bool: Whether myself is `unknown`.
        """
        return self.label == CommonLabel.UNKNOWN

    def __eq__(self, other: SemanticLabel) -> bool:
        return self.label == other.label
