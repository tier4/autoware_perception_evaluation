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

from dataclasses import dataclass
from enum import Enum
import logging
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


class LabelBase(Enum):
    def __str__(self) -> str:
        return self.value

    @staticmethod
    def get_pairs() -> List[Tuple[LabelBase, str]]:
        raise NotImplementedError


class AutowareLabel(LabelBase):
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

    @staticmethod
    def get_pairs(merge_similar_labels: bool) -> List[Tuple[AutowareLabel, str]]:
        """[summary]
        Set pairs of AutowareLabel and str as list.

        Args:
            merge_similar_labels (bool): Whether merge similar labels.
                If True,
                    - BUS, TRUCK, TRAILER -> CAR
                    - MOTORBIKE, CYCLIST -> BICYCLE

        Returns:
            List[Tuple[AutowareLabel, str]]: The list of pair.
        """
        pair_list: List[Tuple[AutowareLabel, str]] = [
            (AutowareLabel.BICYCLE, "bicycle"),
            (AutowareLabel.BICYCLE, "vehicle.bicycle"),
            (AutowareLabel.CAR, "car"),
            (AutowareLabel.CAR, "vehicle.car"),
            (AutowareLabel.CAR, "vehicle.construction"),
            (AutowareLabel.CAR, "vehicle.emergency (ambulance & police)"),
            (AutowareLabel.CAR, "vehicle.police"),
            (AutowareLabel.CAR, "vehicle.fire"),
            (AutowareLabel.CAR, "vehicle.ambulance"),
            (AutowareLabel.PEDESTRIAN, "pedestrian"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.adult"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.child"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.construction_worker"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.personal_mobility"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.police_officer"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.stroller"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.wheelchair"),
            (AutowareLabel.UNKNOWN, "animal"),
            (AutowareLabel.UNKNOWN, "unknown"),
            (AutowareLabel.UNKNOWN, "movable_object.barrier"),
            (AutowareLabel.UNKNOWN, "movable_object.debris"),
            (AutowareLabel.UNKNOWN, "movable_object.pushable_pullable"),
            (AutowareLabel.UNKNOWN, "movable_object.trafficcone"),
            (AutowareLabel.UNKNOWN, "movable_object.traffic_cone"),
            (AutowareLabel.UNKNOWN, "static_object.bicycle rack"),
            (AutowareLabel.UNKNOWN, "static_object.bollard"),
            (AutowareLabel.UNKNOWN, "forklift"),
        ]
        if merge_similar_labels:
            pair_list += [
                (AutowareLabel.CAR, "bus"),
                (AutowareLabel.CAR, "vehicle.bus (bendy & rigid)"),
                (AutowareLabel.CAR, "vehicle.bus"),
                (AutowareLabel.CAR, "truck"),
                (AutowareLabel.CAR, "vehicle.truck"),
                (AutowareLabel.CAR, "trailer"),
                (AutowareLabel.CAR, "vehicle.trailer"),
                (AutowareLabel.BICYCLE, "motorbike"),
                (AutowareLabel.BICYCLE, "vehicle.motorcycle"),
            ]
        else:
            pair_list += [
                (AutowareLabel.BUS, "bus"),
                (AutowareLabel.BUS, "vehicle.bus (bendy & rigid)"),
                (AutowareLabel.BUS, "vehicle.bus"),
                (AutowareLabel.TRUCK, "truck"),
                (AutowareLabel.TRUCK, "vehicle.truck"),
                (AutowareLabel.TRUCK, "trailer"),
                (AutowareLabel.TRUCK, "vehicle.trailer"),
                (AutowareLabel.MOTORBIKE, "motorbike"),
                (AutowareLabel.MOTORBIKE, "vehicle.motorcycle"),
            ]
        return pair_list


class TrafficLightLabel(LabelBase):
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
    UNKNOWN = "unknown"

    @staticmethod
    def get_pairs() -> List[Tuple[TrafficLightLabel, str]]:
        pair_list: List[Tuple[TrafficLightLabel, str]] = [
            (TrafficLightLabel.GREEN, "green"),
            (TrafficLightLabel.RED, "red"),
            (TrafficLightLabel.YELLOW, "yellow"),
            (TrafficLightLabel.RED_STRAIGHT, "red_straight"),
            (TrafficLightLabel.RED_LEFT, "red_left"),
            (TrafficLightLabel.RED_LEFT_STRAIGHT, "red_left_straight"),
            (TrafficLightLabel.RED_LEFT_DIAGONAL, "red_left_diagonal"),
            (TrafficLightLabel.RED_RIGHT, "red_right"),
            (TrafficLightLabel.RED_RIGHT_STRAIGHT, "red_right_straight"),
            (TrafficLightLabel.RED_RIGHT_DIAGONAL, "red_right_diagonal"),
            (TrafficLightLabel.YELLOW_RIGHT, "yellow_right"),
            (TrafficLightLabel.UNKNOWN, "unknown"),
        ]
        return pair_list


LabelType = Union[AutowareLabel, TrafficLightLabel]


@dataclass
class LabelInfo:
    """[summary]
    Label data class

    Attributes:
        self.autoware_label (AutowareLabel): Corresponded Autoware label
        label (str): Label before converted
        num (int): The number of a label
    """

    label: LabelType
    name: str
    num: int = 0


class LabelConverter:
    """A class to convert string label name to LabelType instance.

    Attributes:
        self.labels (List[LabelInfo]): The list of label to convert.
        self.label_type (LabelType): This is determined by `label_prefix`.

    Args:
        merge_similar_labels (bool): Whether merge similar labels.
            If True,
                - BUS, TRUCK, TRAILER -> CAR
                - MOTORBIKE, CYCLIST -> BICYCLE
        label_prefix (str): Prefix of label, [autoware, traffic_light]. Defaults to autoware.

    Raises:
        NotImplementedError: For prefix named `blinker` and `brake_lamp` is under construction.
        ValueError: When unexpected prefix is specified.
    """

    def __init__(self, merge_similar_labels: bool, label_prefix: str = "autoware") -> None:
        if label_prefix == "autoware":
            self.label_type = AutowareLabel
            pair_list: List[Tuple[AutowareLabel, str]] = AutowareLabel.get_pairs(
                merge_similar_labels=merge_similar_labels,
            )
        elif label_prefix == "traffic_light":
            self.label_type = TrafficLightLabel
            pair_list: List[Tuple[TrafficLightLabel, str]] = TrafficLightLabel.get_pairs()
        elif label_prefix in ("blinker", "brake_lamp"):
            raise NotImplementedError(f"{label_prefix} is under construction.")
        else:
            raise ValueError(f"Unexpected `label_prefix`: {label_prefix}")

        self.labels: List[LabelInfo] = [LabelInfo(label, name) for label, name in pair_list]

        logging.debug(f"label {self.labels}")

    def convert_label(
        self,
        label: str,
        count_label_number: bool = False,
    ) -> LabelType:
        """Convert to LabelType instance from label name in string.

        Args:
            label (str): Label name you want to convert to any LabelType object.
            count_label_number (bool): Whether to count how many labels have been converted.
                Defaults to False.

        Returns:
            LabelType: Converted label.

        Examples:
            >>> converter = LabelConverter(False, "autoware")
            >>> converter.convert_label("car")
            <AutowareLabel.CAR: 'car'>
        """
        return_label: Optional[LabelType] = None
        for label_class in self.labels:
            if label.lower() == label_class.name:
                if count_label_number:
                    label_class.num += 1
                    if return_label is not None:
                        logging.error(f"Label {label} is already converted to {return_label}.")
                return_label = label_class.label
                break
        if return_label is None:
            logging.warning(f"Label {label} is not registered.")
            return_label = self.label_type.UNKNOWN
        return return_label


def set_target_lists(
    target_labels: Optional[List[str]],
    label_converter: LabelConverter,
) -> List[LabelType]:
    """Returns a LabelType list from a list of label names in string.

    If no label is specified, returns all LabelType elements.

    Args:
        target_labels (List[str]): The target class to evaluate
        label_converter (LabelConverter): Label Converter class

    Returns:
        List[LabelType]:  LabelType instance list.

    Examples:
        >>> converter = LabelConverter(False, "autoware")
        >>> set_target_lists(["car", "pedestrian"], converter)
        [<AutowareLabel.CAR: 'car'>, <AutowareLabel.PEDESTRIAN: 'pedestrian'>]
    """
    if target_labels is None or len(target_labels) == 0:
        return [label for label in label_converter.label_type]
    target_autoware_labels = []
    for target_label in target_labels:
        target_autoware_labels.append(label_converter.convert_label(target_label))
    return target_autoware_labels
