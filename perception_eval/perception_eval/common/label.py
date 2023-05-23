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

from perception_eval.common.evaluation_task import EvaluationTask


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

    def __str__(self) -> str:
        return self.value


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


class Label:
    """
    Args:
        label (LabelType): LabelType instance.
        name (str): Original label name.
        attributes (List[str]): List of attributes. Defaults to [].
    """

    def __init__(self, label: LabelType, name: str, attributes: List[str] = []) -> None:
        self.label: LabelType = label
        self.name: str = name
        self.attributes: List[str] = attributes

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

    def __eq__(self, other: Label) -> bool:
        return self.label == other.label

    def __hash__(self):
        return hash(tuple((self.label, self.name, tuple(self.attributes))))


class LabelConverter:
    """A class to convert string label name to LabelType instance.

    Attributes:
        evaluation_task (EvaluationTask): EvaluationTask instance.
        labels (List[LabelInfo]): The list of label to convert.
        label_type (LabelType): This is determined by `label_prefix`.

    Args:
        evaluation_task (EvaluationTask): EvaluationTask instance.
        merge_similar_labels (bool): Whether merge similar labels.
            If True,
                - BUS, TRUCK, TRAILER -> CAR
                - MOTORBIKE, CYCLIST -> BICYCLE
        label_prefix (str): Prefix of label, [autoware, traffic_light]. Defaults to autoware.

    Raises:
        NotImplementedError: For prefix named `blinker` and `brake_lamp` is under construction.
        ValueError: When unexpected prefix is specified.
    """

    def __init__(
        self,
        evaluation_task: Union[str, EvaluationTask],
        merge_similar_labels: bool,
        label_prefix: str,
        count_label_number: bool = False,
    ) -> None:

        self.evaluation_task: EvaluationTask = (
            evaluation_task
            if isinstance(evaluation_task, EvaluationTask)
            else EvaluationTask.from_value(evaluation_task)
        )
        self.count_label_number: bool = count_label_number

        if label_prefix == "autoware":
            self.label_type = AutowareLabel
            pair_list = _get_autoware_pairs(merge_similar_labels)
        elif label_prefix == "traffic_light":
            self.label_type = TrafficLightLabel
            pair_list = _get_traffic_light_paris(evaluation_task)
        elif label_prefix in ("blinker", "brake_lamp"):
            raise NotImplementedError(f"{label_prefix} is under construction.")
        else:
            raise ValueError(f"Unexpected `label_prefix`: {label_prefix}")

        self.label_infos: List[LabelInfo] = [LabelInfo(label, name) for label, name in pair_list]

        logging.debug(f"label {self.label_infos}")

    def convert_label(
        self,
        name: str,
        attributes: List[str] = [],
    ) -> Label:
        """Convert label name and attributes to Label instance.

        Args:
            label (str): Label name you want to convert to any LabelType object.
            count_label_number (bool): Whether to count how many labels have been converted.
                Defaults to False.

        Returns:
            Label: Converted label.
        """
        return_label: Optional[Label] = None
        for label_info in self.label_infos:
            if name.lower() == label_info.name:
                if self.count_label_number:
                    label_info.num += 1
                    if return_label is not None:
                        logging.error(f"Label {name} is already converted to {return_label}.")
                return_label = Label(label_info.label, name, attributes)
                break
        if return_label is None:
            logging.warning(f"Label {name} is not registered.")
            return_label = Label(self.label_type.UNKNOWN, name, attributes)
        return return_label

    def convert_name(self, name: str) -> LabelType:
        """Convert label name to LabelType instance.

        Args:
            label (str): Label name you want to convert to any LabelType object.
            count_label_number (bool): Whether to count how many labels have been converted.
                Defaults to False.

        Returns:
            Label: Converted label.
        """
        return_label: Optional[LabelType] = None
        for label_info in self.label_infos:
            if name.lower() == label_info.name:
                if self.count_label_number:
                    label_info.num += 1
                return_label = label_info.label
        if return_label is None:
            logging.warning(f"Label {name} is not registered.")
            return_label = self.label_type.UNKNOWN
        return return_label


def _get_autoware_pairs(merge_similar_labels: bool) -> List[Tuple[AutowareLabel, str]]:
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
        (AutowareLabel.PEDESTRIAN, "construction_worker"),
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


def _get_traffic_light_paris(
    evaluation_task: EvaluationTask,
) -> List[Tuple[TrafficLightLabel, str]]:
    if evaluation_task == EvaluationTask.CLASSIFICATION2D:
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
    else:
        pair_list: List[Tuple[TrafficLightLabel, str]] = [
            (TrafficLightLabel.TRAFFIC_LIGHT, "traffic_light"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "green"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "yellow"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_straight"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_left"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_left_straight"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_left_diagonal"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_right"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_right_straight"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_right_diagonal"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "yellow_right"),
            (TrafficLightLabel.UNKNOWN, "unknown"),
        ]
    return pair_list


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
    return [label_converter.convert_name(name) for name in target_labels]
