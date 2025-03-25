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

# cspell: ignore leftdiagonal, rightdiagonal

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from perception_eval.common.evaluation_task import EvaluationTask

if TYPE_CHECKING:
    from perception_eval.common import ObjectType


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

    # for label type
    LABEL_TYPE = "AutowareLabel"

    def __str__(self) -> str:
        return self.value

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        return {"label_type": AutowareLabel.LABEL_TYPE.value, "value": self.value}

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> AutowareLabel:
        """Deserialize the data to AutowareLabel."""
        return cls(data["value"])


class TrafficLightLabel(Enum):
    # except of classification
    TRAFFIC_LIGHT = "traffic_light"

    # classification
    # === for vehicle ===
    GREEN = "green"
    GREEN_STRAIGHT = "green_straight"
    GREEN_LEFT = "green_left"
    GREEN_RIGHT = "green_right"
    YELLOW = "yellow"
    YELLOW_STRAIGHT = "yellow_straight"
    YELLOW_LEFT = "yellow_left"
    YELLOW_RIGHT = "yellow_right"
    YELLOW_STRAIGHT_LEFT = "yellow_straight_left"
    YELLOW_STRAIGHT_RIGHT = "yellow_straight_right"
    YELLOW_STRAIGHT_LEFT_RIGHT = "yellow_straight_left_right"
    RED = "red"
    RED_STRAIGHT = "red_straight"
    RED_LEFT = "red_left"
    RED_RIGHT = "red_right"
    RED_STRAIGHT_LEFT = "red_straight_left"
    RED_STRAIGHT_RIGHT = "red_straight_right"
    RED_STRAIGHT_LEFT_RIGHT = "red_straight_left_right"
    RED_LEFT_DIAGONAL = "red_left_diagonal"
    RED_RIGHT_DIAGONAL = "red_right_diagonal"

    # unknown is used in both detection and classification
    UNKNOWN = "unknown"

    # for FP validation
    FP = "false_positive"

    # for label type
    LABEL_TYPE = "TrafficLightLabel"

    def __str__(self) -> str:
        return self.value

    def label_type(self) -> str:
        """Label type."""
        return "TrafficLightLabel"

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        return {"label_type": TrafficLightLabel.LABEL_TYPE.value, "value": self.value}

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> AutowareLabel:
        """Deserialize the data to AutowareLabel."""
        return cls(data["value"])


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


@dataclass
class LabelInfo:
    """Label data class.

    Attributes:
        label (LabelType): Corresponding label.
        name (str): Label before converted.
        num (int): The number of each label.
    """

    label: LabelType
    name: str
    num: int = 0


class Label:
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
        self.label: LabelType = label
        self.name: str = name
        self.attributes: List[str] = attributes

    def __reduce__(self) -> Tuple[Label, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling."""
        return (self.__class__, (self.label, self.name, self.attributes))

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

    def __eq__(self, other: Label) -> bool:
        return self.label == other.label

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        return {
            "label": self.label.value,
            "name": self.name,
            "attributes": self.attributes,
        }

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> Label:
        """Deserialize data to this class object."""
        return cls(
            label=LabelType(data["label"]),
            name=data["name"],
            attributes=data["attributes"],
        )


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
        (AutowareLabel.CAR, "vehicle.emergency (ambulance & police)"),
        (AutowareLabel.CAR, "vehicle.police"),
        (AutowareLabel.CAR, "vehicle.fire"),
        (AutowareLabel.CAR, "vehicle.ambulance"),
        (AutowareLabel.PEDESTRIAN, "pedestrian"),
        (AutowareLabel.PEDESTRIAN, "stroller"),
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
        (AutowareLabel.FP, "false_positive"),
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
            (AutowareLabel.CAR, "vehicle.construction"),
            (AutowareLabel.BICYCLE, "motorbike"),
            (AutowareLabel.BICYCLE, "motorcycle"),
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
            (AutowareLabel.TRUCK, "vehicle.construction"),
            (AutowareLabel.MOTORBIKE, "motorbike"),
            (AutowareLabel.MOTORBIKE, "motorcycle"),
            (AutowareLabel.MOTORBIKE, "vehicle.motorcycle"),
        ]
    return pair_list


def _get_traffic_light_paris(
    evaluation_task: EvaluationTask,
) -> List[Tuple[TrafficLightLabel, str]]:
    if evaluation_task == EvaluationTask.CLASSIFICATION2D:
        pair_list: List[Tuple[TrafficLightLabel, str]] = [
            (TrafficLightLabel.GREEN, "green"),
            (TrafficLightLabel.GREEN_STRAIGHT, "green_straight"),
            (TrafficLightLabel.GREEN_LEFT, "green_left"),
            (TrafficLightLabel.GREEN_RIGHT, "green_right"),
            (TrafficLightLabel.YELLOW, "yellow"),
            (TrafficLightLabel.YELLOW_STRAIGHT, "yellow_straight"),
            (TrafficLightLabel.YELLOW_LEFT, "yellow_left"),
            (TrafficLightLabel.YELLOW_RIGHT, "yellow_right"),
            (TrafficLightLabel.YELLOW_STRAIGHT_LEFT, "yellow_straight_left"),
            (TrafficLightLabel.YELLOW_STRAIGHT_LEFT_RIGHT, "yellow_straight_right"),
            (TrafficLightLabel.RED, "red"),
            (TrafficLightLabel.RED_STRAIGHT, "red_straight"),
            (TrafficLightLabel.RED_LEFT, "red_left"),
            (TrafficLightLabel.RED_RIGHT, "red_right"),
            (TrafficLightLabel.RED_STRAIGHT_LEFT, "red_straight_left"),
            (TrafficLightLabel.RED_STRAIGHT_RIGHT, "red_straight_right"),
            (TrafficLightLabel.RED_STRAIGHT_LEFT_RIGHT, "red_straight_left_right"),
            (TrafficLightLabel.RED_RIGHT_DIAGONAL, "red_right_diagonal"),
            (TrafficLightLabel.RED_RIGHT_DIAGONAL, "red_rightdiagonal"),
            (TrafficLightLabel.RED_LEFT_DIAGONAL, "red_left_diagonal"),
            (TrafficLightLabel.RED_LEFT_DIAGONAL, "red_leftdiagonal"),
            (TrafficLightLabel.UNKNOWN, "unknown"),
            (TrafficLightLabel.RED, "crosswalk_red"),
            (TrafficLightLabel.GREEN, "crosswalk_green"),
            (TrafficLightLabel.UNKNOWN, "crosswalk_unknown"),
            (TrafficLightLabel.FP, "false_positive"),
        ]
    else:
        pair_list: List[Tuple[TrafficLightLabel, str]] = [
            (TrafficLightLabel.TRAFFIC_LIGHT, "traffic_light"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "green"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "green_straight"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "green_left"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "green_right"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "yellow"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "yellow_straight"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "yellow_left"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "yellow_right"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "yellow_straight_left"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "yellow_straight_right"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_straight"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_left"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_right"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_straight_left"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_straight_right"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_straight_left_right"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_right_diagonal"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_rightdiagonal"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_left_diagonal"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "red_leftdiagonal"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "crosswalk_red"),
            (TrafficLightLabel.TRAFFIC_LIGHT, "crosswalk_green"),
            (TrafficLightLabel.UNKNOWN, "unknown"),
            (TrafficLightLabel.UNKNOWN, "crosswalk_unknown"),
            (TrafficLightLabel.FP, "false_positive"),
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


def is_same_label(object1: ObjectType, object2: ObjectType) -> bool:
    """Indicate whether both objects have same label.

    Args:
    ----
        object1 (ObjectType): An object.
        object2 (ObjectType): An object.

    Returns:
    -------
        bool: Return True if both labels are same.
    """
    return object1.semantic_label == object2.semantic_label
