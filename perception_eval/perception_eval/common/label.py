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
from logging import getLogger
from typing import List
from typing import Optional
from typing import Tuple

logger = getLogger(__name__)


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


@dataclass
class Label:
    """[summary]
    Label convert class

    Attributes:
        self.autoware_label (AutowareLabel): Corresponded Autoware label
        label (str): Label before converted
        num (int): The number of a label
    """

    autoware_label: AutowareLabel
    label: str
    num: int = 0


class LabelConverter:
    """[summary]
    Label lapper between Autoware label and Tier4 dataset
    Convert from self.labels[i].label to self.labels[i].autoware_label

    Attributes:
        self.labels (List[Label]): The list of label to convert
    """

    def __init__(self, merge_similar_labels: bool) -> None:
        """
        Args:
            merge_similar_labels (bool): Whether merge similar labels.
                If True,
                    - BUS, TRUCK, TRAILER -> CAR
                    - MOTORBIKE, CYCLIST -> BICYCLE
        """
        self.labels: List[Label] = []
        pair_list: List[Tuple[AutowareLabel, str]] = LabelConverter._set_pair_list(
            merge_similar_labels
        )
        for autoware_label, label in pair_list:
            self.labels.append(Label(autoware_label, label))
        logger.debug(f"label {self.labels}")

    def convert_label(
        self,
        label: str,
        count_label_number: bool = False,
    ) -> AutowareLabel:
        """[summary]
        Convert from Nuscenes label to autoware label

        Args:
            label (str): The label you want to convert from predicted object
            count_label_number (bool): The flag of counting the number of labels. Default is False

        Returns:
            AutowareLabel: Converted label
        """
        return_label: Optional[AutowareLabel] = None
        for label_class in self.labels:
            if label == label_class.label:
                if count_label_number:
                    label_class.num += 1
                    if return_label is not None:
                        logger.error(f"Label {label} is already converted to {return_label}")
                return_label = label_class.autoware_label
        if return_label is None:
            logger.warning(f"Label {label} is not registered")
            return_label = AutowareLabel.UNKNOWN
        return return_label

    @staticmethod
    def _set_pair_list(merge_similar_labels: bool) -> List[Tuple[AutowareLabel, str]]:
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
            (AutowareLabel.BICYCLE, "BICYCLE"),
            (AutowareLabel.BICYCLE, "vehicle.bicycle"),
            (AutowareLabel.CAR, "car"),
            (AutowareLabel.CAR, "CAR"),
            (AutowareLabel.CAR, "vehicle.car"),
            (AutowareLabel.CAR, "vehicle.construction"),
            (AutowareLabel.CAR, "vehicle.emergency (ambulance & police)"),
            (AutowareLabel.PEDESTRIAN, "pedestrian"),
            (AutowareLabel.PEDESTRIAN, "PEDESTRIAN"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.adult"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.child"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.construction_worker"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.personal_mobility"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.police_officer"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.stroller"),
            (AutowareLabel.PEDESTRIAN, "pedestrian.wheelchair"),
            (AutowareLabel.UNKNOWN, "animal"),
            (AutowareLabel.UNKNOWN, "ANIMAL"),
            (AutowareLabel.UNKNOWN, "unknown"),
            (AutowareLabel.UNKNOWN, "UNKNOWN"),
            (AutowareLabel.UNKNOWN, "movable_object.barrier"),
            (AutowareLabel.UNKNOWN, "movable_object.debris"),
            (AutowareLabel.UNKNOWN, "movable_object.pushable_pullable"),
            (AutowareLabel.UNKNOWN, "movable_object.trafficcone"),
            (AutowareLabel.UNKNOWN, "movable_object.traffic_cone"),
            (AutowareLabel.UNKNOWN, "static_object.bicycle rack"),
        ]
        if merge_similar_labels:
            pair_list += [
                (AutowareLabel.CAR, "bus"),
                (AutowareLabel.CAR, "BUS"),
                (AutowareLabel.CAR, "vehicle.bus (bendy & rigid)"),
                (AutowareLabel.CAR, "vehicle.bus"),
                (AutowareLabel.CAR, "truck"),
                (AutowareLabel.CAR, "TRUCK"),
                (AutowareLabel.CAR, "vehicle.truck"),
                (AutowareLabel.CAR, "trailer"),
                (AutowareLabel.CAR, "TRAILER"),
                (AutowareLabel.CAR, "vehicle.trailer"),
                (AutowareLabel.BICYCLE, "motorbike"),
                (AutowareLabel.BICYCLE, "MOTORBIKE"),
                (AutowareLabel.BICYCLE, "vehicle.motorcycle"),
            ]
        else:
            pair_list += [
                (AutowareLabel.BUS, "bus"),
                (AutowareLabel.BUS, "BUS"),
                (AutowareLabel.BUS, "vehicle.bus (bendy & rigid)"),
                (AutowareLabel.BUS, "vehicle.bus"),
                (AutowareLabel.TRUCK, "truck"),
                (AutowareLabel.TRUCK, "TRUCK"),
                (AutowareLabel.TRUCK, "vehicle.truck"),
                (AutowareLabel.TRUCK, "trailer"),
                (AutowareLabel.TRUCK, "TRAILER"),
                (AutowareLabel.TRUCK, "vehicle.trailer"),
                (AutowareLabel.MOTORBIKE, "motorbike"),
                (AutowareLabel.MOTORBIKE, "MOTORBIKE"),
                (AutowareLabel.MOTORBIKE, "vehicle.motorcycle"),
            ]
        return pair_list


def set_target_lists(
    target_labels: List[str],
    label_converter: LabelConverter,
) -> List[AutowareLabel]:
    """[summary]
    Set the target class configure

    Args:
        target_labels (List[str]): The target class to evaluate
        label_converter (LabelConverter): Label Converter class

    Returns:
        List[AutowareLabel]:  The list of target class
    """
    target_autoware_labels = []
    for target_label in target_labels:
        target_autoware_labels.append(label_converter.convert_label(target_label))
    return target_autoware_labels
