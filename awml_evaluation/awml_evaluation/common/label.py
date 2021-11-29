from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import List
from typing import Optional
from typing import Union

logger = getLogger(__name__)


class AutowareLabel(Enum):
    """[summary]
    Autoware lable enum.
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


@dataclass
class Label:
    """[summary]
    Label convert class

    Attributes:
        self.autoware_label (AutowareLabel): Corresponded Autoware label
        label (str): Label before converted
        num (int): The number of a lable
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

    def __init__(self) -> None:
        self.labels: List[Label] = []
        pair_list: List[Union[AutowareLabel, str]] = LabelConverter._set_pair_list()
        for label, autoware_label in pair_list:
            self.labels.append(Label(label, autoware_label))
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
                    if return_label:
                        logger.warning(f"Label {label} is already converted to {return_label}")
                return_label = label_class.autoware_label
        if not return_label:
            logger.warning(f"Label {label} is not registered")
            return_label = AutowareLabel.UNKNOWN
        return return_label

    @staticmethod
    def _set_pair_list() -> List[Union[AutowareLabel, str]]:
        pair_list: List[Union[AutowareLabel, str]] = [
            [AutowareLabel.BICYCLE, "bicycle"],
            [AutowareLabel.BICYCLE, "BICYCLE"],
            [AutowareLabel.BICYCLE, "vehicle.bicycle"],
            [AutowareLabel.BUS, "bus"],
            [AutowareLabel.BUS, "BUS"],
            [AutowareLabel.BUS, "vehicle.bus (bendy & rigid)"],
            [AutowareLabel.BUS, "vehicle.bus"],
            [AutowareLabel.CAR, "car"],
            [AutowareLabel.CAR, "CAR"],
            [AutowareLabel.CAR, "vehicle.car"],
            [AutowareLabel.CAR, "vehicle.construction"],
            [AutowareLabel.CAR, "vehicle.emergency (ambulance & police)"],
            [AutowareLabel.MOTORBIKE, "motorbike"],
            [AutowareLabel.MOTORBIKE, "MOTORBIKE"],
            [AutowareLabel.MOTORBIKE, "vehicle.motorcycle"],
            [AutowareLabel.PEDESTRIAN, "pedestrian"],
            [AutowareLabel.PEDESTRIAN, "PEDESTRIAN"],
            [AutowareLabel.PEDESTRIAN, "pedestrian.adult"],
            [AutowareLabel.PEDESTRIAN, "pedestrian.child"],
            [AutowareLabel.PEDESTRIAN, "pedestrian.construction_worker"],
            [AutowareLabel.PEDESTRIAN, "pedestrian.personal_mobility"],
            [AutowareLabel.PEDESTRIAN, "pedestrian.police_officer"],
            [AutowareLabel.PEDESTRIAN, "pedestrian.stroller"],
            [AutowareLabel.PEDESTRIAN, "pedestrian.wheelchair"],
            [AutowareLabel.TRUCK, "truck"],
            [AutowareLabel.TRUCK, "TRUCK"],
            [AutowareLabel.TRUCK, "vehicle.truck"],
            [AutowareLabel.TRUCK, "vehicle.trailer"],
            [AutowareLabel.UNKNOWN, "animal"],
            [AutowareLabel.UNKNOWN, "ANIMAL"],
            [AutowareLabel.UNKNOWN, "unknown"],
            [AutowareLabel.UNKNOWN, "UNKNOWN"],
            [AutowareLabel.UNKNOWN, "movable_object.barrier"],
            [AutowareLabel.UNKNOWN, "movable_object.debris"],
            [AutowareLabel.UNKNOWN, "movable_object.pushable_pullable"],
            [AutowareLabel.UNKNOWN, "movable_object.trafficcone"],
            [AutowareLabel.UNKNOWN, "movable_object.traffic_cone"],
            [AutowareLabel.UNKNOWN, "static_object.bicycle rack"],
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
