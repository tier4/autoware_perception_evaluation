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

from enum import Enum
import pprint
import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import Label
from perception_eval.common.label import LabelType
from perception_eval.common.label import TrafficLightLabel
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.shape import Shape
from perception_eval.common.transform import TransformDict
from pyquaternion.quaternion import Quaternion


def format_class_for_log(
    object: object,
    abbreviation: Optional[int] = None,
) -> str:
    """[summary]
    Convert class object to str to save log

    Args:
        object (object): Class object which you want to convert for str
        abbreviation (Optional[int]): If len(list_object) > abbreviation,
                                      then abbreviate the result.

    Returns:
        str: str converted from class object

    Note:
        Reference is below.
        https://stackoverflow.com/questions/1036409/recursively-convert-python-object-graph-to-dictionary

    """
    return format_dict_for_log(class_to_dict(object, abbreviation))


def class_to_dict(
    object: object,
    abbreviation: Optional[int] = None,
    class_key: Optional[str] = None,
) -> dict:
    """[summary]
    Convert class object to dict

    Args:
        object (object): Class object which you want to convert to dict
        abbreviation (Optional[int]): If len(list_object) > abbreviation,
                                      then abbreviate the result
        class_key (Optional[str]): class key for dict

    Returns:
        dict: Dict converted from class object

    Note:
        Reference is below.
        https://stackoverflow.com/questions/1036409/recursively-convert-python-object-graph-to-dictionary

    """

    if isinstance(object, dict):
        data = {}
        for k, v in object.items():
            data[k] = class_to_dict(v, abbreviation, class_key)
        return data
    elif isinstance(object, Enum):
        return str(object)  # type: ignore
    elif hasattr(object, "_ast"):
        return class_to_dict(object._ast(), abbreviation)  # type: ignore
    elif hasattr(object, "__iter__") and not isinstance(object, str):
        if abbreviation and len(object) > abbreviation:  # type: ignore
            return f" --- length of element {len(object)} ---,"  # type: ignore
        return [class_to_dict(v, abbreviation, class_key) for v in object]  # type: ignore
    elif hasattr(object, "__dict__"):
        data = dict(
            [
                (key, class_to_dict(value, abbreviation, class_key))
                for key, value in object.__dict__.items()
                if not callable(value) and not key.startswith("_")
            ]
        )
        if class_key is not None and hasattr(object, "__class__"):
            data[class_key] = object.__class__.__name__  # type: ignore
        return data
    else:
        return object  # type: ignore


def format_dict_for_log(
    dict: dict,
) -> str:
    """
    Format dict class to str for logger

    Args:
        dict (dict): dict which you want to format for logger

    Returns:
        (str) formatted str
    """
    formatted_str: str = "\n" + pprint.pformat(dict, indent=1, width=120, depth=None, compact=True) + "\n"
    return formatted_str


def get_objects_with_difference(
    ground_truth_objects: List[DynamicObject],
    diff_distance: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    diff_yaw: float = 0.0,
    is_confidence_with_distance: Optional[bool] = None,
    label_to_unknown_rate: float = 1.0,
    transforms: Optional[TransformDict] = None,
    label_candidates: Optional[List[LabelType]] = None,
) -> List[DynamicObject]:
    """Get objects with distance and yaw difference for test.

    Args:
        ground_truth_objects (List[DynamicObject]):
                The ground truth objects.
        diff_distance (Tuple[float, float, float], optional):
                The parameter for difference of position. Defaults to
                (0.0, 0.0, 0.0).
        diff_yaw (float, optional):
                The parameter for difference of yaw angle. Defaults to 0.0.
        is_confidence_with_distance (Optional[bool], optional):
                If this param is None, confidence is same as input.
                If this param is True, confidence is lower (0.2 - 0.8 times)
                according to distance from base_link. Near object is higher
                coefficient like 0.8 and far object is lower like 0.2.
                If this param is False, confidence is lower (0.2 - 0.8 times).
                Near object is lower coefficient like 0.2 and far object is
                higher like 0.8.
                Defaults is None.
        label_unknown_rate (float): Rate to convert label into unknown randomly. Defaults to 0.5.

    Returns:
        List[DynamicObject]: objects with distance and yaw difference.
    """

    output_objects: List[DynamicObject] = []
    for object_ in ground_truth_objects:
        position: Tuple[float, float, float] = (
            object_.state.position[0] + diff_distance[0],
            object_.state.position[1] + diff_distance[1],
            object_.state.position[2] + diff_distance[2],
        )

        semantic_score: float = 0.0
        if is_confidence_with_distance is None:
            semantic_score = object_.semantic_score
        else:
            distance_coefficient: float = object_.get_distance_bev(transforms) / 100.0
            distance_coefficient = max(min(distance_coefficient, 0.8), 0.2)
            if is_confidence_with_distance:
                semantic_score = object_.semantic_score * (1 - distance_coefficient)
            else:
                semantic_score = object_.semantic_score * distance_coefficient

        orientation: Quaternion = Quaternion(
            axis=object_.state.orientation.axis,
            radians=object_.state.orientation.radians + diff_yaw,
        )

        relative_timestamps, predicted_positions, predicted_orientations, predicted_scores = _get_prediction_params(
            object_,
            diff_distance,
            diff_yaw,
        )

        shape: Shape = Shape(shape_type=object_.state.shape_type, size=object_.state.size)

        if label_to_unknown_rate < random.uniform(0.0, 1.0):
            if isinstance(object_.semantic_label.label, AutowareLabel):
                label = AutowareLabel.UNKNOWN
            elif isinstance(object_.semantic_label.label, TrafficLightLabel):
                label = TrafficLightLabel.UNKNOWN
            semantic_label = Label(label, "unknown")
        if label_candidates is not None:
            label: LabelType = random.choice(label_candidates)
            semantic_label = Label(label, label.value)
        else:
            semantic_label = object_.semantic_label

        test_object_: DynamicObject = DynamicObject(
            unix_time=object_.unix_time,
            frame_id=object_.frame_id,
            position=position,
            orientation=orientation,
            shape=shape,
            velocity=object_.state.velocity,
            semantic_score=semantic_score,
            semantic_label=semantic_label,
            pointcloud_num=object_.pointcloud_num,
            uuid=object_.uuid,
            relative_timestamps=relative_timestamps,
            predicted_positions=predicted_positions,
            predicted_orientations=predicted_orientations,
            predicted_scores=predicted_scores,
        )

        output_objects.append(test_object_)
    return output_objects


def _get_prediction_params(
    object_: DynamicObject,
    diff_distance: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    diff_yaw: float = 0.0,
) -> Tuple[
    Optional[List[List[int]]],
    Optional[List[List[Tuple[float]]]],
    Optional[List[List[Quaternion]]],
    Optional[List[float]],
]:
    """
    Get object's prediction parameters with distance and yaw difference for test.

    Args:
        object_ (DynamicObject): dynamic object.
        diff_distance (Tuple[float, float, float], optional):
                The parameter for difference of position. Defaults to
                (0.0, 0.0, 0.0).
        diff_yaw (float, optional):
                The parameter for difference of yaw angle. Defaults to 0.0.

    Returns:
        If the attribute of dynamic object named predicted_paths is None, returns None, None, None.
        relative_timestamps (List[List[int]]): List of relative timestamps.
        predicted_positions (List[List[Tuple[float]]]): List of positions
        predicted_orientations (List[List[Quaternion]]): List of quaternions.
        predicted_scores (List[float]): List of scores.
    """
    if object_.predicted_paths is None:
        return None, None, None, None

    relative_timestamps: List[List[int]] = []
    predicted_positions: List[List[Tuple[float]]] = []
    predicted_orientations: List[List[Quaternion]] = []
    predicted_scores: List[float] = []
    for path in object_.predicted_paths:
        positions = []
        orientations = []
        relative_timestamps.append(path.relative_timestamps)
        for state in path.states:
            positions.append(
                (
                    state.position[0] + diff_distance[0],
                    state.position[1] + diff_distance[1],
                    state.position[2] + diff_distance[2],
                )
            )
            orientations.append(
                Quaternion(
                    axis=state.orientation.axis,
                    radians=state.orientation.radians + diff_yaw,
                )
                if state.orientation is not None
                else None
            )
        predicted_positions.append(positions)
        predicted_orientations.append(orientations)
        predicted_scores.append(path.confidence)

    return relative_timestamps, predicted_positions, predicted_orientations, predicted_scores


def get_objects_with_difference2d(
    objects: List[DynamicObject2D],
    translate: Tuple[int, int] = None,
    label_to_unknown_rate: float = 1.0,
    label_candidates: Optional[List[LabelType]] = None,
) -> List[DynamicObject2D]:
    """Returns translated 2D objects.

    Args:
        objects (List[DynamicObject2D])
        translate (Optional[Tuple[int, int]]): Translation vector [tx, ty][px]. Defaults to None.
        label_unknown_rate (float): Rate to convert label into unknown randomly. Defaults to 0.5.

    Returns:
        List[DynamicObject2D]: List of translated objects.
    """
    output_objects: List[DynamicObject2D] = []
    for object_ in objects:
        if translate is not None and object_.roi is not None:
            offset_: Tuple[int, int] = (
                object_.roi.offset[0] + translate[0],
                object_.roi.offset[1] + translate[1],
            )
            roi = (*offset_, *object_.roi.size)
        else:
            roi = object_.roi

        if label_to_unknown_rate < random.uniform(0.0, 1.0):
            if isinstance(object_.semantic_label.label, AutowareLabel):
                label = AutowareLabel.UNKNOWN
            elif isinstance(object_.semantic_label.label, TrafficLightLabel):
                label = TrafficLightLabel.UNKNOWN
            semantic_label = Label(label, "unknown")
        else:
            semantic_label = object_.semantic_label

        if label_candidates is not None:
            label: LabelType = random.choice(label_candidates)
            semantic_label = Label(label, label.value)
        else:
            semantic_label = object_.semantic_label

        output_objects.append(
            DynamicObject2D(
                unix_time=object_.unix_time,
                frame_id=object_.frame_id,
                semantic_score=object_.semantic_score,
                semantic_label=semantic_label,
                roi=roi,
                uuid=object_.uuid,
                visibility=object_.visibility,
            )
        )
    return output_objects


def get_random_position_map(frame_ground_truth: FrameGroundTruth) -> Dict[str, ArrayLike]:
    """Return a container storing positions of corresponding uuids randomly.

    The positions are computed as ego position + random * 100.0.

    Args:
        frame_ground_truth (FrameGroundTruth):

    Returns:
        Dict[str, ArrayLike]: Position map.
    """
    output: Dict[str, ArrayLike] = {}
    ego2map = frame_ground_truth.transforms[(FrameID.BASE_LINK, FrameID.MAP)]
    for obj in frame_ground_truth.objects:
        output[obj.uuid] = np.random.rand(3) * 100.0 + ego2map.position
    return output
