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
from typing import List
from typing import Optional
from typing import Tuple

from perception_eval.common.object import DynamicObject
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
        for (k, v) in object.items():
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
    formatted_str: str = (
        "\n" + pprint.pformat(dict, indent=1, width=120, depth=None, compact=True) + "\n"
    )
    return formatted_str


def get_objects_with_difference(
    ground_truth_objects: List[DynamicObject],
    diff_distance: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    diff_yaw: float = 0.0,
    is_confidence_with_distance: Optional[bool] = None,
) -> List[DynamicObject]:
    """[summary]
    Get objects with distance and yaw difference for test.

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
            distance_coefficient: float = object_.get_distance_bev() / 100.0
            distance_coefficient = max(min(distance_coefficient, 0.8), 0.2)
            if is_confidence_with_distance:
                semantic_score = object_.semantic_score * (1 - distance_coefficient)
            else:
                semantic_score = object_.semantic_score * distance_coefficient

        orientation: Quaternion = Quaternion(
            axis=object_.state.orientation.axis,
            radians=object_.state.orientation.radians + diff_yaw,
        )

        predicted_positions, predicted_orientations, predicted_confidences = _get_prediction_params(
            object_,
            diff_distance,
            diff_yaw,
        )

        test_object_: DynamicObject = DynamicObject(
            unix_time=object_.unix_time,
            position=position,
            orientation=orientation,
            size=object_.state.size,
            velocity=object_.state.velocity,
            semantic_score=semantic_score,
            semantic_label=object_.semantic_label,
            pointcloud_num=object_.pointcloud_num,
            uuid=object_.uuid,
            predicted_positions=predicted_positions,
            predicted_orientations=predicted_orientations,
            predicted_confidences=predicted_confidences,
        )

        output_objects.append(test_object_)
    return output_objects


def _get_prediction_params(
    object_: DynamicObject,
    diff_distance: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    diff_yaw: float = 0.0,
) -> Tuple[
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
        predicted_positions (List[List[Tuple[float]]]): List of positions
        predicted_orientations (List[List[Quaternion]]): List of quaternions.
        predicted_confidences (List[float]): List of confidences.
    """
    if object_.predicted_paths is None:
        return None, None, None

    predicted_positions: List[List[Tuple[float]]] = []
    predicted_orientations: List[List[Quaternion]] = []
    predicted_confidences: List[float] = []
    for paths in object_.predicted_paths:
        positions = []
        orientations = []
        for path in paths:
            positions.append(
                (
                    path.position[0] + diff_distance[0],
                    path.position[1] + diff_distance[1],
                    path.position[2] + diff_distance[2],
                )
            )
            orientations.append(
                Quaternion(
                    axis=path.orientation.axis,
                    radians=path.orientation.radians + diff_yaw,
                )
            )
        predicted_positions.append(positions)
        predicted_orientations.append(orientations)
        predicted_confidences.append(paths.confidence)

    return predicted_positions, predicted_orientations, predicted_confidences
