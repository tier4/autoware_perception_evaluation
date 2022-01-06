from enum import Enum
import pprint
from typing import List
from typing import Optional
from typing import Tuple

from pyquaternion.quaternion import Quaternion

from awml_evaluation.common.object import DynamicObject


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
        return str(object)
    elif hasattr(object, "_ast"):
        return class_to_dict(object._ast(), abbreviation)
    elif hasattr(object, "__iter__") and not isinstance(object, str):
        if abbreviation and len(object) > abbreviation:
            return f" --- length of element {len(object)} ---,"
        return [class_to_dict(v, abbreviation, class_key) for v in object]
    elif hasattr(object, "__dict__"):
        data = dict(
            [
                (key, class_to_dict(value, abbreviation, class_key))
                for key, value in object.__dict__.items()
                if not callable(value) and not key.startswith("_")
            ]
        )
        if class_key is not None and hasattr(object, "__class__"):
            data[class_key] = object.__class__.__name__
        return data
    else:
        return object


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

    output_objects = []
    for object_ in ground_truth_objects:
        position = (
            object_.state.position[0] + diff_distance[0],
            object_.state.position[1] + diff_distance[1],
            object_.state.position[2] + diff_distance[2],
        )

        if is_confidence_with_distance is None:
            semantic_score = object_.semantic_score
        else:
            distance_coefficient = object_.get_distance_bev() / 100.0
            distance_coefficient = min(distance_coefficient, 0.8)
            distance_coefficient = max(distance_coefficient, 0.2)
            if is_confidence_with_distance:
                semantic_score = object_.semantic_score * (1 - distance_coefficient)
            else:
                semantic_score = object_.semantic_score * distance_coefficient

        orientation = Quaternion(
            axis=object_.state.orientation.axis,
            radians=object_.state.orientation.radians + diff_yaw,
        )

        test_object_ = DynamicObject(
            unix_time=object_.unix_time,
            position=position,
            orientation=orientation,
            size=object_.state.size,
            velocity=object_.state.velocity,
            semantic_score=semantic_score,
            semantic_label=object_.semantic_label,
            uuid=object_.uuid,
        )

        output_objects.append(test_object_)
    return output_objects
