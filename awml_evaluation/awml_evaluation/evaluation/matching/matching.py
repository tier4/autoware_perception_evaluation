from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.object_result import DynamicObjectWithResult


class MatchingMode(Enum):
    """[summary]
    The mode enum for matching algorithm.

    CENTERDISTANCE: The center distance
    IOU3d : 3d IoU (Intersection over Union)
    """

    CENTERDISTANCE = "center_distance"
    IOU3d = "iou_3d"


def filter_tp_objects(
    object_results: List[DynamicObjectWithResult],
    target_labels: Optional[List[AutowareLabel]] = None,
    min_pos_distance: Optional[float] = None,
    max_pos_distance: Optional[float] = None,
    threshold_confidence: Optional[float] = None,
    matching_mode: Optional[MatchingMode] = None,
    matching_threshold: Optional[float] = None,
) -> List[DynamicObjectWithResult]:
    """[summary]
    Filter DynamicObjectWithResult to TP object results

    Args:
        object_results (List[DynamicObjectWithResult]): The object results you want to filter
        target_labels (List[str]): The target label to evaluate. If object label is in
                                   this parameter, this function appends to return objects.
                                   Defaults to None.
        threshold_confidence (float): The confidence threshold. If predicted object's confidence is
                                      higher than this parameter, this function appends to return
                                      objects. It is used to visualization. Defaults to None.
        matching_mode (MatchingMode): The matching mode to evaluate. Defaults to None.
        matching_threshold (float): The matching threshold to evaluate. Defaults to None.
                                    For example, if matching_mode = IOU3d and
                                    matching_threshold = 0.5, and IoU of the object is higher
                                    than "matching_threshold", this function appends to return
                                    objects.
        min_pos_distance (float): Minimum distance for object. Defaults to None.
        max_pos_distance (float): Maximum distance for object. Defaults to None.

    Returns:
        List[DynamicObjectWithResult]: Filtered object result

    Example
        This function is used for use AP calculation to choose matching TP object
        or FP object (like low IoU)

        # filter predicted object and results by iou_threshold and target_labels
        filtered_object_results: List[DynamicObjectWithResult] = filter_tp_objects(
            object_results=object_results,
            target_labels=self.target_labels,
            matching_mode=self.matching_mode,
            matching_threshold=self.matching_threshold,
        )

    """
    filtered_objects: List[DynamicObjectWithResult] = []
    for object_result in object_results:
        is_target = True
        is_target_object = _is_target_object(
            object_result.predicted_object, target_labels, min_pos_distance, max_pos_distance
        )
        is_target = is_target and is_target_object
        if matching_mode == MatchingMode.CENTERDISTANCE:
            is_target = is_target and object_result.center_distance < matching_threshold
        if matching_mode == MatchingMode.IOU3d:
            is_target = is_target and object_result.iou_3d > matching_threshold
        if threshold_confidence:
            is_target = (
                is_target and object_result.predicted_object.semantic_score > threshold_confidence
            )

        if is_target:
            filtered_objects.append(object_result)

    return filtered_objects


def filter_ground_truth_objects(
    objects: List[DynamicObject],
    target_labels: Optional[List[AutowareLabel]] = None,
    min_pos_distance: Optional[float] = None,
    max_pos_distance: Optional[float] = None,
) -> List[DynamicObject]:
    """[summary]
    Filter DynamicObject fo filter ground truth objects

    Args:
        objects (List[DynamicObject]): The objects you want to filter
        target_labels (List[str]): The target label to evaluate. If object label is in
                                   this parameter, this function appends to return objects.
                                   Defaults to None.
        min_pos_distance (float): Minimum distance for object. Defaults to None.
        max_pos_distance (float): Maximum distance for object. Defaults to None.

    Returns:
        List[DynamicObject]: Filtered object
    """
    filtered_objects: List[DynamicObject] = []
    for object_ in objects:
        is_target = True
        is_target_object = _is_target_object(
            object_, target_labels, min_pos_distance, max_pos_distance
        )
        is_target = is_target and is_target_object
        if is_target:
            filtered_objects.append(object_)
    return filtered_objects


def _is_target_object(
    dynamic_object: DynamicObject,
    target_labels: Optional[List[AutowareLabel]] = None,
    min_pos_distance: Optional[float] = None,
    max_pos_distance: Optional[float] = None,
) -> bool:
    """[summary]
    The function judging whether filter target or not.

    Args:
        dynamic_object (DynamicObject): The dynamic object
        target_labels (List[str]): The target label to evaluate. If object label is in
                                   this parameter, this function appends to return objects.
                                   Defaults to None.
        min_pos_distance (float): Minimum distance for object. Defaults to None.
        max_pos_distance (float): Maximum distance for object. Defaults to None.

    Returns:
        bool: If the object is filter target, return True
    """
    is_target = True
    if target_labels:
        is_target = is_target and dynamic_object.semantic_label in target_labels
    if max_pos_distance:
        distance = dynamic_object.get_distance_bev()
        is_target = is_target and distance < max_pos_distance
    if min_pos_distance:
        distance = dynamic_object.get_distance_bev()
        is_target = is_target and distance > min_pos_distance

    return is_target


def divide_tp_fp_objects(
    object_results: List[DynamicObjectWithResult],
) -> Tuple[List[DynamicObjectWithResult], List[DynamicObjectWithResult]]:
    """[summary]
    Divide TP objects and FP objects.

    Args:
        object_results (List[DynamicObjectWithResult]): The object results you want to filter

    Returns:
        Tuple[List[DynamicObjectWithResult], List[DynamicObjectWithResult]]: tp_objects, fp_objects

    """
    tp_objects = []
    fp_objects = []
    for object_result in object_results:
        if object_result.is_label_correct:
            tp_objects.append(object_result)
        else:
            fp_objects.append(object_result)
    return tp_objects, fp_objects


def get_fn_objects(
    tp_objects: List[DynamicObjectWithResult],
    ground_truth_objects: List[DynamicObjectWithResult],
) -> DynamicObjectWithResult:
    """[summary]

    Args:
        tp_objects (List[DynamicObjectWithResult]): [description]
        ground_truth_objects (List[DynamicObjectWithResult]): [description]

    Returns:
        DynamicObjectWithResult: fn objects
    """
    raise NotImplementedError()
