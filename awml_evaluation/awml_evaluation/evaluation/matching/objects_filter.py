"""[summary]
This module has filter function for objects.

function(
    object_results: List[DynamicObjectWithResult],
    params*,
) -> List[DynamicObjectWithResult]

or

function(
    objects: List[DynamicObject],
    params*,
) -> List[DynamicObject]
"""


from logging import getLogger
from typing import List
from typing import Optional
from typing import Tuple

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.threshold import LabelThreshold
from awml_evaluation.common.threshold import get_label_threshold
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithResult

logger = getLogger(__name__)


def filter_object_results(
    object_results: List[DynamicObjectWithResult],
    target_labels: Optional[List[AutowareLabel]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_pos_distance_list: Optional[List[float]] = None,
    min_pos_distance_list: Optional[List[float]] = None,
) -> List[DynamicObjectWithResult]:
    """[summary]
    Filter DynamicObjectWithResult to filter ground truth objects.

    Args:
        object_results (List[DynamicObjectWithResult]): The object results
        max_x_position_list (Optional[List[float]], optional):
                The threshold list of maximum x-axis position for each object.
                Return the object that
                - max_x_position < object x-axis position < max_x_position.
                This param use for range limitation of detection algorithm.
        max_y_position_list (Optional[List[float]], optional):
                The threshold list of maximum y-axis position for each object.
                Return the object that
                - max_y_position < object y-axis position < max_y_position.
                This param use for range limitation of detection algorithm.
        max_pos_distance_list (Optional[List[float]], optional):
                Maximum distance threshold list for object. Defaults to None.
        min_pos_distance_list (Optional[List[float]], optional):
                Minimum distance threshold list for object. Defaults to None.
    """
    # list handling
    if isinstance(target_labels, AutowareLabel):
        target_labels_ = [target_labels]
    else:
        target_labels_ = target_labels

    filtered_object_results: List[DynamicObjectWithResult] = []
    for object_result in object_results:
        is_target = True
        is_target = _is_target_object(
            dynamic_object=object_result.predicted_object,
            target_labels=target_labels_,
            max_x_position_list=max_x_position_list,
            max_y_position_list=max_y_position_list,
            max_pos_distance_list=max_pos_distance_list,
            min_pos_distance_list=min_pos_distance_list,
        )
        if is_target:
            filtered_object_results.append(object_result)
    return filtered_object_results


def filter_ground_truth_objects(
    objects: List[DynamicObject],
    target_labels: Optional[List[AutowareLabel]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_pos_distance_list: Optional[List[float]] = None,
    min_pos_distance_list: Optional[List[float]] = None,
) -> List[DynamicObject]:
    """[summary]
    Filter DynamicObject to filter ground truth objects.

    Args:
        objects (List[DynamicObject]): The objects you want to filter
        target_labels Optional[List[AutowareLabel]], optional):
                The target label to evaluate. If object label is in this parameter,
                this function appends to return objects. Defaults to None.
        max_pos_distance_list (Optional[List[float]], optional):
                Maximum distance threshold list for object. Defaults to None.
        min_pos_distance_list (Optional[List[float]], optional):
                Minimum distance threshold list for object. Defaults to None.

    Returns:
        List[DynamicObject]: Filtered object
    """
    # list handling
    if isinstance(target_labels, AutowareLabel):
        target_labels_ = [target_labels]
    else:
        target_labels_ = target_labels

    filtered_objects: List[DynamicObject] = []
    for object_ in objects:
        is_target = True
        is_target = _is_target_object(
            dynamic_object=object_,
            target_labels=target_labels_,
            max_x_position_list=max_x_position_list,
            max_y_position_list=max_y_position_list,
            max_pos_distance_list=max_pos_distance_list,
            min_pos_distance_list=min_pos_distance_list,
        )
        if is_target:
            filtered_objects.append(object_)
    return filtered_objects


def divide_tp_fp_objects(
    object_results: List[DynamicObjectWithResult],
    target_labels: Optional[List[AutowareLabel]],
    matching_mode: Optional[MatchingMode] = None,
    matching_threshold_list: Optional[List[float]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
) -> Tuple[List[DynamicObjectWithResult], List[DynamicObjectWithResult]]:
    """[summary]
    Divide TP (True Positive) objects and FP (False Positive) objects
    from Prediction condition positive objects.

    Args:
        object_results (List[DynamicObjectWithResult]): The object results you want to filter
        target_labels Optional[List[AutowareLabel]], optional):
                The target label to evaluate. If object label is in this parameter,
                this function appends to return objects. Defaults to None.
        matching_mode (Optional[MatchingMode], optional):
                The matching mode to evaluate. Defaults to None.
        matching_threshold_list (Optional[List[float]], optional):
                The matching threshold to evaluate. Defaults to None.
                For example, if matching_mode = IOU3d and matching_threshold = 0.5,
                and IoU of the object is higher than "matching_threshold",
                this function appends to return objects.
        confidence_threshold_list (Optional[List[float]], optional):
                The confidence threshold list. If predicted object's confidence is higher than
                this parameter, this function appends to return objects.
                It is often used to visualization.
                Defaults to None.

    Returns:
        Tuple[List[DynamicObjectWithResult], List[DynamicObjectWithResult]]: tp_objects, fp_objects
    """

    tp_objects: List[DynamicObjectWithResult] = []
    fp_objects: List[DynamicObjectWithResult] = []
    for object_result in object_results:
        matching_threshold_ = get_label_threshold(
            semantic_label=object_result.predicted_object.semantic_label,
            target_labels=target_labels,
            threshold_list=matching_threshold_list,
        )
        # Whether label is correct
        is_correct: bool = object_result.is_result_correct(
            matching_mode=matching_mode,
            matching_threshold=matching_threshold_,
        )

        # Whether cofidence is above threshold
        if confidence_threshold_list:
            confidence_threshold: Optional[float] = get_label_threshold(
                semantic_label=object_result.predicted_object.semantic_label,
                target_labels=target_labels,
                threshold_list=confidence_threshold_list,
            )
            if confidence_threshold:
                is_confidence: bool = (
                    object_result.predicted_object.semantic_score > confidence_threshold
                )
                is_correct = is_correct and is_confidence

        if is_correct:
            tp_objects.append(object_result)
        else:
            fp_objects.append(object_result)
    return tp_objects, fp_objects


def get_fn_objects(
    ground_truth_objects: List[DynamicObject],
    object_results: List[DynamicObjectWithResult],
    target_labels: Optional[List[AutowareLabel]] = None,
    matching_mode: Optional[MatchingMode] = None,
    matching_threshold_list: Optional[List[float]] = None,
) -> List[DynamicObject]:
    """[summary]
    Get FN (False Negative) objects from ground truth objects by using object result

    Args:
        ground_truth_objects (List[DynamicObject]): The ground truth objects
        object_results (List[DynamicObjectWithResult]): The object results
        target_labels Optional[List[AutowareLabel]], optional):
                The target label to evaluate. If object label is in this parameter,
                this function appends to return objects. Defaults to None.
        matching_mode (Optional[MatchingMode], optional):
                The matching mode to evaluate. Defaults to None.
        matching_threshold_list (Optional[List[float]], optional):
                The matching threshold to evaluate. Defaults to None.
                For example, if matching_mode = IOU3d and matching_threshold = 0.5,
                and IoU of the object is higher than "matching_threshold",
                this function appends to return objects.

    Returns:
        List[DynamicObject]: FN (False Negative) objects
    """

    fn_objects: List[DynamicObject] = []
    for ground_truth_object in ground_truth_objects:
        correspond_result: Optional[DynamicObjectWithResult] = _get_correspond_object_result(
            ground_truth_object,
            object_results,
        )
        if correspond_result:
            matching_threshold_ = get_label_threshold(
                semantic_label=correspond_result.predicted_object.semantic_label,
                target_labels=target_labels,
                threshold_list=matching_threshold_list,
            )
            is_result_correct_: bool = correspond_result.is_result_correct(
                matching_mode=matching_mode,
                matching_threshold=matching_threshold_,
            )
            if not is_result_correct_:
                fn_objects.append(ground_truth_object)
        else:
            fn_objects.append(ground_truth_object)
    return fn_objects


def _get_correspond_object_result(
    ground_truth_object: List[DynamicObject],
    object_results: List[DynamicObjectWithResult],
) -> Optional[DynamicObjectWithResult]:
    """[summary]
    Get the correspond object result to ground truth object

    Args:
        ground_truth_object (List[DynamicObject]): Ground truth object
        object_results (List[DynamicObjectWithResult]): object result

    Returns:
        Optional[DynamicObjectWithResult]: Correspond object result to ground truth object
    """

    correspond_result: Optional[DynamicObjectWithResult] = None
    for object_result in object_results:
        if ground_truth_object == object_result.ground_truth_object:
            if correspond_result:
                if (
                    correspond_result.get_distance_error_bev()
                    < object_result.get_distance_error_bev()
                ):
                    correspond_result = object_result
            else:
                correspond_result = object_result
    return correspond_result


def _is_target_object(
    dynamic_object: DynamicObject,
    target_labels: Optional[List[AutowareLabel]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_pos_distance_list: Optional[List[float]] = None,
    min_pos_distance_list: Optional[List[float]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
) -> bool:
    """[summary]
    The function judging whether the dynamic object is target or not.
    This function used to filtering for both of ground truths and object results.

    Args:
        dynamic_object (DynamicObject): The dynamic object
        target_labels Optional[List[AutowareLabel]], optional):
                The target label to evaluate. If object label is in this parameter,
                this function appends to return objects. Defaults to None.
        max_x_position_list (Optional[List[float]], optional):
                The threshold list of maximum x-axis position for each object.
                Return the object that
                - max_x_position < object x-axis position < max_x_position.
                This param use for range limitation of detection algorithm.
        max_y_position_list (Optional[List[float]], optional):
                The threshold list of maximum y-axis position for each object.
                Return the object that
                - max_y_position < object y-axis position < max_y_position.
                This param use for range limitation of detection algorithm.
        max_pos_distance_list (Optional[List[float]], optional):
                Maximum distance threshold list for object. Defaults to None.
        min_pos_distance_list (Optional[List[float]], optional):
                Minimum distance threshold list for object. Defaults to None.
        confidence_threshold_list (Optional[List[float]], optional):
                The confidence threshold list. If predicted object's confidence is higher than
                this parameter, this function appends to return objects.
                It is often used to visualization.
                Defaults to None.

    Returns:
        bool: If the object is filter target, return True
    """
    label_threshold = LabelThreshold(
        semantic_label=dynamic_object.semantic_label,
        target_labels=target_labels,
    )
    is_target = True

    if target_labels:
        is_target = is_target and dynamic_object.semantic_label in target_labels

    if is_target and confidence_threshold_list:
        confidence_threshold = label_threshold.get_label_threshold(confidence_threshold_list)
        is_target = is_target and dynamic_object.semantic_score > confidence_threshold

    if is_target and max_x_position_list:
        max_x_position = label_threshold.get_label_threshold(max_x_position_list)
        is_target = is_target and abs(dynamic_object.state.position[0]) < max_x_position

    if is_target and max_y_position_list:
        max_y_position = label_threshold.get_label_threshold(max_y_position_list)
        is_target = is_target and abs(dynamic_object.state.position[1]) < max_y_position

    if is_target and max_pos_distance_list:
        max_pos_distance = label_threshold.get_label_threshold(max_pos_distance_list)
        is_target = is_target and dynamic_object.get_distance_bev() < max_pos_distance

    if is_target and min_pos_distance_list:
        min_pos_distance = label_threshold.get_label_threshold(min_pos_distance_list)
        is_target = is_target and dynamic_object.get_distance_bev() > min_pos_distance

    return is_target
