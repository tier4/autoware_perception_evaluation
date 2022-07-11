"""[summary]
This module has filter function for objects.

function(
    object_results: List[DynamicObjectWithPerceptionResult],
    params*,
) -> List[DynamicObjectWithPerceptionResult]

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
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
import numpy as np

logger = getLogger(__name__)


def filter_object_results(
    frame_id: str,
    object_results: List[DynamicObjectWithPerceptionResult],
    target_labels: Optional[List[AutowareLabel]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_pos_distance_list: Optional[List[float]] = None,
    min_pos_distance_list: Optional[List[float]] = None,
    ego2map: Optional[np.ndarray] = None,
) -> List[DynamicObjectWithPerceptionResult]:
    """[summary]
    Filter DynamicObjectWithPerceptionResult to filter ground truth objects.

    Args:
        object_results (List[DynamicObjectWithPerceptionResult]): The object results
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

    filtered_object_results: List[DynamicObjectWithPerceptionResult] = []
    for object_result in object_results:
        is_target: bool = _is_target_object(
            frame_id=frame_id,
            dynamic_object=object_result.estimated_object,
            target_labels=target_labels,
            max_x_position_list=max_x_position_list,
            max_y_position_list=max_y_position_list,
            max_pos_distance_list=max_pos_distance_list,
            min_pos_distance_list=min_pos_distance_list,
            ego2map=ego2map,
        )
        if is_target:
            filtered_object_results.append(object_result)
    return filtered_object_results


def filter_ground_truth_objects(
    frame_id: str,
    objects: List[DynamicObject],
    target_labels: Optional[List[AutowareLabel]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_pos_distance_list: Optional[List[float]] = None,
    min_pos_distance_list: Optional[List[float]] = None,
    min_point_numbers: List[int] = None,
    ego2map: Optional[np.ndarray] = None,
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
        min_point_numbers (List[int]):
                Min point numbers.
                For example, if target_labels is ["car", "bike", "pedestrian"],
                min_point_numbers [5, 0, 0] means
                Car bboxes including 4 points are filtered out.
                Car bboxes including 5 points are NOT filtered out.
                Bike and Pedestrian bboxes are not filtered out(All bboxes are used when calculating metrics.)

    Returns:
        List[DynamicObject]: Filtered object
    """

    filtered_objects: List[DynamicObject] = []
    for object_ in objects:
        is_target: bool = _is_target_object(
            frame_id=frame_id,
            dynamic_object=object_,
            target_labels=target_labels,
            max_x_position_list=max_x_position_list,
            max_y_position_list=max_y_position_list,
            max_pos_distance_list=max_pos_distance_list,
            min_pos_distance_list=min_pos_distance_list,
            min_point_numbers=min_point_numbers,
            ego2map=ego2map,
        )
        if is_target:
            filtered_objects.append(object_)
    return filtered_objects


def divide_tp_fp_objects(
    object_results: List[DynamicObjectWithPerceptionResult],
    target_labels: Optional[List[AutowareLabel]],
    matching_mode: Optional[MatchingMode] = None,
    matching_threshold_list: Optional[List[float]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
) -> Tuple[List[DynamicObjectWithPerceptionResult], List[DynamicObjectWithPerceptionResult]]:
    """[summary]
    Divide TP (True Positive) objects and FP (False Positive) objects
    from Prediction condition positive objects.

    Args:
        object_results (List[DynamicObjectWithPerceptionResult]): The object results you want to filter
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
                The confidence threshold list. If estimated object's confidence is higher than
                this parameter, this function appends to return objects.
                It is often used to visualization.
                Defaults to None.

    Returns:
        Tuple[List[DynamicObjectWithPerceptionResult], List[DynamicObjectWithPerceptionResult]]: tp_objects, fp_objects
    """

    tp_objects: List[DynamicObjectWithPerceptionResult] = []
    fp_objects: List[DynamicObjectWithPerceptionResult] = []
    for object_result in object_results:
        matching_threshold_: Optional[float] = get_label_threshold(
            semantic_label=object_result.estimated_object.semantic_label,
            target_labels=target_labels,
            threshold_list=matching_threshold_list,
        )

        # matching threshold
        is_correct: bool = True
        if matching_threshold_ is None:
            is_correct = object_result.is_label_correct
        else:
            is_correct = object_result.is_result_correct(
                matching_mode=matching_mode,
                matching_threshold=matching_threshold_,
            )

        # confidence threshold
        confidence_threshold_: Optional[float] = get_label_threshold(
            semantic_label=object_result.estimated_object.semantic_label,
            target_labels=target_labels,
            threshold_list=confidence_threshold_list,
        )
        if confidence_threshold_ is not None:
            is_confidence: bool = (
                object_result.estimated_object.semantic_score > confidence_threshold_
            )
            is_correct = is_correct and is_confidence

        if is_correct:
            tp_objects.append(object_result)
        else:
            fp_objects.append(object_result)
    return tp_objects, fp_objects


def get_fn_objects(
    ground_truth_objects: List[DynamicObject],
    object_results: Optional[List[DynamicObjectWithPerceptionResult]],
    tp_objects: Optional[List[DynamicObjectWithPerceptionResult]],
) -> List[DynamicObject]:
    """[summary]
    Get FN (False Negative) objects from ground truth objects by using object result

    Args:
        ground_truth_objects (List[DynamicObject]): The ground truth objects
        object_results (Optional[List[DynamicObjectWithPerceptionResult]]): The object results
        tp_objects (Optional[List[DynamicObjectWithPerceptionResult]]): TP results in object results

    Returns:
        List[DynamicObject]: FN (False Negative) objects
    """

    if object_results is None:
        return ground_truth_objects

    fn_objects: List[DynamicObject] = []
    for ground_truth_object in ground_truth_objects:
        is_fn_object: bool = _is_fn_object(
            ground_truth_object=ground_truth_object,
            object_results=object_results,
            tp_objects=tp_objects,
        )
        if is_fn_object:
            fn_objects.append(ground_truth_object)
    return fn_objects


def _is_fn_object(
    ground_truth_object: DynamicObject,
    object_results: List[DynamicObjectWithPerceptionResult],
    tp_objects: List[DynamicObjectWithPerceptionResult],
) -> bool:
    """[summary]
    Judge whether ground truth object is FN (False Negative) object.
    If there are TP TN object

    Args:
        ground_truth_object (DynamicObject): A ground truth object
        object_results (List[DynamicObjectWithPerceptionResult]): object result
        tp_objects (Optional[List[DynamicObjectWithPerceptionResult]]): TP results in object results

    Returns:
        bool: Whether ground truth object is FN (False Negative) object.
    """

    for object_result in object_results:
        if ground_truth_object == object_result.ground_truth_object and object_result in tp_objects:
            return False
    return True


def _is_target_object(
    frame_id: str,
    dynamic_object: DynamicObject,
    target_labels: Optional[List[AutowareLabel]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_pos_distance_list: Optional[List[float]] = None,
    min_pos_distance_list: Optional[List[float]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
    min_point_numbers: Optional[List[int]] = None,
    ego2map: Optional[np.ndarray] = None,
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
                The confidence threshold list. If estimated object's confidence is higher than
                this parameter, this function appends to return objects.
                It is often used to visualization.
                Defaults to None.
        min_point_numbers (List[int]):
                Min point numbers.
                For example, if target_labels is ["car", "bike", "pedestrian"],
                min_point_numbers [5, 0, 0] means
                Car bboxes including 4 points are filtered out.
                Car bboxes including 5 points are NOT filtered out.
                Bike and Pedestrian bboxes are not filtered out(All bboxes are used when calculating metrics.)

    Returns:
        bool: If the object is filter target, return True
    """
    label_threshold = LabelThreshold(
        semantic_label=dynamic_object.semantic_label,
        target_labels=target_labels,
    )
    is_target: bool = True

    if target_labels is not None:
        is_target = is_target and dynamic_object.semantic_label in target_labels

    if is_target and confidence_threshold_list is not None:
        confidence_threshold = label_threshold.get_label_threshold(confidence_threshold_list)
        is_target = is_target and dynamic_object.semantic_score > confidence_threshold

    assert frame_id in (
        "map",
        "base_link",
    ), f"frame_id must be in (map, base_link), but got {frame_id}"
    position_: Tuple[float, float, float] = dynamic_object.state.position
    if frame_id == "map":
        assert ego2map is not None, "When frame_id is map, ego2map must be specified"
        pos_arr: np.ndarray = np.append(position_, 1.0)
        position_ = tuple(np.linalg.inv(ego2map).dot(pos_arr)[:3].tolist())

    if is_target and max_x_position_list is not None:
        max_x_position = label_threshold.get_label_threshold(max_x_position_list)
        is_target = is_target and abs(position_[0]) < max_x_position

    if is_target and max_y_position_list is not None:
        max_y_position = label_threshold.get_label_threshold(max_y_position_list)
        is_target = is_target and abs(position_[1]) < max_y_position

    if is_target and max_pos_distance_list is not None:
        max_pos_distance = label_threshold.get_label_threshold(max_pos_distance_list)
        is_target = is_target and dynamic_object.get_distance_bev() < max_pos_distance

    if is_target and min_pos_distance_list is not None:
        min_pos_distance = label_threshold.get_label_threshold(min_pos_distance_list)
        is_target = is_target and dynamic_object.get_distance_bev() > min_pos_distance

    if is_target and min_point_numbers is not None:
        min_point_number = label_threshold.get_label_threshold(min_point_numbers)
        is_target = is_target and dynamic_object.pointcloud_num >= min_point_number

    return is_target


def filter_object_results_by_confidence(
    object_results: List[DynamicObjectWithPerceptionResult], confidence_thereshold: float
) -> List[DynamicObjectWithPerceptionResult]:
    """[summary]
    Filter object_results by confidence

    Args:
        object_results (List[DynamicObjectWithPerceptionResult]): The objects you want to filter
        confidence_thereshold (float): Confidence used by filtering. Remove objects which does not have low confidence

    Returns:
        object_results_with_high_confidence (List[DynamicObjectWithPerceptionResult]): The objects whose confidences are higher than confidence_thereshold
    """
    object_results_with_high_confidence: List[DynamicObjectWithPerceptionResult] = []
    for object_result in object_results:
        if object_result.estimated_object.semantic_score > confidence_thereshold:
            object_results_with_high_confidence.append(object_result)
    return object_results_with_high_confidence
