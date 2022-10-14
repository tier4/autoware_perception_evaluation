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

from logging import getLogger
import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from perception_eval.common.label import AutowareLabel
from perception_eval.common.object import DynamicObject
from perception_eval.common.object import RoiObject
from perception_eval.common.threshold import LabelThreshold
from perception_eval.common.threshold import get_label_threshold
from perception_eval.evaluation.matching.object_matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

logger = getLogger(__name__)


def filter_object_results(
    frame_id: str,
    object_results: List[DynamicObjectWithPerceptionResult],
    target_labels: Optional[List[AutowareLabel]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_distance_list: Optional[List[float]] = None,
    min_distance_list: Optional[List[float]] = None,
    min_point_numbers: Optional[List[int]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
    target_uuids: Optional[str] = None,
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
        max_distance_list (Optional[List[float]], optional):
                Maximum distance threshold list for object. Defaults to None.
        min_distance_list (Optional[List[float]], optional):
                Minimum distance threshold list for object. Defaults to None.
        target_uuids (Optional[List[str]]): The list of ground truths' target uuid.
                This is unused is_gt=True. Defaults to None.
        ego2map (Optional[np.ndarray]): The array of matrix to transform from ego coords to map coords.
                This is only needed when frame_id=map. Defaults to None.
    """

    filtered_object_results: List[DynamicObjectWithPerceptionResult] = []
    for object_result in object_results:
        is_target: bool = _is_target_object(
            frame_id=frame_id,
            dynamic_object=object_result.estimated_object,
            target_labels=target_labels,
            max_x_position_list=max_x_position_list,
            max_y_position_list=max_y_position_list,
            max_distance_list=max_distance_list,
            min_distance_list=min_distance_list,
            confidence_threshold_list=confidence_threshold_list,
            ego2map=ego2map,
        )
        if is_target and object_result.ground_truth_object:
            is_target = is_target and _is_target_object(
                frame_id=frame_id,
                dynamic_object=object_result.ground_truth_object,
                target_labels=target_labels,
                max_x_position_list=max_x_position_list,
                max_y_position_list=max_y_position_list,
                max_distance_list=max_distance_list,
                min_distance_list=min_distance_list,
                min_point_numbers=min_point_numbers,
                target_uuids=target_uuids,
                ego2map=ego2map,
            )
        elif target_uuids and object_result.ground_truth_object is None:
            is_target = False

        if is_target:
            filtered_object_results.append(object_result)

    return filtered_object_results


def filter_objects(
    frame_id: str,
    objects: List[Union[DynamicObject, RoiObject]],
    is_gt: bool,
    target_labels: Optional[List[AutowareLabel]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_distance_list: Optional[List[float]] = None,
    min_distance_list: Optional[List[float]] = None,
    min_point_numbers: Optional[List[int]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
    target_uuids: Optional[List[str]] = None,
    ego2map: Optional[np.ndarray] = None,
) -> List[Union[DynamicObject, RoiObject]]:
    """[summary]
    Filter DynamicObject to filter ground truth objects.

    Args:
        frame_id (str): Frame id.
        objects (List[DynamicObject]): The objects you want to filter.
        is_gt (bool)
        target_labels Optional[List[AutowareLabel]], optional):
                The target label to evaluate. If object label is in this parameter,
                this function appends to return objects. Defaults to None.
        max_distance_list (Optional[List[float]], optional):
                Maximum distance threshold list for object. Defaults to None.
        min_distance_list (Optional[List[float]], optional):
                Minimum distance threshold list for object. Defaults to None.
        min_point_numbers (Optional[List[int]]):
                Min point numbers. This is only used if is_gt=True.
                For example, if target_labels is ["car", "bike", "pedestrian"],
                min_point_numbers [5, 0, 0] means
                Car bboxes including 4 points are filtered out.
                Car bboxes including 5 points are NOT filtered out.
                Bike and Pedestrian bboxes are not filtered out(All bboxes are used when calculating metrics.)
        target_uuids (Optional[List[str]]): The list of ground truths' target uuid.
                This is only used if is_gt=True. Defaults to None.
        ego2map (Optional[np.ndarray]): The array of matrix to transform from ego coords to map coords.
                This is only needed when frame_id=map. Defaults to None.

    Returns:
        List[DynamicObject]: Filtered object
    """

    filtered_objects: List[DynamicObject] = []
    for object_ in objects:
        if is_gt:
            is_target: bool = _is_target_object(
                frame_id=frame_id,
                dynamic_object=object_,
                target_labels=target_labels,
                max_x_position_list=max_x_position_list,
                max_y_position_list=max_y_position_list,
                max_distance_list=max_distance_list,
                min_distance_list=min_distance_list,
                min_point_numbers=min_point_numbers,
                target_uuids=target_uuids,
                ego2map=ego2map,
            )
        else:
            is_target: bool = _is_target_object(
                frame_id=frame_id,
                dynamic_object=object_,
                target_labels=target_labels,
                max_x_position_list=max_x_position_list,
                max_y_position_list=max_y_position_list,
                max_distance_list=max_distance_list,
                min_distance_list=min_distance_list,
                confidence_threshold_list=confidence_threshold_list,
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
    ground_truth_objects: List[Union[DynamicObject, RoiObject]],
    object_results: Optional[List[DynamicObjectWithPerceptionResult]],
    tp_objects: Optional[List[DynamicObjectWithPerceptionResult]],
) -> List[Union[DynamicObject, RoiObject]]:
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

    fn_objects: List[Union[DynamicObject, RoiObject]] = []
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
    ground_truth_object: Union[DynamicObject, RoiObject],
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
    dynamic_object: Union[DynamicObject, RoiObject],
    target_labels: Optional[List[AutowareLabel]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_distance_list: Optional[List[float]] = None,
    min_distance_list: Optional[List[float]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
    min_point_numbers: Optional[List[int]] = None,
    target_uuids: Optional[List[str]] = None,
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
        max_distance_list (Optional[List[float]], optional):
                Maximum distance threshold list for object. Defaults to None.
        min_distance_list (Optional[List[float]], optional):
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
        target_uuids (Optional[List[str]]): The list of target uuid. Defaults to None.

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

    if isinstance(dynamic_object, DynamicObject):
        assert frame_id in ("map", "base_link"), f"Unexpected frame id: {frame_id}"
        position_: Tuple[float, float, float] = dynamic_object.state.position
        if frame_id == "map":
            assert ego2map is not None, "When frame_id is map, ego2map must be specified"
            pos_arr: np.ndarray = np.append(position_, 1.0)
            position_ = tuple(np.linalg.inv(ego2map).dot(pos_arr)[:3].tolist())
            # TODO: DynamicObject.get_distance_bev() doesn't support map coords
            bev_distance_: float = math.hypot(position_[0], position_[1])
        else:
            bev_distance_: float = dynamic_object.get_distance_bev()

    if is_target and max_x_position_list is not None:
        max_x_position = label_threshold.get_label_threshold(max_x_position_list)
        is_target = is_target and abs(position_[0]) < max_x_position

    if is_target and max_y_position_list is not None:
        max_y_position = label_threshold.get_label_threshold(max_y_position_list)
        is_target = is_target and abs(position_[1]) < max_y_position

    if is_target and max_distance_list is not None:
        max_distance = label_threshold.get_label_threshold(max_distance_list)
        is_target = is_target and bev_distance_ < max_distance

    if is_target and min_distance_list is not None:
        min_distance = label_threshold.get_label_threshold(min_distance_list)
        is_target = is_target and bev_distance_ > min_distance

    if is_target and min_point_numbers is not None:
        min_point_number = label_threshold.get_label_threshold(min_point_numbers)
        is_target = is_target and dynamic_object.pointcloud_num >= min_point_number

    if is_target and target_uuids is not None:
        assert isinstance(target_uuids, list)
        assert all([isinstance(uuid, str) for uuid in target_uuids])
        is_target = is_target and dynamic_object.uuid in target_uuids

    return is_target


def divide_objects(
    objects: List[Union[DynamicObject, RoiObject, DynamicObjectWithPerceptionResult]],
    target_labels: Optional[List[AutowareLabel]] = None,
) -> Dict[AutowareLabel, List[Union[DynamicObject, RoiObject, DynamicObjectWithPerceptionResult]]]:
    """[summary]
    Divide DynamicObject or DynamicObjectWithPerceptionResult for each label as dict.

    Args:
        objects (List[Union[DynamicObject, DynamicObjectWithPerceptionResult]]):
            The list of DynamicObject or DynamicObjectWithPerceptionResult.
        target_labels (Optional[List[AutowareLabel]]): If this is specified, create empty list even
            if there is no object having specified label. Defaults to None.

    Returns:
        ret (Dict[AutowareLabel, List[Union[DynamicObject, DynamicObjectWithPerceptionResult]]]):
            key is label, item is list of DynamicObject or DynamicObjectWithPerceptionResult.
            It depends on the input type of object.
    """
    if target_labels is not None:
        ret = {label: [] for label in target_labels}
    else:
        ret: Dict[AutowareLabel, List[Union[DynamicObject, RoiObject]]] = {}

    for obj in objects:
        if isinstance(obj, (DynamicObject, RoiObject)):
            label: AutowareLabel = obj.semantic_label
        elif isinstance(obj, DynamicObjectWithPerceptionResult):
            label: AutowareLabel = obj.estimated_object.semantic_label
        else:
            raise TypeError(f"Unexpected object type: {type(obj)}")

        if label not in ret.keys():
            ret[label] = [obj]
        else:
            ret[label].append(obj)
    return ret


def divide_objects_to_num(
    objects: List[Union[DynamicObject, RoiObject, DynamicObjectWithPerceptionResult]],
    target_labels: Optional[List[AutowareLabel]] = None,
) -> Dict[AutowareLabel, int]:
    """[summary]
    Divide objects to the number of them for each label as dict.

    Args:
        objects (List[Union[DynamicObject, DynamicObjectWithPerceptionResult]]):
            The list of DynamicObject or DynamicObjectWithPerceptionResult.
        target_labels (Optional[List[AutowareLabel]]): If this is specified, create empty list even
            if there is no object having specified label. Defaults to None.

    Returns:
        ret (Dict[AutowareLabel, int]): key is label, item is the number of objects.
    """
    if target_labels is not None:
        ret = {label: 0 for label in target_labels}
    else:
        ret: Dict[AutowareLabel, int] = {}

    for obj in objects:
        if isinstance(obj, (DynamicObject, RoiObject)):
            label: AutowareLabel = obj.semantic_label
        elif isinstance(obj, DynamicObjectWithPerceptionResult):
            label: AutowareLabel = obj.estimated_object.semantic_label
        else:
            raise TypeError(f"Unexpected object type: {type(obj)}")

        if label not in ret.keys():
            ret[label] = 1
        else:
            ret[label] += 1
    return ret
