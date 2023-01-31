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

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from perception_eval.common import ObjectType
from perception_eval.common.label import LabelType
from perception_eval.common.object import DynamicObject
from perception_eval.common.status import FrameID
from perception_eval.common.threshold import get_label_threshold
from perception_eval.common.threshold import LabelThreshold
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.matching import MatchingMode


def filter_object_results(
    object_results: List[DynamicObjectWithPerceptionResult],
    target_labels: Optional[List[LabelType]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_distance_list: Optional[List[float]] = None,
    min_distance_list: Optional[List[float]] = None,
    min_point_numbers: Optional[List[int]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
    target_uuids: Optional[str] = None,
    ego2map: Optional[np.ndarray] = None,
) -> List[DynamicObjectWithPerceptionResult]:
    """Filter DynamicObjectWithPerceptionResult considering both estimated and ground truth objects.

    If any of `target_labels`, `max_x_position_list`, `max_y_position_list`, `max_distance_list`, `min_distance_list`,
    `min_point_numbers` or `confidence_threshold_list` are specified, each of them must be same length list.

    It first filters `object_results` with input parameters considering estimated objects.
    After that, remained `object_results` are filtered with input parameters considering ground truth objects.

    Args:
        object_results (List[DynamicObjectWithPerceptionResult]): Object results list.
        target_labels Optional[List[LabelType]]): Filter target list of labels.
            Keep all `object_results` that both of their `estimated_object` and `ground_truth_object`
            have same label in this list. Defaults to None.
        max_x_position_list (Optional[List[float]]): Thresholds list of maximum x-axis position from ego vehicle.
            Keep all `object_results` that their each x position are in [`-max_x_position`, `max_x_position`]
            for both of their `estimated_object` and `ground_truth_object`. Defaults to None.
        max_y_position_list (Optional[List[float]]): Thresholds list of maximum y-axis position from ego vehicle.
            Keep all `object_results` that their each y position are in [`-max_y_position`, `max_y_position`]
            for both of their `estimated_object` and `ground_truth_object`. Defaults to None.
        max_distance_list (Optional[List[float]]): Thresholds list of maximum distance range from ego vehicle.
            Keep all `object_results` that their each distance is smaller than `max_distance`
            for both of their `estimated_object` and `ground_truth_object`. Defaults to None.
        min_distance_list (Optional[List[float]]): Thresholds list of minimum distance range from ego vehicle.
            Keep all `object_results` that their each distance is bigger than `min_distance`
            for both of their `estimated_object` and `ground_truth_object`. Defaults to None.
        min_point_numbers (Optional[List[int]]): Thresholds list of minimum number of points
            must be contained in object's box. Keep all `object_results` that their boxes contain more points than
            `min_point_number` only considering their `ground_truth_object`. Defaults to None.
            For example, `target_labels=["car", "bike", "pedestrian"]` and `min_point_numbers=[5, 0, 0]`,
            Then objects that has car label and their boxes contain 4 or less points are filtered.
            Otherwise, all objects that has bike or pedestrian label are not filtered.
        confidence_threshold_list (Optional[List[float]]): Thresholds list of minimum confidence score.
            Keep all `object_results` that their confidence is bigger than `confidence_threshold`
            only considering their `estimated_object`. Defaults to None.
        target_uuids (Optional[List[str]]): Filter target list of ground truths' uuids.
            Keep all `object_results` that their each uuid is in `target_uuids`
            only considering their `ground_truth_object`.
            Defaults to None.
        ego2map (Optional[numpy.ndarray]): Array of 4x4 matrix to transform objects' coordinates from ego to map.
            This is only needed when `frame_id=map`. Defaults to None.

    Returns:
        filtered_object_results (List[DynamicObjectWithPerceptionResult]): Filtered object results list.
    """
    filtered_object_results: List[DynamicObjectWithPerceptionResult] = []
    for object_result in object_results:
        is_target: bool = _is_target_object(
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
    objects: List[ObjectType],
    is_gt: bool,
    target_labels: Optional[List[LabelType]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_distance_list: Optional[List[float]] = None,
    min_distance_list: Optional[List[float]] = None,
    min_point_numbers: Optional[List[int]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
    target_uuids: Optional[List[str]] = None,
    ego2map: Optional[np.ndarray] = None,
) -> List[ObjectType]:
    """Filter DynamicObject considering ground truth objects.

    If any of `target_labels`, `max_x_position_list`, `max_y_position_list`, `max_distance_list`, `min_distance_list`,
    `min_point_numbers` or `confidence_threshold_list` are specified, each of them must be same length list.

    Args:
        objects (List[ObjectType]: The objects you want to filter.
        is_gt (bool): Flag if input object is ground truth.
        target_labels Optional[List[LabelType]]): Filter target list of labels.
            Keep all `objects` that have same label in this list. Defaults to None.
        max_x_position_list (Optional[List[float]]): Thresholds list of maximum x-axis position from ego vehicle.
            Keep all `objects` that their each x position are in [`-max_x_position`, `max_x_position`].
            Defaults to None.
        max_y_position_list (Optional[List[float]]): Thresholds list of maximum y-axis position from ego vehicle.
            Keep all `objects` that their each y position are in [`-max_y_position`, `max_y_position`].
            Defaults to None.
        max_distance_list (Optional[List[float]]): Thresholds list of maximum distance range from ego vehicle.
            Keep all `objects` that their each distance is smaller than `max_distance`. Defaults to None.
        min_distance_list (Optional[List[float]]): Thresholds list of minimum distance range from ego vehicle.
            Keep all `objects` that their each distance is bigger than `min_distance`. Defaults to None.
        min_point_numbers (Optional[List[int]]): Thresholds list of minimum number of points
            must be contained in object's box. Keep all `objects` that their boxes contain more points than
            `min_point_number`. This is only used when `is_gr=True`. Defaults to None.
            For example, `target_labels=["car", "bike", "pedestrian"]` and `min_point_numbers=[5, 0, 0]`,
            Then objects that has car label and their boxes contain 4 or less points are filtered.
            Otherwise, all objects that has bike or pedestrian label are not filtered.
        confidence_threshold_list (Optional[List[float]]): Thresholds list of minimum confidence score.
            Keep all `objects` that their confidence is bigger than `confidence_threshold`.
            This is only used when `is_gt=False`. Defaults to None.
        target_uuids (Optional[List[str]]): Filter target list of ground truths' uuids.
            Keep all `objects` that their each uuid is in `target_uuids`. This is only used when `is_gt=True`.
            Defaults to None.
        ego2map (Optional[numpy.ndarray]): Array of 4x4 matrix to transform objects' coordinates from ego to map.
            This is only needed when `frame_id=map`. Defaults to None.

    Returns:
        List[Union[DynamicObject, Object2DBase]]: Filtered objects.
    """

    filtered_objects: List[DynamicObject] = []
    for object_ in objects:
        if is_gt:
            is_target: bool = _is_target_object(
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
    target_labels: Optional[List[LabelType]],
    matching_mode: Optional[MatchingMode] = None,
    matching_threshold_list: Optional[List[float]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
) -> Tuple[List[DynamicObjectWithPerceptionResult], List[DynamicObjectWithPerceptionResult]]:
    """Divide input `object_results` into TP (True Positive) and FP (False Positive) object results.

    This function judge whether input `object_results` is TP or FP with `matching_threshold` when
    `matching_threshold_list` is specified.

    Otherwise, determine it considering whether labels between `estimated_object` and `ground_truth_object`
    that are member variables of `object_results` are same.

    And also, judge with `confidence_threshold` when `confidence_threshold_list` is specified.

    Args:
        object_results (List[DynamicObjectWithPerceptionResult]): The object results you want to filter
        target_labels Optional[List[AutowareLabel]]): Target labels list.
            Get threshold value from `matching_threshold_list` at corresponding label index.
        matching_mode (Optional[MatchingMode]): MatchingMode instance.
            When `matching_threshold_list=None`, this is not have to be specified. Defaults to None.
        matching_threshold_list (Optional[List[float]]): Matching thresholds list. Defaults to None.
            For example, if `matching_mode=MatchingMode.IOU3D` and `matching_threshold=0.5`,
            and `object_result.is_result_correct(matching_mode, matching_threshold)=True`,
            then those `object_results` are regarded as TP object results. Defaults to None.
        confidence_threshold_list (Optional[List[float]]): Confidence thresholds list.
            All `object_results` are regarded as TP when their `estimated_object` has higher confidence
            than `confidence_threshold`. Defaults to None.

    Returns:
        List[DynamicObjectWithPerceptionResult]]: TP object results.
        List[DynamicObjectWithPerceptionResult]]: FP object results.
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
    ground_truth_objects: List[ObjectType],
    object_results: Optional[List[DynamicObjectWithPerceptionResult]],
    tp_objects: Optional[List[DynamicObjectWithPerceptionResult]],
) -> List[ObjectType]:
    """Get FN (False Negative) objects from ground truth objects by using object result.

    This function returns a set of `ground_truth_objects` that are not contained in TP object results.

    Args:
        ground_truth_objects (List[ObjectType]): Ground truth objects list.
        object_results (Optional[List[DynamicObjectWithPerceptionResult]]): Object results list.
        tp_objects (Optional[List[DynamicObjectWithPerceptionResult]]): TP results list in object results.

    Returns:
        List[ObjectType]: FN (False Negative) objects list.
    """

    if object_results is None:
        return ground_truth_objects

    fn_objects: List[ObjectType] = []
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
    ground_truth_object: ObjectType,
    object_results: List[DynamicObjectWithPerceptionResult],
    tp_objects: List[DynamicObjectWithPerceptionResult],
) -> bool:
    """Judge whether ground truth object is FN (False Negative) object.

    Args:
        ground_truth_object (ObjectType): Ground truth object.
        object_results (List[DynamicObjectWithPerceptionResult]): object results list.
        tp_objects (Optional[List[DynamicObjectWithPerceptionResult]]): TP results list in object results.

    Returns:
        bool: Whether ground truth object is FN (False Negative) object.
    """

    for object_result in object_results:
        if ground_truth_object == object_result.ground_truth_object and object_result in tp_objects:
            return False
    return True


def _is_target_object(
    dynamic_object: ObjectType,
    target_labels: Optional[List[LabelType]] = None,
    max_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    max_distance_list: Optional[List[float]] = None,
    min_distance_list: Optional[List[float]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
    min_point_numbers: Optional[List[int]] = None,
    target_uuids: Optional[List[str]] = None,
    ego2map: Optional[np.ndarray] = None,
) -> bool:
    """Judge whether the input `dynamic_object` is target or not.

    This function used to filtering for both of ground truths and object results.

    Args:
        dynamic_object (ObjectType): The dynamic object
        target_labels Optional[List[LabelType]]): Target labels list.
            Keep all `dynamic_object` that have same labels in this list. Defaults to None.
        max_x_position_list (Optional[List[float]]): Thresholds list of maximum x-axis position from ego vehicle.
            Keep all `dynamic_object` that their each x position are in [`-max_x_position`, `max_x_position`].
            Defaults to None.
        max_y_position_list (Optional[List[float]]): Thresholds list of maximum y-axis position from ego vehicle.
            Keep all `dynamic_object` that their each y position are in [`-max_y_position`, `max_y_position`].
            Defaults to None.
        max_distance_list (Optional[List[float]]): Thresholds list of maximum distance range from ego vehicle.
            Keep all `dynamic_object` that their each distance is smaller than `max_distance`. Defaults to None.
        min_distance_list (Optional[List[float]]): Thresholds list of minimum distance range from ego vehicle.
            Keep all `dynamic_object` that their each distance is bigger than `min_distance`. Defaults to None.
        min_point_numbers (Optional[List[int]]): Thresholds list of minimum number of points
            must be contained in object's box. Keep all `dynamic_objects` that their boxes contain more points than
            `min_point_number`. Defaults to None.
            For example, `target_labels=["car", "bike", "pedestrian"]` and `min_point_numbers=[5, 0, 0]`,
            Then objects that has car label and their boxes contain 4 or less points are filtered.
            Otherwise, all objects that has bike or pedestrian label are not filtered.
        confidence_threshold_list (Optional[List[float]]): Thresholds list of minimum confidence score.
            Keep all `dynamic_objects` that their confidence is bigger than `confidence_threshold`. Defaults to None.
        target_uuids (Optional[List[str]]): Filter target list of ground truths' uuids.
            Keep all `dynamic_objects` that their each uuid is in `target_uuids`. Defaults to None.
        ego2map (Optional[numpy.ndarray]): Array of 4x4 matrix to transform objects' coordinates from ego to map.
            This is only needed when `frame_id=map`. Defaults to None.

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
        position_: Tuple[float, float, float] = dynamic_object.state.position
        if dynamic_object.frame_id == FrameID.MAP:
            assert ego2map is not None, "When frame_id is map, ego2map must be specified"
            pos_arr: np.ndarray = np.append(position_, 1.0)
            position_ = tuple(np.linalg.inv(ego2map).dot(pos_arr)[:3].tolist())
        bev_distance_: float = dynamic_object.get_distance_bev(ego2map)

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
    objects: List[Union[ObjectType, DynamicObjectWithPerceptionResult]],
    target_labels: Optional[List[LabelType]] = None,
) -> Dict[LabelType, List[Union[ObjectType, DynamicObjectWithPerceptionResult]]]:
    """Divide DynamicObject or DynamicObjectWithPerceptionResult into dict mapped by their labels.

    Args:
        objects (List[Union[ObjectType, DynamicObjectWithPerceptionResult]]):
            List of ObjectType or DynamicObjectWithPerceptionResult.
        target_labels (Optional[List[LabelType]]): If this is specified, create empty list even
            if there is no object having specified label. Defaults to None.

    Returns:
        ret (Dict[LabelType, List[Union[ObjectType, DynamicObjectWithPerceptionResult]]]):
            Dict that are list of ObjectType or DynamicObjectWithPerceptionResult mapped by their labels.
            It depends on the type of input object.
    """
    if target_labels is not None:
        ret = {label: [] for label in target_labels}
    else:
        ret: Dict[LabelType, List[ObjectType]] = {}

    for obj in objects:
        if isinstance(obj, DynamicObjectWithPerceptionResult):
            label: LabelType = obj.estimated_object.semantic_label
        else:
            label: LabelType = obj.semantic_label

        if label not in ret.keys():
            ret[label] = [obj]
        else:
            ret[label].append(obj)
    return ret


def divide_objects_to_num(
    objects: List[Union[ObjectType, DynamicObjectWithPerceptionResult]],
    target_labels: Optional[List[LabelType]] = None,
) -> Dict[LabelType, int]:
    """Divide the number of input `objects` mapped by their labels.

    Args:
        objects (List[Union[ObjectType, DynamicObjectWithPerceptionResult]]):
            List of ObjectType or DynamicObjectWithPerceptionResult.
        target_labels (Optional[List[LabelType]]): If this is specified, create empty list even
            if there is no object having specified label. Defaults to None.

    Returns:
        ret (Dict[LabelType, int]): Dict that are number of ObjectType or DynamicObjectWithPerceptionResult
            mapped by their labels.
    """
    if target_labels is not None:
        ret = {label: 0 for label in target_labels}
    else:
        ret: Dict[LabelType, int] = {}

    for obj in objects:
        if isinstance(obj, DynamicObjectWithPerceptionResult):
            label: LabelType = obj.estimated_object.semantic_label
        else:
            label: LabelType = obj.semantic_label

        if label not in ret.keys():
            ret[label] = 1
        else:
            ret[label] += 1
    return ret
