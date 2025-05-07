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

from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import warnings

import numpy as np
from perception_eval.common import ObjectType
from perception_eval.common.label import CommonLabel
from perception_eval.common.label import Label
from perception_eval.common.label import LabelType
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.status import MatchingStatus
from perception_eval.common.threshold import get_label_threshold
from perception_eval.common.threshold import LabelThreshold
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching.matching_config import MatchingConfig
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


def filter_object_results(
    object_results: List[DynamicObjectWithPerceptionResult],
    target_labels: Optional[List[LabelType]] = None,
    ignore_attributes: Optional[List[str]] = None,
    max_x_position_list: Optional[List[float]] = None,
    min_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    min_y_position_list: Optional[List[float]] = None,
    max_distance_list: Optional[List[float]] = None,
    min_distance_list: Optional[List[float]] = None,
    min_point_numbers: Optional[List[int]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
    target_uuids: Optional[str] = None,
    transforms: Optional[TransformDict] = None,
    *args,
    **kwargs,
) -> List[DynamicObjectWithPerceptionResult]:
    """Filter DynamicObjectWithPerceptionResult considering both estimated and ground truth objects.

    If any of `target_labels`, `max_x_position_list`, `min_x_position_list`, `max_y_position_list`, `min_y_position_list`,
    `max_distance_list`, `min_distance_list`, `min_point_numbers` or `confidence_threshold_list`
    are specified, each of them must be same length list.

    It first filters `object_results` with input parameters considering estimated objects.
    After that, remained `object_results` are filtered with input parameters considering ground truth objects.

    Args:
        object_results (List[DynamicObjectWithPerceptionResult]): Object results list.
        target_labels (Optional[List[LabelType]]): Filter target list of labels.
            Keep all `object_results` that both of their `estimated_object` and `ground_truth_object`
            have same label in this list. Defaults to None.
        ignore_attributes (Optional[List[str]]): List of attributes to be ignored. Defaults to None.
        max_x_position_list (Optional[List[float]]): Thresholds list of maximum x-axis position from ego vehicle.
            Keep all `dynamic_object` that their each x position is smaller than `max_x_position`
            for both of their `estimated_object` and `ground_truth_object`. Defaults to None.
            If `min_x_position_list` is not specified, keep them that each x position are in [`-max_x_position`, `max_x_position`].
        min_x_position_list (Optional[List[float]]): Thresholds list of minimum x-axis position from ego vehicle.
            Keep all `dynamic_object` that their each x position is bigger than `min_x_position`
            for both of their `estimated_object` and `ground_truth_object`. Defaults to None.
        max_y_position_list (Optional[List[float]]): Thresholds list of maximum y-axis position from ego vehicle.
            Keep all `dynamic_object` that their each y position is smaller than `max_y_position`
            for both of their `estimated_object` and `ground_truth_object`. Defaults to None.
            If `min_y_position_list` is not specified, keep them that each y position are in [`-max_y_position`, `max_y_position`].
        min_y_position_list (Optional[List[float]]): Thresholds list of minimum y-axis position from ego vehicle.
            Keep all `dynamic_object` that their each y position is bigger than `min_y_position`
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

    Returns:
        filtered_object_results (List[DynamicObjectWithPerceptionResult]): Filtered object results list.
    """
    filtered_object_results: List[DynamicObjectWithPerceptionResult] = []
    for object_result in object_results:
        is_target: bool = _is_target_object(
            dynamic_object=object_result.estimated_object,
            is_gt=False,
            target_labels=target_labels,
            max_x_position_list=max_x_position_list,
            min_x_position_list=min_x_position_list,
            max_y_position_list=max_y_position_list,
            min_y_position_list=min_y_position_list,
            max_distance_list=max_distance_list,
            min_distance_list=min_distance_list,
            confidence_threshold_list=confidence_threshold_list,
            transforms=transforms,
        )
        if is_target and object_result.ground_truth_object:
            is_target = is_target and _is_target_object(
                dynamic_object=object_result.ground_truth_object,
                is_gt=True,
                target_labels=target_labels,
                ignore_attributes=ignore_attributes,
                max_x_position_list=max_x_position_list,
                min_x_position_list=min_x_position_list,
                max_y_position_list=max_y_position_list,
                min_y_position_list=min_y_position_list,
                max_distance_list=max_distance_list,
                min_distance_list=min_distance_list,
                min_point_numbers=min_point_numbers,
                target_uuids=target_uuids,
                transforms=transforms,
            )
        elif target_uuids and object_result.ground_truth_object is None:
            is_target = False

        if is_target:
            filtered_object_results.append(object_result)

    return filtered_object_results


def filter_objects(
    objects: List[ObjectType],
    is_gt: bool,
    target_labels: Optional[List[Label]] = None,
    ignore_attributes: Optional[List[str]] = None,
    max_x_position_list: Optional[List[float]] = None,
    min_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    min_y_position_list: Optional[List[float]] = None,
    max_distance_list: Optional[List[float]] = None,
    min_distance_list: Optional[List[float]] = None,
    min_point_numbers: Optional[List[int]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
    target_uuids: Optional[List[str]] = None,
    transforms: Optional[TransformDict] = None,
    *args,
    **kwargs,
) -> List[ObjectType]:
    """Filter DynamicObject considering ground truth objects.

    If any of `target_labels`, `max_x_position_list`, `min_x_position_list`, `max_y_position_list`, `min_y_position_list`,
    `max_distance_list`, `min_distance_list`, `min_point_numbers` or `confidence_threshold_list`
    are specified, each of them must be same length list.

    Args:
        objects (List[ObjectType]: The objects you want to filter.
        is_gt (bool): Flag if input object is ground truth.
        target_labels Optional[List[Label]]): Filter target list of labels.
            Keep all `objects` that have same label in this list. Defaults to None.
        attributes_ignore (Optional[List[str]]): List of attributes to be ignored. Defaults to None.
        max_x_position_list (Optional[List[float]]): Thresholds list of maximum x-axis position from ego vehicle.
            Keep all `objects` that their each x position are in [`-max_x_position`, `max_x_position`].
            If `min_x_position_list` is not specified, keep them that each x position are smaller than `max_x_position`.
            Defaults to None.
        min_x_position_list (Optional[List[float]]): Thresholds list of minimum x-axis position from ego vehicle.
            Keep all `objects` that their each x position are bigger than `min_x_position`.
            Defaults to None.
        max_y_position_list (Optional[List[float]]): Thresholds list of maximum y-axis position from ego vehicle.
            Keep all `objects` that their each y posiition are in [`-max_y_position`, `max_y_position`].
            If `min_y_position_list` is not specified, keep them that each y position are smaller than `max_y_position`.
            Defaults to None.
        min_y_position_list (Optional[List[float]]): Thresholds list of minimum y-axis position from ego vehicle.
            Keep all `objects` that their each y position are bigger than `min_y_position`.
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

    Returns:
        List[ObjectType]: Filtered objects.
    """
    filtered_objects: List[ObjectType] = []
    for object_ in objects:
        is_target: bool = _is_target_object(
            dynamic_object=object_,
            is_gt=is_gt,
            target_labels=target_labels,
            ignore_attributes=ignore_attributes,
            max_x_position_list=max_x_position_list,
            min_x_position_list=min_x_position_list,
            max_y_position_list=max_y_position_list,
            min_y_position_list=min_y_position_list,
            max_distance_list=max_distance_list,
            min_distance_list=min_distance_list,
            min_point_numbers=min_point_numbers,
            target_uuids=target_uuids,
            confidence_threshold_list=confidence_threshold_list,
            transforms=transforms,
        )
        if is_target:
            filtered_objects.append(object_)
    return filtered_objects


def get_positive_objects(
    object_results: List[DynamicObjectWithPerceptionResult],
    target_labels: List[Label],
    matching_mode: Optional[MatchingMode] = None,
    matching_threshold_list: Optional[List[float]] = None,
) -> Tuple[List[DynamicObjectWithPerceptionResult], List[DynamicObjectWithPerceptionResult]]:
    """Returns TP (True Positive) and FP (False Positive) object results as `tuple`.

    If an object result has better matching score than the matching threshold, it is TP, otherwise FP.

    Args:
        object_results (List[DynamicObjectWithPerceptionResult]): List of matched estimation and GT objects.
        target_labels (Optional[List[Label]]): List of labels should be evaluated.
        matching_mode (Optional[MatchingMode]): Matching policy, i.e. center or plane distance, iou.
        matching_threshold_list (Optional[List[float]]): List of matching thresholds,
            each element corresponds to target label.

    Returns:
        tp_object_results (List[DynamicObjectWithPerceptionResult]): List of TP.
        fp_object_results (List[DynamicObjectWithPerceptionResult]): List of FP.
    """
    tp_object_results: List[DynamicObjectWithPerceptionResult] = []
    fp_object_results: List[DynamicObjectWithPerceptionResult] = []
    for object_result in object_results:
        if object_result.ground_truth_object is None:
            fp_object_results.append(object_result)
            continue

        # in case of matching (Est, GT) = (unknown, except of unknown)
        # use GT label
        matching_threshold = get_label_threshold(
            semantic_label=object_result.ground_truth_object.semantic_label,
            target_labels=target_labels,
            threshold_list=matching_threshold_list,
        )

        est_status, gt_status = object_result.get_status(matching_mode, matching_threshold)

        if est_status == MatchingStatus.FP:
            if gt_status == MatchingStatus.TN:
                fp_object_results.append(
                    DynamicObjectWithPerceptionResult(
                        object_result.estimated_object,
                        None,
                        object_result.matching_label_policy,
                    )
                )
            else:
                fp_object_results.append(object_result)
        elif est_status == MatchingStatus.TP and gt_status == MatchingStatus.TP:
            tp_object_results.append(object_result)

    return tp_object_results, fp_object_results


def get_negative_objects(
    ground_truth_objects: List[DynamicObject],
    object_results: List[DynamicObjectWithPerceptionResult],
    target_labels: List[Label],
    matching_mode: Optional[MatchingMode] = None,
    matching_threshold_list: Optional[List[float]] = None,
) -> Tuple[List[DynamicObject], List[DynamicObject]]:
    """Returns TN (True Negative) and FN (False Negative) objects as `tuple`.

    If a ground truth object is contained in object results, it is TP or FP.
    Otherwise, the label of ground truth is 'FP', which means this object should not estimated, it is TN.

    Args:
        ground_truth_objects (List[DynamicObject]): List of ground truth objects.
        object_results (List[DynamicObjectWithPerceptionResult]): List of object results.
        target_labels (Optional[List[Label]]): List of labels should be evaluated.
        matching_mode (Optional[MatchingMode]): Matching policy, i.e. center or plane distance, iou.
        matching_threshold_list (Optional[List[float]]): List of matching thresholds,
            each element corresponds to target label.

    Returns:
        tn_objects (List[DynamicObject]): List of TN.
        fn_objects (List[DynamicObject]): List of FN.
    """
    tn_objects: List[DynamicObject] = []
    fn_objects: List[DynamicObject] = []

    non_candidates: List[ObjectType] = []
    for object_result in object_results:
        matching_threshold: Optional[float] = get_label_threshold(
            (
                object_result.ground_truth_object.semantic_label
                if object_result.ground_truth_object is not None
                else object_result.estimated_object.semantic_label
            ),
            target_labels,
            matching_threshold_list,
        )
        _, gt_status = object_result.get_status(matching_mode, matching_threshold)

        if gt_status == MatchingStatus.TN:
            tn_objects.append(object_result.ground_truth_object)
        elif gt_status == MatchingStatus.FN:
            fn_objects.append(object_result.ground_truth_object)

        if gt_status is not None:
            non_candidates.append(object_result.ground_truth_object)

    for ground_truth_object in ground_truth_objects:
        if ground_truth_object in non_candidates:
            continue

        if ground_truth_object.semantic_label.is_fp():
            tn_objects.append(ground_truth_object)
        else:
            fn_objects.append(ground_truth_object)

    return tn_objects, fn_objects


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
        target_labels Optional[List[Label]]): Target labels list.
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
    warnings.warn(
        "`divide_tp_fp_objects()` is removed in next minor update, please use `get_positive_objects()`",
        DeprecationWarning,
    )

    tp_object_results: List[DynamicObjectWithPerceptionResult] = []
    fp_object_results: List[DynamicObjectWithPerceptionResult] = []
    for object_result in object_results:
        if object_result.ground_truth_object is None:
            fp_object_results.append(object_result)
            continue

        # in case of matching (Est, GT) = (unknown, except of unknown)
        # use GT label
        matching_threshold_ = get_label_threshold(
            semantic_label=object_result.ground_truth_object.semantic_label,
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
            is_confidence: bool = object_result.estimated_object.semantic_score > confidence_threshold_
            is_correct = is_correct and is_confidence

        if is_correct:
            tp_object_results.append(object_result)
        else:
            fp_object_results.append(object_result)
    return tp_object_results, fp_object_results


def get_fn_objects(
    ground_truth_objects: List[ObjectType],
    object_results: List[DynamicObjectWithPerceptionResult],
    tp_object_results: List[DynamicObjectWithPerceptionResult],
) -> List[ObjectType]:
    """Get FN (False Negative) objects from ground truth objects by using object result.

    This function returns a set of `ground_truth_objects` that are not contained in TP object results.

    Args:
        ground_truth_objects (List[ObjectType]): Ground truth objects list.
        object_results (Optional[List[DynamicObjectWithPerceptionResult]]): Object results list.
        tp_object_results (Optional[List[DynamicObjectWithPerceptionResult]]): TP results list in object results.

    Returns:
        List[ObjectType]: FN (False Negative) objects list.
    """
    warnings.warn(
        "`get_fn_objects()` is removed in next minor update, please use `get_negative_objects()`",
        DeprecationWarning,
    )

    fn_objects: List[ObjectType] = []
    for ground_truth_object in ground_truth_objects:
        is_fn_object: bool = _is_fn_object(
            ground_truth_object=ground_truth_object,
            object_results=object_results,
            tp_object_results=tp_object_results,
        )
        if is_fn_object:
            fn_objects.append(ground_truth_object)
    return fn_objects


def _is_fn_object(
    ground_truth_object: ObjectType,
    object_results: List[DynamicObjectWithPerceptionResult],
    tp_object_results: List[DynamicObjectWithPerceptionResult],
) -> bool:
    """Judge whether ground truth object is FN (False Negative) object.

    Args:
        ground_truth_object (ObjectType): Ground truth object.
        object_results (List[DynamicObjectWithPerceptionResult]): object results list.
        tp_object_results (List[DynamicObjectWithPerceptionResult]): TP results list in object results.

    Returns:
        bool: Whether ground truth object is FN (False Negative) object.
    """
    warnings.warn(
        "`_is_fn_object()` is removed in next minor update, please use `get_negative_objects()`",
        DeprecationWarning,
    )
    for object_result in object_results:
        if ground_truth_object == object_result.ground_truth_object and object_result in tp_object_results:
            return False
    return True


def _is_target_object(
    dynamic_object: ObjectType,
    is_gt: bool,
    target_labels: Optional[List[LabelType]] = None,
    ignore_attributes: Optional[List[str]] = None,
    max_x_position_list: Optional[List[float]] = None,
    min_x_position_list: Optional[List[float]] = None,
    max_y_position_list: Optional[List[float]] = None,
    min_y_position_list: Optional[List[float]] = None,
    max_distance_list: Optional[List[float]] = None,
    min_distance_list: Optional[List[float]] = None,
    confidence_threshold_list: Optional[List[float]] = None,
    min_point_numbers: Optional[List[int]] = None,
    target_uuids: Optional[List[str]] = None,
    transforms: Optional[TransformDict] = None,
) -> bool:
    """Judge whether the input `dynamic_object` is target or not.

    This function used to filtering for both of ground truths and object results.

    Args:
        dynamic_object (ObjectType): The dynamic object
        is_gt (bool): Whether input object is GT or not.
        target_labels Optional[List[LabelType]]): Target labels list.
            Keep all `dynamic_object` that have same labels in this list. Defaults to None.
        ignore_attributes (Optional[List[str]]): List of attributes to be ignored. Defaults to None.
        max_x_position_list (Optional[List[float]]): Thresholds list of maximum x-axis position from ego vehicle.
            Keep all `dynamic_object` that their each x position are in [`-max_x_position`, `max_x_position`].
            If `min_x_position_list` is not specified, keep them that each x position are smaller than `max_x_position`.
            Defaults to None.
        min_x_position_list (Optional[List[float]]): Thresholds list of minimum x-axis position from ego vehicle.
            Keep all `dynamic_object` that their each x position are bigger than `min_x_position`.
        max_y_position_list (Optional[List[float]]): Thresholds list of maximum y-axis position from ego vehicle.
            Keep all `dynamic_object` that their each y position are in [`-max_y_position`, `max_y_position`].
            If `min_y_position_list` is not specified, keep them that each y position are smaller than `max_y_position`.
            Defaults to None.
        min_y_position_list (Optional[List[float]]): Thresholds list of minimum y-axis position from ego vehicle.
            Keep all `dynamic_object` that their each y position are bigger than `min_y_position`.
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

    Returns:
        bool: If the object is filter target, return True
    """
    if dynamic_object.semantic_label.is_fp():
        return True

    # For estimated objected, skip filtering out if it has unknown label
    is_unknown_estimation: bool = dynamic_object.semantic_label.is_unknown() and is_gt is False

    # Whether unknown is contained in target labels
    is_contained_unknown: bool = (
        any([label == CommonLabel.UNKNOWN for label in target_labels]) if target_labels is not None else False
    )

    # use special threshold for unknown labeled estimations
    use_unknown_threshold: bool = is_unknown_estimation and not is_contained_unknown

    label_threshold = LabelThreshold(
        semantic_label=dynamic_object.semantic_label,
        target_labels=target_labels,
    )
    is_target: bool = True

    if target_labels:
        is_target = (
            is_target and True
            if use_unknown_threshold
            else is_target and dynamic_object.semantic_label.label in target_labels
        )

    if ignore_attributes is not None:
        is_target = (
            is_target and True
            if use_unknown_threshold
            else is_target and not dynamic_object.semantic_label.contains_any(ignore_attributes)
        )

    if is_target and confidence_threshold_list is not None:
        confidence_threshold = (
            0.0 if use_unknown_threshold else label_threshold.get_label_threshold(confidence_threshold_list)
        )
        is_target = is_target and dynamic_object.semantic_score > confidence_threshold

    if transforms is None and dynamic_object.frame_id == FrameID.BASE_LINK:
        position_ = dynamic_object.state.position
        bev_distance_ = dynamic_object.get_distance_bev()
    elif dynamic_object.state.position is not None and transforms is not None:
        position_ = transforms.transform((dynamic_object.frame_id, FrameID.BASE_LINK), dynamic_object.state.position)
        bev_distance_ = dynamic_object.get_distance_bev(transforms)
    else:
        position_ = bev_distance_ = None

    if position_ is not None:
        if is_target and max_x_position_list is not None:
            max_x_position = (
                np.mean(max_x_position_list)
                if use_unknown_threshold
                else label_threshold.get_label_threshold(max_x_position_list)
            )
            if min_x_position_list is not None:
                is_target = is_target and position_[0] < max_x_position
                min_x_position = (
                    np.mean(min_x_position_list)
                    if use_unknown_threshold
                    else label_threshold.get_label_threshold(min_x_position_list)
                )
                is_target = is_target and position_[0] > min_x_position
            else:
                is_target = is_target and abs(position_[0]) < max_x_position

        if is_target and max_y_position_list is not None:
            max_y_position = (
                np.mean(max_y_position_list)
                if use_unknown_threshold
                else label_threshold.get_label_threshold(max_y_position_list)
            )
            if min_y_position_list is not None:
                is_target = is_target and position_[1] < max_y_position
                min_y_position = (
                    np.mean(min_y_position_list)
                    if use_unknown_threshold
                    else label_threshold.get_label_threshold(min_y_position_list)
                )
                is_target = is_target and position_[1] > min_y_position
            else:
                is_target = is_target and abs(position_[1]) < max_y_position

        if is_target and min_x_position_list is not None:
            min_x_position = (
                np.mean(min_x_position_list)
                if use_unknown_threshold
                else label_threshold.get_label_threshold(min_x_position_list)
            )
            is_target = is_target and position_[0] > min_x_position

        if is_target and min_y_position_list is not None:
            min_y_position = (
                np.mean(min_y_position_list)
                if use_unknown_threshold
                else label_threshold.get_label_threshold(min_y_position_list)
            )
            is_target = is_target and position_[1] > min_y_position

    if bev_distance_ is not None:
        if is_target and max_distance_list is not None:
            max_distance = (
                np.mean(max_distance_list)
                if use_unknown_threshold
                else label_threshold.get_label_threshold(max_distance_list)
            )
            is_target = is_target and bev_distance_ < max_distance

        if is_target and min_distance_list is not None:
            min_distance = (
                np.mean(min_distance_list)
                if use_unknown_threshold
                else label_threshold.get_label_threshold(min_distance_list)
            )
            is_target = is_target and bev_distance_ > min_distance

        if is_target and min_point_numbers is not None and is_gt:
            min_point_number = 0 if use_unknown_threshold else label_threshold.get_label_threshold(min_point_numbers)
            is_target = is_target and dynamic_object.pointcloud_num >= min_point_number

    if is_target and target_uuids is not None and is_gt:
        assert isinstance(target_uuids, list)
        assert all([isinstance(uuid, str) for uuid in target_uuids])
        is_target = is_target and dynamic_object.uuid in target_uuids

    return is_target


def resolve_label(
    obj: Union[ObjectType, DynamicObjectWithPerceptionResult],
    target_labels: Optional[List[LabelType]],
) -> Optional[LabelType]:
    """Resolve the label from an object, trying fallback to GT label if needed."""
    estimated_label = (
        obj.estimated_object.semantic_label.label
        if isinstance(obj, DynamicObjectWithPerceptionResult)
        else obj.semantic_label.label
    )

    if target_labels is None or estimated_label in target_labels:
        return estimated_label

    if isinstance(obj, DynamicObjectWithPerceptionResult) and obj.ground_truth_object is not None:
        gt_label = obj.ground_truth_object.semantic_label.label
        if gt_label in target_labels:
            return gt_label

    return None


# TODO(vividf): change the naming for better understanding
def divide_objects(
    objects: List[Union[ObjectType, DynamicObjectWithPerceptionResult]],
    target_labels: Optional[List[LabelType]] = None,
) -> Dict[LabelType, List[Union[ObjectType, DynamicObjectWithPerceptionResult]]]:
    """
    Divide a list of objects into a dictionary grouped by their semantic labels.

    This function is used for classification, tracking, and similar tasks where
    a simple one-to-one matching between estimated and ground truth objects is performed.

    If `target_labels` is specified, only objects whose labels match the target list
    will be included in the output. If an estimated label does not match but a ground
    truth label is available (for DynamicObjectWithPerceptionResult), the ground truth
    label is used as a fallback.

    Args:
        objects (List[Union[ObjectType, DynamicObjectWithPerceptionResult]]):
            List of estimated or matched objects.
        target_labels (Optional[List[LabelType]]):
            List of labels to filter. If None, all labels are included.

    Returns:
        Dict[LabelType, List[Union[ObjectType, DynamicObjectWithPerceptionResult]]]:
            A dictionary mapping each label to the list of corresponding objects.
            If `target_labels` is provided, all specified labels will exist in the dictionary,
            even if the list is empty.
    """
    result: Dict[LabelType, List[Union[ObjectType, DynamicObjectWithPerceptionResult]]] = (
        {label: [] for label in target_labels} if target_labels else {}
    )

    for obj in objects:
        label = resolve_label(obj, target_labels)
        if label is None:
            continue
        result.setdefault(label, []).append(obj)

    return result


def divide_nuscene_objects_by_label(
    objects: Dict[MatchingConfig, List[DynamicObjectWithPerceptionResult]],
    target_labels: Optional[List[LabelType]] = None,
) -> Dict[LabelType, Dict[MatchingConfig, List[DynamicObjectWithPerceptionResult]]]:
    """
    Divide a dictionary of detection matching results into a dictionary grouped by label and matching config.

    This function is used for detection evaluation, where estimated objects are matched
    against ground truth objects across multiple matching modes (e.g., center distance, IoU)
    and thresholds.

    The input is a dictionary where each key is a MatchingConfig (mode, threshold) and
    the value is a list of matched DynamicObjectWithPerceptionResult.

    If `target_labels` is specified, only objects whose labels match the target list
    will be included in the output. If an estimated label does not match but a ground
    truth label is available, the ground truth label is used as a fallback.

    Args:
        objects (Dict[MatchingConfig, List[DynamicObjectWithPerceptionResult]]):
            Dictionary of detection matching results grouped by matching config.
        target_labels (Optional[List[LabelType]]):
            List of labels to filter. If None, all labels are included.

    Returns:
        Dict[LabelType, Dict[MatchingConfig, List[DynamicObjectWithPerceptionResult]]]:
            A dictionary mapping each label to another dictionary that maps matching config
            to the list of corresponding matched results. If `target_labels` is provided,
            all specified labels will exist in the dictionary, even if empty.
    """
    result: Dict[LabelType, Dict[MatchingConfig, List[DynamicObjectWithPerceptionResult]]] = (
        {label: defaultdict(list) for label in target_labels} if target_labels else {}
    )

    for matching_config, object_list in objects.items():
        for obj in object_list:
            label = resolve_label(obj, target_labels)
            if label is None:
                continue
            result[label][matching_config].append(obj)

    return result


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
            label: LabelType = obj.estimated_object.semantic_label.label
        else:
            label: LabelType = obj.semantic_label.label

        if target_labels is not None and label not in target_labels:
            if isinstance(obj, DynamicObjectWithPerceptionResult) and obj.ground_truth_object is not None:
                label = obj.ground_truth_object.semantic_label.label
            else:
                continue

        if label not in ret.keys():
            ret[label] = 1
        else:
            ret[label] += 1
    return ret
