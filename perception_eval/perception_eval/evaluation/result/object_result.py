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

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from perception_eval.common import distance_objects
from perception_eval.common import distance_objects_bev
from perception_eval.common import DynamicObject
from perception_eval.common import DynamicObject2D
from perception_eval.common import ObjectType
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import LabelType
from perception_eval.common.label import TrafficLightLabel
from perception_eval.common.schema import FrameID
from perception_eval.common.status import MatchingStatus
from perception_eval.common.threshold import get_label_threshold
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.matching import CenterDistanceBEVMatching
from perception_eval.evaluation.matching import CenterDistanceMatching
from perception_eval.evaluation.matching import IOU2dMatching
from perception_eval.evaluation.matching import IOU3dMatching
from perception_eval.evaluation.matching import MatchingLabelPolicy
from perception_eval.evaluation.matching import MatchingMethod
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching import PlaneDistanceMatching


class DynamicObjectWithPerceptionResult:
    """Object result class for perception evaluation.

    Attributes:
        estimated_object (ObjectType): Estimated object.
        ground_truth_object (Optional[ObjectType]): Ground truth object.
        is_label_correct (bool): Whether the label both of `estimated_object` and `ground_truth_object` are same.
        center_distance (Optional[CenterDistanceMatching]): CenterDistanceMatching instance.
        center_distance_bev (Optional[CenterDistanceBEVMatching]): CenterDistanceBEVMatching instance.
        plane_distance (Optional[PlaneDistanceMatching]): PlaneDistanceMatching instance.
            In 2D evaluation, this is None.
        iou_2d (IOU2dMatching): IOU2dMatching instance.
        iou_3d (IOU3dMatching): IOU3dMatching instance. In 2D evaluation, this is None.
    """

    def __init__(
        self,
        estimated_object: ObjectType,
        ground_truth_object: Optional[ObjectType],
        matching_label_policy: MatchingLabelPolicy = MatchingLabelPolicy.DEFAULT,
        transforms: Optional[TransformDict] = None,
    ) -> None:
        """[summary]
        Evaluation result for an object estimated object.

        Args:
            estimated_object (ObjectType): The estimated object by inference like CenterPoint
            ground_truth_objects (Optional[ObjectType]): The list of Ground truth objects
            matching_label_policy (MatchingLabelPolicy, optional): Matching policy considering labels between estimation and GT.
        """
        if ground_truth_object is not None:
            assert isinstance(
                estimated_object, type(ground_truth_object)
            ), f"Input objects type must be same, but got {type(estimated_object)} and {type(ground_truth_object)}"

        self.estimated_object: ObjectType = estimated_object
        self.ground_truth_object: Optional[ObjectType] = ground_truth_object
        self.matching_label_policy = matching_label_policy
        self.transforms: Optional[TransformDict] = transforms

        if isinstance(self.estimated_object, DynamicObject2D) and self.estimated_object.roi is None:
            self.center_distance = None
            self.iou_2d = None
        else:
            self.center_distance: CenterDistanceMatching = CenterDistanceMatching(
                self.estimated_object,
                self.ground_truth_object,
            )
            self.iou_2d: IOU2dMatching = IOU2dMatching(
                self.estimated_object,
                self.ground_truth_object,
            )

        if isinstance(estimated_object, DynamicObject):
            self.center_distance_bev: CenterDistanceBEVMatching = CenterDistanceBEVMatching(
                self.estimated_object,
                self.ground_truth_object,
            )
            self.iou_3d: IOU3dMatching = IOU3dMatching(
                self.estimated_object,
                self.ground_truth_object,
            )
            self.plane_distance: PlaneDistanceMatching = PlaneDistanceMatching(
                self.estimated_object,
                self.ground_truth_object,
                transforms=transforms,
            )
        else:
            self.center_distance_bev = None
            self.iou_3d = None
            self.plane_distance = None

    def __reduce__(self) -> Tuple[DynamicObjectWithPerceptionResult, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling."""
        return (
            self.__class__,
            (self.estimated_object, self.ground_truth_object, self.matching_label_policy, self.transforms),
        )

    def get_status(
        self,
        matching_mode: MatchingMode,
        matching_threshold: Optional[float],
    ) -> Tuple[MatchingStatus, Optional[MatchingStatus]]:
        """Returns matching status both of estimation and GT as `tuple`.

        Args:
            matching_mode (MatchingMode): Matching policy.
            matching_threshold (float): Matching threshold.

        Returns:
            Tuple[MatchingStatus, Optional[MatchingStatus]]: Matching status of estimation and GT.
        """
        if self.ground_truth_object is None:
            return (MatchingStatus.FP, None)

        if self.is_result_correct(matching_mode, matching_threshold):
            return (
                (MatchingStatus.FP, MatchingStatus.TN)
                if self.ground_truth_object.semantic_label.is_fp()
                else (MatchingStatus.TP, MatchingStatus.TP)
            )
        else:
            return (
                (MatchingStatus.FP, MatchingStatus.FP)
                if self.ground_truth_object.semantic_label.is_fp()
                else (MatchingStatus.FP, MatchingStatus.FN)
            )

    def is_result_correct(
        self,
        matching_mode: MatchingMode,
        matching_threshold: Optional[float],
    ) -> bool:
        """The function judging whether the result is target or not.
        Return `False`, if label of GT is "FP" and matching.

        Args:
            matching_mode (MatchingMode):
                    The matching mode to evaluate.
            matching_threshold (float):
                    The matching threshold to evaluate.
                    For example, if matching_mode = IOU3d and matching_threshold = 0.5,
                    and IoU of the object is higher than "matching_threshold",
                    this function appends to return objects.

        Returns:
            bool: If label is correct and satisfy matching threshold, return True
        """
        if self.ground_truth_object is None:
            return False

        if matching_threshold is None:
            return self.is_label_correct

        # Whether is matching to ground truth
        matching: Optional[MatchingMethod] = self.get_matching(matching_mode)
        if matching is None:
            return self.is_label_correct

        is_matching: bool = matching.is_better_than(matching_threshold)
        # Whether both label is true and matching is true
        return (
            not is_matching
            if self.ground_truth_object.semantic_label.is_fp()
            else is_matching and self.is_label_correct
        )

    def get_matching(self, matching_mode: MatchingMode) -> Optional[MatchingMethod]:
        """Get MatchingMethod instance with corresponding MatchingMode.

        Args:
            matching_mode (MatchingMode): MatchingMode instance.

        Raises:
            NotImplementedError: When unexpected MatchingMode is input.

        Returns:
            Optional[MatchingMethod]: Corresponding MatchingMethods instance.
        """
        if matching_mode == MatchingMode.CENTERDISTANCE:
            return self.center_distance
        elif matching_mode == MatchingMode.CENTERDISTANCEBEV:
            return self.center_distance_bev
        elif matching_mode == MatchingMode.PLANEDISTANCE:
            return self.plane_distance
        elif matching_mode == MatchingMode.IOU2D:
            return self.iou_2d
        elif matching_mode == MatchingMode.IOU3D:
            return self.iou_3d
        else:
            raise NotImplementedError

    @property
    def distance_error_bev(self) -> Optional[float]:
        """Get error center distance between ground truth and estimated object in BEV space.

        If `self.ground_truth_object=None`, returns None.

        Returns:
            Optional[float]: error center distance between ground truth and estimated object.
        """
        if self.ground_truth_object is None:
            return None
        return distance_objects_bev(self.estimated_object, self.ground_truth_object)

    @property
    def distance_error(self) -> Optional[float]:
        """Get error center distance between ground truth and estimated object.

        If `self.ground_truth_object=None`, returns None.

        Returns:
            Optional[float]: error center distance between ground truth and estimated object.
        """
        if self.ground_truth_object is None:
            return None
        return distance_objects(self.estimated_object, self.ground_truth_object)

    @property
    def position_error(self) -> Optional[Tuple[float, float, float]]:
        """Get the position error vector from estimated to ground truth object.

        If `self.ground_truth_object=None`, returns None.

        Returns:
            float: x-axis position error[m].
            float: y-axis position error[m].
            float: z-axis position error[m].
        """
        return self.estimated_object.get_position_error(self.ground_truth_object)

    @property
    def heading_error(self) -> Optional[Tuple[float, float, float]]:
        """Get the heading error vector from estimated to ground truth object.

        If `self.ground_truth_object=None`, returns None.

        Returns:
            float: Roll error, in [-pi, pi].
            float: Pitch error, in [-pi, pi].
            float: Yaw error, in [-pi, pi].
        """
        return self.estimated_object.get_heading_error(self.ground_truth_object)

    @property
    def velocity_error(self) -> Optional[Tuple[float, float, float]]:
        """Get the velocity error vector from estimated to ground truth object.

        If `self.ground_truth_object=None`, returns None.
        Also, velocity of estimated or ground truth object is None, returns None too.

        Returns:
            float: x-axis velocity error[m/s].
            float: y-axis velocity error[m/s].
            float: z-axis velocity error[m/s].
        """
        return self.estimated_object.get_velocity_error(self.ground_truth_object)

    @property
    def is_label_correct(self) -> bool:
        """Get whether label is correct.

        Returns:
            bool: Whether label is correct
        """
        if self.ground_truth_object:
            return self.matching_label_policy.is_matchable(self.estimated_object, self.ground_truth_object)
        else:
            return False

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        return {
            "estimated_object": self.estimated_object.serialization(),
            "ground_truth_object": self.ground_truth_object.serialization() if self.ground_truth_object else None,
            "matching_label_policy": self.matching_label_policy.value,
            "transforms": self.transforms if self.transforms else None,
        }

    def deserialization(cls, data: Dict[str, Any]) -> DynamicObjectWithPerceptionResult:
        """Deserialize the data to DynamicObjectWithPerceptionResult."""
        if data["opbject_type"] == DynamicObject2D.__name__:
            object_type = DynamicObject2D
        elif data["object_type"] == DynamicObject.__name__:
            object_type = DynamicObject
        else:
            raise ValueError(f"Unsupported object type: {data['object_type']}")

        return cls(
            estimated_object=object_type.deserialization(data["estimated_object"]),
            ground_truth_object=object_type.deserialization(data["ground_truth_object"])
            if data["ground_truth_object"]
            else None,
            matching_label_policy=MatchingLabelPolicy(data["matching_label_policy"]),
            transforms=data["transforms"],
        )


def get_object_results(
    evaluation_task: EvaluationTask,
    estimated_objects: List[ObjectType],
    ground_truth_objects: List[ObjectType],
    target_labels: Optional[List[LabelType]] = None,
    matching_label_policy: MatchingLabelPolicy = MatchingLabelPolicy.DEFAULT,
    matching_mode: MatchingMode = MatchingMode.CENTERDISTANCE,
    matchable_thresholds: Optional[List[float]] = None,
    transforms: Optional[TransformDict] = None,
    uuid_matching_first: bool = False,
) -> List[DynamicObjectWithPerceptionResult]:
    """Returns list of DynamicObjectWithPerceptionResult.

    For classification, matching objects their uuid.
    Otherwise, matching them depending on their center distance by default.

    In case of FP validation, estimated objects, which have no matching GT, will be ignored.
    Otherwise, they all are FP.

    Args:
        evaluation_task (EvaluationTask): Evaluation task.
        estimated_objects (List[ObjectType]): Estimated objects list.
        ground_truth_objects (List[ObjectType]): Ground truth objects list.
        target_labels (Optional[List[LabelType]]): List of labels.
        matching_label_policy (MatchingLabelPolicy, optional): Policy of matching objects.
            Defaults to MatchingLabelPolicy.DEFAULT.
        matching_mode (MatchingMode): MatchingMode instance.
        matchable_thresholds (Optional[List[float]]): Thresholds to be.
        transforms (Optional[TransformDict]): Transforms to be applied.
        uuid_matching_first (bool): Whether matching based on uuid first or not.

    Returns:
        object_results (List[DynamicObjectWithPerceptionResult]): Object results list.
    """
    # There is no estimated object (= all FN)
    if not estimated_objects:
        return []

    # There is no GT and not FP validation (= all FP)
    if not ground_truth_objects and evaluation_task.is_fp_validation() is False:
        return _get_fp_object_results(estimated_objects)

    assert isinstance(
        ground_truth_objects[0], type(estimated_objects[0])
    ), f"Type of estimation and ground truth must be same, but got {type(estimated_objects[0])} and {type(ground_truth_objects[0])}"

    if (
        isinstance(estimated_objects[0], DynamicObject2D)
        and (estimated_objects[0].roi is None or ground_truth_objects[0].roi is None)
        and isinstance(estimated_objects[0].semantic_label.label, TrafficLightLabel)
    ):
        return _get_object_results_for_tlr(estimated_objects, ground_truth_objects, uuid_matching_first)
    elif isinstance(estimated_objects[0], DynamicObject2D) and (
        estimated_objects[0].roi is None or ground_truth_objects[0].roi is None
    ):
        return _get_object_results_with_id(estimated_objects, ground_truth_objects)

    matching_method_module, maximize = _get_matching_module(matching_mode)
    score_table: np.ndarray = _get_score_table(
        estimated_objects,
        ground_truth_objects,
        matching_label_policy,
        matching_method_module,
        target_labels,
        matchable_thresholds,
        transforms,
    )

    scores = score_table[..., 0]
    is_valid = score_table[..., 1]
    masked_scores = np.where(is_valid, scores, np.nan)

    object_results: List[DynamicObjectWithPerceptionResult] = []
    estimated_objects_: List[ObjectType] = estimated_objects.copy()
    ground_truth_objects_: List[ObjectType] = ground_truth_objects.copy()
    # 1. Matching the nearest estimated objects and GTs which have the same label
    num_estimation, *_ = score_table.shape
    for _ in range(num_estimation):
        if np.isnan(masked_scores).all():
            break

        est_idx, gt_idx = (
            np.unravel_index(np.nanargmax(masked_scores), masked_scores.shape)
            if maximize
            else np.unravel_index(np.nanargmin(masked_scores), masked_scores.shape)
        )

        est_obj = estimated_objects_.pop(est_idx)
        gt_obj = ground_truth_objects_.pop(gt_idx)
        result = DynamicObjectWithPerceptionResult(est_obj, gt_obj, matching_label_policy, transforms=transforms)
        object_results.append(result)

        # Remove corresponding estimated objects and GTs from the score table.
        masked_scores = np.delete(masked_scores, est_idx, axis=0)
        masked_scores = np.delete(masked_scores, gt_idx, axis=1)

        score_table = np.delete(score_table, est_idx, axis=0)
        score_table = np.delete(score_table, gt_idx, axis=1)

    # 2. Matching the nearest estimated objects and GTs regardless of their label
    rest_scores = score_table[..., 0]
    num_rest_estimation, *_ = score_table.shape
    for _ in range(num_rest_estimation):
        if np.isnan(rest_scores).all():
            break

        est_idx, gt_idx = (
            np.unravel_index(np.nanargmax(rest_scores), rest_scores.shape)
            if maximize
            else np.unravel_index(np.nanargmin(rest_scores), rest_scores.shape)
        )

        est_obj = estimated_objects_.pop(est_idx)
        gt_obj = ground_truth_objects_.pop(gt_idx)
        result = DynamicObjectWithPerceptionResult(est_obj, gt_obj, matching_label_policy, transforms=transforms)
        object_results.append(result)

        # Remove corresponding estimated objects and GTs from the score table
        rest_scores = np.delete(rest_scores, est_idx, axis=0)
        rest_scores = np.delete(rest_scores, gt_idx, axis=1)

    # In case of evaluation task is not FP validation,
    # when there are rest of estimated objects, they all are FP.
    # Otherwise, they all are ignored
    if len(estimated_objects_) > 0 and evaluation_task.is_fp_validation() is False:
        object_results += _get_fp_object_results(estimated_objects_)

    return object_results


def _get_object_results_with_id(
    estimated_objects: List[DynamicObject2D],
    ground_truth_objects: List[DynamicObject2D],
) -> List[DynamicObjectWithPerceptionResult]:
    """Returns the list of DynamicObjectWithPerceptionResult considering their uuids.

    This function is used in 2D classification evaluation.

    Args:
        estimated_objects (List[DynamicObject2D]): Estimated objects list.
        ground_truth_objects (List[DynamicObject2D]): Ground truth objects list.

    Returns:
        object_results (List[DynamicObjectWithPerceptionEvaluation]): Object results list.
    """
    object_results: List[DynamicObjectWithPerceptionResult] = []
    estimated_objects_ = estimated_objects.copy()
    ground_truth_objects_ = ground_truth_objects.copy()
    for est_object in estimated_objects:
        for gt_object in ground_truth_objects:
            if est_object.uuid is None or gt_object.uuid is None:
                raise RuntimeError(
                    f"uuid of estimation and ground truth must be set, but got {est_object.uuid} and {gt_object.uuid}"
                )
            if est_object.uuid == gt_object.uuid and est_object.frame_id == gt_object.frame_id:
                object_results.append(
                    DynamicObjectWithPerceptionResult(estimated_object=est_object, ground_truth_object=gt_object)
                )
                estimated_objects_.remove(est_object)
                ground_truth_objects_.remove(gt_object)

    # when there are rest of estimated objects, they all are FP.
    if len(estimated_objects_) > 0 and not any(
        [est.frame_id == FrameID.CAM_TRAFFIC_LIGHT for est in estimated_objects_]
    ):
        object_results += _get_fp_object_results(estimated_objects_)

    return object_results


def _get_object_results_for_tlr(
    estimated_objects: List[DynamicObject2D],
    ground_truth_objects: List[DynamicObject2D],
    uuid_matching_first: bool = False,
) -> List[DynamicObjectWithPerceptionResult]:
    """Returns the list of DynamicObjectWithPerceptionResult for TLR classification.

    This function is used in 2D classification evaluation.

    Args:
        estimated_objects (List[DynamicObject2D]): Estimated objects list.
        ground_truth_objects (List[DynamicObject2D]): Ground truth objects list.
        uuid_matching_first (bool): Whether matching based on uuid first or not.
            if True, Evaluation score is not affected by topic which mix up pedestrian signals, but is affected when the part of traffic light is occluded.
            if False, Reverse of the above.

    Returns:
        object_results (List[DynamicObjectWithPerceptionEvaluation]): Object results list.
    """

    def match_condition(est_object: DynamicObject2D, gt_object: DynamicObject2D, uuid_matching_first: bool) -> bool:
        if uuid_matching_first:
            return (
                est_object.semantic_label == gt_object.semantic_label
                and est_object.uuid == gt_object.uuid
                and est_object.frame_id == gt_object.frame_id
                and est_object in estimated_objects_
                and gt_object in ground_truth_objects_
            )
        else:
            return (
                est_object.semantic_label == gt_object.semantic_label
                and est_object.frame_id == gt_object.frame_id
                and est_object in estimated_objects_
                and gt_object in ground_truth_objects_
            )

    object_results: List[DynamicObjectWithPerceptionResult] = []
    estimated_objects_ = estimated_objects.copy()
    ground_truth_objects_ = ground_truth_objects.copy()
    # 1. matching based on same label primary
    for est_object in estimated_objects:
        for gt_object in ground_truth_objects:
            if est_object.uuid is None or gt_object.uuid is None:
                raise RuntimeError(
                    f"uuid of estimation and ground truth must be set, but got {est_object.uuid} and {gt_object.uuid}"
                )

            if match_condition(est_object, gt_object, uuid_matching_first):
                object_results.append(
                    DynamicObjectWithPerceptionResult(estimated_object=est_object, ground_truth_object=gt_object)
                )
                estimated_objects_.remove(est_object)
                ground_truth_objects_.remove(gt_object)

    # 2. matching based on same ID
    rest_estimated_objects_ = estimated_objects_.copy()
    rest_ground_truth_objects_ = ground_truth_objects_.copy()
    for est_object in rest_estimated_objects_:
        for gt_object in rest_ground_truth_objects_:
            if est_object.uuid is None or gt_object.uuid is None:
                raise RuntimeError(
                    f"uuid of estimation and ground truth must be set, but got {est_object.uuid} and {gt_object.uuid}"
                )

            if (
                est_object.uuid == gt_object.uuid
                and est_object.frame_id == gt_object.frame_id
                and est_object in estimated_objects_
                and gt_object in ground_truth_objects_
            ):
                object_results.append(
                    DynamicObjectWithPerceptionResult(estimated_object=est_object, ground_truth_object=gt_object)
                )
                estimated_objects_.remove(est_object)
                ground_truth_objects_.remove(gt_object)
    return object_results


def _get_fp_object_results(
    estimated_objects: List[ObjectType],
) -> List[DynamicObjectWithPerceptionResult]:
    """Returns the list of DynamicObjectWithPerceptionResult that have no ground truth.

    Args:
        estimated_objects (List[ObjectType]): Estimated objects list.

    Returns:
        object_results (List[DynamicObjectWithPerceptionResult]): FP object results list.
    """
    object_results: List[DynamicObjectWithPerceptionResult] = []
    for est_obj_ in estimated_objects:
        object_result_ = DynamicObjectWithPerceptionResult(estimated_object=est_obj_, ground_truth_object=None)
        object_results.append(object_result_)

    return object_results


def _get_matching_module(matching_mode: MatchingMode) -> Tuple[Callable, bool]:
    """Returns the matching function and boolean flag whether choose maximum value or not.

    Args:
        matching_mode (MatchingMode): MatchingMode instance.

    Returns:
        matching_method_module (Callable): MatchingMethod instance.
        maximize (bool): Whether much bigger is better.
    """
    if matching_mode == MatchingMode.CENTERDISTANCE:
        matching_method_module: CenterDistanceMatching = CenterDistanceMatching
        maximize: bool = False
    elif matching_mode == MatchingMode.CENTERDISTANCEBEV:
        matching_method_module: CenterDistanceBEVMatching = CenterDistanceBEVMatching
        maximize: bool = False
    elif matching_mode == MatchingMode.PLANEDISTANCE:
        matching_method_module: PlaneDistanceMatching = PlaneDistanceMatching
        maximize: bool = False
    elif matching_mode == MatchingMode.IOU2D:
        matching_method_module: IOU2dMatching = IOU2dMatching
        maximize: bool = True
    elif matching_mode == MatchingMode.IOU3D:
        matching_method_module: IOU3dMatching = IOU3dMatching
        maximize: bool = True
    else:
        raise ValueError(f"Unsupported matching mode: {matching_mode}")

    return matching_method_module, maximize


def _get_score_table(
    estimated_objects: List[ObjectType],
    ground_truth_objects: List[ObjectType],
    matching_label_policy: MatchingLabelPolicy,
    matching_method_module: Callable,
    target_labels: Optional[List[LabelType]],
    matchable_thresholds: Optional[List[float]],
    transforms: Optional[TransformDict],
) -> np.ndarray:
    """Returns score table, in shape `(num_estimation, num_ground_truth, 2)`.
    Each element represents `(score, is_same_label)`.

    Args:
        estimated_objects (List[ObjectType]): Estimated objects list.
        ground_truth_objects (List[ObjectType]): Ground truth objects list.
        matching_label_policy (MatchingLabelPolicy): Indicates whether allow to match with unknown label or any.
        matching_method_module (Callable): MatchingMethod instance.
        target_labels (Optional[List[LabelType]]): Target labels to be evaluated.
        matching_thresholds (Optional[List[float]]): List of thresholds

    Returns:
        score_table (numpy.ndarray): in shape (num_estimation, num_ground_truth, 2).
    """
    # fill matching score table, in shape (NumEst, NumGT)
    num_row: int = len(estimated_objects)
    num_col: int = len(ground_truth_objects)
    score_table: np.ndarray = np.full((num_row, num_col, 2), (np.nan, False))
    for i, est_obj in enumerate(estimated_objects):
        for j, gt_obj in enumerate(ground_truth_objects):
            is_same_frame_id: bool = est_obj.frame_id == gt_obj.frame_id

            if is_same_frame_id:
                threshold: Optional[float] = (
                    get_label_threshold(
                        est_obj.semantic_label,
                        target_labels,
                        matchable_thresholds,
                    )
                    if gt_obj.semantic_label.is_fp()
                    else get_label_threshold(
                        gt_obj.semantic_label,
                        target_labels,
                        matchable_thresholds,
                    )
                )

                matching_method: MatchingMethod = matching_method_module(
                    estimated_object=est_obj, ground_truth_object=gt_obj, transforms=transforms
                )

                if threshold is None or (threshold is not None and matching_method.is_better_than(threshold)):
                    is_label_ok = matching_label_policy.is_matchable(est_obj, gt_obj)
                    score_table[i, j] = (matching_method.value, is_label_ok)

    return score_table
