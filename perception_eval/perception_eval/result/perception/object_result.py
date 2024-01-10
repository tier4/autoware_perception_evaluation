# Copyright 2022-2024 TIER IV, Inc.

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

from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import is_same_label
from perception_eval.common.schema import is_same_frame_id
from perception_eval.common.status import MatchingStatus
from perception_eval.matching import CenterDistanceMatching
from perception_eval.matching import IOU2dMatching
from perception_eval.matching import IOU3dMatching
from perception_eval.matching import MatchingMode
from perception_eval.matching import MatchingPolicy
from perception_eval.matching import PlaneDistanceMatching
from perception_eval.object import distance_objects
from perception_eval.object import distance_objects_bev
from perception_eval.object import DynamicObject
from perception_eval.object import DynamicObject2D

if TYPE_CHECKING:
    from perception_eval.common.label import LabelType
    from perception_eval.matching import MatchingMethod
    from perception_eval.object import ObjectType


class PerceptionObjectResult:
    """Object level result for perception evaluation.

    This class consists of the pair of estimated and ground truth objects.
    In case of ground truth object is `None`, it means the estimated object is FP regardless of the matching score.

    Args:
    -----
        estimated_object (ObjectType): Estimated object.
        ground_truth_object (Optional[ObjectType]): Ground truth object. `None` is allowed.
    """

    def __init__(
        self,
        estimated_object: ObjectType,
        ground_truth_object: Optional[ObjectType],
    ) -> None:
        if ground_truth_object is not None:
            assert type(estimated_object) is type(
                ground_truth_object
            ), f"Input objects type must be same, but got {type(estimated_object)} and {type(ground_truth_object)}"

        self.estimated_object = estimated_object
        self.ground_truth_object = ground_truth_object

        if isinstance(self.estimated_object, DynamicObject2D) and self.estimated_object.roi is None:
            self.center_distance = None
            self.iou_2d = None
        else:
            self.center_distance = CenterDistanceMatching(
                self.estimated_object,
                self.ground_truth_object,
            )
            self.iou_2d = IOU2dMatching(
                self.estimated_object,
                self.ground_truth_object,
            )

        if isinstance(estimated_object, DynamicObject):
            self.iou_3d = IOU3dMatching(
                self.estimated_object,
                self.ground_truth_object,
            )
            self.plane_distance = PlaneDistanceMatching(
                self.estimated_object,
                self.ground_truth_object,
            )
        else:
            self.iou_3d = None
            self.plane_distance = None

    def get_status(
        self,
        matching_mode: MatchingMode,
        matching_threshold: Optional[float],
    ) -> Tuple[MatchingStatus, Optional[MatchingStatus]]:
        """Returns matching status both of estimation and GT as `tuple`.

        In case of the ground truth is `None`, the status of GT is `None` and estimation is FP.
        Otherwise, the status is determined by the following rules.
        1. The matching score is better than the threshold:
            a. Evaluation task is FP validation:    (FP, TN)
            b. Otherwise:                           (TP, TP)
        2. Otherwise:
            a. Evaluation task is FP validation:    (FP, FP)
            b. Otherwise:                           (FP, FN)

        Args:
        -----
            matching_mode (MatchingMode): Matching mode.
            matching_threshold (float): Matching threshold.

        Returns:
        --------
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

    def is_result_correct(self, matching_mode: MatchingMode, matching_threshold: Optional[float]) -> bool:
        """The method judging whether the result is target or not.

        Returns `False`, if label of GT is "FP" and matching.

        Args:
        -----
            matching_mode (MatchingMode): The matching mode to evaluate.
            matching_threshold (float): The matching threshold to evaluate.

        Returns:
        --------
            bool: If label is correct and satisfy matching threshold, return True
        """
        if self.ground_truth_object is None:
            return False

        if matching_threshold is None:
            return self.is_label_correct

        # Whether is matching to ground truth
        matching = self.get_matching(matching_mode)
        if matching is None:
            return self.is_label_correct

        is_matching: bool = matching.is_better_than(matching_threshold)
        # Whether both label is true and matching is true
        return not is_matching if self.ground_truth_object.semantic_label.is_fp() else is_matching

    def get_matching(self, matching_mode: MatchingMode) -> Optional[MatchingMethod]:
        """Returns the matching module for corresponding matching mode.

        Note that,
            - 2D detection/tracking tasks, input `PLANEDISTANCE` and `IOU3D` return `None`.
            - 2D classification task, all return `None`.

        Args:
        -----
            matching_mode (MatchingMode): The matching mode.

        Returns:
        --------
            Optional[MatchingMethod]: Corresponding MatchingMethods instance.
        """
        if matching_mode == MatchingMode.CENTERDISTANCE:
            return self.center_distance
        elif matching_mode == MatchingMode.PLANEDISTANCE:
            return self.plane_distance
        elif matching_mode == MatchingMode.IOU2D:
            return self.iou_2d
        elif matching_mode == MatchingMode.IOU3D:
            return self.iou_3d
        else:
            raise NotImplementedError(f"Unexpected matching mode: {matching_mode}")

    @property
    def distance_error_bev(self) -> Optional[float]:
        """Returns the center distance error in BEV coords between estimation and GT.

        If GT is `None` returns `None`.

        Returns:
        --------
            Optional[float]: Calculated error [m].
        """
        if self.ground_truth_object is None:
            return None
        return distance_objects_bev(self.estimated_object, self.ground_truth_object)

    @property
    def distance_error(self) -> Optional[float]:
        """Returns the center distance error in BEV coords between estimation and GT.

        If GT is `None` returns `None`.

        Returns:
        --------
            Optional[float]: Calculated error [m].
        """
        if self.ground_truth_object is None:
            return None
        return distance_objects(self.estimated_object, self.ground_truth_object)

    @property
    def position_error(self) -> Optional[Tuple[float, float, float]]:
        """Returns the position error vector from estimated to ground truth object.

        If GT is `None` returns `None`.

        Returns:
        --------
            Optional[Tuple[float, float, float]]: Errors ordering (x, y, z) [m].
        """
        return self.estimated_object.get_position_error(self.ground_truth_object)

    @property
    def heading_error(self) -> Optional[Tuple[float, float, float]]:
        """Returns the heading error vector from estimated to ground truth object.

        If GT is `None` returns `None`.

        Returns:
            Optional[Tuple[float, float, float]]: Errors ordering (roll, pitch, yaw) [rad] in `[-pi, pi]`.
        """
        return self.estimated_object.get_heading_error(self.ground_truth_object)

    @property
    def velocity_error(self) -> Optional[Tuple[float, float, float]]:
        """Returns the velocity error vector from estimated to ground truth object.

        If GT is `None` returns `None`.
        Also, velocity of estimated or ground truth object is None, returns None too.

        Returns:
            Optional[Tuple[float, float, float]]: Errors ordering (x, y, z) [m/s].
        """
        return self.estimated_object.get_velocity_error(self.ground_truth_object)

    @property
    def is_label_correct(self) -> bool:
        """Get whether label is correct.

        Returns:
            bool: Whether label is correct
        """
        return is_same_label(self.estimated_object, self.ground_truth_object) if self.ground_truth_object else False


def get_object_results(
    evaluation_task: EvaluationTask,
    estimated_objects: List[ObjectType],
    ground_truth_objects: List[ObjectType],
    target_labels: Optional[List[LabelType]] = None,
    matching_policy: MatchingPolicy = MatchingPolicy(),
) -> List[PerceptionObjectResult]:
    """Returns list of object results.

    For classification, matching objects their uuid.
    Otherwise, matching them depending on their center distance by default.

    In case of FP validation, estimated objects, which have no matching GT, will be ignored.
    Otherwise, they all are FP.

    Args:
    -----
        evaluation_task (EvaluationTask): Evaluation task.
        estimated_objects (List[ObjectType]): List of estimated objects.
        ground_truth_objects (List[ObjectType]): List of ground truth objects.
        target_labels (Optional[List[LabelType]]): List of target labels to match. Defaults to None.
        matching_policy (MatchingPolicy): Policy of matching.

    Returns:
    --------
        object_results (List[PerceptionObjectResult]): List of object results.
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

    if evaluation_task == EvaluationTask.CLASSIFICATION2D:
        return _get_object_results_with_id(estimated_objects, ground_truth_objects)

    score_table = _get_score_table(estimated_objects, ground_truth_objects, target_labels, matching_policy)

    # assign correspond GT to estimated objects
    object_results: List[PerceptionObjectResult] = []
    estimated_objects_ = estimated_objects.copy()
    ground_truth_objects_ = ground_truth_objects.copy()
    num_estimation: int = score_table.shape[0]
    for _ in range(num_estimation):
        if np.isnan(score_table).all():
            break

        est_idx, gt_idx = (
            np.unravel_index(np.nanargmax(score_table), score_table.shape)
            if matching_policy.maximize
            else np.unravel_index(np.nanargmin(score_table), score_table.shape)
        )

        # remove corresponding estimated and GT objects
        est_obj_: ObjectType = estimated_objects_.pop(est_idx)
        gt_obj_: ObjectType = ground_truth_objects_.pop(gt_idx)
        score_table = np.delete(score_table, obj=est_idx, axis=0)
        score_table = np.delete(score_table, obj=gt_idx, axis=1)
        object_result_ = PerceptionObjectResult(
            estimated_object=est_obj_,
            ground_truth_object=gt_obj_,
        )
        object_results.append(object_result_)

    # In case of evaluation task is not FP validation,
    # when there are rest of estimated objects, they all are FP.
    # Otherwise, they all are ignored
    if len(estimated_objects_) > 0 and evaluation_task.is_fp_validation() is False:
        object_results += _get_fp_object_results(estimated_objects_)

    return object_results


def _get_object_results_with_id(
    estimated_objects: List[DynamicObject2D],
    ground_truth_objects: List[DynamicObject2D],
) -> List[PerceptionObjectResult]:
    """Returns the list of object results considering their uuids.

    This function is used in 2D classification evaluation.

    Args:
    -----
        estimated_objects (List[DynamicObject2D]): List of estimated objects.
        ground_truth_objects (List[DynamicObject2D]): List of ground truth objects.

    Returns:
    --------
        object_results (List[PerceptionObjectResult]): List of object results.
    """
    object_results: List[PerceptionObjectResult] = []
    estimated_objects_ = estimated_objects.copy()
    ground_truth_objects_ = ground_truth_objects.copy()
    for est_object in estimated_objects:
        for gt_object in ground_truth_objects_:
            if est_object.uuid is None or gt_object.uuid is None:
                raise RuntimeError(
                    f"uuid of estimation and ground truth must be set, but got {est_object.uuid} and {gt_object.uuid}"
                )
            if (
                est_object.uuid == gt_object.uuid
                and is_same_label(est_object, gt_object)
                and is_same_frame_id(est_object, gt_object)
            ):
                object_results.append(
                    PerceptionObjectResult(
                        estimated_object=est_object,
                        ground_truth_object=gt_object,
                    )
                )
                estimated_objects_.remove(est_object)
                ground_truth_objects_.remove(gt_object)

    # when there are rest of estimated objects, they all are FP.
    if len(estimated_objects_) > 0:
        object_results += _get_fp_object_results(estimated_objects_)

    return object_results


def _get_fp_object_results(estimated_objects: List[ObjectType]) -> List[PerceptionObjectResult]:
    """Returns the list of FP object results that have no ground truth.

    Args:
    -----
        estimated_objects (List[ObjectType]): List of object results.

    Returns:
    --------
        List[PerceptionObjectResult]: List of FP object results.
    """
    return [PerceptionObjectResult(est, None) for est in estimated_objects]


def _get_score_table(
    estimated_objects: List[ObjectType],
    ground_truth_objects: List[ObjectType],
    target_labels: List[LabelType],
    matching_policy: MatchingPolicy,
) -> np.ndarray:
    """Returns score table, in shape (num_estimation, num_ground_truth).

    Args:
    -----
        estimated_objects (List[ObjectType]): Estimated objects list.
        ground_truth_objects (List[ObjectType]): Ground truth objects list.
        target_labels (List[LabelType]): Target labels to be evaluated.
        matching_policy (MatchingPolicy): Policy of matching.

    Returns:
        np.ndarray: Array in shape (num_estimation, num_ground_truth).
    """
    num_row: int = len(estimated_objects)
    num_col: int = len(ground_truth_objects)
    score_table = np.full((num_row, num_col), np.nan)
    for i, estimation in enumerate(estimated_objects):
        for j, ground_truth in enumerate(ground_truth_objects):
            if matching_policy.is_matchable(estimation, ground_truth):
                score_table[i, j] = matching_policy.get_matching_score(estimation, ground_truth, target_labels)

    return score_table
