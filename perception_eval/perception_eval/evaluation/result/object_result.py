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

from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from perception_eval.common import distance_objects
from perception_eval.common import distance_objects_bev
from perception_eval.common import DynamicObject
from perception_eval.common import DynamicObject2D
from perception_eval.common import ObjectType
from perception_eval.common.label import CommonLabel
from perception_eval.evaluation.matching import CenterDistanceMatching
from perception_eval.evaluation.matching import IOU2dMatching
from perception_eval.evaluation.matching import IOU3dMatching
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
        plane_distance (Optional[PlaneDistanceMatching]): PlaneDistanceMatching instance.
            In 2D evaluation, this is None.
        iou_2d (IOU2dMatching): IOU2dMatching instance.
        iou_3d (IOU3dMatching): IOU3dMatching instance. In 2D evaluation, this is None.
    """

    def __init__(
        self,
        estimated_object: ObjectType,
        ground_truth_object: Optional[ObjectType],
    ) -> None:
        """[summary]
        Evaluation result for an object estimated object.

        Args:
            estimated_object (ObjectType): The estimated object by inference like CenterPoint
            ground_truth_objects (Optional[ObjectType]): The list of Ground truth objects
        """
        if ground_truth_object is not None:
            assert type(estimated_object) == type(
                ground_truth_object
            ), f"Input objects type must be same, but got {type(estimated_object)} and {type(ground_truth_object)}"

        self.estimated_object: ObjectType = estimated_object
        self.ground_truth_object: Optional[ObjectType] = ground_truth_object

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
            self.iou_3d: IOU3dMatching = IOU3dMatching(
                self.estimated_object,
                self.ground_truth_object,
            )
            self.plane_distance: PlaneDistanceMatching = PlaneDistanceMatching(
                self.estimated_object,
                self.ground_truth_object,
            )
        else:
            self.iou_3d = None
            self.plane_distance = None

    def is_result_correct(
        self,
        matching_mode: MatchingMode,
        matching_threshold: float,
    ) -> bool:
        """The function judging whether the result is target or not.

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
        # Whether is matching to ground truth
        matching: Optional[MatchingMethod] = self.get_matching(matching_mode)

        return (
            self.is_label_correct
            if matching is None
            else matching.is_better_than(matching_threshold)
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
            return self.estimated_object.semantic_label == self.ground_truth_object.semantic_label
        else:
            return False


def get_object_results(
    estimated_objects: List[ObjectType],
    ground_truth_objects: List[ObjectType],
    matching_mode: MatchingMode = MatchingMode.CENTERDISTANCE,
) -> List[DynamicObjectWithPerceptionResult]:
    """Returns list of DynamicObjectWithPerceptionResult.

    Args:
        estimated_objects (List[ObjectType]): Estimated objects list.
        ground_truth_objects (List[ObjectType]): Ground truth objects list.
        matching_mode (MatchingMode): MatchingMode instance.

    Returns:
        object_results (List[DynamicObjectWithPerceptionResult]): Object results list.
    """
    # There is no estimated object (= all FN)
    if not estimated_objects:
        return []

    # There is no GT (= all FP)
    if not ground_truth_objects:
        return _get_fp_object_results(estimated_objects)

    assert isinstance(
        ground_truth_objects[0], type(estimated_objects[0])
    ), f"Type of estimation and ground truth must be same, but got {type(estimated_objects[0])} and {type(ground_truth_objects[0])}"

    if isinstance(estimated_objects[0], DynamicObject2D) and (
        estimated_objects[0].roi is None or ground_truth_objects[0].roi is None
    ):
        return _get_object_results_with_id(estimated_objects, ground_truth_objects)

    matching_method_module, maximize = _get_matching_module(matching_mode)
    score_table: np.ndarray = _get_score_table(
        estimated_objects,
        ground_truth_objects,
        matching_method_module,
    )

    # assign correspond GT to estimated objects
    object_results: List[DynamicObjectWithPerceptionResult] = []
    estimated_objects_: List[DynamicObject] = estimated_objects.copy()
    ground_truth_objects_: List[DynamicObject] = ground_truth_objects.copy()
    num_estimation: int = score_table.shape[0]
    for _ in range(num_estimation):
        if np.isnan(score_table).all():
            break

        est_idx, gt_idx = (
            np.unravel_index(np.nanargmax(score_table), score_table.shape)
            if maximize
            else np.unravel_index(np.nanargmin(score_table), score_table.shape)
        )

        # remove corresponding estimated and GT objects
        est_obj_: DynamicObject = estimated_objects_.pop(est_idx)
        gt_obj_: DynamicObject = ground_truth_objects_.pop(gt_idx)
        score_table = np.delete(score_table, obj=est_idx, axis=0)
        score_table = np.delete(score_table, obj=gt_idx, axis=1)
        object_result_: DynamicObjectWithPerceptionResult = DynamicObjectWithPerceptionResult(
            estimated_object=est_obj_,
            ground_truth_object=gt_obj_,
        )
        object_results.append(object_result_)

    # when there are rest of estimated objects, they all are FP.
    if len(estimated_objects_) > 0:
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
        for gt_object in ground_truth_objects_:
            if est_object.uuid is None or gt_object.uuid is None:
                raise RuntimeError(
                    f"uuid of estimation and ground truth must be set, but got {est_object.uuid} and {gt_object.uuid}"
                )
            if (
                est_object.uuid == gt_object.uuid
                and est_object.semantic_label == gt_object.semantic_label
                and est_object.frame_id == gt_object.frame_id
            ):
                object_results.append(
                    DynamicObjectWithPerceptionResult(
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
        object_result_: DynamicObjectWithPerceptionResult = DynamicObjectWithPerceptionResult(
            estimated_object=est_obj_,
            ground_truth_object=None,
        )
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
    matching_method_module: Callable,
) -> np.ndarray:
    """Returns score table, in shape (num_estimation, num_ground_truth).

    Args:
        estimated_objects (List[ObjectType]): Estimated objects list.
        ground_truth_objects (List[ObjectType]): Ground truth objects list.
        matching_method_module (Callable): MatchingMethod instance.

    Returns:
        score_table (numpy.ndarray): in shape (num_estimation, num_ground_truth).
    """
    # fill matching score table, in shape (NumEst, NumGT)
    num_row: int = len(estimated_objects)
    num_col: int = len(ground_truth_objects)
    score_table: np.ndarray = np.full((num_row, num_col), np.nan)
    for i, est_obj in enumerate(estimated_objects):
        for j, gt_obj in enumerate(ground_truth_objects):
            if (
                est_obj.semantic_label == gt_obj.semantic_label
                or any(
                    label == CommonLabel.UNKNOWN
                    for label in (est_obj.semantic_label.label, gt_obj.semantic_label.label)
                )
            ) and est_obj.frame_id == gt_obj.frame_id:
                matching_method: MatchingMethod = matching_method_module(
                    estimated_object=est_obj, ground_truth_object=gt_obj
                )
                score_table[i, j] = matching_method.value
    return score_table
