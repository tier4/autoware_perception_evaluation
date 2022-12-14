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

from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from perception_eval.common.object import DynamicObject
from perception_eval.common.object import distance_objects
from perception_eval.common.object import distance_objects_bev
from perception_eval.evaluation.matching.object_matching import CenterDistanceMatching
from perception_eval.evaluation.matching.object_matching import IOU3dMatching
from perception_eval.evaluation.matching.object_matching import IOUBEVMatching
from perception_eval.evaluation.matching.object_matching import MatchingMethod
from perception_eval.evaluation.matching.object_matching import MatchingMode
from perception_eval.evaluation.matching.object_matching import PlaneDistanceMatching


class DynamicObjectWithPerceptionResult:
    """[summary]
    Evaluation result for a estimated object

    Attributes:
        self.estimated_object (DynamicObject):
                The estimated object by inference like CenterPoint.
        self.ground_truth_object (Optional[DynamicObject]):
                Ground truth object corresponding to estimated object.
        self.is_label_correct (bool):
                Whether the label of estimated_object is same as the label of ground truth object
        self.center_distance (CenterDistanceMatching):
                The center distance between estimated object and ground truth object
        self.plane_distance (PlaneDistanceMatching):
                The plane distance for use case evaluation
        self.iou_bev (IOUBEVMatching):
                The bev IoU between estimated object and ground truth object
        self.iou_3d (IOU3dMatching):
                The 3d IoU between estimated object and ground truth object
    """

    def __init__(
        self,
        estimated_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
    ) -> None:
        """[summary]
        Evaluation result for an object estimated object.

        Args:
            estimated_object (DynamicObject): The estimated object by inference like CenterPoint
            ground_truth_objects (Optional[DynamicObject]): The list of Ground truth objects
        """
        self.estimated_object: DynamicObject = estimated_object
        self.ground_truth_object: Optional[DynamicObject] = ground_truth_object

        # detection
        self.center_distance: CenterDistanceMatching = CenterDistanceMatching(
            self.estimated_object,
            self.ground_truth_object,
        )
        self.iou_bev: IOUBEVMatching = IOUBEVMatching(
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
        )

    def is_result_correct(
        self,
        matching_mode: MatchingMode,
        matching_threshold: float,
    ) -> bool:
        """[summary]
        The function judging whether the result is target or not.

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
        matching: MatchingMethod = self.get_matching(matching_mode)
        is_matching_: bool = matching.is_better_than(matching_threshold)
        # Whether both label is true and matching is true
        is_correct: bool = self.is_label_correct and is_matching_
        return is_correct

    def get_matching(
        self,
        matching_mode: MatchingMode,
    ) -> MatchingMethod:
        """[summary]
        Get matching class

        Args:
            matching_mode (MatchingMode):
                    The matching mode to evaluate. Defaults to None.

        Raises:
            NotImplementedError: Not implemented matching class

        Returns:
            Matching: Matching class
        """
        if matching_mode == MatchingMode.CENTERDISTANCE:
            return self.center_distance
        elif matching_mode == MatchingMode.PLANEDISTANCE:
            return self.plane_distance
        elif matching_mode == MatchingMode.IOUBEV:
            return self.iou_bev
        elif matching_mode == MatchingMode.IOU3D:
            return self.iou_3d
        else:
            raise NotImplementedError

    @property
    def distance_error_bev(self) -> Optional[float]:
        """[summary]
        Get error center distance between ground truth and estimated object in BEV space.

        Returns:
            Optional[float]: error center distance between ground truth and estimated object.
        """
        if self.ground_truth_object is None:
            return None
        return distance_objects_bev(self.estimated_object, self.ground_truth_object)

    @property
    def distance_error(self) -> Optional[float]:
        """[summary]
        Get error center distance between ground truth and estimated object.

        Returns:
            Optional[float]: error center distance between ground truth and estimated object.
        """
        if self.ground_truth_object is None:
            return None
        return distance_objects(self.estimated_object, self.ground_truth_object)

    @property
    def position_error(self) -> Optional[Tuple[float, float, float]]:
        """[summary]
        Get the position error vector from estimated to ground truth object.
        If ground truth object is None, returns None.

        Returns:
            err_x (float): Error of x[m].
            err_y (float): Error of y[m].
            err_z (float): Error of z[m].
        """
        return self.estimated_object.get_position_error(self.ground_truth_object)

    @property
    def heading_error(self) -> Optional[Tuple[float, float, float]]:
        """[summary]
        Get the heading error vector from estimated to ground truth object.
        If ground truth object is None, returns None.

        Returns:
            err_x (float): Error of rotation angle around x-axis, in the range [-pi, pi].
            err_y (float): Error of rotation angle around y-axis, in the range [-pi, pi].
            err_z (float): Error of rotation angle around z-axis, in the range [-pi, pi].
        """
        return self.estimated_object.get_heading_error(self.ground_truth_object)

    @property
    def velocity_error(self) -> Optional[Tuple[float, float, float]]:
        """[summary]
        Get the velocity error vector from estimated to ground truth object.
        If ground truth object is None, returns None.
        Also, velocity of estimated or ground truth object is None, returns None too.

        Returns:
            err_vx (float): Error of vx[m].
            err_vy (float): Error of vy[m].
            err_vz (float): Error of vz[m].
        """
        return self.estimated_object.get_velocity_error(self.ground_truth_object)

    @property
    def is_label_correct(self) -> bool:
        """[summary]
        Get whether label is correct.

        Returns:
            bool: Whether label is correct
        """
        if self.ground_truth_object:
            return self.estimated_object.semantic_label == self.ground_truth_object.semantic_label
        else:
            return False


def get_object_results(
    estimated_objects: List[DynamicObject],
    ground_truth_objects: List[DynamicObject],
    matching_mode: MatchingMode = MatchingMode.CENTERDISTANCE,
) -> List[DynamicObjectWithPerceptionResult]:
    """[summary]
    Returns list of DynamicObjectWithPerceptionResult.

    Args:
        estimated_objects (List[DynamicObject]): The list of estimated object.
        ground_truth_objects (List[DynamicObject]): The list of ground truth object.
        matching_mode (MatchingMode): The MatchingMode instance.

    Returns:
        object_results (List[DynamicObjectWithPerceptionResult]): The list of object result.
    """
    # There is no estimated object (= all FN)
    if not estimated_objects:
        return []

    # There is no GT (= all FP)
    object_results: List[DynamicObjectWithPerceptionResult] = []
    if not ground_truth_objects:
        for estimated_object_ in estimated_objects:
            object_results.append(
                DynamicObjectWithPerceptionResult(
                    estimated_object=estimated_object_,
                    ground_truth_object=None,
                )
            )
        return object_results

    if matching_mode == MatchingMode.CENTERDISTANCE:
        matching_method_module: CenterDistanceMatching = CenterDistanceMatching
        maximize: bool = False
    elif matching_mode == MatchingMode.PLANEDISTANCE:
        matching_method_module: PlaneDistanceMatching = PlaneDistanceMatching
        maximize: bool = False
    elif matching_mode == MatchingMode.IOUBEV:
        matching_method_module: IOUBEVMatching = IOUBEVMatching
        maximize: bool = True
    elif matching_mode == MatchingMode.IOU3D:
        matching_method_module: IOU3dMatching = IOU3dMatching
        maximize: bool = True
    else:
        raise ValueError(f"Unsupported matching mode: {matching_mode}")

    # fill matching score table, in shape (NumEst, NumGT)
    num_row: int = len(estimated_objects)
    num_col: int = len(ground_truth_objects)
    score_table: np.ndarray = np.full((num_row, num_col), np.nan)
    for i, estimated_object_ in enumerate(estimated_objects):
        for j, ground_truth_object_ in enumerate(ground_truth_objects):
            if estimated_object_.semantic_label == ground_truth_object_.semantic_label:
                matching_method: MatchingMethod = matching_method_module(
                    estimated_object=estimated_object_,
                    ground_truth_object=ground_truth_object_,
                )
                score_table[i, j] = matching_method.value

    # assign correspond GT to estimated objects
    estimated_objects_: List[DynamicObject] = estimated_objects.copy()
    ground_truth_objects_: List[DynamicObject] = ground_truth_objects.copy()
    for _ in range(num_row):
        if np.isnan(score_table).all():
            break

        if maximize:
            est_idx, gt_idx = np.unravel_index(
                np.nanargmax(score_table),
                score_table.shape,
            )
        else:
            est_idx, gt_idx = np.unravel_index(
                np.nanargmin(score_table),
                score_table.shape,
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
        for est_obj_ in estimated_objects_:
            object_result_: DynamicObjectWithPerceptionResult = DynamicObjectWithPerceptionResult(
                estimated_object=est_obj_,
                ground_truth_object=None,
            )
            object_results.append(object_result_)

    return object_results
