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
from typing import Dict
from typing import Optional
from typing import Tuple

from perception_eval.common import distance_objects
from perception_eval.common import distance_objects_bev
from perception_eval.common import DynamicObject
from perception_eval.common import DynamicObject2D
from perception_eval.common import ObjectType
from perception_eval.common.status import MatchingStatus
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

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> DynamicObjectWithPerceptionResult:
        """Deserialize the data to DynamicObjectWithPerceptionResult."""
        if data["object_type"] == DynamicObject2D.__name__:
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
