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

from abc import ABC
from abc import abstractmethod
from enum import Enum
import math
from typing import Optional
from typing import overload
from typing import Tuple

import numpy as np
from perception_eval.common import distance_objects
from perception_eval.common import distance_objects_bev
from perception_eval.common import distance_points_bev
from perception_eval.common import ObjectType
from perception_eval.common.label import is_same_label
from perception_eval.common.object import DynamicObject
from perception_eval.common.point import get_point_left_right_index
from perception_eval.common.point import polygon_to_list
from perception_eval.common.schema import FrameID
from perception_eval.common.shape import ShapeType
from perception_eval.common.transform import TransformDict
from shapely.geometry import Polygon


class MatchingLabelPolicy(Enum):
    DEFAULT = "DEFAULT"
    ALLOW_UNKNOWN = "ALLOW_UNKNOWN"
    ALLOW_ANY = "ALLOW_ANY"

    @classmethod
    def from_str(cls, name: str) -> MatchingLabelPolicy:
        """Construct an enum member from str name.

        Args:
            name (str): Name of a member.

        Returns:
            MatchingLanePolicy: Constructed member.
        """
        name = name.upper()
        assert name in cls.__members__, f"{name} is not in enum members."
        return cls.__members__[name]

    def is_matchable(self, estimation: ObjectType, ground_truth: ObjectType) -> bool:
        """Indicate whether input estimation and GT is matchable considering their label.

        Args:
            estimation (ObjectType): Estimated object.
            ground_truth (ObjectType): GT object.

        Returns:
            bool: Return True if they are matchable.
        """
        if ground_truth.semantic_label.is_fp() or self == MatchingLabelPolicy.ALLOW_ANY:
            return True
        elif self == MatchingLabelPolicy.ALLOW_UNKNOWN:
            return is_same_label(estimation, ground_truth) or estimation.semantic_label.is_unknown()
        else:  # STRICT
            return is_same_label(estimation, ground_truth)


class MatchingMode(Enum):
    """[summary]
    The mode enum for matching algorithm.

    CENTERDISTANCE: Center distance in meters for 3D objects and in pixels[px] for 2D objects.
    CENTERDISTANCEBEV: Center distance in meters in Bird Eye View for 3D objects.
    IOU2D : IoU (Intersection over Union) in BEV (Bird Eye View) for 3D objects, pixel for 2D objects.
    IOU3D : IoU (Intersection over Union) in 3D
    PLANEDISTANCE: The plane distance
        TLR_CLASSIFICATION: Traffic Light Recognition (TLR) in 2D classification task
        CLASSIFICATION_2D: Classification in 2D task
    """

    CENTERDISTANCE = "Center Distance"
    CENTERDISTANCEBEV = "Center Distance BEV"
    IOU2D = "IoU 2D"
    IOU3D = "IoU 3D"
    PLANEDISTANCE = "Plane Distance"
    TLR_CLASSIFICATION = "TLR Classification 2D"
    CLASSIFICATION_2D = "Classification 2D"

    def __str__(self) -> str:
        return self.value


class MatchingMethod(ABC):
    """A base class for matching method class.

    Attributes:
        mode (MatchingMode): MatchingMode instance.
        value (Optional[float]): Matching score.

    Args:
        estimated_object (ObjectType): Estimated object.
        ground_truth_object (Optional[ObjectType]): Ground truth object.

    Raises:
        AssertionError: When types of input objects are not same.
    """

    mode: MatchingMode

    @overload
    def __init__(self, estimated_object: ObjectType, ground_truth_object: Optional[ObjectType]) -> None:
        ...

    @overload
    def __init__(
        self,
        estimated_object: ObjectType,
        ground_truth_object: Optional[ObjectType],
        transforms: Optional[TransformDict] = None,
    ) -> None:
        ...

    def __init__(
        self,
        estimated_object: ObjectType,
        ground_truth_object: Optional[ObjectType],
        transforms: Optional[TransformDict] = None,
    ) -> None:
        if ground_truth_object is not None:
            assert isinstance(estimated_object, type(ground_truth_object))
        self.value: Optional[float] = self._calculate_matching_score(
            estimated_object=estimated_object, ground_truth_object=ground_truth_object, transforms=transforms
        )

    @abstractmethod
    def _calculate_matching_score(
        self,
        estimated_object: ObjectType,
        ground_truth_object: Optional[ObjectType],
        transforms: Optional[TransformDict] = None,
    ) -> Optional[float]:
        pass

    @abstractmethod
    def is_better_than(
        self,
        threshold_value: float,
    ) -> bool:
        """Judge whether value is better than threshold.

        This function must be implemented in each inherited class.
        If input `self.value=None`, it always returns False.

        Args:
            threshold_value (float): Threshold value.

        Returns:
            bool: If value is better than threshold, return True.
        """
        pass


class CenterDistanceMatching(MatchingMethod):
    """A class for matching objects by center distance.

    If input `ground_truth_object=None`, `self.value=None`

    Attributes:
        mode (MatchingMode): Matching mode that is `MatchingMode.CENTERDISTANCE`.
        value (Optional[float]): Center distance score.

    Args:
        estimated_object (ObjectType): Estimated object.
        ground_truth_object (Optional[ObjectType]): Ground truth object.
    """

    mode: MatchingMode = MatchingMode.CENTERDISTANCE

    def is_better_than(
        self,
        threshold_value: float,
    ) -> bool:
        """[summary]
        Judge whether value is better than threshold.

        If `self.value=None`, always returns False.

        Args:
            threshold_value (float): Threshold value.

        Returns:
            bool: If value is better than threshold, return True.
        """
        if self.value is None:
            return False

        return self.value < threshold_value

    def _calculate_matching_score(
        self,
        estimated_object: ObjectType,
        ground_truth_object: Optional[ObjectType],
        transforms: Optional[TransformDict] = None,
    ) -> Optional[float]:
        """Get center distance.

        If input `ground_truth_object=None`, it always returns None.

        Args:
            estimated_object (ObjectType): Estimated object.
            ground_truth_object (Optional[ObjectType]): Ground truth object.

        Returns:
            Optional[float]: Center distance between 2 objects.
        """
        if ground_truth_object is None:
            return None
        return distance_objects(estimated_object, ground_truth_object)


class CenterDistanceBEVMatching(MatchingMethod):
    """A class for matching objects by center distance.

    If input `ground_truth_object=None`, `self.value=None`

    Attributes:
        mode (MatchingMode): Matching mode that is `MatchingMode.CENTERDISTANCEBEV`.
        value (Optional[float]): Center distance score.

    Args:
        estimated_object (ObjectType): Estimated object.
        ground_truth_object (Optional[ObjectType]): Ground truth object.
    """

    mode: MatchingMode = MatchingMode.CENTERDISTANCEBEV

    def is_better_than(
        self,
        threshold_value: float,
    ) -> bool:
        """[summary]
        Judge whether value is better than threshold.

        If `self.value=None`, always returns False.

        Args:
            threshold_value (float): Threshold value.

        Returns:
            bool: If value is better than threshold, return True.
        """
        if self.value is None:
            return False

        return self.value < threshold_value

    def _calculate_matching_score(
        self,
        estimated_object: ObjectType,
        ground_truth_object: Optional[ObjectType],
        transforms: Optional[TransformDict] = None,
    ) -> Optional[float]:
        """Get center distance.

        If input `ground_truth_object=None`, it always returns None.

        Args:
            estimated_object (ObjectType): Estimated object.
            ground_truth_object (Optional[ObjectType]): Ground truth object.

        Returns:
            Optional[float]: Center distance between 2 objects.
        """
        if ground_truth_object is None:
            return None
        return distance_objects_bev(estimated_object, ground_truth_object)


_PlanePointType = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]


class PlaneDistanceMatching(MatchingMethod):
    """A class for matching objects by plane distance.

    Input objects of this class must be 3D that means `DynamicObject3D`.
    If input `ground_truth_object=None`, `self.value` and each NN plane attribute is None.

    Attributes:
        mode (MatchingMode): Matching mode that is `MatchingMode.PLANEDISTANCE`.
        value (Optional[float]): Plane distance[m].
        ground_truth_nn_plane (_PlanePointType):
            Vertices of NN plane of ground truth object ((x1, y1), (x2, y2)).
        estimated_nn_plane (_PlanePointType):
            Vertices of NN plane for estimation ((x1, y1), (x2, y2)).

    Args:
        estimated_object (DynamicObject): Estimated object.
        ground_truth_object (Optional[DynamicObject]): Ground truth object.
    """

    mode: MatchingMode = MatchingMode.PLANEDISTANCE

    def __init__(
        self,
        estimated_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
        transforms: Optional[TransformDict] = None,
    ) -> None:
        self.ground_truth_nn_plane: _PlanePointType = ((np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan))
        self.estimated_nn_plane: _PlanePointType = ((np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan))
        super().__init__(
            estimated_object=estimated_object, ground_truth_object=ground_truth_object, transforms=transforms
        )

    def is_better_than(
        self,
        threshold_value: float,
    ) -> bool:
        """Judge whether value is better than threshold.

        If `self.value=None`, it always returns False.

        Args:
            threshold_value (float): Threshold value.

        Returns:
            bool: If value is better than threshold, return True.
        """
        if self.value is None:
            return False

        return self.value < threshold_value

    def _calculate_matching_score(
        self,
        estimated_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
        transforms: Optional[TransformDict] = None,
    ) -> Optional[float]:
        """Calculate plane distance between estimation and ground truth and NN planes.

        This function also set NN plane attributes for estimation and ground truth.
        If input `ground_truth_object=None`, it always returns None.

        If the shapes of both estimated object and GT are BOUNDING_BOX, calculates plane distance.
        Otherwise, calculates center distance in BEV.

        Args:
            estimated_object (DynamicObject): Estimated object.
            ground_truth_object (Optional[DynamicObject]): Ground truth object.
            transforms (Optional[TransformDict]): Transforms.

        Returns:
            Optional[float]: Plane distance.
        """
        if ground_truth_object is None:
            return None

        # Get corner points of estimated object from footprint
        est_footprint: Polygon = estimated_object.get_footprint()
        est_corners = np.array(polygon_to_list(est_footprint))

        # Get corner points of ground truth object from footprint
        gt_footprint: Polygon = ground_truth_object.get_footprint()
        gt_corners = np.array(polygon_to_list(gt_footprint))

        if (
            estimated_object.state.shape_type == ShapeType.BOUNDING_BOX
            and ground_truth_object.state.shape_type == ShapeType.BOUNDING_BOX
        ):
            _, _, error_yaw = estimated_object.get_heading_error(ground_truth_object)
            if abs(error_yaw) > np.pi / 2:
                est_corners = est_corners[[2, 3, 0, 1]]  # based on reverse clockwise order from left top

            # Calculate min distance from ego vehicle
            if ground_truth_object.frame_id != FrameID.BASE_LINK:
                assert transforms is not None, f"`transforms` must be specified for {ground_truth_object.frame_id}"
                gt_corners_base_link = np.array(
                    [
                        transforms.transform((ground_truth_object.frame_id, FrameID.BASE_LINK), corner)
                        for corner in gt_corners
                    ]
                )
                gt_distances = np.linalg.norm(gt_corners_base_link[:, :2], axis=1)
            else:
                gt_distances = np.linalg.norm(gt_corners[:, :2], axis=1)
            sort_idx = np.argsort(gt_distances)

            gt_corners = gt_corners[sort_idx]
            est_corners = est_corners[sort_idx]

            est_plane_points = est_corners[:2].tolist()
            gt_plane_points = gt_corners[:2].tolist()
            left_idx, right_idx = get_point_left_right_index(gt_plane_points[0], gt_plane_points[1])
            gt_left_point, gt_right_point = gt_plane_points[left_idx], gt_plane_points[right_idx]
            est_left_point, est_right_point = est_plane_points[left_idx], est_plane_points[right_idx]
            distance_left_point: float = distance_points_bev(est_left_point, gt_left_point)
            distance_right_point: float = distance_points_bev(est_right_point, gt_right_point)
            distance_squared = distance_left_point**2 + distance_right_point**2
            plane_distance = math.sqrt(0.5 * distance_squared)
            # NOTE: round because the distance become 0.9999999... expecting 1.0
            distance = round(plane_distance, 10)
            self.ground_truth_nn_plane = (gt_left_point, gt_right_point)
            self.estimated_nn_plane = (est_left_point, est_right_point)
        else:
            distance = distance_points_bev(estimated_object.state.position, ground_truth_object.state.position)

        return distance


class IOU2dMatching(MatchingMethod):
    """A class for Matching by 2D IoU in BEV for 3D objects or in pixels for 2D objects.

    This class allows either `DynamicObject` or `DynamicObject2D` as input type.

    Attributes:
        mode (MatchingMode): Matching mode that is `MatchingMode.IOU2D`.
        value (Optional[float]): 2D IoU score.

    Args:
        estimated_object (DynamicObject): Estimated object.
        ground_truth_object (Optional[DynamicObject]): Ground truth object.
    """

    mode: MatchingMode = MatchingMode.IOU2D

    def is_better_than(
        self,
        threshold_value: float,
    ) -> bool:
        """Judge whether value is better than threshold.

        If input `self.value=None`, always returns False.
        Input `threshold_value` must be in [0.0, 1.0].

        Args:
            threshold_value (float): The threshold value

        Returns:
            bool: If value is better than threshold, return True.

        Raises:
            AssertionError: When `threshold_value` is not in [0.0, 1.0].
        """
        assert 0.0 <= threshold_value <= 1.0, f"threshold must be [0.0, 1.0], but got {threshold_value}."

        if self.value is None:
            return False
        else:
            return self.value > threshold_value

    def _calculate_matching_score(
        self,
        estimated_object: ObjectType,
        ground_truth_object: Optional[ObjectType],
        transforms: Optional[TransformDict] = None,
    ) -> float:
        """Calculate 2D IoU score.

        If input `ground_truth_object=None`, always returns 0.0.

        NOTE:
            If Object's size is tiny, it returns wrong IoU score.

        Args:
            estimated_object (ObjectType): Estimated object
            ground_truth_object (Optional[ObjectType]): Ground truth object.

        Returns:
            float: The value of 2D IoU score. If ground truth object is None, returns 0.0.

        Reference:
            https://github.com/lyft/nuscenes-devkit/blob/49c36da0a85da6bc9e8f2a39d5d967311cd75069/lyft_dataset_sdk/eval/detection/mAP_evaluation.py
        """

        if ground_truth_object is None:
            return 0.0

        # TODO: if tiny box dim seen return 0.0 IOU
        if isinstance(estimated_object, DynamicObject):
            estimated_object_area: float = estimated_object.get_area_bev()
            ground_truth_object_area: float = ground_truth_object.get_area_bev()
        else:
            estimated_object_area: float = estimated_object.get_area()
            ground_truth_object_area: float = ground_truth_object.get_area()
        intersection_area: float = _get_area_intersection(estimated_object, ground_truth_object)
        union_area: float = estimated_object_area + ground_truth_object_area - intersection_area
        iou_bev: float = intersection_area / union_area
        return iou_bev


class IOU3dMatching(MatchingMethod):
    """A class for matching by 3d IoU.

    This class only allows `DynamicObject` as input.

    Attributes:
        mode (MatchingMode): Matching mode that is `MatchingMode.IOU3D`.
        value (Optional[float]): 3D IoU score.

    Args:
        estimated_object (DynamicObject): Estimated object.
        ground_truth_object (Optional[DynamicObject]): Ground truth object.
    """

    mode: MatchingMode = MatchingMode.IOU3D

    def is_better_than(
        self,
        threshold_value: float,
    ) -> bool:
        """Judge whether value is better than threshold.

        If `self.value` is None, it always returns `False`.

        Args:
            threshold_value (float): Threshold value that must be in [0.0, 1.0].

        Returns:
            bool: If value is better than threshold, return True.

        Raises:
            AssertionError: When `threshold_value` is not in [0.0, 1.0].
        """
        assert 0.0 <= threshold_value <= 1.0, f"threshold must be [0.0, 1.0], but got {threshold_value}"

        if self.value is None:
            return False
        else:
            return self.value > threshold_value

    def _calculate_matching_score(
        self,
        estimated_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
        transforms: Optional[TransformDict] = None,
    ) -> float:
        """Calculate 3D IoU score.

        If input `ground_truth_object=None`, it always returns 0.0.

        Args:
            estimated_object (DynamicObject): The estimated object
            ground_truth_object (DynamicObject): The corresponded ground truth object

        Returns:
            Optional[float]: The value of 3D IoU.
                            If estimated_object do not have corresponded ground truth object,
                            return 0.0.
        """
        if ground_truth_object is None:
            return 0.0

        estimated_object_volume: float = estimated_object.get_volume()
        ground_truth_object_volume: float = ground_truth_object.get_volume()
        intersection: float = _get_volume_intersection(estimated_object, ground_truth_object)
        union: float = estimated_object_volume + ground_truth_object_volume - intersection
        iou_3d: float = intersection / union
        return iou_3d


def _get_volume_intersection(
    estimated_object: DynamicObject,
    ground_truth_object: DynamicObject,
) -> float:
    """Get the volume at intersected area.

    Args:
        estimated_object (DynamicObject): Estimated object.
        ground_truth_object (DynamicObject): Corresponding ground truth object.

    Returns:
        float: Volume at intersected area.
    """
    area_intersection = _get_area_intersection(estimated_object, ground_truth_object)
    height_intersection = _get_height_intersection(estimated_object, ground_truth_object)
    return area_intersection * height_intersection


def _get_height_intersection(
    estimated_object: DynamicObject,
    ground_truth_object: DynamicObject,
) -> float:
    """Get the height at intersection.

    Args:
        estimated_object (DynamicObject): Estimated object
        ground_truth_object (DynamicObject): Corresponding ground truth object.

    Returns:
        float: The height at intersection
    """
    min_z = max(
        estimated_object.state.position[2] - estimated_object.state.size[2] / 2,
        ground_truth_object.state.position[2] - ground_truth_object.state.size[2] / 2,
    )
    max_z = min(
        estimated_object.state.position[2] + estimated_object.state.size[2] / 2,
        ground_truth_object.state.position[2] + ground_truth_object.state.size[2] / 2,
    )
    return max(0, max_z - min_z)


def _get_area_intersection(estimated_object: ObjectType, ground_truth_object: ObjectType) -> float:
    """Get the area at intersection.

    This function allows either 3D or 2D object as inputs.
    For 3D object, get footprint of 3D box as polygon.
    For 2D object, get ROI of 2D box as a polygon.

    Args:
        estimated_object (ObjectType): Estimated object.
        ground_truth_object (ObjectType): Corresponding ground truth object.

    Returns:
        float: Area at intersection.
    """
    # estimated object footprint and Ground truth object footprint
    if isinstance(estimated_object, DynamicObject):
        pr_polygon: Polygon = estimated_object.get_footprint()
        gt_polygon: Polygon = ground_truth_object.get_footprint()
    else:
        pr_polygon: Polygon = estimated_object.get_polygon()
        gt_polygon: Polygon = ground_truth_object.get_polygon()
    area_intersection: float = pr_polygon.intersection(gt_polygon).area
    return area_intersection
