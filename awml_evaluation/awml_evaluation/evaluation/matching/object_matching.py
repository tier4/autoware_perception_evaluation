"""[summary]
This module has matching class.

Matching(
    predicted_object: DynamicObject,
    ground_truth_object: Optional[DynamicObject],
)
"""

from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
import math
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.object import distance_objects
from awml_evaluation.common.object import distance_points_bev
from awml_evaluation.common.point import get_point_left_right
from awml_evaluation.common.point import polygon_to_list
from shapely.geometry import Polygon


class MatchingMode(Enum):
    """[summary]
    The mode enum for matching algorithm.

    CENTERDISTANCE: center distance in 3d
    IOUBEV : IoU (Intersection over Union) in BEV (Bird Eye View)
    IOU3D : IoU (Intersection over Union) in 3D
    PLANEDISTANCE: The plane distance
    """

    CENTERDISTANCE = "Center Distance 3d [m]"
    IOUBEV = "IoU BEV"
    IOU3D = "IoU 3D"
    PLANEDISTANCE = "Plane Distance [m]"


class Matching(metaclass=ABCMeta):
    """[summary]
    Meta class for matching
    """

    @abstractmethod
    def __init__(
        self,
        predicted_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
    ) -> None:
        """[summary]
        Args:
            predicted_object (DynamicObject): The predicted obeject
            ground_truth_object (Optional[DynamicObject]): The ground truth object
        """
        self.mode: MatchingMode = MatchingMode.CENTERDISTANCE
        self.value: Optional[float] = None

    @abstractmethod
    def is_better_than(
        self,
        threshold_value: float,
    ) -> bool:
        """[summary]
        Judge whether value is better than threshold.

        Args:
            threshold_value (float): The threshold value

        Returns:
            bool: If value is better than threshold, return True.
        """
        pass


class CenterDistanceMatching(Matching):
    """[summary]
    Matching by center distance

    Attributes:
        self.mode (MatchingMode): Matching mode
        self.value (Optional[float]): Center distance
    """

    def __init__(
        self,
        predicted_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
    ) -> None:
        """[summary]
        Args:
            predicted_object (DynamicObject): The predicted object
            ground_truth_object (Optional[DynamicObject]): The ground truth object
        """
        self.mode: MatchingMode = MatchingMode.CENTERDISTANCE
        self.value: Optional[float] = self._get_center_distance(
            predicted_object,
            ground_truth_object,
        )

    def is_better_than(
        self,
        threshold_value: float,
    ) -> bool:
        """[summary]
        Judge whether value is better than threshold.

        Args:
            threshold_value (float): The threshold value

        Returns:
            bool: If value is better than threshold, return True.
        """
        if self.value is None:
            return False
        else:
            return self.value < threshold_value

    def _get_center_distance(
        self,
        predicted_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
    ) -> Optional[float]:
        """[summary]
        Get center distance
        Args:
            predicted_object (DynamicObject): The predicted object
            ground_truth_object (Optional[DynamicObject]): The ground truth object
        """
        if ground_truth_object is None:
            return None
        return distance_objects(predicted_object, ground_truth_object)


class PlaneDistanceMatching(metaclass=ABCMeta):
    """[summary]
    Matching by plane distance

    Attributes:
        self.mode (MatchingMode): Matching mode
        self.value (Optional[float]):
                Plane distance value [m].
                If predicted_object do not have corresponded ground truth object, value is None.
        self.ground_truth_nn_plane (Optional[Tuple[Tuple[float, float]]]):
                The nearest neighbor plane coordinate of ground truth object ((x1, y1), (x2, y2)).
                If predicted_object do not have corresponded ground truth object, value is None.
        self.predicted_nn_plane (Optional[Tuple[Tuple[float, float]]])]:
                The nearest neighbor plane coordinate of predicted object ((x1, y1), (x2, y2)).
                If predicted_object do not have corresponded ground truth object, value is None.
    """

    def __init__(
        self,
        predicted_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
    ) -> None:
        """[summary]
        Args:
            predicted_object (DynamicObject): The predicted object
            ground_truth_object (Optional[DynamicObject]): The ground truth object
        """
        self.mode: MatchingMode = MatchingMode.PLANEDISTANCE
        self.value: Optional[float] = None
        self.ground_truth_nn_plane: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = None
        self.predicted_nn_plane: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = None
        self.value, self.ground_truth_nn_plane, self.predicted_nn_plane = self._get_plane_distance(
            predicted_object,
            ground_truth_object,
        )

    def is_better_than(
        self,
        threshold_value: float,
    ) -> bool:
        """[summary]
        Judge whether value is better than threshold.

        Args:
            threshold_value (float): The threshold value

        Returns:
            bool: If value is better than threshold, return True.
        """
        if self.value is None:
            return False
        else:
            return self.value < threshold_value

    def _get_plane_distance(
        self,
        predicted_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
    ) -> Tuple[
        Optional[float],
        Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
        Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    ]:

        """[summary]
        Calculate plane distance for use case evaluation.

        Args:
            predicted_object (DynamicObject): A predicted object
            ground_truth_object (Optional[DynamicObject]): The correspond ground truth object

        Returns:
            Tuple[value, ground_truth_nn_plane, predicted_nn_plane]
            See class attribute in detail
        """
        if ground_truth_object is None:
            return None, None, None

        # Get corner_points of predicted object from footprint
        pr_footprint_polygon: Polygon = predicted_object.get_footprint()
        pr_corner_points: List[Tuple[float, float, float]] = polygon_to_list(pr_footprint_polygon)

        # Get corner_points of ground truth object from footprint
        gt_footprint_polygon: Polygon = ground_truth_object.get_footprint()
        gt_corner_points: List[Tuple[float, float, float]] = polygon_to_list(gt_footprint_polygon)

        # Sort by 2d distance
        lambda_func: Callable[[Tuple[float, float, float]], float] = lambda x: math.hypot(
            x[0], x[1]
        )
        pr_corner_points.sort(key=lambda_func)
        gt_corner_points.sort(key=lambda_func)

        pr_left_point, pr_right_point = get_point_left_right(
            pr_corner_points[0], pr_corner_points[1]
        )
        gt_left_point, gt_right_point = get_point_left_right(
            gt_corner_points[0], gt_corner_points[1]
        )

        # Calculate plane distance
        distance_left_point: float = abs(distance_points_bev(pr_left_point, gt_left_point))
        distance_right_point: float = abs(distance_points_bev(pr_right_point, gt_right_point))
        distance_squared: float = distance_left_point**2 + distance_right_point**2
        plane_distance: float = math.sqrt(distance_squared / 2.0)

        ground_truth_nn_plane: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
            gt_left_point,
            gt_right_point,
        )
        predicted_nn_plane: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
            pr_left_point,
            pr_right_point,
        )
        return plane_distance, ground_truth_nn_plane, predicted_nn_plane


class IOUBEVMatching(metaclass=ABCMeta):
    """[summary]
    Matching by IoU BEV

    Attributes:
        self.mode (MatchingMode): Matching mode
        self.value (Optional[float]): IoU BEV
    """

    def __init__(
        self,
        predicted_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
    ) -> None:
        """[summary]
        Args:
            predicted_object (DynamicObject): The predicted object
            ground_truth_object (Optional[DynamicObject]): The ground truth object
        """
        self.mode: MatchingMode = MatchingMode.IOUBEV
        self.value: Optional[float] = self._get_iou_bev(
            predicted_object,
            ground_truth_object,
        )

    def is_better_than(
        self,
        threshold_value: float,
    ) -> bool:
        """[summary]
        Judge whether value is better than threshold.

        Args:
            threshold_value (float): The threshold value

        Returns:
            bool: If value is better than threshold, return True.
        """
        if self.value is None:
            return False
        else:
            return self.value > threshold_value

    def _get_iou_bev(
        self,
        predicted_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
    ) -> float:
        """[summary]
        Calculate BEV IoU

        Args:
            predicted_object (DynamicObject): The predicted object
            ground_truth_object (DynamicObject): The corresponded ground truth object

        Returns:
            Optional[float]: The value of BEV IoU.
                            If predicted_object do not have corresponded ground truth object,
                            return 0.0.
        Reference:
            https://github.com/lyft/nuscenes-devkit/blob/49c36da0a85da6bc9e8f2a39d5d967311cd75069/lyft_dataset_sdk/eval/detection/mAP_evaluation.py
        """

        if ground_truth_object is None:
            return 0.0

        # TODO: if tiny box dim seen return 0.0 IOU
        predicted_object_area: float = predicted_object.get_area_bev()
        ground_truth_object_area: float = ground_truth_object.get_area_bev()
        intersection_area: float = _get_area_intersection(predicted_object, ground_truth_object)
        union_area: float = predicted_object_area + ground_truth_object_area - intersection_area
        iou_bev: float = intersection_area / union_area
        return iou_bev


class IOU3dMatching(metaclass=ABCMeta):
    """[summary]
    Matching by IoU 3d

    Attributes:
        self.mode (MatchingMode): Matching mode
        self.value (Optional[float]): IoU 3d
    """

    def __init__(
        self,
        predicted_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
    ) -> None:
        """[summary]
        Args:
            predicted_object (DynamicObject): The predicted object
            ground_truth_object (Optional[DynamicObject]): The ground truth object
        """
        self.mode: MatchingMode = MatchingMode.IOU3D
        self.value: Optional[float] = self._get_iou_3d(
            predicted_object,
            ground_truth_object,
        )

    def is_better_than(
        self,
        threshold_value: float,
    ) -> bool:
        """[summary]
        Judge whether value is better than threshold.

        Args:
            threshold_value (float): The threshold value

        Returns:
            bool: If value is better than threshold, return True.
        """
        if self.value is None:
            return False
        else:
            return self.value > threshold_value

    def _get_iou_3d(
        self,
        predicted_object: DynamicObject,
        ground_truth_object: Optional[DynamicObject],
    ) -> float:
        """[summary]
        Calculate 3D IoU

        Args:
            predicted_object (DynamicObject): The predicted object
            ground_truth_object (DynamicObject): The corresponded ground truth object

        Returns:
            Optional[float]: The value of 3D IoU.
                            If predicted_object do not have corresponded ground truth object,
                            return 0.0.
        """
        if ground_truth_object is None:
            return 0.0

        predicted_object_volume: float = predicted_object.get_volume()
        ground_truth_object_volume: float = ground_truth_object.get_volume()
        intersection: float = _get_volume_intersection(predicted_object, ground_truth_object)
        union: float = predicted_object_volume + ground_truth_object_volume - intersection
        iou_3d: float = intersection / union
        return iou_3d


def _get_volume_intersection(
    predicted_object: DynamicObject,
    ground_truth_object: DynamicObject,
) -> float:
    """[summary]
    Get the volume at intersection

    Args:
        predicted_object (DynamicObject): The predicted object
        ground_truth_object (DynamicObject): The corresponded ground truth object

    Returns:
        float: The volume at intersection

    """
    area_intersection = _get_area_intersection(predicted_object, ground_truth_object)
    height_intersection = _get_height_intersection(predicted_object, ground_truth_object)
    return area_intersection * height_intersection


def _get_height_intersection(
    predicted_object: DynamicObject,
    ground_truth_object: DynamicObject,
) -> float:
    """[summary]
    Get the height at intersection

    Args:
        predicted_object (DynamicObject): The predicted object
        ground_truth_object (DynamicObject): The corresponded ground truth object

    Returns:
        float: The height at intersection

    """
    min_z = max(
        predicted_object.state.position[2] - predicted_object.state.size[2] / 2,
        ground_truth_object.state.position[2] - ground_truth_object.state.size[2] / 2,
    )
    max_z = min(
        predicted_object.state.position[2] + predicted_object.state.size[2] / 2,
        ground_truth_object.state.position[2] + ground_truth_object.state.size[2] / 2,
    )
    return max(0, max_z - min_z)


def _get_area_intersection(
    predicted_object: DynamicObject,
    ground_truth_object: DynamicObject,
) -> float:
    """[summary]
    Get the area at intersection

    Args:
        predicted_object (DynamicObject): The predicted object
        ground_truth_object (DynamicObject): The corresponded ground truth object

    Returns:
        float: The area at intersection
    """
    # Predicted object footprint and Ground truth object footprint
    pr_footprint_polygon: Polygon = predicted_object.get_footprint()
    gt_footprint_polygon: Polygon = ground_truth_object.get_footprint()
    area_intersection: float = pr_footprint_polygon.intersection(gt_footprint_polygon).area
    return area_intersection
