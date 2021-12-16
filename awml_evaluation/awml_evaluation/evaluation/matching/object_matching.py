"""[summary]
This module has matching function.

function(
    object_1: DynamicObject,
    object_2: DynamicObject,
) -> Any

"""

from enum import Enum
import math
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

from shapely.geometry import Polygon

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.object import distance_points_bev


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


def get_uc_plane_distance(
    predicted_object: DynamicObject,
    ground_truth_object: Optional[DynamicObject],
) -> Optional[float]:
    """[summary]
    Calculate plane distance for use case evaluation.

    Args:
        predicted_object (DynamicObject): A predicted object
        ground_truth_object (Optional[DynamicObject]): The correspond ground truth object

    Returns:
        Optional[float]: The value of plane distance.
                         If predicted_object do not have corresponded ground truth object,
                         return None.
    """
    if not ground_truth_object:
        return None

    # Predicted object footprint and Ground truth object footprint
    pr_footprint: List[Tuple[float]] = predicted_object.get_footprint()
    gt_footprint: List[Tuple[float]] = ground_truth_object.get_footprint()

    # Sort by 2d distance
    lambda_func: Callable[[Tuple[float]], float] = lambda x: math.hypot(x[0], x[1])
    pr_footprint.sort(key=lambda_func)
    gt_footprint.sort(key=lambda_func)

    # Calculate plane distance
    distance_1_1: float = abs(distance_points_bev(pr_footprint[0], gt_footprint[0]))
    distance_1_2: float = abs(distance_points_bev(pr_footprint[1], gt_footprint[1]))
    distance_1: float = distance_1_1 + distance_1_2
    distance_2_1: float = abs(distance_points_bev(pr_footprint[0], gt_footprint[1]))
    distance_2_2: float = abs(distance_points_bev(pr_footprint[1], gt_footprint[0]))
    distance_2: float = distance_2_1 + distance_2_2
    uc_plane_distance: float = min(distance_1, distance_2) / 2.0
    # logger.info(f"Distance {uc_plane_distance}")

    return uc_plane_distance


def get_area_intersection(
    predicted_object: DynamicObject,
    ground_truth_object: Optional[DynamicObject],
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
    pr_footprint_polygon: Polygon = predicted_object.get_footprint_polygon()
    gt_footprint_polygon: Polygon = ground_truth_object.get_footprint_polygon()
    area_intersection: float = pr_footprint_polygon.intersection(gt_footprint_polygon).area
    return area_intersection


def get_height_intersection(
    predicted_object: DynamicObject,
    ground_truth_object: Optional[DynamicObject],
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


def get_intersection(
    predicted_object: DynamicObject,
    ground_truth_object: Optional[DynamicObject],
) -> float:
    """[summary]
    Get the volume at intersection

    Args:
        predicted_object (DynamicObject): The predicted object
        ground_truth_object (DynamicObject): The corresponded ground truth object

    Returns:
        float: The volume at intersection

    """
    area_intersection = get_area_intersection(predicted_object, ground_truth_object)
    height_intersection = get_height_intersection(predicted_object, ground_truth_object)
    return area_intersection * height_intersection


def get_iou_bev(
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

    if not ground_truth_object:
        return 0.0

    # TODO: if tiny box dim seen return 0.0 IOU
    predicted_object_area: float = predicted_object.area_bev
    ground_truth_object_area: float = ground_truth_object.area_bev
    intersection_area: float = get_area_intersection(predicted_object, ground_truth_object)
    union_area: float = predicted_object_area + ground_truth_object_area - intersection_area
    iou_bev: float = intersection_area / union_area
    return iou_bev


def get_iou_3d(
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
    if not ground_truth_object:
        return 0.0

    predicted_object_volume: float = predicted_object.volume
    ground_truth_object_volume: float = ground_truth_object.volume
    intersection: float = get_intersection(predicted_object, ground_truth_object)
    union: float = predicted_object_volume + ground_truth_object_volume - intersection
    iou_3d: float = intersection / union
    return iou_3d
