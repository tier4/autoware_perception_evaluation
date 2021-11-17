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

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.object import distance_points_2d


class MatchingMode(Enum):
    """[summary]
    The mode enum for matching algorithm.

    CENTERDISTANCE: The center distance
    IOU3d : 3d IoU (Intersection over Union)
    PLANEDISTANCE = The plane distance
    """

    CENTERDISTANCE = "center_distance"
    IOU3d = "iou_3d"
    PLANEDISTANCE = "plane_distance"


def get_uc_plane_distance(
    predicted_object: DynamicObject,
    ground_truth_object: Optional[DynamicObject],
) -> Optional[float]:
    """[summary]
    Calculate use case plane distance

    Returns:
        Optional[float]: [description]
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
    distance_1_1: float = abs(distance_points_2d(pr_footprint[0], gt_footprint[0]))
    distance_1_2: float = abs(distance_points_2d(pr_footprint[1], gt_footprint[1]))
    distance_1: float = distance_1_1 + distance_1_2
    distance_2_1: float = abs(distance_points_2d(pr_footprint[0], gt_footprint[1]))
    distance_2_2: float = abs(distance_points_2d(pr_footprint[1], gt_footprint[0]))
    distance_2: float = distance_2_1 + distance_2_2
    uc_plane_distance: float = min(distance_1, distance_2) / 2.0
    # logger.info(f"Distance {uc_plane_distance}")

    return uc_plane_distance


def get_iou_3d(
    predicted_object: DynamicObject,
    ground_truth_object: Optional[DynamicObject],
) -> float:
    """[summary]
    Calculate 3d IoU

    Args:
        predicted_object (DynamicObject): The predicted object
        ground_truth_object (DynamicObject): The corresponded ground truth object

    Returns:
        Optional[float]: The value of 3d IoU.
                         If predicted_object do not have corresponded ground truth object,
                         return 0.0.
    """

    # TODO impl
    if not ground_truth_object:
        return 0.0
    return 0.0
