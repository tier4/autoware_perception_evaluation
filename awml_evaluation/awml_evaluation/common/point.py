import math
from typing import Tuple


def distance_points(point_1: Tuple[float], point_2: Tuple[float]) -> float:
    """[summary]
    Calculate the center distance between two points.
    Args:
        point_1 (Tuple[float]): A point
        point_2 (Tuple[float]): A point
    Returns: float: The distance between two points
    """
    return math.dist(point_1, point_2)


def distance_points_bev(point_1: Tuple[float], point_2: Tuple[float]) -> float:
    """[summary]
    Calculate the 2d center distance between two points.
    Args:
        point_1 (Tuple[float]): A point
        point_2 (Tuple[float]): A point
    Returns: float: The distance between two points
    """
    return math.dist(to_bev(point_1), to_bev(point_2))


def to_bev(point_1: Tuple[float, float, float]) -> Tuple[float, float]:
    """[summary]
    (x, y, z) -> (x, y)
    Args:
         point_1 (Tuple[float, float, float]): A point
    Returns: Tuple[float, float]: The 2d point of point_1.
    """
    return (point_1[0], point_1[1])
