import math
from typing import Tuple

from shapely.geometry import Polygon


def distance_points(
    point_1: Tuple[float, float, float],
    point_2: Tuple[float, float, float],
) -> float:
    """[summary]
    Calculate the center distance between two points.
    Args:
        point_1 (Tuple[float]): A point
        point_2 (Tuple[float]): A point
    Returns: float: The distance between two points
    """
    if not (len(point_1) == 3 and len(point_2) == 3):
        raise RuntimeError(
            f"The length of a point is {len(point_1)} and {len(point_2)}, it needs 3."
        )
    return math.dist(point_1, point_2)


def distance_points_bev(
    point_1: Tuple[float, float, float],
    point_2: Tuple[float, float, float],
) -> float:
    """[summary]
    Calculate the 2d center distance between two points.
    Args:
        point_1 (Tuple[float]): A point
        point_2 (Tuple[float]): A point
    Returns: float: The distance between two points
    """
    if not (len(point_1) == 3 and len(point_2) == 3):
        raise RuntimeError(
            f"The length of a point is {len(point_1)} and {len(point_2)}, it needs 3."
        )
    return math.dist(to_bev(point_1), to_bev(point_2))


def to_bev(point_1: Tuple[float, float, float]) -> Tuple[float, float]:
    """[summary]
    (x, y, z) -> (x, y)
    Args:
         point_1 (Tuple[float, float, float]): A point
    Returns: Tuple[float, float]: The 2d point of point_1.
    """
    if not len(point_1) == 3:
        raise RuntimeError(f"The length of a point is {len(point_1)}, it needs 3.")
    return (point_1[0], point_1[1])


def polygon_to_list(polygon: Polygon):
    """[summary]
    Convert from polygon to list.
    from Polygon[(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x0, y0, z0)]
    to List[(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]

    Args:
        polygon (Polygon): Polygon

    Returns:
        [type]: List of coordinates
    """
    return list(set(polygon.exterior.coords))
