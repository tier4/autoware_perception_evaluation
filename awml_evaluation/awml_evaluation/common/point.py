import math
from typing import List
from typing import Tuple

import numpy as np
from shapely.geometry import Polygon


def distance_points(
    point_1: np.ndarray,
    point_2: np.ndarray,
) -> float:
    """[summary]
    Calculate the center distance between two points.
    Args:
        point_1 (numpy.ndarray[3,]): A point
        point_2 (numpy.ndarray[3,]): A point
    Returns: float: The distance between two points
    """
    if not (len(point_1) == 3 and len(point_2) == 3):
        raise RuntimeError(
            f"The length of a point is {len(point_1)} and {len(point_2)}, it needs 3."
        )
    return math.dist(point_1, point_2)


def distance_points_bev(
    point_1: np.ndarray,
    point_2: np.ndarray,
) -> float:
    """[summary]
    Calculate the 2d center distance between two points.
    Args:
        point_1 (numpy.ndarray[3,]): A point
        point_2 (numpy.ndarray[3,]): A point
    Returns: float: The distance between two points
    """
    if not (len(point_1) == 3 and len(point_2) == 3):
        raise RuntimeError(
            f"The length of a point is {len(point_1)} and {len(point_2)}, it needs 3."
        )
    return math.dist(to_bev(point_1), to_bev(point_2))


def to_bev(point_1: np.ndarray) -> np.ndarray:
    """[summary]
    (x, y, z) -> (x, y)
    Args:
         point_1 (numpy.ndarray[3,]): A point
    Returns: numpy.ndarray[2,]: The 2d point of point_1.
    """
    if not len(point_1) == 3:
        raise RuntimeError(f"The length of a point is {len(point_1)}, it needs 3.")
    return point_1[:2]


def crop_pointcloud(
    pointcloud: np.ndarray,
    area: List[Tuple[float, float, float]],
) -> np.ndarray:
    """Crop pointcloud from (N, 3) to (M, 3) with Crossing Number Algorithm.

    TODO:
        Implement to support the case area min/max height is not constant.

    Args:
        pointcloud (numpy.ndarray): The array of pointcloud, in shape (N, 3)
        area (List[Tuple[float, float, float]]): The 3D-polygon area to be cropped

    Returns:
        numpy.ndarray: The  of cropped pointcloud, in shape (M, 3)
    """
    if pointcloud.ndim != 2 or pointcloud.shape[1] < 2:
        raise RuntimeError(
            f"The shape of pointcloud is {pointcloud.shape}, it needs (N, k>=2) and k is (x,y,z,i) order."
        )
    if len(area) < 6 or len(area) % 2 != 0:
        raise RuntimeError(
            f"The area must be a 3D-polygon, it needs the edges more than 6, but got {len(area)}."
        )

    z_min: float = min(area, key=(lambda x: x[2]))[2]
    z_max: float = max(area, key=(lambda x: x[2]))[2]

    # crop with polygon in xy-plane
    num_vertices = len(area)
    cnt_arr_: np.ndarray = np.zeros(pointcloud.shape[0], dtype=np.uint8)
    for i in range(num_vertices // 2 - 1):
        flags_ = ((area[i][1] <= pointcloud[:, 1]) * (area[i + 1][1] > pointcloud[:, 1])) + (
            (area[i][1] > pointcloud[:, 1]) * (area[i + 1][1] <= pointcloud[:, 1])
        )

        if area[i + 1][1] != area[i][1]:
            vt = (pointcloud[:, 1] - area[i][1]) / (area[i + 1][1] - area[i][1])
        else:
            vt = pointcloud[:, 0]

        flags_ *= pointcloud[:, 0] < (area[i][0] + (vt * (area[i + 1][0] - area[i][0])))
        cnt_arr_[flags_] += 1

    xy_cropped: np.ndarray = pointcloud[cnt_arr_ % 2 != 0]

    return xy_cropped[(z_min <= xy_cropped[:, 2]) * (z_max >= xy_cropped[:, 2])]


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
    return list(polygon.exterior.coords)[:4]


def get_point_left_right(
    point_1: Tuple[float, float, float],
    point_2: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float]]:
    """[summary]
    Examine the 2D geometric location of a point1 and a point2.
    Args:
        point_1 (Tuple[float, float, float]): A point
        point_2 (Tuple[float, float, float]): A point
    Returns: Tuple[Tuple[float, float, float]]: Returns [left_point, right_point]
    """
    if not (len(point_1) > 2 and len(point_2) > 2):
        raise RuntimeError(
            f"The length of a point is {len(point_1)} and {len(point_2)}, they must be greater than 2."
        )
    cross_product = point_1[0] * point_2[1] - point_1[1] * point_2[0]
    if cross_product < 0:
        return (point_1, point_2)
    else:
        return (point_2, point_1)
