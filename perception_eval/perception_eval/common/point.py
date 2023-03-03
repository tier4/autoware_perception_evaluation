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

from typing import List
from typing import Tuple

import numpy as np
from shapely.geometry import Polygon


def distance_points(
    point_1: np.ndarray,
    point_2: np.ndarray,
) -> float:
    """Calculate the center distance between two points.
    Args:
        point_1 (numpy.ndarray[3,]): 3D point.
        point_2 (numpy.ndarray[3,]): 3D point.
    Returns: float: The distance between two points
    """
    if not (len(point_1) == 3 and len(point_2) == 3):
        raise RuntimeError(
            f"The length of a point is {len(point_1)} and {len(point_2)}, it needs 3."
        )
    vec: np.ndarray = np.array(point_1) - np.array(point_2)
    return np.linalg.norm(vec, ord=2, axis=0).item()


def distance_points_bev(
    point_1: np.ndarray,
    point_2: np.ndarray,
) -> float:
    """Calculate the 2d center distance between two points.
    Args:
        point_1 (numpy.ndarray[3,]): 3D point.
        point_2 (numpy.ndarray[3,]): 3D point.
    Returns:
        float: Distance between two points in BEV space.
    """
    if not (len(point_1) == 3 and len(point_2) == 3):
        raise RuntimeError(
            f"The length of a point is {len(point_1)} and {len(point_2)}, it needs 3."
        )
    p1_bev: np.ndarray = np.array(to_bev(point_1))
    p2_bev: np.ndarray = np.array(to_bev(point_2))
    vec: np.ndarray = p1_bev - p2_bev
    return np.linalg.norm(vec, ord=2, axis=0).item()


def to_bev(point_1: np.ndarray) -> np.ndarray:
    """(x, y, z) -> (x, y).
    Args:
        point_1 (np.ndarray): 3D point.
    Returns:
        np.ndarray: (x, y) point of input `point_1`.
    """
    if not len(point_1) == 3:
        raise RuntimeError(f"The length of a point is {len(point_1)}, it needs 3.")
    return point_1[:2]


def crop_pointcloud(
    pointcloud: np.ndarray,
    area: List[Tuple[float, float, float]],
    inside: bool = True,
) -> np.ndarray:
    """Crop pointcloud from (N, 3) to (M, 3) with Winding Number Algorithm.

    TODO:
        Implement to support the case area min/max height is not constant.

    Args:
        pointcloud (numpy.ndarray): The array of pointcloud, in shape (N, 3)
        area (List[Tuple[float, float, float]]): The 3D-polygon area to be cropped.
        inside (bool): Whether output inside pointcloud. Defaults to True.

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
    num_vertices = len(area) // 2
    cnt_arr_: np.ndarray = np.zeros(pointcloud.shape[0], dtype=np.uint8)
    for i in range(num_vertices):
        next_idx: int = (i + 1) % num_vertices
        incremental_flags = (area[i][1] <= pointcloud[:, 1]) * (
            area[next_idx][1] > pointcloud[:, 1]
        )
        decremental_flags = (area[i][1] > pointcloud[:, 1]) * (
            area[next_idx][1] <= pointcloud[:, 1]
        )

        if area[i + 1][1] != area[i][1]:
            vt = (pointcloud[:, 1] - area[i][1]) / (area[next_idx][1] - area[i][1])
        else:
            vt = pointcloud[:, 0]

        valid_idx = pointcloud[:, 0] < (area[i][0] + (vt * (area[next_idx][0] - area[i][0])))
        incremental_flags *= valid_idx
        decremental_flags *= valid_idx
        cnt_arr_[incremental_flags] += 1
        cnt_arr_[decremental_flags] -= 1

    if inside:
        xy_idx: np.ndarray = 0 < cnt_arr_
        z_idx: np.ndarray = (z_min <= pointcloud[:, 2]) * (pointcloud[:, 2] <= z_max)
    else:
        xy_idx: np.ndarray = cnt_arr_ <= 0
        z_idx: np.ndarray = ~((pointcloud[:, 2] < z_min) * (z_max < pointcloud[:, 2]))
    return pointcloud[xy_idx * z_idx]


def polygon_to_list(polygon: Polygon):
    """Convert polygon to list.

    Polygon: [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x0, y0, z0)]
    List: [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]

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
    """Examine the 2D geometric location of a point1 and a point2.
    Args:
        point_1 (Tuple[float, float, float]): A point
        point_2 (Tuple[float, float, float]): A point
    Returns:
        Tuple[Tuple[float, float, float]]: Returns [left_point, right_point]
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
