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

import math
from typing import Sequence

import numpy as np


def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    """[summary]
    Convert rotation matrix to euler angles.

    Args:
        rotation_matrix (np.ndarray): The 3x3 array of rotation matrix.

    Returns:
        np.ndarray: The 3D array, [roll, pitch, yaw].
    """
    assert _is_rotation_matrix(rotation_matrix)

    sy: float = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular: bool = sy < 1e-6

    if not singular:
        roll: float = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch: float = math.atan2(-rotation_matrix[2, 0], sy)
        yaw: float = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll: float = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch: float = math.atan2(-rotation_matrix[2, 0], sy)
        yaw: float = 0.0

    return np.array([roll, pitch, yaw])


def _is_rotation_matrix(rotation_matrix: np.ndarray) -> bool:
    """[summary]
    Check whether input matrix is rotation matrix.

    Args:
        rotation_matrix (np.ndarray): 3x3 array.

    Returns:
        bool: Whether input matrix is rotation matrix.
    """
    rot_mat_t = np.transpose(rotation_matrix)
    should_be_identity: np.ndarray = np.dot(rot_mat_t, rotation_matrix)
    identity: np.ndarray = np.identity(3, dtype=rotation_matrix.dtype)
    norm: float = np.linalg.norm(identity - should_be_identity)
    return norm < 1e-6


def get_bbox_scale(distance: float, box_scale_0m: float, box_scale_100m: float) -> float:
    """Calculate scale factor linearly for bounding box at specified distance.

    Note:
        scale = ((box_scale_100m - box_scale_0m) / (100 - 0)) * (distance - 0) + box_scale_0m

    Args:
        distance (float): The distance from vehicle to target bounding box.
        box_scale_0m (float): Scale factor for bbox at 0m.
        box_scale_100m (float): Scale factor for bbox at 100m.

    Returns:
        float: Calculated scale factor.
    """
    slope: float = 0.01 * (box_scale_100m - box_scale_0m)
    return slope * distance + box_scale_0m


def get_skew_matrix(arr: np.ndarray, use_tril: bool = True) -> np.ndarray:
    """Returns the skew-symmetric matrix.
    Args:
        arr (numpy.ndarray)
        use_tril (bool)
    Returns:
        numpy.ndarray
    """
    ret: np.ndarray = arr.copy()
    if use_tril:
        ret = np.tril(ret)
    else:
        ret = np.triu(ret)
    return ret - ret.T


def get_pose_transform_matrix(position: Sequence, rotation: np.ndarray) -> np.ndarray:
    """Returns 4x4 homogeneous transformation matrix for pose.
    Args:
        position (Sequence): In shape (3,)
        rotation (numpy.ndarray): In shape (3, 3).
    Returns:
        ret (numpy.ndarray)
    """
    if not isinstance(position, np.ndarray):
        position = np.array(position)

    ret: np.ndarray = np.eye(4)
    ret[:3, :3] = rotation
    ret[:3, 3] = position
    return ret


def get_velocity_transform_matrix(position: Sequence, rotation: np.ndarray) -> np.ndarray:
    """Returns 6x6 homogeneous transformation matrix for pose.
    Args:
        position (Sequence): In shape (3,)
        rotation (numpy.ndarray): In shape (3, 3).
    Returns:
        ret (numpy.ndarray): In shape (6, 6).
    """
    if not isinstance(position, np.ndarray):
        position = np.array(position)

    ret: np.ndarray = np.zeros((6, 6))
    ret[:3, :3] = rotation
    ret[3:, 3:] = rotation
    ret[3:, :3] = np.matmul(get_skew_matrix(position), rotation)
    return ret
