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

from copy import deepcopy
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.common.object2d import Roi
from perception_eval.common.object import DynamicObject
from perception_eval.common.object import ObjectState
from perception_eval.common.point import distance_points
from perception_eval.common.point import distance_points_bev
from pyquaternion import Quaternion

# Type aliases
ObjectType = Union[DynamicObject, DynamicObject2D]


def interpolate_hopmogeneous_matrix(
    matrix_1: np.ndarray, matrix_2: np.ndarray, t1: float, t2: float, t: float
) -> np.ndarray:
    """[summary]
    Interpolate the state of object_1 to the time of object_2.
    Args:
            matrix_1 (np.ndarray): Homogeneous matrix
            matrix_2 (np.ndarray): Homogeneous matrix
    Returns: np.ndarray: The interpolated state.
    """
    assert t1 <= t <= t2
    assert matrix_1.shape == matrix_2.shape
    assert matrix_1.shape == (4, 4)
    R1 = matrix_1[:3, :3]
    R2 = matrix_2[:3, :3]
    T1 = matrix_1[:3, 3]
    T2 = matrix_2[:3, 3]
    # interpolation
    T = T1 + (T2 - T1) * (t - t1) / (t2 - t1)
    q1 = Quaternion(matrix=R1)
    q2 = Quaternion(matrix=R2)
    q = Quaternion.slerp(q1, q2, (t - t1) / (t2 - t1))
    R = q.rotation_matrix
    # put them together
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = T
    return matrix


def interpolate_list(list_1: List[float], list_2: List[float], t1: float, t2: float, t: float) -> List[float]:
    """[summary]
    Interpolate the state of object_1 to the time of object_2.
    Args:
         list_1 (List[float]): An object
         list_2 (List[float]): An object
    Returns: List[float]: The interpolated state.
    """
    assert t1 <= t <= t2
    assert len(list_1) == len(list_2)
    state = []
    for i in range(len(list_1)):
        state.append(list_1[i] + (list_2[i] - list_1[i]) * (t - t1) / (t2 - t1))
    return state


def interpolate_quaternion(quat_1: Quaternion, quat_2: Quaternion, t1: float, t2: float, t: float) -> Quaternion:
    """Interpolate a quaternion between two given times to a specific time."""
    assert t1 <= t <= t2
    alpha = (t - t1) / (t2 - t1)
    interpolated_quat = quat_1.slerp(quat_1, quat_2, alpha)
    return interpolated_quat


def interpolate_state(state_1: ObjectState, state_2: ObjectState, t1: float, t2: float, t: float) -> ObjectState:
    """[summary]
    Interpolate the state of object_1 to the time of object_2.
    Args:
         state_1 (np.ndarray): An object
         state_2 (np.ndarray): An object
    Returns: np.ndarray: The interpolated state.
    """
    assert t1 <= t <= t2
    # state has position, Orientation, shape, velocity
    interp_position = tuple(interpolate_list(state_1.position, state_2.position, t1, t2, t))
    interp_orientation = interpolate_quaternion(state_1.orientation, state_2.orientation, t1, t2, t)
    interp_shape = state_1.shape  # shape will not change
    interp_velocity = tuple(interpolate_list(state_1.velocity, state_2.velocity, t1, t2, t))
    return ObjectState(
        position=interp_position, orientation=interp_orientation, shape=interp_shape, velocity=interp_velocity
    )


def interpolate_object_list(
    object_list1: List[ObjectType], object_list2: List[ObjectType], t1: float, t2: float, t: float
) -> List[ObjectType]:
    """[summary]
    Interpolate object list from time t1 to time t2 to time t.

    Args:
        object_list1 (List[ObjectType]): _description_
        object_list2 (List[ObjectType]): _description_
        t1 (float): _description_
        t2 (float): _description_
        t (float): _description_

    Returns:
        List[ObjectType]: _description_
    """
    assert t1 <= t <= t2
    output_object_list = []
    id_list = []
    for object1 in object_list1:
        found: bool = False
        for object2 in object_list2:
            if object1.uuid == object2.uuid:
                output_object_list.append(interpolate_object(object1, object2, t1, t2, t))
                id_list.append(object1.uuid)
                found = True
                break
        # not found in object_list2
        if not found:
            output_object_list.append(deepcopy(object1))
            id_list.append(object1.uuid)

    for object2 in object_list2:
        if object2.uuid not in id_list:
            output_object_list.append(deepcopy(object2))
            id_list.append(object2.uuid)

    return output_object_list


def interpolate_object(object_1: ObjectType, object_2: ObjectType, t1: float, t2: float, t: float) -> ObjectType:
    """[summary]
    Interpolate the state of object_1 to the time of object_2.
    Args:
         object_1 (ObjectType): An object
         object_2 (ObjectType): An object
    Returns: ObjectType: The interpolated object.
    """
    if type(object_1) != type(object_2):
        raise TypeError(f"objects' type must be same, but got {type(object_1) and {type(object_2)}}")

    if isinstance(object_1, DynamicObject):
        return interpolate_dynamicobject(object_1, object_2, t1, t2, t)
    elif isinstance(object_1, DynamicObject2D):
        return interpolate_dynamicobject2d(object_1, object_2, t1, t2, t)
    else:
        raise TypeError(f"object type must be DynamicObject or DynamicObject2D, but got {type(object_1)}")


def interpolate_dynamicobject(
    object_1: DynamicObject, object_2: DynamicObject, t1: float, t2: float, t: float
) -> DynamicObject:
    """[summary]
    Interpolate the state of object_1 to the time of object_2.
    Args:
         object_1 (DynamicObject): An object
         object_2 (DynamicObject): An object
    Returns: DynamicObject: The interpolated object.
    """
    assert t1 <= t <= t2
    assert object_1.uuid == object_2.uuid
    # 面倒なので基本的にcopyで済ます
    # TODO: 他の要素も補間する
    output_object = deepcopy(object_1)
    interp_state = interpolate_state(object_1.state, object_2.state, t1, t2, t)
    output_object.state = interp_state
    output_object.unix_time = int(t)
    return output_object


def interpolate_dynamicobject2d(
    object_1: DynamicObject2D, object_2: DynamicObject2D, t1: float, t2: float, t: float
) -> DynamicObject2D:
    """[summary]
    Interpolate the state of object_1 to the time of object_2.
    Args:
         object_1 (DynamicObject2D): An object
         object_2 (DynamicObject2D): An object
    Returns: DynamicObject2D: The interpolated object.
    """
    # TODO: implement
    pass
