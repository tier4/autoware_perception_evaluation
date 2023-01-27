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

from typing import Union

import numpy as np
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.common.object import DynamicObject
from perception_eval.common.point import distance_points
from perception_eval.common.point import distance_points_bev

# Type aliases
ObjectType = Union[DynamicObject, DynamicObject2D]


def distance_objects(object_1: ObjectType, object_2: ObjectType) -> float:
    """[summary]
    Calculate the 3D/2D center distance between two objects.
    Args:
         object_1 (ObjectType): An object
         object_2 (ObjectType): An object
    Returns: float: The center distance between object_1 and object_2.
    """
    if type(object_1) != type(object_2):
        raise TypeError(
            f"objects' type must be same, but got {type(object_1) and {type(object_2)}}"
        )

    if isinstance(object_1, DynamicObject):
        return distance_points(object_1.state.position, object_2.state.position)
    return np.linalg.norm(np.array(object_1.roi.center) - np.array(object_2.roi.center))


def distance_objects_bev(object_1: DynamicObject, object_2: DynamicObject) -> float:
    """[summary]
    Calculate the BEV 2d center distance between two objects.
    Args:
         object_1 (DynamicObject): An object
         object_2 (DynamicObject): An object
    Returns: float: The 2d center distance from object_1 to object_2.
    """
    assert isinstance(object_1, DynamicObject) and isinstance(object_2, DynamicObject)
    return distance_points_bev(object_1.state.position, object_2.state.position)
