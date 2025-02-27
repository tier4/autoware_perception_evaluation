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

from __future__ import annotations

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Generator

import numpy as np
from perception_eval.common.shape import Shape
from perception_eval.common.shape import ShapeType
from pyquaternion import Quaternion
from shapely.geometry import Polygon


class ObjectState:
    """Object state class."""

    def __init__(
        self,
        position: Tuple[float, float, float],
        orientation: Quaternion,
        shape: Optional[Shape] = None,
        velocity: Optional[Tuple[float, float, float]] = None,
        pose_covariance: Optional[np.ndarray] = None,
        twist_covariance: Optional[np.ndarray] = None,
    ) -> None:
        """

        Args:
            position (Tuple[float, float, float]) : (center_x, center_y, center_z) [m]
            orientation (Quaternion) : Quaternion instance.
            shape (Optional[Shape]): Shape instance. Defaults to None.
            velocity (Optional[Tuple[float, float, float]]): velocity of (vx, vy, vz) [m/s]. Defaults to None.
             pose_covariance (Optional[np.ndarray]): Covariance matrix for pose (x, y, z, roll, pitch, yaw).
                Defaults to None.
            twist_covariance (Optional[np.ndarray]): Covariance matrix for twist (x, y, z, roll, pitch, yaw).
                Defaults to None.
        """
        self.position = position
        self.orientation = orientation
        self.shape = shape
        self.velocity = velocity
        self.pose_covariance = pose_covariance
        self.twist_covariance = twist_covariance

    @property
    def shape_type(self) -> Optional[ShapeType]:
        return self.shape.type if self.shape is not None else None

    @property
    def size(self) -> Optional[tuple[float, float, float]]:
        return self.shape.size if self.shape is not None else None

    @property
    def footprint(self) -> Polygon:
        return self.shape.footprint if self.shape is not None else None

    @property
    def has_pose_covariance(self) -> bool:
        return self.pose_covariance is not None

    @property
    def has_twist_covariance(self) -> bool:
        return self.twist_covariance is not None

    def get_position_error(self, other: ObjectState) -> Tuple[float, float, float]:
        """Returns the position error between other and itself.

        Args:
            other (ObjectState): Other state.

        Returns:
            Tuple[float, float, float]: Position error (x, y, z).
        """
        return (
            other.position[0] - self.position[0],
            other.position[1] - self.position[1],
            other.position[2] - self.position[2],
        )

    def get_size_error(self, other: ObjectState) -> Tuple[float, float, float]:
        """Returns the size error between other and itself.

        Args:
            other (ObjectState): Other state.

        Returns:
            Tuple[float, float, float]: Size error (w, l, h).
        """
        return (
            other.size[0] - self.size[0],
            other.size[1] - self.size[1],
            other.size[2] - self.size[2],
        )

    def get_velocity_error(self, other: ObjectState) -> Tuple[float, float, float]:
        """Returns the velocity error between other and itself.

        Args:
            other (ObjectState): Other state.

        Returns:
            Tuple[float, float, float]: Velocity error (vx, vy, vz).
        """
        return (
            other.velocity[0] - self.velocity[0],
            other.velocity[1] - self.velocity[1],
            other.velocity[2] - self.velocity[2],
        )


class ObjectPath:
    """Object path class.

    You can access corresponding state with its index as shown below.
    ```
    >>> path = ObjectPath(states, confidence)
    >>> for idx in range(len(path)):
            state = path[idx]
    ```

    Attributes:
        states (List[ObjectState]): List of ObjectState instances.
        confidence (float): Path confidence in [0, 1].

    Args:
        states (List[ObjectState]): List of ObjectState instances.
        confidence (float): Path confidence in [0, 1].
    """

    def __init__(self, states: List[ObjectState], confidence: float) -> None:
        self.states: List[ObjectState] = states
        self.confidence: float = confidence

    def __len__(self) -> int:
        """Returns length of states.

        Returns:
            int: length of states.
        """
        return len(self.states)
    
    def get_path_error(self, other: ObjectPath) -> np.ndarray:
        """Return the displacement error at each waypoint of the path.

        Args:
            other (ObjectPath): Other path.

        Returns:
            np.ndarray: Displacement error array in the shape of (T, 3).
        """
        min_length = min(len(self), len(other))
        self_states = self.states[:min_length]
        other_states = other.states[:min_length]
        return np.array([self_s.get_position_error(other_s) for self_s, other_s in zip(self_states, other_states)]).reshape(-1, 3)


def set_object_states(
    positions: Optional[List[Tuple[float, float, float]]] = None,
    orientations: Optional[List[Quaternion]] = None,
    shapes: Optional[List[Shape]] = None,
    velocities: Optional[List[Tuple[float, float, float]]] = None,
) -> Optional[List[ObjectState]]:
    """[summary]
    Set list of object states.

    Args:
        positions (Optional[List[Tuple[float]]])
        orientations (Optional[List[Tuple[float]]])
        shapes (Optional[List[Shape]])
        velocities (Optional[List[Tuple[float]]])

    Returns:
        Optional[List[ObjectState]]
    """
    if positions is None or orientations is None:
        return None

    if len(positions) != len(orientations):
        raise RuntimeError(
            "length of positions and orientations must be same, " f"but got {len(positions)} and {len(orientations)}"
        )

    return [
        ObjectState(
            position=pos,
            orientation=orient,
            shape=shapes[i] if shapes else None,
            velocity=velocities[i] if velocities else None,
        )
        for i, (pos, orient) in enumerate(zip(positions, orientations))
    ]


def set_object_path(
    positions: Optional[List[Tuple[float, float, float]]] = None,
    orientations: Optional[List[Quaternion]] = None,
    velocities: Optional[List[Tuple[float, float, float]]] = None,
    confidence: Optional[float] = None,
) -> ObjectPath:
    """[summary]
    Set single path.
    """
    states: Optional[List[ObjectState]] = set_object_states(
        positions=positions,
        orientations=orientations,
        velocities=velocities,
    )
    return ObjectPath(states, confidence) if states else None


def set_object_paths(
    positions: Optional[List[List[Tuple[float, float, float]]]] = None,
    orientations: Optional[List[List[Quaternion]]] = None,
    twists: Optional[List[List[Tuple[float, float, float]]]] = None,
    confidences: Optional[List[float]] = None,
) -> Optional[List[ObjectPath]]:
    """[summary]
    Set multiple paths.
    """
    if positions is None or orientations is None or confidences is None:
        return None

    if len(positions) != len(orientations) != len(confidences):
        raise RuntimeError(
            "length of positions and orientations must be same, " f"but got {len(positions)} and {len(orientations)}"
        )

    paths: List[ObjectPath] = []
    for i, (poses, orients, confidence) in enumerate(zip(positions, orientations, confidences)):
        path = set_object_path(
            positions=poses,
            orientations=orients,
            velocities=twists[i] if twists else None,
            confidence=confidence,
        )
        if path:
            paths.append(path)
    return paths
