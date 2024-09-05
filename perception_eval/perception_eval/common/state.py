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

from pyquaternion import Quaternion


class ObjectState:
    """Object state class.

    Attributes:
        position (Tuple[float, float, float]) : center_x, center_y, center_z [m]
        orientation (Quaternion) : Quaternion class.
            See reference http://kieranwynn.github.io/pyquaternion/
        size (Tuple[float, float, float]): bounding box size of (wx, wy, wz) [m]
        velocity (Tuple[float, float, float]): velocity of (vx, vy, vz) [m/s]

    Args:
        position (Tuple[float, float, float]) : center_x, center_y, center_z [m]
        orientation (Quaternion) : Quaternion class.
            See reference http://kieranwynn.github.io/pyquaternion/
        size (Tuple[float, float, float]): bounding box size of (wx, wy, wz) [m]
        velocity (Tuple[float, float, float]): velocity of (vx, vy, vz) [m/s]
    """

    def __init__(
        self,
        position: Tuple[float, float, float],
        orientation: Quaternion,
        size: Optional[Tuple[float, float, float]] = None,
        velocity: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.position: Tuple[float, float, float] = position
        self.orientation: Quaternion = orientation
        self.size: Tuple[float, float, float] = size
        self.velocity: Tuple[float, float, float] = velocity

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

    def __getitem__(self, idx: int) -> Union[ObjectState, List[ObjectState]]:
        """Returns Nth state.

        Args:
            idx (int): Index.

        Returns:
            Union[ObjectState, List[ObjectState]]
        """
        return self.states[idx]

    def __iter__(self) -> ObjectPath:
        self.__n = 0
        return self

    def __next__(self):
        if self.__n < len(self):
            self.__n += 1
            return self[self.__n - 1]
        raise StopIteration

    def __len__(self) -> int:
        """Returns length of states.

        Returns:
            int: length of states.
        """
        return len(self.states)


def set_object_states(
    positions: Optional[List[Tuple[float, float, float]]] = None,
    orientations: Optional[List[Quaternion]] = None,
    sizes: Optional[List[Tuple[float, float, float]]] = None,
    velocities: Optional[List[Tuple[float, float, float]]] = None,
) -> Optional[List[ObjectState]]:
    """[summary]
    Set list of object states.

    Args:
        positions (Optional[List[Tuple[float]]])
        orientations (Optional[List[Tuple[float]]])
        sizes (Optional[List[Tuple[float]]])
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
            size=sizes[i] if sizes else None,
            velocity=velocities[i] if velocities else None,
        )
        for i, (pos, orient) in enumerate(zip(positions, orientations))
    ]


def set_object_path(
    positions: Optional[List[Tuple[float, float, float]]] = None,
    orientations: Optional[List[Quaternion]] = None,
    sizes: Optional[List[Tuple[float, float, float]]] = None,
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
        sizes=sizes,
    )
    return ObjectPath(states, confidence) if states else None


def set_object_paths(
    positions: Optional[List[List[Tuple[float, float, float]]]] = None,
    orientations: Optional[List[List[Quaternion]]] = None,
    sizes: Optional[List[List[Tuple[float, float, float]]]] = None,
    velocities: Optional[List[List[Tuple[float, float, float]]]] = None,
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
            velocities=velocities[i] if velocities else None,
            sizes=sizes[i] if sizes else None,
            confidence=confidence,
        )
        if path:
            paths.append(path)
    return paths
