from typing import List
from typing import Optional
from typing import Tuple

from pyquaternion import Quaternion

from awml_evaluation.common.label import AutowareLabel


class ObjectState:
    """[summary]
    Object state class
    """

    def __init__(
        self,
        position: Tuple[float, float, float],
        orientation: Quaternion,
        size: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
    ) -> None:
        """
        Args:
            position (Tuple[float, float, float]) : center_x, center_y, center_z [m]
            orientation (Quaternion) : Quaternion class.
                                       See reference http://kieranwynn.github.io/pyquaternion/
            size (Tuple[float, float, float]): bounding box size of (wx, wy, wz) [m]
            velocity (Tuple[float, float, float]): velocity of (vx, vy, vz) [m/s]
        """

        self.position: Tuple[float, float, float] = position
        self.orientation: Quaternion = orientation
        self.size: Tuple[float, float, float] = size
        self.velocity: Tuple[float, float, float] = velocity


class DynamicObject:
    """
    Dynamic object class

    Args:
        self.unix_time (int) : Unix time [us]

        # Detection
        self.state (ObjectState): The state of object
        self.semantic_score (float): Detection score (0.0-1.0)
        self.semantic_label (AutowareLabel): The object label

        # Use case object evaluation for detection
        self.pointcloud_num (Optional[int]): Pointcloud number inside bounding box
        self.will_collide_within_5s (Optional[bool]): Use case evaluation for example

        # Tracking
        self.uuid (Optional[str]): The uuid for tracking
        self.tracked_path (Optional[List[ObjectState]]): List of the past states

        # Prediction
        self.predicted_confidence (Optional[float]): Prediction score
        self.predicted_path (Optional[List[ObjectState]]): List of the future states
    """

    def __init__(
        self,
        unix_time: int,
        position: Tuple[float, float, float],
        orientation: Quaternion,
        size: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
        semantic_score: float,
        semantic_label: AutowareLabel,
        pointcloud_num: Optional[int] = None,
        will_collide_within_5s: Optional[bool] = None,
        uuid: Optional[str] = None,
        tracked_positions: Optional[List[Tuple[float, float, float]]] = None,
        tracked_orientations: Optional[List[Quaternion]] = None,
        tracked_sizes: Optional[List[Tuple[float, float, float]]] = None,
        tracked_twists: Optional[List[Tuple[float, float, float]]] = None,
        predicted_positions: Optional[List[Tuple[float, float, float]]] = None,
        predicted_orientations: Optional[List[Quaternion]] = None,
        predicted_sizes: Optional[List[Tuple[float, float, float]]] = None,
        predicted_twists: Optional[List[Tuple[float, float, float]]] = None,
        predicted_confidence: Optional[float] = None,
    ) -> None:
        """[summary]

        Args:
            unix_time (int): Unix time [us]
            position (Tuple[float, float, float]): The position
            orientation (Quaternion): [description]
            size (Tuple[float, float, float]): [description]
            velocity (Tuple[float, float, float]): [description]
            semantic_score (float): [description]
            semantic_label (AutowareLabel): [description]
            pointcloud_num (Optional[int], optional):
                    Pointcloud number inside bounding box. Defaults to None.
            will_collide_within_5s (Optional[bool], optional):
                    Use case evaluation for example. Defaults to None.
            uuid (Optional[str], optional): The uuid for tracking. Defaults to None.
            tracked_positions (Optional[List[Tuple[float, float, float]]], optional):
                    The list of position for tracked object. Defaults to None.
            tracked_orientations (Optional[List[Quaternion]], optional):
                    The list of quaternion for tracked object. Defaults to None.
            tracked_sizes (Optional[List[Tuple[float, float, float]]], optional):
                    The list of bounding box size for tracked object. Defaults to None.
            tracked_twists (Optional[List[Tuple[float, float, float]]], optional):
                    The list of twist for tracked object. Defaults to None.
            predicted_positions (Optional[List[Tuple[float, float, float]]], optional):
                    The list of position for predicted object. Defaults to None.
            predicted_orientations (Optional[List[Quaternion]], optional):
                    The list of quaternion for predicted object. Defaults to None.
            predicted_sizes (Optional[List[Tuple[float, float, float]]], optional):
                    The list of bounding box size for predicted object. Defaults to None.
            predicted_twists (Optional[List[Tuple[float, float, float]]], optional):
                    The list of twist for predicted object. Defaults to None.
            predicted_confidence (Optional[float], optional): Prediction score. Defaults to None.
        """

        # detection
        self.unix_time: int = unix_time
        self.state: ObjectState = ObjectState(
            position,
            orientation,
            size,
            velocity,
        )
        self.semantic_score: float = semantic_score
        self.semantic_label: AutowareLabel = semantic_label

        # for detection label for case evaluation
        # pointcloud number inside bounding box
        self.pointcloud_num: Optional[int] = pointcloud_num
        self.will_collide_within_5s: Optional[bool] = will_collide_within_5s

        # tracking
        self.uuid: Optional[str] = uuid
        self.tracked_path: Optional[List[ObjectState]] = self._set_states(
            tracked_positions,
            tracked_orientations,
            tracked_sizes,
            tracked_twists,
        )

        # prediction
        self.predicted_confidence: Optional[float] = predicted_confidence
        self.predicted_path: Optional[List[ObjectState]] = self._set_states(
            predicted_positions,
            predicted_orientations,
            predicted_sizes,
            predicted_twists,
        )

    def __eq__(self, other) -> bool:
        eq = True
        eq = eq and self.semantic_label == other.semantic_label
        eq = eq and self.state.position == other.state.position
        eq = eq and self.state.shape_dimension == other.state.shape_dimension
        return eq

    def get_distance_bev(self) -> float:
        """[summary]
        Get the distance to the object from ego vehicle in bird eye view

        Returns:
            float: The distance to the object from ego vehicle in bird eye view
        """
        return self.state.position[0] ** 2 + self.state.position[1] ** 2

    @staticmethod
    def _set_states(
        positions: Optional[List[Tuple[float, float, float]]] = None,
        orientations: Optional[List[Tuple[float, float, float, float]]] = None,
        sizes: Optional[List[Tuple[float, float, float]]] = None,
        twists: Optional[List[Tuple[float, float, float]]] = None,
    ) -> Optional[List[ObjectState]]:
        """[summary]
        Set object state from positions, orientations, sizes, and twists.

        Args:
            positions (Optional[List[Tuple[float, float, float]]], optional):
                    The list of positions for object. Defaults to None.
            orientations (Optional[List[Quaternion]], optional):
                    The list of quaternions for object. Defaults to None.
            sizes (Optional[List[Tuple[float, float, float]]], optional):
                    The list of bounding box sizes for object. Defaults to None.
            twists (Optional[List[Tuple[float, float, float]]], optional):
                    The list of twists for object. Defaults to None.

        Returns:
            Optional[List[ObjectState]]: The list of ObjectState
        """

        if positions and orientations and sizes and twists:
            states: List[ObjectState] = []
            for position, orientation, size, twist in positions, orientations, sizes, twists:
                states.append(ObjectState(position, orientation, size, twist))
            return states
        else:
            return None


def distance_between(
    object_1: DynamicObject,
    object_2: DynamicObject,
) -> float:
    """[summary]
    Calculate the center distance between DynamicObject.
    Calculate the center distance from object_1 to object_2.

    Args:
        object_1 (DynamicObject):
        object_2 (DynamicObject):

    Returns:
        float: The center distance
    """
    x: float = object_1.state.position[0] - object_2.state.position[0]
    y: float = object_1.state.position[1] - object_2.state.position[1]
    z: float = object_1.state.position[2] - object_2.state.position[2]

    return x * x + y * y + z * z
