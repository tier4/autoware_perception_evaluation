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

from logging import getLogger
import math
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from perception_eval.common.label import AutowareLabel
from perception_eval.common.point import distance_points
from perception_eval.common.point import distance_points_bev
from perception_eval.common.status import Visibility
from pyquaternion import Quaternion
from shapely.geometry import Polygon

logger = getLogger(__name__)


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

        # Tracking
        self.uuid (Optional[str]): The uuid for tracking
        self.tracked_path (Optional[List[ObjectState]]): List of the past states

        # Prediction
        self.predicted_confidence (Optional[float]): Prediction score
        self.predicted_path (Optional[List[ObjectState]]): List of the future states

        self.visibility (Optional[Visibility]): Visibility status. Defaults to None.
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
        visibility: Optional[Visibility] = None,
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
            visibility (Optional[Visibility]): Visibility status. Defaults to None.
        """

        # detection
        self.unix_time: int = unix_time
        self.state: ObjectState = ObjectState(
            position=position,
            orientation=orientation,
            size=size,
            velocity=velocity,
        )
        self.semantic_score: float = semantic_score
        self.semantic_label: AutowareLabel = semantic_label

        # for detection label for case evaluation
        # pointcloud number inside bounding box
        self.pointcloud_num: Optional[int] = pointcloud_num

        # tracking
        self.uuid: Optional[str] = uuid
        self.tracked_path: Optional[List[ObjectState]] = DynamicObject._set_states(
            positions=tracked_positions,
            orientations=tracked_orientations,
            sizes=tracked_sizes,
            twists=tracked_twists,
        )

        # prediction
        self.predicted_confidence: Optional[float] = predicted_confidence
        self.predicted_path: Optional[List[ObjectState]] = DynamicObject._set_states(
            positions=predicted_positions,
            orientations=predicted_orientations,
            sizes=predicted_sizes,
            twists=predicted_twists,
        )

        self.visibility: Optional[Visibility] = visibility

    def __eq__(self, other: Optional[DynamicObject]) -> bool:
        """[summary]
        Check if other equals this object.
        If other is not None and does NOT has attributes unix_time, semantic_label, state.position, state.orientation \
        it causes error.

        Returns:
            bool
        """
        if other is None:
            return False
        else:
            eq: bool = True
            eq = eq and self.unix_time == other.unix_time
            eq = eq and self.semantic_label == other.semantic_label  # type: ignore
            eq = eq and self.state.position == other.state.position  # type: ignore
            eq = eq and self.state.orientation == other.state.orientation  # type: ignore
            return eq

    def get_distance(self) -> float:
        """[summary]
        Get the 3d distance to the object from ego vehicle in bird eye view

        Returns:
            float: The 3d distance to the object from ego vehicle in bird eye view
        """
        return np.linalg.norm(self.state.position)

    def get_distance_bev(self) -> float:
        """[summary]
        Get the 2d distance to the object from ego vehicle in bird eye view

        Returns:
            float: The 2d distance to the object from ego vehicle in bird eye view
        """
        return math.hypot(self.state.position[0], self.state.position[1])

    def get_heading_bev(self) -> float:
        """[summary]
        Get the object heading from ego vehicle in bird eye view

        Returns:
            float: The heading (radian)
        """
        rots: float = self.state.orientation.radians
        trans_rots: float = -rots - math.pi / 2
        trans_rots = float(np.where(trans_rots > math.pi, trans_rots - 2 * math.pi, trans_rots))
        trans_rots = float(np.where(trans_rots < -math.pi, trans_rots + 2 * math.pi, trans_rots))
        return trans_rots

    def get_corners(self, bbox_scale: float) -> np.ndarray:
        """[summary]
        Get the bounding box corners.

        Args:
            bbox_scale (float): The factor to scale the box. Defaults to 1.0.

        Returns:
            corners (numpy.ndarray)
        """
        width = self.state.size[0] * bbox_scale
        length = self.state.size[1] * bbox_scale
        height = self.state.size[2] * bbox_scale

        # 3D bounding box corners.
        # (Convention: x points forward, y to the left, z up.)
        # upper plane -> lower plane
        x_corners = length / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = height / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        corners = np.stack((x_corners, y_corners, z_corners), axis=-1)

        # Rotate to the object heading
        corners = np.dot(corners, self.state.orientation.rotation_matrix)

        # Translate by object position
        corners = corners + self.state.position

        return corners

    def get_footprint(self) -> Polygon:
        """[summary]
        Get footprint polygon from an object

        Returns:
            Polygon: The footprint polygon of object. It consists of 4 corner 2d position of
                     the object and  start and end points are same point.
                     ((x0, y0, 0), (x1, y1, 0), (x2, y2, 0), (x3, y3, 0), (x0, y0, 0))
        Notes:
            center_position: (xc, yc)
            vector_center_to_corners[0]: (x0 - xc, y0 - yc)
        """
        corner_points: List[Tuple[float, float]] = []
        vector_center_to_corners: List[np.ndarray] = [
            np.array([self.state.size[1], self.state.size[0], 0.0]) / 2.0,
            np.array([-self.state.size[1], self.state.size[0], 0.0]) / 2.0,
            np.array([-self.state.size[1], -self.state.size[0], 0.0]) / 2.0,
            np.array([self.state.size[1], -self.state.size[0], 0.0]) / 2.0,
        ]
        # rotate vector_center_to_corners
        for vector_center_to_corner in vector_center_to_corners:
            rotated_vector: np.ndarray = self.state.orientation.rotate(vector_center_to_corner)
            corner_point: np.ndarray = self.state.position + rotated_vector
            corner_points.append(corner_point.tolist())
        # corner point to footprint
        footprint: Polygon = Polygon(
            [
                corner_points[0],
                corner_points[1],
                corner_points[2],
                corner_points[3],
                corner_points[0],
            ]
        )
        return footprint

    def get_position_error(
        self,
        other: Optional[DynamicObject],
    ) -> Optional[Tuple[float, float, float]]:
        """[summary]
        Get the position error between myself and other. If other is None, returns None.

        Returns:
            err_x (float): Error of x[m].
            err_y (float): Error of y[m].
            err_z (float): Error of z[m].
        """
        if other is None:
            return None
        err_x: float = abs(other.state.position[0] - self.state.position[0])
        err_y: float = abs(other.state.position[1] - self.state.position[1])
        err_z: float = abs(other.state.position[2] - self.state.position[2])
        return (err_x, err_y, err_z)

    def get_heading_error(
        self,
        other: Optional[DynamicObject],
    ) -> Optional[Tuple[float, float, float]]:
        """[summary]
        Get the heading error between myself and other. If other is None, returns None.

        Returns:
            err_x (float): Error of rotation angle around x-axis, in the range [-pi, pi].
            err_y (float): Error of rotation angle around y-axis, in the range [-pi, pi].
            err_z (float): Error of rotation angle around z-axis, in the range [-pi, pi].
        """
        if other is None:
            return None

        def _clip(err: float) -> float:
            """Clip [-2pi, 2pi] to [0, pi]"""
            if err < 0:
                err += -np.pi * (err // np.pi)
            elif err > np.pi:
                err -= 2 * np.pi
            return err

        yaw1, pitch1, roll1 = self.state.orientation.yaw_pitch_roll
        yaw2, pitch2, roll2 = other.state.orientation.yaw_pitch_roll
        err_x: float = _clip(roll2 - roll1)
        err_y: float = _clip(pitch2 - pitch1)
        err_z: float = _clip(yaw2 - yaw1)

        return (err_x, err_y, err_z)

    def get_velocity_error(
        self,
        other: Optional[DynamicObject],
    ) -> Optional[Tuple[float, float, float]]:
        """[summary]
        Get the velocity error between myself and other.
        If other is None, returns None. Also, velocity of myself or other is None, returns None too.

        Returns:
            err_vx (float): Error of vx[m].
            err_vy (float): Error of vy[m].
            err_vz (float): Error of vz[m].
        """
        if other is None:
            return None

        if self.state.velocity is None or other.state.velocity is None:
            return None

        err_vx: float = abs(other.state.velocity[0] - self.state.velocity[0])
        err_vy: float = abs(other.state.velocity[1] - self.state.velocity[1])
        err_vz: float = abs(other.state.velocity[2] - self.state.velocity[2])

        return err_vx, err_vy, err_vz

    def get_area_bev(self) -> float:
        """[summary]
        Get area of object BEV.

        Returns:
            float: The 2d area from object.
        """
        return self.state.size[0] * self.state.size[1]

    def get_volume(self) -> float:
        return self.get_area_bev() * self.state.size[2]

    def crop_pointcloud(
        self,
        pointcloud: np.ndarray,
        bbox_scale: float,
        inside: bool = True,
    ) -> np.ndarray:
        """[summary]
        Get pointcloud inside of bounding box.

        Args:
            pointcloud (np.ndarray): The Array of pointcloud, in shape (N, 3).
            bbox_scale (float): Scale factor for bounding box.

        Returns:
            numpy.ndarray: The array of pointcloud in bounding box.
        """
        scaled_bbox_size_object_coords: np.ndarray = np.array(
            [
                bbox_scale * self.state.size[1],
                bbox_scale * self.state.size[0],
                bbox_scale * self.state.size[2],
            ]
        )

        # Convert pointcloud coordinates from ego pose to relative to object
        pointcloud_object_coords: np.ndarray = pointcloud[:, :3] - self.state.position

        # Calculate the indices of pointcloud in bounding box
        inside_idx: np.ndarray = (
            (pointcloud_object_coords >= -0.5 * scaled_bbox_size_object_coords)
            * (pointcloud_object_coords <= 0.5 * scaled_bbox_size_object_coords)
        ).all(axis=1)

        if inside:
            return pointcloud[inside_idx]
        return pointcloud[~inside_idx]

    def get_inside_pointcloud_num(
        self,
        pointcloud: np.ndarray,
        bbox_scale: float,
    ) -> int:
        """[summary]
        Calculate the number of pointcloud inside of bounding box.

        Args:
            pointcloud (np.ndarray): The Array of pointcloud, in shape (N, 3).
            bbox_scale (float): Scale factor for bounding box.

        Returns:
            int: The number of points in bounding box.
        """
        inside_pointcloud: np.ndarray = self.crop_pointcloud(pointcloud, bbox_scale)
        return len(inside_pointcloud)

    def point_exist(self, pointcloud: np.ndarray, bbox_scale: float) -> bool:
        """Evaluate whether any input points are inside of bounding box.

        Args:
            pointcloud (numpy.ndarray): The array of pointcloud.
            bbox_scale (float): Scale factor for bounding box.

        Returns:
            bool: Return True if exists.
        """
        num_inside: int = self.get_inside_pointcloud_num(pointcloud, bbox_scale)
        return num_inside > 0

    @staticmethod
    def _set_states(
        positions: Optional[List[Tuple[float, float, float]]] = None,
        orientations: Optional[List[Quaternion]] = None,
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

        if (
            positions is not None
            and orientations is not None
            and sizes is not None
            and twists is not None
        ):
            states: List[ObjectState] = []
            for position, orientation, size, twist in zip(positions, orientations, sizes, twists):
                states.append(
                    ObjectState(
                        position=position, orientation=orientation, size=size, velocity=twist
                    )
                )
            return states
        else:
            return None


class Roi:
    """Region of Interest; ROI class."""

    def __init__(
        self,
        offset: Tuple[int, int],
        size: Tuple[int, int],
    ) -> None:
        self.offset: Tuple[int, int] = offset
        self.size: Tuple[int, int] = size

    @property
    def center(self) -> Tuple[int, int]:
        return (self.offset[0] + self.width // 2, self.offset[1] + self.height // 2)

    @property
    def height(self) -> int:
        return self.size[0]

    @property
    def width(self) -> int:
        return self.size[1]

    @property
    def area(self) -> int:
        return self.size[0] * self.size[1]


class RoiObject:
    """ROI object class."""

    def __init__(
        self,
        unix_time: int,
        offset: Tuple[int, int],
        size: Tuple[int, int],
        semantic_score: float,
        semantic_label: AutowareLabel,
        uuid: Optional[str] = None,
        visibility: Optional[Visibility] = None,
    ) -> None:
        """[summary]
        Args:
            unix_time (int)
            offset (Tuple[int, int]): (x, y) order.
            size (Tuple[int, int]): (height, width) order.
            semantic_score (float)
            semantic_label (AutowareLabel)
        """
        self.unix_time: int = unix_time
        self.roi: Roi = Roi(offset, size)
        self.semantic_score: float = semantic_score
        self.semantic_label: AutowareLabel = semantic_label
        self.visibility: Optional[Visibility] = visibility

    def get_corners(self) -> np.ndarray:
        """[summary]
        Returns the corners of bounding box in pixel.

        Returns:
            numpy.ndarray: (top_left, top_right, bottom_right, bottom_left), in shape (4, 2).
        """
        top_left: Tuple[int, int] = self.roi.offset
        top_right: Tuple[int, int] = (
            self.roi.offset[0] + self.roi.width,
            self.roi.offset[1],
        )
        bottom_right: Tuple[int, int] = (
            self.roi.offset[0] + self.roi.width,
            self.roi.offset[1] + self.roi.height,
        )
        bottom_left: Tuple[int, int] = (
            self.roi.offset[0],
            self.roi.offset[1] + self.roi.height,
        )
        return np.array([top_left, top_right, bottom_right, bottom_left])

    def get_area(self) -> int:
        """[summary]
        Returns the area of bounding box in pixel.

        Returns:
            int: Area of bounding box[px].
        """
        return self.roi.area

    def get_polygon(self) -> Polygon:
        """[summary]
        Returns the corners as polygon.

        Returns:
            Polygon
        """
        corners: List[List[float]] = self.get_corners().tolist()
        corners.append(corners[0])
        return Polygon(corners)


def distance_objects(
    object_1: Union[DynamicObject, RoiObject],
    object_2: Union[DynamicObject, RoiObject],
) -> float:
    """[summary]
    Calculate the 3d center distance between two objects.
    Args:
         object_1 (Union[DynamicObject, RoiObject]): An object
         object_2 (Union[DynamicObject, RoiObject]): An object
    Returns: float: The center distance from object_1 (DynamicObject) to object_2.
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
