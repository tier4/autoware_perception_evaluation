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

import math
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from perception_eval.common.label import LabelType
from perception_eval.common.state import ObjectPath
from perception_eval.common.state import ObjectState
from perception_eval.common.state import set_object_paths
from perception_eval.common.state import set_object_states
from perception_eval.common.status import Visibility
from pyquaternion import Quaternion
from shapely.geometry import Polygon


class DynamicObject:
    """Dynamic object class for 3D object.

    Attributes:
        unix_time (int) : Unix time [us].

        # Detection
        state (ObjectState): State of object.
        semantic_score (float): Detection score in [0.0, 1.0].
        semantic_label (LabelType): Object label

        # Use case object evaluation for detection
        pointcloud_num (Optional[int]): Pointcloud number inside of bounding box.

        # Tracking
        uuid (Optional[str]): The uuid for tracking.
        self.tracked_path (Optional[List[ObjectState]]): List of the past states.

        # Prediction
        predicted_paths (Optional[List[ObjectPath]]): List of the future states.

        visibility (Optional[Visibility]): Visibility status. Defaults to None.

    Args:
        unix_time (int): Unix time [us]
        position (Tuple[float, float, float]): (center_x, center_y, center_z)[m].
        orientation (Quaternion) : Quaternion instance.
        size (Tuple[float, float, float]): Bounding box size, (wx, wy, wz)[m].
        velocity (Optional[Tuple[float, float, float]]): Velocity, (vx, vy, vz)[m/s].
        semantic_score (float): [description]
        semantic_label (LabelType): [description]
        pointcloud_num (Optional[int]): Number of points inside of bounding box. Defaults to None.
        uuid (Optional[str]): Unique ID. Defaults to None.
        tracked_positions (Optional[List[Tuple[float, float, float]]]):
                Sequence of positions for tracked object. Defaults to None.
        tracked_orientations (Optional[List[Quaternion]]):
                Sequence of quaternions for tracked object. Defaults to None.
        tracked_sizes (Optional[List[Tuple[float, float, float]]]):
                Sequence of bounding box sizes for tracked object. Defaults to None.
        tracked_velocities (Optional[List[Tuple[float, float, float]]]):
                Sequence of velocities for tracked object. Defaults to None.
        predicted_positions (Optional[List[Tuple[float, float, float]]]):
                Sequence of positions for predicted object. Defaults to None.
        predicted_orientations (Optional[List[Quaternion]]):
                Sequence of quaternions for predicted object. Defaults to None.
        predicted_sizes (Optional[List[Tuple[float, float, float]]]):
                Sequence of bounding box sizes for predicted object. Defaults to None.
        predicted_velocities (Optional[List[Tuple[float, float, float]]]):
                Sequence of velocities for predicted object. Defaults to None.
        predicted_confidence (Optional[float]): Prediction score. Defaults to None.
        visibility (Optional[Visibility]): Visibility status. Defaults to None.
    """

    def __init__(
        self,
        unix_time: int,
        position: Tuple[float, float, float],
        orientation: Quaternion,
        size: Tuple[float, float, float],
        velocity: Optional[Tuple[float, float, float]],
        semantic_score: float,
        semantic_label: LabelType,
        pointcloud_num: Optional[int] = None,
        uuid: Optional[str] = None,
        tracked_positions: Optional[List[Tuple[float, float, float]]] = None,
        tracked_orientations: Optional[List[Quaternion]] = None,
        tracked_sizes: Optional[List[Tuple[float, float, float]]] = None,
        tracked_velocities: Optional[List[Tuple[float, float, float]]] = None,
        predicted_positions: Optional[List[List[Tuple[float, float, float]]]] = None,
        predicted_orientations: Optional[List[List[Quaternion]]] = None,
        predicted_sizes: Optional[List[List[Tuple[float, float, float]]]] = None,
        predicted_velocities: Optional[List[List[Tuple[float, float, float]]]] = None,
        predicted_confidences: Optional[List[float]] = None,
        visibility: Optional[Visibility] = None,
    ) -> None:
        # detection
        self.unix_time: int = unix_time
        self.state: ObjectState = ObjectState(
            position=position,
            orientation=orientation,
            size=size,
            velocity=velocity,
        )
        self.semantic_score: float = semantic_score
        self.semantic_label: LabelType = semantic_label

        # for detection label for case evaluation
        # pointcloud number inside bounding box
        self.pointcloud_num: Optional[int] = pointcloud_num

        # tracking
        self.uuid: Optional[str] = uuid
        self.tracked_path: Optional[List[ObjectState]] = set_object_states(
            positions=tracked_positions,
            orientations=tracked_orientations,
            sizes=tracked_sizes,
            velocities=tracked_velocities,
        )

        # prediction
        self.predicted_paths: Optional[List[ObjectPath]] = set_object_paths(
            positions=predicted_positions,
            orientations=predicted_orientations,
            sizes=predicted_sizes,
            velocities=predicted_velocities,
            confidences=predicted_confidences,
        )

        self.visibility: Optional[Visibility] = visibility

    def __eq__(self, other: Optional[DynamicObject]) -> bool:
        """Check if other equals this object.

        If other is not None and does NOT has attributes unix_time, semantic_label, state.position, state.orientation
        it causes error.

        Args:
            other (Optional[DynamicObject]): Another object.

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
        """Get the 3d distance to the object from ego vehicle in bird eye view.

        Returns:
            float: The 3d distance to the object from ego vehicle in bird eye view.
        """
        return np.linalg.norm(self.state.position)

    def get_distance_bev(self) -> float:
        """Get the 2d distance to the object from ego vehicle in bird eye view.

        Returns:
            float: The 2d distance to the object from ego vehicle in bird eye view.
        """
        return math.hypot(self.state.position[0], self.state.position[1])

    def get_heading_bev(self) -> float:
        """Get the object heading from ego vehicle in bird eye view

        Returns:
            float: The heading (radian)
        """
        rots: float = self.state.orientation.radians
        trans_rots: float = -rots - math.pi / 2
        trans_rots = float(np.where(trans_rots > math.pi, trans_rots - 2 * math.pi, trans_rots))
        trans_rots = float(np.where(trans_rots < -math.pi, trans_rots + 2 * math.pi, trans_rots))
        return trans_rots

    def get_corners(self, bbox_scale: float) -> np.ndarray:
        """Get the bounding box corners.

        Args:
            bbox_scale (float): The factor to scale the box. Defaults to 1.0.

        Returns:
            corners (numpy.ndarray): Objects corners array.
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
        """Get footprint polygon from an object

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
        """Get the position error between myself and other. If other is None, returns None.

        Returns:
            err_x (float): x-axis position error[m].
            err_y (float): y-axis position error[m].
            err_z (float): z-axis position error[m].
        """
        if other is None:
            return None
        err_x: float = other.state.position[0] - self.state.position[0]
        err_y: float = other.state.position[1] - self.state.position[1]
        err_z: float = other.state.position[2] - self.state.position[2]
        return (err_x, err_y, err_z)

    def get_heading_error(
        self,
        other: Optional[DynamicObject],
    ) -> Optional[Tuple[float, float, float]]:
        """Get the heading error between myself and other. If other is None, returns None.

        Returns:
            err_x (float): Roll error[rad], in [-pi, pi].
            err_y (float): Pitch error[rad], in [-pi, pi].
            err_z (float): Yaw error[rad], in [-pi, pi].
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
        """Get the velocity error between myself and other.

        If other is None, returns None. Also, velocity of myself or other is None, returns None too.

        Returns:
            err_vx (float): x-axis velocity error[m/s].
            err_vy (float): y-axis velocity error[m/s].
            err_vz (float): z-axis velocity error[m/s].
        """
        if other is None:
            return None

        if self.state.velocity is None or other.state.velocity is None:
            return None

        err_vx: float = other.state.velocity[0] - self.state.velocity[0]
        err_vy: float = other.state.velocity[1] - self.state.velocity[1]
        err_vz: float = other.state.velocity[2] - self.state.velocity[2]

        return err_vx, err_vy, err_vz

    def get_path_error(
        self,
        other: Optional[DynamicObject],
        num_waypoints: Optional[int] = None,
        padding: float = np.nan,
    ) -> Optional[np.ndarray]:
        """Returns errors of path as numpy.ndarray.

        Args:
            other (Optional[DynamicObject]): DynamicObject instance.
            num_waypoints (optional[int]): Number of waypoints. Defaults to None.
            padding (float): Padding value. Defaults to numpy.nan.

        Returns:
            numpy.ndarray: in shape (K, T, 3)
        """
        if other is None:
            return None

        self_paths: List[ObjectPath] = self.predicted_paths.copy()
        other_paths: List[ObjectPath] = other.predicted_paths.copy()

        path_errors: List[List[List[float]]] = []
        for self_path, other_path in zip(self_paths, other_paths):
            if self_path is None or other_path is None:
                continue
            num_waypoints_ = (
                num_waypoints if num_waypoints else min(len(self_path), len(other_path))
            )
            self_path_, other_path_ = self_path[:num_waypoints_], other_path[:num_waypoints_]
            err: List[Tuple[float, float, float]] = [
                self_state.get_position_error(other_state)
                for self_state, other_state in zip(self_path_, other_path_)
            ]
            path_errors.append(err)
        return np.array(path_errors)

    def get_area_bev(self) -> float:
        """Get area of object BEV.

        Returns:
            float: Area of footprint.
        """
        return self.state.size[0] * self.state.size[1]

    def get_volume(self) -> float:
        """Get volume of bounding box.

        Returns:
            float: Box volume.
        """
        return self.get_area_bev() * self.state.size[2]

    def crop_pointcloud(
        self,
        pointcloud: np.ndarray,
        bbox_scale: float,
        inside: bool = True,
    ) -> np.ndarray:
        """Returns pointcloud inside or outsize of own bounding box.

        If `inside=True`, returns pointcloud array inside of box.
        Otherwise, returns pointcloud array outside of box.

        Args:
            pointcloud (numpy.ndarray): Pointcloud array, in shape (N, 3).
            bbox_scale (float): Scale factor for bounding box size.
            inside (bool): Whether crop pointcloud inside or outside of box.

        Returns:
            numpy.ndarray: Pointcloud array inside or outside box.
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
        """Calculate the number of pointcloud inside of own bounding box.

        Args:
            pointcloud (numpy.ndarray): Pointcloud array, in shape (N, 3).
            bbox_scale (float): Scale factor for bounding box size.

        Returns:
            int: Number of points inside of the own box.
        """
        inside_pointcloud: np.ndarray = self.crop_pointcloud(pointcloud, bbox_scale)
        return len(inside_pointcloud)

    def point_exist(self, pointcloud: np.ndarray, bbox_scale: float) -> bool:
        """Evaluate whether any input points inside of own bounding box.

        Args:
            pointcloud (numpy.ndarray): Pointcloud array.
            bbox_scale (float): Scale factor for bounding box size.

        Returns:
            bool: Return True if any points exist.
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
        """Set object state from positions, orientations, sizes, and twists.

        Args:
            positions (Optional[List[Tuple[float, float, float]]]): Sequence of positions. Defaults to None.
            orientations (Optional[List[Quaternion]], optional): Sequence of orientations. Defaults to None.
            sizes (Optional[List[Tuple[float, float, float]]]): Sequence of boxes sizes. Defaults to None.
            twists (Optional[List[Tuple[float, float, float]]]): Sequence of velocities. Defaults to None.

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
