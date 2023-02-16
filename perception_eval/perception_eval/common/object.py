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
from perception_eval.common.status import FrameID
from perception_eval.common.status import Visibility
from perception_eval.util.math import rotation_matrix_to_euler
from pyquaternion import Quaternion
from shapely.geometry import Polygon


class ObjectState:
    """Object state class.

    Attributes:
        position (Tuple[float, float, float]): (center_x, center_y, center_z)[m].
        orientation (Quaternion) : Quaternion instance.
        size (Tuple[float, float, float]): Bounding box size, (wx, wy, wz)[m].
        velocity (Optional[Tuple[float, float, float]]): Velocity, (vx, vy, vz)[m/s].

    Args:
        position (Tuple[float, float, float]): (center_x, center_y, center_z)[m].
        orientation (Quaternion) : Quaternion instance.
        size (Tuple[float, float, float]): Bounding box size, (wx, wy, wz)[m].
        velocity (Optional[Tuple[float, float, float]]): Velocity, (vx, vy, vz)[m/s].
    """

    def __init__(
        self,
        position: Tuple[float, float, float],
        orientation: Quaternion,
        size: Tuple[float, float, float],
        velocity: Optional[Tuple[float, float, float]],
    ) -> None:
        self.position: Tuple[float, float, float] = position
        self.orientation: Quaternion = orientation
        self.size: Tuple[float, float, float] = size
        self.velocity: Optional[Tuple[float, float, float]] = velocity


class DynamicObject:
    """Dynamic object class for 3D object.

    Attributes:
        unix_time (int) : Unix time [us].
        frame_id (FrameID): FrameID instance, where 3D objects are with respect, BASE_LINK or MAP.

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
        predicted_confidence (Optional[float]): Prediction score.
        predicted_path (Optional[List[ObjectState]]): List of the future states.

        visibility (Optional[Visibility]): Visibility status. Defaults to None.

    Args:
        unix_time (int): Unix time [us].
        frame_id (FrameID): FrameID instance, where 3D objects are with respect, BASE_LINK or MAP.
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
                The list of bounding box size for tracked object. Defaults to None.
        tracked_twists (Optional[List[Tuple[float, float, float]]]):
                The list of twist for tracked object. Defaults to None.
        predicted_positions (Optional[List[Tuple[float, float, float]]]):
                The list of position for predicted object. Defaults to None.
        predicted_orientations (Optional[List[Quaternion]]):
                The list of quaternion for predicted object. Defaults to None.
        predicted_sizes (Optional[List[Tuple[float, float, float]]]):
                The list of bounding box size for predicted object. Defaults to None.
        predicted_twists (Optional[List[Tuple[float, float, float]]]):
                The list of twist for predicted object. Defaults to None.
        predicted_confidence (Optional[float]): Prediction score. Defaults to None.
        visibility (Optional[Visibility]): Visibility status. Defaults to None.
    """

    def __init__(
        self,
        unix_time: int,
        frame_id: FrameID,
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
        tracked_twists: Optional[List[Tuple[float, float, float]]] = None,
        predicted_positions: Optional[List[Tuple[float, float, float]]] = None,
        predicted_orientations: Optional[List[Quaternion]] = None,
        predicted_sizes: Optional[List[Tuple[float, float, float]]] = None,
        predicted_twists: Optional[List[Tuple[float, float, float]]] = None,
        predicted_confidence: Optional[float] = None,
        visibility: Optional[Visibility] = None,
    ) -> None:
        # detection
        self.unix_time: int = unix_time
        self.frame_id: FrameID = frame_id
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

    def get_distance(self, ego2map: Optional[np.ndarray] = None) -> float:
        """Get the 3d distance to the object from ego vehicle in bird eye view.

        Args:
            ego2map (Optional[numpy.ndarray]):4x4 Transform matrix
                from base_link coordinate system to map coordinate system.

        Returns:
            float: The 3d distance to the object from ego vehicle in bird eye view.
        """
        if self.frame_id == FrameID.BASE_LINK:
            return np.linalg.norm(self.state.position)

        if ego2map is None:
            raise RuntimeError(
                "For objects with respect to map coordinate system, ego2map must be specified."
            )

        pos_arr: np.ndarray = np.append(self.state.position, 1.0)
        return np.linalg.norm(np.linalg.inv(ego2map).dot(pos_arr)[:3])

    def get_distance_bev(self, ego2map: Optional[np.ndarray] = None) -> float:
        """Get the 2d distance to the object from ego vehicle in bird eye view.

        Args:
            ego2map (Optional[numpy.ndarray]):4x4 Transform matrix
                from base_link coordinate system to map coordinate system.

        Returns:
            float: The 2d distance to the object from ego vehicle in bird eye view.
        """
        if self.frame_id == FrameID.BASE_LINK:
            return math.hypot(self.state.position[0], self.state.position[1])

        if ego2map is None:
            raise RuntimeError(
                "For objects with respect to map coordinate system, ego2map must be specified."
            )

        pos_arr: np.ndarray = np.append(self.state.position, 1.0)
        return np.linalg.norm(np.linalg.inv(ego2map).dot(pos_arr)[:2])

    def get_heading_bev(self, ego2map: Optional[np.ndarray] = None) -> float:
        """Get the object heading from ego vehicle in bird eye view.

        Args:
            ego2map (Optional[numpy.ndarray]):4x4 Transform matrix.
                from base_link coordinate system to map coordinate system.

        Returns:
            float: The heading (radian)
        """
        if self.frame_id == FrameID.MAP:
            if ego2map is None:
                raise RuntimeError(
                    "For objects with respect to map coordinate system, ego2map must be specified."
                )
            src: np.ndarray = np.eye(4, 4)
            src[:3, :3] = self.state.orientation.rotation_matrix
            src[:3, 3] = self.state.position
            dst: np.ndarray = np.linalg.inv(ego2map).dot(src)
            rots: float = rotation_matrix_to_euler(dst[:3, :3])[-1].item()
        else:
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
        """Get the position error between myself and other. If other is None, returns None.

        Returns:
            err_x (float): x-axis position error[m].
            err_y (float): y-axis position error[m].
            err_z (float): z-axis position error[m].
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

        err_vx: float = abs(other.state.velocity[0] - self.state.velocity[0])
        err_vy: float = abs(other.state.velocity[1] - self.state.velocity[1])
        err_vz: float = abs(other.state.velocity[2] - self.state.velocity[2])

        return err_vx, err_vy, err_vz

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
        bbox_scale: float = 1.0,
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
