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
from perception_eval.common.label import Label
from perception_eval.common.point import crop_pointcloud
from perception_eval.common.point import polygon_to_list
from perception_eval.common.schema import FrameID
from perception_eval.common.schema import Visibility
from perception_eval.common.shape import Shape
from perception_eval.common.state import ObjectPath
from perception_eval.common.state import ObjectState
from perception_eval.common.state import set_object_paths
from perception_eval.common.transform import TransformDict
from pyquaternion import Quaternion
from shapely.geometry import Polygon


class DynamicObject:
    """Dynamic object class for 3D object.

    Attributes:
        unix_time (int) : Unix time [us].
        frame_id (FrameID): FrameID instance, where 3D objects are with respect, BASE_LINK or MAP.

        # Detection
        state (ObjectState): State of object.
        semantic_score (float): Detection score in [0.0, 1.0].
        semantic_label (Label): Object label.

        # Use case object evaluation for detection
        pointcloud_num (Optional[int]): Pointcloud number inside of bounding box.

        # Tracking
        uuid (Optional[str]): The uuid for tracking.
        self.tracked_path (Optional[List[ObjectState]]): List of the past states.

        # Prediction
        predicted_paths (Optional[List[ObjectPath]]): List of the future states.

        visibility (Optional[Visibility]): Visibility status. Defaults to None.

    Args:
        unix_time (int): Unix time [us].
        frame_id (FrameID): FrameID instance, where 3D objects are with respect, BASE_LINK or MAP.
        position (Tuple[float, float, float]): (center_x, center_y, center_z)[m].
        orientation (Quaternion) : Quaternion instance.
        size (Tuple[float, float, float]): Bounding box size, (wx, wy, wz)[m].
        velocity (Optional[Tuple[float, float, float]]): Velocity, (vx, vy, vz)[m/s].
        semantic_score (float): Detection score in [0.0, 1.0].
        semantic_label (Label): Object label.
        pointcloud_num (Optional[int]): Number of points inside of bounding box. Defaults to None.
        uuid (Optional[str]): Unique ID. Defaults to None.
        tracked_positions (Optional[List[Tuple[float, float, float]]]):
                Sequence of positions for tracked object. Defaults to None.
        tracked_orientations (Optional[List[Quaternion]]):
                Sequence of quaternions for tracked object. Defaults to None.
        tracked_shapes (Optional[List[Shape]):
                The list of bounding box size for tracked object. Defaults to None.
        tracked_twists (Optional[List[Tuple[float, float, float]]]):
                The list of twist for tracked object. Defaults to None.
        predicted_positions (Optional[List[Tuple[float, float, float]]]):
                Sequence of positions for predicted object. Defaults to None.
        predicted_orientations (Optional[List[Quaternion]]):
                The list of quaternion for predicted object. Defaults to None.
        predicted_shapes (Optional[List[Shape]]):
                The list of bounding box size for predicted object. Defaults to None.
        predicted_twists (Optional[List[Tuple[float, float, float]]]):
                The list of twist for predicted object. Defaults to None.
        predicted_scores (Optional[List[float]]): Prediction scores for each mode. Defaults to None.
        visibility (Optional[Visibility]): Visibility status. Defaults to None.
    """

    def __init__(
        self,
        unix_time: int,
        frame_id: FrameID,
        position: Tuple[float, float, float],
        orientation: Quaternion,
        shape: Shape,
        velocity: Optional[Tuple[float, float, float]],
        semantic_score: float,
        semantic_label: Label,
        pointcloud_num: Optional[int] = None,
        uuid: Optional[str] = None,
        tracked_positions: Optional[List[Tuple[float, float, float]]] = None,
        tracked_orientations: Optional[List[Quaternion]] = None,
        tracked_shapes: Optional[List[Shape]] = None,
        tracked_twists: Optional[List[Tuple[float, float, float]]] = None,
        predicted_positions: Optional[List[List[Tuple[float, float, float]]]] = None,
        predicted_orientations: Optional[List[List[Quaternion]]] = None,
        predicted_twists: Optional[List[List[Tuple[float, float, float]]]] = None,
        predicted_scores: Optional[List[float]] = None,
        visibility: Optional[Visibility] = None,
    ) -> None:
        # detection
        self.unix_time: int = unix_time
        self.frame_id: FrameID = frame_id
        self.state: ObjectState = ObjectState(
            position=position,
            orientation=orientation,
            shape=shape,
            velocity=velocity,
        )
        self.semantic_score: float = semantic_score
        self.semantic_label: Label = semantic_label

        # for detection label for case evaluation
        # pointcloud number inside bounding box
        self.pointcloud_num: Optional[int] = pointcloud_num

        # tracking
        self.uuid: Optional[str] = uuid
        self.tracked_path: Optional[List[ObjectState]] = self._set_states(
            positions=tracked_positions,
            orientations=tracked_orientations,
            shapes=tracked_shapes,
            twists=tracked_twists,
        )

        # prediction
        self.predicted_paths: Optional[List[ObjectPath]] = set_object_paths(
            positions=predicted_positions,
            orientations=predicted_orientations,
            twists=predicted_twists,
            confidences=predicted_scores,
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

    def get_distance(self, transforms: Optional[TransformDict] = None) -> float:
        """Get the 3d distance to the object from ego vehicle in bird eye view.

        Args:

        Returns:
            float: The 3d distance to the object from ego vehicle in bird eye view.
        """
        if self.frame_id == FrameID.BASE_LINK:
            position = self.state.position
        else:
            if transforms is None:
                raise ValueError("transforms must be specified.")
            position = transforms.transform((self.frame_id, FrameID.BASE_LINK), self.state.position)
        return np.linalg.norm(position)

    def get_distance_bev(self, transforms: Optional[TransformDict] = None) -> float:
        """Get the 2d distance to the object from ego vehicle in bird eye view.

        Args:

        Returns:
            float: The 2d distance to the object from ego vehicle in bird eye view.
        """
        if self.frame_id == FrameID.BASE_LINK:
            position = self.state.position
        else:
            if transforms is None:
                raise ValueError("transforms must be specified.")
            position = transforms.transform((self.frame_id, FrameID.BASE_LINK), self.state.position)
        return math.hypot(position[0], position[1])

    def get_heading_bev(self, transforms: Optional[TransformDict] = None) -> float:
        """Get the object heading from ego vehicle in bird eye view.

        Args:

        Returns:
            float: The heading (radian)
        """
        if self.frame_id == FrameID.BASE_LINK:
            rots: float = self.state.orientation.radians
        else:
            if transforms is None:
                raise ValueError("transforms must be specified.")
            _, rotation = transforms.transform(
                (self.frame_id, FrameID.BASE_LINK), self.state.position, self.state.orientation
            )
            rots, _, _ = rotation.yaw_pitch_roll

        trans_rots: float = -rots - math.pi / 2
        trans_rots = float(np.where(trans_rots > math.pi, trans_rots - 2 * math.pi, trans_rots))
        trans_rots = float(np.where(trans_rots < -math.pi, trans_rots + 2 * math.pi, trans_rots))
        return trans_rots

    def get_corners(self, scale: float = 1.0) -> np.ndarray:
        """Get the bounding box corners.

        Args:
            scale (float): Scale factor to scale the corners. Defaults to 1.0.

        Returns:
            corners (numpy.ndarray): Objects corners array.
        """
        # NOTE: This is with respect to base_link or map coordinate system.
        footprint: np.ndarray = np.array(polygon_to_list(self.get_footprint(scale=scale)))

        # 3D bounding box corners.
        # (Convention: x points forward, y to the left, z up.)
        # upper plane -> lower plane
        upper: np.ndarray = footprint.copy()
        lower: np.ndarray = footprint.copy()
        upper[:, 2] = self.state.position[2] + (self.state.size[2] / 2)
        lower[:, 2] = self.state.position[2] - (self.state.size[2] / 2)
        corners = np.vstack((upper, lower))
        return corners

    def get_footprint(self, scale: float = 1.0) -> Polygon:
        """[summary]
        Get footprint polygon from an object with respect ot base_link or map coordinate system.

        Args:
            scale (float): Scale factor for footprint. Defaults to 1.0.

        Returns:
            Polygon: The footprint polygon of object with respect to base_link or map coordinate system.
                It consists of 4 corner 2d position of the object and  start and end points are same point.
                ((x0, y0, 0), (x1, y1, 0), (x2, y2, 0), (x3, y3, 0), (x0, y0, 0))
        Notes:
            center_position: (xc, yc)
            vector_center_to_corners[0]: (x0 - xc, y0 - yc)
        """
        # NOTE: This is with respect to each object's coordinate system.
        footprint: np.ndarray = np.array(polygon_to_list(self.state.footprint))
        # Translate to (0, 0, 0) and scale
        footprint = footprint * scale
        rotated_footprint: List[Tuple[float, float, float]] = []
        for point in footprint:
            rotate_point: np.ndarray = self.state.orientation.rotate(point)
            rotate_point[:2] = rotate_point[:2] + self.state.position[:2]
            rotated_footprint.append(rotate_point.tolist())
        poly: List[List[float]] = [f for f in rotated_footprint]
        poly.append(rotated_footprint[0])
        return Polygon(poly)

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
        num_waypoints: int,
    ) -> Optional[np.ndarray]:
        """Returns displacement errors of path as numpy.ndarray.

        Args:
            other (Optional[DynamicObject]): DynamicObject instance.
            num_waypoints (optional[int]): Number of waypoints. Defaults to None.

        Returns:
            numpy.ndarray: in shape (K, T, 3)
        """
        if other is None:
            return None

        path_errors: List[List[List[float]]] = []
        for self_path, other_path in zip(self.predicted_paths, other.predicted_paths):
            if self_path is None or other_path is None:
                continue
            self_path, other_path = self_path[:num_waypoints], other_path[:num_waypoints]
            err: List[Tuple[float, float, float]] = [
                self_state.get_position_error(other_state) for self_state, other_state in zip(self_path, other_path)
            ]  # (T, 3)
            path_errors.append(err)
        return np.array(path_errors)

    def get_area_bev(self) -> float:
        """Get area of object BEV.

        Returns:
            float: Area of footprint.
        """
        return self.state.footprint.area

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
            pointcloud (np.ndarray): The Array of pointcloud, in shape (N, 3).
            bbox_scale (float): Scale factor for bounding box. Defaults to 1.0.
            inside (bool): Whether crop inside pointcloud or outside. Defaults to True.

        Returns:
            numpy.ndarray: Pointcloud array inside or outside box.
        """
        corners: np.ndarray = self.get_corners(scale=bbox_scale)
        return crop_pointcloud(pointcloud, corners.tolist(), inside=inside)

    def get_inside_pointcloud_num(
        self,
        pointcloud: np.ndarray,
        bbox_scale: float = 1.0,
    ) -> int:
        """Calculate the number of pointcloud inside of own bounding box.

        Args:
            pointcloud (np.ndarray): The Array of pointcloud, in shape (N, 3).
            bbox_scale (float): Scale factor for bounding box. Defaults to 1.0.

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
        shapes: List[Shape] = None,
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
        if positions is None or orientations is None:
            return None

        if len(positions) != len(orientations):
            raise RuntimeError(
                "Length of positions and orientations must be same, "
                f"but got {len(positions)} and {len(orientations)}"
            )

        states: List[ObjectState] = []
        for i, (position, orientation) in enumerate(zip(positions, orientations)):
            shape = shapes[i] if shapes else None
            velocity = twists[i] if twists else None
            states.append(
                ObjectState(
                    position=position,
                    orientation=orientation,
                    shape=shape,
                    velocity=velocity,
                )
            )
        return states
