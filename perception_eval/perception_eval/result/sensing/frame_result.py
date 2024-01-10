# Copyright 2022-2024 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import List
from typing import Tuple

import numpy as np
from perception_eval.common.point import crop_pointcloud
from perception_eval.object import DynamicObject

from .frame_config import SensingFrameConfig
from .object_result import SensingObjectResult


class SensingFrameResult:
    """Object result class for sensing evaluation at current frame.

    In sensing evaluation we defined words `detection area` and `non-detection area`
    - `detection area`: The area pointcloud should be detected, that means, the area in objects' boxes.
    - `non-detection area`: The area pointcloud should not be detected, that means, the area there is no objects.

    Attributes:
        sensing_frame_config (SensingFrameConfig): Configuration of sensing evaluation at current frame.
        unix_time (int): Unix time [us].
        frame_name (str): The name of frame.
        detection_success_results (list[SensingObjectResult]): Container for succeeded results
            in detection area.
        detection_fail_results (list[SensingObjectResult]): Container for failed results in detection area.
        detection_warning_results (List[SensingObjectResult]): Container for warned results
            in detection area. This is used when objects are occluded.
        pointcloud_failed_non_detection (List[numpy.ndarray]): Container for array of detected pointcloud
            in non-detection area.

    Args:
    -----
        sensing_frame_config (SensingFrameConfig): Frame level configuration for sensing evaluation.
        unix_time (int): Unix time [us].
        frame_number (int): Number of frame.
    """

    def __init__(self, sensing_frame_config: SensingFrameConfig, unix_time: int, frame_number: int) -> None:
        # Config
        self.sensing_frame_config = sensing_frame_config

        # Frame information
        self.unix_time: int = unix_time
        self.frame_number: int = frame_number

        # Containers for results
        self.detection_success_results: List[SensingObjectResult] = []
        self.detection_fail_results: List[SensingObjectResult] = []
        self.detection_warning_results: List[SensingObjectResult] = []
        self.pointcloud_failed_non_detection: List[np.ndarray] = []

    def evaluate_frame(
        self,
        ground_truth_objects: List[DynamicObject],
        pointcloud_for_detection: np.ndarray,
        pointcloud_for_non_detection: List[np.ndarray],
    ) -> None:
        """Evaluate each object at current frame.

        Args:
        -----
            ground_truth_objects (list[DynamicObject]): Ground truth objects list.
            pointcloud_for_detection (numpy.ndarray): Array of pointcloud in detection area.
            pointcloud_for_non_detection (List[numpy.ndarray]): List of pointcloud array in non-detection area.
        """
        self._evaluate_pointcloud_for_detection(
            ground_truth_objects,
            pointcloud_for_detection,
        )

        self._evaluate_pointcloud_for_non_detection(
            ground_truth_objects,
            pointcloud_for_non_detection,
        )

    def _evaluate_pointcloud_for_detection(
        self,
        ground_truth_objects: List[DynamicObject],
        pointcloud_for_detection: np.ndarray,
    ) -> None:
        """Evaluate if pointcloud are detected.

        If the object is occluded, the result is appended to `self.detection_warning_results`.

        Args:
        -----
            ground_truth_objects (list[DynamicObject]): Ground truth objects list.
            pointcloud_for_detection (numpy.ndarray): Array of pointcloud in detection area.
        """
        if len(ground_truth_objects) == 0:
            logging.warn("There is no annotated objects")
            return

        for ground_truth_object in ground_truth_objects:
            scale_factor_: float = self.sensing_frame_config.get_scale_factor(ground_truth_object.get_distance())
            sensing_result = SensingObjectResult(
                ground_truth_object,
                pointcloud_for_detection,
                scale_factor=scale_factor_,
                min_points_threshold=self.sensing_frame_config.min_points_threshold,
            )

            if sensing_result.is_occluded:
                self.detection_warning_results.append(sensing_result)
            elif sensing_result.is_detected:
                self.detection_success_results.append(sensing_result)
            else:
                self.detection_fail_results.append(sensing_result)

    def _evaluate_pointcloud_for_non_detection(
        self,
        ground_truth_objects: List[DynamicObject],
        pointcloud_for_non_detection: List[np.ndarray],
    ) -> None:
        """Evaluate if there are any pointcloud were detection in non-detection area.

        First of all, input `pointcloud_for_non_detection` is cropped by objects' boxes.
        Then count the number of points, and if points are remained, these points are appended to
        `self.pointcloud_failed_non_detection`.

        Args:
        -----
            ground_truth_objects (list[DynamicObject]): Ground truth objects list.
            pointcloud_for_non_detection (list[numpy.ndarray]): List of pointcloud array in non-detection area.
        """
        for point_non_detection in pointcloud_for_non_detection:
            for ground_truth_object in ground_truth_objects:
                # Get bbox scale factor
                scale_factor_: float = self.sensing_frame_config.get_scale_factor(ground_truth_object.get_distance())
                # Get object area, the start position is looped
                object_area_: List[List[float]] = ground_truth_object.get_corners(scale_factor_).tolist()
                # Convert object_area_ to tuple and make it un-looped
                object_area_: List[Tuple[float]] = [tuple(e) for e in object_area_]
                # Remove pointcloud in bounding boxes
                point_non_detection = crop_pointcloud(
                    point_non_detection,
                    object_area_,
                    inside=False,
                )
            if len(point_non_detection) != 0:
                self.pointcloud_failed_non_detection.append(point_non_detection)
