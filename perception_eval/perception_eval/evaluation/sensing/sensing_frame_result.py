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

import logging
from typing import List
from typing import Tuple

import numpy as np
from perception_eval.common.object import DynamicObject
from perception_eval.common.point import crop_pointcloud
from perception_eval.evaluation.sensing.sensing_frame_config import SensingFrameConfig
from perception_eval.evaluation.sensing.sensing_result import DynamicObjectWithSensingResult


class SensingFrameResult:
    """[summary]
    The result for 1 frame (the pair of detected points and ground truth)

    Attributes:
        self.sensing_frame_config (SensingFrameConfig):
            The configuration of sensing evaluation.
        self.unix_time (int): Unix time [us]
        self.frame_name (str): The name of frame.
        self.detection_success_results (list[DynamicObjectWithSensingResult]):
            The container for succeeded results of detection.
        self.detection_fail_results (list[DynamicObjectWithSensingResult]):
            The container for failed results of detection.
        self.detection_warning_results (List[DynamicObjectWithSensingResult]):
            The container for warned
        self.pointcloud_failed_non_detection (np.ndarray): The array of pointcloud for non-detected.
    """

    def __init__(
        self,
        sensing_frame_config: SensingFrameConfig,
        unix_time: int,
        frame_name: str,
    ) -> None:
        """[summary]
        Args:
            sensing_frame_config (SensingFrameConfig): The configuration of sensing evaluation.
            unix_time (int): Unix time [us]
            frame_name (str): The name of frame.
        """
        # Config
        self.sensing_frame_config = sensing_frame_config

        # Frame information
        self.unix_time: int = unix_time
        self.frame_name: str = frame_name

        # Containers for results
        self.detection_success_results: List[DynamicObjectWithSensingResult] = []
        self.detection_fail_results: List[DynamicObjectWithSensingResult] = []
        self.detection_warning_results: List[DynamicObjectWithSensingResult] = []
        self.pointcloud_failed_non_detection: List[np.ndarray] = []

    def evaluate_frame(
        self,
        ground_truth_objects: List[DynamicObject],
        pointcloud_for_detection: np.ndarray,
        pointcloud_for_non_detection: List[np.ndarray],
    ) -> None:
        """[summary]
        Evaluate each object per frame.

        Args:
            ground_truth_objects (list[DynamicObject]): The list of ground truth objects.
            pointcloud_for_detection (numpy.ndarray): The array of pointcloud for detection.
            pointcloud_for_non_detection (List[numpy.ndarray]):
                The array of pointcloud for non-detection.
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
        """[summary]
        Evaluate if pointcloud are detected.
        If the object is occluded, the result is appended to warning.

        Args:
            ground_truth_objects (list[DynamicObject]): The list of ground truth objects.
            pointcloud_for_detection (numpy.ndarray): The array of pointcloud for detection.
        """
        if len(ground_truth_objects) == 0:
            logging.warn("There is no annotated objects")
            return

        for ground_truth_object in ground_truth_objects:
            scale_factor_: float = self.sensing_frame_config.get_scale_factor(
                ground_truth_object.get_distance()
            )
            sensing_result: DynamicObjectWithSensingResult = DynamicObjectWithSensingResult(
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
        """[summary]
        Evaluate if pointcloud are not detected.

        Args:
            ground_truth_objects (list[DynamicObject]): The list of ground truth objects.
            pointcloud_for_non_detection (list[numpy.ndarray]):
                The list of pointcloud array for non-detection.
        """
        for point_non_detection in pointcloud_for_non_detection:
            for ground_truth_object in ground_truth_objects:
                # Get bbox scale factor
                scale_factor_: float = self.sensing_frame_config.get_scale_factor(
                    ground_truth_object.get_distance()
                )
                # Get object area, the start position is looped
                object_area_: List[List[float]] = ground_truth_object.get_corners(
                    scale_factor_
                ).tolist()
                # Convert object_area_ to tuple and make it un-looped
                object_area_: List[Tuple[float]] = [tuple(e) for e in object_area_]
                # Remove pointcloud in bounding boxes
                point_non_detection = crop_pointcloud(
                    point_non_detection,
                    object_area_,
                )
            if len(point_non_detection) != 0:
                self.pointcloud_failed_non_detection.append(point_non_detection)
