from typing import List

import numpy as np

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.sensing.sensing_frame_config import SensingFrameConfig
from awml_evaluation.evaluation.sensing.sensing_result import DynamicObjectWithSensingResult


class SensingFrameResult:
    def __init__(
        self,
        sensing_frame_config: SensingFrameConfig,
        unix_time: int,
        frame_name: str,
    ) -> None:
        # Config
        self.sensing_frame_config = sensing_frame_config

        # Frame information
        self.unix_time: int = unix_time
        self.frame_name: str = frame_name

        # Result
        self.detection_success_results: List[DynamicObjectWithSensingResult] = []
        self.detection_fail_results: List[DynamicObjectWithSensingResult] = []
        self.non_detected_pointcloud: List[np.ndarray] = []

    def evaluate_frame(
        self,
        ground_truth_objects: List[DynamicObject],
        pointcloud_for_detection: np.ndarray,
        pointcloud_for_non_detection: List[np.ndarray],
    ) -> None:
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
    ):
        # [TODO] implement
        for ground_truth_object in ground_truth_objects:
            sensing_result: DynamicObjectWithSensingResult = DynamicObjectWithSensingResult(
                ground_truth_object,
                pointcloud_for_detection,
            )
            if sensing_result.is_detected:
                self.detection_success_results.append(sensing_result)
            else:
                self.detection_fail_results.append(sensing_result)

    def _evaluate_pointcloud_for_non_detection(
        self,
        ground_truth_objects: List[DynamicObject],
        pointcloud_for_non_detection: List[np.ndarray],
    ):
        # [TODO] implement
        for point_non_detection in pointcloud_for_non_detection:
            is_non_detected = False
            for ground_truth_object in ground_truth_objects:
                is_non_detected = is_non_detected and ground_truth_object.point_exist(
                    point_non_detection
                )
            if not is_non_detected:
                self.non_detected_pointcloud.append(point_non_detection)
