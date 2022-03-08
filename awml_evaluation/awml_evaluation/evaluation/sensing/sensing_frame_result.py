from typing import List
from typing import Tuple

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
        self.non_detected_pointcloud: List[Tuple[float]] = []

    def evaluate_frame(
        self,
        ground_truth_objects: List[DynamicObject],
        pointcloud_for_detection: List[Tuple[float]],
        pointcloud_for_non_detection: List[Tuple[float]],
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
        pointcloud_for_detection: List[Tuple[float]],
    ):
        # [TODO] implement
        for ground_truth_object in ground_truth_objects:
            sensing_result: DynamicObjectWithSensingResult = DynamicObjectWithSensingResult(
                ground_truth_object,
                pointcloud_for_detection,
            )
            if sensing_result.pointcloud_exist(pointcloud_for_detection):
                self.detection_success_results.append(ground_truth_object)
            else:
                self.detection_fail_results.append(ground_truth_object)

    def _evaluate_pointcloud_for_non_detection(
        self,
        ground_truth_objects: List[DynamicObject],
        pointcloud_for_non_detection: List[Tuple[float]],
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
