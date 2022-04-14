from typing import List
from typing import Tuple

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.point import crop_pointcloud
from awml_evaluation.evaluation.sensing.sensing_frame_config import SensingFrameConfig
from awml_evaluation.evaluation.sensing.sensing_result import DynamicObjectWithSensingResult
import numpy as np


class SensingFrameResult:
    """[summary]
    The result for 1 frame (the pair of detected points and ground truth)

    Attributes:
        self.sensing_frame_config (SensingFrameConfig):
            The configuration of sensing evaluation.
        self.unix_time (int): Unix time [us]
        self.frame_name (str): The name of frame.
        self.detection_success_results (list[DynamicObjectWithSensingResult]):
            The container for succeded results of detection.
        self.detection_fail_results (list[DynamicObjectWithSensingResult]):
            The container for failed results of detection.
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

        Args:
            ground_truth_objects (list[DynamicObject]): The list of ground truth objects.
            pointcloud_for_detection (numpy.ndarray): The array of pointcloud for detection.
        """
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
            if sensing_result.is_detected:
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
