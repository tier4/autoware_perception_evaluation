from typing import List
from typing import Tuple

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.dataset import load_all_datasets
from awml_evaluation.common.point import crop_pointcloud
from awml_evaluation.config.sensing_evaluation_config import SensingEvaluationConfig
from awml_evaluation.evaluation.sensing.sensing_frame_result import SensingFrameResult
import numpy as np

from ._evaluation_manager_base import _EvaluationMangerBase


class SensingEvaluationManager(_EvaluationMangerBase):
    """The class to manage sensing evaluation.

    Attributes:
        self.evaluator_config (SensingEvaluationConfig): Configuration for sensing evaluation.
        self.ground_truth_frames (List[FrameGroundTruth]): The list of ground truths per frame.
        self.frame_results (List[SensingFrameResult]): The list of sensing result per frame.
    """

    def __init__(
        self,
        evaluation_config: SensingEvaluationConfig,
    ) -> None:
        """
        Args:
            evaluation_config (SensingEvaluationConfig): The configuration for sensing evaluation.
        """
        super().__init__(evaluation_config)
        self.evaluator_config = evaluation_config
        self.ground_truth_frames: List[FrameGroundTruth] = load_all_datasets(
            dataset_paths=self.evaluator_config.dataset_paths,
            frame_id=self.evaluator_config.frame_id,
            does_use_pointcloud=self.evaluator_config.does_use_pointcloud,
            evaluation_task="sensing",
            label_converter=self.evaluator_config.label_converter,
            target_uuids=self.evaluator_config.target_uuids,
        )
        self.frame_results: List[SensingFrameResult] = []

    def add_frame_result(
        self,
        unix_time: int,
        ground_truth_now_frame: FrameGroundTruth,
        pointcloud_for_detection: np.ndarray,
        pointcloud_for_non_detection: List[np.ndarray],
    ) -> SensingFrameResult:
        """[summary]
        Get sensing result at current frame.
        The results is kept in self.frame_results.

        Args:
            unix_time (int)
            ground_truth_now_frame (FrameGroundTruth)
            pointcloud_for_detection (numpy.ndarray)
            pointcloud_for_non_detection (List[numpy.ndarray])

        Returns:
            result (SensingFrameResult)
        """
        result = SensingFrameResult(
            sensing_frame_config=self.evaluator_config.sensing_frame_config,
            unix_time=unix_time,
            frame_name=ground_truth_now_frame.frame_name,
        )
        result.evaluate_frame(
            ground_truth_objects=ground_truth_now_frame.objects,
            pointcloud_for_detection=pointcloud_for_detection,
            pointcloud_for_non_detection=pointcloud_for_non_detection,
        )
        self.frame_results.append(result)
        return result

    @staticmethod
    def crop_pointcloud(
        pointcloud: np.ndarray,
        non_detection_areas: List[List[Tuple[float, float, float]]],
    ) -> List[np.ndarray]:
        """Crop pointcloud from (N, 3) to (M, 3) with the non-detection area
        specified in SensingEvaluationConfig.

        Args:
            pointcloud (numpy.ndarray): The array of pointcloud, in shape (N, 3).
            non_detection_areas (List[List[Tuple[float, float, float]]]):
                The list of 3D-polygon areas for non-detection.

        Returns:
            cropped_pointcloud (List[numpy.ndarray]): The list of cropped pointcloud.
        """
        cropped_pointcloud = []
        for non_detection_area in non_detection_areas:
            cropped_pointcloud.append(
                crop_pointcloud(
                    pointcloud=pointcloud,
                    area=non_detection_area,
                )
            )
        return cropped_pointcloud
