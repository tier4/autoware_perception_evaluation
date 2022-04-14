from typing import List
from typing import Tuple

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.dataset import get_now_frame
from awml_evaluation.common.dataset import load_all_datasets
from awml_evaluation.common.point import crop_pointcloud
from awml_evaluation.evaluation.sensing.sensing_frame_result import SensingFrameResult
from awml_evaluation.sensing_evaluation_config import SensingEvaluationConfig
from awml_evaluation.visualization.visualization import VisualizationBEV
import numpy as np


class SensingEvaluationManager:
    """The class to manage sensing evaluation.

    Attributes:
        self.evalator_config (SensingEvaluationConfig): Configuration for sensing evaluation.
        self.ground_truth_frames (List[FrameGroundTruth]): The list of ground truths per frame.
        self.frame_results (List[SensingFrameResult]): The list of sensing result per frame.
        self.visualization (VisualizationBEV): The visualizor.
    """

    def __init__(
        self,
        evaluation_config: SensingEvaluationConfig,
    ) -> None:
        """
        Args:
            evaluation_config (SensingEvaluationConfig): The configuration for sensing evaluation.
        """
        self.evaluator_config: SensingEvaluationConfig = evaluation_config
        self.ground_truth_frames: List[FrameGroundTruth] = load_all_datasets(
            dataset_paths=self.evaluator_config.dataset_paths,
            does_use_pointcloud=self.evaluator_config.does_use_pointcloud,
            evaluation_tasks=["sensing"],
            label_converter=self.evaluator_config.label_converter,
            target_uuids=self.evaluator_config.target_uuids,
        )
        self.frame_results: List[SensingFrameResult] = []
        self.visualization: VisualizationBEV = VisualizationBEV(
            self.evaluator_config.visualization_directory
        )

    def get_ground_truth_now_frame(
        self,
        unix_time: int,
        threshold_min_time: int = 75000,
    ) -> FrameGroundTruth:
        """[summary]
        Get ground truth at current frame.

        Args:
            unix_time (int): Unix time of frame to evaluate.
            threshold_min_time (int, optional):
                    Min time for unix time difference [us].
                    Default is 75000 usec = 75 ms.

        Returns:
            FrameGroundTruth: Ground truth at current frame.
        """
        ground_truth_now_frame: FrameGroundTruth = get_now_frame(
            ground_truth_frames=self.ground_truth_frames,
            unix_time=unix_time,
            threshold_min_time=threshold_min_time,
        )
        return ground_truth_now_frame

    def add_sensing_frame_result(
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

    def visualize_bev_all(self) -> None:
        """[summary]
        Visualize objects and pointcloud from bird eye view.
        """
        # TODO
        # for frame_result in self.frame_results:
        #     self.visualization.visualize_bev(
        #         ground_truth_objects,
        #         pointcloud,
        #     )
        pass

    def crop_pointcloud(
        self,
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
            cropped_pointcloud (List[numpy.ndarry]): The list of cropped pointcloud.
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
