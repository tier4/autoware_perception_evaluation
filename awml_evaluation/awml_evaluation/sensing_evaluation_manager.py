from typing import List
from typing import Tuple

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.dataset import get_now_frame
from awml_evaluation.common.dataset import load_all_datasets
from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.evaluation.sensing.sensing_frame_result import SensingFrameResult
from awml_evaluation.sensing_evaluation_config import SensingEvaluationConfig
from awml_evaluation.visualization.visualization import VisualizationBEV


class SensingEvaluationManager:
    def __init__(
        self,
        evaluation_config: SensingEvaluationConfig,
    ) -> None:
        self.evaluator_config: SensingEvaluationConfig = evaluation_config
        self.ground_truth_frames: List[FrameGroundTruth] = load_all_datasets(
            dataset_paths=self.evaluator_config.dataset_paths,
            does_use_pointcloud=False,
            evaluation_tasks=EvaluationTask.SENSING,
            label_converter=self.evaluator_config.label_converter,
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
        Get now frame of ground truth

        Args:
            unix_time (int): Unix time of frame to evaluate.
            threshold_min_time (int, optional):
                    Min time for unix time difference [us].
                    Default is 75000 usec = 75 ms.

        Returns:
            FrameGroundTruth: Now frame of ground truth
        """
        ground_truth_now_frame: FrameGroundTruth = get_now_frame(
            ground_truth_frames=self.ground_truth_frames,
            unix_time=unix_time,
            threshold_min_time=threshold_min_time,
        )
        return ground_truth_now_frame

    def add_sensing_result(
        self,
        unix_time: int,
        ground_truth_now_frame: FrameGroundTruth,
        pointcloud_for_detection: List[Tuple[float]],
        pointcloud_for_non_detection: List[Tuple[float]],
    ) -> SensingFrameResult:
        result = SensingFrameResult(
            sensing_frame_config=self.evaluator_config.sensing_frame_config,
            unix_time=unix_time,
            frame_name=ground_truth_now_frame.frame_name,
        )
        result.evaluate_frame(
            pointcloud_for_detection=pointcloud_for_detection,
            pointcloud_for_non_detection=pointcloud_for_non_detection,
        )
        self.frame_results.append(result)
        return result

    def visualize_bev_all(self) -> None:
        """[summary]
        Visualize objects and pointcloud from bird eye view.
        """

        for frame_result in self.frame_results:
            self.visualization.visualize_bev(
                ground_truth_objects=frame_result.ground_truth_objects,
                pointcloud=frame_result.pointcloud,
            )
