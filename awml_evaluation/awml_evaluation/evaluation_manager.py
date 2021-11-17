from typing import List

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.dataset import get_now_frame
from awml_evaluation.common.dataset import load_datasets
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.configure import EvaluatorConfiguration
from awml_evaluation.evaluation.frame_result import FrameResult
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.object_result import DynamicObjectWithResult
from awml_evaluation.visualization.visualization import VisualizationBEV


class EvaluationManager:
    """[summary]
    EvaluationManager class
    This class is management interface for perception interface.

    Attributes:
        self.evaluator_config (EvaluatorConfiguration): config for evaluation
        self.ground_truth_frames (List[FrameGroundTruth]): Ground truth frames from datasets
        self.frame_results (List[FrameResult]): Evaluation result
        self.visualization (VisualizationBEV): Visualization class

    """

    def __init__(
        self,
        dataset_path: str,
        does_use_pointcloud: bool,
        result_root_directory: str,
        log_directory: str,
        visualization_directory: str,
        evaluation_tasks: List[str],
        target_labels: List[str],
        map_thresholds_center_distance: List[float],
        map_thresholds_plane_distance: List[float],
        map_thresholds_iou: List[float],
    ) -> None:
        """[summary]

        Args:
            dataset_path (str): The path of dataset
            does_use_pointcloud (bool): The flag for loading pointcloud data from dataset
            result_root_directory (str): The path to result directory
            log_directory (str): The path to sub directory for log
            visualization_directory (str): The path to sub directory for visualization
            evaluation_tasks (List[str]): Tasks for evaluation. Choose from common.EvaluationTask
                                          classes (ex. ["detection", "tracking", "prediction"])
            target_labels (List[str]): Target labels to evaluate. Choose from label
            map_thresholds_distance (List[float]): The mAP detection threshold of center distance
                                                   for matching
            map_thresholds_iou3d (List[float]): The mAP detection threshold of 3d iou for matching
        """

        self.evaluator_config = EvaluatorConfiguration(
            result_root_directory,
            log_directory,
            visualization_directory,
            evaluation_tasks,
            target_labels,
            map_thresholds_center_distance,
            map_thresholds_plane_distance,
            map_thresholds_iou,
        )
        self.ground_truth_frames: List[FrameGroundTruth] = load_datasets(
            dataset_path,
            does_use_pointcloud,
            self.evaluator_config.metrics_config.evaluation_tasks,
            self.evaluator_config.label_converter,
        )
        self.frame_results: List[FrameResult] = []
        self.visualization: VisualizationBEV = VisualizationBEV(
            self.evaluator_config.visualization_directory
        )

    def add_frame_result(
        self,
        unix_time: int,
        predicted_objects: List[DynamicObject],
    ) -> FrameResult:
        """[summary]
        Evaluate one frame

        Args:
            unix_time (int): Unix time of frame to evaluate
            predicted_objects (List[DynamicObject]): Predicted object which you want to evaluate

        Returns:
            FrameResult: Evaluation result

        Example
            evaluator = EvaluationManager()
            predicted_objects : List[DynamicObject] = set_from_ros_topic(objects_from_topic)
            frame_result = evaluator.add_frame_result(
                unix_time,
                predicted_objects,
            )
            logger.debug(f"metrics result {frame_result.metrics_score}")
        """

        ground_truth_now_frame: FrameGroundTruth = get_now_frame(
            self.ground_truth_frames, unix_time
        )
        result = FrameResult(
            self.evaluator_config.metrics_config, unix_time, ground_truth_now_frame.frame_name
        )
        result.evaluate_frame(predicted_objects, ground_truth_now_frame.objects)
        self.frame_results.append(result)
        return result

    def get_scenario_result(self) -> MetricsScore:
        """[summary]
        Evaluate scenario

        Returns:
            MetricsScore: Metrics score

        Example
            evaluator = EvaluationManager()
            for frame in frames:
                predicted_objects : List[DynamicObject] = set_from_ros_topic(
                    frame.objects_from_topic
                )
                frame_result = evaluator.add_frame_result(
                    unix_time,
                    predicted_objects,
                )
            final_score = evaluator.get_scenario_result()
            logger.debug(f"Final result metrics {final_score}")
        """

        # gather objects from frame results
        all_frame_results: List[DynamicObjectWithResult] = []
        all_ground_truths: List[DynamicObject] = []
        for frame in self.frame_results:
            all_frame_results += frame.object_results
            all_ground_truths += frame.ground_truth_objects
        # calculate results
        scene_metrics_score: MetricsScore = MetricsScore(
            self.evaluator_config.metrics_config,
        )
        scene_metrics_score.evaluate(all_frame_results, all_ground_truths)
        return scene_metrics_score

    def visualize_bev_all(self) -> None:
        """[summary]
        Visualize objects and pointcloud from bird eye view.
        """

        for frame_result in self.frame_results:
            self.visualization.visualize_bev(
                frame_result.object_results,
                frame_result.ground_truth_objects,
            )
