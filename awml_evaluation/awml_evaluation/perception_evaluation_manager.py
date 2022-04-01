from typing import List

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.dataset import get_now_frame
from awml_evaluation.common.dataset import load_all_datasets
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.evaluation.result.perception_pass_fail_result import CriticalObjectFilterConfig
from awml_evaluation.evaluation.result.perception_pass_fail_result import PerceptionPassFailConfig
from awml_evaluation.perception_evaluation_config import PerceptionEvaluationConfig
from awml_evaluation.visualization.visualization import VisualizationBEV


class PerceptionEvaluationManager:
    """[summary]
    PerceptionEvaluationManager class.
    This class is management interface for perception interface.

    Attributes:
        self.evaluator_config (EvaluatorConfig): config for evaluation
        self.ground_truth_frames (List[FrameGroundTruth]): Ground truth frames from datasets
        self.frame_results (List[PerceptionFrameResult]): Evaluation result
        self.visualization (VisualizationBEV): Visualization class
    """

    def __init__(
        self,
        evaluation_config: PerceptionEvaluationConfig,
    ) -> None:
        """[summary]

        Args:
            evaluation_config (EvaluatorConfig): Evaluation config
        """
        self.evaluator_config: PerceptionEvaluationConfig = evaluation_config
        self.ground_truth_frames: List[FrameGroundTruth] = load_all_datasets(
            self.evaluator_config.dataset_paths,
            self.evaluator_config.does_use_pointcloud,
            self.evaluator_config.metrics_config.evaluation_tasks,
            self.evaluator_config.label_converter,
        )
        self.frame_results: List[PerceptionFrameResult] = []
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

    def add_perception_frame_result(
        self,
        unix_time: int,
        ground_truth_now_frame: FrameGroundTruth,
        predicted_objects: List[DynamicObject],
        ros_critical_ground_truth_objects: List[DynamicObject],
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
    ) -> PerceptionFrameResult:
        """[summary]
        Evaluate one frame

        Args:
            unix_time (int): Unix time of frame to evaluate [us]
            ground_truth_now_frame (FrameGroundTruth): Now frame ground truth
            predicted_objects (List[DynamicObject]): Predicted object which you want to evaluate
            ros_critical_ground_truth_objects (List[DynamicObject]):
                    Critical ground truth objects filtered by ROS node to evaluate pass fail result
            critical_object_filter_config (CriticalObjectFilterConfig):
                    The parameter config to choose critical ground truth objects
            frame_pass_fail_config (PerceptionPassFailConfig):
                    The parameter config to evaluate

        Returns:
            PerceptionFrameResult: Evaluation result
        """

        result = PerceptionFrameResult(
            metrics_config=self.evaluator_config.metrics_config,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
            unix_time=unix_time,
            frame_name=ground_truth_now_frame.frame_name,
            pointcloud=ground_truth_now_frame.pointcloud,
        )
        result.evaluate_frame(
            predicted_objects=predicted_objects,
            ground_truth_objects=ground_truth_now_frame.objects,
            ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
        )
        self.frame_results.append(result)
        return result

    def get_scene_result(self) -> MetricsScore:
        """[summary]
        Evaluate scenario

        Returns:
            MetricsScore: Metrics score

        Example
            evaluator = PerceptionEvaluationManager()
            for frame in frames:
                # write in your application
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
        all_frame_results: List[DynamicObjectWithPerceptionResult] = []
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
