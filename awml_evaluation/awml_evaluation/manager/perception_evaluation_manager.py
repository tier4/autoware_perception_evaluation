from typing import List

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.dataset import load_all_datasets
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.config.perception_evaluation_config import PerceptionEvaluationConfig
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.evaluation.result.perception_pass_fail_result import CriticalObjectFilterConfig
from awml_evaluation.evaluation.result.perception_pass_fail_result import PerceptionPassFailConfig

from ._evaluation_manager_base import _EvaluationMangerBase


class PerceptionEvaluationManager(_EvaluationMangerBase):
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
        super().__init__(evaluation_config=evaluation_config)
        """[summary]

        Args:
            evaluation_config (EvaluatorConfig): Evaluation config
        """
        self.evaluator_config = evaluation_config
        self.ground_truth_frames: List[FrameGroundTruth] = load_all_datasets(
            dataset_paths=self.evaluator_config.dataset_paths,
            frame_id=self.evaluator_config.frame_id,
            does_use_pointcloud=self.evaluator_config.does_use_pointcloud,
            evaluation_task=self.evaluator_config.metrics_config.evaluation_task,
            label_converter=self.evaluator_config.label_converter,
        )
        self.frame_results: List[PerceptionFrameResult] = []

    def add_frame_result(
        self,
        unix_time: int,
        ground_truth_now_frame: FrameGroundTruth,
        estimated_objects: List[DynamicObject],
        ros_critical_ground_truth_objects: List[DynamicObject],
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
    ) -> PerceptionFrameResult:
        """[summary]
        Evaluate one frame

        Args:
            unix_time (int): Unix time of frame to evaluate [us]
            ground_truth_now_frame (FrameGroundTruth): Now frame ground truth
            estimated_objects (List[DynamicObject]): estimated object which you want to evaluate
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
            frame_ground_truth=ground_truth_now_frame,
        )
        previous_results: List[DynamicObjectWithPerceptionResult]
        if self.frame_results:
            previous_results = self.frame_results[-1].object_results
        else:
            previous_results = []

        result.evaluate_frame(
            estimated_objects=estimated_objects,
            ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
            previous_results=previous_results,
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
                estimated_objects : List[DynamicObject] = set_from_ros_topic(
                    frame.objects_from_topic
                )
                frame_result = evaluator.add_frame_result(
                    unix_time,
                    estimated_objects,
                )
            final_score = evaluator.get_scenario_result()
            logger.debug(f"Final result metrics {final_score}")
        """

        # gather objects from frame results
        # NOTE: The first object result is empty list, frame ground truth is None.
        all_frame_results: List[List[DynamicObjectWithPerceptionResult]] = [[]]
        all_ground_truths: List[FrameGroundTruth] = [None]
        for frame in self.frame_results:
            all_frame_results.append(frame.object_results)
            all_ground_truths.append(frame.frame_ground_truth)
        # calculate results
        scene_metrics_score: MetricsScore = MetricsScore(
            self.evaluator_config.metrics_config,
        )
        if self.evaluator_config.metrics_config.detection_config is not None:
            scene_metrics_score.evaluate_detection(all_frame_results, all_ground_truths)
        if self.evaluator_config.metrics_config.tracking_config is not None:
            scene_metrics_score.evaluate_tracking(all_frame_results, all_ground_truths)
        if self.evaluator_config.metrics_config.prediction_config is not None:
            pass

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
