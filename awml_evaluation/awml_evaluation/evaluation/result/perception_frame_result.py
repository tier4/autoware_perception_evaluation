from typing import List

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.metrics.metrics_score_config import MetricsScoreConfig
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from awml_evaluation.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from awml_evaluation.evaluation.result.perception_pass_fail_result import PassFailResult


class PerceptionFrameResult:
    """[summary]
    The result for 1 frame (the pair of estimated objects and ground truth objects)

    Attributes:
        self.frame_name (str):
            The file name of frame in the datasets.
        self.unix_time (int):
            The unix time for frame [us].
        self.frame_ground_truth (FrameGroundTruth)
        self.object_results (List[DynamicObjectWithPerceptionResult]):
            The results to each estimated object.
        self.metrics_score (MetricsScore):
            Metrics score results.
        self.pass_fail_result (PassFailResult):
            Pass fail results.
    """

    def __init__(
        self,
        metrics_config: MetricsScoreConfig,
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
        unix_time: int,
        frame_ground_truth: FrameGroundTruth,
    ):
        """[summary]
        Args:
            metrics_config (MetricsScoreConfig): Metrics config class
            critical_object_filter_config (CriticalObjectFilterConfig):
                    Critical object filter config.
            frame_pass_fail_config (PerceptionPassFailConfig):
                    Frame pass fail config.
            unix_time (int): The unix time for frame [us]
            ground_truth_objects (FrameGroundTruth)
        """

        # frame information
        self.unix_time: int = unix_time
        self.frame_ground_truth: FrameGroundTruth = frame_ground_truth

        # results to each estimated object
        self.ground_truth_objects: List[DynamicObject] = []
        self.object_results: List[DynamicObjectWithPerceptionResult] = []

        # init evaluation
        self.metrics_score: MetricsScore = MetricsScore(metrics_config)
        self.pass_fail_result: PassFailResult = PassFailResult(
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
            frame_id=frame_ground_truth.frame_id,
            ego2map=frame_ground_truth.ego2map,
        )

    def evaluate_frame(
        self,
        estimated_objects: List[DynamicObject],
        ros_critical_ground_truth_objects: List[DynamicObject],
        previous_results: List[DynamicObjectWithPerceptionResult],
    ) -> None:
        """[summary]
        Evaluate a frame from the pair of estimated objects and ground truth objects
        Args:
            estimated_objects (List[DynamicObject]):
                    estimated object which you want to evaluate
            ros_critical_ground_truth_objects (List[DynamicObject]):
                    Ground truth objects filtered by ROS node.
            previous_results (List[DynamicObjectWithPerceptionResult]): The previous object results.
        """
        self.object_results = self.get_object_results(
            estimated_objects,
            self.frame_ground_truth.objects,
        )
        if self.metrics_score.detection_config is not None:
            self.metrics_score.evaluate_detection(
                object_results=[self.object_results],
                frame_ground_truths=[self.frame_ground_truth],
            )
        if self.metrics_score.tracking_config is not None:
            object_results = [previous_results, self.object_results]
            # Dummy FrameGroundTruth for The first one
            frame_ground_truths: List[FrameGroundTruth] = [None, self.frame_ground_truth]
            self.metrics_score.evaluate_tracking(
                object_results=object_results,
                frame_ground_truths=frame_ground_truths,
            )
        if self.metrics_score.prediction_config is not None:
            pass

        self.pass_fail_result.evaluate(
            object_results=self.object_results,
            ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
        )

    @staticmethod
    def get_object_results(
        estimated_objects: List[DynamicObject],
        ground_truth_objects: List[DynamicObject],
    ) -> List[DynamicObjectWithPerceptionResult]:
        """[summary]
        Get object results from the pair of estimated_objects and ground_truth_objects in a frame.

        Args:
            estimated_objects (List[DynamicObject]):
                    estimated object which you want to evaluate
            ground_truth_objects (List[DynamicObject]):
                    The ground truth objects in the same time frame as estimated object

        Returns:
            List[DynamicObjectWithPerceptionResult]: List of Object results
        """
        object_results: List[DynamicObjectWithPerceptionResult] = []
        for estimated_object in estimated_objects:
            # calculate to results for each object
            object_results.append(
                DynamicObjectWithPerceptionResult(
                    estimated_object=estimated_object,
                    ground_truth_objects=ground_truth_objects,
                )
            )
        return object_results
