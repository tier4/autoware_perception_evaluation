from typing import List
from typing import Optional

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.metrics.metrics_config import MetricsScoreConfig
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithResult
from awml_evaluation.evaluation.result.pass_fail_result import CriticalObjectFilterConfig
from awml_evaluation.evaluation.result.pass_fail_result import FramePassFailConfig
from awml_evaluation.evaluation.result.pass_fail_result import PassFailResult


class FrameResult:
    """[summary]
    The result for 1 frame (the pair of predicted objects and ground truth objects)

    Attributes:
        self.frame_name (str): The file name of frame in the datasets
        self.ground_truth_objects (List[DynamicObject]): The ground truth objects for the frame
        self.pointcloud (Optional[List[float]]): The pointcloud for the frame
        self.unix_time (int): The unix time for frame [us]
        self.object_results (List[DynamicObjectWithResult]): The results to each predicted object
        self.metrics_score (MetricsScore): Metrics score results
        self.pass_fail_result (PassFailResult): Pass fail results
    """

    def __init__(
        self,
        metrics_config: MetricsScoreConfig,
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: FramePassFailConfig,
        unix_time: int,
        frame_name: str,
        pointcloud: Optional[List[float]] = None,
    ):
        """[summary]
        Args:
            metrics_config (MetricsConfiguration): Metrics config class
            critical_object_filter_config (CriticalObjectFilterConfig):
                    Critical object filter config.
            frame_pass_fail_config (FramePassFailConfig):
                    Frame pass fail config.
            unix_time (int): The unix time for frame [us]
            frame_name (str): The file name of frame in the datasets
            pointcloud (Optional[List[float]], optional):
                    The pointcloud for the frame. Defaults to None.
        """

        # frame information
        self.unix_time: int = unix_time
        self.frame_name: str = frame_name
        self.pointcloud: Optional[List[float]] = pointcloud

        # results to each predicted object
        self.ground_truth_objects: List[DynamicObject] = []
        self.object_results: List[DynamicObjectWithResult] = []

        # init evaluation
        self.metrics_score: MetricsScore = MetricsScore(metrics_config)
        self.pass_fail_result: PassFailResult = PassFailResult(
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
        )

    def evaluate_frame(
        self,
        predicted_objects: List[DynamicObject],
        ground_truth_objects: List[DynamicObject],
        ros_critical_ground_truth_objects: List[DynamicObject],
    ) -> None:
        """[summary]
        Evaluate a frame from the pair of predicted objects and ground truth objects
        Args:
            predicted_objects (List[DynamicObject]):
                    Predicted object which you want to evaluate
            ground_truth_objects (List[DynamicObject]):
                    The ground truth objects in the same time frame as predicted object
            ros_critical_ground_truth_objects (List[DynamicObject]):
                    Ground truth objects filtered by ROS node.
        """
        self.ground_truth_objects = ground_truth_objects
        self.object_results = FrameResult.get_object_results(
            predicted_objects,
            ground_truth_objects,
        )
        self.metrics_score.evaluate(
            object_results=self.object_results,
            ground_truth_objects=ground_truth_objects,
        )
        self.pass_fail_result.evaluate(
            object_results=self.object_results,
            ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
        )

    @staticmethod
    def get_object_results(
        predicted_objects: List[DynamicObject],
        ground_truth_objects: List[DynamicObject],
    ) -> List[DynamicObjectWithResult]:
        """[summary]
        Get object results from the pair of predicted_objects and ground_truth_objects in a frame.

        Args:
            predicted_objects (List[DynamicObject]):
                    Predicted object which you want to evaluate
            ground_truth_objects (List[DynamicObject]):
                    The ground truth objects in the same time frame as predicted object

        Returns:
            List[DynamicObjectWithResult]: List of Object results
        """

        object_results: List[DynamicObjectWithResult] = []
        for predicted_object in predicted_objects:
            # calculate to results for each object
            object_results.append(
                DynamicObjectWithResult(
                    predicted_object=predicted_object,
                    ground_truth_objects=ground_truth_objects,
                )
            )
        return object_results