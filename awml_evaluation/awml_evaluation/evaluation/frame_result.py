from typing import List
from typing import Optional

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.metrics.configure import MetricsScoreConfig
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.object_result import DynamicObjectWithResult


class FrameResult:
    """[summary]
    The result for 1 frame (the pair of predicted objects and ground truth objects)

    Attributes:
        self.frame_name (str): The file name of frame in the datasets
        self.ground_truth_objects (List[DynamicObject]) : The ground truth objects for the frame
        self.pointcloud (Optional[List[float]]) : The pointcloud for the frame
        self.unix_time (int): The unix time for frame
        self.object_results (List[DynamicObjectWithResult]) : The results to each predicted object
        self.metrics_score (MetricsScore) : Metrics score results
    """

    def __init__(
        self,
        metrics_config: MetricsScoreConfig,
        unix_time: int,
        frame_name: str,
        pointcloud: Optional[List[float]] = None,
    ):
        """[summary]
        Args:
            metrics_config (MetricsConfiguration): Metrics config class
            unix_time (int): The unix time for frame
            frame_name (str): The file name of frame in the datasets
            pointcloud (Optional[List[float]], optional): The pointcloud for the frame.
                                                          Defaults to None.
        """

        # frame information
        self.unix_time: int = unix_time
        self.frame_name: str = frame_name
        self.pointcloud: Optional[List[float]] = pointcloud

        # results to each predicted object
        self.ground_truth_objects: List[DynamicObject] = []
        self.object_results: List[DynamicObjectWithResult] = []

        # init metrics score results
        self.metrics_score: MetricsScore = MetricsScore(metrics_config)

    def evaluate_frame(
        self,
        predicted_objects: List[DynamicObject],
        ground_truth_objects: List[DynamicObject],
    ) -> None:
        """[summary]
        Evaluate a frame from the pair of predicted objects and ground truth objects
        Args:
            predicted_objects (List[DynamicObject]): Predicted object which you want to evaluate
            ground_truth_objects (List[DynamicObject]): The ground truth objects in the same
                                                        time frame as predicted object
        """
        self.ground_truth_objects = ground_truth_objects
        # set object results information
        for predicted_object in predicted_objects:
            # caliculate to results for each object
            self.object_results.append(
                DynamicObjectWithResult(predicted_object, ground_truth_objects)
            )
        self.metrics_score.evaluate(self.object_results, ground_truth_objects)
