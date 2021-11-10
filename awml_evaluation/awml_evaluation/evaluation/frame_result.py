import os
from typing import List
from typing import Optional

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.configure import EvaluatorConfiguration
from awml_evaluation.evaluation.matching.matching import MatchingMode
from awml_evaluation.evaluation.matching.matching import divide_tp_fp_objects
from awml_evaluation.evaluation.matching.matching import filter_tp_objects
from awml_evaluation.evaluation.matching.matching import get_fn_objects
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.object_result import DynamicObjectWithResult
from awml_evaluation.visualization.visualization import Color
from awml_evaluation.visualization.visualization import VisualizationBEV


class FrameResult:
    """[summary]
    The result for 1 frame (the pair of predicted objects and ground truth objects)

    Attributes:
        self.config (EvaluatorConfiguration): Config class
        self.unix_time (int): The unix time for frame
        self.frame_name (str): The file name of frame in the datasets
        self.pointcloud (Optional[List[float]]) : The pointcloud for the frame
        self.ground_truth_objects (List[DynamicObject]) : The ground truth objects for the frame
        self.object_results (List[DynamicObjectWithResult]) : The results to each predicted object
        self.visualization (VisualizationBEV): Visualization interface
        self.metrics_score (MetricsScore) : Metrics score results
    """

    def __init__(
        self,
        config: EvaluatorConfiguration,
        unix_time: int,
        frame_name: str,
        pointcloud: Optional[List[float]] = None,
    ):
        """[summary]

        Args:
            config (EvaluatorConfiguration): Config class
            unix_time (int): The unix time for frame
            frame_name (str): The file name of frame in the datasets
            pointcloud (Optional[List[float]], optional): The pointcloud for the frame.
                                                          Defaults to None.
        """

        # setting
        self.config: EvaluatorConfiguration = config

        # frame information
        self.unix_time: int = unix_time
        self.frame_name: str = frame_name
        self.pointcloud: Optional[List[float]] = pointcloud

        # results to each predicted object
        self.ground_truth_objects: List[DynamicObject] = []
        self.object_results: List[DynamicObjectWithResult] = []

        # init visualization
        self.visualization = VisualizationBEV()

        # init metrics score results
        self.metrics_score: MetricsScore = MetricsScore(
            config.target_labels,
            config.detection_thresholds_distance,
            config.detection_thresholds_iou3d,
        )

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

    def visualize_bev(self, file_name: Optional[str] = None):
        """[summary]
        Visualize the frame result from bird eye view

        Args:
            file_name (Optional[str]): File name. Defaults to None.
        """

        if file_name is None:
            file_name_: str = f"{self.unix_time}_{self.frame_name}.png"
        else:
            file_name_: str = file_name
        file_path: str = os.join("bev_pictures", file_name_)
        full_path: str = os.join(self.config.result_root_directory, file_path)

        filtered_predicted_objects: List[DynamicObjectWithResult] = filter_tp_objects(
            object_results=self.object_results,
            target_labels=self.config.target_labels[0],
            matching_mode=MatchingMode.CENTERDISTANCE,
            matching_threshold=self.config.detection_thresholds_distance,
        )
        tp_objects, fp_objects = divide_tp_fp_objects(filtered_predicted_objects)
        fn_objects = get_fn_objects(tp_objects, self.ground_truth_objects)

        self.visualization.visualize_bev(
            full_path,
            self.pointcloud,
            [tp_objects, fp_objects, fn_objects, self.ground_truth_objects],
            [Color.GREEN, Color.YELLOW, Color.RED, Color.WHITE],
            [4.0, 4.0, 4.0, 1.0],
        )
