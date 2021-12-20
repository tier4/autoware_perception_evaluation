import datetime
from logging import getLogger
import os
from typing import List
from typing import Union

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.label import LabelConverter
from awml_evaluation.common.label import set_target_lists
from awml_evaluation.evaluation.metrics.metrics_config import MetricsScoreConfig

logger = getLogger(__name__)


class EvaluationConfigError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


class EvaluationConfig:
    """[summary]
    Evaluation configure class

    Attributes:
        self.result_root_directory (str): The path to result directory
        self.log_directory (str): The path to sub directory for log
        self.visualization_directory (str): The path to sub directory for visualization
        self.label_converter (LabelConverter): The label convert class
        self.metrics_config (MetricsScoreConfig): The config for metrics
    """

    def __init__(
        self,
        dataset_paths: List[str],
        does_use_pointcloud: bool,
        result_root_directory: str,
        log_directory: str,
        visualization_directory: str,
        evaluation_tasks: List[str],
        target_labels: List[str],
        max_x_position: float,
        max_y_position: float,
        map_thresholds_center_distance: Union[List[List[float]], List[float]],
        map_thresholds_plane_distance: Union[List[List[float]], List[float]],
        map_thresholds_iou_bev: Union[List[List[float]], List[float]],
        map_thresholds_iou_3d: Union[List[List[float]], List[float]],
    ) -> None:
        """[summary]

        Args:
            dataset_paths (List[str]): The paths of dataset
            does_use_pointcloud (bool): The flag for loading pointcloud data from dataset
            result_root_directory (str): The path to result directory
            log_directory (str): The path to sub directory for log
            visualization_directory (str): The path to sub directory for visualization
            evaluation_tasks (List[str]): Tasks for evaluation. Choose from common.EvaluationTask
                                          classes (ex. ["detection", "tracking", "prediction"])
            target_labels (List[str]): Target labels to evaluate.
            max_x_position (float): Maximum x-axis position for object. Return the object that
                                    - max_x_position < object x-axis position < max_x_position.
                                    This param use for range limitation of detection algorithm.
            max_y_position (float): Maximum y-axis position for object. Return the object that
                                    - max_y_position < object y-axis position < max_y_position.
            map_thresholds_center_distance (Union[List[List[float]], List[float]]):
                    The mAP threshold of center distance.
                    This parameter can be List[List[float]] or List[float]

                    1. List[List[float]]
                    List[List[float]] means mAP threshold list whose list has for each
                    object threshold.
                    For example, if target_labels is ["car", "bike", "pedestrian"],
                    List[List[float]] : [[1.0, 0.5, 0.5], [2.0, 1.0, 1.0] means
                    First mAP thresholds of center distance threshold for a car is 1.0m,
                    one for bike is 0.5m, and one for a pedestrian is 0.5m.
                    Second mAP thresholds are [2.0, 1.0, 1.0]

                    2. List[float]
                    List[float] means mAP thresholds when you use same parameter for each objects
                    For example, if target_labels is ["car", "bike", "pedestrian"],
                    List[float] : [0.5, 1.0, 2.0] means that mAP thresholds of
                    center distance threshold are 0.5m, 1.0m, and 2.0m.
                    The threshold apply to all object list.
            map_thresholds_plane_distance (Union[List[List[float]], List[float]]):
                    The mAP threshold of plane distance.
                    The specification is same as map_thresholds_center_distance
            map_thresholds_iou_bev (Union[List[List[float]], List[float]]]:
                    The mAP threshold of BEV iou for matching
                    The specification is same as map_thresholds_center_distance
            map_thresholds_iou_3d (Union[List[List[float]], List[float]]]:
                    The mAP threshold of 3D iou for matching
                    The specification is same as map_thresholds_center_distance
        """
        # dataset
        self.dataset_paths: List[str] = dataset_paths
        self.does_use_pointcloud: bool = does_use_pointcloud

        # directory
        time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        self.result_root_directory: str = result_root_directory.format(TIME=time)
        self.log_directory: str = log_directory
        self.visualization_directory: str = visualization_directory

        # Labels
        self.label_converter = LabelConverter()

        # Config for Metrics
        autoware_target_labels: List[AutowareLabel] = set_target_lists(
            target_labels,
            self.label_converter,
        )
        evaluation_tasks_: List[EvaluationTask] = EvaluationConfig._set_task_lists(
            evaluation_tasks
        )

        # Set for thresholds Union[List[List[float]], List[float]]
        map_thresholds_center_distance_ = EvaluationConfig._set_thresholds(
            map_thresholds_center_distance,
            len(autoware_target_labels),
        )
        map_thresholds_plane_distance_ = EvaluationConfig._set_thresholds(
            map_thresholds_plane_distance,
            len(autoware_target_labels),
        )
        map_thresholds_iou_bev_ = EvaluationConfig._set_thresholds(
            map_thresholds_iou_bev,
            len(autoware_target_labels),
        )
        map_thresholds_iou_3d_ = EvaluationConfig._set_thresholds(
            map_thresholds_iou_3d,
            len(autoware_target_labels),
        )

        self.metrics_config: MetricsScoreConfig = MetricsScoreConfig(
            target_labels=autoware_target_labels,
            evaluation_tasks=evaluation_tasks_,
            max_x_position_list=[max_x_position] * len(autoware_target_labels),
            max_y_position_list=[max_y_position] * len(autoware_target_labels),
            map_thresholds_center_distance=map_thresholds_center_distance_,
            map_thresholds_plane_distance=map_thresholds_plane_distance_,
            map_thresholds_iou_bev=map_thresholds_iou_bev_,
            map_thresholds_iou_3d=map_thresholds_iou_3d_,
        )

    @staticmethod
    def _set_task_lists(evaluation_tasks: List[str]) -> List[EvaluationTask]:
        """[summary]
        Convert str to EvaluationTask class

        Args:
            evaluation_tasks (List[str]): The tasks to evaluate

        Returns:
            List[EvaluationTask]: The tasks to evaluate
        """
        output = []
        for evaluation_task in evaluation_tasks:
            if evaluation_task == EvaluationTask.DETECTION.value:
                output.append(EvaluationTask.DETECTION)
            elif evaluation_task == EvaluationTask.TRACKING.value:
                output.append(EvaluationTask.TRACKING)
            elif evaluation_task == EvaluationTask.PREDICTION.value:
                output.append(EvaluationTask.PREDICTION)
            else:
                raise EvaluationConfigError(f"{evaluation_task} is not proper setting")
        return output

    def get_result_log_directory(self) -> str:
        """[summary]
        Get the full path to put logs
        Returns:
            str: The full path to put logs
        """
        return os.path.join(self.result_root_directory, self.log_directory)

    def get_result_visualization_directory(self) -> str:
        """[summary]
        Get the full path to put the visualization images
        Returns:
            str: The full path to put the visualization images"""
        return os.path.join(self.result_root_directory, self.visualization_directory)

    @staticmethod
    def _set_thresholds(
        thresholds: Union[List[List[float]], List[float]],
        target_objects_num: int,
    ) -> List[List[float]]:
        """[summary]
        Set List[List[float]] thresholds
        If the threshold is List[float], convert to List[List[float]].

        Args:
            thresholds (Union[List[List[float]], List[float]]): THresholds to convert
            target_objects_num (int): The number of targets

        Returns:
            List[List[float]]: Thresholds list

        Examples:
            _set_thresholds([1.0, 2.0], 3)
            # [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
            _set_thresholds([[1.0, 2.0], [3.0, 4.0]] 2)
            # [[1.0, 2.0], [3.0, 4.0]]
        """
        thresholds_list: List[List[float]] = []
        if len(thresholds) == 0:
            thresholds_list = [[]]
        elif isinstance(thresholds[0], float):
            for float_value in thresholds:
                thresholds_list.append([float_value] * target_objects_num)
        else:
            thresholds_list = thresholds
        return thresholds_list
