import datetime
from logging import getLogger
import os
from typing import List

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.label import LabelConverter

logger = getLogger(__name__)


class EvaluatorConfiguration:
    def __init__(
        self,
        result_root_directory: str,
        log_directory: str,
        visualization_directory: str,
        evaluation_tasks: List[str],
        target_labels: List[str],
        detection_thresholds_distance: List[float],
        detection_thresholds_iou3d: List[float],
    ) -> None:
        """[summary]

        Args:
            result_root_directory (str): The path to result directory
            log_directory (str): The path to sub directory for log
            visualization_directory (str): The path to sub directory for visualization
            evaluation_tasks (List[str]): Tasks for evaluation. Choose from common.EvaluationTask
                                          classes (ex. ["detection", "tracking", "prediction"])
            target_labels (List[str]): Target labels to evaluate. Choose from label
            detection_thresholds_distance (List[float]): The detection threshold of center
                                                         distance for matching
            detection_thresholds_iou3d (List[float]): The detection threshold of 3d iou
                                                      for matching
        """

        # directory
        time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        self.result_root_directory: str = result_root_directory.format(TIME=time)

        self.log_directory: str = log_directory
        self.visualization_directory: str = visualization_directory

        # task
        self.evaluation_tasks: List[EvaluationTask] = self._set_task_lists(evaluation_tasks)

        # label converter
        self.label_converter = LabelConverter()
        # self.target_labels: List[List[AutowareLabel]] = self._set_target_lists(target_labels)
        self.target_labels: List[AutowareLabel] = self._set_target_lists(
            target_labels, self.label_converter
        )

        # config for Evaluation
        self.detection_thresholds_distance: List[float] = detection_thresholds_distance
        self.detection_thresholds_iou3d: List[float] = detection_thresholds_iou3d

    def _set_task_lists(self, evaluation_tasks: List[str]) -> List[EvaluationTask]:
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
                logger.error(f"{evaluation_task} is not proper setting")
        return output

    @staticmethod
    def _set_target_lists(
        target_labels: List[str], label_converter: LabelConverter
    ) -> List[AutowareLabel]:
        """[summary]
        Set the target class configure

        Args:
            target_labels (List[str]): The target class to evaluate
            label_converter (LabelConverter): Label Converter class

        Returns:
            List[AutowareLabel]:  The list of target class
        """
        target_autoware_labels = []
        for target_label in target_labels:
            target_autoware_labels.append(label_converter.convert_label(target_label))
        return target_autoware_labels

    @property
    def result_log_directory(self) -> str:
        """[summary]
        Get the full path to put logs
        Returns:
            str: The full path to put logs
        """
        return os.path.join(self.result_root_directory, self.log_directory)

    @property
    def result_visualization_directory(self) -> str:
        """[summary]
        Get the full path to put the visualization images
        Returns:
            str: The full path to put the visualization images"""
        return os.path.join(self.result_root_directory, self.visualization_directory)
