from typing import List

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel


class MetricsScoreConfig:
    """[summary]
    The config for evaluation metrics

    Attributes:
        self.target_labels (List[AutowareLabel]): The list of targets to evaluate
        self.map_thresholds_center_distance (List[float]): The mAP threshold of center distance
        self.map_thresholds_plane_distance (List[float]): The mAP threshold of plane distance
        self.map_thresholds_iou (List[float]): The mAP threshold of 3d iou for matching
    """

    def __init__(
        self,
        target_labels: List[AutowareLabel],
        evaluation_tasks: List[EvaluationTask],
        map_thresholds_center_distance: List[float],
        map_thresholds_plane_distance: List[float],
        map_thresholds_iou: List[float],
    ) -> None:
        """[summary]
        Args:
            target_labels (List[AutowareLabel]): The list of targets to evaluate
            map_thresholds_center_distance (List[float]): The mAP threshold of center distance
            map_thresholds_plane_distance (List[float]): The mAP threshold of plane distance
            map_thresholds_iou (List[float]): The mAP threshold of 3d iou for matching
        """
        self.target_labels: List[AutowareLabel] = target_labels
        self.evaluation_tasks: List[EvaluationTask] = evaluation_tasks
        # detection
        # mAP
        self.map_thresholds_center_distance: List[float] = map_thresholds_center_distance
        self.map_thresholds_plane_distance: List[float] = map_thresholds_plane_distance
        self.map_thresholds_iou: List[float] = map_thresholds_iou
