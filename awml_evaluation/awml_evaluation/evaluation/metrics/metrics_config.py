from typing import List
from typing import Optional

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel


class MetricsScoreConfig:
    """[summary]
    The config for evaluation metrics.

    Attributes:
        self.target_labels (List[AutowareLabel]): The list of targets to evaluate
        self.map_thresholds_center_distance (List[float]): The mAP threshold of center distance
        self.map_thresholds_plane_distance (List[float]): The mAP threshold of plane distance
        self.map_thresholds_iou_bev (List[float]): The mAP threshold of bev iou for matching
        self.map_thresholds_iou_3d (List[float]): The mAP threshold of 3d iou for matching
    """

    def __init__(
        self,
        target_labels: List[AutowareLabel],
        evaluation_tasks: List[EvaluationTask],
        max_x_position_list: List[float],
        max_y_position_list: List[float],
        map_thresholds_center_distance: List[List[float]],
        map_thresholds_plane_distance: List[List[float]],
        map_thresholds_iou_bev: List[List[float]],
        map_thresholds_iou_3d: List[List[float]],
    ) -> None:
        """[summary]
        Args:
            target_labels (List[AutowareLabel]): The list of targets to evaluate.
            evaluation_tasks: List[EvaluationTask]:
            max_x_position_list (List[float]):
                    The threshold list of maximum x-axis position for each object.
                    Return the object that
                    - max_x_position < object x-axis position < max_x_position.
                    This param use for range limitation of detection algorithm.
            max_y_position_list (List[float]):
                    The threshold list of maximum y-axis position for each object.
                    Return the object that
                    - max_y_position < object y-axis position < max_y_position.
                    This param use for range limitation of detection algorithm.
            map_thresholds_center_distance (List[List[float]]):
                    The mAP threshold of center distance.
                    For example, if target_labels is ["car", "bike", "pedestrian"],
                    List[float] : [1.0, 0.5, 0.5] means
                    center distance threshold for a car is 1.0.
                    center distance threshold for a bike is 0.5.
                    center distance threshold for a pedestrian is 0.5.
            map_thresholds_plane_distance (List[List[float]]):
                    The mAP threshold of plane distance as map_thresholds_center_distance.
            map_thresholds_iou_bev (List[List[float])]:
                    The mAP threshold of BEV iou for matching as map_thresholds_center_distance.
            map_thresholds_iou_3d (List[List[float])]:
                    The mAP threshold of 3D iou for matching as map_thresholds_center_distance.
        """
        self.target_labels: List[AutowareLabel] = target_labels
        self.evaluation_tasks: List[EvaluationTask] = evaluation_tasks

        self.max_x_position_list: List[float] = MetricsScoreConfig.set_thresholds(
            max_x_position_list,
            self.target_labels,
        )
        self.max_y_position_list: List[float] = MetricsScoreConfig.set_thresholds(
            max_y_position_list,
            self.target_labels,
        )

        # detection
        # mAP
        self.map_thresholds_center_distance: List[
            List[float]
        ] = MetricsScoreConfig.set_thresholds_list(
            map_thresholds_center_distance,
            self.target_labels,
        )
        self.map_thresholds_plane_distance: List[
            List[float]
        ] = MetricsScoreConfig.set_thresholds_list(
            map_thresholds_plane_distance,
            self.target_labels,
        )
        self.map_thresholds_iou_bev: List[List[float]] = MetricsScoreConfig.set_thresholds_list(
            map_thresholds_iou_bev,
            self.target_labels,
        )
        self.map_thresholds_iou_3d: List[List[float]] = MetricsScoreConfig.set_thresholds_list(
            map_thresholds_iou_3d,
            self.target_labels,
        )

    @staticmethod
    def set_thresholds(
        thresholds: List[Optional[float]],
        target_labels: List[AutowareLabel],
    ) -> List[Optional[float]]:
        """[summary]
        Check the config and set the thresholds.

        Args:
            thresholds (Optional[List[float]]): Thresholds
            target_labels (List[AutowareLabel]): Target labels

        Raises:
            UseCaseThresholdsError: Error for use case thresholds

        Returns:
            List[Optional[List[float]]]: A thresholds
        """
        if len(thresholds) != len(target_labels):
            raise MetricThresholdsError(
                "Error: Metrics threshold is not proper! \
                The length of the threshold is not same as target labels"
            )
        return thresholds

    @staticmethod
    def set_thresholds_list(
        thresholds_list: List[List[float]],
        target_labels: List[AutowareLabel],
    ) -> List[List[float]]:
        """[summary]
        Check the config and set the thresholds.

        Args:
            thresholds_list (List[List[float]]): A thresholds list
            target_labels (List[AutowareLabel]): Target labels

        Raises:
            MetricThresholdsError: Error for metrics thresholds

        Returns:
            List[List[float]]: A thresholds list
        """
        for thresholds in thresholds_list:
            if len(thresholds) != 0 and len(thresholds) != len(target_labels):
                raise MetricThresholdsError(
                    "Error: Metrics threshold is not proper! \
                    The length of the threshold is not same as target labels"
                )
        return thresholds_list


class MetricThresholdsError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)
