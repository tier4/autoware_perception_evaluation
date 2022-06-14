from typing import List

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel

from ._metrics_config_base import _MetricsConfigBase


class DetectionMetricsConfig(_MetricsConfigBase):
    """[summary]
    The config for detection evaluation metrics.

    Attributes:
        self.target_labels (List[AutowareLabel]): The list of targets to evaluate
        self.max_x_position_list (List[float])
        self.max_y_position_list (List[float])
        self.center_distance_thresholds (List[float]): The threshold list of center distance for matching
        self.plane_distance_thresholds (List[float]): The threshold list of plane distance for matching
        self.iou_bev_thresholds (List[float]): The threshold list of bev iou for matching
        self.iou_3d_thresholds (List[float]): The threshold list of 3d iou for matching
    """

    evaluation_task = EvaluationTask.DETECTION

    def __init__(
        self,
        target_labels: List[AutowareLabel],
        max_x_position: float,
        max_y_position: float,
        center_distance_thresholds: List[List[float]],
        plane_distance_thresholds: List[List[float]],
        iou_bev_thresholds: List[List[float]],
        iou_3d_thresholds: List[List[float]],
    ) -> None:
        """[summary]
        Args:
            target_labels (List[AutowareLabel]): The list of targets to evaluate.
            max_x_position (float):
                    The threshold of maximum x-axis position for each object.
                    Return the object that
                    - max_x_position < object x-axis position < max_x_position.
                    This param use for range limitation of detection algorithm.
            max_y_position (float):
                    The threshold list of maximum y-axis position for each object.
                    Return the object that
                    - max_y_position < object y-axis position < max_y_position.
                    This param use for range limitation of detection algorithm.
            center_distance_thresholds (List[List[float]]):
                    The threshold List of center distance.
                    For example, if target_labels is ["car", "bike", "pedestrian"],
                    List[float] : [1.0, 0.5, 0.5] means
                    center distance threshold for a car is 1.0.
                    center distance threshold for a bike is 0.5.
                    center distance threshold for a pedestrian is 0.5.
            plane_distance_thresholds (List[List[float]]):
                    The mAP threshold of plane distance as map_thresholds_center_distance.
            iou_bev_thresholds (List[List[float])]:
                    The threshold List of BEV iou for matching as map_thresholds_center_distance.
            iou_3d_thresholds (List[List[float])]:
                    The threshold list of 3D iou for matching as map_thresholds_center_distance.
        """
        super().__init__(
            target_labels=target_labels,
            max_x_position=max_x_position,
            max_y_position=max_y_position,
            center_distance_thresholds=center_distance_thresholds,
            plane_distance_thresholds=plane_distance_thresholds,
            iou_bev_thresholds=iou_bev_thresholds,
            iou_3d_thresholds=iou_3d_thresholds,
        )
