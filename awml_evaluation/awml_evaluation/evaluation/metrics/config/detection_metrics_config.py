from typing import List

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel

from ._metrics_config_base import _MetricsConfigBase


class DetectionMetricsConfig(_MetricsConfigBase):
    """[summary]
    The config for detection evaluation metrics.

    Attributes:
        self.evaluation_task (EvaluationTask.DETECTION)
        self.target_labels (List[AutowareLabel]): The list of targets to evaluate
        self.center_distance_thresholds (List[float]): The threshold list of center distance for matching
        self.plane_distance_thresholds (List[float]): The threshold list of plane distance for matching
        self.iou_bev_thresholds (List[float]): The threshold list of bev iou for matching
        self.iou_3d_thresholds (List[float]): The threshold list of 3d iou for matching
    """

    evaluation_task = EvaluationTask.DETECTION

    def __init__(
        self,
        target_labels: List[AutowareLabel],
        center_distance_thresholds: List[List[float]],
        plane_distance_thresholds: List[List[float]],
        iou_bev_thresholds: List[List[float]],
        iou_3d_thresholds: List[List[float]],
    ) -> None:
        """[summary]
        Args:
            target_labels (List[AutowareLabel]): The list of targets to evaluate.
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
            min_point_numbers (List[int]):
                    Min point numbers.
                    For example, if target_labels is ["car", "bike", "pedestrian"],
                    min_point_numbers [5, 0, 0] means
                    Car bboxes including 4 points are filtered out.
                    Car bboxes including 5 points are NOT filtered out.
                    Bike and Pedestrian bboxes are not filtered out(All bboxes are used when calculating metrics.)

        """
        super().__init__(
            target_labels=target_labels,
            center_distance_thresholds=center_distance_thresholds,
            plane_distance_thresholds=plane_distance_thresholds,
            iou_bev_thresholds=iou_bev_thresholds,
            iou_3d_thresholds=iou_3d_thresholds,
        )
