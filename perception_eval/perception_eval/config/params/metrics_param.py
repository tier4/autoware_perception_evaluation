from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from .base import BaseParam

if TYPE_CHECKING:
    from perception_eval.common.evaluation_task import EvaluationTask
    from perception_eval.common.label import LabelType


__all__ = ("PerceptionMetricsParam", "SensingMetricsParam")


@dataclass
class PerceptionMetricsParam(BaseParam):
    evaluation_task: EvaluationTask
    target_labels: List[LabelType]
    center_distance_thresholds: Optional[Union[List[float], List[List[float]]]] = None
    plane_distance_thresholds: Optional[Union[List[float], List[List[float]]]] = None
    iou_2d_thresholds: Optional[Union[List[float], List[List[float]]]] = None
    iou_3d_thresholds: Optional[Union[List[float], List[List[float]]]] = None

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> PerceptionMetricsParam:
        evaluation_task: EvaluationTask = cfg["evaluation_task"]
        target_labels: list[LabelType] = cfg["target_labels"]
        center_distance_thresholds = cfg.get("center_distance_thresholds")
        plane_distance_thresholds = cfg.get("plane_distance_thresholds")
        iou_2d_thresholds = cfg.get("iou_2d_thresholds")
        iou_3d_thresholds = cfg.get("iou_3d_thresholds")
        return cls(
            evaluation_task,
            target_labels,
            center_distance_thresholds,
            plane_distance_thresholds,
            iou_2d_thresholds,
            iou_3d_thresholds,
        )


@dataclass
class SensingMetricsParam(BaseParam):
    box_scale_0m: float = 1.0
    box_scale_100m: float = 1.0
    min_points_threshold: int = 1

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> SensingMetricsParam:
        box_scale_0m = cfg.get("box_scale_0m", 1.0)
        box_scale_100m = cfg.get("box_scale_100m", 1.0)
        min_point_numbers = cfg.get("min_points_threshold", 1)
        return cls(box_scale_0m, box_scale_100m, min_point_numbers)
