from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.threshold import set_thresholds

if TYPE_CHECKING:
    from perception_eval.common.label import LabelType

__all__ = ("PerceptionFilterParam", "SensingFilterParam")


class FilterParamBase(ABC):
    @classmethod
    @abstractmethod
    def from_dict(
        cls,
        cfg: Dict[str, Any],
        evaluation_task: EvaluationTask,
        target_labels: List[LabelType],
    ) -> FilterParamBase:
        pass

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerceptionFilterParam(FilterParamBase):
    evaluation_task: EvaluationTask
    target_labels: List[LabelType]
    max_x_position_list: Optional[List[float]] = None
    max_y_position_list: Optional[List[float]] = None
    min_distance_list: Optional[List[float]] = None
    max_distance_list: Optional[List[float]] = None
    min_point_numbers: Optional[List[float]] = None
    confidence_threshold_list: Optional[List[float]] = None
    target_uuids: Optional[List[str]] = None
    ignore_attributes: Optional[List[str]] = None

    @classmethod
    def from_dict(
        cls,
        cfg: Dict[str, Any],
        evaluation_task: EvaluationTask,
        target_labels: List[LabelType],
    ) -> PerceptionFilterParam:
        max_x_position: Optional[float] = cfg.get("max_x_position")
        max_y_position: Optional[float] = cfg.get("max_y_position")
        max_distance: Optional[float] = cfg.get("max_distance")
        min_distance: Optional[float] = cfg.get("min_distance")

        num_elements: int = len(target_labels)
        max_x_position_list = None
        max_y_position_list = None
        min_distance_list = None
        max_distance_list = None
        if max_x_position and max_y_position:
            max_x_position_list: List[float] = set_thresholds(max_x_position, num_elements, False)
            max_y_position_list: List[float] = set_thresholds(max_y_position, num_elements, False)
        elif max_distance and min_distance:
            max_distance_list: List[float] = set_thresholds(max_distance, num_elements, False)
            min_distance_list: List[float] = [min_distance] * len(target_labels)
        elif not evaluation_task.is_2d():
            raise RuntimeError("Either max x/y position or max/min distance should be specified")

        min_point_numbers: Optional[List[int]] = cfg.get("min_point_numbers")
        if min_point_numbers is not None:
            min_point_numbers = set_thresholds(min_point_numbers, num_elements, False)

        if evaluation_task == EvaluationTask.DETECTION and min_point_numbers is None:
            raise RuntimeError("In detection task, min point numbers must be specified")

        confidence_threshold: Optional[float] = cfg.get("confidence_threshold")
        if confidence_threshold is not None:
            confidence_threshold_list: List[float] = set_thresholds(confidence_threshold, num_elements, False)
        else:
            confidence_threshold_list = None

        target_uuids: Optional[List[str]] = cfg.get("target_uuids")
        ignore_attributes: Optional[List[str]] = cfg.get("ignore_attributes")

        return cls(
            evaluation_task,
            target_labels,
            max_x_position_list,
            max_y_position_list,
            min_distance_list,
            max_distance_list,
            min_point_numbers,
            confidence_threshold_list,
            target_uuids,
            ignore_attributes,
        )


@dataclass
class SensingFilterParam(FilterParamBase):
    target_uuids: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], **kwargs) -> SensingFilterParam:
        target_uuids: Optional[List[str]] = cfg.get("target_uuids")
        return cls(target_uuids)
