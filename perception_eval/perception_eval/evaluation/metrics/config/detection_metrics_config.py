# Copyright 2022 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import LabelType
from perception_eval.common.label import TrafficLightLabel

from ._metrics_config_base import _MetricsConfigBase


class DetectionMetricsConfig(_MetricsConfigBase):
    """Configuration class for detection evaluation metrics.

    Attributes:
        evaluation_task (EvaluationTask.DETECTION)
        target_labels (List[LabelType]): Target labels list.
        center_distance_thresholds (List[float]): Thresholds list of center distance matching.
        center_distance_bev_thresholds (List[float]): Thresholds list of center distance bev matching.
        plane_distance_thresholds (List[float]): Threshold list of plane distance matching.
        iou_2d_thresholds (List[float]): Thresholds list of 2d iou matching.
        iou_3d_thresholds (List[float]): Thresholds list of 3d iou matching.

    Args:
        target_labels (List[LabelType]): Target labels list.
        center_distance_thresholds (List[float]): Thresholds list of center distance matching.
        center_distance_bev_thresholds (List[float]): Thresholds list of center distance bev matching.
        plane_distance_thresholds (List[float]): Threshold list of plane distance matching.
        iou_2d_thresholds (List[float]): Thresholds list of 2d iou matching.
        iou_3d_thresholds (List[float]): Thresholds list of 3d iou matching.
        advanced_detection_metrics (Optional[Dict[str, Any]]): Optional opt-in driving-aware metrics section
            (ranges, class_groups, filters, components, map). Absent means legacy mAP/mAPH behaviour only.
    """

    evaluation_task: EvaluationTask = [EvaluationTask.DETECTION, EvaluationTask.DETECTION2D]

    def __init__(
        self,
        target_labels: List[LabelType],
        center_distance_thresholds: Optional[List[float]] = None,
        center_distance_bev_thresholds: Optional[List[float]] = None,
        plane_distance_thresholds: Optional[List[float]] = None,
        iou_2d_thresholds: Optional[List[float]] = None,
        iou_3d_thresholds: Optional[List[float]] = None,
        advanced_detection_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            target_labels=target_labels,
            center_distance_thresholds=center_distance_thresholds,
            center_distance_bev_thresholds=center_distance_bev_thresholds,
            plane_distance_thresholds=plane_distance_thresholds,
            iou_2d_thresholds=iou_2d_thresholds,
            iou_3d_thresholds=iou_3d_thresholds,
        )
        # Raw section kept for pickling/serialization; the typed config is derived from it.
        self.advanced_detection_metrics_dict: Optional[Dict[str, Any]] = (
            dict(advanced_detection_metrics) if advanced_detection_metrics else None
        )
        self.advanced_detection_metrics = None
        if self.advanced_detection_metrics_dict is not None:
            from perception_eval.evaluation.metrics.detection.config import AdvancedDetectionMetricsConfig

            self.advanced_detection_metrics = AdvancedDetectionMetricsConfig.from_dict(
                self.advanced_detection_metrics_dict, target_labels
            )

    def __reduce__(self) -> Tuple[DetectionMetricsConfig, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling.

        The seventh positional argument is new; pickles produced before it existed carry six and
        therefore load with ``advanced_detection_metrics=None``.
        """
        return (
            self.__class__,
            (
                self.target_labels,
                self.center_distance_thresholds,
                self.center_distance_bev_thresholds,
                self.plane_distance_thresholds,
                self.iou_2d_thresholds,
                self.iou_3d_thresholds,
                self.advanced_detection_metrics_dict,
            ),
        )

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        data = super().serialization()
        data["advanced_detection_metrics"] = self.advanced_detection_metrics_dict
        return data

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> DetectionMetricsConfig:
        """Deserialize the data to DetectionMetricsConfig (old dicts without the new key load fine)."""
        target_labels = []
        for label in data["target_labels"]:
            label_type = label["label_type"]
            # `LABEL_TYPE` is an enum member; accept both the member and its string value.
            if label_type in (AutowareLabel.LABEL_TYPE, AutowareLabel.LABEL_TYPE.value):
                label_class = AutowareLabel
            elif label_type in (TrafficLightLabel.LABEL_TYPE, TrafficLightLabel.LABEL_TYPE.value):
                label_class = TrafficLightLabel
            else:
                raise ValueError(f"Invalid label type: {label_type}")
            target_labels.append(label_class.deserialization(label))
        return cls(
            target_labels=target_labels,
            center_distance_thresholds=data["center_distance_thresholds"],
            center_distance_bev_thresholds=data["center_distance_bev_thresholds"],
            plane_distance_thresholds=data["plane_distance_thresholds"],
            iou_2d_thresholds=data["iou_2d_thresholds"],
            iou_3d_thresholds=data["iou_3d_thresholds"],
            advanced_detection_metrics=data.get("advanced_detection_metrics"),
        )
