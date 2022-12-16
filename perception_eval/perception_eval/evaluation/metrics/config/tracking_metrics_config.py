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

from typing import List

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import LabelType

from ._metrics_config_base import _MetricsConfigBase


class TrackingMetricsConfig(_MetricsConfigBase):
    """[summary]
    The config for tracking evaluation metrics.

    Attributes:
        self.evaluation_task (EvaluationTask.TRACKING)
        self.target_labels (List[LabelType]): The list of targets to evaluate
        self.center_distance_thresholds (List[float]): The threshold list of center distance for matching
        self.plane_distance_thresholds (List[float]): The threshold list of plane distance for matching
        self.iou_bev_thresholds (List[float]): The threshold list of bev iou for matching
        self.iou_3d_thresholds (List[float]): The threshold list of 3d iou for matching
    """

    evaluation_task = [EvaluationTask.TRACKING, EvaluationTask.TRACKING2D]

    def __init__(
        self,
        target_labels: List[LabelType],
        center_distance_thresholds: List[List[float]],
        plane_distance_thresholds: List[List[float]],
        iou_2d_thresholds: List[List[float]],
        iou_3d_thresholds: List[List[float]],
    ) -> None:
        """[summary]
        Args:
            target_labels (List[LabelType]): The list of targets to evaluate.
            center_distance_thresholds (List[List[float]]):
                    The threshold List of center distance.
                    For example, if target_labels is ["car", "bike", "pedestrian"],
                    List[float] : [1.0, 0.5, 0.5] means
                    center distance threshold for a car is 1.0.
                    center distance threshold for a bike is 0.5.
                    center distance threshold for a pedestrian is 0.5.
            plane_distance_thresholds (List[List[float]]):
                    The mAP threshold of plane distance as map_thresholds_center_distance.
            iou_2d_thresholds (List[List[float])]:
                    The threshold List of BEV iou for matching as map_thresholds_center_distance.
            iou_3d_thresholds (List[List[float])]:
                    The threshold list of 3D iou for matching as map_thresholds_center_distance.
        """
        super().__init__(
            target_labels=target_labels,
            center_distance_thresholds=center_distance_thresholds,
            plane_distance_thresholds=plane_distance_thresholds,
            iou_2d_thresholds=iou_2d_thresholds,
            iou_3d_thresholds=iou_3d_thresholds,
        )
