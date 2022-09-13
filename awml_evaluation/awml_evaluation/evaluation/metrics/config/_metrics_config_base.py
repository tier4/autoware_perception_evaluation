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

from abc import ABC
from abc import abstractmethod
from typing import List

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.threshold import check_thresholds_list
from awml_evaluation.common.threshold import set_thresholds


class _MetricsConfigBase(ABC):

    evaluation_task: EvaluationTask

    @abstractmethod
    def __init__(
        self,
        target_labels: List[AutowareLabel],
        center_distance_thresholds,
        plane_distance_thresholds,
        iou_bev_thresholds,
        iou_3d_thresholds,
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
        """
        super().__init__()

        self.target_labels: List[AutowareLabel] = target_labels

        center_distance_thresholds_ = set_thresholds(
            center_distance_thresholds,
            len(target_labels),
        )
        plane_distance_thresholds_ = set_thresholds(
            plane_distance_thresholds,
            len(target_labels),
        )
        iou_bev_thresholds_ = set_thresholds(
            iou_bev_thresholds,
            len(target_labels),
        )
        iou_3d_thresholds_ = set_thresholds(
            iou_3d_thresholds,
            len(target_labels),
        )

        self.center_distance_thresholds: List[List[float]] = check_thresholds_list(
            center_distance_thresholds_,
            self.target_labels,
            MetricThresholdsError,
        )
        self.plane_distance_thresholds: List[List[float]] = check_thresholds_list(
            plane_distance_thresholds_,
            self.target_labels,
            MetricThresholdsError,
        )
        self.iou_bev_thresholds: List[List[float]] = check_thresholds_list(
            iou_bev_thresholds_,
            self.target_labels,
            MetricThresholdsError,
        )
        self.iou_3d_thresholds: List[List[float]] = check_thresholds_list(
            iou_3d_thresholds_,
            self.target_labels,
            MetricThresholdsError,
        )


class MetricThresholdsError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)
