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
from typing import Optional

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import LabelType
from perception_eval.common.threshold import set_thresholds


class _MetricsConfigBase(ABC):
    """Abstract base class of MetricsConfig for each evaluation task.

    Input thresholds are used like following.
    In case of `target_labels=["car", "bike", "pedestrian"]` and `thresholds=[1.0, 0.5, 0.5]`.
    Then, threshold for car objects is 1.0. threshold for bike objects is 0.5. threshold for pedestrian object is 0.5.

    Args:
        target_labels (List[LabelType]): Target labels list.
        center_distance_thresholds (List[List[float]]):
                Thresholds List of center distance. Defaults to None.
        plane_distance_thresholds (List[List[float]]):
                Thresholds list of plane distance. Defaults to None.
        iou_2d_thresholds (List[List[float])]:
                The threshold List of BEV iou for matching as map_thresholds_center_distance.
        iou_3d_thresholds (List[List[float])]:
                The threshold list of 3D iou for matching as map_thresholds_center_distance.
    """

    evaluation_task: EvaluationTask

    @abstractmethod
    def __init__(
        self,
        target_labels: List[LabelType],
        center_distance_thresholds: Optional[List[float]] = None,
        plane_distance_thresholds: Optional[List[float]] = None,
        iou_2d_thresholds: Optional[List[float]] = None,
        iou_3d_thresholds: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        self.target_labels: List[LabelType] = target_labels

        num_targets: int = len(target_labels)
        if center_distance_thresholds:
            self.center_distance_thresholds = set_thresholds(center_distance_thresholds, num_targets, True)
        else:
            self.center_distance_thresholds = []

        if plane_distance_thresholds:
            self.plane_distance_thresholds = set_thresholds(plane_distance_thresholds, num_targets, True)
        else:
            self.plane_distance_thresholds = []

        if iou_2d_thresholds:
            self.iou_2d_thresholds = set_thresholds(iou_2d_thresholds, num_targets, True)
        else:
            self.iou_2d_thresholds = []

        if iou_3d_thresholds:
            self.iou_3d_thresholds = set_thresholds(iou_3d_thresholds, num_targets, True)
        else:
            self.iou_3d_thresholds = []
