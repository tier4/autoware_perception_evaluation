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
from typing import Optional

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import LabelType

from ._metrics_config_base import _MetricsConfigBase


class ClassificationMetricsConfig(_MetricsConfigBase):
    evaluation_task: EvaluationTask = [EvaluationTask.CLASSIFICATION2D]

    def __init__(
        self,
        target_labels: List[LabelType],
        center_distance_thresholds: Optional[List[float]] = None,
        plane_distance_thresholds: Optional[List[float]] = None,
        iou_2d_thresholds: Optional[List[float]] = None,
        iou_3d_thresholds: Optional[List[float]] = None,
    ) -> None:
        super().__init__(
            target_labels,
            center_distance_thresholds,
            plane_distance_thresholds,
            iou_2d_thresholds,
            iou_3d_thresholds,
        )