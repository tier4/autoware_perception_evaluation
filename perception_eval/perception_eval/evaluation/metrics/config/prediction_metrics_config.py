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


class PredictionMetricsConfig(_MetricsConfigBase):
    """Configuration class for prediction evaluation metrics."""

    evaluation_task = EvaluationTask.PREDICTION

    def __init__(
        self,
        target_labels: List[LabelType],
        top_ks: List[int] = [1, 3, 6],
        miss_tolerance: float = 2.0,
        center_distance_thresholds: Optional[List[List[float]]] = None,
        plane_distance_thresholds: Optional[List[List[float]]] = None,
        iou_2d_thresholds: Optional[List[List[float]]] = None,
        iou_3d_thresholds: Optional[List[List[float]]] = None,
    ) -> None:
        """Construct a new object.

        Args:
            target_labels (List[LabelType]): List of target label names.
            top_ks (List[int], optional): List of top K modes to be evaluated. Defaults to [1, 3, 6].
            miss_tolerance (float, optional): Threshold value to determine miss. Defaults to 2.0.

        NOTE:
            `**_thresholds` are not used, just need to input.
        """
        super().__init__(
            target_labels=target_labels,
            center_distance_thresholds=center_distance_thresholds,
            plane_distance_thresholds=plane_distance_thresholds,
            iou_2d_thresholds=iou_2d_thresholds,
            iou_3d_thresholds=iou_3d_thresholds,
        )
        self.top_ks = top_ks
        self.miss_tolerance = miss_tolerance
