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

from typing import Any
from typing import List
from typing import Optional

from perception_eval.common.label import LabelType
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetrics
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp

from ._metrics_base import _TrackingMetricsBase


class HOTA(_TrackingMetricsBase):
    """==== TODO ====
    HOTA metrics class that has HOTA, LocA, DetA and AssA as sub metrics.

    Attributes:
        self.target_labels (List[LabelType]): The list of target label.
        self.matching_mode (MatchingMode): The target matching mode.
        self.metrics_field (Optional[List[str]]): The list of target metrics name. If not specified, set default supported metrics.
        self.ground_truth_objects_num (int): The number of ground truth.
        self.tp_metrics (TPMetrics): The way of calculating TP value.
        self.support_metrics (List[str]): The list of supported metrics name.
    """

    _support_metrics: List[str] = ["HOTA", "DetA", "AssA"]

    def __init__(
        self,
        object_results: List[List[DynamicObjectWithPerceptionResult]],
        num_ground_truth: int,
        target_labels: List[LabelType],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
        tp_metrics: TPMetrics = TPMetricsAp(),
        metrics_field: Optional[List[str]] = None,
    ) -> None:
        """[summary]
        HOTA metrics.

        NOTE: objects_results, ground_truth_objects
            If evaluate 1-frame, index 0 is previous object results.
            If evaluate all frames, index 0 is empty list.

        Args:
            object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object results for each frames.
            num_ground_truth (int): The number of ground truth.
            target_labels (List[LabelType]): The list of target labels.
            matching_mode: (MatchingMode): Matching mode class.
            tp_metrics (TPMetrics): The way of calculating TP value. Defaults to TPMetricsAP.
            metrics_field: List[str]: The list of target sub metrics.
        """
        super().__init__(
            num_ground_truth=num_ground_truth,
            target_labels=target_labels,
            matching_mode=matching_mode,
            matching_threshold_list=matching_threshold_list,
            tp_metrics=tp_metrics,
            metrics_field=metrics_field,
        )

        self.tp: float = 0.0
        self.fp: float = 0.0
        self.tpa: float = 0.0
        self.fpa: float = 0.0
        self.tp_matching_score: float = 0.0

        num_frame: int = len(object_results)
        for i in range(1, num_frame):
            # Comparing matching pair
            tp_t, fp_t, tpa_t, fpa_t = self._calculate_tp_fp(
                cur_object_results=object_results[i],
                prev_object_results=object_results[i - 1],
            )
            self.tp += tp_t
            self.fp += fp_t
            self.tpa += tpa_t
            self.fpa += fpa_t

    def _calculate_tp_fp(
        self,
        cur_object_results: List[DynamicObjectWithPerceptionResult],
        prev_object_results: List[DynamicObjectWithPerceptionResult],
    ) -> Any:
        """Calculate TP/FP and TPA/FPA

        Args:
            cur_object_results
            prev_object_results (List[DynamicObjectWithPerceptionResult])
            prev_object_results (List[DynamicObjectWithPerceptionResult])
            matching_threshold (float)
        Returns:
            tp_list (List[float])
            fp_list (List[float])
            tpa_list (List[float])
            fpa_list (List[float])
        """
        pass
