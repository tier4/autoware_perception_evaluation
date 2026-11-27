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

from collections import Counter
from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional

from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetrics
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

from ._metrics_base import _TrackingMetricsBase


class HOTA(_TrackingMetricsBase):
    """HOTA metrics (Luiten et al., 2020) at a single matching threshold.

    HOTA = sqrt(DetA * AssA)
        DetA = TP / (TP + FN + FP)
        AssA = mean over TP of  TPA / (TPA + FNA + FPA), where for a TP pair (gt g, est e)
            TPA = #TP with (g, e), FNA = #gt detections of g not matched to e,
            FPA = #est detections of e not matched to g.

    shortcut: one threshold instead of averaging over alpha in 0.05..0.95, because the
    library thresholds are in the matching mode's own unit; average by running several
    thresholds. GT ids that are never matched are not in `object_results`, so they are
    missing from FNA (AssA slightly optimistic); DetA uses `num_ground_truth` and is exact.
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
        """
        NOTE: Index 0 of `object_results` is skipped, as in CLEAR.
        """
        super().__init__(
            num_ground_truth=num_ground_truth,
            target_labels=target_labels,
            matching_mode=matching_mode,
            matching_threshold_list=matching_threshold_list,
            tp_metrics=tp_metrics,
            metrics_field=metrics_field,
        )
        threshold = matching_threshold_list[0]

        gt_count: Counter = Counter()  # gt id -> #TP detections
        est_count: Counter = Counter()  # est id -> #detections (TP and FP)
        pair_count: Counter = Counter()  # (gt id, est id) -> #TP
        self.tp = 0
        self.fp = 0
        for frame in object_results[1:]:
            for res in frame:
                est_count[res.estimated_object.uuid] += 1
                if res.ground_truth_object is not None and res.is_result_correct(matching_mode, threshold):
                    self.tp += 1
                    gt_count[res.ground_truth_object.uuid] += 1
                    pair_count[(res.ground_truth_object.uuid, res.estimated_object.uuid)] += 1
                else:
                    self.fp += 1
        self.fn = max(0, num_ground_truth - self.tp)

        denom = self.tp + self.fn + self.fp
        self.det_a: float = self.tp / denom if denom else float("inf")
        if self.tp == 0:
            self.ass_a: float = float("inf") if denom == 0 else 0.0
        else:
            # sum over TP of TPA/(TPA+FNA+FPA) == sum over pairs of pair^2/(gt+est-pair)
            self.ass_a = (
                sum(n * n / (gt_count[g] + est_count[e] - n) for (g, e), n in pair_count.items()) / self.tp
            )
        self.hota: float = (
            float("inf") if self.det_a == float("inf") else (self.det_a * self.ass_a) ** 0.5
        )

    @property
    def results(self) -> Dict[str, float]:
        return OrderedDict({"HOTA": self.hota, "DetA": self.det_a, "AssA": self.ass_a})

    def _calculate_tp_fp(self, cur_object_results, prev_object_results):
        raise NotImplementedError("HOTA is computed over all frames at once.")
