# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/collision_weighted_map.py) at fcf86419.

"""Collision-risk-weighted mAP (metric B2).

Every object's contribution to precision and recall is weighted by ``w = e^(-decay * TTC)``, with
TTC the reachability time-to-collision to ego computed once per frame by the suite's collision
provider. An object that cannot be reached within the horizon has TTC = inf and weight 0, so a
same-speed lead or off-road scenery buys no score, while an imminent in-path object dominates it.

Matching is the usual per-frame greedy center distance, only the precision and recall accumulation
is weighted. Reported alongside the unweighted mAP, never instead.
"""

from __future__ import annotations

from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.component import mean_valid
from perception_eval.evaluation.metrics.detection.criticality import greedy_match_thresholds
from perception_eval.evaluation.metrics.detection.criticality import weighted_average_precision
from perception_eval.evaluation.metrics.detection.state import DEFAULT_MATCH_THRESHOLDS
from perception_eval.evaluation.metrics.detection.state import DetectionState
from perception_eval.evaluation.metrics.geometry.reachability import collision_weights
from perception_eval.evaluation.metrics.naming import label_metric_name


class CollisionWeightedMeanAP:
    """Class-mean AP with each object weighted by its reachability collision risk."""

    needs_ttc = True

    def __init__(
        self,
        thresholds: Sequence[float] = DEFAULT_MATCH_THRESHOLDS,
        decay: float = 0.5,
    ) -> None:
        """Validate the risk decay.

        Args:
            thresholds (Sequence[float]): Center-distance match thresholds in meters, the weighted AP
                is averaged over them.
            decay (float): Exponential risk decay per second of TTC, must be non-negative.
        """
        self.thresholds: Tuple[float, ...] = tuple(float(threshold) for threshold in thresholds)
        if not self.thresholds:
            raise ValueError("thresholds must not be empty.")
        self.decay = float(decay)
        if self.decay < 0.0:
            raise ValueError("decay must be >= 0.")

    def evaluate(self, state: DetectionState) -> Dict[str, float]:
        """Weighted AP per class, averaged over thresholds, then over classes."""
        labels = state.labels(full=True)
        total_gt_weight = {label: 0.0 for label in labels}
        scores: Dict[Tuple[int, float], List[NDArray]] = {
            (label, threshold): [] for label in labels for threshold in self.thresholds
        }
        weights: Dict[Tuple[int, float], List[NDArray]] = {key: [] for key in scores}
        is_tp: Dict[Tuple[int, float], List[NDArray]] = {key: [] for key in scores}
        for sample in state.samples:
            if not sample.has_ttc:
                raise ValueError(
                    "CollisionWeightedMeanAP needs per-box TTC, build the suite with a collision provider."
                )
            # A frame without a lanelet map has no TTC basis: excluded from the weighted mass.
            if not sample.ttc_covered:
                continue
            gt_centers = sample.gt_boxes[:, :2]
            gt_weight_all = collision_weights(sample.gt_ttc, self.decay)
            pred_centers = sample.pred_boxes[:, :2]
            pred_weight_all = collision_weights(sample.pred_ttc, self.decay)
            for label in labels:
                gt_mask = sample.gt_labels == label
                gt_weight = gt_weight_all[gt_mask]
                total_gt_weight[label] += float(gt_weight.sum())

                pred_mask = sample.pred_labels == label
                pred_scores = sample.pred_scores[pred_mask]
                if pred_scores.shape[0] == 0:
                    continue
                matches = greedy_match_thresholds(
                    gt_centers[gt_mask], pred_centers[pred_mask], pred_scores, self.thresholds
                )
                for threshold, (tp, matched) in matches.items():
                    # A true positive inherits its matched GT's weight, a false positive keeps its own.
                    weight = pred_weight_all[pred_mask].copy()
                    weight[tp] = gt_weight[matched[tp]]
                    scores[(label, threshold)].append(pred_scores)
                    weights[(label, threshold)].append(weight)
                    is_tp[(label, threshold)].append(tp)

        per_class: Dict[int, float] = {}
        for label in labels:
            per_class[label] = mean_valid(
                [
                    self._assemble_ap(
                        weights[(label, threshold)],
                        is_tp[(label, threshold)],
                        scores[(label, threshold)],
                        total_gt_weight[label],
                    )
                    for threshold in self.thresholds
                ]
            )
        report = {"cw_mAP": mean_valid(list(per_class.values()))}
        for label, value in per_class.items():
            report[f"cw_mAP_{label_metric_name(label, state.class_names)}"] = value
        return report

    @staticmethod
    def _assemble_ap(
        weights: List[NDArray],
        is_tp: List[NDArray],
        scores: List[NDArray],
        total_gt_weight: float,
    ) -> float:
        if not scores:
            return weighted_average_precision(np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0), total_gt_weight)
        return weighted_average_precision(
            np.concatenate(weights), np.concatenate(is_tp), np.concatenate(scores), total_gt_weight
        )
