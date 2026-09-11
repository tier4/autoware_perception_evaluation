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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/criticality.py) at fcf86419.

"""Shared frame-local matching and weighted-AP helpers for the detection metrics.

The reachability time-to-collision and its risk weight live in
:mod:`perception_eval.evaluation.metrics.geometry.reachability`, and the criticality metrics
(B1, B2) read a per-box TTC the suite has already computed. This module holds only the stateless
NumPy pieces several metrics share: the class-agnostic greedy center-distance match and the
weighted average precision.
"""

from __future__ import annotations

from typing import Dict
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.detection.matching import MIN_PRECISION
from perception_eval.evaluation.metrics.detection.matching import MIN_RECALL
from perception_eval.evaluation.metrics.detection.matching import NUM_RECALL_POINTS


def greedy_match_thresholds(
    gt_centers: NDArray,
    pred_centers: NDArray,
    pred_scores: NDArray,
    thresholds: Tuple[float, ...],
) -> Dict[float, Tuple[NDArray[np.bool_], NDArray[np.int64]]]:
    """Score-ordered greedy center-distance matching at several thresholds.

    Returns ``{threshold: (is_tp, matched_gt)}`` aligned to the predictions in their original
    order (``matched_gt`` is ``-1`` for non-matches).
    """
    gt_centers = np.asarray(gt_centers, dtype=np.float64).reshape(-1, gt_centers.shape[-1] if len(gt_centers) else 2)
    pred_centers = np.asarray(pred_centers, dtype=np.float64).reshape(
        -1, pred_centers.shape[-1] if len(pred_centers) else 2
    )
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    num_pred = pred_centers.shape[0]
    results = {
        float(threshold): (np.zeros(num_pred, dtype=bool), np.full(num_pred, -1, dtype=np.int64))
        for threshold in thresholds
    }
    if num_pred == 0 or gt_centers.shape[0] == 0:
        return results
    distances = np.linalg.norm(pred_centers[:, None, :2] - gt_centers[None, :, :2], axis=2)
    candidates = np.argsort(distances, axis=1, kind="stable").tolist()
    score_order = np.argsort(-pred_scores, kind="stable")
    for threshold, (is_tp, matched_gt) in results.items():
        claimed = np.zeros(gt_centers.shape[0], dtype=bool)
        for pred_index in score_order:
            row = distances[pred_index]
            for candidate in candidates[pred_index]:
                if row[candidate] > threshold:
                    break
                if claimed[candidate]:
                    continue
                is_tp[pred_index] = True
                matched_gt[pred_index] = candidate
                claimed[candidate] = True
                break
    return results


def greedy_match(
    gt_centers: NDArray, pred_centers: NDArray, pred_scores: NDArray, threshold: float
) -> Tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """Score-ordered greedy center-distance matching for one frame (class-agnostic)."""
    return greedy_match_thresholds(gt_centers, pred_centers, pred_scores, (float(threshold),))[float(threshold)]


def weighted_average_precision(
    weights: NDArray,
    is_tp: NDArray,
    scores: NDArray,
    total_gt_weight: float,
    min_recall: float = MIN_RECALL,
    min_precision: float = MIN_PRECISION,
) -> float:
    """nuScenes-style interpolated AP with per-prediction weights.

    ``weights`` weighs each prediction's TP/FP contribution and ``total_gt_weight`` is the summed
    weight of all ground truth (the recall denominator). With unit weights this equals the
    unweighted AP convention of :class:`perception_eval.evaluation.metrics.detection.ap.Ap`.
    """
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    is_tp = np.asarray(is_tp, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if total_gt_weight <= 0.0:
        return float("nan")
    if scores.shape[0] == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp_weight = np.where(is_tp[order], weights[order], 0.0)
    fp_weight = np.where(is_tp[order], 0.0, weights[order])
    cum_tp = np.cumsum(tp_weight)
    cum_fp = np.cumsum(fp_weight)
    denominator = cum_tp + cum_fp
    precision = np.divide(cum_tp, denominator, out=np.zeros_like(cum_tp), where=denominator != 0.0)
    recall = cum_tp / total_gt_weight
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_grid = np.linspace(0.0, 1.0, NUM_RECALL_POINTS)
    interpolated = np.interp(recall_grid, recall, envelope, right=0.0)
    first = int(round(100 * min_recall)) + 1
    filtered = interpolated[first:] - min_precision
    filtered[filtered < 0.0] = 0.0
    return float(np.mean(filtered)) / (1.0 - min_precision)
