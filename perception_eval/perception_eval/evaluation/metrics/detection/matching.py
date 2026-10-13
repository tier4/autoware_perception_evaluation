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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/matching.py) at fcf86419.

"""Score-ordered greedy matching and AP helpers for the advanced detection metrics.

This engine is used only by the driving-aware components; the existing mAP/mAPH path
(``NuscenesObjectMatcher`` / ``Map`` / ``Ap``) is untouched. Matching is greedy and score-ordered
with a configurable cost: nuScenes-style BEV center distance by default, corner distance optionally.
Predictions are matched frame by frame; a prediction claims the nearest unclaimed GT of its frame
(stable ascending cost order, so the lowest GT index wins ties) when that cost is within the
threshold.
"""

from __future__ import annotations

from math import pi
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.component import mean_valid
from perception_eval.evaluation.metrics.detection.geometry import corner_displacement_matrix
from perception_eval.evaluation.metrics.detection.geometry import corner_displacements
from perception_eval.evaluation.metrics.detection.geometry import nearest_surface_distances
from perception_eval.evaluation.metrics.detection.state import CurveMetrics
from perception_eval.evaluation.metrics.detection.state import DetectionSample
from perception_eval.evaluation.metrics.detection.state import ERROR_NAMES
from perception_eval.evaluation.metrics.detection.state import MatchCurve

NUM_RECALL_POINTS = 101
MIN_RECALL = 0.1
MIN_PRECISION = 0.1


def cost_center(pred_boxes: NDArray, gt_boxes: NDArray) -> NDArray[np.float64]:
    """BEV center distance for every prediction and ground-truth pair, ``(P, G)``."""
    return np.linalg.norm(pred_boxes[:, None, :2] - gt_boxes[None, :, :2], axis=2)


def cost_corner(pred_boxes: NDArray, gt_boxes: NDArray) -> NDArray[np.float64]:
    """Mean BEV corner distance for every prediction and ground-truth pair, ``(P, G)``."""
    return corner_displacement_matrix(pred_boxes, gt_boxes)


MATCH_COST_FUNCTIONS: Dict[str, Callable[[NDArray, NDArray], NDArray]] = {"center": cost_center, "corner": cost_corner}


def resolve_match_cost(name: str) -> Callable[[NDArray, NDArray], NDArray]:
    """Return the matching cost function for a configured name (fail-loud)."""
    if name not in MATCH_COST_FUNCTIONS:
        raise ValueError(f"Unknown match cost {name!r}, expected one of {sorted(MATCH_COST_FUNCTIONS)}.")
    return MATCH_COST_FUNCTIONS[name]


class PreparedLabel:
    """One class's boxes across all frames, flattened in global score order.

    Everything threshold-independent is done once here (masking, the per-frame cost matrices, and
    the global stable score sort), so a ``DetectionState`` builds this once per label and matches
    every threshold against it. Greedy claims never cross frames, so the flat order matches
    per-frame greedy matching exactly.
    """

    __slots__ = ("total_gt", "pred_boxes", "scores", "cost_rows", "candidate_rows", "gt_flat", "gt_flat_index")

    def __init__(self, samples: Sequence[DetectionSample], label: int, match_cost: str = "center") -> None:
        cost_fn = resolve_match_cost(match_cost)
        pred_chunks: List[NDArray] = []
        score_chunks: List[NDArray] = []
        cost_row_chunks: List[List[NDArray]] = []
        candidate_row_chunks: List[List[List[int]]] = []
        gt_index_chunks: List[NDArray] = []
        gt_frames: List[NDArray] = []
        self.total_gt = 0
        gt_offset = 0
        for sample in samples:
            gt_boxes = sample.gt_boxes[sample.gt_labels == label]
            self.total_gt += int(gt_boxes.shape[0])
            pred_mask = sample.pred_labels == label
            if not bool(pred_mask.any()):
                continue
            pred_boxes = sample.pred_boxes[pred_mask]
            cost = cost_fn(pred_boxes, gt_boxes)  # (P, G)
            pred_chunks.append(pred_boxes)
            score_chunks.append(sample.pred_scores[pred_mask])
            cost_row_chunks.append(list(cost))
            # Candidates in ascending cost order, once per row. The stable sort keeps the lowest
            # index first among equal costs (the argmin tie rule).
            candidate_row_chunks.append(np.argsort(cost, axis=1, kind="stable").tolist())
            gt_frames.append(gt_boxes)
            gt_index_chunks.append(np.full(pred_boxes.shape[0], gt_offset, dtype=np.int64))
            gt_offset += gt_boxes.shape[0]

        if not pred_chunks:
            self.pred_boxes = np.zeros((0, 9), dtype=np.float64)
            self.scores = np.zeros(0, dtype=np.float64)
            self.cost_rows: List[NDArray] = []
            self.candidate_rows: List[List[int]] = []
            self.gt_flat = np.zeros((0, 9), dtype=np.float64)
            self.gt_flat_index = np.zeros(0, dtype=np.int64)
            return

        scores = np.concatenate(score_chunks)
        order = np.argsort(-scores, kind="stable")
        self.pred_boxes = np.concatenate(pred_chunks)[order]
        self.scores = scores[order]
        cost_rows = [row for chunk in cost_row_chunks for row in chunk]
        self.cost_rows = [cost_rows[index] for index in order]
        candidate_rows = [row for chunk in candidate_row_chunks for row in chunk]
        self.candidate_rows = [candidate_rows[index] for index in order]
        self.gt_flat = np.concatenate(gt_frames) if gt_frames else np.zeros((0, 9), dtype=np.float64)
        self.gt_flat_index = np.concatenate(gt_index_chunks)[order]

    def match(self, threshold: float) -> MatchCurve:
        """Greedy matching at one threshold over the prepared, score-ordered boxes."""
        num_pred = self.scores.shape[0]
        true_positive = np.zeros(num_pred, dtype=np.float64)
        matched_flat_gt = np.full(num_pred, -1, dtype=np.int64)
        claimed = np.zeros(self.gt_flat.shape[0], dtype=bool)
        for index in range(num_pred):
            costs = self.cost_rows[index]
            offset = int(self.gt_flat_index[index])
            for candidate in self.candidate_rows[index]:
                if costs[candidate] > threshold:
                    break
                flat = offset + candidate
                if claimed[flat]:
                    continue
                true_positive[index] = 1.0
                matched_flat_gt[index] = flat
                claimed[flat] = True
                break

        heading_score = np.zeros(num_pred, dtype=np.float64)
        errors = {name: np.full(num_pred, np.nan, dtype=np.float64) for name in ERROR_NAMES}
        corner_error = np.full(num_pred, np.nan, dtype=np.float64)
        nearest_surface = np.full(num_pred, np.nan, dtype=np.float64)
        matched = matched_flat_gt >= 0
        if matched.any():
            pred_boxes = self.pred_boxes[matched]
            gt_boxes = self.gt_flat[matched_flat_gt[matched]]
            errors["ATE"][matched] = np.linalg.norm(pred_boxes[:, :2] - gt_boxes[:, :2], axis=1)
            errors["AOE"][matched] = orientation_errors(pred_boxes, gt_boxes)
            errors["ASE"][matched] = scale_errors(pred_boxes, gt_boxes)
            errors["AVE"][matched] = velocity_errors(pred_boxes, gt_boxes)
            errors["AAE"][matched] = 1.0
            heading_score[matched] = heading_scores(errors["AOE"][matched])
            corner_error[matched] = corner_displacements(pred_boxes, gt_boxes)
            nearest_surface[matched] = nearest_surface_distances(pred_boxes) - nearest_surface_distances(gt_boxes)

        return MatchCurve(
            total_gt=self.total_gt,
            scores=self.scores,
            true_positive=true_positive,
            false_positive=1.0 - true_positive,
            heading_score=heading_score,
            translation_error=errors["ATE"],
            orientation_error=errors["AOE"],
            scale_error=errors["ASE"],
            velocity_error=errors["AVE"],
            attribute_error=errors["AAE"],
            corner_error=corner_error,
            nearest_surface_error=nearest_surface,
        )


def match_by_cost(
    samples: Sequence[DetectionSample], label: int, threshold: float, match_cost: str = "center"
) -> MatchCurve:
    """One-threshold form of :class:`PreparedLabel`."""
    return PreparedLabel(samples, label, match_cost).match(threshold)


def curve_metrics(curve: MatchCurve) -> CurveMetrics:
    """AP, APH, max-F1 and the optimal-confidence operating point for a curve."""
    cumulative_fp = curve.cumulative_fp
    precision, recall = precision_recall(curve.cumulative_tp, cumulative_fp, curve.total_gt)
    ap = interpolated_ap(precision, recall, curve.total_gt, curve.num_predictions)

    heading_precision, heading_recall = precision_recall(curve.cumulative_heading_tp, cumulative_fp, curve.total_gt)
    f1_scores = _f1_scores(precision, recall)
    optimal_index = _max_f1_index(f1_scores)

    if optimal_index >= 0:
        max_f1 = float(f1_scores[optimal_index])
        optimal_conf = float(curve.scores[optimal_index])
        optimal_recall = float(recall[optimal_index])
        optimal_precision = float(precision[optimal_index])
    else:
        max_f1 = optimal_conf = optimal_recall = optimal_precision = float("nan")

    return CurveMetrics(
        ap=ap,
        aph=interpolated_ap(heading_precision, heading_recall, curve.total_gt, curve.num_predictions),
        max_f1=max_f1,
        optimal_conf=optimal_conf,
        optimal_index=optimal_index,
        optimal_recall=optimal_recall,
        optimal_precision=optimal_precision,
    )


def precision_recall(
    cumulative_tp: NDArray, cumulative_fp: NDArray, total_gt: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Precision and recall curves from cumulative TP/FP counts (or weights)."""
    denominator = cumulative_tp + cumulative_fp
    precision = np.divide(
        cumulative_tp, denominator, out=np.zeros_like(cumulative_tp, dtype=np.float64), where=denominator != 0.0
    )
    recall = cumulative_tp / float(total_gt) if total_gt > 0 else np.zeros_like(cumulative_tp, dtype=np.float64)
    return precision, recall


def interpolated_ap(
    precision: NDArray,
    recall: NDArray,
    total_gt: float,
    num_predictions: int,
    min_recall: float = MIN_RECALL,
    min_precision: float = MIN_PRECISION,
) -> float:
    """nuScenes-style interpolated AP: 101 recall points, recall through ``min_recall`` ignored,
    ``min_precision`` subtracted and clamped at zero, normalized by ``1 - min_precision``.

    This is the same convention as :class:`perception_eval.evaluation.metrics.detection.ap.Ap`.
    """
    if total_gt <= 0 and num_predictions == 0:
        return float("nan")
    if precision.shape[0] == 0:
        return 0.0
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_grid = np.linspace(0.0, 1.0, NUM_RECALL_POINTS)
    precision_interp = np.interp(recall_grid, recall, precision_envelope, right=0.0)
    first_index = int(round(100 * min_recall)) + 1
    filtered_precision = precision_interp[first_index:] - min_precision
    filtered_precision[filtered_precision < 0.0] = 0.0
    return float(np.mean(filtered_precision)) / (1.0 - min_precision)


def _f1_scores(precision: NDArray, recall: NDArray) -> NDArray[np.float64]:
    denominator = precision + recall
    return np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=np.float64),
        where=denominator != 0.0,
    )


def _max_f1_index(f1_scores: NDArray) -> int:
    if f1_scores.shape[0] == 0 or np.all(np.isnan(f1_scores)):
        return -1
    return int(np.nanargmax(f1_scores))


def orientation_errors(pred_boxes: NDArray, gt_boxes: NDArray) -> NDArray[np.float64]:
    """Absolute yaw error wrapped to ``[0, pi]``, per matched pair."""
    diff = np.abs(pred_boxes[:, 6] - gt_boxes[:, 6])
    return np.abs((diff + pi) % (2.0 * pi) - pi)


def heading_scores(orientation_error: NDArray) -> NDArray[np.float64]:
    """nuScenes heading weight ``1 - |yaw error| / pi`` clipped to ``[0, 1]``."""
    return np.round(np.clip(1.0 - orientation_error / pi, 0.0, 1.0), 10)


def scale_errors(pred_boxes: NDArray, gt_boxes: NDArray) -> NDArray[np.float64]:
    """``1 - IoU`` of the aligned (center- and yaw-matched) 3D dimensions."""
    pred_dims = np.maximum(pred_boxes[:, 3:6], 0.0)
    gt_dims = np.maximum(gt_boxes[:, 3:6], 0.0)
    intersection = np.prod(np.minimum(pred_dims, gt_dims), axis=1)
    union = np.prod(pred_dims, axis=1) + np.prod(gt_dims, axis=1) - intersection
    errors = np.ones(pred_boxes.shape[0], dtype=np.float64)
    positive = union > 0.0
    errors[positive] = 1.0 - intersection[positive] / union[positive]
    return errors


def velocity_errors(pred_boxes: NDArray, gt_boxes: NDArray) -> NDArray[np.float64]:
    """Planar velocity error, or 1.0 when the boxes carry no velocity columns."""
    if pred_boxes.shape[1] < 9 or gt_boxes.shape[1] < 9:
        return np.ones(pred_boxes.shape[0], dtype=np.float64)
    return np.linalg.norm(pred_boxes[:, 7:9] - gt_boxes[:, 7:9], axis=1)


def mean_tp_errors(error_dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each error name across the given per-class error dicts."""
    return {error_name: mean_valid([errors[error_name] for errors in error_dicts]) for error_name in ERROR_NAMES}
