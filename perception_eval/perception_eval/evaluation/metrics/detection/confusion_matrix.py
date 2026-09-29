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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/confusion_matrix.py) at fcf86419.

"""Detection confusion matrix (matched detections only).

Class-agnostic greedy center-distance matching pairs each prediction to its nearest unclaimed
ground-truth box within ``match_threshold``, and every matched pair adds one to
``confusion[true_class, pred_class]``. Predictions below ``min_score`` and unmatched boxes are
dropped: this is the label-confusion view among detections that did match, not a recall matrix.
In a grouped view the state's labels are already folded, so the matrix is over behaviour groups.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from perception_eval.evaluation.metrics.confusion_report import confusion_cells
from perception_eval.evaluation.metrics.detection.criticality import greedy_match
from perception_eval.evaluation.metrics.detection.state import DEFAULT_TP_THRESHOLD
from perception_eval.evaluation.metrics.detection.state import DetectionState


class ConfusionMatrix:
    """True vs. predicted class counts over class-agnostically matched detections."""

    needs_ttc: bool = False

    def __init__(self, match_threshold: float = DEFAULT_TP_THRESHOLD, min_score: float = 0.1) -> None:
        """
        Args:
            match_threshold (float): Class-agnostic center-distance match threshold in meters.
            min_score (float): Reporting floor, predictions below it never enter the matrix.
        """
        self.match_threshold = float(match_threshold)
        self.min_score = float(min_score)
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1].")

    def evaluate(self, state: DetectionState) -> Dict[str, float]:
        """Accumulate matched (true, predicted) label pairs into a confusion matrix."""
        if state.class_names is None:
            raise ValueError("ConfusionMatrix needs class_names to label its cells.")
        if state.match_cost != "center":
            raise ValueError(
                "ConfusionMatrix matches class-agnostically by center distance, a state "
                f"configured with match_cost={state.match_cost!r} would report matched "
                "pairs inconsistent with its AP family."
            )
        num_classes = len(state.class_names)
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for sample in state.samples:
            if sample.gt_boxes.shape[0] == 0 or sample.pred_boxes.shape[0] == 0:
                continue
            keep = sample.pred_scores >= self.min_score
            pred_centers = sample.pred_boxes[keep][:, :2]
            if pred_centers.shape[0] == 0:
                continue
            is_tp, matched_gt = greedy_match(
                sample.gt_boxes[:, :2], pred_centers, sample.pred_scores[keep], self.match_threshold
            )
            true_labels = sample.gt_labels[matched_gt[is_tp]]
            pred_labels = sample.pred_labels[keep][is_tp]
            pairs = np.concatenate([true_labels, pred_labels])
            if pairs.shape[0] and (pairs.min() < 0 or pairs.max() >= num_classes):
                raise ValueError(
                    "matched labels outside the configured class range, a folding or configuration bug upstream."
                )
            np.add.at(matrix, (true_labels, pred_labels), 1)
        return confusion_cells(matrix, state.class_names)
