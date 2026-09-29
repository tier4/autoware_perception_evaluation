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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/critical_fp_fn.py) at fcf86419.

"""Critical FP / FN over the ego reachable set (metric B1).

Two numbers, never averaged: false positives in ego's path cause phantom braking (usability),
false negatives in ego's path mean driving toward something unseen (safety). "In ego's path" is
the one reachability model: an object is critical when it can collide with ego within the horizon,
i.e. its per-box TTC (computed by the suite's collision provider) is finite. Both are reported as a
function of the confidence threshold.

Matching is class-agnostic (a box detected where an object is avoids both a phantom and a miss),
and only the unmatched boxes that are critical are counted. Frames whose scene has no lanelet map
carry no TTC and are excluded from the per-frame denominator.
"""

from __future__ import annotations

from typing import Dict
from typing import Sequence
from typing import Tuple

import numpy as np
from perception_eval.evaluation.metrics.detection.criticality import greedy_match
from perception_eval.evaluation.metrics.detection.state import DEFAULT_TP_THRESHOLD
from perception_eval.evaluation.metrics.detection.state import DetectionState
from perception_eval.evaluation.metrics.naming import label_metric_name
from perception_eval.evaluation.metrics.naming import threshold_token


class CriticalFPFN:
    """Per-frame false positives / negatives in the ego reachable set, per confidence."""

    needs_ttc = True

    def __init__(
        self,
        confidences: Sequence[float] = (0.3, 0.5),
        match_threshold: float = DEFAULT_TP_THRESHOLD,
    ) -> None:
        """Store the confidence operating points and the match threshold.

        Args:
            confidences (Sequence[float]): Score floors, the FP and FN counts are reported at each.
            match_threshold (float): Class-agnostic center-distance match threshold in meters.
        """
        self.confidences: Tuple[float, ...] = tuple(float(c) for c in confidences)
        if not self.confidences:
            raise ValueError("confidences must not be empty.")
        if match_threshold <= 0.0:
            raise ValueError("match_threshold must be > 0.")
        self.match_threshold = float(match_threshold)

    def evaluate(self, state: DetectionState) -> Dict[str, float]:
        """Count critical FP / FN (finite TTC) at each confidence threshold."""
        covered_frames = 0
        fp_total = {conf: 0 for conf in self.confidences}
        fn_total = {conf: 0 for conf in self.confidences}
        fp_class: Dict[Tuple[float, int], int] = {}
        fn_class: Dict[Tuple[float, int], int] = {}
        for sample in state.samples:
            if not sample.has_ttc:
                raise ValueError("CriticalFPFN needs per-box TTC, build the suite with a collision provider.")
            if not sample.ttc_covered:
                continue
            covered_frames += 1
            gt_boxes = sample.gt_boxes
            gt_critical = np.isfinite(sample.gt_ttc)
            pred_critical = np.isfinite(sample.pred_ttc)
            for conf in self.confidences:
                keep = sample.pred_scores >= conf
                kept_boxes = sample.pred_boxes[keep]
                is_tp, matched_gt = greedy_match(
                    gt_boxes[:, :2] if gt_boxes.shape[0] else np.zeros((0, 2)),
                    kept_boxes[:, :2] if kept_boxes.shape[0] else np.zeros((0, 2)),
                    sample.pred_scores[keep],
                    self.match_threshold,
                )
                # Critical false positives: unmatched predictions that are in-path.
                false_positive = (~is_tp) & pred_critical[keep]
                matched = np.zeros(gt_boxes.shape[0], dtype=bool)
                matched[matched_gt[is_tp]] = True
                # Critical false negatives: unmatched ground truth that is in-path.
                false_negative = (~matched) & gt_critical
                fp_total[conf] += int(false_positive.sum())
                fn_total[conf] += int(false_negative.sum())
                for label in sample.pred_labels[keep][false_positive]:
                    fp_class[(conf, int(label))] = fp_class.get((conf, int(label)), 0) + 1
                for label in sample.gt_labels[false_negative]:
                    fn_class[(conf, int(label))] = fn_class.get((conf, int(label)), 0) + 1

        # No covered frame means the slice has no basis: report NaN, never a fake zero.
        denominator = covered_frames if covered_frames else float("nan")
        report: Dict[str, float] = {}
        for conf in self.confidences:
            token = threshold_token(conf)
            report[f"critical_fp_{token}"] = fp_total[conf] / denominator
            report[f"critical_fn_{token}"] = fn_total[conf] / denominator
        for label in state.labels(full=True):
            name = label_metric_name(label, state.class_names)
            for conf in self.confidences:
                token = threshold_token(conf)
                report[f"critical_fp_{name}_{token}"] = fp_class.get((conf, label), 0) / denominator
                report[f"critical_fn_{name}_{token}"] = fn_class.get((conf, label), 0) / denominator
        return report
