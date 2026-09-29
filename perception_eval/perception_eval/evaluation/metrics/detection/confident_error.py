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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/confident_error.py) at fcf86419.

"""Detection confident-error rate (metric E3, detection variant).

Of the boxes the model got wrong, what fraction was it sure about? A high-score false positive is
the confident phantom that triggers a hard brake. This is false-positive only by nature: a false
negative carries no prediction and therefore no score (see the critical FN of B1).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from perception_eval.evaluation.metrics.detection.state import DEFAULT_TP_THRESHOLD
from perception_eval.evaluation.metrics.detection.state import DetectionState


class ConfidentErrorRate:
    """Confident-FP rate: of the false positives at or above ``min_score`` (the reporting floor),
    the fraction scored at or above ``score_threshold``."""

    needs_ttc: bool = False

    def __init__(
        self,
        tp_threshold: float = DEFAULT_TP_THRESHOLD,
        score_threshold: float = 0.5,
        min_score: float = 0.1,
    ) -> None:
        """
        Args:
            tp_threshold (float): Match threshold in meters deciding false positives.
            score_threshold (float): Score at or above which a false positive counts as confident.
            min_score (float): Reporting floor, false positives below it leave the denominator.
        """
        self.tp_threshold = float(tp_threshold)
        self.score_threshold = float(score_threshold)
        self.min_score = float(min_score)
        if not 0.0 <= self.min_score <= self.score_threshold <= 1.0:
            raise ValueError("expected 0 <= min_score <= score_threshold <= 1.")

    def evaluate(self, state: DetectionState) -> Dict[str, float]:
        """Report the confident false-positive rate over all classes."""
        false_positive_total = 0
        confident_total = 0
        for label in state.labels(full=True):
            curve = state.match_curve(label, self.tp_threshold)
            false_positive_scores = curve.scores[curve.false_positive == 1.0]
            false_positive_total += int(np.sum(false_positive_scores >= self.min_score))
            confident_total += int(np.sum(false_positive_scores >= self.score_threshold))
        rate = confident_total / false_positive_total if false_positive_total else float("nan")
        num_frames = state.num_frames if state.num_frames else float("nan")
        return {
            "confident_error_rate": rate,
            "confident_error_count": float(confident_total),
            "confident_errors_per_frame": confident_total / num_frames,
        }
