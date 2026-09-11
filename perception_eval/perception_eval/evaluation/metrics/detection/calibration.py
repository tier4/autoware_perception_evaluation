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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/calibration.py) at fcf86419.

"""Detection calibration error (metric E1, detection variant).

A detection score should match the empirical precision: of the boxes scored about 0.9, roughly 90%
should be true positives. Predictions are binned by score and the gap between the mean score and
the precision in each bin is measured. Correctness is TP status at ``tp_threshold``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.component import mean_valid
from perception_eval.evaluation.metrics.detection.state import DEFAULT_TP_THRESHOLD
from perception_eval.evaluation.metrics.detection.state import DetectionState


def expected_calibration_error(scores: NDArray, correct: NDArray, num_bins: int) -> float:
    """Prediction-weighted mean gap between confidence and correctness over equal-width score bins.

    Raises:
        ValueError: If any score falls outside [0, 1] (scores must be probabilities, not logits).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if scores.shape[0] == 0:
        return float("nan")
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise ValueError(
            "detection scores must be probabilities in [0, 1] but values fall outside, "
            "calibrate against probabilities, not raw logits."
        )
    total = scores.shape[0]
    bin_index = np.minimum((scores * num_bins).astype(np.int64), num_bins - 1)
    count = np.bincount(bin_index, minlength=num_bins).astype(np.float64)
    score_sum = np.bincount(bin_index, weights=scores, minlength=num_bins)
    correct_sum = np.bincount(bin_index, weights=correct, minlength=num_bins)
    nonempty = count > 0
    gaps = np.abs(correct_sum[nonempty] / count[nonempty] - score_sum[nonempty] / count[nonempty])
    return float(np.sum(count[nonempty] / total * gaps))


class CalibrationError:
    """Expected calibration error of the detection score against precision.

    ``ece`` pools every prediction and ``ece_macro`` averages the per-class ECE so a dominant class
    does not hide a badly calibrated rare one.
    """

    needs_ttc: bool = False

    def __init__(self, tp_threshold: float = DEFAULT_TP_THRESHOLD, num_bins: int = 15) -> None:
        """
        Args:
            tp_threshold (float): Match threshold in meters deciding correctness.
            num_bins (int): Number of equal-width score bins.
        """
        self.tp_threshold = float(tp_threshold)
        self.num_bins = int(num_bins)
        if self.num_bins < 1:
            raise ValueError("num_bins must be >= 1.")

    def evaluate(self, state: DetectionState) -> Dict[str, float]:
        """Report overall and macro calibration error at the TP threshold."""
        all_scores = []
        all_correct = []
        per_class = []
        for label in state.labels(full=True):
            curve = state.match_curve(label, self.tp_threshold)
            if curve.scores.shape[0] == 0:
                continue
            all_scores.append(curve.scores)
            all_correct.append(curve.true_positive)
            per_class.append(expected_calibration_error(curve.scores, curve.true_positive, self.num_bins))
        if not all_scores:
            return {"ece": float("nan"), "ece_macro": float("nan")}
        scores = np.concatenate(all_scores)
        correct = np.concatenate(all_correct)
        return {
            "ece": expected_calibration_error(scores, correct, self.num_bins),
            "ece_macro": mean_valid(per_class),
        }
