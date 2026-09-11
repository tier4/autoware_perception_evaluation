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
# Ported from tier4/autoware-ml (autoware_ml/metrics/segmentation3d/entropy_auroc.py) at fcf86419.

"""Uncertainty usefulness, entropy as an error detector (metric E2).

Measures how well per-point normalized predictive entropy separates correct from misclassified
points, via the AUROC of entropy for the binary "is this point wrong?" task. 0.5 means the
uncertainty is meaningless, higher means it is a usable error flag. Entropy is in ``[0, 1]``, so
the AUROC is computed from two fixed-width histograms (wrong / correct) accumulated per frame:
memory stays O(num_bins) and the tie-aware closed form quantizes entropy to ``1 / num_bins``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.segmentation.state import SegmentationView


def binned_auroc(positive_hist: NDArray, negative_hist: NDArray) -> float:
    """Tie-aware AUROC of a score for the positive class, from two histograms.

    Treats all values inside one bin as tied (the mid-rank convention):
    ``P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)``.
    """
    positive_hist = np.asarray(positive_hist, dtype=np.float64)
    negative_hist = np.asarray(negative_hist, dtype=np.float64)
    num_positive = float(positive_hist.sum())
    num_negative = float(negative_hist.sum())
    if num_positive == 0.0 or num_negative == 0.0:
        return float("nan")
    negative_below = np.concatenate(([0.0], np.cumsum(negative_hist)[:-1]))
    wins = float(np.sum(positive_hist * negative_below))
    ties = float(np.sum(positive_hist * negative_hist))
    return (wins + 0.5 * ties) / (num_positive * num_negative)


def exact_auroc(scores: NDArray, positive: NDArray) -> float:
    """Exact tie-aware (mid-rank) AUROC, for tests and small inputs."""
    from scipy.stats import rankdata

    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class UncertaintyUsefulness:
    """AUROC of entropy as a misclassification detector, plus the mean entropy of wrong/correct points."""

    kind = "point"
    supports_grouped = True
    needs_boxes = False

    def __init__(self, num_bins: int = 8192) -> None:
        self.num_bins = int(num_bins)
        if self.num_bins < 2:
            raise ValueError("num_bins must be >= 2.")
        self.reset()

    def reset(self) -> None:
        self._wrong_hist = np.zeros(self.num_bins, dtype=np.float64)
        self._correct_hist = np.zeros(self.num_bins, dtype=np.float64)
        self._entropy_sum = {True: 0.0, False: 0.0}
        self._count = {True: 0, False: 0}

    def update(self, view: SegmentationView) -> None:
        if view.num_points == 0:
            return
        wrong = view.wrong
        bins = np.minimum((view.entropy * self.num_bins).astype(np.int64), self.num_bins - 1)
        self._wrong_hist += np.bincount(bins[wrong], minlength=self.num_bins)
        self._correct_hist += np.bincount(bins[~wrong], minlength=self.num_bins)
        for flag in (True, False):
            selected = view.entropy[wrong] if flag else view.entropy[~wrong]
            self._entropy_sum[flag] += float(selected.sum(dtype=np.float64))
            self._count[flag] += int(selected.shape[0])

    def compute(self) -> Dict[str, float]:
        return {
            "entropy_auroc": binned_auroc(self._wrong_hist, self._correct_hist),
            "mean_entropy_wrong": (self._entropy_sum[True] / self._count[True]) if self._count[True] else float("nan"),
            "mean_entropy_correct": (
                (self._entropy_sum[False] / self._count[False]) if self._count[False] else float("nan")
            ),
        }
