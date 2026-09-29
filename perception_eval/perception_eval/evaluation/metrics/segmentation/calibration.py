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
# Ported from tier4/autoware-ml (autoware_ml/metrics/segmentation3d/calibration.py) at fcf86419.

"""Segmentation calibration error (metric E1).

A model that says "90% road" should be right about 90% of the time. Expected Calibration Error
bins points by the confidence of the reported class and sums the gap between confidence and
accuracy. Reported overall and macro-averaged per *predicted* class, so the dominant class does
not hide poor calibration on rare classes. Streaming: only ``(K, num_bins)`` sums are retained.
"""

from __future__ import annotations

from typing import Dict
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.segmentation.state import SegmentationView


def ece_from_bins(conf_sum: NDArray, correct_sum: NDArray, count: NDArray) -> float:
    """Count-weighted mean gap between mean confidence and accuracy over the bins."""
    total = float(np.sum(count))
    if total == 0:
        return float("nan")
    nonempty = count > 0
    accuracy = correct_sum[nonempty] / count[nonempty]
    mean_conf = conf_sum[nonempty] / count[nonempty]
    weight = count[nonempty] / total
    return float(np.sum(weight * np.abs(accuracy - mean_conf)))


class CalibrationError:
    """Expected Calibration Error, overall (``ece``) and macro over predicted classes (``ece_macro``)."""

    kind = "point"
    supports_grouped = True
    needs_boxes = False

    def __init__(self, num_bins: int = 15) -> None:
        self.num_bins = int(num_bins)
        if self.num_bins < 1:
            raise ValueError("num_bins must be >= 1.")
        self._conf_sum: Optional[NDArray] = None
        self._correct_sum: Optional[NDArray] = None
        self._count: Optional[NDArray] = None

    def reset(self) -> None:
        self._conf_sum = self._correct_sum = self._count = None

    def _ensure(self, num_classes: int) -> None:
        if self._count is None:
            self._conf_sum = np.zeros((num_classes, self.num_bins), dtype=np.float64)
            self._correct_sum = np.zeros((num_classes, self.num_bins), dtype=np.float64)
            self._count = np.zeros((num_classes, self.num_bins), dtype=np.float64)

    def update(self, view: SegmentationView) -> None:
        self._ensure(view.num_classes)
        if view.num_points == 0:
            return
        correct = (view.pred == view.target).astype(np.float64)
        bin_index = np.clip((view.confidence * self.num_bins).astype(np.int64), 0, self.num_bins - 1)
        np.add.at(self._conf_sum, (view.pred, bin_index), view.confidence)
        np.add.at(self._correct_sum, (view.pred, bin_index), correct)
        np.add.at(self._count, (view.pred, bin_index), 1.0)

    def compute(self) -> Dict[str, float]:
        if self._count is None:
            return {"ece": float("nan"), "ece_macro": float("nan")}
        overall = ece_from_bins(self._conf_sum.sum(0), self._correct_sum.sum(0), self._count.sum(0))
        per_class = [
            ece_from_bins(self._conf_sum[c], self._correct_sum[c], self._count[c])
            for c in range(self._count.shape[0])
            if self._count[c].sum() > 0
        ]
        macro = float(np.mean(per_class)) if per_class else float("nan")
        return {"ece": overall, "ece_macro": macro}
