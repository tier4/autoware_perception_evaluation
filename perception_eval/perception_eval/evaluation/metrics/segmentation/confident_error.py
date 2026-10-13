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
# Ported from tier4/autoware-ml (autoware_ml/metrics/segmentation3d/confident_error.py) at fcf86419.

"""Segmentation confident-error rate (metric E3).

Over the misclassified points, the fraction made at high confidence, i.e. normalized entropy
below ``entropy_threshold``. An error the model is unsure about is recoverable, a confident error
is what triggers a hard brake or hides an obstacle.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from perception_eval.evaluation.metrics.segmentation.state import SegmentationView


class ConfidentErrorRate:
    """Fraction of misclassified points made at high confidence (low entropy)."""

    kind = "point"
    supports_grouped = True
    needs_boxes = False

    def __init__(self, entropy_threshold: float = 0.3) -> None:
        self.entropy_threshold = float(entropy_threshold)
        if not 0.0 <= self.entropy_threshold <= 1.0:
            raise ValueError("entropy_threshold must be in [0, 1].")
        self.reset()

    def reset(self) -> None:
        self._error_total = 0
        self._confident_total = 0

    def update(self, view: SegmentationView) -> None:
        if view.num_points == 0:
            return
        wrong = view.wrong
        if not wrong.any():
            return
        entropy = view.entropy[wrong]
        self._error_total += int(wrong.sum())
        self._confident_total += int(np.sum(entropy < self.entropy_threshold))

    def compute(self) -> Dict[str, float]:
        return {
            "confident_error_rate": (self._confident_total / self._error_total) if self._error_total else float("nan"),
            "confident_error_count": float(self._confident_total),
        }
