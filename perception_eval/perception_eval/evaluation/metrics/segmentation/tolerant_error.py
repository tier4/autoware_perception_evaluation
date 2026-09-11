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
# Ported from tier4/autoware-ml (autoware_ml/metrics/segmentation3d/tolerant_error.py) at fcf86419.

"""Neighbourhood-tolerant error rate (metric D3).

A misclassified point counts as an error only when it is wrong *and* the model predicted its true
class on no point within ``radius``. The tolerance only ever removes errors, so this rate is at
most the strict error rate, and equals it at ``radius = 0``. Reported whole-scene and per class.
"""

from __future__ import annotations

from typing import Dict
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.naming import label_metric_name
from perception_eval.evaluation.metrics.segmentation.spatial import tolerant_error_mask
from perception_eval.evaluation.metrics.segmentation.state import SegmentationView


class NeighbourhoodTolerantErrorRate:
    """Fraction of valid points that are wrong with no correct-class neighbour within ``radius``."""

    kind = "point"
    supports_grouped = True
    needs_boxes = False

    def __init__(self, radius: float = 0.2) -> None:
        self.radius = float(radius)
        if self.radius < 0.0:
            raise ValueError("radius must be >= 0.")
        self.reset()

    def reset(self) -> None:
        self._error_total = 0
        self._valid_total = 0
        self._class_names: Optional[tuple] = None
        self._per_error: Optional[NDArray] = None
        self._per_valid: Optional[NDArray] = None

    def update(self, view: SegmentationView) -> None:
        if self._per_error is None:
            self._class_names = view.class_names
            self._per_error = np.zeros(view.num_classes, dtype=np.int64)
            self._per_valid = np.zeros(view.num_classes, dtype=np.int64)
        if view.num_points == 0:
            return
        errors = tolerant_error_mask(view.xyz, view.pred, view.target, self.radius)
        self._valid_total += view.num_points
        self._error_total += int(errors.sum())
        self._per_valid += np.bincount(view.target, minlength=view.num_classes)[: view.num_classes]
        self._per_error += np.bincount(view.target[errors], minlength=view.num_classes)[: view.num_classes]

    def compute(self) -> Dict[str, float]:
        report: Dict[str, float] = {
            "tolerant_error_rate": (self._error_total / self._valid_total) if self._valid_total else float("nan"),
            "tolerant_error_count": float(self._error_total),
        }
        if self._per_error is None:
            return report
        for class_index in range(self._per_error.shape[0]):
            name = label_metric_name(class_index, self._class_names)
            valid = int(self._per_valid[class_index])
            report[f"tolerant_error_rate_{name}"] = (
                (int(self._per_error[class_index]) / valid) if valid else float("nan")
            )
            report[f"tolerant_error_count_{name}"] = float(self._per_error[class_index])
        return report
