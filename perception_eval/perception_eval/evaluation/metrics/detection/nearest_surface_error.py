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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/nearest_surface_error.py) at fcf86419.

"""Signed nearest-surface error (metric A3).

Stopping distance is computed against the nearest face of an object, not its center. The sign is
the safety signal: positive means the predicted near face sits farther than the truth (ego brakes
late), negative means nearer (over-caution). Reported per class at the TP threshold.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from perception_eval.evaluation.metrics.component import mean_valid
from perception_eval.evaluation.metrics.detection.state import DEFAULT_TP_THRESHOLD
from perception_eval.evaluation.metrics.detection.state import DetectionState
from perception_eval.evaluation.metrics.naming import label_metric_name


class NearestSurfaceError:
    """Signed near-face error tails per class (meters)."""

    needs_ttc: bool = False

    def __init__(
        self,
        tp_threshold: float = DEFAULT_TP_THRESHOLD,
        low_percentile: float = 5.0,
        high_percentile: float = 95.0,
    ) -> None:
        """
        Args:
            tp_threshold (float): Match threshold in meters the errors are read at.
            low_percentile (float): Signed percentile reported as the over-caution tail.
            high_percentile (float): Signed percentile reported as the late-braking tail.
        """
        self.tp_threshold = float(tp_threshold)
        self.low_percentile = float(low_percentile)
        self.high_percentile = float(high_percentile)
        if not 0.0 <= self.low_percentile < self.high_percentile <= 100.0:
            raise ValueError("expected 0 <= low_percentile < high_percentile <= 100.")

    def evaluate(self, state: DetectionState) -> Dict[str, float]:
        """Report per-class signed nearest-surface error tails at the TP threshold."""
        report: Dict[str, float] = {}
        per_class_high = []
        per_class_absmax = []
        for label in state.labels(full=True):
            curve = state.match_curve(label, self.tp_threshold)
            errors = curve.nearest_surface_error[curve.true_positive == 1.0]
            errors = errors[~np.isnan(errors)]
            name = label_metric_name(label, state.class_names)
            if errors.shape[0] == 0:
                report[f"nsurf_mean_{name}"] = float("nan")
                report[f"nsurf_low_{name}"] = float("nan")
                report[f"nsurf_high_{name}"] = float("nan")
                report[f"nsurf_absmax_{name}"] = float("nan")
                continue
            high = float(np.percentile(errors, self.high_percentile))
            absmax = float(np.max(np.abs(errors)))
            report[f"nsurf_mean_{name}"] = float(np.mean(errors))
            report[f"nsurf_low_{name}"] = float(np.percentile(errors, self.low_percentile))
            report[f"nsurf_high_{name}"] = high
            report[f"nsurf_absmax_{name}"] = absmax
            per_class_high.append(high)
            per_class_absmax.append(absmax)

        report["mnsurf_high"] = mean_valid(per_class_high)
        report["mnsurf_absmax"] = mean_valid(per_class_absmax)
        return report
