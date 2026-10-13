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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/corner_error.py) at fcf86419.

"""Corner displacement error (metric A1).

Corner displacement couples position, size, and yaw into one distance in meters, measured where
planning collides with things: the box outline. It also exposes footprint inflation that center
distance is blind to. Reported per class at the TP threshold.
"""

from __future__ import annotations

from typing import Dict
from typing import Sequence

import numpy as np
from perception_eval.evaluation.metrics.component import mean_valid
from perception_eval.evaluation.metrics.detection.state import DEFAULT_TP_THRESHOLD
from perception_eval.evaluation.metrics.detection.state import DetectionState
from perception_eval.evaluation.metrics.naming import label_metric_name
from perception_eval.evaluation.metrics.naming import number_token


class CornerError:
    """Mean / percentile / max true-positive corner displacement, per class (meters)."""

    needs_ttc: bool = False

    def __init__(self, tp_threshold: float = DEFAULT_TP_THRESHOLD, percentiles: Sequence[float] = (95.0,)) -> None:
        """
        Args:
            tp_threshold (float): Match threshold in meters the errors are read at.
            percentiles (Sequence[float]): Percentiles in [0, 100] reported next to the mean and max.
        """
        self.tp_threshold = float(tp_threshold)
        self.percentiles = tuple(float(p) for p in percentiles)
        for p in self.percentiles:
            if not 0.0 <= p <= 100.0:
                raise ValueError(f"percentile {p} outside [0, 100].")

    def evaluate(self, state: DetectionState) -> Dict[str, float]:
        """Report per-class corner-displacement statistics at the TP threshold."""
        report: Dict[str, float] = {}
        per_class_mean = []
        per_class_max = []
        for label in state.labels(full=True):
            curve = state.match_curve(label, self.tp_threshold)
            errors = curve.corner_error[curve.true_positive == 1.0]
            errors = errors[~np.isnan(errors)]
            name = label_metric_name(label, state.class_names)
            if errors.shape[0] == 0:
                report[f"corner_mean_{name}"] = float("nan")
                report[f"corner_max_{name}"] = float("nan")
                for p in self.percentiles:
                    report[f"corner_p{number_token(p)}_{name}"] = float("nan")
                continue
            mean_value = float(np.mean(errors))
            maximum = float(np.max(errors))
            report[f"corner_mean_{name}"] = mean_value
            report[f"corner_max_{name}"] = maximum
            per_class_mean.append(mean_value)
            per_class_max.append(maximum)
            for p in self.percentiles:
                report[f"corner_p{number_token(p)}_{name}"] = float(np.percentile(errors, p))

        report["mcorner_mean"] = mean_valid(per_class_mean)
        report["mcorner_max"] = mean_valid(per_class_max)
        return report
