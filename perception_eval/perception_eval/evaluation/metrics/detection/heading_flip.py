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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/heading_flip.py) at fcf86419.

"""Heading-flip rate (metric A2).

Corner displacement (A1) forgives a 180-degree heading flip, yet a flipped heading inverts the
object's velocity direction and breaks tracking, so it is penalized on its own: every true positive
whose absolute wrapped yaw error exceeds ``flip_threshold`` counts as one flip.
"""

from __future__ import annotations

from math import pi
from typing import Dict

import numpy as np
from perception_eval.evaluation.metrics.component import mean_valid
from perception_eval.evaluation.metrics.detection.state import DEFAULT_TP_THRESHOLD
from perception_eval.evaluation.metrics.detection.state import DetectionState
from perception_eval.evaluation.metrics.naming import label_metric_name


class HeadingFlipRate:
    """Per-class fraction (and count) of true positives with a reversed heading."""

    needs_ttc: bool = False

    def __init__(self, tp_threshold: float = DEFAULT_TP_THRESHOLD, flip_threshold: float = pi / 2.0) -> None:
        """
        Args:
            tp_threshold (float): Match threshold in meters the errors are read at.
            flip_threshold (float): Absolute wrapped yaw error in radians above which a TP counts as flipped.
        """
        self.tp_threshold = float(tp_threshold)
        self.flip_threshold = float(flip_threshold)

    def evaluate(self, state: DetectionState) -> Dict[str, float]:
        """Report per-class heading-flip rate and count at the TP threshold."""
        report: Dict[str, float] = {}
        rates = []
        for label in state.labels(full=True):
            curve = state.match_curve(label, self.tp_threshold)
            errors = curve.orientation_error[curve.true_positive == 1.0]
            errors = errors[~np.isnan(errors)]
            name = label_metric_name(label, state.class_names)
            if errors.shape[0] == 0:
                report[f"flip_rate_{name}"] = float("nan")
                report[f"flip_count_{name}"] = 0.0
                continue
            flips = float(np.sum(errors > self.flip_threshold))
            rate = flips / float(errors.shape[0])
            report[f"flip_count_{name}"] = flips
            report[f"flip_rate_{name}"] = rate
            rates.append(rate)

        report["mflip_rate"] = mean_valid(rates)
        return report
