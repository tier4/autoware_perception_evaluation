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
# Ported from tier4/autoware-ml (autoware_ml/metrics/segmentation3d/{iou,accuracy,precision_recall_f1}.py)
# at fcf86419.

"""Baseline segmentation scores derived from the confusion matrix (IoU, accuracy, P/R/F1).

These are acceptance checks rather than new metrics of the ported change; they read the same
:class:`ConfusionState` as the confusion-matrix component.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.naming import label_metric_name
from perception_eval.evaluation.metrics.segmentation.confusion_matrix import ConfusionState


def macro(values: NDArray, has_support: NDArray) -> float:
    """Mean over classes with ground-truth support, or NaN when none have it."""
    if not bool(np.any(has_support)):
        return float("nan")
    return float(np.mean(values[has_support]))


def _safe_ratio(numerator: NDArray, denominator: NDArray) -> NDArray[np.float64]:
    zeros = np.zeros_like(numerator, dtype=np.float64)
    return np.divide(numerator, denominator, out=zeros, where=denominator > 0)


class IoU:
    """Macro mean IoU, frequency-weighted IoU and per-class IoU for supported classes."""

    kind = "confusion"

    def evaluate(self, state: ConfusionState) -> Dict[str, float]:
        union = state.predicted + state.actual - state.true_positive
        iou = _safe_ratio(state.true_positive, union)
        report = {"mIoU": macro(iou, state.has_support)}
        if bool(state.has_support.any()):
            report["fwIoU"] = float((state.frequency * iou)[state.has_support].sum())
        else:
            report["fwIoU"] = float("nan")
        for class_index in range(state.num_classes):
            if not bool(state.has_support[class_index]):
                continue
            report[f"iou_{label_metric_name(class_index, state.class_names)}"] = float(iou[class_index])
        return report


class Accuracy:
    """Micro point accuracy: correct points over all valid points."""

    kind = "confusion"

    def evaluate(self, state: ConfusionState) -> Dict[str, float]:
        if state.total > 0:
            return {"acc": float(state.true_positive.sum() / state.total)}
        return {"acc": float("nan")}


class PrecisionRecallF1:
    """Macro recall/precision/F1 plus the per-class breakdown for supported classes."""

    kind = "confusion"

    def evaluate(self, state: ConfusionState) -> Dict[str, float]:
        recall = _safe_ratio(state.true_positive, state.actual)
        precision = _safe_ratio(state.true_positive, state.predicted)
        f1 = _safe_ratio(2.0 * state.true_positive, state.predicted + state.actual)
        report = {
            "mRecall": macro(recall, state.has_support),
            "mPrecision": macro(precision, state.has_support),
            "mF1": macro(f1, state.has_support),
        }
        for class_index in range(state.num_classes):
            if not bool(state.has_support[class_index]):
                continue
            name = label_metric_name(class_index, state.class_names)
            report[f"recall_{name}"] = float(recall[class_index])
            report[f"precision_{name}"] = float(precision[class_index])
            report[f"f1_{name}"] = float(f1[class_index])
        return report
