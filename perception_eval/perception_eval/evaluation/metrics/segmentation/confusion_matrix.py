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
# Ported from tier4/autoware-ml (autoware_ml/metrics/segmentation3d/confusion.py, suite.py,
# confusion_matrix.py) at fcf86419.

"""Confusion-matrix accumulation and state shared by the confusion-backed segmentation metrics.

A confusion matrix is a bounded sufficient statistic: every confusion-backed metric is an exact
closed form of its integer counts. :class:`ConfusionAccumulator` keeps one ``(C, C)`` int64 matrix
per ``(filter, range)`` bucket and never retains points. :class:`ConfusionState` is what the suite
hands to each metric, with the cheap marginals cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.class_groups import fold_confusion
from perception_eval.evaluation.metrics.confusion_report import confusion_cells


def confusion_counts(targets: NDArray, predictions: NDArray, num_classes: int) -> NDArray[np.int64]:
    """``(C, C)`` counts (rows = true class, columns = prediction) of already-valid label pairs."""
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.shape[0] == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    indices = targets * num_classes + predictions
    return np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes).astype(np.int64)


class ConfusionAccumulator:
    """Stacked ``(F+1, R+1, C, C)`` integer confusion matrices over filter and range buckets."""

    def __init__(self, num_filters: int, num_ranges: int, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.confusion = np.zeros((num_filters + 1, num_ranges + 1, num_classes, num_classes), dtype=np.int64)

    def add(self, filter_bucket: int, range_bucket: int, targets: NDArray, predictions: NDArray) -> None:
        self.confusion[filter_bucket, range_bucket] += confusion_counts(targets, predictions, self.num_classes)

    def matrix(self, filter_bucket: int, range_bucket: int) -> NDArray[np.int64]:
        return self.confusion[filter_bucket, range_bucket]

    def reset(self) -> None:
        self.confusion[...] = 0


@dataclass
class ConfusionState:
    """Confusion matrix for one bucket with cached marginals.

    In a grouped view the matrix is already folded onto the behaviour taxonomy, so ``class_names``
    are the grouped names and every metric reads it exactly as it reads the per-class matrix.
    """

    confusion: NDArray[np.int64]
    class_names: Optional[Tuple[str, ...]]
    num_classes: int

    @cached_property
    def _double(self) -> NDArray[np.float64]:
        return np.asarray(self.confusion, dtype=np.float64)

    @cached_property
    def true_positive(self) -> NDArray[np.float64]:
        return np.diag(self._double)

    @cached_property
    def predicted(self) -> NDArray[np.float64]:
        return self._double.sum(axis=0)

    @cached_property
    def actual(self) -> NDArray[np.float64]:
        return self._double.sum(axis=1)

    @cached_property
    def total(self) -> float:
        return float(self._double.sum())

    @cached_property
    def has_support(self) -> NDArray[np.bool_]:
        return self.actual > 0

    @cached_property
    def frequency(self) -> NDArray[np.float64]:
        if self.total > 0:
            return self.actual / self.total
        return np.zeros_like(self.actual)

    @classmethod
    def from_matrix(
        cls,
        confusion: NDArray,
        class_names: Sequence[str],
        lut: Optional[NDArray[np.int64]] = None,
        grouped_names: Optional[Sequence[str]] = None,
    ) -> ConfusionState:
        """State for the raw taxonomy, or folded onto ``grouped_names`` when ``lut`` is given."""
        if lut is not None:
            if grouped_names is None:
                raise ValueError("grouped_names are required with a fold LUT.")
            folded = fold_confusion(np.asarray(confusion, dtype=np.int64), lut, len(grouped_names))
            return cls(confusion=folded, class_names=tuple(grouped_names), num_classes=len(grouped_names))
        return cls(
            confusion=np.asarray(confusion, dtype=np.int64),
            class_names=tuple(class_names),
            num_classes=len(class_names),
        )


class ConfusionMatrix:
    """Raw point-level confusion counts as ``confusion_<true>__<pred>`` keys."""

    kind = "confusion"

    def __init__(self) -> None:
        pass

    def evaluate(self, state: ConfusionState) -> Dict[str, float]:
        return confusion_cells(state.confusion, state.class_names)
