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
# Ported from tier4/autoware-ml (autoware_ml/metrics/confusion_report.py) at fcf86419.

"""Shared emission of a confusion matrix as flat metric keys.

Both the detection (matched-pair) and segmentation (point) confusion metrics turn their
``(C, C)`` matrix (rows = true class, columns = predicted class) into ``confusion_<true>__<pred>``
count keys. The ``__`` separator is safe because no class name contains a double underscore.
Raw counts are emitted (not fractions) so both the absolute count and the per-true-class rate stay
recoverable.
"""

from __future__ import annotations

from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.naming import metric_token

CONFUSION_PREFIX = "confusion_"
CONFUSION_SEPARATOR = "__"


def confusion_class_token(index: int, class_names: Optional[Sequence[str]]) -> str:
    """Sanitized per-class key token, ``class_{index}`` only when no names exist.

    With ``class_names`` configured an out-of-range index is a folding or configuration bug and
    raises instead of silently minting a phantom class key.
    """
    if class_names is None:
        return f"class_{index}"
    if not 0 <= index < len(class_names):
        raise ValueError(f"class index {index} is outside the configured class set ({len(class_names)} classes).")
    return metric_token(class_names[index])


def confusion_key(true_index: int, pred_index: int, class_names: Optional[Sequence[str]]) -> str:
    """The ``confusion_<true>__<pred>`` key for one cell."""
    true_name = confusion_class_token(true_index, class_names)
    pred_name = confusion_class_token(pred_index, class_names)
    return f"{CONFUSION_PREFIX}{true_name}{CONFUSION_SEPARATOR}{pred_name}"


def confusion_cells(matrix: NDArray, class_names: Optional[Sequence[str]]) -> Dict[str, float]:
    """Flatten a ``(C, C)`` true vs. pred matrix into ``confusion_<true>__<pred>`` counts."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"confusion matrix must be square, got shape {matrix.shape}.")
    if class_names is not None and len(class_names) != matrix.shape[0]:
        raise ValueError(f"confusion matrix has {matrix.shape[0]} classes but {len(class_names)} names were given.")
    num_classes = int(matrix.shape[0])
    cells: Dict[str, float] = {}
    for true_index in range(num_classes):
        for pred_index in range(num_classes):
            cells[confusion_key(true_index, pred_index, class_names)] = float(matrix[true_index, pred_index])
    return cells
