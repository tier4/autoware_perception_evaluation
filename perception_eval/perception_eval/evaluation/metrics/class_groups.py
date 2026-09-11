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
# Ported from tier4/autoware-ml (autoware_ml/metrics/class_groups.py) at fcf86419.

"""Behaviour-taxonomy class folding, shared by the detection and segmentation suites.

A ``class_groups`` spec maps a grouped class name to the trained class names it contains, e.g.
``{"grouped_vehicle": ["car", "truck", "bus"]}``. It is a *full* taxonomy: every trained class
belongs to exactly one group and singleton groups are allowed, so the whole label space is
remapped onto behaviour-equivalent classes and intra-group confusion counts as correct.

A member that is not a trained class is a forward-compatibility slot: it is skipped as long as its
group keeps at least one trained member.

There is no "additive" mode: grouping always replaces. A suite is evaluated twice, once without
``class_groups`` (per class) and once with it (grouped), so both taxonomies are reported side by side.
"""

from __future__ import annotations

from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.naming import metric_token


def resolve_class_groups(
    class_names: Optional[Sequence[str]],
    class_groups: Mapping[str, Sequence[str]],
) -> Tuple[NDArray[np.int64], Tuple[str, ...]]:
    """Return the trained-index to grouped-index LUT and the ordered grouped names.

    The groups must partition every trained class exactly once (singletons allowed), and a member
    absent from ``class_names`` is skipped as a forward-compatibility slot as long as its group
    keeps at least one trained member. Grouped classes are ordered by definition order.
    """
    if class_names is None:
        raise ValueError("class_groups requires class_names so member names can be resolved.")
    if not class_groups:
        raise ValueError("class_groups must define at least one group.")
    class_names = tuple(str(name) for name in class_names)
    name_to_index = {name: index for index, name in enumerate(class_names)}
    lut = np.full(len(class_names), -1, dtype=np.int64)
    grouped_names: list[str] = []
    claimed: dict[int, str] = {}
    for group_name, members in class_groups.items():
        group_name = str(group_name)
        if group_name in name_to_index:
            raise ValueError(f"Grouped class name {group_name!r} collides with a trained class.")
        indices = []
        for member in members:
            member = str(member)
            if member not in name_to_index:
                continue  # forward-compatibility slot for a not-yet-trained class
            index = name_to_index[member]
            if index in claimed:
                raise ValueError(f"Class {member!r} appears in both {claimed[index]!r} and {group_name!r}.")
            claimed[index] = group_name
            indices.append(index)
        if not indices:
            raise ValueError(
                f"Grouped class {group_name!r} has no trained member, at least one of "
                f"{list(members)} must be a trained class."
            )
        position = len(grouped_names)
        grouped_names.append(group_name)
        for index in indices:
            lut[index] = position
    unassigned = [class_names[i] for i in range(len(class_names)) if lut[i] < 0]
    if unassigned:
        raise ValueError(f"class_groups must cover every trained class exactly once, unassigned: {unassigned}.")
    validate_name_tokens(grouped_names)
    return lut, tuple(grouped_names)


def validate_name_tokens(names: Optional[Sequence[str]]) -> None:
    """Reject class-name sets whose metric-key tokens collide."""
    if not names:
        return
    tokens = [metric_token(name) for name in names]
    duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
    if duplicates:
        raise ValueError(
            f"class names collapse to duplicate metric-key tokens {duplicates}, "
            "per-class values would silently overwrite each other."
        )


def fold_labels(labels: NDArray, lut: NDArray[np.int64]) -> NDArray[np.int64]:
    """Relabel integer class indices through the LUT, out-of-range (ignore) untouched."""
    labels = np.asarray(labels, dtype=np.int64)
    folded = labels.copy()
    in_range = (labels >= 0) & (labels < lut.shape[0])
    folded[in_range] = lut[labels[in_range]]
    return folded


def fold_matrix(lut: NDArray[np.int64], num_grouped: int) -> NDArray[np.float64]:
    """``(C, G)`` 0/1 matrix marginalizing per-class probabilities onto the grouped classes."""
    fold = np.zeros((lut.shape[0], num_grouped), dtype=np.float64)
    fold[np.arange(lut.shape[0]), lut] = 1.0
    return fold


def fold_confusion(confusion: NDArray, lut: NDArray[np.int64], num_grouped: int) -> NDArray:
    """Fold a ``(C, C)`` confusion matrix's rows and columns through the grouped-class LUT."""
    confusion = np.asarray(confusion)
    rows = np.zeros((num_grouped, confusion.shape[1]), dtype=confusion.dtype)
    np.add.at(rows, lut, confusion)
    folded = np.zeros((num_grouped, num_grouped), dtype=confusion.dtype)
    np.add.at(folded.T, lut, rows.T)
    return folded
