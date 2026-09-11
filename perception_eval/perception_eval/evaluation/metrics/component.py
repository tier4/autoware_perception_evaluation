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
# Ported from tier4/autoware-ml (autoware_ml/metrics/base.py) at fcf86419.

"""Task-agnostic primitives of the driving-aware metric framework.

The framework separates two concerns:

* A **suite** owns the accumulated frames and builds a task ``state`` for each
  ``(taxonomy, filter, range)`` view. It knows nothing about which metrics run.
* A **component** is a small, stateless object that computes its own numbers from that state.

This module holds only the shared pieces: :class:`MetricRange` (radial distance buckets), the
component protocols, :class:`MetricReport` (the values every suite emits) and the deterministic
metric-key composition ``<task>/<taxonomy?>/<filter?>/<range?>/<metric-key>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.naming import distance_token

KEY_SEPARATOR = "/"
GROUPED_TAXONOMY = "grouped"


@dataclass(frozen=True)
class MetricRange:
    """Radial (BEV) distance window in meters used to bucket metrics by range.

    Attributes:
        name (str): Human-readable label, kept for config clarity (not used in keys).
        min_distance (float): Inclusive lower bound in meters.
        max_distance (float | None): Exclusive upper bound in meters, or ``None`` for unbounded.
    """

    name: str
    min_distance: float = 0.0
    max_distance: Optional[float] = None

    def __post_init__(self) -> None:
        if self.min_distance < 0.0:
            raise ValueError(f"MetricRange {self.name!r}: min_distance must be >= 0, got {self.min_distance}.")
        if self.max_distance is not None and self.max_distance <= self.min_distance:
            raise ValueError(
                f"MetricRange {self.name!r}: max_distance ({self.max_distance}) must be greater than "
                f"min_distance ({self.min_distance})."
            )

    @property
    def suffix(self) -> str:
        """Collision-free key suffix, e.g. ``0m_50m`` or ``90m_inf``."""
        return range_suffix(self)

    def contains(self, distances: NDArray) -> NDArray[np.bool_]:
        """Boolean mask of ``distances`` (meters) inside ``[min_distance, max_distance)``."""
        distances = np.asarray(distances, dtype=np.float64)
        mask = distances >= self.min_distance
        if self.max_distance is not None:
            mask &= distances < self.max_distance
        return mask


def range_suffix(metric_range: MetricRange) -> str:
    """Collision-free key suffix for a range, e.g. ``0m_50m`` or ``90m_inf``."""
    lower = distance_token(metric_range.min_distance)
    if metric_range.max_distance is None:
        return f"{lower}_inf"
    return f"{lower}_{distance_token(metric_range.max_distance)}"


def validate_ranges(ranges: Sequence[MetricRange]) -> Tuple[MetricRange, ...]:
    """Return ``ranges`` as a tuple after rejecting duplicate names or key suffixes."""
    ranges = tuple(ranges)
    names = [metric_range.name for metric_range in ranges]
    duplicated_names = sorted({name for name in names if names.count(name) > 1})
    if duplicated_names:
        raise ValueError(f"Range names must be unique: {duplicated_names}")
    suffixes = [range_suffix(metric_range) for metric_range in ranges]
    duplicated_suffixes = sorted({suffix for suffix in suffixes if suffixes.count(suffix) > 1})
    if duplicated_suffixes:
        raise ValueError(f"Range metric suffixes must be unique: {duplicated_suffixes}")
    return ranges


StateT = TypeVar("StateT")
ViewT = TypeVar("ViewT")


class MetricComponent(Protocol[StateT]):
    """One batch-oriented metric. Owns its computation and holds no accumulated state."""

    def evaluate(self, state: StateT) -> Dict[str, float]:
        """Compute this component's keys from the suite's ``state``."""
        ...


class StreamingComponent(Protocol[ViewT]):
    """One streaming metric: consumes one frame view at a time into bounded statistics."""

    def update(self, view: ViewT) -> None:
        """Accumulate one frame view."""
        ...

    def compute(self) -> Dict[str, float]:
        """Return this component's keys from the accumulated statistics."""
        ...

    def reset(self) -> None:
        """Drop the accumulated statistics."""
        ...


def compose_key(
    task: str,
    metric_key: str,
    taxonomy: Optional[str] = None,
    filter_name: Optional[str] = None,
    range_suffix_: Optional[str] = None,
) -> str:
    """Compose ``<task>/<taxonomy?>/<filter?>/<range?>/<metric-key>``; empty levels are skipped."""
    levels = [task, taxonomy, filter_name, range_suffix_, metric_key]
    parts = [str(level) for level in levels if level]
    if not metric_key:
        raise ValueError("metric_key must not be empty.")
    for part in parts:
        if KEY_SEPARATOR in part:
            raise ValueError(f"metric-key level {part!r} must not contain {KEY_SEPARATOR!r}.")
    return KEY_SEPARATOR.join(parts)


def flatten_key(key: str, separator: str = "_") -> str:
    """Legacy flat form of a slash-separated key, e.g. ``detection_road_0m_30m_corner_mean_car``."""
    return key.replace(KEY_SEPARATOR, separator)


def mean_valid(values: Iterable[float]) -> float:
    """Mean of the non-NaN values, or NaN when none are valid."""
    valid_values = [float(value) for value in values if not math.isnan(float(value))]
    if not valid_values:
        return float("nan")
    return float(sum(valid_values) / len(valid_values))


@dataclass
class MetricReport:
    """Values emitted by one metric suite for one scene.

    Attributes:
        values (dict[str, float]): Slash-separated metric key to value (NaN when undefined).
        coverage (dict[str, tuple[int, int]]): Availability-gated view name to
            ``(covered_frames, seen_frames)``.
        warnings (list[str]): Human-readable notes (partial coverage, skipped frames, ...).
    """

    values: Dict[str, float] = field(default_factory=dict)
    coverage: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def add(self, key: str, value: float) -> None:
        """Add one value, refusing to overwrite an existing key."""
        if key in self.values:
            raise ValueError(f"Two metrics emit the same key {key!r}. Give one a distinct name.")
        self.values[key] = float(value)

    def update(self, values: Dict[str, float]) -> None:
        """Add many values, refusing to overwrite existing keys."""
        for key, value in values.items():
            self.add(key, value)

    def warn(self, message: str) -> None:
        """Record a warning once."""
        if message not in self.warnings:
            self.warnings.append(message)

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Value for ``key`` or ``default``."""
        return self.values.get(key, default)

    def select(self, prefix: str) -> Dict[str, float]:
        """Values whose key starts with ``prefix``."""
        return {key: value for key, value in self.values.items() if key.startswith(prefix)}

    def to_flat_keys(self, separator: str = "_") -> Dict[str, float]:
        """Values with slashes replaced by ``separator``; raises when two keys collide."""
        flat: Dict[str, float] = {}
        for key, value in self.values.items():
            flat_key = flatten_key(key, separator)
            if flat_key in flat:
                raise ValueError(f"Flat key collision for {flat_key!r}.")
            flat[flat_key] = value
        return flat

    def serialization(self) -> Dict[str, Any]:
        """JSON-friendly dict (NaN is kept as ``float('nan')``)."""
        return {
            "values": dict(self.values),
            "coverage": {name: list(pair) for name, pair in self.coverage.items()},
            "warnings": list(self.warnings),
        }

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> MetricReport:
        """Inverse of :meth:`serialization`."""
        return cls(
            values={key: float(value) for key, value in data.get("values", {}).items()},
            coverage={name: (int(pair[0]), int(pair[1])) for name, pair in data.get("coverage", {}).items()},
            warnings=list(data.get("warnings", [])),
        )

    def summary(self, max_rows: int = 40, exclude_prefixes: Tuple[str, ...] = ("confusion_",)) -> str:
        """Concise multi-line summary; confusion cells are excluded by default."""
        lines: List[str] = []
        if self.coverage:
            lines.append("Coverage (covered/seen frames):")
            for name, (covered, seen) in self.coverage.items():
                lines.append(f"  {name}: {covered}/{seen}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        rows = [
            (key, value)
            for key, value in self.values.items()
            if not any(key.rsplit(KEY_SEPARATOR, 1)[-1].startswith(prefix) for prefix in exclude_prefixes)
        ]
        lines.append(f"Values ({len(rows)} shown of {len(self.values)}):")
        for key, value in rows[:max_rows]:
            lines.append(f"  {key}: {value:.4f}" if not math.isnan(value) else f"  {key}: nan")
        if len(rows) > max_rows:
            lines.append(f"  ... {len(rows) - max_rows} more")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
