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

"""Typed configuration of the point-cloud segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.metrics.class_groups import resolve_class_groups
from perception_eval.evaluation.metrics.class_groups import validate_name_tokens
from perception_eval.evaluation.metrics.component import MetricRange
from perception_eval.evaluation.metrics.filter_config import filter_config_serialization
from perception_eval.evaluation.metrics.filter_config import FilterConfigType
from perception_eval.evaluation.metrics.filter_config import MapConfig
from perception_eval.evaluation.metrics.filter_config import MetricsConfigError
from perception_eval.evaluation.metrics.filter_config import needs_map
from perception_eval.evaluation.metrics.filter_config import parse_filters
from perception_eval.evaluation.metrics.filter_config import parse_map
from perception_eval.evaluation.metrics.filter_config import parse_ranges
from perception_eval.evaluation.metrics.segmentation.registry import ComponentSpec
from perception_eval.evaluation.metrics.segmentation.registry import parse_components

SEGMENTATION_METRIC_KEYS: Tuple[str, ...] = (
    "class_names",
    "ignore_index",
    "ranges",
    "class_groups",
    "filters",
    "components",
    "map",
    "box_label_to_seg_class",
    "check_argmax",
    "probability_tolerance",
    "probability_sum_tolerance",
    "include_confusion_cells",
)

_AUTOWARE_LABEL_VALUES = tuple(label.value for label in AutowareLabel if label is not AutowareLabel.LABEL_TYPE)


@dataclass(frozen=True)
class SegmentationMetricsConfig:
    """Configuration of the segmentation metric suite.

    Attributes:
        class_names (tuple[str, ...]): Trained class names; their order is the probability column order.
        ignore_index (int): Target label excluded from every metric.
        ranges (tuple[MetricRange, ...]): Radial BEV windows every metric is also reported for.
        class_groups (dict[str, tuple[str, ...]] | None): Behaviour groups (a full partition of
            ``class_names``); when set, a ``grouped`` view is reported alongside the trained one.
        filters (tuple): Spatial filters (corridor / region / collision); identity is implicit.
        components (tuple[ComponentSpec, ...]): Metric components to run.
        map (MapConfig | None): Lanelet map resolution, required by map-dependent filters.
        box_label_to_seg_class (dict[str, str]): ``AutowareLabel`` value -> segmentation class name
            whose points a detection box should carry (partial-detection metric).
        check_argmax (bool): Require ``predictions == argmax(probabilities)``.
        probability_tolerance (float): Allowed excursion of probabilities outside ``[0, 1]``.
        probability_sum_tolerance (float): Allowed deviation of each row sum from one.
        include_confusion_cells (bool): Emit ``confusion_<true>__<pred>`` keys.
    """

    class_names: Tuple[str, ...]
    ignore_index: int = -1
    ranges: Tuple[MetricRange, ...] = ()
    class_groups: Optional[Dict[str, Tuple[str, ...]]] = None
    filters: Tuple[FilterConfigType, ...] = ()
    components: Tuple[ComponentSpec, ...] = ()
    map: Optional[MapConfig] = None
    box_label_to_seg_class: Dict[str, str] = field(default_factory=dict)
    check_argmax: bool = True
    probability_tolerance: float = 1e-6
    probability_sum_tolerance: float = 1e-3
    include_confusion_cells: bool = True

    def __post_init__(self) -> None:
        class_names = tuple(str(name) for name in self.class_names)
        if len(class_names) < 2:
            raise MetricsConfigError("class_names must list at least two classes (entropy is normalized by log C).")
        if len(set(class_names)) != len(class_names):
            raise MetricsConfigError(f"class_names must be unique, got {list(class_names)}.")
        try:
            validate_name_tokens(class_names)
        except ValueError as error:
            raise MetricsConfigError(str(error)) from error
        object.__setattr__(self, "class_names", class_names)
        if 0 <= int(self.ignore_index) < len(class_names):
            raise MetricsConfigError(f"ignore_index {self.ignore_index} collides with a class index.")
        object.__setattr__(self, "ignore_index", int(self.ignore_index))
        object.__setattr__(self, "ranges", parse_ranges(self.ranges))
        object.__setattr__(self, "filters", parse_filters(self.filters))
        object.__setattr__(self, "components", parse_components(self.components))
        object.__setattr__(self, "map", parse_map(self.map))
        if self.class_groups is not None:
            groups = {
                str(name): tuple(str(member) for member in members) for name, members in self.class_groups.items()
            }
            try:
                _, grouped_names = resolve_class_groups(class_names, groups)
            except ValueError as error:
                raise MetricsConfigError(f"class_groups: {error}") from error
            if len(grouped_names) < 2:
                raise MetricsConfigError("class_groups must keep >= 2 groups (entropy normalization).")
            object.__setattr__(self, "class_groups", groups)
        mapping = {str(key): str(value) for key, value in dict(self.box_label_to_seg_class or {}).items()}
        for key, value in mapping.items():
            if key not in _AUTOWARE_LABEL_VALUES:
                raise MetricsConfigError(
                    f"box_label_to_seg_class key {key!r} is not an AutowareLabel value; valid: {list(_AUTOWARE_LABEL_VALUES)}."
                )
            if value not in class_names:
                raise MetricsConfigError(f"box_label_to_seg_class value {value!r} is not in class_names.")
        object.__setattr__(self, "box_label_to_seg_class", mapping)
        if self.needs_boxes and not mapping:
            raise MetricsConfigError("partial_detection needs a non-empty box_label_to_seg_class mapping.")
        if needs_map(self.filters) and self.map is None:
            raise MetricsConfigError("a region/collision filter is configured but the `map` section is missing.")
        if float(self.probability_tolerance) < 0.0 or float(self.probability_sum_tolerance) < 0.0:
            raise MetricsConfigError("probability tolerances must be >= 0.")

    # ------------------------------------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def needs_boxes(self) -> bool:
        return any(spec.needs_boxes for spec in self.components)

    @property
    def needs_map(self) -> bool:
        return needs_map(self.filters)

    @property
    def det_class_names(self) -> Tuple[str, ...]:
        return tuple(self.box_label_to_seg_class.keys())

    def resolve_groups(self) -> Tuple[Optional[NDArray[np.int64]], Optional[Tuple[str, ...]]]:
        """``(lut, grouped_names)`` or ``(None, None)`` when no groups are configured."""
        if self.class_groups is None:
            return None, None
        return resolve_class_groups(self.class_names, self.class_groups)

    # ------------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> SegmentationMetricsConfig:
        """Build from a plain dict (unknown keys are rejected)."""
        unknown = sorted(set(cfg) - set(SEGMENTATION_METRIC_KEYS))
        if unknown:
            raise MetricsConfigError(
                f"unknown segmentation metrics keys {unknown}; allowed: {list(SEGMENTATION_METRIC_KEYS)}."
            )
        if "class_names" not in cfg:
            raise MetricsConfigError("segmentation metrics need class_names.")
        kwargs: Dict[str, Any] = {key: cfg[key] for key in SEGMENTATION_METRIC_KEYS if key in cfg}
        kwargs["class_names"] = tuple(kwargs["class_names"])
        return cls(**kwargs)

    def serialization(self) -> Dict[str, Any]:
        return {
            "class_names": list(self.class_names),
            "ignore_index": self.ignore_index,
            "ranges": [
                {"name": r.name, "min_distance": r.min_distance, "max_distance": r.max_distance} for r in self.ranges
            ],
            "class_groups": None if self.class_groups is None else {k: list(v) for k, v in self.class_groups.items()},
            "filters": [filter_config_serialization(f) for f in self.filters],
            "components": [spec.serialization() for spec in self.components],
            "map": None if self.map is None else self.map.serialization(),
            "box_label_to_seg_class": dict(self.box_label_to_seg_class),
            "check_argmax": self.check_argmax,
            "probability_tolerance": self.probability_tolerance,
            "probability_sum_tolerance": self.probability_sum_tolerance,
            "include_confusion_cells": self.include_confusion_cells,
        }

    @classmethod
    def deserialization(cls, data: Mapping[str, Any]) -> SegmentationMetricsConfig:
        return cls.from_dict(data)

    def __reduce__(self):
        return (_deserialize, (self.serialization(),))


def _deserialize(data: Dict[str, Any]) -> SegmentationMetricsConfig:
    return SegmentationMetricsConfig.from_dict(data)


def _parse_class_groups(entry: Optional[Mapping[str, Sequence[str]]]) -> Optional[Dict[str, Tuple[str, ...]]]:
    if entry is None:
        return None
    return {str(k): tuple(v) for k, v in entry.items()}
