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

"""Typed, closed configuration of the opt-in advanced detection metrics.

The ``advanced_detection_metrics`` section of the evaluation config is parsed into frozen
dataclasses. ``type`` tokens map to exactly one component config; nothing is instantiated from
arbitrary strings. Absence of the section means legacy behaviour (mAP/mAPH only).

Example::

    advanced_detection_metrics:
      ranges: [{name: 0_30, min_distance: 0.0, max_distance: 30.0}]
      class_groups: {grouped_vehicle: [car, truck, bus], grouped_vru: [pedestrian, bicycle, motorbike]}
      filters: [{name: corridor, type: corridor, width_m: 3.0}]
      components:
        - {type: corner_error, tp_threshold: 2.0, percentiles: [95.0]}
        - {type: heading_flip}
      map: {resolver: t4_scene_directory}
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from math import pi
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

from perception_eval.common.label import LabelType
from perception_eval.evaluation.metrics.class_groups import resolve_class_groups
from perception_eval.evaluation.metrics.component import MetricRange
from perception_eval.evaluation.metrics.detection.adapter import class_names_of
from perception_eval.evaluation.metrics.detection.state import DEFAULT_MATCH_THRESHOLDS
from perception_eval.evaluation.metrics.detection.state import DEFAULT_TP_THRESHOLD
from perception_eval.evaluation.metrics.detection.state import MATCH_COSTS
from perception_eval.evaluation.metrics.filter_config import filter_config_serialization
from perception_eval.evaluation.metrics.filter_config import FilterConfigType
from perception_eval.evaluation.metrics.filter_config import MapConfig
from perception_eval.evaluation.metrics.filter_config import MetricsConfigError
from perception_eval.evaluation.metrics.filter_config import needs_map
from perception_eval.evaluation.metrics.filter_config import parse_filters
from perception_eval.evaluation.metrics.filter_config import parse_map
from perception_eval.evaluation.metrics.filter_config import parse_ranges

# ------------------------------------------------------------------------------------------------
# Component configs (one frozen dataclass per ``type`` token)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CornerErrorConfig:
    tp_threshold: float = DEFAULT_TP_THRESHOLD
    percentiles: Tuple[float, ...] = (95.0,)

    needs_ttc = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "percentiles", tuple(float(p) for p in self.percentiles))
        for p in self.percentiles:
            if not 0.0 <= p <= 100.0:
                raise ValueError(f"percentile {p} outside [0, 100].")


@dataclass(frozen=True)
class HeadingFlipConfig:
    tp_threshold: float = DEFAULT_TP_THRESHOLD
    flip_threshold: float = pi / 2.0

    needs_ttc = False


@dataclass(frozen=True)
class NearestSurfaceErrorConfig:
    tp_threshold: float = DEFAULT_TP_THRESHOLD
    low_percentile: float = 5.0
    high_percentile: float = 95.0

    needs_ttc = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_percentile < self.high_percentile <= 100.0:
            raise ValueError("expected 0 <= low_percentile < high_percentile <= 100.")


@dataclass(frozen=True)
class CalibrationConfig:
    tp_threshold: float = DEFAULT_TP_THRESHOLD
    num_bins: int = 15

    needs_ttc = False

    def __post_init__(self) -> None:
        if self.num_bins < 1:
            raise ValueError("num_bins must be >= 1.")


@dataclass(frozen=True)
class ConfidentErrorConfig:
    tp_threshold: float = DEFAULT_TP_THRESHOLD
    score_threshold: float = 0.5
    min_score: float = 0.1

    needs_ttc = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= self.score_threshold <= 1.0:
            raise ValueError("expected 0 <= min_score <= score_threshold <= 1.")


@dataclass(frozen=True)
class ConfusionMatrixConfig:
    match_threshold: float = DEFAULT_TP_THRESHOLD
    min_score: float = 0.1

    needs_ttc = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1].")


@dataclass(frozen=True)
class CriticalFPFNConfig:
    confidences: Tuple[float, ...] = (0.3, 0.5)
    match_threshold: float = DEFAULT_TP_THRESHOLD

    needs_ttc = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidences", tuple(float(c) for c in self.confidences))
        if not self.confidences:
            raise ValueError("confidences must not be empty.")


@dataclass(frozen=True)
class CollisionWeightedMapConfig:
    thresholds: Tuple[float, ...] = DEFAULT_MATCH_THRESHOLDS
    decay: float = 0.5

    needs_ttc = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "thresholds", tuple(float(t) for t in self.thresholds))
        if not self.thresholds:
            raise ValueError("thresholds must not be empty.")
        if self.decay < 0.0:
            raise ValueError("decay must be >= 0.")


ComponentConfigType = Union[
    CornerErrorConfig,
    HeadingFlipConfig,
    NearestSurfaceErrorConfig,
    CalibrationConfig,
    ConfidentErrorConfig,
    ConfusionMatrixConfig,
    CriticalFPFNConfig,
    CollisionWeightedMapConfig,
]

COMPONENT_TYPES: Dict[str, Type[Any]] = {
    "corner_error": CornerErrorConfig,
    "heading_flip": HeadingFlipConfig,
    "nearest_surface_error": NearestSurfaceErrorConfig,
    "calibration": CalibrationConfig,
    "confident_error": ConfidentErrorConfig,
    "confusion_matrix": ConfusionMatrixConfig,
    "critical_fp_fn": CriticalFPFNConfig,
    "collision_weighted_map": CollisionWeightedMapConfig,
}


def component_token(config: ComponentConfigType) -> str:
    """The ``type`` token of a component config."""
    for token, cls in COMPONENT_TYPES.items():
        if isinstance(config, cls):
            return token
    raise MetricsConfigError(f"unsupported component config {type(config).__name__}.")


def component_kwargs(config: ComponentConfigType) -> Dict[str, Any]:
    """Constructor kwargs of the component class for one config."""
    return {f.name: getattr(config, f.name) for f in fields(config)}


def parse_components(
    entries: Optional[Sequence[Union[Mapping[str, Any], ComponentConfigType]]],
) -> Tuple[ComponentConfigType, ...]:
    """Parse ``[{type, ...params}, ...]`` into validated component configs (duplicates rejected)."""
    if not entries:
        raise MetricsConfigError("advanced_detection_metrics.components must list at least one component.")
    configs: List[ComponentConfigType] = []
    seen: List[str] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, tuple(COMPONENT_TYPES.values())):
            config = entry
        else:
            if not isinstance(entry, Mapping):
                raise MetricsConfigError(f"components[{index}] must be a mapping, got {type(entry).__name__}.")
            token = entry.get("type")
            if token not in COMPONENT_TYPES:
                raise MetricsConfigError(
                    f"components[{index}]: unknown component type {token!r}; valid: {sorted(COMPONENT_TYPES)}."
                )
            config_cls = COMPONENT_TYPES[token]
            allowed = {f.name for f in fields(config_cls)}
            unknown = sorted(set(entry) - allowed - {"type"})
            if unknown:
                raise MetricsConfigError(
                    f"components[{index}] ({token}): unknown keys {unknown}; allowed keys are {sorted(allowed)}."
                )
            try:
                config = config_cls(**{k: v for k, v in entry.items() if k != "type"})
            except (TypeError, ValueError) as error:
                raise MetricsConfigError(f"components[{index}] ({token}): {error}") from error
        token = component_token(config)
        if token in seen:
            raise MetricsConfigError(f"components[{index}]: component type {token!r} is configured twice.")
        seen.append(token)
        configs.append(config)
    return tuple(configs)


# ------------------------------------------------------------------------------------------------
# Top-level config
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AdvancedDetectionMetricsConfig:
    """Validated ``advanced_detection_metrics`` section.

    Attributes:
        class_names (tuple[str, ...]): Metric-key class names in ``target_labels`` order.
        ranges (tuple[MetricRange, ...]): Radial BEV windows; empty means no range level in keys.
        class_groups (dict[str, tuple[str, ...]] | None): Full partition of ``class_names`` defining
            the ``grouped`` taxonomy view, or ``None`` for no grouped view.
        filters (tuple): Non-identity filter configs (the identity view is always evaluated).
        components (tuple): Component configs to run on every view.
        map (MapConfig | None): Lanelet map resolution, required by map-dependent filters and TTC.
        match_cost (str): ``center`` (default) or ``corner`` matching cost for the TP-based metrics.
    """

    class_names: Tuple[str, ...]
    ranges: Tuple[MetricRange, ...] = ()
    class_groups: Optional[Dict[str, Tuple[str, ...]]] = None
    filters: Tuple[FilterConfigType, ...] = ()
    components: Tuple[ComponentConfigType, ...] = ()
    map: Optional[MapConfig] = None
    match_cost: str = "center"

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_names", tuple(str(name) for name in self.class_names))
        if not self.class_names:
            raise MetricsConfigError("advanced_detection_metrics needs at least one target label.")
        if self.match_cost not in MATCH_COSTS:
            raise MetricsConfigError(f"match_cost must be one of {list(MATCH_COSTS)}, got {self.match_cost!r}.")
        if not self.components:
            raise MetricsConfigError("advanced_detection_metrics.components must list at least one component.")
        if self.class_groups is not None:
            groups = {
                str(name): tuple(str(member) for member in members) for name, members in self.class_groups.items()
            }
            object.__setattr__(self, "class_groups", groups)
            try:
                resolve_class_groups(self.class_names, groups)
            except ValueError as error:
                raise MetricsConfigError(f"class_groups: {error}") from error
            unknown_members = sorted(
                {member for members in groups.values() for member in members} - set(self.class_names)
            )
            if unknown_members:
                raise MetricsConfigError(
                    f"class_groups reference classes {unknown_members} that are not target labels "
                    f"{list(self.class_names)} (use AutowareLabel names such as 'motorbike', 'hazard', 'unknown')."
                )
        if self.requires_map and self.map is None:
            raise MetricsConfigError(
                "region/collision filters and critical_fp_fn/collision_weighted_map components need the `map` section."
            )

    @property
    def needs_ttc(self) -> bool:
        return any(getattr(component, "needs_ttc", False) for component in self.components) or any(
            type(config).__name__ == "CollisionFilterConfig" for config in self.filters
        )

    @property
    def requires_map(self) -> bool:
        return needs_map(self.filters) or any(getattr(component, "needs_ttc", False) for component in self.components)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], target_labels: Sequence[LabelType]) -> AdvancedDetectionMetricsConfig:
        """Parse the raw section for the configured ``target_labels``."""
        if not isinstance(raw, Mapping):
            raise MetricsConfigError(f"advanced_detection_metrics must be a mapping, got {type(raw).__name__}.")
        allowed = {"ranges", "class_groups", "filters", "components", "map", "match_cost"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise MetricsConfigError(f"advanced_detection_metrics: unknown keys {unknown}; allowed {sorted(allowed)}.")
        class_groups = raw.get("class_groups")
        if class_groups is not None and not isinstance(class_groups, Mapping):
            raise MetricsConfigError("advanced_detection_metrics.class_groups must be a mapping {group: [classes]}.")
        return cls(
            class_names=class_names_of(target_labels),
            ranges=parse_ranges(raw.get("ranges")),
            class_groups=None if class_groups is None else {k: tuple(v) for k, v in class_groups.items()},
            filters=parse_filters(raw.get("filters")),
            components=parse_components(raw.get("components")),
            map=parse_map(raw.get("map")),
            match_cost=str(raw.get("match_cost", "center")),
        )

    def serialization(self) -> Dict[str, Any]:
        """Raw-dict form (without ``class_names``, which come from ``target_labels``)."""
        data: Dict[str, Any] = {
            "ranges": [
                {"name": r.name, "min_distance": r.min_distance, "max_distance": r.max_distance} for r in self.ranges
            ],
            "filters": [filter_config_serialization(config) for config in self.filters],
            "components": [{"type": component_token(c), **component_kwargs(c)} for c in self.components],
            "match_cost": self.match_cost,
        }
        for component in data["components"]:
            for key, value in list(component.items()):
                if isinstance(value, tuple):
                    component[key] = list(value)
        if self.class_groups is not None:
            data["class_groups"] = {name: list(members) for name, members in self.class_groups.items()}
        if self.map is not None:
            data["map"] = self.map.serialization()
        return data
