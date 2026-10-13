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

"""Typed configuration of the shared evaluation axes (ranges, filters, lanelet maps).

Configuration never instantiates arbitrary Python classes: every ``type`` token maps to one
validated frozen dataclass here, and :func:`build_filter` turns a dataclass into the filter object.
Both the advanced detection metrics and the segmentation metrics parse their ``ranges``,
``filters`` and ``map`` sections through this module so the two tasks accept the same YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

from perception_eval.evaluation.metrics.component import MetricRange
from perception_eval.evaluation.metrics.component import validate_ranges
from perception_eval.evaluation.metrics.filters import CorridorFilter
from perception_eval.evaluation.metrics.filters import IDENTITY
from perception_eval.evaluation.metrics.filters import MetricFilter
from perception_eval.evaluation.metrics.filters import validate_filters


class MetricsConfigError(ValueError):
    """Raised for an invalid advanced-metrics configuration."""


def _reject_unknown_keys(entry: Mapping[str, Any], config_cls: type, context: str) -> None:
    allowed = {f.name for f in fields(config_cls)} | {"type"}
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise MetricsConfigError(f"{context}: unknown keys {unknown}; allowed keys are {sorted(allowed - {'type'})}.")


def _build_dataclass(config_cls: type, entry: Mapping[str, Any], context: str) -> Any:
    _reject_unknown_keys(entry, config_cls, context)
    kwargs = {key: value for key, value in entry.items() if key != "type"}
    try:
        return config_cls(**kwargs)
    except TypeError as error:
        raise MetricsConfigError(f"{context}: {error}") from error
    except ValueError as error:
        raise MetricsConfigError(f"{context}: {error}") from error


# --------------------------------------------------------------------------------------------
# Ranges
# --------------------------------------------------------------------------------------------


def parse_ranges(entries: Optional[Sequence[Union[Mapping[str, Any], MetricRange]]]) -> Tuple[MetricRange, ...]:
    """Parse ``[{name, min_distance, max_distance}, ...]`` into validated :class:`MetricRange`s."""
    if not entries:
        return ()
    ranges: List[MetricRange] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, MetricRange):
            ranges.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise MetricsConfigError(f"ranges[{index}] must be a mapping, got {type(entry).__name__}.")
        if "name" not in entry:
            raise MetricsConfigError(f"ranges[{index}] needs a name.")
        allowed = {"name", "min_distance", "max_distance"}
        unknown = sorted(set(entry) - allowed)
        if unknown:
            raise MetricsConfigError(f"ranges[{index}]: unknown keys {unknown}.")
        try:
            ranges.append(
                MetricRange(
                    name=str(entry["name"]),
                    min_distance=float(entry.get("min_distance", 0.0)),
                    max_distance=None if entry.get("max_distance") is None else float(entry["max_distance"]),
                )
            )
        except ValueError as error:
            raise MetricsConfigError(str(error)) from error
    try:
        return validate_ranges(ranges)
    except ValueError as error:
        raise MetricsConfigError(str(error)) from error


# --------------------------------------------------------------------------------------------
# Lanelet map
# --------------------------------------------------------------------------------------------

MAP_RESOLVER_TOKENS = ("t4_scene_directory", "explicit")


@dataclass(frozen=True)
class MapConfig:
    """Lanelet map resolution and the collision-model class table.

    Attributes:
        resolver (str): ``t4_scene_directory`` (``<scene>/map/lanelet2_map.osm``) or ``explicit``.
        data_root (str | None): Optional dataset root the scene ids are relative to.
        mapping (dict[str, str]): ``scene_id -> osm path`` for the explicit resolver.
        region (tuple[str, ...]): Drivable lanelet tokens the collision model is clipped to.
        collision_kinds (dict[str, str]): Class name -> ``wheeled``/``vru``/``static`` overrides.
        vru_speeds (dict[str, float]): Class name -> VRU run speed [m/s] overrides.
        max_speed_mps (float): Off-map wheeled speed fallback.
        ego_body_m (float): Ego collision half-extent [m].
    """

    resolver: str = "t4_scene_directory"
    data_root: Optional[str] = None
    mapping: Dict[str, str] = None  # type: ignore[assignment]
    region: Tuple[str, ...] = ("road", "road_shoulder", "crosswalk")
    collision_kinds: Dict[str, str] = None  # type: ignore[assignment]
    vru_speeds: Dict[str, float] = None  # type: ignore[assignment]
    max_speed_mps: float = 16.7
    ego_body_m: float = 1.0

    def __post_init__(self) -> None:
        if self.resolver not in MAP_RESOLVER_TOKENS:
            raise ValueError(f"map.resolver must be one of {list(MAP_RESOLVER_TOKENS)}, got {self.resolver!r}.")
        object.__setattr__(self, "mapping", dict(self.mapping or {}))
        object.__setattr__(self, "collision_kinds", dict(self.collision_kinds or {}))
        object.__setattr__(self, "vru_speeds", {k: float(v) for k, v in (self.vru_speeds or {}).items()})
        object.__setattr__(self, "region", tuple(str(token) for token in self.region))
        if self.resolver == "explicit" and not self.mapping:
            raise ValueError("map.resolver 'explicit' needs a non-empty mapping {scene_id: osm_path}.")
        if self.max_speed_mps <= 0.0:
            raise ValueError("map.max_speed_mps must be > 0.")
        if self.ego_body_m <= 0.0:
            raise ValueError("map.ego_body_m must be > 0.")

    def build_provider(self) -> Any:
        """Instantiate the lanelet map provider (imports the geometry package lazily)."""
        from perception_eval.evaluation.metrics.geometry.lanelet import build_map_provider

        return build_map_provider(self.resolver, data_root=self.data_root, mapping=self.mapping)

    def serialization(self) -> Dict[str, Any]:
        return {
            "resolver": self.resolver,
            "data_root": self.data_root,
            "mapping": dict(self.mapping),
            "region": list(self.region),
            "collision_kinds": dict(self.collision_kinds),
            "vru_speeds": dict(self.vru_speeds),
            "max_speed_mps": self.max_speed_mps,
            "ego_body_m": self.ego_body_m,
        }


def parse_map(entry: Optional[Union[Mapping[str, Any], MapConfig]]) -> Optional[MapConfig]:
    """Parse the optional ``map`` section."""
    if entry is None:
        return None
    if isinstance(entry, MapConfig):
        return entry
    if not isinstance(entry, Mapping):
        raise MetricsConfigError(f"map must be a mapping, got {type(entry).__name__}.")
    return _build_dataclass(MapConfig, entry, "map")


# --------------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityFilterConfig:
    """The whole-scene view (no key prefix)."""

    name: str = ""

    requires_map = False


@dataclass(frozen=True)
class CorridorFilterConfig:
    """A straight forward strip ``width_m`` wide centered on the ego x axis."""

    name: str = "corridor"
    width_m: float = 3.0

    requires_map = False

    def __post_init__(self) -> None:
        if self.width_m <= 0.0:
            raise ValueError("width_m must be > 0.")


@dataclass(frozen=True)
class RegionFilterConfig:
    """Elements inside a union of lanelet2 regions (footprint overlap for boxes)."""

    name: str
    regions: Tuple[str, ...]
    margin_m: float = 0.0
    expand: bool = False

    requires_map = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", tuple(str(token) for token in self.regions))
        if not self.regions:
            raise ValueError("regions must list at least one lanelet2 token.")
        if self.margin_m < 0.0:
            raise ValueError("margin_m must be >= 0.")


@dataclass(frozen=True)
class CollisionFilterConfig:
    """Elements inside the ego reachable region clipped to the drivable lanelets."""

    name: str = "collision"
    horizon_s: float = 4.0
    dt_s: float = 0.1
    max_lateral_accel_mps2: float = 3.0
    min_radius_m: float = 3.0
    arc_samples: int = 21

    requires_map = True


FilterConfigType = Union[IdentityFilterConfig, CorridorFilterConfig, RegionFilterConfig, CollisionFilterConfig]

FILTER_TYPES: Dict[str, Type[Any]] = {
    "identity": IdentityFilterConfig,
    "corridor": CorridorFilterConfig,
    "region": RegionFilterConfig,
    "collision": CollisionFilterConfig,
}


def parse_filters(
    entries: Optional[Sequence[Union[Mapping[str, Any], FilterConfigType]]],
) -> Tuple[FilterConfigType, ...]:
    """Parse ``[{name, type, ...}, ...]`` into validated filter configs (identity is implicit)."""
    if not entries:
        return ()
    configs: List[FilterConfigType] = []
    names: List[str] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, tuple(FILTER_TYPES.values())):
            config = entry
        else:
            if not isinstance(entry, Mapping):
                raise MetricsConfigError(f"filters[{index}] must be a mapping, got {type(entry).__name__}.")
            token = entry.get("type")
            if token not in FILTER_TYPES:
                raise MetricsConfigError(
                    f"filters[{index}]: unknown filter type {token!r}; valid: {sorted(FILTER_TYPES)}."
                )
            config = _build_dataclass(FILTER_TYPES[token], entry, f"filters[{index}] ({token})")
        if isinstance(config, IdentityFilterConfig):
            continue
        if not config.name:
            raise MetricsConfigError(f"filters[{index}]: a non-identity filter needs a non-empty name.")
        if config.name in names:
            raise MetricsConfigError(f"filters[{index}]: duplicate filter name {config.name!r}.")
        names.append(config.name)
        configs.append(config)
    return tuple(configs)


def needs_map(filter_configs: Sequence[FilterConfigType]) -> bool:
    """Whether any configured filter depends on a lanelet map."""
    return any(getattr(config, "requires_map", False) for config in filter_configs)


def build_filter(config: FilterConfigType, map_provider: Any = None) -> MetricFilter:
    """Instantiate the filter for one config; map-dependent filters need a ``map_provider``."""
    if isinstance(config, IdentityFilterConfig):
        return IDENTITY
    if isinstance(config, CorridorFilterConfig):
        return CorridorFilter(width_m=config.width_m, name=config.name)
    if map_provider is None:
        raise MetricsConfigError(f"filter {config.name!r} needs a lanelet map; configure the `map` section.")
    if isinstance(config, RegionFilterConfig):
        from perception_eval.evaluation.metrics.filters_map import RegionFilter

        return RegionFilter(
            region=config.regions,
            map_provider=map_provider,
            margin=config.margin_m,
            expand=config.expand,
            name=config.name,
        )
    if isinstance(config, CollisionFilterConfig):
        from perception_eval.evaluation.metrics.filters_map import CollisionFilter
        from perception_eval.evaluation.metrics.geometry.reachability import ReachabilityParams

        return CollisionFilter(
            map_provider=map_provider,
            params=ReachabilityParams(
                horizon_s=config.horizon_s,
                dt_s=config.dt_s,
                max_lateral_accel_mps2=config.max_lateral_accel_mps2,
                min_radius_m=config.min_radius_m,
                arc_samples=config.arc_samples,
            ),
            name=config.name,
        )
    raise MetricsConfigError(f"unsupported filter config {type(config).__name__}.")


def build_filters(
    filter_configs: Sequence[FilterConfigType],
    map_provider: Any = None,
    map_config: Optional[MapConfig] = None,
) -> List[MetricFilter]:
    """Identity plus one filter per config, validated for unique names."""
    filters: List[MetricFilter] = [IDENTITY]
    for config in filter_configs:
        if isinstance(config, CollisionFilterConfig) and map_config is not None:
            from perception_eval.evaluation.metrics.filters_map import CollisionFilter
            from perception_eval.evaluation.metrics.geometry.reachability import ReachabilityParams

            if map_provider is None:
                raise MetricsConfigError(f"filter {config.name!r} needs a lanelet map; configure the `map` section.")
            filters.append(
                CollisionFilter(
                    map_provider=map_provider,
                    region=map_config.region,
                    params=ReachabilityParams(
                        horizon_s=config.horizon_s,
                        dt_s=config.dt_s,
                        max_lateral_accel_mps2=config.max_lateral_accel_mps2,
                        min_radius_m=config.min_radius_m,
                        arc_samples=config.arc_samples,
                    ),
                    max_speed_mps=map_config.max_speed_mps,
                    ego_body_m=map_config.ego_body_m,
                    name=config.name,
                )
            )
            continue
        filters.append(build_filter(config, map_provider))
    return validate_filters(filters)


def filter_config_serialization(config: FilterConfigType) -> Dict[str, Any]:
    """Dict form of one filter config including its ``type`` token."""
    token = next(token for token, cls in FILTER_TYPES.items() if isinstance(config, cls))
    data: Dict[str, Any] = {"type": token}
    for f in fields(config):
        value = getattr(config, f.name)
        data[f.name] = list(value) if isinstance(value, tuple) else value
    return data
