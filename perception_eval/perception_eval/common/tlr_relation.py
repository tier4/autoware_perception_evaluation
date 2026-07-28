# Copyright 2026 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resolve a 2D traffic-light `object_ann.instance_token` to a Regulatory Element (RE) ID.

Two dataset conventions exist:

    legacy (no `traffic_light_instance_map.json`):
        object_ann.instance_token -> instance.instance_name (`...:<RE ID>`)

    current (`annotation/traffic_light_instance_map.json` present):
        object_ann.instance_token -> traffic_light_instance_map.json.instance_token
            -> traffic_light_linestring_id -> Lanelet2 map Regulatory Element ID

`DynamicObject2D.uuid` is always the resolved RE ID, so downstream signal-aggregation
and `driving_log_replayer_v2` code does not need to know which convention a given
dataset uses.

`build_traffic_light_id_resolver` decides which resolver to construct, once per
dataset load (see `perception_eval.common.dataset._load_dataset`), and the resolver is
then reused for every frame/annotation in that dataset.

Map parsing (Lanelet2 -> LineString-to-RegulatoryElement reverse index) is delegated to
`t4_devkit.lanelet`, which is the single shared implementation of that logic (see
`t4_devkit.lanelet.build_linestring_to_regulatory_element_index`); this module does not
re-implement a Lanelet2 parser. `t4-devkit` is only required when a dataset actually
carries `traffic_light_instance_map.json` and no `linestring_to_re_id` mapping is injected explicitly
- datasets that only use the legacy convention never need it installed.
"""
from __future__ import annotations

import json
import logging
import os.path as osp
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from typing import runtime_checkable

__all__ = [
    "TRAFFIC_LIGHT_RELATION_FILENAME",
    "LegacyInstanceNameResolver",
    "TrafficLightIdResolver",
    "TrafficLightLineStringResolver",
    "TrafficLightRelationError",
    "build_linestring_to_regulatory_element_index",
    "build_traffic_light_id_resolver",
]

TRAFFIC_LIGHT_RELATION_FILENAME = "traffic_light_instance_map.json"


class TrafficLightRelationError(Exception):
    """Raised when TLR relation/map data required to resolve an RE ID is missing or
    inconsistent.

    Kept distinct from `perception_eval.common.dataset.DatasetLoadingError` so callers
    can tell a broken TLR relation apart from other dataset load failures.
    """


@runtime_checkable
class TrafficLightIdResolver(Protocol):
    """Resolves a 2D traffic-light `object_ann.instance_token` to a Regulatory Element ID.

    Implementations are constructed once per dataset load by
    `build_traffic_light_id_resolver` and reused across every frame/annotation of that
    dataset; `resolve_re_id` itself must not re-scan any table.
    """

    def resolve_re_id(self, instance_token: str) -> str:
        """Return the Regulatory Element ID related to `instance_token`.

        Raises:
            TrafficLightRelationError: If `instance_token` cannot be resolved.
        """
        ...


class LegacyInstanceNameResolver:
    """Resolves the RE ID from `Instance.instance_name` (`"...:<RE ID>"`).

    This is the pre-`traffic_light_instance_map.json` convention: the RE ID is the text after the
    last `:` in `instance_name`.
    """

    def __init__(self, instance_records: List[Dict[str, Any]]) -> None:
        """
        Args:
            instance_records (List[Dict[str, Any]]): Raw `instance.json` records
                (e.g. `nusc.instance`).
        """
        self._instance_by_token: Dict[str, Dict[str, Any]] = {record["token"]: record for record in instance_records}

    def resolve_re_id(self, instance_token: str) -> str:
        record = self._instance_by_token.get(instance_token)
        if record is None:
            raise TrafficLightRelationError(f"instance_token '{instance_token}' is not found in instance.json.")

        instance_name: str = record.get("instance_name", "")
        if ":" not in instance_name:
            raise TrafficLightRelationError(
                "Legacy TLR instance_name must contain ':' followed by the Regulatory "
                f"Element ID, but got instance_name={instance_name!r} for "
                f"instance_token={instance_token!r}."
            )

        re_id = instance_name.rsplit(":", 1)[-1]
        if re_id == "":
            raise TrafficLightRelationError(
                "Legacy TLR instance_name has an empty Regulatory Element ID: "
                f"instance_name={instance_name!r} (instance_token={instance_token!r})."
            )
        return re_id


class TrafficLightLineStringResolver:
    """Resolves the RE ID via `traffic_light_instance_map.json` + a Lanelet2 map reverse index.

        instance_token -> traffic_light_linestring_id -> Regulatory Element ID

    All indices are built once at construction time and never rescanned by
    `resolve_re_id`.
    """

    def __init__(
        self,
        traffic_light_records: List[Dict[str, Any]],
        linestring_to_re_id: Dict[str, str],
        *,
        legacy_fallback: Optional[LegacyInstanceNameResolver] = None,
    ) -> None:
        """
        Args:
            traffic_light_records (List[Dict[str, Any]]): Raw `traffic_light_instance_map.json` records:
                `{"token", "instance_token", "traffic_light_linestring_id"}`.
            linestring_to_re_id (Dict[str, str]): Reverse index from Lanelet2 LineString
                ID to Regulatory Element ID (see `build_linestring_to_regulatory_element_index`).
            legacy_fallback (Optional[LegacyInstanceNameResolver]): If given, an
                `instance_token` with no `traffic_light_instance_map.json` relation falls back to
                legacy `instance_name` resolution instead of raising. Disabled by
                default (`allow_legacy_tlr_fallback=False`); every per-annotation
                fallback is logged and counted in `fallback_count`.

        Raises:
            TrafficLightRelationError: If one `instance_token` maps to more than one
                distinct `traffic_light_linestring_id`, or if a `traffic_light_linestring_id`
                does not resolve to any Regulatory Element in `linestring_to_re_id`.
        """
        self._linestring_to_re_id = linestring_to_re_id
        self._legacy_fallback = legacy_fallback
        self._fallback_count = 0

        instance_to_linestring: Dict[str, str] = {}
        for record in traffic_light_records:
            instance_token = record["instance_token"]
            linestring_id = str(record["traffic_light_linestring_id"])
            existing = instance_to_linestring.get(instance_token)
            if existing is not None:
                if existing != linestring_id:
                    raise TrafficLightRelationError(
                        f"instance_token '{instance_token}' maps to multiple "
                        f"traffic_light_linestring_id values: {existing!r} and "
                        f"{linestring_id!r}."
                    )
                continue
            instance_to_linestring[instance_token] = linestring_id

        for instance_token, linestring_id in instance_to_linestring.items():
            if linestring_id not in linestring_to_re_id:
                raise TrafficLightRelationError(
                    f"traffic_light_linestring_id '{linestring_id}' (instance_token="
                    f"{instance_token!r}) does not resolve to a Regulatory Element in "
                    "the map: either the LineString is missing from the map, or it is "
                    "not referred to by any traffic-light Regulatory Element."
                )

        self._instance_to_linestring = instance_to_linestring

    @property
    def fallback_count(self) -> int:
        """Number of `resolve_re_id` calls that fell back to legacy `instance_name`."""
        return self._fallback_count

    def resolve_re_id(self, instance_token: str) -> str:
        linestring_id = self._instance_to_linestring.get(instance_token)
        if linestring_id is None:
            if self._legacy_fallback is not None:
                self._fallback_count += 1
                logging.warning(
                    "TLR relation missing for instance_token=%s; falling back to legacy "
                    "instance_name (allow_legacy_tlr_fallback=True). fallback_count=%d",
                    instance_token,
                    self._fallback_count,
                )
                return self._legacy_fallback.resolve_re_id(instance_token)
            raise TrafficLightRelationError(
                f"No traffic_light_instance_map.json relation for instance_token={instance_token!r}. "
                "This dataset uses the relation+map format (traffic_light_instance_map.json is "
                "present), so per-annotation fallback to legacy instance_name is "
                "disabled by default. Pass allow_legacy_tlr_fallback=True to permit it "
                "during migration."
            )
        return self._linestring_to_re_id[linestring_id]


def build_linestring_to_regulatory_element_index(map_path: str) -> Dict[str, str]:
    """Build the Lanelet2 `LineString ID -> Regulatory Element ID` reverse index.

    Delegates the actual Lanelet2 parsing to `t4_devkit.lanelet`, so this package does
    not maintain a second Lanelet2 map parser. `t4-devkit` is only imported here, i.e.
    only when the new relation format is actually in use.

    Args:
        map_path (str): Path to the Lanelet2 OSM map file.

    Returns:
        Mapping from LineString ID (way ID, as string) to Regulatory Element ID.

    Raises:
        TrafficLightRelationError: If `t4-devkit` is not installed, or if a LineString
            is referred to by more than one traffic-light Regulatory Element.
    """
    try:
        from t4_devkit.lanelet import AmbiguousRegulatoryElementError
        from t4_devkit.lanelet import build_linestring_to_regulatory_element_index as _build_index
        from t4_devkit.lanelet import LaneletParser
    except ImportError as e:
        raise TrafficLightRelationError(
            "t4-devkit is required to resolve the Lanelet2 map for TLR relation "
            "resolution (`pip install t4-devkit`), or pass a prebuilt "
            "`linestring_to_re_id` mapping to `build_traffic_light_id_resolver` "
            "explicitly."
        ) from e

    parser = LaneletParser(map_path)
    try:
        return _build_index(parser)
    except AmbiguousRegulatoryElementError as e:
        raise TrafficLightRelationError(str(e)) from e


def _load_traffic_light_relation_records(dataset_path: str) -> Optional[List[Dict[str, Any]]]:
    """Return raw `traffic_light_instance_map.json` records, or None if the file is absent.

    `nusc`/`nuim` (nuscenes-devkit) do not know about this table, so it is read
    directly from `<dataset_path>/annotation/traffic_light_instance_map.json`.
    """
    filepath = osp.join(dataset_path, "annotation", TRAFFIC_LIGHT_RELATION_FILENAME)
    if not osp.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)


def build_traffic_light_id_resolver(
    nusc,
    dataset_path: str,
    *,
    allow_legacy_tlr_fallback: bool = False,
    linestring_to_re_id: Optional[Dict[str, str]] = None,
) -> TrafficLightIdResolver:
    """Build the TLR ID resolver for one dataset load.

    - If `annotation/traffic_light_instance_map.json` exists, this is a B' dataset: use
      `TrafficLightLineStringResolver` (relation + Lanelet2 map).
    - Otherwise, this is a legacy dataset: use `LegacyInstanceNameResolver`
      (`instance.instance_name`).

    Call this once per dataset (see `perception_eval.common.dataset._load_dataset`) and
    reuse the returned resolver for every frame/annotation; do not call this per
    annotation.

    Args:
        nusc (NuScenes): NuScenes instance already loaded for this dataset
            (`nusc.instance` is used by both resolvers).
        dataset_path (str): Root path of the T4 dataset.
        allow_legacy_tlr_fallback (bool): Migration-only escape hatch. When the
            relation+map format is used but a specific `instance_token` has no
            `traffic_light_instance_map.json` relation, fall back to legacy `instance_name`
            resolution for that annotation instead of raising. Defaults to False;
            every fallback is logged.
        linestring_to_re_id (Optional[Dict[str, str]]): Prebuilt LineString-to-RE
            reverse index. Supply this to avoid the `t4-devkit` dependency (e.g. in
            tests, or if the caller already parses the map itself). If omitted, it is
            built from the dataset's `map.json` + Lanelet2 map via `t4-devkit`.

    Returns:
        A `TrafficLightIdResolver`, ready to call `resolve_re_id` per annotation.

    Raises:
        TrafficLightRelationError: See `TrafficLightLineStringResolver` and
            `build_linestring_to_regulatory_element_index`.
    """
    traffic_light_records = _load_traffic_light_relation_records(dataset_path)

    if traffic_light_records is None:
        return LegacyInstanceNameResolver(nusc.instance)

    if linestring_to_re_id is None:
        map_records = getattr(nusc, "map", [])
        if not map_records:
            raise TrafficLightRelationError(
                "traffic_light_instance_map.json is present but no 'map' record exists in this "
                "dataset; cannot resolve Regulatory Element IDs without a Lanelet2 map."
            )
        map_path = osp.join(dataset_path, map_records[0]["filename"])
        linestring_to_re_id = build_linestring_to_regulatory_element_index(map_path)

    legacy_fallback = LegacyInstanceNameResolver(nusc.instance) if allow_legacy_tlr_fallback else None

    return TrafficLightLineStringResolver(
        traffic_light_records,
        linestring_to_re_id,
        legacy_fallback=legacy_fallback,
    )
