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
"""Resolve a 2D traffic-light `object_ann.instance_token` to the Regulatory Element
(RE) ID(s) it corresponds to.

`DynamicObject2D.uuid` for a traffic-light annotation is a Regulatory Element ID, and
matching between estimated and ground truth objects for TLR classification is done by
`uuid` equality (see `evaluation.result.object_result_matching`), not by geometry. The
live Autoware node reports one state per Regulatory Element, and a single physical
traffic light can legitimately be attached to more than one Regulatory Element (e.g.
one lamp shared by two lanelets, each with its own stop line) -- confirmed against a
real production map, where this is common rather than an edge case. So resolving an
`instance_token` can yield more than one RE ID, and ground truth must be duplicated
across all of them for `uuid`-based matching to line up with what the live system
reports per RE.

Two dataset conventions are supported:

    legacy (no `annotation/traffic_light.json`):
        object_ann.instance_token -> instance.instance_name (`"...:<RE ID>"`)

    relation table (`annotation/traffic_light.json` present):
        object_ann.instance_token -> traffic_light.json.instance_token
            -> traffic_light_linestring_id -> Lanelet2 map Regulatory Element ID(s)

`build_default_traffic_light_id_resolver` picks between them by file presence and is
used automatically by `perception_eval.common.dataset._load_dataset` -- callers
(including `driving_log_replayer_v2`) do not need to pass anything to get correct
resolution for either format. Lanelet2 map parsing is delegated to `t4_devkit.lanelet`
(imported only when `traffic_light.json` is actually present), so this package does not
maintain a second Lanelet2 parser.

A caller with a different relation source can still override resolution entirely via
`perception_eval.common.dataset.load_all_datasets(..., traffic_light_id_resolver_factory=...)`.
"""
from __future__ import annotations

import json
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
    "TrafficLightRelationError",
    "TrafficLightTableResolver",
    "build_default_traffic_light_id_resolver",
]

TRAFFIC_LIGHT_RELATION_FILENAME = "traffic_light.json"

# T4 dataset convention when `map.filename` is not populated (seen in real datasets).
_DEFAULT_MAP_FILENAME = "map/lanelet2_map.osm"


class TrafficLightRelationError(Exception):
    """Raised when a traffic-light `object_ann.instance_token` cannot be resolved to at
    least one RE ID."""


@runtime_checkable
class TrafficLightIdResolver(Protocol):
    """Resolves a 2D traffic-light `object_ann.instance_token` to the Regulatory
    Element ID(s) it corresponds to.

    Implementations are constructed once per dataset load (see
    `perception_eval.common.dataset._load_dataset`) and reused across every
    frame/annotation of that dataset; `resolve_re_ids` itself must not re-scan any
    table.
    """

    def resolve_re_ids(self, instance_token: str) -> List[str]:
        """Return the Regulatory Element ID(s) related to `instance_token`.

        Almost always a single-element list; more than one only when the physical
        traffic light is attached to more than one Regulatory Element (see module
        docstring). Never empty.

        Raises:
            TrafficLightRelationError: If `instance_token` cannot be resolved.
        """
        ...


class LegacyInstanceNameResolver:
    """Resolves the RE ID from `Instance.instance_name` (`"...:<RE ID>"`).

    The RE ID is the text after the last `:` in `instance_name`. This convention only
    ever encodes a single RE ID per instance.
    """

    def __init__(self, instance_records: List[Dict[str, Any]]) -> None:
        """
        Args:
            instance_records (List[Dict[str, Any]]): Raw `instance.json` records
                (e.g. `nusc.instance`).
        """
        self._instance_by_token: Dict[str, Dict[str, Any]] = {record["token"]: record for record in instance_records}

    def resolve_re_ids(self, instance_token: str) -> List[str]:
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
        return [re_id]


class TrafficLightTableResolver:
    """Resolves RE ID(s) via `traffic_light.json` + a Lanelet2 LineString-to-RE index.

        instance_token -> traffic_light_linestring_id -> Regulatory Element ID(s)

    All indices are built once at construction time and never rescanned by
    `resolve_re_ids`. Construction fails loud (see `TrafficLightRelationError`) on a
    dangling `instance_token`/`traffic_light_linestring_id`, on one `instance_token`
    mapping to more than one distinct `traffic_light_linestring_id`, or on a
    `traffic_light_linestring_id` that does not resolve to any Regulatory Element --
    all of these indicate a data problem that should stop dataset load, not fail
    silently on whichever annotation happens to hit it first.
    """

    def __init__(
        self,
        traffic_light_records: List[Dict[str, Any]],
        linestring_to_re_ids: Dict[str, List[str]],
    ) -> None:
        """
        Args:
            traffic_light_records (List[Dict[str, Any]]): Raw `traffic_light.json`
                records: `{"token", "instance_token", "traffic_light_linestring_id"}`.
            linestring_to_re_ids (Dict[str, List[str]]): Reverse index from Lanelet2
                LineString ID to the Regulatory Element ID(s) that refer to it (see
                `t4_devkit.lanelet.group_traffic_light_linestrings`). A LineString may
                legitimately map to more than one RE.

        Raises:
            TrafficLightRelationError: See class docstring.
        """
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
            if not linestring_to_re_ids.get(linestring_id):
                raise TrafficLightRelationError(
                    f"traffic_light_linestring_id '{linestring_id}' (instance_token="
                    f"{instance_token!r}) does not resolve to any Regulatory Element: "
                    "either the LineString is missing from the map, or it is not "
                    "referred to by any traffic-light Regulatory Element."
                )

        self._instance_to_linestring = instance_to_linestring
        self._linestring_to_re_ids = linestring_to_re_ids

    def resolve_re_ids(self, instance_token: str) -> List[str]:
        linestring_id = self._instance_to_linestring.get(instance_token)
        if linestring_id is None:
            raise TrafficLightRelationError(
                f"No {TRAFFIC_LIGHT_RELATION_FILENAME} relation for " f"instance_token={instance_token!r}."
            )
        return sorted(set(self._linestring_to_re_ids[linestring_id]))


def _build_linestring_to_re_ids(map_path: str) -> Dict[str, List[str]]:
    """Build the Lanelet2 `LineString ID -> Regulatory Element ID(s)` reverse index.

    Delegates the actual Lanelet2 parsing to `t4_devkit.lanelet`, so this package does
    not maintain a second Lanelet2 map parser. `t4-devkit` is only imported here, i.e.
    only when `traffic_light.json` is actually present in the dataset.
    """
    try:
        from t4_devkit.lanelet import group_traffic_light_linestrings
        from t4_devkit.lanelet import LaneletParser
    except ImportError as e:
        raise TrafficLightRelationError(
            "t4-devkit is required to resolve the Lanelet2 map for TLR relation "
            "resolution (`pip install t4-devkit`), or pass a prebuilt "
            "`linestring_to_re_ids` mapping via a custom "
            "`traffic_light_id_resolver_factory` instead."
        ) from e

    parser = LaneletParser(map_path)
    return group_traffic_light_linestrings(parser)


def build_default_traffic_light_id_resolver(
    nusc,
    dataset_path: str,
    *,
    linestring_to_re_ids: Optional[Dict[str, List[str]]] = None,
) -> TrafficLightIdResolver:
    """Build the default TLR ID resolver for one dataset load.

    Used automatically by `perception_eval.common.dataset._load_dataset` when no
    `traffic_light_id_resolver_factory` is supplied, so this is what
    `driving_log_replayer_v2` and any other caller gets without wiring anything
    through: `traffic_light.json` + the Lanelet2 map when present, otherwise
    `LegacyInstanceNameResolver`.

    Args:
        nusc (NuScenes): NuScenes instance already loaded for this dataset
            (`nusc.instance`/`nusc.map` are used).
        dataset_path (str): Root path of the T4 dataset.
        linestring_to_re_ids (Optional[Dict[str, List[str]]]): Prebuilt LineString-to-RE
            reverse index, to avoid the `t4-devkit` dependency (e.g. in tests). If
            omitted, it is built from the dataset's `map.json` + Lanelet2 map via
            `t4-devkit`.

    Returns:
        A `TrafficLightIdResolver`, ready to call `resolve_re_ids` per annotation.

    Raises:
        TrafficLightRelationError: See `TrafficLightTableResolver` and
            `_build_linestring_to_re_ids`.
    """
    traffic_light_path = osp.join(dataset_path, "annotation", TRAFFIC_LIGHT_RELATION_FILENAME)
    if not osp.exists(traffic_light_path):
        return LegacyInstanceNameResolver(nusc.instance)

    with open(traffic_light_path) as f:
        traffic_light_records = json.load(f)

    if linestring_to_re_ids is None:
        map_records = getattr(nusc, "map", [])
        if not map_records:
            raise TrafficLightRelationError(
                f"{TRAFFIC_LIGHT_RELATION_FILENAME} is present but no 'map' record "
                "exists in this dataset; cannot resolve Regulatory Element IDs "
                "without a Lanelet2 map."
            )
        # `map.filename` is blank in some real datasets; fall back to the standard
        # T4 dataset convention instead of failing.
        map_filename = map_records[0].get("filename") or _DEFAULT_MAP_FILENAME
        map_path = osp.join(dataset_path, map_filename)
        linestring_to_re_ids = _build_linestring_to_re_ids(map_path)

    return TrafficLightTableResolver(traffic_light_records, linestring_to_re_ids)
