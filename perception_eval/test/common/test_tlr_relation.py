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
"""Unit tests for `perception_eval.common.tlr_relation`.

Covers the cases listed in the TLR relation design doc (section 4.1):
    1. Legacy `instance_name` resolves to an RE ID.
    2. Relation + map resolves to an RE ID.
    3. Missing relation is detected.
    4. LineString absent from the map is detected.
    5. LineString not referenced by any Regulatory Element is detected.
    6. LineString referenced by multiple Regulatory Elements (ambiguous) is detected.
    7. The index is built exactly once per dataset load.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from perception_eval.common.tlr_relation import (
    LegacyInstanceNameResolver,
    TrafficLightLineStringResolver,
    TrafficLightRelationError,
    build_linestring_to_regulatory_element_index,
    build_traffic_light_id_resolver,
)

_HAS_T4_DEVKIT = importlib.util.find_spec("t4_devkit") is not None
requires_t4_devkit = pytest.mark.skipif(not _HAS_T4_DEVKIT, reason="t4-devkit is not installed")

# ---------------------------------------------------------------------------
# 1. Legacy `instance.instance_name` resolver
# ---------------------------------------------------------------------------


def test_legacy_resolver_resolves_re_id() -> None:
    instances = [
        {"token": "inst_0", "instance_name": "camera0:tl0:1234"},
        {"token": "inst_1", "instance_name": "camera1:5678"},
    ]
    resolver = LegacyInstanceNameResolver(instances)

    assert resolver.resolve_re_id("inst_0") == "1234"
    assert resolver.resolve_re_id("inst_1") == "5678"


def test_legacy_resolver_missing_instance_token() -> None:
    resolver = LegacyInstanceNameResolver([{"token": "inst_0", "instance_name": "a:1"}])

    with pytest.raises(TrafficLightRelationError, match="not found in instance.json"):
        resolver.resolve_re_id("does_not_exist")


def test_legacy_resolver_malformed_instance_name() -> None:
    resolver = LegacyInstanceNameResolver([{"token": "inst_0", "instance_name": "no_colon_here"}])

    with pytest.raises(TrafficLightRelationError, match="must contain ':'"):
        resolver.resolve_re_id("inst_0")


def test_legacy_resolver_empty_re_id() -> None:
    resolver = LegacyInstanceNameResolver([{"token": "inst_0", "instance_name": "camera0:"}])

    with pytest.raises(TrafficLightRelationError, match="empty Regulatory Element ID"):
        resolver.resolve_re_id("inst_0")


# ---------------------------------------------------------------------------
# 2. Relation + map resolver
# ---------------------------------------------------------------------------


def test_linestring_resolver_resolves_re_id() -> None:
    traffic_light_records = [
        {"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"},
        {"token": "rel_1", "instance_token": "inst_1", "traffic_light_linestring_id": "401"},
    ]
    linestring_to_re_id = {"400": "2000", "401": "2000"}  # two lamps, same RE
    resolver = TrafficLightLineStringResolver(traffic_light_records, linestring_to_re_id)

    assert resolver.resolve_re_id("inst_0") == "2000"
    assert resolver.resolve_re_id("inst_1") == "2000"


def test_linestring_resolver_rejects_instance_with_conflicting_linestrings() -> None:
    traffic_light_records = [
        {"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"},
        {"token": "rel_1", "instance_token": "inst_0", "traffic_light_linestring_id": "401"},
    ]
    with pytest.raises(TrafficLightRelationError, match="maps to multiple"):
        TrafficLightLineStringResolver(traffic_light_records, {"400": "2000", "401": "2001"})


# ---------------------------------------------------------------------------
# 3. Missing relation is detected (strict by default, fallback opt-in)
# ---------------------------------------------------------------------------


def test_linestring_resolver_missing_relation_raises_by_default() -> None:
    traffic_light_records = [
        {"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"},
    ]
    resolver = TrafficLightLineStringResolver(traffic_light_records, {"400": "2000"})

    with pytest.raises(TrafficLightRelationError, match="No traffic_light_instance_map.json relation"):
        resolver.resolve_re_id("inst_missing")


def test_linestring_resolver_missing_relation_falls_back_when_allowed() -> None:
    traffic_light_records = [
        {"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"},
    ]
    legacy_fallback = LegacyInstanceNameResolver(
        [{"token": "inst_missing", "instance_name": "camera0:9999"}]
    )
    resolver = TrafficLightLineStringResolver(
        traffic_light_records, {"400": "2000"}, legacy_fallback=legacy_fallback
    )

    assert resolver.resolve_re_id("inst_missing") == "9999"
    assert resolver.fallback_count == 1

    # A second fallback keeps incrementing the counter (each fallback is logged).
    resolver.resolve_re_id("inst_missing")
    assert resolver.fallback_count == 2


# ---------------------------------------------------------------------------
# 4. LineString absent from the map is detected
# ---------------------------------------------------------------------------


def test_linestring_resolver_rejects_linestring_missing_from_map_index() -> None:
    traffic_light_records = [
        {"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "999999"},
    ]
    with pytest.raises(TrafficLightRelationError, match="does not resolve to a Regulatory"):
        TrafficLightLineStringResolver(traffic_light_records, linestring_to_re_id={})


# ---------------------------------------------------------------------------
# 5 & 6: map-derived index edge cases, via the t4-devkit-backed map index builder.
# Skipped (per-test, not module-wide) if t4-devkit is not installed (it is an
# optional dependency of this resolver; see `build_linestring_to_regulatory_element_index`).
# ---------------------------------------------------------------------------

_OSM_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" lat="0.0" lon="0.0"><tag k="local_x" v="0.0"/><tag k="local_y" v="0.0"/></node>
  <node id="2" lat="0.0" lon="0.0"><tag k="local_x" v="1.0"/><tag k="local_y" v="0.0"/></node>
"""
_OSM_FOOTER = "</osm>\n"


def _way(way_id: str) -> str:
    return f'  <way id="{way_id}"><nd ref="1"/><nd ref="2"/></way>\n'


def _regulatory_element(re_id: str, way_ids: list) -> str:
    members = "\n".join(f'    <member type="way" ref="{w}" role="ref_line"/>' for w in way_ids)
    return (
        f'  <relation id="{re_id}">\n{members}\n'
        '    <tag k="type" v="regulatory_element"/>\n'
        '    <tag k="subtype" v="traffic_light"/>\n'
        "  </relation>\n"
    )


def _write_osm(tmp_path: Path, body: str) -> str:
    path = tmp_path / "lanelet2_map.osm"
    path.write_text(_OSM_HEADER + body + _OSM_FOOTER, encoding="utf-8")
    return path.as_posix()


@requires_t4_devkit
def test_build_linestring_to_regulatory_element_index_resolves(tmp_path: Path) -> None:
    """(2, end-to-end) A well-formed map resolves the LineString to its RE."""
    map_path = _write_osm(tmp_path, _way("400") + _regulatory_element("2000", ["400"]))

    index = build_linestring_to_regulatory_element_index(map_path)

    assert index == {"400": "2000"}


@requires_t4_devkit
def test_build_linestring_to_regulatory_element_index_not_referenced(tmp_path: Path) -> None:
    """(5) A LineString with no Regulatory Element referring to it is simply absent."""
    map_path = _write_osm(tmp_path, _way("400"))  # no regulatory_element relation at all

    index = build_linestring_to_regulatory_element_index(map_path)

    assert "400" not in index


@requires_t4_devkit
def test_build_linestring_to_regulatory_element_index_ambiguous(tmp_path: Path) -> None:
    """(6) A LineString referred to by two Regulatory Elements is ambiguous."""
    map_path = _write_osm(
        tmp_path,
        _way("400") + _regulatory_element("2000", ["400"]) + _regulatory_element("2001", ["400"]),
    )

    with pytest.raises(TrafficLightRelationError, match="multiple traffic-light"):
        build_linestring_to_regulatory_element_index(map_path)


def test_build_linestring_to_regulatory_element_index_requires_t4_devkit(monkeypatch) -> None:
    """Missing `t4-devkit` raises a clear, actionable error instead of an ImportError."""
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "t4_devkit.lanelet":
            raise ImportError("simulated missing t4-devkit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    with pytest.raises(TrafficLightRelationError, match="t4-devkit is required"):
        build_linestring_to_regulatory_element_index("unused.osm")


# ---------------------------------------------------------------------------
# 7. Index is built once per dataset load, not per annotation.
# ---------------------------------------------------------------------------


def test_build_traffic_light_id_resolver_builds_map_index_once(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset"
    (dataset_path / "annotation").mkdir(parents=True)
    (dataset_path / "annotation" / "traffic_light_instance_map.json").write_text(
        json.dumps(
            [{"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"}]
        ),
        encoding="utf-8",
    )

    nusc = MagicMock()
    nusc.instance = []
    nusc.map = [{"filename": "map/lanelet2_map.osm"}]

    build_index_mock = MagicMock(return_value={"400": "2000"})
    monkeypatch.setattr(
        "perception_eval.common.tlr_relation.build_linestring_to_regulatory_element_index",
        build_index_mock,
    )

    resolver = build_traffic_light_id_resolver(nusc, dataset_path.as_posix())

    # Resolve several annotations against the same, already-built resolver.
    for _ in range(5):
        assert resolver.resolve_re_id("inst_0") == "2000"

    build_index_mock.assert_called_once()


def test_build_traffic_light_id_resolver_uses_legacy_when_no_relation_file(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    (dataset_path / "annotation").mkdir(parents=True)  # no traffic_light_instance_map.json

    nusc = MagicMock()
    nusc.instance = [{"token": "inst_0", "instance_name": "camera0:1234"}]

    resolver = build_traffic_light_id_resolver(nusc, dataset_path.as_posix())

    assert isinstance(resolver, LegacyInstanceNameResolver)
    assert resolver.resolve_re_id("inst_0") == "1234"


def test_build_traffic_light_id_resolver_accepts_injected_index(tmp_path) -> None:
    """Adapter injection: callers may skip the t4-devkit map parse entirely."""
    dataset_path = tmp_path / "dataset"
    (dataset_path / "annotation").mkdir(parents=True)
    (dataset_path / "annotation" / "traffic_light_instance_map.json").write_text(
        json.dumps(
            [{"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"}]
        ),
        encoding="utf-8",
    )

    nusc = MagicMock()
    nusc.instance = []

    resolver = build_traffic_light_id_resolver(
        nusc, dataset_path.as_posix(), linestring_to_re_id={"400": "2000"}
    )

    assert resolver.resolve_re_id("inst_0") == "2000"
