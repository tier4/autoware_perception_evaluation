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
"""Unit tests for `perception_eval.common.tlr_relation` and its injection point in
`perception_eval.common.dataset`.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

from perception_eval.common.tlr_relation import build_default_traffic_light_id_resolver
from perception_eval.common.tlr_relation import LegacyInstanceNameResolver
from perception_eval.common.tlr_relation import TrafficLightRelationError
from perception_eval.common.tlr_relation import TrafficLightTableResolver
import pytest

_HAS_T4_DEVKIT = importlib.util.find_spec("t4_devkit") is not None
requires_t4_devkit = pytest.mark.skipif(not _HAS_T4_DEVKIT, reason="t4-devkit is not installed")


def test_legacy_resolver_resolves_re_id() -> None:
    instances = [
        {"token": "inst_0", "instance_name": "camera0:tl0:1234"},
        {"token": "inst_1", "instance_name": "camera1:5678"},
    ]
    resolver = LegacyInstanceNameResolver(instances)

    assert resolver.resolve_re_ids("inst_0") == ["1234"]
    assert resolver.resolve_re_ids("inst_1") == ["5678"]


def test_legacy_resolver_missing_instance_token() -> None:
    resolver = LegacyInstanceNameResolver([{"token": "inst_0", "instance_name": "a:1"}])

    with pytest.raises(TrafficLightRelationError, match="not found in instance.json"):
        resolver.resolve_re_ids("does_not_exist")


def test_legacy_resolver_malformed_instance_name() -> None:
    resolver = LegacyInstanceNameResolver([{"token": "inst_0", "instance_name": "no_colon_here"}])

    with pytest.raises(TrafficLightRelationError, match="must contain ':'"):
        resolver.resolve_re_ids("inst_0")


def test_legacy_resolver_empty_re_id() -> None:
    resolver = LegacyInstanceNameResolver([{"token": "inst_0", "instance_name": "camera0:"}])

    with pytest.raises(TrafficLightRelationError, match="empty Regulatory Element ID"):
        resolver.resolve_re_ids("inst_0")


# ---------------------------------------------------------------------------
# TrafficLightTableResolver: traffic_light.json (instance_token ->
# traffic_light_linestring_id) + a LineString -> RE ID(s) reverse index.
# ---------------------------------------------------------------------------


def test_table_resolver_resolves_single_re() -> None:
    records = [
        {"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"},
    ]
    resolver = TrafficLightTableResolver(records, {"400": ["2000"]})

    assert resolver.resolve_re_ids("inst_0") == ["2000"]


def test_table_resolver_fans_out_to_multiple_res_sorted_and_deduped() -> None:
    records = [
        {"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"},
    ]
    # Duplicate RE id in the index itself must not produce a duplicate in the output.
    resolver = TrafficLightTableResolver(records, {"400": ["2001", "2000", "2000"]})

    assert resolver.resolve_re_ids("inst_0") == ["2000", "2001"]


def test_table_resolver_same_re_from_multiple_instances() -> None:
    """Two different physical-light instances attached to the same RE resolve
    independently to that RE (this is the normal, non-fan-out case of multiple
    lamps belonging to one group)."""
    records = [
        {"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"},
        {"token": "rel_1", "instance_token": "inst_1", "traffic_light_linestring_id": "401"},
    ]
    resolver = TrafficLightTableResolver(records, {"400": ["2000"], "401": ["2000"]})

    assert resolver.resolve_re_ids("inst_0") == ["2000"]
    assert resolver.resolve_re_ids("inst_1") == ["2000"]


def test_table_resolver_rejects_instance_with_conflicting_linestrings() -> None:
    records = [
        {"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"},
        {"token": "rel_1", "instance_token": "inst_0", "traffic_light_linestring_id": "401"},
    ]
    with pytest.raises(TrafficLightRelationError, match="maps to multiple"):
        TrafficLightTableResolver(records, {"400": ["2000"], "401": ["2001"]})


def test_table_resolver_rejects_linestring_with_no_regulatory_element() -> None:
    records = [
        {"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"},
    ]
    with pytest.raises(TrafficLightRelationError, match="does not resolve to any Regulatory"):
        TrafficLightTableResolver(records, {"400": []})


def test_table_resolver_missing_relation_raises() -> None:
    resolver = TrafficLightTableResolver([], {})

    with pytest.raises(TrafficLightRelationError, match="No traffic_light.json relation"):
        resolver.resolve_re_ids("inst_missing")


# ---------------------------------------------------------------------------
# build_default_traffic_light_id_resolver: what perception_eval.common.dataset
# actually calls by default (no factory needed from the caller, e.g. DLR).
# ---------------------------------------------------------------------------


def test_build_default_resolver_uses_table_when_file_present(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    (dataset_path / "annotation").mkdir(parents=True)
    (dataset_path / "annotation" / "traffic_light.json").write_text(
        json.dumps([{"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"}]),
        encoding="utf-8",
    )

    nusc = MagicMock()
    nusc.instance = []
    nusc.map = [{"filename": "map/lanelet2_map.osm"}]

    resolver = build_default_traffic_light_id_resolver(
        nusc, dataset_path.as_posix(), linestring_to_re_ids={"400": ["2000", "2001"]}
    )

    assert isinstance(resolver, TrafficLightTableResolver)
    assert resolver.resolve_re_ids("inst_0") == ["2000", "2001"]


def test_build_default_resolver_uses_legacy_when_file_absent(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    (dataset_path / "annotation").mkdir(parents=True)  # no traffic_light.json

    nusc = MagicMock()
    nusc.instance = [{"token": "inst_0", "instance_name": "camera0:1234"}]

    resolver = build_default_traffic_light_id_resolver(nusc, dataset_path.as_posix())

    assert isinstance(resolver, LegacyInstanceNameResolver)
    assert resolver.resolve_re_ids("inst_0") == ["1234"]


def test_build_default_resolver_raises_without_map_record(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    (dataset_path / "annotation").mkdir(parents=True)
    (dataset_path / "annotation" / "traffic_light.json").write_text("[]", encoding="utf-8")

    nusc = MagicMock()
    nusc.map = []

    with pytest.raises(TrafficLightRelationError, match="no 'map' record"):
        build_default_traffic_light_id_resolver(nusc, dataset_path.as_posix())


def test_build_default_resolver_falls_back_to_convention_map_path_when_filename_blank(
    tmp_path: Path, monkeypatch
) -> None:
    """Real datasets have been seen with `map.filename == ""`; the standard T4
    convention path must be used instead of failing."""
    dataset_path = tmp_path / "dataset"
    (dataset_path / "annotation").mkdir(parents=True)
    (dataset_path / "annotation" / "traffic_light.json").write_text("[]", encoding="utf-8")

    nusc = MagicMock()
    nusc.map = [{"filename": ""}]

    captured = {}

    def _fake_build_index(map_path):
        captured["map_path"] = map_path
        return {}

    monkeypatch.setattr(
        "perception_eval.common.tlr_relation._build_linestring_to_re_ids",
        _fake_build_index,
    )

    build_default_traffic_light_id_resolver(nusc, dataset_path.as_posix())

    assert captured["map_path"] == dataset_path.joinpath("map/lanelet2_map.osm").as_posix()


@requires_t4_devkit
def test_build_default_resolver_end_to_end_with_real_lanelet2_map(tmp_path: Path) -> None:
    """No injected index: parses a real (synthetic) Lanelet2 map via t4-devkit."""
    dataset_path = tmp_path / "dataset"
    (dataset_path / "annotation").mkdir(parents=True)
    (dataset_path / "map").mkdir(parents=True)
    (dataset_path / "annotation" / "traffic_light.json").write_text(
        json.dumps([{"token": "rel_0", "instance_token": "inst_0", "traffic_light_linestring_id": "400"}]),
        encoding="utf-8",
    )
    (dataset_path / "map" / "lanelet2_map.osm").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<osm version="0.6" generator="test">\n'
        '  <node id="1" lat="0.0" lon="0.0"><tag k="local_x" v="0.0"/><tag k="local_y" v="0.0"/></node>\n'
        '  <node id="2" lat="0.0" lon="0.0"><tag k="local_x" v="1.0"/><tag k="local_y" v="0.0"/></node>\n'
        '  <way id="400"><nd ref="1"/><nd ref="2"/></way>\n'
        '  <relation id="2000">\n'
        '    <member type="way" ref="400" role="ref_line"/>\n'
        '    <tag k="type" v="regulatory_element"/>\n'
        '    <tag k="subtype" v="traffic_light"/>\n'
        "  </relation>\n"
        "</osm>\n",
        encoding="utf-8",
    )

    nusc = MagicMock()
    nusc.map = [{"filename": "map/lanelet2_map.osm"}]

    resolver = build_default_traffic_light_id_resolver(nusc, dataset_path.as_posix())

    assert isinstance(resolver, TrafficLightTableResolver)
    assert resolver.resolve_re_ids("inst_0") == ["2000"]


# ---------------------------------------------------------------------------
# Injection point: perception_eval.common.dataset._load_dataset builds the
# resolver once per dataset load, defaulting to
# build_default_traffic_light_id_resolver, and uses a caller-supplied
# `traffic_light_id_resolver_factory` instead when given.
# ---------------------------------------------------------------------------


def _fake_label_converter(label_type) -> MagicMock:
    converter = MagicMock()
    converter.label_type = label_type
    return converter


def test_load_dataset_defaults_to_legacy_resolver(monkeypatch) -> None:
    from perception_eval.common import dataset as dataset_module
    from perception_eval.common.evaluation_task import EvaluationTask
    from perception_eval.common.label import TrafficLightLabel
    from perception_eval.common.schema import FrameID

    nusc = MagicMock()
    nusc.instance = [{"token": "inst_0", "instance_name": "camera0:1234"}]
    nusc.visibility = ["dummy"]
    nusc.category = []
    nusc.sample = [{"token": "sample_0"}]
    nusc.get = MagicMock(return_value={"data": {}, "timestamp": 0})
    monkeypatch.setattr(dataset_module, "NuScenes", MagicMock(return_value=nusc))
    monkeypatch.setattr(dataset_module, "NuImages", MagicMock())
    dataset_module.NuImages.return_value.get = MagicMock(return_value={"timestamp": 0})
    dataset_module.NuImages.return_value.object_ann = []
    monkeypatch.setattr(dataset_module, "PredictHelper", MagicMock())
    monkeypatch.setattr(dataset_module, "_get_sample_tokens", MagicMock(return_value=["sample_0"]))

    captured = {}
    original_sample_to_frame_2d = dataset_module._sample_to_frame_2d

    def _spy_sample_to_frame_2d(*args, **kwargs):
        captured["resolver"] = kwargs.get("traffic_light_id_resolver")
        return original_sample_to_frame_2d(*args, **kwargs)

    monkeypatch.setattr(dataset_module, "_sample_to_frame_2d", _spy_sample_to_frame_2d)

    dataset_module._load_dataset(
        dataset_path="unused",
        evaluation_task=EvaluationTask.CLASSIFICATION2D,
        label_converter=_fake_label_converter(TrafficLightLabel),
        frame_ids=[FrameID.CAM_FRONT],
        load_raw_data=False,
    )

    assert isinstance(captured["resolver"], LegacyInstanceNameResolver)
    assert captured["resolver"].resolve_re_ids("inst_0") == ["1234"]


def test_load_dataset_uses_injected_resolver_factory(monkeypatch) -> None:
    from perception_eval.common import dataset as dataset_module
    from perception_eval.common.evaluation_task import EvaluationTask
    from perception_eval.common.label import TrafficLightLabel
    from perception_eval.common.schema import FrameID

    nusc = MagicMock()
    nusc.instance = []
    nusc.visibility = ["dummy"]
    nusc.category = []
    nusc.sample = [{"token": "sample_0"}]
    nusc.get = MagicMock(return_value={"data": {}, "timestamp": 0})
    monkeypatch.setattr(dataset_module, "NuScenes", MagicMock(return_value=nusc))
    monkeypatch.setattr(dataset_module, "NuImages", MagicMock())
    dataset_module.NuImages.return_value.get = MagicMock(return_value={"timestamp": 0})
    dataset_module.NuImages.return_value.object_ann = []
    monkeypatch.setattr(dataset_module, "PredictHelper", MagicMock())
    monkeypatch.setattr(dataset_module, "_get_sample_tokens", MagicMock(return_value=["sample_0"]))

    injected_resolver = MagicMock()
    injected_resolver.resolve_re_ids.return_value = ["9999"]
    factory = MagicMock(return_value=injected_resolver)

    captured = {}
    original_sample_to_frame_2d = dataset_module._sample_to_frame_2d

    def _spy_sample_to_frame_2d(*args, **kwargs):
        captured["resolver"] = kwargs.get("traffic_light_id_resolver")
        return original_sample_to_frame_2d(*args, **kwargs)

    monkeypatch.setattr(dataset_module, "_sample_to_frame_2d", _spy_sample_to_frame_2d)

    dataset_module._load_dataset(
        dataset_path="some/dataset/path",
        evaluation_task=EvaluationTask.CLASSIFICATION2D,
        label_converter=_fake_label_converter(TrafficLightLabel),
        frame_ids=[FrameID.CAM_FRONT],
        load_raw_data=False,
        traffic_light_id_resolver_factory=factory,
    )

    factory.assert_called_once_with(nusc, "some/dataset/path")
    assert captured["resolver"] is injected_resolver


# ---------------------------------------------------------------------------
# Fan-out: one instance_token resolving to multiple RE IDs must produce one
# ground truth DynamicObject2D per RE ID (see module docstring for why: the live
# Autoware node reports state per RE, and matching is done by uuid equality).
# ---------------------------------------------------------------------------


def _run_sample_to_frame_2d_with_one_tlr_annotation(resolver, *, instance_token="inst_0"):
    """Drive `_sample_to_frame_2d` with exactly one TrafficLightLabel `object_ann`
    resolved through `resolver`. Returns the resulting `FrameGroundTruth`."""
    from perception_eval.common.dataset_utils import _sample_to_frame_2d
    from perception_eval.common.evaluation_task import EvaluationTask
    from perception_eval.common.label import TrafficLightLabel
    from perception_eval.common.schema import FrameID

    sample_token = "sample_0"
    sample_data_token = "sd_0"

    def _nusc_get(table, token):
        if table == "sample":
            return {"data": {"CAM_FRONT": sample_data_token}}
        if table == "sample_data":
            return {"is_key_frame": False}
        raise AssertionError(f"unexpected nusc.get({table!r}, {token!r})")

    nusc = MagicMock()
    nusc.get.side_effect = _nusc_get

    def _nuim_get(table, token):
        if table == "sample":
            return {"timestamp": 0}
        if table == "category":
            return {"name": "red"}
        raise AssertionError(f"unexpected nuim.get({table!r}, {token!r})")

    nuim = MagicMock()
    nuim.get.side_effect = _nuim_get
    nuim.object_ann = [
        {
            "sample_data_token": sample_data_token,
            "instance_token": instance_token,
            "category_token": "cat_red",
            "attribute_tokens": [],
        }
    ]

    return _sample_to_frame_2d(
        nusc=nusc,
        nuim=nuim,
        sample_token=sample_token,
        evaluation_task=EvaluationTask.CLASSIFICATION2D,
        label_converter=_fake_label_converter(TrafficLightLabel),
        frame_ids=[FrameID.CAM_FRONT],
        frame_name="0",
        load_raw_data=False,
        traffic_light_id_resolver=resolver,
    )


def test_sample_to_frame_2d_fans_out_instance_to_multiple_regulatory_elements() -> None:
    resolver = MagicMock()
    resolver.resolve_re_ids.return_value = ["re_a", "re_b"]

    frame = _run_sample_to_frame_2d_with_one_tlr_annotation(resolver)

    resolver.resolve_re_ids.assert_called_once_with("inst_0")
    assert {obj.uuid for obj in frame.objects} == {"re_a", "re_b"}
    assert len(frame.objects) == 2


def test_sample_to_frame_2d_deduplicates_resolver_output() -> None:
    """A resolver returning a duplicate RE id must not produce duplicate GT
    objects for the same RE (defense in depth on top of each resolver's own
    normalization)."""
    resolver = MagicMock()
    resolver.resolve_re_ids.return_value = ["re_a", "re_a", "re_b"]

    frame = _run_sample_to_frame_2d_with_one_tlr_annotation(resolver)

    assert {obj.uuid for obj in frame.objects} == {"re_a", "re_b"}
    assert len(frame.objects) == 2


def test_sample_to_frame_2d_raises_when_resolver_returns_empty() -> None:
    resolver = MagicMock()
    resolver.resolve_re_ids.return_value = []

    with pytest.raises(TrafficLightRelationError, match="returned no Regulatory Element IDs"):
        _run_sample_to_frame_2d_with_one_tlr_annotation(resolver)
