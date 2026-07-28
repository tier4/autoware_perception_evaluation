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

from unittest.mock import MagicMock

from perception_eval.common.tlr_relation import LegacyInstanceNameResolver
from perception_eval.common.tlr_relation import TrafficLightRelationError
import pytest


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
# Injection point: perception_eval.common.dataset._load_dataset builds the
# resolver once per dataset load, defaulting to LegacyInstanceNameResolver, and
# uses a caller-supplied `traffic_light_id_resolver_factory` instead when given.
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
    assert captured["resolver"].resolve_re_id("inst_0") == "1234"


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
    injected_resolver.resolve_re_id.return_value = "9999"
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
