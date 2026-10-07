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

"""Prepared samples: prepare/evaluate_prepared equivalence, the npz file, pooling and its errors."""

from dataclasses import replace
import json
import math
import os
import pickle
import tempfile
from test.evaluation.metrics.detection.test_suite import _frame as _plain_frame
from test.evaluation.metrics.detection.test_suite import TARGET_LABELS as PLAIN_LABELS
from test.evaluation.metrics.detection.test_suite import TestAdvancedDetectionSuiteInputs
from test.evaluation.metrics.detection.test_suite_map import _frame as _map_frame
from test.evaluation.metrics.detection.test_suite_map import _object
from test.evaluation.metrics.detection.test_suite_map import _road_map
from test.evaluation.metrics.detection.test_suite_map import _StubProvider
from test.evaluation.metrics.detection.test_suite_map import SECTION as MAP_SECTION
from test.evaluation.metrics.detection.test_suite_map import TARGET_LABELS as MAP_LABELS
from test.manager.test_perception_evaluation_manager_advanced import _build_manager
from test.manager.test_perception_evaluation_manager_advanced import _run_scene
from test.manager.test_perception_evaluation_manager_advanced import ADVANCED_SECTION
from test.util.dummy_object import make_dummy_data
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import unittest
from unittest import mock

import numpy as np
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.shape import Shape
from perception_eval.common.shape import ShapeType
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.metrics.component import MetricReport
from perception_eval.evaluation.metrics.detection.config import AdvancedDetectionMetricsConfig
from perception_eval.evaluation.metrics.detection.frame import DetectionFrame
from perception_eval.evaluation.metrics.detection.prepared import BOX_COLUMNS
from perception_eval.evaluation.metrics.detection.prepared import config_digest
from perception_eval.evaluation.metrics.detection.prepared import COUNT_KEYS
from perception_eval.evaluation.metrics.detection.prepared import evaluate_prepared
from perception_eval.evaluation.metrics.detection.prepared import PoolingError
from perception_eval.evaluation.metrics.detection.prepared import PREPARED_SCHEMA_VERSION
from perception_eval.evaluation.metrics.detection.prepared import PreparedSamples
from perception_eval.evaluation.metrics.detection.suite import AdvancedDetectionSuite
from perception_eval.evaluation.metrics.filter_config import MapConfig
from perception_eval.tool import PerceptionAnalyzer3D
from pyquaternion import Quaternion
from shapely.geometry import Polygon

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
POOLED_GOLDEN = os.path.join(FIXTURE_DIR, "pooled_scenes_expected.json")


# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------


def _polygon_object() -> DynamicObject:
    return DynamicObject(
        unix_time=100,
        frame_id=FrameID.BASE_LINK,
        position=(1.0, 1.0, 1.0),
        orientation=Quaternion(),
        shape=Shape(ShapeType.POLYGON, (0.0, 0.0, 1.0), footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
        velocity=None,
        semantic_score=0.9,
        semantic_label=Label(AutowareLabel.CAR, "car"),
    )


def _plain_cases() -> List[Tuple[str, Dict[str, Any], List[DetectionFrame]]]:
    """``(name, section, frames)`` for the map-free fixtures of ``test_suite.py``."""
    estimated, ground_truth = make_dummy_data()
    two_frames = [_plain_frame("0", estimated, ground_truth), _plain_frame("1", estimated, ground_truth)]
    nan_estimated, _ = make_dummy_data()
    nan_estimated[0].semantic_score = float("nan")
    existence_estimated, _ = make_dummy_data()
    for obj in existence_estimated:
        obj.existence_probability = 0.25
    return [
        (
            "identity",
            {
                "components": [
                    {"type": "corner_error"},
                    {"type": "heading_flip"},
                    {"type": "calibration"},
                    {"type": "confusion_matrix"},
                ]
            },
            two_frames,
        ),
        (
            "grouped_filter_range",
            {
                "ranges": [{"name": "near", "max_distance": 30.0}, {"name": "far", "min_distance": 30.0}],
                "class_groups": {"grouped_vehicle": ["car", "motorbike"], "grouped_vru": ["bicycle", "pedestrian"]},
                "filters": [{"name": "corridor", "type": "corridor", "width_m": 3.0}],
                "components": [{"type": "corner_error"}],
            },
            two_frames,
        ),
        (
            "zero_coverage",
            {
                "filters": [{"name": "road", "type": "region", "regions": ["road"]}],
                "components": [{"type": "corner_error"}],
                "map": {"resolver": "explicit", "mapping": {"other_scene": "/nonexistent/lanelet2_map.osm"}},
            },
            [_plain_frame("0", *make_dummy_data(), scene_id="scene_without_map")],
        ),
        (
            "no_pose",
            {
                "filters": [{"name": "road", "type": "region", "regions": ["road"]}],
                "components": [{"type": "heading_flip"}],
                "map": {"resolver": "explicit", "mapping": {"s": "/nonexistent.osm"}},
            },
            [_plain_frame("0", *make_dummy_data(), scene_id="s", with_pose=False)],
        ),
        (
            "polygon_skip",
            {"components": [{"type": "confident_error"}]},
            [two_frames[0], _plain_frame("1", estimated + [_polygon_object()], ground_truth)],
        ),
        (
            "polygon_as_box",
            {"components": [{"type": "confident_error"}], "polygon_shapes": "as_box"},
            [TestAdvancedDetectionSuiteInputs._polygon_frame()],
        ),
        (
            "nan_score",
            {"components": [{"type": "calibration"}, {"type": "confident_error"}]},
            [_plain_frame("0", nan_estimated, ground_truth)],
        ),
        (
            "existence_probability",
            {"components": [{"type": "confident_error"}], "score_source": "existence_probability"},
            [_plain_frame("0", existence_estimated, ground_truth)],
        ),
        (
            "missing_existence_probability",
            {"components": [{"type": "confident_error"}], "score_source": "existence_probability"},
            [_plain_frame("0", estimated, ground_truth)],
        ),
    ]


def _map_config(section: Optional[Dict[str, Any]] = None) -> AdvancedDetectionMetricsConfig:
    return AdvancedDetectionMetricsConfig.from_dict(MAP_SECTION if section is None else section, MAP_LABELS)


def _map_provider(no_map: Sequence[str] = ("unmapped",)) -> _StubProvider:
    return _StubProvider(_road_map(), no_map=no_map)


def _scene_a() -> List[DetectionFrame]:
    """Two frames on a mapped scene: a perfect car and a phantom pedestrian on the road ahead."""
    gt = [_object(20.0, 0.0, AutowareLabel.CAR), _object(30.0, 40.0, AutowareLabel.PEDESTRIAN, size=(0.6, 0.6, 1.7))]
    est = [
        _object(20.0, 0.0, AutowareLabel.CAR, 0.9),
        _object(30.0, 0.0, AutowareLabel.PEDESTRIAN, 0.8, size=(0.6, 0.6, 1.7)),
    ]
    return [_map_frame("0", est, gt), _map_frame("1", est, gt)]


def _scene_b() -> List[DetectionFrame]:
    """Two frames on a scene without a lanelet map: map views and TTC are uncovered."""
    gt = [_object(20.0, 0.0, AutowareLabel.CAR), _object(12.0, 2.0, AutowareLabel.PEDESTRIAN, size=(0.6, 0.6, 1.7))]
    est = [_object(20.5, 0.3, AutowareLabel.CAR, 0.7), _object(5.0, -3.0, AutowareLabel.CAR, 0.6)]
    return [_map_frame("0", est, gt, scene_id="unmapped"), _map_frame("1", est, gt, scene_id="unmapped")]


def _scene_c() -> List[DetectionFrame]:
    """Three frames that all contain a POLYGON-shaped estimation: every frame is skipped."""
    gt = [_object(20.0, 0.0, AutowareLabel.CAR)]
    est = [_object(20.0, 0.0, AutowareLabel.CAR, 0.9), _polygon_object()]
    return [_map_frame(str(index), est, gt) for index in range(3)]


def _report_to_json(report: MetricReport, prepared: PreparedSamples) -> Dict[str, Any]:
    return {
        "values": {key: (None if math.isnan(value) else value) for key, value in report.values.items()},
        "coverage": {name: list(pair) for name, pair in report.coverage.items()},
        "counts": dict(prepared.counts),
        "num_frames": prepared.num_frames,
        "warnings": list(report.warnings),
    }


# ------------------------------------------------------------------------------------------------
# Assertions shared by the test classes
# ------------------------------------------------------------------------------------------------


class _Assertions(unittest.TestCase):
    def assert_same_values(self, actual: Dict[str, float], expected: Dict[str, float]) -> None:
        self.assertEqual(list(actual), list(expected))
        for key, value in expected.items():
            if math.isnan(value):
                self.assertTrue(math.isnan(actual[key]), key)
            else:
                self.assertEqual(actual[key], value, key)

    def assert_same_report(self, actual: MetricReport, expected: MetricReport, warnings: bool = True) -> None:
        self.assert_same_values(actual.values, expected.values)
        self.assertEqual(actual.coverage, expected.coverage)
        if warnings:
            self.assertEqual(actual.warnings, expected.warnings)

    def assert_same_samples(self, actual: PreparedSamples, expected: PreparedSamples) -> None:
        self.assertEqual(actual.schema_version, expected.schema_version)
        self.assertEqual(actual.config_digest, expected.config_digest)
        self.assertEqual(actual.class_names, expected.class_names)
        self.assertEqual(actual.filter_names, expected.filter_names)
        self.assertEqual(actual.has_ttc, expected.has_ttc)
        self.assertEqual(actual.counts, expected.counts)
        self.assertEqual(actual.warnings, expected.warnings)
        self.assertEqual(actual.num_frames, expected.num_frames)
        for got, want in zip(actual.frames, expected.frames):
            self.assertEqual(got.sample.frame_name, want.sample.frame_name)
            self.assertEqual(got.unix_time, want.unix_time)
            self.assertEqual(got.sample.ttc_covered, want.sample.ttc_covered)
            self.assertEqual(dict(got.available), dict(want.available))
            for attribute in ("pred_boxes", "pred_scores", "pred_labels", "gt_boxes", "gt_labels"):
                self.assertTrue(
                    np.array_equal(getattr(got.sample, attribute), getattr(want.sample, attribute)), attribute
                )
                self.assertEqual(getattr(got.sample, attribute).dtype, getattr(want.sample, attribute).dtype)
            for attribute in ("pred_ttc", "gt_ttc"):
                got_ttc, want_ttc = getattr(got.sample, attribute), getattr(want.sample, attribute)
                if want_ttc is None:
                    self.assertIsNone(got_ttc)
                else:
                    self.assertTrue(np.array_equal(got_ttc, want_ttc), attribute)  # inf == inf
            for name in expected.filter_names:
                self.assertTrue(np.array_equal(got.pred_masks[name], want.pred_masks[name]))
                self.assertTrue(np.array_equal(got.gt_masks[name], want.gt_masks[name]))


# ------------------------------------------------------------------------------------------------
# prepare + evaluate_prepared == evaluate
# ------------------------------------------------------------------------------------------------


class TestPrepareEvaluateEquivalence(_Assertions):
    def test_plain_fixtures_are_bit_identical(self):
        for name, section, frames in _plain_cases():
            with self.subTest(name):
                config = AdvancedDetectionMetricsConfig.from_dict(section, PLAIN_LABELS)
                suite = AdvancedDetectionSuite(config)
                prepared = suite.prepare(frames, PLAIN_LABELS)
                self.assertEqual(prepared.schema_version, PREPARED_SCHEMA_VERSION)
                self.assertEqual(prepared.config_digest, config_digest(config))
                self.assertEqual(set(prepared.counts), set(COUNT_KEYS))
                self.assertEqual(prepared.counts["input_frames"], len(frames))
                by_prepared = suite.evaluate_prepared(prepared)
                by_frames = AdvancedDetectionSuite(config).evaluate(frames, PLAIN_LABELS)
                self.assert_same_report(by_prepared, by_frames)
                # The module-level entry point builds a fresh suite and reaches the same numbers.
                self.assert_same_report(evaluate_prepared(config, prepared), by_frames)

    def test_map_fixtures_are_bit_identical(self):
        config = _map_config()
        cases = {
            "mapped": (_scene_a(), ()),
            "partial": (_scene_a()[:1] + _scene_b()[:1], ("unmapped",)),
            "no_coverage": (_scene_b()[:1], ("unmapped",)),
        }
        for name, (frames, no_map) in cases.items():
            with self.subTest(name):
                suite = AdvancedDetectionSuite(config, map_provider=_map_provider(no_map))
                prepared = suite.prepare(frames, MAP_LABELS)
                self.assertTrue(prepared.has_ttc)
                by_frames = AdvancedDetectionSuite(config, map_provider=_map_provider(no_map)).evaluate(
                    frames, MAP_LABELS
                )
                self.assert_same_report(suite.evaluate_prepared(prepared), by_frames)
                self.assert_same_report(evaluate_prepared(config, prepared), by_frames)

    def test_counts_and_dropped_prediction_warning(self):
        estimated, ground_truth = make_dummy_data()
        estimated[0].semantic_score = 0.0
        config = AdvancedDetectionMetricsConfig.from_dict(
            {"components": [{"type": "confident_error"}], "score_source": "existence_probability"}, PLAIN_LABELS
        )
        prepared = AdvancedDetectionSuite(config).prepare([_plain_frame("0", estimated, ground_truth)], PLAIN_LABELS)
        # No estimation carries an existence probability: everything is dropped and said so.
        self.assertEqual(prepared.counts["pred_total"], len(estimated))
        self.assertEqual(prepared.counts["pred_kept"], 0)
        self.assertEqual(prepared.counts["non_finite_score"], len(estimated))
        self.assertIn(f"all {len(estimated)} predictions dropped: no existence_probability", prepared.warnings)
        self.assertEqual(prepared.counts["zero_score"], 0)

        config = AdvancedDetectionMetricsConfig.from_dict({"components": [{"type": "confident_error"}]}, PLAIN_LABELS)
        prepared = AdvancedDetectionSuite(config).prepare([_plain_frame("0", estimated, ground_truth)], PLAIN_LABELS)
        self.assertEqual(prepared.counts["pred_total"], len(estimated))
        self.assertEqual(prepared.counts["pred_kept"], len(estimated))
        self.assertEqual(prepared.counts["zero_score"], 1)
        self.assertFalse(any("all" in warning for warning in prepared.warnings))

    def test_unavailable_filter_masks_are_all_false(self):
        config = _map_config()
        prepared = AdvancedDetectionSuite(config, map_provider=_map_provider()).prepare(_scene_b(), MAP_LABELS)
        for frame in prepared.frames:
            self.assertEqual(frame.available, {"road": False, "collision": False})
            self.assertFalse(frame.sample.ttc_covered)
            for name in prepared.filter_names:
                self.assertEqual(frame.pred_masks[name].shape, (frame.sample.num_predictions,))
                self.assertFalse(frame.pred_masks[name].any())
                self.assertFalse(frame.gt_masks[name].any())
            self.assertTrue(np.all(np.isinf(frame.sample.gt_ttc)))


# ------------------------------------------------------------------------------------------------
# npz
# ------------------------------------------------------------------------------------------------


class TestNpzRoundTrip(_Assertions):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _path(self, name: str = "advanced_detection_samples.npz") -> str:
        return os.path.join(self.tmpdir.name, name)

    def test_round_trip_with_ttc_keeps_every_array(self):
        config = _map_config()
        frames = _scene_a() + _scene_b()
        prepared = AdvancedDetectionSuite(config, map_provider=_map_provider()).prepare(frames, MAP_LABELS)
        digest = prepared.to_npz(self._path(), producer={"name": "test", "version": "0"})
        self.assertEqual(len(digest), 64)
        loaded = PreparedSamples.from_npz(self._path())
        self.assert_same_samples(loaded, prepared)
        # Unreachable objects keep their inf TTC through the file.
        self.assertTrue(any(np.isinf(frame.sample.gt_ttc).any() for frame in loaded.frames))
        self.assert_same_report(evaluate_prepared(config, loaded), evaluate_prepared(config, prepared))

    def test_round_trip_without_ttc(self):
        name, section, frames = _plain_cases()[1]
        config = AdvancedDetectionMetricsConfig.from_dict(section, PLAIN_LABELS)
        prepared = AdvancedDetectionSuite(config).prepare(frames, PLAIN_LABELS)
        prepared.to_npz(self._path())
        loaded = PreparedSamples.from_npz(self._path())
        self.assertFalse(loaded.has_ttc)
        self.assertIsNone(loaded.frames[0].sample.gt_ttc)
        self.assert_same_samples(loaded, prepared)
        self.assert_same_report(evaluate_prepared(config, loaded), AdvancedDetectionSuite(config).evaluate(frames))

    def test_file_layout_and_no_object_arrays(self):
        config = _map_config()
        prepared = AdvancedDetectionSuite(config, map_provider=_map_provider()).prepare(_scene_a(), MAP_LABELS)
        prepared.to_npz(self._path())
        with np.load(self._path(), allow_pickle=False) as data:
            keys = set(data.files)
            expected = {
                "header",
                "frame_name",
                "unix_time",
                "ttc_covered",
                "available",
                "pred_offsets",
                "gt_offsets",
                "pred_boxes",
                "gt_boxes",
                "pred_scores",
                "pred_labels",
                "gt_labels",
                "pred_ttc",
                "gt_ttc",
                "pred_masks",
                "gt_masks",
            }
            self.assertEqual(keys, expected)
            for key in keys:
                self.assertNotEqual(data[key].dtype.kind, "O", key)
            header = json.loads(bytes(data["header"]).decode("utf-8"))
            self.assertEqual(header["schema_version"], PREPARED_SCHEMA_VERSION)
            self.assertEqual(header["config_digest"], config_digest(config))
            self.assertEqual(header["box_columns"], list(BOX_COLUMNS))
            self.assertEqual(header["class_names"], list(MAP_LABELS[i].value for i in range(len(MAP_LABELS))))
            self.assertEqual(header["filter_names"], ["road", "collision"])
            self.assertEqual(header["producer"], {})
            num_frames = len(prepared.frames)
            self.assertEqual(data["frame_name"].dtype, np.dtype("<U64"))
            self.assertEqual(data["unix_time"].dtype, np.int64)
            self.assertEqual(data["available"].shape, (num_frames, 2))
            self.assertEqual(data["pred_offsets"].shape, (num_frames + 1,))
            self.assertEqual(int(data["pred_offsets"][-1]), data["pred_boxes"].shape[0])
            self.assertEqual(int(data["gt_offsets"][-1]), data["gt_boxes"].shape[0])
            self.assertEqual(data["pred_boxes"].shape[1], 9)
            self.assertEqual(data["pred_boxes"].dtype, np.float64)
            self.assertEqual(data["pred_masks"].shape, (data["pred_boxes"].shape[0], 2))
            self.assertEqual(data["gt_masks"].dtype, bool)

    def test_empty_samples_round_trip(self):
        config = _map_config()
        empty = AdvancedDetectionSuite(config, map_provider=_map_provider()).prepare([], MAP_LABELS)
        self.assertEqual(empty.num_frames, 0)
        empty.to_npz(self._path())
        loaded = PreparedSamples.from_npz(self._path())
        self.assert_same_samples(loaded, empty)
        report = evaluate_prepared(config, loaded)
        self.assertEqual(report.coverage, {"road": (0, 0), "collision": (0, 0), "ttc": (0, 0)})
        self.assertTrue(all(math.isnan(value) for value in report.values.values()))
        self.assertEqual(report.warnings, [])

    def test_unknown_schema_version_is_refused(self):
        config = _map_config()
        prepared = AdvancedDetectionSuite(config, map_provider=_map_provider()).prepare(_scene_a(), MAP_LABELS)
        arrays = prepared._to_arrays(None)
        header = json.loads(bytes(arrays["header"]).decode("utf-8"))
        header["schema_version"] = PREPARED_SCHEMA_VERSION + 1
        arrays["header"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
        with open(self._path("future.npz"), "wb") as stream:
            np.savez_compressed(stream, **arrays)
        with self.assertRaises(PoolingError):
            PreparedSamples.from_npz(self._path("future.npz"))

    def test_path_is_used_verbatim(self):
        config = AdvancedDetectionMetricsConfig.from_dict({"components": [{"type": "heading_flip"}]}, PLAIN_LABELS)
        prepared = AdvancedDetectionSuite(config).prepare(_plain_cases()[0][2], PLAIN_LABELS)
        path = self._path("samples.bin")
        prepared.to_npz(path)
        self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(path + ".npz"))


# ------------------------------------------------------------------------------------------------
# Pooling
# ------------------------------------------------------------------------------------------------


class TestPooling(_Assertions):
    def setUp(self):
        self.config = _map_config()
        self.scenes = {"A": _scene_a(), "B": _scene_b(), "C": _scene_c()}
        self.prepared = {
            name: AdvancedDetectionSuite(self.config, map_provider=_map_provider()).prepare(frames, MAP_LABELS)
            for name, frames in self.scenes.items()
        }

    def test_pooled_equals_evaluate_over_all_frames(self):
        parts = [self.prepared["A"], self.prepared["B"], self.prepared["C"]]
        pooled = PreparedSamples.concat(parts)
        self.assertEqual(pooled.num_frames, 4)  # scene C is skipped entirely
        self.assertEqual(pooled.counts["input_frames"], 7)
        self.assertEqual(pooled.counts["skipped_frames"], 3)
        self.assertEqual(
            pooled.counts["pred_total"],
            self.prepared["A"].counts["pred_total"] + self.prepared["B"].counts["pred_total"],
        )
        all_frames = self.scenes["A"] + self.scenes["B"] + self.scenes["C"]
        by_frames = AdvancedDetectionSuite(self.config, map_provider=_map_provider()).evaluate(all_frames, MAP_LABELS)
        by_parts = evaluate_prepared(self.config, parts)
        self.assert_same_report(by_parts, by_frames, warnings=False)
        self.assertEqual(by_parts.coverage, {"road": (2, 4), "collision": (2, 4), "ttc": (2, 4)})
        # Warnings are the deduplicated union of the per-scene warnings plus the pooled coverage notes.
        for warning in self.prepared["C"].warnings:
            self.assertIn(warning, by_parts.warnings)
        self.assertTrue(any("covered 2/4 frames" in warning for warning in by_parts.warnings))

    def test_pooled_matches_committed_golden_values(self):
        with open(POOLED_GOLDEN, encoding="utf-8") as stream:
            golden = json.load(stream)
        parts = [self.prepared["A"], self.prepared["B"], self.prepared["C"]]
        pooled = PreparedSamples.concat(parts)
        report = evaluate_prepared(self.config, pooled)
        self.assertEqual(golden["config_digest"], config_digest(self.config))
        self.assertEqual(golden["num_frames"], pooled.num_frames)
        self.assertEqual(golden["counts"], dict(pooled.counts))
        self.assertEqual({name: tuple(pair) for name, pair in golden["coverage"].items()}, report.coverage)
        self.assertEqual(list(golden["values"]), list(report.values))
        for key, expected in golden["values"].items():
            actual = report.values[key]
            if expected is None:
                self.assertTrue(math.isnan(actual), key)
            else:
                self.assertTrue(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), (key, actual, expected))
        self.assertEqual(golden["warnings"], list(report.warnings))

    def test_canonical_order_reproduces_evaluate_in_both_directions(self):
        # Equal scores across scenes make the greedy matcher's tie-break depend on frame order, so
        # pooling [A, B] and [B, A] may legitimately differ. Each order reproduces ``evaluate`` over
        # the frames in that same order: callers pool in the canonical order (dataset, scenario).
        config = AdvancedDetectionMetricsConfig.from_dict(
            {"components": [{"type": "corner_error"}, {"type": "calibration"}, {"type": "confusion_matrix"}]},
            MAP_LABELS,
        )
        gt = [_object(10.0, 0.0, AutowareLabel.CAR)]
        scene_a = [_map_frame("0", [_object(10.5, 0.0, AutowareLabel.CAR, 0.9)], gt)]
        scene_b = [_map_frame("0", [_object(11.0, 0.0, AutowareLabel.CAR, 0.9)], gt)]
        prepared_a = AdvancedDetectionSuite(config).prepare(scene_a, MAP_LABELS)
        prepared_b = AdvancedDetectionSuite(config).prepare(scene_b, MAP_LABELS)
        forward = evaluate_prepared(config, [prepared_a, prepared_b])
        backward = evaluate_prepared(config, [prepared_b, prepared_a])
        self.assert_same_report(forward, AdvancedDetectionSuite(config).evaluate(scene_a + scene_b, MAP_LABELS))
        self.assert_same_report(backward, AdvancedDetectionSuite(config).evaluate(scene_b + scene_a, MAP_LABELS))
        pooled = PreparedSamples.concat([prepared_b, prepared_a])
        self.assertEqual([frame.sample.pred_boxes[0, 0] for frame in pooled.frames], [11.0, 10.5])

    def test_single_part_sequence_equals_single_samples(self):
        single = evaluate_prepared(self.config, self.prepared["A"])
        self.assert_same_report(evaluate_prepared(self.config, [self.prepared["A"]]), single)

    def test_evaluate_prepared_never_builds_a_map_provider(self):
        section = dict(MAP_SECTION)
        section["map"] = {
            "resolver": "explicit",
            "mapping": {"scene": "/definitely/missing/lanelet2_map.osm"},
            "max_speed_mps": 10.0,
        }
        config = _map_config(section)
        self.assertEqual(config_digest(config), config_digest(self.config))
        with mock.patch.object(MapConfig, "build_provider", side_effect=AssertionError("map was resolved")):
            report = evaluate_prepared(config, [self.prepared["A"], self.prepared["B"]])
        self.assertIn("detection/critical_fp_0p5m", report.values)
        # Preparing with that configuration, on the other hand, does need the map.
        with mock.patch.object(MapConfig, "build_provider", side_effect=AssertionError("map was resolved")):
            with self.assertRaises(AssertionError):
                AdvancedDetectionSuite(config).prepare(self.scenes["A"], MAP_LABELS)

    def test_pooling_errors(self):
        reference = self.prepared["A"]
        with self.assertRaises(PoolingError):
            PreparedSamples.concat([])
        with self.assertRaises(PoolingError):
            PreparedSamples.concat([reference, replace(reference, class_names=("car", "bus"))])
        with self.assertRaises(PoolingError):
            PreparedSamples.concat([reference, replace(reference, filter_names=("road",))])
        with self.assertRaises(PoolingError):
            PreparedSamples.concat([reference, replace(reference, config_digest="sha256:" + "0" * 64)])
        with self.assertRaises(PoolingError):
            PreparedSamples.concat([reference, replace(reference, schema_version=PREPARED_SCHEMA_VERSION + 1)])
        with self.assertRaises(PoolingError):
            evaluate_prepared(self.config, replace(reference, schema_version=PREPARED_SCHEMA_VERSION + 1))
        other = dict(MAP_SECTION)
        other["components"] = [{"type": "corner_error", "tp_threshold": 1.0}]
        with self.assertRaises(PoolingError):
            evaluate_prepared(_map_config(other), reference)
        with self.assertRaises(PoolingError):
            AdvancedDetectionSuite(_map_config(other)).evaluate_prepared(reference)
        plain = AdvancedDetectionMetricsConfig.from_dict({"components": [{"type": "corner_error"}]}, PLAIN_LABELS)
        with self.assertRaises(PoolingError):
            evaluate_prepared(plain, reference)


# ------------------------------------------------------------------------------------------------
# config_digest
# ------------------------------------------------------------------------------------------------


class TestConfigDigest(unittest.TestCase):
    def test_format(self):
        digest = config_digest(_map_config())
        self.assertTrue(digest.startswith("sha256:"))
        self.assertEqual(len(digest), len("sha256:") + 64)
        self.assertEqual(digest, config_digest(_map_config()))

    def test_invariant_to_map_paths_and_key_order(self):
        section_a = {
            "map": {"resolver": "explicit", "mapping": {"scene": "/a.osm"}, "data_root": "/data/a"},
            "components": [{"type": "corner_error"}, {"type": "critical_fp_fn", "confidences": [0.5]}],
            "filters": [{"name": "road", "type": "region", "regions": ["road"]}],
        }
        section_b = {
            "filters": [{"type": "region", "regions": ["road"], "name": "road"}],
            "components": [{"type": "corner_error"}, {"confidences": [0.5], "type": "critical_fp_fn"}],
            "map": {"data_root": "/data/b", "mapping": {"other": "/b.osm"}, "resolver": "explicit"},
        }
        digest_a = config_digest(AdvancedDetectionMetricsConfig.from_dict(section_a, MAP_LABELS))
        digest_b = config_digest(AdvancedDetectionMetricsConfig.from_dict(section_b, MAP_LABELS))
        self.assertEqual(digest_a, digest_b)
        section_c = dict(section_a)
        section_c["map"] = {"resolver": "t4_scene_directory", "data_root": "/data/c"}
        digest_c = config_digest(AdvancedDetectionMetricsConfig.from_dict(section_c, MAP_LABELS))
        self.assertNotEqual(digest_a, digest_c)  # the resolver is part of the configuration proper

    def test_changes_with_component_parameter_and_class_names(self):
        base = {"components": [{"type": "corner_error", "tp_threshold": 2.0}]}
        changed = {"components": [{"type": "corner_error", "tp_threshold": 1.5}]}
        self.assertNotEqual(
            config_digest(AdvancedDetectionMetricsConfig.from_dict(base, PLAIN_LABELS)),
            config_digest(AdvancedDetectionMetricsConfig.from_dict(changed, PLAIN_LABELS)),
        )
        self.assertNotEqual(
            config_digest(AdvancedDetectionMetricsConfig.from_dict(base, PLAIN_LABELS)),
            config_digest(AdvancedDetectionMetricsConfig.from_dict(base, MAP_LABELS)),
        )
        self.assertNotEqual(
            config_digest(AdvancedDetectionMetricsConfig.from_dict(base, PLAIN_LABELS)),
            config_digest(
                AdvancedDetectionMetricsConfig.from_dict(dict(base, score_source="existence_probability"), PLAIN_LABELS)
            ),
        )


# ------------------------------------------------------------------------------------------------
# MetricsScore and the analyzer
# ------------------------------------------------------------------------------------------------


class TestMetricsScoreAndAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.manager = _build_manager(self.tmpdir.name, ADVANCED_SECTION)
        self.frames = _run_scene(self.manager)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_manager_score_carries_transient_prepared_samples(self):
        score = self.manager.get_scene_result()
        self.assertIsNotNone(score.detection_prepared)
        self.assertEqual(score.detection_prepared.num_frames, len(self.frames))
        self.assertEqual(score.detection_prepared.filter_names, ("corridor",))
        fresh = MetricsScore(config=self.manager.metrics_config, used_frame=[0, 1, 2])
        fresh.detection_metric_report = score.detection_metric_report
        fresh.detection_prepared = score.detection_prepared
        restored: MetricsScore = pickle.loads(pickle.dumps(fresh))
        self.assertIsNone(restored.detection_prepared)  # transient
        self.assertIsNotNone(restored.detection_metric_report)
        state = fresh.__reduce__()[2]
        self.assertNotIn("detection_prepared", state)

    def test_analyzer_runs_the_suite_only_on_request(self):
        analyzer = PerceptionAnalyzer3D(self.manager.evaluator_config)
        default_score = analyzer.get_metrics_score(self.frames)
        self.assertIsNone(default_score.detection_metric_report)
        self.assertIsNone(default_score.detection_prepared)
        self.assertGreater(len(default_score.mean_ap_values), 0)
        advanced_score = analyzer.get_metrics_score(self.frames, advanced=True)
        self.assertIsNotNone(advanced_score.detection_metric_report)
        self.assertIsNotNone(advanced_score.detection_prepared)
        self.assertIn("detection/corner_mean_car", advanced_score.detection_metric_report.values)


if __name__ == "__main__":
    unittest.main()
