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

import json
import math
import os
import pickle
import tempfile
from test.evaluation.metrics.segmentation.helpers import make_frame
from test.evaluation.metrics.segmentation.helpers import scores_for
from test.util.dummy_object import make_dummy_data
import tracemalloc
import unittest

import numpy as np
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.schema import FrameID
from perception_eval.common.transform import HomogeneousMatrix
from perception_eval.config import SegmentationEvaluationConfig
from perception_eval.evaluation.metrics.segmentation import SegmentationFrame
from perception_eval.evaluation.metrics.segmentation import SegmentationFrameSummary
from perception_eval.evaluation.metrics.segmentation import SegmentationMetricsReport
from perception_eval.evaluation.metrics.segmentation import SegmentationMetricSuite
from perception_eval.manager import SegmentationEvaluationManager
from pyquaternion import Quaternion

NAMES = ["car", "truck", "road", "sidewalk"]


def _config_dict(**overrides):
    cfg = {
        "evaluation_task": "segmentation",
        "class_names": NAMES,
        "ranges": [{"name": "near", "min_distance": 0.0, "max_distance": 30.0}],
        "class_groups": {"vehicle": ["car", "truck"], "flat": ["road", "sidewalk"]},
        "filters": [{"name": "corridor", "type": "corridor", "width_m": 3.0}],
        "components": [
            "confusion_matrix",
            "iou",
            "accuracy",
            {"type": "calibration", "num_bins": 15},
            "uncertainty_usefulness",
            "confident_error",
            "error_clusters",
            "tolerant_error",
        ],
    }
    cfg.update(overrides)
    return cfg


def _random_frame(rng, n, frame_name):
    target = rng.integers(0, 4, n)
    pred = target.copy()
    flip = rng.random(n) < 0.1
    pred[flip] = rng.integers(0, 4, int(flip.sum()))
    return make_frame(
        [],
        target,
        pred,
        num_classes=4,
        scores=scores_for(pred, 0.8, 4),
        frame_name=frame_name,
        coords=rng.uniform(-60, 60, (n, 3)),
    )


class TestSegmentationEvaluationManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config = SegmentationEvaluationConfig(
            dataset_paths=[],
            frame_id="base_link",
            result_root_directory=self.tmpdir.name,
            evaluation_config_dict=_config_dict(),
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_streaming_matches_suite_and_keeps_no_frames(self):
        manager = SegmentationEvaluationManager(self.config)
        suite = SegmentationMetricSuite(self.config.metrics_config)
        rng = np.random.default_rng(11)
        for index in range(3):
            frame = _random_frame(rng, 150, str(index))
            summary = manager.add_frame(frame)
            suite.update(frame)
            self.assertIsInstance(summary, SegmentationFrameSummary)
        self.assertEqual(manager.num_frames, 3)
        self.assertEqual(len(manager.frame_summaries), 3)
        self.assertFalse(hasattr(manager, "frame_results"))
        report = manager.get_scene_result(save_report=True)
        expected = suite.compute()
        self.assertEqual(set(report.values), set(expected.values))
        for key, value in expected.values.items():
            if math.isnan(value):
                self.assertTrue(math.isnan(report.values[key]))
            else:
                self.assertAlmostEqual(report.values[key], value, places=9, msg=key)
        json_path = os.path.join(self.config.log_directory, "segmentation_metrics.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path) as f:
            data = json.load(f)
        self.assertEqual(data["num_frames"], 3)
        manager.reset()
        self.assertEqual(manager.num_frames, 0)
        with self.assertRaises(NotImplementedError):
            _ = manager.visualizer

    def test_report_round_trips(self):
        manager = SegmentationEvaluationManager(self.config)
        rng = np.random.default_rng(12)
        manager.add_frame_result(_random_frame(rng, 100, "0"))
        report = manager.get_scene_result()
        pickled = pickle.loads(pickle.dumps(report))
        self.assertEqual(set(pickled.values), set(report.values))
        for key, value in report.values.items():
            self.assertTrue(math.isnan(pickled.values[key]) if math.isnan(value) else pickled.values[key] == value, key)
        np.testing.assert_array_equal(pickled.confusion, report.confusion)
        restored = SegmentationMetricsReport.deserialization(report.serialization())
        self.assertEqual(set(restored.values), set(report.values))
        self.assertEqual(restored.class_names, tuple(NAMES))
        self.assertEqual(restored.group_names, ("vehicle", "flat"))
        np.testing.assert_array_equal(restored.confusion_matrix("grouped"), report.confusion_matrix("grouped"))
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.to_json(path, nan_as_null=True)
            with open(path) as f:
                data = json.load(f)
            self.assertIn("report", data)
            for value in data["report"]["values"].values():
                self.assertTrue(value is None or isinstance(value, float))
        finally:
            os.remove(path)
        text = str(report)
        self.assertIn("segmentation/mIoU", text)
        self.assertNotIn("confusion_car__truck", text)
        self.assertIn("segmentation/mIoU", report.to_flat_keys("/"))
        config_copy = pickle.loads(pickle.dumps(self.config.metrics_config))
        self.assertEqual(config_copy.class_names, tuple(NAMES))

    def test_frame_from_ground_truth(self):
        _, ground_truth_objects = make_dummy_data()
        ego2map = HomogeneousMatrix(
            (1.0, 2.0, 0.0), Quaternion(axis=[0, 0, 1], angle=0.0), FrameID.BASE_LINK, FrameID.MAP
        )
        raw = np.zeros((10, 4), dtype=np.float32)
        raw[:, 0] = np.arange(10)
        ground_truth = FrameGroundTruth(
            unix_time=100,
            frame_name="3",
            objects=ground_truth_objects,
            transforms=[ego2map],
            raw_data={FrameID.LIDAR_TOP: raw},
        )
        manager = SegmentationEvaluationManager(self.config)
        targets = np.zeros(10, dtype=np.int64)
        frame = manager.frame_from_ground_truth(
            ground_truth, targets, targets, scores_for(targets, 0.9, 4), scene_id="scene-a"
        )
        self.assertIsInstance(frame, SegmentationFrame)
        self.assertEqual(frame.frame_name, "3")
        self.assertEqual(frame.scene_id, "scene-a")
        self.assertEqual(frame.num_points, 10)
        self.assertEqual(frame.coordinates.shape, (10, 3))
        self.assertEqual(len(frame.gt_objects), len(ground_truth_objects))
        np.testing.assert_allclose(frame.ego2map[:3, 3], [1.0, 2.0, 0.0])
        summary = manager.add_frame(frame)
        self.assertEqual(summary.num_valid, 10)
        no_raw = FrameGroundTruth(unix_time=100, frame_name="4", objects=[], transforms=[ego2map])
        with self.assertRaises(ValueError):
            manager.frame_from_ground_truth(no_raw, targets, targets, scores_for(targets, 0.9, 4))
        explicit = manager.frame_from_ground_truth(
            no_raw, targets, targets, scores_for(targets, 0.9, 4), coordinates=raw[:, :3]
        )
        self.assertEqual(explicit.num_points, 10)

    def test_memory_stays_bounded(self):
        manager = SegmentationEvaluationManager(self.config)
        rng = np.random.default_rng(13)
        num_points = 200_000
        for index in range(2):
            manager.add_frame(_random_frame(rng, num_points, str(index)))
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        for index in range(2, 22):
            manager.add_frame(_random_frame(rng, num_points, str(index)))
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        growth = sum(stat.size_diff for stat in after.compare_to(before, "filename") if stat.size_diff > 0)
        self.assertLess(growth, 2 * 1024 * 1024, f"retained memory grew by {growth / 1e6:.1f} MB over 20 frames")
        self.assertEqual(manager.num_frames, 22)


if __name__ == "__main__":
    unittest.main()
