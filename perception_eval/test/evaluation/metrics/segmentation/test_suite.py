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

import math
from test.evaluation.metrics.segmentation.helpers import config
from test.evaluation.metrics.segmentation.helpers import make_frame
from test.evaluation.metrics.segmentation.helpers import scores_for
from test.evaluation.metrics.segmentation.helpers import StubMapFilter
import unittest

import numpy as np
from perception_eval.evaluation.metrics.segmentation.suite import SegmentationMetricSuite

NAMES = ("car", "truck", "road", "sidewalk")
GROUPS = {"vehicle": ["car", "truck"], "flat": ["road", "sidewalk"]}
ALL_COMPONENTS = [
    "confusion_matrix",
    "iou",
    "accuracy",
    "precision_recall_f1",
    "calibration",
    "uncertainty_usefulness",
    "confident_error",
    "error_clusters",
    "tolerant_error",
]


def _random_frame(rng, n=200, frame_name="0", scene_id=None, num_classes=4):
    target = rng.integers(0, num_classes, n)
    pred = target.copy()
    flip = rng.random(n) < 0.2
    pred[flip] = rng.integers(0, num_classes, int(flip.sum()))
    return make_frame(
        [],
        target,
        pred,
        num_classes=num_classes,
        scores=scores_for(pred, 0.8, num_classes),
        frame_name=frame_name,
        scene_id=scene_id,
        coords=rng.uniform(-60, 60, (n, 3)),
    )


def _with_stub(suite: SegmentationMetricSuite) -> SegmentationMetricSuite:
    """Append an availability-gated stub filter (as a map-dependent filter would be)."""
    suite.filters.append(StubMapFilter("stub"))
    suite.filter_names = tuple(f.name for f in suite.filters)
    suite.reset()
    return suite


class TestSuiteFanOut(unittest.TestCase):
    def test_key_snapshot(self):
        cfg = config(
            ALL_COMPONENTS,
            class_names=NAMES,
            class_groups=GROUPS,
            ranges=[{"name": "0_30", "min_distance": 0.0, "max_distance": 30.0}],
            filters=[{"name": "corridor", "type": "corridor", "width_m": 3.0}],
        )
        suite = SegmentationMetricSuite(cfg)
        rng = np.random.default_rng(0)
        suite.update(_random_frame(rng))
        report = suite.compute()
        keys = set(report.values)
        # Whole scene, raw taxonomy
        self.assertIn("segmentation/mIoU", keys)
        self.assertIn("segmentation/confusion_car__truck", keys)
        self.assertIn("segmentation/ece", keys)
        self.assertIn("segmentation/entropy_auroc", keys)
        self.assertIn("segmentation/confident_error_rate", keys)
        self.assertIn("segmentation/error_clusters_per_frame", keys)
        self.assertIn("segmentation/tolerant_error_rate_car", keys)
        # Range, filter, grouped combinations
        self.assertIn("segmentation/0m_30m/mIoU", keys)
        self.assertIn("segmentation/corridor/mIoU", keys)
        self.assertIn("segmentation/corridor/0m_30m/error_rate", keys)
        self.assertIn("segmentation/grouped/mIoU", keys)
        self.assertIn("segmentation/grouped/corridor/0m_30m/tolerant_error_rate_vehicle", keys)
        self.assertIn("segmentation/grouped/confusion_vehicle__flat", keys)
        self.assertNotIn("segmentation/grouped/iou_car", keys)
        # Every key is a slash key with the task prefix and no duplicates in flat form.
        self.assertTrue(all(key.startswith("segmentation/") for key in keys))
        self.assertEqual(len(report.to_flat_keys()), len(keys))
        self.assertEqual(report.coverage, {})  # corridor is never availability-gated
        self.assertEqual(report.num_frames, 1)

    def test_range_and_filter_slices_are_subsets(self):
        cfg = config(
            ["accuracy", "error_clusters"],
            class_names=NAMES,
            ranges=[{"name": "near", "max_distance": 30.0}],
            filters=[{"name": "corridor", "type": "corridor", "width_m": 3.0}],
        )
        suite = SegmentationMetricSuite(cfg)
        rng = np.random.default_rng(1)
        frame = _random_frame(rng, n=500)
        suite.update(frame)
        report = suite.compute()
        total = int(report.confusion_matrix("raw", "", "").sum())
        near = int(report.confusion_matrix("raw", "", "near").sum())
        corridor = int(report.confusion_matrix("raw", "corridor", "").sum())
        radius = np.linalg.norm(frame.coordinates[:, :2], axis=1)
        self.assertEqual(total, 500)
        self.assertEqual(near, int(np.sum(radius < 30.0)))
        self.assertEqual(
            corridor, int(np.sum((frame.coordinates[:, 0] >= 0) & (np.abs(frame.coordinates[:, 1]) <= 1.5)))
        )
        self.assertLessEqual(near, total)
        self.assertLessEqual(corridor, total)

    def test_ignore_index_excluded(self):
        cfg = config(["accuracy", "confusion_matrix"], class_names=("road", "obstacle"))
        suite = SegmentationMetricSuite(cfg)
        suite.update(make_frame([0, 1, 2], target=[0, -1, 1], pred=[0, 1, 0]))
        report = suite.compute()
        self.assertEqual(int(report.confusion_matrix().sum()), 2)
        self.assertAlmostEqual(report.values["segmentation/acc"], 0.5)

    def test_include_confusion_cells_false(self):
        cfg = config(["confusion_matrix", "accuracy"], include_confusion_cells=False)
        suite = SegmentationMetricSuite(cfg)
        suite.update(make_frame([0, 1], [0, 1], [0, 1]))
        report = suite.compute()
        self.assertFalse(any("confusion_" in key for key in report.values))
        self.assertIn("segmentation/acc", report.values)


class TestCoverage(unittest.TestCase):
    def test_uncovered_filter_reports_nan_and_warning(self):
        cfg = config(["accuracy", "confusion_matrix", "error_clusters"])
        suite = _with_stub(SegmentationMetricSuite(cfg))
        suite.update(make_frame([1, 2], [0, 1], [0, 1], scene_id=None))
        suite.update(make_frame([1, 2], [0, 1], [0, 0], scene_id=None))
        report = suite.compute()
        self.assertEqual(report.coverage["stub"], (0, 2))
        self.assertTrue(math.isnan(report.values["segmentation/stub/acc"]))
        self.assertTrue(math.isnan(report.values["segmentation/stub/confusion_road__road"]))
        self.assertTrue(math.isnan(report.values["segmentation/stub/error_cluster_count"]))
        self.assertAlmostEqual(report.values["segmentation/acc"], 0.75)  # whole scene keeps every frame
        self.assertEqual(len(report.warnings), 1)
        self.assertIn("0/2", report.warnings[0])

    def test_partial_coverage_excludes_only_uncovered_frames(self):
        cfg = config(["accuracy"])
        suite = _with_stub(SegmentationMetricSuite(cfg))
        suite.update(make_frame([1, 2], [0, 1], [0, 1], scene_id="scene"))  # covered, all correct
        suite.update(make_frame([1, 2], [0, 1], [1, 0], scene_id=None))  # uncovered, all wrong
        suite.update(make_frame([-1, 2], [0, 1], [1, 1], scene_id="scene"))  # covered; x=-1 dropped by the stub
        report = suite.compute()
        self.assertEqual(report.coverage["stub"], (2, 3))
        self.assertAlmostEqual(report.values["segmentation/stub/acc"], 3.0 / 3.0)
        self.assertAlmostEqual(report.values["segmentation/acc"], 3.0 / 6.0)
        self.assertEqual(len(report.warnings), 1)
        self.assertIn("2/3", report.warnings[0])
        summaries_available = suite.coverage_covered["stub"]
        self.assertEqual(summaries_available, 2)

    def test_summary_reports_frame_and_filter_state(self):
        cfg = config(["accuracy"])
        suite = _with_stub(SegmentationMetricSuite(cfg))
        summary = suite.update(make_frame([1, 2, 3], [0, 1, -1], [0, 0, 0], scene_id="s"))
        self.assertEqual(summary.num_points, 3)
        self.assertEqual(summary.num_valid, 2)
        self.assertEqual(summary.num_errors, 1)
        self.assertEqual(summary.filter_available, {"stub": True})


class TestStreamingEqualsBatch(unittest.TestCase):
    def test_three_frames_streamed_match_component_over_concatenation(self):
        cfg = config(["accuracy", "calibration", "confident_error", "uncertainty_usefulness"], class_names=NAMES)
        rng = np.random.default_rng(7)
        frames = [_random_frame(rng, n=100, frame_name=str(i)) for i in range(3)]
        streamed = SegmentationMetricSuite(cfg)
        for frame in frames:
            streamed.update(frame)
        merged = make_frame(
            [],
            np.concatenate([f.targets for f in frames]),
            np.concatenate([f.predictions for f in frames]),
            num_classes=4,
            scores=np.concatenate([f.probabilities for f in frames]),
            coords=np.concatenate([f.coordinates for f in frames]),
        )
        batch = SegmentationMetricSuite(cfg)
        batch.update(merged)
        a, b = streamed.compute(), batch.compute()
        for key in (
            "segmentation/acc",
            "segmentation/ece",
            "segmentation/ece_macro",
            "segmentation/confident_error_rate",
            "segmentation/entropy_auroc",
        ):
            self.assertAlmostEqual(a.values[key], b.values[key], places=9, msg=key)


if __name__ == "__main__":
    unittest.main()
