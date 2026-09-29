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
import unittest

import numpy as np
from perception_eval.evaluation.metrics.class_groups import resolve_class_groups
from perception_eval.evaluation.metrics.segmentation.confusion_matrix import confusion_counts
from perception_eval.evaluation.metrics.segmentation.confusion_matrix import ConfusionAccumulator
from perception_eval.evaluation.metrics.segmentation.confusion_matrix import ConfusionMatrix
from perception_eval.evaluation.metrics.segmentation.confusion_matrix import ConfusionState
from perception_eval.evaluation.metrics.segmentation.confusion_scores import Accuracy
from perception_eval.evaluation.metrics.segmentation.confusion_scores import IoU
from perception_eval.evaluation.metrics.segmentation.confusion_scores import PrecisionRecallF1
from perception_eval.evaluation.metrics.segmentation.suite import SegmentationMetricSuite

NAMES = ("car", "truck", "road", "sidewalk")
GROUPS = {"vehicle": ["car", "truck"], "flat": ["road", "sidewalk"]}


class TestConfusionCounts(unittest.TestCase):
    def test_counts_match_brute_force(self):
        rng = np.random.default_rng(1)
        target = rng.integers(0, 3, 500)
        pred = rng.integers(0, 3, 500)
        counts = confusion_counts(target, pred, 3)
        expected = np.zeros((3, 3), dtype=np.int64)
        for t, p in zip(target, pred):
            expected[t, p] += 1
        np.testing.assert_array_equal(counts, expected)
        self.assertEqual(confusion_counts(np.zeros(0, int), np.zeros(0, int), 3).shape, (3, 3))

    def test_accumulator_buckets(self):
        acc = ConfusionAccumulator(num_filters=1, num_ranges=2, num_classes=2)
        self.assertEqual(acc.confusion.shape, (2, 3, 2, 2))
        acc.add(1, 2, np.array([0, 1]), np.array([1, 1]))
        self.assertEqual(acc.matrix(1, 2).tolist(), [[0, 1], [0, 1]])
        self.assertEqual(int(acc.matrix(0, 0).sum()), 0)
        acc.reset()
        self.assertEqual(int(acc.confusion.sum()), 0)


class TestConfusionScores(unittest.TestCase):
    def setUp(self):
        # Class 0: TP=1, FN=1; class 1: TP=1, FP=1 (source fixture).
        self.state = ConfusionState(confusion=np.array([[1, 1], [0, 1]]), class_names=("car", "road"), num_classes=2)

    def test_cells(self):
        cells = ConfusionMatrix().evaluate(self.state)
        self.assertEqual(cells["confusion_car__road"], 1.0)
        self.assertEqual(cells["confusion_road__car"], 0.0)

    def test_iou(self):
        out = IoU().evaluate(self.state)
        self.assertAlmostEqual(out["iou_car"], 0.5)
        self.assertAlmostEqual(out["iou_road"], 0.5)
        self.assertAlmostEqual(out["mIoU"], 0.5)
        self.assertAlmostEqual(out["fwIoU"], 0.5)

    def test_accuracy(self):
        self.assertAlmostEqual(Accuracy().evaluate(self.state)["acc"], 2.0 / 3.0)
        self.assertTrue(math.isnan(Accuracy().evaluate(ConfusionState(np.zeros((2, 2), int), ("a", "b"), 2))["acc"]))

    def test_precision_recall_f1(self):
        out = PrecisionRecallF1().evaluate(self.state)
        self.assertAlmostEqual(out["recall_car"], 0.5)
        self.assertAlmostEqual(out["precision_car"], 1.0)
        self.assertAlmostEqual(out["f1_car"], 2.0 / 3.0)
        self.assertAlmostEqual(out["recall_road"], 1.0)
        self.assertAlmostEqual(out["precision_road"], 0.5)
        self.assertAlmostEqual(out["mRecall"], 0.75)

    def test_absent_class_is_skipped_and_macro_nan_without_support(self):
        state = ConfusionState(
            confusion=np.array([[3, 0, 0], [0, 0, 0], [1, 0, 0]]), class_names=("a", "b", "c"), num_classes=3
        )
        out = IoU().evaluate(state)
        self.assertNotIn("iou_b", out)
        self.assertAlmostEqual(out["iou_a"], 0.75)
        self.assertAlmostEqual(out["iou_c"], 0.0)
        empty = ConfusionState(np.zeros((2, 2), int), ("a", "b"), 2)
        self.assertTrue(math.isnan(IoU().evaluate(empty)["mIoU"]))
        self.assertTrue(math.isnan(IoU().evaluate(empty)["fwIoU"]))
        self.assertTrue(math.isnan(PrecisionRecallF1().evaluate(empty)["mF1"]))


class TestGroupedConfusion(unittest.TestCase):
    def test_intra_group_confusion_folds_to_grouped_diagonal(self):
        # car<->truck and road<->sidewalk swapped: raw all wrong, grouped all correct.
        cfg = config(["confusion_matrix", "iou"], class_names=NAMES, class_groups=GROUPS)
        suite = SegmentationMetricSuite(cfg)
        suite.update(make_frame([0, 1, 2, 3], target=[1, 0, 3, 2], pred=[0, 1, 2, 3], num_classes=4))
        report = suite.compute()
        self.assertEqual(report.values["segmentation/grouped/confusion_vehicle__vehicle"], 2.0)
        self.assertEqual(report.values["segmentation/grouped/confusion_vehicle__flat"], 0.0)
        self.assertEqual(report.values["segmentation/grouped/confusion_flat__flat"], 2.0)
        self.assertAlmostEqual(report.values["segmentation/grouped/mIoU"], 1.0)
        self.assertAlmostEqual(report.values["segmentation/iou_car"], 0.0)
        self.assertNotIn("segmentation/confusion_grouped_vehicle__grouped_vehicle", report.values)
        self.assertNotIn("segmentation/grouped/iou_car", report.values)
        lut, names = resolve_class_groups(NAMES, GROUPS)
        np.testing.assert_array_equal(report.confusion_matrix("grouped"), [[2, 0], [0, 2]])
        np.testing.assert_array_equal(report.confusion_matrix("raw"), suite.confusion.matrix(0, 0))

    def test_update_order_invariance(self):
        cfg = config(["confusion_matrix", "iou", "accuracy"], class_names=NAMES, class_groups=GROUPS)
        rng = np.random.default_rng(2)
        frames = []
        for index in range(4):
            n = 50
            target = rng.integers(0, 4, n)
            pred = target.copy()
            flip = rng.random(n) < 0.3
            pred[flip] = rng.integers(0, 4, int(flip.sum()))
            frames.append(make_frame(rng.uniform(-40, 40, n), target, pred, num_classes=4, frame_name=str(index)))
        suite_a = SegmentationMetricSuite(cfg)
        suite_b = SegmentationMetricSuite(cfg)
        for frame in frames:
            suite_a.update(frame)
        for frame in reversed(frames):
            suite_b.update(frame)
        a, b = suite_a.compute(), suite_b.compute()
        np.testing.assert_array_equal(a.confusion, b.confusion)
        for key, value in a.values.items():
            if math.isnan(value):
                self.assertTrue(math.isnan(b.values[key]))
            else:
                self.assertAlmostEqual(value, b.values[key], places=9, msg=key)


if __name__ == "__main__":
    unittest.main()
