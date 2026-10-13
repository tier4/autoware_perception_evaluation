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
from test.evaluation.metrics.segmentation.helpers import make_frame
from test.evaluation.metrics.segmentation.helpers import make_view
from test.evaluation.metrics.segmentation.helpers import scores_for
import unittest

import numpy as np
from perception_eval.evaluation.metrics.segmentation.calibration import CalibrationError
from perception_eval.evaluation.metrics.segmentation.calibration import ece_from_bins


class TestCalibrationError(unittest.TestCase):
    def test_zero_when_confidence_matches_accuracy(self):
        target = [0, 0, 1, 1]
        metric = CalibrationError(num_bins=10)
        metric.update(make_view(make_frame([0, 1, 2, 3], target, target, scores=scores_for(target, 1.0))))
        out = metric.compute()
        self.assertAlmostEqual(out["ece"], 0.0)
        self.assertAlmostEqual(out["ece_macro"], 0.0)

    def test_penalizes_overconfident_mistakes(self):
        target = [0, 0, 1, 1]
        pred = [0, 0, 0, 0]
        metric = CalibrationError(num_bins=10)
        metric.update(make_view(make_frame([0, 1, 2, 3], target, pred, scores=scores_for(pred, 0.95))))
        out = metric.compute()
        # Every point in the 0.95 bin, accuracy 0.5 -> |0.5 - 0.95| = 0.45
        self.assertAlmostEqual(out["ece"], 0.45)
        self.assertGreater(out["ece"], 0.4)

    def test_hand_computed_bins_and_streaming(self):
        conf = np.array([0.95, 0.85, 0.65, 0.35])
        correct = np.array([1, 1, 0, 0])
        pred = [0, 0, 1, 1]
        target = [0, 0, 0, 0]  # points 2,3 predicted 1 are wrong
        scores = np.zeros((4, 2))
        scores[np.arange(4), pred] = conf
        scores[np.arange(4), 1 - np.asarray(pred)] = 1.0 - conf
        frame = make_frame([0, 1, 2, 3], target, pred, scores=scores)
        metric = CalibrationError(num_bins=15)
        metric.update(make_view(frame))
        out = metric.compute()
        bins = np.minimum((conf * 15).astype(int), 14)
        expected = sum(
            np.sum(bins == b) / 4.0 * abs(correct[bins == b].mean() - conf[bins == b].mean()) for b in np.unique(bins)
        )
        self.assertAlmostEqual(out["ece"], expected)
        # Macro over predicted classes: class 0 is perfectly calibrated-ish, class 1 fully wrong.
        ece_0 = abs(1.0 - 0.95) * 0.5 + abs(1.0 - 0.85) * 0.5
        ece_1 = abs(0.0 - 0.65) * 0.5 + abs(0.0 - 0.35) * 0.5
        self.assertAlmostEqual(out["ece_macro"], (ece_0 + ece_1) / 2)
        # Two half frames give the same as one frame.
        streamed = CalibrationError(num_bins=15)
        streamed.update(make_view(make_frame([0, 1], target[:2], pred[:2], scores=scores[:2])))
        streamed.update(make_view(make_frame([2, 3], target[2:], pred[2:], scores=scores[2:])))
        self.assertAlmostEqual(streamed.compute()["ece"], out["ece"])

    def test_empty_and_validation(self):
        self.assertTrue(math.isnan(CalibrationError().compute()["ece"]))
        self.assertTrue(math.isnan(ece_from_bins(np.zeros(3), np.zeros(3), np.zeros(3))))
        with self.assertRaises(ValueError):
            CalibrationError(num_bins=0)
        metric = CalibrationError()
        metric.update(make_view(make_frame([], [], [])))
        self.assertTrue(math.isnan(metric.compute()["ece_macro"]))


if __name__ == "__main__":
    unittest.main()
