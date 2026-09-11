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
import unittest

import numpy as np
from perception_eval.evaluation.metrics.segmentation.entropy_auroc import binned_auroc
from perception_eval.evaluation.metrics.segmentation.entropy_auroc import exact_auroc
from perception_eval.evaluation.metrics.segmentation.entropy_auroc import UncertaintyUsefulness


class TestUncertaintyUsefulness(unittest.TestCase):
    def test_perfect_separation(self):
        # Wrong points get high entropy (uniform), correct points get low entropy.
        target = [0, 0, 1, 1]
        pred = [0, 0, 0, 0]
        scores = np.array([[0.99, 0.01], [0.99, 0.01], [0.5, 0.5], [0.5, 0.5]])
        metric = UncertaintyUsefulness()
        metric.update(make_view(make_frame([0, 1, 2, 3], target, pred, scores=scores)))
        out = metric.compute()
        self.assertAlmostEqual(out["entropy_auroc"], 1.0)
        self.assertAlmostEqual(out["mean_entropy_wrong"], 1.0)
        self.assertLess(out["mean_entropy_correct"], 0.1)

    def test_all_correct_or_all_wrong_is_nan(self):
        metric = UncertaintyUsefulness()
        metric.update(make_view(make_frame([0, 1], [0, 1], [0, 1])))
        out = metric.compute()
        self.assertTrue(math.isnan(out["entropy_auroc"]))
        self.assertTrue(math.isnan(out["mean_entropy_wrong"]))
        self.assertAlmostEqual(out["mean_entropy_correct"], 1.0)

    def test_binned_matches_exact_rank_auroc(self):
        rng = np.random.default_rng(5)
        entropy = rng.beta(2.0, 5.0, 50_000)
        wrong = rng.random(50_000) < 0.2
        entropy[wrong] += 0.15
        entropy = np.clip(entropy, 0.0, 1.0)
        exact = exact_auroc(entropy, wrong)
        num_bins = 8192
        bins = np.minimum((entropy * num_bins).astype(np.int64), num_bins - 1)
        wrong_hist = np.bincount(bins[wrong], minlength=num_bins).astype(np.float64)
        correct_hist = np.bincount(bins[~wrong], minlength=num_bins).astype(np.float64)
        self.assertLess(abs(binned_auroc(wrong_hist, correct_hist) - exact), 1e-3)
        # Small random case within one bin width.
        small = rng.uniform(0, 1, 20)
        flags = rng.random(20) < 0.5
        if flags.any() and (~flags).any():
            bins = np.minimum((small * num_bins).astype(np.int64), num_bins - 1)
            approx = binned_auroc(
                np.bincount(bins[flags], minlength=num_bins), np.bincount(bins[~flags], minlength=num_bins)
            )
            self.assertAlmostEqual(approx, exact_auroc(small, flags), delta=1.0 / num_bins * 20)
        # Exact equality when scores sit on distinct bins.
        centers = (np.arange(6) + 0.5) / num_bins
        pos = np.array([False, False, False, True, True, True])
        bins = (centers * num_bins).astype(np.int64)
        self.assertAlmostEqual(
            binned_auroc(np.bincount(bins[pos], minlength=num_bins), np.bincount(bins[~pos], minlength=num_bins)),
            exact_auroc(centers, pos),
        )

    def test_validation(self):
        with self.assertRaises(ValueError):
            UncertaintyUsefulness(num_bins=1)
        self.assertTrue(math.isnan(binned_auroc(np.zeros(4), np.ones(4))))


if __name__ == "__main__":
    unittest.main()
