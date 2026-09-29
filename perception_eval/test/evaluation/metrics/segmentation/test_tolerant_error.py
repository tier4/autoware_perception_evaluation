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
from perception_eval.evaluation.metrics.segmentation.tolerant_error import NeighbourhoodTolerantErrorRate


class TestNeighbourhoodTolerantErrorRate(unittest.TestCase):
    def test_forgives_correct_neighbour(self):
        frame = make_frame([0, 1], target=[0, 0], pred=[1, 0])
        seen = NeighbourhoodTolerantErrorRate(radius=1.5)
        seen.update(make_view(frame))
        out = seen.compute()
        self.assertAlmostEqual(out["tolerant_error_rate"], 0.0)
        self.assertAlmostEqual(out["tolerant_error_count"], 0.0)
        unseen = NeighbourhoodTolerantErrorRate(radius=0.5)
        unseen.update(make_view(frame))
        out = unseen.compute()
        self.assertAlmostEqual(out["tolerant_error_rate"], 0.5)
        self.assertAlmostEqual(out["tolerant_error_count"], 1.0)

    def test_radius_zero_is_strict_error_rate(self):
        rng = np.random.default_rng(4)
        n = 300
        target = rng.integers(0, 2, n)
        pred = target.copy()
        flip = rng.random(n) < 0.3
        pred[flip] = 1 - pred[flip]
        frame = make_frame([], target, pred, coords=rng.uniform(-10, 10, (n, 3)))
        metric = NeighbourhoodTolerantErrorRate(radius=0.0)
        metric.update(make_view(frame))
        out = metric.compute()
        self.assertEqual(out["tolerant_error_count"], float(np.sum(pred != target)))
        self.assertAlmostEqual(out["tolerant_error_rate"], float(np.mean(pred != target)))

    def test_emits_global_and_per_class(self):
        frame = make_frame([0, 10, 20, 30], target=[0, 0, 1, 1], pred=[1, 0, 1, 1])
        metric = NeighbourhoodTolerantErrorRate(radius=0.5)
        metric.update(make_view(frame))
        out = metric.compute()
        self.assertAlmostEqual(out["tolerant_error_rate"], 0.25)
        self.assertAlmostEqual(out["tolerant_error_rate_road"], 0.5)
        self.assertAlmostEqual(out["tolerant_error_rate_obstacle"], 0.0)
        self.assertAlmostEqual(out["tolerant_error_count_road"], 1.0)

    def test_empty_and_validation(self):
        self.assertTrue(math.isnan(NeighbourhoodTolerantErrorRate().compute()["tolerant_error_rate"]))
        with self.assertRaises(ValueError):
            NeighbourhoodTolerantErrorRate(radius=-0.1)


if __name__ == "__main__":
    unittest.main()
