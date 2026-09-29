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
from perception_eval.evaluation.metrics.segmentation.confident_error import ConfidentErrorRate


class TestConfidentErrorRate(unittest.TestCase):
    def test_flags_confident_mistakes(self):
        target = [0, 0, 0, 0]
        pred = [1, 1, 0, 0]
        metric = ConfidentErrorRate(entropy_threshold=0.3)
        metric.update(make_view(make_frame([0, 1, 2, 3], target, pred, scores=scores_for(pred, 0.99))))
        out = metric.compute()
        self.assertAlmostEqual(out["confident_error_rate"], 1.0)
        self.assertAlmostEqual(out["confident_error_count"], 2.0)

    def test_uncertain_mistakes_and_threshold_boundary(self):
        target = [0, 0]
        pred = [1, 1]
        metric = ConfidentErrorRate(entropy_threshold=0.3)
        metric.update(make_view(make_frame([0, 1], target, pred)))  # uniform -> entropy 1.0
        self.assertAlmostEqual(metric.compute()["confident_error_rate"], 0.0)
        # An error exactly at the threshold is NOT confident (strict `<`, as in the source).
        p = 0.9
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)
        boundary = ConfidentErrorRate(entropy_threshold=float(entropy))
        boundary.update(make_view(make_frame([0], [0], [1], scores=np.array([[1 - p, p]]))))
        self.assertAlmostEqual(boundary.compute()["confident_error_rate"], 0.0)
        above = ConfidentErrorRate(entropy_threshold=float(entropy) + 1e-9)
        above.update(make_view(make_frame([0], [0], [1], scores=np.array([[1 - p, p]]))))
        self.assertAlmostEqual(above.compute()["confident_error_rate"], 1.0)

    def test_no_errors_is_nan_and_counts_add(self):
        metric = ConfidentErrorRate()
        metric.update(make_view(make_frame([0, 1], [0, 1], [0, 1])))
        out = metric.compute()
        self.assertTrue(math.isnan(out["confident_error_rate"]))
        self.assertEqual(out["confident_error_count"], 0.0)
        metric.update(make_view(make_frame([0], [0], [1], scores=scores_for([1], 0.99))))
        metric.update(make_view(make_frame([0], [0], [1], scores=scores_for([1], 0.99))))
        self.assertEqual(metric.compute()["confident_error_count"], 2.0)
        with self.assertRaises(ValueError):
            ConfidentErrorRate(entropy_threshold=1.5)


if __name__ == "__main__":
    unittest.main()
