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
from math import pi
from test.evaluation.metrics.detection.component_fixtures import empty_state
from test.evaluation.metrics.detection.component_fixtures import yaw_state
import unittest

from perception_eval.evaluation.metrics.detection.heading_flip import HeadingFlipRate


class TestHeadingFlipRate(unittest.TestCase):
    def test_no_flip_for_small_error(self):
        out = HeadingFlipRate().evaluate(yaw_state(0.1))
        self.assertEqual(out["flip_count_car"], 0.0)
        self.assertAlmostEqual(out["flip_rate_car"], 0.0)
        self.assertAlmostEqual(out["mflip_rate"], 0.0)

    def test_counts_a_reversal(self):
        out = HeadingFlipRate().evaluate(yaw_state(pi - 0.1))
        self.assertEqual(out["flip_count_car"], 1.0)
        self.assertAlmostEqual(out["flip_rate_car"], 1.0)
        self.assertAlmostEqual(out["mflip_rate"], 1.0)

    def test_threshold_is_configurable(self):
        out = HeadingFlipRate(flip_threshold=0.05).evaluate(yaw_state(0.1))
        self.assertEqual(out["flip_count_car"], 1.0)

    def test_class_without_tp(self):
        out = HeadingFlipRate().evaluate(empty_state())
        self.assertTrue(math.isnan(out["flip_rate_car"]))
        self.assertEqual(out["flip_count_car"], 0.0)
        self.assertTrue(math.isnan(out["mflip_rate"]))
        self.assertIs(HeadingFlipRate.needs_ttc, False)


if __name__ == "__main__":
    unittest.main()
