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
from test.evaluation.metrics.detection.component_fixtures import box
from test.evaluation.metrics.detection.component_fixtures import empty_state
from test.evaluation.metrics.detection.component_fixtures import sample
from test.evaluation.metrics.detection.component_fixtures import yaw_state
import unittest

from perception_eval.evaluation.metrics.detection.corner_error import CornerError
from perception_eval.evaluation.metrics.detection.state import DetectionState


class TestCornerError(unittest.TestCase):
    def test_positive_under_yaw(self):
        out = CornerError().evaluate(yaw_state(0.1))
        self.assertGreater(out["corner_mean_car"], 0.0)
        self.assertGreaterEqual(out["corner_max_car"], out["corner_mean_car"])
        self.assertIn("corner_p95_car", out)
        self.assertAlmostEqual(out["mcorner_mean"], out["corner_mean_car"])
        self.assertAlmostEqual(out["mcorner_max"], out["corner_max_car"])

    def test_translation_is_exact(self):
        state = DetectionState(samples=[sample([box(11.0)], [0.9], [0], [box(10.0)], [0])], class_names=("car",))
        out = CornerError(percentiles=(50.0, 95.0)).evaluate(state)
        self.assertAlmostEqual(out["corner_mean_car"], 1.0)
        self.assertAlmostEqual(out["corner_p50_car"], 1.0)
        self.assertAlmostEqual(out["corner_p95_car"], 1.0)

    def test_corner_matching_mode_still_matches(self):
        out = CornerError().evaluate(yaw_state(0.05, match_cost="corner"))
        self.assertFalse(math.isnan(out["corner_mean_car"]))

    def test_class_without_tp_is_nan(self):
        out = CornerError().evaluate(empty_state(("car", "truck")))
        for name in ("car", "truck"):
            self.assertTrue(math.isnan(out[f"corner_mean_{name}"]))
            self.assertTrue(math.isnan(out[f"corner_max_{name}"]))
            self.assertTrue(math.isnan(out[f"corner_p95_{name}"]))
        self.assertTrue(math.isnan(out["mcorner_mean"]))
        self.assertTrue(math.isnan(out["mcorner_max"]))

    def test_validation(self):
        with self.assertRaises(ValueError):
            CornerError(percentiles=(101.0,))
        self.assertIs(CornerError.needs_ttc, False)


if __name__ == "__main__":
    unittest.main()
