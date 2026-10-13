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
from test.evaluation.metrics.detection.component_fixtures import no_frames_state
from test.evaluation.metrics.detection.component_fixtures import sample
from test.evaluation.metrics.detection.component_fixtures import tp_fp_state
import unittest

from perception_eval.evaluation.metrics.detection.confident_error import ConfidentErrorRate
from perception_eval.evaluation.metrics.detection.state import DetectionState


class TestConfidentErrorRate(unittest.TestCase):
    def test_flags_high_score_false_positive(self):
        out = ConfidentErrorRate(score_threshold=0.5).evaluate(tp_fp_state())
        self.assertAlmostEqual(out["confident_error_rate"], 1.0)  # the one FP (0.8) is confident
        self.assertAlmostEqual(out["confident_error_count"], 1.0)
        self.assertAlmostEqual(out["confident_errors_per_frame"], 1.0)
        lifted = ConfidentErrorRate(score_threshold=0.9).evaluate(tp_fp_state())
        self.assertAlmostEqual(lifted["confident_error_rate"], 0.0)  # 0.8 < 0.9
        self.assertAlmostEqual(lifted["confident_error_count"], 0.0)

    def test_min_score_floor_drops_low_fps(self):
        state = DetectionState(
            samples=[sample([box(0.0), box(100.0)], [0.9, 0.05], [0, 0], [box(0.0)], [0])],
            class_names=("car",),
        )
        out = ConfidentErrorRate(min_score=0.1).evaluate(state)
        self.assertTrue(math.isnan(out["confident_error_rate"]))  # no FP above the floor
        self.assertEqual(out["confident_error_count"], 0.0)

    def test_no_false_positive_is_nan_rate(self):
        out = ConfidentErrorRate().evaluate(empty_state())
        self.assertTrue(math.isnan(out["confident_error_rate"]))
        self.assertEqual(out["confident_error_count"], 0.0)
        self.assertEqual(out["confident_errors_per_frame"], 0.0)

    def test_no_frames_per_frame_is_nan(self):
        out = ConfidentErrorRate().evaluate(no_frames_state())
        self.assertTrue(math.isnan(out["confident_errors_per_frame"]))

    def test_validation(self):
        with self.assertRaises(ValueError):
            ConfidentErrorRate(score_threshold=0.5, min_score=0.6)
        with self.assertRaises(ValueError):
            ConfidentErrorRate(score_threshold=1.5)
        self.assertIs(ConfidentErrorRate.needs_ttc, False)


if __name__ == "__main__":
    unittest.main()
