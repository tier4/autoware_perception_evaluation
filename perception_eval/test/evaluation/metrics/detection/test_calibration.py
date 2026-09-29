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
from test.evaluation.metrics.detection.component_fixtures import tp_fp_state
import unittest

import numpy as np
from perception_eval.evaluation.metrics.detection.calibration import CalibrationError
from perception_eval.evaluation.metrics.detection.calibration import expected_calibration_error
from perception_eval.evaluation.metrics.detection.state import DetectionState


class TestCalibrationError(unittest.TestCase):
    def test_matches_hand_computed_ece(self):
        # TP score 0.9 (correct) -> bin 13, gap |1-0.9|=0.1. FP score 0.8 (wrong) -> bin 12,
        # gap |0-0.8|=0.8. Each bin weight 1/2 -> ECE = 0.5*0.1 + 0.5*0.8 = 0.45.
        out = CalibrationError(num_bins=15).evaluate(tp_fp_state())
        self.assertAlmostEqual(out["ece"], 0.45)
        self.assertAlmostEqual(out["ece_macro"], 0.45)

    def test_function_edge_cases(self):
        self.assertTrue(math.isnan(expected_calibration_error(np.zeros(0), np.zeros(0), 15)))
        with self.assertRaises(ValueError):
            expected_calibration_error(np.array([1.2]), np.array([1.0]), 15)
        with self.assertRaises(ValueError):
            expected_calibration_error(np.array([-0.1]), np.array([0.0]), 15)
        # Score exactly 1.0 lands in the last bin.
        self.assertAlmostEqual(expected_calibration_error(np.array([1.0]), np.array([1.0]), 15), 0.0)

    def test_invalid_scores_raise(self):
        state = DetectionState(samples=[sample([box()], [1.5], [0], [box()], [0])], class_names=("car",))
        with self.assertRaises(ValueError):
            CalibrationError().evaluate(state)

    def test_no_predictions_is_nan(self):
        out = CalibrationError().evaluate(empty_state())
        self.assertTrue(math.isnan(out["ece"]))
        self.assertTrue(math.isnan(out["ece_macro"]))

    def test_validation(self):
        with self.assertRaises(ValueError):
            CalibrationError(num_bins=0)
        self.assertIs(CalibrationError.needs_ttc, False)


if __name__ == "__main__":
    unittest.main()
