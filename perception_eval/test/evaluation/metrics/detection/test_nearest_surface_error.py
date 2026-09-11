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

from perception_eval.evaluation.metrics.detection.nearest_surface_error import NearestSurfaceError
from perception_eval.evaluation.metrics.detection.state import DetectionState


class TestNearestSurfaceError(unittest.TestCase):
    def test_present(self):
        out = NearestSurfaceError().evaluate(yaw_state(0.1))
        self.assertIn("nsurf_absmax_car", out)
        self.assertGreaterEqual(out["nsurf_absmax_car"], 0.0)
        for key in ("nsurf_mean_car", "nsurf_low_car", "nsurf_high_car", "mnsurf_high", "mnsurf_absmax"):
            self.assertIn(key, out)

    def test_sign_of_a_far_prediction(self):
        # Predicting the near face 1 m too far away is a positive (late-braking) error.
        state = DetectionState(samples=[sample([box(11.0)], [0.9], [0], [box(10.0)], [0])], class_names=("car",))
        out = NearestSurfaceError().evaluate(state)
        self.assertAlmostEqual(out["nsurf_mean_car"], 1.0)
        self.assertAlmostEqual(out["nsurf_low_car"], 1.0)
        self.assertAlmostEqual(out["nsurf_high_car"], 1.0)
        self.assertAlmostEqual(out["nsurf_absmax_car"], 1.0)
        self.assertAlmostEqual(out["mnsurf_high"], 1.0)
        self.assertAlmostEqual(out["mnsurf_absmax"], 1.0)

    def test_class_without_tp(self):
        out = NearestSurfaceError().evaluate(empty_state())
        for key in (
            "nsurf_mean_car",
            "nsurf_low_car",
            "nsurf_high_car",
            "nsurf_absmax_car",
            "mnsurf_high",
            "mnsurf_absmax",
        ):
            self.assertTrue(math.isnan(out[key]))

    def test_validation(self):
        with self.assertRaises(ValueError):
            NearestSurfaceError(low_percentile=95.0, high_percentile=5.0)
        with self.assertRaises(ValueError):
            NearestSurfaceError(low_percentile=-1.0)
        self.assertIs(NearestSurfaceError.needs_ttc, False)


if __name__ == "__main__":
    unittest.main()
