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

from test.evaluation.metrics.detection.component_fixtures import box
from test.evaluation.metrics.detection.component_fixtures import empty_state
from test.evaluation.metrics.detection.component_fixtures import sample
import unittest

from perception_eval.evaluation.metrics.detection.components import COMPONENT_CLASSES
from perception_eval.evaluation.metrics.detection.components import get_component_class
from perception_eval.evaluation.metrics.detection.confusion_matrix import ConfusionMatrix
from perception_eval.evaluation.metrics.detection.state import DetectionState


def _det_state(pred_xy_label, gt_xy_label, scores=None, match_cost="center") -> DetectionState:
    scores = [0.9] * len(pred_xy_label) if scores is None else scores
    return DetectionState(
        samples=[
            sample(
                [box(x, y) for x, y, _ in pred_xy_label],
                scores,
                [lbl for _, _, lbl in pred_xy_label],
                [box(x, y) for x, y, _ in gt_xy_label],
                [lbl for _, _, lbl in gt_xy_label],
            )
        ],
        class_names=("car", "truck"),
        match_cost=match_cost,
    )


class TestConfusionMatrix(unittest.TestCase):
    def test_counts_matched_label_pair(self):
        # A car prediction (label 0) sits on a truck GT (label 1) -> confusion[truck, car].
        out = ConfusionMatrix(match_threshold=2.0).evaluate(_det_state([(10.0, 0.0, 0)], [(10.0, 0.0, 1)]))
        self.assertEqual(out["confusion_truck__car"], 1.0)
        self.assertEqual(out["confusion_truck__truck"], 0.0)
        self.assertEqual(out["confusion_car__car"], 0.0)
        self.assertEqual(out["confusion_car__truck"], 0.0)
        self.assertEqual(len(out), 4)

    def test_drops_unmatched(self):
        out = ConfusionMatrix(match_threshold=2.0).evaluate(_det_state([(100.0, 0.0, 0)], [(10.0, 0.0, 0)]))
        self.assertTrue(all(v == 0.0 for v in out.values()))

    def test_min_score_drops_low_score_predictions(self):
        state = _det_state([(10.0, 0.0, 0)], [(10.0, 0.0, 1)], scores=[0.05])
        out = ConfusionMatrix(min_score=0.1).evaluate(state)
        self.assertEqual(out["confusion_truck__car"], 0.0)
        kept = ConfusionMatrix(min_score=0.0).evaluate(state)
        self.assertEqual(kept["confusion_truck__car"], 1.0)

    def test_empty_inputs(self):
        out = ConfusionMatrix().evaluate(empty_state(("car", "truck")))
        self.assertEqual(len(out), 4)
        self.assertTrue(all(v == 0.0 for v in out.values()))

    def test_requires_center_cost_and_class_names(self):
        with self.assertRaises(ValueError):
            ConfusionMatrix().evaluate(_det_state([(10.0, 0.0, 0)], [(10.0, 0.0, 1)], match_cost="corner"))
        with self.assertRaises(ValueError):
            ConfusionMatrix().evaluate(DetectionState(samples=[], class_names=None))
        with self.assertRaises(ValueError):
            ConfusionMatrix(min_score=1.5)
        self.assertIs(ConfusionMatrix.needs_ttc, False)


class TestComponentRegistry(unittest.TestCase):
    def test_map_free_tokens_resolve_and_declare_no_ttc(self):
        for token in (
            "corner_error",
            "heading_flip",
            "nearest_surface_error",
            "calibration",
            "confident_error",
            "confusion_matrix",
        ):
            cls = get_component_class(token)
            self.assertIs(cls, COMPONENT_CLASSES[token])
            self.assertIs(cls.needs_ttc, False)
            self.assertTrue(callable(getattr(cls(), "evaluate")))
        self.assertIn("critical_fp_fn", COMPONENT_CLASSES)
        with self.assertRaises(ValueError):
            get_component_class("unknown")


if __name__ == "__main__":
    unittest.main()
