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
#
# Ported from tier4/autoware-ml (autoware_ml/tests/metrics/test_criticality_metrics.py) at fcf86419.

from dataclasses import replace
import math
from math import inf
import unittest

import numpy as np
from perception_eval.evaluation.metrics.detection.collision_weighted_map import CollisionWeightedMeanAP
from perception_eval.evaluation.metrics.detection.critical_fp_fn import CriticalFPFN
from perception_eval.evaluation.metrics.detection.criticality import greedy_match
from perception_eval.evaluation.metrics.detection.criticality import greedy_match_thresholds
from perception_eval.evaluation.metrics.detection.criticality import weighted_average_precision
from perception_eval.evaluation.metrics.detection.state import DetectionSample
from perception_eval.evaluation.metrics.detection.state import DetectionState


def _box(x: float):
    return [x, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]


def _sample(preds, pred_scores, pred_labels, gts, gt_labels, pred_ttc=None, gt_ttc=None) -> DetectionSample:
    """A frame with per-box TTC; default TTC = 1.0 s (all boxes critical/weighted)."""
    n_pred, n_gt = len(preds), len(gts)
    pred_ttc = [1.0] * n_pred if pred_ttc is None else pred_ttc
    gt_ttc = [1.0] * n_gt if gt_ttc is None else gt_ttc
    return DetectionSample(
        pred_boxes=np.array(preds, dtype=np.float64).reshape(-1, 9),
        pred_scores=np.array(pred_scores, dtype=np.float64),
        pred_labels=np.array(pred_labels, dtype=np.int64),
        gt_boxes=np.array(gts, dtype=np.float64).reshape(-1, 9),
        gt_labels=np.array(gt_labels, dtype=np.int64),
        pred_ttc=np.array(pred_ttc, dtype=np.float64),
        gt_ttc=np.array(gt_ttc, dtype=np.float64),
    )


class TestHelpers(unittest.TestCase):
    def test_greedy_match_basic(self):
        tp, matched = greedy_match(np.array([[0.0, 0.0]]), np.array([[0.5, 0.0]]), np.array([0.9]), 2.0)
        self.assertEqual(tp.tolist(), [True])
        self.assertEqual(matched.tolist(), [0])

    def test_greedy_match_is_score_ordered_and_shares_costs_across_thresholds(self):
        gt = np.array([[0.0, 0.0]])
        preds = np.array([[0.4, 0.0], [0.1, 0.0]])  # the nearer prediction has the lower score
        result = greedy_match_thresholds(gt, preds, np.array([0.9, 0.8]), (0.2, 2.0))
        self.assertEqual(result[2.0][0].tolist(), [True, False])  # high score claims first
        self.assertEqual(result[0.2][0].tolist(), [False, True])  # 0.4 exceeds 0.2, so the other one matches
        empty_tp, empty_gt = greedy_match(np.zeros((0, 2)), preds, np.array([0.9, 0.8]), 2.0)
        self.assertEqual(empty_tp.tolist(), [False, False])
        self.assertEqual(empty_gt.tolist(), [-1, -1])

    def test_weighted_ap_all_true_positive_is_one(self):
        ap = weighted_average_precision(np.array([1.0]), np.array([True]), np.array([0.9]), total_gt_weight=1.0)
        self.assertAlmostEqual(ap, 1.0)
        self.assertTrue(math.isnan(weighted_average_precision(np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0), 0.0)))
        self.assertEqual(weighted_average_precision(np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0), 1.0), 0.0)

    def test_unit_weights_reproduce_unweighted_ap(self):
        from perception_eval.evaluation.metrics.detection.matching import interpolated_ap
        from perception_eval.evaluation.metrics.detection.matching import precision_recall

        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        is_tp = np.array([True, False, True, True, False])
        total_gt = 4
        cumulative_tp = np.cumsum(is_tp.astype(float))
        cumulative_fp = np.cumsum((~is_tp).astype(float))
        precision, recall = precision_recall(cumulative_tp, cumulative_fp, total_gt)
        expected = interpolated_ap(precision, recall, total_gt, scores.shape[0])
        actual = weighted_average_precision(np.ones(5), is_tp, scores, float(total_gt))
        self.assertAlmostEqual(actual, expected, places=12)


class TestCriticalFPFN(unittest.TestCase):
    def test_counts_phantoms_and_misses(self):
        state = DetectionState(
            samples=[
                _sample([_box(10.0)], [0.9], [0], [_box(10.0)], [0]),
                _sample([_box(10.0), _box(30.0)], [0.9, 0.9], [0, 0], [_box(10.0)], [0]),  # +1 FP
                _sample([], [], [], [_box(10.0)], [0]),  # +1 FN
            ],
            class_names=("car",),
        )
        out = CriticalFPFN(confidences=(0.5,), match_threshold=2.0).evaluate(state)
        self.assertAlmostEqual(out["critical_fp_0p5m"], 1 / 3)
        self.assertAlmostEqual(out["critical_fn_0p5m"], 1 / 3)
        self.assertAlmostEqual(out["critical_fp_car_0p5m"], 1 / 3)
        self.assertAlmostEqual(out["critical_fn_car_0p5m"], 1 / 3)

    def test_excludes_unreachable(self):
        state = DetectionState(
            samples=[
                _sample([_box(30.0)], [0.9], [0], [], [], pred_ttc=[inf]),
                _sample([], [], [], [_box(30.0)], [0], gt_ttc=[inf]),
            ],
            class_names=("car",),
        )
        out = CriticalFPFN(confidences=(0.5,)).evaluate(state)
        self.assertAlmostEqual(out["critical_fp_0p5m"], 0.0)
        self.assertAlmostEqual(out["critical_fn_0p5m"], 0.0)

    def test_excludes_uncovered_frames(self):
        covered_fp = _sample([_box(10.0), _box(30.0)], [0.9, 0.9], [0, 0], [_box(10.0)], [0])
        uncovered = replace(
            _sample([_box(30.0)], [0.9], [0], [_box(50.0)], [0], pred_ttc=[inf], gt_ttc=[inf]),
            ttc_covered=False,
        )
        state = DetectionState(samples=[covered_fp, uncovered], class_names=("car",))
        out = CriticalFPFN(confidences=(0.5,), match_threshold=2.0).evaluate(state)
        self.assertAlmostEqual(out["critical_fp_0p5m"], 1.0)  # denominator is 1, not 2
        weighted = CollisionWeightedMeanAP(thresholds=(2.0,), decay=0.5).evaluate(state)
        self.assertGreater(weighted["cw_mAP"], 0.0)

    def test_no_coverage_is_nan(self):
        frame = replace(_sample([_box(10.0)], [0.9], [0], [_box(10.0)], [0]), ttc_covered=False)
        state = DetectionState(samples=[frame], class_names=("car",))
        out = CriticalFPFN(confidences=(0.5,)).evaluate(state)
        self.assertTrue(math.isnan(out["critical_fp_0p5m"]))
        self.assertTrue(math.isnan(out["critical_fn_car_0p5m"]))

    def test_confidence_gate(self):
        state = DetectionState(samples=[_sample([_box(30.0)], [0.4], [0], [], [])], class_names=("car",))
        out = CriticalFPFN(confidences=(0.5,)).evaluate(state)
        self.assertAlmostEqual(out["critical_fp_0p5m"], 0.0)

    def test_requires_ttc_and_validates_params(self):
        sample = DetectionSample(np.array([_box(1.0)]), [0.9], [0], np.array([_box(1.0)]), [0])
        with self.assertRaisesRegex(ValueError, "TTC"):
            CriticalFPFN().evaluate(DetectionState(samples=[sample], class_names=("car",)))
        with self.assertRaises(ValueError):
            CriticalFPFN(confidences=())
        self.assertTrue(CriticalFPFN.needs_ttc)
        keys = CriticalFPFN().evaluate(DetectionState(samples=[], class_names=("car",)))
        self.assertEqual(
            set(keys),
            {
                "critical_fp_0p3m",
                "critical_fn_0p3m",
                "critical_fp_0p5m",
                "critical_fn_0p5m",
                "critical_fp_car_0p3m",
                "critical_fn_car_0p3m",
                "critical_fp_car_0p5m",
                "critical_fn_car_0p5m",
            },
        )


class TestCollisionWeightedMeanAP(unittest.TestCase):
    def test_perfect_detection(self):
        state = DetectionState(samples=[_sample([_box(10.0)], [0.9], [0], [_box(10.0)], [0])], class_names=("car",))
        out = CollisionWeightedMeanAP(thresholds=(2.0,), decay=0.5).evaluate(state)
        self.assertAlmostEqual(out["cw_mAP"], 1.0)
        self.assertAlmostEqual(out["cw_mAP_car"], 1.0)

    def test_unreachable_gt_has_no_weight(self):
        state = DetectionState(
            samples=[_sample([_box(10.0)], [0.9], [0], [_box(10.0)], [0], gt_ttc=[inf], pred_ttc=[inf])],
            class_names=("car",),
        )
        out = CollisionWeightedMeanAP(thresholds=(2.0,), decay=0.5).evaluate(state)
        self.assertTrue(math.isnan(out["cw_mAP_car"]))

    def test_far_phantom_weighs_less_than_imminent_one(self):
        # Same detections; the FP with a long TTC costs less precision than one about to collide.
        near_phantom = _sample(
            [_box(10.0), _box(30.0)], [0.9, 0.8], [0, 0], [_box(10.0)], [0], pred_ttc=[1.0, 0.5], gt_ttc=[1.0]
        )
        far_phantom = _sample(
            [_box(10.0), _box(30.0)], [0.9, 0.8], [0, 0], [_box(10.0)], [0], pred_ttc=[1.0, 3.5], gt_ttc=[1.0]
        )
        metric = CollisionWeightedMeanAP(thresholds=(2.0,), decay=0.5)
        near = metric.evaluate(DetectionState(samples=[near_phantom], class_names=("car",)))["cw_mAP"]
        far = metric.evaluate(DetectionState(samples=[far_phantom], class_names=("car",)))["cw_mAP"]
        self.assertGreater(far, near)

    def test_validation(self):
        with self.assertRaises(ValueError):
            CollisionWeightedMeanAP(decay=-1.0)
        with self.assertRaises(ValueError):
            CollisionWeightedMeanAP(thresholds=())
        sample = DetectionSample(np.array([_box(1.0)]), [0.9], [0], np.array([_box(1.0)]), [0])
        with self.assertRaisesRegex(ValueError, "TTC"):
            CollisionWeightedMeanAP().evaluate(DetectionState(samples=[sample], class_names=("car",)))
        self.assertTrue(CollisionWeightedMeanAP.needs_ttc)


if __name__ == "__main__":
    unittest.main()
