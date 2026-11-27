import unittest
from types import SimpleNamespace

from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.tracking.hota import HOTA


def _res(gt, est, ok=True):
    gt_obj = None if gt is None else SimpleNamespace(uuid=gt)
    return SimpleNamespace(
        ground_truth_object=gt_obj,
        estimated_object=SimpleNamespace(uuid=est),
        is_result_correct=lambda mode, thr: ok,
    )


def _hota(frames, num_gt):
    return HOTA([[]] + frames, num_gt, [AutowareLabel.CAR], MatchingMode.CENTERDISTANCE, [1.0])


class TestHOTA(unittest.TestCase):
    def test_perfect(self):
        h = _hota([[_res("g", "e")], [_res("g", "e")]], 2)
        self.assertAlmostEqual(h.hota, 1.0)

    def test_id_switch(self):
        # gt g tracked by e1 then e2: AssA = 0.5, DetA = 1
        h = _hota([[_res("g", "e1")], [_res("g", "e2")]], 2)
        self.assertAlmostEqual(h.det_a, 1.0)
        self.assertAlmostEqual(h.ass_a, 0.5)
        self.assertAlmostEqual(h.hota, 0.5**0.5)

    def test_fp_and_fn(self):
        # 1 TP, 1 FP, 2 GT -> 1 FN: DetA = 1/3
        h = _hota([[_res("g", "e"), _res(None, "x", ok=False)]], 2)
        self.assertAlmostEqual(h.det_a, 1 / 3)


if __name__ == "__main__":
    unittest.main()
