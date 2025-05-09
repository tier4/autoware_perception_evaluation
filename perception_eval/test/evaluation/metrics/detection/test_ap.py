# Copyright 2025 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from cmath import isclose
import math
from test.util.dummy_object import make_dummy_data
from test.util.object_diff import DiffTranslation
from test.util.object_diff import DiffYaw
from typing import List
from typing import Tuple
import unittest

import numpy as np
from perception_eval.common import DynamicObject
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_nuscene_object_results_by_label
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.metrics.detection.ap import Ap
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from perception_eval.evaluation.metrics.metrics_score_config import MetricsScoreConfig
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.object_result_matching import get_nuscene_object_results
from perception_eval.util.debug import get_objects_with_difference


class AnswerAP:
    """Answer class to compare AP result.

    Attributes:
        self.ap (float)
        self.tp_list (List[float])
        self.fp_list (List[float])
        self.precision_list (List[float])
        self.recall_list (List[float])
        self.max_precision_list (List[float])
        self.max_precision_recall_list (List[float])
    """

    def __init__(
        self,
        ap: float,
        tp_list: List[float],
        fp_list: List[float],
        precision_list: List[float],
        recall_list: List[float],
    ) -> None:
        """[summary]
        Args:
            ap (float)
            tp_list (List[float])
            fp_list (List[float])
            precision_list (List[float])
            recall_list (List[float])

        """
        assert len(tp_list) == len(
            fp_list
        ), f"length of TP/FP list must be same, but got {len(tp_list)} and {len(fp_list)}"
        assert len(precision_list) == len(
            recall_list
        ), f"length of precision/recall list must be same, but got {len(precision_list)} and {len(recall_list)}"

        self.ap: float = ap
        self.tp_list: List[float] = tp_list
        self.fp_list: List[float] = fp_list
        self.precision_list: List[float] = precision_list
        self.recall_list: List[float] = recall_list

    @classmethod
    def from_ap(cls, ap: Ap) -> AnswerAP:
        """[summary]
        Generate AnswerAP class from AP.

        Args:
            ap (AP)

        Returns:
            AnswerAP
        """
        precision_list, recall_list = ap.get_precision_recall()

        return AnswerAP(
            ap.ap,
            ap.tp_list,
            ap.fp_list,
            precision_list,
            recall_list,
        )

    def __eq__(self, other: AnswerAP) -> bool:
        return (
            isclose(self.ap, other.ap)
            and np.allclose(self.tp_list, other.tp_list)
            and np.allclose(self.fp_list, other.fp_list)
            and np.allclose(self.precision_list, other.precision_list)
            and np.allclose(self.recall_list, other.recall_list)
        )

    def __str__(self) -> str:
        str_: str = "\n("
        str_ += f"ap: {self.ap}, "
        str_ += f"tp_list: {self.tp_list}, "
        str_ += f"fp_list: {self.fp_list}, "
        str_ += f"precision_list: {self.precision_list}, "
        str_ += f"recall_list: {self.recall_list}, "
        str_ += ")"
        return str_


class TestAp(unittest.TestCase):
    def setUp(self):
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()
        self.evaluation_task: EvaluationTask = EvaluationTask.DETECTION
        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]

    def _evaluate_ap_aph_for_label(
        self,
        estimated_objects: List[DynamicObject],
        ground_truth_objects: List[DynamicObject],
        label: AutowareLabel,
        matching_mode: MatchingMode,
        matching_threshold: float,
    ) -> Tuple[Ap, Ap]:
        metrics_config = MetricsScoreConfig(
            evaluation_task="detection",
            target_labels=self.target_labels,
            center_distance_thresholds=[matching_threshold],
            center_distance_bev_thresholds=[matching_threshold],
            iou_2d_thresholds=[matching_threshold],
            iou_3d_thresholds=[matching_threshold],
        )
        object_results = get_nuscene_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=estimated_objects,
            ground_truth_objects=ground_truth_objects,
            metrics_config=metrics_config,
        )
        object_results_by_label = divide_nuscene_object_results_by_label(object_results, self.target_labels)
        num_gt_dict = divide_objects_to_num(ground_truth_objects, self.target_labels)

        result = object_results_by_label[label][(matching_mode, matching_threshold)]
        num_gt = num_gt_dict[label]

        ap = Ap(
            tp_metrics=TPMetricsAp(),
            object_results=result,
            num_ground_truth=num_gt,
            target_labels=label,
            matching_mode=matching_mode,
            matching_threshold=matching_threshold,
        )
        aph = Ap(
            tp_metrics=TPMetricsAph(),
            object_results=result,
            num_ground_truth=num_gt,
            target_labels=label,
            matching_mode=matching_mode,
            matching_threshold=matching_threshold,
        )
        return ap, aph

    # Test center distance
    def test_ap_center_distance_translation_difference(self):
        """[summary]
        Test AP and APH with center distance matching for translation difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_distance, check if ap and aph are almost correct.
        """
        # patterns: (diff_trans, ans_ap, ans_aph)
        patterns: List[Tuple[DiffTranslation, AnswerAP, AnswerAP]] = [
            # Given no diff_trans, ap and aph is 1.0.
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
            ),
            # Given 1.0 diff_distance for one axis, ap and aph are equal to 0.0
            # since both are over the metrics threshold.
            (
                DiffTranslation((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
        ]
        for n, (diff_trans, ans_ap, ans_aph) in enumerate(patterns):
            with self.subTest(f"Test AP and APH with center distance matching for translation difference: {n + 1}"):
                est_objs = get_objects_with_difference(self.dummy_estimated_objects, diff_trans.diff_estimated, 0.0)
                gt_objs = get_objects_with_difference(
                    self.dummy_ground_truth_objects, diff_trans.diff_ground_truth, 0.0
                )
                ap, aph = self._evaluate_ap_aph_for_label(
                    est_objs, gt_objs, AutowareLabel.CAR, MatchingMode.CENTERDISTANCE, 0.5
                )
                out_ap: AnswerAP = AnswerAP.from_ap(ap)
                out_aph: AnswerAP = AnswerAP.from_ap(aph)

                self.assertEqual(out_ap, ans_ap, f"out_ap = {str(out_ap)}, ans_ap = {str(ans_ap)}")
                self.assertEqual(out_aph, ans_aph, f"out_aph = {str(out_aph)}, ans_aph = {str(ans_aph)}")

    def test_ap_center_distance_yaw_difference(self):
        """[summary]
        Test AP and APH with center distance matching for yaw difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_yaw, check if ap and aph are almost correct.
        """
        # patterns: (diff_yaw, ans_ap, ans_aph)
        patterns: List[Tuple[DiffYaw, AnswerAP, AnswerAP]] = [
            # Given no diff_yaw, ap and aph is 1.0.
            (
                DiffYaw(0.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
            ),
            # Given vertical diff_yaw
            # since precision and recall of aph is 0.5 times those of ap.
            (
                DiffYaw(math.pi / 2.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.43621399176954734,
                    [0.5, 0.5],
                    [0.0, 1.0],
                    [1.0, 0.3333333333333333],
                    [0.5, 0.5],
                ),
            ),
            # Given vertical diff_yaw, aph is 0.5**2 times ap
            # since precision and recall of aph is 0.5 times those of ap.
            (
                DiffYaw(-math.pi / 2.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.43621399176954734,
                    [0.5, 0.5],
                    [0.0, 1.0],
                    [1.0, 0.3333333333333333],
                    [0.5, 0.5],
                ),
            ),
            # Given opposite direction, aph is 0.0.
            (
                DiffYaw(math.pi, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
            (
                DiffYaw(-math.pi, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
            # Given diff_yaw is pi/4, aph is 0.75**2 times ap
            (
                DiffYaw(math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.715167548500882,
                    [0.75, 0.75],
                    [0.0, 1.0],
                    [1.0, 0.42857142857142855],
                    [0.75, 0.75],
                ),
            ),
            (
                DiffYaw(-math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.715167548500882,
                    [0.75, 0.75],
                    [0.0, 1.0],
                    [1.0, 0.42857142857142855],
                    [0.75, 0.75],
                ),
            ),
            # Given diff_yaw is 3*pi/4, aph is 0.25**2 times ap
            (
                DiffYaw(3 * math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.16666666666666666,
                    [0.25000000000000033, 0.25000000000000033],
                    [0.0, 1.0],
                    [1.0, 0.2000000000000002],
                    [0.25000000000000033, 0.25000000000000033],
                ),
            ),
            (
                DiffYaw(-3 * math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.1567901234567901,
                    [0.25, 0.25],
                    [0.0, 1.0],
                    [1.0, 0.2],
                    [0.25, 0.25],
                ),
            ),
        ]

        for n, (diff_yaw, ans_ap, ans_aph) in enumerate(patterns):
            with self.subTest(f"Test AP and APH with center distance matching for yaw difference: {n + 1}"):
                est_objs = get_objects_with_difference(
                    self.dummy_estimated_objects, (0.0, 0.0, 0.0), diff_yaw.diff_estimated
                )
                gt_objs = get_objects_with_difference(
                    self.dummy_ground_truth_objects, (0.0, 0.0, 0.0), diff_yaw.diff_ground_truth
                )
                ap, aph = self._evaluate_ap_aph_for_label(
                    est_objs, gt_objs, AutowareLabel.CAR, MatchingMode.CENTERDISTANCE, 0.5
                )
                out_ap: AnswerAP = AnswerAP.from_ap(ap)
                out_aph: AnswerAP = AnswerAP.from_ap(aph)

                self.assertEqual(out_ap, ans_ap, f"out_ap = {str(out_ap)}, ans_ap = {str(ans_ap)}")
                self.assertEqual(out_aph, ans_aph, f"out_aph = {str(out_aph)}, ans_aph = {str(ans_aph)}")

    def test_ap_center_distance_random_objects(self):
        """[summary]
        Test AP and APH with center distance matching for random objects.

        test objects:
            dummy_estimated_objects (List[DynamicObject])
            dummy_ground_truth_objects (List[DynamicObject])

        test target_labels:
            MOTORBIKE

        test patterns:
            Check if ap and aph are almost correct.
        """
        # ap and aph is nan since no MOTORBIKE in the estimated_objects
        ans_ap = 0.0
        ans_aph = 0.0

        ap, aph = self._evaluate_ap_aph_for_label(
            self.dummy_estimated_objects,
            self.dummy_ground_truth_objects,
            AutowareLabel.MOTORBIKE,
            MatchingMode.CENTERDISTANCE,
            0.5,
        )

        self.assertAlmostEqual(ap.ap, ans_ap)
        self.assertAlmostEqual(aph.ap, ans_aph)

    # Test BEV center distance
    def test_ap_center_distance_bev_translation_difference(self):
        """[summary]
        Test AP and APH with center distance matching for translation difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_distance, check if ap and aph are almost correct.
        """
        # patterns: (diff_trans, ans_ap, ans_aph)
        patterns: List[Tuple[DiffTranslation, AnswerAP, AnswerAP]] = [
            # Given no diff_trans, ap and aph is 1.0.
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
            ),
            # Given 1.0 diff_distance for one axis, ap and aph are equal to 0.0
            # since both are over the metrics threshold.
            (
                DiffTranslation((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
        ]
        for n, (diff_trans, ans_ap, ans_aph) in enumerate(patterns):
            with self.subTest(f"Test AP and APH with center distance matching for translation difference: {n + 1}"):
                est_objs = get_objects_with_difference(self.dummy_estimated_objects, diff_trans.diff_estimated, 0.0)
                gt_objs = get_objects_with_difference(
                    self.dummy_ground_truth_objects, diff_trans.diff_ground_truth, 0.0
                )
                ap, aph = self._evaluate_ap_aph_for_label(
                    est_objs, gt_objs, AutowareLabel.CAR, MatchingMode.CENTERDISTANCEBEV, 0.5
                )
                out_ap: AnswerAP = AnswerAP.from_ap(ap)
                out_aph: AnswerAP = AnswerAP.from_ap(aph)

                self.assertEqual(out_ap, ans_ap, f"out_ap = {str(out_ap)}, ans_ap = {str(ans_ap)}")
                self.assertEqual(out_aph, ans_aph, f"out_aph = {str(out_aph)}, ans_aph = {str(ans_aph)}")

    def test_ap_center_distance_bev_yaw_difference(self):
        """[summary]
        Test AP and APH with center distance matching for yaw difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_yaw, check if ap and aph are almost correct.
        """
        # patterns: (diff_yaw, ans_ap, ans_aph)
        patterns: List[Tuple[DiffYaw, AnswerAP, AnswerAP]] = [
            # Given no diff_yaw, ap and aph is 1.0.
            (
                DiffYaw(0.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
            ),
            # Given vertical diff_yaw
            # since precision and recall of aph is 0.5 times those of ap.
            (
                DiffYaw(math.pi / 2.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.43621399176954734,
                    [0.5, 0.5],
                    [0.0, 1.0],
                    [1.0, 0.3333333333333333],
                    [0.5, 0.5],
                ),
            ),
            # Given vertical diff_yaw
            # since precision and recall of aph is 0.5 times those of ap.
            (
                DiffYaw(-math.pi / 2.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.43621399176954734,
                    [0.5, 0.5],
                    [0.0, 1.0],
                    [1.0, 0.3333333333333333],
                    [0.5, 0.5],
                ),
            ),
            # Given opposite direction, aph is 0.0.
            (
                DiffYaw(math.pi, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
            (
                DiffYaw(-math.pi, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
            # Given diff_yaw is pi/4, aph is 0.75**2 times ap
            (
                DiffYaw(math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.715167548500882,
                    [0.75, 0.75],
                    [0.0, 1.0],
                    [1.0, 0.42857142857142855],
                    [0.75, 0.75],
                ),
            ),
            (
                DiffYaw(-math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.715167548500882,
                    [0.75, 0.75],
                    [0.0, 1.0],
                    [1.0, 0.42857142857142855],
                    [0.75, 0.75],
                ),
            ),
            # Given diff_yaw is 3*pi/4, aph is 0.25**2 times ap
            (
                DiffYaw(3 * math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.16666666666666666,
                    [0.25000000000000033, 0.25000000000000033],
                    [0.0, 1.0],
                    [1.0, 0.2000000000000002],
                    [0.25000000000000033, 0.25000000000000033],
                ),
            ),
            (
                DiffYaw(-3 * math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.1567901234567901,
                    [0.25, 0.25],
                    [0.0, 1.0],
                    [1.0, 0.2],
                    [0.25, 0.25],
                ),
            ),
        ]

        for n, (diff_yaw, ans_ap, ans_aph) in enumerate(patterns):
            with self.subTest(f"Test AP and APH with center distance matching for yaw difference: {n + 1}"):
                est_objs = get_objects_with_difference(
                    self.dummy_estimated_objects, (0.0, 0.0, 0.0), diff_yaw.diff_estimated
                )
                gt_objs = get_objects_with_difference(
                    self.dummy_ground_truth_objects, (0.0, 0.0, 0.0), diff_yaw.diff_ground_truth
                )
                ap, aph = self._evaluate_ap_aph_for_label(
                    est_objs, gt_objs, AutowareLabel.CAR, MatchingMode.CENTERDISTANCEBEV, 0.5
                )
                out_ap: AnswerAP = AnswerAP.from_ap(ap)
                out_aph: AnswerAP = AnswerAP.from_ap(aph)

                self.assertEqual(out_ap, ans_ap, f"out_ap = {str(out_ap)}, ans_ap = {str(ans_ap)}")
                self.assertEqual(out_aph, ans_aph, f"out_aph = {str(out_aph)}, ans_aph = {str(ans_aph)}")

    def test_ap_center_distance_bev_random_objects(self):
        """[summary]
        Test AP and APH with center distance matching for random objects.

        test objects:
            dummy_estimated_objects (List[DynamicObject])
            dummy_ground_truth_objects (List[DynamicObject])

        test target_labels:
            MOTORBIKE

        test patterns:
            Check if ap and aph are almost correct.
        """
        # ap and aph is nan since no MOTORBIKE in the estimated_objects
        ans_ap = 0.0
        ans_aph = 0.0

        ap, aph = self._evaluate_ap_aph_for_label(
            self.dummy_estimated_objects,
            self.dummy_ground_truth_objects,
            AutowareLabel.MOTORBIKE,
            MatchingMode.CENTERDISTANCEBEV,
            0.5,
        )

        self.assertAlmostEqual(ap.ap, ans_ap)
        self.assertAlmostEqual(aph.ap, ans_aph)

    # Test IoU 2D
    def test_ap_iou_2d_translation_difference(self):
        """[summary]
        Test AP and APH with iou 2d matching for translation difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_distance, check if ap and aph are almost correct.
        """
        # patterns: (diff_distance, ans_ap, ans_aph)
        patterns: List[Tuple[DiffTranslation, AnswerAP, AnswerAP]] = [
            # Given no diff_distance, IOU 2D is 0.44444
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
            ),
            # Given 1.0 diff_distance for one axis, ap and aph are equal to 0.0
            # since iou_bev is under the threshold.
            (
                DiffTranslation((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
        ]
        for n, (diff_trans, ans_ap, ans_aph) in enumerate(patterns):
            with self.subTest(f"Test AP and APH with iou 2d matching for translation difference: {n + 1}"):
                est_objs = get_objects_with_difference(self.dummy_estimated_objects, diff_trans.diff_estimated, 0.0)
                gt_objs = get_objects_with_difference(
                    self.dummy_ground_truth_objects, diff_trans.diff_ground_truth, 0.0
                )

                ap, aph = self._evaluate_ap_aph_for_label(est_objs, gt_objs, AutowareLabel.CAR, MatchingMode.IOU2D, 0.4)

                out_ap: AnswerAP = AnswerAP.from_ap(ap)
                out_aph: AnswerAP = AnswerAP.from_ap(aph)

                self.assertEqual(out_ap, ans_ap, f"out_ap = {str(out_ap)}, ans_ap = {str(ans_ap)}")
                self.assertEqual(out_aph, ans_aph, f"out_aph = {str(out_aph)}, ans_aph = {str(ans_aph)}")

    def test_ap_iou_2d_yaw_difference(self):
        """[summary]
        Test AP and APH with iou 2d matching for yaw difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_yaw, check if ap and aph are almost correct.
        """
        # patterns: (diff_yaw, ans_ap, ans_aph)
        patterns: List[Tuple[DiffYaw, AnswerAP, AnswerAP]] = [
            # Given no diff_yaw, ap and aph is 1.0.
            (
                DiffYaw(0.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
            ),
            # Given vertical diff_yaw
            # since precision and recall of aph is 0.5 times those of ap.
            (
                DiffYaw(math.pi / 2.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.43621399176954734,
                    [0.5, 0.5],
                    [0.0, 1.0],
                    [1.0, 0.3333333333333333],
                    [0.5, 0.5],
                ),
            ),
            # Given vertical diff_yaw
            # since precision and recall of aph is 0.5 times those of ap.
            (
                DiffYaw(-math.pi / 2.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.43621399176954734,
                    [0.5, 0.5],
                    [0.0, 1.0],
                    [1.0, 0.3333333333333333],
                    [0.5, 0.5],
                ),
            ),
            # Given opposite direction, aph is 0.0.
            (
                DiffYaw(math.pi, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
            (
                DiffYaw(-math.pi, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
            # Given diff_yaw is pi/4, aph is 0.75**2 times ap
            (
                DiffYaw(math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.715167548500882,
                    [0.75, 0.75],
                    [0.0, 1.0],
                    [1.0, 0.42857142857142855],
                    [0.75, 0.75],
                ),
            ),
            (
                DiffYaw(-math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.715167548500882,
                    [0.75, 0.75],
                    [0.0, 1.0],
                    [1.0, 0.42857142857142855],
                    [0.75, 0.75],
                ),
            ),
            # Given diff_yaw is 3*pi/4, aph is 0.25**2 times ap
            (
                DiffYaw(3 * math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.16666666666666666,
                    [0.25000000000000033, 0.25000000000000033],
                    [0.0, 1.0],
                    [1.0, 0.2000000000000002],
                    [0.25000000000000033, 0.25000000000000033],
                ),
            ),
            (
                DiffYaw(-3 * math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.1567901234567901,
                    [0.25, 0.25],
                    [0.0, 1.0],
                    [1.0, 0.2],
                    [0.25, 0.25],
                ),
            ),
        ]

        for n, (diff_yaw, ans_ap, ans_aph) in enumerate(patterns):
            with self.subTest(f"Test AP and APH with iou 2d matching for yaw difference: {n + 1}"):
                est_objs = get_objects_with_difference(
                    self.dummy_estimated_objects, (0.0, 0.0, 0.0), diff_yaw.diff_estimated
                )
                gt_objs = get_objects_with_difference(
                    self.dummy_ground_truth_objects, (0.0, 0.0, 0.0), diff_yaw.diff_ground_truth
                )

                ap, aph = self._evaluate_ap_aph_for_label(est_objs, gt_objs, AutowareLabel.CAR, MatchingMode.IOU2D, 0.4)

                out_ap: AnswerAP = AnswerAP.from_ap(ap)
                out_aph: AnswerAP = AnswerAP.from_ap(aph)
                self.assertEqual(out_ap, ans_ap, f"out_ap = {str(out_ap)}, ans_ap = {str(ans_ap)}")
                self.assertEqual(out_aph, ans_aph, f"out_aph = {str(out_aph)}, ans_aph = {str(ans_aph)}")

    def test_ap_iou_2d_random_objects(self):
        """[summary]
        Test AP and APH with iou 2d matching for random objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Check if ap and aph are almost correct.
        """
        # iou_bev is 0.4444444444444444
        # which is over the threshold (0.4)
        ans_ap = 0.9938271604938275
        ans_aph = 0.9938271604938275
        ap, aph = self._evaluate_ap_aph_for_label(
            self.dummy_estimated_objects,
            self.dummy_ground_truth_objects,
            AutowareLabel.CAR,
            MatchingMode.IOU2D,
            0.4,
        )

        self.assertAlmostEqual(ap.ap, ans_ap)
        self.assertAlmostEqual(aph.ap, ans_aph)

    # Test IOU 3d
    def test_ap_iou_3d_translation_difference(self):
        """[summary]
        Test AP and APH with iou 3d matching for translation difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_distance, check if ap and aph are almost correct.
        """
        # patterns: (diff_distance, ans_ap, ans_aph)
        patterns: List[Tuple[DiffTranslation, AnswerAP, AnswerAP]] = [
            # Given no diff_distance, IOU 3D is 0.806
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
            ),
            # Given 1.0 diff_distance for one axis, ap and aph are equal to 0.0
            # since iou_bev is under the threshold.
            (
                DiffTranslation((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
        ]
        for n, (diff_trans, ans_ap, ans_aph) in enumerate(patterns):
            with self.subTest(f"Test AP and APH with iou 3d matching for translation difference: {n + 1}"):
                est_objs = get_objects_with_difference(self.dummy_estimated_objects, diff_trans.diff_estimated, 0.0)
                gt_objs = get_objects_with_difference(
                    self.dummy_ground_truth_objects, diff_trans.diff_ground_truth, 0.0
                )

                ap, aph = self._evaluate_ap_aph_for_label(est_objs, gt_objs, AutowareLabel.CAR, MatchingMode.IOU3D, 0.2)

                out_ap: AnswerAP = AnswerAP.from_ap(ap)
                out_aph: AnswerAP = AnswerAP.from_ap(aph)

                self.assertEqual(out_ap, ans_ap, f"out_ap = {str(out_ap)}, ans_ap = {str(ans_ap)}")
                self.assertEqual(out_aph, ans_aph, f"out_aph = {str(out_aph)}, ans_aph = {str(ans_aph)}")

    def test_ap_iou_3d_yaw_difference(self):
        """[summary]
        Test AP and APH with iou 3d matching for yaw difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_yaw, check if ap and aph are almost correct.
        """
        # patterns: (diff_yaw, ans_ap, ans_aph)
        patterns: List[Tuple[DiffYaw, AnswerAP, AnswerAP]] = [
            # Given no diff_yaw, ap and aph is 1.0.
            (
                DiffYaw(0.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
            ),
            # Given vertical diff_yaw
            (
                DiffYaw(math.pi / 2.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.43621399176954734,
                    [0.5, 0.5],
                    [0.0, 1.0],
                    [1.0, 0.3333333333333333],
                    [0.5, 0.5],
                ),
            ),
            # Given vertical diff_yaw
            (
                DiffYaw(-math.pi / 2.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.43621399176954734,
                    [0.5, 0.5],
                    [0.0, 1.0],
                    [1.0, 0.3333333333333333],
                    [0.5, 0.5],
                ),
            ),
            # Given opposite direction, aph is 0.0.
            (
                DiffYaw(math.pi, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
            (
                DiffYaw(-math.pi, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.0,
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ),
            ),
            # Given diff_yaw is pi/4
            (
                DiffYaw(math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.715167548500882,
                    [0.75, 0.75],
                    [0.0, 1.0],
                    [1.0, 0.42857142857142855],
                    [0.75, 0.75],
                ),
            ),
            (
                DiffYaw(-math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.715167548500882,
                    [0.75, 0.75],
                    [0.0, 1.0],
                    [1.0, 0.42857142857142855],
                    [0.75, 0.75],
                ),
            ),
            # Given diff_yaw is 3*pi/4
            (
                DiffYaw(3 * math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.16666666666666666,
                    [0.25000000000000033, 0.25000000000000033],
                    [0.0, 1.0],
                    [1.0, 0.2000000000000002],
                    [0.25000000000000033, 0.25000000000000033],
                ),
            ),
            (
                DiffYaw(-3 * math.pi / 4.0, 0.0),
                AnswerAP(
                    0.9938271604938275,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                ),
                AnswerAP(
                    0.1567901234567901,
                    [0.25, 0.25],
                    [0.0, 1.0],
                    [1.0, 0.2],
                    [0.25, 0.25],
                ),
            ),
        ]

        for n, (diff_yaw, ans_ap, ans_aph) in enumerate(patterns):
            with self.subTest(f"Test AP and APH with iou 2d matching for yaw difference: {n + 1}"):
                est_objs = get_objects_with_difference(
                    self.dummy_estimated_objects, (0.0, 0.0, 0.0), diff_yaw.diff_estimated
                )
                gt_objs = get_objects_with_difference(
                    self.dummy_ground_truth_objects, (0.0, 0.0, 0.0), diff_yaw.diff_ground_truth
                )

                ap, aph = self._evaluate_ap_aph_for_label(est_objs, gt_objs, AutowareLabel.CAR, MatchingMode.IOU3D, 0.2)

                out_ap: AnswerAP = AnswerAP.from_ap(ap)
                out_aph: AnswerAP = AnswerAP.from_ap(aph)

                self.assertEqual(out_ap, ans_ap, f"out_ap = {str(out_ap)}, ans_ap = {str(ans_ap)}")
                self.assertEqual(out_aph, ans_aph, f"out_aph = {str(out_aph)}, ans_aph = {str(ans_aph)}")

    def test_ap_iou_3d_random_objects(self):
        """[summary]
        Test AP and APH with iou 2d matching for random objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Check if ap and aph are almost correct.
        """
        ans_ap = 0.0
        ans_aph = 0.0
        ap, aph = self._evaluate_ap_aph_for_label(
            self.dummy_estimated_objects,
            self.dummy_ground_truth_objects,
            AutowareLabel.CAR,
            MatchingMode.IOU3D,
            0.4,
        )

        self.assertAlmostEqual(ap.ap, ans_ap)
        self.assertAlmostEqual(aph.ap, ans_aph)

    # Test plance distance
    # def test_ap_plane_distance_translation_difference(self):
    #     """[summary]
    #     Test AP and APH with plane distance matching for translation difference.

    #     test objects:
    #         dummy_ground_truth_objects (List[DynamicObject])
    #         dummy_ground_truth_objects with diff_distance (List[DynamicObject])

    #     test patterns:
    #         Given diff_distance, check if ap and aph are almost correct.
    #     """
    #     # patterns: (diff_distance, ans_ap, ans_aph)
    #     patterns: List[Tuple[float, float, float]] = [
    #         # Given no diff_distance, ap and aph is 1.0.
    #         (0.0, 1.0, 1.0),
    #         # Given 1.0 diff_distance for one axis, ap and aph are equal to 0.0
    #         # since both are over the metrics threshold.
    #         (1.0, 0.0, 0.0),
    #     ]
    #     for diff_distance, ans_ap, ans_aph in patterns:
    #         with self.subTest("Test AP and APH with plane distance matching for translation difference."):
    #             diff_distance_dummy_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
    #                 ground_truth_objects=self.dummy_ground_truth_objects,
    #                 diff_distance=(diff_distance, 0.0, 0.0),
    #                 diff_yaw=0,
    #             )

    #             object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
    #                 evaluation_task=self.evaluation_task,
    #                 estimated_objects=diff_distance_dummy_ground_truth_objects,
    #                 ground_truth_objects=self.dummy_ground_truth_objects,
    #             )
    #             num_ground_truth: int = len(self.dummy_ground_truth_objects)
    #             ap: Ap = Ap(
    #                 tp_metrics=TPMetricsAp(),
    #                 object_results=object_results,
    #                 num_ground_truth=num_ground_truth,
    #                 target_labels=self.target_labels,
    #                 matching_mode=MatchingMode.PLANEDISTANCE,
    #                 matching_threshold=0.1,
    #             )
    #             aph: Ap = Ap(
    #                 tp_metrics=TPMetricsAph(),
    #                 object_results=object_results,
    #                 num_ground_truth=num_ground_truth,
    #                 target_labels=self.target_labels,
    #                 matching_mode=MatchingMode.PLANEDISTANCE,
    #                 matching_threshold=1.0,
    #             )

    #             self.assertAlmostEqual(ap.ap, ans_ap)
    #             self.assertAlmostEqual(aph.ap, ans_aph)

    # def test_ap_plane_distance_yaw_difference(self):
    #     """[summary]
    #     Test AP and APH with plane distance matching for yaw difference.

    #     test objects:
    #         dummy_ground_truth_objects (List[DynamicObject])
    #         dummy_ground_truth_objects with diff_distance (List[DynamicObject])

    #     test patterns:
    #         Given diff_yaw, check if ap and aph are almost correct.
    #     """
    #     # patterns: (diff_yaw, ans_ap, ans_aph)
    #     patterns: List[Tuple[float, float, float]] = [
    #         # Given no diff_yaw, ap and aph is 1.0.
    #         (0.0, 1.0, 1.0),
    #         # Given vertical diff_yaw
    #         # since precision and recall of aph is 0.5 times those of ap.
    #         (math.pi / 2.0, 1.0, 0.25),
    #         (-math.pi / 2.0, 1.0, 0.25),
    #         # Given opposite direction, aph is 0.0.
    #         (math.pi, 1.0, 0.0),
    #         (-math.pi, 1.0, 0.0),
    #         # Given diff_yaw is pi/4, aph is 0.75**2 times ap
    #         (math.pi / 4, 1.0, 0.5625),
    #         (-math.pi / 4, 1.0, 0.5625),
    #         # Given diff_yaw is 3*pi/4, aph is 0.25**2 times ap
    #         (3 * math.pi / 4, 1.0, 0.0625),
    #         (-3 * math.pi / 4, 1.0, 0.0625),
    #     ]

    #     for diff_yaw, ans_ap, ans_aph in patterns:
    #         with self.subTest("Test AP and APH with plane distance matching for yaw difference."):
    #             diff_yaw_dummy_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
    #                 ground_truth_objects=self.dummy_ground_truth_objects,
    #                 diff_distance=(0.0, 0.0, 0.0),
    #                 diff_yaw=diff_yaw,
    #             )

    #             object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
    #                 evaluation_task=self.evaluation_task,
    #                 estimated_objects=diff_yaw_dummy_ground_truth_objects,
    #                 ground_truth_objects=self.dummy_ground_truth_objects,
    #             )
    #             num_ground_truth: int = len(self.dummy_ground_truth_objects)
    #             ap: Ap = Ap(
    #                 tp_metrics=TPMetricsAp(),
    #                 object_results=object_results,
    #                 num_ground_truth=num_ground_truth,
    #                 target_labels=self.target_labels,
    #                 matching_mode=MatchingMode.PLANEDISTANCE,
    #                 matching_threshold=1.5,
    #             )
    #             aph: Ap = Ap(
    #                 tp_metrics=TPMetricsAph(),
    #                 object_results=object_results,
    #                 num_ground_truth=num_ground_truth,
    #                 target_labels=self.target_labels,
    #                 matching_mode=MatchingMode.PLANEDISTANCE,
    #                 matching_threshold=1.5,
    #             )

    #             self.assertAlmostEqual(ap.ap, ans_ap)
    #             self.assertAlmostEqual(aph.ap, ans_aph)

    # def test_ap_plane_distance_random_objects(self):
    #     """[summary]
    #     Test AP and APH with plane distance matching for random objects.

    #     test objects:
    #         dummy_ground_truth_objects (List[DynamicObject])
    #         dummy_ground_truth_objects with diff_distance (List[DynamicObject])

    #     test patterns:
    #         Check if ap and aph are almost correct.
    #     """
    #     # dummy_estimated_objects[0] (CAR) and dummy_ground_truth_objects[0] (CAR):
    #     #   pr_corner_points(left, right) = [(0.25, 0.25, 1.0), (0.25, 1.75, 1.0)]
    #     #   gt_corner_points(left, right) = [(0.5, 0.5, 1.0), (0.5, 1.5, 1.0)]
    #     #   plane_distance = 0.3535533905932738
    #     ans_ap_tp: float = 1.0
    #     ans_ap_tn: float = 0.0
    #     ans_aph_tp: float = 1.0
    #     ans_aph_tn: float = 0.0

    #     object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
    #         evaluation_task=self.evaluation_task,
    #         estimated_objects=self.dummy_estimated_objects,
    #         ground_truth_objects=self.dummy_ground_truth_objects,
    #     )

    #     num_ground_truth: int = len(self.dummy_ground_truth_objects)

    #     ap_tp: Ap = Ap(
    #         tp_metrics=TPMetricsAp(),
    #         object_results=object_results,
    #         num_ground_truth=num_ground_truth,
    #         target_labels=self.target_labels,
    #         matching_mode=MatchingMode.PLANEDISTANCE,
    #         matching_threshold=1.0,
    #     )
    #     aph_tp: Ap = Ap(
    #         tp_metrics=TPMetricsAph(),
    #         object_results=object_results,
    #         num_ground_truth=num_ground_truth,
    #         target_labels=self.target_labels,
    #         matching_mode=MatchingMode.PLANEDISTANCE,
    #         matching_threshold=1.0,
    #     )
    #     ap_tn: Ap = Ap(
    #         tp_metrics=TPMetricsAp(),
    #         object_results=object_results,
    #         num_ground_truth=num_ground_truth,
    #         target_labels=self.target_labels,
    #         matching_mode=MatchingMode.PLANEDISTANCE,
    #         matching_threshold=0.2,
    #     )
    #     aph_tn: Ap = Ap(
    #         tp_metrics=TPMetricsAph(),
    #         object_results=object_results,
    #         num_ground_truth=num_ground_truth,
    #         target_labels=self.target_labels,
    #         matching_mode=MatchingMode.PLANEDISTANCE,
    #         matching_threshold=0.2,
    #     )

    # print(f"[DEBUG #{n + 1}] out_ap = {out_ap}")
    # print(f"[DEBUG #{n + 1}] out_aph = {out_aph}")
    #     self.assertAlmostEqual(ap_tp.ap, ans_ap_tp)
    #     self.assertAlmostEqual(aph_tp.ap, ans_aph_tp)
    #     self.assertAlmostEqual(ap_tn.ap, ans_ap_tn)
    #     self.assertAlmostEqual(aph_tn.ap, ans_aph_tn)


if __name__ == "__main__":
    unittest.main()
