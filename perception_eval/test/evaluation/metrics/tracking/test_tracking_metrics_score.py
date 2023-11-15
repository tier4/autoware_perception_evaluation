# Copyright 2022 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from test.evaluation.metrics.tracking.test_clear import AnswerCLEAR
from test.util.dummy_object import make_dummy_data
from test.util.object_diff import DiffTranslation
from test.util.object_diff import DiffYaw
from typing import Dict
from typing import List
from typing import Tuple
import unittest

from perception_eval.common import DynamicObject
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.matching.object_matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics.tracking.tracking_metrics_score import TrackingMetricsScore
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.object_result import get_object_results
from perception_eval.util.debug import get_objects_with_difference


class TestTrackingMetricsScore(unittest.TestCase):
    """The class to test TrackingMetricsScore."""

    def setUp(self) -> None:
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()

        self.evaluation_task: EvaluationTask = EvaluationTask.TRACKING
        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]
        self.max_x_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]
        self.max_y_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]

    def test_sum_clear(self):
        """[summary]
        Test summing up CLEAR scores. _sum_clear() returns total MOTA, MOTP and IDsw for same frames.

        test patterns:
            Check the summed up MOTA/MOTP and ID switch score with translated previous and current results.
            NOTE:
                - Estimated object is only matched with GT that has same label.
                - The estimations & GTs are following (number represents the index)
                    Estimation = 3
                        (0): CAR, (1): BICYCLE, (2): CAR
                    GT = 4
                        (0): CAR, (1): BICYCLE, (2): PEDESTRIAN, (3): MOTORBIKE
        """
        # patterns: (prev_diff_trans, cur_diff_trans, ans_mota, ans_motp, ans_id_switch)
        patterns: List[Tuple[DiffTranslation, DiffTranslation, float, float, int]] = [
            # (1)
            # -> previous   : TP=2.0((Est[0], GT[0]), (Est[1], GT[1])), FP=1.0(Est[2])
            #       MOTA=(1.0-1.0)/2+1.0/1+0.0/1+0.0/1=0.25, MOTP=(0.0/1.0+0.0/1.0)=0.0, IDsw=0
            # -> current    : TP=2.0((Est[0], GT[2]), (Est[0], GT[2])), FP=1.0(Est[2])
            #       MOTA=(1.0-1.0)/2+1.0/1+0.0/1+0.0/1=0.25, MOTP=(0.0/1.0+0.0/1.0)=0.0, IDsw=0
            # [TOTAL]
            #       MOTA=(0.25*4 + 0.25*4)/8=0.25, MOTP=(0.0*1+0.0*1)/2=0.0, IDsw=0
            (
                # prev: (trans est, trans gt)
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                # cur: (trans est, trans gt)
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                0.25,
                0.0,
                0,
            ),
            # (2)
            # -> previous   : TP=1.0(Est[1], GT[1]), FP=2.0(Est[0], Est[2])
            #       MOTA=(1.0-2.0)/2->0.0, MOTP=(0.0/1.0)=0.0, IDsw=0
            # -> current    : TP=2.0((Est[0], GT[2]), (Est[0], GT[2])), FP=1.0(Est[2])
            #       MOTA=(2.0-1.0)/4=0.25, MOTP=(0.0/1.0+0.0/1.0)=0.0, IDsw=0
            # [TOTAL]
            #       MOTA=(0.25*4+0.25*4)/8=0.25, MOTP=(0.0*2+0.0*2)/4, IDsw=0
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.5, 2.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                0.25,
                0.0,
                0,
            ),
            # (3)
            # -> previous   : TP=2.0((Est[0], GT[0]), (Est[1], Est[1])), FP=1.0(Est[2])
            #       MOTA=(2.0-1.0)/4=0.25, MOTP=(0.0/1.0+0.0/1.0)=0.0, IDsw=0
            # -> current    : TP=2.0((Est[0], Est[2]), (Est[0], Est[2])), FP=1.0(Est[2])
            #       MOTA=(2.0-1.0)/4=0.25, MOTP=(0.0/1.0+0.0/1.0)=0.0, IDsw=0
            # [TOTAL]
            #       MOTA=(0.25*4+0.25*4)/8=0.25, MOTP=(0.0*2+0.0*2)/4=0.25, IDsw=0
            (
                DiffTranslation((1.0, 0.0, 0.0), (0.2, 0, 0.0)),
                DiffTranslation((0.25, 0.0, 0.0), (0.0, 0.0, 0.0)),
                0.25,
                0.25,
                0,
            ),
        ]
        for n, (prev_diff_trans, cur_diff_trans, ans_mota, ans_motp, ans_id_switch) in enumerate(patterns):
            with self.subTest(f"Test sum CLEAR: {n + 1}"):
                prev_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_estimated_objects,
                    diff_distance=prev_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                # Previous ground truth objects
                prev_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=prev_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )
                # Filter previous objects
                prev_estimated_objects = filter_objects(
                    objects=prev_estimated_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                prev_ground_truth_objects = filter_objects(
                    objects=prev_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                # Previous object results
                prev_object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=prev_estimated_objects,
                    ground_truth_objects=prev_ground_truth_objects,
                )
                prev_object_results_dict = divide_objects(prev_object_results, self.target_labels)

                # Current estimated objects
                cur_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_estimated_objects,
                    diff_distance=cur_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                # Current ground truth objects
                cur_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=cur_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )
                # Filter current objects
                cur_estimated_objects = filter_objects(
                    objects=cur_estimated_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                cur_ground_truth_objects = filter_objects(
                    objects=cur_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                # Current object results
                cur_object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )
                cur_object_results_dict = divide_objects(cur_object_results, self.target_labels)

                object_results_dict = {}
                for label in self.target_labels:
                    object_results_dict[label] = [
                        prev_object_results_dict[label],
                        cur_object_results_dict[label],
                    ]

                num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
                    cur_ground_truth_objects, self.target_labels
                )

                tracking_score: TrackingMetricsScore = TrackingMetricsScore(
                    object_results_dict=object_results_dict,
                    num_ground_truth_dict=num_ground_truth_dict,
                    target_labels=self.target_labels,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[0.5, 0.5, 0.5, 0.5],
                )
                mota, motp, id_switch = tracking_score._sum_clear()
                self.assertAlmostEqual(mota, ans_mota)
                self.assertAlmostEqual(motp, ans_motp)
                self.assertEqual(id_switch, ans_id_switch)

    def test_center_distance_translation_difference(self):
        """[summary]
        Test TrackingMetricsScore with center distance matching, when each object result is translated by xy axis.

        Test patterns:
            Check the clear score for each target label with translated previous and current results.
        """
        # patterns: (prev_diff_difference, cur_diff_difference, ans_clears)
        patterns: List[Tuple[DiffTranslation, DiffTranslation, Tuple[AnswerCLEAR]]] = [
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                (
                    AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0.0, 0.0),
                    AnswerCLEAR(1, 1.0, 0.0, 0, 0.0, 1.0, 0.0),
                    AnswerCLEAR(1, 0.0, 0.0, 0, 0.0, 0.0, float("inf")),
                    AnswerCLEAR(1, 0.0, 0.0, 0, 0.0, 0.0, float("inf")),
                ),
            ),
        ]
        for n, (prev_diff_trans, cur_diff_trans, ans_clears) in enumerate(patterns):
            with self.subTest(f"Test tracking score with center distance: {n + 1}"):
                prev_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_estimated_objects,
                    diff_distance=prev_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                # Previous ground truth objects
                prev_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=prev_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )
                # Filter previous objects
                prev_estimated_objects = filter_objects(
                    objects=prev_estimated_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                prev_ground_truth_objects = filter_objects(
                    objects=prev_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                # Previous object results
                prev_object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=prev_estimated_objects,
                    ground_truth_objects=prev_ground_truth_objects,
                )
                prev_object_results_dict = divide_objects(
                    prev_object_results,
                    self.target_labels,
                )

                # Current estimated objects
                cur_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_estimated_objects,
                    diff_distance=cur_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                # Current ground truth objects
                cur_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=cur_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )
                # Filter current objects
                cur_estimated_objects = filter_objects(
                    objects=cur_estimated_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                cur_ground_truth_objects = filter_objects(
                    objects=cur_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                # Current object results
                cur_object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )
                cur_object_results_dict = divide_objects(
                    cur_object_results,
                    self.target_labels,
                )

                object_results_dict = {}
                for label in self.target_labels:
                    object_results_dict[label] = [
                        prev_object_results_dict[label],
                        cur_object_results_dict[label],
                    ]

                num_ground_truth_dict = divide_objects_to_num(
                    cur_ground_truth_objects,
                    self.target_labels,
                )

                tracking_score: TrackingMetricsScore = TrackingMetricsScore(
                    object_results_dict=object_results_dict,
                    num_ground_truth_dict=num_ground_truth_dict,
                    target_labels=self.target_labels,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[0.5, 0.5, 0.5, 0.5],
                )
                for clear_, ans_clear_ in zip(tracking_score.clears, ans_clears):
                    out_clear_: AnswerCLEAR = AnswerCLEAR.from_clear(clear_)
                    self.assertEqual(
                        out_clear_,
                        ans_clear_,
                        f"out_clear = {str(out_clear_)}, ans_clear = {str(ans_clear_)}",
                    )

    def test_center_distance_yaw_difference(self):
        """[summary]
        Test TrackingMetricsScore with center distance matching, when each object result is rotated by yaw angle.

        Test patterns:
            Check the clear score for each target label with rotated previous and current results around yaw angle.
        """
        # patterns: (prev_diff_yaw, cur_diff_yaw, ans_clears)
        patterns: List[Tuple[DiffYaw, DiffYaw, List[AnswerCLEAR]]] = [
            (
                DiffYaw(60.0, 0.0, deg2rad=True),
                DiffYaw(0.0, 45.0, deg2rad=True),
                (
                    AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0.0, 0.0),
                    AnswerCLEAR(1, 1.0, 0.0, 0, 0.0, 1.0, 0.0),
                    AnswerCLEAR(1, 0.0, 0.0, 0, 0.0, 0.0, float("inf")),
                    AnswerCLEAR(1, 0.0, 0.0, 0, 0.0, 0.0, float("inf")),
                ),
            ),
        ]
        for n, (prev_diff_yaw, cur_diff_yaw, ans_clears) in enumerate(patterns):
            with self.subTest(f"Test tracking score with center distance matching translated by yaw: {n + 1}"):
                prev_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_estimated_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=prev_diff_yaw.diff_estimated,
                )
                # Previous ground truth objects
                prev_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=prev_diff_yaw.diff_ground_truth,
                )
                # Filter previous objects
                prev_estimated_objects = filter_objects(
                    objects=prev_estimated_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                prev_ground_truth_objects = filter_objects(
                    objects=prev_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                # Previous object results
                prev_object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=prev_estimated_objects,
                    ground_truth_objects=prev_ground_truth_objects,
                )
                prev_object_results_dict = divide_objects(prev_object_results, self.target_labels)

                # Current estimated objects
                cur_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_estimated_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=cur_diff_yaw.diff_estimated,
                )
                # Current ground truth objects
                cur_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=cur_diff_yaw.diff_ground_truth,
                )
                # Filter current objects
                cur_estimated_objects = filter_objects(
                    objects=cur_estimated_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                cur_ground_truth_objects = filter_objects(
                    objects=cur_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                # Current object results
                cur_object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )
                cur_object_results_dict = divide_objects(cur_object_results, self.target_labels)

                object_results_dict = {}
                for label in self.target_labels:
                    object_results_dict[label] = [
                        prev_object_results_dict[label],
                        cur_object_results_dict[label],
                    ]

                num_ground_truth_dict = divide_objects_to_num(cur_ground_truth_objects, self.target_labels)

                tracking_score: TrackingMetricsScore = TrackingMetricsScore(
                    object_results_dict=object_results_dict,
                    num_ground_truth_dict=num_ground_truth_dict,
                    target_labels=self.target_labels,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[0.5, 0.5, 0.5, 0.5],
                )

                # Check scores for each target label
                for clear_, ans_clear_ in zip(tracking_score.clears, ans_clears):
                    out_clear_: AnswerCLEAR = AnswerCLEAR.from_clear(clear_)
                    self.assertEqual(
                        out_clear_,
                        ans_clear_,
                        f"out_clear = {str(out_clear_)}, ans_clear = {str(ans_clear_)}",
                    )
