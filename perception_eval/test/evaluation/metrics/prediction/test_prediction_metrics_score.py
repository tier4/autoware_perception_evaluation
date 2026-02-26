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
from perception_eval.evaluation.matching.object_matching import MatchingLabelPolicy
from perception_eval.evaluation.matching.object_matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics.metrics_score_config import MetricsScoreConfig
from perception_eval.evaluation.metrics.prediction.prediction_metrics_score import DisplacementErrorScores
from perception_eval.evaluation.metrics.prediction.prediction_metrics_score import PredictionMetricsScore
from perception_eval.evaluation.metrics.tracking.tracking_metrics_score import TrackingMetricsScore
from perception_eval.evaluation.result.object_result_matching import NuscenesObjectMatcher
from perception_eval.util.debug import get_objects_with_difference


class TestTrackingMetricsScore(unittest.TestCase):
    """The class to test TrackingMetricsScore."""

    def setUp(self) -> None:
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()

        self.evaluation_task: EvaluationTask = EvaluationTask.PREDICTION
        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]
        self.max_x_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]
        self.max_y_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]
        self.metric_score_config = MetricsScoreConfig(
            evaluation_task=self.evaluation_task,
            target_labels=self.target_labels,
            center_distance_thresholds=[0.5],
            center_distance_bev_thresholds=None,
            plane_distance_thresholds=None,
            iou_2d_thresholds=None,
            iou_3d_thresholds=None,
            top_ks=[1, 3],
        )

    def test_sum_clear_with_class_agnostic_fps(self):
        """[summary]
        Test summing up CLEAR scores. _sum_clear() returns total MOTA, MOTP and IDsw for same frames.
        Matching is class agnostic FPS.

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
        matcher = NuscenesObjectMatcher(
            evaluation_task=self.evaluation_task,
            metrics_config=self.metric_score_config,
            uuid_matching_first=False,
            matching_class_agnostic_fps=False,
        )
        # patterns: (PrevTrans, CurrTrans, MOTA, MOTP, IDsw)
        patterns: List[Tuple[DiffTranslation, DiffTranslation, float, float, int]] = [
            # (1)
            # -> current    : ((Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2]))
            (
                # cur: (trans est, trans gt)
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                [
                    # top_k = 1
                    DisplacementErrorScores(
                        ade=2.6235, top_k=1, fde=3.9241, miss_rate=0.6666, predict_num=3, ground_truth_num=4
                    ),
                    # top_k = 3
                    DisplacementErrorScores(
                        ade=1.3118, top_k=3, fde=1.9620, miss_rate=0.3333, predict_num=3, ground_truth_num=4
                    ),
                ],
            ),
            # (2)
            # -> current    : (Est[2], GT[0]), (Est[2], None), (Est[1], None)
            (
                DiffTranslation((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
                [
                    # top_k = 1
                    DisplacementErrorScores(
                        ade=2.8284, top_k=1, fde=4.2426, miss_rate=0.6666, predict_num=3, ground_truth_num=4
                    ),
                    # top_k = 3
                    DisplacementErrorScores(
                        ade=1.4142, top_k=3, fde=2.1213, miss_rate=0.3333, predict_num=3, ground_truth_num=4
                    ),
                ],
            ),
        ]
        for n, (cur_diff_trans, displacement_error_answers) in enumerate(patterns):
            with self.subTest(f"Test sum CLEAR: {n + 1}"):
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
                    dynamic_objects=cur_estimated_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                cur_ground_truth_objects = filter_objects(
                    dynamic_objects=cur_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )

                # Current object results
                cur_object_results = matcher.match(
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )

                num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
                    cur_ground_truth_objects, self.target_labels
                )

                prediction_scores_ = []
                for top_k in self.metric_score_config.prediction_config.top_ks:
                    prediction_score_ = PredictionMetricsScore(
                        nuscene_object_results=cur_object_results,
                        num_ground_truth_dict=num_ground_truth_dict,
                        target_labels=self.target_labels,
                        top_k=top_k,
                        miss_tolerance=2.0,
                    )
                    prediction_scores_.append(prediction_score_)

                    # String representation test
                    prediction_score_str = str(prediction_score_)
                    self.assertIn(f"Matching Mode: {MatchingMode.CENTERDISTANCE}, Top K: {top_k}", prediction_score_str)
                    self.assertIn(f"Predict Num", prediction_score_str)
                    self.assertIn(f"Ground Truth Num", prediction_score_str)
                    self.assertIn(f"ADE", prediction_score_str)
                    self.assertIn(f"FDE", prediction_score_str)
                    self.assertIn(f"Miss Rate", prediction_score_str)
                    for target_label in self.target_labels:
                        self.assertIn(f"{target_label}", prediction_score_str)

                for index, prediction_score_ in enumerate(prediction_scores_):
                    for _, thresholds in prediction_score_.displacement_error_scores.items():
                        for _, scores in thresholds.items():
                            self.assertAlmostEqual(scores.ade, displacement_error_answers[index].ade, delta=1e-4)
                            self.assertAlmostEqual(scores.fde, displacement_error_answers[index].fde, delta=1e-4)
                            self.assertAlmostEqual(
                                scores.miss_rate, displacement_error_answers[index].miss_rate, delta=1e-4
                            )
                            self.assertEqual(scores.predict_num, displacement_error_answers[index].predict_num)
                            self.assertEqual(
                                scores.ground_truth_num, displacement_error_answers[index].ground_truth_num
                            )
