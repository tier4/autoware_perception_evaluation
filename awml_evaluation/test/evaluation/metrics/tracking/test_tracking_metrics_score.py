from test.evaluation.metrics.tracking.test_clear import AnswerCLEAR
from test.util.dummy_object import make_dummy_data
from test.util.object_diff import DiffTranslation
from test.util.object_diff import DiffYaw
from typing import List
from typing import Tuple
import unittest

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.tracking.tracking_metrics_score import TrackingMetricsScore
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.util.debug import get_objects_with_difference


class TestTrackingMetricsScore(unittest.TestCase):
    """The class to test TrackingMetricsScore."""

    def setUp(self) -> None:
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()

        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]
        self.max_x_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]
        self.max_y_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]

    def test_sum_clear(self):
        """Test summing up CLEAR scores.

        Test patterns:
            Check the summed up MOTA/MOTP and ID switch score with translated previous and current results.
        """
        # patterns: (prev_diff_trans, cur_diff_trans, ans_mota, ans_motp, ans_id_switch)
        patterns: List[Tuple[DiffTranslation, DiffTranslation, float, float, int]] = [
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                0.25,
                0.0,
                0,
            ),
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.5, 2.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                0.25,
                0.0,
                1,
            ),
            (
                DiffTranslation((1.0, 0.0, 0.0), (0.2, 0, 0.0)),
                DiffTranslation((0.25, 0.0, 0.0), (0.0, 0.0, 0.0)),
                0.25,
                0.25,
                0,
            ),
        ]
        for prev_diff_trans, cur_diff_trans, ans_mota, ans_motp, ans_id_switch in patterns:
            with self.subTest("Test sum CLEAR"):
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
                # Previous object results
                prev_object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=prev_estimated_objects,
                    ground_truth_objects=prev_ground_truth_objects,
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
                # Current object results
                cur_object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )

                tracking_score: TrackingMetricsScore = TrackingMetricsScore(
                    object_results=[prev_object_results, cur_object_results],
                    ground_truth_objects=[prev_ground_truth_objects, cur_ground_truth_objects],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[0.5, 0.5, 0.5, 0.5],
                )
                mota, motp, id_switch = tracking_score._sum_clear()
                self.assertAlmostEqual(mota, ans_mota)
                self.assertAlmostEqual(motp, ans_motp)
                self.assertEqual(id_switch, ans_id_switch)

    def test_center_distance_translation_difference(self):
        """Test TrackingMetricsScore with center distance matching, when each object result is translated by xy axis.

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
                # Previous object results
                prev_object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=prev_estimated_objects,
                    ground_truth_objects=prev_ground_truth_objects,
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
                # Current object results
                cur_object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )

                tracking_score: TrackingMetricsScore = TrackingMetricsScore(
                    object_results=[prev_object_results, cur_object_results],
                    ground_truth_objects=[prev_ground_truth_objects, cur_ground_truth_objects],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
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
            with self.subTest(f"Test sum CLEAR: {n + 1}"):
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
                # Previous object results
                prev_object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=prev_estimated_objects,
                    ground_truth_objects=prev_ground_truth_objects,
                )

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
                # Current object results
                cur_object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )

                tracking_score: TrackingMetricsScore = TrackingMetricsScore(
                    object_results=[prev_object_results, cur_object_results],
                    ground_truth_objects=[prev_ground_truth_objects, cur_ground_truth_objects],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
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
