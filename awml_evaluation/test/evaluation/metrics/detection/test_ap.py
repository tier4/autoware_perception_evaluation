import math
from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple
import unittest

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.detection.ap import Ap
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.util.debug import get_objects_with_difference
import numpy as np


class TestAp(unittest.TestCase):
    def setUp(self):
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()

        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
        ]
        self.max_x_position_list: List[float] = [100.0]
        self.max_y_position_list: List[float] = [100.0]

    def test_ap_center_distance_translation_difference(self):
        """[summary]
        Test AP and APH with center distance matching for translation difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_distance, check if ap and aph are almost correct.
        """
        # patterns: (diff_distance, ans_ap, ans_aph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_distance, ap and aph is 1.0.
            (0.0, 1.0, 1.0),
            # Given 1.0 diff_distance for one axis, ap and aph are equal to 0.0
            # since both are over the metrics threshold.
            (1.0, 0.0, 0.0),
        ]
        for diff_distance, ans_ap, ans_aph in patterns:
            with self.subTest(
                "Test AP and APH with center distance matching for translation difference."
            ):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
                    unix_time=0,
                    frame_name="0",
                    frame_id="base_link",
                    objects=diff_distance_dummy_ground_truth_objects,
                    ego2map=np.eye(4),
                )
                ap: Ap = Ap(
                    tp_metrics=TPMetricsAp(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[0.5],
                    min_point_numbers=[0],
                )

                aph: Ap = Ap(
                    tp_metrics=TPMetricsAph(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[0.5],
                    min_point_numbers=[0],
                )
                self.assertAlmostEqual(ap.ap, ans_ap)
                self.assertAlmostEqual(aph.ap, ans_aph)

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
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_yaw, ap and aph is 1.0.
            (0.0, 1.0, 1.0),
            # Given vertical diff_yaw, aph is 0.5**2 times ap
            # since precision and recall of aph is 0.5 times those of ap.
            (math.pi / 2.0, 1.0, 0.25),
            (-math.pi / 2.0, 1.0, 0.25),
            # Given opposite direction, aph is 0.0.
            (math.pi, 1.0, 0.0),
            (-math.pi, 1.0, 0.0),
            # Given diff_yaw is pi/4, aph is 0.75**2 times ap
            (math.pi / 4, 1.0, 0.5625),
            (-math.pi / 4, 1.0, 0.5625),
            # Given diff_yaw is 3*pi/4, aph is 0.25**2 times ap
            (3 * math.pi / 4, 1.0, 0.0625),
            (-3 * math.pi / 4, 1.0, 0.0625),
        ]

        for diff_yaw, ans_ap, ans_aph in patterns:
            with self.subTest("Test AP and APH with center distance matching for yaw difference."):
                diff_yaw_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=diff_yaw,
                )
                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
                    unix_time=0,
                    frame_name="0",
                    frame_id="base_link",
                    objects=diff_yaw_dummy_ground_truth_objects,
                    ego2map=np.eye(4),
                )
                ap: Ap = Ap(
                    tp_metrics=TPMetricsAp(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[0.1],
                    min_point_numbers=[0],
                )
                aph: Ap = Ap(
                    tp_metrics=TPMetricsAph(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[0.1],
                    min_point_numbers=[0],
                )

                self.assertAlmostEqual(ap.ap, ans_ap)
                self.assertAlmostEqual(aph.ap, ans_aph)

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
        # ap and aph is 0.0 since no MOTORBIKE in the estimated_objects
        ans_ap: float = 0.0
        ans_aph: float = 0.0

        object_results: List[
            DynamicObjectWithPerceptionResult
        ] = PerceptionFrameResult.get_object_results(
            estimated_objects=self.dummy_estimated_objects,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
            unix_time=0,
            frame_name="0",
            frame_id="base_link",
            objects=self.dummy_ground_truth_objects,
            ego2map=np.eye(4),
        )
        ap: Ap = Ap(
            tp_metrics=TPMetricsAp(),
            object_results=[object_results],
            frame_ground_truths=[frame_ground_truth],
            target_labels=[AutowareLabel.MOTORBIKE],
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.CENTERDISTANCE,
            matching_threshold_list=[0.1],
            min_point_numbers=[0],
        )
        aph: Ap = Ap(
            tp_metrics=TPMetricsAph(),
            object_results=[object_results],
            frame_ground_truths=[frame_ground_truth],
            target_labels=[AutowareLabel.MOTORBIKE],
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.CENTERDISTANCE,
            matching_threshold_list=[0.1],
            min_point_numbers=[0],
        )

        self.assertAlmostEqual(ap.ap, ans_ap)
        self.assertAlmostEqual(aph.ap, ans_aph)

    def test_ap_iou_bev_translation_difference(self):
        """[summary]
        Test AP and APH with iou bev matching for translation difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_distance, check if ap and aph are almost correct.
        """
        # patterns: (diff_distance, ans_ap, ans_aph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_distance, ap and aph is 1.0.
            (0.0, 1.0, 1.0),
            # Given 0.3 diff_distance for one axis, ap and aph are equal to 0.0
            # since iou_bev is under the threshold.
            (0.3, 0.0, 0.0),
        ]
        for diff_distance, ans_ap, ans_aph in patterns:
            with self.subTest("Test AP and APH with iou bev matching for translation difference."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )

                frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
                    unix_time=0,
                    frame_name="0",
                    frame_id="base_link",
                    objects=diff_distance_dummy_ground_truth_objects,
                    ego2map=np.eye(4),
                )

                ap: Ap = Ap(
                    tp_metrics=TPMetricsAp(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOUBEV,
                    matching_threshold_list=[0.7],
                    min_point_numbers=[0],
                )
                aph: Ap = Ap(
                    tp_metrics=TPMetricsAph(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOUBEV,
                    matching_threshold_list=[0.7],
                    min_point_numbers=[0],
                )

                self.assertAlmostEqual(ap.ap, ans_ap)
                self.assertAlmostEqual(aph.ap, ans_aph)

    def test_ap_iou_bev_yaw_difference(self):
        """[summary]
        Test ap and APH with iou bev matching for yaw difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_yaw, check if ap and aph are almost correct.
        """
        # patterns: (diff_yaw, ans_ap, ans_aph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_yaw, ap and aph is 1.0.
            # (iou_bev is 1.0)
            (0.0, 1.0, 1.0),
            # Given vertical diff_yaw, aph is 0.25 times ap
            # since precision and recall of aph is 0.5 times those of ap.
            # (iou_bev is 1.0)
            (math.pi / 2.0, 1.0, 0.25),
            (-math.pi / 2.0, 1.0, 0.25),
            # Given opposite direction, aph is 0.0.
            # (iou_bev is 1.0)
            (math.pi, 1.0, 0.0),
            (-math.pi, 1.0, 0.0),
            # Given diff_yaw is pi/4, aph is 0.75**2 times ap
            # iou_bev is 0.7071067811865472
            # which is under the threshold (0.8) for ap
            # and over the threshold (0.7) for aph
            (math.pi / 4, 0.0, 0.5625),
            (-math.pi / 4, 0.0, 0.5625),
            # Given diff_yaw is 3*pi/4, aph is 0.25**2 times ap
            # iou_bev is 0.7071067811865472
            # which is under the threshold (0.8) for ap
            # and over the threshold (0.7) for aph
            (3 * math.pi / 4, 0.0, 0.0625),
            (-3 * math.pi / 4, 0.0, 0.0625),
        ]

        for diff_yaw, ans_ap, ans_aph in patterns:
            with self.subTest("Test AP and APH with iou bev matching for yaw difference."):
                diff_yaw_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=diff_yaw,
                )
                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
                    unix_time=0,
                    frame_name="0",
                    frame_id="base_link",
                    objects=diff_yaw_dummy_ground_truth_objects,
                    ego2map=np.eye(4),
                )
                ap: Ap = Ap(
                    tp_metrics=TPMetricsAp(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOUBEV,
                    matching_threshold_list=[0.8],
                    min_point_numbers=[0],
                )
                aph: Ap = Ap(
                    tp_metrics=TPMetricsAph(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOUBEV,
                    matching_threshold_list=[0.7],
                    min_point_numbers=[0],
                )

                self.assertAlmostEqual(ap.ap, ans_ap)
                self.assertAlmostEqual(aph.ap, ans_aph)

    def test_ap_iou_bev_random_objects(self):
        """[summary]
        Test AP and APH with iou bev matching for random objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Check if ap and aph are almost correct.
        """
        # iou_bev is 0.4444444444444444
        # which is over the threshold (0.4)
        ans_ap: float = 1.0
        ans_aph: float = 1.0
        object_results: List[
            DynamicObjectWithPerceptionResult
        ] = PerceptionFrameResult.get_object_results(
            estimated_objects=self.dummy_estimated_objects,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
            unix_time=0,
            frame_name="0",
            frame_id="base_link",
            objects=self.dummy_ground_truth_objects,
            ego2map=np.eye(4),
        )
        ap: Ap = Ap(
            tp_metrics=TPMetricsAp(),
            object_results=[object_results],
            frame_ground_truths=[frame_ground_truth],
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.IOUBEV,
            matching_threshold_list=[0.4],
            min_point_numbers=[0],
        )
        aph: Ap = Ap(
            tp_metrics=TPMetricsAph(),
            object_results=[object_results],
            frame_ground_truths=[frame_ground_truth],
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.IOUBEV,
            matching_threshold_list=[0.4],
            min_point_numbers=[0],
        )

        self.assertAlmostEqual(ap.ap, ans_ap)
        self.assertAlmostEqual(aph.ap, ans_aph)

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
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_distance, ap and aph is 1.0.
            (0.0, 1.0, 1.0),
            # Given 0.5 diff_distance for one axis, ap and aph are equal to 0.0
            # since iou_3d is 0.5384615384615382 which is under the threshold (0.6).
            (0.3, 0.0, 0.0),
        ]
        for diff_distance, ans_ap, ans_aph in patterns:
            with self.subTest("Test AP and APH with iou 3d matching for translation difference."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
                    unix_time=0,
                    frame_name="0",
                    frame_id="base_link",
                    objects=diff_distance_dummy_ground_truth_objects,
                    ego2map=np.eye(4),
                )

                ap: Ap = Ap(
                    tp_metrics=TPMetricsAp(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOU3D,
                    matching_threshold_list=[0.6],
                    min_point_numbers=[0],
                )
                aph: Ap = Ap(
                    tp_metrics=TPMetricsAph(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOU3D,
                    matching_threshold_list=[0.6],
                    min_point_numbers=[0],
                )

                self.assertAlmostEqual(ap.ap, ans_ap)
                self.assertAlmostEqual(aph.ap, ans_aph)

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
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_yaw, ap and aph is 1.0.
            (0.0, 1.0, 1.0),
            # Given vertical diff_yaw, aph is 0.25 times ap
            # since precision and recall of aph is 0.5 times those of ap.
            (math.pi / 2.0, 1.0, 0.25),
            (-math.pi / 2.0, 1.0, 0.25),
            # Given opposite direction, aph is 0.0.
            (math.pi, 1.0, 0.0),
            (-math.pi, 1.0, 0.0),
            # Given diff_yaw is pi/4, aph is 0.75**2 times ap
            # iou_3d is 0.7071067811865472
            # which is under the threshold (0.8)
            (math.pi / 4, 0.0, 0.0),
            (-math.pi / 4, 0.0, 0.0),
            # Given diff_yaw is 3*pi/4, aph is 0.25**2 times ap
            # iou_3d is 0.7071067811865472
            # which is under the threshold (0.8)
            (3 * math.pi / 4, 0.0, 0.0),
            (-3 * math.pi / 4, 0.0, 0.0),
        ]

        for diff_yaw, ans_ap, ans_aph in patterns:
            with self.subTest("Test AP and APH with iou 3d matching for yaw difference."):
                diff_yaw_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=diff_yaw,
                )
                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
                    unix_time=0,
                    frame_name="0",
                    frame_id="base_link",
                    objects=diff_yaw_dummy_ground_truth_objects,
                    ego2map=np.eye(4),
                )
                ap: Ap = Ap(
                    tp_metrics=TPMetricsAp(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOU3D,
                    matching_threshold_list=[0.8],
                    min_point_numbers=[0],
                )
                aph: Ap = Ap(
                    tp_metrics=TPMetricsAph(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOU3D,
                    matching_threshold_list=[0.8],
                    min_point_numbers=[0],
                )

                self.assertAlmostEqual(ap.ap, ans_ap)
                self.assertAlmostEqual(aph.ap, ans_aph)

    def test_ap_iou_3d_random_objects(self):
        """[summary]
        Test AP and APH with iou 3d matching for random objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Check if ap and aph are almost correct.
        """
        # iou_3d is 0.2962962962962963
        # which is under the threshold for ap (0.3)
        # and over the threshold for aph (0.2).
        ans_ap: float = 0.0
        ans_aph: float = 1.0

        object_results: List[
            DynamicObjectWithPerceptionResult
        ] = PerceptionFrameResult.get_object_results(
            estimated_objects=self.dummy_estimated_objects,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
            unix_time=0,
            frame_name="0",
            frame_id="base_link",
            objects=self.dummy_ground_truth_objects,
            ego2map=np.eye(4),
        )
        ap: Ap = Ap(
            tp_metrics=TPMetricsAp(),
            object_results=[object_results],
            frame_ground_truths=[frame_ground_truth],
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.IOU3D,
            matching_threshold_list=[0.3],
            min_point_numbers=[0],
        )
        aph: Ap = Ap(
            tp_metrics=TPMetricsAph(),
            object_results=[object_results],
            frame_ground_truths=[frame_ground_truth],
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.IOU3D,
            matching_threshold_list=[0.2],
            min_point_numbers=[0],
        )

        self.assertAlmostEqual(ap.ap, ans_ap)
        self.assertAlmostEqual(aph.ap, ans_aph)

    def test_ap_plane_distance_translation_difference(self):
        """[summary]
        Test AP and APH with plane distance matching for translation difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_distance, check if ap and aph are almost correct.
        """
        # patterns: (diff_distance, ans_ap, ans_aph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_distance, ap and aph is 1.0.
            (0.0, 1.0, 1.0),
            # Given 1.0 diff_distance for one axis, ap and aph are equal to 0.0
            # since both are over the metrics threshold.
            (1.0, 0.0, 0.0),
        ]
        for diff_distance, ans_ap, ans_aph in patterns:
            with self.subTest(
                "Test AP and APH with plane distance matching for translation difference."
            ):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
                    unix_time=0,
                    frame_name="0",
                    frame_id="base_link",
                    objects=diff_distance_dummy_ground_truth_objects,
                    ego2map=np.eye(4),
                )
                ap: Ap = Ap(
                    tp_metrics=TPMetricsAp(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.PLANEDISTANCE,
                    matching_threshold_list=[0.1],
                    min_point_numbers=[0],
                )
                aph: Ap = Ap(
                    tp_metrics=TPMetricsAph(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.PLANEDISTANCE,
                    matching_threshold_list=[1.0],
                    min_point_numbers=[0],
                )

                self.assertAlmostEqual(ap.ap, ans_ap)
                self.assertAlmostEqual(aph.ap, ans_aph)

    def test_ap_plane_distance_yaw_difference(self):
        """[summary]
        Test AP and APH with plane distance matching for yaw difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_yaw, check if ap and aph are almost correct.
        """
        # patterns: (diff_yaw, ans_ap, ans_aph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_yaw, ap and aph is 1.0.
            (0.0, 1.0, 1.0),
            # Given vertical diff_yaw, aph is 0.5**2 times ap
            # since precision and recall of aph is 0.5 times those of ap.
            (math.pi / 2.0, 1.0, 0.25),
            (-math.pi / 2.0, 1.0, 0.25),
            # Given opposite direction, aph is 0.0.
            (math.pi, 1.0, 0.0),
            (-math.pi, 0.0, 0.0),
            # Given diff_yaw is pi/4, aph is 0.75**2 times ap
            (math.pi / 4, 1.0, 0.5625),
            (-math.pi / 4, 1.0, 0.5625),
            # Given diff_yaw is 3*pi/4, aph is 0.25**2 times ap
            (3 * math.pi / 4, 1.0, 0.0625),
            (-3 * math.pi / 4, 1.0, 0.0625),
        ]

        for diff_yaw, ans_ap, ans_aph in patterns:
            with self.subTest("Test AP and APH with plane distance matching for yaw difference."):
                diff_yaw_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=diff_yaw,
                )
                object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
                    unix_time=0,
                    frame_name="0",
                    frame_id="base_link",
                    objects=diff_yaw_dummy_ground_truth_objects,
                    ego2map=np.eye(4),
                )
                ap: Ap = Ap(
                    tp_metrics=TPMetricsAp(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.PLANEDISTANCE,
                    matching_threshold_list=[1.0],
                    min_point_numbers=[0],
                )
                aph: Ap = Ap(
                    tp_metrics=TPMetricsAph(),
                    object_results=[object_results],
                    frame_ground_truths=[frame_ground_truth],
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.PLANEDISTANCE,
                    matching_threshold_list=[1.0],
                    min_point_numbers=[0],
                )

                self.assertAlmostEqual(ap.ap, ans_ap)
                self.assertAlmostEqual(aph.ap, ans_aph)

    def test_ap_plane_distance_random_objects(self):
        """[summary]
        Test AP and APH with plane distance matching for random objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Check if ap and aph are almost correct.
        """
        # dummy_estimated_objects[0] (CAR) and dummy_ground_truth_objects[0] (CAR):
        #   pr_corner_points(left, right) = [(0.25, 0.25, 1.0), (0.25, 1.75, 1.0)]
        #   gt_corner_points(left, right) = [(0.5, 0.5, 1.0), (0.5, 1.5, 1.0)]
        #   plane_distance = 0.3535533905932738
        ans_ap_tp: float = 1.0
        ans_ap_tn: float = 0.0
        ans_aph_tp: float = 1.0
        ans_aph_tn: float = 0.0

        object_results: List[
            DynamicObjectWithPerceptionResult
        ] = PerceptionFrameResult.get_object_results(
            estimated_objects=self.dummy_estimated_objects,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
            unix_time=0,
            frame_name="0",
            frame_id="base_link",
            objects=self.dummy_ground_truth_objects,
            ego2map=np.eye(4),
        )

        ap_tp: Ap = Ap(
            tp_metrics=TPMetricsAp(),
            object_results=[object_results],
            frame_ground_truths=[frame_ground_truth],
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.PLANEDISTANCE,
            matching_threshold_list=[1.0],
            min_point_numbers=[0],
        )
        aph_tp: Ap = Ap(
            tp_metrics=TPMetricsAph(),
            object_results=[object_results],
            frame_ground_truths=[frame_ground_truth],
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.PLANEDISTANCE,
            matching_threshold_list=[1.0],
            min_point_numbers=[0],
        )
        ap_tn: Ap = Ap(
            tp_metrics=TPMetricsAp(),
            object_results=[object_results],
            frame_ground_truths=[frame_ground_truth],
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.PLANEDISTANCE,
            matching_threshold_list=[0.2],
            min_point_numbers=[0],
        )
        aph_tn: Ap = Ap(
            tp_metrics=TPMetricsAph(),
            object_results=[object_results],
            frame_ground_truths=[frame_ground_truth],
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.PLANEDISTANCE,
            matching_threshold_list=[0.2],
            min_point_numbers=[0],
        )
        self.assertAlmostEqual(ap_tp.ap, ans_ap_tp)
        self.assertAlmostEqual(aph_tp.ap, ans_aph_tp)
        self.assertAlmostEqual(ap_tn.ap, ans_ap_tn)
        self.assertAlmostEqual(aph_tn.ap, ans_aph_tn)


if __name__ == "__main__":
    unittest.main()
