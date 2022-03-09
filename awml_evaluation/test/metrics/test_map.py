import math
from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple
import unittest

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.detection.map import Map
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.util.debug import get_objects_with_difference


class TestMap(unittest.TestCase):
    def setUp(self):
        self.dummy_predicted_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_predicted_objects, self.dummy_ground_truth_objects = make_dummy_data()

        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]
        self.max_x_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]
        self.max_y_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]

    def test_map_center_distance_translation_difference(self):
        """[summary]
        Test mAP and mAPH with center distance matching for translation difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_distance, check if map and maph are almost correct.
        """
        # patterns: (diff_distance, ans_map, ans_maph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_distance, map and maph is 1.0.
            (0.0, 1.0, 1.0),
            # Given 0.5 diff_distance for one axis, map and maph are equal to 1.0
            # since both are under the metrics threshold.
            (0.5, 1.0, 1.0),
            # Given 2.5 diff_distance for one axis, map and maph are equal to 0.0
            # since both are beyond the metrics threshold.
            (2.5, 0.0, 0.0),
        ]
        for diff_distance, ans_map, ans_maph in patterns:
            with self.subTest(
                "Test mAP and mAPH with center distance matching for translation difference."
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
                    predicted_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                map: Map = Map(
                    object_results=object_results,
                    ground_truth_objects=diff_distance_dummy_ground_truth_objects,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[1.0, 1.0, 1.0, 1.0],
                )
                self.assertAlmostEqual(map.map, ans_map)
                self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_center_distance_yaw_difference(self):
        """[summary]
        Test mAP and mAPH with center distance matching for yaw difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_yaw, check if map and maph are almost correct.
        """
        # patterns: (diff_yaw, ans_map, ans_maph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_yaw, map and maph is 1.0.
            (0.0, 1.0, 1.0),
            # Given vertical diff_yaw, maph is 0.5**2 times map
            # since precision and recall of maph is 0.5 times those of map.
            (math.pi / 2.0, 1.0, 0.25),
            (-math.pi / 2.0, 1.0, 0.25),
            # Given opposite direction, maph is 0.0.
            (math.pi, 1.0, 0.0),
            (-math.pi, 1.0, 0.0),
            # Given diff_yaw is pi/4, maph is 0.75**2 times map
            (math.pi / 4, 1.0, 0.5625),
            (-math.pi / 4, 1.0, 0.5625),
            # Given diff_yaw is 3*pi/4, maph is 0.25**2 times map
            (3 * math.pi / 4, 1.0, 0.0625),
            (-3 * math.pi / 4, 1.0, 0.0625),
        ]

        for diff_yaw, ans_map, ans_maph in patterns:
            with self.subTest(
                "Test mAP and mAPH with center distance matching for yaw difference."
            ):
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
                    predicted_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                map: Map = Map(
                    object_results=object_results,
                    ground_truth_objects=diff_yaw_dummy_ground_truth_objects,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[1.0, 1.0, 1.0, 1.0],
                )
                self.assertAlmostEqual(map.map, ans_map)
                self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_center_distance_random_objects(self):
        """[summary]
        Test mAP and mAPH with center distance matching for random objects.

        test objects:
            dummy_predicted_objects (List[DynamicObject])
            dummy_ground_truth_objects (List[DynamicObject])

        test patterns:
            Check if map and maph are almost correct.
        """
        ans_map: float = (1.0 + 1.0 + 0.0 + 0.0) / 4.0
        ans_maph: float = (1.0 + 1.0 + 0.0 + 0.0) / 4.0

        object_results: List[
            DynamicObjectWithPerceptionResult
        ] = PerceptionFrameResult.get_object_results(
            predicted_objects=self.dummy_predicted_objects,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        map: Map = Map(
            object_results=object_results,
            ground_truth_objects=self.dummy_ground_truth_objects,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.CENTERDISTANCE,
            matching_threshold_list=[1.0, 1.0, 1.0, 1.0],
        )
        self.assertAlmostEqual(map.map, ans_map)
        self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_iou_bev_translation_difference(self):
        """[summary]
        Test mAP and mAPH with iou bev matching for translation difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_distance, check if map and maph are almost correct.
        """
        # patterns: (diff_distance, ans_map, ans_maph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_distance, map and maph is 1.0.
            (0.0, 1.0, 1.0),
            # Given 0.3 diff_distance for one axis, map and maph are equal to 1.0
            # since both are over the metrics threshold.
            (0.3, 1.0, 1.0),
            # Given 0.5 diff_distance for one axis, map and maph are equal to 0.0
            # since both are under the metrics threshold.
            (0.5, 0.0, 0.0),
        ]
        for diff_distance, ans_map, ans_maph in patterns:
            with self.subTest(
                "Test mAP and mAPH with iou bev matching for translation difference."
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
                    predicted_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                map: Map = Map(
                    object_results=object_results,
                    ground_truth_objects=diff_distance_dummy_ground_truth_objects,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOUBEV,
                    matching_threshold_list=[0.5, 0.5, 0.5, 0.5],
                )
                self.assertAlmostEqual(map.map, ans_map)
                self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_iou_bev_yaw_difference(self):
        """[summary]
        Test mAP and mAPH with iou bev matching for yaw difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_yaw, check if map and maph are almost correct.
        """
        # patterns: (diff_yaw, ans_map, ans_maph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_yaw, map and maph is 1.0.
            (0.0, 1.0, 1.0),
            # Given vertical diff_yaw, maph is 0.25 times map
            # since precision and recall of maph is 0.5 times those of map.
            (math.pi / 2.0, 1.0, 0.25),
            (-math.pi / 2.0, 1.0, 0.25),
            # Given opposite direction, maph is 0.0.
            (math.pi, 1.0, 0.0),
            (-math.pi, 1.0, 0.0),
            # Given diff_yaw is pi/4, maph is 0.75**2 times map
            (math.pi / 4, 1.0, 0.5625),
            (-math.pi / 4, 1.0, 0.5625),
            # Given diff_yaw is 3*pi/4, maph is 0.25**2 times map
            (3 * math.pi / 4, 1.0, 0.0625),
            (-3 * math.pi / 4, 1.0, 0.0625),
        ]

        for diff_yaw, ans_map, ans_maph in patterns:
            with self.subTest("Test mAP and mAPH with iou bev matching for yaw difference."):
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
                    predicted_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                map: Map = Map(
                    object_results=object_results,
                    ground_truth_objects=diff_yaw_dummy_ground_truth_objects,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOUBEV,
                    matching_threshold_list=[0.5, 0.5, 0.5, 0.5],
                )
                self.assertAlmostEqual(map.map, ans_map)
                self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_iou_bev_random_objects(self):
        """[summary]
        Test mAP and mAPH with iou bev matching for random objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Check if map and maph are almost correct.
        """
        ans_map: float = (1.0 + 0.0 + 0.0 + 0.0) / 4.0
        ans_maph: float = (1.0 + 0.0 + 0.0 + 0.0) / 4.0

        object_results: List[
            DynamicObjectWithPerceptionResult
        ] = PerceptionFrameResult.get_object_results(
            predicted_objects=self.dummy_predicted_objects,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        map: Map = Map(
            object_results=object_results,
            ground_truth_objects=self.dummy_ground_truth_objects,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.IOUBEV,
            matching_threshold_list=[0.2, 0.5, 0.5, 0.5],
        )
        self.assertAlmostEqual(map.map, ans_map)
        self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_iou_3d_translation_difference(self):
        """[summary]
        Test mAP and mAPH with iou 3d matching for translation difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_distance, check if map and maph are almost correct.
        """
        # patterns: (diff_distance, ans_map, ans_maph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_distance, map and maph is 1.0.
            (0.0, 1.0, 1.0),
            # Given 0.3 diff_distance for one axis, map and maph are equal to 1.0
            # since both are under the metrics threshold.
            (0.3, 1.0, 1.0),
            # Given 0.5 diff_distance for one axis, map and maph are equal to 0.0
            # since both are beyond the metrics threshold.
            (0.5, 0.0, 0.0),
        ]
        for diff_distance, ans_map, ans_maph in patterns:
            with self.subTest(
                "Test mAP and mAPH with iou 3d matching for translation difference."
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
                    predicted_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                map: Map = Map(
                    object_results=object_results,
                    ground_truth_objects=diff_distance_dummy_ground_truth_objects,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOU3D,
                    matching_threshold_list=[0.5, 0.5, 0.5, 0.5],
                )
                self.assertAlmostEqual(map.map, ans_map)
                self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_iou_3d_yaw_difference(self):
        """[summary]
        Test mAP and mAPH with iou 3d matching for yaw difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_yaw, check if map and maph are almost correct.
        """
        # patterns: (diff_yaw, ans_map, ans_maph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_yaw, map and maph is 1.0.
            (0.0, 1.0, 1.0),
            # Given vertical diff_yaw, maph is 0.25 times map
            # since precision and recall of maph is 0.5 times those of map.
            (math.pi / 2.0, 1.0, 0.25),
            (-math.pi / 2.0, 1.0, 0.25),
            # Given opposite direction, maph is 0.0.
            (math.pi, 1.0, 0.0),
            (-math.pi, 1.0, 0.0),
            # Given diff_yaw is pi/4, maph is 0.75**2 times map
            (math.pi / 4, 1.0, 0.5625),
            (-math.pi / 4, 1.0, 0.5625),
            # Given diff_yaw is 3*pi/4, maph is 0.25**2 times map
            (3 * math.pi / 4, 1.0, 0.0625),
            (-3 * math.pi / 4, 1.0, 0.0625),
        ]

        for diff_yaw, ans_map, ans_maph in patterns:
            with self.subTest("Test mAP and mAPH with iou 3d matching for yaw difference."):
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
                    predicted_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                map: Map = Map(
                    object_results=object_results,
                    ground_truth_objects=diff_yaw_dummy_ground_truth_objects,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.IOU3D,
                    matching_threshold_list=[0.5, 0.5, 0.5, 0.5],
                )
                self.assertAlmostEqual(map.map, ans_map)
                self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_iou_3d_random_objects(self):
        """[summary]
        Test mAP and mAPH with iou 3d matching for random objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Check if map and maph are almost correct.
        """
        ans_map: float = (1.0 + 0.0 + 0.0 + 0.0) / 4.0
        ans_maph: float = (1.0 + 0.0 + 0.0 + 0.0) / 4.0

        object_results: List[
            DynamicObjectWithPerceptionResult
        ] = PerceptionFrameResult.get_object_results(
            predicted_objects=self.dummy_predicted_objects,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        map: Map = Map(
            object_results=object_results,
            ground_truth_objects=self.dummy_ground_truth_objects,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.IOU3D,
            matching_threshold_list=[0.2, 0.5, 0.5, 0.5],
        )
        self.assertAlmostEqual(map.map, ans_map)
        self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_plane_distance_translation_difference(self):
        """[summary]
        Test mAP and mAPH with plane distance matching for translation difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_distance, check if map and maph are almost correct.
        """
        # patterns: (diff_distance, ans_map, ans_maph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_distance, map and maph is 1.0.
            (0.0, 1.0, 1.0),
            # Given 0.5 diff_distance for one axis, map and maph are equal to 1.0
            # since both are under the metrics threshold.
            (0.5, 1.0, 1.0),
            # Given 2.5 diff_distance for one axis, map and maph are equal to 0.0
            # since both are beyond the metrics threshold.
            (2.5, 0.0, 0.0),
        ]
        for diff_distance, ans_map, ans_maph in patterns:
            with self.subTest(
                "Test mAP and mAPH with plane distance matching for translation difference."
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
                    predicted_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                map: Map = Map(
                    object_results=object_results,
                    ground_truth_objects=diff_distance_dummy_ground_truth_objects,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.PLANEDISTANCE,
                    matching_threshold_list=[1.0, 1.0, 1.0, 1.0],
                )
                self.assertAlmostEqual(map.map, ans_map)
                self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_plane_distance_yaw_difference(self):
        """[summary]
        Test mAP and mAPH with plane distance matching for yaw difference.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Given diff_yaw, check if map and maph are almost correct.
        """
        # patterns: (diff_yaw, ans_map, ans_maph)
        patterns: List[Tuple[float, float, float]] = [
            # Given no diff_yaw, map and maph is 1.0.
            (0.0, 1.0, 1.0),
            # Given vertical diff_yaw, maph is 0.5**2 times map
            # since precision and recall of maph is 0.5 times those of map.
            (math.pi / 2.0, 1.0, 0.25),
            (-math.pi / 2.0, 1.0, 0.25),
            # Given opposite direction, maph is 0.0.
            (math.pi, 1.0, 0.0),
            (-math.pi, 1.0, 0.0),
            # Given diff_yaw is pi/4, maph is 0.75**2 times map
            (math.pi / 4, 1.0, 0.5625),
            (-math.pi / 4, 1.0, 0.5625),
            # Given diff_yaw is 3*pi/4, maph is 0.25**2 times map
            (3 * math.pi / 4, 1.0, 0.0625),
            (-3 * math.pi / 4, 1.0, 0.0625),
        ]

        for diff_yaw, ans_map, ans_maph in patterns:
            with self.subTest(
                "Test mAP and mAPH with plane distance matching for yaw difference."
            ):
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
                    predicted_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                map: Map = Map(
                    object_results=object_results,
                    ground_truth_objects=diff_yaw_dummy_ground_truth_objects,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    matching_mode=MatchingMode.PLANEDISTANCE,
                    matching_threshold_list=[1.0, 1.0, 1.0, 1.0],
                )
                self.assertAlmostEqual(map.map, ans_map)
                self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_plane_distance_random_objects(self):
        """[summary]
        Test mAP and mAPH with plane distance matching for random objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Check if map and maph are almost correct.
        """
        # dummy_predicted_objects[0] (CAR) and dummy_ground_truth_objects[0] (CAR):
        #   sorted pr_corner_points[:2] = [(0.25, 0.25, 1.0), (0.25, 1.75, 1.0)]
        #   sorted gt_corner_points[:2] = [(0.5, 0.5, 1.0), (1.5, 0.5, 1.0)]
        #   plane_distance = 1.2747548783981963
        #
        # dummy_predicted_objects[1] (BICYCLE) and dummy_ground_truth_objects[1] (BICYCLE):
        #   sorted pr_corner_points[:2] = [(0.75, -0.75, 1.0), (1.25, -0.75, 1.0)]
        #   sorted gt_corner_points[:2] = [(0.5, -0.5, 1.0), (1.5, -0.5, 1.0)]
        #   plane_distance = 0.3535533905932738
        #
        # dummy_predicted_objects[2] (PEDESTRIAN) and dummy_ground_truth_objects[2] (CAR):
        #   sorted pr_corner_points[:2] = [(-0.5, 0.5, 1.0), (-0.5, 1.5, 1.0)]
        #   sorted gt_corner_points[:2] = [(-0.5, 0.5, 1.0), (-0.5, 1.5, 1.0)]
        #   plane_distance = 0.0

        # CAR: beyond the threshold
        #   ap and aph: 0.0
        # BICYCLE: under the threshold
        #   ap and aph: 1.0
        # PEDESTRIAN: under the threshold but label does not match
        #   ap and aph: 0.0
        # MOTORBIKE: not predicted
        #   ap and aph: 0.0
        ans_map: float = (0.0 + 1.0 + 0.0 + 0.0) / 4.0
        ans_maph: float = (0.0 + 1.0 + 0.0 + 0.0) / 4.0

        object_results: List[
            DynamicObjectWithPerceptionResult
        ] = PerceptionFrameResult.get_object_results(
            predicted_objects=self.dummy_predicted_objects,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        map: Map = Map(
            object_results=object_results,
            ground_truth_objects=self.dummy_ground_truth_objects,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            matching_mode=MatchingMode.PLANEDISTANCE,
            matching_threshold_list=[1.0, 1.0, 1.0, 1.0],
        )

        self.assertAlmostEqual(map.map, ans_map)
        self.assertAlmostEqual(map.maph, ans_maph)


if __name__ == "__main__":
    unittest.main()
