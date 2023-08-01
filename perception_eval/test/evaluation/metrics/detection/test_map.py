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

import math
from test.util.dummy_object import make_dummy_data
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
from perception_eval.evaluation.metrics.detection.map import Map
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.object_result import get_object_results
from perception_eval.util.debug import get_objects_with_difference


class TestMap(unittest.TestCase):
    def setUp(self):
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()

        self.evaluation_task: EvaluationTask = EvaluationTask.DETECTION
        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]
        self.max_x_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]
        self.max_y_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]
        self.min_point_numbers: List[int] = [0, 0, 0, 0]

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
                # Filter objects
                diff_distance_dummy_ground_truth_objects = filter_objects(
                    objects=diff_distance_dummy_ground_truth_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                dummy_ground_truth_objects = filter_objects(
                    objects=self.dummy_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    min_point_numbers=self.min_point_numbers,
                )

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=dummy_ground_truth_objects,
                )
                object_results_dict: Dict[
                    AutowareLabel, List[DynamicObjectWithPerceptionResult]
                ] = divide_objects(
                    object_results,
                    self.target_labels,
                )

                num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
                    dummy_ground_truth_objects,
                    self.target_labels,
                )

                map: Map = Map(
                    object_results_dict=object_results_dict,
                    num_ground_truth_dict=num_ground_truth_dict,
                    target_labels=self.target_labels,
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
                # Filter objects
                diff_yaw_dummy_ground_truth_objects = filter_objects(
                    objects=diff_yaw_dummy_ground_truth_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                dummy_ground_truth_objects = filter_objects(
                    objects=self.dummy_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    min_point_numbers=self.min_point_numbers,
                )

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=dummy_ground_truth_objects,
                )
                object_results_dict: Dict[
                    AutowareLabel, List[DynamicObjectWithPerceptionResult]
                ] = divide_objects(
                    object_results,
                    self.target_labels,
                )

                num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
                    dummy_ground_truth_objects,
                    self.target_labels,
                )

                map: Map = Map(
                    object_results_dict=object_results_dict,
                    num_ground_truth_dict=num_ground_truth_dict,
                    target_labels=self.target_labels,
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[1.0, 1.0, 1.0, 1.0],
                )
                self.assertAlmostEqual(map.map, ans_map)
                self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_center_distance_random_objects(self):
        """[summary]
        Test mAP and mAPH with center distance matching for random objects.

        test objects:
            dummy_estimated_objects (List[DynamicObject])
            dummy_ground_truth_objects (List[DynamicObject])

        test patterns:
            Check if map and maph are almost correct.
        """
        # PEDESTRIAN and MOTORBIKE is not included -> each AP is `inf` and skipped in mAP computation.
        ans_map: float = (1.0 + 1.0) / 2.0
        ans_maph: float = (1.0 + 1.0) / 2.0

        # Filter objects
        dummy_estimated_objects = filter_objects(
            objects=self.dummy_estimated_objects,
            is_gt=False,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
        )
        dummy_ground_truth_objects = filter_objects(
            objects=self.dummy_ground_truth_objects,
            is_gt=True,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            min_point_numbers=self.min_point_numbers,
        )

        object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=dummy_estimated_objects,
            ground_truth_objects=dummy_ground_truth_objects,
        )
        object_results_dict: Dict[
            AutowareLabel, List[DynamicObjectWithPerceptionResult]
        ] = divide_objects(
            object_results,
            self.target_labels,
        )

        num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
            dummy_ground_truth_objects,
            self.target_labels,
        )

        map: Map = Map(
            object_results_dict=object_results_dict,
            num_ground_truth_dict=num_ground_truth_dict,
            target_labels=self.target_labels,
            matching_mode=MatchingMode.CENTERDISTANCE,
            matching_threshold_list=[1.0, 1.0, 1.0, 1.0],
        )
        self.assertAlmostEqual(map.map, ans_map)
        self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_iou_2d_translation_difference(self):
        """[summary]
        Test mAP and mAPH with iou 2d (bev) matching for translation difference.

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
                # Filter objects
                diff_distance_dummy_ground_truth_objects = filter_objects(
                    objects=diff_distance_dummy_ground_truth_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                dummy_ground_truth_objects = filter_objects(
                    objects=self.dummy_ground_truth_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    min_point_numbers=self.min_point_numbers,
                )

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=dummy_ground_truth_objects,
                )

                object_results_dict: Dict[
                    AutowareLabel, List[DynamicObjectWithPerceptionResult]
                ] = divide_objects(
                    object_results,
                    self.target_labels,
                )

                num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
                    dummy_ground_truth_objects,
                    self.target_labels,
                )

                map: Map = Map(
                    object_results_dict=object_results_dict,
                    num_ground_truth_dict=num_ground_truth_dict,
                    target_labels=self.target_labels,
                    matching_mode=MatchingMode.IOU2D,
                    matching_threshold_list=[0.5, 0.5, 0.5, 0.5],
                )
                self.assertAlmostEqual(map.map, ans_map)
                self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_iou_2d_yaw_difference(self):
        """[summary]
        Test mAP and mAPH with iou 2d (bev) matching for yaw difference.

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
                # Filter objects
                diff_yaw_dummy_ground_truth_objects = filter_objects(
                    objects=diff_yaw_dummy_ground_truth_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                dummy_ground_truth_objects = filter_objects(
                    objects=self.dummy_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    min_point_numbers=self.min_point_numbers,
                )

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=dummy_ground_truth_objects,
                )
                object_results_dict: Dict[
                    AutowareLabel, List[DynamicObjectWithPerceptionResult]
                ] = divide_objects(
                    object_results,
                    self.target_labels,
                )

                num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
                    dummy_ground_truth_objects,
                    self.target_labels,
                )

                map: Map = Map(
                    object_results_dict=object_results_dict,
                    num_ground_truth_dict=num_ground_truth_dict,
                    target_labels=self.target_labels,
                    matching_mode=MatchingMode.IOU2D,
                    matching_threshold_list=[0.5, 0.5, 0.5, 0.5],
                )
                self.assertAlmostEqual(map.map, ans_map)
                self.assertAlmostEqual(map.maph, ans_maph)

    def test_map_iou_2d_random_objects(self):
        """[summary]
        Test mAP and mAPH with iou 2d (bev) matching for random objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_objects with diff_distance (List[DynamicObject])

        test patterns:
            Check if map and maph are almost correct.
        """
        # PEDESTRIAN and MOTORBIKE is not included -> each AP is `inf` and skipped in mAP computation.
        ans_map: float = (1.0 + 0.0) / 2.0
        ans_maph: float = (1.0 + 0.0) / 2.0

        # Filter objects
        dummy_estimated_objects = filter_objects(
            objects=self.dummy_estimated_objects,
            is_gt=False,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
        )
        dummy_ground_truth_objects = filter_objects(
            objects=self.dummy_ground_truth_objects,
            is_gt=True,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            min_point_numbers=self.min_point_numbers,
        )
        object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=dummy_estimated_objects,
            ground_truth_objects=dummy_ground_truth_objects,
        )
        object_results_dict: Dict[
            AutowareLabel, List[DynamicObjectWithPerceptionResult]
        ] = divide_objects(
            object_results,
            self.target_labels,
        )

        num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
            dummy_ground_truth_objects,
            self.target_labels,
        )

        map: Map = Map(
            object_results_dict=object_results_dict,
            num_ground_truth_dict=num_ground_truth_dict,
            target_labels=self.target_labels,
            matching_mode=MatchingMode.IOU2D,
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
            with self.subTest("Test mAP and mAPH with iou 3d matching for translation difference."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                # Filter objects
                diff_distance_dummy_ground_truth_objects = filter_objects(
                    objects=diff_distance_dummy_ground_truth_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                dummy_ground_truth_objects = filter_objects(
                    objects=self.dummy_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    min_point_numbers=self.min_point_numbers,
                )

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=dummy_ground_truth_objects,
                )
                object_results_dict: Dict[
                    AutowareLabel, List[DynamicObjectWithPerceptionResult]
                ] = divide_objects(
                    object_results,
                    self.target_labels,
                )

                num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
                    dummy_ground_truth_objects,
                    self.target_labels,
                )

                map: Map = Map(
                    object_results_dict=object_results_dict,
                    num_ground_truth_dict=num_ground_truth_dict,
                    target_labels=self.target_labels,
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
                # Filter objects
                diff_yaw_dummy_ground_truth_objects = filter_objects(
                    objects=diff_yaw_dummy_ground_truth_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                dummy_ground_truth_objects = filter_objects(
                    objects=self.dummy_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    min_point_numbers=self.min_point_numbers,
                )

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=dummy_ground_truth_objects,
                )
                object_results_dict: Dict[
                    AutowareLabel, List[DynamicObjectWithPerceptionResult]
                ] = divide_objects(
                    object_results,
                    self.target_labels,
                )

                num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
                    dummy_ground_truth_objects,
                    self.target_labels,
                )

                map: Map = Map(
                    object_results_dict=object_results_dict,
                    num_ground_truth_dict=num_ground_truth_dict,
                    target_labels=self.target_labels,
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
        # PEDESTRIAN and MOTORBIKE is not included -> each AP is `inf` and skipped in mAP computation.
        ans_map: float = (1.0 + 0.0) / 2.0
        ans_maph: float = (1.0 + 0.0) / 2.0

        # Filter objects
        dummy_estimated_objects = filter_objects(
            objects=self.dummy_estimated_objects,
            is_gt=False,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
        )
        dummy_ground_truth_objects = filter_objects(
            objects=self.dummy_ground_truth_objects,
            is_gt=True,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            min_point_numbers=self.min_point_numbers,
        )

        object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=dummy_estimated_objects,
            ground_truth_objects=dummy_ground_truth_objects,
        )

        object_results_dict: Dict[
            AutowareLabel, List[DynamicObjectWithPerceptionResult]
        ] = divide_objects(
            object_results,
            self.target_labels,
        )

        num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
            dummy_ground_truth_objects,
            self.target_labels,
        )

        map: Map = Map(
            object_results_dict=object_results_dict,
            num_ground_truth_dict=num_ground_truth_dict,
            target_labels=self.target_labels,
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
                # Filter objects
                diff_distance_dummy_ground_truth_objects = filter_objects(
                    objects=diff_distance_dummy_ground_truth_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                dummy_ground_truth_objects = filter_objects(
                    objects=self.dummy_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    min_point_numbers=self.min_point_numbers,
                )

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=dummy_ground_truth_objects,
                )

                object_results_dict: Dict[
                    AutowareLabel, List[DynamicObjectWithPerceptionResult]
                ] = divide_objects(
                    object_results,
                    self.target_labels,
                )

                num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
                    dummy_ground_truth_objects,
                    self.target_labels,
                )

                map: Map = Map(
                    object_results_dict=object_results_dict,
                    num_ground_truth_dict=num_ground_truth_dict,
                    target_labels=self.target_labels,
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
            (-math.pi, 0.0, 0.0),
            # Given diff_yaw is pi/4, maph is 0.75**2 times map
            (math.pi / 4, 1.0, 0.5625),
            (-math.pi / 4, 1.0, 0.5625),
            # Given diff_yaw is 3*pi/4, maph is 0.25**2 times map
            (3 * math.pi / 4, 1.0, 0.0625),
            (-3 * math.pi / 4, 1.0, 0.0625),
        ]

        for diff_yaw, ans_map, ans_maph in patterns:
            with self.subTest("Test mAP and mAPH with plane distance matching for yaw difference."):
                diff_yaw_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=diff_yaw,
                )
                # Filter objects
                diff_yaw_dummy_ground_truth_objects = filter_objects(
                    objects=diff_yaw_dummy_ground_truth_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                dummy_ground_truth_objects = filter_objects(
                    objects=self.dummy_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                    min_point_numbers=self.min_point_numbers,
                )
                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=dummy_ground_truth_objects,
                )

                object_results_dict: Dict[
                    AutowareLabel, List[DynamicObjectWithPerceptionResult]
                ] = divide_objects(
                    object_results,
                    self.target_labels,
                )
                num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
                    dummy_ground_truth_objects,
                    self.target_labels,
                )

                map: Map = Map(
                    object_results_dict=object_results_dict,
                    num_ground_truth_dict=num_ground_truth_dict,
                    target_labels=self.target_labels,
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
        # dummy_estimated_objects[0] (CAR) and dummy_ground_truth_objects[0] (CAR):
        #   pr_corner_points(left, right) = [(0.25, 0.25, 1.0), (0.25, 1.75, 1.0)]
        #   gt_corner_points(left, right) = [(0.5, 0.5, 1.0), (0.5, 1.5, 1.0)]
        #   plane_distance = 0.3535533905932738
        #
        # dummy_estimated_objects[1] (BICYCLE) and dummy_ground_truth_objects[1] (BICYCLE):
        #   pr_corner_points(left, right) = [(1.25, -0.75, 1.0), (0.75, -0.75, 1.0)]
        #   gt_corner_points(left, right) = [(1.5, -0.5, 1.0), (0.5, -0.5, 1.0)]
        #   plane_distance = 0.3535533905932738
        #
        # dummy_estimated_objects[2] (PEDESTRIAN) and dummy_ground_truth_objects[2] (CAR):
        #   pr_corner_points(left, right) = [(-0.5, 0.5, 1.0), (-0.5, 1.5, 1.0)]
        #   gt_corner_points(left, right) = [(-0.5, 0.5, 1.0), (-0.5, 1.5, 1.0)]
        #   plane_distance = 0.0

        # CAR: under the threshold
        #   ap and aph: 1.0
        # BICYCLE: under the threshold
        #   ap and aph: 1.0
        # PEDESTRIAN: under the threshold but label does not match
        #   ap and aph: 0.0
        # MOTORBIKE: not estimated
        #   ap and aph: 0.0
        # PEDESTRIAN and MOTORBIKE is not included -> each AP is `inf` and skipped in mAP computation.
        ans_map: float = (1.0 + 1.0) / 2.0
        ans_maph: float = (1.0 + 1.0) / 2.0

        # Filter objects
        dummy_estimated_objects = filter_objects(
            objects=self.dummy_estimated_objects,
            is_gt=False,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
        )
        dummy_ground_truth_objects = filter_objects(
            objects=self.dummy_ground_truth_objects,
            is_gt=True,
            target_labels=self.target_labels,
            max_x_position_list=self.max_x_position_list,
            max_y_position_list=self.max_y_position_list,
            min_point_numbers=self.min_point_numbers,
        )

        object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=dummy_estimated_objects,
            ground_truth_objects=dummy_ground_truth_objects,
        )
        object_results_dict: Dict[
            AutowareLabel, List[DynamicObjectWithPerceptionResult]
        ] = divide_objects(
            object_results,
            self.target_labels,
        )

        num_ground_truth_dict: Dict[AutowareLabel, int] = divide_objects_to_num(
            dummy_ground_truth_objects,
            self.target_labels,
        )

        map: Map = Map(
            object_results_dict=object_results_dict,
            num_ground_truth_dict=num_ground_truth_dict,
            target_labels=self.target_labels,
            matching_mode=MatchingMode.PLANEDISTANCE,
            matching_threshold_list=[1.0, 1.0, 1.0, 1.0],
        )

        self.assertAlmostEqual(map.map, ans_map)
        self.assertAlmostEqual(map.maph, ans_maph)


if __name__ == "__main__":
    unittest.main()
