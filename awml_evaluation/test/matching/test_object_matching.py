import math
from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple
import unittest

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import CenterDistanceMatching
from awml_evaluation.evaluation.matching.object_matching import IOU3dMatching
from awml_evaluation.evaluation.matching.object_matching import IOUBEVMatching
from awml_evaluation.evaluation.matching.object_matching import PlaneDistanceMatching
from awml_evaluation.evaluation.matching.object_matching import _get_area_intersection
from awml_evaluation.evaluation.matching.object_matching import _get_height_intersection
from awml_evaluation.evaluation.matching.object_matching import _get_volume_intersection
from awml_evaluation.util.debug import get_objects_with_difference


class TestObjectMatching(unittest.TestCase):
    def setUp(self):
        self.dummy_predicted_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_predicted_objects, self.dummy_ground_truth_objects = make_dummy_data()

    def test_get_area_intersection(self):
        """[summary]
        Test getting the area at intersection.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if area_intersection and ans_area_intersection are equal.
        """
        # patterns: (diff_distance, List[ans_area_intersection])
        patterns: List[Tuple[float, float]] = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        for diff_distance, ans_area_intersection in patterns:
            with self.subTest("Test get_area_intersection."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                for predicted_object, ground_truth_object in zip(
                    diff_distance_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    area_intersection = _get_area_intersection(
                        predicted_object, ground_truth_object
                    )
                    self.assertAlmostEqual(area_intersection, ans_area_intersection)

    def test_get_height_intersection(self):
        """[summary]
        Test getting the height at intersection.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if height_intersection and ans_height_intersection
            are equal.
        """
        # patterns: (diff_distance, List[ans_height_intersection])
        patterns: List[Tuple[float, float]] = [(0.0, 1.0), (-0.5, 0.5), (-1.0, 0.0)]
        for diff_distance, ans_height_intersection in patterns:
            with self.subTest("Test get_height_intersection."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, diff_distance),
                    diff_yaw=0,
                )
                for predicted_object, ground_truth_object in zip(
                    diff_distance_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    height_intersection = _get_height_intersection(
                        predicted_object,
                        ground_truth_object,
                    )
                    self.assertAlmostEqual(height_intersection, ans_height_intersection)

    def test_volume_intersection(self):
        """[summary]
        Test getting the volume at intersection.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if intersection and ans_intersection are equal.
        """
        # patterns: (diff_distance, List[ans_intersection])
        patterns: List[Tuple[float, float]] = [(0.0, 1.0), (0.5, 0.125), (1.5, 0.0)]
        for diff_distance, ans_intersection in patterns:
            with self.subTest("Test get_intersection."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, diff_distance, diff_distance),
                    diff_yaw=0,
                )
                for predicted_object, ground_truth_object in zip(
                    diff_distance_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    intersection = _get_volume_intersection(predicted_object, ground_truth_object)
                    self.assertAlmostEqual(intersection, ans_intersection)

    def test_plane_distance_matching(self):
        """[summary]
        Test calculating plane distance for use case evaluation.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if plane_distance and ans_plane_distance are equal.
        """
        # patterns: (diff_distance, List[ans_plane_distance])
        patterns: List[Tuple[float, List[float]]] = [
            (0.0, [0.0, 0.0, 0.0, 0.0]),
            (1.0, [1.0, 1.0, 1.0, 1.0]),
            # (2.0, [2.0, 2.0, (1.0+math.sqrt(5.0))/2.0, (1.0+math.sqrt(5.0))/2.0]) # previous version
            (
                2.0,
                [2.0, 2.0, math.sqrt((1.0 + 5.0) / 2.0), math.sqrt((1.0 + 5.0) / 2.0)],
            ),  # new version
        ]
        fixed_dummy_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
            ground_truth_objects=self.dummy_ground_truth_objects,
            # Shift all objects a bit to exclude the effect of numerical error.
            diff_distance=(0.01, 0.0, 0.0),
            diff_yaw=0,
        )
        for diff_distance, ans_plane_distance_list in patterns:
            with self.subTest("Test get_uc_plane_distance."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=fixed_dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                for predicted_object, ground_truth_object, ans_plane_distance in zip(
                    diff_distance_dummy_ground_truth_objects,
                    fixed_dummy_ground_truth_objects,
                    ans_plane_distance_list,
                ):
                    plane_distance = PlaneDistanceMatching(
                        predicted_object,
                        ground_truth_object,
                    )
                    self.assertAlmostEqual(
                        plane_distance.value,
                        ans_plane_distance,
                    )

    def test_iou_bev_matching(self):
        """[summary]
        Test calculating BEV IoU.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if iou_bev and ans_iou_bev are equal.
        """
        # patterns: (diff_distance, List[ans_iou_bev])
        patterns: List[Tuple[float, float]] = [(0.0, 1.0), (0.5, 1.0 / 3.0), (1.0, 0.0)]
        for diff_distance, ans_iou_bev in patterns:
            with self.subTest("Test get_iou_bev."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                for predicted_object, ground_truth_object in zip(
                    diff_distance_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    iou_bev = IOUBEVMatching(predicted_object, ground_truth_object)
                    self.assertAlmostEqual(iou_bev.value, ans_iou_bev)

    def test_iou_3d_matching(self):
        """[summary]
        Test calculating 3d IoU.

        test objects:
            dummy_ground_truth_objects

        test patterns:
            Given diff_distance, check if iou_3d and ans_iou_3d are equal.
        """
        # patterns: (diff_distance, List[ans_iou_3d])
        patterns: List[Tuple[float, float]] = [(0.0, 1.0), (-0.5, 1.0 / 3.0), (-1.0, 0.0)]
        for diff_distance, ans_iou_3d in patterns:
            with self.subTest("Test get_iou_3d."):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                for predicted_object, ground_truth_object in zip(
                    diff_distance_dummy_ground_truth_objects, self.dummy_ground_truth_objects
                ):
                    iou_3d = IOU3dMatching(predicted_object, ground_truth_object)
                    self.assertAlmostEqual(iou_3d.value, ans_iou_3d)


if __name__ == "__main__":
    unittest.main()
