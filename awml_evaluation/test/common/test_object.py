import math
from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple
import unittest

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.object import distance_objects
from awml_evaluation.common.object import distance_objects_bev
from awml_evaluation.util.debug import get_objects_with_difference
from shapely.geometry import Polygon


class TestObject(unittest.TestCase):
    def setUp(self):
        self.dummy_predicted_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_predicted_objects, self.dummy_ground_truth_objects = make_dummy_data()

    def test_get_distance(self):
        """[summary]
        Test getting the 3d distance to the dummy_ground_truth_object from ego vehicle.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])

        test patterns:
            Test if the 3d distance is almost equal to ans_distance.
        """
        # patterns: (ans_distance)
        patterns: List[float] = [
            # 3d_distance to the dummy_ground_truth_object from ego vehicle.
            math.sqrt(3 * 1.0**2),
            math.sqrt(3 * 1.0**2),
            math.sqrt(3 * 1.0**2),
            math.sqrt(3 * 1.0**2),
        ]
        for ans_distance in patterns:
            with self.subTest("Test get_distance"):
                for dummy_ground_truth_object in self.dummy_ground_truth_objects:
                    distance = dummy_ground_truth_object.get_distance()
                    self.assertAlmostEqual(distance, ans_distance)

    def test_get_distance_bev(self):
        """[summary]
        Test getting the 2d distance to the dummy_ground_truth_object from ego vehicle.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])

        test patterns:
            Test if the 2d distance is almost equal to ans_distance_bev.
        """
        # patterns: (ans_distance_bev)
        patterns: List[float] = [
            math.sqrt(2 * 1.0**2),
            math.sqrt(2 * 1.0**2),
            math.sqrt(2 * 1.0**2),
            math.sqrt(2 * 1.0**2),
        ]
        for ans_distance_bev in patterns:
            with self.subTest("Test get_distance_bev"):
                for dummy_ground_truth_object in self.dummy_ground_truth_objects:
                    distance_bev = dummy_ground_truth_object.get_distance_bev()
                    self.assertAlmostEqual(distance_bev, ans_distance_bev)

    def test_get_heading_bev(self):
        """[summary]
        Test getting the object heading from ego vehicle in bird eye view.

        test objects:
            dummy_ground_truth_object with diff_yaw (List[DynamicObject])

        test patterns:
            Test if the object heading is almost equal to ans_heading_bev.
        """
        # patterns: (diff_yaw, ans_heading_bev)
        patterns: List[Tuple[float, float]] = [
            (0.0, math.pi / 2),
            (math.pi / 2, 0.0),
            (-math.pi / 2, -math.pi),
            (math.pi, -math.pi / 2),
            (-math.pi, -math.pi / 2),
        ]
        for diff_yaw, ans_heading_bev in patterns:
            with self.subTest("Test get_heading_bev"):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=diff_yaw,
                )
                for diff_distance_dummy_object in diff_distance_dummy_ground_truth_objects:
                    heading_bev = diff_distance_dummy_object.get_heading_bev()
                    self.assertAlmostEqual(heading_bev, ans_heading_bev)

    def test_get_footprint(self):
        """[summary]
        Test getting footprint polygon from an object.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])

        test patterns:
            Test if the object footprint polygon is equal to ans_polygon.
        """
        # patterns: (ans_polygon)
        patterns: List[Polygon] = [
            Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)]),
            Polygon([(0.5, -1.5), (1.5, -1.5), (1.5, -0.5), (0.5, -0.5), (0.5, -1.5)]),
            Polygon([(-1.5, 0.5), (-0.5, 0.5), (-0.5, 1.5), (-1.5, 1.5), (-1.5, 0.5)]),
            Polygon([(-1.5, -1.5), (-0.5, -1.5), (-0.5, -0.5), (-1.5, -0.5), (-1.5, -1.5)]),
        ]
        with self.subTest("Test get_footprint"):
            for dummy_object, ans_polygon in zip(self.dummy_ground_truth_objects, patterns):
                footprint = dummy_object.get_footprint()
                self.assertTrue(footprint.equals(ans_polygon))

    def test_get_area_bev(self):
        """[summary]
        Test getting the 2d area of the dummy_ground_truth_objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])

        test patterns:
            Test if the 2d area is almost equal to ans_area_bev.
        """
        # patterns: (ans_area_bev)
        patterns: List[float] = [
            math.sqrt(1.0 * 1.0),
            math.sqrt(1.0 * 1.0),
            math.sqrt(1.0 * 1.0),
            math.sqrt(1.0 * 1.0),
        ]
        for ans_area_bev in patterns:
            with self.subTest("Test get_area_bev"):
                for dummy_ground_truth_object in self.dummy_ground_truth_objects:
                    area_bev = dummy_ground_truth_object.get_area_bev()
                    self.assertAlmostEqual(area_bev, ans_area_bev)

    def test_get_volume(self):
        """[summary]
        Test getting the volume of the dummy_ground_truth_objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])

        test patterns:
            Test if the volume is almost equal to ans_volume.
        """
        # patterns: (ans_volume)
        patterns: List[float] = [
            math.sqrt(1.0**3),
            math.sqrt(1.0**3),
            math.sqrt(1.0**3),
            math.sqrt(1.0**3),
        ]
        for ans_volume in patterns:
            with self.subTest("Test get_volume"):
                for dummy_ground_truth_object in self.dummy_ground_truth_objects:
                    volume = dummy_ground_truth_object.get_volume()
                    self.assertAlmostEqual(volume, ans_volume)

    def test_distance_objects(self):
        """[summary]
        Test calculating the 3d center distance between two objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_object with diff_distance (List[DynamicObject])

        test patterns:
            Test if distace is almost equal to ans_distance.
        """
        # patterns: (diff_distance, ans_distance)
        patterns: List[Tuple[float, float]] = [
            (0.0, 0.0),
            (2.0, math.sqrt(3 * 2.0**2)),
            (-2.0, math.sqrt(3 * 2.0**2)),
        ]
        for diff_distance, ans_distance in patterns:
            with self.subTest("Test distance_objects"):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, diff_distance, diff_distance),
                    diff_yaw=0,
                )
                for dummy_object, diff_dummy_object in zip(
                    self.dummy_ground_truth_objects, diff_distance_dummy_ground_truth_objects
                ):
                    distance = distance_objects(dummy_object, diff_dummy_object)
                    self.assertAlmostEqual(distance, ans_distance)

    def test_distance_objects_bev(self):
        """[summary]
        Test calculating the 2d center distance between two objects.

        test objects:
            dummy_ground_truth_objects (List[DynamicObject])
            dummy_ground_truth_object with diff_distance (List[DynamicObject])

        test patterns:
            Test if 2d distace is almost equal to ans_distance_bev.
        """
        # patterns: (diff_distance, ans_distance_bev)
        patterns: List[Tuple[float, float]] = [
            (0.0, 0.0),
            (2.0, math.sqrt(2 * 2.0**2)),
            (-2.0, math.sqrt(2 * 2.0**2)),
        ]
        for diff_distance, ans_distance_bev in patterns:
            with self.subTest("Test distance_objects_bev"):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, diff_distance, diff_distance),
                    diff_yaw=0,
                )
                for dummy_object, diff_dummy_object in zip(
                    self.dummy_ground_truth_objects, diff_distance_dummy_ground_truth_objects
                ):
                    distance_bev = distance_objects_bev(dummy_object, diff_dummy_object)
                    self.assertAlmostEqual(distance_bev, ans_distance_bev)


if __name__ == "__main__":
    unittest.main()
