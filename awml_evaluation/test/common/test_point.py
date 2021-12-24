import math
from typing import List
from typing import Tuple
import unittest

from awml_evaluation.common.point import distance_points, distance_points_bev, to_bev


class TestPoint(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_distance_points(self):
        """[summary]
        Test calculating the 3d center distance between two points.

        test patterns:
            Test if the 3d distance between point_1 and point_2 is
            almost equal to ans_distance.
        """
        # patterns: (point_1, point_2, ans_distance)
        patterns: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], float]] = [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), math.sqrt(3*1.0**2)),
            ((0.0, 0.0, 0.0), (1.0, -1.0, 1.0), math.sqrt(3*1.0**2)),
            ((0.0, 0.0, 0.0), (-1.0, -1.0, 1.0), math.sqrt(3*1.0**2)),
            ((0.0, 0.0, 0.0), (-1.0, -1.0, -1.0), math.sqrt(3*1.0**2)),
        ]
        for point_1, point_2, ans_distance in patterns:
            with self.subTest("Test get_distance_points"):
                distance = distance_points(point_1, point_2)
                self.assertAlmostEqual(distance, ans_distance)

    def test_get_distance_points_bev(self):
        """[summary]
        Test calculating the 2d center distance between two points.

        test patterns:
            Test if the 2d distance between point_1 and point_2 is
            almost equal to ans_distance.
        """
        # patterns: (point_1, point_2, ans_distance)
        patterns: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], float]] = [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), math.sqrt(2*1.0**2)),
            ((0.0, 0.0, 0.0), (1.0, -1.0, 1.0), math.sqrt(2*1.0**2)),
            ((0.0, 0.0, 0.0), (-1.0, -1.0, 5.0), math.sqrt(2*1.0**2)),
            ((0.0, 0.0, 0.0), (-1.0, -1.0, -10.0), math.sqrt(2*1.0**2)),
        ]
        for point_1, point_2, ans_distance in patterns:
            with self.subTest("Test get_distance_points_bev"):
                distance = distance_points_bev(point_1, point_2)
                self.assertAlmostEqual(distance, ans_distance)

    def test_to_bev(self):
        """[summary]
        Test (x, y, z) -> (x, y)

        test patterns:
            Test if 3d point_1 is converted to 2d point_1 (=ans_point).
        """
        # patterns: (point_1,  ans_point)
        patterns: List[Tuple[Tuple[float, float, float], Tuple[float, float]]] = [
            ((0.0, 0.0, 0.0), (0.0, 0.0)),
            ((1.0, 0.0, 1.0), (1.0, 0.0)),
            ((0.0, -1.0, 5.0), (0.0, -1.0))
        ]
        for point_1, ans_point in patterns:
            with self.subTest("Test to_bev"):
                point = to_bev(point_1)
                self.assertEqual(point, ans_point)


if __name__ == "__main__":
    unittest.main()
