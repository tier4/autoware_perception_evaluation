# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
import unittest

import numpy as np
from perception_eval.evaluation.metrics.detection.geometry import bev_corners
from perception_eval.evaluation.metrics.detection.geometry import bev_corners_batch
from perception_eval.evaluation.metrics.detection.geometry import clip_forward
from perception_eval.evaluation.metrics.detection.geometry import corner_displacement
from perception_eval.evaluation.metrics.detection.geometry import corner_displacement_matrix
from perception_eval.evaluation.metrics.detection.geometry import corner_displacements
from perception_eval.evaluation.metrics.detection.geometry import nearest_surface_distance
from perception_eval.evaluation.metrics.detection.geometry import nearest_surface_distances
from perception_eval.evaluation.metrics.detection.geometry import points_in_bev_box
from perception_eval.evaluation.metrics.detection.geometry import signed_nearest_surface_error
from perception_eval.evaluation.metrics.detection.geometry import validate_boxes
from perception_eval.evaluation.metrics.detection.geometry import wrap_angle


def _box(cx=10.0, cy=0.0, dx=4.0, dy=2.0, yaw=0.0) -> np.ndarray:
    return np.array([cx, cy, 0.0, dx, dy, 1.5, yaw, 0.0, 0.0], dtype=np.float64)


class TestGeometry(unittest.TestCase):
    def test_bev_corners_axis_aligned(self):
        corners = bev_corners(_box(cx=0.0, cy=0.0, dx=2.0, dy=2.0, yaw=0.0))
        expected = {(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)}
        self.assertEqual({tuple(np.round(c, 6)) for c in corners}, expected)
        np.testing.assert_allclose(bev_corners_batch(np.stack([_box(), _box(cx=1.0)]))[0], bev_corners(_box()))
        self.assertEqual(bev_corners_batch(np.zeros((0, 9))).shape, (0, 4, 2))

    def test_corner_displacement_identity_and_translation(self):
        self.assertAlmostEqual(corner_displacement(_box(), _box()), 0.0)
        # Pure 1 m translation shifts every corner by exactly 1 m.
        self.assertAlmostEqual(corner_displacement(_box(cx=11.0), _box(cx=10.0)), 1.0)
        # A 180-degree flip of a box is forgiven by the cyclic assignment.
        self.assertAlmostEqual(corner_displacement(_box(yaw=pi), _box()), 0.0)
        np.testing.assert_allclose(corner_displacements(np.stack([_box(cx=11.0)]), np.stack([_box()])), [1.0])
        matrix = corner_displacement_matrix(np.stack([_box(cx=11.0), _box()]), np.stack([_box()]))
        np.testing.assert_allclose(matrix, [[1.0], [0.0]])
        self.assertEqual(corner_displacement_matrix(np.zeros((0, 9)), np.zeros((2, 9))).shape, (0, 2))

    def test_nearest_surface_distance_and_sign(self):
        # Box spans x in [8, 12], y in [-1, 1], nearest face to the origin is x = 8.
        self.assertAlmostEqual(nearest_surface_distance(_box()), 8.0)
        # Predicting the box 1 m farther => +1 m signed near-face error (brakes late).
        self.assertAlmostEqual(signed_nearest_surface_error(_box(cx=11.0), _box(cx=10.0)), 1.0)
        # Origin inside the footprint is well defined as 0.
        self.assertAlmostEqual(nearest_surface_distance(_box(cx=0.0, cy=0.0)), 0.0)
        np.testing.assert_allclose(nearest_surface_distances(np.stack([_box(), _box(cx=0.0)])), [8.0, 0.0])
        # A rotated box: 2 m wide box centered 10 m ahead rotated 90 deg -> nearest face at 10 - 1.
        self.assertAlmostEqual(nearest_surface_distance(_box(yaw=pi / 2)), 9.0)

    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(3 * pi), -pi)
        self.assertAlmostEqual(wrap_angle(0.5), 0.5)

    def test_points_in_bev_box_is_yaw_aware(self):
        box = _box(cx=0.0, cy=0.0, dx=4.0, dy=2.0, yaw=pi / 2)
        # Under yaw pi/2 the long axis lies along y.
        np.testing.assert_array_equal(
            points_in_bev_box(np.array([[0.0, 1.9], [1.9, 0.0], [0.0, 0.0]]), box), [True, False, True]
        )
        self.assertEqual(points_in_bev_box(np.zeros((0, 2)), box).shape, (0,))

    def test_clip_forward(self):
        clipped = clip_forward(bev_corners(_box(cx=-1.0, cy=0.0, dx=4.0, dy=2.0)))
        self.assertTrue((clipped[:, 0] >= -1e-9).all())
        self.assertAlmostEqual(clipped[:, 0].max(), 1.0)
        self.assertEqual(clip_forward(bev_corners(_box(cx=-10.0))).shape, (0, 2))

    def test_validate_boxes(self):
        with self.assertRaises(ValueError):
            validate_boxes(np.zeros((2, 6)))
        self.assertEqual(validate_boxes(np.zeros((2, 7))).dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
