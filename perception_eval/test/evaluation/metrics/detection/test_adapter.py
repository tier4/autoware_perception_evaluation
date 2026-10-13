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
from test.util.dummy_object import make_dummy_data
import unittest

import numpy as np
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.shape import Shape
from perception_eval.common.shape import ShapeType
from perception_eval.common.transform import HomogeneousMatrix
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.metrics.detection.adapter import class_names_of
from perception_eval.evaluation.metrics.detection.adapter import label_index
from perception_eval.evaluation.metrics.detection.adapter import MissingTransformError
from perception_eval.evaluation.metrics.detection.adapter import object_score
from perception_eval.evaluation.metrics.detection.adapter import object_to_base_link_row
from perception_eval.evaluation.metrics.detection.adapter import objects_to_arrays
from perception_eval.evaluation.metrics.detection.adapter import objects_to_box_rows
from perception_eval.evaluation.metrics.detection.adapter import UnsupportedShapeError
from pyquaternion import Quaternion
from shapely.geometry import Polygon


def _object(
    frame_id=FrameID.BASE_LINK, position=(10.0, 2.0, 0.5), yaw=0.3, size=(1.5, 4.0, 1.7), velocity=None, shape=None
):
    return DynamicObject(
        unix_time=0,
        frame_id=frame_id,
        position=position,
        orientation=Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw),
        shape=shape if shape is not None else Shape(ShapeType.BOUNDING_BOX, size),
        velocity=velocity,
        semantic_score=0.7,
        semantic_label=Label(AutowareLabel.CAR, "car"),
    )


class TestAdapter(unittest.TestCase):
    def test_size_convention_is_swapped_to_length_width(self):
        row = object_to_base_link_row(_object(), None)
        np.testing.assert_allclose(row[:3], [10.0, 2.0, 0.5])
        self.assertAlmostEqual(row[3], 4.0)  # dx = length
        self.assertAlmostEqual(row[4], 1.5)  # dy = width
        self.assertAlmostEqual(row[5], 1.7)
        self.assertAlmostEqual(row[6], 0.3)
        np.testing.assert_allclose(row[7:], [0.0, 0.0])

    def test_velocity_is_rotated_from_object_frame(self):
        row = object_to_base_link_row(_object(yaw=pi / 2, velocity=(1.0, 0.0, 0.0)), None)
        np.testing.assert_allclose(row[7:], [0.0, 1.0], atol=1e-12)

    def test_map_frame_object_is_transformed_to_base_link(self):
        ego2map = HomogeneousMatrix(
            (10.0, 0.0, 0.0), Quaternion(axis=[0, 0, 1], angle=pi / 2), src=FrameID.BASE_LINK, dst=FrameID.MAP
        )
        obj = _object(frame_id=FrameID.MAP, position=(11.0, 2.0, 0.0), yaw=0.3 + pi / 2)
        row = object_to_base_link_row(obj, TransformDict([ego2map]))
        np.testing.assert_allclose(row[:2], [2.0, -1.0], atol=1e-9)
        self.assertAlmostEqual(row[6], 0.3)

    def test_errors(self):
        with self.assertRaises(MissingTransformError):
            object_to_base_link_row(_object(frame_id=FrameID.MAP), None)
        with self.assertRaises(MissingTransformError):
            object_to_base_link_row(_object(frame_id=FrameID.MAP), TransformDict())
        polygon = Shape(ShapeType.POLYGON, (0.0, 0.0, 1.0), footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
        with self.assertRaises(UnsupportedShapeError):
            object_to_base_link_row(_object(shape=polygon), None)

    def test_objects_to_arrays_follows_target_label_order(self):
        estimated, ground_truth = make_dummy_data()
        target_labels = [AutowareLabel.CAR, AutowareLabel.BICYCLE, AutowareLabel.PEDESTRIAN, AutowareLabel.MOTORBIKE]
        boxes, labels, scores = objects_to_arrays(ground_truth, None, target_labels)
        self.assertEqual(boxes.shape, (4, 9))
        self.assertEqual(labels.tolist(), [0, 1, 2, 3])
        np.testing.assert_allclose(scores, 1.0)
        # Non-target labels are dropped.
        boxes, labels, _ = objects_to_arrays(ground_truth, None, [AutowareLabel.CAR])
        self.assertEqual(boxes.shape, (1, 9))
        self.assertEqual(labels.tolist(), [0])
        self.assertEqual(objects_to_box_rows([], None).shape, (0, 9))
        empty_boxes, empty_labels, empty_scores = objects_to_arrays([], None, target_labels)
        self.assertEqual((empty_boxes.shape, empty_labels.shape, empty_scores.shape), ((0, 9), (0,), (0,)))

    def test_label_helpers(self):
        self.assertEqual(
            class_names_of([AutowareLabel.CAR, AutowareLabel.HAZARD, AutowareLabel.MOTORBIKE]),
            ("car", "hazard", "motorbike"),
        )
        self.assertEqual(label_index(Label(AutowareLabel.HAZARD, "cone"), [AutowareLabel.CAR, AutowareLabel.HAZARD]), 1)
        self.assertEqual(label_index(Label(AutowareLabel.BUS, "bus"), [AutowareLabel.CAR]), -1)


if __name__ == "__main__":
    unittest.main()


class TestAdapterInputPolicies(unittest.TestCase):
    @staticmethod
    def _polygon(yaw=0.0, footprint=((1.0, -0.5), (3.0, -0.5), (3.0, 1.5), (1.0, 1.5))):
        return _object(
            position=(10.0, 0.0, 0.5),
            yaw=yaw,
            shape=Shape(ShapeType.POLYGON, (0.0, 0.0, 1.2), footprint=Polygon(footprint)),
        )

    def test_polygon_is_rejected_by_default(self):
        with self.assertRaises(UnsupportedShapeError):
            object_to_base_link_row(self._polygon(), None)

    def test_polygon_as_box_uses_footprint_extent_and_offset(self):
        counts = {}
        row = object_to_base_link_row(self._polygon(), None, polygon_shapes="as_box", counts=counts)
        # Footprint spans x in [1, 3] and y in [-0.5, 1.5] around the object origin.
        np.testing.assert_allclose(row[:3], [12.0, 0.5, 0.5])
        self.assertAlmostEqual(row[3], 2.0)  # dx = length (x extent)
        self.assertAlmostEqual(row[4], 2.0)  # dy = width (y extent)
        self.assertAlmostEqual(row[5], 1.2)
        self.assertEqual(counts, {"polygon_as_box": 1})

    def test_polygon_as_box_offset_follows_object_yaw(self):
        row = object_to_base_link_row(self._polygon(yaw=pi / 2), None, polygon_shapes="as_box")
        # The (+2, +0.5) footprint centre rotates by +90 deg into (-0.5, +2).
        np.testing.assert_allclose(row[:2], [9.5, 2.0], atol=1e-9)
        self.assertAlmostEqual(row[6], pi / 2)

    def test_score_source(self):
        obj = _object()
        self.assertAlmostEqual(object_score(obj), 0.7)
        self.assertTrue(np.isnan(object_score(obj, "existence_probability")))
        obj.existence_probability = 0.4
        self.assertAlmostEqual(object_score(obj, "existence_probability"), 0.4)
        with self.assertRaises(ValueError):
            object_score(obj, "classification")

    def test_non_finite_scores_are_dropped_and_counted(self):
        good, bad = _object(), _object()
        bad.semantic_score = float("nan")
        counts = {}
        _, labels, scores = objects_to_arrays(
            [good, bad], None, [AutowareLabel.CAR], counts=counts, drop_non_finite_scores=True
        )
        self.assertEqual(labels.tolist(), [0])
        np.testing.assert_allclose(scores, [0.7])
        self.assertEqual(counts, {"non_finite_score": 1})
        # Without the flag the scores are passed through unchanged (ground truth path).
        _, _, scores = objects_to_arrays([good, bad], None, [AutowareLabel.CAR])
        self.assertEqual(len(scores), 2)
