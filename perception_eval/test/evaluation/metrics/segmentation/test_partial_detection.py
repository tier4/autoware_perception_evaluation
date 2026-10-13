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

import math
from math import pi
from test.evaluation.metrics.segmentation.helpers import CLASS_NAMES
from test.evaluation.metrics.segmentation.helpers import config
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
from perception_eval.evaluation.metrics.segmentation.boxes import gt_boxes_from_objects
from perception_eval.evaluation.metrics.segmentation.partial_detection import PartialDetectionScore
from perception_eval.evaluation.metrics.segmentation.state import RAW_TAXONOMY
from perception_eval.evaluation.metrics.segmentation.state import SegmentationFrame
from perception_eval.evaluation.metrics.segmentation.state import SegmentationView
from perception_eval.evaluation.metrics.segmentation.suite import SegmentationMetricSuite
from pyquaternion import Quaternion


def _object(
    x: float,
    y: float,
    label: AutowareLabel = AutowareLabel.PEDESTRIAN,
    yaw: float = 0.0,
    size=(2.0, 2.0, 2.0),
    frame_id=FrameID.BASE_LINK,
):
    return DynamicObject(
        unix_time=0,
        frame_id=frame_id,
        position=(x, y, 0.0),
        orientation=Quaternion(axis=[0, 0, 1], angle=yaw),
        shape=Shape(ShapeType.BOUNDING_BOX, size),  # (width, length, height)
        velocity=None,
        semantic_score=1.0,
        semantic_label=Label(label, label.value),
    )


def _view(coord, pred, boxes, seg_class, det_label, taxonomy=RAW_TAXONOMY) -> SegmentationView:
    n = coord.shape[0]
    return SegmentationView(
        taxonomy=taxonomy,
        filter_name="",
        range_name=None,
        frame_name="0",
        pred=np.asarray(pred, dtype=np.int64),
        target=np.ones(n, dtype=np.int64),
        confidence=np.full(n, 0.5),
        entropy=np.full(n, 1.0),
        xyz=np.asarray(coord, dtype=np.float64),
        gt_boxes=np.asarray(boxes, dtype=np.float64),
        gt_box_seg_class=np.asarray(seg_class, dtype=np.int64),
        gt_box_det_label=np.asarray(det_label, dtype=np.int64),
        num_classes=2,
        class_names=CLASS_NAMES,
    )


def _box_row(x, y, dx, dy, yaw=0.0):
    return [x, y, 0.0, dx, dy, 2.0, yaw, 0.0, 0.0]


class TestPartialDetectionScore(unittest.TestCase):
    def test_credit_landmarks(self):
        metric = PartialDetectionScore(det_class_names=("ped",))
        self.assertAlmostEqual(metric.credit(0, 4), 0.0)
        self.assertAlmostEqual(metric.credit(4, 4), 1.0)
        self.assertAlmostEqual(metric.credit(1, 3), (1 / 2) / (3 / 4))
        # One box containing 4 points, 1 correct -> (1/2)/(4/5) = 0.625.
        coord = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])
        metric.update(_view(coord, [1, 0, 0, 0], [_box_row(0.0, 0.0, 2.0, 2.0)], [1], [0]))
        out = metric.compute()
        self.assertAlmostEqual(out["pd_score_ped"], 0.625)
        self.assertAlmostEqual(out["mpd_score"], 0.625)
        self.assertEqual(out["pd_skipped_low_point_boxes"], 0.0)

    def test_skips_low_point_boxes_and_reports_nan_without_boxes(self):
        metric = PartialDetectionScore(min_points=2, det_class_names=("ped", "car"))
        coord = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        metric.update(
            _view(coord, [1, 1], [_box_row(0.0, 0.0, 2.0, 2.0), _box_row(50.0, 0.0, 2.0, 2.0)], [1, 1], [0, 0])
        )
        out = metric.compute()
        self.assertEqual(out["pd_skipped_low_point_boxes"], 2.0)
        self.assertTrue(math.isnan(out["pd_score_ped"]))
        self.assertTrue(math.isnan(out["pd_score_car"]))
        self.assertTrue(math.isnan(out["mpd_score"]))
        # A frame with boxes but no valid points skips every mapped box.
        metric.update(_view(np.zeros((0, 3)), [], [_box_row(0.0, 0.0, 2.0, 2.0)], [1], [0]))
        self.assertEqual(metric.compute()["pd_skipped_low_point_boxes"], 3.0)

    def test_yaw_aware_membership(self):
        # 4 m long, 1 m wide box rotated 90 deg: a point at (0, 1.5) is inside only under the yaw-aware test.
        metric = PartialDetectionScore(det_class_names=("ped",))
        coord = np.array([[0.0, 1.5, 0.0], [1.5, 0.0, 0.0]])
        metric.update(_view(coord, [1, 1], [_box_row(0.0, 0.0, 4.0, 1.0, yaw=pi / 2)], [1], [0]))
        out = metric.compute()
        self.assertAlmostEqual(out["pd_score_ped"], 1.0)  # exactly one interior point, correct
        self.assertEqual(out["pd_skipped_low_point_boxes"], 0.0)

    def test_refuses_grouped_view_and_unmapped_boxes_ignored(self):
        metric = PartialDetectionScore(det_class_names=("ped",))
        with self.assertRaisesRegex(ValueError, "raw"):
            metric.update(_view(np.zeros((1, 3)), [1], [_box_row(0, 0, 2, 2)], [1], [0], taxonomy="grouped"))
        metric.update(_view(np.zeros((1, 3)), [1], [_box_row(0, 0, 2, 2)], [-1], [-1]))
        self.assertTrue(math.isnan(metric.compute()["pd_score_ped"]))
        with self.assertRaises(ValueError):
            PartialDetectionScore(min_points=0)
        with self.assertRaises(ValueError):
            PartialDetectionScore(half_saturation=0.0)


class TestBoxes(unittest.TestCase):
    def test_gt_boxes_from_objects_maps_labels_and_transforms(self):
        mapping = {"pedestrian": "obstacle", "car": "obstacle"}
        ego2map = HomogeneousMatrix(
            (10.0, 0.0, 0.0), Quaternion(axis=[0, 0, 1], angle=0.0), FrameID.BASE_LINK, FrameID.MAP
        )
        objects = [
            _object(1.0, 2.0, size=(1.0, 4.0, 2.0)),  # base_link, width 1 length 4
            _object(15.0, 0.0, label=AutowareLabel.CAR, frame_id=FrameID.MAP),  # map frame -> base_link x = 5
            _object(0.0, 0.0, label=AutowareLabel.BUS),  # unmapped
        ]
        rows, seg_class, det_label = gt_boxes_from_objects(objects, TransformDict([ego2map]), mapping, CLASS_NAMES)
        self.assertEqual(rows.shape, (3, 9))
        np.testing.assert_allclose(rows[0, :2], [1.0, 2.0])
        self.assertAlmostEqual(rows[0, 3], 4.0)  # dx = length
        self.assertAlmostEqual(rows[0, 4], 1.0)  # dy = width
        np.testing.assert_allclose(rows[1, :2], [5.0, 0.0])
        self.assertEqual(seg_class.tolist(), [1, 1, -1])
        self.assertEqual(det_label.tolist(), [0, 1, -1])
        empty = gt_boxes_from_objects([], TransformDict(), mapping, CLASS_NAMES)
        self.assertEqual(empty[0].shape, (0, 9))


class TestPartialDetectionInSuite(unittest.TestCase):
    def test_suite_runs_partial_detection_only_on_raw_taxonomy(self):
        cfg = config(
            ["partial_detection"],
            class_names=("road", "obstacle"),
            class_groups={"flat": ["road"], "thing": ["obstacle"]},
            box_label_to_seg_class={"pedestrian": "obstacle"},
        )
        suite = SegmentationMetricSuite(cfg)
        coord = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])
        frame = SegmentationFrame(
            "0",
            None,
            coord,
            np.array([1, 1, 1, 1]),
            np.array([1, 0, 0, 0]),
            np.array([[0.2, 0.8], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]),
            gt_objects=(_object(0.0, 0.0),),
        )
        suite.update(frame)
        report = suite.compute()
        self.assertAlmostEqual(report.values["segmentation/pd_score_pedestrian"], 0.625)
        self.assertNotIn("segmentation/grouped/pd_score_pedestrian", report.values)
        with self.assertRaises(ValueError):
            config(["partial_detection"], class_names=("road", "obstacle"))  # mapping required


if __name__ == "__main__":
    unittest.main()
