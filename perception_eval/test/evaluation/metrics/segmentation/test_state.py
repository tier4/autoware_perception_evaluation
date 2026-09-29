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

from test.evaluation.metrics.segmentation.helpers import config
from test.evaluation.metrics.segmentation.helpers import make_frame
from test.evaluation.metrics.segmentation.helpers import scores_for
import unittest

import numpy as np
from perception_eval.common.schema import FrameID
from perception_eval.common.transform import HomogeneousMatrix
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.metrics.class_groups import resolve_class_groups
from perception_eval.evaluation.metrics.segmentation.state import build_grouped_taxonomy
from perception_eval.evaluation.metrics.segmentation.state import build_raw_taxonomy
from perception_eval.evaluation.metrics.segmentation.state import confidence_of
from perception_eval.evaluation.metrics.segmentation.state import normalized_entropy
from perception_eval.evaluation.metrics.segmentation.state import SegmentationFrame
from perception_eval.evaluation.metrics.segmentation.state import valid_point_mask
from perception_eval.evaluation.metrics.segmentation.state import validate_frame
from pyquaternion import Quaternion

NAMES = ("car", "truck", "road", "sidewalk")
GROUPS = {"vehicle": ["car", "truck"], "flat": ["road", "sidewalk"]}


class TestSegmentationFrame(unittest.TestCase):
    def test_structural_validation(self):
        good = make_frame([0, 1], [0, 1], [0, 1])
        self.assertEqual(good.num_points, 2)
        self.assertEqual(good.num_classes, 2)
        self.assertIsInstance(good.transforms, TransformDict)
        self.assertIsNone(good.ego2map)
        with self.assertRaises(ValueError):
            SegmentationFrame("0", None, np.zeros((2, 2)), np.zeros(2, int), np.zeros(2, int), np.full((2, 2), 0.5))
        with self.assertRaises(ValueError):
            SegmentationFrame("0", None, np.zeros((2, 3)), np.zeros(3, int), np.zeros(2, int), np.full((2, 2), 0.5))
        with self.assertRaises(ValueError):
            SegmentationFrame("0", None, np.zeros((2, 3)), np.zeros(2), np.zeros(2, int), np.full((2, 2), 0.5))
        with self.assertRaises(ValueError):
            SegmentationFrame("0", None, np.zeros((2, 3)), np.zeros(2, int), np.zeros(2, int), np.full((3, 2), 0.5))
        with self.assertRaises(TypeError):
            SegmentationFrame(
                "0", None, np.zeros((2, 3)), np.zeros(2, int), np.zeros(2, int), np.full((2, 2), 0.5), gt_objects=(1,)
            )
        empty = SegmentationFrame("0", None, np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2)))
        self.assertEqual(empty.num_points, 0)

    def test_ego2map(self):
        ego2map = HomogeneousMatrix(
            (10.0, 0.0, 0.0), Quaternion(axis=[0, 0, 1], angle=0.0), FrameID.BASE_LINK, FrameID.MAP
        )
        frame = SegmentationFrame(
            "0",
            "scene",
            np.zeros((1, 3)),
            np.zeros(1, int),
            np.zeros(1, int),
            np.full((1, 2), 0.5),
            TransformDict([ego2map]),
        )
        np.testing.assert_allclose(frame.ego2map[:3, 3], [10.0, 0.0, 0.0])

    def test_config_validation(self):
        cfg = config(["accuracy"], check_argmax=True)
        validate_frame(make_frame([0, 1], [0, 1], [0, 1], scores=scores_for([0, 1], 0.9)), cfg)
        with self.assertRaisesRegex(ValueError, "columns"):
            validate_frame(make_frame([0], [0], [0], num_classes=3), cfg)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            validate_frame(make_frame([0], [0], [0], scores=np.array([[1.2, -0.2]])), cfg)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_frame(make_frame([0], [0], [0], scores=np.array([[0.6, 0.6]])), cfg)
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_frame(make_frame([0], [0], [0], scores=np.array([[np.nan, 1.0]])), cfg)
        with self.assertRaisesRegex(ValueError, "argmax"):
            validate_frame(make_frame([0], [0], [1], scores=np.array([[0.9, 0.1]])), cfg)
        relaxed = config(["accuracy"], check_argmax=False)
        validate_frame(make_frame([0], [0], [1], scores=np.array([[0.9, 0.1]])), relaxed)


class TestDerivedScalars(unittest.TestCase):
    def test_entropy_and_confidence(self):
        uniform = np.full((1, 4), 0.25)
        one_hot = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.assertAlmostEqual(normalized_entropy(uniform)[0], 1.0)
        self.assertAlmostEqual(normalized_entropy(one_hot)[0], 0.0, places=9)
        probs = np.array([[0.4, 0.0, 0.35, 0.25]])
        expected = -sum(p * np.log(p) for p in (0.4, 0.35, 0.25)) / np.log(4)
        self.assertAlmostEqual(normalized_entropy(probs)[0], expected)
        np.testing.assert_allclose(confidence_of(probs, np.array([0])), [0.4])
        np.testing.assert_allclose(confidence_of(probs, np.array([7])), [0.0])
        with self.assertRaises(ValueError):
            normalized_entropy(np.ones((2, 1)))

    def test_valid_point_mask(self):
        mask = valid_point_mask(np.array([0, -1, 1, 5]), np.array([0, 0, 3, 0]), 2, -1)
        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_grouped_confidence_is_predicted_group_mass(self):
        lut, names = resolve_class_groups(NAMES, GROUPS)
        # Raw probabilities: car 0.4 (argmax), truck 0.0, road 0.35, sidewalk 0.25.
        # Folded: vehicle 0.4, flat 0.6. The reported group is vehicle, so the grouped
        # confidence must be 0.4, never the 0.6 max.
        frame = make_frame([0], [2], [0], num_classes=4, scores=np.array([[0.4, 0.0, 0.35, 0.25]]))
        raw = build_raw_taxonomy(frame, NAMES, -1)
        grouped = build_grouped_taxonomy(frame, lut, names, -1)
        self.assertAlmostEqual(raw.confidence[0], 0.4)
        self.assertEqual(grouped.pred[0], 0)  # vehicle
        self.assertEqual(grouped.target[0], 1)  # flat
        self.assertAlmostEqual(grouped.confidence[0], 0.4)
        expected_entropy = -(0.4 * np.log(0.4) + 0.6 * np.log(0.6)) / np.log(2)
        self.assertAlmostEqual(grouped.entropy[0], expected_entropy)
        self.assertEqual(grouped.num_classes, 2)
        self.assertEqual(grouped.class_names, ("vehicle", "flat"))

    def test_grouped_taxonomy_leaves_ignore_untouched(self):
        lut, names = resolve_class_groups(NAMES, GROUPS)
        frame = make_frame([0, 1], [-1, 3], [1, 2], num_classes=4)
        grouped = build_grouped_taxonomy(frame, lut, names, -1)
        self.assertEqual(grouped.target.tolist(), [-1, 1])
        self.assertEqual(grouped.valid.tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
