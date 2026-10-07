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
from perception_eval.evaluation.metrics.detection.config import AdvancedDetectionMetricsConfig
from perception_eval.evaluation.metrics.detection.frame import DetectionFrame
from perception_eval.evaluation.metrics.detection.suite import AdvancedDetectionSuite
from pyquaternion import Quaternion
from shapely.geometry import Polygon

TARGET_LABELS = [AutowareLabel.CAR, AutowareLabel.BICYCLE, AutowareLabel.PEDESTRIAN, AutowareLabel.MOTORBIKE]


def _frame(name: str, estimated, ground_truth, scene_id=None, with_pose=True) -> DetectionFrame:
    transforms = None
    if with_pose:
        transforms = TransformDict(
            [HomogeneousMatrix((0.0, 0.0, 0.0), Quaternion(), src=FrameID.BASE_LINK, dst=FrameID.MAP)]
        )
    return DetectionFrame(
        frame_name=name,
        unix_time=int(name) * 100,
        scene_id=scene_id,
        estimated_objects=tuple(estimated),
        ground_truth_objects=tuple(ground_truth),
        transforms=transforms,
    )


class TestAdvancedDetectionSuite(unittest.TestCase):
    def setUp(self):
        estimated, ground_truth = make_dummy_data()
        self.frames = [_frame("0", estimated, ground_truth), _frame("1", estimated, ground_truth)]

    def test_identity_view_keys_and_values(self):
        config = AdvancedDetectionMetricsConfig.from_dict(
            {
                "components": [
                    {"type": "corner_error"},
                    {"type": "heading_flip"},
                    {"type": "calibration"},
                    {"type": "confusion_matrix"},
                ]
            },
            TARGET_LABELS,
        )
        report = AdvancedDetectionSuite(config).evaluate(self.frames, TARGET_LABELS)
        self.assertIn("detection/mcorner_mean", report.values)
        self.assertIn("detection/corner_mean_car", report.values)
        self.assertIn("detection/flip_rate_bicycle", report.values)
        self.assertIn("detection/ece", report.values)
        self.assertIn("detection/confusion_car__car", report.values)
        # Dummy estimations sit on their ground truths with a different size: a finite corner error,
        # no heading flips, and every matched car lands on the confusion diagonal.
        self.assertGreater(report.values["detection/corner_mean_car"], 0.0)
        self.assertLess(report.values["detection/corner_mean_car"], 2.0)
        self.assertAlmostEqual(report.values["detection/mflip_rate"], 0.0)
        self.assertEqual(report.values["detection/confusion_car__car"], 2.0)  # 1 car GT x 2 frames
        # Pedestrian has GT but no estimation in the dummy data -> NaN corner error.
        self.assertTrue(math.isnan(report.values["detection/corner_mean_pedestrian"]))
        self.assertEqual(report.coverage, {})
        self.assertEqual(report.warnings, [])

    def test_grouped_filter_and_range_views(self):
        config = AdvancedDetectionMetricsConfig.from_dict(
            {
                "ranges": [{"name": "near", "max_distance": 30.0}, {"name": "far", "min_distance": 30.0}],
                "class_groups": {"grouped_vehicle": ["car", "motorbike"], "grouped_vru": ["bicycle", "pedestrian"]},
                "filters": [{"name": "corridor", "type": "corridor", "width_m": 3.0}],
                "components": [{"type": "corner_error"}],
            },
            TARGET_LABELS,
        )
        report = AdvancedDetectionSuite(config).evaluate(self.frames, TARGET_LABELS)
        keys = set(report.values)
        self.assertIn("detection/corner_mean_car", keys)
        self.assertIn("detection/0m_30m/corner_mean_car", keys)
        self.assertIn("detection/30m_inf/corner_mean_car", keys)
        self.assertIn("detection/corridor/corner_mean_car", keys)
        self.assertIn("detection/corridor/0m_30m/corner_mean_car", keys)
        self.assertIn("detection/grouped/corner_mean_grouped_vehicle", keys)
        self.assertIn("detection/grouped/corridor/30m_inf/mcorner_mean", keys)
        self.assertNotIn("detection/grouped/corner_mean_car", keys)
        # Every view emits the same per-view key set: macro mean/max + 4 classes x (mean, max, p95).
        per_view = {key.rsplit("/", 1)[-1] for key in keys if key.startswith("detection/corridor/0m_30m/")}
        self.assertEqual(len(per_view), 2 + 4 * 3)
        grouped_view = {key.rsplit("/", 1)[-1] for key in keys if key.startswith("detection/grouped/corridor/0m_30m/")}
        self.assertEqual(len(grouped_view), 2 + 2 * 3)
        # Objects far away in the dummy data are all within 30 m, so the far window has NaNs only.
        self.assertTrue(math.isnan(report.values["detection/30m_inf/mcorner_mean"]))
        # Map-free filters are always covered.
        self.assertEqual(report.coverage, {"corridor": (2, 2)})

    def test_zero_coverage_view_is_nan_with_warning(self):
        config = AdvancedDetectionMetricsConfig.from_dict(
            {
                "filters": [{"name": "road", "type": "region", "regions": ["road"]}],
                "components": [{"type": "corner_error"}],
                "map": {"resolver": "explicit", "mapping": {"other_scene": "/nonexistent/lanelet2_map.osm"}},
            },
            TARGET_LABELS,
        )
        frames = [_frame("0", *make_dummy_data(), scene_id="scene_without_map")]
        report = AdvancedDetectionSuite(config).evaluate(frames, TARGET_LABELS)
        self.assertEqual(report.coverage["road"], (0, 1))
        self.assertTrue(math.isnan(report.values["detection/road/corner_mean_car"]))
        self.assertFalse(math.isnan(report.values["detection/corner_mean_car"]))
        self.assertTrue(any("road" in warning for warning in report.warnings))

    def test_frame_without_pose_is_uncovered_for_map_filters(self):
        config = AdvancedDetectionMetricsConfig.from_dict(
            {
                "filters": [{"name": "road", "type": "region", "regions": ["road"]}],
                "components": [{"type": "heading_flip"}],
                "map": {"resolver": "explicit", "mapping": {"s": "/nonexistent.osm"}},
            },
            TARGET_LABELS,
        )
        frames = [_frame("0", *make_dummy_data(), scene_id="s", with_pose=False)]
        report = AdvancedDetectionSuite(config).evaluate(frames, TARGET_LABELS)
        self.assertEqual(report.coverage["road"], (0, 1))

    def test_polygon_frame_is_skipped_with_warning(self):
        estimated, ground_truth = make_dummy_data()
        polygon_object = DynamicObject(
            unix_time=100,
            frame_id=FrameID.BASE_LINK,
            position=(1.0, 1.0, 1.0),
            orientation=Quaternion(),
            shape=Shape(ShapeType.POLYGON, (0.0, 0.0, 1.0), footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
            velocity=None,
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.CAR, "car"),
        )
        frames = [_frame("0", estimated, ground_truth), _frame("1", estimated + [polygon_object], ground_truth)]
        config = AdvancedDetectionMetricsConfig.from_dict({"components": [{"type": "confident_error"}]}, TARGET_LABELS)
        report = AdvancedDetectionSuite(config).evaluate(frames, TARGET_LABELS)
        self.assertEqual(len(report.warnings), 2)
        self.assertIn("skipped", report.warnings[0])
        # Frame names repeat in practice, so the total is reported once as well.
        self.assertIn("1/2 frames skipped", report.warnings[1])
        # Only the surviving frame counts in the per-frame denominator.
        self.assertEqual(
            report.values["detection/confident_errors_per_frame"], report.values["detection/confident_error_count"]
        )

    def test_labels_derived_from_class_names_when_omitted(self):
        config = AdvancedDetectionMetricsConfig.from_dict({"components": [{"type": "heading_flip"}]}, TARGET_LABELS)
        report = AdvancedDetectionSuite(config).evaluate(self.frames)
        self.assertAlmostEqual(report.values["detection/flip_count_car"], 0.0)

    def test_flat_keys(self):
        config = AdvancedDetectionMetricsConfig.from_dict({"components": [{"type": "heading_flip"}]}, TARGET_LABELS)
        report = AdvancedDetectionSuite(config).evaluate(self.frames, TARGET_LABELS)
        self.assertIn("detection_mflip_rate", report.to_flat_keys())


if __name__ == "__main__":
    unittest.main()


class TestAdvancedDetectionSuiteInputs(unittest.TestCase):
    """Score source, POLYGON policy and non-finite scores (inputs as produced by a log replayer)."""

    @staticmethod
    def _polygon_frame(name="1"):
        estimated, ground_truth = make_dummy_data()
        polygon_object = DynamicObject(
            unix_time=100,
            frame_id=FrameID.BASE_LINK,
            position=(1.0, 1.0, 1.0),
            orientation=Quaternion(),
            shape=Shape(ShapeType.POLYGON, (0.0, 0.0, 1.0), footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
            velocity=None,
            semantic_score=0.9,
            semantic_label=Label(AutowareLabel.CAR, "car"),
        )
        return _frame(name, estimated + [polygon_object], ground_truth)

    def test_as_box_keeps_polygon_frames(self):
        config = AdvancedDetectionMetricsConfig.from_dict(
            {"components": [{"type": "confident_error"}], "polygon_shapes": "as_box"}, TARGET_LABELS
        )
        report = AdvancedDetectionSuite(config).evaluate([self._polygon_frame()], TARGET_LABELS)
        self.assertFalse(any("skipped" in warning for warning in report.warnings))
        self.assertTrue(any("1 POLYGON-shaped objects" in warning for warning in report.warnings))

    def test_nan_score_is_dropped_not_fatal(self):
        estimated, ground_truth = make_dummy_data()
        estimated[0].semantic_score = float("nan")
        config = AdvancedDetectionMetricsConfig.from_dict(
            {"components": [{"type": "calibration"}, {"type": "confident_error"}]}, TARGET_LABELS
        )
        report = AdvancedDetectionSuite(config).evaluate([_frame("0", estimated, ground_truth)], TARGET_LABELS)
        self.assertTrue(np.isfinite(report.values["detection/ece"]))
        self.assertTrue(any("1 predictions dropped" in warning for warning in report.warnings))

    def test_existence_probability_score_source(self):
        estimated, ground_truth = make_dummy_data()
        for obj in estimated:
            obj.existence_probability = 0.25
        frames = [_frame("0", estimated, ground_truth)]
        by_semantic = AdvancedDetectionSuite(
            AdvancedDetectionMetricsConfig.from_dict({"components": [{"type": "confident_error"}]}, TARGET_LABELS)
        ).evaluate(frames, TARGET_LABELS)
        by_existence = AdvancedDetectionSuite(
            AdvancedDetectionMetricsConfig.from_dict(
                {"components": [{"type": "confident_error"}], "score_source": "existence_probability"},
                TARGET_LABELS,
            )
        ).evaluate(frames, TARGET_LABELS)
        # Every existence probability sits below the 0.5 threshold, so no confident errors remain.
        self.assertEqual(by_existence.values["detection/confident_error_count"], 0.0)
        self.assertGreaterEqual(
            by_semantic.values["detection/confident_error_count"],
            by_existence.values["detection/confident_error_count"],
        )

    def test_missing_existence_probability_is_reported(self):
        estimated, ground_truth = make_dummy_data()
        config = AdvancedDetectionMetricsConfig.from_dict(
            {"components": [{"type": "confident_error"}], "score_source": "existence_probability"}, TARGET_LABELS
        )
        report = AdvancedDetectionSuite(config).evaluate([_frame("0", estimated, ground_truth)], TARGET_LABELS)
        self.assertTrue(any("missing existence_probability" in warning for warning in report.warnings))
