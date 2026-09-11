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

import pickle
import unittest

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.metrics.config.detection_metrics_config import DetectionMetricsConfig
from perception_eval.evaluation.metrics.detection.config import AdvancedDetectionMetricsConfig
from perception_eval.evaluation.metrics.detection.config import CornerErrorConfig
from perception_eval.evaluation.metrics.detection.config import parse_components
from perception_eval.evaluation.metrics.filter_config import MetricsConfigError
from perception_eval.evaluation.metrics.metrics_score_config import MetricsScoreConfig

TARGET_LABELS = [
    AutowareLabel.CAR,
    AutowareLabel.TRUCK,
    AutowareLabel.BUS,
    AutowareLabel.PEDESTRIAN,
    AutowareLabel.BICYCLE,
    AutowareLabel.MOTORBIKE,
    AutowareLabel.HAZARD,
    AutowareLabel.UNKNOWN,
]

FULL_SECTION = {
    "ranges": [
        {"name": "0_30", "min_distance": 0.0, "max_distance": 30.0},
        {"name": "30_60", "min_distance": 30.0, "max_distance": 60.0},
    ],
    "class_groups": {
        "grouped_vehicle": ["car", "truck", "bus"],
        "grouped_vru": ["pedestrian", "bicycle", "motorbike"],
        "grouped_static": ["hazard", "unknown"],
    },
    "filters": [
        {"name": "corridor", "type": "corridor", "width_m": 3.0},
        {"name": "road", "type": "region", "regions": ["road", "road_shoulder", "crosswalk"]},
        {"name": "collision", "type": "collision"},
    ],
    "components": [
        {"type": "corner_error", "tp_threshold": 2.0, "percentiles": [95.0]},
        {"type": "heading_flip", "tp_threshold": 2.0, "flip_threshold": 1.57079632679},
        {"type": "nearest_surface_error", "tp_threshold": 2.0},
        {"type": "calibration", "tp_threshold": 2.0, "num_bins": 15},
        {"type": "confident_error", "tp_threshold": 2.0, "min_score": 0.1, "score_threshold": 0.5},
        {"type": "confusion_matrix", "match_threshold": 2.0, "min_score": 0.1},
        {"type": "critical_fp_fn", "confidences": [0.3, 0.5], "match_threshold": 2.0},
        {"type": "collision_weighted_map", "thresholds": [0.5, 1.0, 2.0, 4.0], "decay": 0.5},
    ],
    "map": {"resolver": "t4_scene_directory"},
}


class TestAdvancedDetectionMetricsConfig(unittest.TestCase):
    def test_full_section_parses(self):
        config = AdvancedDetectionMetricsConfig.from_dict(FULL_SECTION, TARGET_LABELS)
        self.assertEqual(config.class_names[:3], ("car", "truck", "bus"))
        self.assertEqual([r.suffix for r in config.ranges], ["0m_30m", "30m_60m"])
        self.assertEqual(len(config.filters), 3)
        self.assertEqual(len(config.components), 8)
        self.assertTrue(config.needs_ttc)
        self.assertTrue(config.requires_map)
        self.assertEqual(config.map.resolver, "t4_scene_directory")
        # Serialization round trip reparses to an equal config.
        again = AdvancedDetectionMetricsConfig.from_dict(config.serialization(), TARGET_LABELS)
        self.assertEqual(again, config)

    def test_minimal_section(self):
        config = AdvancedDetectionMetricsConfig.from_dict({"components": [{"type": "corner_error"}]}, TARGET_LABELS[:2])
        self.assertEqual(config.ranges, ())
        self.assertIsNone(config.class_groups)
        self.assertEqual(config.filters, ())
        self.assertFalse(config.needs_ttc)
        self.assertFalse(config.requires_map)
        self.assertEqual(config.components, (CornerErrorConfig(),))

    def test_errors(self):
        with self.assertRaises(MetricsConfigError):
            AdvancedDetectionMetricsConfig.from_dict({"components": []}, TARGET_LABELS)
        with self.assertRaises(MetricsConfigError):
            AdvancedDetectionMetricsConfig.from_dict({"components": [{"type": "nope"}]}, TARGET_LABELS)
        with self.assertRaises(MetricsConfigError):
            AdvancedDetectionMetricsConfig.from_dict(
                {"components": [{"type": "corner_error", "foo": 1}]}, TARGET_LABELS
            )
        with self.assertRaises(MetricsConfigError):
            AdvancedDetectionMetricsConfig.from_dict(
                {"components": [{"type": "corner_error"}, {"type": "corner_error"}]}, TARGET_LABELS
            )
        with self.assertRaises(MetricsConfigError):
            AdvancedDetectionMetricsConfig.from_dict(
                {"components": [{"type": "corner_error"}], "typo": 1}, TARGET_LABELS
            )
        # nuScenes vocabulary is rejected: the repo uses AutowareLabel names.
        with self.assertRaisesRegex(MetricsConfigError, "motorbike"):
            AdvancedDetectionMetricsConfig.from_dict(
                {
                    "components": [{"type": "corner_error"}],
                    "class_groups": {
                        "vehicle": ["car", "truck", "bus"],
                        "vru": ["pedestrian", "bicycle", "motorcycle"],
                    },
                },
                TARGET_LABELS[:6],
            )
        # Partition must be complete.
        with self.assertRaises(MetricsConfigError):
            AdvancedDetectionMetricsConfig.from_dict(
                {"components": [{"type": "corner_error"}], "class_groups": {"vehicle": ["car"]}}, TARGET_LABELS[:2]
            )
        # Map-dependent pieces need the map section.
        with self.assertRaises(MetricsConfigError):
            AdvancedDetectionMetricsConfig.from_dict(
                {
                    "components": [{"type": "corner_error"}],
                    "filters": [{"type": "region", "name": "r", "regions": ["road"]}],
                },
                TARGET_LABELS,
            )
        with self.assertRaises(MetricsConfigError):
            AdvancedDetectionMetricsConfig.from_dict({"components": [{"type": "critical_fp_fn"}]}, TARGET_LABELS)
        with self.assertRaises(MetricsConfigError):
            AdvancedDetectionMetricsConfig.from_dict(
                {"components": [{"type": "corner_error", "percentiles": [150]}]}, TARGET_LABELS
            )
        with self.assertRaises(MetricsConfigError):
            parse_components(None)


class TestDetectionMetricsConfigIntegration(unittest.TestCase):
    def _config(self, advanced=None) -> DetectionMetricsConfig:
        return DetectionMetricsConfig(
            target_labels=TARGET_LABELS[:2],
            center_distance_thresholds=[[1.0, 1.0]],
            advanced_detection_metrics=advanced,
        )

    def test_absent_means_legacy(self):
        config = self._config()
        self.assertIsNone(config.advanced_detection_metrics)
        self.assertIsNone(config.advanced_detection_metrics_dict)
        restored = pickle.loads(pickle.dumps(config))
        self.assertIsNone(restored.advanced_detection_metrics)
        # A six-argument tuple (pickles produced before the field existed) still constructs.
        legacy = DetectionMetricsConfig(*pickle.loads(pickle.dumps(config)).__reduce__()[1][:6])
        self.assertIsNone(legacy.advanced_detection_metrics)
        self.assertNotIn("advanced_detection_metrics", config.serialization()["target_labels"][0])
        self.assertIsNone(DetectionMetricsConfig.deserialization(config.serialization()).advanced_detection_metrics)

    def test_present_round_trips(self):
        section = {"components": [{"type": "corner_error"}], "ranges": [{"name": "near", "max_distance": 30.0}]}
        config = self._config(section)
        self.assertIsNotNone(config.advanced_detection_metrics)
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.advanced_detection_metrics, config.advanced_detection_metrics)
        deserialized = DetectionMetricsConfig.deserialization(config.serialization())
        self.assertEqual(deserialized.advanced_detection_metrics, config.advanced_detection_metrics)

    def test_metrics_score_config_accepts_the_key(self):
        score_config = MetricsScoreConfig(
            EvaluationTask.DETECTION,
            target_labels=TARGET_LABELS[:2],
            center_distance_thresholds=[[1.0, 1.0]],
            advanced_detection_metrics={"components": [{"type": "heading_flip"}]},
        )
        self.assertIsNotNone(score_config.detection_config.advanced_detection_metrics)
        legacy = MetricsScoreConfig(
            EvaluationTask.DETECTION, target_labels=TARGET_LABELS[:2], center_distance_thresholds=[[1.0, 1.0]]
        )
        self.assertIsNone(legacy.detection_config.advanced_detection_metrics)
        self.assertIsNotNone(pickle.loads(pickle.dumps(score_config)).detection_config.advanced_detection_metrics)


if __name__ == "__main__":
    unittest.main()
