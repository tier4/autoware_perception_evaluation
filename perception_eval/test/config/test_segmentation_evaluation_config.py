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
import tempfile
import unittest

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.schema import FrameID
from perception_eval.config import SegmentationEvaluationConfig
from perception_eval.evaluation.metrics.config.segmentation_metrics_config import SegmentationMetricsConfig
from perception_eval.evaluation.metrics.filter_config import MetricsConfigError


def _dict(**overrides):
    cfg = {
        "evaluation_task": "segmentation",
        "class_names": ["car", "truck", "pedestrian", "road", "sidewalk", "vegetation"],
        "ignore_index": -1,
        "ranges": [{"name": "0_30", "min_distance": 0.0, "max_distance": 30.0}],
        "class_groups": {
            "vehicle": ["car", "truck"],
            "vru": ["pedestrian"],
            "flat": ["road", "sidewalk"],
            "other": ["vegetation"],
        },
        "filters": [{"name": "corridor", "type": "corridor", "width_m": 3.0}],
        "components": [
            {"type": "confusion_matrix"},
            {"type": "iou"},
            {"type": "calibration", "num_bins": 15},
            {"type": "uncertainty_usefulness"},
            {"type": "confident_error", "entropy_threshold": 0.3},
            {"type": "error_clusters", "cluster_radius": 0.5},
            {"type": "tolerant_error", "radius": 0.2},
            {"type": "partial_detection", "half_saturation": 1.0, "min_points": 1},
        ],
        "box_label_to_seg_class": {"car": "car", "truck": "truck", "pedestrian": "pedestrian"},
        "check_argmax": True,
    }
    cfg.update(overrides)
    return cfg


class TestSegmentationEvaluationConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _build(self, frame_id="base_link", **overrides):
        return SegmentationEvaluationConfig(
            dataset_paths=[],
            frame_id=frame_id,
            result_root_directory=self.tmpdir.name,
            evaluation_config_dict=_dict(**overrides),
        )

    def test_happy_path(self):
        config = self._build()
        self.assertEqual(config.evaluation_task, EvaluationTask.SEGMENTATION)
        self.assertEqual(config.frame_ids, [FrameID.BASE_LINK])
        self.assertEqual(config.metrics_config.num_classes, 6)
        self.assertEqual(config.metrics_config.det_class_names, ("car", "truck", "pedestrian"))
        self.assertTrue(config.metrics_config.needs_boxes)
        self.assertFalse(config.metrics_config.needs_map)
        self.assertEqual(len(config.metrics_config.components), 8)
        self.assertEqual(config.filtering_params, {"target_uuids": None})
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.metrics_config.class_names, config.metrics_config.class_names)

    def test_rejects_unsupported_task_and_frame_ids(self):
        with self.assertRaises(ValueError):
            self._build(evaluation_task="detection")
        with self.assertRaises(ValueError):
            self._build(frame_id=["base_link", "map"])

    def test_metrics_config_validation(self):
        with self.assertRaises(MetricsConfigError):
            self._build(components=[{"type": "nope"}])
        with self.assertRaises(MetricsConfigError):
            self._build(components=[{"type": "calibration", "bins": 3}])
        with self.assertRaises(MetricsConfigError):
            self._build(box_label_to_seg_class={"car": "bicycle_lane"})
        with self.assertRaises(MetricsConfigError):
            self._build(box_label_to_seg_class={"motorcycle": "car"})
        with self.assertRaises(MetricsConfigError):
            self._build(ranges=[{"name": "a", "max_distance": 10.0}, {"name": "a", "max_distance": 20.0}])
        with self.assertRaises(MetricsConfigError):
            self._build(ignore_index=2)
        with self.assertRaises(MetricsConfigError):
            self._build(class_groups={"vehicle": ["car", "truck"]})
        with self.assertRaises(MetricsConfigError):
            self._build(filters=[{"name": "road", "type": "region", "regions": ["road"]}])  # needs map
        with self.assertRaises(MetricsConfigError):
            SegmentationMetricsConfig.from_dict({"class_names": ["a", "b"], "unknown_key": 1})
        with self.assertRaises(MetricsConfigError):
            SegmentationMetricsConfig.from_dict({"class_names": ["only"]})
        with self.assertRaises(MetricsConfigError):
            self._build(components=[{"type": "iou"}, {"type": "iou"}])

    def test_serialization_round_trip(self):
        metrics_config = self._build().metrics_config
        data = metrics_config.serialization()
        restored = SegmentationMetricsConfig.deserialization(data)
        self.assertEqual(restored, metrics_config)
        self.assertEqual(data["filters"][0]["type"], "corridor")
        self.assertEqual(data["components"][2], {"type": "calibration", "num_bins": 15})


if __name__ == "__main__":
    unittest.main()
