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

"""Integration test: enabling the advanced detection metrics leaves mAP/mAPH bit-identical."""

from copy import deepcopy
import math
import pickle
import tempfile
from test.util.dummy_object import make_dummy_data
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import unittest

from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.schema import FrameID
from perception_eval.common.transform import HomogeneousMatrix
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.metrics.detection.frame import DetectionFrame
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.manager import PerceptionEvaluationManager
from perception_eval.util.debug import get_objects_with_difference
from pyquaternion import Quaternion

TARGET_LABELS = ["car", "bicycle", "pedestrian", "motorbike"]

ADVANCED_SECTION: Dict[str, Any] = {
    "ranges": [{"name": "near", "min_distance": 0.0, "max_distance": 30.0}],
    "class_groups": {"grouped_vehicle": ["car", "motorbike"], "grouped_vru": ["bicycle", "pedestrian"]},
    "filters": [{"name": "corridor", "type": "corridor", "width_m": 3.0}],
    "components": [
        {"type": "corner_error"},
        {"type": "heading_flip"},
        {"type": "nearest_surface_error"},
        {"type": "calibration"},
        {"type": "confident_error"},
        {"type": "confusion_matrix"},
    ],
}


def _evaluation_config_dict(advanced: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = {
        "evaluation_task": "detection",
        "target_labels": TARGET_LABELS,
        "max_x_position": 102.4,
        "max_y_position": 102.4,
        "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
        "center_distance_bev_thresholds": [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
        "plane_distance_thresholds": [2.0, 3.0],
        "iou_2d_thresholds": [0.5, 0.5, 0.5, 0.5],
        "iou_3d_thresholds": [0.5, 0.5, 0.5, 0.5],
        "min_point_numbers": [0, 0, 0, 0],
        "label_prefix": "autoware",
        "merge_similar_labels": False,
    }
    if advanced is not None:
        config["advanced_detection_metrics"] = advanced
    return config


def _build_manager(root: str, advanced: Optional[Dict[str, Any]]) -> PerceptionEvaluationManager:
    config = PerceptionEvaluationConfig(
        dataset_paths=[root],
        frame_id="base_link",
        result_root_directory=root,
        evaluation_config_dict=_evaluation_config_dict(advanced),
        load_raw_data=False,
    )
    return PerceptionEvaluationManager(config, load_ground_truth=False)


def _run_scene(manager: PerceptionEvaluationManager, num_frames: int = 3) -> List[PerceptionFrameResult]:
    critical = CriticalObjectFilterConfig(
        evaluator_config=manager.evaluator_config,
        target_labels=TARGET_LABELS,
        max_x_position_list=[100.0] * 4,
        max_y_position_list=[100.0] * 4,
    )
    pass_fail = PerceptionPassFailConfig(
        evaluator_config=manager.evaluator_config, target_labels=TARGET_LABELS, matching_threshold_list=[2.0] * 4
    )
    ego2map = HomogeneousMatrix((0.0, 0.0, 0.0), Quaternion(), src=FrameID.BASE_LINK, dst=FrameID.MAP)
    results = []
    for index in range(num_frames):
        estimated, ground_truth = make_dummy_data()
        estimated = get_objects_with_difference(estimated, (0.1 * index, 0.0, 0.0), 0.05 * index)
        frame_gt = FrameGroundTruth(
            unix_time=100 * (index + 1),
            frame_name=str(index),
            objects=ground_truth,
            transforms=[ego2map],
            scene_id="scene_a",
        )
        results.append(manager.add_frame_result(100 * (index + 1), frame_gt, estimated, critical, pass_fail))
    return results


class TestAdvancedMetricsIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_map_values_are_bit_identical_and_report_is_populated(self):
        legacy = _build_manager(self.tmpdir.name, None)
        advanced = _build_manager(self.tmpdir.name, ADVANCED_SECTION)
        legacy_frames = _run_scene(legacy)
        advanced_frames = _run_scene(advanced)

        self.assertTrue(all(frame.detection_frame is None for frame in legacy_frames))
        self.assertTrue(all(isinstance(frame.detection_frame, DetectionFrame) for frame in advanced_frames))
        self.assertEqual(advanced_frames[0].detection_frame.scene_id, "scene_a")
        self.assertEqual(advanced_frames[0].detection_frame.num_ground_truth, 4)

        legacy_score = legacy.get_scene_result()
        advanced_score = advanced.get_scene_result()
        self.assertIsNone(legacy_score.detection_metric_report)
        self.assertIsNotNone(advanced_score.detection_metric_report)

        self.assertEqual(len(legacy_score.mean_ap_values), len(advanced_score.mean_ap_values))
        for legacy_map, advanced_map in zip(legacy_score.mean_ap_values, advanced_score.mean_ap_values):
            self.assertEqual(legacy_map.map, advanced_map.map)
            self.assertEqual(legacy_map.maph, advanced_map.maph)
            self.assertEqual(legacy_map.label_mean_to_ap, advanced_map.label_mean_to_ap)
            self.assertEqual(str(legacy_map), str(advanced_map))

        report = advanced_score.detection_metric_report
        self.assertIn("detection/corner_mean_car", report.values)
        self.assertIn("detection/grouped/corridor/0m_30m/mcorner_mean", report.values)
        self.assertIn("detection/confusion_car__car", report.values)
        self.assertFalse(math.isnan(report.values["detection/ece"]))
        self.assertIn("Advanced detection metrics", str(advanced_score))
        self.assertNotIn("Advanced detection metrics", str(legacy_score))

    def test_pickle_round_trips_and_legacy_state(self):
        manager = _build_manager(self.tmpdir.name, ADVANCED_SECTION)
        frames = _run_scene(manager, num_frames=2)
        score = manager.get_scene_result()

        restored_frames = pickle.loads(pickle.dumps(frames))
        self.assertIsNotNone(restored_frames[0].detection_frame)
        self.assertEqual(restored_frames[0].detection_frame.frame_name, "0")
        # `MetricsScore` produced by the manager holds defaultdict lambdas (pre-existing, unpicklable);
        # the report itself and a fresh score with the report attached round-trip.
        restored_report = pickle.loads(pickle.dumps(score.detection_metric_report))
        self._assert_same_values(restored_report.values, score.detection_metric_report.values)
        fresh = MetricsScore(config=manager.metrics_config, used_frame=[0, 1])
        fresh.detection_metric_report = score.detection_metric_report
        restored_score: MetricsScore = pickle.loads(pickle.dumps(fresh))
        self._assert_same_values(restored_score.detection_metric_report.values, score.detection_metric_report.values)

        # Legacy pickles carry state dicts without the new keys.
        frame_copy = deepcopy(frames[0])
        frame_copy.__setstate__(
            {"pass_fail_result": frames[0].pass_fail_result, "metric_score": frames[0].metrics_score}
        )
        self.assertIsNotNone(frame_copy.detection_frame)  # keeps the constructor value
        score_copy = deepcopy(score)
        score_copy.__setstate__({"mean_ap_values": score.mean_ap_values})
        self.assertIsNone(score_copy.detection_metric_report)

    def _assert_same_values(self, actual: Dict[str, float], expected: Dict[str, float]) -> None:
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            if math.isnan(value):
                self.assertTrue(math.isnan(actual[key]), key)
            else:
                self.assertEqual(actual[key], value, key)

    def test_preprocess_path_builds_the_same_detection_frame(self):
        manager = _build_manager(self.tmpdir.name, ADVANCED_SECTION)
        critical = CriticalObjectFilterConfig(
            evaluator_config=manager.evaluator_config,
            target_labels=TARGET_LABELS,
            max_x_position_list=[100.0] * 4,
            max_y_position_list=[100.0] * 4,
        )
        pass_fail = PerceptionPassFailConfig(
            evaluator_config=manager.evaluator_config, target_labels=TARGET_LABELS, matching_threshold_list=[2.0] * 4
        )
        estimated, ground_truth = make_dummy_data()
        frame_gt = FrameGroundTruth(unix_time=100, frame_name="0", objects=ground_truth, scene_id="scene_a")
        preprocessed = manager.preprocess_object_results(100, frame_gt, estimated, critical, pass_fail)
        manager.evaluate_perception_frame(preprocessed)
        self.assertIsNotNone(preprocessed.detection_frame)
        self.assertEqual(preprocessed.detection_frame.num_estimated, 3)
        self.assertEqual(preprocessed.detection_frame.num_ground_truth, 4)
        self.assertIsNone(preprocessed.detection_frame.ego2map)


if __name__ == "__main__":
    unittest.main()
