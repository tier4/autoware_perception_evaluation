# Copyright 2022 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
import unittest

from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation.matching import MatchingLabelPolicy


class TestPerceptionEvaluationConfig(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_check_tasks(self):
        """Test if it can detect the exception."""
        evaluation_config_dict = {
            "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
            "max_x_position": 102.4,
            "max_y_position": 102.4,
            "center_distance_thresholds": [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            "center_distance_bev_thresholds": [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            "plane_distance_thresholds": [2.0, 3.0],
            "iou_bev_thresholds": [0.5],
            "iou_3d_thresholds": [0.5],
            "min_point_numbers": [0, 0, 0, 0],
            "label_prefix": "autoware",
            "merge_similar_labels": False,
            "allow_matching_unknown": True,
        }
        # patterns: (frame_id, evaluation_task)
        patterns: List[Tuple(str, Dict[str, Any])] = [
            ("map", {"evaluation_task": "foo"}),
            ("base_link", {"evaluation_task": "foo"}),
            (["base_link", "map"], {"evaluation_task": "detection"}),
            (["cam_front", "cam_back"], {"evaluation_task": "detection"}),
        ]
        for n, (frame_id, evaluation_task) in enumerate(patterns):
            with self.subTest(f"Test if it can detect the exception of task keys: {n + 1}"):
                with self.assertRaises(ValueError):
                    evaluation_config_dict.update(evaluation_task)
                    _ = PerceptionEvaluationConfig(
                        dataset_paths="/tmp",
                        frame_id=frame_id,
                        result_root_directory="/tmp",
                        evaluation_config_dict=evaluation_config_dict,
                    )

    def test_matching_label_policy(self) -> None:
        """Test if `MatchingLabelPolicy` is set as expected."""
        evaluation_config_dict = {
            "evaluation_task": "detection",
            "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
            "max_x_position": 102.4,
            "max_y_position": 102.4,
            "center_distance_thresholds": [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            "center_distance_bev_thresholds": [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            "plane_distance_thresholds": [2.0, 3.0],
            "iou_bev_thresholds": [0.5],
            "iou_3d_thresholds": [0.5],
            "min_point_numbers": [0, 0, 0, 0],
            "label_prefix": "autoware",
            "merge_similar_labels": False,
        }
        # patterns: (policy, expect)
        patterns: List[Tuple[Dict[str, Any], MatchingLabelPolicy]] = [
            ({}, MatchingLabelPolicy.DEFAULT),
            ({"allow_matching_unknown": False}, MatchingLabelPolicy.DEFAULT),
            ({"allow_matching_unknown": True}, MatchingLabelPolicy.ALLOW_UNKNOWN),
            ({"matching_label_policy": "DEFAULT"}, MatchingLabelPolicy.DEFAULT),
            ({"matching_label_policy": "default"}, MatchingLabelPolicy.DEFAULT),
            ({"matching_label_policy": "ALLOW_UNKNOWN"}, MatchingLabelPolicy.ALLOW_UNKNOWN),
            ({"matching_label_policy": "allow_unknown"}, MatchingLabelPolicy.ALLOW_UNKNOWN),
            ({"matching_label_policy": "ALLOW_ANY"}, MatchingLabelPolicy.ALLOW_ANY),
            ({"matching_label_policy": "allow_any"}, MatchingLabelPolicy.ALLOW_ANY),
        ]
        for n, (policy, expect) in enumerate(patterns):
            with self.subTest(f"Test case: {n + 1}"):
                evaluation_config_dict.update(policy)
                config = PerceptionEvaluationConfig(
                    dataset_paths="/tmp",
                    frame_id="base_link",
                    result_root_directory="/tmp",
                    evaluation_config_dict=evaluation_config_dict,
                )

                assert expect == config.label_params["matching_label_policy"], f"Fail@{n + 1}"
