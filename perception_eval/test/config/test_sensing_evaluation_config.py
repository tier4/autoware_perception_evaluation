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

import unittest
from typing import Any, Dict, List, Tuple

import pytest

from perception_eval.config import SensingEvaluationConfig


class TestSensingEvaluationConfig(unittest.TestCase):
    def test_check_tasks(self):
        """Test if it can detect the exception."""
        evaluation_config_dict = {
            "evaluation_task": "sensing",
            "target_uuids": ["1b40c0876c746f96ac679a534e1037a2"],
            "box_scale_0m": 1.0,
            "box_scale_100m": 1.0,
            "min_points_threshold": 1,
            "label_prefix": "autoware",
        }
        patterns: List[Tuple(str, Dict[str, Any])] = [
            ("map", {"evaluation_task": "foo"}),
            ("base_link", {"evaluation_task": "foo"}),
            (["base_link", "map"], {"evaluation_task": "sensing"}),
        ]
        for n, (frame_id, evaluation_task) in enumerate(patterns):
            with self.subTest(f"Test if it can detect the exception of task keys: {n + 1}"), pytest.raises(ValueError):
                evaluation_config_dict.update(evaluation_task)
                _ = SensingEvaluationConfig(
                    dataset_paths="/tmp/path",
                    frame_id=frame_id,
                    result_root_directory="/tmp/path",
                    evaluation_config_dict=evaluation_config_dict,
                )
