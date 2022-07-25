from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
import unittest

from awml_evaluation.config.sensing_evaluation_config import SensingEvaluationConfig


class TestSensingEvaluationConfig(unittest.TestCase):
    def test_check_tasks(self):
        """Test if it can detect the exception."""
        evaluation_config_dict = {
            "evaluation_task": "sensing",
            "target_uuids": ["1b40c0876c746f96ac679a534e1037a2"],
            "box_scale_0m": 1.0,
            "box_scale_100m": 1.0,
            "min_points_threshold": 1,
        }
        # patterns: (frame_id, evaluation_task)
        patterns: List[Tuple(str, Dict[str, Any])] = [
            ("map", {"evaluation_task": "foo"}),
            ("base_link", {"evaluation_task": "foo"}),
            ("foo", {"evaluation_task": "sensing"}),
        ]
        for n, (frame_id, evaluation_task) in enumerate(patterns):
            with self.subTest(f"Test if it can detect the exception of task keys: {n + 1}"):
                with self.assertRaises(ValueError):
                    evaluation_config_dict.update(evaluation_task)
                    _ = SensingEvaluationConfig(
                        dataset_paths="/tmp/path",
                        does_use_pointcloud=False,
                        frame_id=frame_id,
                        result_root_directory="/tmp/path",
                        evaluation_config_dict=evaluation_config_dict,
                    )
