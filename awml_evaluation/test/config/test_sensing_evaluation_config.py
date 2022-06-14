from typing import Any
from typing import Dict
from typing import List
import unittest

from awml_evaluation.config.sensing_evaluation_config import SensingEvaluationConfig


class TestSensingEvaluationConfig(unittest.TestCase):
    def test_check_tasks(self):
        """Test if it can detect the exception."""
        patterns: List[Dict[str, Dict[str, Any]]] = [
            {"evaluation_task": "foo", "target_uuids": ["001"]},
        ]
        for n, evaluation_config_dict in enumerate(patterns):
            with self.subTest(f"Test if it can detect the exception of task keys: {n + 1}"):
                with self.assertRaises(ValueError):
                    _ = SensingEvaluationConfig(
                        dataset_paths="/tmp/path",
                        does_use_pointcloud=False,
                        result_root_directory="/tmp/path",
                        log_directory="/tmp/path",
                        visualization_directory="/tmp/path",
                        evaluation_config_dict=evaluation_config_dict,
                    )
