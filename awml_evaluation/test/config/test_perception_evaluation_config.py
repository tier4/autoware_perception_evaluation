from typing import Any
from typing import Dict
from typing import List
import unittest

from awml_evaluation.config.perception_evaluation_config import PerceptionEvaluationConfig


class TestPerceptionEvaluationConfig(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_check_tasks(self):
        """Test if it can detect the exception."""
        patterns: List[Dict[str, Any]] = [
            {"evaluation_task": "foo", "target_labels": ["car"]},
        ]
        for n, evaluation_config_dict in enumerate(patterns):
            with self.subTest(f"Test if it can detect the exception of task keys: {n + 1}"):
                with self.assertRaises(ValueError):
                    _ = PerceptionEvaluationConfig(
                        dataset_paths="/tmp",
                        does_use_pointcloud=False,
                        result_root_directory="/tmp",
                        log_directory="/tmp",
                        visualization_directory="/tmp",
                        evaluation_config_dict=evaluation_config_dict,
                    )
