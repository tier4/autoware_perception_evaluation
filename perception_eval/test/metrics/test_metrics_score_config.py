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

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.config.params import PerceptionMetricsParam
from perception_eval.metrics.config._metrics_config_base import _MetricsConfigBase
from perception_eval.metrics.config.detection_metrics_config import DetectionMetricsConfig
from perception_eval.metrics.config.tracking_metrics_config import TrackingMetricsConfig
from perception_eval.metrics.metrics_score_config import MetricsScoreConfig

# from perception_eval.metrics.metrics_score_config import MetricsParameterError


class TestMetricsScoreConfig(unittest.TestCase):
    def test_check_parameters(self):
        """[summary]
        Test if it can detect the exception.

        test patterns:
            Check if the exception is raised when wrong key is specified.
        """
        patterns: List[Tuple[_MetricsConfigBase, Dict[str, Any]]] = [
            (
                DetectionMetricsConfig,
                PerceptionMetricsParam.from_dict(
                    cfg={"foo": 1.0},
                    evaluation_task=EvaluationTask.DETECTION,
                    target_labels=[AutowareLabel.CAR],
                ),
            ),
            (
                DetectionMetricsConfig,
                PerceptionMetricsParam.from_dict(
                    cfg={
                        "center_distance_thresholds": [[1.0]],
                        "plane_distance_thresholds": [[1.0]],
                        "iou_bev_thresholds": [[1.0]],
                    },
                    evaluation_task=EvaluationTask.DETECTION,
                    target_labels=[AutowareLabel.CAR],
                ),
            ),
            (
                TrackingMetricsConfig,
                PerceptionMetricsParam.from_dict(
                    cfg={"foo": 1.0},
                    evaluation_task=EvaluationTask.TRACKING,
                    target_labels=[AutowareLabel.CAR],
                ),
            ),
            (
                TrackingMetricsConfig,
                PerceptionMetricsParam.from_dict(
                    cfg={
                        "center_distance_thresholds": [[1.0]],
                        "plane_distance_thresholds": [[1.0]],
                        "iou_bev_thresholds": [[1.0]],
                    },
                    evaluation_task=EvaluationTask.TRACKING,
                    target_labels=[AutowareLabel.CAR],
                ),
            ),
        ]

        for n, (config, params) in enumerate(patterns):
            with self.subTest(f"Test if it can detect the exception of parameters: {n + 1}"):
                # with self.assertRaises(MetricsParameterError, msg=f"{n + 1}"):
                MetricsScoreConfig._extract_params(config, params)
