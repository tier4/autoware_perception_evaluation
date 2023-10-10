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
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import pytest

from perception_eval.evaluation.metrics.config.detection_metrics_config import DetectionMetricsConfig
from perception_eval.evaluation.metrics.config.tracking_metrics_config import TrackingMetricsConfig
from perception_eval.evaluation.metrics.metrics_score_config import MetricsParameterError, MetricsScoreConfig

if TYPE_CHECKING:
    from perception_eval.evaluation.metrics.config._metrics_config_base import _MetricsConfigBase


class TestMetricsScoreConfig(unittest.TestCase):
    def test_check_parameters(self):
        """[summary]
        Test if it can detect the exception.

        test patterns:
            Check if the exception is raised when wrong key is specified.
        """
        patterns: List[Tuple[_MetricsConfigBase, Dict[str, Any]]] = [
            (DetectionMetricsConfig, {"foo": 0.0}),
            (
                DetectionMetricsConfig,
                {
                    "target_labels": ["car"],
                    "center_distance_thresholds": [[1.0]],
                    "plane_distance_thresholds": [[1.0]],
                    "iou_bev_thresholds": [[1.0]],
                    "iou_3d_thresholds": [[1.0]],
                    "foo": 1.0,
                },
            ),
            (TrackingMetricsConfig, {"foo": 0.0}),
            (
                TrackingMetricsConfig,
                {
                    "target_labels": ["car"],
                    "center_distance_thresholds": [[1.0]],
                    "plane_distance_thresholds": [[1.0]],
                    "iou_bev_thresholds": [[1.0]],
                    "iou_3d_thresholds": [[1.0]],
                    "foo": 1.0,
                },
            ),
        ]

        for n, (config, params) in enumerate(patterns):
            with self.subTest(f"Test if it can detect the exception of parameters: {n + 1}"):
                with pytest.raises(MetricsParameterError):
                    MetricsScoreConfig._check_parameters(config, params)
