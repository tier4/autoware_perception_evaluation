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

from inspect import signature
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel

from .config._metrics_config_base import _MetricsConfigBase
from .config.detection_metrics_config import DetectionMetricsConfig
from .config.prediction_metrics_config import PredictionMetricsConfig
from .config.tracking_metrics_config import TrackingMetricsConfig


class MetricsScoreConfig:
    """[summary]
    The config for each evaluation task metrics.

    Attributes:
        self.detection_config (Optional[DetectionMetricsConfig])
        self.tracking_config (Optional[DetectionMetricsConfig])
        self.prediction_config (Optional[PredictionMetricsConfig])
        self.evaluation_tasks (List[EvaluationTask])
    """

    def __init__(self, evaluation_task: EvaluationTask, **cfg) -> None:
        """[summary]

        Args:
            evaluation_task (EvaluationTask)
        """
        self.detection_config: Optional[DetectionMetricsConfig] = None
        self.tracking_config: Optional[TrackingMetricsConfig] = None

        # NOTE: prediction_config is under construction
        self.prediction_config = None

        self.evaluation_task: EvaluationTask = evaluation_task
        self.target_labels: List[AutowareLabel] = cfg["target_labels"]
        if self.evaluation_task in (EvaluationTask.DETECTION2D, EvaluationTask.DETECTION):
            self._check_parameters(DetectionMetricsConfig, cfg)
            self.detection_config = DetectionMetricsConfig(**cfg)
        elif self.evaluation_task == EvaluationTask.TRACKING:
            self._check_parameters(TrackingMetricsConfig, cfg)
            self.tracking_config = TrackingMetricsConfig(**cfg)
            # NOTE: In tracking, evaluate mAP too
            # TODO: Check and extract parameters for detection from parameters for tracking
            self.detection_config = DetectionMetricsConfig(**cfg)
        elif self.evaluation_task == EvaluationTask.PREDICTION:
            self._check_parameters(PredictionMetricsConfig, cfg)
            raise NotImplementedError("Prediction config is under construction")
            # TODO
            # self.evaluation_tasks.append(task)
        else:
            raise KeyError(f"Unsupported perception evaluation task: {self.evaluation_task}")

    @staticmethod
    def _check_parameters(config: _MetricsConfigBase, params: Dict[str, Any]):
        """Check if input parameters are valid.

        Args:
            config (_MetricsConfigBase): The config for metrics.
            params (Dict[str, any]): The parameters for metrics.

        Raises:
            KeyError: When got invalid parameter names.
        """
        valid_parameters: Set = set(signature(config).parameters)
        input_params: Set = set(params.keys())
        if not input_params <= valid_parameters:
            raise MetricsParameterError(
                f"MetricsConfig for '{config.evaluation_task}'\n"
                f"Unexpected parameters: {input_params - valid_parameters} \n"
                f"Usage: {valid_parameters} \n"
            )


class MetricsParameterError(Exception):
    def __init__(self, *args) -> None:
        super(MetricsParameterError, self).__init__(*args)
