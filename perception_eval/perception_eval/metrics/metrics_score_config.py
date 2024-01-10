# Copyright 2022-2024 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from inspect import signature
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING

from perception_eval.common.evaluation_task import EvaluationTask

from .config._metrics_config_base import _MetricsConfigBase
from .config.classification_metrics_config import ClassificationMetricsConfig
from .config.detection_metrics_config import DetectionMetricsConfig
from .config.prediction_metrics_config import PredictionMetricsConfig
from .config.tracking_metrics_config import TrackingMetricsConfig

if TYPE_CHECKING:
    from perception_eval.config.params import PerceptionMetricsParam


class MetricsScoreConfig:
    """A configuration class for each evaluation task metrics.

    Args:
    -----
        params (PerceptionMetricsParam): Parameters for metrics score.
    """

    def __init__(self, params: PerceptionMetricsParam) -> None:
        self.detection_config: Optional[DetectionMetricsConfig] = None
        self.tracking_config: Optional[TrackingMetricsConfig] = None
        self.classification_config: Optional[ClassificationMetricsConfig] = None

        # NOTE: prediction_config is under construction
        self.prediction_config = None

        self.evaluation_task = params.evaluation_task
        self.target_labels = params.target_labels
        if self.evaluation_task in (EvaluationTask.DETECTION2D, EvaluationTask.DETECTION):
            inputs = self._extract_params(DetectionMetricsConfig, params)
            self.detection_config = DetectionMetricsConfig(**inputs)
        elif self.evaluation_task in (EvaluationTask.TRACKING2D, EvaluationTask.TRACKING):
            inputs = self._extract_params(TrackingMetricsConfig, params)
            self.tracking_config = TrackingMetricsConfig(**inputs)
            # NOTE: In tracking, evaluate mAP too
            # TODO: Check and extract parameters for detection from parameters for tracking
            self.detection_config = DetectionMetricsConfig(**inputs)
        elif self.evaluation_task == EvaluationTask.PREDICTION:
            inputs = self._extract_params(PredictionMetricsConfig, params)
            raise NotImplementedError("Prediction config is under construction")
            # TODO
            # self.evaluation_tasks.append(task)
        elif self.evaluation_task == EvaluationTask.CLASSIFICATION2D:
            inputs = self._extract_params(ClassificationMetricsConfig, params)
            self.classification_config = ClassificationMetricsConfig(**inputs)

    @staticmethod
    def _extract_params(config: _MetricsConfigBase, params: PerceptionMetricsParam) -> Dict[str, Any]:
        """Check if input parameters are valid.

        Args:
        -----
            config (_MetricsConfigBase): Metrics score instance.
            params (PerceptionMetricsParam): Parameters for metrics.

        Returns:
        --------
            Dict[str, Any]: Extracted params.

        Raises:
        -------
            KeyError: When got invalid parameter names.
        """
        input_params_dict = params.as_dict()
        valid_params: Set = set(signature(config).parameters)
        input_params: Set = set(input_params_dict.keys())
        if valid_params > input_params:
            raise MetricsParameterError(
                f"MetricsConfig for '{config.evaluation_task}'\n"
                f"Unexpected parameters: {input_params - valid_params} \n"
                f"Usage: {valid_params} \n"
            )
        return {key: input_params_dict[key] for key in valid_params}


class MetricsParameterError(Exception):
    def __init__(self, *args) -> None:
        super(MetricsParameterError, self).__init__(*args)
