from inspect import signature
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set

from awml_evaluation.common.evaluation_task import EvaluationTask

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

    def __init__(self, metrics_config_dict: Dict[str, Any]) -> None:
        """[summary]

        Args:
            metrics_config_dict (Dict[str, Any]):
        """
        self.detection_config: Optional[DetectionMetricsConfig] = None
        self.tracking_config: Optional[TrackingMetricsConfig] = None

        # NOTE: prediction_config is under construction
        self.prediction_config = None

        self.evaluation_task: EvaluationTask = metrics_config_dict.pop("evaluation_task")
        if self.evaluation_task == EvaluationTask.DETECTION:
            self._check_parameters(DetectionMetricsConfig, metrics_config_dict)
            self.detection_config = DetectionMetricsConfig(**metrics_config_dict)
        elif self.evaluation_task == EvaluationTask.TRACKING:
            self._check_parameters(TrackingMetricsConfig, metrics_config_dict)
            self.tracking_config = TrackingMetricsConfig(**metrics_config_dict)
            # NOTE: In tracking, evaluate mAP too
            # TODO: Check and extract parameters for detection from parameters for tracking
            detection_metrics_config_dict = metrics_config_dict.copy()
            detection_metrics_config_dict.update(
                {"min_point_numbers": [0] * len(metrics_config_dict["target_labels"])}
            )
            self.detection_config = DetectionMetricsConfig(**detection_metrics_config_dict)
        elif self.evaluation_task == EvaluationTask.PREDICTION:
            self._check_parameters(PredictionMetricsConfig, metrics_config_dict)
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
