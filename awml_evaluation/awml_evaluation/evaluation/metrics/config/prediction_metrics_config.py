from typing import List

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel

from ._metrics_config_base import _MetricsConfigBase


class PredictionMetricsConfig(_MetricsConfigBase):
    """[summary]
    The config for prediction evaluation metrics.

    Attributes:
        self.target_labels (List[AutowareLabel]): The list of targets to evaluate
        self.max_x_position_list (List[float])
        self.max_y_position_list (List[float])
    """

    evaluation_task = EvaluationTask.PREDICTION

    def __init__(
        self,
        target_labels: List[AutowareLabel],
        max_x_position: float,
        max_y_position: float,
    ) -> None:
        """[summary]
        Args:
            target_labels (List[AutowareLabel]): The list of targets to evaluate.
            max_x_position (float):
                    The threshold of maximum x-axis position for each object.
                    Return the object that
                    - max_x_position < object x-axis position < max_x_position.
                    This param use for range limitation of detection algorithm.
            max_y_position (float):
                    The threshold list of maximum y-axis position for each object.
                    Return the object that
                    - max_y_position < object y-axis position < max_y_position.
                    This param use for range limitation of detection algorithm.
        """
        super().__init__(
            target_labels=target_labels,
            max_x_position=max_x_position,
            max_y_position=max_y_position,
        )
