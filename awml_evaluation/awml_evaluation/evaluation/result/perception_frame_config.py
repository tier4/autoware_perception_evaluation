from typing import List

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.label import set_target_lists
from awml_evaluation.common.threshold import check_thresholds
from awml_evaluation.config.perception_evaluation_config import PerceptionEvaluationConfig


class CriticalObjectFilterConfig:
    """[summary]
    Config class for critical object filter

    Attributes:
        self.target_labels (List[str]): Target list
        self.max_pos_distance_list (List[float]]):
                Maximum distance threshold list for object. Defaults to None.
        self.min_pos_distance_list (List[float]):
                Minimum distance threshold list for object. Defaults to None.
    """

    def __init__(
        self,
        evaluator_config: PerceptionEvaluationConfig,
        target_labels: List[str],
        max_x_position_list: List[float],
        max_y_position_list: List[float],
    ) -> None:
        """[summary]

        Args:
            evaluator_config (PerceptionEvaluationConfig): Evaluation config
            target_labels (List[str]): Target list
            max_pos_distance_list (List[float]]):
                    Maximum distance threshold list for object. Defaults to None.
            min_pos_distance_list (List[float]):
                    Minimum distance threshold list for object. Defaults to None.
        """
        self.target_labels: List[AutowareLabel] = set_target_lists(
            target_labels,
            evaluator_config.label_converter,
        )
        self.max_x_position_list: List[float] = check_thresholds(
            max_x_position_list,
            self.target_labels,
        )
        self.max_y_position_list: List[float] = check_thresholds(
            max_y_position_list,
            self.target_labels,
        )


class PerceptionPassFailConfig:
    """[summary]
    Config filter for pass fail to frame result

    Attributes:
        self.target_labels (List[str]): Target list
        self.threshold_plane_distance_list (List[float]): The threshold list for plane distance
    """

    def __init__(
        self,
        evaluator_config: PerceptionEvaluationConfig,
        target_labels: List[str],
        threshold_plane_distance_list: List[float],
    ) -> None:
        """[summary]
        Args:
            evaluator_config (PerceptionEvaluationConfig): Evaluation config
            target_labels (List[str]): Target list
            threshold_plane_distance_list (List[float]): The threshold list for plane distance
        """
        self.target_labels: List[AutowareLabel] = set_target_lists(
            target_labels,
            evaluator_config.label_converter,
        )
        self.threshold_plane_distance_list: List[float] = check_thresholds(
            threshold_plane_distance_list,
            self.target_labels,
        )


class UseCaseThresholdsError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)
