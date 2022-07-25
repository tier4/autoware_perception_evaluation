from typing import List
from typing import Optional

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.threshold import check_thresholds


class PerceptionVisualizationConfig:
    """The config of to visualize perception results.

    Attributes:
        self.visualization_directory_path (str): The directory path to save visualized results.
        evaluation_task (EvaluationTask)
        self.height (int): The height of image.
        self.width (int): The width of image.
        self.visualize_box (bool): Whether visualize bounding boxes.
        self.visualize_tracking (bool): Whether visualize tracked paths.
        self.visualize_prediction (bool): Whether visualize predicted paths.
    """

    def __init__(
        self,
        visualization_directory_path: str,
        frame_id: str,
        evaluation_task: EvaluationTask,
        height: int = 640,
        width: int = 640,
        target_labels: Optional[List[AutowareLabel]] = None,
        max_x_position: Optional[float] = None,
        max_y_position: Optional[float] = None,
    ) -> None:
        self.visualization_directory_path: str = visualization_directory_path
        self.frame_id: str = frame_id
        self.evaluation_task: EvaluationTask = evaluation_task
        self.height: int = height
        self.width: int = width

        self.target_labels: Optional[List[AutowareLabel]] = target_labels
        self.max_x_position: Optional[float] = max_x_position
        self.max_y_position: Optional[float] = max_y_position

        if target_labels is not None:
            if max_x_position is not None:
                max_x_position_list = [max_x_position] * len(target_labels)
                self.max_x_position_list: List[float] = check_thresholds(
                    max_x_position_list,
                    self.target_labels,
                )

            if max_y_position is not None:
                max_y_position_list = [max_y_position] * len(target_labels)
                self.max_y_position_list: List[float] = check_thresholds(
                    max_y_position_list,
                    self.target_labels,
                )
        else:
            if max_x_position is not None:
                self.max_x_position_list = [max_x_position]

            if max_y_position is not None:
                max_y_position_list = [max_y_position] * len(target_labels)
                self.max_y_position_list: List[float] = check_thresholds(
                    max_y_position_list,
                    self.target_labels,
                )
