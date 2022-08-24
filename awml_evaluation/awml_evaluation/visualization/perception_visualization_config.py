from typing import List
from typing import Optional

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel


class PerceptionVisualizationConfig:
    """The config of to visualize perception results.

    Attributes:
        self.visualization_directory_path (str): The directory path to save visualized results.
        evaluation_task (EvaluationTask)
        self.height (int): The height of image.
        self.width (int): The width of image.
        self.target_labels (Optional[List[AutowareLabel]]): target_labels
        self.max_x_position_list (Optional[List[float]]): max_x_position_list
        self.max_y_position_list (Optional[List[float]]): max_y_position_list
        self.max_distance_list (Optional[List[float]]): max_distance_list
        self.min_distance_list (Optional[List[float]]): min_distance_list
        self.min_point_numbers (Optional[List[int]]): min_point_numbers
        self.target_uuids (Optional[List[str]]) target_uuids
        self.xlim (float): The limit of x range defined by max_x_position of max_distance.
            When both of them are None, set 100.0.
        self.ylim (float): The limit of y range defined by max_y_position of max_distance.
            When both of them are None, set 100.0.
    """

    def __init__(
        self,
        visualization_directory_path: str,
        frame_id: str,
        evaluation_task: EvaluationTask,
        height: int = 640,
        width: int = 640,
        target_labels: Optional[List[AutowareLabel]] = None,
        max_x_position_list: Optional[List[float]] = None,
        max_y_position_list: Optional[List[float]] = None,
        max_distance_list: Optional[List[float]] = None,
        min_distance_list: Optional[List[float]] = None,
        min_point_numbers: Optional[List[int]] = None,
        confidence_threshold_list: Optional[List[float]] = None,
        target_uuids: Optional[List[str]] = None,
    ) -> None:
        """[summary]
        Args:
            visualization_directory_path (str): Path to save visualized result.
            frame_id (str): base_link  or map.
            evaluation_task (EvaluationTask): Name of evaluation.
            height (int): Image height. Defaults to 640.
            width (int): Image width. Defaults to 640.
            target_labels (Optional[List[AutowareLabel]]): The list of target label. Defaults to None.
            max_x_position_list (Optional[List[float]]): The list of max x position. Defaults to None.
            max_y_position_list (Optional[List[float]]): The list of max y position. Defaults to None.
            max_distance_list (Optional[List[float]]): The list of max distance. Defaults to None.
            min_distance_list (Optional[List[float]]): The list of min distance. Defaults to None.
            min_point_numbers (Optional[List[int]]): The list of min point numbers. Defaults to None.
            confidence_threshold_list (Optional[List[float]]): The list of confidence threshold. Defaults to None.
            target_uuids (Optional[List[str]]): The list of uuid. Defaults to None.
        """
        self.visualization_directory_path: str = visualization_directory_path
        self.frame_id: str = frame_id
        self.evaluation_task: EvaluationTask = evaluation_task
        self.height: int = height
        self.width: int = width

        self.target_labels: Optional[List[AutowareLabel]] = target_labels
        self.max_x_position_list: Optional[List[float]] = max_x_position_list
        self.max_y_position_list: Optional[List[float]] = max_y_position_list
        self.max_distance_list: Optional[List[float]] = max_distance_list
        self.min_distance_list: Optional[List[float]] = min_distance_list
        self.min_point_numbers: Optional[List[int]] = min_point_numbers
        self.confidence_threshold_list: Optional[List[float]] = confidence_threshold_list
        self.target_uuids: Optional[List[str]] = target_uuids

        if max_x_position_list is None:
            self.xlim: float = max(max_distance_list)
            self.ylim: float = max(max_distance_list)
        elif max_distance_list is None:
            self.xlim: float = max(max_x_position_list)
            self.ylim: float = max(max_y_position_list)
        else:
            self.xlim: float = 100.0
            self.ylim: float = 100.0
