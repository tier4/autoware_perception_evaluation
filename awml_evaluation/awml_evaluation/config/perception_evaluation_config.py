from typing import Any
from typing import Dict
from typing import List

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.evaluation_task import set_task
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.label import set_target_lists
from awml_evaluation.evaluation.metrics.metrics_score_config import MetricsScoreConfig

from ._evaluation_config_base import _EvaluationConfigBase


class PerceptionEvaluationConfig(_EvaluationConfigBase):
    """[summary]
    Evaluation configure class

    Attributes:
    self.dataset_paths (List[str]): The list of dataset path.
        self.frame_id (str): The frame_id, base_link or map.
        self.does_use_pointcloud (bool): The boolean flag if load pointcloud data from dataset.
        self.result_root_directory (str): The path to result directory
        self.log_directory (str): The path to sub directory for log
        self.visualization_directory (str): The path to sub directory for visualization
        self.label_converter (LabelConverter): The label convert class
        self.metrics_config (MetricsScoreConfig): The config for metrics
    """

    _support_tasks: List[str] = ["detection", "tracking", "prediction"]

    def __init__(
        self,
        dataset_paths: List[str],
        frame_id: str,
        does_use_pointcloud: bool,
        result_root_directory: str,
        evaluation_config_dict: Dict[str, Any],
    ) -> None:
        """[summary]

        Args:
            dataset_paths (List[str]): The paths of dataset
            frame_id (str): The frame_id base_link or map
            does_use_pointcloud (bool): The flag for loading pointcloud data from dataset
            result_root_directory (str): The path to result directory
            evaluation_config_dict (Dict[str, Dict[str, Any]]): The dictionary of evaluation config for each task.
                                          This has a key of evaluation task name which support in EvaluationTask class(ex. ["detection", "tracking", "prediction"])
        """
        super().__init__(
            dataset_paths=dataset_paths,
            frame_id=frame_id,
            does_use_pointcloud=does_use_pointcloud,
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
        )

        # Covert labels to autoware labels for Metrics
        autoware_target_labels: List[AutowareLabel] = set_target_lists(
            evaluation_config_dict["target_labels"],
            self.label_converter,
        )
        evaluation_config_dict["target_labels"] = autoware_target_labels

        # Convert task names to EvaluationTask
        self.evaluation_task: EvaluationTask = set_task(evaluation_config_dict["evaluation_task"])
        evaluation_config_dict["evaluation_task"] = self.evaluation_task

        # Set for thresholds Union[List[List[float]], List[float]]
        self.metrics_config: MetricsScoreConfig = MetricsScoreConfig(
            metrics_config_dict=evaluation_config_dict,
        )
