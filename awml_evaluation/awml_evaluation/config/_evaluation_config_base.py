from abc import ABC
from abc import abstractmethod
import datetime
import os.path as osp
from typing import Any
from typing import Dict
from typing import List

from awml_evaluation.common.label import LabelConverter


class _EvaluationConfigBase(ABC):
    """Abstract base class for evaluation config

    Attributes:
        self.dataset_paths (List[str]): The list of dataset path.
        self.frame_id (str): The frame_id, base_link or map.
        self.does_use_pointcloud (bool): Whether use pointcloud of dataset.
        self.result_root_directory (str): The directory path to save result.
        self.log_directory (str): The directory path to save log.
        self.visualization_directory (str): The directory path to save visualization result.
        self.label_converter (LabelConverter): The converter to convert string label to autoware format.

    properties:
        self.support_tasks (List[str]): The list of supported task of EvaluationManager.
            (e.g.)
            - PerceptionEvaluationManager: ["detection", "tracking", "prediction"]
            - SensingEvaluationManager: ["sensing"]
    """

    _support_tasks: List[str] = []

    @abstractmethod
    def __init__(
        self,
        dataset_paths: List[str],
        frame_id: str,
        does_use_pointcloud: bool,
        result_root_directory: str,
        log_directory: str,
        visualization_directory: str,
        evaluation_config_dict: Dict[str, Any],
    ) -> None:
        """[summary]
        Args:
            dataset_paths (List[str]): The list of dataset path.
            does_use_pointcloud (bool): Whether use pointcloud of dataset.
            result_root_directory (str): The directory path to save result.
            log_directory (str): The directory path to save log.
            visualization_directory (str): The directory path to save visualization result.
            evaluation_config_dict (Dict[str, Any]): The config for each evaluation task. The key represents task name.
        """
        super().__init__()
        # Check tasks are supported
        self._check_tasks(evaluation_config_dict)

        # dataset
        self.dataset_paths: List[str] = dataset_paths

        if frame_id not in ("base_link", "map"):
            raise ValueError(f"Unexpected frame_id: {frame_id}")
        self.frame_id: str = frame_id
        self.does_use_pointcloud: bool = does_use_pointcloud

        # directory
        time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        self.result_root_directory: str = result_root_directory.format(TIME=time)
        self.log_directory: str = log_directory
        self.visualization_directory: str = visualization_directory

        # Labels
        self.label_converter = LabelConverter()

    @property
    def support_tasks(cls) -> List[str]:
        return cls._support_tasks

    def _check_tasks(self, evaluation_config_dict: Dict[str, Any]):
        """[summary]
        Check if specified tasks are supported.

        Args:
            evaluation_config_dict (Dict[str, Any]): The keys of config must be in supported task names.

        Raises:
            ValueError: If the keys of input config are unsupported.
        """
        task: str = evaluation_config_dict["evaluation_task"]
        if task not in self.support_tasks:
            raise ValueError(f"Unsupported task: {task}\nSupported tasks: {self.support_tasks}")

    def get_result_log_directory(self) -> str:
        """[summary]
        Get the full path to put logs
        Returns:
            str: The full path to put logs
        """
        return osp.join(self.result_root_directory, self.log_directory)

    def get_result_visualization_directory(self) -> str:
        """[summary]
        Get the full path to put the visualization images
        Returns:
            str: The full path to put the visualization images"""
        return osp.join(self.result_root_directory, self.visualization_directory)
