from typing import Any
from typing import Dict
from typing import List

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.evaluation_task import set_task
from awml_evaluation.evaluation.sensing.sensing_frame_config import SensingFrameConfig

from ._evaluation_config_base import _EvaluationConfigBase


class SensingEvaluationConfig(_EvaluationConfigBase):
    """The class of config for sensing evaluation.

    Attributes:
        self.dataset_paths (List[str]): The path(s) of dataset(s).
        self.frame_id (str): The frame_id, base_link or map.
        self.does_use_pointcloud (bool): The boolean flag if load pointcloud data from dataset.
        self.result_root_directory (str): The directory path to save result.
        self.log_directory (str): The directory path to save log.
        self.visualization_directory (str): The directory path to save visualization result.
        self.target_uuids (List[str]): The list of objects' uuids should be detected.
        self.label_converter (LabelConverter): The instance to convert label names.
    """

    _support_tasks: List[str] = ["sensing"]

    def __init__(
        self,
        dataset_paths: List[str],
        frame_id: str,
        does_use_pointcloud: bool,
        result_root_directory: str,
        evaluation_config_dict: Dict[str, Dict[str, Any]],
    ):
        """
        Args:
            dataset_paths (List[str]): The list of dataset paths.
            does_use_pointcloud (bool): The boolean flag if load pointcloud data from dataset.
            result_root_directory (str): The directory path to save result.
            evaluation_config_dict (Dict[str, Dict[str, Any]]): The dictionary of evaluation config for each task.
                                          This has a key of evaluation task name which support in EvaluationTask class(ex. ["sensing"])
        """
        super().__init__(
            dataset_paths=dataset_paths,
            frame_id=frame_id,
            does_use_pointcloud=does_use_pointcloud,
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
        )
        self.evaluation_task: EvaluationTask = set_task(
            evaluation_config_dict.pop("evaluation_task")
        )
        # target uuids
        self.target_uuids: List[str] = evaluation_config_dict.pop("target_uuids")

        # config per frame
        self.sensing_frame_config: SensingFrameConfig = SensingFrameConfig(**evaluation_config_dict)
