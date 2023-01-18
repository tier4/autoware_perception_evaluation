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

from abc import ABC
from abc import abstractmethod
import datetime
import os
import os.path as osp
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.evaluation_task import set_task
from perception_eval.common.label import LabelConverter


class _EvaluationConfigBase(ABC):
    """Abstract base class for evaluation config

    Attributes:
        self.dataset_paths (List[str]): The list of dataset path.
        self.frame_id (str): The frame_id, base_link or map.
        self.merge_similar_labels (bool): Whether merge similar labels.
        self.load_raw_data (bool): Whether load pointcloud/image data.
        self.result_root_directory (str): The directory path to save result.
        self.log_directory (str): The directory path to save log.
        self.visualization_directory (str): The directory path to save visualization result.
        self.label_converter (LabelConverter): The converter to convert string label to autoware format.
        self.evaluation_config_dict (Dict[str, Any]): The original config dict.

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
        merge_similar_labels: bool,
        result_root_directory: str,
        evaluation_config_dict: Dict[str, Any],
        label_prefix: str = "autoware",
        camera_type: Optional[str] = None,
        load_raw_data: bool = False,
    ) -> None:
        """[summary]
        Args:
            dataset_paths (List[str]): The list of dataset path.
            frame_id (str): Frame ID, base_link or map.
            merge_similar_labels (bool): Whether merge similar labels.
                If True,
                    - BUS, TRUCK, TRAILER -> CAR
                    - MOTORBIKE, CYCLIST -> BICYCLE
            result_root_directory (str): The directory path to save result.
            evaluation_config_dict (Dict[str, Any]): The config for each evaluation task. The key represents task name.
            label_prefix (str): Defaults to autoware.
            load_raw_data (bool): Defaults to False.
        """
        super().__init__()
        # Check tasks are supported
        self.evaluation_task: EvaluationTask = self._check_tasks(evaluation_config_dict)
        self.evaluation_config_dict: Dict[str, Any] = evaluation_config_dict

        # dataset
        self.dataset_paths: List[str] = dataset_paths

        if frame_id not in ("base_link", "map"):
            raise ValueError(f"Unexpected frame_id: {frame_id}")
        self.frame_id: str = frame_id
        self.merge_similar_labels: bool = merge_similar_labels
        self.label_prefix: str = label_prefix
        self.camera_type: Optional[str] = camera_type
        self.load_raw_data: bool = load_raw_data

        # directory
        time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        self.result_root_directory: str = result_root_directory.format(TIME=time)
        self.__log_directory: str = osp.join(self.result_root_directory, "log")
        self.__visualization_directory: str = osp.join(self.result_root_directory, "visualization")
        if not osp.exists(self.log_directory):
            os.makedirs(self.log_directory)
        if not osp.exists(self.visualization_directory):
            os.makedirs(self.visualization_directory)

        # Labels
        self.label_converter = LabelConverter(merge_similar_labels, label_prefix)

    @property
    def support_tasks(self) -> List[str]:
        return self._support_tasks

    def _check_tasks(self, evaluation_config_dict: Dict[str, Any]) -> EvaluationTask:
        """[summary]
        Check if specified tasks are supported.

        Args:
            evaluation_config_dict (Dict[str, Any]): The config has params as dict.

        Returns:
            evaluation_task (EvaluationTask): Evaluation task.

        Raises:
            ValueError: If the keys of input config are unsupported.
        """
        task: str = evaluation_config_dict["evaluation_task"]
        if task not in self.support_tasks:
            raise ValueError(f"Unsupported task: {task}\nSupported tasks: {self.support_tasks}")

        # evaluation task
        evaluation_task: EvaluationTask = set_task(task)
        return evaluation_task

    @abstractmethod
    def _extract_params(
        self,
        evaluation_config_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """[summary]
        Extract filtering and metrics parameters from evaluation config.

        Args:
            evaluation_config_dict (Dict[str, Any])

        Returns:
            filter_params (Dict[str, Any]): filtering parameters.
            metrics_params (Dict[str, Any]): metrics parameters.
        """
        pass

    @property
    def log_directory(self) -> str:
        return self.__log_directory

    @property
    def visualization_directory(self) -> str:
        return self.__visualization_directory
