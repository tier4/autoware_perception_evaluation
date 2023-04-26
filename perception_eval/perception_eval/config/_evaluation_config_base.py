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
from typing import Tuple

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.evaluation_task import set_task
from perception_eval.common.label import LabelConverter
from perception_eval.common.status import FrameID


class _EvaluationConfigBase(ABC):
    """Abstract base class for evaluation config

    Directory structure to save log and visualization result is following
    - result_root_directory/
        ├── log_directory/
        └── visualization_directory/

    Attributes:
        dataset_paths (List[str]): Dataset paths list.
        frame_id (FrameID): FrameID instance, where objects are with respect.
        result_root_directory (str): Directory path to save result.
        log_directory (str): Directory Directory path to save log.
        visualization_directory (str): Directory path to save visualization result.
        label_converter (LabelConverter): LabelConverter instance.
        evaluation_task (EvaluationTask): EvaluationTask instance.
        label_prefix (str): Prefix of label type. Choose from [`autoware", `traffic_light`]. Defaults to autoware.
        load_raw_data (bool): Whether load pointcloud/image data. Defaults to False.
        target_labels (List[LabelType]): Target labels list.

    properties:
        support_tasks (List[str]): The list of supported task of EvaluationManager.
            PerceptionEvaluationManager: [
                "detection",
                "tracking",
                "prediction",
                "detection2d",
                "tracking2d",
                "classification2d"
            ]
            SensingEvaluationManager: ["sensing"]

    Args:
        dataset_paths (List[str]): Dataset paths list.
        frame_id (str): FrameID in string, where objects are with respect.
        merge_similar_labels (bool): Whether merge similar labels.
            If True,
                - BUS, TRUCK, TRAILER -> CAR
                - MOTORBIKE, CYCLIST -> BICYCLE
        result_root_directory (str): Directory path to save result.
        evaluation_config_dict (Dict[str, Dict[str, Any]]): Dict that items are evaluation config for each task.
        label_prefix (str): Prefix of label type. Choose from `autoware` or `traffic_light`. Defaults to autoware.
        load_raw_data (bool): Whether load pointcloud/image data. Defaults to False.
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
        load_raw_data: bool = False,
    ) -> None:
        super().__init__()
        # Check tasks are supported
        self.evaluation_task: EvaluationTask = self._check_tasks(evaluation_config_dict)
        self.evaluation_config_dict: Dict[str, Any] = evaluation_config_dict

        # dataset
        self.dataset_paths: List[str] = dataset_paths

        self.frame_id: FrameID = FrameID.from_value(frame_id)
        self.merge_similar_labels: bool = merge_similar_labels
        self.label_prefix: str = label_prefix
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
        self.label_converter = LabelConverter(
            self.evaluation_task,
            merge_similar_labels,
            label_prefix,
            count_label_number=True,
        )

    @property
    def support_tasks(self) -> List[str]:
        return self._support_tasks

    def _check_tasks(self, evaluation_config_dict: Dict[str, Any]) -> EvaluationTask:
        """Check if specified tasks are supported.

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
        """Extract filtering and metrics parameters from evaluation config.

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
