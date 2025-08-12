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
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import datetime
import os
import os.path as osp
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.evaluation_task import set_task
from perception_eval.common.label import LabelConverter
from perception_eval.common.schema import FrameID


class _EvaluationConfigBase(ABC):
    """Abstract base class for evaluation config

    Directory structure to save log and visualization result is following
    ```
    result_root_directory/
        ├── log_directory/
        └── visualization_directory/
    ```

    Attributes:
        dataset_paths (List[str]): Dataset paths list.
        frame_ids (List[FrameID]): List of FrameID instances, where objects are with respect.
        result_root_directory (str): Directory path to save result.
        log_directory (str): Directory Directory path to save log.
        visualization_directory (str): Directory path to save visualization result.
        label_converter (LabelConverter): LabelConverter instance.
        evaluation_task (EvaluationTask): EvaluationTask instance.
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
        frame_id (Union[str, Sequence[str]]): FrameID(s) in string, where objects are with respect.
        result_root_directory (str): Directory path to save result.
        evaluation_config_dict (Dict[str, Dict[str, Any]]): Dict that items are evaluation config for each task.
        load_raw_data (bool): Whether load pointcloud/image data. Defaults to False.
    """

    _support_tasks: List[str] = []

    @abstractmethod
    def __init__(
        self,
        dataset_paths: List[str],
        frame_id: Union[str, Sequence[str]],
        result_root_directory: str,
        evaluation_config_dict: Dict[str, Any],
        load_raw_data: bool = False,
    ) -> None:
        super().__init__()
        # Check tasks are supported
        self.evaluation_task: EvaluationTask = self._check_tasks(evaluation_config_dict)
        self.evaluation_config_dict: Dict[str, Any] = evaluation_config_dict

        # Labels
        self.label_params = self._extract_label_params(evaluation_config_dict)
        self.label_converter = LabelConverter(
            self.evaluation_task,
            self.label_params["merge_similar_labels"],
            self.label_params["label_prefix"],
            self.label_params["count_label_number"],
        )

        self.filtering_params, self.metrics_params = self._extract_params(evaluation_config_dict)

        # dataset
        self.dataset_paths: List[str] = dataset_paths

        self.frame_id = frame_id
        self.frame_ids: List[FrameID] = (
            [FrameID.from_value(frame_id)] if isinstance(frame_id, str) else [FrameID.from_value(f) for f in frame_id]
        )
        if self.evaluation_task.is_3d() and len(self.frame_ids) != 1:
            raise ValueError(f"For 3D task, FrameID must be 1, but got {len(self.frame_ids)}")

        self.load_raw_data: bool = load_raw_data

        # directory
        time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        self.result_root_directory: str = result_root_directory.format(TIME=time)
        self.__log_directory: str = osp.join(self.result_root_directory, "log")
        self.__visualization_directory: str = osp.join(self.result_root_directory, "visualization")
        if not osp.exists(self.log_directory):
            try:
                os.makedirs(self.log_directory)
            except Exception as e:
                print(e)
        if not osp.exists(self.visualization_directory):
            try:
                os.makedirs(self.visualization_directory)
            except Exception as e:
                print(e)

    def __reduce__(self) -> Tuple[_EvaluationConfigBase, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling."""
        return (
            self.__class__,
            (
                self.dataset_paths,
                self.frame_id,
                self.result_root_directory,
                self.evaluation_config_dict,
                self.load_raw_data,
            ),
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

    @staticmethod
    @abstractmethod
    def _extract_label_params(evaluation_config_dict: Dict[str, Any]) -> Dict[str, Any]:
        pass

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

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        return {
            "dataset_paths": self.dataset_paths,
            "frame_id": self.frame_id,
            "result_root_directory": self.result_root_directory,
            "evaluation_config_dict": self.evaluation_config_dict,
            "load_raw_data": self.load_raw_data,
        }

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> _EvaluationConfigBase:
        """Deserialize the data to _EvaluationConfigBase."""
        return cls(
            dataset_paths=data["dataset_paths"],
            frame_id=data["frame_id"],
            result_root_directory=data["result_root_directory"],
            evaluation_config_dict=data["evaluation_config_dict"],
            load_raw_data=data["load_raw_data"],
        )
