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
from typing import TYPE_CHECKING
from typing import Union

from perception_eval.common.evaluation_task import set_task
from perception_eval.common.label import LabelConverter
from perception_eval.common.label import set_target_lists
from perception_eval.common.schema import FrameID

from .params import LabelParam

if TYPE_CHECKING:
    from perception_eval.common.evaluation_task import EvaluationTask

    from .params import FilterParamType
    from .params import MetricsParamType


class EvaluationConfigBase(ABC):
    """Abstract base class for evaluation config

    Directory structure to save log and visualization result is following
    - result_root_directory/
        ├── log_directory/
        └── visualization_directory/

    Args:
        dataset_paths (List[str]): Dataset paths list.
        frame_id (Union[str, Sequence[str]]): FrameID(s) in string, where objects are with respect.
        result_root_directory (str): Directory path to save result.
        evaluation_config_dict (Dict[str, Dict[str, Any]]): Dict that items are evaluation config for each task.
        load_raw_data (bool): Whether load pointcloud/image data. Defaults to False.
    """

    @abstractmethod
    def __init__(
        self,
        dataset_paths: List[str],
        frame_id: Union[str, Sequence[str]],
        result_root_directory: str,
        config_dict: Dict[str, Any],
        load_raw_data: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_paths: List[str] = dataset_paths
        self.load_raw_data: bool = load_raw_data

        # convert string to object
        self.evaluation_task: EvaluationTask = self._load_task(config_dict)
        self.label_param = LabelParam.from_dict(config_dict)
        self.label_converter = LabelConverter(self.evaluation_task, **self.label_param.as_dict())
        self.target_labels = set_target_lists(config_dict.get("target_labels"), self.label_converter)
        config_dict["evaluation_task"] = self.evaluation_task
        config_dict["target_labels"] = self.target_labels

        self.filter_param, self.metrics_param = self._build_params(config_dict)

        # frame id
        self.frame_ids: List[FrameID] = (
            [FrameID.from_value(frame_id)] if isinstance(frame_id, str) else [FrameID.from_value(f) for f in frame_id]
        )
        if self.evaluation_task.is_3d() and len(self.frame_ids) != 1:
            raise ValueError(f"For 3D task, FrameID must be 1, but got {len(self.frame_ids)}")

        # directory
        time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        self.result_root_directory: str = result_root_directory.format(TIME=time)
        self.__log_directory: str = osp.join(self.result_root_directory, "log")
        self.__visualization_directory: str = osp.join(self.result_root_directory, "visualization")
        if not osp.exists(self.log_directory):
            os.makedirs(self.log_directory)
        if not osp.exists(self.visualization_directory):
            os.makedirs(self.visualization_directory)

    @property
    def support_tasks(self) -> List[str]:
        return self._support_tasks

    def _load_task(self, cfg: Dict[str, Any]) -> EvaluationTask:
        """Check if specified tasks are supported.

        Args:
            evaluation_config_dict (Dict[str, Any]): The config has params as dict.

        Returns:
            evaluation_task (EvaluationTask): Evaluation task.

        Raises:
            ValueError: If the keys of input config are unsupported.
        """
        task_name: str = cfg["evaluation_task"]
        if not self._validate_task(task_name):
            raise ValueError(f"Unsupported task: {task_name}\nSupported tasks: {self.support_tasks}")
        return set_task(task_name)

    @abstractmethod
    def _validate_task(self, task: str | EvaluationTask) -> bool:
        pass

    @abstractmethod
    def _build_params(self, cfg: Dict[str, Any]) -> Tuple[FilterParamType, MetricsParamType]:
        """
        Build parameters from config dict.

        Args:
            cfg (Dict[str, Any]):

        Returns:
            Tuple[FilterParamType, MetricsParamType]: _description_
        """
        pass

    @property
    def log_directory(self) -> str:
        return self.__log_directory

    @property
    def visualization_directory(self) -> str:
        return self.__visualization_directory

    def dump_label_params(self) -> Dict[str, Any]:
        return self.label_param.as_dict()

    def dump_filter_params(self) -> Dict[str, Any]:
        return self.filter_param.as_dict()

    def dump_metrics_params(self) -> Dict[str, Any]:
        return self.metrics_param.as_dict()
