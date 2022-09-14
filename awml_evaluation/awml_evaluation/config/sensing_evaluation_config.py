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

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from awml_evaluation.evaluation.sensing.sensing_frame_config import SensingFrameConfig

from ._evaluation_config_base import _EvaluationConfigBase


class SensingEvaluationConfig(_EvaluationConfigBase):
    """The class of config for sensing evaluation.

    Attributes:
        - By _EvaluationConfigBase:
        self.dataset_paths (List[str]): The path(s) of dataset(s).
        self.frame_id (str): The frame_id, base_link or map.
        self.does_use_pointcloud (bool): The boolean flag if load pointcloud data from dataset.
        self.result_root_directory (str): The directory path to save result.
        self.log_directory (str): The directory path to save log.
        self.visualization_directory (str): The directory path to save visualization result.
        self.label_converter (LabelConverter): The instance to convert label names.
        self.evaluation_task (EvaluationTask): The instance of EvaluationTask

        - By SensingEvaluationConfig
        self.filtering_params (Dict[str, Any]): Filtering parameters.
        self.metrics_params (Dict[str, Any]): Metrics parameters.
        self.sensing_frame_config (SensingFrameConfig)
    """

    _support_tasks: List[str] = ["sensing"]

    def __init__(
        self,
        dataset_paths: List[str],
        frame_id: str,
        merge_similar_labels: bool,
        does_use_pointcloud: bool,
        result_root_directory: str,
        evaluation_config_dict: Dict[str, Dict[str, Any]],
    ):
        """
        Args:
            dataset_paths (List[str]): The list of dataset paths.
            frame_id (str): Frame ID, base_link or map.
            merge_similar_labels (bool): Whether merge similar labels.
                If True,
                    - BUS, TRUCK, TRAILER -> CAR
                    - MOTORBIKE, CYCLIST -> BICYCLE
            does_use_pointcloud (bool): The boolean flag if load pointcloud data from dataset.
            result_root_directory (str): The directory path to save result.
            evaluation_config_dict (Dict[str, Dict[str, Any]]): The dictionary of evaluation config for each task.
                                          This has a key of evaluation task name which support in EvaluationTask class(ex. ["sensing"])
        """
        super().__init__(
            dataset_paths=dataset_paths,
            frame_id=frame_id,
            merge_similar_labels=merge_similar_labels,
            does_use_pointcloud=does_use_pointcloud,
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
        )
        self.filtering_params, self.metrics_params = self._extract_params(evaluation_config_dict)
        self.sensing_frame_config: SensingFrameConfig = SensingFrameConfig(**self.metrics_params)

    def _extract_params(
        self,
        evaluation_config_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """"""
        e_cfg: Dict[str, Any] = evaluation_config_dict.copy()
        f_params: Dict[str, Any] = {"target_uuids": e_cfg.get("target_uuids", None)}
        m_params: Dict[str, Any] = {
            "box_scale_0m": e_cfg.get("box_scale_0m", 1.0),
            "box_scale_100m": e_cfg.get("box_scale_100m", 1.0),
            "min_points_threshold": e_cfg.get("min_points_threshold", 1),
        }
        return f_params, m_params
