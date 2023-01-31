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

from perception_eval.common.status import FrameID

from ._evaluation_config_base import _EvaluationConfigBase


class SensingEvaluationConfig(_EvaluationConfigBase):
    """The class of config for sensing evaluation.

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
        camera_type (Optional[str]): Camera name. Specify in 2D evaluation. Defaults to None.
        load_raw_data (bool): Whether load pointcloud/image data. Defaults to False.
        target_labels (List[LabelType]): Target labels list.
        filtering_params (Dict[str, Any]): Filtering parameters.
        metrics_params (Dict[str, Any]): Metrics parameters.

    Args:
        dataset_paths (List[str]): Dataset paths list.
        frame_id (FrameID): FrameID instance, where objects are with respect.
        merge_similar_labels (bool): Whether merge similar labels.
            If True,
                - BUS, TRUCK, TRAILER -> CAR
                - MOTORBIKE, CYCLIST -> BICYCLE
        result_root_directory (str): Directory path to save result.
        evaluation_config_dict (Dict[str, Dict[str, Any]]): Dict that items are evaluation config for each task.
        load_raw_data (bool): Whether load pointcloud/image data. Defaults to False.
    """

    _support_tasks: List[str] = ["sensing"]

    def __init__(
        self,
        dataset_paths: List[str],
        frame_id: FrameID,
        merge_similar_labels: bool,
        result_root_directory: str,
        evaluation_config_dict: Dict[str, Dict[str, Any]],
        load_raw_data: bool = False,
    ):
        super().__init__(
            dataset_paths=dataset_paths,
            frame_id=frame_id,
            merge_similar_labels=merge_similar_labels,
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
            load_raw_data=load_raw_data,
        )
        self.filtering_params, self.metrics_params = self._extract_params(evaluation_config_dict)

    def _extract_params(
        self,
        evaluation_config_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract parameters.
        Args:
            evaluation_config_dict (Dict[str, Any]): Configuration as dict.
        Returns:
            f_params (Dict[str, Any]): Parameters for filtering.
            m_params (Dict[str, Any]): Parameters for metrics.
        """
        e_cfg: Dict[str, Any] = evaluation_config_dict.copy()
        f_params: Dict[str, Any] = {"target_uuids": e_cfg.get("target_uuids", None)}
        m_params: Dict[str, Any] = {
            "box_scale_0m": e_cfg.get("box_scale_0m", 1.0),
            "box_scale_100m": e_cfg.get("box_scale_100m", 1.0),
            "min_points_threshold": e_cfg.get("min_points_threshold", 1),
        }
        return f_params, m_params
