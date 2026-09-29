# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from perception_eval.evaluation.metrics.config.segmentation_metrics_config import SEGMENTATION_METRIC_KEYS
from perception_eval.evaluation.metrics.config.segmentation_metrics_config import SegmentationMetricsConfig

from ._evaluation_config_base import _EvaluationConfigBase


class SegmentationEvaluationConfig(_EvaluationConfigBase):
    """Configuration of the point-cloud semantic segmentation evaluation.

    Directory structure to save log and visualization result is following
    ```
    result_root_directory/
        ├── log_directory/
        └── visualization_directory/
    ```

    Attributes:
        dataset_paths (List[str]): Dataset paths list (may be empty when frames are supplied as arrays).
        frame_ids (List[FrameID]): Always ``[FrameID.BASE_LINK]`` for segmentation.
        result_root_directory (str): Directory path to save result.
        log_directory (str): Directory path to save log.
        visualization_directory (str): Directory path to save visualization result.
        evaluation_task (EvaluationTask): ``EvaluationTask.SEGMENTATION``.
        metrics_config (SegmentationMetricsConfig): Typed metrics configuration.
        filtering_params (Dict[str, Any]): ``{"target_uuids": ...}`` applied to ground-truth boxes.
        metrics_params (Dict[str, Any]): Raw metrics configuration dict.

    Args:
        dataset_paths (List[str]): Dataset paths list.
        frame_id (Union[str, Sequence[str]]): FrameID in string, ``"base_link"``.
        result_root_directory (str): Directory path to save result.
        evaluation_config_dict (Dict[str, Any]): ``{"evaluation_task": "segmentation", "class_names": [...], ...}``;
            see :class:`SegmentationMetricsConfig` for the accepted keys.
        load_raw_data (bool): Whether to load point clouds when datasets are loaded. Defaults to False.
    """

    _support_tasks: List[str] = ["segmentation"]

    def __init__(
        self,
        dataset_paths: List[str],
        frame_id: Union[str, Sequence[str]],
        result_root_directory: str,
        evaluation_config_dict: Dict[str, Any],
        load_raw_data: bool = False,
    ) -> None:
        super().__init__(
            dataset_paths=dataset_paths,
            frame_id=frame_id,
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict,
            load_raw_data=load_raw_data,
        )
        self.metrics_config: SegmentationMetricsConfig = SegmentationMetricsConfig.from_dict(self.metrics_params)

    @staticmethod
    def _extract_label_params(evaluation_config_dict: Dict[str, Any]) -> Dict[str, Any]:
        e_cfg: Dict[str, Any] = evaluation_config_dict.copy()
        return {
            "label_prefix": e_cfg.get("label_prefix", "autoware"),
            "merge_similar_labels": e_cfg.get("merge_similar_labels", False),
            "count_label_number": e_cfg.get("count_label_number", False),
        }

    def _extract_params(self, evaluation_config_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        e_cfg: Dict[str, Any] = evaluation_config_dict.copy()
        f_params: Dict[str, Any] = {"target_uuids": e_cfg.get("target_uuids", None)}
        m_params: Dict[str, Any] = {key: e_cfg[key] for key in SEGMENTATION_METRIC_KEYS if key in e_cfg}
        return f_params, m_params
