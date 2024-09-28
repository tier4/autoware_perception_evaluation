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
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import LabelType
from perception_eval.common.label import set_target_lists
from perception_eval.common.threshold import set_thresholds
from perception_eval.evaluation.matching import MatchingLabelPolicy
from perception_eval.evaluation.metrics import MetricsScoreConfig

from ._evaluation_config_base import _EvaluationConfigBase


class PerceptionEvaluationConfig(_EvaluationConfigBase):
    """Configuration class for perception evaluation.

    Directory structure to save log and visualization result is following
    - result_root_directory/
        ├── log_directory/
        └── visualization_directory/

    Attributes:
        dataset_paths (List[str]): Dataset paths list.
        frame_ids (List[FrameID]): List of FrameID instance, where objects are with respect.
        result_root_directory (str): Directory path to save result.
        log_directory (str): Directory Directory path to save log.
        visualization_directory (str): Directory path to save visualization result.
        label_converter (LabelConverter): LabelConverter instance.
        evaluation_task (EvaluationTask): EvaluationTask instance.
        label_prefix (str): Prefix of label type. Choose from [`autoware", `traffic_light`]. Defaults to autoware.
        load_raw_data (bool): Whether load pointcloud/image data. Defaults to False.
        target_labels (List[LabelType]): Target labels list.
        filtering_params (Dict[str, Any]): Filtering parameters.
        metrics_params (Dict[str, Any]): Metrics parameters.
        metrics_config (MetricsScoreConfig): MetricsScoreConfig instance.

    Args:
        dataset_paths (List[str]): Dataset paths list.
        frame_id (Union[str, Sequence[str]]): FrameID(s) in string, where objects are with respect.
        result_root_directory (str): Directory path to save result.
        evaluation_config_dict (Dict[str, Dict[str, Any]]): Dict that items are evaluation config for each task.
        load_raw_data (bool): Whether load pointcloud/image data. Defaults to False.
    """

    _support_tasks: List[str] = [
        "detection2d",
        "tracking2d",
        "classification2d",
        "fp_validation2d",
        "detection",
        "tracking",
        "prediction",
        "fp_validation",
    ]

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

        self.metrics_config: MetricsScoreConfig = MetricsScoreConfig(
            self.evaluation_task,
            **self.metrics_params,
        )

    @staticmethod
    def _extract_label_params(evaluation_config_dict: Dict[str, Any]) -> Dict[str, Any]:
        e_cfg = evaluation_config_dict.copy()
        if e_cfg.get("matching_label_policy"):
            matching_label_policy = MatchingLabelPolicy.from_str(e_cfg.get("matching_label_policy"))
        else:
            allow_matching_unknown = e_cfg.get("allow_matching_unknown", False)
            matching_label_policy = (
                MatchingLabelPolicy.ALLOW_UNKNOWN if allow_matching_unknown else MatchingLabelPolicy.DEFAULT
            )

        l_params: Dict[str, Any] = {
            "label_prefix": e_cfg["label_prefix"],
            "merge_similar_labels": e_cfg.get("merge_similar_labels", False),
            "matching_label_policy": matching_label_policy,
            "count_label_number": e_cfg.get("count_label_number", True),
        }
        return l_params

    def _extract_params(
        self,
        evaluation_config_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract and divide parameters from evaluation_config_dict into filtering and metrics parameters.

        Args:
            evaluation_config_dict (Dict[str, Any]): Dict that items are evaluation config for each task.

        Returns:
            f_params (Dict[str, Any]): Parameters for filtering.
            m_params (Dict[str, Any]): Parameters for metrics.
            l_params (Dict[str, Any]): Parameters for label.
        """
        e_cfg = evaluation_config_dict.copy()

        # Covert labels to autoware labels for Metrics
        target_labels: List[LabelType] = set_target_lists(
            e_cfg.get("target_labels"),
            self.label_converter,
        )
        self.target_labels: List[LabelType] = target_labels

        max_x_position: Optional[float] = e_cfg.get("max_x_position")
        max_y_position: Optional[float] = e_cfg.get("max_y_position")
        max_distance: Optional[float] = e_cfg.get("max_distance")
        min_distance: Optional[float] = e_cfg.get("min_distance")

        num_elements: int = len(target_labels)
        if None not in (max_x_position, max_y_position):
            max_x_position_list: List[float] = set_thresholds(max_x_position, num_elements, False)
            max_y_position_list: List[float] = set_thresholds(max_y_position, num_elements, False)
            max_distance_list = None
            min_distance_list = None
        elif None not in (max_distance, min_distance):
            max_distance_list: List[float] = set_thresholds(max_distance, num_elements, False)
            min_distance_list: List[float] = [min_distance] * len(target_labels)
            max_x_position_list = None
            max_y_position_list = None
        elif self.evaluation_task.is_2d():
            max_x_position_list = None
            max_y_position_list = None
            max_distance_list = None
            min_distance_list = None
        else:
            raise RuntimeError("Either max x/y position or max/min distance should be specified")

        max_matchable_radii: Optional[Union[float, List[float]]] = e_cfg.get("max_matchable_radii")
        if max_matchable_radii is not None:
            max_matchable_radii: List[float] = set_thresholds(max_matchable_radii, num_elements, False)

        min_point_numbers: Optional[List[int]] = e_cfg.get("min_point_numbers")
        if min_point_numbers is not None:
            min_point_numbers: List[int] = set_thresholds(min_point_numbers, num_elements, False)

        if self.evaluation_task == EvaluationTask.DETECTION and min_point_numbers is None:
            raise RuntimeError("In detection task, min point numbers must be specified")

        conf_thresh: Optional[float] = e_cfg.get("confidence_threshold")
        if conf_thresh is not None:
            confidence_threshold_list: List[float] = set_thresholds(conf_thresh, num_elements, False)
        else:
            confidence_threshold_list = None

        target_uuids: Optional[List[str]] = e_cfg.get("target_uuids")
        ignore_attributes: Optional[List[str]] = e_cfg.get("ignore_attributes")

        f_params: Dict[str, Any] = {
            "target_labels": target_labels,
            "ignore_attributes": ignore_attributes,
            "max_x_position_list": max_x_position_list,
            "max_y_position_list": max_y_position_list,
            "max_distance_list": max_distance_list,
            "min_distance_list": min_distance_list,
            "max_matchable_radii": max_matchable_radii,
            "min_point_numbers": min_point_numbers,
            "confidence_threshold_list": confidence_threshold_list,
            "target_uuids": target_uuids,
            "first_uuid_matching": e_cfg.get("first_uuid_matching", False),
        }

        m_params: Dict[str, Any] = {
            "target_labels": target_labels,
            "center_distance_thresholds": e_cfg.get("center_distance_thresholds"),
            "plane_distance_thresholds": e_cfg.get("plane_distance_thresholds"),
            "iou_2d_thresholds": e_cfg.get("iou_2d_thresholds"),
            "iou_3d_thresholds": e_cfg.get("iou_3d_thresholds"),
        }

        return f_params, m_params
