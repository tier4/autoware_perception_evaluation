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

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import set_target_lists
from perception_eval.common.threshold import ThresholdsError
from perception_eval.common.threshold import check_thresholds
from perception_eval.evaluation.metrics.metrics_score_config import MetricsScoreConfig

from ._evaluation_config_base import _EvaluationConfigBase


class PerceptionEvaluationConfig(_EvaluationConfigBase):
    """[summary]
    Evaluation configure class

    Attributes:
        - By _EvaluationConfigBase
        self.dataset_paths (List[str]): The list of dataset path.
        self.frame_id (str): The frame_id, base_link or map.
        self.does_use_pointcloud (bool): The boolean flag if load pointcloud data from dataset.
        self.result_root_directory (str): The path to result directory
        self.log_directory (str): The path to sub directory for log
        self.visualization_directory (str): The path to sub directory for visualization
        self.label_converter (LabelConverter): The label convert class
        self.evaluation_task (EvaluationTask): The instance of EvaluationTask

        - By PerceptionEvaluationConfig
        self.target_labels (List[AutowareLabel]): The list of target label.
        self.filtering_params (Dict[str, Any]): Filtering parameters.
        self.metrics_params (Dict[str, Any]): Metrics parameters.
        self.metrics_config (MetricsScoreConfig): The config for metrics
    """

    _support_tasks: List[str] = ["detection", "tracking", "prediction"]

    def __init__(
        self,
        dataset_paths: List[str],
        frame_id: str,
        merge_similar_labels: bool,
        does_use_pointcloud: bool,
        result_root_directory: str,
        evaluation_config_dict: Dict[str, Any],
    ) -> None:
        """[summary]

        Args:
            dataset_paths (List[str]): The paths of dataset
            frame_id (str): Frame ID, base_link or map
            merge_similar_labels (bool): Whether merge similar labels.
                If True,
                    - BUS, TRUCK, TRAILER -> CAR
                    - MOTORBIKE, CYCLIST -> BICYCLE
            does_use_pointcloud (bool): The flag for loading pointcloud data from dataset
            result_root_directory (str): The path to result directory
            evaluation_config_dict (Dict[str, Dict[str, Any]]): The dictionary of evaluation config for each task.
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

        self.metrics_config: MetricsScoreConfig = MetricsScoreConfig(
            self.evaluation_task,
            **self.metrics_params,
        )

    def _extract_params(self, evaluation_config_dict: Dict[str, Any]) -> None:
        """[summary]
        Extract and divide parameters from evaluation_config_dict.

        Args:
            evaluation_config_dict (Dict[str, Any])

        Returns:
            f_params (Dict[str, Any]): Parameters for filtering objects.
            m_params (Dict[str, Any]): Parameters for metrics config.
        """
        e_cfg = evaluation_config_dict.copy()

        # Covert labels to autoware labels for Metrics
        target_labels: List[AutowareLabel] = set_target_lists(
            e_cfg["target_labels"],
            self.label_converter,
        )
        self.target_labels: List[AutowareLabel] = target_labels

        max_x_position: Optional[float] = e_cfg.get("max_x_position")
        max_y_position: Optional[float] = e_cfg.get("max_y_position")
        max_distance: Optional[float] = e_cfg.get("max_distance")
        min_distance: Optional[float] = e_cfg.get("min_distance")

        if max_x_position and max_y_position:
            max_x_position_list: List[float] = [max_x_position] * len(target_labels)
            max_y_position_list: List[float] = [max_y_position] * len(target_labels)

            max_x_position_list = check_thresholds(
                max_x_position_list,
                target_labels,
                ThresholdsError,
            )
            max_y_position_list: List[float] = check_thresholds(
                max_y_position_list,
                target_labels,
                ThresholdsError,
            )
            max_distance_list = None
            min_distance_list = None
        elif max_distance and min_distance:
            max_distance_list: List[float] = [max_distance] * len(target_labels)
            min_distance_list: List[float] = [min_distance] * len(target_labels)

            max_distance_list = check_thresholds(
                max_distance_list,
                target_labels,
                ThresholdsError,
            )
            min_distance_list = check_thresholds(
                min_distance_list,
                target_labels,
                ThresholdsError,
            )
            max_x_position_list = None
            max_y_position_list = None
        else:
            raise RuntimeError("Either max x/y position or max/min distance should be specified")

        min_point_numbers: Optional[List[int]] = e_cfg.get("min_point_numbers")
        if min_point_numbers is not None:
            min_point_numbers: List[int] = check_thresholds(
                min_point_numbers,
                target_labels,
                ThresholdsError,
            )

        if self.evaluation_task == EvaluationTask.DETECTION and min_point_numbers is None:
            raise RuntimeError("In detection task, min point numbers must be specified")

        conf_thresh: Optional[float] = e_cfg.get("confidence_threshold")
        if conf_thresh is not None:
            confidence_threshold_list: List[float] = [conf_thresh] * len(target_labels)
            confidence_threshold_list = check_thresholds(confidence_threshold_list, target_labels)
        else:
            confidence_threshold_list = None

        target_uuids: Optional[List[str]] = e_cfg.get("target_uuids")

        f_params: Dict[str, Any] = {
            "target_labels": target_labels,
            "max_x_position_list": max_x_position_list,
            "max_y_position_list": max_y_position_list,
            "max_distance_list": max_distance_list,
            "min_distance_list": min_distance_list,
            "min_point_numbers": min_point_numbers,
            "confidence_threshold_list": confidence_threshold_list,
            "target_uuids": target_uuids,
        }

        m_params: Dict[str, Any] = {
            "target_labels": target_labels,
            "center_distance_thresholds": e_cfg["center_distance_thresholds"],
            "plane_distance_thresholds": e_cfg["plane_distance_thresholds"],
            "iou_bev_thresholds": e_cfg["iou_bev_thresholds"],
            "iou_3d_thresholds": e_cfg["iou_3d_thresholds"],
        }

        return f_params, m_params
