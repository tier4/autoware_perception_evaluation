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
from perception_eval.common.threshold import check_thresholds
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig


class CriticalObjectFilterConfig:
    """[summary]
    Config class for critical object filter

    Attributes:
        self.target_labels (List[str]): Target list
        self.max_x_position_list (Optional[List[float]]):
            Maximum x position threshold list for each label. Defaults to None.
        self.max_y_position_list (Optional[List[float]]):
            Maximum y position threshold list for each label. Defaults to None.
        self.max_distance_list (Optional[List[float]]]):
            Maximum distance threshold list for each label. Defaults to None.
        self.min_distance_list (Optional[List[float]]):
            Minimum distance threshold list for object. Defaults to None.
        self.min_point_numbers (Optional[List[int]]):
            Minimum number of points to be included in object's box. Defaults to None.
        self.confidence_threshold_list (Optional[List[float]]):
            The list of confidence threshold for each label. Defaults to None.
        self.target_uuids (Optional[List[str]]): The list of target uuid. Defaults to None.
        self.filtering_params: (Dict[str, Any]): The container of filtering parameters.
    """

    def __init__(
        self,
        evaluator_config: PerceptionEvaluationConfig,
        target_labels: List[str],
        max_x_position_list: Optional[List[float]] = None,
        max_y_position_list: Optional[List[float]] = None,
        max_distance_list: Optional[List[float]] = None,
        min_distance_list: Optional[List[float]] = None,
        min_point_numbers: Optional[List[int]] = None,
        confidence_threshold_list: Optional[List[float]] = None,
        target_uuids: Optional[List[str]] = None,
    ) -> None:
        """[summary]

        Args:
            evaluator_config (PerceptionEvaluationConfig): Evaluation config
            target_labels (List[str]): The list of target label.
            max_x_position_list (Optional[List[float]]):
                Maximum x position threshold list for each label. Defaults to None.
            max_y_position_list (Optional[List[float]]):
                Maximum y position threshold list for each label. Defaults to None.
            max_distance_list (Optional[List[float]]]):
                Maximum distance threshold list for each label. Defaults to None.
            min_distance_list (Optional[List[float]]):
                Minimum distance threshold list for object. Defaults to None.
            min_point_numbers (Optional[List[int]]):
                Minimum number of points to be included in object's box. Defaults to None.
            confidence_threshold_list (Optional[List[float]]):
                The list of confidence threshold for each label. Defaults to None.
            target_uuids (Optional[List[str]]): The list of target uuid. Defaults to None.
        """
        self.target_labels: List[AutowareLabel] = set_target_lists(
            target_labels,
            evaluator_config.label_converter,
        )
        if max_x_position_list and max_y_position_list:
            self.max_x_position_list: List[float] = check_thresholds(
                max_x_position_list,
                self.target_labels,
            )
            self.max_y_position_list: List[float] = check_thresholds(
                max_y_position_list,
                self.target_labels,
            )
            self.max_distance_list = None
            self.min_distance_list = None
        elif max_distance_list and min_distance_list:
            self.max_distance_list: List[float] = check_thresholds(
                max_distance_list,
                self.target_labels,
            )
            self.min_distance_list: List[float] = check_thresholds(
                min_distance_list,
                self.target_labels,
            )
            self.max_x_position_list = None
            self.max_y_position_list = None
        elif evaluator_config.evaluation_task.is_2d():
            self.max_x_position_list = None
            self.max_y_position_list = None
            self.max_distance_list = None
            self.min_distance_list = None
        else:
            raise RuntimeError("Either max x/y position or max/min distance should be specified")

        if min_point_numbers is None:
            self.min_point_numbers = None
        else:
            self.min_point_numbers: List[int] = check_thresholds(
                min_point_numbers,
                self.target_labels,
            )

        if confidence_threshold_list is None:
            self.confidence_threshold_list = None
        else:
            self.confidence_threshold_list: List[float] = check_thresholds(
                confidence_threshold_list,
                self.target_labels,
            )

        self.target_uuids: Optional[List[str]] = target_uuids

        self.filtering_params: Dict[str, Any] = {
            "target_labels": self.target_labels,
            "max_x_position_list": self.max_x_position_list,
            "max_y_position_list": self.max_y_position_list,
            "max_distance_list": self.max_distance_list,
            "min_distance_list": self.min_distance_list,
            "min_point_numbers": self.min_point_numbers,
            "confidence_threshold_list": self.confidence_threshold_list,
            "target_uuids": self.target_uuids,
        }


class PerceptionPassFailConfig:
    """[summary]
    Config filter for pass fail to frame result

    Attributes:
        self.evaluation_task (EvaluationTask): Evaluation task.
        self.target_labels (List[str]): The list of target label.
        self.matching_distance_list (List[float]): The threshold list for Pass/Fail.
            For 2D evaluation, IOU2D, for 3D evaluation, PLANEDISTANCE will be used.
        self.confidence_threshold_list (Optional[List[float]]): The list of confidence threshold.
    """

    def __init__(
        self,
        evaluator_config: PerceptionEvaluationConfig,
        target_labels: List[str],
        matching_threshold_list: List[float],
        confidence_threshold_list: Optional[List[float]] = None,
    ) -> None:
        """[summary]
        Args:
            evaluator_config (PerceptionEvaluationConfig): Evaluation config
            target_labels (List[str]): Target list
            matching_threshold_list (List[float]): The threshold list for Pass/Fail.
                For 2D evaluation, IOU2D, for 3D evaluation, PLANEDISTANCE will be used.
            confidence_threshold_list (Optional[List[float]]): The list of confidence threshold.
                Defaults to None.
        """
        self.evaluation_task: EvaluationTask = evaluator_config.evaluation_task
        self.target_labels: List[AutowareLabel] = set_target_lists(
            target_labels,
            evaluator_config.label_converter,
        )
        self.matching_threshold_list: List[float] = check_thresholds(
            matching_threshold_list,
            self.target_labels,
        )
        if confidence_threshold_list is None:
            self.confidence_threshold_list = None
        else:
            self.confidence_threshold_list: List[float] = check_thresholds(
                confidence_threshold_list,
                self.target_labels,
            )


class UseCaseThresholdsError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)
