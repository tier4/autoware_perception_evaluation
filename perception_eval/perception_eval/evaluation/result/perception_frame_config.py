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

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import LabelType
from perception_eval.common.label import set_target_lists
from perception_eval.common.threshold import check_thresholds
from perception_eval.evaluation.matching import MatchingMode

# from perception_eval.config import PerceptionEvaluationConfig


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
        evaluator_config,  #: PerceptionEvaluationConfig,
        target_labels: List[str],
        ignore_attributes: Optional[List[str]] = None,
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
        self.target_labels: List[LabelType] = set_target_lists(
            target_labels,
            evaluator_config.label_converter,
        )
        self.ignore_attributes: Optional[List[str]] = ignore_attributes

        num_elements: int = len(self.target_labels)
        if max_x_position_list and max_y_position_list:
            self.max_x_position_list: List[float] = check_thresholds(max_x_position_list, num_elements)
            self.max_y_position_list: List[float] = check_thresholds(max_y_position_list, num_elements)
            self.max_distance_list = None
            self.min_distance_list = None
        elif max_distance_list and min_distance_list:
            self.max_distance_list: List[float] = check_thresholds(max_distance_list, num_elements)
            self.min_distance_list: List[float] = check_thresholds(min_distance_list, num_elements)
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
            self.min_point_numbers: List[int] = check_thresholds(min_point_numbers, num_elements)

        if confidence_threshold_list is None:
            self.confidence_threshold_list = None
        else:
            self.confidence_threshold_list: List[float] = check_thresholds(confidence_threshold_list, num_elements)

        self.target_uuids: Optional[List[str]] = target_uuids

        self.filtering_params: Dict[str, Any] = {
            "target_labels": self.target_labels,
            "ignore_attributes": self.ignore_attributes,
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
        self.matching_distance_list (Optional[List[float]]): The threshold list for Pass/Fail.
            For 2D evaluation, IOU2D, for 3D evaluation, PLANEDISTANCE will be used.
        self.confidence_threshold_list (Optional[List[float]]): The list of confidence threshold.
    """

    def __init__(
        self,
        evaluator_config,  #: PerceptionEvaluationConfig,
        target_labels: Optional[List[str]],
        matching_mode_list: Optional[List[MatchingMode]] = None,
        matching_threshold_list_for_labels: Union[List[List[float]], List[float], None] = None,
        confidence_threshold_list: Optional[List[float]] = None,
    ) -> None:
        """[summary]
        Args:
            evaluator_config (PerceptionEvaluationConfig): Evaluation config
            target_labels (List[str]): Target list. If None or empty list is specified, all labels will be evaluated.
            matching_mode_list (Optional[List[MatchingMode]]): The list of matching mode.
            matching_threshold_list_for_labels (Union[List[List[float]], List[float], None]):
                The list of matching threshold for each matching mode. Defaults to None.
                The shape of matching_threshold_list_for_labels is
                [
                    [matching_threshold_list_for_matching_mode_1],
                    [matching_threshold_list_for_matching_mode_2],
                    ...
                ]
                where matching_threshold_list_for_mode_i is the list of matching threshold for each label.
            # MEMO: 後で消す。matching_threshold_list_for_labelsが1次元で、かつmatching_mode_listがNoneの場合（つまり現状の殆どのscenario.yaml）は、
            # 必ずmatching_modeが1つになる（2D IOUまたは平面距離）ので、matching_threshold_list_for_labelsをそのまま2次元にしている。が、ここはなくしても良いかも。
            confidence_threshold_list (Optional[List[float]]): The list of confidence threshold. Defaults to None.
        """
        self.evaluation_task: EvaluationTask = evaluator_config.evaluation_task
        self.target_labels: List[LabelType] = set_target_lists(
            target_labels,
            evaluator_config.label_converter,
        )

        num_elements: int = len(self.target_labels)
        if matching_mode_list is None:
            self.matching_mode_list: List[MatchingMode] = [
                MatchingMode.IOU2D if self.evaluation_task.is_2d() else MatchingMode.PLANEDISTANCE
            ]
        else:
            self.matching_mode_list: List[MatchingMode] = matching_mode_list
        if matching_threshold_list_for_labels is None:
            self.matching_threshold_list_for_labels = None
        elif all(isinstance(elem, float) for elem in matching_threshold_list_for_labels):
            self.matching_threshold_list_for_labels: List[List[float]] = [
                check_thresholds(matching_threshold_list_for_labels, num_elements)
            ]
        else:
            assert len(matching_threshold_list_for_labels) == len(self.matching_mode_list), (
                f"Length of matching_threshold_list_for_labels ({len(matching_threshold_list_for_labels)}) "
                f"must be equal to length of matching_mode_list ({len(self.matching_mode_list)})"
            )
            self.matching_threshold_list_for_labels: List[List[float]] = List(
                check_thresholds(matching_threshold_list, num_elements)
                for matching_threshold_list in matching_threshold_list_for_labels
            )
        if confidence_threshold_list is None:
            self.confidence_threshold_list = None
        else:
            self.confidence_threshold_list: List[float] = check_thresholds(confidence_threshold_list, num_elements)


class UseCaseThresholdsError(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)
