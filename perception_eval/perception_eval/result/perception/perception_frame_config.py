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

from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from perception_eval.common.label import set_target_lists
from perception_eval.common.threshold import check_thresholds
from perception_eval.config.params import PerceptionFilterParam

if TYPE_CHECKING:
    from perception_eval.common.label import LabelType
    from perception_eval.config import PerceptionEvaluationConfig


class PerceptionFrameConfig:
    """
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
        target_labels: List[Union[str, LabelType]],
        max_x_position_list: Optional[List[float]] = None,
        max_y_position_list: Optional[List[float]] = None,
        min_distance_list: Optional[List[float]] = None,
        max_distance_list: Optional[List[float]] = None,
        min_point_numbers: Optional[List[int]] = None,
        confidence_threshold_list: Optional[List[float]] = None,
        target_uuids: Optional[List[str]] = None,
        ignore_attributes: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None,
    ) -> None:
        """
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
        self.evaluation_task = evaluator_config.evaluation_task
        if all([isinstance(label, str) for label in target_labels]):
            self.target_labels = set_target_lists(target_labels, evaluator_config.label_converter)
        else:
            self.target_labels = target_labels

        self.filter_param = PerceptionFilterParam(
            evaluation_task=self.evaluation_task,
            target_labels=self.target_labels,
            max_x_position_list=max_x_position_list,
            max_y_position_list=max_y_position_list,
            min_distance_list=min_distance_list,
            max_distance_list=max_distance_list,
            min_point_numbers=min_point_numbers,
            confidence_threshold_list=confidence_threshold_list,
            target_uuids=target_uuids,
            ignore_attributes=ignore_attributes,
        )

        num_elements: int = len(self.target_labels)
        if thresholds is None:
            self.thresholds = None
        else:
            self.thresholds = check_thresholds(thresholds, num_elements)

    @classmethod
    def from_eval_cfg(cls, eval_cfg: PerceptionEvaluationConfig) -> PerceptionFrameConfig:
        return cls(
            evaluator_config=eval_cfg,
            target_labels=eval_cfg.target_labels,
            max_x_position_list=eval_cfg.filter_param.max_x_position_list,
            max_y_position_list=eval_cfg.filter_param.max_y_position_list,
            min_distance_list=eval_cfg.filter_param.min_distance_list,
            max_distance_list=eval_cfg.filter_param.max_distance_list,
            min_point_numbers=eval_cfg.filter_param.min_point_numbers,
            confidence_threshold_list=eval_cfg.filter_param.confidence_threshold_list,
            target_uuids=eval_cfg.filter_param.target_uuids,
            ignore_attributes=eval_cfg.filter_param.ignore_attributes,
            thresholds=eval_cfg.metrics_param.plane_distance_thresholds,
        )
