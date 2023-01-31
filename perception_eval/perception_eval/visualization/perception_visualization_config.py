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

from typing import List
from typing import Optional

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import LabelType
from perception_eval.common.status import FrameID


class PerceptionVisualizationConfig:
    """The config of to visualize perception results.

    Attributes:
        self.visualization_directory_path (str): The directory path to save visualized results.
        self.frame_id (FrameID): BASE_LINK or MAP.
        self.evaluation_task (EvaluationTask): Evaluation task.
        self.height (int): The height of image.
        self.width (int): The width of image.
        self.target_labels (Optional[List[LabelType]]): target_labels
        self.max_x_position_list (Optional[List[float]]): max_x_position_list
        self.max_y_position_list (Optional[List[float]]): max_y_position_list
        self.max_distance_list (Optional[List[float]]): max_distance_list
        self.min_distance_list (Optional[List[float]]): min_distance_list
        self.min_point_numbers (Optional[List[int]]): min_point_numbers
        self.target_uuids (Optional[List[str]]) target_uuids
        self.xlim (float): The limit of x range defined by max_x_position of max_distance.
            When both of them are None, set 100.0.
        self.ylim (float): The limit of y range defined by max_y_position of max_distance.
            When both of them are None, set 100.0.
    """

    def __init__(
        self,
        visualization_directory_path: str,
        frame_id: FrameID,
        evaluation_task: EvaluationTask,
        height: int = 640,
        width: int = 640,
        target_labels: Optional[List[LabelType]] = None,
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
            visualization_directory_path (str): Path to save visualized result.
            frame_id (FrameID): BASE_LINK or MAP.
            evaluation_task (EvaluationTask): Name of evaluation.
            height (int): Image height. Defaults to 640.
            width (int): Image width. Defaults to 640.
            target_labels (Optional[List[LabelType]]): The list of target label. Defaults to None.
            max_x_position_list (Optional[List[float]]): The list of max x position. Defaults to None.
            max_y_position_list (Optional[List[float]]): The list of max y position. Defaults to None.
            max_distance_list (Optional[List[float]]): The list of max distance. Defaults to None.
            min_distance_list (Optional[List[float]]): The list of min distance. Defaults to None.
            min_point_numbers (Optional[List[int]]): The list of min point numbers. Defaults to None.
            confidence_threshold_list (Optional[List[float]]): The list of confidence threshold. Defaults to None.
            target_uuids (Optional[List[str]]): The list of uuid. Defaults to None.
        """
        self.visualization_directory_path: str = visualization_directory_path
        self.frame_id: str = frame_id
        self.evaluation_task: EvaluationTask = evaluation_task
        self.height: int = height
        self.width: int = width

        self.target_labels: Optional[List[LabelType]] = target_labels
        self.max_x_position_list: Optional[List[float]] = max_x_position_list
        self.max_y_position_list: Optional[List[float]] = max_y_position_list
        self.max_distance_list: Optional[List[float]] = max_distance_list
        self.min_distance_list: Optional[List[float]] = min_distance_list
        self.min_point_numbers: Optional[List[int]] = min_point_numbers
        self.confidence_threshold_list: Optional[List[float]] = confidence_threshold_list
        self.target_uuids: Optional[List[str]] = target_uuids

        if max_x_position_list is None:
            self.xlim: float = max(max_distance_list)
            self.ylim: float = max(max_distance_list)
        elif max_distance_list is None:
            self.xlim: float = max(max_x_position_list)
            self.ylim: float = max(max_y_position_list)
        else:
            self.xlim: float = 100.0
            self.ylim: float = 100.0
