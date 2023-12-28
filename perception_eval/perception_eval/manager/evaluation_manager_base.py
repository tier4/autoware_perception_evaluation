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
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from perception_eval.dataset import get_now_frame
from perception_eval.dataset import load_all_datasets

if TYPE_CHECKING:
    from perception_eval.config import EvaluationConfigType
    from perception_eval.config.params import FilterParamType
    from perception_eval.config.params import LabelParam
    from perception_eval.config.params import MetricsParamType
    from perception_eval.dataset import FrameGroundTruth
    from perception_eval.result import FrameResultType
    from perception_eval.visualization import VisualizerType


class EvaluationMangerBase(ABC):
    """Abstract base class for EvaluationManager.

    Attributes:
        evaluator_config (EvaluationConfigType): Configuration for specified evaluation task.
        ground_truth_frames (List[FrameGroundTruth]): List of ground truths per frame.

    Args:
        evaluation_config (EvaluationConfigType): Parameter config for EvaluationManager.
    """

    @abstractmethod
    def __init__(self, config: EvaluationConfigType) -> None:
        super().__init__()

        self.config = config
        self.ground_truth_frames = load_all_datasets(
            dataset_paths=self.config.dataset_paths,
            evaluation_task=self.config.evaluation_task,
            label_converter=self.config.label_converter,
            frame_id=self.config.frame_ids,
            load_raw_data=self.config.load_raw_data,
        )

        self.frame_results: List[FrameResultType] = []

    @property
    def evaluation_task(self):
        return self.config.evaluation_task

    @property
    def frame_ids(self):
        return self.config.frame_ids

    @property
    def label_param(self) -> LabelParam:
        return self.config.label_param

    @property
    def filter_param(self) -> FilterParamType:
        return self.config.filter_param

    @property
    def metrics_param(self) -> MetricsParamType:
        return self.config.metrics_param

    @property
    @abstractmethod
    def visualizer(self) -> VisualizerType:
        ...

    @abstractmethod
    def add_frame_result(self) -> FrameResultType:
        """Add perception/sensing frame result to `self.frame_results`

        Returns:
            FrameResultType: Frame result at current frame.
        """
        pass

    @abstractmethod
    def _filter_objects(self):
        """Filter objects with `self.filtering_params`"""
        pass

    def get_ground_truth_now_frame(
        self,
        unix_time: int,
        threshold_min_time: int = 75000,
    ) -> Optional[FrameGroundTruth]:
        """Returns a FrameGroundTruth instance that has the closest timestamp with `unix_time`.

        If there is no corresponding ground truth, returns None.

        Args:
            unix_time (int): Unix time of frame to evaluate.
            threshold_min_time (int, optional): Minimum timestamp threshold[s]. Defaults to 75000[s]=75[ms].

        Returns:
            Optional[FrameGroundTruth]: FrameGroundTruth instance at current frame.
                If there is no corresponding ground truth, returns None.
        """
        ground_truth_now_frame = get_now_frame(
            ground_truth_frames=self.ground_truth_frames,
            unix_time=unix_time,
            threshold_min_time=threshold_min_time,
        )
        return ground_truth_now_frame

    def visualize_all(self) -> None:
        """Visualize object result in BEV space for all frames."""
        self.visualizer.visualize_all(self.frame_results)

    def visualize_frame(self, frame_index: int = -1) -> None:
        """[summary]
        Visualize object result in BEV space at specified frame.

        Args:
            frame_index (int): The index of frame to be visualized. Defaults to -1 (latest frame).
        """
        self.visualizer.visualize_frame(self.frame_results[frame_index])
