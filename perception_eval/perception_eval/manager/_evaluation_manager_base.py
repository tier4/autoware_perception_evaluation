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

from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.dataset import get_now_frame
from perception_eval.common.dataset import load_all_datasets
from perception_eval.config import EvaluationConfigType
from perception_eval.evaluation import FrameResultType


class _EvaluationMangerBase(ABC):
    """Abstract base class for EvaluationManager.

    Attributes:
        evaluator_config (EvaluationConfigType): Configuration for specified evaluation task.
        ground_truth_frames (List[FrameGroundTruth]): List of ground truths per frame.

    Args:
        evaluation_config (EvaluationConfigType): Parameter config for EvaluationManager.
    """

    @abstractmethod
    def __init__(
        self,
        evaluation_config: EvaluationConfigType,
    ) -> None:
        super().__init__()

        self.evaluator_config = evaluation_config
        self.ground_truth_frames: List[FrameGroundTruth] = load_all_datasets(
            dataset_paths=self.evaluator_config.dataset_paths,
            evaluation_task=self.evaluator_config.evaluation_task,
            label_converter=self.evaluator_config.label_converter,
            frame_id=self.evaluator_config.frame_id,
            load_raw_data=self.evaluator_config.load_raw_data,
        )

    @property
    def evaluation_task(self):
        return self.evaluator_config.evaluation_task

    @property
    def frame_id(self):
        return self.evaluator_config.frame_id

    @property
    def filtering_params(self):
        return self.evaluator_config.filtering_params

    @property
    def metrics_params(self):
        return self.evaluator_config.metrics_params

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
        ground_truth_now_frame: FrameGroundTruth = get_now_frame(
            ground_truth_frames=self.ground_truth_frames,
            unix_time=unix_time,
            threshold_min_time=threshold_min_time,
        )
        return ground_truth_now_frame
