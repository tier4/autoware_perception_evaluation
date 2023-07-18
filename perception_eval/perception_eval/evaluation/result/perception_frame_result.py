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

from typing import Dict
from typing import List
from typing import Optional

from perception_eval.common import ObjectType
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import LabelType
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation import PerceptionFrameConfig
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.metrics import MetricsScoreConfig
from perception_eval.evaluation.result.perception_pass_fail_result import PassFailResult


class PerceptionFrameResult:
    """The result for 1 frame (the pair of estimated objects and ground truth objects)

    Attributes:
        object_results (List[DynamicObjectWithPerceptionResult]): Filtered object results to each estimated object.
        frame_ground_truth (FrameGroundTruth): Filtered ground truth of frame.
        frame_name (str): The file name of frame in the datasets.
        unix_time (int): The unix time for frame [us].
        target_labels (List[AutowareLabel]): The list of target label.
        metrics_score (MetricsScore): Metrics score results.
        pass_fail_result (PassFailResult): Pass fail results.

    Args:
        object_results (List[DynamicObjectWithPerceptionResult]): The list of object result.
        frame_ground_truth (FrameGroundTruth): FrameGroundTruth instance.
        metrics_config (MetricsScoreConfig): Metrics config class.
        frame_config (FrameConfig): Critical object filter config.
        unix_time (int): The unix time for frame [us]
        target_labels (List[AutowareLabel]): The list of target label.
    """

    def __init__(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        frame_ground_truth: FrameGroundTruth,
        metrics_config: MetricsScoreConfig,
        frame_config: PerceptionFrameConfig,
        unix_time: int,
        target_labels: List[LabelType],
    ):
        # frame information
        self.frame_name: str = frame_ground_truth.frame_name
        self.unix_time: int = unix_time
        self.target_labels: List[LabelType] = target_labels

        self.object_results: List[DynamicObjectWithPerceptionResult] = object_results
        self.frame_ground_truth: FrameGroundTruth = frame_ground_truth

        # init evaluation
        self.metrics_score: MetricsScore = MetricsScore(
            metrics_config,
            used_frame=[int(self.frame_name)],
        )
        self.pass_fail_result: PassFailResult = PassFailResult(
            frame_config=frame_config,
            ego2map=frame_ground_truth.ego2map,
        )

    def evaluate_frame(
        self,
        ros_critical_ground_truth_objects: List[ObjectType],
        previous_result: Optional[PerceptionFrameResult] = None,
    ) -> None:
        """[summary]
        Evaluate a frame from the pair of estimated objects and ground truth objects
        Args:
            ros_critical_ground_truth_objects (List[ObjectType]): The list of Ground truth objects filtered by ROS node.
            previous_result (Optional[PerceptionFrameResult]): The previous frame result. If None, set it as empty list []. Defaults to None.
        """
        # Divide objects by label to dict
        object_results_dict: Dict[
            LabelType, List[DynamicObjectWithPerceptionResult]
        ] = divide_objects(self.object_results, self.target_labels)

        num_ground_truth_dict: Dict[LabelType, int] = divide_objects_to_num(
            self.frame_ground_truth.objects, self.target_labels
        )

        if self.metrics_score.detection_config is not None:
            self.metrics_score.evaluate_detection(object_results_dict, num_ground_truth_dict)
        if self.metrics_score.tracking_config is not None:
            if previous_result is None:
                previous_results_dict = {label: [] for label in self.target_labels}
            else:
                previous_results_dict = divide_objects(
                    previous_result.object_results, self.target_labels
                )
            tracking_results: Dict[
                LabelType, List[DynamicObjectWithPerceptionResult]
            ] = object_results_dict.copy()
            for label, prev_results in previous_results_dict.items():
                tracking_results[label] = [prev_results, tracking_results[label]]
            self.metrics_score.evaluate_tracking(tracking_results, num_ground_truth_dict)
        if self.metrics_score.prediction_config is not None:
            pass
        if self.metrics_score.classification_config is not None:
            self.metrics_score.evaluate_classification(object_results_dict, num_ground_truth_dict)

        self.pass_fail_result.evaluate(
            object_results=self.object_results,
            ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
        )
