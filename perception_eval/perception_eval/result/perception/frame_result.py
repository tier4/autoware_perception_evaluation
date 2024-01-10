# Copyright 2022-2024 TIER IV, Inc.

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
from typing import TYPE_CHECKING

from perception_eval.common.status import GroundTruthStatus
from perception_eval.common.status import MatchingStatus
import perception_eval.matching.objects_filter as objects_filter
import perception_eval.metrics as metrics

from .pass_fail_result import PassFailResult

if TYPE_CHECKING:
    from perception_eval.common.label import LabelType
    from perception_eval.dataset import FrameGroundTruth
    from perception_eval.metrics import MetricsScoreConfig
    from perception_eval.object import ObjectType

    from .frame_config import PerceptionFrameConfig
    from .object_result import PerceptionObjectResult


class PerceptionFrameResult:
    """Frame level result for perception evaluation.

    Args:
    -----
        unix_time (int): Unix timestamp.
        frame_config (PerceptionFrameConfig): Frame level configuration.
        metrics_config (MetricsScoreConfig): Configuration for metrics score.
        object_results (List[PerceptionObjectResult]): List of object results.
        frame_ground_truth (FrameGroundTruth): Ground truth at one frame.
    """

    def __init__(
        self,
        unix_time: int,
        frame_config: PerceptionFrameConfig,
        metrics_config: MetricsScoreConfig,
        object_results: List[PerceptionObjectResult],
        frame_ground_truth: FrameGroundTruth,
    ) -> None:
        self.unix_time = unix_time
        self.frame_config = frame_config
        self.metrics_config = metrics_config

        self.object_results = object_results
        self.frame_ground_truth = frame_ground_truth

        # init evaluation
        self.metrics_score = metrics.MetricsScore(
            self.metrics_config,
            used_frame=[int(self.frame_number)],
        )
        self.pass_fail_result = PassFailResult(
            unix_time=self.unix_time,
            frame_number=self.frame_number,
            frame_config=self.frame_config,
            ego2map=self.frame_ground_truth.ego2map,
        )

    @property
    def target_labels(self) -> List[LabelType]:
        return self.frame_config.target_labels

    @property
    def frame_number(self) -> int:
        return self.frame_ground_truth.frame_number

    def evaluate_frame(
        self,
        critical_ground_truth_objects: List[ObjectType],
        previous_result: Optional[PerceptionFrameResult] = None,
    ) -> None:
        """
        Evaluate one frame.

        Internally, this method runs the following.
        1. Calculate metrics score if corresponding metrics config exists.
        2. Evaluate pass fail result.

        Args:
        -----
            critical_ground_truth_objects (List[ObjectType]): List of ground truth objects.
            previous_result (Optional[PerceptionFrameResult]): If the previous result exist
                it should be input to perform time series evaluation. Defaults to None.
        """
        # Divide objects by label to dict
        object_results_dict = objects_filter.divide_objects(self.object_results, self.target_labels)

        num_ground_truth_dict = objects_filter.divide_objects_to_num(
            self.frame_ground_truth.objects, self.target_labels
        )

        # If evaluation task is FP validation, only evaluate pass/fail result.
        if self.metrics_score.detection_config is not None:
            self.metrics_score.evaluate_detection(object_results_dict, num_ground_truth_dict)
        if self.metrics_score.tracking_config is not None:
            if previous_result is None:
                previous_results_dict = {label: [] for label in self.target_labels}
            else:
                previous_results_dict = objects_filter.divide_objects(
                    previous_result.object_results, self.target_labels
                )
            tracking_results: Dict[LabelType, List[PerceptionObjectResult]] = object_results_dict.copy()
            for label, prev_results in previous_results_dict.items():
                tracking_results[label] = [prev_results, tracking_results[label]]
            self.metrics_score.evaluate_tracking(tracking_results, num_ground_truth_dict)
        if self.metrics_score.prediction_config is not None:
            pass
        if self.metrics_score.classification_config is not None:
            self.metrics_score.evaluate_classification(object_results_dict, num_ground_truth_dict)

        self.pass_fail_result.evaluate(
            object_results=self.object_results,
            critical_ground_truth_objects=critical_ground_truth_objects,
        )


def get_object_status(frame_results: List[PerceptionFrameResult]) -> List[GroundTruthStatus]:
    """Returns the number of TP/FP/TN/FN ratios per frame as tuple.

    Args:
    -----
        frame_results (List[PerceptionFrameResult]): List of frame results.

    Returns:
    --------
        List[GroundTruthStatus]: List of matching status ratios for each GT.
    """
    status_infos: List[GroundTruthStatus] = []
    for frame_result in frame_results:
        frame_num = int(frame_result.frame_number)
        # TP
        for tp_object_result in frame_result.pass_fail_result.tp_object_results:
            if tp_object_result.ground_truth_object.uuid not in status_infos:
                tp_status = GroundTruthStatus(tp_object_result.ground_truth_object.uuid)
                tp_status.add_status(MatchingStatus.TP, frame_num)
                status_infos.append(tp_status)
            else:
                index = status_infos.index(tp_object_result.ground_truth_object.uuid)
                tp_status = status_infos[index]
                tp_status.add_status(MatchingStatus.TP, frame_num)
        # FP
        for fp_object_result in frame_result.pass_fail_result.fp_object_results:
            if fp_object_result.ground_truth_object is None:
                continue
            if fp_object_result.ground_truth_object.uuid not in status_infos:
                fp_status = GroundTruthStatus(fp_object_result.ground_truth_object.uuid)
                fp_status.add_status(MatchingStatus.FP, frame_num)
                status_infos.append(fp_status)
            else:
                index = status_infos.index(fp_object_result.ground_truth_object.uuid)
                fp_status = status_infos[index]
                fp_status.add_status(MatchingStatus.FP, frame_num)
        # TN
        for tn_object in frame_result.pass_fail_result.tn_objects:
            if tn_object.uuid not in status_infos:
                tn_status = GroundTruthStatus(tn_object.uuid)
                tn_status.add_status(MatchingStatus.TN, frame_num)
                status_infos.append(tn_status)
            else:
                index = status_infos.index(tn_object.uuid)
                tn_status = status_infos[index]
                tn_status.add_status(MatchingStatus.TN, frame_num)

        # FN
        for fn_object in frame_result.pass_fail_result.fn_objects:
            if fn_object.uuid not in status_infos:
                fn_status = GroundTruthStatus(fn_object.uuid)
                fn_status.add_status(MatchingStatus.FN, frame_num)
                status_infos.append(fn_status)
            else:
                index = status_infos.index(fn_object.uuid)
                fn_status = status_infos[index]
                fn_status.add_status(MatchingStatus.FN, frame_num)

    return status_infos
