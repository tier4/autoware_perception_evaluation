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
from perception_eval.common.schema import GroundTruthStatus
from perception_eval.common.schema import MatchingStatus
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.metrics import MetricsScoreConfig
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
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
        critical_object_filter_config (CriticalObjectFilterConfig): Critical object filter config.
        frame_pass_fail_config (PerceptionPassFailConfig): Frame pass fail config.
        unix_time (int): The unix time for frame [us]
        target_labels (List[AutowareLabel]): The list of target label.
    """

    def __init__(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        frame_ground_truth: FrameGroundTruth,
        metrics_config: MetricsScoreConfig,
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
        unix_time: int,
        target_labels: List[LabelType],
    ) -> None:
        # TODO(ktro2828): rename `frame_name` into `frame_number`
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
            unix_time=unix_time,
            frame_number=frame_ground_truth.frame_name,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
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

        # If evaluation task is FP validation, only evaluate pass/fail result.
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


def get_object_status(frame_results: List[PerceptionFrameResult]) -> List[GroundTruthStatus]:
    """Returns the number of TP/FP/TN/FN ratios per frame as tuple.

    Args:
        frame_results (List[PerceptionFrameResult]): List of frame results.

    Returns:
        List[GroundTruthStatus]: Sequence of matching status ratios for each GT.
    """
    status_infos: List[GroundTruthStatus] = []
    for frame_result in frame_results:
        frame_num: int = int(frame_result.frame_name)
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
                fp_status = GroundTruthStatus(tp_object_result.ground_truth_object.uuid)
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
