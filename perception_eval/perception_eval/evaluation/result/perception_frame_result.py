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
from typing import Tuple

from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import LabelType
from perception_eval.common.label import TrafficLightLabel
from perception_eval.common.status import GroundTruthStatus
from perception_eval.common.status import MatchingStatus
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_nuscene_object_results
from perception_eval.evaluation.matching.objects_filter import filter_object_results
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.metrics import MetricsScoreConfig
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.evaluation.result.perception_pass_fail_nuscene_result import PassFailNusceneResult
from perception_eval.evaluation.result.perception_pass_fail_result import PassFailResult


class PerceptionFrameResult:
    """The result for 1 frame (the pair of estimated objects and ground truth objects)

    Attributes:
        object_results (List[DynamicObjectWithPerceptionResult]): Filtered object results to each estimated object.
        nuscene_object_results (Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]):
            Object results grouped by matching mode, target label, and threshold.
        frame_ground_truth (FrameGroundTruth): Filtered ground truth of frame.
        frame_name (str): The file name of frame in the datasets.
        unix_time (int): The unix time for frame [us].
        target_labels (List[AutowareLabel]): The list of target label.
        metrics_score (MetricsScore): Metrics score results.
        pass_fail_result (PassFailResult): Pass fail results.

    Args:
        object_results (List[DynamicObjectWithPerceptionResult]): The list of object result.
        nuscene_object_results (Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]):
            A dictionary storing object results grouped by matching mode, target label, and threshold.
        frame_ground_truth (FrameGroundTruth): FrameGroundTruth instance.
        metrics_config (MetricsScoreConfig): Metrics config class.
        critical_object_filter_config (CriticalObjectFilterConfig): Critical object filter config.
        frame_pass_fail_config (PerceptionPassFailConfig): Frame pass fail config.
        unix_time (int): The unix time for frame [us]
        target_labels (List[AutowareLabel]): The list of target label.
    """

    def __init__(
        self,
        object_results: Optional[List[DynamicObjectWithPerceptionResult]],
        nuscene_object_results: Optional[
            Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]
        ],
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

        self.object_results: Optional[List[DynamicObjectWithPerceptionResult]] = object_results
        self.nuscene_object_results: Optional[
            Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]
        ] = nuscene_object_results
        self.frame_ground_truth: FrameGroundTruth = frame_ground_truth

        # init evaluation
        self.metrics_config = metrics_config
        self.metrics_score: MetricsScore = MetricsScore(
            metrics_config,
            used_frame=[int(self.frame_name)],
        )
        self.critical_object_filter_config = critical_object_filter_config
        self.frame_pass_fail_config = frame_pass_fail_config

        # For tracking and detection, we use PassFailNusceneResult
        if self.metrics_score.detection_config or self.metrics_score.tracking_config:
            self.pass_fail_result: PassFailNusceneResult = PassFailNusceneResult(
                unix_time=unix_time,
                frame_number=frame_ground_truth.frame_name,
                critical_object_filter_config=critical_object_filter_config,
                frame_pass_fail_config=frame_pass_fail_config,
                transforms=frame_ground_truth.transforms,
            )
        else:
            self.pass_fail_result: PassFailResult = PassFailResult(
                unix_time=unix_time,
                frame_number=frame_ground_truth.frame_name,
                critical_object_filter_config=critical_object_filter_config,
                frame_pass_fail_config=frame_pass_fail_config,
                transforms=frame_ground_truth.transforms,
            )

    def evaluate_perception_frame(
        self,
        previous_result: Optional[PerceptionFrameResult] = None,
    ) -> None:
        """
        Evaluate a frame from the pair of estimated objects and ground truth objects
        Args:
            previous_result (Optional[PerceptionFrameResult]): The previous frame result. If None, set it as empty list []. Defaults to None.
        """
        # TODO(KS): This should be done in preprocessing step
        num_ground_truth_dict: Dict[LabelType, int] = divide_objects_to_num(
            self.frame_ground_truth.objects,
            self.critical_object_filter_config.target_labels,
        )
        self._core_evaluation(
            num_ground_truth_dict=num_ground_truth_dict,
            previous_object_results=previous_result.object_results if previous_result else None,
        )

    def evaluate_frame(
        self,
        previous_result: Optional[PerceptionFrameResult] = None,
    ) -> None:
        """[summary]
        Evaluate a frame from the pair of estimated objects and ground truth objects
        Args:
            previous_result (Optional[PerceptionFrameResult]): The previous frame result. If None, set it as empty list []. Defaults to None.
        """
        self.frame_ground_truth.objects = filter_objects(
            self.frame_ground_truth.objects,
            is_gt=True,
            transforms=self.frame_ground_truth.transforms,
            **self.critical_object_filter_config.filtering_params,
        )

        num_ground_truth_dict: Dict[LabelType, int] = divide_objects_to_num(
            self.frame_ground_truth.objects,
            self.critical_object_filter_config.target_labels,
        )

        if self.nuscene_object_results is not None:
            # Filter objects by critical object filter config
            self.nuscene_object_results = filter_nuscene_object_results(
                self.nuscene_object_results,
                transforms=self.frame_ground_truth.transforms,
                **self.critical_object_filter_config.filtering_params,
            )

        if self.object_results is not None:
            # Filter objects by critical object filter config
            self.object_results: List[DynamicObjectWithPerceptionResult] = filter_object_results(
                self.object_results,
                transforms=self.frame_ground_truth.transforms,
                **self.critical_object_filter_config.filtering_params,
            )

        self._core_evaluation(
            num_ground_truth_dict=num_ground_truth_dict,
            previous_object_results=previous_result.object_results if previous_result else None,
        )

    def _core_evaluation(
        self,
        num_ground_truth_dict: Dict[LabelType, int],
        previous_object_results: Optional[List[DynamicObjectWithPerceptionResult]],
    ) -> None:
        """Core evaluation logic for the frame result."""
        # Only detection
        if self.metrics_score.detection_config is not None and self.metrics_score.tracking_config is None:
            self.metrics_score.evaluate_detection(self.nuscene_object_results, num_ground_truth_dict)
            # Calculate TP/FP/TN/FN based on plane distance
            self.pass_fail_result.evaluate(self.nuscene_object_results, self.frame_ground_truth.objects)

        assert self.object_results is not None, "Object results must be provided for evaluation in the following tasks."

        # TODO(KS): This should be done in preprocessing step
        # Group objects results based on the Label
        object_results_dict: Dict[LabelType, List[DynamicObjectWithPerceptionResult]] = divide_objects(
            self.object_results,
            self.critical_object_filter_config.target_labels,
        )

        # Detection and Tracking
        if self.metrics_score.detection_config is not None and self.metrics_score.tracking_config is not None:
            # Use object_results for tracking and nuscene_object_results for detection/pass_fail
            if previous_object_results is None:
                previous_results_dict = {label: [] for label in self.critical_object_filter_config.target_labels}
            else:
                previous_results_dict = divide_objects(
                    previous_object_results,
                    self.critical_object_filter_config.target_labels,
                )

            tracking_results: Dict[LabelType, List[DynamicObjectWithPerceptionResult]] = object_results_dict.copy()
            for label, prev_results in previous_results_dict.items():
                tracking_results[label] = [prev_results, tracking_results[label]]
            self.metrics_score.evaluate_detection(self.nuscene_object_results, num_ground_truth_dict)
            self.metrics_score.evaluate_tracking(tracking_results, num_ground_truth_dict)
            # Calculate TP/FP/TN/FN based on plane distance
            self.pass_fail_result.evaluate(self.nuscene_object_results, self.frame_ground_truth.objects)

        # Classification
        elif self.metrics_score.classification_config is not None:
            self.metrics_score.evaluate_classification(object_results_dict, num_ground_truth_dict)
            self.pass_fail_result.evaluate(self.object_results, self.frame_ground_truth.objects)

        # Prediction
        elif self.metrics_score.prediction_config is not None:
            self.metrics_score.evaluate_prediction(object_results_dict, num_ground_truth_dict)
            self.pass_fail_result.evaluate(self.object_results, self.frame_ground_truth.objects)
        else:
            raise ValueError(
                "No matched metrics config. "
                "Please ensure that at least one of detection, tracking, classification, or prediction configs is set."
            )

    def __reduce__(self) -> Tuple[PerceptionFrameResult, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling."""
        init_args = (
            self.object_results,
            self.nuscene_object_results,
            self.frame_ground_truth,
            self.metrics_config,
            self.critical_object_filter_config,
            self.frame_pass_fail_config,
            self.unix_time,
            self.target_labels,
        )
        state = {"pass_fail_result": self.pass_fail_result}
        return (self.__class__, init_args, state)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the object after deserialization."""
        self.pass_fail_result = state.get("pass_fail_result", None)

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        return {
            "object_results": [object_result.serialization() for object_result in self.object_results],
            "nuscene_object_results": [
                nuscene_object_result.serialization() for nuscene_object_result in (self.nuscene_object_results or [])
            ],
            "frame_ground_truth": self.frame_ground_truth.serialization(),
            "frame_name": self.frame_name,
            "unix_time": self.unix_time,
            "target_labels": [target_label.serialization() for target_label in self.target_labels],
            "metrics_config": self.metrics_config.serialization(),
            "frame_pass_fail_config": self.frame_pass_fail_config.serialization(),
            "critical_object_filter_config": self.critical_object_filter_config.serialization(),
        }

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> PerceptionFrameResult:
        """Deserialize the data to PerceptionFrameResult."""
        nuscene_object_results_list = data.get("nuscene_object_results", [])
        nuscene_object_results = (
            [DynamicObjectWithPerceptionResult.deserialization(obj) for obj in nuscene_object_results_list]
            if nuscene_object_results_list
            else None
        )

        target_labels = []
        for label in data["target_labels"]:
            label_type = label["label_type"]
            if label_type == AutowareLabel.LABEL_TYPE:
                label_class = AutowareLabel
            elif label_type == TrafficLightLabel.LABEL_TYPE:
                label_class = TrafficLightLabel
            else:
                raise ValueError(f"Invalid label type: {label_type}")

            target_labels.append(label_class.deserialization(label))

        return cls(
            object_results=[DynamicObjectWithPerceptionResult.deserialization(obj) for obj in data["object_results"]],
            nuscene_object_results=nuscene_object_results,
            frame_ground_truth=FrameGroundTruth.deserialization(data["frame_ground_truth"]),
            metrics_config=MetricsScoreConfig.deserialization(data["metrics_config"]),
            critical_object_filter_config=CriticalObjectFilterConfig.deserialization(
                data["critical_object_filter_config"]
            ),
            frame_pass_fail_config=PerceptionPassFailConfig.deserialization(data["frame_pass_fail_config"]),
            target_labels=target_labels,
            unix_time=data["unix_time"],
        )


# TODO(vividf): Put this function into a class
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
