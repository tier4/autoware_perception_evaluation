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

from itertools import chain
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

from .classification import ClassificationMetricsScore
from .detection import Map
from .metrics_score_config import MetricsScoreConfig
from .prediction import PredictionMetricsScore
from .tracking import TrackingMetricsScore


class MetricsScore:
    """Metrics score class.

    Attributes:
        detection_config (Optional[DetectionMetricsConfig]): Config for detection evaluation.
        tracking_config (Optional[DetectionMetricsConfig]): Config for tracking evaluation.
        prediction_config (Optional[PredictionMetricsConfig]): Config for prediction evaluation.
        classification_config (Optional[ClassificationMetricsConfig]): Config for classification evaluation.
        evaluation_task (EvaluationTask): EvaluationTask instance.
        mean_ap_values (List[Map]): List of mAP instances. Each mAP is different from threshold
                               for matching (ex. IoU 0.3).
        tracking_scores (List[TrackingMetricsScore]): List of TrackingMetricsScore instances.
        prediction_scores (List[TODO]): TBD
        classification_score (List[ClassificationMetricsScore]): List of ClassificationMetricsScore instances.

    Args:
        config (MetricsScoreConfig): MetricsScoreConfig instance.
        used_frame: List[int]: List of frame numbers loaded to evaluate.
    """

    def __init__(
        self,
        config: MetricsScoreConfig,
        used_frame: List[int],
    ) -> None:
        self.config = config
        self.detection_config = config.detection_config
        self.tracking_config = config.tracking_config
        self.prediction_config = config.prediction_config
        self.classification_config = config.classification_config
        # detection metrics scores for each matching method
        self.mean_ap_values: List[Map] = []
        # tracking metrics scores for each matching method
        self.tracking_scores: List[TrackingMetricsScore] = []
        # prediction metrics scores for each matching method
        self.prediction_scores: List = []
        self.classification_scores: List[ClassificationMetricsScore] = []
        self.evaluation_task = config.evaluation_task

        self.__num_frame: int = len(used_frame)
        self.__num_gt: int = 0
        self.__used_frame: List[int] = used_frame

    def __reduce__(self) -> Tuple[MetricsScore, Tuple[Any], Dict[Any]]:
        """Serialization and deserialization of the object with pickling."""
        init_args = (
            self.config,
            self.__used_frame,
        )
        state = {
            "mean_ap_values": self.mean_ap_values,
            "tracking_scores": self.tracking_scores,
            "prediction_scores": self.prediction_scores,
            "classification_scores": self.classification_scores,
            "__num_gt": self.__num_gt,
        }
        return (self.__class__, init_args, state)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the object to preserve states after deserialization."""
        state_keys = ["mean_ap_values", "tracking_scores", "prediction_scores", "classification_scores", "__num_gt"]

        for state_key in state_keys:
            value = state.get(state_key, None)
            if state_key.startswith("__"):  # private
                mangled = f"_{self.__class__.__name__}{state_key}"
                setattr(self, mangled, value)
            else:
                setattr(self, state_key, value)

    def __str__(self) -> str:
        """__str__ method

        Returns:
            str: Formatted string.
        """
        str_: str = "\n\n"
        # total frame
        str_ += "Frame:\n"
        str_ += f" Total Num: {self.num_frame}\n"
        str_ += f" Skipped Frames: {self.skipped_frame}\n"
        str_ += f" Skipped Frames Count: {len(self.skipped_frame)}\n"

        str_ += "\n"

        # object num
        str_ += f"Ground Truth Num: {self.num_ground_truth}\n"

        # detection
        for mean_ap in self.mean_ap_values:
            # whole result
            str_ += str(mean_ap)

        # tracking
        for track_score in self.tracking_scores:
            str_ += str(track_score)

        # prediction
        for pred_score in self.prediction_scores:
            str_ += str(pred_score)

        for classify_score in self.classification_scores:
            str_ += str(classify_score)

        return str_

    @property
    def num_frame(self) -> int:
        return self.__num_frame

    @property
    def used_frame(self) -> List[int]:
        return self.__used_frame

    @property
    def skipped_frame(self) -> List[int]:
        return [n for n in range(self.num_frame) if n not in self.used_frame]

    @property
    def num_ground_truth(self) -> int:
        return self.__num_gt

    def evaluate_detection(
        self,
        nuscene_object_results: Dict[
            MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]
        ],
        num_ground_truth: Dict[LabelType, int],
    ) -> None:
        """
        Evaluate detection performance and calculate detection-related metrics such as mAP.

        For each MatchingMode (e.g., IoU, center distance) and each label, this function
        calculates per-threshold detection metrics. The results are passed to Map instances for
        mAP and mAPH computation.

        Args:
            object_results: A nested dictionary of detection results where:
                - The first key is a MatchingMode (e.g., IoU),
                - The second key is a label (e.g., car, pedestrian),
                - The third key is a threshold (e.g., 0.5),
                - The value is either a list of DynamicObjectWithPerceptionResult instances.
            num_ground_truth: A dictionary mapping each label to the number of ground truth objects.
        """
        if self.tracking_config is None:
            self.__num_gt += sum(num_ground_truth.values())

        if self.detection_config is None:
            return
        
        target_labels = self.detection_config.target_labels 
        for matching_mode, label_to_threshold_map in nuscene_object_results.items():
            num_gt_dict = {label: num_ground_truth.get(label, 0) for label in target_labels}
            self.mean_ap_values.append(
                Map(
                    object_results_dict=label_to_threshold_map,
                    num_ground_truth_dict=num_gt_dict,
                    target_labels=target_labels,
                    matching_mode=matching_mode,
                    is_detection_2d=self.evaluation_task.is_2d(),
                )
            )

    def evaluate_tracking(
        self,
        object_results: Dict[LabelType, List[List[DynamicObjectWithPerceptionResult]]],
        num_ground_truth: Dict[LabelType, int],
    ) -> None:
        """[summary]
        Calculate tracking metrics.

        NOTE:
            object_results and ground_truth_objects must be nested list.
            In case of evaluating single frame, [[previous], [current]].
            In case of evaluating multi frame, [[], [t1], [t2], ..., [tn]]

        Args:
            object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object result for each frame.
        """
        self.__num_gt += sum(num_ground_truth.values())

        for distance_threshold_ in self.tracking_config.center_distance_thresholds:
            tracking_score_ = TrackingMetricsScore(
                object_results_dict=object_results,
                num_ground_truth_dict=num_ground_truth,
                target_labels=self.tracking_config.target_labels,
                matching_mode=MatchingMode.CENTERDISTANCE,
                matching_threshold_list=distance_threshold_,
            )
            self.tracking_scores.append(tracking_score_)
        for iou_threshold_2d_ in self.tracking_config.iou_2d_thresholds:
            tracking_score_ = TrackingMetricsScore(
                object_results_dict=object_results,
                num_ground_truth_dict=num_ground_truth,
                target_labels=self.tracking_config.target_labels,
                matching_mode=MatchingMode.IOU2D,
                matching_threshold_list=iou_threshold_2d_,
            )
            self.tracking_scores.append(tracking_score_)

        if self.evaluation_task.is_3d():
            for distance_bev_threshold_ in self.tracking_config.center_distance_bev_thresholds:
                tracking_score_ = TrackingMetricsScore(
                    object_results_dict=object_results,
                    num_ground_truth_dict=num_ground_truth,
                    target_labels=self.tracking_config.target_labels,
                    matching_mode=MatchingMode.CENTERDISTANCEBEV,
                    matching_threshold_list=distance_bev_threshold_,
                )
                self.tracking_scores.append(tracking_score_)
            for iou_threshold_3d_ in self.tracking_config.iou_3d_thresholds:
                tracking_score_ = TrackingMetricsScore(
                    object_results_dict=object_results,
                    num_ground_truth_dict=num_ground_truth,
                    target_labels=self.tracking_config.target_labels,
                    matching_mode=MatchingMode.IOU3D,
                    matching_threshold_list=iou_threshold_3d_,
                )
                self.tracking_scores.append(tracking_score_)
            for plane_distance_threshold_ in self.tracking_config.plane_distance_thresholds:
                tracking_score_ = TrackingMetricsScore(
                    object_results_dict=object_results,
                    num_ground_truth_dict=num_ground_truth,
                    target_labels=self.tracking_config.target_labels,
                    matching_mode=MatchingMode.PLANEDISTANCE,
                    matching_threshold_list=plane_distance_threshold_,
                )
                self.tracking_scores.append(tracking_score_)

    def evaluate_prediction(
        self,
        object_results: Dict[LabelType, List[DynamicObjectWithPerceptionResult]],
        num_ground_truth: Dict[LabelType, int],
    ) -> None:
        """[summary]
        Calculate prediction metrics

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): The list of object result
        """
        self.__num_gt += sum(num_ground_truth.values())

        for top_k in self.prediction_config.top_ks:
            prediction_score_ = PredictionMetricsScore(
                object_results_dict=object_results,
                num_ground_truth_dict=num_ground_truth,
                target_labels=self.prediction_config.target_labels,
                top_k=top_k,
                miss_tolerance=self.prediction_config.miss_tolerance,
            )
            self.prediction_scores.append(prediction_score_)

    def evaluate_classification(
        self,
        object_results: Dict[LabelType, List[List[DynamicObjectWithPerceptionResult]]],
        num_ground_truth: Dict[LabelType, int],
    ) -> None:
        self.__num_gt += sum(num_ground_truth.values())
        classification_score_ = ClassificationMetricsScore(
            object_results_dict=object_results,
            num_ground_truth_dict=num_ground_truth,
            target_labels=self.classification_config.target_labels,
        )
        self.classification_scores.append(classification_score_)
