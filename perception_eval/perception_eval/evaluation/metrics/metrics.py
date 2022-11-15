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

from typing import Dict
from typing import List

from perception_eval.common.label import AutowareLabel
from perception_eval.evaluation.matching.object_matching import MatchingMode
from perception_eval.evaluation.metrics.detection.map import Map
from perception_eval.evaluation.metrics.metrics_score_config import MetricsScoreConfig
from perception_eval.evaluation.metrics.tracking.tracking_metrics_score import TrackingMetricsScore
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class MetricsScore:
    """[summary]
    Metrics score class.

    Attributes:
        self.detection_config (Optional[DetectionMetricsConfig])
        self.tracking_config (Optional[TrackingMetricsConfig])
        self.prediction_config (Optional[PredictionMetricsConfig]): TBD
        self.maps (List[Map]): The list of mAP class object. Each mAP is different from threshold
                               for matching (ex. IoU 0.3).
        self.tracking_scores (List[TrackingMetricsScore])
        self.prediction_scores (List[TODO]): TBD
    """

    def __init__(
        self,
        config: MetricsScoreConfig,
        used_frame: List[int],
    ) -> None:
        """[summary]
        Args:
            metrics_config (MetricsScoreConfig): A config for metrics
            used_frame: List[int]: The list of chosen frame.
        """
        self.detection_config = config.detection_config
        self.tracking_config = config.tracking_config
        self.prediction_config = config.prediction_config
        # detection metrics scores for each matching method
        self.maps: List[Map] = []
        # tracking metrics scores for each matching method
        self.tracking_scores: List[TrackingMetricsScore] = []
        # TODO: prediction metrics scores for each matching method
        self.prediction_scores: List = []
        self.evaluation_task = config.evaluation_task

        self.__num_frame: int = len(used_frame)
        self.__num_gt: int = 0
        self.__used_frame: List[int] = used_frame

    def __str__(self) -> str:
        """[summary]
        Str method
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
        for map_ in self.maps:
            # whole result
            str_ += str(map_)

        # tracking
        for track_score in self.tracking_scores:
            str_ += str(track_score)

        # TODO: prediction

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
        object_results: Dict[AutowareLabel, List[DynamicObjectWithPerceptionResult]],
        num_ground_truth: Dict[AutowareLabel, int],
    ) -> None:
        """[summary]
        Calculate detection metrics

        Args:
            object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object result
        """
        if self.tracking_config is None:
            self.__num_gt += sum(num_ground_truth.values())

        for distance_threshold_ in self.detection_config.center_distance_thresholds:
            map_ = Map(
                object_results_dict=object_results,
                num_ground_truth_dict=num_ground_truth,
                target_labels=self.detection_config.target_labels,
                matching_mode=MatchingMode.CENTERDISTANCE,
                matching_threshold_list=distance_threshold_,
                is_detection_2d=self.evaluation_task.is_2d(),
            )
            self.maps.append(map_)
        for iou_threshold_2d_ in self.detection_config.iou_2d_thresholds:
            map_ = Map(
                object_results_dict=object_results,
                num_ground_truth_dict=num_ground_truth,
                target_labels=self.detection_config.target_labels,
                matching_mode=MatchingMode.IOU2D,
                matching_threshold_list=iou_threshold_2d_,
                is_detection_2d=self.evaluation_task.is_2d(),
            )
            self.maps.append(map_)

        if self.evaluation_task.is_3d():
            # Only for Detection3D
            for iou_threshold_3d_ in self.detection_config.iou_3d_thresholds:
                map_ = Map(
                    object_results_dict=object_results,
                    num_ground_truth_dict=num_ground_truth,
                    target_labels=self.detection_config.target_labels,
                    matching_mode=MatchingMode.IOU3D,
                    matching_threshold_list=iou_threshold_3d_,
                )
                self.maps.append(map_)
            for plane_distance_threshold_ in self.detection_config.plane_distance_thresholds:
                map_ = Map(
                    object_results_dict=object_results,
                    num_ground_truth_dict=num_ground_truth,
                    target_labels=self.detection_config.target_labels,
                    matching_mode=MatchingMode.PLANEDISTANCE,
                    matching_threshold_list=plane_distance_threshold_,
                )
                self.maps.append(map_)

    def evaluate_tracking(
        self,
        object_results: Dict[AutowareLabel, List[List[DynamicObjectWithPerceptionResult]]],
        num_ground_truth: Dict[AutowareLabel, int],
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
        object_results: Dict[AutowareLabel, List[DynamicObjectWithPerceptionResult]],
        num_ground_truth: Dict[AutowareLabel, int],
    ) -> None:
        """[summary]
        Calculate prediction metrics

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): The list of object result
        """
        pass
