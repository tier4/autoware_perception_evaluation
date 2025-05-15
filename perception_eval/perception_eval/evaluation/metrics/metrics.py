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
from typing import Union

from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

from .classification import ClassificationMetricsScore
from .detection import Map
from .metrics_score_config import MetricsScoreConfig
from .metrics_utils import flatten_and_group_object_results_by_match_config
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
        self.detection_config = config.detection_config
        self.tracking_config = config.tracking_config
        self.prediction_config = config.prediction_config
        self.classification_config = config.classification_config
        # detection metrics scores for each matching method
        self.mean_ap_values: List[Map] = []
        # tracking metrics scores for each matching method
        self.tracking_scores: List[TrackingMetricsScore] = []
        # TODO: prediction metrics scores for each matching method
        self.prediction_scores: List = []
        self.classification_scores: List[ClassificationMetricsScore] = []
        self.evaluation_task = config.evaluation_task

        self.__num_frame: int = len(used_frame)
        self.__num_gt: int = 0
        self.__used_frame: List[int] = used_frame

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

        # TODO: prediction

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

    def flatten_object_results(
        self,
        object_results: Union[
            Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]],
            Dict[MatchingMode, Dict[LabelType, Dict[float, List[List[DynamicObjectWithPerceptionResult]]]]],
        ],
    ) -> Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]:
        """
        Flatten nested multi-frame object results into a unified single-frame format.

        This function supports both:
          - Single-frame results: List[DynamicObjectWithPerceptionResult]
          - Multi-frame aggregated results: List[List[DynamicObjectWithPerceptionResult]]

        It detects whether each result list is nested (i.e., a list of lists),
        and flattens it accordingly while preserving the structure across:
          - Matching mode (e.g., CENTERDISTANCE, IOU3D)
          - Object label (e.g., car, pedestrian)
          - Matching threshold (e.g., 0.5, 1.0)

        Args:
            object_results: Nested or flat dictionary containing detection results
                grouped by MatchingMode, LabelType, and float threshold.

        Returns:
            Flattened object_results in the format:
                Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]
        """
        flattened_results: Dict[
            MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]
        ] = {}

        for mode, label_map in object_results.items():
            if not isinstance(label_map, dict):
                raise TypeError(f"Expected Dict for object_results[{mode}], got {type(label_map)}: {label_map}")
            flattened_results[mode] = {}
            for label, threshold_map in label_map.items():
                flattened_results[mode][label] = {}
                for threshold, result_list in threshold_map.items():
                    # Check if nested: List[List[...]]
                    if all(isinstance(r, list) for r in result_list):
                        flat_list = list(chain.from_iterable(result_list))  # type: ignore
                    else:
                        flat_list = result_list  # type: ignore
                    flattened_results[mode][label][threshold] = flat_list

        return flattened_results

    # TODO(vividf): Refactor this function to always accept flattened input (List[DynamicObjectWithPerceptionResult]).
    # Move multi-frame handling and scenario-level aggregation logic to a new function
    # reference: https://github.com/tier4/autoware_perception_evaluation/pull/212#discussion_r2079100088
    def evaluate_detection(
        self,
        nuscene_object_results: Union[
            Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]],  # Single frame
            Dict[
                MatchingMode, Dict[LabelType, Dict[float, List[List[DynamicObjectWithPerceptionResult]]]]
            ],  # Multiple frames
        ],
        num_ground_truth: Dict[LabelType, int],
    ) -> None:
        """
        Evaluate detection metrics (e.g., AP, APH, mAP) from object-level matching results.

        This function supports both single-frame and multi-frame formats:
            - Single-frame: List[DynamicObjectWithPerceptionResult]
            - Multi-frame: List[List[DynamicObjectWithPerceptionResult]]

        It first flattens nested lists if necessary, and then constructs a Map instance per
        MatchingMode (e.g., CENTERDISTANCE, IOU3D), aggregating results per label and threshold.

        Args:
            nuscene_object_results: Hierarchical matching results structured as:
                Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]
                or
                Dict[MatchingMode, Dict[LabelType, Dict[float, List[List[DynamicObjectWithPerceptionResult]]]]]
                where thresholds are used to group multiple matching configurations.
            num_ground_truth: A dictionary mapping LabelType to the total number of ground truth
                objects for that label (aggregated across all frames if multi-frame).
        """
        if self.tracking_config is None:
            self.__num_gt += sum(num_ground_truth.values())

        # Flatten List[List[...]] if multi-frame input
        flattened_results = self.flatten_object_results(nuscene_object_results)

        for matching_mode, label_to_threshold_map in flattened_results.items():
            target_labels = list(label_to_threshold_map.keys())
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
        pass

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
