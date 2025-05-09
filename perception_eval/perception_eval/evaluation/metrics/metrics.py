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

from collections import defaultdict
from itertools import chain
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching.matching_config import MatchingConfig
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

from .classification import ClassificationMetricsScore
from .detection import Map
from .metrics_score_config import MetricsScoreConfig
from .tracking import TrackingMetricsScore


class MetricsScore:
    """Metrics score class.

    Attributes:
        detection_config (Optional[DetectionMetricsConfig]): Config for detection evaluation.
        tracking_config (Optional[DetectionMetricsConfig]): Config for tracking evaluation.
        prediction_config (Optional[PredictionMetricsConfig]): Config for prediction evaluation.
        classification_config (Optional[ClassificationMetricsConfig]): Config for classification evaluation.
        evaluation_task (EvaluationTask): EvaluationTask instance.
        maps (List[Map]): List of mAP instances. Each mAP is different from threshold
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
        self.maps: List[Map] = []
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
        for map_ in self.maps:
            # whole result
            str_ += str(map_)

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

    def flatten_and_group_object_results_by_match_config(
        self,
        object_results: Union[
            Dict[LabelType, Dict[MatchingConfig, List[DynamicObjectWithPerceptionResult]]],
            Dict[LabelType, Dict[MatchingConfig, List[List[DynamicObjectWithPerceptionResult]]]],
        ],
    ) -> Dict[MatchingConfig, Dict[LabelType, List[DynamicObjectWithPerceptionResult]]]:
        """
        Flatten nested detection results and regroup them by matching configuration and label.

        This method supports both single-frame and multi-frame object results.
        In the case of multi-frame results (i.e., lists of lists), the method flattens the
        nested lists into a single list per (label, MatchingConfig) pair.
        It then reorganizes the data into a structure grouped first by MatchingConfig
        and then by LabelType, which simplifies downstream metric computations.

        Args:
            object_results: A dictionary where the first key is a LabelType, the second key is a MatchingConfig,
                and the value is either a list of DynamicObjectWithPerceptionResult (for single-frame input) or
                a list of lists of them (for multi-frame input).

        Returns:
            A dictionary where each key is a MatchingConfig, and each value is another dictionary
            mapping LabelType to a flattened list of DynamicObjectWithPerceptionResult.
        """
        results_by_match_config: Dict[
            MatchingConfig, Dict[LabelType, List[DynamicObjectWithPerceptionResult]]
        ] = defaultdict(dict)

        for label, method_results in object_results.items():
            for matching_config, result_list in method_results.items():
                if all(isinstance(r, list) for r in result_list):
                    result_list_flat = list(chain.from_iterable(result_list))
                else:
                    result_list_flat = result_list

                results_by_match_config[matching_config][label] = result_list_flat

        return results_by_match_config

    # TODO(vividf): Refactor this function to always accept flattened input (List[DynamicObjectWithPerceptionResult]).
    # Move multi-frame handling and scenario-level aggregation logic to a new function
    # reference: https://github.com/tier4/autoware_perception_evaluation/pull/212#discussion_r2079100088
    def evaluate_detection(
        self,
        object_results: Union[
            Dict[LabelType, Dict[MatchingConfig, List[DynamicObjectWithPerceptionResult]]],  # Single frame
            Dict[LabelType, Dict[MatchingConfig, List[List[DynamicObjectWithPerceptionResult]]]],  # Multiple frames
        ],
        num_ground_truth: Dict[LabelType, int],
    ) -> None:
        """
        Evaluate detection performance and calculate detection-related metrics such as mAP.

        This function processes detection results grouped by label and matching configuration.
        It supports both single-frame and multi-frame input formats. Internally, the object results
        are first flattened and regrouped by MatchingConfig using
        `flatten_and_group_object_results_by_match_config()`.

        For each unique (MatchingMode, threshold) pair, it computes per-label statistics and
        passes them to a Map instance for final mAP (mean Average Precision) calculation.

        Args:
            object_results: A nested dictionary of detection results where:
                - The first key is a LabelType (e.g., car, pedestrian),
                - The second key is a MatchingConfig (e.g., IoU@0.5),
                - The value is either a list (single frame) or list of lists (multi-frame)
                of DynamicObjectWithPerceptionResult instances.
            num_ground_truth: A dictionary mapping each label to the number of ground truth objects.
        """

        if self.tracking_config is None:
            self.__num_gt += sum(num_ground_truth.values())

        # Group by (matching_mode_str, threshold): Dict[LabelType -> List[DynamicObjectWithPerceptionResult]]
        # Example: (center_distance_bev, 0,5): {car: List[DynamicObjectWithPerceptionResult]}
        results_by_match_config = self.flatten_and_group_object_results_by_match_config(object_results)

        # For each (matching_mode, threshold), group labels and compute MAP
        for (matching_mode, threshold), label_results in results_by_match_config.items():
            target_labels = list(label_results.keys())
            num_gt_dict = {label: num_ground_truth.get(label, 0) for label in target_labels}
            thresholds = [threshold] * len(target_labels)

            # TODO(vividf): Rename variable to avoid shadowing Python built-in keywords like 'map'
            self.maps.append(
                Map(
                    object_results_dict=label_results,
                    num_ground_truth_dict=num_gt_dict,
                    target_labels=target_labels,
                    matching_mode=matching_mode,
                    # TODO(vividf): Rename matching_threshold_list to avoid variable name with data type
                    matching_threshold_list=thresholds,
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
