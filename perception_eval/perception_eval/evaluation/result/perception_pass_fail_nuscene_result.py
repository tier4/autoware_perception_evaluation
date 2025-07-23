# Copyright 2025 TIER IV, Inc.

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
from typing import Optional
import warnings

from perception_eval.common import ObjectType
from perception_eval.common.label import LabelType
from perception_eval.common.status import MatchingStatus
from perception_eval.common.threshold import get_label_threshold
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig


# Change the class name back to PassFailResult after deleting the old PassFailResult class
class PassFailNusceneResult:
    """
    Pass/Fail evaluator using nuscene_object_results as input.
    This class is used for keeping TP/FP/TN/FP object results and GT objects for critical GT objects.

    Attributes:
        critical_object_filter_config (CriticalObjectFilterConfig): Critical object filter config.
        frame_pass_fail_config (PerceptionPassFailConfig): Frame pass fail config.
        tn_objects (List[ObjectType]): TN ground truth objects list.
        fn_objects (List[ObjectType]): FN ground truth objects list.
        fp_object_results (List[DynamicObjectWithPerceptionResult]): FP object results list.
        tp_object_results (List[DynamicObjectWithPerceptionResult]): TP object results list.

    Args:
        unix_time (int): UNIX timestamp.
        frame_number (int): The Number of frame.
        critical_object_filter_config (CriticalObjectFilterConfig): Critical object filter config.
        frame_pass_fail_config (PerceptionPassFailConfig): Frame pass fail config.
        frame_id (str): `base_link` or `map`.
        transforms (Optional[TransformDict]): Array of 4x4 matrix to transform coordinates from ego to map.
            Defaults to None.
    """

    def __init__(
        self,
        unix_time: int,
        frame_number: int,
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
        transforms: Optional[TransformDict] = None,
    ) -> None:
        self.unix_time: int = unix_time
        self.frame_number: int = frame_number
        # TODO(ktro2828): merge CriticalObjectFilterConfig and FramePassFailConfig into one
        self.critical_object_filter_config: CriticalObjectFilterConfig = critical_object_filter_config
        self.frame_pass_fail_config: PerceptionPassFailConfig = frame_pass_fail_config
        self.transforms = transforms
        # Build label-to-threshold mapping
        self.label_thresholds: Dict[LabelType, float] = self._build_label_thresholds()
        # Decide matching mode once
        self.mode = (
            MatchingMode.IOU2D if self.frame_pass_fail_config.evaluation_task.is_2d() else MatchingMode.PLANEDISTANCE
        )
        self.selected_object_results: List[DynamicObjectWithPerceptionResult] = []

        # Results after evaluation
        self.tn_objects: List[ObjectType] = []
        self.fn_objects: List[ObjectType] = []
        self.fp_object_results: List[DynamicObjectWithPerceptionResult] = []
        self.tp_object_results: List[DynamicObjectWithPerceptionResult] = []

    def _build_label_thresholds(self) -> Dict[LabelType, float]:
        """
        Build a label-to-threshold mapping from PerceptionPassFailConfig.
        Returns:
            Dict[LabelType, float]: Dict mapping each LabelType to its threshold.
        """
        label_list: List[LabelType] = self.frame_pass_fail_config.target_labels
        threshold_list: List[float] = self.frame_pass_fail_config.matching_threshold_list
        return {label: threshold for label, threshold in zip(label_list, threshold_list)}

    def evaluate(
        self,
        nuscene_object_results: Dict[
            MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]
        ],
        ground_truth_objects: List[ObjectType],
    ) -> None:
        """
        Evaluate pass/fail using nuscene_object_results and ground truth objects.
        This method selects the relevant object results only once and stores them for later use.
        """

        self.selected_object_results = self._select_object_results(nuscene_object_results)
        self.tp_object_results, self.fp_object_results = self.get_positive_objects(self.selected_object_results)
        self.tn_objects, self.fn_objects = self.get_negative_objects(ground_truth_objects, self.selected_object_results)

    def _select_object_results(self, nuscene_object_results) -> List[DynamicObjectWithPerceptionResult]:
        """
        Select object results for each label and its configured threshold.
        Raises an error if the required matching mode is not present.
        Args:
            nuscene_object_results: Nested dict of object results by mode, label, and threshold.
        Returns:
            List of selected DynamicObjectWithPerceptionResult.
        Raises:
            ValueError: If the required matching mode is not present in nuscene_object_results.
        """
        selected_object_results = []
        if nuscene_object_results is None:
            return selected_object_results
        if self.mode not in nuscene_object_results:
            raise ValueError(
                f"Required matching mode {self.mode} not found in nuscene_object_results. Please specify matching mode in PerceptionEvaluationConfig."
            )
        for label in self.frame_pass_fail_config.target_labels:
            threshold = self.label_thresholds[label]
            label_dict = nuscene_object_results[self.mode].get(label)
            if label_dict is None:
                continue
            results_for_label_and_threshold = label_dict.get(threshold)
            if results_for_label_and_threshold is None:
                continue
            selected_object_results.extend(results_for_label_and_threshold)
        return selected_object_results

    def get_positive_objects(self, object_results: List[DynamicObjectWithPerceptionResult]):
        """
        Returns TP (True Positive) and FP (False Positive) object results as a tuple.
        Args:
            object_results: List of matched estimation and GT objects.
        Returns:
            (tp_object_results, fp_object_results): Tuple of TP and FP lists.
        """
        tp_object_results: List[DynamicObjectWithPerceptionResult] = []
        fp_object_results: List[DynamicObjectWithPerceptionResult] = []
        for object_result in object_results:
            if object_result.ground_truth_object is None:
                fp_object_results.append(object_result)
                continue
            # Use GT label for threshold, but use EST label if GT is FP validation
            semantic_label = (
                object_result.estimated_object.semantic_label
                if object_result.ground_truth_object.semantic_label.is_fp()
                else object_result.ground_truth_object.semantic_label
            )
            matching_threshold: Optional[float] = get_label_threshold(
                semantic_label=semantic_label,
                target_labels=self.frame_pass_fail_config.target_labels,
                threshold_list=self.frame_pass_fail_config.matching_threshold_list,
            )
            est_status, gt_status = object_result.get_status(self.mode, matching_threshold)
            if est_status == MatchingStatus.FP:
                if gt_status == MatchingStatus.TN:
                    continue
                fp_object_results.append(object_result)
            elif est_status == MatchingStatus.TP and gt_status == MatchingStatus.TP:
                tp_object_results.append(object_result)
        return tp_object_results, fp_object_results

    def get_negative_objects(
        self, ground_truth_objects: List[ObjectType], object_results: List[DynamicObjectWithPerceptionResult]
    ):
        """
        Returns TN (True Negative) and FN (False Negative) objects as a tuple.
        Args:
            ground_truth_objects: List of ground truth objects.
            object_results: List of object results.
        Returns:
            (tn_objects, fn_objects): Tuple of TN and FN lists.
        """
        tn_objects: List[ObjectType] = []
        fn_objects: List[ObjectType] = []
        non_candidates: List[ObjectType] = []
        for object_result in object_results:
            semantic_label = (
                object_result.estimated_object.semantic_label
                if (
                    object_result.ground_truth_object is None
                    or object_result.ground_truth_object.semantic_label.is_fp()
                )
                else object_result.ground_truth_object.semantic_label
            )
            matching_threshold: Optional[float] = get_label_threshold(
                semantic_label,
                self.frame_pass_fail_config.target_labels,
                self.frame_pass_fail_config.matching_threshold_list,
            )
            _, gt_status = object_result.get_status(self.mode, matching_threshold)
            if gt_status == MatchingStatus.TN:
                tn_objects.append(object_result.ground_truth_object)
            elif gt_status == MatchingStatus.FN:
                fn_objects.append(object_result.ground_truth_object)
            if gt_status is not None:
                non_candidates.append(object_result.ground_truth_object)
        for ground_truth_object in ground_truth_objects:
            if ground_truth_object in non_candidates:
                continue
            if ground_truth_object.semantic_label.is_fp():
                tn_objects.append(ground_truth_object)
            else:
                fn_objects.append(ground_truth_object)
        return tn_objects, fn_objects

    def get_num_success(self) -> int:
        """
        Returns the number of successful detections (TP + TN).
        """
        return len(self.tp_object_results) + len(self.tn_objects)

    def get_num_fail(self) -> int:
        """
        Returns the number of failed detections (FP + FN).
        """
        return len(self.fp_object_results) + len(self.fn_objects)

    def get_num_gt(self) -> int:
        """
        Returns the number of ground truth objects.
        """
        if self.frame_pass_fail_config.evaluation_task.is_fp_validation():
            return len(self.fp_object_results) + len(self.tn_objects)
        else:
            return len(self.tp_object_results) + len(self.fn_objects)

    def get_fail_object_num(self) -> int:
        """
        Returns the number of fail objects (deprecated, use get_num_fail).
        """
        warnings.warn(
            "`get_fail_object_num()` is removed in next minor update, please use `get_num_fail()`",
            DeprecationWarning,
        )
        return len(self.fn_objects) + len(self.fp_object_results)
