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

from typing import List
from typing import Optional
import warnings

from perception_eval.common import ObjectType
from perception_eval.common.status import MatchingStatus
from perception_eval.common.threshold import get_label_threshold
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig


# TODO(vividf): Delete this class after we use nuscene object results for all evaluation tasks
class PassFailResult:
    """Class for keeping TP/FP/TN/FP object results and GT objects for critical GT objects.

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

        self.tn_objects: List[ObjectType] = []
        self.fn_objects: List[ObjectType] = []
        self.fp_object_results: List[DynamicObjectWithPerceptionResult] = []
        self.tp_object_results: List[DynamicObjectWithPerceptionResult] = []

    def evaluate(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        ground_truth_objects: List[ObjectType],
    ) -> None:
        """Evaluate object results' pass fail.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): Object results list.
            ground_truth_objects (List[ObjectType]): Ground truth objects which must be evaluated at current frame.
        """
        self.tp_object_results, self.fp_object_results = self._get_positive_objects(object_results)
        self.tn_objects, self.fn_objects = self._get_negative_objects(ground_truth_objects, object_results)

    def _get_positive_objects(self, object_results):
        """Returns TP (True Positive) and FP (False Positive) object results as `tuple`.

        If an object result has better matching score than the matching threshold, it is TP, otherwise FP.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): List of matched estimation and GT objects.
            target_labels (Optional[List[Label]]): List of labels should be evaluated.

        Returns:
            tp_object_results (List[DynamicObjectWithPerceptionResult]): List of TP.
            fp_object_results (List[DynamicObjectWithPerceptionResult]): List of FP.
        """
        tp_object_results: List[DynamicObjectWithPerceptionResult] = []
        fp_object_results: List[DynamicObjectWithPerceptionResult] = []
        for object_result in object_results:
            if object_result.ground_truth_object is None:
                fp_object_results.append(object_result)
                continue

            # to get label threshold, use GT label basically,
            # but use EST label if GT is FP validation
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
            est_status, gt_status = object_result.get_status(
                MatchingMode.IOU2D
                if self.frame_pass_fail_config.evaluation_task.is_2d()
                else MatchingMode.PLANEDISTANCE,
                matching_threshold,
            )
            if est_status == MatchingStatus.FP:
                if gt_status == MatchingStatus.TN:
                    continue
                fp_object_results.append(object_result)
            elif est_status == MatchingStatus.TP and gt_status == MatchingStatus.TP:
                tp_object_results.append(object_result)
        return tp_object_results, fp_object_results

    def _get_negative_objects(self, ground_truth_objects, object_results):
        """Returns TN (True Negative) and FN (False Negative) objects as `tuple`.

        If a ground truth object is contained in object results, it is TP or FP.
        Otherwise, the label of ground truth is 'FP', which means this object should not estimated, it is TN.

        Args:
            ground_truth_objects (List[DynamicObject]): List of ground truth objects.
            object_results (List[DynamicObjectWithPerceptionResult]): List of object results.

        Returns:
            tn_objects (List[DynamicObject]): List of TN.
            fn_objects (List[DynamicObject]): List of FN.
        """
        tn_objects: List[ObjectType] = []
        fn_objects: List[ObjectType] = []

        non_candidates: List[ObjectType] = []
        for object_result in object_results:
            # to get label threshold, use GT label basically,
            # but use EST label if GT is FP validation
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
            _, gt_status = object_result.get_status(
                MatchingMode.IOU2D
                if self.frame_pass_fail_config.evaluation_task.is_2d()
                else MatchingMode.PLANEDISTANCE,
                matching_threshold,
            )
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
        """Returns the number of success.

        Returns:
            int: Number of success.
        """
        return len(self.tp_object_results) + len(self.tn_objects)

    def get_num_fail(self) -> int:
        """Returns the number of fail.

        Returns:
            int: Number of fail.
        """
        return len(self.fp_object_results) + len(self.fn_objects)

    def get_num_gt(self) -> int:
        """Get the number of ground truth objects.

        Returns:
            int: Number of ground truth objects.
        """
        if self.frame_pass_fail_config.evaluation_task.is_fp_validation():
            return len(self.fp_object_results) + len(self.tn_objects)
        else:
            return len(self.tp_object_results) + len(self.fn_objects)

    def get_fail_object_num(self) -> int:
        """Get the number of fail objects.

        Returns:
            int: Number of fail objects.
        """
        warnings.warn(
            "`get_fail_object_num()` is removed in next minor update, please use `get_num_fail()`",
            DeprecationWarning,
        )
        return len(self.fn_objects) + len(self.fp_object_results)
