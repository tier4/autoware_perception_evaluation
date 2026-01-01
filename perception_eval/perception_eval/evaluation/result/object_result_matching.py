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

from cProfile import label
from collections import defaultdict
from dataclasses import dataclass
import functools
from re import L, Match
from token import OP
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from perception_eval.common import DynamicObject2D
from perception_eval.common import ObjectType
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import Label
from perception_eval.common.label import LabelType
from perception_eval.common.label import TrafficLightLabel
from perception_eval.common.schema import FrameID
from perception_eval.common.threshold import get_label_threshold
from perception_eval.common.transform import TransformDict
from perception_eval.common.object import DynamicObject
from perception_eval.evaluation.matching import CenterDistanceBEVMatching
from perception_eval.evaluation.matching import CenterDistanceMatching
from perception_eval.evaluation.matching import IOU2dMatching
from perception_eval.evaluation.matching import IOU3dMatching
from perception_eval.evaluation.matching import MatchingLabelPolicy
from perception_eval.evaluation.matching import MatchingMethod
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching import PlaneDistanceMatching
from perception_eval.evaluation.metrics.metrics_score_config import MetricsScoreConfig
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

@dataclass(frozen=True)
class MatchingMatrices:

    matching_scores: np.ndarray # [num_est_box, num_gt_box]
    matching_valid_masks: np.ndarray    # [est_box, gt_box]


# TODO(vividf): Remove this after we define threshold in Dict
def convert_nested_thresholds(thresholds: List[List[float]], labels: List[LabelType]) -> Dict[LabelType, List[float]]:
    """
    Convert nested thresholds [num_thresholds][num_labels]
    into a flat dict: {label: List[float]}.

    Args:
        thresholds: 2D list of thresholds (row = threshold group, col = label).
        labels: List of labels in order of columns.

    Returns:
        Dict mapping each label to its list of thresholds.
    """
    result: Dict[LabelType, List[float]] = {label: [] for label in labels}
    for row in thresholds:
        for i, value in enumerate(row):
            result[labels[i]].append(value)

    return result


class NuscenesObjectMatcher:
    """
    Class to perform NuScenes-style matching between estimated and ground truth objects
    using various matching modes (e.g., center distance, IoU) and thresholds.

    This class supports greedy 1-to-1 matching with configurable label policies,
    and returns results grouped by matching mode → label → threshold.

    To improve efficiency, it computes a pairwise cost matrix once per matching mode and label,
    and reuses this matrix across all thresholds for that configuration. This avoids redundant
    recomputation of the same matching mode.

    Note that NuscenesObjectMatcher is not supporting for different label_policy in current stage.

    Attributes:
        evaluation_task (EvaluationTask): The evaluation task type (e.g., DETECTION).
        metrics_config (MetricsScoreConfig): Configuration for evaluation thresholds and target labels.
        transforms (Optional[TransformDict]): Optional transform dictionary for coordinate conversion.
        uuid_matching_first (bool): Whether matching based on uuid first or not. Note that this is only used for classification2D.
        matching_class_agnostic_fps (bool): Set True to allow matching FPs to any ground truths without matching their label.
    """

    def __init__(
        self,
        evaluation_task: EvaluationTask,
        metrics_config: MetricsScoreConfig,
        matching_label_policy: MatchingLabelPolicy = MatchingLabelPolicy.DEFAULT,
        uuid_matching_first: bool = False,
        matching_class_agnostic_fps: bool = False,
        transforms: Optional[TransformDict] = None,
    ):
        self.evaluation_task = evaluation_task
        self.metrics_config = metrics_config
        self.matching_label_policy = matching_label_policy
        self.transforms = transforms
        self.matching_config_map = self._build_matching_config_map()

        self.uuid_matching_first = uuid_matching_first
        # If the matching label policy is DEFAULT, then it allows matching class_agnostic fps
        self.matching_class_agnostic_fps = matching_class_agnostic_fps if matching_label_policy in [MatchingLabelPolicy.DEFAULT] else False

    def _build_matching_config_map(
            self) -> Dict[MatchingMode, Dict[LabelType, List[float]]]:
        """
        Convert threshold lists from metrics config to label-to-threshold mappings per matching mode.

        Returns:
            Dict mapping each MatchingMode to a label-wise threshold dictionary.
        """
        label_list: List[LabelType] = self.metrics_config.target_labels
        return {
            mode: convert_nested_thresholds(thresholds, label_list)
            for mode, thresholds in [
                (MatchingMode.CENTERDISTANCE, self.metrics_config.
                 detection_config.center_distance_thresholds),
                (MatchingMode.CENTERDISTANCEBEV, self.metrics_config.
                 detection_config.center_distance_bev_thresholds),
                (MatchingMode.PLANEDISTANCE,
                 self.metrics_config.detection_config.plane_distance_thresholds
                 ),
                (MatchingMode.IOU2D,
                 self.metrics_config.detection_config.iou_2d_thresholds),
                (MatchingMode.IOU3D,
                 self.metrics_config.detection_config.iou_3d_thresholds),
            ] if thresholds
        }

    def _match_tlr_classification2d(
        self, 
        estimated_objects: List[ObjectType],
        ground_truth_objects: List[ObjectType],
        default_threshold: float = -1.0
    ) -> Dict[MatchingMode, Dict[LabelType, Dict[
            float, List[DynamicObjectWithPerceptionResult]]]]:
        """Returns the {MatchingMode.TLR_CLASSIFICATION: {LabelType.TLR_CLASSIFICATION: {threshold: [DynamicObjectWithPerceptionResult, ...], ...}, ...}} for TLR classification.

            This function is used in 2D TLR classification matching.

        Args:
            estimated_objects (List[DynamicObject2D]): Estimated objects list.
            ground_truth_objects (List[DynamicObject2D]): Ground truth objects list.
        Returns:
            Nested dict grouped as:
            {
                MatchingMode.TLR_CLASSIFICATION: {
                    LabelType.traffic_light: {
                        -1.0: [DynamicObjectWithPerceptionResult, ...],
                        ...
                    },
                    ...
                },
                ...
            }

        """
        matching_object_results: Dict[MatchingMode, Dict[LabelType, Dict[
            float, List[DynamicObjectWithPerceptionResult]]]] = defaultdict(
                lambda: defaultdict(lambda: defaultdict(list)))

        # Set only uses O(1) for searching, so we should use Set instead of removing from a list, since removing in List takes O(n)
        matched_est_indices: set = set()
        matched_gt_indices: set = set()

        # There is no ground truths and not FP validation (= all FP)
        if not ground_truth_objects and self.evaluation_task != EvaluationTask.FP_VALIDATION2D:
            fp_object_results = self._add_fps(
                estimated_objects=estimated_objects,
                label_to_thresholds_map=None,
                matched_est_indices=matched_est_indices,
                available_thresholds=[default_threshold],
            )
            matching_object_results[MatchingMode.TLR_CLASSIFICATION].update(fp_object_results)
            return matching_object_results

        # 1. matching based on same label primary
        for est_index, est_object in enumerate(estimated_objects):
            for gt_index, gt_object in enumerate(ground_truth_objects):
                if gt_index in matched_gt_indices:
                    continue

                if est_object.uuid is None or gt_object.uuid is None:
                    raise RuntimeError(
                        f"uuid of estimation and ground truth must be set, but got {est_object.uuid} and {gt_object.uuid}"
                    )

                label_matching = est_object.semantic_label == gt_object.semantic_label and est_object.uuid == gt_object.uuid
                uuid_matching = est_object.uuid == gt_object.uuid if self.uuid_matching_first else True
                if label_matching and uuid_matching:
                    # For TLR classification, there is no threshold, so it always sets to the default threshold
                    matching_object_results[MatchingMode.TLR_CLASSIFICATION][
                        gt_object.semantic_label.label][default_threshold].append(
                            DynamicObjectWithPerceptionResult(
                                estimated_object=est_object,
                                ground_truth_object=gt_object))

                    matched_est_indices.add(est_index)
                    matched_gt_indices.add(gt_index)

        # 2. matching based on same ID
        for est_index, est_object in enumerate(estimated_objects):
            if est_index in matched_est_indices:
                continue
            for gt_index, gt_object in enumerate(ground_truth_objects):
                if gt_index in matched_gt_indices:
                    continue

                # uuid checking is done in the first step
                uuid_matching = est_object.uuid == gt_object.uuid and est_object.frame_id == gt_object.frame_id

                if uuid_matching:
                    # For TLR classification, there is no threshold, so it always sets to the default threshold
                    # When uuid is matched, we report metrics based on the ground truth labels
                    matching_object_results[MatchingMode.TLR_CLASSIFICATION][
                        gt_object.semantic_label.label][default_threshold].append(
                            DynamicObjectWithPerceptionResult(
                                estimated_object=est_object,
                                ground_truth_object=gt_object))
                    
                    matched_est_indices.add(est_index)
                    matched_gt_indices.add(gt_index)

        return matching_object_results

    def _match_classification2d_uuid(
        self, estimated_objects: List[ObjectType],
        ground_truth_objects: List[ObjectType],
        default_threshold: float = -1.0
    ) -> Dict[MatchingMode, Dict[LabelType, Dict[
            float, List[DynamicObjectWithPerceptionResult]]]]:
        """Returns the {MatchingMode.CLASSIFICATION_2D: {LabelType.TLR_CLASSIFICATION: {threshold: [DynamicObjectWithPerceptionResult, ...], ...}, ...}} for matching classification2D with UUID.

        Args:
            estimated_objects (List[DynamicObject2D]): Estimated objects list.
            ground_truth_objects (List[DynamicObject2D]): Ground truth objects list.
        Returns:
            Nested dict grouped as:
            {
                MatchingMode.ClASSIFICATION_2D: {
                    LabelType.CAR: {
                        -1.0: [DynamicObjectWithPerceptionResult, ...],
                        ...
                    },
                    ...
                },
                ...
            }

        """
        matching_object_results: Dict[MatchingMode, Dict[LabelType, Dict[
            float, List[DynamicObjectWithPerceptionResult]]]] = defaultdict(
                lambda: defaultdict(lambda: defaultdict(list)))

        # Set only uses O(1) for searching, so we should use Set instead of removing from a list, since removing in List takes O(n)
        matched_est_indices: set = set()
        matched_gt_indices: set = set()

        # There is no ground truths and not FP validation (= all FP)
        if not ground_truth_objects and self.evaluation_task != EvaluationTask.FP_VALIDATION2D:
            fp_object_results = self._add_fps(
                estimated_objects=estimated_objects,
                available_thresholds=[default_threshold],
                label_to_thresholds_map=None,
                matched_est_indices=matched_est_indices,
            )
            matching_object_results[MatchingMode.CLASSIFICATION_2D].update(fp_object_results)
            return matching_object_results

        # 1. matching based on same label primary
        for est_index, est_object in enumerate(estimated_objects):
            for gt_index, gt_object in enumerate(ground_truth_objects):
                if gt_index in matched_gt_indices[default_threshold]:
                    continue

                if est_object.uuid is None or gt_object.uuid is None:
                    raise RuntimeError(
                        f"uuid of estimation and ground truth must be set, but got {est_object.uuid} and {gt_object.uuid}"
                    )

                uuid_matching = est_object.uuid == gt_object.uuid and est_object.frame_id == gt_object.frame_id
                if uuid_matching:
                    # For classification 2d, there is no threshold, so it always sets to the default threshold
                    matching_object_results[MatchingMode.CLASSIFICATION_2D][
                        gt_object.semantic_label.label][default_threshold].append(
                            DynamicObjectWithPerceptionResult(
                                estimated_object=est_object,
                                ground_truth_object=gt_object))

                    matched_est_indices.add(est_index)
                    matched_gt_indices.add(gt_index)

        rest_estimated_objects_nums = len(estimated_objects) - len(matched_est_indices)
        rest_estimated_cam_traffic_light_frame = any([
            est.frame_id == FrameID.CAM_TRAFFIC_LIGHT
            and est_index not in matched_est_indices
            for est_index, est in enumerate(estimated_objects)
        ])

        # when there are rest of estimated objects, they all are FP.
        if rest_estimated_objects_nums > 0 and not rest_estimated_cam_traffic_light_frame:
            fp_object_results = self._add_fps(
                estimated_objects=estimated_objects,
                label_to_thresholds_map=None,
                matched_est_indices=matched_est_indices,
                available_thresholds=[default_threshold],
            )
            for label, threshold_to_results in fp_object_results.items():
                for threshold, results in threshold_to_results.items():
                    matching_object_results[MatchingMode.CLASSIFICATION_2D][label][threshold] += results

        return matching_object_results

    def match(
        self,
        estimated_objects: List[ObjectType],
        ground_truth_objects: List[ObjectType],
    ) -> Dict[MatchingMode, Dict[LabelType, Dict[
            float, List[DynamicObjectWithPerceptionResult]]]]:
        """
        Perform label-wise and threshold-wise matching between estimated and ground truth objects,
        and return results grouped by matching mode → label → threshold.

        Matching is performed with greedy 1-to-1 assignment sorted by semantic score.
        For false positive validation mode, unmatched estimated objects are excluded from the results.
        Otherwise, unmatched objects are recorded as false positives.

        If no estimated objects are given, the result will still include all labels and thresholds
        with empty matching results.

        Returns:
            Nested dict grouped as:
                {
                  MatchingMode.CENTERDISTANCE: {
                      LabelType.CAR: {
                          0.5: [DynamicObjectWithPerceptionResult, ...],
                          ...
                      },
                      ...
                  },
                  ...
                }
        """
        if len(estimated_objects):
            is_dynamic_2d = isinstance(estimated_objects[0], DynamicObject2D)
            is_classification_2d_task = is_dynamic_2d and estimated_objects[
                0].roi is None
            if is_classification_2d_task:
                if isinstance(estimated_objects[0].semantic_label.label,
                              TrafficLightLabel):
                    return self._match_tlr_classification2d(
                        estimated_objects, ground_truth_objects)
                else:
                    return self._match_classification2d_uuid(
                        estimated_objects, ground_truth_objects)

        # Bounding boxes matching
        return self._match_bounding_boxes(
            estimated_objects=estimated_objects,
            ground_truth_objects=ground_truth_objects)

    def _match_bounding_boxes(
        self, estimated_objects: List[ObjectType],
        ground_truth_objects: List[ObjectType]
    ) -> Dict[MatchingMode, Dict[LabelType, Dict[
            float, List[DynamicObjectWithPerceptionResult]]]]:
        """
        Perform label-wise and threshold-wise matching between estimated and ground truth objects in bounding boxes,
        and return results grouped by matching mode → label → threshold.
        This function is used for all matching modes except for TLR_CLASSIFICATION.

        Matching is performed with greedy 1-to-1 assignment sorted by semantic score.
        For false positive validation mode, unmatched estimated objects are excluded from the results.
        Otherwise, unmatched objects are recorded as false positives.

        If no estimated objects are given, the result will still include all labels and thresholds
        with empty matching results.
        Returns:
            Nested dict grouped as:
                {
                  MatchingMode.CENTERDISTANCE: {
                      LabelType.CAR: {
                          0.5: [DynamicObjectWithPerceptionResult, ...],
                          ...
                      },
                      ...
                  },
                  ...
                }
        """
        # Use functools.partial instead of lambda to ensure the nested defaultdict structure is pickleable.
        nuscene_object_results: Dict[MatchingMode, Dict[LabelType, Dict[
            float, List[DynamicObjectWithPerceptionResult]]]] = defaultdict(
                functools.partial(defaultdict,
                                  functools.partial(defaultdict, list)))

        # All FN cases
        if not estimated_objects:
            for label in self.metrics_config.target_labels:
                for matching_mode, label_to_thresholds_map in self.matching_config_map.items(
                ):
                    thresholds = label_to_thresholds_map.get(label, [])
                    if not thresholds:
                        continue
                    for threshold in thresholds:
                        nuscene_object_results[matching_mode][label][
                            threshold] = []
            return nuscene_object_results

        estimated_objects_sorted = sorted(estimated_objects,
                                          key=lambda x: x.semantic_score,
                                          reverse=True)

        for matching_mode, label_to_thresholds_map in self.matching_config_map.items():
            matching_method_module, _ = _get_matching_module(matching_mode)
            matching_matrices = self._compute_matching_matrix(
                estimated_objects=estimated_objects_sorted,
                ground_truth_objects=ground_truth_objects,
                matching_method_module=matching_method_module,
            )

            object_results = self._add_matchable_bounding_boxes(
                estimated_objects_sorted=estimated_objects_sorted,
                ground_truth_objects=ground_truth_objects,
                matching_matrices=matching_matrices,
                label_to_thresholds_map=label_to_thresholds_map,
            )
            nuscene_object_results[matching_mode] = object_results 
            
            label_threshold_pairs = [
                (label, threshold) for label, thresholds in label_to_thresholds_map.items() for threshold in thresholds
            ]

            # Initialize all entries in the nested result dictionary,
            # even if no matching results are found
            for label, threshold in label_threshold_pairs:
                if threshold not in nuscene_object_results[matching_mode][label]:
                    nuscene_object_results[matching_mode][label][threshold] = []
        
        return nuscene_object_results
        
    def _add_fps(
        self, estimated_objects: List[ObjectType],
        available_thresholds: List[float],
        label_to_thresholds_map: Optional[Dict[LabelType, List[float]]],
        matched_est_indices: set[int],
    ) -> Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]:
        """
        Handle cases where there are no predictions or no ground truth:
            - If there are predictions but no ground truth, treat all predictions as false positives (FP).
            - If both are empty or only ground truth is present, return empty results for all thresholds.
        """
        object_results: Dict[LabelType, Dict[
            float, List[DynamicObjectWithPerceptionResult]]] = defaultdict(
                lambda: defaultdict(list))

        for threshold in available_thresholds:
            # If there are predictions, treat all predictions as false positives (FP)
            for est_index, est_obj in enumerate(estimated_objects):
                if est_index in matched_est_indices:
                    continue 
                
                # When label to thresholds map is not None, we need to check if the threshold is in the label to thresholds map since we assume 
                # it's class-to-class matching.
                if label_to_thresholds_map is not None:
                    est_label = est_obj.semantic_label.label
                    if threshold not in label_to_thresholds_map.get(est_label, []):
                        continue 

                object_results[est_obj.semantic_label.label][threshold].append(
                    DynamicObjectWithPerceptionResult(
                        est_obj,
                        None,
                        self.matching_label_policy,
                        transforms=self.transforms))

        return object_results
    
    def _find_best_match(
        self, 
        est_idx: int, 
        ground_truth_objects: List[ObjectType],
        matching_scores: np.ndarray, 
        matched_gt_indices: set, 
        matching_valid_masks: np.ndarray, 
        label_to_thresholds_map: Dict[LabelType, List[float]],
        threshold: float,
        class_agnostic_matching: bool
    ) -> Optional[Tuple[int, MatchingMethod]]:
        """
        Find the best unmatched ground truth object for the given estimated object index.

        This function iterates over all unmatched ground truth objects and evaluates the matching
        quality between the estimated object and each ground truth candidate using a `MatchingMethod`.

        The "best match" refers to the ground truth object that yields the most favorable matching score
        as defined by the `MatchingMethod.is_better_than()` function. The comparison logic varies by matching
        mode, for example:
            - For distance-based methods (e.g., center distance, plane distance), a smaller value is better.
            - For IoU-based methods, a higher value is better.

        The function returns the index of the best unmatched ground truth object and the corresponding
        MatchingMethod instance. If no suitable match is found, it returns None.

        Returns:
            Optional[Tuple[int, MatchingMethod]]: A tuple containing the index of the selected ground truth object
                and its corresponding `MatchingMethod`, or `None` if no valid unmatched match is found.
        """
        best_gt_idx = None
        best_matching = None

        for gt_idx in range(matching_scores.shape[1]):
            ground_truth_label = ground_truth_objects[gt_idx].semantic_label.label
            # When the ground truth object is already matched or the threshold is not in the label to thresholds map, then we skip it
            # Only consider the matching when the threshold is defined for the ground truth label for both class-to-class and class-agnostic matching.
            # It means even an est and a gt have a high matching score, we don't match them if the threshold is not defined for the ground truth label. 
            if gt_idx in matched_gt_indices or threshold not in label_to_thresholds_map[ground_truth_label]:
                continue
            
            # When the matching is not class-agnostic, we need to check if the matching for class-to-class is valid. This is used when we matching fps to 
            # any ground truth without matching their label.
            if not class_agnostic_matching and not matching_valid_masks[est_idx, gt_idx]:
                continue 

            matching = matching_scores[est_idx, gt_idx]
            if best_matching is None or matching.is_better_than(best_matching.value):
                best_matching = matching
                best_gt_idx = gt_idx

        if best_matching is None:
            return None
        return best_gt_idx, best_matching

    def _add_matchable_bounding_boxes(
        self, 
        estimated_objects_sorted: List[ObjectType],
        ground_truth_objects: List[ObjectType],
        matching_matrices: Optional[MatchingMatrices],
        label_to_thresholds_map: Dict[LabelType, List[float]]
    ) -> Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]:
        """
        Add matchable bounding boxes to the object results.
        1) It iterates over all estimated objects and finds the best matching ground truth objects for each threshold.
        2) Then, it updates the matched indices and object results with the best matching ground truth objects.
        3) If it allows matching clas_agnostic fps, then it matches the unmatched estimated objects to unmatched ground truth objects.
        4) Finally, it adds false positives to the object results. If the task is fail/pass validation, don't need to add false positives.
        
        This function is used for all matching modes except for TLR_CLASSIFICATION.
        Returns:
            Nested dict grouped as:
                {
                    LabelType.CAR: {
                        0.5: [DynamicObjectWithPerceptionResult, ...],
                        ...
                    },
                    ...
                }
        """
        object_results: Dict[LabelType, Dict[
            float, List[DynamicObjectWithPerceptionResult]]] = defaultdict(
                lambda: defaultdict(list))

        available_thresholds = sorted(set(threshold for values in label_to_thresholds_map.values() for threshold in values)) 
        # Empty matching matrices
        if matching_matrices is None:
            # When the matching label policy is ALLOW_ANY or ALLOW_UNKNOWN, we don't need to consider the threshold, every est must be considered as FP.
            if self.matching_label_policy in [MatchingLabelPolicy.ALLOW_ANY, MatchingLabelPolicy.ALLOW_UNKNOWN]:
                thresholds_map = None 
            else:
                thresholds_map = label_to_thresholds_map

            return self._add_fps(
                estimated_objects=estimated_objects_sorted,
                available_thresholds=available_thresholds,
                label_to_thresholds_map=thresholds_map,
                matched_est_indices=set(),
            )
        
        # 1) It iterates over all estimated objects and finds the best matching ground truth objects for each threshold.
        for threshold in available_thresholds:
            matched_est_indices = set()
            matched_gt_indices = set()
            
            # 2) Match boxes with a specific threshold and merge the object results.
            object_results_threshold = self._match_boxes_threshold(
                estimated_objects_sorted=estimated_objects_sorted,
                ground_truth_objects=ground_truth_objects,
                matching_matrices=matching_matrices,
                label_to_thresholds_map=label_to_thresholds_map,
                threshold=threshold,
                class_agnostic_matching=False,
                matched_est_indices=matched_est_indices,
                matched_gt_indices=matched_gt_indices,
            )
            object_results = self._merge_object_results(object_results, object_results_threshold)

            # 3) Match class-agnostic False positives and merge the object results if applicable.
            if self.matching_class_agnostic_fps:
                object_results_threshold_class_agnostic = self._match_boxes_threshold(
                    estimated_objects_sorted=estimated_objects_sorted,
                    ground_truth_objects=ground_truth_objects,
                    matching_matrices=matching_matrices,
                    label_to_thresholds_map=label_to_thresholds_map,
                    threshold=threshold,
                    class_agnostic_matching=True,
                    matched_est_indices=matched_est_indices,
                    matched_gt_indices=matched_gt_indices,
                )
                object_results = self._merge_object_results(object_results, object_results_threshold_class_agnostic)

            # 4) Add false positives to the object results if applicable.
            # Add unmatched estimated objects as false positives if applicable
            if self.evaluation_task is not None and self.evaluation_task.is_fp_validation():
                continue
                
            # When the matching label policy is ALLOW_ANY or ALLOW_UNKNOWN, we don't need to consider the threshold, every est must be considered as FP.
            if self.matching_label_policy in [MatchingLabelPolicy.ALLOW_ANY, MatchingLabelPolicy.ALLOW_UNKNOWN]:
                thresholds_map = None
            else:
                thresholds_map = label_to_thresholds_map

            object_results_fps = self._add_fps(
                estimated_objects=estimated_objects_sorted,
                available_thresholds=[threshold],
                label_to_thresholds_map=thresholds_map,
                matched_est_indices=matched_est_indices,
            )
            object_results = self._merge_object_results(object_results, object_results_fps)

        return object_results
    
    def _match_boxes_threshold(
        self, 
        estimated_objects_sorted: List[ObjectType],
        ground_truth_objects: List[ObjectType],
        matching_matrices: Optional[MatchingMatrices],
        label_to_thresholds_map: Dict[LabelType, List[float]],
        threshold: float, 
        class_agnostic_matching: bool,
        matched_est_indices: set[int],
        matched_gt_indices: set[int],
    ) -> Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]:
        """
        Match boxes with a specific threshold.
        Args:
            estimated_objects_sorted (List[ObjectType]): Sorted estimated objects.
            ground_truth_objects (List[ObjectType]): Ground truth objects.
            matching_matrices (Optional[MatchingMatrices]): Matching matrices.
            label_to_thresholds_map (Dict[LabelType, List[float]]): Label to thresholds map.
            threshold (float): Threshold.
            class_agnostic_matching (bool): Whether to use class-agnostic matching.
        Returns:
            Nested dict grouped as:
                {
                    LabelType.CAR: {
                        0.5: [DynamicObjectWithPerceptionResult, ...],
                        ...
                    },
                    ...
                }
        """
        object_results: Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]] = defaultdict(lambda: defaultdict(list))
        for est_idx in range(len(estimated_objects_sorted)):
            # If the matching label policy is default, and the threshold is not in the label to thresholds map, 
            # then we skip this estimated object for this threshold
            # Example, when the label to threshold map is {LabelType.CAR: [0.5, 0.7], LabelType.TRUCK: [0.6, 0.8]}, and the threshold is 0.6, 
            # then we exclude the estimated object with label LabelType.CAR when matching since it's a class-to-class matching, 
            # where threshold 0.6 only considers the matching between TRUCK.
            est_label = estimated_objects_sorted[est_idx].semantic_label.label
            if self.matching_label_policy in [MatchingLabelPolicy.DEFAULT]:
                thresholds_for_label = label_to_thresholds_map.get(est_label, [])
                if threshold not in thresholds_for_label:
                    continue
                
            if est_idx in matched_est_indices:
                continue 

            best_match = self._find_best_match(
                est_idx=est_idx,
                ground_truth_objects=ground_truth_objects, 
                matching_scores=matching_matrices.matching_scores, 
                matched_gt_indices=matched_gt_indices, 
                matching_valid_masks=matching_matrices.matching_valid_masks, 
                label_to_thresholds_map=label_to_thresholds_map, 
                threshold=threshold,
                class_agnostic_matching=class_agnostic_matching,
            )

            if best_match is None:
                continue

            best_gt_idx, best_matching = best_match
            if not best_matching.is_better_than(threshold):
                continue

            matched_est_indices.add(est_idx)
            matched_gt_indices.add(best_gt_idx)

            object_results[ground_truth_objects[best_gt_idx].semantic_label.label][threshold].append(
                DynamicObjectWithPerceptionResult(
                    estimated_objects_sorted[est_idx],
                    ground_truth_objects[best_gt_idx],
                    self.matching_label_policy,
                    transforms=self.transforms,
                )
            )

        return object_results

    @staticmethod
    def _merge_object_results(
        object_results_1: Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]],
        object_results_2: Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]
    ) -> Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]:
        """
        Merge two object results.
        Returns:
            Nested dict grouped as:
                {
                    LabelType.CAR: {
                        0.5: [DynamicObjectWithPerceptionResult, ...],
                        ...
                    },
                    ...
                }
        """
        for label, threshold_to_results in object_results_2.items():
            for threshold, results in threshold_to_results.items():
                object_results_1[label][threshold].extend(results)
        
        return object_results_1

    def _compute_matching_matrix(
        self,
        estimated_objects: List[ObjectType],
        ground_truth_objects: List[ObjectType],
        matching_method_module: Callable,
    ) -> Optional[MatchingMatrices]:
        """
        Compute a matrix of MatchingMethod instances for all est-gt pairs.

        Returns:
            MatchingMatrix:
                matching_score: 2D numpy array with shape (num_est, num_gt) storing MatchingMethod instances and values
                matching_valid_masks: 2D numpy array with shape (num_est, num_gt) to save a boolean value to indicate if a pair is matchable.
            None if either input is empty.
        """
        if not estimated_objects or not ground_truth_objects:
            return None

        matching_scores = np.full(
            (len(estimated_objects), len(ground_truth_objects)), None)
        matching_valid_masks = np.zeros_like(matching_scores)

        for i, est_obj in enumerate(estimated_objects):
            for j, gt_obj in enumerate(ground_truth_objects):
                matching_scores[i, j] = matching_method_module(
                    est_obj, gt_obj, self.transforms)
                matching_valid_masks[
                    i, j] = self.matching_label_policy.is_matchable(
                        est_obj, gt_obj)

        return MatchingMatrices(
            matching_scores=matching_scores,
            matching_valid_masks=matching_valid_masks, 
        )


def get_object_results(
    evaluation_task: EvaluationTask,
    estimated_objects: List[ObjectType],
    ground_truth_objects: List[ObjectType],
    target_labels: Optional[List[LabelType]] = None,
    matching_label_policy: MatchingLabelPolicy = MatchingLabelPolicy.DEFAULT,
    matching_mode: MatchingMode = MatchingMode.CENTERDISTANCE,
    matchable_thresholds: Optional[List[float]] = None,
    transforms: Optional[TransformDict] = None,
    uuid_matching_first: bool = False,
) -> List[DynamicObjectWithPerceptionResult]:
    """Returns list of DynamicObjectWithPerceptionResult.

    For classification, matching objects their uuid.
    Otherwise, matching them depending on their center distance by default.

    In case of FP validation, estimated objects, which have no matching GT, will be ignored.
    Otherwise, they all are FP.

    Args:
        evaluation_task (EvaluationTask): Evaluation task.
        estimated_objects (List[ObjectType]): Estimated objects list.
        ground_truth_objects (List[ObjectType]): Ground truth objects list.
        target_labels (Optional[List[LabelType]]): List of labels.
        matching_label_policy (MatchingLabelPolicy, optional): Policy of matching objects.
            Defaults to MatchingLabelPolicy.DEFAULT.
        matching_mode (MatchingMode): MatchingMode instance.
        matchable_thresholds (Optional[List[float]]): Thresholds to be.
        transforms (Optional[TransformDict]): Transforms to be applied.
        uuid_matching_first (bool): Whether matching based on uuid first or not.

    Returns:
        object_results (List[DynamicObjectWithPerceptionResult]): Object results list.
    """
    # There is no estimated object (= all FN)
    if not estimated_objects:
        return []

    # There is no GT and not FP validation (= all FP)
    if not ground_truth_objects and not evaluation_task.is_fp_validation():
        return _get_fp_object_results(estimated_objects)

    assert isinstance(
        ground_truth_objects[0], type(estimated_objects[0])
    ), f"Type of estimation and ground truth must be same, but got {type(estimated_objects[0])} and {type(ground_truth_objects[0])}"

    if (
        isinstance(estimated_objects[0], DynamicObject2D)
        and (estimated_objects[0].roi is None or ground_truth_objects[0].roi is None)
        and isinstance(estimated_objects[0].semantic_label.label, TrafficLightLabel)
    ):
        return _get_object_results_for_tlr(estimated_objects, ground_truth_objects, uuid_matching_first)
    elif isinstance(estimated_objects[0], DynamicObject2D) and (
        estimated_objects[0].roi is None or ground_truth_objects[0].roi is None
    ):
        return _get_object_results_with_id(estimated_objects, ground_truth_objects)

    matching_method_module, maximize = _get_matching_module(matching_mode)
    score_table: np.ndarray = _get_score_table(
        estimated_objects,
        ground_truth_objects,
        matching_label_policy,
        matching_method_module,
        target_labels,
        matchable_thresholds,
        transforms,
    )

    scores = score_table[..., 0]
    is_valid = score_table[..., 1]
    masked_scores = np.where(is_valid, scores, np.nan)

    object_results: List[DynamicObjectWithPerceptionResult] = []
    estimated_objects_: List[ObjectType] = estimated_objects.copy()
    ground_truth_objects_: List[ObjectType] = ground_truth_objects.copy()
    # 1. Matching the nearest estimated objects and GTs which have the same label
    num_estimation, *_ = score_table.shape
    for _ in range(num_estimation):
        if np.isnan(masked_scores).all():
            break

        est_idx, gt_idx = (
            np.unravel_index(np.nanargmax(masked_scores), masked_scores.shape)
            if maximize
            else np.unravel_index(np.nanargmin(masked_scores), masked_scores.shape)
        )

        est_obj = estimated_objects_.pop(est_idx)
        gt_obj = ground_truth_objects_.pop(gt_idx)
        result = DynamicObjectWithPerceptionResult(est_obj, gt_obj, matching_label_policy, transforms=transforms)
        object_results.append(result)

        # Remove corresponding estimated objects and GTs from the score table.
        masked_scores = np.delete(masked_scores, est_idx, axis=0)
        masked_scores = np.delete(masked_scores, gt_idx, axis=1)

        score_table = np.delete(score_table, est_idx, axis=0)
        score_table = np.delete(score_table, gt_idx, axis=1)

    # 2. Matching the nearest estimated objects and GTs regardless of their label
    rest_scores = score_table[..., 0]
    num_rest_estimation, *_ = score_table.shape
    for _ in range(num_rest_estimation):
        if np.isnan(rest_scores).all():
            break

        est_idx, gt_idx = (
            np.unravel_index(np.nanargmax(rest_scores), rest_scores.shape)
            if maximize
            else np.unravel_index(np.nanargmin(rest_scores), rest_scores.shape)
        )

        est_obj = estimated_objects_.pop(est_idx)
        gt_obj = ground_truth_objects_.pop(gt_idx)
        result = DynamicObjectWithPerceptionResult(est_obj, gt_obj, matching_label_policy, transforms=transforms)
        object_results.append(result)

        # Remove corresponding estimated objects and GTs from the score table
        rest_scores = np.delete(rest_scores, est_idx, axis=0)
        rest_scores = np.delete(rest_scores, gt_idx, axis=1)

    # In case of evaluation task is not FP validation,
    # when there are rest of estimated objects, they all are FP.
    # Otherwise, they all are ignored
    if len(estimated_objects_) > 0 and evaluation_task.is_fp_validation() is False:
        object_results += _get_fp_object_results(estimated_objects_)

    return object_results


def _get_object_results_with_id(
    estimated_objects: List[DynamicObject2D],
    ground_truth_objects: List[DynamicObject2D],
) -> List[DynamicObjectWithPerceptionResult]:
    """Returns the list of DynamicObjectWithPerceptionResult considering their uuids.

    This function is used in 2D classification evaluation.

    Args:
        estimated_objects (List[DynamicObject2D]): Estimated objects list.
        ground_truth_objects (List[DynamicObject2D]): Ground truth objects list.

    Returns:
        object_results (List[DynamicObjectWithPerceptionEvaluation]): Object results list.
    """
    object_results: List[DynamicObjectWithPerceptionResult] = []
    estimated_objects_ = estimated_objects.copy()
    ground_truth_objects_ = ground_truth_objects.copy()
    for est_object in estimated_objects:
        for gt_object in ground_truth_objects:
            if est_object.uuid is None or gt_object.uuid is None:
                raise RuntimeError(
                    f"uuid of estimation and ground truth must be set, but got {est_object.uuid} and {gt_object.uuid}"
                )
            if est_object.uuid == gt_object.uuid and est_object.frame_id == gt_object.frame_id:
                object_results.append(
                    DynamicObjectWithPerceptionResult(estimated_object=est_object, ground_truth_object=gt_object)
                )
                estimated_objects_.remove(est_object)
                ground_truth_objects_.remove(gt_object)

    # when there are rest of estimated objects, they all are FP.
    if len(estimated_objects_) > 0 and not any(
        [est.frame_id == FrameID.CAM_TRAFFIC_LIGHT for est in estimated_objects_]
    ):
        object_results += _get_fp_object_results(estimated_objects_)

    return object_results


def _get_object_results_for_tlr(
    estimated_objects: List[DynamicObject2D],
    ground_truth_objects: List[DynamicObject2D],
    uuid_matching_first: bool = False,
) -> List[DynamicObjectWithPerceptionResult]:
    """Returns the list of DynamicObjectWithPerceptionResult for TLR classification.

    This function is used in 2D classification evaluation.

    Args:
        estimated_objects (List[DynamicObject2D]): Estimated objects list.
        ground_truth_objects (List[DynamicObject2D]): Ground truth objects list.
        uuid_matching_first (bool): Whether matching based on uuid first or not.
            if True, Evaluation score is not affected by topic which mix up pedestrian signals, but is affected when the part of traffic light is occluded.
            if False, Reverse of the above.

    Returns:
        object_results (List[DynamicObjectWithPerceptionEvaluation]): Object results list.
    """

    def match_condition(est_object: DynamicObject2D, gt_object: DynamicObject2D, uuid_matching_first: bool) -> bool:
        if uuid_matching_first:
            return (
                est_object.semantic_label == gt_object.semantic_label
                and est_object.uuid == gt_object.uuid
                and est_object.frame_id == gt_object.frame_id
                and est_object in estimated_objects_
                and gt_object in ground_truth_objects_
            )
        else:
            return (
                est_object.semantic_label == gt_object.semantic_label
                and est_object.frame_id == gt_object.frame_id
                and est_object in estimated_objects_
                and gt_object in ground_truth_objects_
            )

    object_results: List[DynamicObjectWithPerceptionResult] = []
    estimated_objects_ = estimated_objects.copy()
    ground_truth_objects_ = ground_truth_objects.copy()
    # 1. matching based on same label primary
    for est_object in estimated_objects:
        for gt_object in ground_truth_objects:
            if est_object.uuid is None or gt_object.uuid is None:
                raise RuntimeError(
                    f"uuid of estimation and ground truth must be set, but got {est_object.uuid} and {gt_object.uuid}"
                )

            if match_condition(est_object, gt_object, uuid_matching_first):
                object_results.append(
                    DynamicObjectWithPerceptionResult(estimated_object=est_object, ground_truth_object=gt_object)
                )
                estimated_objects_.remove(est_object)
                ground_truth_objects_.remove(gt_object)

    # 2. matching based on same ID
    rest_estimated_objects_ = estimated_objects_.copy()
    rest_ground_truth_objects_ = ground_truth_objects_.copy()
    for est_object in rest_estimated_objects_:
        for gt_object in rest_ground_truth_objects_:
            if est_object.uuid is None or gt_object.uuid is None:
                raise RuntimeError(
                    f"uuid of estimation and ground truth must be set, but got {est_object.uuid} and {gt_object.uuid}"
                )

            if (
                est_object.uuid == gt_object.uuid
                and est_object.frame_id == gt_object.frame_id
                and est_object in estimated_objects_
                and gt_object in ground_truth_objects_
            ):
                object_results.append(
                    DynamicObjectWithPerceptionResult(estimated_object=est_object, ground_truth_object=gt_object)
                )
                estimated_objects_.remove(est_object)
                ground_truth_objects_.remove(gt_object)
    return object_results


def _get_fp_object_results(estimated_objects: List[ObjectType]) -> List[DynamicObjectWithPerceptionResult]:
    """Returns the list of DynamicObjectWithPerceptionResult that have no ground truth.

    Args:
        estimated_objects (List[ObjectType]): Estimated objects list.

    Returns:
        object_results (List[DynamicObjectWithPerceptionResult]): FP object results list.
    """
    object_results: List[DynamicObjectWithPerceptionResult] = []
    for est_obj_ in estimated_objects:
        object_result_ = DynamicObjectWithPerceptionResult(estimated_object=est_obj_, ground_truth_object=None)
        object_results.append(object_result_)

    return object_results


def _get_matching_module(matching_mode: MatchingMode) -> Tuple[Callable, bool]:
    """Returns the matching function and boolean flag whether choose maximum value or not.

    Args:
        matching_mode (MatchingMode): MatchingMode instance.

    Returns:
        matching_method_module (Callable): MatchingMethod instance.
        maximize (bool): Whether much bigger is better.
    """
    if matching_mode == MatchingMode.CENTERDISTANCE:
        matching_method_module: CenterDistanceMatching = CenterDistanceMatching
        maximize: bool = False
    elif matching_mode == MatchingMode.CENTERDISTANCEBEV:
        matching_method_module: CenterDistanceBEVMatching = CenterDistanceBEVMatching
        maximize: bool = False
    elif matching_mode == MatchingMode.PLANEDISTANCE:
        matching_method_module: PlaneDistanceMatching = PlaneDistanceMatching
        maximize: bool = False
    elif matching_mode == MatchingMode.IOU2D:
        matching_method_module: IOU2dMatching = IOU2dMatching
        maximize: bool = True
    elif matching_mode == MatchingMode.IOU3D:
        matching_method_module: IOU3dMatching = IOU3dMatching
        maximize: bool = True
    else:
        raise ValueError(f"Unsupported matching mode: {matching_mode}")

    return matching_method_module, maximize


def _get_score_table(
    estimated_objects: List[ObjectType],
    ground_truth_objects: List[ObjectType],
    matching_label_policy: MatchingLabelPolicy,
    matching_method_module: Callable,
    target_labels: Optional[List[LabelType]],
    matchable_thresholds: Optional[List[float]],
    transforms: Optional[TransformDict],
) -> np.ndarray:
    """Returns score table, in shape `(num_estimation, num_ground_truth, 2)`.
    Each element represents `(score, is_same_label)`.

    Args:
        estimated_objects (List[ObjectType]): Estimated objects list.
        ground_truth_objects (List[ObjectType]): Ground truth objects list.
        matching_label_policy (MatchingLabelPolicy): Indicates whether allow to match with unknown label or any.
        matching_method_module (Callable): MatchingMethod instance.
        target_labels (Optional[List[LabelType]]): Target labels to be evaluated.
        matching_thresholds (Optional[List[float]]): List of thresholds

    Returns:
        score_table (numpy.ndarray): in shape (num_estimation, num_ground_truth, 2).
    """
    # fill matching score table, in shape (NumEst, NumGT)
    num_row: int = len(estimated_objects)
    num_col: int = len(ground_truth_objects)
    score_table: np.ndarray = np.full((num_row, num_col, 2), (np.nan, False))
    for i, est_obj in enumerate(estimated_objects):
        for j, gt_obj in enumerate(ground_truth_objects):
            is_same_frame_id: bool = est_obj.frame_id == gt_obj.frame_id

            if is_same_frame_id:
                threshold: Optional[float] = (
                    get_label_threshold(
                        est_obj.semantic_label,
                        target_labels,
                        matchable_thresholds,
                    )
                    if gt_obj.semantic_label.is_fp()
                    else get_label_threshold(
                        gt_obj.semantic_label,
                        target_labels,
                        matchable_thresholds,
                    )
                )

                matching_method: MatchingMethod = matching_method_module(
                    estimated_object=est_obj, ground_truth_object=gt_obj, transforms=transforms
                )

                if threshold is None or (threshold is not None and matching_method.is_better_than(threshold)):
                    is_label_ok = matching_label_policy.is_matchable(est_obj, gt_obj)
                    score_table[i, j] = (matching_method.value, is_label_ok)

    return score_table
