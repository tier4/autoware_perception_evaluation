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
from typing import Tuple
import warnings

import numpy as np
from perception_eval.common import ObjectType
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.matching.objects_filter import get_negative_objects
from perception_eval.evaluation.matching.objects_filter import get_positive_objects
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig


class PassFailResult:
    """Class for keeping TP/FP/TN/FP object results and GT objects for critical GT objects.

    Attributes:
        critical_object_filter_config (CriticalObjectFilterConfig): Critical object filter config.
        frame_pass_fail_config (PerceptionPassFailConfig): Frame pass fail config.
        critical_ground_truth_objects (Optional[List[DynamicObject]]): Critical ground truth objects
            must be evaluated at current frame.
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
        ego2map (Optional[numpy.ndarray]): Array of 4x4 matrix to transform coordinates from ego to map.
            Defaults to None.
    """

    def __init__(
        self,
        unix_time: int,
        frame_number: int,
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
        ego2map: Optional[np.ndarray] = None,
    ) -> None:
        self.unix_time: int = unix_time
        self.frame_number: int = frame_number
        # TODO(ktro2828): merge CriticalObjectFilterConfig and FramePassFailConfig into one
        self.critical_object_filter_config: CriticalObjectFilterConfig = critical_object_filter_config
        self.frame_pass_fail_config: PerceptionPassFailConfig = frame_pass_fail_config
        self.ego2map: Optional[np.ndarray] = ego2map

        self.critical_ground_truth_objects: List[ObjectType] = []
        self.tn_objects: List[ObjectType] = []
        self.fn_objects: List[ObjectType] = []
        self.fp_object_results: List[DynamicObjectWithPerceptionResult] = []
        self.tp_object_results: List[DynamicObjectWithPerceptionResult] = []

    def evaluate(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        ros_critical_ground_truth_objects: List[ObjectType],
    ) -> None:
        """Evaluate object results' pass fail.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): Object results list.
            ros_critical_ground_truth_objects (List[ObjectType]): Critical ground truth objects
                must be evaluated at current frame.
        """
        self.critical_ground_truth_objects = filter_objects(
            objects=ros_critical_ground_truth_objects,
            is_gt=True,
            ego2map=self.ego2map,
            **self.critical_object_filter_config.filtering_params,
        )
        self.tp_object_results, self.fp_object_results = self.__get_positive_object_results(
            object_results=object_results,
            critical_ground_truth_objects=self.critical_ground_truth_objects,
        )

        self.tn_objects, self.fn_objects = get_negative_objects(
            self.critical_ground_truth_objects,
            object_results,
            self.frame_pass_fail_config.target_labels,
            self.frame_pass_fail_config.matching_mode_list,
            self.frame_pass_fail_config.matching_threshold_list_for_modes,
        )

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

    def __get_positive_object_results(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        critical_ground_truth_objects: List[ObjectType],
    ) -> Tuple[List[DynamicObjectWithPerceptionResult], List[DynamicObjectWithPerceptionResult]]:
        """Get TP and FP object results list from `object_results`.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): Object results list.
            critical_ground_truth_objects (List[ObjectType]): Critical ground truth objects
                must be evaluated at current frame.

        Returns:
            List[DynamicObjectWithPerceptionResult]: TP object results.
            List[DynamicObjectWithPerceptionResult]: FP object results.
        """
        tp_object_results, fp_object_results = get_positive_objects(
            object_results=object_results,
            target_labels=self.frame_pass_fail_config.target_labels,
            matching_mode_list=self.frame_pass_fail_config.matching_mode_list,
            matching_threshold_list_for_modes=self.frame_pass_fail_config.matching_threshold_list_for_modes,
        )

        # filter by critical_ground_truth_objects
        tp_critical_results: List[DynamicObjectWithPerceptionResult] = [
            tp_result
            for tp_result in tp_object_results
            if tp_result.ground_truth_object in critical_ground_truth_objects
        ]
        fp_critical_results: List[DynamicObjectWithPerceptionResult] = [
            fp_result
            for fp_result in fp_object_results
            if fp_result.ground_truth_object in critical_ground_truth_objects
        ]
        return tp_critical_results, fp_critical_results
