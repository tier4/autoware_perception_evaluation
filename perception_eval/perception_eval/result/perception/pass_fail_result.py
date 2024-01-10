# Copyright 2022-2024 TIER IV, Inc.

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

from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np
from perception_eval.matching import MatchingMode
import perception_eval.matching.objects_filter as objects_filter

if TYPE_CHECKING:
    from perception_eval.object import ObjectType

    from .frame_config import PerceptionFrameConfig
    from .object_result import PerceptionObjectResult


class PassFailResult:
    """Class to determine pass fail.

    Args:
    -----
        unix_time (int): UNIX timestamp.
        frame_number (int): The Number of frame.
        frame_config (PerceptionFrameConfig): Frame level configuration.
        ego2map (Optional[np.ndarray]): 4x4 matrix to transform coordinates from ego to map. Defaults to None.
    """

    def __init__(
        self,
        unix_time: int,
        frame_number: int,
        frame_config: PerceptionFrameConfig,
        ego2map: Optional[np.ndarray] = None,
    ) -> None:
        self.unix_time = unix_time
        self.frame_number = frame_number
        self.frame_config = frame_config
        self.ego2map = ego2map

        self.critical_ground_truth_objects: List[ObjectType] = []
        self.tn_objects: List[ObjectType] = []
        self.fn_objects: List[ObjectType] = []
        self.fp_object_results: List[PerceptionObjectResult] = []
        self.tp_object_results: List[PerceptionObjectResult] = []

    def evaluate(
        self,
        object_results: List[PerceptionObjectResult],
        critical_ground_truth_objects: List[ObjectType],
    ) -> None:
        """Evaluate object results' pass fail.

        Args:
        -----
            object_results (List[PerceptionObjectResult]): List of object results.
            critical_ground_truth_objects (List[ObjectType]): Critical ground truth objects
                must be evaluated at current frame.
        """
        critical_ground_truth_objects = objects_filter.filter_objects(
            objects=critical_ground_truth_objects,
            is_gt=True,
            ego2map=self.ego2map,
            **self.frame_config.filter_param.as_dict(),
        )
        self.tp_object_results, self.fp_object_results = self.__get_positive_object_results(
            object_results=object_results,
            critical_ground_truth_objects=critical_ground_truth_objects,
        )

        self.tn_objects, self.fn_objects = objects_filter.get_negative_objects(
            critical_ground_truth_objects,
            object_results,
            self.frame_config.target_labels,
            MatchingMode.IOU2D if self.frame_config.evaluation_task.is_2d() else MatchingMode.PLANEDISTANCE,
            self.frame_config.success_thresholds,
        )

        self.critical_ground_truth_objects = critical_ground_truth_objects

    def get_num_success(self) -> int:
        """Returns the number of success.

        Returns:
        --------
            int: Number of success.
        """
        return len(self.tp_object_results) + len(self.tn_objects)

    def get_num_fail(self) -> int:
        """Returns the number of fail.

        Returns:
        --------
            int: Number of fail.
        """
        return len(self.fp_object_results) + len(self.fn_objects)

    def __get_positive_object_results(
        self,
        object_results: List[PerceptionObjectResult],
        critical_ground_truth_objects: List[ObjectType],
    ) -> Tuple[List[PerceptionObjectResult], List[PerceptionObjectResult]]:
        """Returns list of TP and FP results from object results.

        Args:
            object_results (List[PerceptionObjectResult]): Object results list.
            critical_ground_truth_objects (List[ObjectType]): Critical ground truth objects
                must be evaluated at current frame.

        Returns:
        --------
            Tuple[List[PerceptionObjectResult], List[PerceptionObjectResult]]: TP and FP object results.
        """
        tp_object_results, fp_object_results = objects_filter.get_positive_objects(
            object_results=object_results,
            target_labels=self.frame_config.target_labels,
            matching_mode=MatchingMode.IOU2D
            if self.frame_config.evaluation_task.is_2d()
            else MatchingMode.PLANEDISTANCE,
            matching_threshold_list=self.frame_config.success_thresholds,
        )

        # filter by critical_ground_truth_objects
        tp_critical_results = [
            tp_result
            for tp_result in tp_object_results
            if tp_result.ground_truth_object in critical_ground_truth_objects
        ]
        fp_critical_results = [
            fp_result
            for fp_result in fp_object_results
            if fp_result.ground_truth_object in critical_ground_truth_objects
        ]
        return tp_critical_results, fp_critical_results
