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

import numpy as np
from perception_eval.common import ObjectType
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation import PerceptionFrameConfig
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_tp_fp_objects
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.matching.objects_filter import get_fn_objects


class PassFailResult:
    """
    Attributes:
        frame_config (PerceptionFrameConfig): Parameter config to evaluate pass/fail.
                This allows to specify target ground truth objects dynamically.
        critical_ground_truth_objects (Optional[List[DynamicObject]]): Critical ground truth objects
            must be evaluated at current frame.
        fn_objects ([List[ObjectType]): FN ground truth objects list.
        fp_object_results (List[DynamicObjectWithPerceptionResult]): FP object results list.
        tp_object_results (List[DynamicObjectWithPerceptionResult]): TP object results list.

    Args:
        frame_config (PerceptionFrameConfig): Parameter config to evaluate pass/fail.
                This allows to specify target ground truth objects dynamically.
        ego2map (Optional[numpy.ndarray]): Array of 4x4 matrix to transform coordinates from ego to map.
            Defaults to None.
    """

    def __init__(
        self,
        frame_config: PerceptionFrameConfig,
        ego2map: Optional[np.ndarray] = None,
    ) -> None:
        self.frame_config = frame_config
        self.ego2map: Optional[np.ndarray] = ego2map

        self.critical_ground_truth_objects: List[ObjectType] = []
        self.fn_objects: List[ObjectType] = []
        self.fp_object_results: List[DynamicObjectWithPerceptionResult] = []
        self.tp_object_results: List[DynamicObjectWithPerceptionResult] = []

    def evaluate(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        critical_ground_truth_objects: List[ObjectType],
    ) -> None:
        """Evaluate object results' pass fail.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): Object results list.
            critical_ground_truth_objects (List[ObjectType]): Critical ground truth objects
                must be evaluated at current frame.
        """
        self.critical_ground_truth_objects = filter_objects(
            objects=critical_ground_truth_objects,
            is_gt=True,
            ego2map=self.ego2map,
            **self.frame_config.filtering_params,
        )
        self.tp_object_results, self.fp_object_results = self.get_tp_fp_object_results(
            object_results=object_results,
            critical_ground_truth_objects=self.critical_ground_truth_objects,
        )
        self.fn_objects = get_fn_objects(
            ground_truth_objects=self.critical_ground_truth_objects,
            object_results=object_results,
            tp_object_results=self.tp_object_results,
        )

    def get_fail_object_num(self) -> int:
        """Get the number of fail objects.

        Returns:
            int: Number of fail objects.
        """
        return len(self.fn_objects) + len(self.fp_object_results)

    def get_tp_fp_object_results(
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
        fp_object_results: List[DynamicObjectWithPerceptionResult] = []
        tp_object_results, fp_object_results = divide_tp_fp_objects(
            object_results=object_results,
            target_labels=self.frame_config.target_labels,
            matching_mode=MatchingMode.IOU2D
            if self.frame_config.evaluation_task.is_2d()
            else MatchingMode.PLANEDISTANCE,
            matching_threshold_list=self.frame_config.matching_threshold_list,
        )

        # filter by critical_ground_truth_objects
        fp_critical_object_results: List[DynamicObjectWithPerceptionResult] = []
        for fp_object_result in fp_object_results:
            if fp_object_result.ground_truth_object in critical_ground_truth_objects:
                fp_critical_object_results.append(fp_object_result)
        return tp_object_results, fp_critical_object_results
