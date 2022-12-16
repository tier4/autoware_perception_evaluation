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
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_tp_fp_objects
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.matching.objects_filter import get_fn_objects
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig


class PassFailResult:
    """[summary]
    Attributes:
        self.critical_object_filter_config (CriticalObjectFilterConfig):
                Critical object filter config
        self.frame_pass_fail_config (PerceptionPassFailConfig):
                Frame pass fail config
        self.critical_ground_truth_objects (Optional[List[DynamicObject]]):
                Critical ground truth objects to evaluate for use case
        self.fn_objects (Optional[List[DynamicObject]]):
                The FN (False Negative) ground truth object.
        self.fp_objects (Optional[List[DynamicObjectWithPerceptionResult]]):
                The FP (False Positive) object result.
        self.tp_objects (Optional[List[DynamicObjectWithPerceptionResult]]):
                The TP (True Positive) object result.
    """

    def __init__(
        self,
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
        frame_id: str,
        ego2map: Optional[np.ndarray] = None,
    ) -> None:
        """[summary]

        Args:
            critical_object_filter_config (CriticalObjectFilterConfig):
                    Critical object filter config
            frame_pass_fail_config (PerceptionPassFailConfig):
                    Frame pass fail config
        """
        self.critical_object_filter_config: CriticalObjectFilterConfig = (
            critical_object_filter_config
        )
        self.frame_pass_fail_config: PerceptionPassFailConfig = frame_pass_fail_config
        self.frame_id: str = frame_id
        self.ego2map: Optional[np.ndarray] = ego2map

        self.critical_ground_truth_objects: Optional[List[ObjectType]] = None
        self.fn_objects: Optional[List[ObjectType]] = None
        self.fp_objects_result: Optional[List[DynamicObjectWithPerceptionResult]] = None
        self.tp_objects: Optional[List[DynamicObjectWithPerceptionResult]] = None

    def evaluate(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        ros_critical_ground_truth_objects: List[ObjectType],
    ) -> None:
        """[summary]
        Evaluate pass fail objects.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): The object results
            ros_critical_ground_truth_objects (List[ObjectType]): Ground truth objects filtered by ROS node.
        """
        self.critical_ground_truth_objects = filter_objects(
            frame_id=self.frame_id,
            objects=ros_critical_ground_truth_objects,
            is_gt=True,
            ego2map=self.ego2map,
            **self.critical_object_filter_config.filtering_params,
        )
        if self.critical_ground_truth_objects is not None:
            self.tp_objects, self.fp_objects_result = self.get_tp_fp_objects_result(
                object_results=object_results,
                critical_ground_truth_objects=self.critical_ground_truth_objects,
            )
            self.fn_objects = get_fn_objects(
                ground_truth_objects=self.critical_ground_truth_objects,
                object_results=object_results,
                tp_objects=self.tp_objects,
            )
        else:
            self.fn_objects = None
            self.tp_objects = None
            self.fp_objects_result = None

    def get_fail_object_num(self) -> int:
        """[summary]
        Get the number of fail objects

        Returns:
            int: The number of fail objects
        """
        if self.fn_objects is not None and self.fp_objects_result is not None:
            return len(self.fn_objects) + len(self.fp_objects_result)
        else:
            return 0

    def get_tp_fp_objects_result(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        critical_ground_truth_objects: List[ObjectType],
    ) -> Tuple[List[DynamicObjectWithPerceptionResult], List[DynamicObjectWithPerceptionResult]]:
        """[summary]
        Get FP objects from object results

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]):
                    The object results.
            critical_ground_truth_objects (List[ObjectType]):
                    Ground truth objects to evaluate for use case objects.

        Returns:
            Tuple[List[DynamicObjectWithPerceptionResult], List[DynamicObjectWithPerceptionResult]]: tp_objects, fp_critical_objects
        """
        fp_object_results: List[DynamicObjectWithPerceptionResult] = []
        tp_object_results, fp_object_results = divide_tp_fp_objects(
            object_results=object_results,
            target_labels=self.frame_pass_fail_config.target_labels,
            matching_mode=MatchingMode.IOU2D
            if self.frame_pass_fail_config.evaluation_task.is_2d()
            else MatchingMode.PLANEDISTANCE,
            matching_threshold_list=self.frame_pass_fail_config.matching_threshold_list,
        )

        # filter by critical_ground_truth_objects
        fp_critical_object_results: List[DynamicObjectWithPerceptionResult] = []
        for fp_object_result in fp_object_results:
            if fp_object_result.ground_truth_object in critical_ground_truth_objects:
                fp_critical_object_results.append(fp_object_result)
        return tp_object_results, fp_critical_object_results
