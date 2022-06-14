"""
This module has use case evaluation.

function(
    object_results: List[DynamicObjectWithPerceptionResult]
    ground_truths_for_use_case_evaluation: List[DynamicObject],
) -> Any

"""

from logging import getLogger
from typing import List
from typing import Optional
from typing import Tuple

from awml_evaluation.common.dataset import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.matching.objects_filter import divide_tp_fp_objects
from awml_evaluation.evaluation.matching.objects_filter import filter_ground_truth_objects
from awml_evaluation.evaluation.matching.objects_filter import get_fn_objects
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from awml_evaluation.evaluation.result.perception_frame_config import PerceptionPassFailConfig
import numpy as np

logger = getLogger(__name__)


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

        self.critical_ground_truth_objects: Optional[List[DynamicObject]] = None
        self.fn_objects: Optional[List[DynamicObject]] = None
        self.fp_objects_result: Optional[List[DynamicObjectWithPerceptionResult]] = None
        self.tp_objects: Optional[List[DynamicObjectWithPerceptionResult]] = None

    def evaluate(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
        ros_critical_ground_truth_objects: List[DynamicObject],
    ) -> None:
        """[summary]
        Evaluate pass fail objects.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): The object results
            ros_critical_ground_truth_objects (List[DynamicObject]):
                    Ground truth objects filtered by ROS node.
        """
        self.critical_ground_truth_objects = filter_ground_truth_objects(
            frame_id=self.frame_id,
            objects=ros_critical_ground_truth_objects,
            target_labels=self.critical_object_filter_config.target_labels,
            max_x_position_list=self.critical_object_filter_config.max_x_position_list,
            max_y_position_list=self.critical_object_filter_config.max_y_position_list,
            ego2map=self.ego2map,
        )
        if self.critical_ground_truth_objects is not None:
            self.fn_objects = get_fn_objects(
                ground_truth_objects=self.critical_ground_truth_objects,
                object_results=object_results,
            )
            self.tp_objects, self.fp_objects_result = self.get_tp_fp_objects_result(
                object_results=object_results,
                critical_ground_truth_objects=self.critical_ground_truth_objects,
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
        critical_ground_truth_objects: List[DynamicObject],
    ) -> Tuple[List[DynamicObjectWithPerceptionResult], List[DynamicObjectWithPerceptionResult]]:
        """[summary]
        Get FP objects from object results

        Args:
            object_results (List[DynamicObjectWithPerceptionResult]):
                    The object results.
            critical_ground_truth_objects (List[DynamicObject]):
                    Ground truth objects to evaluate for use case objects.

        Returns:
            Tuple[List[DynamicObjectWithPerceptionResult], List[DynamicObjectWithPerceptionResult]]: tp_objects, fp_critical_objects
        """
        fp_object_results: List[DynamicObjectWithPerceptionResult] = []
        tp_object_results, fp_object_results = divide_tp_fp_objects(
            object_results=object_results,
            target_labels=self.frame_pass_fail_config.target_labels,
            matching_mode=MatchingMode.PLANEDISTANCE,
            matching_threshold_list=self.frame_pass_fail_config.threshold_plane_distance_list,
        )

        # filter by critical_ground_truth_objects
        fp_critical_object_results: List[DynamicObjectWithPerceptionResult] = []
        for fp_object_result in fp_object_results:
            if fp_object_result.ground_truth_object in critical_ground_truth_objects:
                fp_critical_object_results.append(fp_object_result)
        return tp_object_results, fp_critical_object_results
