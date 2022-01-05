"""
This module has use case evaluation.

function(
    object_results: List[DynamicObjectWithResult]
    ground_truths_for_use_case_evaluation: List[DynamicObject],
) -> Any

"""

from logging import getLogger
from typing import List
from typing import Optional

from awml_evaluation.common.dataset import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.matching.objects_filter import filter_ground_truth_objects
from awml_evaluation.evaluation.matching.objects_filter import get_fn_objects
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithResult
from awml_evaluation.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from awml_evaluation.evaluation.result.perception_frame_config import PerceptionPassFailConfig

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
        self.uc_fail_objects (Optional[List[DynamicObject]]):
                The ground truth object which detection failed in use case evaluation.
    """

    def __init__(
        self,
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
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
        self.critical_ground_truth_objects: Optional[List[DynamicObject]] = None
        self.uc_fail_objects: Optional[List[DynamicObject]] = None

    def evaluate(
        self,
        object_results: List[DynamicObjectWithResult],
        ros_critical_ground_truth_objects: List[DynamicObject],
    ) -> None:
        """[summary]
        Evaluate pass fail objects.

        Args:
            object_results (List[DynamicObjectWithResult]): The object results
            ros_critical_ground_truth_objects (List[DynamicObject]):
                    Ground truth objects filtered by ROS node.
        """
        self.critical_ground_truth_objects: List[DynamicObject] = filter_ground_truth_objects(
            objects=ros_critical_ground_truth_objects,
            target_labels=self.critical_object_filter_config.target_labels,
            max_x_position_list=self.critical_object_filter_config.max_x_position_list,
            max_y_position_list=self.critical_object_filter_config.max_y_position_list,
        )
        self.uc_fail_objects = get_fn_objects(
            self.critical_ground_truth_objects,
            object_results,
            self.frame_pass_fail_config.target_labels,
            MatchingMode.PLANEDISTANCE,
            self.frame_pass_fail_config.threshold_plane_distance_list,
        )
