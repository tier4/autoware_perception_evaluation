from logging import getLogger
from typing import List

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.detection.ap import Ap
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithResult

logger = getLogger(__name__)


class MapConfig:
    """[summary]
    Config for mAP calculation

    Attributes:
        self.target_labels (List[AutowareLabel]): Target labels to evaluate
        self.matching_mode (MatchingMode):
                Matching mode like distance between the center of the object, 3d IoU
        self.matching_threshold_list (List[float]):
                Matching thresholds between the predicted object and ground truth for each label
    """

    def __init__(
        self,
        target_labels: List[AutowareLabel],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
    ) -> None:
        """[summary]

        Args:
            target_labels (List[AutowareLabel]): Target labels to evaluate
            matching_mode (MatchingMode):
                    Matching mode like distance between the center of the object, 3d IoU.
            matching_threshold_list (List[float]):
                    Matching thresholds between the predicted object and ground truth object
                    for each label.
        """
        self.target_labels: List[AutowareLabel] = target_labels
        self.matching_mode: MatchingMode = matching_mode
        self.matching_threshold_list: List[float] = matching_threshold_list


class Map:
    """[summary]
    mAP class

    Attributes:
        self.map_config (MapConfig): The config for mAP calculation
        self.aps (List[Ap]): The list of AP (Average Precision) for each label
        self.map (float): mAP value
    """

    def __init__(
        self,
        object_results: List[DynamicObjectWithResult],
        ground_truth_objects: List[DynamicObject],
        target_labels: List[AutowareLabel],
        max_x_position_list: List[float],
        max_y_position_list: List[float],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
    ) -> None:
        """[summary]

        Args:
            object_results (DynamicObjectWithResult): The list of object results
            ground_truth_objects (List[DynamicObject]) : The ground truth objects for the frame
            target_labels (List[AutowareLabel]): Target labels to evaluate mAP
            max_x_position_list (List[float]]):
                    The threshold list of maximum x-axis position for each object.
                    Return the object that
                    - max_x_position < object x-axis position < max_x_position.
                    This param use for range limitation of detection algorithm.
            max_y_position_list (List[float]]):
                    The threshold list of maximum y-axis position for each object.
                    Return the object that
                    - max_y_position < object y-axis position < max_y_position.
                    This param use for range limitation of detection algorithm.
            matching_mode (MatchingMode): Matching mode like distance between the center of
                                           the object, 3d IoU.
            matching_threshold_list (List[float]):
                    The matching threshold to evaluate. Defaults to None.
                    For example, if matching_mode = IOU3d and matching_threshold = 0.5,
                    and IoU of the object is higher than "matching_threshold",
                    this function appends to return objects.
        """
        if not target_labels:
            logger.error(f"target_labels is empty ({target_labels})")
            return

        self.map_config = MapConfig(
            target_labels=target_labels,
            matching_mode=matching_mode,
            matching_threshold_list=matching_threshold_list,
        )

        # calculate AP
        self.aps: List[Ap] = []
        for target_label, max_x_position, max_y_position, matching_threshold in zip(
            self.map_config.target_labels,
            max_x_position_list,
            max_y_position_list,
            matching_threshold_list,
        ):
            ap_ = Ap(
                tp_metrics=TPMetricsAp(),
                object_results=object_results,
                ground_truth_objects=ground_truth_objects,
                target_labels=[target_label],
                max_x_position_list=[max_x_position],
                max_y_position_list=[max_y_position],
                matching_mode=matching_mode,
                matching_threshold_list=[matching_threshold],
            )

            self.aps.append(ap_)

        # calculate mAP
        sum_ap: float = 0.0
        for ap in self.aps:
            sum_ap += ap.ap
        self.map: float = sum_ap / len(target_labels)

        # calculate APH
        self.aphs: List[Ap] = []
        for target_label, max_x_position, max_y_position, matching_threshold in zip(
            self.map_config.target_labels,
            max_x_position_list,
            max_y_position_list,
            matching_threshold_list,
        ):
            aph_ = Ap(
                tp_metrics=TPMetricsAph(),
                object_results=object_results,
                ground_truth_objects=ground_truth_objects,
                target_labels=[target_label],
                max_x_position_list=[max_x_position],
                max_y_position_list=[max_y_position],
                matching_mode=matching_mode,
                matching_threshold_list=[matching_threshold],
            )

            self.aphs.append(aph_)

        # calculate mAPH
        sum_aph: float = 0.0
        for aph in self.aphs:
            sum_aph += aph.ap
        self.maph: float = sum_aph / len(target_labels)
