from typing import Dict
from typing import List

from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.detection.ap import Ap
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult


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
        object_results_dict: Dict[AutowareLabel, List[DynamicObjectWithPerceptionResult]],
        num_ground_truth_dict: Dict[AutowareLabel, int],
        target_labels: List[AutowareLabel],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
    ) -> None:
        """[summary]

        Args:
            object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object results
            target_labels (List[AutowareLabel]): Target labels to evaluate mAP
            matching_mode (MatchingMode): Matching mode like distance between the center of
                                           the object, 3d IoU.
            matching_threshold_list (List[float]):
                    The matching threshold to evaluate. Defaults to None.
                    For example, if matching_mode = IOU3d and matching_threshold = 0.5,
                    and IoU of the object is higher than "matching_threshold",
                    this function appends to return objects.
        """
        self.target_labels: List[AutowareLabel] = target_labels
        self.matching_mode: MatchingMode = matching_mode
        self.matching_threshold_list: List[float] = matching_threshold_list

        # calculate AP & APH
        self.aps: List[Ap] = []
        self.aphs: List[Ap] = []
        for target_label, matching_threshold in zip(target_labels, matching_threshold_list):
            object_results = object_results_dict[target_label]
            num_ground_truth = num_ground_truth_dict[target_label]
            ap_ = Ap(
                tp_metrics=TPMetricsAp(),
                object_results=object_results,
                num_ground_truth=num_ground_truth,
                target_labels=[target_label],
                matching_mode=matching_mode,
                matching_threshold_list=[matching_threshold],
            )
            self.aps.append(ap_)

            aph_ = Ap(
                tp_metrics=TPMetricsAph(),
                object_results=object_results,
                num_ground_truth=num_ground_truth,
                target_labels=[target_label],
                matching_mode=matching_mode,
                matching_threshold_list=[matching_threshold],
            )
            self.aphs.append(aph_)

        # calculate mAP & mAPH
        sum_ap: float = 0.0
        sum_aph: float = 0.0
        for ap, aph in zip(self.aps, self.aphs):
            sum_ap += ap.ap
            sum_aph += aph.ap
        self.map: float = sum_ap / len(target_labels)
        self.maph: float = sum_aph / len(target_labels)

    def __str__(self) -> str:
        """__str__ method"""

        str_: str = "\n"
        str_ += f"mAP: {self.map:.3f}, mAPH: {self.maph:.3f} "
        str_ += f"({self.matching_mode.value})\n"
        # Table
        str_ += "\n"
        # label
        str_ += "|      Label |"
        target_str: str
        for ap_ in self.aps:
            target_str = ""
            for target in ap_.target_labels:
                target_str += target.value
            str_ += f" {target_str}({ap_.matching_threshold_list}) | "
        str_ += "\n"
        str_ += "| :--------: |"
        for ap_ in self.aps:
            str_ += " :---: |"
        str_ += "\n"
        str_ += "| Predict_num |"
        for ap_ in self.aps:
            str_ += f" {ap_.objects_results_num} |"
        # Each label result
        str_ += "\n"
        str_ += "|         AP |"
        for ap_ in self.aps:
            str_ += f" {ap_.ap:.3f} | "
        str_ += "\n"
        str_ += "|        APH |"
        for aph_ in self.aphs:
            target_str = ""
            for target in aph_.target_labels:
                target_str += target.value
            str_ += f" {aph_.ap:.3f} | "
        str_ += "\n"

        return str_
