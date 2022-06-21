from typing import List

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.detection.ap import Ap
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class MapConfig:
    """[summary]
    Config for mAP calculation

    Attributes:
        self.target_labels (List[AutowareLabel]): Target labels to evaluate
        self.matching_mode (MatchingMode):
                Matching mode like distance between the center of the object, 3d IoU
        self.matching_threshold_list (List[float]):
                Matching thresholds between the estimated object and ground truth for each label
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
                    Matching thresholds between the estimated object and ground truth object
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
        object_results: List[List[DynamicObjectWithPerceptionResult]],
        frame_ground_truths: List[FrameGroundTruth],
        target_labels: List[AutowareLabel],
        max_x_position_list: List[float],
        max_y_position_list: List[float],
        min_point_numbers: List[int],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
    ) -> None:
        """[summary]

        Args:
            object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object results
            ground_truth_objects (List[FrameGroundTruth]) : The list of ground truth for each frame
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
            min_point_numbers (List[int]):
                    min point numbers.
                    For example, if target_labels is ["car", "bike", "pedestrian"],
                    min_point_numbers [5, 0, 0] means
                    Car bboxes including 4 points are filtered out.
                    Car bboxes including 5 points are NOT filtered out.
                    Bike and Pedestrian bboxes are not filtered out(All bboxes are used when calculating metrics.)
            matching_mode (MatchingMode): Matching mode like distance between the center of
                                           the object, 3d IoU.
            matching_threshold_list (List[float]):
                    The matching threshold to evaluate. Defaults to None.
                    For example, if matching_mode = IOU3d and matching_threshold = 0.5,
                    and IoU of the object is higher than "matching_threshold",
                    this function appends to return objects.
        """
        assert (
            len(target_labels)
            == len(max_x_position_list)
            == len(max_y_position_list)
            == len(matching_threshold_list)
            == len(min_point_numbers)
        )

        self.map_config = MapConfig(
            target_labels=target_labels,
            matching_mode=matching_mode,
            matching_threshold_list=matching_threshold_list,
        )

        # calculate AP
        self.aps: List[Ap] = []
        for (
            target_label,
            max_x_position,
            max_y_position,
            matching_threshold,
            min_point_number,
        ) in zip(
            self.map_config.target_labels,
            max_x_position_list,
            max_y_position_list,
            matching_threshold_list,
            min_point_numbers,
        ):
            ap_ = Ap(
                tp_metrics=TPMetricsAp(),
                object_results=object_results,
                frame_ground_truths=frame_ground_truths,
                target_labels=[target_label],
                max_x_position_list=[max_x_position],
                max_y_position_list=[max_y_position],
                min_point_numbers=[min_point_number],
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
                frame_ground_truths=frame_ground_truths,
                target_labels=[target_label],
                max_x_position_list=[max_x_position],
                max_y_position_list=[max_y_position],
                min_point_numbers=min_point_numbers,
                matching_mode=matching_mode,
                matching_threshold_list=[matching_threshold],
            )

            self.aphs.append(aph_)

        # calculate mAPH
        sum_aph: float = 0.0
        for aph in self.aphs:
            sum_aph += aph.ap
        self.maph: float = sum_aph / len(target_labels)

    def __str__(self) -> str:
        """__str__ method"""

        str_: str = "\n"
        str_ += f"mAP: {self.map:.3f}, mAPH: {self.maph:.3f} "
        str_ += f"({self.map_config.matching_mode.value})\n"
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
