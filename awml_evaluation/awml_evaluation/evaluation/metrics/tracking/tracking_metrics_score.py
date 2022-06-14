from ctypes import Union
from typing import List
from typing import Tuple

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.tracking.clear import CLEAR
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class TrackingMetricsScore:
    """Metrics score class for tracking.

    Attributes:
        self.matching_mode (MatchingMode): The target matching mode.
        self.clears (List[CLEAR]): The list of CLEAR score.
    """

    def __init__(
        self,
        object_results: List[List[DynamicObjectWithPerceptionResult]],
        frame_ground_truths: List[FrameGroundTruth],
        target_labels: List[AutowareLabel],
        max_x_position_list: List[float],
        max_y_position_list: List[float],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
    ) -> None:
        """[summary]

        Args:
            object_results (List[List[DynamicObjectWithPerceptionResult]]): object results for multi frame.
            frame_ground_truths (List[FrameGroundTruth]): ground truth objects for multi frame.
            target_labels (List[AutowareLabel]): e.g. ["car", "pedestrian", "bus"]
            max_x_position_list (List[float]): The list of max x position threshold for each category. (e.g. [10.0, 5.0, 10.0])
            max_y_position_list (List[float]): The list of max y position threshold for each category. (e.g. [2.0, 1.0, 2.0])
            matching_mode (MatchingMode): The target matching mode.
            matching_threshold_list (List[float]): The list of matching threshold for each category. (e.g. [0.5, 0.3, 0.5])
        """
        assert (
            len(target_labels)
            == len(max_x_position_list)
            == len(max_y_position_list)
            == len(matching_threshold_list)
        )
        self.matching_mode: MatchingMode = matching_mode
        self.frame_num: int = len(object_results) - 1

        # CLEAR results for each class
        self.clears: List[CLEAR] = []
        # Calculate score for each target labels
        for target_label, max_x_position, max_y_position, matching_threshold in zip(
            target_labels,
            max_x_position_list,
            max_y_position_list,
            matching_threshold_list,
        ):
            clear_: CLEAR = CLEAR(
                object_results=object_results,
                frame_ground_truths=frame_ground_truths,
                target_labels=[target_label],
                max_x_position_list=[max_x_position],
                max_y_position_list=[max_y_position],
                matching_mode=matching_mode,
                matching_threshold_list=[matching_threshold],
            )
            self.clears.append(clear_)

    def _sum_clear(self) -> Tuple[float, float, int]:
        """Summing up multi CLEAR result.

        Returns:
            mota (float): MOTA score.
            motp (float): MOTP score.
            num_id_switch (int): The number of ID switched.
        """
        mota: float = 0.0
        motp: float = 0.0
        num_gt: int = 0
        num_tp: int = 0
        num_id_switch: int = 0
        for clear in self.clears:
            if clear.mota != float("inf"):
                mota += clear.mota * clear.ground_truth_objects_num
            if clear.motp != float("inf"):
                motp += clear.motp * clear.tp
            num_gt += clear.ground_truth_objects_num
            num_tp += int(clear.tp)
            num_id_switch += clear.id_switch

        mota = float("inf") if num_gt == 0 else mota / num_gt
        mota = max(0.0, mota)
        # NOTE: If tp_metrics is not TPMetricsAP this calculation cause bug
        motp = float("inf") if num_tp == 0 else motp / num_tp
        return mota, motp, num_id_switch

    def __str__(self) -> str:
        """__str__ method"""
        str_: str = "\n"
        # === Total ===
        str_ += f"Total Frames: {self.frame_num}\n"
        # CLEAR
        mota, motp, id_switch = self._sum_clear()
        str_ += f"[TOTAL] MOTA: {mota:.3f}, MOTP: {motp:.3f}, ID switch: {id_switch:d} "
        str_ += f"({self.matching_mode.value})\n"
        # === For each label ===
        # --- Table ---
        str_ += "\n"
        # --- Label ----
        str_ += "|      Label |"
        # CLEAR
        for clear in self.clears:
            for target_label in clear.target_labels:
                target_str: str = ""
                target_str += target_label.value
            str_ += f" {target_str}({clear.matching_threshold_list}) | "
        str_ += "\n"
        str_ += "| :--------: |"
        for _ in self.clears:
            str_ += " :---: |"
        str_ += "\n"
        for field_name in self.clears[0].metrics_field:
            str_ += f"|   {field_name} |"
            for clear_ in self.clears:
                score: Union[int, float] = clear_.results[field_name]
                if isinstance(score, int):
                    str_ += f" {score:d} |"
                else:
                    str_ += f" {score:.3f} |"
            str_ += "\n"

        return str_
