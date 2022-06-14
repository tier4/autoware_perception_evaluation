from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.threshold import get_label_threshold
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetrics
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from awml_evaluation.evaluation.metrics.tracking._metrics_base import _TrackingMetricsBase
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class CLEAR(_TrackingMetricsBase):
    """CLEAR metrics class that has MOTA and MOTP as sub metrics.
    NOTE: MT, ML, PT is under construction.

    Attributes:
        self.target_labels (List[AutowareLabel]): The list of target label.
        self.max_x_position_list (List[float]): The list of max x position threshold.
        self.max_y_position_list (List[float]): The list of max y position threshold.
        self.matching_mode (MatchingMode): The target matching mode.
        self.tp_metrics (TPMetrics): The way of calculating TP value.
        self.metrics_field (Optional[List[str]]): The list of target metrics name. If not specified, set default supported metrics.
        self.ground_truth_objects_num (int): The number of ground truth.
        self.support_metrics (List[str]): The list of supported metrics name. (["MOTA", "MOTP"])
        self.tp (float): The total value/number of TP.
        self.fp (float): The total value/number of FP.
        self.id_switch (int): The total number of ID switch.
        self.tp_matching_score (float): The total value of matching score in TP.
        self.mota (float): MOTA score.
        self.motp (float): MOTP score.
        self.results (OrderedDict[str, Any]): The dict to keep scores.
    """

    _support_metrics: List[str] = [
        "predict_num",
        "MOTA",
        "MOTP",
        "id_switch",
        "tp",
        "fp",
        "tp_matching_score",
    ]

    def __init__(
        self,
        object_results: List[List[DynamicObjectWithPerceptionResult]],
        frame_ground_truths: List[FrameGroundTruth],
        target_labels: List[AutowareLabel],
        max_x_position_list: List[float],
        max_y_position_list: List[float],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
        tp_metrics: TPMetrics = TPMetricsAp(),
        metrics_field: Optional[List[str]] = None,
    ) -> None:
        """[summary]
        CLEAR metrics.

        NOTE: About objects_results and ground_truth_objects,
            When evaluate 1-frame, index 0 is previous object results.
            When evaluate all frames, index 0 is empty list.

        Args:
            object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object results for each frames.
            frame_ground_truths (List[FrameGroundTruth]): The list of ground truth objects for each frames.
            target_labels (List[AutowareLabel]): The list of target labels.
            max_x_position_list (List[float]): The list of max x positions.
            max_y_position_list (List[float]): The list of max y positions.
            matching_mode (MatchingMode): Matching mode class.
            tp_metrics (TPMetrics): The way of calculating TP value. Defaults to TPMetricsAP.
            metrics_field: List[str]: The list of target sub metrics. Defaults to None.
        """
        super().__init__(
            frame_ground_truths=frame_ground_truths,
            target_labels=target_labels,
            max_x_position_list=max_x_position_list,
            max_y_position_list=max_y_position_list,
            matching_mode=matching_mode,
            matching_threshold_list=matching_threshold_list,
            tp_metrics=tp_metrics,
            metrics_field=metrics_field,
        )

        self.tp: float = 0.0
        self.fp: float = 0.0
        self.id_switch: int = 0
        self.tp_matching_score: float = 0.0
        self.objects_results_num: int = 0

        num_frame: int = len(object_results)
        self.frame_num: int = num_frame - 1
        for i in range(1, num_frame):
            cur_object_results, prev_object_results = self._filter_and_sort(
                frame_id=frame_ground_truths[i].frame_id,
                cur_object_results=object_results[i],
                prev_object_results=object_results[i - 1],
                ego2map=frame_ground_truths[i].ego2map,
            )
            self.objects_results_num += len(cur_object_results)

            # Calculate TP/FP/IDSwitch and total matching score in TP at frame t
            tp_t, fp_t, id_switch_t, tp_matching_score_t = self._calculate_tp_fp(
                cur_object_results=cur_object_results,
                prev_object_results=prev_object_results,
            )
            self.tp += tp_t
            self.fp += fp_t
            self.id_switch += id_switch_t
            self.tp_matching_score += tp_matching_score_t

        self.mota, self.motp = self._calculate_score()

    @property
    def results(self) -> Dict[str, float]:
        return OrderedDict(
            {
                "predict_num": self.objects_results_num,
                "MOTA": self.mota,
                "MOTP": self.motp,
                "id_switch": self.id_switch,
                "tp": self.tp,
                "fp": self.fp,
                "tp_matching_score": self.tp_matching_score,
            }
        )

    def _calculate_score(self):
        """[summary]
        Calculate MOTA and MOTPresult_type.

        NOTE:
            if the number of GT is 0, MOTA returns inf and if the TP score is 0, MOTP returns inf.

        Returns:
            mota (float): MOTA score.
            motp (float): MOTP score.
        """
        if self.ground_truth_objects_num == 0:
            mota: float = float("inf")
        else:
            mota: float = (self.tp - self.fp - self.id_switch) / self.ground_truth_objects_num

        if self.tp == 0.0:
            motp: float = float("inf")
        else:
            motp: float = self.tp_matching_score / self.tp
        mota = max(0.0, mota)

        return mota, motp

    def _calculate_tp_fp(
        self,
        cur_object_results: List[DynamicObjectWithPerceptionResult],
        prev_object_results: List[DynamicObjectWithPerceptionResult],
    ) -> Tuple[float, float, int, float]:
        """[summary]
        Calculate matching compared with previous object results

        Args:
            cur_object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object results at current frame.
            prev_object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object results at previous frame.

        Returns:
            tp (float): Total value of TP. If matching is True, num_tp += 1.0.
            fp (float): Total value of FP. If matching is False, num_fp += 1.0.
            num_id_switch (int): Total number of ID switch. If matching is switched compared with previous result, num_id_switch += 1.
            tp_matching_scores: (float): Total matching score in TP. If matching True, total_tp_scores += matching_score.
        """
        tp: float = 0.0
        fp: float = 0.0
        num_id_switch: int = 0
        tp_matching_score: float = 0.0

        # Assign previous results if same matching pair has
        has_same_tp: bool = False
        is_id_switched: bool = False
        for cur_obj_result in cur_object_results:
            # Assign previous results if same matching pair has
            matching_threshold_: float = get_label_threshold(
                semantic_label=cur_obj_result.estimated_object.semantic_label,
                target_labels=self.target_labels,
                threshold_list=self.matching_threshold_list,
            )
            has_same_tp: bool = False
            is_id_switched: bool = False
            for prev_obj_result in prev_object_results:
                is_id_switched: bool = self._is_id_switched(
                    cur_obj_result,
                    prev_obj_result,
                )
                if is_id_switched:
                    break
                is_same_result: bool = self._is_same_result(
                    cur_obj_result,
                    prev_obj_result,
                )
                if is_same_result is False:
                    continue

                is_tp_prev: bool = prev_obj_result.is_result_correct(
                    self.matching_mode,
                    matching_threshold_,
                )

                has_same_tp = is_same_result * is_tp_prev
                if has_same_tp:
                    # NOTE: The One GT can have multi matching estimated object
                    tp += self.tp_metrics.get_value(cur_obj_result)
                    tp_matching_score += cur_obj_result.get_matching(self.matching_mode).value
                    break
            if has_same_tp:
                continue

            # If there is no same matching pair with previous results
            is_tp_cur: bool = cur_obj_result.is_result_correct(
                self.matching_mode,
                matching_threshold_,
            )
            if is_tp_cur:
                tp += self.tp_metrics.get_value(cur_obj_result)
                tp_matching_score += cur_obj_result.get_matching(self.matching_mode).value
                if is_id_switched:
                    num_id_switch += 1
            else:
                fp += 1.0

        return tp, fp, num_id_switch, tp_matching_score

    @staticmethod
    def _is_id_switched(
        cur_object_result: DynamicObjectWithPerceptionResult,
        prev_object_result: DynamicObjectWithPerceptionResult,
    ) -> bool:
        """Check whether current and previous object result have switched ID.

        NOTE:
            There is a case the label is not same inspite of the same ID is given.
            GT ID is unique between the different labels.

        Args:
            cur_object_result (DynamicObjectWithPerceptionResult): An object result at current frame.
            prev_object_result (DynamicObjectWithPerceptionResult): An object result at previous frame.

        Returns:
            bool: Return True if ID switched.
        """
        if (
            cur_object_result.ground_truth_object is None
            or prev_object_result.ground_truth_object is None
        ):
            return False

        has_same_estimated_id: bool = (
            cur_object_result.estimated_object.uuid == prev_object_result.estimated_object.uuid
        )
        has_same_estimated_label: bool = (
            cur_object_result.estimated_object.semantic_label
            == prev_object_result.estimated_object.semantic_label
        )
        has_same_ground_truth_id: bool = (
            cur_object_result.ground_truth_object.uuid
            == prev_object_result.ground_truth_object.uuid
        )
        if bool(has_same_estimated_id * has_same_estimated_label):
            return not has_same_ground_truth_id
        elif has_same_ground_truth_id:
            return not (has_same_estimated_id * has_same_estimated_label)

        return False

    @staticmethod
    def _is_same_result(
        cur_object_result: DynamicObjectWithPerceptionResult,
        prev_object_result: DynamicObjectWithPerceptionResult,
    ) -> bool:
        """Check whether current and previous object result have same matching pair.

        Args:
            cur_object_result (DynamicObjectWithPerceptionResult): An object result at current frame.
            prev_object_result (DynamicObjectWithPerceptionResult): An object result at previous frame.

        Returns:
            bool: Return True if both estimated and GT ID are same.
        """
        if (
            cur_object_result.ground_truth_object is None
            or prev_object_result.ground_truth_object is None
        ):
            return False

        has_same_estimated_id: bool = (
            cur_object_result.estimated_object.uuid == prev_object_result.estimated_object.uuid
        )
        has_same_estimated_label: bool = (
            cur_object_result.estimated_object.semantic_label
            == prev_object_result.estimated_object.semantic_label
        )
        has_same_ground_truth_id: bool = (
            cur_object_result.ground_truth_object.uuid
            == prev_object_result.ground_truth_object.uuid
        )

        return bool(has_same_estimated_id * has_same_estimated_label * has_same_ground_truth_id)
