from typing import Any
from typing import List
from typing import Optional

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetrics
from awml_evaluation.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from awml_evaluation.evaluation.metrics.tracking._metrics_base import _TrackingMetricsBase
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class HOTA(_TrackingMetricsBase):
    """==== TODO ====
    HOTA metrics class that has HOTA, LocA, DetA and AssA as sub metrics.

    Attributes:
        self.target_labels (List[AutowareLabel]): The list of target label.
        self.max_x_position_list (List[float]): The list of max x position threshold.
        self.max_y_position_list (List[float]): The list of max y position threshold.
        self.matching_mode (MatchingMode): The target matching mode.
        self.metrics_field (Optional[List[str]]): The list of target metrics name. If not specified, set default supported metrics.
        self.ground_truth_objects_num (int): The number of ground truth.
        self.tp_metrics (TPMetrics): The way of calculating TP value.
        self.support_metrics (List[str]): The list of supported metrics name.
    """

    _support_metrics: List[str] = ["HOTA", "DetA", "AssA"]

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
        HOTA metrics.

        NOTE: objects_results, ground_truth_objects
            If evaluate 1-frame, index 0 is previous object results.
            If evaluate all frames, index 0 is empty list.

        Args:
            object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object results for each frames.
            ground_truth_objects (List[List[DynamicObject]]): The list of ground truth objects for each frames.
            target_labels (List[AutowareLabel]): The list of target labels.
            max_x_position_list (List[float]): The list of max x positions.
            max_y_position_list (List[float]): The list of max y positions.
            matching_mode: (MatchingMode): Matching mode class.
            tp_metrics (TPMetrics): The way of calculating TP value. Defaults to TPMetricsAP.
            metrics_field: List[str]: The list of target sub metrics.
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
        self.tpa: float = 0.0
        self.fpa: float = 0.0
        self.tp_matching_score: float = 0.0

        num_frame: int = len(object_results)
        for i in range(1, num_frame):
            cur_object_results, prev_object_results = self._filter_and_sort(
                cur_object_results=object_results[i],
                prev_object_results=object_results[i - 1],
            )

            # Comparing matching pair
            tp_t, fp_t, tpa_t, fpa_t = self._calculate_tp_fp(
                cur_object_results=cur_object_results,
                prev_object_results=prev_object_results,
            )
            self.tp += tp_t
            self.fp += fp_t
            self.tpa += tpa_t
            self.fpa += fpa_t

    def _calculate_tp_fp(
        self,
        cur_object_results: List[DynamicObjectWithPerceptionResult],
        prev_object_results: List[DynamicObjectWithPerceptionResult],
    ) -> Any:
        """Calculate TP/FP and TPA/FPA

        Args:
            cur_object_results
            prev_object_results (List[DynamicObjectWithPerceptionResult])
            prev_object_results (List[DynamicObjectWithPerceptionResult])
            matching_threshold (float)
        Returns:
            tp_list (List[float])
            fp_list (List[float])
            tpa_list (List[float])
            fpa_list (List[float])
        """
        pass
