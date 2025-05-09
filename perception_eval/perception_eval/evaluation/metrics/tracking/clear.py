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

from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from perception_eval.common.label import LabelType
from perception_eval.common.threshold import get_label_threshold
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetrics
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

from ._metrics_base import _TrackingMetricsBase


class CLEAR(_TrackingMetricsBase):
    """CLEAR metrics class that has MOTA and MOTP as sub metrics.

    In About objects_results and ground_truth_objects,
            When evaluate 1-frame, index 0 is previous object results.
            When evaluate all frames, index 0 is empty list.

    NOTE: MT, ML, PT is under construction.

    Attributes:
        target_labels (List[LabelType]): The list of target label.
        matching_mode (MatchingMode): The target matching mode.
        metrics_field (Optional[List[str]]): The list of target metrics name. If not specified, set default supported metrics.
        ground_truth_objects_num (int): The number of ground truth.
        support_metrics (List[str]): The list of supported metrics name. (["MOTA", "MOTP"])
        tp (float): The total value/number of TP.
        fp (float): The total value/number of FP.
        id_switch (int): The total number of ID switch.
        tp_matching_score (float): The total value of matching score in TP.
        mota (float): MOTA score.
        motp (float): MOTP score.
        results (OrderedDict[str, Any]): The dict to keep scores.

    Args:
        object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object results for each frames.
        num_ground_truth (int): The number of ground truth.
        target_labels (List[LabelType]): The list of target labels.
        matching_mode (MatchingMode): Matching mode class.
        tp_metrics (TPMetrics): The way of calculating TP value. Defaults to TPMetricsAP.
        metrics_field: List[str]: The list of target sub metrics. Defaults to None.
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
        # TODO(vivid): change the naming for 'object_results'
        # Should include the naming include 'frame'.
        object_results: List[List[DynamicObjectWithPerceptionResult]],
        num_ground_truth: int,
        target_labels: List[LabelType],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
        tp_metrics: TPMetrics = TPMetricsAp(),
        metrics_field: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            num_ground_truth=num_ground_truth,
            target_labels=target_labels,
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

        for i, cur_object_results in enumerate(object_results[1:], 1):
            prev_object_results: List[DynamicObjectWithPerceptionResult] = object_results[i - 1]
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
        """Calculate MOTA and MOTP.

        NOTE:
            if the number of GT is 0, MOTA returns inf and if the TP score is 0, MOTP returns inf.

        Returns:
            mota (float): MOTA score.
            motp (float): MOTP score.
        """
        if self.num_ground_truth == 0:
            mota: float = float("inf")
        else:
            mota: float = (self.tp - self.fp - self.id_switch) / self.num_ground_truth

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
        """Calculate matching compared with previous object results.

        Args:
            cur_object_results (List[List[DynamicObjectWithPerceptionResult]]): Object results list at current frame.
            prev_object_results (List[List[DynamicObjectWithPerceptionResult]]): Object results list at previous frame.

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

        for cur_obj_result in cur_object_results:
            # Assign previous results if same matching pair has
            # to get label threshold, use GT label basically,
            # but use EST label if GT is FP validation
            semantic_label = (
                cur_obj_result.estimated_object.semantic_label
                if (
                    cur_obj_result.ground_truth_object is None
                    or cur_obj_result.ground_truth_object.semantic_label.is_fp()
                )
                else cur_obj_result.ground_truth_object.semantic_label
            )
            matching_threshold_ = get_label_threshold(
                semantic_label=semantic_label,
                target_labels=self.target_labels,
                threshold_list=self.matching_threshold_list,
            )
            if matching_threshold_ is None:
                continue

            is_same_match: bool = False
            is_id_switched: bool = False
            for prev_obj_result in prev_object_results:
                is_tp_prev: bool = prev_obj_result.is_result_correct(
                    self.matching_mode,
                    matching_threshold_,
                )
                # is_tp_cur: bool = cur_obj_result.is_result_correct(
                #     self.matching_mode,
                #     matching_threshold_,
                # )
                # if is_tp_prev is False or is_tp_cur is False:
                if not is_tp_prev:
                    continue

                is_id_switched = self._is_id_switched(cur_obj_result, prev_obj_result)
                if is_id_switched:
                    break

                is_same_match: bool = self._is_same_match(cur_obj_result, prev_obj_result)
                if is_same_match:
                    # NOTE: If current/previous has same matching pair and both current/previous is TP,
                    #       previous TP score is assigned.
                    tp += self.tp_metrics.get_value(prev_obj_result)
                    tp_matching_score += prev_obj_result.get_matching(self.matching_mode).value
                    break
            if is_same_match:
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
        """Check whether current and previous object result have switched ID for TP pairs.

        NOTE:
            There is a case the label is not same in spite of the same ID is given.
            GT ID is unique between the different labels.

        Args:
            cur_object_result (DynamicObjectWithPerceptionResult): Object result at current frame.
            prev_object_result (DynamicObjectWithPerceptionResult): Object result at previous frame.

        Returns:
            bool: Return True if ID is switched.
        """
        # current GT = None -> FP
        if cur_object_result.ground_truth_object is None or prev_object_result.ground_truth_object is None:
            return False

        # 1. Check whether current/previous estimated objects has same ID.
        # NOTE: There is a case current/previous estimated objects has same ID,
        #       but different label(Checked by 2.)
        has_same_estimated_id: bool = (
            cur_object_result.estimated_object.uuid == prev_object_result.estimated_object.uuid
        )
        # 2. Check whether current/previous estimated objects has same label.
        has_same_estimated_label: bool = (
            cur_object_result.estimated_object.semantic_label == prev_object_result.estimated_object.semantic_label
        )
        # 3. Check whether current/previous GT has same ID.
        # NOTE: There is no case GT has same ID, but different label. (Like 1.)
        has_same_ground_truth_id: bool = (
            cur_object_result.ground_truth_object.uuid == prev_object_result.ground_truth_object.uuid
        )
        if bool(has_same_estimated_id * has_same_estimated_label):
            return not has_same_ground_truth_id
        elif has_same_ground_truth_id:
            return not (has_same_estimated_id * has_same_estimated_label)

        return False

    @staticmethod
    def _is_same_match(
        cur_object_result: DynamicObjectWithPerceptionResult,
        prev_object_result: DynamicObjectWithPerceptionResult,
    ) -> bool:
        """Check whether current and previous object result have same matching pair.
        When previous or current GT is None(=FP), return False regardless the ID of estimated.

        Args:
            cur_object_result (DynamicObjectWithPerceptionResult): Object result at current frame.
            prev_object_result (DynamicObjectWithPerceptionResult):Object result at previous frame.

        Returns:
            bool: Return True if both estimated and GT ID are same.
        """
        if cur_object_result.ground_truth_object is None or prev_object_result.ground_truth_object is None:
            return False

        has_same_estimated_id: bool = (
            cur_object_result.estimated_object.uuid == prev_object_result.estimated_object.uuid
        )
        has_same_estimated_label: bool = (
            cur_object_result.estimated_object.semantic_label == prev_object_result.estimated_object.semantic_label
        )
        has_same_ground_truth_id: bool = (
            cur_object_result.ground_truth_object.uuid == prev_object_result.ground_truth_object.uuid
        )

        return bool(has_same_estimated_id * has_same_estimated_label * has_same_ground_truth_id)
