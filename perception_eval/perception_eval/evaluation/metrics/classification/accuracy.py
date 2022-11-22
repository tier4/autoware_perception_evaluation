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
from typing import Tuple

from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class ClassificationAccuracy:
    """[summary]
    Class to calculate classification accuracy.
    """

    def __init__(
        self,
        object_results,
        num_ground_truth,
        target_labels,
    ) -> None:
        """
        Args:
        """
        self.num_ground_truth: int = num_ground_truth
        self.target_labels = target_labels
        if len(object_results) == 0 or not isinstance(object_results[0], list):
            all_object_results = object_results
        else:
            for obj_results in object_results:
                all_object_results += obj_results
        self.accuracy: float = self.get_accuracy_score(all_object_results)

        self.tp: float = 0.0
        self.fp: float = 0.0
        self.id_switch: int = 0
        self.tp_matching_score: float = 0.0
        self.objects_results_num: int = 0

        for i, cur_object_results in enumerate(object_results[1:], 1):
            prev_object_results: List[DynamicObjectWithPerceptionResult] = object_results[i - 1]
            self.objects_results_num += len(cur_object_results)

            # Calculate TP/FP/IDSwitch and total matching score in TP at frame t
            tp_t, fp_t, id_switch_t = self.get_tsca_score(
                cur_object_results=cur_object_results,
                prev_object_results=prev_object_results,
            )
            self.tp += tp_t
            self.fp += fp_t
            self.id_switch += id_switch_t
        tsca: float = (
            float("inf")
            if self.num_ground_truth == 0
            else (self.tp - self.fp - self.id_switch) / self.num_ground_truth
        )
        self.tsca: float = max(0.0, tsca)

    def get_accuracy_score(
        self,
        object_results: List[DynamicObjectWithPerceptionResult],
    ) -> float:
        """[summary]
        Calculate accuracy score.

        Args:
            object_results (List[DynamicObjectWithPerceptionResult])

        Returns:
            float: Accuracy score.
        """
        sum_acc: int = 0
        for obj_result in object_results:
            if obj_result.is_label_correct:
                sum_acc += 1
        return float("inf") if self.num_ground_truth == 0 else sum_acc / self.num_ground_truth

    def get_tsca_score(
        self,
        cur_object_results: List[DynamicObjectWithPerceptionResult],
        prev_object_results: List[DynamicObjectWithPerceptionResult],
    ) -> Tuple[float, float, int]:
        """[summary]
        Calculate Time-Series Classification Accuracy score.

        Args:
            cur_object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object results at current frame.
            prev_object_results (List[List[DynamicObjectWithPerceptionResult]]): The list of object results at previous frame.

        Returns:
            tp (float): Total value of TP. If matching is True, num_tp += 1.0.
            fp (float): Total value of FP. If matching is False, num_fp += 1.0.
            num_id_switch (int): Total number of ID switch. If matching is switched compared with previous result, num_id_switch += 1.
        """
        tp: float = 0.0
        fp: float = 0.0
        num_id_switch: int = 0

        for cur_obj_result in cur_object_results:
            # Assign previous results if same matching pair has
            is_same_match: bool = False
            is_id_switched: bool = False
            for prev_obj_result in prev_object_results:
                if not prev_obj_result.is_label_correct:
                    continue
                is_id_switched = self._is_id_switched(cur_obj_result, prev_obj_result)
                if is_id_switched:
                    break
                is_same_match: bool = self._is_same_match(cur_obj_result, prev_obj_result)
                if is_same_match:
                    # NOTE: If current/previous has same matching pair and both current/previous is TP,
                    #       previous TP score is assigned.
                    tp += 1.0
                    break
            if is_same_match:
                continue
            # If there is no same matching pair with previous results
            if cur_obj_result.is_label_correct:
                tp += 1.0
                if is_id_switched:
                    num_id_switch += 1
            else:
                fp += 1.0

        return tp, fp, num_id_switch

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
            cur_object_result (DynamicObjectWithPerceptionResult): An object result at current frame.
            prev_object_result (DynamicObjectWithPerceptionResult): An object result at previous frame.

        Returns:
            bool: Return True if ID switched.
        """
        # current GT = None -> FP
        if (
            cur_object_result.ground_truth_object is None
            or prev_object_result.ground_truth_object is None
        ):
            return False

        # 1. Check whether current/previous estimated objects has same ID.
        # NOTE: There is a case current/previous estimated objects has same ID,
        #       but different label(Checked by 2.)
        has_same_estimated_id: bool = (
            cur_object_result.estimated_object.uuid == prev_object_result.estimated_object.uuid
        )
        # 2. Check whether current/previous estimated objects has same label.
        has_same_estimated_label: bool = (
            cur_object_result.estimated_object.semantic_label
            == prev_object_result.estimated_object.semantic_label
        )
        # 3. Check whether current/previous GT has same ID.
        # NOTE: There is no case GT has same ID, but different label. (Like 1.)
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
    def _is_same_match(
        cur_object_result: DynamicObjectWithPerceptionResult,
        prev_object_result: DynamicObjectWithPerceptionResult,
    ) -> bool:
        """Check whether current and previous object result have same matching pair.
        When previous or current GT is None(=FP), return False regardless the ID of estimated.

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
