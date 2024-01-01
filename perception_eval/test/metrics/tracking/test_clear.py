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

from __future__ import annotations

from math import isclose
from test.util.dummy_object import make_dummy_data
from test.util.object_diff import DiffTranslation
from typing import List
from typing import Optional
from typing import Tuple
import unittest

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.matching import MatchingMode
from perception_eval.matching.objects_filter import filter_objects
from perception_eval.metrics.tracking.clear import CLEAR
from perception_eval.object import DynamicObject
from perception_eval.result import get_object_results
from perception_eval.result import PerceptionObjectResult
from perception_eval.util import get_objects_with_difference


class AnswerCLEAR:
    """Answer class for CLEAR to compare result.

    Attributes:
        self.num_ground_truth (int)
        self.tp (float)
        self.fp (float)
        self.id_switch (int)
        self.tp_matching_score (float)
        self.mota (float)
        self.motp (float)
    """

    def __init__(
        self,
        num_ground_truth: int,
        tp: float,
        fp: float,
        id_switch: int,
        tp_matching_score: float,
        mota: float,
        motp: float,
    ):
        """[summary]
        Args:
            num_ground_truth (int)
            tp (float)
            fp (float)
            id_switch (float)
            tp_matching_score (float)
            mota (float)
            motp (float)
        """
        self.num_ground_truth: int = num_ground_truth
        self.tp: float = tp
        self.fp: float = fp
        self.id_switch: int = id_switch
        self.tp_matching_score: float = tp_matching_score
        self.mota: float = mota
        self.motp: float = motp

    @classmethod
    def from_clear(cls, clear: CLEAR) -> AnswerCLEAR:
        """Generate AnswerCLEAR from CLEAR.

        Args:
            clear (CLEAR)

        Returns:
            AnswerCLEAR
        """
        return AnswerCLEAR(
            clear.num_ground_truth,
            clear.tp,
            clear.fp,
            clear.id_switch,
            clear.tp_matching_score,
            clear.mota,
            clear.motp,
        )

    def __eq__(self, other: AnswerCLEAR) -> bool:
        return (
            self.num_ground_truth == other.num_ground_truth
            and isclose(self.tp, other.tp)
            and isclose(self.fp, other.fp)
            and self.id_switch == other.id_switch
            and isclose(self.tp_matching_score, other.tp_matching_score)
            and isclose(self.mota, other.mota)
            and isclose(self.motp, other.motp)
        )

    def __str__(self) -> str:
        str_: str = "\n("
        str_ += f"num_ground_truth: {self.num_ground_truth}, "
        str_ += f"tp: {self.tp}, "
        str_ += f"fp: {self.fp}, "
        str_ += f"id switch: {self.id_switch}, "
        str_ += f"tp_matching_score: {self.tp_matching_score}, "
        str_ += f"mota: {self.mota}, "
        str_ += f"motp: {self.motp}"
        str_ += ")"
        return str_


class TestCLEAR(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_unique_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data(use_unique_id=True)
        self.dummy_estimated_objects, _ = make_dummy_data(use_unique_id=False)

        self.evaluation_task: EvaluationTask = EvaluationTask.TRACKING
        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]
        self.max_x_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]
        self.max_y_position_list: List[float] = [100.0, 100.0, 100.0, 100.0]

    def test_calculate_tp_fp(self):
        """[summary]
        Test calculating TP/FP/ID switch.

        test patterns:
            Check if TP/FP, ID switch, total matching score in TP, MOTA and MOTP are calculated correctly.
            If prev_diff_trans is None, it means there is no previous results. Then, ID switch must be 0.
            - matching mode     : Center distance
            - matching threshold: 0.5
            - tp metrics        : AP(=1.0)
            NOTE:
                - The flag must be same whether use unique ID or not.
                - Estimated object is only matched with GT that has same label.
                - The estimations & GTs are following (number represents the index)
                    Estimation = 3
                        (0): CAR, (1): BICYCLE, (2): CAR
                    GT = 4
                        (0): CAR, (1): BICYCLE, (2): PEDESTRIAN, (3): MOTORBIKE
        """
        # patterns: (prev_diff_trans, cur_diff_trans, target_label, use_unique_id, ans_clear)
        patterns: List[Tuple[Optional[DiffTranslation], DiffTranslation, AutowareLabel, AnswerCLEAR]] = [
            # ========== Test unique ID association ==========
            # (1). Est: 2, GT: 1
            # -> previous   : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> current    : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> TP score=0.0, IDsw=0, TP matching score=0.0, MOTA=0.0, MOTP=0.0
            (
                # prev: (trans est, trans gt)
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                # cur: (trans est, trans gt)
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                # target filtered label
                AutowareLabel.CAR,
                # whether use unique ID
                True,
                # num_gt<int>, tp<float>, fp<float>, id_switch<int>, tp_matching_score<float>, mota<float>, motp<float>
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0.0, 0.0),
            ),
            # (2). Est: 2, GT: 1
            # -> previous   : TP=0.0, FP=2.0(Est[0], Est[2])
            # -> current    : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> TP score=0.0, IDsw=0, TP matching score=0.0, MOTA=0.0, MOTP=0.0
            # Although Est[0] and GT[0] is assigned as match, the matching score(=1.0) is worse than threshold(=0.5)
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0.0, 0.0),
            ),
            # (3). Est: 2, GT: 1
            # -> previous   : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> current    : TP=0.0, FP=2.0(Est[0], Est[2])
            # -> TP score=1.0, IDsw=1, matching score=0.0, MOTA=0.0, MOTP=0.0
            # When current/previous has same match and previous result is TP, previous TP is assigned even though current is TP.
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0.0, 0.0),
            ),
            # (4). Est: 2, GT: 1
            # -> previous   : TP=0.0, FP=2.0(Est[0], Est[2])
            # -> current    : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> TP score=1.0, IDsw=1, matching score=0.2, MOTA=0.0, MOTP=0.2
            # In case of current/previous has same match, current IDsw is 0 if TP even though previous is FP.
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.2, 0.0, 0.2),
            ),
            # (5). Est: 2, GT: 0
            # -> previous   : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> current    : TP=0.0, FP=2.0(Est[0], Est[2])
            # -> TP score=0.0, IDsw=1, TP matching score=0.0, MOTA=inf, MOTP=0.0
            # If there is no GT, MOTA get inf. Also, if there is no TP, MOTP get inf.
            # In this case, GT is filtered by max_x_position(=100.0).
            # When current/previous has same match and previous result is TP, previous TP is assigned even though current is TP.
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (101.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(0, 0.0, 2.0, 0, 0.0, float("inf"), float("inf")),
            ),
            # (6). Est: 2, GT: 1
            # -> previous   : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> current    : TP=0.0, FP=2.0(Est[0], Est[2])
            # -> TP score=0.0, IDsw=1, TP matching score=0.0, MOTA=inf, MOTP=0.0
            # If there is no TP, MOTP get inf.
            (
                DiffTranslation((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(1, 0.0, 2.0, 0, 0.0, 0.0, float("inf")),
            ),
            # --- Test there is no previous result ---
            # (7). Est: 2, GT: 1
            # -> previous   : No result
            # -> current    : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> TP score=0.0, IDsw=0, matching score=0.0, MOTA=0.0, MOTP=0.0
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0, 0.0),
            ),
            # (8). Est: 2, GT: 1
            # -> previous   : No result
            # -> current    : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> TP score=1.0, IDsw=0, matching score=0.2, MOTA=0.0, MOTP=0.2
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.2, 0.0, 0.2),
            ),
            # (9). Est: 1, GT: 1
            # -> previous   : No result
            # -> current    : TP=1.0(Est[1], GT[1]), FP=0.0
            # -> TP score=1.0, IDsw=0, matching score=0.2, MOTA=0.0, MOTP=0.2
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.BICYCLE,
                True,
                AnswerCLEAR(1, 1.0, 0.0, 0, 0.2, 1.0, 0.2),
            ),
            # ========== Test non-unique ID association ==========
            #   -> The result must be same, whether use unique ID or not.
            # (10). Est: 2, GT: 1    = Case(1)
            # -> previous   : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> current    : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> TP score=0.0, IDsw=0, TP matching score=0.0, MOTA=0.0, MOTP=0.0
            (
                # prev: (trans est, trans gt)
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                # cur: (trans est, trans gt)
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                # target filtered label
                AutowareLabel.CAR,
                # whether use unique ID
                False,
                # num_gt<int>, tp<float>, fp<float>, id_switch<int>, tp_matching_score<float>, mota<float>, motp<float>
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0.0, 0.0),
            ),
            # (11). Est: 2, GT: 1    = Case(2)
            # -> previous   : TP=0.0, FP=2.0(Est[0], Est[2])
            # -> current    : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> TP score=0.0, IDsw=0, TP matching score=0.0, MOTA=0.0, MOTP=0.0
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                False,
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0.0, 0.0),
            ),
            # (12). Est: 2, GT: 1   = Case(3)
            # -> previous   : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> current    : TP=0.0, FP=2.0(Est[0], Est[2])
            # -> TP score=1.0, IDsw=1, matching score=0.0, MOTA=0.0, MOTP=0.0
            # When current/previous has same match and previous result is TP, previous TP is assigned.
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
                AutowareLabel.CAR,
                False,
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0.0, 0.0),
            ),
            # (13). Est: 2, GT: 1     = Case(4)
            # -> previous   : TP=0.0, FP=2.0(Est[0], Est[2])
            # -> current    : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> TP score=1.0, IDsw=1, matching score=0.2, MOTA=0.0, MOTP=0.2
            # When current/previous has same match, if current is TP IDsw is 0, though previous is FP.
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.CAR,
                False,
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.2, 0.0, 0.2),
            ),
            # (14). Est: 2, GT: 0   = Case(5)
            # -> previous   : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> current    : TP=0.0, FP=2.0(Est[0], Est[2])
            # -> TP score=0.0, IDsw=1, TP matching score=0.0, MOTA=inf, MOTP=0.0
            # If there is no GT, MOTA get inf. Also, if there is no TP, MOTP get inf.
            # In this case, GT is filtered by max_x_position(=100.0).
            # When current/previous has same match and previous result is TP, previous TP is assigned even though current is FP.
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (101.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(0, 0.0, 2.0, 0, 0.0, float("inf"), float("inf")),
            ),
            # (15). Est: 2, GT: 1   = Case(6)
            # -> previous   : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> current    : TP=0.0, FP=2.0(Est[0], Est[2])
            # -> TP score=0.0, IDsw=1, TP matching score=0.0, MOTA=inf, MOTP=0.0
            # If there is no TP, MOTP get inf.
            (
                DiffTranslation((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                False,
                AnswerCLEAR(1, 0.0, 2.0, 0, 0.0, 0.0, float("inf")),
            ),
            # --- Test there is no previous result ---
            # (16). Est: 2, GT: 1   = Case(5)
            # -> previous   : No result
            # -> current    : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> TP score=0.0, IDsw=0, matching score=0.0, MOTA=0.0, MOTP=0.0
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                False,
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0, 0.0),
            ),
            # (17). Est: 2, GT: 1   = Case(6)
            # -> previous   : No result
            # -> current    : TP=1.0(Est[0], GT[0]), FP=1.0(Est[2])
            # -> TP score=1.0, IDsw=0, matching score=0.2, MOTA=0.0, MOTP=0.2
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.CAR,
                False,
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.2, 0.0, 0.2),
            ),
            # (18). Est: 1, GT: 1   = Case(7)
            # -> previous   : No result
            # -> current    : TP=1.0(Est[1], GT[1]), FP=0.0
            # -> TP score=1.0, IDsw=0, matching score=0.2, MOTA=0.0, MOTP=0.2
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.BICYCLE,
                False,
                AnswerCLEAR(1, 1.0, 0.0, 0, 0.2, 1.0, 0.2),
            ),
        ]
        for n, (
            prev_diff_trans,
            cur_diff_trans,
            target_label,
            use_unique_id,
            ans_clear,
        ) in enumerate(patterns):
            with self.subTest(f"Test calculate TP/FP: {n + 1}"):
                # Previous estimated objects
                prev_object_results: Optional[List[PerceptionObjectResult]] = []

                if prev_diff_trans is not None:
                    # Translate previous estimated objects
                    prev_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                        ground_truth_objects=self.dummy_unique_estimated_objects
                        if use_unique_id
                        else self.dummy_estimated_objects,
                        diff_distance=prev_diff_trans.diff_estimated,
                        diff_yaw=0.0,
                    )
                    prev_ground_truth_objects = get_objects_with_difference(
                        ground_truth_objects=self.dummy_ground_truth_objects,
                        diff_distance=prev_diff_trans.diff_ground_truth,
                        diff_yaw=0.0,
                    )

                    # Filter previous objects
                    prev_estimated_objects = filter_objects(
                        objects=prev_estimated_objects,
                        is_gt=False,
                        target_labels=[target_label],
                        max_x_position_list=self.max_x_position_list,
                        max_y_position_list=self.max_y_position_list,
                    )
                    prev_ground_truth_objects = filter_objects(
                        objects=prev_ground_truth_objects,
                        is_gt=True,
                        target_labels=[target_label],
                        max_x_position_list=self.max_x_position_list,
                        max_y_position_list=self.max_y_position_list,
                    )

                    # Get previous object results
                    prev_object_results = get_object_results(
                        evaluation_task=self.evaluation_task,
                        estimated_objects=prev_estimated_objects,
                        ground_truth_objects=prev_ground_truth_objects,
                    )
                else:
                    prev_object_results = []

                # Translate current objects
                cur_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_unique_estimated_objects
                    if use_unique_id
                    else self.dummy_estimated_objects,
                    diff_distance=cur_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                cur_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=cur_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )

                # Filter current objects
                cur_estimated_objects = filter_objects(
                    objects=cur_estimated_objects,
                    is_gt=False,
                    target_labels=[target_label],
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                cur_ground_truth_objects = filter_objects(
                    objects=cur_ground_truth_objects,
                    is_gt=True,
                    target_labels=[target_label],
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )

                # Current object results
                cur_object_results: List[PerceptionObjectResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )

                num_ground_truth: int = len(cur_ground_truth_objects)

                clear_: CLEAR = CLEAR(
                    object_results=[prev_object_results, cur_object_results],
                    num_ground_truth=num_ground_truth,
                    target_labels=[target_label],
                    matching_mode=MatchingMode.CENTERDISTANCE,
                    matching_threshold_list=[0.5],
                )
                out_clear: AnswerCLEAR = AnswerCLEAR.from_clear(clear_)
                self.assertEqual(
                    out_clear,
                    ans_clear,
                    f"\nout_clear = {str(out_clear)},\nans_clear = {str(ans_clear)}",
                )

    def test_is_id_switched(self):
        """[summary]
        Test the method to check if ID is switched between previous and current TP results.

        test patterns:
            Check if ID was switched between each i-th current object result and j-th previous object.
            ``ans_flags`` represents the answer flag for three object results per one previous result.
            NOTE:
                - The flag must be same whether use unique ID or not.
                - Estimated object is only matched with GT that has same label.
                - The estimations & GTs are following (number represents the index)
                    Estimation = 3
                        (0): CAR, (1): BICYCLE, (2): CAR
                    GT = 4
                        (0): CAR, (1): BICYCLE, (2): PEDESTRIAN, (3): MOTORBIKE
        """
        # patterns: (prev_diff_trans, cur_diff_trans, use_unique_id, ans_flags)
        patterns: List[Tuple[DiffTranslation, DiffTranslation, bool, Tuple[bool]]] = [
            # ========== Test unique ID association ==========
            # (1)
            # -> previous   : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                True,
                # If ID was switched for cur[i] and prev[j]
                ((False, False, False), (False, False, False), (False, False, False)),
            ),
            # (2)
            # -> previous   : (Est[2], GT[0]), (Est[1], GT[1]), (Est[0], None)
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # ((False, False, False), (False, False, False), (False, False, False))
            # The IDsw only is counted for TP pairs both previous and current.
            # If previous result is TP though current result is FP, IDsw is not counted
            (
                DiffTranslation((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                True,
                ((True, False, False), (False, False, False), (False, False, False)),
            ),
            # --- Test there is no previous result ---
            # (3)
            # -> previous   : (Est[0], None), (Est[1], None), (Est[2], None)
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # The IDsw only is counted for TP pairs both previous and current.
            # If current result is TP though previous result is FP, IDsw is not counted
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                True,
                # If ID was switched for cur[i] and prev[j]
                ((False, False, False), (False, False, False), (False, False, False)),
            ),
            # ========== Test non-unique ID association ==========
            # (4)   = Case(1)
            # Est: 3, GT: 4
            # -> previous   : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                False,
                ((False, False, False), (False, False, False), (False, False, False)),
            ),
            # (5)   = Case(2)
            # -> previous   : (Est[2], GT[0]), (Est[1], GT[1]), (Est[0], None)
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # ((False, False, False), (False, False, False), (False, False, False))
            (
                DiffTranslation((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                False,
                ((True, False, False), (False, False, False), (False, False, False)),
            ),
            # --- Test there is no previous result ---
            # (6)   = Case(3)
            # -> previous   : (Est[0], None), (Est[1], None), (Est[2], None)
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                False,
                ((False, False, False), (False, False, False), (False, False, False)),
            ),
        ]
        for n, (prev_diff_trans, cur_diff_trans, use_unique_id, ans_flags) in enumerate(patterns):
            with self.subTest(f"Test is ID switched: {n + 1}"):
                if prev_diff_trans is not None:
                    # Translate previous objects
                    prev_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                        ground_truth_objects=self.dummy_unique_estimated_objects
                        if use_unique_id
                        else self.dummy_estimated_objects,
                        diff_distance=prev_diff_trans.diff_estimated,
                        diff_yaw=0.0,
                    )
                    prev_ground_truth_objects = get_objects_with_difference(
                        ground_truth_objects=self.dummy_ground_truth_objects,
                        diff_distance=prev_diff_trans.diff_ground_truth,
                        diff_yaw=0.0,
                    )

                    # Filter previous objects
                    prev_estimated_objects = filter_objects(
                        objects=prev_estimated_objects,
                        is_gt=False,
                        target_labels=self.target_labels,
                        max_x_position_list=self.max_x_position_list,
                        max_y_position_list=self.max_y_position_list,
                    )
                    prev_ground_truth_objects = filter_objects(
                        objects=prev_ground_truth_objects,
                        is_gt=True,
                        target_labels=self.target_labels,
                        max_x_position_list=self.max_x_position_list,
                        max_y_position_list=self.max_y_position_list,
                    )

                    # Previous object results
                    prev_object_results = get_object_results(
                        evaluation_task=self.evaluation_task,
                        estimated_objects=prev_estimated_objects,
                        ground_truth_objects=prev_ground_truth_objects,
                    )
                else:
                    prev_object_results = []

                # Translate current objects
                cur_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_unique_estimated_objects
                    if use_unique_id
                    else self.dummy_estimated_objects,
                    diff_distance=cur_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                cur_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=cur_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )

                # Filter current objects
                cur_estimated_objects = filter_objects(
                    objects=cur_estimated_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                cur_ground_truth_objects = filter_objects(
                    objects=cur_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )

                # Current object results
                cur_object_results: List[PerceptionObjectResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )
                for i, cur_obj_result in enumerate(cur_object_results):
                    for j, prev_obj_result in enumerate(prev_object_results):
                        flag: bool = CLEAR._is_id_switched(
                            cur_object_result=cur_obj_result,
                            prev_object_result=prev_obj_result,
                        )
                        self.assertEqual(flag, ans_flags[i][j], f"{flag} != ans_flags[{i}][{j}]]")

    def test_is_same_match(self):
        """[summary]
        Test check if previous and current are same result.

        test patterns:
            Check if the object result is same one between each i-th current object result and j-th previous object.
            ``ans_flags`` represents the answer flag for three object results per one previous result.
            NOTE:
                - The flag must be same whether use unique ID or not.
                - Estimated object is only matched with GT that has same label.
                - The estimations & GTs are following (number represents the index)
                    Estimation
                        (0): CAR, (1): BICYCLE, (2): CAR
                    GT
                        (0): CAR, (1): BICYCLE, (2): PEDESTRIAN, (3): MOTORBIKE
        """
        # patterns: (prev_diff_trans, cur_diff_trans, use_unique_id, ans_flags)
        patterns: List[Tuple[DiffTranslation, DiffTranslation, Tuple[bool]]] = [
            # ========== Test unique ID association ==========
            # (1)
            # -> previous   : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # When previous or current GT is None(=FP), return False regardless the ID of estimated.
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                True,
                # If ID was switched for cur[i] and prev[j]
                ((True, False, False), (False, True, False), (False, False, False)),
            ),
            # TODO
            # (2)
            # [Confidence priority]
            # -> previous   : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # ((False, False, False), (False, False, False), (False, False, False))
            # [Matching score priority]
            # -> previous   : (Est[0], None), (Est[1], GT[1]), (Est[2], GT[0])
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # ((False, False, False), (False, False, False), (False, False, False)
            # When previous or current GT is None(=FP), return False regardless the ID of estimated.
            (
                DiffTranslation((0.0, 0.0, 0.0), (-2.5, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                True,
                ((False, False, False), (False, False, False), (False, False, False)),
            ),
            # --- Test there is no previous result ---
            # (3)
            # -> previous   : No matching result
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # If previous estimation is FP, IDsw is not counted
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                True,
                # If ID was switched for cur[i] and prev[j]
                ((False, False, False), (False, False, False), (False, False, False)),
            ),
            # ========== Test non-unique ID association ==========
            # (4)   = Case(1)
            # Est: 3, GT: 4
            # -> previous   : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                False,
                ((True, False, False), (False, True, False), (False, False, False)),
            ),
            # TODO
            # (5)   = Case(2)
            # [Confidence priority]
            # -> previous   : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # ((False, False, False), (False, False, False), (False, False, False))
            # [Matching score priority]
            # -> previous   : (Est[0], None), (Est[1], GT[1]), (Est[2], GT[0])
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # ((False, False, False), (False, False, False), (False, False, False))
            # When previous or current GT is None(=FP), return False regardless the ID of estimated.
            (
                DiffTranslation((0.0, 0.0, 0.0), (-2.5, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                False,
                ((False, False, False), (False, False, False), (False, False, False)),
            ),
            # --- Test there is no previous result ---
            # (6)   = Case(3)
            # -> previous   : No matching result
            # -> current    : (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                False,
                ((False, False, False), (False, False, False), (False, False, False)),
            ),
        ]
        for n, (prev_diff_trans, cur_diff_trans, use_unique_id, ans_flags) in enumerate(patterns):
            with self.subTest(f"Test is same result: {n + 1}"):
                if prev_diff_trans is not None:
                    # Translate previous objects
                    prev_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                        ground_truth_objects=self.dummy_unique_estimated_objects
                        if use_unique_id
                        else self.dummy_estimated_objects,
                        diff_distance=prev_diff_trans.diff_estimated,
                        diff_yaw=0.0,
                    )
                    prev_ground_truth_objects = get_objects_with_difference(
                        ground_truth_objects=self.dummy_ground_truth_objects,
                        diff_distance=prev_diff_trans.diff_ground_truth,
                        diff_yaw=0.0,
                    )

                    # Filter previous objects
                    prev_estimated_objects = filter_objects(
                        objects=prev_estimated_objects,
                        is_gt=False,
                        target_labels=self.target_labels,
                        max_x_position_list=self.max_x_position_list,
                        max_y_position_list=self.max_y_position_list,
                    )
                    prev_ground_truth_objects = filter_objects(
                        objects=prev_ground_truth_objects,
                        is_gt=True,
                        target_labels=self.target_labels,
                        max_x_position_list=self.max_x_position_list,
                        max_y_position_list=self.max_y_position_list,
                    )

                    # Previous object results
                    prev_object_results = get_object_results(
                        evaluation_task=self.evaluation_task,
                        estimated_objects=prev_estimated_objects,
                        ground_truth_objects=prev_ground_truth_objects,
                    )
                else:
                    prev_object_results = []

                # Translate current objects
                cur_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_unique_estimated_objects
                    if use_unique_id
                    else self.dummy_estimated_objects,
                    diff_distance=cur_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                cur_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=cur_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )

                # Filter current objects
                cur_estimated_objects = filter_objects(
                    objects=cur_estimated_objects,
                    is_gt=False,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )
                cur_ground_truth_objects = filter_objects(
                    objects=cur_ground_truth_objects,
                    is_gt=True,
                    target_labels=self.target_labels,
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
                )

                # Current object results
                cur_object_results: List[PerceptionObjectResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )
                for i, cur_obj_result in enumerate(cur_object_results):
                    for j, prev_obj_result in enumerate(prev_object_results):
                        flag: bool = CLEAR._is_same_match(
                            cur_object_result=cur_obj_result,
                            prev_object_result=prev_obj_result,
                        )
                    self.assertEqual(flag, ans_flags[i][j], f"{flag} != ans_flags[{i}][{j}]")
