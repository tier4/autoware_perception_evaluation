from __future__ import annotations

from math import isclose
from test.util.dummy_object import make_dummy_data
from test.util.object_diff import DiffTranslation
from typing import List
from typing import Optional
from typing import Tuple
import unittest

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.matching.object_matching import MatchingMode
from awml_evaluation.evaluation.metrics.tracking.clear import CLEAR
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.util.debug import get_objects_with_difference
import numpy as np


class AnswerCLEAR:
    """Answer class for CLEAR to compare result.

    Attributes:
        self.ground_truth_objects_num (int)
        self.tp (float)
        self.fp (float)
        self.id_switch (int)
        self.tp_matching_score (float)
        self.mota (float)
        self.motp (float)
    """

    def __init__(
        self,
        ground_truth_objects_num: int,
        tp: float,
        fp: float,
        id_switch: int,
        tp_matching_score: float,
        mota: float,
        motp: float,
    ):
        """[summary]
        Args:
            ground_truth_objects_num (int)
            tp (float)
            fp (float)
            id_switch (float)
            tp_matching_score (float)
            mota (float)
            motp (float)
        """
        self.ground_truth_objects_num: int = ground_truth_objects_num
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
            clear.ground_truth_objects_num,
            clear.tp,
            clear.fp,
            clear.id_switch,
            clear.tp_matching_score,
            clear.mota,
            clear.motp,
        )

    def __eq__(self, other: AnswerCLEAR) -> bool:
        return (
            self.ground_truth_objects_num == other.ground_truth_objects_num
            and isclose(self.tp, other.tp)
            and isclose(self.fp, other.fp)
            and self.id_switch == other.id_switch
            and isclose(self.tp_matching_score, other.tp_matching_score)
            and isclose(self.mota, other.mota)
            and isclose(self.motp, other.motp)
        )

    def __str__(self) -> str:
        str_: str = "\n("
        str_ += f"ground_truth_objects_num: {self.ground_truth_objects_num}, "
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
        self.dummy_unique_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data(
            use_unique_id=True
        )
        self.dummy_estimated_objects, _ = make_dummy_data(use_unique_id=False)

        self.max_x_position_list: List[float] = [100.0]
        self.max_y_position_list: List[float] = [100.0]

    def test_calculate_tp_fp(self):
        """[summary]
        Test calculating TP/FP/ID switch

        test patterns:
            Check if TP/FP, ID switch, total matching score in TP, MOTA and MOTP are calculated correctly.
            If prev_diff_trans is None, it means there is no previous results. Then, ID switch must be 0.
        """
        # patterns: (prev_diff_trans, cur_diff_trans, target_label, use_unique_id, ans_clear)
        patterns: List[
            Tuple[Optional[DiffTranslation], DiffTranslation, AutowareLabel, AnswerCLEAR]
        ] = [
            # === Test unique ID association ===
            (
                # prev: (trans est, trans gt)
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                # cur: (trans est, trans gt)
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                # num_gt<int>, tp<float>, fp<float>, id_switch<int>, tp_matching_score<float>, mota<float>, motp<float>
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0.0, 0.0),
            ),
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(1, 1.0, 1.0, 1, 0.0, 0.0, 0.0),
            ),
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(0, 1.0, 1.0, 0, 0.0, float("inf"), 0.0),
            ),
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(1, 1.0, 1.0, 0, 2.0, 0.0, 2.0),
            ),
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(1, 1.0, 1.0, 1, 0.2, 0.0, 0.2),
            ),
            # --- Test no previous case ---
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(0, 1.0, 1.0, 0, 0.2, float("inf"), 0.2),
            ),
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.BICYCLE,
                True,
                AnswerCLEAR(0, 1.0, 0.0, 0, 0.2, float("inf"), 0.2),
            ),
            # == Test non-unique ID association ===
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                True,
                AnswerCLEAR(1, 1.0, 1.0, 0, 0.0, 0.0, 0.0),
            ),
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                False,
                AnswerCLEAR(1, 1.0, 1.0, 1, 0.0, 0.0, 0.0),
            ),
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                AutowareLabel.CAR,
                False,
                AnswerCLEAR(0, 1.0, 1.0, 0, 0.0, float("inf"), 0.0),
            ),
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
                AutowareLabel.CAR,
                False,
                AnswerCLEAR(1, 1.0, 1.0, 0, 2.0, 0.0, 2.0),
            ),
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.CAR,
                False,
                AnswerCLEAR(1, 1.0, 1.0, 1, 0.2, 0.0, 0.2),
            ),
            # --- Test no previous case ---
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.CAR,
                False,
                AnswerCLEAR(0, 1.0, 1.0, 0, 0.2, float("inf"), 0.2),
            ),
            (
                None,
                DiffTranslation((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
                AutowareLabel.BICYCLE,
                False,
                AnswerCLEAR(0, 1.0, 0.0, 0, 0.2, float("inf"), 0.2),
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
                prev_object_results: List[DynamicObjectWithPerceptionResult] = []
                prev_ground_truth_objects: List[DynamicObject] = []

                if prev_diff_trans is not None:
                    prev_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                        ground_truth_objects=self.dummy_unique_estimated_objects
                        if use_unique_id
                        else self.dummy_estimated_objects,
                        diff_distance=prev_diff_trans.diff_estimated,
                        diff_yaw=0.0,
                    )
                    # Previous ground truth objects
                    prev_ground_truth_objects = get_objects_with_difference(
                        ground_truth_objects=self.dummy_ground_truth_objects,
                        diff_distance=prev_diff_trans.diff_ground_truth,
                        diff_yaw=0.0,
                    )
                    # Previous object results
                    prev_object_results = PerceptionFrameResult.get_object_results(
                        estimated_objects=prev_estimated_objects,
                        ground_truth_objects=prev_ground_truth_objects,
                    )

                # Current estimated objects
                cur_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_unique_estimated_objects
                    if use_unique_id
                    else self.dummy_estimated_objects,
                    diff_distance=cur_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                # Current ground truth objects
                cur_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=cur_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )
                # Current object results
                cur_object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )
                prev_frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
                    unix_time=0,
                    frame_name="0",
                    frame_id="base_link",
                    objects=prev_ground_truth_objects,
                    ego2map=np.eye(4),
                )
                cur_frame_ground_truth: FrameGroundTruth = FrameGroundTruth(
                    unix_time=0,
                    frame_name="0",
                    frame_id="base_link",
                    objects=prev_ground_truth_objects,
                    ego2map=np.eye(4),
                )

                clear_: CLEAR = CLEAR(
                    object_results=[prev_object_results, cur_object_results],
                    frame_ground_truths=[prev_frame_ground_truth, cur_frame_ground_truth],
                    target_labels=[target_label],
                    max_x_position_list=self.max_x_position_list,
                    max_y_position_list=self.max_y_position_list,
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
        Test check if ID is switched between previous and current results.

        test patterns:
            Check if ID was switched between each i-th current object result and j-th previous object.
            ``ans_flags`` represents the answer flag for three object results per one previous result.
        """
        # patterns: (prev_diff_trans, cur_diff_trans, use_unique_id, ans_flags)
        patterns: List[Tuple[DiffTranslation, DiffTranslation, bool, Tuple[bool]]] = [
            # === Test unique ID association ===
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                True,
                # If ID was switched for cur[i] and prev[j]
                ((False, False, True), (False, False, False), (True, False, False)),
            ),
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                True,
                ((False, False, True), (False, False, False), (True, False, False)),
            ),
            # === Test non-unique ID association ===
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                False,
                ((False, False, True), (False, False, False), (True, False, False)),
            ),
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                False,
                ((False, False, True), (False, False, False), (True, False, False)),
            ),
        ]
        for n, (prev_diff_trans, cur_diff_trans, use_unique_id, ans_flags) in enumerate(patterns):
            with self.subTest(f"Test is ID switched: {n + 1}"):
                # Previous estimated objects
                prev_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_unique_estimated_objects
                    if use_unique_id
                    else self.dummy_estimated_objects,
                    diff_distance=prev_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                # Previous ground truth objects
                prev_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=prev_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )
                # Previous object results
                prev_object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=prev_estimated_objects,
                    ground_truth_objects=prev_ground_truth_objects,
                )

                # Current esimated objects
                cur_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_unique_estimated_objects
                    if use_unique_id
                    else self.dummy_estimated_objects,
                    diff_distance=cur_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                # Current ground truth objects
                cur_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=cur_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )
                # Current object results
                cur_object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
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

    def test_is_same_result(self):
        """[summary]
        Test check if previous and current are same result.

        test patterns:
            Check if the object result is same one between each i-th current object result and j-th previous object.
            ``ans_flags`` represents the answer flag for three object results per one previous result.
        """
        # patterns: (prev_diff_trans, cur_diff_trans, use_unique_id, ans_flags)
        patterns: List[Tuple[DiffTranslation, DiffTranslation, Tuple[bool]]] = [
            # === Test unique ID association ===
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                True,
                # If the object result is same one for cur[i] and prev[j]
                ((True, False, False), (False, True, False), (False, False, True)),
            ),
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                True,
                ((False, False, False), (False, True, False), (False, False, True)),
            ),
            # === Test non-unique ID association ===
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                False,
                ((True, False, False), (False, True, False), (False, False, True)),
            ),
            (
                DiffTranslation((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                False,
                ((False, False, False), (False, True, False), (False, False, True)),
            ),
        ]
        for n, (prev_diff_trans, cur_diff_trans, use_unique_id, ans_flags) in enumerate(patterns):
            with self.subTest(f"Test is same result: {n + 1}"):
                # Previous estimated objects
                prev_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_unique_estimated_objects
                    if use_unique_id
                    else self.dummy_estimated_objects,
                    diff_distance=prev_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                # Previous ground truth objects
                prev_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=prev_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )
                # Previous object results
                prev_object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=prev_estimated_objects,
                    ground_truth_objects=prev_ground_truth_objects,
                )

                # Current esimated objects
                cur_estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_unique_estimated_objects
                    if use_unique_id
                    else self.dummy_estimated_objects,
                    diff_distance=cur_diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                # Current ground truth objects
                cur_ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=cur_diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )
                # Current object results
                cur_object_results: List[
                    DynamicObjectWithPerceptionResult
                ] = PerceptionFrameResult.get_object_results(
                    estimated_objects=cur_estimated_objects,
                    ground_truth_objects=cur_ground_truth_objects,
                )
                for i, cur_obj_result in enumerate(cur_object_results):
                    for j, prev_obj_result in enumerate(prev_object_results):
                        flag: bool = CLEAR._is_same_result(
                            cur_object_result=cur_obj_result,
                            prev_object_result=prev_obj_result,
                        )
                    self.assertEqual(flag, ans_flags[i][j], f"{flag} != ans_flags[{i}][{j}]")
