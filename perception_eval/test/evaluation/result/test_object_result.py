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

from copy import deepcopy
from test.util.dummy_object import make_dummy_data
from test.util.object_diff import DiffTranslation
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import unittest

from perception_eval.common import DynamicObject
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import Label
from perception_eval.evaluation.matching import MatchingLabelPolicy
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.object_result import get_object_results
from perception_eval.util.debug import get_objects_with_difference


class TestObjectResult(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()
        self.evaluation_task: EvaluationTask = EvaluationTask.DETECTION

    def test_get_object_results(self):
        """[summary]
        Test matching estimated objects and ground truth objects.

        test patterns:
            NOTE:
                - The estimations & GTs are following (number represents the index)
                        Estimation = 3
                            (0): CAR, (1): BICYCLE, (2): CAR
                        GT = 4
                            (0): CAR, (1): BICYCLE, (2): PEDESTRIAN, (3): MOTORBIKE
        """
        patterns: List[Tuple[DiffTranslation, List[Tuple[int, Optional[int]]]]] = [
            # (1)
            # (Est[0], GT[0]), (Est[1], GT[1]), (Est[0], GT[2])
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                [(0, 0), (1, 1), (2, 2)],
            ),
            # (2)
            # (Est[0], GT[2]), (Est[1], GT[1]), (Est[2], GT[0])
            (
                DiffTranslation((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
                [(0, 2), (1, 1), (2, 0)],
            ),
        ]
        for n, (diff_trans, ans_pair_index) in enumerate(patterns):
            with self.subTest(f"Test matching objects: {n + 1}"):
                estimated_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_estimated_objects,
                    diff_distance=diff_trans.diff_estimated,
                    diff_yaw=0.0,
                )
                ground_truth_objects: List[DynamicObject] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=diff_trans.diff_ground_truth,
                    diff_yaw=0.0,
                )

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=estimated_objects,
                    ground_truth_objects=ground_truth_objects,
                )

                for i, object_result_ in enumerate(object_results):
                    self.assertIn(
                        object_result_.estimated_object,
                        estimated_objects,
                        f"Unexpected estimated object at {i}",
                    )
                    estimated_object_index: int = estimated_objects.index(object_result_.estimated_object)
                    gt_idx = ans_pair_index[estimated_object_index][1]
                    if gt_idx is not None:
                        self.assertEqual(
                            object_result_.ground_truth_object,
                            ground_truth_objects[gt_idx],
                            f"Unexpected ground truth object at {i}",
                        )
                    else:
                        # In this case, there is no threshold
                        self.assertIsNone(
                            object_result_.ground_truth_object,
                            "ground truth must be None",
                        )

    def test_matching_label_policy_default(self) -> None:
        """Test retrieving object results with `MatchingLabelPolicy.DEFAULT`.

        Patterns:
        --------
            1. Est: (CAR, BICYCLE, CAR), GT: (CAR, BICYCLE, CAR)
                -> TP: [(CAR, CAR), (BICYCLE, BICYCLE), (CAR, CAR)]
            2. Est: (CAR, BICYCLE, UNKNOWN), GT: (CAR, BICYCLE, CAR)
                -> TP: [(CAR, CAR), (BICYCLE, BICYCLE)], FP [(UNKNOWN, CAR)]
            3. Est: (CAR, BICYCLE, PEDESTRIAN), GT: (CAR, BICYCLE, CAR)
                -> TP: [(CAR, CAR), (BICYCLE, BICYCLE)], FP [(PEDESTRIAN, CAR)]
        """
        patterns: List[Tuple[Dict[int, Label], List[Tuple[Label, Label, bool]]]] = [
            (
                {},
                [
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                    (Label(AutowareLabel.BICYCLE, "bicycle"), Label(AutowareLabel.BICYCLE, "bicycle"), True),
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                ],
            ),
            (
                {2: Label(AutowareLabel.UNKNOWN, "unknown")},
                [
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                    (Label(AutowareLabel.BICYCLE, "bicycle"), Label(AutowareLabel.BICYCLE, "bicycle"), True),
                    (Label(AutowareLabel.UNKNOWN, "unknown"), Label(AutowareLabel.CAR, "car"), False),
                ],
            ),
            (
                {2: Label(AutowareLabel.PEDESTRIAN, "pedestrian")},
                [
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                    (Label(AutowareLabel.BICYCLE, "bicycle"), Label(AutowareLabel.BICYCLE, "bicycle"), True),
                    (Label(AutowareLabel.PEDESTRIAN, "pedestrian"), Label(AutowareLabel.CAR, "car"), False),
                ],
            ),
        ]
        for n, (label_remap, expect_list) in enumerate(patterns):
            with self.subTest(f"Test matching objects: {n + 1}"):
                estimated_objects = deepcopy(self.dummy_estimated_objects)
                ground_truth_objects = deepcopy(self.dummy_estimated_objects)
                for idx, label in label_remap.items():
                    estimated_objects[idx].semantic_label = label

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=estimated_objects,
                    ground_truth_objects=ground_truth_objects,
                    matching_label_policy=MatchingLabelPolicy.DEFAULT,
                )

                for result, expect in zip(object_results, expect_list):
                    est_label, gt_label, is_label_correct = expect
                    assert est_label == result.estimated_object.semantic_label
                    assert gt_label == result.ground_truth_object.semantic_label
                    assert is_label_correct == result.is_label_correct

    def test_matching_label_policy_allow_unknown(self) -> None:
        """Test retrieving object results with `MatchingLabelPolicy.ALLOW_UNKNOWN`.

        Patterns:
        --------
            1. Est: (CAR, BICYCLE, CAR), GT: (CAR, BICYCLE, CAR)
                -> TP: [(CAR, CAR), (BICYCLE, BICYCLE), (CAR, CAR)]
            2. Est: (CAR, BICYCLE, UNKNOWN), GT: (CAR, BICYCLE, CAR)
                -> TP: [(CAR, CAR), (BICYCLE, BICYCLE), (UNKNOWN, CAR)]
            3. Est: (CAR, BICYCLE, PEDESTRIAN), GT: (CAR, BICYCLE, CAR)
                -> TP: [(CAR, CAR), (BICYCLE, BICYCLE)], FP [(PEDESTRIAN, CAR)]
        """
        patterns: List[Tuple[Dict[int, Label], List[Tuple[Label, Label, bool]]]] = [
            (
                {},
                [
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                    (Label(AutowareLabel.BICYCLE, "bicycle"), Label(AutowareLabel.BICYCLE, "bicycle"), True),
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                ],
            ),
            (
                {2: Label(AutowareLabel.UNKNOWN, "unknown")},
                [
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                    (Label(AutowareLabel.BICYCLE, "bicycle"), Label(AutowareLabel.BICYCLE, "bicycle"), True),
                    (Label(AutowareLabel.UNKNOWN, "unknown"), Label(AutowareLabel.CAR, "car"), True),
                ],
            ),
            (
                {2: Label(AutowareLabel.PEDESTRIAN, "pedestrian")},
                [
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                    (Label(AutowareLabel.BICYCLE, "bicycle"), Label(AutowareLabel.BICYCLE, "bicycle"), True),
                    (Label(AutowareLabel.PEDESTRIAN, "pedestrian"), Label(AutowareLabel.CAR, "car"), False),
                ],
            ),
        ]
        for n, (label_remap, expect_list) in enumerate(patterns):
            with self.subTest(f"Test matching objects: {n + 1}"):
                estimated_objects = deepcopy(self.dummy_estimated_objects)
                ground_truth_objects = deepcopy(self.dummy_estimated_objects)
                for idx, label in label_remap.items():
                    estimated_objects[idx].semantic_label = label

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=estimated_objects,
                    ground_truth_objects=ground_truth_objects,
                    matching_label_policy=MatchingLabelPolicy.ALLOW_UNKNOWN,
                )

                for result, expect in zip(object_results, expect_list):
                    est_label, gt_label, is_label_correct = expect
                    assert est_label == result.estimated_object.semantic_label
                    assert gt_label == result.ground_truth_object.semantic_label
                    assert is_label_correct == result.is_label_correct

    def test_matching_label_policy_allow_any(self) -> None:
        """Test retrieving object results with `MatchingLabelPolicy.ALLOW_ANY`.

        Patterns:
        --------
            1. Est: (CAR, BICYCLE, CAR), GT: (CAR, BICYCLE, CAR)
                -> TP: [(CAR, CAR), (BICYCLE, BICYCLE), (CAR, CAR)]
            2. Est: (CAR, BICYCLE, UNKNOWN), GT: (CAR, BICYCLE, CAR)
                -> TP: [(CAR, CAR), (BICYCLE, BICYCLE), (UNKNOWN, CAR)]
            3. Est: (CAR, BICYCLE, PEDESTRIAN), GT: (CAR, BICYCLE, CAR)
                -> TP: [(CAR, CAR), (BICYCLE, BICYCLE), (PEDESTRIAN, CAR)]
        """
        patterns: List[Tuple[Dict[int, Label], List[Tuple[Label, Label, bool]]]] = [
            (
                {},
                [
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                    (Label(AutowareLabel.BICYCLE, "bicycle"), Label(AutowareLabel.BICYCLE, "bicycle"), True),
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                ],
            ),
            (
                {2: Label(AutowareLabel.UNKNOWN, "unknown")},
                [
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                    (Label(AutowareLabel.BICYCLE, "bicycle"), Label(AutowareLabel.BICYCLE, "bicycle"), True),
                    (Label(AutowareLabel.UNKNOWN, "unknown"), Label(AutowareLabel.CAR, "car"), True),
                ],
            ),
            (
                {2: Label(AutowareLabel.PEDESTRIAN, "pedestrian")},
                [
                    (Label(AutowareLabel.CAR, "car"), Label(AutowareLabel.CAR, "car"), True),
                    (Label(AutowareLabel.BICYCLE, "bicycle"), Label(AutowareLabel.BICYCLE, "bicycle"), True),
                    (Label(AutowareLabel.PEDESTRIAN, "pedestrian"), Label(AutowareLabel.CAR, "car"), True),
                ],
            ),
        ]
        for n, (label_remap, expect_list) in enumerate(patterns):
            with self.subTest(f"Test matching objects: {n + 1}"):
                estimated_objects = deepcopy(self.dummy_estimated_objects)
                ground_truth_objects = deepcopy(self.dummy_estimated_objects)
                for idx, label in label_remap.items():
                    estimated_objects[idx].semantic_label = label

                object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
                    evaluation_task=self.evaluation_task,
                    estimated_objects=estimated_objects,
                    ground_truth_objects=ground_truth_objects,
                    matching_label_policy=MatchingLabelPolicy.ALLOW_ANY,
                )

                for result, expect in zip(object_results, expect_list):
                    est_label, gt_label, is_label_correct = expect
                    assert est_label == result.estimated_object.semantic_label
                    assert gt_label == result.ground_truth_object.semantic_label
                    assert is_label_correct == result.is_label_correct
