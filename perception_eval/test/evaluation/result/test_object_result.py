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

from test.util.dummy_object import make_dummy_data
from test.util.object_diff import DiffTranslation
from typing import List
from typing import Optional
from typing import Tuple
import unittest

from perception_eval.common import DynamicObject
from perception_eval.common.evaluation_task import EvaluationTask
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
            # (Est[0], GT[0]), (Est[1], GT[1]), (Est[0], None)
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                [(0, 0), (1, 1), (2, None)],
            ),
            # (2)
            # (Est[0], None), (Est[1], GT[1]), (Est[0], GT[0])
            (
                DiffTranslation((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
                [(0, None), (1, 1), (2, 0)],
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
