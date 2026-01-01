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
from perception_eval.evaluation.matching import MatchingLabelPolicy, MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.object_result_matching import NuscenesObjectMatcher, get_object_results
from perception_eval.util.debug import get_objects_with_difference
from perception_eval.evaluation.metrics.config.detection_metrics_config import DetectionMetricsConfig
from perception_eval.evaluation.metrics.metrics_score_config import MetricsScoreConfig


class TestNuSceneObjectResult(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_estimated_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_estimated_objects, self.dummy_ground_truth_objects = make_dummy_data()
        self.evaluation_task: EvaluationTask = EvaluationTask.DETECTION
        self.target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BICYCLE,
            AutowareLabel.PEDESTRIAN,
            AutowareLabel.MOTORBIKE,
        ]
        self.metric_configs = MetricsScoreConfig(
            evaluation_task=self.evaluation_task,
            target_labels=self.target_labels,
            center_distance_thresholds=[0.5, 2.0, 4.0],
            center_distance_bev_thresholds=[0.5, 2.0, 4.0],
            plane_distance_thresholds=[0.5, 2.0, 4.0],
        )

    def _test_object_results(
        self, 
        matcher: NuscenesObjectMatcher, 
        diff_trans: DiffTranslation, 
        threshold_to_ans_pair_index: Dict[MatchingMode, Dict[float, Optional[int]]]) -> None:
        """
        Test matching estimated objects and ground truth objects.

        Args:
            matcher (NuscenesObjectMatcher): Nuscenes object matcher.
            diff_trans (DiffTranslation): Difference translation to add to estimated and ground truth objects.
            threshold_to_ans_pair_index (Dict[MatchingMode, Dict[float, Optional[int]]]): Threshold to expected matching results pair index (est_idx: gt_idx).
        """
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
        object_results = matcher.match(estimated_objects, ground_truth_objects)
        for matching_mode, label_to_threshold_to_object_results in object_results.items():
            for label, threshold_to_object_results in label_to_threshold_to_object_results.items():
                for threshold, object_results in threshold_to_object_results.items():
                    ans_pair_index = threshold_to_ans_pair_index[matching_mode][threshold]
                    for object_result in object_results:
                        self.assertIn(
                            object_result.estimated_object,
                            estimated_objects,
                            f"Unexpected estimated object at Matching mode: {matching_mode}, Label: {label}, Threshold: {threshold}",
                        )
                        estimated_object_index: int = estimated_objects.index(object_result.estimated_object)
                        gt_idx = ans_pair_index[estimated_object_index][1]
                        if gt_idx is not None:  
                            self.assertEqual(
                                object_result.ground_truth_object,
                                ground_truth_objects[gt_idx],
                                f"Unexpected ground truth object at Matching mode: {matching_mode}, Label: {label}, Threshold: {threshold}",
                            )
                        else:
                            # In this case, there is no threshold
                            self.assertIsNone(
                                object_result.ground_truth_object,
                                f"Ground truth must be None at Matching mode: {matching_mode}, Label: {label}, Threshold: {threshold}",
                            )
    def test_matching_objects_default(self):
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
            # Center Distance/Center Distance BEV:
            # {
                # Threshold: 0.5
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
                # Threshold: 2.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # } 
            # PLANEDISTANCE:
            # {
                # Threshold: 0.5
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
                # Threshold: 2.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # } 
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                # MatchignMode: Threshold: expected matching results
                {
                    MatchingMode.CENTERDISTANCE: {
                        0.5: [(0, 0), (1, 1), (2, None)],
                        2.0: [(0, 0), (1, 1), (2, None)],
                        4.0: [(0, 0), (1, 1), (2, None)],
                    },
                    MatchingMode.CENTERDISTANCEBEV: {
                        0.5: [(0, 0), (1, 1), (2, None)],
                        2.0: [(0, 0), (1, 1), (2, None)],
                        4.0: [(0, 0), (1, 1), (2, None)],
                    },
                    MatchingMode.PLANEDISTANCE: {
                        0.5: [(0, 0), (1, 1), (2, None)],
                        2.0: [(0, 0), (1, 1), (2, None)],
                        4.0: [(0, 0), (1, 1), (2, None)],
                    },
                },
            ),
            # (2)
            # Center Distance/Center Distance BEV:
            # {
                # Threshold: 0.5
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 2.0
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # } 
            # PLANEDISTANCE:
            # {
                # Threshold: 0.5
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 2.0
                # (Est[0], None), (Est[1], GT[1]), (Est[2], GT[0])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], None)
            # } 
            (
                DiffTranslation((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
                {
                    # MatchignMode: Threshold: expected matching results
                    MatchingMode.CENTERDISTANCE: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, None), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, None)],
                    },
                    MatchingMode.CENTERDISTANCEBEV: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, None), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, None)],
                    },
                    MatchingMode.PLANEDISTANCE: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, 1), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, None)],
                    },
                }
            ),
        ]
        matcher = NuscenesObjectMatcher(
            evaluation_task=self.evaluation_task,
            metrics_config=self.metric_configs,
            matching_label_policy=MatchingLabelPolicy.DEFAULT,
            transforms=None,
            matching_class_agnostic_fps=False,
        )
        for n, (diff_trans, threshold_to_ans_pair_index) in enumerate(patterns):
            with self.subTest(f"Test matching objects: {n + 1}"):
                self._test_object_results(matcher, diff_trans, threshold_to_ans_pair_index)

    def test_matching_fps_class_agnostic(self):
        """[summary]
        Test matching estimated objects and ground truth objects with class agnostic fps.

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
            # Center Distance/Center Distance BEV:
            # {
                # Threshold: 0.5
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 2.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            # PLANEDISTANCE:
            # {
                # Threshold: 0.5
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 2.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                # MatchignMode: Threshold: expected matching results
                {
                    MatchingMode.CENTERDISTANCE: {
                        0.5: [(0, 0), (1, 1), (2, 2)],
                        2.0: [(0, 0), (1, 1), (2, 2)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.CENTERDISTANCEBEV: {
                        0.5: [(0, 0), (1, 1), (2, 2)],
                        2.0: [(0, 0), (1, 1), (2, 2)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.PLANEDISTANCE: {
                        0.5: [(0, 0), (1, 1), (2, 2)],
                        2.0: [(0, 0), (1, 1), (2, 2)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                },
            ),
            # (2)
            # Center Distance/Center Distance BEV:
            # {
                # Threshold: 0.5
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 2.0
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            # PLANEDISTANCE:
            # {
                # Threshold: 0.5
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 2.0
                # (Est[0], None), (Est[1], GT[1]), (Est[2], GT[0])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            (
                DiffTranslation((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
                {
                    # MatchignMode: Threshold: expected matching results
                    MatchingMode.CENTERDISTANCE: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, None), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.CENTERDISTANCEBEV: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, None), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.PLANEDISTANCE: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, 1), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                }
            ),
        ]
        matcher = NuscenesObjectMatcher(
            evaluation_task=self.evaluation_task,
            metrics_config=self.metric_configs,
            matching_label_policy=MatchingLabelPolicy.DEFAULT,
            transforms=None,
            matching_class_agnostic_fps=True,
        )
        for n, (diff_trans, threshold_to_ans_pair_index) in enumerate(patterns):
            with self.subTest(f"Test matching objects: {n + 1}"):
                self._test_object_results(matcher, diff_trans, threshold_to_ans_pair_index)

    def test_matching_objects_allow_any(self):
        """[summary]
        Test matching estimated objects and ground truth objects with ALLOW_ANY matching label policy.

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
            # Center Distance/Center Distance BEV:
            # {
                # Threshold: 0.5
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 2.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            # PLANEDISTANCE:
            # {
                # Threshold: 0.5
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 2.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                # MatchignMode: Threshold: expected matching results
                {
                    MatchingMode.CENTERDISTANCE: {
                        0.5: [(0, 0), (1, 1), (2, 2)],
                        2.0: [(0, 0), (1, 1), (2, 2)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.CENTERDISTANCEBEV: {
                        0.5: [(0, 0), (1, 1), (2, 2)],
                        2.0: [(0, 0), (1, 1), (2, 2)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.PLANEDISTANCE: {
                        0.5: [(0, 0), (1, 1), (2, 2)],
                        2.0: [(0, 0), (1, 1), (2, 2)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                },
            ),
            # (2)
            # Center Distance/Center Distance BEV:
            # {
                # Threshold: 0.5
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 2.0
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            # PLANEDISTANCE:
            # {
                # Threshold: 0.5
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 2.0
                # (Est[0], None), (Est[1], GT[1]), (Est[2], GT[0])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            (
                DiffTranslation((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
                {
                    # MatchignMode: Threshold: expected matching results
                    MatchingMode.CENTERDISTANCE: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, None), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.CENTERDISTANCEBEV: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, None), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.PLANEDISTANCE: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, 1), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                }
            ),
        ]
        matcher = NuscenesObjectMatcher(
            evaluation_task=self.evaluation_task,
            metrics_config=self.metric_configs,
            matching_label_policy=MatchingLabelPolicy.ALLOW_ANY,
            transforms=None,
            matching_class_agnostic_fps=False,
        )
        for n, (diff_trans, threshold_to_ans_pair_index) in enumerate(patterns):
            with self.subTest(f"Test matching objects: {n + 1}"):
                self._test_object_results(matcher, diff_trans, threshold_to_ans_pair_index)

    def test_matching_objects_allow_any_fps_class_agnostic(self):
        """[summary]
        Test matching estimated objects and ground truth objects with ALLOW_ANY matching label policy and class agnostic fps. 
        However, it doesn't support this combination since ALLOW_ANY matching label policy match detections without matching their label. 
        Therefore, matching_class_agnostic_fps will set to False in NuSceneObjectMatcher, and it will have similar behavior 
        as ALLOW_ANY matching label policy.
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
            # Center Distance/Center Distance BEV:
            # {
                # Threshold: 0.5
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 2.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            # PLANEDISTANCE:
            # {
                # Threshold: 0.5
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 2.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            (
                DiffTranslation((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                # MatchignMode: Threshold: expected matching results
                {
                    MatchingMode.CENTERDISTANCE: {
                        0.5: [(0, 0), (1, 1), (2, 2)],
                        2.0: [(0, 0), (1, 1), (2, 2)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.CENTERDISTANCEBEV: {
                        0.5: [(0, 0), (1, 1), (2, 2)],
                        2.0: [(0, 0), (1, 1), (2, 2)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.PLANEDISTANCE: {
                        0.5: [(0, 0), (1, 1), (2, 2)],
                        2.0: [(0, 0), (1, 1), (2, 2)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                },
            ),
            # (2)
            # Center Distance/Center Distance BEV:
            # {
                # Threshold: 0.5
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 2.0
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            # PLANEDISTANCE:
            # {
                # Threshold: 0.5
                # (Est[0], None), (Est[1], None), (Est[2], GT[0])
                # Threshold: 2.0
                # (Est[0], None), (Est[1], GT[1]), (Est[2], GT[0])
                # Threshold: 4.0
                # (Est[0], GT[0]), (Est[1], GT[1]), (Est[2], GT[2])
            # } 
            (
                DiffTranslation((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
                {
                    # MatchignMode: Threshold: expected matching results
                    MatchingMode.CENTERDISTANCE: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, None), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.CENTERDISTANCEBEV: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, None), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                    MatchingMode.PLANEDISTANCE: {
                        0.5: [(0, None), (1, None), (2, 0)],
                        2.0: [(0, None), (1, 1), (2, 0)],
                        4.0: [(0, 0), (1, 1), (2, 2)],
                    },
                }
            ),
        ]
        matcher = NuscenesObjectMatcher(
            evaluation_task=self.evaluation_task,
            metrics_config=self.metric_configs,
            matching_label_policy=MatchingLabelPolicy.ALLOW_ANY,
            transforms=None,
            matching_class_agnostic_fps=True,
        )
        for n, (diff_trans, threshold_to_ans_pair_index) in enumerate(patterns):
            with self.subTest(f"Test matching objects: {n + 1}"):
                self._test_object_results(matcher, diff_trans, threshold_to_ans_pair_index)
