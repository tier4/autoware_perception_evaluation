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
import unittest

from perception_eval.common.label import AutowareLabel
from perception_eval.common.threshold import ThresholdsError
from perception_eval.common.threshold import check_thresholds
from perception_eval.common.threshold import check_thresholds_list
from perception_eval.common.threshold import get_label_threshold
from perception_eval.common.threshold import set_thresholds


class TestThreshold(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_get_label_threshold(self):
        """[summary]
        Test getting threshold for each label in threshold list.

        Test patterns:
            Test if return threshold at correct index same with specified target labels.
        """
        target_labels: List[AutowareLabel] = [
            AutowareLabel.CAR,
            AutowareLabel.BUS,
            AutowareLabel.PEDESTRIAN,
        ]
        threshold_list: List[float] = [0.1, 0.2, 0.3]

        # patterns: (semantic_label,  ans_threshold)
        patterns: List[Tuple[AutowareLabel, float]] = [
            (AutowareLabel.CAR, 0.1),
            (AutowareLabel.BUS, 0.2),
            (AutowareLabel.PEDESTRIAN, 0.3),
        ]
        for semantic_label, ans_threshold in patterns:
            with self.subTest("Test get_label_thresholds"):
                threshold: float = get_label_threshold(
                    semantic_label,
                    target_labels,
                    threshold_list,
                )
                self.assertEqual(threshold, ans_threshold)

    def test_set_thresholds(self):
        """[summary]
        Test setting thresholds list.

        Test patterns:
        """
        # patterns: (thresholds, num_object, ans_thresholds)
        patterns: List = [
            ([1.0, 2.0], 3, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            ([[1.0, 2.0], [3.0, 4.0]], 2, [[1.0, 2.0], [3.0, 4.0]]),
            ([[1.0, 2.0, 3.0]], 3, [[1.0, 2.0, 3.0]]),
            ([0.5], 3, [[0.5, 0.5, 0.5]]),
        ]
        for n, (thresholds, num_object, ans_thresholds) in enumerate(patterns):
            with self.subTest(f"Test set_thresholds: {n + 1}"):
                out_thresholds = set_thresholds(thresholds, num_object)
                self.assertEqual(out_thresholds, ans_thresholds)

        # AssertionError: The empty list is not allowed
        fail_patters: List = [([], 1), ([[1.0, 2.0], [3.0, 4.0]], 3)]
        for n, (thresholds, target_object_num) in enumerate(fail_patters):
            with self.subTest(f"Test fail pattern for set thresholds: {n + 1}"):
                with self.assertRaises(AssertionError):
                    _ = set_thresholds(thresholds, target_object_num)

        # ThresholdsError: Check type error
        with self.subTest("Test fail patterns for set thresholds"):
            with self.assertRaises(ThresholdsError):
                _ = set_thresholds([(1.0, 2.0, 3.0)], 3)

    def test_check_thresholds(self):
        """[summary]
        Test if it can detect the exception.
        """
        # patterns: (thresholds, target_labels, exception)
        patterns: List = [
            (
                [1.0, 2.0],
                [AutowareLabel.CAR, AutowareLabel.BICYCLE, AutowareLabel.PEDESTRIAN],
                ThresholdsError,
            ),
        ]
        for n, (thresholds, target_labels, exception) in enumerate(patterns):
            with self.subTest(f"Test {n + 1}"):
                with self.assertRaises(exception):
                    _ = check_thresholds(thresholds, target_labels)

    def test_check_thresholds_list(self):
        """[summary]
        Test if it can detect the exception.
        """
        # patterns: (thresholds_list, target_labels, exception)
        patterns: List = [
            (
                [[1.0, 1.0, 1.0], [2.0, 2.0]],
                [AutowareLabel.CAR, AutowareLabel.BICYCLE, AutowareLabel.PEDESTRIAN],
                ThresholdsError,
            )
        ]
        for n, (thresholds_list, target_labels, exception) in enumerate(patterns):
            with self.subTest(f"Test {n + 1}"):
                with self.assertRaises(exception):
                    _ = check_thresholds_list(thresholds_list, target_labels, exception)
