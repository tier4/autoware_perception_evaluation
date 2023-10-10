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

import unittest
from typing import List, Tuple

import pytest

from perception_eval.common.label import AutowareLabel, Label
from perception_eval.common.threshold import (
    ThresholdError,
    check_nested_thresholds,
    check_thresholds,
    get_label_threshold,
    set_thresholds,
)


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

        patterns: List[Tuple[AutowareLabel, float]] = [
            (Label(AutowareLabel.CAR, "car", []), 0.1),
            (Label(AutowareLabel.BUS, "bus", []), 0.2),
            (Label(AutowareLabel.PEDESTRIAN, "pedestrian", []), 0.3),
        ]
        for semantic_label, ans_threshold in patterns:
            with self.subTest("Test get_label_thresholds"):
                threshold: float = get_label_threshold(
                    semantic_label,
                    target_labels,
                    threshold_list,
                )
                assert threshold == ans_threshold

    def test_set_thresholds(self):
        """[summary]
        Test setting thresholds list.

        Test patterns:
        """
        success_patterns: List = [
            (1.0, 3, False, [1.0, 1.0, 1.0]),
            ([1.0], 3, False, [1.0, 1.0, 1.0]),
            ([1.0, 2.0, 3.0], 3, False, [1.0, 2.0, 3.0]),
            (1.0, 3, True, [[1.0, 1.0, 1.0]]),
            ([1.0], 3, True, [[1.0, 1.0, 1.0]]),
            ([1.0, 2.0], 3, True, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            ([1.0, 2.0, 3.0], 3, True, [[1.0, 2.0, 3.0]]),
            ([[1.0, 2.0], [3.0, 4.0]], 2, True, [[1.0, 2.0], [3.0, 4.0]]),
            ([[1.0], [2.0, 3.0]], 2, True, [[1.0, 1.0], [2.0, 3.0]]),
        ]
        for n, (thresholds, num_object, nest, expect) in enumerate(success_patterns):
            with self.subTest(f"Test set_thresholds: {n + 1}"):
                answer = set_thresholds(thresholds, num_object, nest)
                assert answer == expect

        fail_patters: List = [
            ([], 2, False),
            ([1.0, [2.0]], 2, False),
            ([1.0, 2.0], 3, False),
            ([], 3, True),
            ([1.0, [2.0]], 2, True),
            ([[1.0, 2.0, 3.0]], 2, True),
        ]
        for n, (thresholds, num_object, nest) in enumerate(fail_patters):
            with self.subTest(f"Test fail pattern for set thresholds: {n + 1}"), pytest.raises(ThresholdError):
                set_thresholds(thresholds, num_object, nest)

    def test_check_thresholds(self):
        """[summary]
        Test if it can detect the exception.
        """
        patterns: List = [([1.0, 2.0, 3.0], 2)]
        for n, (thresholds, target_labels) in enumerate(patterns):
            with self.subTest(f"Test {n + 1}"), pytest.raises(ThresholdError):
                check_thresholds(thresholds, target_labels)

    def test_check_nested_thresholds(self):
        """[summary]
        Test if it can detect the exception.
        """
        patterns: List = [([[1.0, 1.0, 1.0], [2.0, 2.0]], 3)]
        for n, (thresholds_list, target_labels) in enumerate(patterns):
            with self.subTest(f"Test {n + 1}"), pytest.raises(ThresholdError):
                check_nested_thresholds(thresholds_list, target_labels)
