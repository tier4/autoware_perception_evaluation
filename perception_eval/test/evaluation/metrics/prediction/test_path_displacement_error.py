# Copyright 2025 TIER IV, Inc.

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

from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import LabelType
from perception_eval.common.schema import FrameID
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.metrics.prediction.path_displacement_error import PathDisplacementError
from perception_eval.evaluation.result.object_result import get_object_results
import pytest


@pytest.fixture(scope="module")
def setup() -> Tuple[List[DynamicObjectWithPerceptionResult], int, List[LabelType]]:
    estimations, ground_truths = make_dummy_data(frame_id=FrameID.MAP)
    return (
        get_object_results(
            evaluation_task=EvaluationTask.PREDICTION,
            estimated_objects=estimations,
            ground_truth_objects=ground_truths,
        ),
        len(ground_truths),
        [AutowareLabel.CAR, AutowareLabel.PEDESTRIAN, AutowareLabel.BICYCLE],
    )


def test_all_path_displacement_error(setup) -> None:
    object_results, num_ground_truth, target_labels = setup
    displacement_err = PathDisplacementError(
        object_results=object_results,
        num_ground_truth=num_ground_truth,
        target_labels=target_labels,
    )
    print(displacement_err.ade)


def test_min_path_displacement_error(setup) -> None:
    object_results, num_ground_truth, target_labels = setup
    displacement_err = PathDisplacementError(
        object_results=object_results,
        num_ground_truth=num_ground_truth,
        target_labels=target_labels,
        kernel="min",
    )
    print(displacement_err.ade)


def test_max_path_displacement_error(setup) -> None:
    object_results, num_ground_truth, target_labels = setup
    displacement_err = PathDisplacementError(
        object_results=object_results,
        num_ground_truth=num_ground_truth,
        target_labels=target_labels,
        kernel="max",
    )
    print(displacement_err.ade)


def test_highest_path_displacement_error(object_results) -> None:
    object_results, num_ground_truth, target_labels = setup
    displacement_err = PathDisplacementError(
        object_results=object_results,
        num_ground_truth=num_ground_truth,
        target_labels=target_labels,
        kernel="highest",
    )
    print(displacement_err.ade)
