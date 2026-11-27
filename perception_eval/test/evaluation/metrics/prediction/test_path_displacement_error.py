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

from copy import deepcopy
import math
from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple

from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import LabelType
from perception_eval.common.schema import FrameID
from perception_eval.common.transform import HomogeneousMatrix
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.metrics.prediction.path_displacement_error import PathDisplacementError
from perception_eval.evaluation.metrics.prediction.utils import prepare_path
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.object_result_matching import get_object_results
import pytest


@pytest.fixture(scope="module")
def setup() -> Tuple[List[DynamicObjectWithPerceptionResult], int, List[LabelType]]:
    estimations, ground_truths = make_dummy_data(frame_id=FrameID.MAP, use_unique_id=False)
    transforms = TransformDict(
        HomogeneousMatrix(position=(0, 0, 0), rotation=(1, 0, 0, 0), src=FrameID.BASE_LINK, dst=FrameID.MAP)
    )
    return (
        get_object_results(
            evaluation_task=EvaluationTask.PREDICTION,
            estimated_objects=estimations,
            ground_truth_objects=ground_truths,
            transforms=transforms,
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
    assert math.isclose(displacement_err.ade, 1.31175, rel_tol=1e-3)
    assert math.isclose(displacement_err.fde, 1.96204, rel_tol=1e-3)
    assert math.isclose(displacement_err.miss_rate, 0.333, rel_tol=1e-3)


def test_min_path_displacement_error(setup) -> None:
    object_results, num_ground_truth, target_labels = setup
    displacement_err = PathDisplacementError(
        object_results=object_results,
        num_ground_truth=num_ground_truth,
        target_labels=target_labels,
        kernel="min",
    )
    assert math.isclose(displacement_err.ade, 0.0, rel_tol=1e-3)
    assert math.isclose(displacement_err.fde, 0.0, rel_tol=1e-3)
    assert math.isclose(displacement_err.miss_rate, 0.0, rel_tol=1e-3)


def test_max_path_displacement_error(setup) -> None:
    object_results, num_ground_truth, target_labels = setup
    displacement_err = PathDisplacementError(
        object_results=object_results,
        num_ground_truth=num_ground_truth,
        target_labels=target_labels,
        kernel="max",
    )
    assert math.isclose(displacement_err.ade, 2.624, rel_tol=1e-3)
    assert math.isclose(displacement_err.fde, 3.924, rel_tol=1e-3)
    assert math.isclose(displacement_err.miss_rate, 0.667, rel_tol=1e-3)


def test_highest_path_displacement_error(setup) -> None:
    object_results, num_ground_truth, target_labels = setup
    displacement_err = PathDisplacementError(
        object_results=object_results,
        num_ground_truth=num_ground_truth,
        target_labels=target_labels,
        kernel="highest",
    )
    assert math.isclose(displacement_err.ade, 2.624, rel_tol=1e-3)
    assert math.isclose(displacement_err.fde, 3.924, rel_tol=1e-3)
    assert math.isclose(displacement_err.miss_rate, 0.667, rel_tol=1e-3)


def test_path_displacement_error_skips_estimation_without_paths(setup) -> None:
    """An estimation whose predicted paths were all dropped (a path that held only the current pose has no
    future waypoint) used to make ``np.stack`` raise "need at least one array to stack" and end the whole
    evaluation; the pair is skipped instead."""
    object_results, num_ground_truth, target_labels = setup
    matched = [r for r in object_results if r.ground_truth_object is not None and r.is_label_correct]
    assert matched, "the dummy data must hold a correctly labelled match"
    pathless = deepcopy(matched[0])
    # what the log-replay conversion yields for a ROS path holding only the current pose: one mode with no
    # future waypoint, which set_object_paths drops (kept in the constructor arguments, as deepcopy
    # rebuilds the object from them)
    estimation = pathless.estimated_object
    estimation.relative_timestamps = [[]]
    estimation.predicted_positions = [[]]
    estimation.predicted_orientations = [[]]
    estimation.predicted_twists = None
    estimation.predicted_scores = [1.0]
    estimation.predicted_paths = []

    estimation, ground_truth = prepare_path(pathless, top_k=1)
    assert estimation.get_path_error(ground_truth).shape[0] == 0
    displacement_err = PathDisplacementError(
        object_results=[pathless],
        num_ground_truth=num_ground_truth,
        target_labels=target_labels,
    )
    # nothing was measured: the metric is the empty-set value, not an exception
    assert math.isnan(displacement_err.ade) or displacement_err.ade == 0.0
