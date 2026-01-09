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

import math
from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple

import numpy as np
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import LabelType
from perception_eval.common.schema import FrameID
from perception_eval.common.transform import HomogeneousMatrix
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.metrics.metrics_score_config import MetricsScoreConfig
from perception_eval.evaluation.metrics.prediction.path_displacement_error import PathDisplacementError
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.object_result_matching import NuscenesObjectMatcher
import pytest


@pytest.fixture(scope="module")
def setup() -> Tuple[List[DynamicObjectWithPerceptionResult], int, List[LabelType]]:
    estimations, ground_truths = make_dummy_data(frame_id=FrameID.MAP, use_unique_id=False)
    transforms = TransformDict(
        HomogeneousMatrix(position=(0, 0, 0), rotation=(1, 0, 0, 0), src=FrameID.BASE_LINK, dst=FrameID.MAP)
    )
    target_labels: List[AutowareLabel] = [
        AutowareLabel.CAR,
        AutowareLabel.BICYCLE,
        AutowareLabel.PEDESTRIAN,
        AutowareLabel.MOTORBIKE,
    ]
    metric_score_config = MetricsScoreConfig(
        evaluation_task=EvaluationTask.PREDICTION,
        target_labels=target_labels,
        center_distance_thresholds=[0.5],
        center_distance_bev_thresholds=None,
        plane_distance_thresholds=None,
        iou_2d_thresholds=None,
        iou_3d_thresholds=None,
    )
    matcher = NuscenesObjectMatcher(
        evaluation_task=EvaluationTask.PREDICTION,
        metrics_config=metric_score_config,
        transforms=transforms,
        uuid_matching_first=False,
        matching_class_agnostic_fps=True,
    )
    object_results = matcher.match(estimations, ground_truths)
    num_ground_truth = divide_objects_to_num(ground_truths, target_labels)
    return (
        object_results,
        num_ground_truth,
        [AutowareLabel.CAR, AutowareLabel.PEDESTRIAN, AutowareLabel.BICYCLE],
    )


def test_all_path_displacement_error(setup) -> None:
    object_results, num_ground_truths, target_labels = setup
    # Flatten the object results
    selected_object_results = []
    num_gts = 0
    for _, label_object_result in object_results.items():
        for target_label in target_labels:
            threshold_object_results = label_object_result[target_label]
            num_ground_truth = num_ground_truths[target_label]
            num_gts += num_ground_truth
            for _, object_results in threshold_object_results.items():
                selected_object_results.extend(object_results)

    displacement_err = PathDisplacementError(
        object_results=selected_object_results,
        num_ground_truth=num_gts,
        target_labels=target_labels,
    )
    assert math.isclose(displacement_err.ade, 1.31175, rel_tol=1e-3)
    assert math.isclose(displacement_err.fde, 1.96204, rel_tol=1e-3)
    assert math.isclose(displacement_err.miss_rate, 0.333, rel_tol=1e-3)


def test_min_path_displacement_error(setup) -> None:
    object_results, num_ground_truths, target_labels = setup
    # Flatten the object results
    selected_object_results = []
    num_gts = 0
    for _, label_object_result in object_results.items():
        for target_label in target_labels:
            threshold_object_results = label_object_result[target_label]
            num_ground_truth = num_ground_truths[target_label]
            num_gts += num_ground_truth
            for _, object_results in threshold_object_results.items():
                selected_object_results.extend(object_results)

    displacement_err = PathDisplacementError(
        object_results=selected_object_results,
        num_ground_truth=num_gts,
        target_labels=target_labels,
        kernel="min",
    )
    assert math.isclose(displacement_err.ade, 0.0, rel_tol=1e-3)
    assert math.isclose(displacement_err.fde, 0.0, rel_tol=1e-3)
    assert math.isclose(displacement_err.miss_rate, 0.0, rel_tol=1e-3)


def test_max_path_displacement_error(setup) -> None:
    object_results, num_ground_truths, target_labels = setup
    # Flatten the object results
    selected_object_results = []
    num_gts = 0
    for _, label_object_result in object_results.items():
        for target_label in target_labels:
            threshold_object_results = label_object_result[target_label]
            num_ground_truth = num_ground_truths[target_label]
            num_gts += num_ground_truth
            for _, object_results in threshold_object_results.items():
                selected_object_results.extend(object_results)

    displacement_err = PathDisplacementError(
        object_results=selected_object_results,
        num_ground_truth=num_gts,
        target_labels=target_labels,
        kernel="max",
    )
    assert math.isclose(displacement_err.ade, 2.624, rel_tol=1e-3)
    assert math.isclose(displacement_err.fde, 3.924, rel_tol=1e-3)
    assert math.isclose(displacement_err.miss_rate, 0.667, rel_tol=1e-3)


def test_highest_path_displacement_error(setup) -> None:
    object_results, num_ground_truths, target_labels = setup
    # Flatten the object results
    selected_object_results = []
    num_gts = 0
    for _, label_object_result in object_results.items():
        for target_label in target_labels:
            threshold_object_results = label_object_result[target_label]
            num_ground_truth = num_ground_truths[target_label]
            num_gts += num_ground_truth
            for _, object_results in threshold_object_results.items():
                selected_object_results.extend(object_results)

    displacement_err = PathDisplacementError(
        object_results=selected_object_results,
        num_ground_truth=num_gts,
        target_labels=target_labels,
        kernel="highest",
    )
    assert math.isclose(displacement_err.ade, 2.624, rel_tol=1e-3)
    assert math.isclose(displacement_err.fde, 3.924, rel_tol=1e-3)
    assert math.isclose(displacement_err.miss_rate, 0.667, rel_tol=1e-3)


def test_all_path_displacement_error_thresholds(setup) -> None:
    object_results, num_ground_truths, target_labels = setup

    ans = [
        # ADE, FDE, Miss Rate
        # CAR
        (1.4142, 2.1213, 0.3333),
        # BICYCLE
        (np.nan, np.nan, np.nan),
        # PEDESTRIAN
        (1.2093, 1.8027, 0.3333),
    ]
    for _, label_object_result in object_results.items():
        for label_index, target_label in enumerate(target_labels):
            num_ground_truth = num_ground_truths[target_label]
            threshold_object_results = label_object_result[target_label]
            for _, object_results in threshold_object_results.items():
                displacement_err = PathDisplacementError(
                    object_results=object_results,
                    num_ground_truth=num_ground_truth,
                    target_labels=[target_label],
                )
                if ans[label_index][0] is np.nan:
                    assert math.isnan(displacement_err.ade)
                else:
                    assert math.isclose(displacement_err.ade, ans[label_index][0], rel_tol=1e-3)

                if ans[label_index][1] is np.nan:
                    assert math.isnan(displacement_err.fde)
                else:
                    assert math.isclose(displacement_err.fde, ans[label_index][1], rel_tol=1e-3)

                if ans[label_index][2] is np.nan:
                    assert math.isnan(displacement_err.miss_rate)
                else:
                    assert math.isclose(displacement_err.miss_rate, ans[label_index][2], rel_tol=1e-3)


def test_min_path_displacement_error_thresholds(setup) -> None:
    object_results, num_ground_truths, target_labels = setup

    ans = [
        # ADE, FDE, Miss Rate
        # CAR
        (0.0, 0.0, 0.0),
        # BICYCLE
        (np.nan, np.nan, np.nan),
        # PEDESTRIAN
        (0.0, 0.0, 0.0),
    ]
    for _, label_object_result in object_results.items():
        for label_index, target_label in enumerate(target_labels):
            num_ground_truth = num_ground_truths[target_label]
            threshold_object_results = label_object_result[target_label]
            for _, object_results in threshold_object_results.items():
                displacement_err = PathDisplacementError(
                    object_results=object_results,
                    num_ground_truth=num_ground_truth,
                    target_labels=[target_label],
                    kernel="min",
                )
                if ans[label_index][0] is np.nan:
                    assert math.isnan(displacement_err.ade)
                else:
                    assert math.isclose(displacement_err.ade, ans[label_index][0], rel_tol=1e-3)

                if ans[label_index][1] is np.nan:
                    assert math.isnan(displacement_err.fde)
                else:
                    assert math.isclose(displacement_err.fde, ans[label_index][1], rel_tol=1e-3)

                if ans[label_index][2] is np.nan:
                    assert math.isnan(displacement_err.miss_rate)
                else:
                    assert math.isclose(displacement_err.miss_rate, ans[label_index][2], rel_tol=1e-3)


def test_max_path_displacement_error_thresholds(setup) -> None:
    object_results, num_ground_truths, target_labels = setup

    ans = [
        # ADE, FDE, Miss Rate
        # CAR
        (2.8284, 4.2424, 0.6666),
        # BICYCLE
        (np.nan, np.nan, np.nan),
        # PEDESTRIAN
        (2.4186, 3.6055, 0.6666),
    ]
    for _, label_object_result in object_results.items():
        for label_index, target_label in enumerate(target_labels):
            num_ground_truth = num_ground_truths[target_label]
            threshold_object_results = label_object_result[target_label]
            for _, object_results in threshold_object_results.items():
                displacement_err = PathDisplacementError(
                    object_results=object_results,
                    num_ground_truth=num_ground_truth,
                    target_labels=[target_label],
                    kernel="max",
                )
                if ans[label_index][0] is np.nan:
                    assert math.isnan(displacement_err.ade)
                else:
                    assert math.isclose(displacement_err.ade, ans[label_index][0], rel_tol=1e-3)

                if ans[label_index][1] is np.nan:
                    assert math.isnan(displacement_err.fde)
                else:
                    assert math.isclose(displacement_err.fde, ans[label_index][1], rel_tol=1e-3)

                if ans[label_index][2] is np.nan:
                    assert math.isnan(displacement_err.miss_rate)
                else:
                    assert math.isclose(displacement_err.miss_rate, ans[label_index][2], rel_tol=1e-3)


def test_highest_path_displacement_error_thresholds(setup) -> None:
    object_results, num_ground_truths, target_labels = setup

    ans = [
        # ADE, FDE, Miss Rate
        # CAR
        (2.8284, 4.2424, 0.6666),
        # BICYCLE
        (np.nan, np.nan, np.nan),
        # PEDESTRIAN
        (2.4186, 3.6055, 0.6666),
    ]
    for _, label_object_result in object_results.items():
        for label_index, target_label in enumerate(target_labels):
            num_ground_truth = num_ground_truths[target_label]
            threshold_object_results = label_object_result[target_label]
            for _, object_results in threshold_object_results.items():
                displacement_err = PathDisplacementError(
                    object_results=object_results,
                    num_ground_truth=num_ground_truth,
                    target_labels=[target_label],
                    kernel="highest",
                )
                if ans[label_index][0] is np.nan:
                    assert math.isnan(displacement_err.ade)
                else:
                    assert math.isclose(displacement_err.ade, ans[label_index][0], rel_tol=1e-3)

                if ans[label_index][1] is np.nan:
                    assert math.isnan(displacement_err.fde)
                else:
                    assert math.isclose(displacement_err.fde, ans[label_index][1], rel_tol=1e-3)

                if ans[label_index][2] is np.nan:
                    assert math.isnan(displacement_err.miss_rate)
                else:
                    assert math.isclose(displacement_err.miss_rate, ans[label_index][2], rel_tol=1e-3)
