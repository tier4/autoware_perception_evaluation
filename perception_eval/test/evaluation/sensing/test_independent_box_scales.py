# Copyright 2026 TIER IV, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
import pickle
from types import SimpleNamespace

import numpy as np
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.shape import Shape
from perception_eval.common.shape import ShapeType
from perception_eval.evaluation.sensing.bounding_box_scale_config import BoundingBoxScaleConfig
from perception_eval.evaluation.sensing.sensing_frame_config import SensingFrameConfig
from perception_eval.evaluation.sensing.sensing_frame_result import SensingFrameResult
from perception_eval.manager.sensing_evaluation_manager import SensingEvaluationManager
from pyquaternion import Quaternion
import pytest


def object_at(x=50.0, uuid="selected"):
    return DynamicObject(
        unix_time=1,
        frame_id=FrameID.BASE_LINK,
        position=(x, 0.0, 0.0),
        orientation=Quaternion(),
        shape=Shape(ShapeType.BOUNDING_BOX, (2.0, 2.0, 2.0)),
        velocity=(0.0, 0.0, 0.0),
        semantic_score=1.0,
        semantic_label=Label(AutowareLabel.CAR, "car"),
        pointcloud_num=1,
        uuid=uuid,
    )


def manager(common=1.0, uuids=None):
    # Isolate sensing logic from dataset loading and filesystem visualizers.
    result = SensingEvaluationManager.__new__(SensingEvaluationManager)
    result.evaluator_config = SimpleNamespace(
        metrics_params={
            "box_scale_0m": common,
            "box_scale_100m": common,
            "min_points_threshold": 1,
        },
        filtering_params={"target_uuids": uuids},
    )
    result.frame_results = []
    return result


def area():
    obj = object_at()
    obj.state.shape = Shape(ShapeType.BOUNDING_BOX, (10.0, 100.0, 4.0))
    return obj.get_corners().tolist()


@pytest.mark.parametrize("d,n,detected,remaining", [(2, 1, True, 1), (1, 2, False, 0)])
def test_manager_keeps_detection_and_both_non_detection_stages_independent(
    d, n, detected, remaining
):
    mgr = manager(common=3.0)
    result = mgr.add_frame_result(
        1,
        FrameGroundTruth(1, "1", [object_at()]),
        np.array([[51.5, 0.0, 0.0]]),
        [area()],
        SensingFrameConfig(None, d, d, 1),
        BoundingBoxScaleConfig(n, n),
    )
    assert bool(result.detection_success_results) == detected
    assert sum(len(a) for a in result.pointcloud_failed_non_detection) == remaining
    assert len(mgr.frame_results) == 1


@pytest.mark.parametrize("n,remaining", [(1, 1), (2, 0)])
def test_each_non_detection_stage_uses_explicit_scale(n, remaining):
    points = np.array([[51.5, 0.0, 0.0]])
    mgr = manager(common=3.0)
    cropped = mgr.crop_pointcloud(
        [object_at()], points, [area()], None, BoundingBoxScaleConfig(n, n)
    )
    assert sum(map(len, cropped)) == remaining
    # Feed the point directly to stage two, so stage one cannot hide an incorrect D.
    result = SensingFrameResult(
        SensingFrameConfig(None, 3.0, 3.0, 1), 1, "1", BoundingBoxScaleConfig(n, n)
    )
    result._evaluate_pointcloud_for_non_detection([object_at()], [points])
    assert sum(map(len, result.pointcloud_failed_non_detection)) == remaining


def test_legacy_api_preserves_distinct_scale_sources():
    mgr = manager(common=1.0)
    points = np.array([[51.5, 0.0, 0.0]])
    assert len(mgr.crop_pointcloud([object_at()], points, [area()])[0]) == 1
    result = mgr.add_frame_result(
        1,
        FrameGroundTruth(1, "1", [object_at()]),
        points,
        [area()],
        SensingFrameConfig(None, 2.0, 2.0, 1),
    )
    assert not result.pointcloud_failed_non_detection
    assert result.detection_success_results[0].inside_pointcloud_num == 1


def test_stage_one_excludes_unselected_ground_truth():
    mgr = manager(uuids=["selected"])
    result = mgr.add_frame_result(
        1,
        FrameGroundTruth(1, "1", [object_at(), object_at(60.0, "unselected")]),
        np.array([[61.5, 0.0, 0.0]]),
        [area()],
        SensingFrameConfig(["selected"], 1.0, 1.0, 1),
        BoundingBoxScaleConfig(2.0, 2.0),
    )
    assert not result.pointcloud_failed_non_detection
    assert len(result.detection_fail_results) == 1
    assert result.detection_fail_results[0].ground_truth_object.uuid == "selected"


def test_old_pickle_without_scale_attribute_retains_legacy_behavior():
    result = SensingFrameResult(SensingFrameConfig(None, 2.0, 2.0, 1), 1, "1")
    del result.non_detection_box_scale
    restored = pickle.loads(pickle.dumps(result))
    restored._evaluate_pointcloud_for_non_detection([object_at()], [np.array([[51.5, 0.0, 0.0]])])
    assert not restored.pointcloud_failed_non_detection


@pytest.mark.parametrize("distance,expected", [(0, 1), (50, 1.5), (100, 2), (200, 3)])
def test_scale_config_uses_existing_linear_extrapolation(distance, expected):
    scale = BoundingBoxScaleConfig(1.0, 2.0)
    assert scale.get_scale_factor(distance) == expected
    with pytest.raises(FrozenInstanceError):
        scale.box_scale_0m = 3


def test_legacy_values_are_not_revalidated_and_decreasing_extrapolation_is_not_clamped():
    assert BoundingBoxScaleConfig(0.0, 0.0).get_scale_factor(50) == 0
    assert BoundingBoxScaleConfig(2.0, 1.0).get_scale_factor(300) == -1


def test_scale_does_not_expand_height_and_new_pickle_retains_exclusion_scale():
    mgr = manager()
    result = mgr.add_frame_result(
        1,
        FrameGroundTruth(1, "1", [object_at()]),
        np.array([[50.0, 0.0, 1.5]]),
        [area()],
        SensingFrameConfig(None, 2.0, 2.0, 1),
        BoundingBoxScaleConfig(2.0, 2.0),
    )
    assert not result.detection_success_results
    assert len(result.pointcloud_failed_non_detection[0]) == 1
    restored = pickle.loads(pickle.dumps(result))
    assert restored.non_detection_box_scale == BoundingBoxScaleConfig(2.0, 2.0)


def test_empty_ground_truth_still_evaluates_non_detection_points():
    mgr = manager()
    result = mgr.add_frame_result(
        1,
        FrameGroundTruth(1, "1", []),
        np.array([[51.5, 0.0, 0.0]]),
        [area()],
        SensingFrameConfig(None, 1.0, 1.0, 1),
        BoundingBoxScaleConfig(2.0, 2.0),
    )
    assert not result.detection_success_results
    assert not result.detection_fail_results
    assert len(result.pointcloud_failed_non_detection[0]) == 1
