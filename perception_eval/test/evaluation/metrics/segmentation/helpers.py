# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for the segmentation metric tests."""

from __future__ import annotations

from typing import Optional
from typing import Sequence

import numpy as np
from perception_eval.evaluation.metrics.config.segmentation_metrics_config import SegmentationMetricsConfig
from perception_eval.evaluation.metrics.filters import FrameMetricContext
from perception_eval.evaluation.metrics.filters import MetricFilter
from perception_eval.evaluation.metrics.segmentation.state import RAW_TAXONOMY
from perception_eval.evaluation.metrics.segmentation.state import SegmentationFrame
from perception_eval.evaluation.metrics.segmentation.state import SegmentationView

CLASS_NAMES = ("road", "obstacle")


def scores_for(pred: Sequence[int], confidence: float, num_classes: int = 2) -> np.ndarray:
    """Probability rows whose argmax is ``pred`` at the given max probability."""
    rest = (1.0 - confidence) / (num_classes - 1)
    scores = np.full((len(pred), num_classes), rest, dtype=np.float64)
    scores[np.arange(len(pred)), np.asarray(pred)] = confidence
    return scores


def make_frame(
    xs: Sequence[float],
    target: Sequence[int],
    pred: Sequence[int],
    num_classes: int = 2,
    scores: Optional[np.ndarray] = None,
    frame_name: str = "0",
    scene_id: Optional[str] = None,
    coords: Optional[np.ndarray] = None,
    gt_objects=(),
) -> SegmentationFrame:
    """Frame with points along the x axis (or explicit ``coords``); uniform probabilities by default."""
    if coords is None:
        coords = np.array([[float(x), 0.0, 0.0] for x in xs], dtype=np.float64).reshape(-1, 3)
    if scores is None:
        scores = np.full((coords.shape[0], num_classes), 1.0 / num_classes, dtype=np.float64)
    return SegmentationFrame(
        frame_name=frame_name,
        scene_id=scene_id,
        coordinates=coords,
        targets=np.asarray(target, dtype=np.int64),
        predictions=np.asarray(pred, dtype=np.int64),
        probabilities=scores,
        gt_objects=tuple(gt_objects),
    )


def make_view(frame: SegmentationFrame, class_names=CLASS_NAMES, ignore_index: int = -1) -> SegmentationView:
    """Whole-scene raw view of a frame with the valid mask applied (as the suite would build it)."""
    from perception_eval.evaluation.metrics.segmentation.state import build_raw_taxonomy

    taxonomy = build_raw_taxonomy(frame, class_names, ignore_index)
    valid = taxonomy.valid
    return SegmentationView(
        taxonomy=RAW_TAXONOMY,
        filter_name="",
        range_name=None,
        frame_name=frame.frame_name,
        pred=taxonomy.pred[valid],
        target=taxonomy.target[valid],
        confidence=taxonomy.confidence[valid],
        entropy=taxonomy.entropy[valid],
        xyz=np.asarray(frame.coordinates[:, :3], dtype=np.float64)[valid],
        gt_boxes=np.zeros((0, 9)),
        gt_box_seg_class=np.zeros(0, dtype=np.int64),
        gt_box_det_label=np.zeros(0, dtype=np.int64),
        num_classes=len(class_names),
        class_names=tuple(class_names),
    )


def config(components, class_names=CLASS_NAMES, **kwargs) -> SegmentationMetricsConfig:
    """Config with ``check_argmax`` off by default: test frames often use uniform probabilities."""
    kwargs.setdefault("check_argmax", False)
    return SegmentationMetricsConfig.from_dict({"class_names": list(class_names), "components": components, **kwargs})


class StubMapFilter(MetricFilter):
    """Availability-gated filter: unavailable when the frame has no scene id, else keeps x >= 0."""

    requires_map = True
    requires_pose = False

    def __init__(self, name: str = "stub") -> None:
        self.name = name

    def available(self, context: FrameMetricContext) -> bool:
        return context.scene_id is not None

    def keep(self, elements: np.ndarray, context: FrameMetricContext) -> np.ndarray:
        elements = np.asarray(elements, dtype=np.float64)
        return elements[:, 0] >= 0.0
