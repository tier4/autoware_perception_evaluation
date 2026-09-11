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

"""Hand-built detection states shared by the component tests."""

from __future__ import annotations

from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from perception_eval.evaluation.metrics.detection.state import DetectionSample
from perception_eval.evaluation.metrics.detection.state import DetectionState


def box(cx: float = 10.0, cy: float = 0.0, dx: float = 4.0, dy: float = 2.0, yaw: float = 0.0) -> List[float]:
    return [cx, cy, 0.0, dx, dy, 1.5, yaw, 0.0, 0.0]


def sample(
    preds: Sequence[Sequence[float]],
    scores: Sequence[float],
    pred_labels: Sequence[int],
    gts: Sequence[Sequence[float]],
    gt_labels: Sequence[int],
) -> DetectionSample:
    return DetectionSample(
        pred_boxes=np.array(preds, dtype=np.float64).reshape(-1, 9),
        pred_scores=np.array(scores, dtype=np.float64),
        pred_labels=np.array(pred_labels, dtype=np.int64),
        gt_boxes=np.array(gts, dtype=np.float64).reshape(-1, 9),
        gt_labels=np.array(gt_labels, dtype=np.int64),
    )


def yaw_state(yaw_err: float, match_cost: str = "center") -> DetectionState:
    """One car GT and one prediction differing only by yaw."""
    return DetectionState(
        samples=[sample([box(yaw=yaw_err)], [0.9], [0], [box()], [0])],
        class_names=("car",),
        match_cost=match_cost,
    )


def tp_fp_state(class_names: Tuple[str, ...] = ("car",)) -> DetectionState:
    """One TP (score 0.9) on the GT and one FP (score 0.8) 100 m away."""
    return DetectionState(
        samples=[sample([box(0.0), box(100.0)], [0.9, 0.8], [0, 0], [box(0.0)], [0])],
        class_names=class_names,
    )


def empty_state(class_names: Optional[Tuple[str, ...]] = ("car",)) -> DetectionState:
    return DetectionState(samples=[DetectionSample.empty()], class_names=class_names)


def no_frames_state() -> DetectionState:
    return DetectionState(samples=[], class_names=("car",))
