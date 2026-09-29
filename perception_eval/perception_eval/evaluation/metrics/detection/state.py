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
#
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/structures.py, matching.py) at fcf86419.

"""Per-frame detection samples and the memoizing state handed to the advanced metric components.

``DetectionSample`` is the per-frame input (NumPy box rows in base_link). ``DetectionState`` owns
a list of samples for one ``(taxonomy, filter, range)`` view and memoizes the score-ordered greedy
matching per ``(label, threshold)`` so every component shares one matching pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.component import MetricRange
from perception_eval.evaluation.metrics.detection.geometry import validate_boxes

ERROR_NAMES = ("ATE", "AOE", "ASE", "AVE", "AAE")

# nuScenes center-distance AP thresholds (meters) and the single TP operating point.
DEFAULT_MATCH_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_TP_THRESHOLD = 2.0

MATCH_COSTS = ("center", "corner")


def _empty_boxes() -> NDArray[np.float64]:
    return np.zeros((0, 9), dtype=np.float64)


@dataclass(frozen=True)
class DetectionSample:
    """Prediction and ground-truth box rows for one detection frame (base_link).

    Rows are ``[cx, cy, cz, dx(length), dy(width), dz(height), yaw, vx, vy]``. ``gt_ttc`` /
    ``pred_ttc`` are the per-box reachability time-to-collision to ego in seconds (``inf`` means
    unreachable), present only when a collision provider ran, and read by the criticality metrics
    (B1, B2). ``ttc_covered`` is False for frames whose scene has no lanelet map: the criticality
    metrics exclude such frames from their denominators.
    """

    pred_boxes: NDArray[np.float64]
    pred_scores: NDArray[np.float64]
    pred_labels: NDArray[np.int64]
    gt_boxes: NDArray[np.float64]
    gt_labels: NDArray[np.int64]
    gt_ttc: Optional[NDArray[np.float64]] = None
    pred_ttc: Optional[NDArray[np.float64]] = None
    ttc_covered: bool = True
    frame_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "pred_boxes", validate_boxes(self.pred_boxes, "pred_boxes"))
        object.__setattr__(self, "gt_boxes", validate_boxes(self.gt_boxes, "gt_boxes"))
        object.__setattr__(self, "pred_scores", np.asarray(self.pred_scores, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "pred_labels", np.asarray(self.pred_labels, dtype=np.int64).reshape(-1))
        object.__setattr__(self, "gt_labels", np.asarray(self.gt_labels, dtype=np.int64).reshape(-1))
        if self.gt_ttc is not None:
            object.__setattr__(self, "gt_ttc", np.asarray(self.gt_ttc, dtype=np.float64).reshape(-1))
        if self.pred_ttc is not None:
            object.__setattr__(self, "pred_ttc", np.asarray(self.pred_ttc, dtype=np.float64).reshape(-1))
        num_pred, num_gt = self.pred_boxes.shape[0], self.gt_boxes.shape[0]
        if self.pred_scores.shape[0] != num_pred or self.pred_labels.shape[0] != num_pred:
            raise ValueError("pred_boxes, pred_scores and pred_labels must have the same length.")
        if self.gt_labels.shape[0] != num_gt:
            raise ValueError("gt_boxes and gt_labels must have the same length.")
        if self.pred_ttc is not None and self.pred_ttc.shape[0] != num_pred:
            raise ValueError("pred_ttc must align with pred_boxes.")
        if self.gt_ttc is not None and self.gt_ttc.shape[0] != num_gt:
            raise ValueError("gt_ttc must align with gt_boxes.")

    @property
    def num_predictions(self) -> int:
        return int(self.pred_boxes.shape[0])

    @property
    def num_ground_truths(self) -> int:
        return int(self.gt_boxes.shape[0])

    @property
    def has_ttc(self) -> bool:
        return self.gt_ttc is not None and self.pred_ttc is not None

    def select(self, pred_keep: NDArray[np.bool_], gt_keep: NDArray[np.bool_]) -> DetectionSample:
        """Return a new sample with predictions and GT sliced by boolean masks."""
        pred_keep = np.asarray(pred_keep, dtype=bool)
        gt_keep = np.asarray(gt_keep, dtype=bool)
        return DetectionSample(
            pred_boxes=self.pred_boxes[pred_keep],
            pred_scores=self.pred_scores[pred_keep],
            pred_labels=self.pred_labels[pred_keep],
            gt_boxes=self.gt_boxes[gt_keep],
            gt_labels=self.gt_labels[gt_keep],
            gt_ttc=None if self.gt_ttc is None else self.gt_ttc[gt_keep],
            pred_ttc=None if self.pred_ttc is None else self.pred_ttc[pred_keep],
            ttc_covered=self.ttc_covered,
            frame_name=self.frame_name,
        )

    def fold_labels(self, lut: NDArray[np.int64]) -> DetectionSample:
        """Return a new sample with labels folded through the class-group LUT."""
        from perception_eval.evaluation.metrics.class_groups import fold_labels

        return DetectionSample(
            pred_boxes=self.pred_boxes,
            pred_scores=self.pred_scores,
            pred_labels=fold_labels(self.pred_labels, lut),
            gt_boxes=self.gt_boxes,
            gt_labels=fold_labels(self.gt_labels, lut),
            gt_ttc=self.gt_ttc,
            pred_ttc=self.pred_ttc,
            ttc_covered=self.ttc_covered,
            frame_name=self.frame_name,
        )

    def clip_to_range(self, metric_range: MetricRange) -> DetectionSample:
        """Clip both GT and predictions to a radial BEV distance window on box centers."""
        return self.select(
            metric_range.contains(np.linalg.norm(self.pred_boxes[:, :2], axis=1)),
            metric_range.contains(np.linalg.norm(self.gt_boxes[:, :2], axis=1)),
        )

    @classmethod
    def empty(cls, frame_name: str = "") -> DetectionSample:
        return cls(
            pred_boxes=_empty_boxes(),
            pred_scores=np.zeros(0),
            pred_labels=np.zeros(0, dtype=np.int64),
            gt_boxes=_empty_boxes(),
            gt_labels=np.zeros(0, dtype=np.int64),
            frame_name=frame_name,
        )


@dataclass(frozen=True)
class MatchCurve:
    """Per-class, per-threshold score-ordered matching results.

    ``corner_error`` and ``nearest_surface_error`` are per-prediction arrays (``NaN`` for
    non-matches), filled for true positives alongside the nuScenes error channels so the
    driving-aware components read them without re-matching.
    """

    total_gt: int
    scores: NDArray[np.float64]
    true_positive: NDArray[np.float64]
    false_positive: NDArray[np.float64]
    heading_score: NDArray[np.float64]
    translation_error: NDArray[np.float64]
    orientation_error: NDArray[np.float64]
    scale_error: NDArray[np.float64]
    velocity_error: NDArray[np.float64]
    attribute_error: NDArray[np.float64]
    corner_error: NDArray[np.float64]
    nearest_surface_error: NDArray[np.float64]

    @property
    def num_predictions(self) -> int:
        return int(self.scores.shape[0])

    @property
    def num_match(self) -> int:
        return int(np.sum(self.true_positive))

    @property
    def cumulative_tp(self) -> NDArray[np.float64]:
        return np.cumsum(self.true_positive)

    @property
    def cumulative_fp(self) -> NDArray[np.float64]:
        return np.cumsum(self.false_positive)

    @property
    def cumulative_heading_tp(self) -> NDArray[np.float64]:
        return np.cumsum(self.heading_score)


@dataclass(frozen=True)
class CurveMetrics:
    """AP-style summary values derived from one match curve."""

    ap: float
    aph: float
    max_f1: float
    optimal_conf: float
    optimal_index: int
    optimal_recall: float
    optimal_precision: float


def labels_to_evaluate(samples: Sequence[DetectionSample], class_names: Optional[Sequence[str]]) -> List[int]:
    """All class indices when ``class_names`` is given, else only labels present in the GT."""
    if class_names is not None:
        return list(range(len(class_names)))
    if not samples:
        return []
    labels = np.unique(np.concatenate([sample.gt_labels.reshape(-1) for sample in samples]))
    return [int(label) for label in labels.tolist() if int(label) >= 0]


@dataclass
class DetectionState:
    """Detection state handed to each metric component for one view.

    Attributes:
        samples (list[DetectionSample]): Per-frame samples (already filtered/range-clipped).
        class_names (tuple[str, ...] | None): Ordered class names for metric keys, or ``None``.
        match_cost (str): Matching cost name (``center`` or ``corner``).
    """

    samples: List[DetectionSample]
    class_names: Optional[Tuple[str, ...]]
    match_cost: str = "center"
    _curve_cache: Dict[Tuple[int, float], MatchCurve] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _label_cache: Dict[int, object] = field(default_factory=dict, init=False, repr=False, compare=False)
    _labels_cache: Dict[bool, List[int]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _metrics_cache: Dict[Tuple[int, float], CurveMetrics] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.match_cost not in MATCH_COSTS:
            raise ValueError(f"Unknown match cost {self.match_cost!r}, expected one of {list(MATCH_COSTS)}.")
        self.samples = list(self.samples)
        if self.class_names is not None:
            self.class_names = tuple(self.class_names)

    @property
    def num_frames(self) -> int:
        return len(self.samples)

    def labels(self, full: bool = True) -> List[int]:
        """Class labels to report; every configured class when ``full`` else only GT-present ones."""
        cached = self._labels_cache.get(bool(full))
        if cached is None:
            cached = labels_to_evaluate(self.samples, self.class_names if full else None)
            self._labels_cache[bool(full)] = cached
        return cached

    def match_curve(self, label: int, threshold: float) -> MatchCurve:
        """Score-ordered match curve for a class and threshold, memoized."""
        from perception_eval.evaluation.metrics.detection.matching import PreparedLabel

        key = (int(label), float(threshold))
        curve = self._curve_cache.get(key)
        if curve is None:
            prepared = self._label_cache.get(int(label))
            if prepared is None:
                prepared = PreparedLabel(self.samples, int(label), self.match_cost)
                self._label_cache[int(label)] = prepared
            curve = prepared.match(float(threshold))
            self._curve_cache[key] = curve
        return curve

    def curve_metrics(self, label: int, threshold: float) -> CurveMetrics:
        """AP/APH/F1 summary of a match curve, memoized like the curve itself."""
        from perception_eval.evaluation.metrics.detection.matching import curve_metrics

        key = (int(label), float(threshold))
        metrics = self._metrics_cache.get(key)
        if metrics is None:
            metrics = curve_metrics(self.match_curve(label, threshold))
            self._metrics_cache[key] = metrics
        return metrics
