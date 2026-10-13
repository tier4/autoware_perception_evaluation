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
# Ported from tier4/autoware-ml (autoware_ml/metrics/segmentation3d/point_cloud.py) at fcf86419.

"""Public input contract and derived per-frame state of the point-cloud segmentation metrics.

``SegmentationFrame`` is the model-independent input: one frame's points in base_link with their
target class, predicted class and per-class probabilities. From it the suite derives, once per
frame, the per-point uncertainty scalars (confidence of the reported class and normalized
predictive entropy) for the trained taxonomy and, when class groups are configured, for the
behaviour-group taxonomy (probabilities are folded *before* the uncertainty is derived, so the
grouped confidence is the summed probability mass of the predicted group).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.shape import ShapeType
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.metrics.class_groups import fold_labels
from perception_eval.evaluation.metrics.class_groups import fold_matrix

RAW_TAXONOMY = "raw"
ENTROPY_EPSILON = 1e-12


@dataclass(frozen=True)
class SegmentationFrame:
    """Per-point segmentation input for one frame.

    Attributes:
        frame_name (str): Frame name.
        scene_id (str | None): Scene identifier used to resolve a lanelet map, or ``None``.
        coordinates (NDArray): ``(N, 3+)`` point coordinates in base_link.
        targets (NDArray): ``(N,)`` integer ground-truth class per point (``ignore_index`` allowed).
        predictions (NDArray): ``(N,)`` integer predicted class per point.
        probabilities (NDArray): ``(N, C)`` per-class probabilities (rows sum to one).
        transforms (TransformDict | None): Frame transforms; ``(BASE_LINK, MAP)`` gives the ego pose.
            ``None`` is stored as an empty ``TransformDict``.
        gt_objects (tuple[DynamicObject, ...]): Detection ground-truth boxes for the partial-detection metric.
        unix_time (int | None): Optional timestamp passthrough.
    """

    frame_name: str
    scene_id: Optional[str]
    coordinates: NDArray
    targets: NDArray
    predictions: NDArray
    probabilities: NDArray
    transforms: Optional[TransformDict] = None
    gt_objects: Tuple[DynamicObject, ...] = ()
    unix_time: Optional[int] = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates)
        targets = np.asarray(self.targets)
        predictions = np.asarray(self.predictions)
        probabilities = np.asarray(self.probabilities)
        if coordinates.ndim != 2 or coordinates.shape[1] < 3:
            raise ValueError(f"coordinates must have shape (N, 3+), got {coordinates.shape}.")
        num_points = coordinates.shape[0]
        for name, array in (("targets", targets), ("predictions", predictions)):
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"{name} must have an integer dtype, got {array.dtype}.")
            if array.shape[0] != num_points:
                raise ValueError(f"{name} has {array.shape[0]} entries for {num_points} points.")
        if probabilities.ndim != 2:
            raise ValueError(f"probabilities must have shape (N, C), got {probabilities.shape}.")
        if probabilities.shape[0] != num_points:
            raise ValueError(f"probabilities has {probabilities.shape[0]} rows for {num_points} points.")
        transforms = self.transforms if self.transforms is not None else TransformDict()
        if not isinstance(transforms, TransformDict):
            raise TypeError(f"transforms must be a TransformDict or None, got {type(transforms).__name__}.")
        gt_objects = tuple(self.gt_objects)
        for obj in gt_objects:
            if not isinstance(obj, DynamicObject):
                raise TypeError(f"gt_objects must contain DynamicObject, got {type(obj).__name__}.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "targets", targets.astype(np.int64, copy=False))
        object.__setattr__(self, "predictions", predictions.astype(np.int64, copy=False))
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "transforms", transforms)
        object.__setattr__(self, "gt_objects", gt_objects)

    @property
    def num_points(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.probabilities.shape[1])

    @property
    def ego2map(self) -> Optional[NDArray]:
        """4x4 base_link -> map matrix, or ``None`` when the frame carries no ego pose."""
        matrix = self.transforms.get((FrameID.BASE_LINK, FrameID.MAP))
        if matrix is None:
            inverse = self.transforms.get((FrameID.MAP, FrameID.BASE_LINK))
            if inverse is None:
                return None
            matrix = inverse.inv()
        return np.asarray(matrix.matrix, dtype=np.float64)


def normalized_entropy(probabilities: NDArray) -> NDArray[np.float64]:
    """Per-row Shannon entropy ``-sum(p log p) / log(C)`` in ``[0, 1]`` (zero terms treated as zero)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape (N, C) with C >= 2 for normalized entropy.")
    clipped = np.clip(probabilities, ENTROPY_EPSILON, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return entropy / np.log(probabilities.shape[1])


def confidence_of(probabilities: NDArray, predictions: NDArray) -> NDArray[np.float64]:
    """Probability of the reported prediction per row (``0`` where the prediction is out of range)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    confidence = np.zeros(predictions.shape[0], dtype=np.float64)
    in_range = (predictions >= 0) & (predictions < probabilities.shape[1])
    rows = np.flatnonzero(in_range)
    confidence[rows] = probabilities[rows, predictions[rows]]
    return confidence


def valid_point_mask(targets: NDArray, predictions: NDArray, num_classes: int, ignore_index: int) -> NDArray[np.bool_]:
    """Mask of points with an in-range target and prediction (ignore excluded)."""
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    return (
        (targets != ignore_index)
        & (targets >= 0)
        & (targets < num_classes)
        & (predictions >= 0)
        & (predictions < num_classes)
    )


@dataclass(frozen=True)
class TaxonomyFrame:
    """One frame's labels and uncertainty scalars in one taxonomy (trained classes or groups)."""

    name: str
    pred: NDArray[np.int64]
    target: NDArray[np.int64]
    valid: NDArray[np.bool_]
    confidence: NDArray[np.float64]
    entropy: NDArray[np.float64]
    num_classes: int
    class_names: Tuple[str, ...]


def build_raw_taxonomy(frame: SegmentationFrame, class_names: Sequence[str], ignore_index: int) -> TaxonomyFrame:
    """Trained-class view: confidence is ``p[pred]``, entropy is over the full distribution."""
    num_classes = len(class_names)
    probabilities = np.asarray(frame.probabilities, dtype=np.float64)
    return TaxonomyFrame(
        name=RAW_TAXONOMY,
        pred=frame.predictions,
        target=frame.targets,
        valid=valid_point_mask(frame.targets, frame.predictions, num_classes, ignore_index),
        confidence=confidence_of(probabilities, frame.predictions),
        entropy=normalized_entropy(probabilities) if frame.num_points else np.zeros(0, dtype=np.float64),
        num_classes=num_classes,
        class_names=tuple(class_names),
    )


def build_grouped_taxonomy(
    frame: SegmentationFrame,
    lut: NDArray[np.int64],
    grouped_names: Sequence[str],
    ignore_index: int,
    taxonomy_name: str = "grouped",
) -> TaxonomyFrame:
    """Behaviour-group view: probabilities are folded first, then confidence/entropy are derived."""
    num_grouped = len(grouped_names)
    probabilities = np.asarray(frame.probabilities, dtype=np.float64)
    folded = probabilities @ fold_matrix(lut, num_grouped) if frame.num_points else np.zeros((0, num_grouped))
    pred = fold_labels(frame.predictions, lut)
    target = fold_labels(frame.targets, lut)
    return TaxonomyFrame(
        name=taxonomy_name,
        pred=pred,
        target=target,
        valid=valid_point_mask(target, pred, num_grouped, ignore_index),
        confidence=confidence_of(folded, pred),
        entropy=normalized_entropy(folded) if frame.num_points else np.zeros(0, dtype=np.float64),
        num_classes=num_grouped,
        class_names=tuple(grouped_names),
    )


@dataclass(frozen=True)
class FrameGeometry:
    """Taxonomy-independent geometry of one frame."""

    xyz: NDArray[np.float64]  # (N, 3) base_link
    bev_radius: NDArray[np.float64]  # (N,)
    gt_boxes: NDArray[np.float64]  # (M, 9) base_link rows
    gt_box_seg_class: NDArray[np.int64]  # (M,) segmentation class index, -1 when unmapped
    gt_box_det_label: NDArray[np.int64]  # (M,) index into the detection class list, -1 when unmapped

    @property
    def box_radius(self) -> NDArray[np.float64]:
        if self.gt_boxes.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        return np.linalg.norm(self.gt_boxes[:, :2], axis=1)


@dataclass(frozen=True)
class SegmentationView:
    """Masked arrays of one frame for one ``(taxonomy, filter, range)`` view.

    Only valid points that pass the filter and range masks are included; boxes are those whose
    footprint passes the filter and whose center lies in the range.
    """

    taxonomy: str
    filter_name: str
    range_name: Optional[str]
    frame_name: str
    pred: NDArray[np.int64]
    target: NDArray[np.int64]
    confidence: NDArray[np.float64]
    entropy: NDArray[np.float64]
    xyz: NDArray[np.float64]
    gt_boxes: NDArray[np.float64]
    gt_box_seg_class: NDArray[np.int64]
    gt_box_det_label: NDArray[np.int64]
    num_classes: int
    class_names: Tuple[str, ...]

    @property
    def num_points(self) -> int:
        return int(self.pred.shape[0])

    @property
    def wrong(self) -> NDArray[np.bool_]:
        return self.pred != self.target


@dataclass(frozen=True)
class SegmentationFrameSummary:
    """Small per-frame record returned by the streaming manager (no arrays retained)."""

    frame_name: str
    scene_id: Optional[str]
    num_points: int
    num_valid: int
    num_errors: int
    filter_available: Dict[str, bool] = field(default_factory=dict)


def validate_frame(frame: SegmentationFrame, config: Any) -> None:
    """Configuration-dependent validation of a frame (probabilities, argmax, class count, boxes).

    Args:
        frame (SegmentationFrame): The frame to validate.
        config: A ``SegmentationMetricsConfig`` (duck-typed: ``class_names``, ``check_argmax``,
            ``probability_tolerance``, ``probability_sum_tolerance``, ``needs_boxes``).
    """
    num_classes = len(config.class_names)
    if frame.num_classes != num_classes:
        raise ValueError(
            f"frame {frame.frame_name!r}: probabilities carry {frame.num_classes} columns for "
            f"{num_classes} configured classes."
        )
    probabilities = np.asarray(frame.probabilities, dtype=np.float64)
    if frame.num_points:
        if not np.all(np.isfinite(probabilities)):
            raise ValueError(f"frame {frame.frame_name!r}: probabilities must be finite.")
        tolerance = float(config.probability_tolerance)
        if float(probabilities.min()) < -tolerance or float(probabilities.max()) > 1.0 + tolerance:
            raise ValueError(
                f"frame {frame.frame_name!r}: probabilities must lie in [0, 1] (tolerance {tolerance:g}); "
                "pass probabilities, not logits."
            )
        row_sums = probabilities.sum(axis=1)
        sum_tolerance = float(config.probability_sum_tolerance)
        if np.any(np.abs(row_sums - 1.0) > sum_tolerance):
            worst = float(np.max(np.abs(row_sums - 1.0)))
            raise ValueError(
                f"frame {frame.frame_name!r}: probability rows must sum to one (tolerance {sum_tolerance:g}, "
                f"worst deviation {worst:g})."
            )
        if config.check_argmax:
            argmax = probabilities.argmax(axis=1)
            mismatch = frame.predictions != argmax
            if bool(mismatch.any()):
                raise ValueError(
                    f"frame {frame.frame_name!r}: {int(mismatch.sum())} predictions differ from "
                    "argmax(probabilities); set check_argmax=False if the reported class is chosen otherwise."
                )
    if getattr(config, "needs_boxes", False):
        for obj in frame.gt_objects:
            shape = obj.state.shape
            if shape is None or shape.type != ShapeType.BOUNDING_BOX:
                raise ValueError(
                    f"frame {frame.frame_name!r}: partial_detection needs BOUNDING_BOX ground truth, "
                    f"got {None if shape is None else shape.type}."
                )
