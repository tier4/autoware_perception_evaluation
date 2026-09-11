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

"""Adapters from ``DynamicObject`` to the base_link box rows the advanced metrics consume.

Every object is normalized to base_link exactly once. The local shape convention
``Shape.size == (width, length, height)`` is translated explicitly into the row convention
``[cx, cy, cz, dx=length, dy=width, dz=height, yaw, vx, vy]``; copying the tuple would rotate the
dimensions and corrupt corner errors, nearest-surface errors, region overlap and TTC body radii.
"""

from __future__ import annotations

from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.common.label import Label
from perception_eval.common.label import LabelType
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.shape import ShapeType
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.metrics.detection.geometry import BOX_DIM

VELOCITY_IS_OBJECT_FRAME = True
"""Dataset ground truth stores ``velocity`` in the object frame (``dataset_utils._sample_to_frame``).

The adapter rotates it by the object yaw into base_link. Flip this switch if estimations carry a
base_link twist instead; only the reachability-based metrics read the velocity columns.
"""


class UnsupportedShapeError(ValueError):
    """Raised for objects whose shape is not an axis-defined bounding box."""


class MissingTransformError(KeyError):
    """Raised when an object is not in base_link and no transform to base_link is available."""


def class_names_of(target_labels: Sequence[LabelType]) -> Tuple[str, ...]:
    """Metric-key class names (``AutowareLabel.value``) in ``target_labels`` order."""
    return tuple(str(label.value) for label in target_labels)


def label_index(label: Label, target_labels: Sequence[LabelType]) -> int:
    """Index of ``label`` in ``target_labels``, or ``-1`` when it is not a target."""
    try:
        return list(target_labels).index(label.label)
    except ValueError:
        return -1


def object_to_base_link_row(obj: DynamicObject, transforms: Optional[TransformDict]) -> NDArray[np.float64]:
    """Convert one object to ``[cx, cy, cz, dx, dy, dz, yaw, vx, vy]`` in base_link.

    Raises:
        UnsupportedShapeError: the shape is not ``ShapeType.BOUNDING_BOX``.
        MissingTransformError: the object is not in base_link and no transform is available.
    """
    if not isinstance(obj, DynamicObject):
        raise UnsupportedShapeError(f"advanced detection metrics need 3D DynamicObject, got {type(obj).__name__}")
    shape = obj.state.shape
    if shape is None or shape.type != ShapeType.BOUNDING_BOX:
        shape_type = None if shape is None else shape.type
        raise UnsupportedShapeError(f"advanced detection metrics require ShapeType.BOUNDING_BOX, got {shape_type}")

    position = np.asarray(obj.state.position, dtype=np.float64)
    orientation = obj.state.orientation
    if obj.frame_id != FrameID.BASE_LINK:
        if transforms is None:
            raise MissingTransformError(f"object is in {obj.frame_id} but no transforms were given")
        try:
            position, orientation = transforms.transform((obj.frame_id, FrameID.BASE_LINK), position, orientation)
        except KeyError as error:
            raise MissingTransformError(f"no transform {obj.frame_id}->{FrameID.BASE_LINK}: {error}") from error
        position = np.asarray(position, dtype=np.float64)

    yaw = float(orientation.yaw_pitch_roll[0]) if orientation is not None else 0.0
    width, length, height = (float(value) for value in shape.size)
    vx, vy = _velocity_xy(obj.state.velocity, yaw)
    return np.array(
        [position[0], position[1], position[2], length, width, height, yaw, vx, vy],
        dtype=np.float64,
    )


def _velocity_xy(velocity: Optional[Sequence[float]], yaw: float) -> Tuple[float, float]:
    if velocity is None:
        return 0.0, 0.0
    vx, vy = float(velocity[0]), float(velocity[1])
    if not np.isfinite(vx) or not np.isfinite(vy):
        return 0.0, 0.0
    if not VELOCITY_IS_OBJECT_FRAME:
        return vx, vy
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    return float(vx * cos_yaw - vy * sin_yaw), float(vx * sin_yaw + vy * cos_yaw)


def objects_to_box_rows(objects: Sequence[DynamicObject], transforms: Optional[TransformDict]) -> NDArray[np.float64]:
    """Box rows ``(N, 9)`` for many objects (see :func:`object_to_base_link_row`)."""
    if len(objects) == 0:
        return np.zeros((0, BOX_DIM), dtype=np.float64)
    return np.stack([object_to_base_link_row(obj, transforms) for obj in objects])


def objects_to_arrays(
    objects: Sequence[DynamicObject],
    transforms: Optional[TransformDict],
    target_labels: Sequence[LabelType],
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Box rows, label indices (``target_labels`` order) and scores for many objects.

    Objects whose label is not a target are dropped (the manager filter normally removes them).
    """
    rows: List[NDArray] = []
    labels: List[int] = []
    scores: List[float] = []
    for obj in objects:
        index = label_index(obj.semantic_label, target_labels)
        if index < 0:
            continue
        rows.append(object_to_base_link_row(obj, transforms))
        labels.append(index)
        scores.append(float(obj.semantic_score))
    if not rows:
        return (
            np.zeros((0, BOX_DIM), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
        )
    return np.stack(rows), np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)
