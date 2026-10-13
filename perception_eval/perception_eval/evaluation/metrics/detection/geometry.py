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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/geometry.py) at fcf86419.

"""Pure box geometry shared by the driving-aware detection metrics.

These helpers operate on boxes in the ego/base_link frame, laid out as
``[cx, cy, cz, dx, dy, dz, yaw, ...]`` with ``dx`` the length along the heading and ``dy`` the
width. Everything here is stateless NumPy math so the metric components, the filters and the
matching core share one definition of corners, corner displacement, and the nearest box surface.
The single-box functions are the readable reference form; the ``*_batch`` / paired forms are their
broadcast equivalents used on the matching hot path.
"""

from __future__ import annotations

from math import pi

import numpy as np
from numpy.typing import NDArray

BOX_DIM = 9
"""Columns of a box row: cx, cy, cz, dx(length), dy(width), dz(height), yaw, vx, vy."""


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (float(angle) + pi) % (2.0 * pi) - pi


def validate_boxes(boxes: NDArray, name: str = "boxes") -> NDArray[np.float64]:
    """Return ``boxes`` as a float64 ``(N, 7+)`` array, raising on any other shape."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"{name} must have shape (N, 7+) but got {tuple(boxes.shape)}.")
    return boxes


def bev_corners(box: NDArray) -> NDArray[np.float64]:
    """Return the four BEV corners of a box as a ``(4, 2)`` array.

    Corners are ordered starting from the ``(+dx/2, +dy/2)`` corner in the box frame, then
    ``(+dx/2, -dy/2)``, ``(-dx/2, -dy/2)``, ``(-dx/2, +dy/2)``, rotated by ``yaw`` and shifted to
    the center.
    """
    box = np.asarray(box, dtype=np.float64)
    center = box[:2]
    half = box[3:5] / 2.0
    yaw = float(box[6])
    local = np.array(
        [
            [half[0], half[1]],
            [half[0], -half[1]],
            [-half[0], -half[1]],
            [-half[0], half[1]],
        ],
        dtype=np.float64,
    )
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    return local @ rotation.T + center


def corner_displacement(pred_box: NDArray, gt_box: NDArray) -> float:
    """Mean BEV corner distance under the best cyclic corner assignment.

    Couples position, size, and yaw into one distance in meters. The cyclic ``argmin`` over the
    four rotations keeps a box's 90-degree parameterization ambiguity from being punished as a
    gross error.
    """
    pred_corners = bev_corners(pred_box)
    gt_corners = bev_corners(gt_box)
    best = np.inf
    for shift in range(4):
        rolled = np.roll(pred_corners, shift, axis=0)
        mean_distance = float(np.mean(np.linalg.norm(rolled - gt_corners, axis=1)))
        best = min(best, mean_distance)
    return best


def nearest_surface_distance(box: NDArray) -> float:
    """Distance from the ego origin to the nearest point of the box's BEV outline.

    The ego origin is the frame origin ``(0, 0)``. Returns ``0.0`` when the origin lies inside the
    footprint (unphysical for a real object, but kept well defined).
    """
    box = np.asarray(box, dtype=np.float64)
    center = box[:2]
    half = box[3:5] / 2.0
    yaw = float(box[6])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    # Express the origin in the box-local frame: rotate (-yaw) about the center.
    offset = -center
    local = np.array(
        [
            offset[0] * cos_yaw + offset[1] * sin_yaw,
            -offset[0] * sin_yaw + offset[1] * cos_yaw,
        ],
        dtype=np.float64,
    )
    clamped = np.clip(local, -half, half)
    return float(np.linalg.norm(local - clamped))


def signed_nearest_surface_error(pred_box: NDArray, gt_box: NDArray) -> float:
    """Signed near-face error ``d_pred - d_gt`` in meters.

    Positive means the predicted near face sits farther than the truth (ego would brake late),
    negative means nearer (over-caution).
    """
    return nearest_surface_distance(pred_box) - nearest_surface_distance(gt_box)


def bev_corners_batch(boxes: NDArray) -> NDArray[np.float64]:
    """:func:`bev_corners` for ``(N, 7+)`` boxes at once, as ``(N, 4, 2)``."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.zeros((0, 4, 2), dtype=np.float64)
    half_dx, half_dy = boxes[:, 3] / 2.0, boxes[:, 4] / 2.0
    local_x = np.stack([half_dx, half_dx, -half_dx, -half_dx], axis=1)
    local_y = np.stack([half_dy, -half_dy, -half_dy, half_dy], axis=1)
    cos_yaw = np.cos(boxes[:, 6])[:, None]
    sin_yaw = np.sin(boxes[:, 6])[:, None]
    corners_x = local_x * cos_yaw - local_y * sin_yaw + boxes[:, 0:1]
    corners_y = local_x * sin_yaw + local_y * cos_yaw + boxes[:, 1:2]
    return np.stack([corners_x, corners_y], axis=2)


def corner_displacements(pred_boxes: NDArray, gt_boxes: NDArray) -> NDArray[np.float64]:
    """:func:`corner_displacement` for paired boxes, ``(N, 7+)`` x ``(N, 7+)`` to ``(N,)``."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    if pred_boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    pred_corners = bev_corners_batch(pred_boxes)
    gt_corners = bev_corners_batch(gt_boxes)
    best = np.full(pred_boxes.shape[0], np.inf, dtype=np.float64)
    for shift in range(4):
        rolled = np.roll(pred_corners, shift, axis=1)
        mean_distance = np.mean(np.linalg.norm(rolled - gt_corners, axis=2), axis=1)
        best = np.minimum(best, mean_distance)
    return best


def corner_displacement_matrix(pred_boxes: NDArray, gt_boxes: NDArray) -> NDArray[np.float64]:
    """:func:`corner_displacement` for every pair, ``(P, 7+)`` x ``(G, 7+)`` to ``(P, G)``."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float64)
    pred_corners = bev_corners_batch(pred_boxes)  # (P, 4, 2)
    gt_corners = bev_corners_batch(gt_boxes)  # (G, 4, 2)
    best = np.full((pred_boxes.shape[0], gt_boxes.shape[0]), np.inf, dtype=np.float64)
    for shift in range(4):
        rolled = np.roll(pred_corners, shift, axis=1)[:, None]  # (P, 1, 4, 2)
        mean_distance = np.mean(np.linalg.norm(rolled - gt_corners[None], axis=3), axis=2)
        best = np.minimum(best, mean_distance)
    return best


def nearest_surface_distances(boxes: NDArray) -> NDArray[np.float64]:
    """:func:`nearest_surface_distance` for ``(N, 7+)`` boxes at once, as ``(N,)``."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    half = boxes[:, 3:5] / 2.0
    cos_yaw, sin_yaw = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    offset = -boxes[:, :2]
    local = np.stack(
        [
            offset[:, 0] * cos_yaw + offset[:, 1] * sin_yaw,
            -offset[:, 0] * sin_yaw + offset[:, 1] * cos_yaw,
        ],
        axis=1,
    )
    clamped = np.clip(local, -half, half)
    return np.linalg.norm(local - clamped, axis=1)


def points_in_bev_box(coord_xy: NDArray, box: NDArray) -> NDArray[np.bool_]:
    """Boolean mask of the given ``(N, 2)`` points inside the yaw-aware BEV footprint of ``box``."""
    coord_xy = np.asarray(coord_xy, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    if coord_xy.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    center = box[:2]
    half = box[3:5] / 2.0
    yaw = float(box[6])
    offset = coord_xy - center
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    local_x = offset[:, 0] * cos_yaw + offset[:, 1] * sin_yaw
    local_y = -offset[:, 0] * sin_yaw + offset[:, 1] * cos_yaw
    return (np.abs(local_x) <= half[0]) & (np.abs(local_y) <= half[1])


def clip_forward(corners: NDArray) -> NDArray[np.float64]:
    """Clip a convex BEV polygon to the forward half-plane ``x >= 0``."""
    corners = np.asarray(corners, dtype=np.float64)
    clipped: list = []
    count = corners.shape[0]
    for index in range(count):
        current = corners[index]
        following = corners[(index + 1) % count]
        if current[0] >= 0.0:
            clipped.append(current)
        if (current[0] >= 0.0) != (following[0] >= 0.0):
            t = current[0] / (current[0] - following[0])
            clipped.append(current + t * (following - current))
    return np.asarray(clipped, dtype=np.float64) if clipped else np.zeros((0, 2), dtype=np.float64)
