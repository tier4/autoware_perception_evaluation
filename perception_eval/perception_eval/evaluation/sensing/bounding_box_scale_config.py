# Copyright 2026 TIER IV, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resolved bbox scale values, independent of sensing targets and thresholds."""

from dataclasses import dataclass

from perception_eval.util.math import get_bbox_scale


@dataclass(frozen=True)
class BoundingBoxScaleConfig:
    """Immutable scale endpoints; preserve the existing extrapolation and height rules.

    Validation of new user overrides belongs to the caller. Legacy common values
    must retain their existing acceptance rules.
    """

    box_scale_0m: float
    box_scale_100m: float

    def get_scale_factor(self, distance: float) -> float:
        """Return the existing linear scale, including extrapolation beyond 100 m."""
        return get_bbox_scale(distance, self.box_scale_0m, self.box_scale_100m)
