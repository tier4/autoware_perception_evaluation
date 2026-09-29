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

"""Closed registry from configuration ``type`` tokens to detection metric component classes.

Components are never instantiated from arbitrary class paths; only the tokens listed in
:data:`COMPONENT_TOKENS` are accepted. Map/TTC-dependent components are imported lazily so the
map-free metrics stay importable without the reachability stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict
from typing import Tuple

COMPONENT_TOKENS: Dict[str, Tuple[str, str]] = {
    "corner_error": ("perception_eval.evaluation.metrics.detection.corner_error", "CornerError"),
    "heading_flip": ("perception_eval.evaluation.metrics.detection.heading_flip", "HeadingFlipRate"),
    "nearest_surface_error": (
        "perception_eval.evaluation.metrics.detection.nearest_surface_error",
        "NearestSurfaceError",
    ),
    "calibration": ("perception_eval.evaluation.metrics.detection.calibration", "CalibrationError"),
    "confident_error": ("perception_eval.evaluation.metrics.detection.confident_error", "ConfidentErrorRate"),
    "confusion_matrix": ("perception_eval.evaluation.metrics.detection.confusion_matrix", "ConfusionMatrix"),
    "critical_fp_fn": ("perception_eval.evaluation.metrics.detection.critical_fp_fn", "CriticalFPFN"),
    "collision_weighted_map": (
        "perception_eval.evaluation.metrics.detection.collision_weighted_map",
        "CollisionWeightedMeanAP",
    ),
}
"""Token to (module, class name)."""

TTC_COMPONENT_TOKENS = frozenset({"critical_fp_fn", "collision_weighted_map"})
"""Tokens whose component reads per-box TTC (needs a collision provider and a lanelet map)."""


def get_component_class(token: str) -> type:
    """Resolve a configuration token to its component class (fail-loud on unknown tokens)."""
    if token not in COMPONENT_TOKENS:
        raise ValueError(f"Unknown component type {token!r}, expected one of {sorted(COMPONENT_TOKENS)}.")
    module_name, class_name = COMPONENT_TOKENS[token]
    try:
        module = import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"component {token!r} is unavailable: {module_name} could not be imported ({error})."
        ) from error
    return getattr(module, class_name)


class _LazyComponentClasses(dict):
    """Dict view over :func:`get_component_class` that resolves tokens on access."""

    def __missing__(self, token: str) -> type:
        cls = get_component_class(token)
        self[token] = cls
        return cls

    def __contains__(self, token: object) -> bool:
        return token in COMPONENT_TOKENS

    def keys(self):  # noqa: D102
        return COMPONENT_TOKENS.keys()

    def __iter__(self):
        return iter(COMPONENT_TOKENS)

    def __len__(self) -> int:
        return len(COMPONENT_TOKENS)


COMPONENT_CLASSES: Dict[str, type] = _LazyComponentClasses()
"""Token to component class, resolved lazily on first access."""
