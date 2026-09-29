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

"""Closed registry from configuration ``type`` tokens to segmentation metric components.

Configuration never instantiates arbitrary classes: only the tokens below are accepted, and the
parameters of each entry are validated against the component constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import inspect
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from perception_eval.evaluation.metrics.filter_config import MetricsConfigError
from perception_eval.evaluation.metrics.segmentation.calibration import CalibrationError
from perception_eval.evaluation.metrics.segmentation.confident_error import ConfidentErrorRate
from perception_eval.evaluation.metrics.segmentation.confusion_matrix import ConfusionMatrix
from perception_eval.evaluation.metrics.segmentation.confusion_scores import Accuracy
from perception_eval.evaluation.metrics.segmentation.confusion_scores import IoU
from perception_eval.evaluation.metrics.segmentation.confusion_scores import PrecisionRecallF1
from perception_eval.evaluation.metrics.segmentation.entropy_auroc import UncertaintyUsefulness
from perception_eval.evaluation.metrics.segmentation.error_clusters import ErrorClusters
from perception_eval.evaluation.metrics.segmentation.partial_detection import PartialDetectionScore
from perception_eval.evaluation.metrics.segmentation.tolerant_error import NeighbourhoodTolerantErrorRate

COMPONENT_TYPES: Dict[str, type] = {
    "confusion_matrix": ConfusionMatrix,
    "iou": IoU,
    "accuracy": Accuracy,
    "precision_recall_f1": PrecisionRecallF1,
    "calibration": CalibrationError,
    "uncertainty_usefulness": UncertaintyUsefulness,
    "confident_error": ConfidentErrorRate,
    "error_clusters": ErrorClusters,
    "tolerant_error": NeighbourhoodTolerantErrorRate,
    "partial_detection": PartialDetectionScore,
}

# Constructor parameters injected by the suite rather than taken from the config entry.
_INJECTED_PARAMS = {"det_class_names"}


@dataclass(frozen=True)
class ComponentSpec:
    """One validated ``components`` entry: a registry token and its constructor parameters."""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in COMPONENT_TYPES:
            raise MetricsConfigError(
                f"unknown segmentation component type {self.type!r}; valid: {sorted(COMPONENT_TYPES)}."
            )
        allowed = {
            name
            for name in inspect.signature(COMPONENT_TYPES[self.type].__init__).parameters
            if name != "self" and name not in _INJECTED_PARAMS
        }
        unknown = sorted(set(self.params) - allowed)
        if unknown:
            raise MetricsConfigError(
                f"component {self.type!r}: unknown parameters {unknown}; allowed: {sorted(allowed)}."
            )
        object.__setattr__(self, "params", dict(self.params))
        # Fail loud on invalid values at configuration time.
        self.instantiate()

    @property
    def kind(self) -> str:
        return COMPONENT_TYPES[self.type].kind

    @property
    def supports_grouped(self) -> bool:
        return bool(getattr(COMPONENT_TYPES[self.type], "supports_grouped", True))

    @property
    def needs_boxes(self) -> bool:
        return bool(getattr(COMPONENT_TYPES[self.type], "needs_boxes", False))

    def instantiate(self, det_class_names: Optional[Sequence[str]] = None) -> Any:
        """A fresh component instance (streaming components hold state, so one per view)."""
        kwargs = dict(self.params)
        if self.type == "partial_detection":
            kwargs["det_class_names"] = det_class_names
        try:
            return COMPONENT_TYPES[self.type](**kwargs)
        except (TypeError, ValueError) as error:
            raise MetricsConfigError(f"component {self.type!r}: {error}") from error

    def serialization(self) -> Dict[str, Any]:
        return {"type": self.type, **self.params}


def parse_components(
    entries: Optional[Sequence[Union[Mapping[str, Any], ComponentSpec, str]]],
) -> Tuple[ComponentSpec, ...]:
    """Parse ``[{type: ..., **params}, ...]`` (a bare string is a type with default params)."""
    if not entries:
        return ()
    specs = []
    seen = set()
    for index, entry in enumerate(entries):
        if isinstance(entry, ComponentSpec):
            spec = entry
        elif isinstance(entry, str):
            spec = ComponentSpec(type=entry)
        elif isinstance(entry, Mapping):
            if "type" not in entry:
                raise MetricsConfigError(f"components[{index}] needs a type token.")
            spec = ComponentSpec(type=str(entry["type"]), params={k: v for k, v in entry.items() if k != "type"})
        else:
            raise MetricsConfigError(
                f"components[{index}] must be a mapping or type token, got {type(entry).__name__}."
            )
        if spec.type in seen:
            raise MetricsConfigError(f"components[{index}]: component {spec.type!r} is configured twice.")
        seen.add(spec.type)
        specs.append(spec)
    return tuple(specs)
