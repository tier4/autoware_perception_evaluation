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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/naming.py, base.py) at fcf86419.

"""Metric-key token helpers shared by the driving-aware metric components.

Keeps key formatting in one place so components stay focused on selecting values.
"""

from __future__ import annotations

from typing import Optional
from typing import Sequence


def metric_token(value: str) -> str:
    """Lowercase, underscore-separated token safe for a metric key."""
    return str(value).lower().replace(" ", "_").replace("/", "_")


def label_metric_name(label: int, class_names: Optional[Sequence[str]]) -> str:
    """Token for a class label, ``class_{label}`` only when no names exist.

    With ``class_names`` configured, an out-of-range label is a folding or configuration bug
    and raises instead of silently minting a phantom class key.
    """
    if class_names is None:
        return f"class_{label}"
    if not 0 <= label < len(class_names):
        raise ValueError(f"label {label} is outside the configured {len(class_names)} class names.")
    return metric_token(class_names[label])


def number_token(value: float) -> str:
    """Key-safe token for a plain number, e.g. ``99p5`` for 99.5."""
    return f"{float(value):g}".replace("-", "minus").replace(".", "p")


def distance_token(distance: float) -> str:
    """Key-safe token for a distance in meters, e.g. ``0p5m`` for 0.5."""
    return f"{number_token(distance)}m"


def threshold_token(threshold: float) -> str:
    """Collision-free token for a distance threshold, e.g. ``0p5m`` for 0.5.

    Delegates to the shared distance-token rule so range suffixes and threshold tokens can never
    drift apart.
    """
    return distance_token(threshold)
