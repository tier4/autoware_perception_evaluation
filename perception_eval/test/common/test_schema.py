# Copyright 2026 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for `perception_eval.common.schema.FrameID`."""
from __future__ import annotations

import pytest
from perception_eval.common.schema import FrameID


@pytest.mark.parametrize(
    "value",
    ["cam_front", "CAM_FRONT", "cam_traffic_light_far", "lidar_concat"],
)
def test_from_value_resolves_canonical_names(value: str) -> None:
    assert FrameID.from_value(value) == FrameID(value.lower())


@pytest.mark.parametrize("alias", ["cam_front_far", "CAM_FRONT_FAR", "Cam_Front_Far"])
def test_from_value_resolves_cam_front_far_alias(alias: str) -> None:
    """Some real T4 datasets name the far/telephoto TLR camera "cam_front_far"
    instead of the canonical "cam_traffic_light_far"; both must resolve to the same
    FrameID so `_get_transforms()` doesn't reject a dataset over a naming choice."""
    assert FrameID.from_value(alias) == FrameID.CAM_TRAFFIC_LIGHT_FAR


def test_from_value_raises_for_unknown_channel() -> None:
    with pytest.raises(ValueError, match="Unexpected value"):
        FrameID.from_value("totally_unknown_channel")
