# Copyright 2022-2024 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from typing import Optional


class SensingFrameConfig:
    """Frame level configuration for sensing evaluation.

    Args:
    -----
        target_uuids (Optional[List[str]]): List of target uuids to be filtered.
        box_scale_0m (float): Scale factor for bounding box at 0m.
        box_scale_100m (float): Scale factor for bounding box at 100m.
        min_points_threshold (int): The minimum number of points should be detected in bounding box.
    """

    def __init__(
        self,
        target_uuids: Optional[List[str]],
        box_scale_0m: float,
        box_scale_100m: float,
        min_points_threshold: int,
    ) -> None:
        self.target_uuids: Optional[List[str]] = target_uuids
        self.box_scale_0m: float = box_scale_0m
        self.box_scale_100m: float = box_scale_100m
        self.min_points_threshold: int = min_points_threshold

        self.scale_slope_: float = 0.01 * (box_scale_100m - box_scale_0m)

    def get_scale_factor(self, distance: float) -> float:
        """Calculate scale factor linearly for bounding box at specified distance.

        Note:
            scale = ((box_scale_100m - box_scale_0m) / (100 - 0)) * (distance - 0) + box_scale_0m

        Args:
        -----
            distance (float): The distance from vehicle to target bounding box.

        Returns:
        --------
            float: Calculated scale factor.
        """
        return self.scale_slope_ * distance + self.box_scale_0m
