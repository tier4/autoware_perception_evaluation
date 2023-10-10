# Copyright 2022 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Tuple


class DiffTranslation:
    """Differences of translation class for estimated and ground truth object.

    Attributes:
    ----------
        self.diff_estimated (Tuple[float, float, float]): The translation difference of estimated object.
        self.diff_ground_truth (Tuple[float, float, float]): The translation difference of ground truth object.
    """

    def __init__(
        self,
        diff_estimated: Tuple[float, float, float],
        diff_ground_truth: Tuple[float, float, float],
    ) -> None:
        """[summary].

        Args:
        ----
            diff_estimated (Tuple[float, float, float]): The translation difference of estimated object.
            diff_ground_truth (Tuple[float, float, float]): The translation difference of ground truth object.
        """
        self.diff_estimated: Tuple[float, float, float] = diff_estimated
        self.diff_ground_truth: Tuple[float, float, float] = diff_ground_truth


class DiffYaw:
    """Differences of yaw class for estimated and ground truth object.

    Attributes:
    ----------
        self.diff_estimated (float): The yaw difference of estimated object.
        self.diff_ground_truth (float): The yaw difference of ground truth object.
    """

    def __init__(
        self,
        diff_estimated: float,
        diff_ground_truth: float,
        deg2rad: bool = False,
    ) -> None:
        """[summary].

        Args:
        ----
            diff_estimated (float): The yaw difference of estimated object.
            diff_ground_truth (float): The yaw difference of ground truth object.
            deg2rad (bool): Whether convert degrees to radians. Defaults to False.
        """
        self.diff_estimated: float = math.radians(diff_estimated) if deg2rad else diff_estimated
        self.diff_ground_truth: float = math.radians(diff_ground_truth) if deg2rad else diff_ground_truth
