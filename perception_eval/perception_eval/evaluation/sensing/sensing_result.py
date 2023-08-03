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

from typing import Optional

import numpy as np
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import Visibility


class DynamicObjectWithSensingResult:
    """[summary]
    The class to evaluate sensing result for dynamic object.

    Attributes:
        self.ground_truth_object (DynamicObject): The target DynamicObject.
        self.inside_pointcloud (numpy.ndarray): The array of pointcloud in bounding box.
        self.inside_pointcloud_num (int): The number of pointcloud in bounding box.
        self.is_detected (bool): The boolean flag indicates whether pointcloud is in bounding box.
        self.nearest_point (np.ndarray): The nearest point from base_link.
        self.is_occluded (bool): Whether the object is occluded.
    """

    def __init__(
        self,
        ground_truth_object: DynamicObject,
        pointcloud: np.ndarray,
        scale_factor: float,
        min_points_threshold: int,
    ) -> None:
        """[summary]
        Args:
            ground_truth_object (DynamicObject): Ground truth object.
            pointcloud (numpy.ndarray): Array of pointcloud after removing ground.
            scale_factor (float): Scale factor for bounding box.
            min_points_threshold (int): The minimum number of points should be detected in bounding box.
        """
        self.ground_truth_object: DynamicObject = ground_truth_object

        # Evaluate
        self.inside_pointcloud: np.ndarray = self.ground_truth_object.crop_pointcloud(
            pointcloud,
            scale_factor,
        )
        self.inside_pointcloud_num: int = len(self.inside_pointcloud)
        self.is_detected: bool = self.inside_pointcloud_num >= min_points_threshold
        self.nearest_point: Optional[np.ndarray] = self._get_nearest_point()
        self.is_occluded: bool = ground_truth_object.visibility == Visibility.NONE

    def _get_nearest_point(self) -> Optional[np.ndarray]:
        """[summary]
        Returns the nearest point from base_link. The pointcloud must be base_link coords.

        Returns:
            Optional[np.ndarray]: The nearest point included in the object's bbox, in shape (3,).
                If there is no point in bbox, returns None.
        """
        if self.inside_pointcloud_num == 0:
            return None

        points: np.ndarray = self.inside_pointcloud[:, :3]
        idx: int = np.argmin(np.linalg.norm(points, ord=2, axis=1)).item()
        return points[idx]
