from typing import Optional

from awml_evaluation.common.object import DynamicObject
from awml_evaluation.common.status import Visibility
import numpy as np


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
