from logging import getLogger

from awml_evaluation.common.object import DynamicObject
import numpy as np

logger = getLogger(__name__)


class DynamicObjectWithSensingResult:
    """[summary]
    The class to evaluate sensing result for dynamic object.

    Attributes:
        self.ground_truth_object (DynamicObject): The target DynamicObject.
        self.inside_pointcloud (numpy.ndarray): The array of pointcloud in bounding box.
        self.inside_pointcloud_num (int): The number of pointcloud in bounding box.
        self.is_detected (bool): The boolean flag indicates whether pointcloud is in bounding box.
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
            min_points_threshold (int): The minibmum number of points should be detected in bounding box.
        """
        self.ground_truth_object: DynamicObject = ground_truth_object

        # Evaluate
        self.inside_pointcloud: np.ndarray = self.ground_truth_object.get_inside_pointcloud(
            pointcloud,
            scale_factor,
        )
        self.inside_pointcloud_num: int = len(self.inside_pointcloud)
        self.is_detected = self.inside_pointcloud_num >= min_points_threshold
