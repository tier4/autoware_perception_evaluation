from logging import getLogger

import numpy as np

from awml_evaluation.common.object import DynamicObject

logger = getLogger(__name__)


class DynamicObjectWithSensingResult:
    def __init__(
        self,
        ground_truth_object: DynamicObject,
        pointcloud: np.ndarray,
        scale_factor: float = 1.0,
    ) -> None:
        self.ground_truth_object: DynamicObject = []
        self.inside_pointcloud_num: int = self.ground_truth_object.get_inside_pointcloud_num(
            pointcloud
        )
        self.is_detected = self._is_detected()

    def _is_detected(self) -> bool:
        if self.inside_pointcloud_num > 0:
            return True
        else:
            return False
