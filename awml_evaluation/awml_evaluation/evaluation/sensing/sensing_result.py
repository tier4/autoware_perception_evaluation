from logging import getLogger
from typing import Tuple

from awml_evaluation.common.object import DynamicObject

logger = getLogger(__name__)


class DynamicObjectWithSensingResult:
    def __init__(
        self,
        ground_truth_object: DynamicObject,
        pointcloud: Tuple[Tuple[float]],
        scale_factor: float = 1.0,
    ) -> None:
        pass

    def pointcloud_exist(self) -> bool:
        pass
