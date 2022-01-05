from typing import List
from typing import Tuple

from awml_evaluation.evaluation.sensing.sensing_frame_config import SensingFrameConfig


class SensingFrameResult:
    def __init__(
        self,
        sensing_frame_config: SensingFrameConfig,
        unix_time: int,
        frame_name: str,
    ) -> None:
        # config
        self.sensing_frame_config = sensing_frame_config

        # frame information
        self.unix_time: int = unix_time
        self.frame_name: str = frame_name

    def evaluate_frame(
        self,
        pointcloud_for_detection: List[Tuple[float]],
        pointcloud_for_non_detection: List[Tuple[float]],
    ) -> None:
        self._evaluate_pointcloud_for_detection()
        self._evaluate_pointcloud_for_non_detection()

    def _evaluate_pointcloud_for_detection(
        self,
        pointcloud_for_detection: List[Tuple[float]],
    ):
        raise NotImplementedError

    def _evaluate_pointcloud_for_non_detection(
        self,
        pointcloud_for_non_detection: List[Tuple[float]],
    ):
        raise NotImplementedError
