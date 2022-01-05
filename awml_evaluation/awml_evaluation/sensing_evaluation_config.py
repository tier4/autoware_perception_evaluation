import datetime
from logging import getLogger
from typing import List

from awml_evaluation.common.label import LabelConverter
from awml_evaluation.evaluation.sensing.sensing_frame_config import SensingFrameConfig

logger = getLogger(__name__)


class SensingEvaluationConfig:
    def __init__(
        self,
        dataset_paths: List[str],
        result_root_directory: str,
        log_directory: str,
        visualization_directory: str,
        box_margin_for_0m_detection: float,
        box_margin_for_100m_detection: float,
        box_margin_for_0m_non_detection: float,
        box_margin_for_100m_non_detection: float,
    ):
        # dataset
        self.dataset_paths: List[str] = dataset_paths

        # directory
        time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        self.result_root_directory: str = result_root_directory.format(TIME=time)
        self.log_directory: str = log_directory
        self.visualization_directory: str = visualization_directory

        # Labels
        self.label_converter = LabelConverter()

        # margin
        self.sensing_frame_config = SensingFrameConfig(
            box_margin_for_0m_detection=box_margin_for_0m_detection,
            box_margin_for_100m_detection=box_margin_for_100m_detection,
            box_margin_for_0m_non_detection=box_margin_for_0m_non_detection,
            box_margin_for_100m_non_detection=box_margin_for_100m_non_detection,
        )
