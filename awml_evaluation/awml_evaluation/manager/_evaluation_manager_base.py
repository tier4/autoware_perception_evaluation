from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Union

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.dataset import get_now_frame
from awml_evaluation.config._evaluation_config_base import _EvaluationConfigBase
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.evaluation.sensing.sensing_frame_result import SensingFrameResult
from awml_evaluation.visualization.visualization import VisualizationBEV


class _EvaluationMangerBase(ABC):
    """Abstract base class for EvaluationManager

    Attributes:
        self.ground_truth_frames (List[FrameGroundTruth]): The list of ground truths per frame.
        self.visualization (VisualizationBEV): The BEV visualizer.
    """

    @abstractmethod
    def __init__(self, evaluation_config: _EvaluationConfigBase) -> None:
        """[summary]
        Args:
            evaluation_config (_EvaluationConfig): The config for EvaluationManager.
        """
        super().__init__()

        self.ground_truth_frames: List[FrameGroundTruth] = []
        self.visualization: VisualizationBEV = VisualizationBEV(
            evaluation_config.visualization_directory
        )

    @abstractmethod
    def add_frame_result(self) -> Union[PerceptionFrameResult, SensingFrameResult]:
        """[summary]
        Add perception/sensing frame result.
        """
        pass

    def get_ground_truth_now_frame(
        self,
        unix_time: int,
        threshold_min_time: int = 75000,
    ) -> FrameGroundTruth:
        """[summary]
        Get now frame of ground truth

        Args:
            unix_time (int): Unix time of frame to evaluate.
            threshold_min_time (int, optional):
                    Min time for unix time difference [us].
                    Default is 75000 sec = 75 ms.

        Returns:
            FrameGroundTruth: Now frame of ground truth
        """
        ground_truth_now_frame: FrameGroundTruth = get_now_frame(
            ground_truth_frames=self.ground_truth_frames,
            unix_time=unix_time,
            threshold_min_time=threshold_min_time,
        )
        return ground_truth_now_frame
