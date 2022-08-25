from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import Union

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.dataset import get_now_frame
from awml_evaluation.common.dataset import load_all_datasets
from awml_evaluation.config.perception_evaluation_config import PerceptionEvaluationConfig
from awml_evaluation.config.sensing_evaluation_config import SensingEvaluationConfig
from awml_evaluation.evaluation.result.perception_frame_result import PerceptionFrameResult
from awml_evaluation.evaluation.sensing.sensing_frame_result import SensingFrameResult


class _EvaluationMangerBase(ABC):
    """Abstract base class for EvaluationManager

    Attributes:
        self.evaluator_config (Union[PerceptionEvaluationConfig, SensingEvaluationConfig]):
            Configuration for specified evaluation task.
        self.ground_truth_frames (List[FrameGroundTruth]): The list of ground truths per frame.
    """

    @abstractmethod
    def __init__(
        self,
        evaluation_config: Union[PerceptionEvaluationConfig, SensingEvaluationConfig],
    ) -> None:
        """[summary]
        Args:
            evaluation_config (Union[PerceptionEvaluationConfig, SensingEvaluationConfig]):
                The config for EvaluationManager.
        """
        super().__init__()

        self.evaluator_config = evaluation_config
        self.ground_truth_frames: List[FrameGroundTruth] = load_all_datasets(
            dataset_paths=self.evaluator_config.dataset_paths,
            frame_id=self.evaluator_config.frame_id,
            does_use_pointcloud=self.evaluator_config.does_use_pointcloud,
            evaluation_task=self.evaluator_config.evaluation_task,
            label_converter=self.evaluator_config.label_converter,
        )

    @abstractmethod
    def add_frame_result(self) -> Union[PerceptionFrameResult, SensingFrameResult]:
        """[summary]
        Add perception/sensing frame result.
        """
        pass

    @abstractmethod
    def _filter_objects(self):
        """[summary]"""
        pass

    def get_ground_truth_now_frame(
        self,
        unix_time: int,
        threshold_min_time: int = 75000,
    ) -> Optional[FrameGroundTruth]:
        """[summary]
        Get now frame of ground truth

        Args:
            unix_time (int): Unix time of frame to evaluate.
            threshold_min_time (int, optional):
                    Min time for unix time difference [us].
                    Default is 75000 sec = 75 ms.

        Returns:
            Optional[FrameGroundTruth]: Now frame of ground truth.
                If there is no corresponding ground truth, returns None.
        """
        ground_truth_now_frame: FrameGroundTruth = get_now_frame(
            ground_truth_frames=self.ground_truth_frames,
            unix_time=unix_time,
            threshold_min_time=threshold_min_time,
        )
        return ground_truth_now_frame
