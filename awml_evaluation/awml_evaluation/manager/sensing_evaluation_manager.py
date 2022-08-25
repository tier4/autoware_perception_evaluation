from typing import List
from typing import Tuple

from awml_evaluation.common.dataset import FrameGroundTruth
from awml_evaluation.common.point import crop_pointcloud
from awml_evaluation.config.sensing_evaluation_config import SensingEvaluationConfig
from awml_evaluation.evaluation.matching.objects_filter import filter_objects
from awml_evaluation.evaluation.sensing.sensing_frame_result import SensingFrameResult
from awml_evaluation.util.math import get_bbox_scale
import numpy as np

from ._evaluation_manager_base import _EvaluationMangerBase
from ..common.object import DynamicObject


class SensingEvaluationManager(_EvaluationMangerBase):
    """The class to manage sensing evaluation.

    Attributes:
        - By _EvaluationMangerBase
        self.evaluator_config (SensingEvaluationConfig): Configuration for sensing evaluation.
        self.ground_truth_frames (List[FrameGroundTruth]): The list of ground truths per frame.

        - By SensingEvaluationManger
        self.frame_results (List[SensingFrameResult]): The list of sensing result per frame.
    """

    def __init__(
        self,
        evaluation_config: SensingEvaluationConfig,
    ) -> None:
        """
        Args:
            evaluation_config (SensingEvaluationConfig): The configuration for sensing evaluation.
        """
        super().__init__(evaluation_config)
        self.frame_results: List[SensingFrameResult] = []

    def add_frame_result(
        self,
        unix_time: int,
        ground_truth_now_frame: FrameGroundTruth,
        pointcloud: np.ndarray,
        non_detection_areas: List[List[Tuple[float, float, float]]],
    ) -> SensingFrameResult:
        """[summary]
        Get sensing result at current frame.
        The results is kept in self.frame_results.

        Args:
            unix_time (int)
            ground_truth_now_frame (FrameGroundTruth): If there is no corresponding annotation, allow None.
            pointcloud (np.ndarray): Observed pointcloud.
            non_detection_area (List[List[Tuple[float, float, float]]]):
                List of non-detection areas.

        Returns:
            result (SensingFrameResult)
        """
        ground_truth_objects: List[DynamicObject] = self._filter_objects(ground_truth_now_frame)
        frame_name: str = ground_truth_now_frame.frame_name

        # Crop pointcloud for non-detection area
        pointcloud_for_non_detection: np.ndarray = self.crop_pointcloud(
            ground_truth_objects=ground_truth_objects,
            pointcloud=pointcloud,
            non_detection_areas=non_detection_areas,
        )

        result = SensingFrameResult(
            sensing_frame_config=self.evaluator_config.sensing_frame_config,
            unix_time=unix_time,
            frame_name=frame_name,
        )

        result.evaluate_frame(
            ground_truth_objects=ground_truth_objects,
            pointcloud_for_detection=pointcloud,
            pointcloud_for_non_detection=pointcloud_for_non_detection,
        )
        self.frame_results.append(result)

        return result

    def _filter_objects(self, frame_ground_truth: FrameGroundTruth) -> List[DynamicObject]:
        return filter_objects(
            frame_id=self.evaluator_config.frame_id,
            objects=frame_ground_truth.objects,
            is_gt=True,
            ego2map=frame_ground_truth.ego2map,
            **self.evaluator_config.filtering_params,
        )

    def crop_pointcloud(
        self,
        ground_truth_objects: List[DynamicObject],
        pointcloud: np.ndarray,
        non_detection_areas: List[List[Tuple[float, float, float]]],
    ) -> List[np.ndarray]:
        """Crop pointcloud from (N, 3) to (M, 3) with the non-detection area
        specified in SensingEvaluationConfig.

        Args:
            ground_truth_objects: List[DynamicObject]
            pointcloud (numpy.ndarray): The array of pointcloud, in shape (N, 3).
            non_detection_areas (List[List[Tuple[float, float, float]]]):
                The list of 3D-polygon areas for non-detection.

        Returns:
            cropped_pointcloud (List[numpy.ndarray]): The list of cropped pointcloud.
        """
        cropped_pointcloud: List[np.ndarray] = []
        for non_detection_area in non_detection_areas:
            cropped_pointcloud.append(
                crop_pointcloud(
                    pointcloud=pointcloud,
                    area=non_detection_area,
                )
            )

        # Crop pointcloud for non-detection outside of objects' bbox
        box_scale_0m: float = self.evaluator_config.metrics_params["box_scale_0m"]
        box_scale_100m: float = self.evaluator_config.metrics_params["box_scale_100m"]
        for i, points in enumerate(cropped_pointcloud):
            outside_points: np.ndarray = points.copy()
            for ground_truth in ground_truth_objects:
                bbox_scale: float = get_bbox_scale(
                    distance=ground_truth.get_distance(),
                    box_scale_0m=box_scale_0m,
                    box_scale_100m=box_scale_100m,
                )
                outside_points: np.ndarray = ground_truth.crop_pointcloud(
                    pointcloud=outside_points,
                    bbox_scale=bbox_scale,
                    inside=False,
                )
            cropped_pointcloud[i] = outside_points
        return cropped_pointcloud
