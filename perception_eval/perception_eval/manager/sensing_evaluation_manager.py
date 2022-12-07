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

from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.point import crop_pointcloud
from perception_eval.config.sensing_evaluation_config import SensingEvaluationConfig
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.sensing.sensing_frame_config import SensingFrameConfig
from perception_eval.evaluation.sensing.sensing_frame_result import SensingFrameResult
from perception_eval.util.math import get_bbox_scale

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
        sensing_frame_config: Optional[SensingFrameConfig] = None,
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
            sensing_frame_config (Optional[SensingFrameConfig]): Evaluation config for one frame.
                If not specified, parameters of metrics config will be used. Defaults to None.

        Returns:
            result (SensingFrameResult)
        """
        if sensing_frame_config is None:
            sensing_frame_config = SensingFrameConfig(
                **self.evaluator_config.filtering_params,
                **self.evaluator_config.metrics_params,
            )

        ground_truth_objects: List[DynamicObject] = self._filter_objects(
            ground_truth_now_frame,
            sensing_frame_config,
        )

        # Crop pointcloud for non-detection area
        pointcloud_for_non_detection: np.ndarray = self.crop_pointcloud(
            ground_truth_objects=ground_truth_objects,
            pointcloud=pointcloud,
            non_detection_areas=non_detection_areas,
        )

        result = SensingFrameResult(
            sensing_frame_config=sensing_frame_config,
            unix_time=unix_time,
            frame_name=ground_truth_now_frame.frame_name,
        )

        result.evaluate_frame(
            ground_truth_objects=ground_truth_objects,
            pointcloud_for_detection=pointcloud,
            pointcloud_for_non_detection=pointcloud_for_non_detection,
        )
        self.frame_results.append(result)

        return result

    def _filter_objects(
        self,
        frame_ground_truth: FrameGroundTruth,
        sensing_frame_config: SensingFrameConfig,
    ) -> List[DynamicObject]:
        """Filter ground truth objects
        Args:
            frame_ground_truth (FrameGroundTruth): FrameGroundTruth instance.
            sensing_frame_config (SensingFrameConfig): SensingFrameConfig instance.
        Returns:
            List[DynamicObject]: Filtered ground truth objects
        """
        return filter_objects(
            frame_id=self.evaluator_config.frame_id,
            objects=frame_ground_truth.objects,
            is_gt=True,
            target_uuids=sensing_frame_config.target_uuids,
            ego2map=frame_ground_truth.ego2map,
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
