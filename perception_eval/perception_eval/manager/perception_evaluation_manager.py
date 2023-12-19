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

from __future__ import annotations

from typing import List
from typing import Tuple
from typing import TYPE_CHECKING

from perception_eval.matching.objects_filter import divide_objects
from perception_eval.matching.objects_filter import divide_objects_to_num
from perception_eval.matching.objects_filter import filter_object_results
from perception_eval.matching.objects_filter import filter_objects
from perception_eval.metrics import MetricsScore
from perception_eval.result import get_object_results
from perception_eval.result import PerceptionFrameResult
from perception_eval.visualization import PerceptionVisualizer2D
from perception_eval.visualization import PerceptionVisualizer3D

from ._evaluation_manager_base import _EvaluationMangerBase

if TYPE_CHECKING:
    from perception_eval.common.label import LabelType
    from perception_eval.config import PerceptionEvaluationConfig
    from perception_eval.dataset import FrameGroundTruth
    from perception_eval.object import ObjectType
    from perception_eval.result import CriticalObjectFilterConfig
    from perception_eval.result import DynamicObjectWithPerceptionResult
    from perception_eval.result import PerceptionPassFailConfig
    from perception_eval.visualization import PerceptionVisualizerType


class PerceptionEvaluationManager(_EvaluationMangerBase):
    """A manager class to evaluate perception task.

    Attributes:
        evaluator_config (PerceptionEvaluatorConfig): Configuration for perception evaluation.
        ground_truth_frames (List[FrameGroundTruth]): Ground truth frames from datasets
        target_labels (List[LabelType]): List of target labels.
        frame_results (List[PerceptionFrameResult]): Perception results list at each frame.
        visualizer (Optional[PerceptionVisualizerType]): Visualization class for perception result.
            If `self.evaluation_task.is_2d()=True`, this is None.

    Args:
        evaluator_config (PerceptionEvaluatorConfig): Configuration for perception evaluation.
    """

    def __init__(
        self,
        evaluation_config: PerceptionEvaluationConfig,
    ) -> None:
        super().__init__(evaluation_config=evaluation_config)
        self.__visualizer = (
            PerceptionVisualizer2D(self.evaluator_config)
            if self.evaluation_task.is_2d()
            else PerceptionVisualizer3D(self.evaluator_config)
        )

    @property
    def target_labels(self) -> List[LabelType]:
        return self.evaluator_config.target_labels

    @property
    def metrics_config(self):
        return self.evaluator_config.metrics_config

    @property
    def visualizer(self) -> PerceptionVisualizerType:
        return self.__visualizer

    def add_frame_result(
        self,
        unix_time: int,
        ground_truth_now_frame: FrameGroundTruth,
        estimated_objects: List[ObjectType],
        ros_critical_ground_truth_objects: List[ObjectType],
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
    ) -> PerceptionFrameResult:
        """Get perception result at current frame.

        Evaluated result is appended to `self.frame_results`.

        TODO:
        - Arrange `CriticalObjectFilterConfig` and `PerceptionPassFailConfig` to `PerceptionFrameConfig`.
        - Allow input `PerceptionFrameConfig` and `ros_critical_ground_truth_objects` are None.

        Args:
            unix_time (int): Unix timestamp [us].
            ground_truth_now_frame (FrameGroundTruth): FrameGroundTruth instance that has the closest
                timestamp with `unix_time`.
            estimated_objects (List[ObjectType]): Estimated objects list.
            ros_critical_ground_truth_objects (List[ObjectType]): Critical ground truth objects filtered by ROS
                node to evaluate pass fail result.
            critical_object_filter_config (CriticalObjectFilterConfig): Parameter config to filter objects.
            frame_pass_fail_config (PerceptionPassFailConfig):Parameter config to evaluate pass/fail.

        Returns:
            PerceptionFrameResult: Evaluation result.
        """
        object_results, ground_truth_now_frame = self._filter_objects(
            estimated_objects,
            ground_truth_now_frame,
        )

        result = PerceptionFrameResult(
            object_results=object_results,
            frame_ground_truth=ground_truth_now_frame,
            metrics_config=self.metrics_config,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
            unix_time=unix_time,
            target_labels=self.target_labels,
        )

        if len(self.frame_results) > 0:
            result.evaluate_frame(
                ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
                previous_result=self.frame_results[-1],
            )
        else:
            result.evaluate_frame(
                ros_critical_ground_truth_objects=ros_critical_ground_truth_objects,
            )

        self.frame_results.append(result)
        return result

    def _filter_objects(
        self,
        estimated_objects: List[ObjectType],
        frame_ground_truth: FrameGroundTruth,
    ) -> Tuple[List[DynamicObjectWithPerceptionResult], FrameGroundTruth]:
        """Returns filtered list of DynamicObjectResult and FrameGroundTruth instance.

        First of all, filter `estimated_objects` and `frame_ground_truth`.
        Then generate a list of DynamicObjectResult as `object_results`.
        Finally, filter `object_results` when `target_uuids` is specified.

        Args:
            estimated_objects (List[ObjectType]): Estimated objects list.
            frame_ground_truth (FrameGroundTruth): FrameGroundTruth instance.

        Returns:
            object_results (List[DynamicObjectWithPerceptionResult]): Filtered object results list.
            frame_ground_truth (FrameGroundTruth): Filtered FrameGroundTruth instance.
        """
        estimated_objects = filter_objects(
            objects=estimated_objects,
            is_gt=False,
            ego2map=frame_ground_truth.ego2map,
            **self.filtering_params,
        )

        frame_ground_truth.objects = filter_objects(
            objects=frame_ground_truth.objects,
            is_gt=True,
            ego2map=frame_ground_truth.ego2map,
            **self.filtering_params,
        )

        object_results = get_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=estimated_objects,
            ground_truth_objects=frame_ground_truth.objects,
            target_labels=self.target_labels,
            allow_matching_unknown=self.evaluator_config.label_params["allow_matching_unknown"],
            matchable_thresholds=self.filtering_params["max_matchable_radii"],
        )

        if self.evaluator_config.filtering_params.get("target_uuids"):
            object_results = filter_object_results(
                object_results=object_results,
                ego2map=frame_ground_truth.ego2map,
                target_uuids=self.filtering_params["target_uuids"],
            )
        return object_results, frame_ground_truth

    def get_scene_result(self) -> MetricsScore:
        """Evaluate metrics score thorough a scene.

        Returns:
            scene_metrics_score (MetricsScore): MetricsScore instance.
        """
        # Gather objects from frame results
        target_labels: List[LabelType] = self.target_labels
        all_frame_results = {label: [[]] for label in target_labels}
        all_num_gt = {label: 0 for label in target_labels}
        used_frame: List[int] = []
        for frame in self.frame_results:
            obj_result_dict = divide_objects(frame.object_results, target_labels)
            num_gt_dict = divide_objects_to_num(frame.frame_ground_truth.objects, target_labels)
            for label in target_labels:
                all_frame_results[label].append(obj_result_dict[label])
                all_num_gt[label] += num_gt_dict[label]
            used_frame.append(int(frame.frame_name))

        # Calculate score
        scene_metrics_score = MetricsScore(
            config=self.metrics_config,
            used_frame=used_frame,
        )
        if self.evaluator_config.metrics_config.detection_config is not None:
            scene_metrics_score.evaluate_detection(all_frame_results, all_num_gt)
        if self.evaluator_config.metrics_config.tracking_config is not None:
            scene_metrics_score.evaluate_tracking(all_frame_results, all_num_gt)
        if self.evaluator_config.metrics_config.prediction_config is not None:
            pass
        if self.evaluator_config.metrics_config.classification_config is not None:
            scene_metrics_score.evaluate_classification(all_frame_results, all_num_gt)

        return scene_metrics_score
