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
from typing import Tuple

from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel
from perception_eval.common.object import DynamicObject
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_object_results
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.evaluation.result.perception_pass_fail_result import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_pass_fail_result import PerceptionPassFailConfig
from perception_eval.visualization.perception_visualizer import PerceptionVisualizer

from ._evaluation_manager_base import _EvaluationMangerBase
from ..evaluation.result.object_result import DynamicObjectWithPerceptionResult
from ..evaluation.result.object_result import get_object_results


class PerceptionEvaluationManager(_EvaluationMangerBase):
    """[summary]
    PerceptionEvaluationManager class.
    This class is management interface for perception interface.

    Attributes:
        - By _EvaluationMangerBase
        self.evaluator_config (PerceptionEvaluatorConfig): Configuration for perception evaluation.
        self.ground_truth_frames (List[FrameGroundTruth]): Ground truth frames from datasets

        - By PerceptionEvaluationManger
        self.frame_results (List[PerceptionFrameResult]): Evaluation result
        self.visualizer (PerceptionVisualizer): Visualization class for perception result.
    """

    def __init__(
        self,
        evaluation_config: PerceptionEvaluationConfig,
    ) -> None:
        super().__init__(evaluation_config=evaluation_config)
        """[summary]

        Args:
            evaluator_config (PerceptionEvaluatorConfig): Configuration for perception evaluation.
        """
        self.target_labels: List[AutowareLabel] = evaluation_config.target_labels
        self.frame_results: List[PerceptionFrameResult] = []
        self.visualizer = PerceptionVisualizer.from_eval_cfg(self.evaluator_config)

    def add_frame_result(
        self,
        unix_time: int,
        ground_truth_now_frame: FrameGroundTruth,
        estimated_objects: List[DynamicObject],
        ros_critical_ground_truth_objects: List[DynamicObject],
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
    ) -> PerceptionFrameResult:
        """[summary]
        Evaluate one frame

        Args:
            unix_time (int): Unix time of frame to evaluate [us]
            ground_truth_now_frame (FrameGroundTruth): Now frame ground truth
            estimated_objects (List[DynamicObject]): estimated object which you want to evaluate
            ros_critical_ground_truth_objects (List[DynamicObject]):
                    Critical ground truth objects filtered by ROS node to evaluate pass fail result
            critical_object_filter_config (CriticalObjectFilterConfig):
                    The parameter config to choose critical ground truth objects
            frame_pass_fail_config (PerceptionPassFailConfig):
                    The parameter config to evaluate

        Returns:
            PerceptionFrameResult: Evaluation result
        """
        object_results, ground_truth_now_frame = self._filter_objects(
            estimated_objects,
            ground_truth_now_frame,
        )

        result = PerceptionFrameResult(
            object_results=object_results,
            frame_ground_truth=ground_truth_now_frame,
            metrics_config=self.evaluator_config.metrics_config,
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
        estimated_objects: List[DynamicObject],
        frame_ground_truth: FrameGroundTruth,
    ) -> Tuple[List[DynamicObjectWithPerceptionResult], FrameGroundTruth]:
        """[summary]
        Filtering estimated and ground truth objects.
        Args:
            estimated_objects (List[DynamicObject])
            frame_ground_truth (FrameGroundTruth)
        Returns:
            estimated_objects (List[DynamicObject])
            frame_ground_truth (FrameGroundTruth)
        """
        estimated_objects = filter_objects(
            objects=estimated_objects,
            is_gt=False,
            ego2map=frame_ground_truth.ego2map,
            **self.evaluator_config.filtering_params,
        )

        frame_ground_truth.objects = filter_objects(
            objects=frame_ground_truth.objects,
            is_gt=True,
            ego2map=frame_ground_truth.ego2map,
            **self.evaluator_config.filtering_params,
        )
        object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
            estimated_objects=estimated_objects,
            ground_truth_objects=frame_ground_truth.objects,
        )
        if self.evaluator_config.filtering_params.get("target_uuids"):
            object_results = filter_object_results(
                object_results=object_results,
                ego2map=frame_ground_truth.ego2map,
                target_uuids=self.evaluator_config.filtering_params["target_uuids"],
            )
        return object_results, frame_ground_truth

    def get_scene_result(self) -> MetricsScore:
        """[summary]
        Evaluate metrics score thorough a scene.

        Returns:
            MetricsScore: Metrics score
        """
        # Gather objects from frame results
        target_labels: List[AutowareLabel] = self.evaluator_config.target_labels
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
        scene_metrics_score: MetricsScore = MetricsScore(
            config=self.evaluator_config.metrics_config,
            used_frame=used_frame,
        )
        if self.evaluator_config.metrics_config.detection_config is not None:
            scene_metrics_score.evaluate_detection(all_frame_results, all_num_gt)
        if self.evaluator_config.metrics_config.tracking_config is not None:
            scene_metrics_score.evaluate_tracking(all_frame_results, all_num_gt)
        if self.evaluator_config.metrics_config.prediction_config is not None:
            pass

        return scene_metrics_score

    def visualize_all(self, animation: bool = False) -> None:
        """[summary]
        Visualize object result in BEV space for all frames.

        Args:
            animation (bool): Whether make animation. Defaults to True.
        """
        self.visualizer.visualize_all(self.frame_results, animation=animation)

    def visualize_frame(self, frame_index: int = -1) -> None:
        """[summary]
        Visualize object result in BEV space at specified frame.

        Args:
            frame_index (int): The index of frame to be visualized. Defaults to -1 (latest frame).
        """
        self.visualizer.visualize_frame(self.frame_results[frame_index])
