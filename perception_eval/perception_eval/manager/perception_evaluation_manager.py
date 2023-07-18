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

from perception_eval.common import ObjectType
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import LabelType
from perception_eval.common.threshold import set_thresholds
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import PerceptionFrameConfig
from perception_eval.evaluation import PerceptionFrameResult
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_object_results
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.visualization import PerceptionVisualizer2D
from perception_eval.visualization import PerceptionVisualizer3D
from perception_eval.visualization import PerceptionVisualizerType

from ._evaluation_manager_base import _EvaluationMangerBase
from ..evaluation.result.object_result import DynamicObjectWithPerceptionResult
from ..evaluation.result.object_result import get_object_results


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
        self.frame_results: List[PerceptionFrameResult] = []
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
        frame_config: Optional[PerceptionFrameConfig] = None,
        critical_ground_truth_objects: Optional[List[ObjectType]] = None,
    ) -> PerceptionFrameResult:
        """Get perception result at current frame.

        Evaluated result is appended to `self.frame_results`.

        TODO:
        - Allow input `PerceptionFrameConfig` and `ros_critical_ground_truth_objects` are None.

        Args:
            unix_time (int): Unix timestamp [us].
            ground_truth_now_frame (FrameGroundTruth): FrameGroundTruth instance that has the closest
                timestamp with `unix_time`.
            estimated_objects (List[ObjectType]): Estimated objects list.
            frame_config (PerceptionFrameConfig): Parameter config to evaluate pass/fail.
                This allows to specify target ground truth objects dynamically.
            critical_ground_truth_objects (List[ObjectType]): Critical ground truth objects filtered node to evaluate pass fail result.

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
            frame_config=frame_config,
            unix_time=unix_time,
            target_labels=self.target_labels,
        )

        if frame_config is None:
            frame_config = __get_frame_config(self.evaluator_config)

        if critical_ground_truth_objects is None:
            critical_ground_truth_objects = ground_truth_now_frame.objects

        if len(self.frame_results) > 0:
            result.evaluate_frame(
                critical_ground_truth_objects=critical_ground_truth_objects,
                previous_result=self.frame_results[-1],
            )
        else:
            result.evaluate_frame(
                critical_ground_truth_objects=critical_ground_truth_objects,
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
        object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
            estimated_objects=estimated_objects,
            ground_truth_objects=frame_ground_truth.objects,
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
        scene_metrics_score: MetricsScore = MetricsScore(
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


def __get_frame_config(eval_cfg: PerceptionEvaluationConfig) -> PerceptionFrameConfig:
    """Returns frame config from `PerceptionEvaluationConfig`.

    Args:
        eval_cfg (PerceptionEvaluationConfig): Evaluation config for scene.

    Returns:
        PerceptionFrameConfig: Frame config.
    """
    if eval_cfg.evaluation_task.is_3d():
        matching_thresholds = eval_cfg.metrics_params["plane_distance_threshold"]
    elif eval_cfg.evaluation_task.is_2d():
        matching_thresholds = eval_cfg.metrics_params["iou2d_thresholds"]
    else:
        matching_thresholds: Optional[List[float]] = None

    matching_threshold_list = (
        set_thresholds(matching_thresholds, len(eval_cfg.target_labels))[0]
        if matching_thresholds is not None
        else None
    )

    return PerceptionFrameConfig(
        eval_cfg.evaluation_task,
        eval_cfg.target_labels,
        matching_threshold_list=matching_threshold_list,
        **eval_cfg.filtering_params,
    )
