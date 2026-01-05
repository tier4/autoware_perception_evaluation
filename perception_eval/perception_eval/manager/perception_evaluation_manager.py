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


from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from perception_eval.common import ObjectType
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import LabelType
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_nuscene_object_results
from perception_eval.evaluation.matching.objects_filter import filter_object_results
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.util.aggregation_results import accumulate_nuscene_results
from perception_eval.visualization import PerceptionVisualizer2D
from perception_eval.visualization import PerceptionVisualizer3D
from perception_eval.visualization import PerceptionVisualizerType
from perception_eval.visualization.detection_confusion_matrix import DetectionConfusionMatrix

from ._evaluation_manager_base import _EvaluationManagerBase
from ..evaluation.result.object_result import DynamicObjectWithPerceptionResult
from ..evaluation.result.object_result_matching import get_object_results
from ..evaluation.result.object_result_matching import NuscenesObjectMatcher


class PerceptionEvaluationManager(_EvaluationManagerBase):
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
        load_ground_truth (bool): Whether to automatically load ground truth annotations during initialization.
    Defaults to True. Set to False if you prefer to handle ground truth loading manually — for example, in the Autoware ML evaluation pipeline.
        output_dir: Main directory to save any artifacts from running metrics.
    """

    def __init__(
        self,
        evaluation_config: PerceptionEvaluationConfig,
        load_ground_truth: bool = True,
        metric_output_dir: Optional[str] = None,
    ) -> None:
        super().__init__(evaluation_config=evaluation_config, load_ground_truth=load_ground_truth)
        self.frame_results: List[PerceptionFrameResult] = []
        self.__visualizer = (
            PerceptionVisualizer2D(self.evaluator_config)
            if self.evaluation_task.is_2d()
            else PerceptionVisualizer3D(self.evaluator_config)
        )
        self._metric_output_dir = Path(metric_output_dir) if metric_output_dir is not None else None

    @property
    def target_labels(self) -> List[LabelType]:
        return self.evaluator_config.target_labels

    @property
    def metrics_config(self):
        return self.evaluator_config.metrics_config

    @property
    def visualizer(self) -> PerceptionVisualizerType:
        return self.__visualizer

    @property
    def metric_output_dir(self) -> Path:
        return self._metric_output_dir

    def preprocess_object_results(
        self,
        unix_time: int,
        ground_truth_now_frame: FrameGroundTruth,
        estimated_objects: List[ObjectType],
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
    ) -> PerceptionFrameResult:
        """Preprocess perception result at current frame without appending to `self.frame_results`."""

        # Filter estimated and ground truth objects
        filtered_estimated_objects, filtered_ground_truth = self.filter_objects(
            estimated_objects, ground_truth_now_frame
        )

        # Match for detection metrics (Based on matching policy)
        nuscene_object_results = (
            self.match_nuscene_objects(filtered_estimated_objects, filtered_ground_truth)
            if self.metrics_config.detection_config
            else None
        )

        # Validate that at least one matching method was performed
        if nuscene_object_results is None:
            raise ValueError(
                "No object matching performed. At least one metric configuration "
                "(detection, tracking, prediction, or classification) must be enabled."
            )

        filtered_ground_truth.objects = filter_objects(
            filtered_ground_truth.objects,
            is_gt=True,
            transforms=filtered_ground_truth.transforms,
            **critical_object_filter_config.filtering_params,
        )

        if nuscene_object_results is not None:
            # Filter objects by critical object filter config
            nuscene_object_results = filter_nuscene_object_results(
                nuscene_object_results,
                transforms=filtered_ground_truth.transforms,
                **critical_object_filter_config.filtering_params,
            )

        # Create PerceptionFrameResult
        return PerceptionFrameResult(
            nuscene_object_results=nuscene_object_results,
            frame_ground_truth=filtered_ground_truth,
            metrics_config=self.metrics_config,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
            unix_time=unix_time,
            target_labels=self.target_labels,
        )

    def evaluate_perception_frame(
        self,
        perception_frame_result: PerceptionFrameResult,
        previous_perception_frame_result: PerceptionFrameResult = None,
    ) -> PerceptionFrameResult:
        """Evaluate perception frame result."""
        if previous_perception_frame_result is not None:
            perception_frame_result.evaluate_perception_frame(previous_result=previous_perception_frame_result)
        else:
            perception_frame_result.evaluate_perception_frame()

        return perception_frame_result

    def add_frame_result(
        self,
        unix_time: int,
        ground_truth_now_frame: FrameGroundTruth,
        estimated_objects: List[ObjectType],
        critical_object_filter_config: CriticalObjectFilterConfig,
        frame_pass_fail_config: PerceptionPassFailConfig,
    ) -> PerceptionFrameResult:
        """Get perception result at current frame.

        Evaluated result is appended to `self.frame_results`.

        TODO:
        - Arrange `CriticalObjectFilterConfig` and `PerceptionPassFailConfig` to `PerceptionFrameConfig`.
        - Allow input `PerceptionFrameConfig` is None.

        Args:
            unix_time (int): Unix timestamp [us].
            ground_truth_now_frame (FrameGroundTruth): FrameGroundTruth instance that has the closest
                timestamp with `unix_time`.
            estimated_objects (List[ObjectType]): Estimated objects list.
            critical_object_filter_config (CriticalObjectFilterConfig): Parameter config to filter objects.
            frame_pass_fail_config (PerceptionPassFailConfig):Parameter config to evaluate pass/fail.

        Returns:
            PerceptionFrameResult: Evaluation result.
        """
        # Filter estimated and ground truth objects
        filtered_estimated_objects, filtered_ground_truth = self.filter_objects(
            estimated_objects, ground_truth_now_frame
        )

        # Match objects based on enabled metrics
        nuscene_object_results = self.match_nuscene_objects(filtered_estimated_objects, filtered_ground_truth)

        # Validate that at least one matching method was performed
        if nuscene_object_results is None:
            raise ValueError(
                "No object matching performed. At least one metric configuration "
                "(detection, tracking, prediction, classification, or fp_validation) must be enabled."
            )

        # Create PerceptionFrameResult
        perception_frame_result = PerceptionFrameResult(
            nuscene_object_results=nuscene_object_results,
            frame_ground_truth=filtered_ground_truth,
            metrics_config=self.metrics_config,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
            unix_time=unix_time,
            target_labels=self.target_labels,
        )

        if self.frame_results:
            perception_frame_result.evaluate_frame(previous_result=self.frame_results[-1])
        else:
            perception_frame_result.evaluate_frame()

        self.frame_results.append(perception_frame_result)
        return perception_frame_result

    def filter_objects(
        self, estimated_objects: List[ObjectType], frame_ground_truth: FrameGroundTruth
    ) -> Tuple[List[ObjectType], FrameGroundTruth]:
        """
        Apply spatial and semantic filters to estimated and ground truth objects.

        Args:
            estimated_objects (List[ObjectType]): The list of estimated perception objects.
            frame_ground_truth (FrameGroundTruth): The ground truth objects and transformation for the current frame.

        Returns:
            filtered_estimated_objects (List[ObjectType]): Filtered list of estimated objects.
            filtered_frame_ground_truth (FrameGroundTruth): Ground truth frame with filtered objects.
        """
        estimated_objects = filter_objects(
            dynamic_objects=estimated_objects,
            is_gt=False,
            transforms=frame_ground_truth.transforms,
            **self.filtering_params,
        )

        frame_ground_truth.objects = filter_objects(
            dynamic_objects=frame_ground_truth.objects,
            is_gt=True,
            transforms=frame_ground_truth.transforms,
            **self.filtering_params,
        )

        return estimated_objects, frame_ground_truth

    def match_nuscene_objects(
        self,
        estimated_objects: List[ObjectType],
        frame_ground_truth: FrameGroundTruth,
    ) -> Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]:
        """
        Perform NuScenes-style matching between estimated and ground truth objects.

        This function matches estimated and ground truth objects using configurable matching modes
        (e.g., center distance, IoU) and multiple thresholds, producing results categorized by (mode, threshold).

        Args:
            estimated_objects (List[ObjectType]): Filtered estimated perception objects.
            frame_ground_truth (FrameGroundTruth): Filtered ground truth objects and transformations for the current frame.

        Returns:
            nuscene_object_results (Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]):
                A nested dictionary mapping from matching mode → label → threshold
                to a list of matched object results.
        """

        matcher = NuscenesObjectMatcher(
            evaluation_task=self.evaluation_task,
            metrics_config=self.metrics_config,
            matching_label_policy=self.evaluator_config.label_params["matching_label_policy"],
            transforms=frame_ground_truth.transforms,
            uuid_matching_first=self.filtering_params["uuid_matching_first"],
            matching_class_agnostic_fps=self.evaluator_config.label_params["matching_class_agnostic_fps"],
        )
        return matcher.match(estimated_objects, frame_ground_truth.objects)

    def get_scene_result(self) -> MetricsScore:
        """Evaluate metrics score thorough a scene.

        Returns:
            scene_metrics_score (MetricsScore): MetricsScore instance.
        """
        aggregated_nuscene_object_results: Dict[
            MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        aggregated_num_gt = {label: 0 for label in self.target_labels}
        used_frame: List[int] = []

        # Gather objects from frame results
        for frame in self.frame_results:
            num_gt_dict = divide_objects_to_num(frame.frame_ground_truth.objects, self.target_labels)

            for label in self.target_labels:
                aggregated_num_gt[label] += num_gt_dict[label]

            # Only aggregate nuscene_object_results if detection_config exists and frame has nuscene_object_results
            if (
                self.evaluator_config.metrics_config.detection_config is not None
                and frame.nuscene_object_results is not None
            ):
                accumulate_nuscene_results(aggregated_nuscene_object_results, frame.nuscene_object_results)

            used_frame.append(int(frame.frame_name))

        scene_metrics_score: MetricsScore = MetricsScore(
            config=self.metrics_config,
            used_frame=used_frame,
        )

        # Classification
        if self.evaluator_config.metrics_config.classification_config is not None:
            scene_metrics_score.evaluate_classification(aggregated_nuscene_object_results, aggregated_num_gt)

        # Detection
        if self.evaluator_config.metrics_config.detection_config is not None:
            scene_metrics_score.evaluate_detection(aggregated_nuscene_object_results, aggregated_num_gt)

        # Tracking
        if self.evaluator_config.metrics_config.tracking_config is not None:
            scene_metrics_score.evaluate_tracking(aggregated_nuscene_object_results, aggregated_num_gt)

        # Prediction
        if self.evaluator_config.metrics_config.prediction_config is not None:
            scene_metrics_score.evaluate_prediction(aggregated_nuscene_object_results, aggregated_num_gt)

        return scene_metrics_score
