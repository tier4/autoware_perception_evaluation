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

from collections import defaultdict
from dataclasses import dataclass
import functools
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from perception_eval.common import ObjectType
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import LabelType
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_nuscene_object_results
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.util.aggregation_results import accumulate_nuscene_results
from perception_eval.util.aggregation_results import accumulate_nuscene_tracking_results
from perception_eval.visualization import PerceptionVisualizer2D
from perception_eval.visualization import PerceptionVisualizer3D
from perception_eval.visualization import PerceptionVisualizerType
from perception_eval.visualization.detection_confusion_matrix import DetectionConfusionMatrix

from ._evaluation_manager_base import _EvaluationManagerBase
from ..evaluation.result.object_result import DynamicObjectWithPerceptionResult
from ..evaluation.result.object_result_matching import NuscenesObjectMatcher


@dataclass
class AggregatedNusceneObjectResults:
    nuscene_object_results: Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]
    nuscene_object_tracking_results: Dict[MatchingMode, Dict[LabelType, Dict[float, List[List[DynamicObjectWithPerceptionResult]]]]]
    num_gts: Dict[LabelType, int]
    used_frames: List[int]



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
        metric_output_dir: Main directory to save any artifacts from running metrics.
    """

    def __init__(
        self,
        evaluation_config: PerceptionEvaluationConfig,
        enable_visualizer: bool = True,
        load_ground_truth: bool = True,
        metric_output_dir: Optional[str] = None,
    ) -> None:
        super().__init__(evaluation_config=evaluation_config, load_ground_truth=load_ground_truth)
        self.frame_results: List[PerceptionFrameResult] = []
        self.enable_visualizer = enable_visualizer
        if enable_visualizer:
            self.__visualizer = (
                PerceptionVisualizer2D(self.evaluator_config)
                if self.evaluation_task.is_2d()
                else PerceptionVisualizer3D(self.evaluator_config)
            )
        else:
            self.__visualizer = None

        self._metric_output_dir = Path(metric_output_dir) if metric_output_dir is not None else None

    def __reduce__(self) -> Tuple[PerceptionEvaluationManager, Tuple[Any], Dict[Any]]:
        """Serialization and deserialization of the object with pickling."""
        init_args = (
            self.evaluator_config,
            False,  # When pickling, visualizer is disabled
            self.load_ground_truth,
            self.metric_output_dir,
        )
        state = {
            "frame_results": self.frame_results,
        }
        return (self.__class__, init_args, state)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the object to preserve states after deserialization."""
        self.frame_results = state.get("frame_results", self.frame_results)

    def serialization(self) -> Dict[str, Any]:
        """Serialize the object to a dict."""
        return {
            "evaluation_config": self.evaluator_config.serialization(),
            "enable_visualization": self.enable_visualization,
            "load_ground_truth": self.load_ground_truth,
            "metric_output_dir": self.metric_output_dir,
        }

    @classmethod
    def deserialization(cls, data: Dict[str, Any]) -> PerceptionEvaluationManager:
        """Deserialize the data to PerceptionEvaluationManager."""

        return cls(
            evaluation_config=PerceptionEvaluationConfig.deserialization(data["evaluation_config"]),
            load_ground_truth=data["load_ground_truth"],
            metric_output_dir=data["metric_output_dir"],
        )

    @property
    def target_labels(self) -> List[LabelType]:
        return self.evaluator_config.target_labels

    @property
    def metrics_config(self):
        return self.evaluator_config.metrics_config

    @property
    def visualizer(self) -> Optional[PerceptionVisualizerType]:
        return self.__visualizer

    @property
    def metric_output_dir(self) -> Path:
        return self._metric_output_dir

    def preprocess_object_results(
        self,
        unix_time: int,
        ground_truth_now_frame: FrameGroundTruth,
        estimated_objects: List[ObjectType],
        frame_pass_fail_config: PerceptionPassFailConfig,
        critical_object_filter_config: Optional[CriticalObjectFilterConfig] = None,
        frame_prefix: Optional[str] = None,
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

        # Filter ground truth objects by critical object filter config. If critical object filter config is not set, use target labels.
        if critical_object_filter_config is not None:
            filtered_ground_truth.objects = filter_objects(
                filtered_ground_truth.objects,
                is_gt=True,
                transforms=filtered_ground_truth.transforms,
                **critical_object_filter_config.filtering_params,
            )

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
            frame_prefix=frame_prefix,
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
        frame_prefix: Optional[str] = None,
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
            frame_prefix=frame_prefix,
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

        frame_ground_truth_objects = filter_objects(
            dynamic_objects=frame_ground_truth.objects,
            is_gt=True,
            transforms=frame_ground_truth.transforms,
            **self.filtering_params,
        )

        # Create a new FrameGroundTruth instance with filtered objects
        frame_ground_truth = FrameGroundTruth(
            unix_time=frame_ground_truth.unix_time,
            frame_name=frame_ground_truth.frame_name,
            # Replace the objects with filtered objects
            objects=frame_ground_truth_objects,
            transforms=frame_ground_truth.transform_matrices,
            raw_data=frame_ground_truth.raw_data,
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

    def _aggregate_nuscene_object_results(self) -> AggregatedNusceneObjectResults:
        """
        Aggregate nuscene object results.

        Args:
            frame_results (List[PerceptionFrameResult]): List of frame results.

        Returns:
            aggregated_nuscene_object_results (Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]): Aggregated nuscene object results.
        """
        aggregated_nuscene_object_results: Dict[
            MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]
        ] = defaultdict(functools.partial(defaultdict, functools.partial(defaultdict, list)))

        # If the task is Tracking, we need to aggregate previous frame results
        aggregated_nuscene_object_tracking_results: Dict[
            MatchingMode, Dict[LabelType, Dict[float, List[List[DynamicObjectWithPerceptionResult]]]]
        ] = defaultdict(functools.partial(defaultdict, functools.partial(defaultdict, list)))

        aggregated_num_gt = {label: 0 for label in self.target_labels}
        used_frames: List[int] = []
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
                # Aggregate a sequence of frame results
                if self.evaluator_config.metrics_config.tracking_config is not None:
                    accumulate_nuscene_tracking_results(
                        aggregated_nuscene_object_tracking_results, frame.nuscene_object_results
                    )

            used_frames.append(int(frame.frame_name))

        return AggregatedNusceneObjectResults(
            nuscene_object_results=aggregated_nuscene_object_results, 
            num_gts=aggregated_num_gt, 
            used_frames=used_frames,
            nuscene_object_tracking_results=aggregated_nuscene_object_tracking_results
        )

    def get_scene_result(
        self, aggregated_nuscene_object_results: Optional[AggregatedNusceneObjectResults] = None
    ) -> MetricsScore:
        """Evaluate metrics score thorough a scene.

        Args:
            aggregated_nuscene_object_results (Optional[AggregatedNusceneObjectResults]): Aggregated nuscene object results. Defaults to None.

        Returns:
            scene_metrics_score (MetricsScore): MetricsScore instance.
        """
        if aggregated_nuscene_object_results is None:
            aggregated_nuscene_object_results = self._aggregate_nuscene_object_results()

        scene_metrics_score: MetricsScore = MetricsScore(
            config=self.metrics_config,
            used_frame=aggregated_nuscene_object_results.used_frames,
        )

        # Classification
        if self.evaluator_config.metrics_config.classification_config is not None:
            scene_metrics_score.evaluate_classification(
                aggregated_nuscene_object_results.nuscene_object_results, aggregated_nuscene_object_results.num_gts
            )

        # Detection
        if self.evaluator_config.metrics_config.detection_config is not None:
            scene_metrics_score.evaluate_detection(
                aggregated_nuscene_object_results.nuscene_object_results, aggregated_nuscene_object_results.num_gts
            )

            if self.metric_output_dir is not None:
                detection_confusion_matrix = DetectionConfusionMatrix(output_dir=self.metric_output_dir)
                # Draw confusion matrices
                detection_confusion_matrix(
                    nuscene_object_results=aggregated_nuscene_object_results.nuscene_object_results,
                    num_ground_truth=aggregated_nuscene_object_results.num_gts,
                )

        # Tracking
        if self.evaluator_config.metrics_config.tracking_config is not None:
            scene_metrics_score.evaluate_tracking(
                aggregated_nuscene_object_results.nuscene_object_results, aggregated_nuscene_object_results.num_gts
            )
            scene_metrics_score.evaluate_tracking(
                aggregated_nuscene_object_results.nuscene_object_tracking_results, 
                aggregated_nuscene_object_results.num_gts)

        # Prediction
        if self.evaluator_config.metrics_config.prediction_config is not None:
            scene_metrics_score.evaluate_prediction(
                aggregated_nuscene_object_results.nuscene_object_results, aggregated_nuscene_object_results.num_gts
            )

        return scene_metrics_score

    def _group_nuscene_object_results_by_prefix(self) -> Dict[str, AggregatedNusceneObjectResults]:
        """
        Aggregate nuscene object results.

        Args:
            frame_results (List[PerceptionFrameResult]): List of frame results.

        Returns:
            aggregated_nuscene_object_results (Dict[MatchingMode, Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]]]): Aggregated nuscene object results.
        """
        aggregated_nuscene_object_results: Dict[str, AggregatedNusceneObjectResults] = defaultdict(AggregatedNusceneObjectResults)
        # Gather objects from frame results
        for frame in self.frame_results:
            if frame.frame_prefix not in aggregated_nuscene_object_results:
                aggregated_nuscene_object_results[frame.frame_prefix] = AggregatedNusceneObjectResults(
                    nuscene_object_results=defaultdict(functools.partial(defaultdict, functools.partial(defaultdict, list))),
                    nuscene_object_tracking_results=defaultdict(functools.partial(defaultdict, functools.partial(defaultdict, list))),
                    num_gts={label: 0 for label in self.target_labels},
                    used_frames=[]
                )
            
            nuscene_object_results = aggregated_nuscene_object_results[frame.frame_prefix].nuscene_object_results
            nuscene_object_tracking_results = aggregated_nuscene_object_results[frame.frame_prefix].nuscene_object_tracking_results
            num_gts = aggregated_nuscene_object_results[frame.frame_prefix].num_gts
            used_frames = aggregated_nuscene_object_results[frame.frame_prefix].used_frames

            num_gt_dict = divide_objects_to_num(frame.frame_ground_truth.objects, self.target_labels)
            for label in self.target_labels:
                num_gts[label] += num_gt_dict[label]

            # Only aggregate nuscene_object_results if detection_config exists and frame has nuscene_object_results
            if (
                self.evaluator_config.metrics_config.detection_config is not None
                and frame.nuscene_object_results is not None
            ):
                accumulate_nuscene_results(nuscene_object_results, frame.nuscene_object_results)
                # Aggregate a sequence of frame results
                if self.evaluator_config.metrics_config.tracking_config is not None:
                    accumulate_nuscene_tracking_results(
                        nuscene_object_tracking_results, frame.nuscene_object_results
                    )

            used_frames.append(int(frame.frame_name))

        return aggregated_nuscene_object_results
    
    def get_scene_result_with_prefix(self) -> Dict[str, MetricsScore]:
        """
        Get scene result with prefix.

        Args:
            prefix (str): Prefix of the scene result.

        Returns:
            scene_result_with_prefix (Dict[str, MetricsScore]): Scene result with prefix.
        """
        grouped_nuscene_object_results = self._group_nuscene_object_results_by_prefix()
        scene_result_with_prefix: Dict[str, MetricsScore] = defaultdict()
        for prefix, aggregated_nuscene_object_results in grouped_nuscene_object_results.items():
            scene_result_with_prefix[prefix] = self.get_scene_result(aggregated_nuscene_object_results)

        return scene_result_with_prefix
