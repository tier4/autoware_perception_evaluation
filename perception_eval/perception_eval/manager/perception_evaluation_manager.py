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


from typing import Dict
from typing import List
from typing import Tuple

from perception_eval.common import ObjectType
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import LabelType
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation.matching.matching_config import MatchingConfig
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_object_results
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.visualization import PerceptionVisualizer2D
from perception_eval.visualization import PerceptionVisualizer3D
from perception_eval.visualization import PerceptionVisualizerType

from ._evaluation_manager_base import _EvaluationManagerBase
from ..evaluation.result.object_result import DynamicObjectWithPerceptionResult
from ..evaluation.result.object_result_matching import get_nuscene_object_results
from ..evaluation.result.object_result_matching import get_object_results


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
    """

    def __init__(self, evaluation_config: PerceptionEvaluationConfig, load_ground_truth: bool = True) -> None:
        super().__init__(evaluation_config=evaluation_config, load_ground_truth=load_ground_truth)
        self.perception_frame_results: List[PerceptionFrameResult] = []
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
        nuscene_object_results, object_results, ground_truth_now_frame = self.filter_and_match_objects(
            estimated_objects, ground_truth_now_frame
        )

        perception_frame_result = PerceptionFrameResult(
            object_results=object_results,
            nuscene_object_results=nuscene_object_results,
            frame_ground_truth=ground_truth_now_frame,
            metrics_config=self.metrics_config,
            critical_object_filter_config=critical_object_filter_config,
            frame_pass_fail_config=frame_pass_fail_config,
            unix_time=unix_time,
            target_labels=self.target_labels,
        )

        if len(self.perception_frame_results) > 0:
            perception_frame_result.evaluate_frame(previous_result=self.perception_frame_results[-1])
        else:
            perception_frame_result.evaluate_frame()

        self.perception_frame_results.append(perception_frame_result)

        return perception_frame_result

    def filter_and_match_objects(
        self, estimated_objects: List[ObjectType], frame_ground_truth: FrameGroundTruth
    ) -> Tuple[
        Dict[MatchingConfig, List[DynamicObjectWithPerceptionResult]],
        List[DynamicObjectWithPerceptionResult],
        FrameGroundTruth,
    ]:
        """
        Filter estimated and ground truth objects, compute matching results, and optionally filter by UUIDs.

        This function first applies spatial and semantic filters to the estimated objects and ground truth
        objects. It then generates matching results in both NuScenes-style (thresholded multi-key format)
        and Autoware-style (flat list). If `target_uuids` is specified, it will further filter the flat list.

        Args:
            estimated_objects (List[ObjectType]): The list of estimated perception objects.
            frame_ground_truth (FrameGroundTruth): The ground truth objects and transformation for the current frame.

        Returns:
            nuscene_object_results (Dict[MatchingConfig, List[DynamicObjectWithPerceptionResult]]):
                Thresholded object match results for NuScenes-style metrics.
            object_results (List[DynamicObjectWithPerceptionResult]):
                Flat list of matched results for Autoware-style metrics.
            frame_ground_truth (FrameGroundTruth):
                Updated ground truth with filtered objects.
        """

        estimated_objects = filter_objects(
            objects=estimated_objects,
            is_gt=False,
            transforms=frame_ground_truth.transforms,
            **self.filtering_params,
        )

        frame_ground_truth.objects = filter_objects(
            objects=frame_ground_truth.objects,
            is_gt=True,
            transforms=frame_ground_truth.transforms,
            **self.filtering_params,
        )

        nuscene_object_results: Dict[
            MatchingConfig, List[DynamicObjectWithPerceptionResult]
        ] = get_nuscene_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=estimated_objects,
            ground_truth_objects=frame_ground_truth.objects,
            metrics_config=self.metrics_config,
            matching_label_policy=self.evaluator_config.label_params["matching_label_policy"],
            transforms=frame_ground_truth.transforms,
        )

        object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=estimated_objects,
            ground_truth_objects=frame_ground_truth.objects,
            target_labels=self.target_labels,
            matching_label_policy=self.evaluator_config.label_params["matching_label_policy"],
            matchable_thresholds=self.filtering_params["max_matchable_radii"],
            transforms=frame_ground_truth.transforms,
            uuid_matching_first=self.filtering_params["uuid_matching_first"],
        )

        if self.evaluator_config.filtering_params.get("target_uuids"):
            object_results = filter_object_results(
                object_results=object_results,
                transforms=frame_ground_truth.transforms,
                target_uuids=self.filtering_params["target_uuids"],
            )

        return nuscene_object_results, object_results, frame_ground_truth

    def get_scene_result(self) -> MetricsScore:
        """Evaluate metrics score thorough a scene.

        Returns:
            scene_metrics_score (MetricsScore): MetricsScore instance.
        """
        # Gather objects from frame results
        target_labels: List[LabelType] = self.target_labels

        # aggregated results from each frame
        aggregated_object_results_dict = {label: [] for label in target_labels}
        aggregated_nuscene_object_results_dict: Dict[
            LabelType, Dict[MatchingConfig, List[List[DynamicObjectWithPerceptionResult]]]
        ] = {label: {} for label in target_labels}
        aggregated_num_gt = {label: 0 for label in target_labels}
        used_frame: List[int] = []

        for frame in self.perception_frame_results:
            object_results_dict: Dict[LabelType, List[DynamicObjectWithPerceptionResult]] = divide_objects(
                frame.object_results, target_labels
            )
            nuscene_object_results_dict: Dict[
                LabelType, Dict[MatchingConfig, List[DynamicObjectWithPerceptionResult]]
            ] = divide_objects(frame.nuscene_object_results, target_labels)
            num_gt_dict = divide_objects_to_num(frame.frame_ground_truth.objects, target_labels)

            for label in target_labels:
                aggregated_object_results_dict[label].append(object_results_dict[label])

                nuscene_label_result: Dict[
                    MatchingConfig, List[DynamicObjectWithPerceptionResult]
                ] = nuscene_object_results_dict.get(label, {})

                for key, detection_list in nuscene_label_result.items():
                    if key not in aggregated_nuscene_object_results_dict[label]:
                        aggregated_nuscene_object_results_dict[label][key] = []
                    aggregated_nuscene_object_results_dict[label][key].append(detection_list)

                aggregated_num_gt[label] += num_gt_dict[label]

            used_frame.append(int(frame.frame_name))

        scene_metrics_score: MetricsScore = MetricsScore(
            config=self.metrics_config,
            used_frame=used_frame,
        )

        # Classification
        if self.evaluator_config.metrics_config.classification_config is not None:
            scene_metrics_score.evaluate_classification(aggregated_object_results_dict, aggregated_num_gt)

        # Detection
        if self.evaluator_config.metrics_config.detection_config is not None:
            scene_metrics_score.evaluate_detection(aggregated_nuscene_object_results_dict, aggregated_num_gt)

        # Tracking
        if self.evaluator_config.metrics_config.tracking_config is not None:
            scene_metrics_score.evaluate_tracking(aggregated_object_results_dict, aggregated_num_gt)

        # Prediction
        if self.evaluator_config.metrics_config.prediction_config is not None:
            pass

        return scene_metrics_score
