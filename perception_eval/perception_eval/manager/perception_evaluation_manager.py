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
import logging

from perception_eval.common import ObjectType
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import LabelType
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.evaluation import PerceptionFrameResult
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_object_results
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
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
            result.evaluate_frame(previous_result=self.frame_results[-1])
        else:
            result.evaluate_frame()

        self.frame_results.append(result)

        ### debug log ###
        if result.frame_ground_truth is not None:
            logging.info(f"GT Num {len(result.frame_ground_truth.objects)}")
        for i, object in enumerate(result.object_results):
            if object.ground_truth_object is not None:
                if object.ground_truth_object.semantic_label is None:
                    logging.info(f"GT {i}, Label: None")
                else:
                    logging.info(f"GT {i}, Label:{object.ground_truth_object.semantic_label.label}, {object.ground_truth_object.semantic_label.name}")
            if object.estimated_object is not None:
                if object.estimated_object.semantic_label is None:
                    logging.info(f"EO {i}, Label: None")
                else:
                    logging.info(f"EO {i}, Label:{object.estimated_object.semantic_label.label}, {object.estimated_object.semantic_label.name}")
        logging.info(f"Num Success:{result.pass_fail_result.get_num_success()}")
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
            transforms=frame_ground_truth.transforms,
            **self.filtering_params,
        )

        frame_ground_truth.objects = filter_objects(
            objects=frame_ground_truth.objects,
            is_gt=True,
            transforms=frame_ground_truth.transforms,
            **self.filtering_params,
        )

        object_results: List[DynamicObjectWithPerceptionResult] = get_object_results(
            evaluation_task=self.evaluation_task,
            estimated_objects=estimated_objects,
            ground_truth_objects=frame_ground_truth.objects,
            target_labels=self.target_labels,
            matching_label_policy=self.evaluator_config.label_params["matching_label_policy"],
            matchable_thresholds=self.filtering_params["max_matchable_radii"],
            transforms=frame_ground_truth.transforms,
        )

        if self.evaluator_config.filtering_params.get("target_uuids"):
            object_results = filter_object_results(
                object_results=object_results,
                transforms=frame_ground_truth.transforms,
                target_uuids=self.filtering_params["target_uuids"],
            )

        ### debug log ###
        #for i, object in enumerate(frame_ground_truth.objects):
        #    logging.info(f"GT {i}, Label {object.semantic_label}")
        """
        for i, object in enumerate(object_results):
            if object.ground_truth_object is not None:
                if object.ground_truth_object.semantic_label is None:
                    logging.info(f"GT {i}, Label: None")
                else:
                    logging.info(f"GT {i}, Label:{object.ground_truth_object.semantic_label.label}, {object.ground_truth_object.semantic_label.name}")
            if object.estimated_object is not None:
                if object.estimated_object.semantic_label is None:
                    logging.info(f"EO {i}, Label: None")
                else:
                    logging.info(f"EO {i}, Label:{object.estimated_object.semantic_label.label}, {object.estimated_object.semantic_label.name}")
        """
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
