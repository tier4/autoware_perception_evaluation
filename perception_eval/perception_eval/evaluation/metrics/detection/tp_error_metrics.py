from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from perception_eval.common import ObjectType
from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class TPErrorMetric:
    """Tp error metric base class."""

    METRIC_NAME: str = "BaseTPErrorMetric"

    def __init__(self, is_detection_2d: bool = False) -> None:
        self.metric_score: float = float('nan')
        self.is_detection_2d = is_detection_2d
        assert not is_detection_2d, "TPErrorMetric only supports for 3D detection at the moment."

    def __reduce__(self) -> Tuple[TPErrorMetric, Tuple[Any], Dict[Any]]:
        """Serialization and deserialization of the object with pickling."""
        init_args = (self.is_detection_2d,)
        state = {"metric_score": self.metric_score}
        return (self.__class__, init_args, state)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the object to preserve states after deserialization."""
        self.metrics_score = state.get("metric_score", self.metrics_score)

    def __call__(self, ground_truth_dynamic_object: ObjectType, predicted_dynamic_object: ObjectType) -> None:
        """
        Compute tp error metric value.

        Args:
            ground_truth_dynamic_object (ObjectType): Ground truth object to evaluate.
            predicted_dynamic_object (ObjectType): Predicted object to evaluate.
        """
        self.metric_score = self.compute_tp_error_metric(
            ground_truth_dynamic_object=ground_truth_dynamic_object, predicted_dynamic_object=predicted_dynamic_object
        )

    def compute_tp_error_metric(
        self, ground_truth_dynamic_object: ObjectType, predicted_dynamic_object: ObjectType
    ) -> float:
        """
        Compute tp error metric value.

        Args:
            dynamic_object_with_perception_result (DynamicObjectWithPerceptionResult): Object result to evaluate.
            is_detection_2d (bool): Whether the evaluation is for 2D detection.

        Returns:
            float: Computed tp error metric value.
        """
        raise NotImplementedError("Subclasses must implement compute_tp_error_metric method.")

    def compute_mean_tp_error_metric(self, num_tps: int) -> float:
        """
        Compute mean tp error metric value.

        Returns:
            float: Computed mean tp error metric value.
        """
        return self.metric_value / num_tps if num_tps > 0 else float('nan')


class TPTranslationError(TPErrorMetric):
    """Translation error metric class."""

    METRIC_NAME: str = "TPTranslationError"

    def __init__(self, is_detection_2d: bool = False) -> None:
        super().__init__(is_detection_2d=is_detection_2d)

    def compute_tp_error_metric(
        self, ground_truth_dynamic_object: ObjectType, predicted_dynamic_object: ObjectType
    ) -> float:
        """
        Compute bev center distance error between ground truth and predicted objects.

        Args:
            ground_truth_dynamic_object (ObjectType): Ground truth object to evaluate.
            predicted_dynamic_object (ObjectType): Predicted object to evaluate.
        """
        return np.linalg.norm(
            np.array(ground_truth_dynamic_object.position[:2]) - np.array(predicted_dynamic_object.position[:2])
        )


class TPScaleError(TPErrorMetric):
    """Scale error metric class."""

    METRIC_NAME: str = "TPScaleError"

    def __init__(self, is_detection_2d: bool = False) -> None:
        super().__init__(is_detection_2d=is_detection_2d)

    def compute_tp_error_metric(
        self, ground_truth_dynamic_object: ObjectType, predicted_dynamic_object: ObjectType
    ) -> float:
        """
        Compute scale error between ground truth and predicted objects.

        This function is modified from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/utils.py#L86,
        where scale error is defined as 1 - iou. Since two boxes are matched, we consider they are aligned, for example, translation and rotation are identical.

        Args:
            ground_truth_dynamic_object (ObjectType): Ground truth object to evaluate.
            predicted_dynamic_object (ObjectType): Predicted object to evaluate.
        """
        gt_volume = np.prod(ground_truth_dynamic_object.state.shape.size)
        pred_volume = np.prod(predicted_dynamic_object.state.shape.size)

        assert gt_volume > 0, "Ground truth object size must be positive values."
        assert pred_volume > 0, "Predicted object size must be positive values."

        intersection = np.prod(
            np.minimum(ground_truth_dynamic_object.state.shape.size, predicted_dynamic_object.state.shape.size)
        )
        iou = intersection / (gt_volume + pred_volume - intersection)
        return 1 - iou
