from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from perception_eval.common.label import LabelType
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

from .tp_error_metrics import TPErrorMetric
from .tp_error_metrics import TPScaleError
from .tp_error_metrics import TPTranslationError


class TPErrors:
    """
    mAP evaluation class supporting multiple thresholds per label.

    This class calculates Average Precision (AP) and Average Precision with Heading (APH)
    for a set of perception results grouped by label and matching threshold.

    For each label:
        - It computes AP and optionally APH for all given matching thresholds.
        - It then calculates the mean AP (and APH) across thresholds for that label.

    Finally:
        - It averages the per-label mean AP (and APH) across all target labels
          to produce the final mAP and mAPH.

    This class supports both 2D and 3D detection evaluation:
        - In 2D detection, only AP is calculated (APH is skipped).
        - In 3D detection, both AP and APH are calculated.

    Attributes:
        target_labels (List[LabelType]):
            List of target labels evaluated in this instance.
        matching_mode (MatchingMode):
            The matching strategy used for TP/FP calculation (e.g., CENTERDISTANCE, IOU3D).
        is_detection_2d (bool):
            If True, only AP is computed; APH is skipped.
        label_to_aps (Dict[LabelType, List[Ap]]):
            List of AP instances (one per threshold) for each label.
        label_mean_to_ap (Dict[LabelType, float]):
            Mean AP across thresholds for each label. Can be NaN if all AP values are NaN.
        label_to_aphs (Optional[Dict[LabelType, List[Ap]]]):
            List of APH instances (one per threshold) for each label (if 3D detection).
        label_mean_to_aph (Optional[Dict[LabelType, float]]):
            Mean APH across thresholds for each label (if 3D detection). Can be NaN if all APH values are NaN.
        map (float):
            Final mean Average Precision (mAP) across all labels. Can be NaN if all label means are NaN.
        maph (Optional[float]):
            Final mean Average Precision with Heading (mAPH) across all labels,
            or None if `is_detection_2d` is True. Can be NaN if all label means are NaN.
    """

    def __init__(
        self,
        object_results_dict: Dict[LabelType, Dict[float, List[DynamicObjectWithPerceptionResult]]],
        num_ground_truth_dict: Dict[LabelType, int],
        target_labels: List[LabelType],
        matching_mode: MatchingMode,
        is_detection_2d: bool = False,
    ) -> None:
        self.object_results_dict = object_results_dict
        self.num_ground_truth_dict = num_ground_truth_dict
        self.target_labels = target_labels
        self.matching_mode = matching_mode
        self.is_detection_2d = is_detection_2d
        self.num_tps = 0

        # All available matching thresholds
        # TODO (KokSeang): Configurable tp error metrics for each label
        # {Label: {Matching_Threshold: [TPErrorMetric]}}
        self.labels_to_tp_errors: Dict[LabelType, Dict[float, List[TPErrorMetric]]] = {}
        for label in self.target_labels:
            for threshold, _ in self.object_results_dict[label].items():
                self.labels_to_tp_errors[label] = {
                    threshold: [
                        TPTranslationError(is_detection_2d=self.is_detection_2d),
                        TPScaleError(is_detection_2d=self.is_detection_2d),
                    ]
                }

        self.available_tp_error_metric_names = [TPTranslationError.METRIC_NAME, TPScaleError.METRIC_NAME]

    def __call__(self) -> None:
        """
        Compute mAP and mAPH across all target labels and thresholds.

        This method processes the object results for each label and threshold,
        computes AP and APH (if applicable), and aggregates the results to
        calculate mean AP (mAP) and mean APH (mAPH) across all labels.

        It populates the following attributes:
            - label_to_aps
            - label_mean_to_ap
            - label_to_aphs (if 3D detection)
            - label_mean_to_aph (if 3D detection)
            - map
            - maph (if 3D detection)
        """
        for label in self.target_labels:
            label_to_tp_errors = self.labels_to_tp_errors.get(label, None)
            assert label_to_tp_errors is not None, f"TP errors for label {label} not found."

            for threshold, object_results in self.object_results_dict[label].items():
                tp_error_metrics = label_to_tp_errors.get(threshold, None)
                assert tp_error_metrics is not None, f"TP errors for label {label} at threshold {threshold} not found."
                self._compute_tp_errors(object_results=object_results, tp_error_metrics=tp_error_metrics)

    def _compute_tp_errors(
        self, object_results: List[DynamicObjectWithPerceptionResult], tp_error_metrics: List[TPErrorMetric]
    ) -> None:
        for obj in object_results:
            is_tp = obj.ground_truth_object is not None and obj.is_label_correct
            if not is_tp:
                continue

            for tp_error_metric in tp_error_metrics:
                tp_error_metric(
                    ground_truth_dynamic_object=obj.ground_truth_object, predicted_dynamic_object=obj.estimated_object
                )

            self.num_tps += 1

    def __reduce__(self) -> Tuple[TPErrors, Tuple[Any]]:
        """Serialization and deserialization of the object with pickling."""
        init_args = (
            self.object_results_dict,
            self.num_ground_truth_dict,
            self.target_labels,
            self.matching_mode,
            self.is_detection_2d,
        )
        state = {"labels_to_tp_errors": self.labels_to_tp_errors, "num_tps": self.num_tps}
        return (self.__class__, init_args, state)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the object to preserve states after deserialization."""
        self.labels_to_tp_errors = state.get("labels_to_tp_errors", self.labels_to_tp_errors)
        self.num_tps = state.get("num_tps", self.num_tps)

    def compute_mean_tp_errors(self) -> Dict[LabelType, Dict[float, List[float]]]:
        """
        Compute mean TP errors across all target labels and thresholds.

        This method processes the TP error metrics for each label and threshold,
        computes mean TP errors (e.g., translation error, scale error),
        and aggregates the results to calculate mean TP errors across all labels.
        """
        return {
            label: {
                threshold: [
                    tp_error_metric.compute_mean_tp_error_metric(num_tps=self.num_tps)
                    for tp_error_metric in tp_error_metrics
                ]
                for threshold, tp_error_metrics in label_to_tp_errors.items()
            }
            for label, label_to_tp_errors in self.labels_to_tp_errors.items()
        }
