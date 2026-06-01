from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple


class NuScenesDetectionScore:
    """
    Class to compute NuScenes Detection Score (NDS).
    NDS is a metric that combines mAP and mATE.
    NDS is calculated as follows:
        NDS = 1 / weights (ap_weight * mAP + tp_scores
    where mAP is the mean Average Precision and mATE is the mean Average Translation Error.
    """

    def __init__(
        self,
        map: float,
        metric_prefix_name: str,
        mean_tp_error_metrics: Dict[str, float],
        ap_weight: float = 5,
    ) -> None:
        """
        Initialize the NuScenesDetectionScore class.
        """
        self.map = map
        self.metric_prefix_name = metric_prefix_name
        self.mean_tp_error_metrics = mean_tp_error_metrics
        self.ap_weight = ap_weight
        self.total_num_metrics = len(self.mean_tp_error_metrics) + self.ap_weight
        self.nds = self.compute_nds()

    def __reduce__(self) -> Tuple[NuScenesDetectionScore, Tuple[Any]]:
        """
        Serialize the NuScenesDetectionScore class.
        """
        init_args = (
            self.map,
            self.metric_prefix_name,
            self.mean_tp_error_metrics,
            self.ap_weight,
        )
        return (
            self.__class__,
            init_args,
        )

    def compute_tp_scores(self) -> Dict[str, float]:
        """
        Compute TP scores.
        """
        tp_scores = {}
        for metric_name, metric_value in self.mean_tp_error_metrics.items():
            tp_scores[metric_name] = max(0, 1 - metric_value)

        return tp_scores

    def compute_nds(self) -> float:
        """
        Compute NDS.
        """
        tp_scores = self.compute_tp_scores()
        return 1 / self.total_num_metrics * (self.ap_weight * self.map + sum(tp_scores.values()))

    def __str__(self) -> str:
        """
        Return the string representation of the NuScenesDetectionScore class.
        """
        return f"{self.metric_prefix_name} NDS: {self.nds}"
