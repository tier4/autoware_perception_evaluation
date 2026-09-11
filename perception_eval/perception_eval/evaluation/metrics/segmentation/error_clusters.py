# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from tier4/autoware-ml (autoware_ml/metrics/segmentation3d/error_clusters.py) at fcf86419.

"""Misclassification rate and error clusters (metric D1).

An error is a point whose predicted class disagrees with the ground truth. Besides the point rate,
nearby error points (within ``cluster_radius``) are merged into spatial clusters and each cluster
counts as one phantom regardless of size. Reported whole-scene and per true class; clusters are
computed per frame, so only counters are retained across frames.
"""

from __future__ import annotations

from typing import Dict
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.naming import label_metric_name
from perception_eval.evaluation.metrics.segmentation.spatial import count_clusters
from perception_eval.evaluation.metrics.segmentation.state import SegmentationView


class ErrorClusters:
    """Misclassified-point rate plus the count of error clusters, overall and per class."""

    kind = "point"
    supports_grouped = True
    needs_boxes = False

    def __init__(self, cluster_radius: float = 0.5, min_cluster_points: int = 1) -> None:
        self.cluster_radius = float(cluster_radius)
        self.min_cluster_points = int(min_cluster_points)
        if self.cluster_radius < 0.0:
            raise ValueError("cluster_radius must be >= 0.")
        if self.min_cluster_points < 1:
            raise ValueError("min_cluster_points must be >= 1.")
        self.reset()

    def reset(self) -> None:
        self._error_total = 0
        self._valid_total = 0
        self._cluster_total = 0
        self._num_frames = 0
        self._class_names: Optional[tuple] = None
        self._per_error: Optional[NDArray] = None
        self._per_valid: Optional[NDArray] = None
        self._per_cluster: Optional[NDArray] = None

    def _ensure(self, view: SegmentationView) -> None:
        if self._per_error is None:
            self._class_names = view.class_names
            self._per_error = np.zeros(view.num_classes, dtype=np.int64)
            self._per_valid = np.zeros(view.num_classes, dtype=np.int64)
            self._per_cluster = np.zeros(view.num_classes, dtype=np.int64)

    def update(self, view: SegmentationView) -> None:
        self._ensure(view)
        if view.num_points == 0:
            return
        self._num_frames += 1
        wrong = view.wrong
        self._valid_total += view.num_points
        self._error_total += int(wrong.sum())
        self._cluster_total += count_clusters(view.xyz[wrong], self.cluster_radius, self.min_cluster_points)
        self._per_valid += np.bincount(view.target, minlength=view.num_classes)[: view.num_classes]
        self._per_error += np.bincount(view.target[wrong], minlength=view.num_classes)[: view.num_classes]
        for class_index in np.unique(view.target[wrong]):
            wrong_class = wrong & (view.target == class_index)
            self._per_cluster[class_index] += count_clusters(
                view.xyz[wrong_class], self.cluster_radius, self.min_cluster_points
            )

    def compute(self) -> Dict[str, float]:
        report: Dict[str, float] = {
            "error_rate": (self._error_total / self._valid_total) if self._valid_total else float("nan"),
            "error_cluster_count": float(self._cluster_total),
            "error_clusters_per_frame": (self._cluster_total / self._num_frames) if self._num_frames else float("nan"),
        }
        if self._per_error is None:
            return report
        for class_index in range(self._per_error.shape[0]):
            name = label_metric_name(class_index, self._class_names)
            valid = int(self._per_valid[class_index])
            report[f"error_rate_{name}"] = (int(self._per_error[class_index]) / valid) if valid else float("nan")
            report[f"error_cluster_count_{name}"] = float(self._per_cluster[class_index])
        return report
