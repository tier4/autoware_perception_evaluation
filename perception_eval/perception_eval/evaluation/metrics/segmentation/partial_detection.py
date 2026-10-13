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
# Ported from tier4/autoware-ml (autoware_ml/metrics/segmentation3d/partial_detection.py) at fcf86419.

"""Small-object partial-detection score (metric D2, diagnostic).

For a pedestrian or cone, classifying even a few points correctly is far better than none. D2
groups segmentation points inside each mapped detection box and rewards partial hits with a
saturating credit ``(k/(k+h)) / (n/(n+h))`` for ``k`` correct among ``n`` valid points: the first
correct point earns about half, further points add diminishing refinement up to 1 at all-correct,
and zero correct points score exactly 0. Boxes with fewer than ``min_points`` are skipped and
counted. This metric only runs on the trained taxonomy: the box-to-class mapping lives in that
index space.
"""

from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
from perception_eval.evaluation.metrics.component import mean_valid
from perception_eval.evaluation.metrics.detection.geometry import points_in_bev_box
from perception_eval.evaluation.metrics.naming import label_metric_name
from perception_eval.evaluation.metrics.segmentation.state import SegmentationView


class PartialDetectionScore:
    """Saturating credit for partial segmentation of small-object detection boxes."""

    kind = "point"
    supports_grouped = False
    needs_boxes = True

    def __init__(
        self,
        half_saturation: float = 1.0,
        min_points: int = 1,
        det_class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.half_saturation = float(half_saturation)
        self.min_points = int(min_points)
        if self.half_saturation <= 0.0:
            raise ValueError("half_saturation must be > 0.")
        if self.min_points < 1:
            raise ValueError("min_points must be >= 1.")
        self.det_class_names = tuple(det_class_names) if det_class_names is not None else None
        self.reset()

    def reset(self) -> None:
        self._credit_sum: Dict[int, float] = {}
        self._credit_count: Dict[int, int] = {}
        self._skipped = 0

    def credit(self, k: int, n: int) -> float:
        """Saturating credit for ``k`` correct among ``n`` points."""
        h = self.half_saturation
        return (k / (k + h)) / (n / (n + h))

    def update(self, view: SegmentationView) -> None:
        if view.taxonomy != "raw":
            raise ValueError("PartialDetectionScore only runs on the trained (raw) taxonomy.")
        mapped = view.gt_box_seg_class >= 0
        if not bool(mapped.any()):
            return
        boxes = view.gt_boxes[mapped]
        seg_classes = view.gt_box_seg_class[mapped]
        det_labels = view.gt_box_det_label[mapped]
        if view.num_points == 0:
            self._skipped += int(boxes.shape[0])
            return
        coord_xy = view.xyz[:, :2]
        for box, seg_class, det_label in zip(boxes, seg_classes, det_labels):
            # Rectangular prefilter at the footprint's circumradius, then the exact yaw-aware test.
            radius = float(np.linalg.norm(box[3:5])) / 2.0 * (1.0 + 1e-9)
            center = box[:2]
            near = (np.abs(coord_xy[:, 0] - center[0]) <= radius) & (np.abs(coord_xy[:, 1] - center[1]) <= radius)
            candidates = np.flatnonzero(near)
            inside = candidates[points_in_bev_box(coord_xy[candidates], box)] if candidates.shape[0] else candidates
            n = int(inside.shape[0])
            if n < self.min_points:
                self._skipped += 1
                continue
            correct = int(np.sum(view.pred[inside] == seg_class))
            self._credit_sum[int(det_label)] = self._credit_sum.get(int(det_label), 0.0) + self.credit(correct, n)
            self._credit_count[int(det_label)] = self._credit_count.get(int(det_label), 0) + 1

    def compute(self) -> Dict[str, float]:
        report: Dict[str, float] = {"pd_skipped_low_point_boxes": float(self._skipped)}
        labels: List[int] = (
            list(range(len(self.det_class_names))) if self.det_class_names is not None else sorted(self._credit_count)
        )
        per_class_means: List[float] = []
        for label in labels:
            name = label_metric_name(label, self.det_class_names)
            count = self._credit_count.get(label, 0)
            if not count:
                report[f"pd_score_{name}"] = float("nan")
                continue
            mean_credit = self._credit_sum[label] / count
            report[f"pd_score_{name}"] = mean_credit
            per_class_means.append(mean_credit)
        report["mpd_score"] = mean_valid(per_class_means)
        return report
