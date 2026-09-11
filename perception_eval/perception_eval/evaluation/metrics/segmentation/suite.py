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
# Ported from tier4/autoware-ml (autoware_ml/metrics/segmentation3d/{suite,point_cloud}.py) at fcf86419.

"""The streaming segmentation metric suite.

``SegmentationMetricSuite.update(frame)`` derives one frame's taxonomy views and geometry once,
resolves the filter and range masks, folds the frame into the confusion matrices and into every
configured point component for every ``(taxonomy, filter, range)`` view, then discards the frame.
``compute()`` turns the retained bounded statistics into a :class:`SegmentationMetricsReport`.
The results equal an end-of-scene computation over all frames because every metric decomposes into
additive per-frame sufficient statistics (clusters and neighbourhood rescue are per-frame by
definition in the source implementation).
"""

from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.class_groups import fold_labels
from perception_eval.evaluation.metrics.component import compose_key
from perception_eval.evaluation.metrics.component import GROUPED_TAXONOMY
from perception_eval.evaluation.metrics.component import MetricReport
from perception_eval.evaluation.metrics.filter_config import build_filters
from perception_eval.evaluation.metrics.filters import FrameMetricContext
from perception_eval.evaluation.metrics.filters import MetricFilter
from perception_eval.evaluation.metrics.segmentation.boxes import gt_boxes_from_objects
from perception_eval.evaluation.metrics.segmentation.confusion_matrix import ConfusionAccumulator
from perception_eval.evaluation.metrics.segmentation.confusion_matrix import ConfusionState
from perception_eval.evaluation.metrics.segmentation.report import SegmentationMetricsReport
from perception_eval.evaluation.metrics.segmentation.state import build_grouped_taxonomy
from perception_eval.evaluation.metrics.segmentation.state import build_raw_taxonomy
from perception_eval.evaluation.metrics.segmentation.state import FrameGeometry
from perception_eval.evaluation.metrics.segmentation.state import RAW_TAXONOMY
from perception_eval.evaluation.metrics.segmentation.state import SegmentationFrame
from perception_eval.evaluation.metrics.segmentation.state import SegmentationFrameSummary
from perception_eval.evaluation.metrics.segmentation.state import SegmentationView
from perception_eval.evaluation.metrics.segmentation.state import TaxonomyFrame
from perception_eval.evaluation.metrics.segmentation.state import validate_frame

if TYPE_CHECKING:  # pragma: no cover - typing only (the config module imports the registry)
    from perception_eval.evaluation.metrics.config.segmentation_metrics_config import SegmentationMetricsConfig

logger = logging.getLogger(__name__)

TASK_KEY = "segmentation"
ViewKey = Tuple[str, int, int]  # (taxonomy, filter bucket, range bucket)


class SegmentationMetricSuite:
    """Streaming accumulation of every configured segmentation metric over frames.

    Args:
        config (SegmentationMetricsConfig): Metric configuration.
        map_provider: Optional lanelet map provider; built from ``config.map`` when needed and omitted.
    """

    def __init__(self, config: SegmentationMetricsConfig, map_provider: Any = None) -> None:
        self.config = config
        if map_provider is None and config.map is not None and config.needs_map:
            map_provider = config.map.build_provider()
        self.map_provider = map_provider
        self.filters: List[MetricFilter] = build_filters(config.filters, map_provider, config.map)
        self.filter_names: Tuple[str, ...] = tuple(f.name for f in self.filters)  # "" first (identity)
        self.ranges = config.ranges
        self.range_names: Tuple[str, ...] = ("",) + tuple(r.name for r in self.ranges)
        self.range_suffixes: Tuple[Optional[str], ...] = (None,) + tuple(r.suffix for r in self.ranges)
        self.group_lut, self.group_names = config.resolve_groups()
        self.taxonomies: Tuple[str, ...] = (RAW_TAXONOMY,) + ((GROUPED_TAXONOMY,) if self.group_lut is not None else ())
        self.confusion_specs = [spec for spec in config.components if spec.kind == "confusion"]
        self.point_specs = [spec for spec in config.components if spec.kind == "point"]
        self.reset()

    # ------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Drop every accumulated statistic."""
        self.num_frames = 0
        self.confusion = ConfusionAccumulator(len(self.filters) - 1, len(self.ranges), self.config.num_classes)
        self.coverage_seen: Dict[str, int] = {name: 0 for name in self.filter_names if name}
        self.coverage_covered: Dict[str, int] = {name: 0 for name in self.filter_names if name}
        self._components: Dict[ViewKey, List[Any]] = {}
        for taxonomy in self.taxonomies:
            for filter_bucket in range(len(self.filters)):
                for range_bucket in range(len(self.range_names)):
                    components = [
                        spec.instantiate(det_class_names=self.config.det_class_names)
                        for spec in self.point_specs
                        if taxonomy == RAW_TAXONOMY or spec.supports_grouped
                    ]
                    self._components[(taxonomy, filter_bucket, range_bucket)] = components
        self.warnings: List[str] = []

    # ------------------------------------------------------------------------------------------
    def update(self, frame: SegmentationFrame) -> SegmentationFrameSummary:
        """Fold one frame into the statistics and return a small summary."""
        validate_frame(frame, self.config)
        self.num_frames += 1
        geometry = self._geometry(frame)
        taxonomy_frames = self._taxonomy_frames(frame)
        raw = taxonomy_frames[RAW_TAXONOMY]

        context = FrameMetricContext(ego2map=frame.ego2map, scene_id=frame.scene_id)
        point_masks: List[Optional[NDArray]] = []
        box_masks: List[Optional[NDArray]] = []
        available: Dict[str, bool] = {}
        for bucket, metric_filter in enumerate(self.filters):
            if bucket == 0:
                point_masks.append(None)
                box_masks.append(None)
                continue
            is_available = bool(metric_filter.available(context))
            available[metric_filter.name] = is_available
            self.coverage_seen[metric_filter.name] += 1
            if not is_available:
                point_masks.append(None)
                box_masks.append(None)
                continue
            self.coverage_covered[metric_filter.name] += 1
            point_masks.append(np.asarray(metric_filter.keep(geometry.xyz, context), dtype=bool))
            box_masks.append(
                np.asarray(metric_filter.keep(geometry.gt_boxes, context), dtype=bool)
                if geometry.gt_boxes.shape[0]
                else np.zeros(0, dtype=bool)
            )

        range_masks: List[Optional[NDArray]] = [None] + [r.contains(geometry.bev_radius) for r in self.ranges]
        box_range_masks: List[Optional[NDArray]] = [None] + [r.contains(geometry.box_radius) for r in self.ranges]

        for filter_bucket in range(len(self.filters)):
            if filter_bucket > 0 and not available.get(self.filter_names[filter_bucket], False):
                continue
            for range_bucket in range(len(self.range_names)):
                point_mask = _combine(raw.valid, point_masks[filter_bucket], range_masks[range_bucket])
                self.confusion.add(filter_bucket, range_bucket, raw.target[point_mask], raw.pred[point_mask])
                box_mask = _combine(
                    np.ones(geometry.gt_boxes.shape[0], dtype=bool),
                    box_masks[filter_bucket],
                    box_range_masks[range_bucket],
                )
                for taxonomy in self.taxonomies:
                    components = self._components[(taxonomy, filter_bucket, range_bucket)]
                    if not components:
                        continue
                    taxonomy_frame = taxonomy_frames[taxonomy]
                    mask = (
                        point_mask
                        if taxonomy == RAW_TAXONOMY
                        else _combine(taxonomy_frame.valid, point_masks[filter_bucket], range_masks[range_bucket])
                    )
                    view = self._view(frame, taxonomy_frame, geometry, mask, box_mask, filter_bucket, range_bucket)
                    for component in components:
                        component.update(view)

        return SegmentationFrameSummary(
            frame_name=frame.frame_name,
            scene_id=frame.scene_id,
            num_points=frame.num_points,
            num_valid=int(raw.valid.sum()),
            num_errors=int(np.sum(raw.pred[raw.valid] != raw.target[raw.valid])),
            filter_available=available,
        )

    def _geometry(self, frame: SegmentationFrame) -> FrameGeometry:
        xyz = np.asarray(frame.coordinates[:, :3], dtype=np.float64)
        if self.config.needs_boxes and frame.gt_objects:
            gt_boxes, seg_class, det_label = gt_boxes_from_objects(
                frame.gt_objects, frame.transforms, self.config.box_label_to_seg_class, self.config.class_names
            )
        else:
            gt_boxes = np.zeros((0, 9), dtype=np.float64)
            seg_class = np.zeros(0, dtype=np.int64)
            det_label = np.zeros(0, dtype=np.int64)
        return FrameGeometry(
            xyz=xyz,
            bev_radius=np.linalg.norm(xyz[:, :2], axis=1) if xyz.shape[0] else np.zeros(0, dtype=np.float64),
            gt_boxes=gt_boxes,
            gt_box_seg_class=seg_class,
            gt_box_det_label=det_label,
        )

    def _taxonomy_frames(self, frame: SegmentationFrame) -> Dict[str, TaxonomyFrame]:
        frames = {RAW_TAXONOMY: build_raw_taxonomy(frame, self.config.class_names, self.config.ignore_index)}
        if self.group_lut is not None:
            frames[GROUPED_TAXONOMY] = build_grouped_taxonomy(
                frame, self.group_lut, self.group_names, self.config.ignore_index, GROUPED_TAXONOMY
            )
        return frames

    def _view(
        self,
        frame: SegmentationFrame,
        taxonomy_frame: TaxonomyFrame,
        geometry: FrameGeometry,
        point_mask: NDArray,
        box_mask: NDArray,
        filter_bucket: int,
        range_bucket: int,
    ) -> SegmentationView:
        seg_class = geometry.gt_box_seg_class[box_mask]
        if taxonomy_frame.name != RAW_TAXONOMY and self.group_lut is not None:
            seg_class = fold_labels(seg_class, self.group_lut)
        return SegmentationView(
            taxonomy=taxonomy_frame.name,
            filter_name=self.filter_names[filter_bucket],
            range_name=self.range_names[range_bucket] or None,
            frame_name=frame.frame_name,
            pred=taxonomy_frame.pred[point_mask],
            target=taxonomy_frame.target[point_mask],
            confidence=taxonomy_frame.confidence[point_mask],
            entropy=taxonomy_frame.entropy[point_mask],
            xyz=geometry.xyz[point_mask],
            gt_boxes=geometry.gt_boxes[box_mask],
            gt_box_seg_class=seg_class,
            gt_box_det_label=geometry.gt_box_det_label[box_mask],
            num_classes=taxonomy_frame.num_classes,
            class_names=taxonomy_frame.class_names,
        )

    # ------------------------------------------------------------------------------------------
    def compute(self) -> SegmentationMetricsReport:
        """Build the scene report from the accumulated statistics."""
        report = MetricReport()
        for metric_filter in self.filters:
            name = metric_filter.name
            if not name or not (metric_filter.requires_map or metric_filter.requires_pose):
                continue  # only availability-gated filters report coverage
            seen, covered = self.coverage_seen[name], self.coverage_covered[name]
            report.coverage[name] = (covered, seen)
            if seen and covered == 0:
                report.warn(f"Filter {name!r} covered 0/{seen} frames (no lanelet map / ego pose): its values are NaN.")
            elif covered < seen:
                report.warn(
                    f"Filter {name!r} ran on {covered}/{seen} frames; uncovered frames are excluded from its slice only."
                )
        for taxonomy in self.taxonomies:
            taxonomy_token = None if taxonomy == RAW_TAXONOMY else taxonomy
            for filter_bucket, filter_name in enumerate(self.filter_names):
                uncovered = bool(filter_name) and self.coverage_covered[filter_name] == 0
                for range_bucket, range_suffix in enumerate(self.range_suffixes):
                    values: Dict[str, float] = {}
                    state = ConfusionState.from_matrix(
                        self.confusion.matrix(filter_bucket, range_bucket),
                        self.config.class_names,
                        lut=None if taxonomy == RAW_TAXONOMY else self.group_lut,
                        grouped_names=None if taxonomy == RAW_TAXONOMY else self.group_names,
                    )
                    for spec in self.confusion_specs:
                        if spec.type == "confusion_matrix" and not self.config.include_confusion_cells:
                            continue
                        values.update(spec.instantiate().evaluate(state))
                    for component in self._components[(taxonomy, filter_bucket, range_bucket)]:
                        values.update(component.compute())
                    if uncovered:
                        values = {key: float("nan") for key in values}
                    for key, value in values.items():
                        report.add(
                            compose_key(
                                TASK_KEY,
                                key,
                                taxonomy=taxonomy_token,
                                filter_name=filter_name or None,
                                range_suffix_=range_suffix,
                            ),
                            value,
                        )
        for warning in self.warnings:
            report.warn(warning)
        return SegmentationMetricsReport(
            report=report,
            confusion=self.confusion.confusion.copy(),
            class_names=self.config.class_names,
            group_names=self.group_names,
            group_lut=self.group_lut,
            filter_names=self.filter_names,
            range_names=self.range_names,
            num_frames=self.num_frames,
        )


def _combine(base: NDArray, *masks: Optional[NDArray]) -> NDArray[np.bool_]:
    result = np.asarray(base, dtype=bool)
    for mask in masks:
        if mask is not None:
            result = result & np.asarray(mask, dtype=bool)
    return result
