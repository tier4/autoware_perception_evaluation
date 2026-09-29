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
# Ported from tier4/autoware-ml (autoware_ml/metrics/detection3d/suite.py, base.py) at fcf86419.

"""The advanced detection metric suite: a scene-level state engine.

``AdvancedDetectionSuite`` converts the retained :class:`DetectionFrame` objects into NumPy
samples once, precomputes every filter mask (and, when configured, the per-box reachability TTC)
per frame, and then builds a :class:`DetectionState` for each ``(taxonomy, filter, range)`` view
and hands it to the configured components. Keys are composed as
``detection/<taxonomy?>/<filter?>/<range?>/<metric-key>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.class_groups import resolve_class_groups
from perception_eval.evaluation.metrics.component import compose_key
from perception_eval.evaluation.metrics.component import GROUPED_TAXONOMY
from perception_eval.evaluation.metrics.component import MetricRange
from perception_eval.evaluation.metrics.component import MetricReport
from perception_eval.evaluation.metrics.detection.adapter import MissingTransformError
from perception_eval.evaluation.metrics.detection.adapter import objects_to_arrays
from perception_eval.evaluation.metrics.detection.adapter import UnsupportedShapeError
from perception_eval.evaluation.metrics.detection.components import get_component_class
from perception_eval.evaluation.metrics.detection.config import AdvancedDetectionMetricsConfig
from perception_eval.evaluation.metrics.detection.config import component_kwargs
from perception_eval.evaluation.metrics.detection.config import component_token
from perception_eval.evaluation.metrics.detection.frame import DetectionFrame
from perception_eval.evaluation.metrics.detection.state import DetectionSample
from perception_eval.evaluation.metrics.detection.state import DetectionState
from perception_eval.evaluation.metrics.filter_config import build_filters
from perception_eval.evaluation.metrics.filters import FrameMetricContext
from perception_eval.evaluation.metrics.filters import IdentityFilter
from perception_eval.evaluation.metrics.filters import MetricFilter

logger = getLogger(__name__)

TASK_KEY = "detection"
TTC_COVERAGE_KEY = "ttc"


@dataclass
class _PreparedFrame:
    """One frame converted to arrays plus its per-filter masks and availability."""

    sample: DetectionSample
    context: FrameMetricContext
    available: Dict[str, bool]
    pred_masks: Dict[str, NDArray]
    gt_masks: Dict[str, NDArray]


class AdvancedDetectionSuite:
    """Evaluate the configured driving-aware components over a scene's frames."""

    def __init__(self, config: AdvancedDetectionMetricsConfig, map_provider: Any = None) -> None:
        self.config = config
        self.class_names: Tuple[str, ...] = tuple(config.class_names)
        self.ranges: Tuple[MetricRange, ...] = tuple(config.ranges)
        self.map_provider = map_provider
        if self.map_provider is None and config.requires_map and config.map is not None:
            self.map_provider = config.map.build_provider()
        self.filters: List[MetricFilter] = build_filters(config.filters, self.map_provider, config.map)
        self.components = [
            get_component_class(component_token(component_config))(**component_kwargs(component_config))
            for component_config in config.components
        ]
        self._group_lut: Optional[NDArray] = None
        self._grouped_names: Optional[Tuple[str, ...]] = None
        if config.class_groups:
            self._group_lut, self._grouped_names = resolve_class_groups(self.class_names, config.class_groups)
        self._collision = self._build_collision_provider() if config.needs_ttc else None

    # ------------------------------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------------------------------

    def _build_collision_provider(self) -> Any:
        from perception_eval.evaluation.metrics.detection.collision import CollisionTTC
        from perception_eval.evaluation.metrics.detection.collision import DEFAULT_COLLISION_KINDS
        from perception_eval.evaluation.metrics.detection.collision import DEFAULT_VRU_SPEEDS
        from perception_eval.evaluation.metrics.geometry.reachability import ReachabilityParams

        map_config = self.config.map
        if map_config is None or self.map_provider is None:
            raise ValueError("TTC-based components need the `map` section of advanced_detection_metrics.")
        kinds = dict(DEFAULT_COLLISION_KINDS)
        kinds.update(map_config.collision_kinds)
        vru_speeds = dict(DEFAULT_VRU_SPEEDS)
        vru_speeds.update(map_config.vru_speeds)
        params = ReachabilityParams()
        for filter_config in self.config.filters:
            if type(filter_config).__name__ == "CollisionFilterConfig":
                params = ReachabilityParams(
                    horizon_s=filter_config.horizon_s,
                    dt_s=filter_config.dt_s,
                    max_lateral_accel_mps2=filter_config.max_lateral_accel_mps2,
                    min_radius_m=filter_config.min_radius_m,
                    arc_samples=filter_config.arc_samples,
                )
                break
        return CollisionTTC(
            class_names=self.class_names,
            map_provider=self.map_provider,
            kinds=kinds,
            region=map_config.region,
            params=params,
            vru_speeds=vru_speeds,
            max_speed_mps=map_config.max_speed_mps,
            ego_body_radius_m=map_config.ego_body_m,
        )

    # ------------------------------------------------------------------------------------------
    # Frame preparation
    # ------------------------------------------------------------------------------------------

    def _prepare_frame(
        self, frame: DetectionFrame, target_labels: Sequence[Any], report: MetricReport
    ) -> Optional[_PreparedFrame]:
        try:
            pred_boxes, pred_labels, pred_scores = objects_to_arrays(
                frame.estimated_objects, frame.transforms, target_labels
            )
            gt_boxes, gt_labels, _ = objects_to_arrays(frame.ground_truth_objects, frame.transforms, target_labels)
        except (UnsupportedShapeError, MissingTransformError) as error:
            report.warn(f"frame {frame.frame_name!r} skipped by the advanced detection metrics: {error}")
            return None

        context = FrameMetricContext(ego2map=frame.ego2map, scene_id=frame.scene_id)
        gt_ttc = pred_ttc = None
        ttc_covered = True
        if self._collision is not None:
            ttc_covered = context.has_pose and self._collision.available(frame.scene_id)
            if ttc_covered:
                gt_ttc = self._collision.per_box_ttc(gt_boxes, gt_labels, context.ego2map, frame.scene_id)
                pred_ttc = self._collision.per_box_ttc(pred_boxes, pred_labels, context.ego2map, frame.scene_id)
            else:
                gt_ttc = np.full(gt_boxes.shape[0], np.inf)
                pred_ttc = np.full(pred_boxes.shape[0], np.inf)

        sample = DetectionSample(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_ttc=gt_ttc,
            pred_ttc=pred_ttc,
            ttc_covered=ttc_covered,
            frame_name=frame.frame_name,
        )

        available: Dict[str, bool] = {}
        pred_masks: Dict[str, NDArray] = {}
        gt_masks: Dict[str, NDArray] = {}
        for metric_filter in self.filters:
            if isinstance(metric_filter, IdentityFilter):
                continue
            is_available = bool(metric_filter.available(context))
            if getattr(metric_filter, "requires_pose", False) and not context.has_pose:
                is_available = False
            available[metric_filter.name] = is_available
            if is_available:
                pred_masks[metric_filter.name] = np.asarray(metric_filter.keep(pred_boxes, context), dtype=bool)
                gt_masks[metric_filter.name] = np.asarray(metric_filter.keep(gt_boxes, context), dtype=bool)
        return _PreparedFrame(
            sample=sample, context=context, available=available, pred_masks=pred_masks, gt_masks=gt_masks
        )

    # ------------------------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------------------------

    def evaluate(self, frames: Sequence[DetectionFrame], target_labels: Optional[Sequence[Any]] = None) -> MetricReport:
        """Run every component over every ``(taxonomy, filter, range)`` view of ``frames``.

        Args:
            frames (Sequence[DetectionFrame]): Ordered scene frames.
            target_labels (Sequence[LabelType] | None): Label order matching ``config.class_names``;
                derived from the class names when omitted.
        """
        report = MetricReport()
        labels = list(target_labels) if target_labels is not None else self._labels_from_class_names()

        prepared = [self._prepare_frame(frame, labels, report) for frame in frames]
        prepared = [frame for frame in prepared if frame is not None]

        # Coverage per availability-gated filter and for TTC.
        for metric_filter in self.filters:
            if isinstance(metric_filter, IdentityFilter):
                continue
            seen = len(prepared)
            covered = sum(1 for frame in prepared if frame.available.get(metric_filter.name, False))
            report.coverage[metric_filter.name] = (covered, seen)
            if seen and covered < seen:
                report.warn(
                    f"filter {metric_filter.name!r} covered {covered}/{seen} frames; uncovered frames "
                    "(no ego pose or no lanelet map for the scene) are excluded from its view only."
                )
        if self._collision is not None:
            seen = len(prepared)
            covered = sum(1 for frame in prepared if frame.sample.ttc_covered)
            report.coverage[TTC_COVERAGE_KEY] = (covered, seen)
            if seen and covered < seen:
                report.warn(
                    f"collision TTC covered {covered}/{seen} frames; uncovered frames are excluded from the "
                    "criticality metrics' denominators."
                )

        taxonomies: List[Tuple[Optional[str], Optional[NDArray], Tuple[str, ...]]] = [(None, None, self.class_names)]
        if self._group_lut is not None and self._grouped_names is not None:
            taxonomies.append((GROUPED_TAXONOMY, self._group_lut, self._grouped_names))

        for taxonomy, lut, class_names in taxonomies:
            for metric_filter in self.filters:
                filter_name = None if isinstance(metric_filter, IdentityFilter) else metric_filter.name
                samples = self._view_samples(prepared, metric_filter, lut)
                for metric_range in (None, *self.ranges):
                    view_samples = samples if metric_range is None else [s.clip_to_range(metric_range) for s in samples]
                    state = DetectionState(
                        samples=view_samples, class_names=class_names, match_cost=self.config.match_cost
                    )
                    range_suffix = None if metric_range is None else metric_range.suffix
                    zero_coverage = filter_name is not None and report.coverage.get(filter_name, (0, 0))[0] == 0
                    for component in self.components:
                        values = component.evaluate(state)
                        for key, value in values.items():
                            full_key = compose_key(TASK_KEY, key, taxonomy, filter_name, range_suffix)
                            report.add(full_key, float("nan") if zero_coverage else value)
        return report

    def _labels_from_class_names(self) -> List[Any]:
        from perception_eval.common.label import AutowareLabel
        from perception_eval.common.label import TrafficLightLabel

        labels: List[Any] = []
        for name in self.class_names:
            for enum_cls in (AutowareLabel, TrafficLightLabel):
                try:
                    labels.append(enum_cls(name))
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"class name {name!r} is not an AutowareLabel or TrafficLightLabel value.")
        return labels

    def _view_samples(
        self, prepared: Sequence[_PreparedFrame], metric_filter: MetricFilter, lut: Optional[NDArray]
    ) -> List[DetectionSample]:
        samples: List[DetectionSample] = []
        for frame in prepared:
            sample = frame.sample
            if not isinstance(metric_filter, IdentityFilter):
                if not frame.available.get(metric_filter.name, False):
                    continue  # uncovered frame leaves this view entirely
                sample = sample.select(frame.pred_masks[metric_filter.name], frame.gt_masks[metric_filter.name])
            if lut is not None:
                sample = sample.fold_labels(lut)
            samples.append(sample)
        return samples
