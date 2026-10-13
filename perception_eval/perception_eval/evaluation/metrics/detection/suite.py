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

``AdvancedDetectionSuite`` works in two steps. :meth:`AdvancedDetectionSuite.prepare` converts the
retained :class:`DetectionFrame` objects into NumPy samples once and computes every filter mask
(and, when configured, the per-box reachability TTC) per frame; this is the only step that needs
the lanelet map. :meth:`AdvancedDetectionSuite.evaluate_prepared` then builds a
:class:`DetectionState` for each ``(taxonomy, filter, range)`` view of the prepared samples and
hands it to the configured components. Keys are composed as
``detection/<taxonomy?>/<filter?>/<range?>/<metric-key>``. :meth:`AdvancedDetectionSuite.evaluate`
chains both steps; prepared samples of several scenes can be pooled before the second step.
"""

from __future__ import annotations

from logging import getLogger
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

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
from perception_eval.evaluation.metrics.detection.prepared import config_digest
from perception_eval.evaluation.metrics.detection.prepared import COUNT_KEYS
from perception_eval.evaluation.metrics.detection.prepared import PoolingError
from perception_eval.evaluation.metrics.detection.prepared import PREPARED_SCHEMA_VERSION
from perception_eval.evaluation.metrics.detection.prepared import PreparedFrame
from perception_eval.evaluation.metrics.detection.prepared import PreparedSamples
from perception_eval.evaluation.metrics.detection.state import DetectionSample
from perception_eval.evaluation.metrics.detection.state import DetectionState
from perception_eval.evaluation.metrics.filter_config import build_filters
from perception_eval.evaluation.metrics.filters import FrameMetricContext
from perception_eval.evaluation.metrics.filters import IdentityFilter
from perception_eval.evaluation.metrics.filters import MetricFilter

logger = getLogger(__name__)

TASK_KEY = "detection"
TTC_COVERAGE_KEY = "ttc"


class AdvancedDetectionSuite:
    """Evaluate the configured driving-aware components over a scene's frames.

    The lanelet map provider, the filters and the collision (TTC) provider are built lazily on the
    first :meth:`prepare` call, so a suite that only runs :meth:`evaluate_prepared` never resolves
    a map.
    """

    def __init__(self, config: AdvancedDetectionMetricsConfig, map_provider: Any = None) -> None:
        self.config = config
        self.class_names: Tuple[str, ...] = tuple(config.class_names)
        self.ranges: Tuple[MetricRange, ...] = tuple(config.ranges)
        self.filter_names: Tuple[str, ...] = tuple(filter_config.name for filter_config in config.filters)
        self.has_ttc: bool = bool(config.needs_ttc)
        self.config_digest: str = config_digest(config)
        self._map_provider = map_provider
        self._filters: Optional[List[MetricFilter]] = None
        self._collision_provider: Any = None
        self.components = [
            get_component_class(component_token(component_config))(**component_kwargs(component_config))
            for component_config in config.components
        ]
        self._group_lut: Optional[NDArray] = None
        self._grouped_names: Optional[Tuple[str, ...]] = None
        if config.class_groups:
            self._group_lut, self._grouped_names = resolve_class_groups(self.class_names, config.class_groups)

    # ------------------------------------------------------------------------------------------
    # Construction helpers (lazy: nothing here runs for evaluate_prepared)
    # ------------------------------------------------------------------------------------------

    @property
    def map_provider(self) -> Any:
        """The lanelet map provider, built from ``config.map`` on first use when not injected."""
        if self._map_provider is None and self.config.requires_map and self.config.map is not None:
            self._map_provider = self.config.map.build_provider()
        return self._map_provider

    @property
    def filters(self) -> List[MetricFilter]:
        """Identity plus the configured filters, built on first use."""
        if self._filters is None:
            self._filters = build_filters(self.config.filters, self.map_provider, self.config.map)
        return self._filters

    @property
    def _collision(self) -> Any:
        """The collision TTC provider when TTC is needed, built on first use; ``None`` otherwise."""
        if self.has_ttc and self._collision_provider is None:
            self._collision_provider = self._build_collision_provider()
        return self._collision_provider

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

    def prepare(
        self, frames: Sequence[DetectionFrame], target_labels: Optional[Sequence[Any]] = None
    ) -> PreparedSamples:
        """Convert ``frames`` to arrays and precompute every filter mask (needs the map, if any).

        Args:
            frames (Sequence[DetectionFrame]): Ordered scene frames.
            target_labels (Sequence[LabelType] | None): Label order matching ``config.class_names``;
                derived from the class names when omitted.

        Returns:
            PreparedSamples: Map-free samples that :meth:`evaluate_prepared` (or a pooled call)
            turns into a report.
        """
        labels = list(target_labels) if target_labels is not None else self._labels_from_class_names()
        counts: Dict[str, int] = {key: 0 for key in COUNT_KEYS}
        counts["input_frames"] = len(frames)
        warnings: List[str] = []
        prepared = [self._prepare_frame(frame, labels, warnings, counts) for frame in frames]
        prepared = [frame for frame in prepared if frame is not None]
        # Frame names repeat across a scene, so per-frame warnings report too few: also report totals once.
        if counts["skipped_frames"]:
            _warn(
                warnings, f"{counts['skipped_frames']}/{len(frames)} frames skipped by the advanced detection metrics."
            )
        if counts["polygon_as_box"]:
            _warn(warnings, f"{counts['polygon_as_box']} POLYGON-shaped objects evaluated as their footprint box.")
        if counts["non_finite_score"]:
            _warn(
                warnings,
                f"{counts['non_finite_score']} predictions dropped: non-finite or missing {self.config.score_source}.",
            )
        if counts["pred_total"] and not counts["pred_kept"]:
            _warn(warnings, f"all {counts['pred_total']} predictions dropped: no {self.config.score_source}")
        return PreparedSamples(
            schema_version=PREPARED_SCHEMA_VERSION,
            config_digest=self.config_digest,
            class_names=self.class_names,
            filter_names=self.filter_names,
            has_ttc=self.has_ttc,
            frames=tuple(prepared),
            counts=counts,
            warnings=tuple(warnings),
        )

    def _prepare_frame(
        self,
        frame: DetectionFrame,
        target_labels: Sequence[Any],
        warnings: List[str],
        counts: Dict[str, int],
    ) -> Optional[PreparedFrame]:
        non_finite_before = counts.get("non_finite_score", 0)
        try:
            pred_boxes, pred_labels, pred_scores = objects_to_arrays(
                frame.estimated_objects,
                frame.transforms,
                target_labels,
                score_source=self.config.score_source,
                polygon_shapes=self.config.polygon_shapes,
                counts=counts,
                drop_non_finite_scores=True,
            )
            gt_boxes, gt_labels, _ = objects_to_arrays(
                frame.ground_truth_objects,
                frame.transforms,
                target_labels,
                polygon_shapes=self.config.polygon_shapes,
            )
        except (UnsupportedShapeError, MissingTransformError) as error:
            counts["skipped_frames"] = counts.get("skipped_frames", 0) + 1
            _warn(warnings, f"frame {frame.frame_name!r} skipped by the advanced detection metrics: {error}")
            return None
        dropped = counts.get("non_finite_score", 0) - non_finite_before
        counts["pred_kept"] = counts.get("pred_kept", 0) + int(pred_scores.shape[0])
        counts["pred_total"] = counts.get("pred_total", 0) + int(pred_scores.shape[0]) + dropped
        counts["zero_score"] = counts.get("zero_score", 0) + int(np.count_nonzero(pred_scores == 0.0))

        context = FrameMetricContext(ego2map=frame.ego2map, scene_id=frame.scene_id)
        gt_ttc = pred_ttc = None
        ttc_covered = True
        collision = self._collision
        if collision is not None:
            ttc_covered = context.has_pose and collision.available(frame.scene_id)
            if ttc_covered:
                gt_ttc = collision.per_box_ttc(gt_boxes, gt_labels, context.ego2map, frame.scene_id)
                pred_ttc = collision.per_box_ttc(pred_boxes, pred_labels, context.ego2map, frame.scene_id)
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
            else:
                pred_masks[metric_filter.name] = np.zeros(pred_boxes.shape[0], dtype=bool)
                gt_masks[metric_filter.name] = np.zeros(gt_boxes.shape[0], dtype=bool)
        return PreparedFrame(
            sample=sample,
            available=available,
            pred_masks=pred_masks,
            gt_masks=gt_masks,
            unix_time=int(frame.unix_time),
        )

    # ------------------------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------------------------

    def evaluate(self, frames: Sequence[DetectionFrame], target_labels: Optional[Sequence[Any]] = None) -> MetricReport:
        """Run every component over every ``(taxonomy, filter, range)`` view of ``frames``.

        Equivalent to ``evaluate_prepared(prepare(frames, target_labels))``.

        Args:
            frames (Sequence[DetectionFrame]): Ordered scene frames.
            target_labels (Sequence[LabelType] | None): Label order matching ``config.class_names``;
                derived from the class names when omitted.
        """
        return self.evaluate_prepared(self.prepare(frames, target_labels))

    def evaluate_prepared(self, prepared: Union[PreparedSamples, Sequence[PreparedSamples]]) -> MetricReport:
        """Run every component over every view of already prepared (possibly pooled) samples.

        Never touches a lanelet map: filters are identified by name from ``prepared.filter_names``
        and their masks are read from the prepared frames.

        Args:
            prepared (PreparedSamples | Sequence[PreparedSamples]): One scene, or several scenes
                pooled in the given order.

        Raises:
            PoolingError: the samples were prepared with a different configuration.
        """
        if not isinstance(prepared, PreparedSamples):
            prepared = PreparedSamples.concat(list(prepared))
        if prepared.schema_version != PREPARED_SCHEMA_VERSION:
            raise PoolingError(f"prepared samples schema version {prepared.schema_version!r} is not supported.")
        if prepared.config_digest != self.config_digest:
            raise PoolingError(
                f"prepared samples digest {prepared.config_digest} does not match the configuration digest "
                f"{self.config_digest}; evaluate with the configuration they were prepared with."
            )
        if prepared.class_names != self.class_names or prepared.filter_names != self.filter_names:
            raise PoolingError("prepared samples class or filter names do not match the configuration.")
        if prepared.has_ttc != self.has_ttc:
            raise PoolingError("prepared samples TTC availability does not match the configuration.")

        report = MetricReport()
        for warning in prepared.warnings:
            report.warn(warning)
        frames = prepared.frames

        # Coverage per availability-gated filter and for TTC.
        seen = len(frames)
        for filter_name in prepared.filter_names:
            covered = sum(1 for frame in frames if frame.available.get(filter_name, False))
            report.coverage[filter_name] = (covered, seen)
            if seen and covered < seen:
                report.warn(
                    f"filter {filter_name!r} covered {covered}/{seen} frames; uncovered frames "
                    "(no ego pose or no lanelet map for the scene) are excluded from its view only."
                )
        if prepared.has_ttc:
            covered = sum(1 for frame in frames if frame.sample.ttc_covered)
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
            for filter_name in (None, *prepared.filter_names):
                samples = self._view_samples(frames, filter_name, lut)
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

    @staticmethod
    def _view_samples(
        prepared: Sequence[PreparedFrame], filter_name: Optional[str], lut: Optional[NDArray]
    ) -> List[DetectionSample]:
        samples: List[DetectionSample] = []
        for frame in prepared:
            sample = frame.sample
            if filter_name is not None:
                if not frame.available.get(filter_name, False):
                    continue  # uncovered frame leaves this view entirely
                sample = sample.select(frame.pred_masks[filter_name], frame.gt_masks[filter_name])
            if lut is not None:
                sample = sample.fold_labels(lut)
            samples.append(sample)
        return samples


def _warn(warnings: List[str], message: str) -> None:
    """Record a warning once (same semantics as :meth:`MetricReport.warn`)."""
    if message not in warnings:
        warnings.append(message)
