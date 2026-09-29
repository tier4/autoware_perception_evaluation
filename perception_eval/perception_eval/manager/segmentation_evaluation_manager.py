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

"""Streaming evaluation manager for point-cloud semantic segmentation."""

from __future__ import annotations

import logging
import os.path as osp
from typing import Any
from typing import List
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.schema import FrameID
from perception_eval.config.segmentation_evaluation_config import SegmentationEvaluationConfig
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics.segmentation.report import SegmentationMetricsReport
from perception_eval.evaluation.metrics.segmentation.state import SegmentationFrame
from perception_eval.evaluation.metrics.segmentation.state import SegmentationFrameSummary
from perception_eval.evaluation.metrics.segmentation.suite import SegmentationMetricSuite
from perception_eval.manager._evaluation_manager_base import _EvaluationManagerBase

logger = logging.getLogger(__name__)


class SegmentationEvaluationManager(_EvaluationManagerBase):
    """Evaluate per-point semantic segmentation frame by frame with bounded memory.

    Frames are supplied as already-decoded NumPy arrays (:class:`SegmentationFrame`). The manager
    never retains frames: each ``add_frame`` folds the frame into the suite's sufficient statistics
    and returns a small :class:`SegmentationFrameSummary`.

    Args:
        evaluation_config (SegmentationEvaluationConfig): Configuration.
        load_ground_truth (bool): Whether to load the dataset ground truth (boxes, transforms, raw
            point clouds) so :meth:`frame_from_ground_truth` can build frames. Defaults to False.
        map_provider: Optional lanelet map provider (built from the config's ``map`` section otherwise).
    """

    def __init__(
        self,
        evaluation_config: SegmentationEvaluationConfig,
        load_ground_truth: bool = False,
        map_provider: Any = None,
    ) -> None:
        super().__init__(evaluation_config=evaluation_config, load_ground_truth=load_ground_truth)
        if not load_ground_truth:
            self.ground_truth_frames: List[FrameGroundTruth] = []
        self.metrics_config = evaluation_config.metrics_config
        self.suite = SegmentationMetricSuite(self.metrics_config, map_provider=map_provider)
        self.frame_summaries: List[SegmentationFrameSummary] = []

    # ------------------------------------------------------------------------------------------
    @property
    def visualizer(self):
        raise NotImplementedError("Segmentation evaluation has no visualizer yet.")

    def visualize_all(self) -> None:
        raise NotImplementedError("Segmentation evaluation has no visualizer yet.")

    def visualize_frame(self, frame_index: int = -1) -> None:
        raise NotImplementedError("Segmentation evaluation has no visualizer yet.")

    @property
    def num_frames(self) -> int:
        return self.suite.num_frames

    # ------------------------------------------------------------------------------------------
    def add_frame(self, frame: SegmentationFrame) -> SegmentationFrameSummary:
        """Fold one frame into the metrics (streaming) and return its summary."""
        summary = self.suite.update(frame)
        self.frame_summaries.append(summary)
        return summary

    def add_frame_result(self, frame: SegmentationFrame) -> SegmentationFrameSummary:  # type: ignore[override]
        """Alias of :meth:`add_frame` for API parity with the other managers."""
        return self.add_frame(frame)

    def get_scene_result(self, save_report: bool = False) -> SegmentationMetricsReport:
        """Compute the scene report; optionally write ``segmentation_metrics.json`` to the log directory."""
        report = self.suite.compute()
        if save_report:
            path = osp.join(self.evaluator_config.log_directory, "segmentation_metrics.json")
            try:
                report.to_json(path)
            except OSError as error:
                logger.warning("Failed to write %s: %s", path, error)
        return report

    def reset(self) -> None:
        """Drop the accumulated statistics and summaries."""
        self.suite.reset()
        self.frame_summaries = []

    # ------------------------------------------------------------------------------------------
    def frame_from_ground_truth(
        self,
        ground_truth: FrameGroundTruth,
        targets: NDArray,
        predictions: NDArray,
        probabilities: NDArray,
        coordinates: Optional[NDArray] = None,
        scene_id: Optional[str] = None,
    ) -> SegmentationFrame:
        """Build a frame from a loaded ground-truth frame plus per-point arrays (the adapter seam).

        ``coordinates`` default to the LiDAR point cloud in ``ground_truth.raw_data``
        (``LIDAR_CONCAT`` or ``LIDAR_TOP``), so ``load_raw_data=True`` is required in that case.
        Ground-truth boxes are filtered by ``target_uuids`` when configured.
        """
        if coordinates is None:
            raw_data = ground_truth.raw_data or {}
            if FrameID.LIDAR_CONCAT in raw_data:
                coordinates = raw_data[FrameID.LIDAR_CONCAT][:, :3]
            elif FrameID.LIDAR_TOP in raw_data:
                coordinates = raw_data[FrameID.LIDAR_TOP][:, :3]
            else:
                raise ValueError(
                    "coordinates were not given and the ground truth carries no LIDAR_CONCAT/LIDAR_TOP data."
                )
        objects = ground_truth.objects
        target_uuids = self.filtering_params.get("target_uuids")
        if target_uuids:
            objects = filter_objects(objects, is_gt=True, target_uuids=target_uuids)
        return SegmentationFrame(
            frame_name=str(ground_truth.frame_name),
            scene_id=scene_id if scene_id is not None else getattr(ground_truth, "scene_id", None),
            coordinates=np.asarray(coordinates),
            targets=np.asarray(targets),
            predictions=np.asarray(predictions),
            probabilities=np.asarray(probabilities),
            transforms=ground_truth.transforms,
            gt_objects=tuple(objects),
            unix_time=ground_truth.unix_time,
        )
