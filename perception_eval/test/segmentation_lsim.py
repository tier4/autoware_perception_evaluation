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

"""Mock of a point-cloud segmentation evaluation with synthetic frames.

Two frames are evaluated: one with an ego pose and a scene id (map-covered when ``--map_root``
points at a T4 dataset root that ships ``<scene>/map/lanelet2_map.osm``) and one without either,
so the coverage bookkeeping of map-dependent filters is exercised even without real data.
"""

from __future__ import annotations

import argparse
import logging
import os.path as osp
import tempfile
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.shape import Shape
from perception_eval.common.shape import ShapeType
from perception_eval.common.transform import HomogeneousMatrix
from perception_eval.common.transform import TransformDict
from perception_eval.config import SegmentationEvaluationConfig
from perception_eval.evaluation.metrics.segmentation import SegmentationFrame
from perception_eval.evaluation.metrics.segmentation import SegmentationMetricsReport
from perception_eval.manager import SegmentationEvaluationManager
from perception_eval.util.logger_config import configure_logger
from pyquaternion import Quaternion

CLASS_NAMES = ["car", "pedestrian", "road", "sidewalk", "vegetation"]


def evaluation_config_dict(map_root: Optional[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "evaluation_task": "segmentation",
        "class_names": CLASS_NAMES,
        "ignore_index": -1,
        "ranges": [
            {"name": "0_30", "min_distance": 0.0, "max_distance": 30.0},
            {"name": "30_60", "min_distance": 30.0, "max_distance": 60.0},
        ],
        "class_groups": {
            "vehicle": ["car"],
            "vru": ["pedestrian"],
            "flat": ["road", "sidewalk"],
            "other": ["vegetation"],
        },
        "filters": [{"name": "corridor", "type": "corridor", "width_m": 3.0}],
        "components": [
            {"type": "confusion_matrix"},
            {"type": "iou"},
            {"type": "accuracy"},
            {"type": "precision_recall_f1"},
            {"type": "calibration", "num_bins": 15},
            {"type": "uncertainty_usefulness"},
            {"type": "confident_error", "entropy_threshold": 0.3},
            {"type": "error_clusters", "cluster_radius": 0.5},
            {"type": "tolerant_error", "radius": 0.2},
            {"type": "partial_detection", "half_saturation": 1.0, "min_points": 1},
        ],
        "box_label_to_seg_class": {"car": "car", "pedestrian": "pedestrian"},
        "check_argmax": True,
    }
    if map_root is not None:
        cfg["filters"].append({"name": "road", "type": "region", "regions": ["road", "road_shoulder", "crosswalk"]})
        cfg["map"] = {"resolver": "t4_scene_directory", "data_root": map_root}
    return cfg


def _object(x: float, y: float, label: AutowareLabel, size) -> DynamicObject:
    return DynamicObject(
        unix_time=0,
        frame_id=FrameID.BASE_LINK,
        position=(x, y, 0.0),
        orientation=Quaternion(axis=[0, 0, 1], angle=0.0),
        shape=Shape(ShapeType.BOUNDING_BOX, size),
        velocity=None,
        semantic_score=1.0,
        semantic_label=Label(label, label.value),
    )


def make_synthetic_frame(
    rng: np.random.Generator,
    frame_name: str,
    scene_id: Optional[str],
    num_points: int,
    with_pose: bool,
) -> SegmentationFrame:
    """Random points in a 60 m disk with a spatial class rule, 8% flipped predictions and peaked probabilities."""
    radius = 60.0 * np.sqrt(rng.uniform(0.0, 1.0, num_points))
    angle = rng.uniform(-np.pi, np.pi, num_points)
    coords = np.stack([radius * np.cos(angle), radius * np.sin(angle), rng.normal(0.0, 0.2, num_points)], axis=1)
    targets = np.full(num_points, CLASS_NAMES.index("vegetation"), dtype=np.int64)
    targets[np.abs(coords[:, 1]) < 3.0] = CLASS_NAMES.index("road")
    targets[(np.abs(coords[:, 1]) >= 3.0) & (np.abs(coords[:, 1]) < 5.0)] = CLASS_NAMES.index("sidewalk")
    car = _object(15.0, 0.0, AutowareLabel.CAR, (2.0, 4.5, 1.6))
    pedestrian = _object(8.0, 4.0, AutowareLabel.PEDESTRIAN, (0.6, 0.6, 1.7))
    in_car = (np.abs(coords[:, 0] - 15.0) < 2.25) & (np.abs(coords[:, 1]) < 1.0)
    in_ped = (np.abs(coords[:, 0] - 8.0) < 0.3) & (np.abs(coords[:, 1] - 4.0) < 0.3)
    targets[in_car] = CLASS_NAMES.index("car")
    targets[in_ped] = CLASS_NAMES.index("pedestrian")
    predictions = targets.copy()
    flip = rng.random(num_points) < 0.08
    predictions[flip] = rng.integers(0, len(CLASS_NAMES), int(flip.sum()))
    probabilities = rng.dirichlet(np.ones(len(CLASS_NAMES)) * 0.5, num_points)
    probabilities[np.arange(num_points), predictions] += 2.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    transforms = TransformDict(
        [HomogeneousMatrix((100.0, 50.0, 0.0), Quaternion(axis=[0, 0, 1], angle=0.3), FrameID.BASE_LINK, FrameID.MAP)]
        if with_pose
        else None
    )
    return SegmentationFrame(
        frame_name=frame_name,
        scene_id=scene_id,
        coordinates=coords.astype(np.float32),
        targets=targets,
        predictions=predictions,
        probabilities=probabilities.astype(np.float32),
        transforms=transforms,
        gt_objects=(car, pedestrian),
        unix_time=0,
    )


class SegmentationLSimMoc:
    """Moc to evaluate point-cloud segmentation with externally supplied arrays."""

    def __init__(self, result_root_directory: str, map_root: Optional[str] = None) -> None:
        evaluation_config = SegmentationEvaluationConfig(
            dataset_paths=[],
            frame_id="base_link",
            result_root_directory=result_root_directory,
            evaluation_config_dict=evaluation_config_dict(map_root),
        )
        _ = configure_logger(
            log_file_directory=evaluation_config.log_directory,
            console_log_level=logging.INFO,
            file_log_level=logging.INFO,
        )
        self.evaluator = SegmentationEvaluationManager(evaluation_config=evaluation_config)

    def callback(self, frame: SegmentationFrame) -> None:
        summary = self.evaluator.add_frame(frame)
        logging.info(
            "frame %s: %d points, %d valid, %d errors, filters %s",
            summary.frame_name,
            summary.num_points,
            summary.num_valid,
            summary.num_errors,
            summary.filter_available,
        )

    def get_final_result(self) -> SegmentationMetricsReport:
        report = self.evaluator.get_scene_result(save_report=True)
        logging.info("\n%s", str(report))
        logging.info(
            "report written to %s", osp.join(self.evaluator.evaluator_config.log_directory, "segmentation_metrics.json")
        )
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use_tmpdir", action="store_true", help="Whether save results to temporal directory")
    parser.add_argument("--num_points", type=int, default=50_000, help="Points per synthetic frame")
    parser.add_argument("--map_root", type=str, default=None, help="T4 dataset root with <scene>/map/lanelet2_map.osm")
    parser.add_argument("--scene_id", type=str, default="scene_with_map", help="Scene id under --map_root")
    args = parser.parse_args()

    if args.use_tmpdir:
        tmpdir = tempfile.TemporaryDirectory()
        result_root_directory: str = tmpdir.name
    else:
        result_root_directory = "data/result/{TIME}/"

    lsim = SegmentationLSimMoc(result_root_directory, args.map_root)
    rng = np.random.default_rng(0)
    lsim.callback(make_synthetic_frame(rng, "0", args.scene_id, args.num_points, with_pose=True))
    lsim.callback(make_synthetic_frame(rng, "1", None, args.num_points, with_pose=False))
    final_report = lsim.get_final_result()
    logging.info("mIoU: %s", final_report.get("segmentation/mIoU"))
    logging.info("grouped mIoU: %s", final_report.get("segmentation/grouped/mIoU"))
