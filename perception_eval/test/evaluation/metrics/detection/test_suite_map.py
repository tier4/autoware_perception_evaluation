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

"""End-to-end advanced detection suite with a synthetic lanelet map: region/collision filters and B1/B2."""

import math
import unittest

from perception_eval.common.label import AutowareLabel
from perception_eval.common.label import Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.schema import FrameID
from perception_eval.common.shape import Shape
from perception_eval.common.shape import ShapeType
from perception_eval.common.transform import HomogeneousMatrix
from perception_eval.common.transform import TransformDict
from perception_eval.evaluation.metrics.detection.config import AdvancedDetectionMetricsConfig
from perception_eval.evaluation.metrics.detection.frame import DetectionFrame
from perception_eval.evaluation.metrics.detection.suite import AdvancedDetectionSuite
from perception_eval.evaluation.metrics.geometry.lanelet import LaneletMap
from pyquaternion import Quaternion
from shapely.geometry import Polygon

TARGET_LABELS = [AutowareLabel.CAR, AutowareLabel.PEDESTRIAN]


class _StubProvider:
    """One fixed map for every scene except those declared map-less."""

    def __init__(self, lanelet_map: LaneletMap, no_map=()) -> None:
        self._map = lanelet_map
        self._no_map = set(no_map)

    def get(self, scene_id):
        if scene_id in self._no_map:
            raise FileNotFoundError(scene_id)
        return self._map

    def available(self, scene_id) -> bool:
        return scene_id not in self._no_map


def _road_map() -> LaneletMap:
    # Road x in [-20, 80], y in [-6, 6] in the map frame with a 40 km/h limit.
    road = Polygon([(-20, -6), (80, -6), (80, 6), (-20, 6)])
    return LaneletMap({"road": [road]}, [(road, 40.0 / 3.6)])


def _object(
    x: float, y: float, label: AutowareLabel, score: float = 1.0, size=(2.0, 4.0, 1.5), yaw: float = math.pi
) -> DynamicObject:
    """A box in base_link; the default heading is oncoming, because a lead vehicle at the map speed
    limit is unreachable by the collision model (TTC inf) and carries no criticality weight."""
    return DynamicObject(
        unix_time=0,
        frame_id=FrameID.BASE_LINK,
        position=(x, y, 0.0),
        orientation=Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw),
        shape=Shape(ShapeType.BOUNDING_BOX, size),
        velocity=None,
        semantic_score=score,
        semantic_label=Label(label, label.value),
    )


def _frame(name: str, estimated, ground_truth, scene_id="scene") -> DetectionFrame:
    ego2map = HomogeneousMatrix((0.0, 0.0, 0.0), Quaternion(), src=FrameID.BASE_LINK, dst=FrameID.MAP)
    return DetectionFrame(
        frame_name=name,
        unix_time=int(name),
        scene_id=scene_id,
        estimated_objects=tuple(estimated),
        ground_truth_objects=tuple(ground_truth),
        transforms=TransformDict([ego2map]),
    )


SECTION = {
    "filters": [
        {"name": "road", "type": "region", "regions": ["road"]},
        {"name": "collision", "type": "collision"},
    ],
    "components": [
        {"type": "corner_error"},
        {"type": "critical_fp_fn", "confidences": [0.5]},
        {"type": "collision_weighted_map", "thresholds": [2.0]},
    ],
    "map": {"resolver": "explicit", "mapping": {"scene": "/unused/lanelet2_map.osm"}, "max_speed_mps": 10.0},
}


class TestAdvancedDetectionSuiteWithMap(unittest.TestCase):
    def setUp(self):
        self.config = AdvancedDetectionMetricsConfig.from_dict(SECTION, TARGET_LABELS)
        self.provider = _StubProvider(_road_map())

    def test_region_collision_and_criticality_metrics(self):
        gt = [
            _object(20.0, 0.0, AutowareLabel.CAR),
            _object(30.0, 40.0, AutowareLabel.PEDESTRIAN, size=(0.6, 0.6, 1.7)),
        ]
        # A phantom pedestrian on the road ahead is reachable (a same-speed lead car would not be: TTC inf).
        phantom = _object(30.0, 0.0, AutowareLabel.PEDESTRIAN, 0.8, size=(0.6, 0.6, 1.7))
        est = [_object(20.0, 0.0, AutowareLabel.CAR, 0.9), phantom]
        report = AdvancedDetectionSuite(self.config, map_provider=self.provider).evaluate(
            [_frame("0", est, gt), _frame("1", est, gt)], TARGET_LABELS
        )
        self.assertEqual(report.coverage["road"], (2, 2))
        self.assertEqual(report.coverage["collision"], (2, 2))
        self.assertEqual(report.coverage["ttc"], (2, 2))
        # The pedestrian 40 m off the road is outside the road view and unreachable within 4 s.
        self.assertTrue(math.isnan(report.values["detection/road/corner_mean_pedestrian"]))
        self.assertAlmostEqual(report.values["detection/road/corner_mean_car"], 0.0)
        self.assertAlmostEqual(report.values["detection/collision/corner_mean_car"], 0.0)
        # B1: the phantom pedestrian 30 m ahead is in ego's reachable set -> 1 critical FP per frame.
        self.assertAlmostEqual(report.values["detection/critical_fp_0p5m"], 1.0)
        self.assertAlmostEqual(report.values["detection/critical_fn_0p5m"], 0.0)
        self.assertAlmostEqual(report.values["detection/critical_fp_pedestrian_0p5m"], 1.0)
        self.assertAlmostEqual(report.values["detection/critical_fp_car_0p5m"], 0.0)
        # B2: the matched car is perfect; the off-road pedestrian GT has zero weight so its AP is undefined.
        self.assertAlmostEqual(report.values["detection/cw_mAP_car"], 1.0)
        self.assertTrue(math.isnan(report.values["detection/cw_mAP_pedestrian"]))
        self.assertAlmostEqual(report.values["detection/cw_mAP"], 1.0)
        self.assertEqual(report.warnings, [])

    def test_partial_map_coverage(self):
        provider = _StubProvider(_road_map(), no_map=("nomap",))
        gt = [_object(20.0, 0.0, AutowareLabel.CAR)]
        est = [_object(20.0, 0.0, AutowareLabel.CAR, 0.9)]
        frames = [_frame("0", est, gt), _frame("1", est, gt, scene_id="nomap")]
        report = AdvancedDetectionSuite(self.config, map_provider=provider).evaluate(frames, TARGET_LABELS)
        self.assertEqual(report.coverage["road"], (1, 2))
        self.assertEqual(report.coverage["ttc"], (1, 2))
        self.assertGreaterEqual(len(report.warnings), 2)
        # Denominators count covered frames only: no critical errors, a perfect weighted AP.
        self.assertAlmostEqual(report.values["detection/critical_fp_0p5m"], 0.0)
        self.assertAlmostEqual(report.values["detection/cw_mAP_car"], 1.0)

    def test_no_coverage_is_nan(self):
        provider = _StubProvider(_road_map(), no_map=("nomap",))
        gt = [_object(20.0, 0.0, AutowareLabel.CAR)]
        report = AdvancedDetectionSuite(self.config, map_provider=provider).evaluate(
            [_frame("0", gt, gt, scene_id="nomap")], TARGET_LABELS
        )
        self.assertEqual(report.coverage["ttc"], (0, 1))
        self.assertTrue(math.isnan(report.values["detection/critical_fp_0p5m"]))
        self.assertTrue(math.isnan(report.values["detection/cw_mAP"]))
        self.assertTrue(math.isnan(report.values["detection/road/corner_mean_car"]))
        self.assertAlmostEqual(report.values["detection/corner_mean_car"], 0.0)


if __name__ == "__main__":
    unittest.main()
