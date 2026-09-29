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
# Ported from tier4/autoware-ml (autoware_ml/tests/metrics/test_reachability.py) at fcf86419.

from math import inf
from math import isclose
from math import pi
import unittest

import numpy as np
from perception_eval.evaluation.metrics.geometry.reachability import Agent
from perception_eval.evaluation.metrics.geometry.reachability import collision_weights
from perception_eval.evaluation.metrics.geometry.reachability import EgoReachability
from perception_eval.evaluation.metrics.geometry.reachability import ReachabilityParams
from perception_eval.evaluation.metrics.geometry.reachability import reachable_region
from perception_eval.evaluation.metrics.geometry.reachability import reachable_set
from perception_eval.evaluation.metrics.geometry.reachability import STATIC
from perception_eval.evaluation.metrics.geometry.reachability import time_to_collision
from perception_eval.evaluation.metrics.geometry.reachability import VRU
from perception_eval.evaluation.metrics.geometry.reachability import WHEELED
from shapely.geometry import box
from shapely.geometry import Point

PARAMS = ReachabilityParams(horizon_s=4.0, dt_s=0.1)
# A wide open drivable region so wheeled fronts are not clipped in these unit cases.
ROAD = box(-80.0, -50.0, 500.0, 50.0)


def _footprint(x: float, y: float, size: float = 2.0):
    return box(x - size / 2, y - size / 2, x + size / 2, y + size / 2)


class TestReachability(unittest.TestCase):
    def test_same_speed_lead_never_collides(self):
        ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.2)
        lead = Agent(WHEELED, 25.0, 0.0, heading=0.0, speed=10.0, body_radius=1.2)
        self.assertEqual(time_to_collision(ego, lead, ROAD, PARAMS), inf)

    def test_stationary_object_ahead_finite_near_distance_over_speed(self):
        ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
        obj = Agent(STATIC, 30.0, 0.0, footprint=_footprint(30.0, 0.0), body_radius=1.0)
        ttc = time_to_collision(ego, obj, ROAD, PARAMS)
        self.assertNotEqual(ttc, inf)
        self.assertTrue(2.4 <= ttc <= 3.1)

    def test_oncoming_closes_at_combined_speed(self):
        ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
        obj = Agent(WHEELED, 40.0, 0.0, heading=pi, speed=10.0, body_radius=1.0)
        ttc = time_to_collision(ego, obj, ROAD, PARAMS)
        self.assertNotEqual(ttc, inf)
        self.assertTrue(1.7 <= ttc <= 2.1)

    def test_oncoming_beyond_ego_reach_still_collides(self):
        ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
        obj = Agent(WHEELED, 50.0, 0.0, heading=pi, speed=10.0, body_radius=1.0)
        ttc = time_to_collision(ego, obj, ROAD, PARAMS)
        self.assertTrue(2.3 <= ttc <= 2.6)

    def test_crossing_vru_is_finite_within_horizon(self):
        ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
        ped = Agent(VRU, 18.0, 6.0, speed=4.0, body_radius=0.4)
        ttc = time_to_collision(ego, ped, ROAD, PARAMS)
        self.assertNotEqual(ttc, inf)
        self.assertLessEqual(ttc, PARAMS.horizon_s)

    def test_far_object_is_rejected_cheaply(self):
        ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
        obj = Agent(STATIC, 200.0, 0.0, footprint=_footprint(200.0, 0.0))
        self.assertEqual(time_to_collision(ego, obj, ROAD, PARAMS), inf)

    def test_wheeled_front_needs_drivable(self):
        ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0)
        obj = Agent(STATIC, 20.0, 0.0, footprint=_footprint(20.0, 0.0))
        with self.assertRaisesRegex(ValueError, "drivable"):
            time_to_collision(ego, obj, None, PARAMS)

    def test_disconnected_road_is_unreachable(self):
        split_road = box(-80.0, -10.0, 200.0, 10.0).union(box(-80.0, 20.0, 200.0, 40.0))
        ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
        oncoming_across = Agent(WHEELED, 5.0, 30.0, heading=-pi / 2, speed=10.0, body_radius=1.0)
        self.assertEqual(time_to_collision(ego, oncoming_across, split_road, PARAMS), inf)
        off_road = Agent(WHEELED, 0.0, 60.0, heading=0.0, speed=10.0, body_radius=1.0)
        self.assertEqual(time_to_collision(ego, off_road, ROAD, PARAMS), inf)
        region = reachable_region(ego, PARAMS, split_road)
        self.assertTrue(region.intersection(box(-80.0, 20.0, 200.0, 40.0)).is_empty)

    def test_params_validation(self):
        with self.assertRaisesRegex(ValueError, "dt_s must not exceed horizon_s"):
            ReachabilityParams(horizon_s=0.5, dt_s=0.6)
        with self.assertRaises(ValueError):
            ReachabilityParams(arc_samples=2)
        with self.assertRaises(ValueError):
            Agent("plane", 0.0, 0.0)
        with self.assertRaises(ValueError):
            Agent(STATIC, 0.0, 0.0)
        self.assertAlmostEqual(PARAMS.turn_radius(1.0), 3.0)
        self.assertAlmostEqual(PARAMS.turn_radius(6.0), 12.0)

    def test_steps_stay_within_horizon(self):
        ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=1.0, body_radius=0.5)
        self.assertEqual(EgoReachability(ego, ROAD, ReachabilityParams(horizon_s=1.0, dt_s=0.6)).steps, 1)
        self.assertEqual(EgoReachability(ego, ROAD, ReachabilityParams(horizon_s=3.0, dt_s=0.1)).steps, 30)
        vru = Agent(VRU, 7.6, 0.0, speed=5.0, body_radius=0.5)
        self.assertEqual(time_to_collision(ego, vru, ROAD, ReachabilityParams(horizon_s=1.0, dt_s=0.6)), inf)

    def test_collision_weights_monotone_and_bounds(self):
        weights = collision_weights([inf, 0.0, 1.0, 3.0], 0.1)
        self.assertEqual(weights[0], 0.0)
        self.assertTrue(isclose(weights[1], 1.0))
        self.assertTrue(weights[2] > weights[3] > 0.0)
        with self.assertRaises(ValueError):
            collision_weights([1.0], -0.1)

    def test_low_speed_region_stays_valid_past_pi_sweep(self):
        for speed in (0.83, 2.78, 3.0):
            agent = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=speed, body_radius=1.0)
            region = reachable_region(agent, PARAMS, ROAD)
            self.assertTrue(region.is_valid and not region.is_empty)
            self.assertTrue(region.contains(Point(min(speed * PARAMS.horizon_s * 0.9, 10.0), 0.0)))

    def test_ego_must_be_wheeled(self):
        with self.assertRaises(ValueError):
            EgoReachability(Agent(VRU, 0.0, 0.0, speed=1.0), ROAD, PARAMS)

    def test_ego_reachability_matches_bruteforce_stepping(self):
        def brute_force(ego: Agent, obj: Agent) -> float:
            steps = int(PARAMS.horizon_s / PARAMS.dt_s + 1e-9)
            for index in range(1, steps + 1):
                t = index * PARAMS.dt_s
                ego_set = reachable_set(ego, t, PARAMS, ROAD)
                obj_set = reachable_set(obj, t, PARAMS, ROAD)
                if not ego_set.is_empty and not obj_set.is_empty and ego_set.intersects(obj_set):
                    return t
            return inf

        rng = np.random.default_rng(7)
        ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=13.9, body_radius=1.0)
        frame = EgoReachability(ego, ROAD, PARAMS)
        for index in range(40):
            x, y = float(rng.uniform(-30, 130)), float(rng.uniform(-45, 45))
            kind = (WHEELED, VRU, STATIC)[index % 3]
            if kind == STATIC:
                obj = Agent(STATIC, x, y, footprint=_footprint(x, y))
            elif kind == VRU:
                obj = Agent(VRU, x, y, speed=float(rng.uniform(0.0, 6.0)), body_radius=0.4)
            else:
                obj = Agent(
                    WHEELED,
                    x,
                    y,
                    heading=float(rng.uniform(0, 2 * pi)),
                    speed=float(rng.uniform(0.5, 16.7)),
                    body_radius=1.0,
                )
            self.assertEqual(frame.time_to_collision(obj), brute_force(ego, obj), f"agent {index}: {obj}")


if __name__ == "__main__":
    unittest.main()
