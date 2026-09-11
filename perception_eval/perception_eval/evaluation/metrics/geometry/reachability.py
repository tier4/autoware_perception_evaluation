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
# Ported from tier4/autoware-ml (autoware_ml/metrics/geometry/reachability.py) at fcf86419.

"""Reachability time-to-collision for the criticality metrics (B1, B2).

One planner-independent model. Every agent, ego included, travels at the maximum map-legal speed
for its class. The recorded path and observed velocities are never used. Time-to-collision is the
earliest look-ahead time ``t`` at which ego and the object can occupy a common point at the same
time ``t``, each having travelled exactly ``speed * t`` along a feasible path (reachable at ``t``,
not "arrive early and wait"). That is what makes same-direction traffic at matched speed
non-critical: the two fronts stay a constant gap apart and never meet, so TTC = inf and its
collision weight is 0.

Reachable-at-``t`` set by class, in the map frame (metres):

* wheeled (car / truck / bus / motorbike, and ego): the endpoints of every feasible
  constant-curvature arc of length ``v * t`` under bounded steering (minimum turn radius), a curved
  front, clipped to the drivable area (an arc leaving the road is infeasible) and given the
  vehicle's body half-width.
* VRU (pedestrian / animal / bicycle): the disc of radius ``v * t`` about the current position,
  free to move in any direction, over any surface.
* static (hazard / unknown): the fixed footprint, for every ``t``.

The two steps match the metric's contract:

1. prescreen: distance bounds prove most objects can never meet ego within the horizon (their
   largest reachable set misses ego's reachable region, or the straight-line gap cannot close in
   time), giving TTC = inf with no stepping.
2. otherwise step ``t`` up from the earliest feasible step and return the first ``t`` at which the
   two reachable-at-``t`` sets overlap, the fastest path to collision.

:class:`EgoReachability` holds everything that depends only on ego (the localized drivable area,
the filled reachable region used by the prescreen, and the per-step fronts), so one frame's many
objects share it instead of rebuilding identical ego geometry per box.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from math import cos
from math import hypot
from math import inf
from math import sin
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from perception_eval.evaluation.metrics.geometry.shapely_compat import get_parts
from perception_eval.evaluation.metrics.geometry.shapely_compat import Prepared
from perception_eval.evaluation.metrics.geometry.shapely_compat import union_all
from shapely.geometry import box
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

WHEELED = "wheeled"
VRU = "vru"
STATIC = "static"
AGENT_KINDS = frozenset({WHEELED, VRU, STATIC})


@dataclass(frozen=True)
class ReachabilityParams:
    """Spec-versioned parameters shared by every reachability TTC evaluation.

    The wheeled turn radius is bounded by the lateral-acceleration (friction) limit,
    ``R = v^2 / max_lateral_accel``, floored at ``min_radius_m`` for low speed, so a fast vehicle
    cannot turn implausibly tight.
    """

    horizon_s: float = 4.0
    dt_s: float = 0.1
    max_lateral_accel_mps2: float = 3.0
    min_radius_m: float = 3.0
    arc_samples: int = 21

    def __post_init__(self) -> None:
        if self.horizon_s <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("horizon_s and dt_s must be > 0.")
        if self.dt_s > self.horizon_s:
            raise ValueError("dt_s must not exceed horizon_s.")
        if self.max_lateral_accel_mps2 <= 0.0 or self.min_radius_m <= 0.0:
            raise ValueError("max_lateral_accel_mps2 and min_radius_m must be > 0.")
        if self.arc_samples < 3:
            raise ValueError("arc_samples must be >= 3 to trace a front.")

    def turn_radius(self, speed: float) -> float:
        """Minimum feasible turn radius at ``speed`` (friction-limited, floored)."""
        return max(speed * speed / self.max_lateral_accel_mps2, self.min_radius_m)

    @property
    def steps(self) -> int:
        """Number of look-ahead steps inside the horizon (an exact multiple keeps its final step)."""
        return int(self.horizon_s / self.dt_s + 1e-9)


@dataclass(frozen=True)
class Agent:
    """One collision participant in the map frame.

    ``kind`` selects the reachable-set shape. ``heading`` and ``speed`` are used by the wheeled
    kind, the VRU kind uses ``speed`` isotropically, and the static kind ignores both and uses
    ``footprint``. ``body_radius`` is the half-extent added so a collision is a footprint overlap,
    not a point coincidence.
    """

    kind: str
    x: float
    y: float
    heading: float = 0.0
    speed: float = 0.0
    body_radius: float = 0.5
    footprint: Optional[Polygon] = None

    def __post_init__(self) -> None:
        if self.kind not in AGENT_KINDS:
            raise ValueError(f"kind must be one of {sorted(AGENT_KINDS)}, got {self.kind!r}.")
        if self.kind == STATIC and self.footprint is None:
            raise ValueError("a static agent needs a footprint polygon.")
        if self.kind != STATIC and self.speed < 0.0:
            raise ValueError("speed must be >= 0.")


def _arc_endpoint(x: float, y: float, heading: float, kappa: float, length: float) -> Tuple[float, float]:
    """Endpoint of a constant-curvature arc of signed curvature ``kappa`` and ``length``.

    ``kappa > 0`` turns left of the heading, ``kappa == 0`` is straight.
    """
    if abs(kappa) < 1e-9:
        return (x + length * cos(heading), y + length * sin(heading))
    radius = 1.0 / kappa
    # Turn centre is 90 deg to the left of the heading (left of travel for kappa>0).
    cx = x - radius * sin(heading)
    cy = y + radius * cos(heading)
    phi = length * kappa  # signed swept angle
    sx, sy = x - cx, y - cy
    ex = cx + sx * cos(phi) - sy * sin(phi)
    ey = cy + sx * sin(phi) + sy * cos(phi)
    return (ex, ey)


def _as_prepared(drivable: BaseGeometry) -> Prepared:
    return drivable if isinstance(drivable, Prepared) else Prepared(drivable)


def wheeled_front(agent: Agent, t: float, params: ReachabilityParams, drivable: BaseGeometry) -> BaseGeometry:
    """The wheeled reachable-at-``t`` front: buffered endpoint locus of the feasible arcs.

    Every constant-curvature arc is sampled as a polyline at the front's own resolution and kept
    only when the whole path stays on the drivable surface, so an endpoint across a gap in the road
    stays unreachable even when it lands on another drivable polygon. An agent whose reference
    point is off the surface has no feasible arc and an empty front. The endpoints of consecutive
    feasible arcs form the front, buffered by the body radius (capped by the travelled length so
    the sweep starts as a point at the agent's reference position) and clipped to the surface.
    """
    prepared = _as_prepared(drivable)
    length = agent.speed * t
    if length <= 1e-6:
        return Point(agent.x, agent.y).buffer(1e-3)
    kmax = 1.0 / params.turn_radius(agent.speed)
    lengths = np.linspace(0.0, length, params.arc_samples)
    runs: List[List[Tuple[float, float]]] = [[]]
    for kappa in np.linspace(-kmax, kmax, params.arc_samples):
        arc = [_arc_endpoint(agent.x, agent.y, agent.heading, float(kappa), float(s)) for s in lengths]
        if prepared.covers(LineString(arc)):
            runs[-1].append(arc[-1])
        elif runs[-1]:
            runs.append([])
    body = min(agent.body_radius, length)
    pieces = [(LineString(run) if len(run) > 1 else Point(run[0])).buffer(body) for run in runs if run]
    if not pieces:
        return Polygon()
    return union_all(pieces).intersection(prepared.geometry)


def reachable_region(agent: Agent, params: ReachabilityParams, drivable: Optional[BaseGeometry]) -> BaseGeometry:
    """Filled region a wheeled agent can reach within the horizon (the "hat").

    Union of its fronts over ``t`` in ``(0, T]``, whose boundary is the two extreme max-curvature
    arcs plus the far arc at ``t = T``, clipped to ``drivable``. The body buffer tapers at the start
    exactly like :func:`wheeled_front`. Used by the collision filter's in-the-ego-path membership.
    """
    length = agent.speed * params.horizon_s
    if length <= 1e-6:
        region: BaseGeometry = Point(agent.x, agent.y).buffer(1e-3)
    else:
        kmax = 1.0 / params.turn_radius(agent.speed)
        n = params.arc_samples
        x, y, heading = agent.x, agent.y, agent.heading
        left = [_arc_endpoint(x, y, heading, kmax, float(s)) for s in np.linspace(0.0, length, n)]
        far = [_arc_endpoint(x, y, heading, float(k), length) for k in np.linspace(kmax, -kmax, n)]
        right = [_arc_endpoint(x, y, heading, -kmax, float(s)) for s in np.linspace(length, 0.0, n)]
        shell = Polygon(left + far + right)
        if not shell.is_valid:
            # At low speed the max-curvature arcs sweep past pi and the shell ring self-touches.
            shell = shell.buffer(0)
        # Full body width everywhere except within one body radius of the seed point.
        widened = shell.buffer(agent.body_radius).difference(Point(agent.x, agent.y).buffer(agent.body_radius))
        region = shell.union(widened)
    if drivable is not None:
        surface = drivable.geometry if isinstance(drivable, Prepared) else drivable
        region = _seed_component(region.intersection(surface), Point(agent.x, agent.y))
    return region


def _seed_component(region: BaseGeometry, seed: Point) -> BaseGeometry:
    """The connected part of ``region`` closest to ``seed``.

    A clipped sweep can span several disconnected drivable polygons, and only the part
    road-connected to the agent is truly reachable.
    """
    parts = get_parts(region)
    if len(parts) <= 1:
        return region
    return min(parts, key=seed.distance)


def reachable_set(agent: Agent, t: float, params: ReachabilityParams, drivable: Optional[BaseGeometry]) -> BaseGeometry:
    """The agent's reachable-at-``t`` set in the map frame.

    Wheeled fronts keep only arcs that stay on ``drivable`` for their whole path, so a wheeled
    evaluation needs a drivable polygon. VRU discs and static footprints ignore ``drivable``.
    """
    if agent.kind == STATIC:
        return agent.footprint
    if agent.kind == VRU:
        return Point(agent.x, agent.y).buffer(agent.speed * t + agent.body_radius)
    if drivable is None:
        raise ValueError("a wheeled reachable set needs a drivable polygon.")
    return wheeled_front(agent, t, params, drivable)


class EgoReachability:
    """Ego reachable sets for one frame, shared across that frame's objects.

    Everything that depends only on ego is computed once: the drivable area localized to ego's
    horizon disc, the filled reachable region ("hat") the prescreen tests against, and, lazily one
    per step, the prepared reachable-at-``t`` fronts. :meth:`time_to_collision` then runs the
    two-step contract per object.

    The prescreen bounds are supersets of the true reachable sets (every front lies inside the hat,
    and inside the disc of radius ``speed * t + body``), so skipping a step or an object they
    exclude never misses a real overlap.
    """

    def __init__(self, ego: Agent, drivable: BaseGeometry, params: ReachabilityParams) -> None:
        if ego.kind != WHEELED:
            raise ValueError("ego must be a wheeled agent.")
        if drivable is None:
            raise ValueError("ego reachability needs a drivable polygon to clip against.")
        self.ego = ego
        self.params = params
        self.steps = params.steps
        self._surface = drivable.geometry if isinstance(drivable, Prepared) else drivable
        reach = ego.speed * params.horizon_s + ego.body_radius
        self._drivable = Prepared(
            self._surface.intersection(box(ego.x - reach, ego.y - reach, ego.x + reach, ego.y + reach))
        )
        self._hat = Prepared(reachable_region(ego, params, self._drivable))
        self._fronts: List[Optional[Prepared]] = [None] * (self.steps + 1)

    @property
    def hat(self) -> BaseGeometry:
        """Ego's filled reachable region within the horizon (map frame)."""
        return self._hat.geometry

    def _front(self, index: int) -> Prepared:
        """Ego's prepared reachable-at-``index * dt`` front (built once per step)."""
        front = self._fronts[index]
        if front is None:
            front = Prepared(wheeled_front(self.ego, index * self.params.dt_s, self.params, self._drivable))
            self._fronts[index] = front
        return front

    def _first_step(self, distance: float, speed: float, body: float) -> Optional[int]:
        """Earliest step at which a set growing ``speed * t + body`` spans ``distance``."""
        if distance <= body:
            return 1
        if speed <= 0.0:
            return None
        t_min = (distance - body) / speed
        if t_min > self.params.horizon_s:
            return None
        return max(1, ceil(t_min / self.params.dt_s - 1e-9))

    def time_to_collision(self, obj: Agent) -> float:
        """Earliest ``t`` at which ego and ``obj`` can occupy a common point at ``t``."""
        if self._hat.is_empty:
            return inf
        ego, params = self.ego, self.params

        if obj.kind == STATIC:
            # A static set never grows: it must already meet the hat.
            if not self._hat.intersects(obj.footprint):
                return inf
            distance = obj.footprint.distance(Point(ego.x, ego.y))
            start = self._first_step(distance, ego.speed, ego.body_radius)
            if start is None:
                return inf
            for index in range(start, self.steps + 1):
                if self._front(index).intersects(obj.footprint):
                    return index * params.dt_s
            return inf

        # Moving object: its set at t is within speed * t + body of its position, so it must be
        # able to reach the hat, and jointly close the gap to ego, in time.
        hat_distance = self._hat.distance(Point(obj.x, obj.y))
        start_hat = self._first_step(hat_distance, obj.speed, obj.body_radius)
        gap = hypot(obj.x - ego.x, obj.y - ego.y)
        start_gap = self._first_step(gap, ego.speed + obj.speed, ego.body_radius + obj.body_radius)
        if start_hat is None or start_gap is None:
            return inf
        # A wheeled object's whole path must stay on the surface, so its feasibility is checked
        # against the full drivable localized to the object's own reach, never against ego's clip.
        surface: Optional[Prepared] = None
        if obj.kind == WHEELED:
            reach = obj.speed * params.horizon_s + obj.body_radius
            surface = Prepared(
                self._surface.intersection(box(obj.x - reach, obj.y - reach, obj.x + reach, obj.y + reach))
            )
        for index in range(max(start_hat, start_gap), self.steps + 1):
            t = index * params.dt_s
            obj_set = reachable_set(obj, t, params, surface)
            if not obj_set.is_empty and self._front(index).intersects(obj_set):
                return t
        return inf


def time_to_collision(
    ego: Agent,
    obj: Agent,
    drivable: Optional[BaseGeometry],
    params: ReachabilityParams,
) -> float:
    """Earliest ``t`` at which ego and ``obj`` can occupy a common point at time ``t``.

    Returns ``inf`` when no such ``t`` exists within the horizon. ``ego`` must be a wheeled agent
    and needs ``drivable`` (fronts are clipped to it). One-pair form of :class:`EgoReachability`.
    """
    return EgoReachability(ego, drivable, params).time_to_collision(obj)


def collision_weights(ttc: NDArray, decay: float) -> NDArray[np.float64]:
    """Risk weight ``e^(-decay * TTC)`` per entry. An unreachable object (``inf``) weighs 0."""
    if decay < 0.0:
        raise ValueError("decay must be >= 0.")
    ttc = np.asarray(ttc, dtype=np.float64)
    weights = np.zeros_like(ttc)
    finite = np.isfinite(ttc)
    weights[finite] = np.exp(-decay * ttc[finite])
    return weights
