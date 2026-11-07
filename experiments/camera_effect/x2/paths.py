"""Routes from A to B and the moving target point the drone follows.

A Route is a smooth curve through control points [A, waypoints..., B] (centripetal Catmull-Rom: no loops or cusps
inside a segment). With only A and B it is a straight line. The curve's own parameter does not advance at constant
speed, so an arc-length table maps "metres travelled" to the curve. The target point moves along the route at a
constant speed and stops at B; the drone's controller chases it.
"""

import numpy as np

from . import config

SAMPLES_PER_SEGMENT = 200


class Route:
    def __init__(self, points, name: str = "route"):
        self.name = name
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(self.points) < 2:
            raise ValueError("a route needs at least A and B")
        p = self.points
        # Extra end points mirrored through A and B give the end segments a natural direction.
        self._ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
        self._nseg = len(p) - 1
        self._u = np.linspace(0.0, self._nseg, self._nseg * SAMPLES_PER_SEGMENT + 1)
        self.dense = self._curve(self._u)  # fine polyline along the route
        self._s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.dense, axis=0), axis=1))])
        self.length = float(self._s[-1])

    @property
    def start(self):
        return self.points[0].copy()

    @property
    def goal(self):
        return self.points[-1].copy()

    def _curve(self, u):
        """Points at curve parameter u in [0, nseg] (segment index + fraction), centripetal Catmull-Rom."""
        u = np.clip(np.atleast_1d(np.asarray(u, dtype=float)), 0.0, self._nseg)
        i = np.minimum(np.floor(u).astype(int), self._nseg - 1)
        frac = (u - i)[:, None]
        p0, p1, p2, p3 = (self._ext[i + k] for k in range(4))
        # knot spacing = sqrt(distance) (centripetal); a tiny floor avoids division by zero
        k01, k12, k23 = (np.maximum(np.sqrt(np.linalg.norm(b - a, axis=1, keepdims=True)), 1e-9)
                         for a, b in ((p0, p1), (p1, p2), (p2, p3)))
        t0, t1 = 0.0, k01
        t2, t3 = t1 + k12, t1 + k12 + k23
        t = t1 + frac * k12
        a1 = ((t1 - t) * p0 + (t - t0) * p1) / (t1 - t0)
        a2 = ((t2 - t) * p1 + (t - t1) * p2) / (t2 - t1)
        a3 = ((t3 - t) * p2 + (t - t2) * p3) / (t3 - t2)
        b1 = ((t2 - t) * a1 + (t - t0) * a2) / (t2 - t0)
        b2 = ((t3 - t) * a2 + (t - t1) * a3) / (t3 - t1)
        return ((t2 - t) * b1 + (t - t1) * b2) / (t2 - t1)

    def point_at(self, s):
        """Position after travelling s metres along the route (clamped to [0, length])."""
        return self._curve(np.interp(np.clip(s, 0.0, self.length), self._s, self._u))[0]

    def direction_at(self, s):
        """Unit direction of travel at s metres."""
        a = self.point_at(max(s - 1e-3, 0.0))
        b = self.point_at(min(s + 1e-3, self.length))
        return (b - a) / np.linalg.norm(b - a)

    def target(self, t: float, speed: float = config.SPEED):
        """(position, velocity) of the target point at time t; it stops at B."""
        s = speed * t
        vel = speed * self.direction_at(s) if s < self.length else np.zeros(3)
        return self.point_at(s), vel

    def travel_time(self, speed: float = config.SPEED) -> float:
        return self.length / speed

    def distance_to_route(self, positions):
        """Shortest distance from each position (N, 3) to the route polyline."""
        a, b = self.dense[:-1], self.dense[1:]
        ab = b - a
        denom = np.maximum((ab * ab).sum(1), 1e-12)
        out = np.empty(len(positions))
        for k, p in enumerate(np.asarray(positions)):
            w = np.clip(((p - a) * ab).sum(1) / denom, 0.0, 1.0)
            out[k] = np.sqrt((((a + w[:, None] * ab) - p) ** 2).sum(1).min())
        return out

    def max_curvature(self) -> float:
        d = np.diff(self.dense, axis=0)
        ds = np.linalg.norm(d, axis=1)
        unit = d / np.maximum(ds, 1e-12)[:, None]
        angle = np.arccos(np.clip((unit[:-1] * unit[1:]).sum(1), -1.0, 1.0))
        return float((angle / np.maximum(0.5 * (ds[:-1] + ds[1:]), 1e-12)).max())

    def to_dict(self) -> dict:
        return {"name": self.name, "points": self.points.round(6).tolist(), "length_m": round(self.length, 6)}


def straight_route() -> Route:
    return Route([config.STRAIGHT_A, config.STRAIGHT_B], name="straight")


def random_route(seed: int, max_attempts: int = 10000) -> Route:
    """A random smooth route, the same for the same seed.

    A and B are uniform in the flight box; 1-3 waypoints are spaced evenly along A-B and pushed sideways (up to
    WAYPOINT_SPREAD) and up/down (up to WAYPOINT_DZ). Draws are rejected until A and B are far enough apart, the
    length is in LENGTH_RANGE, the whole curve stays in the box and no turn is tighter than MIN_TURN_RADIUS.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([-config.AREA_HALF, -config.AREA_HALF, config.Z_RANGE[0]])
    hi = np.array([config.AREA_HALF, config.AREA_HALF, config.Z_RANGE[1]])
    k = int(rng.integers(config.WAYPOINTS[0], config.WAYPOINTS[1] + 1))
    frac = (np.arange(1, k + 1) / (k + 1))[:, None]
    for _ in range(max_attempts):
        a, b = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if np.linalg.norm(b - a) < config.MIN_AB_DIST:
            continue
        way = a + frac * (b - a)
        way[:, :2] += rng.uniform(-config.WAYPOINT_SPREAD, config.WAYPOINT_SPREAD, size=(k, 2))
        way[:, 2] = np.clip(way[:, 2] + rng.uniform(-config.WAYPOINT_DZ, config.WAYPOINT_DZ, size=k), *config.Z_RANGE)
        route = Route(np.vstack([a, way, b]), name=f"path{seed}")
        inside = (np.abs(route.dense[:, :2]) <= config.AREA_HALF).all() and \
            (route.dense[:, 2] >= config.Z_RANGE[0]).all() and (route.dense[:, 2] <= config.Z_RANGE[1]).all()
        if (config.LENGTH_RANGE[0] <= route.length <= config.LENGTH_RANGE[1] and inside
                and route.max_curvature() <= 1.0 / config.MIN_TURN_RADIUS):
            return route
    raise RuntimeError(f"no valid route for seed {seed} in {max_attempts} attempts")
