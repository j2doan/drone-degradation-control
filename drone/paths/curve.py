"""Seeded random smooth curved paths from A to B, and a target point moving along them.

The target point moves at constant speed.

A Path is a centripetal Catmull-Rom curve (alpha = 0.5: no cusps or self-loops inside a
segment) through its control points [A, waypoints..., B]. The curve's own parameter does
not move at constant speed, so a dense arc-length table (SAMPLES_PER_SEGMENT samples per
segment) maps distance travelled s -> curve parameter. setpoint(t, speed) gives the
target's position and velocity: it moves at `speed` along the curve, then stops at B
(the drone hovers there). sample_path() draws one deterministically from a seed, so
every controller flies the exact same route.

Simulator-independent: numpy only (no isaaclab / isaacsim / omni / warp, no scipy).

Usage (inside the flight loop):
    path = sample_path(seed)
    pos, vel = path.setpoint(t, speed)  # target position and velocity at time t
    duration = path.travel_time(speed) + hold
"""

import numpy as np

AREA_HALF = 3.0  # m; x and y stay in [-AREA_HALF, AREA_HALF] (a 6 x 6 m area)
# m; the whole curve stays in this band (faults cost ~0.75 m; crash zone < 0.30 m)
Z_RANGE = (1.0, 2.5)
LENGTH_RANGE = (3.0, 8.0)  # m; total curve length
WAYPOINT_RANGE = (1, 3)  # number of intermediate waypoints between A and B (inclusive)
# m; max sideways (x, y) push of a waypoint off the straight line A->B
WAYPOINT_SPREAD = 1.0
# m; max height push of a waypoint off that line (then clipped to Z_RANGE)
WAYPOINT_DZ = 0.5
MIN_AB_DIST = 2.0  # m; A and B at least this far apart
# m; reject hairpins (at 0.5 m/s that is at most 0.5 m/s^2 of sideways acceleration)
MIN_TURN_RADIUS = 0.5
SAMPLES_PER_SEGMENT = 200  # arc-length table resolution per curve segment
MAX_ATTEMPTS = 10000  # sample_path gives up after this many rejected draws
ALPHA = 0.5  # 0.5 = centripetal Catmull-Rom


class Path:
    """Smooth curve through control points (K, 3), K >= 2.

    Travelled by arc length s in [0, length].
    """

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(self.points) < 2:
            raise ValueError("A Path needs at least 2 control points (A and B).")
        p = self.points
        # end handling: reflect the neighbour through each endpoint, so the end segments
        # get a natural tangent
        self._ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
        self._nseg = len(p) - 1
        # arc-length table: curve parameter g (segment index + fraction) <-> distance
        # along the curve s
        self._g = np.linspace(0.0, self._nseg, self._nseg * SAMPLES_PER_SEGMENT + 1)
        self._dense = self._eval(self._g)
        self._s = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(self._dense, axis=0), axis=1))]
        )
        self.length = float(self._s[-1])

    @property
    def start(self):
        return self.points[0].copy()

    @property
    def goal(self):
        return self.points[-1].copy()

    def _eval(self, g):
        """Curve position at parameter(s) g in [0, nseg] -> (n, 3).

        Barry-Goldman form of Catmull-Rom.
        """
        g = np.clip(np.atleast_1d(np.asarray(g, dtype=float)), 0.0, self._nseg)
        i = np.minimum(np.floor(g).astype(int), self._nseg - 1)
        f = (g - i)[:, None]
        p0, p1, p2, p3 = (self._ext[i + k] for k in range(4))
        # knots spaced by distance**alpha; tiny floor so coincident control points do
        # not divide by zero
        d01, d12, d23 = (
            np.maximum(np.linalg.norm(b - a, axis=1, keepdims=True) ** ALPHA, 1e-9)
            for a, b in ((p0, p1), (p1, p2), (p2, p3))
        )
        t0, t1 = 0.0, d01
        t2 = t1 + d12
        t3 = t2 + d23
        u = t1 + f * d12
        a1 = ((t1 - u) * p0 + (u - t0) * p1) / (t1 - t0)
        a2 = ((t2 - u) * p1 + (u - t1) * p2) / (t2 - t1)
        a3 = ((t3 - u) * p2 + (u - t2) * p3) / (t3 - t2)
        b1 = ((t2 - u) * a1 + (u - t0) * a2) / (t2 - t0)
        b2 = ((t3 - u) * a2 + (u - t1) * a3) / (t3 - t1)
        return ((t2 - u) * b1 + (u - t1) * b2) / (t2 - t1)

    def _param(self, s):
        return np.interp(np.clip(s, 0.0, self.length), self._s, self._g)

    def point_at(self, s: float):
        """Position (3,) after s metres along the curve (s clamped to [0, length])."""
        return self._eval(self._param(s))[0]

    def tangent_at(self, s: float):
        """Unit direction of travel (3,) at arc length s.

        Central difference in the curve parameter.
        """
        g = float(self._param(s))
        eps = 1e-5
        d = np.diff(
            self._eval([max(g - eps, 0.0), min(g + eps, float(self._nseg))]), axis=0
        )[0]
        return d / np.linalg.norm(d)

    def sample(self, n: int):
        """(n, 3) points evenly spaced along the arc length, from A to B."""
        return self._eval(self._param(np.linspace(0.0, self.length, n)))

    def setpoint(self, t: float, speed: float):
        """(pos (3,), vel (3,)) of the target at time t.

        The target moves at `speed` along the curve from t = 0 and stops at B.
        """
        dist = speed * t
        vel = speed * self.tangent_at(dist) if dist < self.length else np.zeros(3)
        return self.point_at(dist), vel

    def travel_time(self, speed: float) -> float:
        """Seconds the target needs to go from A to B at `speed`."""
        return self.length / speed

    def max_curvature(self) -> float:
        """Largest curvature [1/m] along the arc-length table.

        Turn angle between chords divided by distance.
        """
        d = np.diff(self._dense, axis=0)
        ds = np.linalg.norm(d, axis=1)
        u = d / np.maximum(ds, 1e-12)[:, None]
        angle = np.arccos(np.clip((u[:-1] * u[1:]).sum(axis=1), -1.0, 1.0))
        return float((angle / np.maximum(0.5 * (ds[:-1] + ds[1:]), 1e-12)).max())

    def to_dict(self) -> dict:
        return {"points": self.points.tolist(), "length": self.length}

    @classmethod
    def from_dict(cls, d: dict) -> "Path":
        return cls(d["points"])


def _inside_box(points) -> bool:
    return bool(
        (np.abs(points[:, :2]) <= AREA_HALF).all()
        and (points[:, 2] >= Z_RANGE[0]).all()
        and (points[:, 2] <= Z_RANGE[1]).all()
    )


def sample_path(seed: int) -> Path:
    """Random smooth path, identical for identical seeds.

    The number of waypoints k is uniform in WAYPOINT_RANGE. A and B are uniform in the
    box (x, y in +-AREA_HALF, z in Z_RANGE); waypoint j sits at fraction j/(k+1) along
    A->B, pushed sideways by up to +-WAYPOINT_SPREAD in x and y and by up to
    +-WAYPOINT_DZ in height. (Waypoints uniform in the whole box almost never give a
    curve shorter than 8 m with 3 waypoints.) Draws are rejected until |AB| >=
    MIN_AB_DIST, the length is in LENGTH_RANGE, the whole curve stays in the box and no
    turn is tighter than MIN_TURN_RADIUS. Uses its own RNG stream ([seed, 7]), so it
    does not disturb sample_fault's default_rng(seed).
    """
    rng = np.random.default_rng([seed, 7])
    lo = np.array([-AREA_HALF, -AREA_HALF, Z_RANGE[0]])
    hi = np.array([AREA_HALF, AREA_HALF, Z_RANGE[1]])
    # number of waypoints drawn once, before the rejection loop: otherwise the length
    # limit would favour 1 waypoint
    k = int(rng.integers(WAYPOINT_RANGE[0], WAYPOINT_RANGE[1] + 1))
    frac = (np.arange(1, k + 1) / (k + 1))[:, None]
    for _ in range(MAX_ATTEMPTS):
        a, b = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if np.linalg.norm(b - a) < MIN_AB_DIST:
            continue
        way = a + frac * (b - a)
        way[:, :2] += rng.uniform(-WAYPOINT_SPREAD, WAYPOINT_SPREAD, size=(k, 2))
        way[:, 2] = np.clip(
            way[:, 2] + rng.uniform(-WAYPOINT_DZ, WAYPOINT_DZ, size=k), *Z_RANGE
        )
        path = Path(np.vstack([a, way, b]))
        if not LENGTH_RANGE[0] <= path.length <= LENGTH_RANGE[1] or not _inside_box(
            path._dense
        ):
            continue
        if path.max_curvature() > 1.0 / MIN_TURN_RADIUS:
            continue
        return path
    raise RuntimeError(
        f"sample_path({seed}): no valid path in {MAX_ATTEMPTS} attempts; "
        "loosen the constants "
        f"(LENGTH_RANGE={LENGTH_RANGE}, AREA_HALF={AREA_HALF}, Z_RANGE={Z_RANGE})."
    )
