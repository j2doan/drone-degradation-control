"""Self-test for paths/curve.py and the polyline route metrics.

Run: python paths/test_curve.py (or with pytest)
"""

import pathlib
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # drone/

import numpy as np

from paths.curve import (
    AREA_HALF,
    LENGTH_RANGE,
    MIN_AB_DIST,
    WAYPOINT_RANGE,
    Z_RANGE,
    Path,
    sample_path,
)

DT = 0.01  # s, physics step
SPEED = 0.5  # m/s
SEEDS = range(300)


def test_deterministic():
    assert np.array_equal(sample_path(7).points, sample_path(7).points)


def test_different_seeds_differ():
    starts = {tuple(np.round(sample_path(s).start, 9)) for s in range(50)}
    assert len(starts) == 50


def test_ranges():
    counts = set()
    for s in SEEDS:
        p = sample_path(s)
        dense = p.sample(2000)
        assert LENGTH_RANGE[0] <= p.length <= LENGTH_RANGE[1], (s, p.length)
        assert (np.abs(dense[:, :2]) <= AREA_HALF + 1e-9).all(), s
        assert (
            dense[:, 2].min() >= Z_RANGE[0] - 1e-9
            and dense[:, 2].max() <= Z_RANGE[1] + 1e-9
        ), s
        assert np.linalg.norm(p.goal - p.start) >= MIN_AB_DIST, s
        counts.add(len(p.points) - 2)
    # every waypoint count occurs
    assert counts == set(range(WAYPOINT_RANGE[0], WAYPOINT_RANGE[1] + 1))


def test_constant_speed_and_stop():
    for s in range(20):
        p = sample_path(s)
        T = p.travel_time(SPEED)
        t = np.arange(0.0, T + 2.0, DT)
        pos = np.array([p.setpoint(ti, SPEED)[0] for ti in t])
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        moving = t[1:] <= T - DT
        assert np.allclose(step[moving], SPEED * DT, rtol=0.02), (
            s,
            step[moving].min(),
            step[moving].max(),
        )
        after = t >= T
        assert np.allclose(pos[after], p.goal, atol=1e-9)
        assert all(
            np.array_equal(p.setpoint(ti, SPEED)[1], np.zeros(3)) for ti in t[after]
        )
        # while moving, the velocity has the right size and points along the path
        for ti in t[t < T - DT][::50]:
            pos0, vel = p.setpoint(ti, SPEED)
            assert abs(np.linalg.norm(vel) - SPEED) < 1e-9
            assert np.allclose(
                pos0 + vel * DT, p.setpoint(ti + DT, SPEED)[0], atol=1e-4
            )


def test_passes_through_control_points():
    for s in range(30):
        p = sample_path(s)
        dense = p.sample(20000)
        for c in p.points:
            assert np.linalg.norm(dense - c, axis=1).min() < 1e-3, (s, c)
        assert np.allclose(p.point_at(0.0), p.start) and np.allclose(
            p.point_at(p.length), p.goal
        )


def test_smooth():
    for s in range(50):
        p = sample_path(s)
        n = int(p.length / 0.01)  # 1 cm apart
        tang = np.array([p.tangent_at(si) for si in np.linspace(0.0, p.length, n)])
        turn = np.degrees(
            np.arccos(np.clip((tang[:-1] * tang[1:]).sum(axis=1), -1.0, 1.0))
        )
        assert turn.max() < 3.0, (s, turn.max())  # 3 deg per cm = turn radius >= ~0.2 m


def test_to_dict_roundtrip():
    import json

    p = sample_path(3)
    d = json.loads(json.dumps(p.to_dict()))
    q = Path.from_dict(d)
    assert np.allclose(q.points, p.points) and abs(q.length - d["length"]) < 1e-12


def test_polyline_metrics_on_known_route():
    from analysis.route import progress_and_deviation_polyline, sideways_offset_polyline

    route = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 2.0, 1.0]])  # an L
    pos = np.array(
        [[1.0, 0.5, 1.0], [2.5, 1.0, 1.2], [2.0, 3.0, 1.0], [-1.0, 0.0, 1.0]]
    )
    prog, dev = progress_and_deviation_polyline(pos, route)
    assert np.allclose(prog, [1.0, 3.0, 4.0, 0.0])
    assert np.allclose(dev, [0.5, np.hypot(0.5, 0.2), 1.0, 1.0])
    # sideways: height ignored; overshoot past B / behind A along the route ignored
    assert np.allclose(sideways_offset_polyline(pos, route), [0.5, 0.5, 0.0, 0.0])


def _straight_line_reference(pos, start, goal):
    """Straight-line definitions, as a cross-check.

    Deviation from segment A-B, sideways from the infinite line A-B.
    """
    ab = goal - start
    L = np.linalg.norm(ab)
    prog = np.clip((pos - start) @ (ab / L), 0.0, L)
    dev = np.linalg.norm(pos - (start + prog[:, None] * ab / L), axis=1)
    rel, d = (pos - start)[:, :2], ab[:2] / np.linalg.norm(ab[:2])
    return dev, np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])


def _straight_flight():
    """A=(0,0,1) -> B=(3,0,1) at 0.5 m/s, then a hover; the drone wobbles a few cm."""
    import pandas as pd

    t = np.arange(0.0, 9.0, DT)
    x = np.minimum(0.5 * t, 3.0)
    sp = np.stack([x, np.zeros_like(t), np.ones_like(t)], axis=1)
    wobble = np.stack([0.0 * t, 0.05 * np.sin(2.0 * t), 0.03 * np.cos(3.0 * t)], axis=1)
    pos = sp + wobble
    cols = {"step": np.arange(len(t)), "t": t}
    cols |= {f"pos_{a}": pos[:, k] for k, a in enumerate("xyz")}
    cols |= {f"quat_{a}": np.full_like(t, 1.0 if a == "w" else 0.0) for a in "xyzw"}
    cols |= {f"setpoint_{a}": sp[:, k] for k, a in enumerate("xyz")}
    cols |= {f"thrust_cmd_{i}": np.full_like(t, 3.04) for i in range(4)}
    return pd.DataFrame(cols)


def test_metrics_match_straight_line_definitions():
    import time

    from analysis.metrics import POS, compute_metrics
    from analysis.route import route_from_df

    df = _straight_flight()
    pos = df[POS].to_numpy(dtype=float)
    route = route_from_df(df)
    start, goal = route[0], route[-1]
    assert np.allclose(start, [0, 0, 1]) and np.allclose(goal, [3, 0, 1])
    dev_old, side_old = _straight_line_reference(pos, start, goal)
    t0 = time.perf_counter()
    m = compute_metrics(df)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, elapsed
    assert abs(m["max_deviation"] - dev_old.max()) < 1e-3, (
        m["max_deviation"],
        dev_old.max(),
    )
    assert abs(m["mean_deviation"] - dev_old.mean()) < 1e-3, (
        m["mean_deviation"],
        dev_old.mean(),
    )
    assert abs(m["max_sideways_drift"] - side_old.max()) < 1e-3, (
        m["max_sideways_drift"],
        side_old.max(),
    )
    # explicit endpoints equal to the route's: nothing changes
    explicit = compute_metrics(df, [0, 0, 1], [3, 0, 1])
    assert all(np.isclose(m[k], explicit[k], equal_nan=True) for k in m), explicit
    print(
        f"  straight flight: {len(df)} rows, {len(route)} route points, "
        f"metrics in {elapsed * 1000:.0f} ms; "
        f"max_dev {m['max_deviation']:.4f} (straight {dev_old.max():.4f}), "
        f"mean_dev {m['mean_deviation']:.4f} (straight {dev_old.mean():.4f}), "
        f"sideways {m['max_sideways_drift']:.4f} (straight {side_old.max():.4f})"
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"[OK] {name}")
    for s in range(5):
        p = sample_path(s)
        print(
            f"  seed {s}: {len(p.points) - 2} waypoints, length {p.length:.2f} m, "
            f"travel {p.travel_time(SPEED):.1f} s at {SPEED} m/s, "
            f"A {np.round(p.start, 2)}, B {np.round(p.goal, 2)}"
        )
    print("[OK] all curve tests passed")
