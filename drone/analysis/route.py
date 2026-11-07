"""Geometry of a run's route (any path, given as a polyline), plus its start/goal.

The route is the polyline traced by the target point: the CSV's setpoint_x/y/z columns
(route_from_df).
progress: arc length along the route of the route point closest to the drone.
deviation: 3-D distance from the drone to the route.
sideways: horizontal part of the offset from the drone to its closest route point, i.e.
    drift to the side. The two end segments are extended as lines, so overshooting past
    B (or behind A) along the route is not counted; height and overshoot are left out
    because the altitude plot and distance to B already show them. On a straight route
    this is the horizontal distance from the line through A and B.
"""

import json
import pathlib

import numpy as np

DEFAULT_START = (0.0, 0.0, 1.0)
DEFAULT_GOAL = (3.0, 0.0, 1.0)
SETPOINT = ["setpoint_x", "setpoint_y", "setpoint_z"]
# drone positions per block in the (positions x segments) distances; bounds memory
_CHUNK = 256


def route_from_df(df, start=None, goal=None):
    """(M, 3) route polyline: the setpoint positions with repeats removed.

    The target hovers at B for many rows.

    Optional start/goal are prepended/appended when they differ from the route's own
    ends.
    """
    pts = df[SETPOINT].to_numpy(dtype=float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
    pts = pts[keep]
    if (
        start is not None
        and np.linalg.norm(pts[0] - np.asarray(start, dtype=float)) > 1e-9
    ):
        pts = np.vstack([np.asarray(start, dtype=float), pts])
    if (
        goal is not None
        and np.linalg.norm(pts[-1] - np.asarray(goal, dtype=float)) > 1e-9
    ):
        pts = np.vstack([pts, np.asarray(goal, dtype=float)])
    return pts


def _closest_on_polyline(pos, route, extend_ends=False):
    """pos (N,3) -> (closest route point (N,3), its arc length along the route (N,)).

    extend_ends=True lets the first/last segment run on past A/B as lines (used for the
    sideways offset).
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    route = np.asarray(route, dtype=float).reshape(-1, 3)
    if len(route) == 1:  # degenerate route: a single point
        return np.repeat(route, len(pos), axis=0), np.zeros(len(pos))
    a, seg = route[:-1], np.diff(route, axis=0)  # (M-1, 3) segment starts and vectors
    seg_len2 = np.maximum((seg**2).sum(axis=1), 1e-18)
    cum = np.concatenate([[0.0], np.cumsum(np.sqrt((seg**2).sum(axis=1)))])
    lo = np.zeros(len(seg))
    hi = np.ones(len(seg))
    if extend_ends:
        lo[0], hi[-1] = -np.inf, np.inf
    closest = np.empty_like(pos)
    progress = np.empty(len(pos))
    for i in range(0, len(pos), _CHUNK):
        p = pos[i : i + _CHUNK, None, :]  # (n, 1, 3)
        # (n, M-1) position along each segment
        u = np.clip(((p - a) * seg).sum(axis=2) / seg_len2, lo, hi)
        cand = a + u[..., None] * seg  # (n, M-1, 3) closest point on each segment
        best = ((p - cand) ** 2).sum(axis=2).argmin(axis=1)
        rows = np.arange(len(best))
        closest[i : i + _CHUNK] = cand[rows, best]
        progress[i : i + _CHUNK] = cum[best] + u[rows, best] * np.sqrt(seg_len2[best])
    return closest, np.clip(progress, 0.0, cum[-1])


def progress_and_deviation_polyline(pos, route):
    """pos (N,3), route (M,3) -> (progress (N,), deviation (N,)) on the polyline."""
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    closest, progress = _closest_on_polyline(pos, route)
    return progress, np.linalg.norm(pos - closest, axis=1)


def sideways_offset_polyline(pos, route):
    """pos (N,3), route (M,3) -> (N,) horizontal distance to the closest route point."""
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    closest, _ = _closest_on_polyline(pos, route, extend_ends=True)
    return np.linalg.norm((pos - closest)[:, :2], axis=1)


def distance_to(pos, point):
    """Euclidean distance from each row of pos (N,3) to one point."""
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    return np.linalg.norm(pos - np.asarray(point, dtype=float), axis=1)


def endpoints_for(csv_path, start=None, goal=None):
    """Start/goal for a run.

    Explicit args win, else "start"/"goal" from <stem>.json next to the CSV, else
    defaults.
    """
    meta = {}
    json_path = pathlib.Path(csv_path).with_suffix(".json")
    if json_path.exists():
        meta = json.loads(json_path.read_text())
    start = start if start is not None else meta.get("start", DEFAULT_START)
    goal = goal if goal is not None else meta.get("goal", DEFAULT_GOAL)
    return [float(v) for v in start], [float(v) for v in goal]
