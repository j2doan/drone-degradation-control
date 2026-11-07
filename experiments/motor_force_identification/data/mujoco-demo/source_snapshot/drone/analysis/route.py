"""Geometry of the straight A->B route, plus loading of a run's start/goal.

progress  = how far along A->B the drone is (projection, clamped to [0, |AB|]).
deviation = distance from the drone to the A->B segment.
sideways  = horizontal distance from the A->B line, i.e. drift to the side (height and overshoot along the line
            are left out, because the altitude plot and distance to B already show those).
"""

import json
import pathlib

import numpy as np

DEFAULT_START = (0.0, 0.0, 1.0)
DEFAULT_GOAL = (3.0, 0.0, 1.0)


def progress_and_deviation(pos, start, goal):
    """pos (N,3) -> (progress (N,), deviation (N,)) relative to the segment start->goal."""
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    ab = goal - start
    length = float(np.linalg.norm(ab))
    if length == 0.0:  # degenerate route: no direction, deviation = distance to the point
        return np.zeros(len(pos)), distance_to(pos, start)
    progress = np.clip((pos - start) @ (ab / length), 0.0, length)
    closest = start + progress[:, None] * (ab / length)
    return progress, np.linalg.norm(pos - closest, axis=1)


def sideways_offset(pos, start, goal):
    """pos (N,3) -> (N,) horizontal distance from the (infinite) A->B line, ignoring height."""
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    rel = (pos - start)[:, :2]
    d = (goal - start)[:2]
    length = float(np.linalg.norm(d))
    if length == 0.0:  # vertical or degenerate route: no sideways direction, use horizontal distance from A
        return np.linalg.norm(rel, axis=1)
    d = d / length
    return np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])  # 2-D cross product = perpendicular distance


def distance_to(pos, point):
    """Euclidean distance from each row of pos (N,3) to one point."""
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    return np.linalg.norm(pos - np.asarray(point, dtype=float), axis=1)


def endpoints_for(csv_path, start=None, goal=None):
    """Start/goal for a run: explicit args win, else "start"/"goal" from <stem>.json next to the CSV, else defaults."""
    meta = {}
    json_path = pathlib.Path(csv_path).with_suffix(".json")
    if json_path.exists():
        meta = json.loads(json_path.read_text())
    start = start if start is not None else meta.get("start", DEFAULT_START)
    goal = goal if goal is not None else meta.get("goal", DEFAULT_GOAL)
    return [float(v) for v in start], [float(v) for v in goal]
