"""Scores for one flight log (a DataFrame with flight.COLUMNS).

crashed / crash_t       below CRASH_MIN_Z after CRASH_GRACE_T, or tilted past CRASH_MAX_TILT_DEG; first time it happens
arrived                 not crashed and the final position is within ARRIVE_TOL of B
final_dist_to_B         distance to B at the end of the flight (reported for crashed flights too)
time_to_arrive          first time the drone is within ARRIVE_TOL of B and stays there to the end
mean/max_track_error    distance to the moving target point
mean/max_route_dev      shortest distance to the route itself (ignores lagging behind the target)
max_tilt_deg            largest tilt from vertical
max_alt_loss            largest drop below the target altitude after the fault (whole flight if healthy)
recovery_time           after the fault, time until the altitude is back within RECOVER_TOL of the target altitude
                        and stays there RECOVER_HOLD; 0 if it never left; NaN if it never came back or crashed
saturation_frac         share of steps after the fault (whole flight if healthy) with a motor command at the cap
cmd_smoothness          mean step-to-step change of the motor commands [N]
"""

import numpy as np

from . import config

NAN = float("nan")


def _first(t, mask):
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if len(idx) else NAN


def recovery_time(t, z, z_target, onset):
    t, inside = t[onset:], np.abs(z[onset:] - z_target[onset:]) <= config.RECOVER_TOL
    if inside.all():
        return 0.0
    start = None
    for k in range(int(np.argmin(inside)), len(t)):
        if not inside[k]:
            start = None
        elif start is None:
            start = k
        if start is not None and t[k] - t[start] >= config.RECOVER_HOLD - 1e-9:
            return float(t[start] - t[0])
    return NAN


def compute(log, route) -> dict:
    t = log["t"].to_numpy()
    pos = log[["pos_x", "pos_y", "pos_z"]].to_numpy()
    z, z_target = pos[:, 2], log["target_z"].to_numpy()
    tilt = log["tilt_deg"].to_numpy()
    dist = log["dist_to_B"].to_numpy()

    crash_mask = ((t >= config.CRASH_GRACE_T) & (z < config.CRASH_MIN_Z)) | (tilt > config.CRASH_MAX_TILT_DEG)
    crashed = bool(crash_mask.any())
    hits = np.flatnonzero(log["fault_active"].to_numpy())
    onset = int(hits[0]) if len(hits) else None
    after = slice(onset, None) if onset is not None else slice(None)

    near = dist <= config.ARRIVE_TOL
    if near[-1] and not crashed:
        leave = np.flatnonzero(~near)
        time_to_arrive = float(t[leave[-1] + 1]) if len(leave) else 0.0
    else:
        time_to_arrive = NAN

    commands = log[[f"command_{i}" for i in range(1, 5)]].to_numpy()
    route_dev = route.distance_to_route(pos[::5])  # every 5th step is plenty for these statistics
    rec = NAN if onset is None else (NAN if crashed else recovery_time(t, z, z_target, onset))
    return {
        "crashed": crashed,
        "crash_t": _first(t, crash_mask),
        "arrived": bool(near[-1] and not crashed),
        "final_dist_to_B": float(dist[-1]),
        "time_to_arrive": time_to_arrive,
        "mean_track_error": float(log["track_error"].mean()),
        "max_track_error": float(log["track_error"].max()),
        "mean_route_dev": float(route_dev.mean()),
        "max_route_dev": float(route_dev.max()),
        "max_tilt_deg": float(tilt.max()),
        "max_alt_loss": float(max(0.0, (z_target[after] - z[after]).max())),
        "min_z": float(z.min()),
        "recovery_time": rec,
        "saturation_frac": float((commands[after] >= config.MAX_THRUST - 1e-9).any(axis=1).mean()),
        "cmd_smoothness": float(np.abs(np.diff(commands, axis=0)).mean()),
        "flight_time": float(t[-1] + config.SIM_DT),
    }
