"""Scalar metrics for one flight CSV (one row per physics step, columns as in common.STATE_COLUMNS).

Works on old CSVs (no fault columns): the fault metrics are then NaN.

Run (from this experiment folder):
    python analysis/metrics.py data/<name>.csv [--start X Y Z] [--goal X Y Z]
Start/goal default to "start"/"goal" in data/<name>.json if it exists, else A=(0,0,1), B=(3,0,1).
"""

import pathlib
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # experiment folder

import numpy as np

from analysis.route import distance_to, endpoints_for, progress_and_deviation, sideways_offset

ARRIVE_TOL = 0.10  # m, "arrived" = within this distance of B
RECOVER_TOL = 0.05  # m, altitude band around the target altitude
RECOVER_HOLD = 0.5  # s, must stay in that band this long to count as recovered
# m, below this height (after CRASH_GRACE_T) = crash. The logged z is the drone body's reference point, which sits
# 0.25 m above the ground when the ARL drone rests on it (measured in Isaac), so 0.30 m means "touching the ground".
# Healthy flights dip to ~0.45 m during the start-up sag, so this does not flag them.
CRASH_MIN_Z = 0.30
CRASH_GRACE_T = 0.5  # s, ignore the height rule right after spawn
CRASH_MAX_TILT_DEG = 60.0  # body z more than this far from world z = crash

POS = ["pos_x", "pos_y", "pos_z"]
NAN = float("nan")


def tilt_deg(df):
    """Angle between body z and world z, from quaternion (x, y, z, w). Body z's world-z component = 1 - 2(x^2 + y^2)."""
    qx, qy, qz, qw = (df[f"quat_{a}"].to_numpy() for a in "xyzw")
    norm2 = qx**2 + qy**2 + qz**2 + qw**2  # tolerate slightly unnormalised quaternions
    cos_tilt = 1.0 - 2.0 * (qx**2 + qy**2) / norm2
    return np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))


def fault_onset_index(df):
    """Row index of the first step with fault_active == 1, or None (no fault / old CSV)."""
    if "fault_active" not in df:
        return None
    hits = np.flatnonzero(df["fault_active"].to_numpy() > 0.5)
    return int(hits[0]) if len(hits) else None


def _first_t(t, mask):
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if len(idx) else NAN


def recovery_time(t, z, z_target, onset):
    """Time from onset until |z - z_target| re-enters RECOVER_TOL and stays there >= RECOVER_HOLD.

    Measured against the target altitude, not the altitude at onset: the drone may be over- or undershooting when the
    fault hits, and "recovered" means back where it should be.
    0.0 if z never leaves the band; NaN if it leaves and never settles back (incl. a hold cut off by the end of data).
    """
    t, dz = t[onset:], np.abs(z[onset:] - z_target[onset:])
    inside = dz <= RECOVER_TOL
    if inside.all():
        return 0.0
    first_out = int(np.argmin(inside))
    run_start = None
    for k in range(first_out, len(t)):
        if not inside[k]:
            run_start = None
        elif run_start is None:
            run_start = k
        if run_start is not None and t[k] - t[run_start] >= RECOVER_HOLD:
            return float(t[run_start] - t[0])
    return NAN


def compute_metrics(df, start, goal) -> dict:
    """Metrics dict (plain floats/bools, NaN where undefined) for one run."""
    t = df["t"].to_numpy(dtype=float)
    pos = df[POS].to_numpy(dtype=float)
    z = pos[:, 2]
    dist_b = distance_to(pos, goal)
    _, dev = progress_and_deviation(pos, start, goal)

    thrust = df[[f"thrust_cmd_{i}" for i in range(4)]].to_numpy(dtype=float)
    smoothness = float(np.abs(np.diff(thrust, axis=0)).mean()) if len(df) > 1 else NAN

    crash_mask = ((z < CRASH_MIN_Z) & (t > CRASH_GRACE_T)) | (tilt_deg(df) > CRASH_MAX_TILT_DEG)
    crash_t = _first_t(t, crash_mask)
    crashed = bool(not np.isnan(crash_t))

    onset = fault_onset_index(df)
    if onset is None:
        onset_t = alt_loss = rec_t = NAN
        motor, loss = -1, 0.0
    else:
        onset_t = float(t[onset])
        # target altitude: the setpoint's height if logged, else the height at onset (old CSVs)
        z_target = df["setpoint_z"].to_numpy(dtype=float) if "setpoint_z" in df else np.full_like(z, z[onset])
        alt_loss = float(max(0.0, (z_target[onset:] - z[onset:]).max()))  # deepest drop below the target altitude
        rec_t = recovery_time(t, z, z_target, onset)
        motor = int(df["fault_motor"].iloc[onset]) if "fault_motor" in df else -1
        loss = float(df["fault_loss"].iloc[onset]) if "fault_loss" in df else NAN

    return {
        # where the drone ended up; NaN after a crash (the wreck's position is not a result: a crash is a fail)
        "final_dist_to_B": NAN if crashed else float(dist_b[-1]),
        "path_length": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
        "max_deviation": float(dev.max()),
        "mean_deviation": float(dev.mean()),
        "max_sideways_drift": float(sideways_offset(pos, start, goal).max()),
        "time_to_arrive": _first_t(t, dist_b <= ARRIVE_TOL),
        "smoothness": smoothness,
        "fault_motor": motor,
        "fault_loss": loss,
        "fault_onset_t": onset_t,
        "max_alt_loss_after_fault": alt_loss,
        "recovery_time": rec_t,
        "crashed": crashed,
        "crash_t": crash_t,
    }


def main():
    import argparse
    import json

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Flight CSV.")
    parser.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Point A [m].")
    parser.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Point B [m].")
    args = parser.parse_args()

    start, goal = endpoints_for(args.csv, args.start, args.goal)
    metrics = compute_metrics(pd.read_csv(args.csv), start, goal)
    print(json.dumps({"csv": str(args.csv), "start": start, "goal": goal, **metrics}, indent=2))


if __name__ == "__main__":
    main()
