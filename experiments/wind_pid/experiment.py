"""PD vs PID in a steady wind, and a sweep of the integral settings (resumable).

    python experiment.py            # fly everything not yet done, then write data/summary.csv
    python experiment.py --status   # how many flights are done
    python experiment.py --force    # fly everything again

Flights (all the same 1.5 m square mission, winds 0 / 2 / 5 N along +x):
    PD                                   3 flights
    sweep: ki x integration zone x limit 6 x 6 x 5 = 180 settings x 3 winds = 540 flights
           (zone "inf" = integral always on; limit "inf" = no limit)
The always-on PID (ki 2, always on, no limit) and the near-target PID (ki 2, within 0.5 m, no limit) are
points of the sweep.

Self-contained: everything is in this folder (sim.py, the Skydio X2 model in assets/). Each flight is written to
data/flights/<id>.tmp and renamed when complete, so a run cut off part-way (laptop closed) continues where it stopped;
a flight is flown again if its settings or the code changed.
"""

import argparse
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sim  # noqa: E402

DATA = HERE / "data"
FLIGHTS = DATA / "flights"
WINDS = (0.0, 2.0, 5.0)  # N
KI_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)  # 1/s^3
ZONE_VALUES = (0.1, 0.25, 0.5, 0.75, 1.0, float("inf"))  # m (inf = always integrate)
LIMIT_VALUES = (1.0, 2.0, 4.0, 8.0, float("inf"))  # m/s^2
LOG_EVERY = 5  # store the trajectory at 20 Hz (the simulation runs at 100 Hz)

PD = sim.Gains()
PID = sim.Gains(ki=2.0)  # integral always on
PID_NEAR = sim.Gains(ki=2.0, i_rule="near", i_zone=0.5)  # integral only within 0.5 m of the waypoint


def _num(v):
    return "inf" if not np.isfinite(v) else f"{v:g}"


def make_gains(ki, zone, limit):
    return sim.Gains(ki=ki, i_rule="always" if not np.isfinite(zone) else "near", i_zone=zone, i_limit=limit)


def flight_id(g: sim.Gains, wind):
    name = "pd" if g.ki == 0 else f"ki{_num(g.ki)}_zone{_num(g.i_zone)}_lim{_num(g.i_limit)}"
    return f"{name}_wind{wind:g}"


def all_specs():
    specs = [(PD, w) for w in WINDS]
    for ki in KI_VALUES:
        for zone in ZONE_VALUES:
            for limit in LIMIT_VALUES:
                specs += [(make_gains(ki, zone, limit), w) for w in WINDS]
    return specs


def code_key():
    """Fingerprint of the code and model: a flight is flown again if any of it changes."""
    h = hashlib.sha256()
    for p in [HERE / "sim.py", HERE / "experiment.py"] + sorted(sim.find_scene().parent.rglob("*.xml")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _done(fid, key):
    f = FLIGHTS / fid / "flight.json"
    if not f.exists():
        return False
    return json.loads(f.read_text()).get("code_key") == key


def fly_one(g, wind, model, key):
    fid = flight_id(g, wind)
    tmp, final = FLIGHTS / f"{fid}.tmp", FLIGHTS / fid
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    log, res = sim.fly(g, wind, model, log_every=LOG_EVERY)
    pd.DataFrame(log).to_csv(tmp / "flight.csv", index=False, float_format="%.6g")
    info = {"id": fid, "controller": g.label(), "gains": sim.gains_dict(g), "wind_N": wind, "code_key": key, **res,
            "mean_error_m": None if res["crashed"] else float(np.mean(res["end_errors_m"]))}  # crashed = failed
    (tmp / "flight.json").write_text(json.dumps(info, indent=2))
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)


def summarise():
    rows = []
    for f in sorted(FLIGHTS.glob("*/flight.json")):
        if f.parent.name.endswith(".tmp"):
            continue
        j = json.loads(f.read_text())
        g = j["gains"]
        errs = j["end_errors_m"] + [np.nan] * (len(sim.WAYPOINTS) - 1 - len(j["end_errors_m"]))
        rows.append({"id": j["id"], "controller": j["controller"], "ki": g["ki"], "i_rule": g["i_rule"],
                     "i_zone": g["i_zone"] if g["i_zone"] is not None else np.inf,
                     "i_limit": g["i_limit"] if g["i_limit"] is not None else np.inf,
                     "wind_N": j["wind_N"], **{f"err_wp{i + 1}_m": e for i, e in enumerate(errs)},
                     "mean_error_m": np.inf if j["crashed"] else j["mean_error_m"], "crashed": j["crashed"]})
    df = pd.DataFrame(rows).sort_values(["ki", "i_zone", "i_limit", "wind_N"]).reset_index(drop=True)
    df.to_csv(DATA / "summary.csv", index=False, float_format="%.6g")
    return df


def write_manifest(session):
    path = DATA / "manifest.json"
    m = json.loads(path.read_text()) if path.exists() else {"experiment": "wind_pid", "sessions": []}
    try:  # git revision if this folder is inside a git repo
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
                             timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        rev = None
    m.update({"python": sys.version.split()[0], "platform": platform.platform(),
              "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__, "pandas": pd.__version__},
              "git_revision": rev, "code_key": code_key(), "model": str(sim.find_scene().parent.name),
              "winds_N": WINDS, "ki": KI_VALUES, "zones_m": [_num(z) for z in ZONE_VALUES],
              "limits_mps2": [_num(v) for v in LIMIT_VALUES], "hold_s": sim.HOLD_T,
              "waypoints": sim.WAYPOINTS.tolist(), "pd_gains": sim.gains_dict(PD)})
    m["sessions"].append(session)
    path.write_text(json.dumps(m, indent=2))


def status():
    key = code_key()
    specs = all_specs()
    done = sum(_done(flight_id(g, w), key) for g, w in specs)
    return done, len(specs)


def main(force=False, say=print):
    FLIGHTS.mkdir(parents=True, exist_ok=True)
    key, model = code_key(), sim.load_model()
    start, t0, flown = dt.datetime.now().isoformat(timespec="seconds"), time.time(), 0
    specs = all_specs()
    for i, (g, w) in enumerate(specs):
        if not force and _done(flight_id(g, w), key):
            continue
        fly_one(g, w, model, key)
        flown += 1
        if flown % 50 == 0:
            say(f"  {i + 1}/{len(specs)} ({time.time() - t0:.0f} s)")
    df = summarise()
    write_manifest({"started": start, "finished": dt.datetime.now().isoformat(timespec="seconds"),
                    "flown": flown, "skipped": len(specs) - flown})
    say(f"done: {flown} flown, {len(specs) - flown} already done -> {DATA}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.status:
        d, n = status()
        print(f"{d}/{n} flights done")
    else:
        main(force=a.force, say=lambda s: print(s, flush=True))
