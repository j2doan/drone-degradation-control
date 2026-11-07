"""Camera on/off experiment: does a downward camera improve the X2's recovery from a motor fault?

Three sensing conditions fly the same 4 routes x 5 loss levels on motor 1:
    perfect     the PID reads the true position and velocity (best possible case)
    no_camera   the PID reads a Kalman estimate from IMU + GPS (5 Hz) + barometer (20 Hz)
    camera      the same, plus the downward camera (20 Hz height and ground velocity)
no_camera and camera are flown with 3 noise seeds each, and for a given seed both see exactly the same IMU, GPS and
barometer noise, so any difference between them comes from the camera alone. 20 + 60 + 60 = 140 flights.

Self-contained: the simulation code is in x2/ and the Skydio X2 model in assets/,
so this folder can be moved anywhere. Resumable: each flight is written to "<id>.tmp/", finished with done.json, then renamed. Run again
after an interruption and only missing flights are flown.

    python experiment.py            # run (or resume) everything; works from any working directory
    python experiment.py --status   # progress
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
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent  # this folder; it is self-contained (x2/ and assets/ live here)
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sensing  # noqa: E402
from x2 import config, metrics  # noqa: E402
from x2.controller import PID, Gains  # noqa: E402
from x2.model import drone_facts, find_scene, hover_feasible, model_file_hashes  # noqa: E402
from x2.motors import Fault, Motors  # noqa: E402
from x2.paths import random_route  # noqa: E402
from x2.sensors import ImuNoise  # noqa: E402

DATA = HERE / "data"
CONDITIONS = ("perfect", "no_camera", "camera")
NOISE_SEEDS = (0, 1, 2)
SAVE_FRAMES_SEED = 0  # camera frames are saved for this seed only (all seeds log the camera's readings)


def flight_list():
    """(flight_id, condition, seed, route seed, loss) for every flight, in run order."""
    out = []
    for cond in CONDITIONS:
        for seed in (NOISE_SEEDS if cond != "perfect" else (0,)):
            for p in config.RANDOM_PATH_SEEDS:
                for loss in config.LOSS_LEVELS:
                    out.append((f"{cond}_s{seed}_path{p}_loss{loss * 100:04.1f}", cond, seed, p, loss))
    return out


def fly(condition, seed, route_seed, loss, model, save_frames=False):
    """One flight. Returns (log DataFrame, frames list, info dict). Scoring uses the true state."""
    route = random_route(route_seed)
    fault = None if loss <= 0 else Fault(config.FAULT_MOTOR, loss, round(config.FAULT_AT_FRACTION * route.travel_time(), 6))
    duration = route.travel_time() + config.HOLD_T
    if fault is not None:
        duration = max(duration, fault.onset_t + config.POST_FAULT_T)
    total = int(round(duration / config.SIM_DT))

    data = mujoco.MjData(model)
    facts = drone_facts(model)
    data.qpos[:3], data.qpos[3:7] = route.start, [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    pid = PID(facts.mass, facts.inertia, facts.allocation)
    motors = Motors(facts.hover_thrust, fault)

    # Noise: identical streams for no_camera and camera at the same (route, loss, seed)
    key = zlib.crc32(f"path{route_seed}_loss{loss}_s{seed}".encode())
    imu = ImuNoise(key)
    rng = np.random.default_rng([key, 7])
    est = sensing.Estimator(route.start)
    cam = sensing.DownCamera(model) if condition == "camera" else None
    acc_adr = model.sensor("body_linacc").adr[0]
    gps_every = int(round(sensing.GPS_PERIOD / config.SIM_DT))
    baro_every = int(round(sensing.BARO_PERIOD / config.SIM_DT))
    cam_every = int(round(sensing.CAM_PERIOD / config.SIM_DT))

    rows, frames = [], []
    rot9 = np.zeros(9)
    accel_world = None
    goal = route.goal
    try:
        for k in range(total):
            t = k * config.SIM_DT
            pos, vel = data.qpos[:3].copy(), data.qvel[:3].copy()
            mujoco.mju_quat2Mat(rot9, data.qpos[3:7])
            rot = rot9.reshape(3, 3).copy()
            omega = data.qvel[3:6].copy()

            # --- estimation (always run, so the noise streams are the same in every condition)
            if accel_world is not None:
                est.predict(accel_world)
            gps = np.full(3, np.nan)
            baro = np.nan
            if k % gps_every == 0:
                gps = pos + rng.normal(0.0, sensing.GPS_SIGMA)
                for i in range(3):
                    est.update(i, gps[i], sensing.GPS_SIGMA[i], "pos")
            if k % baro_every == 0:
                baro = pos[2] + rng.normal(0.0, sensing.BARO_SIGMA)
                est.update(2, baro, sensing.BARO_SIGMA, "pos")
            cam_h, cam_v, frame_idx = np.nan, np.full(2, np.nan), -1
            if cam is not None and k % cam_every == 0:
                cam_h, cam_v, gray, _ = cam.measure(data, est.vel[:2])
                if np.isfinite(cam_h):
                    est.update(2, cam_h, 0.01 + 0.02 * cam_h, "pos")
                sigma_v = 0.03 + 0.04 * (cam_h if np.isfinite(cam_h) else 2.0)
                est.update(0, cam_v[0], sigma_v, "vel")
                est.update(1, cam_v[1], sigma_v, "vel")
                if save_frames:
                    frames.append(np.clip(gray, 0, 255).astype(np.uint8))
                    frame_idx = len(frames) - 1

            # --- control
            target_pos, target_vel = route.target(t)
            if condition == "perfect":
                ctrl_pos, ctrl_vel = pos, vel
            else:
                ctrl_pos, ctrl_vel = est.pos, est.vel
            request = pid.act(ctrl_pos, ctrl_vel, rot, omega, target_pos, target_vel)
            command, thrust, applied, active = motors.step(request, t)
            data.ctrl[:] = applied
            mujoco.mj_step(model, data)
            acc_noisy, _ = imu.apply(data.sensordata[acc_adr:acc_adr + 3].copy(), np.zeros(3))
            accel_world = rot @ acc_noisy - np.array([0.0, 0.0, config.GRAVITY])

            tilt = np.degrees(np.arccos(np.clip(rot[2, 2], -1.0, 1.0)))
            rows.append([k, t, *target_pos, *target_vel, *pos, *vel, tilt, *command, *applied, float(active),
                         np.linalg.norm(goal - pos), np.linalg.norm(target_pos - pos),
                         *est.pos, *est.vel, *gps, baro, cam_h, *cam_v, frame_idx])
    finally:
        if cam is not None:
            cam.close()

    cols = (["step", "t"] + [f"target_{a}" for a in "xyz"] + [f"target_v{a}" for a in "xyz"]
            + [f"pos_{a}" for a in "xyz"] + [f"vel_{a}" for a in "xyz"] + ["tilt_deg"]
            + [f"command_{i}" for i in range(1, 5)] + [f"applied_{i}" for i in range(1, 5)]
            + ["fault_active", "dist_to_B", "track_error"]
            + [f"est_{a}" for a in "xyz"] + [f"est_v{a}" for a in "xyz"]
            + [f"gps_{a}" for a in "xyz"] + ["baro_z", "cam_h", "cam_vx", "cam_vy", "cam_frame"])
    log = pd.DataFrame(rows, columns=cols)
    for c in ("step", "fault_active", "cam_frame"):
        log[c] = log[c].astype(int)
    info = {"route": route.to_dict(), "fault": fault.to_dict() if fault else None,
            "hover_feasible": True if fault is None else hover_feasible(facts, fault.motor, fault.loss),
            "estimator_rejected_measurements": est.rejected}
    return log, frames, info, route


def estimation_errors(log):
    pos = log[["pos_x", "pos_y", "pos_z"]].to_numpy()
    vel = log[["vel_x", "vel_y", "vel_z"]].to_numpy()
    e_pos = log[["est_x", "est_y", "est_z"]].to_numpy() - pos
    e_vel = log[["est_vx", "est_vy", "est_vz"]].to_numpy() - vel
    out = {"est_pos_rmse": float(np.sqrt((e_pos ** 2).sum(1).mean())),
           "est_z_rmse": float(np.sqrt((e_pos[:, 2] ** 2).mean())),
           "est_xy_rmse": float(np.sqrt((e_pos[:, :2] ** 2).sum(1).mean())),
           "est_vel_rmse": float(np.sqrt((e_vel ** 2).sum(1).mean())),
           "est_vz_rmse": float(np.sqrt((e_vel[:, 2] ** 2).mean()))}
    cam = log[np.isfinite(log["cam_h"])]
    if len(cam):
        out["cam_h_rmse"] = float(np.sqrt(((cam["cam_h"] - cam["pos_z"]) ** 2).mean()))
        out["cam_h_valid_frac"] = float(len(cam) / max(1, (log["step"] % 5 == 0).sum()))
        cv = cam[np.isfinite(cam["cam_vx"])]
        if len(cv):
            out["cam_v_rmse"] = float(np.sqrt(((cv[["cam_vx", "cam_vy"]].to_numpy()
                                                - cv[["vel_x", "vel_y"]].to_numpy()) ** 2).sum(1).mean()))
    return out


# ---------------------------------------------------------------- resumable runner
def _code_hashes():
    files = [HERE / "sensing.py", HERE / "experiment.py"] + sorted((HERE / "x2").glob("*.py"))
    return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in files}


def _settings():
    s = {k: v for k, v in vars(sensing).items() if k.isupper() and not isinstance(v, np.ndarray) and k != "WINDOW"}
    s["GPS_SIGMA"] = sensing.GPS_SIGMA.tolist()
    return {"sensing": s, "x2_config": {k: v for k, v in vars(config).items() if k.isupper()},
            "conditions": CONDITIONS, "noise_seeds": NOISE_SEEDS}


def _key(fid):
    return hashlib.sha256(json.dumps({"id": fid, "settings": _settings(), "gains": Gains().to_dict(),
                                      "code": _code_hashes()}, sort_keys=True, default=str).encode()).hexdigest()[:16]


def is_done(fid):
    f = DATA / fid / "done.json"
    return f.exists() and json.loads(f.read_text()).get("key") == _key(fid)


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def _manifest(session):
    path = DATA / "manifest.json"
    m = json.loads(path.read_text()) if path.exists() else {"experiment": "camera_effect", "sessions": []}
    try:  # the git revision, if this folder is inside a git repository
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
                             timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        rev = None
    m.update({"python": sys.version.split()[0], "platform": platform.platform(),
              "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__, "pandas": pd.__version__},
              "git_revision": rev, "model_files": model_file_hashes(find_scene()),
              "settings": _settings(), "pid_gains": Gains().to_dict(), "code": _code_hashes()})
    m["sessions"] = [s for s in m["sessions"] if s["started"] != session["started"]] + [session]
    path.write_text(json.dumps(m, indent=2, default=str))


def run(say=print):
    DATA.mkdir(parents=True, exist_ok=True)
    for tmp in DATA.glob("*.tmp"):
        say(f"[resume] removing half-written {tmp.name}")
        shutil.rmtree(tmp)
    session = {"started": _now(), "flown": [], "skipped": []}
    _manifest(session)
    model = sensing.build_model()
    flights = flight_list()
    for n, (fid, cond, seed, p, loss) in enumerate(flights, start=1):
        if is_done(fid):
            session["skipped"].append(fid)
            continue
        t0 = time.time()
        save = cond == "camera" and seed == SAVE_FRAMES_SEED
        log, frames, info, route = fly(cond, seed, p, loss, model, save_frames=save)
        scores = {**metrics.compute(log, route), **estimation_errors(log)}
        tmp = DATA / f"{fid}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        log.to_csv(tmp / "flight.csv", index=False, float_format="%.6g")
        if frames:
            np.savez_compressed(tmp / "frames.npz", frames=np.stack(frames))
        (tmp / "flight.json").write_text(json.dumps(
            {"flight_id": fid, "condition": cond, "noise_seed": seed, "route_seed": p, "loss": loss, **info,
             "metrics": scores}, indent=2, default=str))
        (tmp / "done.json").write_text(json.dumps({"key": _key(fid), "finished": _now()}))
        if (DATA / fid).exists():
            shutil.rmtree(DATA / fid)
        tmp.rename(DATA / fid)
        session["flown"].append(fid)
        _manifest(session)
        out = f"CRASH {scores['crash_t']:.2f}s" if scores["crashed"] else f"{scores['final_dist_to_B']:.3f} m from B"
        say(f"[{n:3d}/{len(flights)}] {fid}: {out}, est z rmse {scores['est_z_rmse']:.3f} m ({time.time() - t0:.1f} s)")
    session["finished"] = _now()
    _manifest(session)
    summary = write_summary()
    say(f"done: {len(session['flown'])} flown, {len(session['skipped'])} already done -> {DATA}")
    return summary


def write_summary():
    rows = []
    for fid, cond, seed, p, loss in flight_list():
        if not (DATA / fid / "done.json").exists():
            continue
        info = json.loads((DATA / fid / "flight.json").read_text())
        rows.append({"flight_id": fid, "condition": cond, "noise_seed": seed, "route": f"path{p}", "loss": loss,
                     "hover_feasible": info["hover_feasible"], **info["metrics"]})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "summary.csv", index=False)
    return df


def status():
    flights = flight_list()
    for cond in CONDITIONS:
        sub = [f for f in flights if f[1] == cond]
        print(f"{cond:10s} {sum(is_done(f[0]) for f in sub)}/{len(sub)} flights done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    status() if args.status else run(say=lambda m: print(m, flush=True))
