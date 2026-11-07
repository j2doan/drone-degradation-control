"""Train the residual PPO policy for the 1D hover task (resumable), and evaluate it.

    python train.py                 # train seeds 0-2 (resumes from checkpoints), evaluate
    python train.py --seeds 0       # one seed only

The folder is self-contained: data/ is always written next to this file, whatever the working directory.

Each seed trains TOTAL_STEPS decisions. A checkpoint is written every CHECKPOINT_EVERY steps; if training is cut off
(laptop closed, crash), running the same command continues from the newest checkpoint. A finished seed has
model_final.zip and is skipped.
"""

import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sim1d  # noqa: E402

DATA = HERE / "data"
TOTAL_STEPS = 300_000
CHECKPOINT_EVERY = 50_000
N_ENVS = 8
PPO_KWARGS = dict(n_steps=1024, batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[32, 32]))
EVAL_LOSSES = (0.0, 0.125, 0.25, 0.375, 0.5)
SWEEP_LOSSES = tuple(np.round(np.arange(0.0, 0.5001, 0.025), 3))
EVAL_ONSET = 3.0


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def train_seed(seed: int, say=print):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecMonitor

    folder = DATA / f"seed{seed}"
    final = folder / "model_final.zip"
    if final.exists():
        say(f"[seed {seed}] already trained, skipped")
        return PPO.load(final, device="cpu")
    folder.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(folder.glob("ckpt_*_steps.zip"), key=lambda p: int(re.findall(r"(\d+)_steps", p.name)[0]))
    session = len(list(folder.glob("monitor_*.monitor.csv")))
    env = VecMonitor(make_vec_env(sim1d.HoverEnv, n_envs=N_ENVS, seed=seed * 1000 + session),
                     filename=str(folder / f"monitor_{session}"))
    if checkpoints:
        model = PPO.load(checkpoints[-1], env=env, device="cpu")
        say(f"[seed {seed}] resuming from {checkpoints[-1].name} ({model.num_timesteps} steps)")
    else:
        model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **PPO_KWARGS)
    remaining = TOTAL_STEPS - model.num_timesteps
    t0 = time.time()
    if remaining > 0:
        cb = CheckpointCallback(save_freq=max(1, CHECKPOINT_EVERY // N_ENVS), save_path=str(folder), name_prefix="ckpt")
        model.learn(total_timesteps=remaining, callback=cb, reset_num_timesteps=False)
    model.save(final)
    log = folder / "train_log.json"
    hist = json.loads(log.read_text()) if log.exists() else []
    hist.append({"session": session, "finished": _now(), "seconds": round(time.time() - t0, 1),
                 "steps_this_session": int(max(remaining, 0)), "total_steps": int(model.num_timesteps)})
    log.write_text(json.dumps(hist, indent=2))
    say(f"[seed {seed}] trained {model.num_timesteps} steps ({time.time() - t0:.0f} s)")
    return model


def learning_curve(seed: int) -> pd.DataFrame:
    """Episode returns in training order (all sessions), with the cumulative decision count."""
    parts = []
    for f in sorted((DATA / f"seed{seed}").glob("monitor_*.monitor.csv")):
        parts.append(pd.read_csv(f, skiprows=1))
    df = pd.concat(parts, ignore_index=True)
    df["steps"] = df["l"].cumsum()
    return df


def stiff_pid_episode(scenario):
    """Baseline with stiffer gains (kp 5, kd 3.5, ki 2); everything else as the PID."""
    old = (sim1d.KP, sim1d.KD, sim1d.KI, sim1d.I_LIMIT)
    sim1d.KP, sim1d.KD, sim1d.KI = 5.0, 3.5, 2.0
    sim1d.I_LIMIT = (sim1d.MAX_THRUST - sim1d.WEIGHT) / (sim1d.MASS * sim1d.KI)
    try:
        return sim1d.run_episode(scenario, "pid")
    finally:
        sim1d.KP, sim1d.KD, sim1d.KI, sim1d.I_LIMIT = old


def evaluate(models: dict, say=print):
    """Fixed test: 5 losses at onset 3 s, plus a fine sweep 0-50 %. Saves per-flight CSVs and summary.csv."""
    out = DATA / "eval"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    controllers = [("pid", None), ("stiff_pid", None), ("oracle", None)] + [(f"ppo_seed{s}", m) for s, m in models.items()]
    for loss in sorted(set(EVAL_LOSSES) | set(SWEEP_LOSSES)):
        sc = sim1d.Scenario(float(loss), EVAL_ONSET)
        for name, model in controllers:
            if name == "stiff_pid":
                log = stiff_pid_episode(sc)
            elif name.startswith("ppo"):
                log = sim1d.run_episode(sc, "ppo", model)
            else:
                log = sim1d.run_episode(sc, name)
            if float(loss) in EVAL_LOSSES:
                pd.DataFrame(log).to_csv(out / f"{name}_loss{loss * 100:04.1f}.csv", index=False, float_format="%.6g")
            rows.append({"controller": name, "loss": float(loss), "in_test_grid": float(loss) in EVAL_LOSSES,
                         **sim1d.score(log, sc)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    say(f"evaluation: {len(df)} episodes -> {out}")
    return df


def write_manifest(seeds):
    import gymnasium
    import stable_baselines3
    import torch
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in (HERE / "sim1d.py", HERE / "train.py")}
    path = DATA / "manifest.json"
    m = json.loads(path.read_text()) if path.exists() else {"experiment": "ppo_1d", "runs": []}
    m.update({"python": sys.version.split()[0], "platform": platform.platform(),
              "versions": {"numpy": np.__version__, "torch": torch.__version__, "gymnasium": gymnasium.__version__,
                           "stable_baselines3": stable_baselines3.__version__},
              "total_steps_per_seed": TOTAL_STEPS, "n_envs": N_ENVS, "ppo": PPO_KWARGS, "seeds": list(seeds),
              "eval_losses": EVAL_LOSSES, "eval_onset_s": EVAL_ONSET, "code": code})
    m["runs"].append({"finished": _now()})
    path.write_text(json.dumps(m, indent=2, default=str))


def main(seeds=(0, 1, 2), say=print):
    DATA.mkdir(exist_ok=True)
    models = {s: train_seed(s, say) for s in seeds}
    summary = evaluate(models, say)
    write_manifest(seeds)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    main(tuple(ap.parse_args().seeds), say=lambda m: print(m, flush=True))
