"""Exploratory 1D screening of internal actuator changes and estimator memory.

Run with python -m drone_adaptation.switching_study --output results/switching-dev.
Classical evaluation uses the standard library only. --models optionally evaluates
existing frozen residual-PPO checkpoints; it never trains or selects checkpoints.
This is not a quadrotor/Isaac benchmark or a test of a new algorithm.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import statistics
from typing import Any

from .core import ExperimentConfig, OracleController, PIDController
from .switching import EffectivenessSchedule, ForgettingController, SwitchingAltitudePlant


METHODS = ("pid", "window_short", "window_long", "rls_fixed", "rls_variable", "oracle")
SCENARIOS = ("healthy", "sustained_loss", "restoration", "severity_change", "repeated_loss")


def make_schedule(kind: str, onset: float, dwell: float) -> EffectivenessSchedule:
    """All methods receive the identical hidden schedule, not its labels."""
    if kind == "healthy":
        events = ()
    elif kind == "sustained_loss":
        events = ((onset, 0.75),)
    elif kind == "restoration":
        events = ((onset, 0.75), (onset + dwell, 1.0))
    elif kind == "severity_change":
        events = ((onset, 0.85), (onset + dwell, 0.70))
    elif kind == "repeated_loss":
        events = tuple((onset + i * dwell, eta) for i, eta in enumerate((0.75, 1.0, 0.75, 1.0)))
    else:
        raise ValueError(f"unknown scenario: {kind}")
    return EffectivenessSchedule(events=events)


def summarize_trace(rows: list[dict], config: ExperimentConfig, schedule: EffectivenessSchedule,
                    *, observation_start_s: float, recovery_horizon_s: float = 4.0,
                    hold_s: float = 1.0) -> tuple[dict, list[dict]]:
    """Evaluate every transition, never allowing a recovery hold across a switch.

    Episode MAE uses a fixed horizon and assigns a declared tail error after a
    boundary failure. Unreached events remain explicit. Short-dwell transitions
    retain excursion/error metrics but are censored for recovery confirmation.
    """
    if not rows or recovery_horizon_s <= 0 or hold_s <= 0:
        raise ValueError("a trace and positive metric durations are required")
    final_time = rows[-1]["time_s"]
    failed = any(r["z_m"] <= config.min_altitude_m or r["z_m"] >= config.max_altitude_m for r in rows)
    failure_tail_error = max(config.target_m - config.min_altitude_m,
                             config.max_altitude_m - config.target_m)
    integral = 0.0
    for left, right in zip(rows, rows[1:]):
        start, end = max(left["time_s"], observation_start_s), right["time_s"]
        if end > start:
            integral += (end - start) * (abs(left["z_m"] - config.target_m)
                                         + abs(right["z_m"] - config.target_m)) / 2
    if failed:
        integral += max(0.0, config.duration_s - max(final_time, observation_start_s)) * failure_tail_error
    duration = config.duration_s - observation_start_s
    selected = [r for r in rows if r["time_s"] >= observation_start_s - 1e-9]
    controls = [r["command_n"] for r in selected if r.get("command_n") is not None]
    episode = {
        "failed": failed,
        "trace_duration_s": final_time,
        "failure_penalized_mae_m": integral / duration,
        "failure_tail_error_m": failure_tail_error,
        "peak_absolute_error_m": max((abs(r["z_m"] - config.target_m) for r in selected), default=None),
        "saturation_fraction": sum(c <= 1e-9 or c >= config.max_thrust_n - 1e-9 for c in controls) / len(controls) if controls else None,
        "control_deviation_rms_n": math.sqrt(statistics.mean((c - config.mass_kg * config.gravity_m_s2) ** 2 for c in controls)) if controls else None,
    }
    events = []
    for i, (onset, eta) in enumerate(schedule.events):
        previous_eta = schedule.initial_eta if i == 0 else schedule.events[i - 1][1]
        next_onset = schedule.events[i + 1][0] if i + 1 < len(schedule.events) else config.duration_s
        deadline = min(onset + recovery_horizon_s, next_onset, config.duration_s)
        # A horizon endpoint can confirm recovery; a new regime's endpoint cannot.
        has_next_switch = i + 1 < len(schedule.events) and math.isclose(deadline, next_onset, abs_tol=1e-9)
        segment = [r for r in rows if r["time_s"] >= onset - 1e-9 and
                   (r["time_s"] < deadline - 1e-9 if has_next_switch else r["time_s"] <= deadline + 1e-9)]
        reached = final_time > onset + 1e-9
        needed_duration = hold_s + (config.dt_s if has_next_switch else 0.0)
        eligible = reached and deadline - onset >= needed_duration - 1e-9
        candidate = recovered_at = confirmed_at = None
        if reached and eligible:
            for row in segment:
                inside = abs(row["z_m"] - config.target_m) <= 0.10 and abs(row["vz_m_s"]) <= 0.15
                if not inside:
                    candidate = None
                else:
                    candidate = row["time_s"] if candidate is None else candidate
                    if row["time_s"] - candidate >= hold_s - 1e-9:
                        recovered_at, confirmed_at = candidate - onset, row["time_s"] - onset
                        break
        if not reached:
            status = "not_reached_after_failure"
        elif not eligible:
            status = "insufficient_dwell"
        elif recovered_at is not None:
            status = "confirmed"
        else:
            status = "not_confirmed"
        if failed and onset <= final_time < deadline:
            recovered_at = confirmed_at = None
            status = "failed_in_window"
        error_area = sum(
            (b["time_s"] - a["time_s"]) * (abs(a["z_m"] - config.target_m) + abs(b["z_m"] - config.target_m)) / 2
            for a, b in zip(segment, segment[1:])
        )
        estimate_rows = [r for r in segment if r.get("eta_estimate") is not None]
        events.append({
            "event_index": i, "event_time_s": onset, "eta_before": previous_eta, "eta_after": eta,
            "event_type": "restoration" if eta > previous_eta else "degradation",
            "window_duration_s": deadline - onset, "recovery_assessable": eligible,
            "recovery_status": status, "recovery_time_s": recovered_at, "confirmation_time_s": confirmed_at,
            "capped_recovery_time_s": (recovered_at if recovered_at is not None else deadline - onset) if eligible else None,
            "observed_error_area_m_s": error_area,
            "observed_peak_absolute_error_m": max((abs(r["z_m"] - config.target_m) for r in segment), default=None),
            "observed_overshoot_m": max((max(0.0, r["z_m"] - config.target_m) for r in segment), default=None),
            "eta_estimation_mae": statistics.mean(abs(r["eta_estimate"] - r["eta_true"]) for r in estimate_rows) if estimate_rows else None,
        })
    return episode, events


def run_classical(config: ExperimentConfig, schedule: EffectivenessSchedule, method: str, seed: int) -> list[dict]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    plant = SwitchingAltitudePlant(config, schedule)
    known = dict(target_m=config.target_m, mass_kg=config.mass_kg,
                 gravity_m_s2=config.gravity_m_s2, max_thrust_n=config.max_thrust_n)
    if method == "pid":
        controller = PIDController(**known)
    elif method == "oracle":
        controller = OracleController(**known, true_eta=lambda: plant.eta_at(plant.state.time_s))
    else:
        controller = ForgettingController(**known, mode=method)
    rng, rows = random.Random(seed), []
    for _ in range(config.num_steps):
        state = plant.state
        observation = plant.observe(rng.gauss(0, config.noise_z_m), rng.gauss(0, config.noise_vz_m_s))
        command = controller.act(observation, config.dt_s)
        after = plant.step(command)
        rows.append(dict(time_s=state.time_s, z_m=state.z_m, vz_m_s=state.vz_m_s,
                         observed_z_m=observation.z_m, observed_vz_m_s=observation.vz_m_s,
                         command_n=plant.last_command_n, applied_thrust_n=plant.last_applied_thrust_n,
                         eta_true=plant.eta_at(state.time_s), eta_estimate=controller.eta_estimate,
                         integral_state=controller.integral_state,
                         forgetting_factor=getattr(controller, "last_forgetting_factor", None)))
        if after.z_m <= config.min_altitude_m or after.z_m >= config.max_altitude_m:
            break
    state = plant.state
    rows.append(dict(time_s=state.time_s, z_m=state.z_m, vz_m_s=state.vz_m_s,
                     command_n=None, applied_thrust_n=None, eta_true=plant.eta_at(state.time_s),
                     eta_estimate=controller.eta_estimate, integral_state=controller.integral_state))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=48000)
    parser.add_argument("--dwells", type=float, nargs="+", default=[0.5, 2.0])
    parser.add_argument("--noise-z", type=float, nargs="+", default=[0.0, 0.01, 0.03])
    parser.add_argument("--models", type=Path, help="Optional existing residual-pilot directory; no training")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory; existing runs are never overwritten")
    if args.episodes < 1 or args.first_seed < 0:
        parser.error("episodes must be positive and first-seed nonnegative")
    if any(not math.isfinite(d) or d <= 0 or d > 4 for d in args.dwells):
        parser.error("dwell times must be finite and in (0,4] seconds")
    if any(not math.isfinite(n) or n < 0 for n in args.noise_z):
        parser.error("noise levels must be finite and nonnegative")
    if len(set(args.dwells)) != len(args.dwells) or len(set(args.noise_z)) != len(args.noise_z):
        parser.error("dwell and noise values must be unique")
    base = ExperimentConfig(duration_s=26.0)
    scenarios = []
    for kind in SCENARIOS:
        # Dwell has no effect on these two scenarios; do not duplicate them.
        dwells = args.dwells if kind not in ("healthy", "sustained_loss") else [args.dwells[0]]
        for dwell in dwells:
            for noise in args.noise_z:
                for seed in range(args.first_seed, args.first_seed + args.episodes):
                    onset = round(random.Random(seed + 1_000_003).uniform(3.6, 4.4) / base.dt_s) * base.dt_s
                    schedule = make_schedule(kind, onset, dwell)
                    cfg = replace(base, noise_z_m=noise, noise_vz_m_s=3 * noise, fault_time_s=onset)
                    if any(eta * cfg.max_thrust_n <= cfg.mass_kg * cfg.gravity_m_s2 for _, eta in schedule.events):
                        parser.error("a schedule leaves insufficient scalar thrust for hover")
                    sid = f"{kind}_d{dwell!r}_n{noise!r}_s{seed}"
                    scenarios.append((sid, kind, dwell, seed, cfg, schedule))
    training = None
    if args.models:
        training = json.loads((args.models / "training_manifest.json").read_text())
        if training.get("status") != "complete":
            parser.error("models must have a complete training manifest")
        if set(training["settings"]["seeds"]) & set(range(args.first_seed, args.first_seed + args.episodes)):
            parser.error("evaluation seeds overlap training seeds")
        for name in ("core.py", "residual_env.py", "residual_policy.py"):
            if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != training["source_sha256"].get(name):
                parser.error(f"original model code changed: {name}")
    args.output.mkdir(parents=True)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), status="running",
                    scope="Exploratory 1D scalar-actuator screening, not Isaac, not individual-rotor evidence",
                    split="development; exposed results are not a blind test", platform=platform.platform(),
                    methods=list(METHODS), config=asdict(base), episodes_per_condition=args.episodes,
                    controller_settings=dict(pid_gains=dict(kp=5.0, kd=3.5, ki=2.0),
                        observation_window_s=0.1, history_windows_s=dict(window_short=0.2, window_long=1.0),
                        fixed_forgetting=0.9, variable_forgetting_bounds=[0.2, 0.995],
                        innovation_tolerance_eta=0.08, innovation_scale_eta=0.2,
                        initial_covariance=10.0, eta_bounds=[0.4, 1.2],
                        residual_scale_weight=0.6, tuning="untuned development defaults"),
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
                    metrics=dict(position_tolerance_m=0.10, velocity_tolerance_m_s=0.15, hold_s=1.0, event_horizon_s=4.0,
                                 insufficient_dwell="censored; retained in error metrics, excluded from recovery denominator",
                                 failure_tail="fixed bound-distance error for remaining episode horizon"),
                    scenarios=[dict(id=sid, kind=kind, dwell_s=dwell, seed=seed, config=asdict(cfg),
                                    initial_eta=schedule.initial_eta, events=schedule.events)
                               for sid, kind, dwell, seed, cfg, schedule in scenarios])
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    episodes, events, traces = [], [], []

    def collect(rows, item, method, training_seed=None):
        sid, kind, dwell, seed, cfg, schedule = item
        episode, transition_rows = summarize_trace(rows, cfg, schedule, observation_start_s=cfg.fault_time_s)
        metadata = dict(scenario_id=sid, scenario=kind, dwell_s=dwell, noise_z_m=cfg.noise_z_m,
                        evaluation_seed=seed, method=method, training_seed=training_seed)
        episodes.append({**metadata, **episode})
        events.extend({**metadata, **row} for row in transition_rows)
        # Predeclared representative traces: first eval seed at 0.01m noise;
        # first training seed only, never a retrospectively chosen best run.
        if seed == args.first_seed and math.isclose(cfg.noise_z_m, 0.01) and (training_seed is None or training_seed == training["settings"]["seeds"][0]):
            traces.extend({**metadata, **row} for row in rows)

    for method in METHODS:
        for item in scenarios:
            collect(run_classical(item[4], item[5], method, item[3]), item, method)
        print(f"Completed {method}: {len(scenarios)} episodes", flush=True)
    if training:
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from .switching_residual import SwitchingResidualEnv
        torch.set_num_threads(1)
        manifest["training_manifest_sha256"] = hashlib.sha256((args.models / "training_manifest.json").read_bytes()).hexdigest()
        manifest["learned_training"] = training
        manifest["methods"].extend(dict.fromkeys(f"ppo_h{entry['history_s']:g}" for entry in training["models"]))
        manifest["checkpoint_sha256"] = {}
        manifest["evaluated_models"] = []
        for entry in training["models"]:
            path = args.models / entry["checkpoint"]
            if not path.is_file():
                path = args.models / entry["name"]
            manifest["checkpoint_sha256"][entry["name"]] = hashlib.sha256(path.read_bytes()).hexdigest()
            model = PPO.load(path, device="cpu")
            manifest["evaluated_models"].append({**entry,
                "parameter_count": sum(p.numel() for p in model.policy.parameters()),
                "inference": "deterministic; frozen weights"})
            envs, obs = [], []
            for item in scenarios:
                env = SwitchingResidualEnv(schedule=item[5], history_s=entry["history_s"], randomize=False,
                    scenario=item[4], residual_scale=training["settings"]["residual_scale_weight"])
                observation, _ = env.reset(seed=item[3])
                envs.append(env)
                obs.append(observation)
            active = list(range(len(envs)))
            while active:
                batch = {key: np.stack([obs[i][key] for i in active]) for key in obs[active[0]]}
                actions, _ = model.predict(batch, deterministic=True)
                remaining = []
                for i, action in zip(active, actions):
                    obs[i], _, terminated, truncated, _ = envs[i].step(action)
                    if terminated or truncated:
                        collect(envs[i].episode_rows(), scenarios[i], f"ppo_h{entry['history_s']:g}", entry["training_seed"])
                        envs[i].close()
                    else:
                        remaining.append(i)
                active = remaining
            print(f"Completed frozen {entry['name']}: {len(scenarios)} episodes", flush=True)
    write_csv(args.output / "episodes.csv", episodes)
    write_csv(args.output / "events.csv", events)
    write_csv(args.output / "traces.csv", traces)
    manifest.update(status="complete", episode_count=len(episodes), event_count=len(events),
                    learned_models="previous single-fault training; switched regimes and longer horizon are development shifts" if training else "none")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Internal adaptation screening — development results", "",
             "Scalar 1D altitude model; not a full-drone or individual-motor result. No new algorithm is claimed.",
             "Classical defaults are untuned references, not exact implementations of cited papers.",
             "Optional learned policies are existing, fixed-budget single-fault checkpoints, not retrained on switching.", "",
             "| Method | Episodes | Failures | Mean failure-penalized MAE (m) |", "|---|---:|---:|---:|"]
    for method in dict.fromkeys(r["method"] for r in episodes):
        group = [r for r in episodes if r["method"] == method]
        lines.append(f"| {method} | {len(group)} | {sum(r['failed'] for r in group)} | {statistics.mean(r['failure_penalized_mae_m'] for r in group):.4f} |")
    lines += ["", "This aggregate is descriptive: inspect scenario/noise/dwell strata before comparing methods.",
              "Evaluation episodes are not independent trained policies. No confidence interval across pooled episodes is a training-seed interval.",
              "Short-dwell events are censored for the one-second recovery hold. Event errors cover observed segments only; failures remain in episode metrics.",
              "A change after these exposed results must be evaluated on fresh held-out conditions. Full protocol: docs/INTERNAL_ADAPTATION_RESEARCH_PLAN.md.", ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines))
    print(f"Saved {len(episodes)} episode and {len(events)} event records to {args.output}")


if __name__ == "__main__":
    main()
