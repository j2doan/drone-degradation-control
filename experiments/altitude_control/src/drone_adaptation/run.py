"""Run the first paired baseline experiment: python3 -m drone_adaptation.run."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import sys

from .core import AdaptiveController, ExperimentConfig, PIDController
from .experiment import METHODS, run_episode


def write_report(output: Path, summaries: list[dict], manifest: dict) -> None:
    text = [
        "# First baseline experiment",
        "",
        "These are generated **one-dimensional model checks**, not Isaac Sim, learned-policy,",
        "motor-failure, or sim-to-real results. Attitude is assumed perfectly stabilized.",
        "PID and the explicit estimator receive noisy measurements and issued commands.",
        "The oracle is intentionally privileged and is only a diagnostic reference.",
        "",
        "| Effectiveness | Altitude noise (m) | Method | Recovered | Failed | Capped recovery (s) | Peak loss (m) |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for eta in manifest["etas"]:
        for noise in manifest["noise_z_levels_m"]:
            for method in METHODS:
                group = [
                    row for row in summaries
                    if row["eta_after"] == eta and row["noise_z_m"] == noise and row["method"] == method
                ]
                text.append(
                    f"| {eta:.2f} | {noise:.3f} | {method} | "
                    f"{sum(row['recovered'] for row in group)}/{len(group)} | "
                    f"{sum(row['failed'] for row in group)}/{len(group)} | "
                    f"{statistics.mean(row['capped_recovery_time_s'] for row in group):.3f} | "
                    f"{statistics.mean(row['peak_altitude_loss_m'] for row in group):.3f} |"
                )
    text += [
        "",
        "## How to read this",
        "",
        "- Effectiveness 1.0 is a no-fault control; its fault time is only an evaluation anchor.",
        "- Recovery requires true altitude error <= 0.10 m and speed <= 0.15 m/s for 1 second.",
        "- Recovery time is the beginning of that sustained interval; confirmation is logged separately.",
        "- Nonrecoveries receive the common horizon (normally 8 s) in capped recovery time.",
        "- Any physical boundary failure disqualifies recovery, even after an earlier settled interval.",
        "- Post-fault RMSE covers only the recorded trace; a failure shortens it. Never compare it alone.",
        "- Seeds here vary evaluation noise and onset, **not independently trained policies**.",
        "- Gains and estimator settings are prototype defaults, not a tuned final benchmark.",
        "",
        "## Files",
        "",
        "- `manifest.json`: exact scenarios, configuration, controller settings, runtime and provenance.",
        "- `summary.csv`: one row per method/scenario, retaining nonrecoveries and failures.",
        "- `trajectories.csv`: true state, measurements, issued commands and hidden diagnostic state.",
        "  A row at t stores state/measurement at t and actuation over [t,t+dt].",
        "  The terminal row has no measurement or command because no further action is taken.",
        "",
        "## Next check",
        "",
        "Inspect the altitude and command traces,",
        "then read this experiment’s `README.md` before adding learned residuals.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/starter"))
    parser.add_argument("--episodes", type=int, default=5, help="Paired evaluation seeds per condition")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--etas", nargs="+", type=float, default=[1.0, 0.9, 0.75])
    parser.add_argument("--noise-z", nargs="+", type=float, default=[0.0, 0.01], help="Altitude standard deviation; velocity noise is 3x this value")
    parser.add_argument("--fault-time", type=float, default=None, help="Optional common onset; otherwise sampled in [3,7] seconds")
    parser.add_argument("--overwrite", action="store_true", help="Replace this runner's files in an existing output directory")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if len(set(args.etas)) != len(args.etas) or len(set(args.noise_z)) != len(args.noise_z):
        parser.error("Effectiveness and noise lists must not contain duplicate conditions")
    base = ExperimentConfig()
    try:
        for eta in args.etas:
            for noise in args.noise_z:
                replace(base, eta_after=eta, noise_z_m=noise, noise_vz_m_s=3.0 * noise)
        if args.fault_time is not None:
            replace(base, fault_time_s=args.fault_time)
            if base.duration_s - args.fault_time < 8.0:
                raise ValueError("--fault-time must leave the complete 8-second evaluation horizon")
    except ValueError as error:
        parser.error(str(error))
    reserved = ("summary.csv", "trajectories.csv", "manifest.json", "RESULTS.md")
    if not args.overwrite and any((args.output / name).exists() for name in reserved):
        parser.error("Output contains an earlier run. Choose a new --output or pass --overwrite.")
    # Check physical feasibility before writing a run: eta*Tmax must exceed weight.
    if any(eta * base.max_thrust_n <= base.mass_kg * base.gravity_m_s2 for eta in args.etas):
        parser.error("A requested effectiveness leaves insufficient thrust to hover; select a recoverable eta.")
    args.output.mkdir(parents=True, exist_ok=True)
    known = {"target_m": base.target_m, "mass_kg": base.mass_kg, "gravity_m_s2": base.gravity_m_s2, "max_thrust_n": base.max_thrust_n}
    pid = PIDController(**known)
    adaptive = AdaptiveController(**known)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "model": "1D vertical point mass, zero actuator lag, perfectly stabilized attitude",
        "status": "baseline engineering checks; no learned policy or Isaac integration",
        "base_config": asdict(base),
        "methods": list(METHODS),
        "etas": args.etas,
        "noise_z_levels_m": args.noise_z,
        "velocity_noise_multiplier": 3.0,
        "episodes_per_condition": args.episodes,
        "first_seed": args.seed,
        "controller_settings": {
            "pid_gains": {name: getattr(pid, name) for name in ("kp", "kd", "ki")},
            "anti_windup": "conditional integration at final command limits",
            "adaptive_estimator": {name: getattr(adaptive, name) for name in ("estimation_window_s", "forgetting_factor", "initial_covariance", "eta_min", "eta_max")},
            "oracle": "Same PID with true effectiveness, diagnostic only",
        },
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob("*.py"))},
        "metrics": {"horizon_s": 8.0, "position_tolerance_m": 0.1, "velocity_tolerance_m_s": 0.15, "hold_s": 1.0},
        "scenarios": [],
    }
    summaries: list[dict] = []
    with (args.output / "trajectories.csv").open("w", newline="", encoding="utf-8") as trace_file:
        writer = None
        for eta in args.etas:
            for noise in args.noise_z:
                for seed in range(args.seed, args.seed + args.episodes):
                    # Independent stream: sampling onset does not shift sensor noise.
                    onset = args.fault_time
                    if onset is None:
                        onset = round(random.Random(seed + 1_000_003).uniform(3.0, 7.0) / base.dt_s) * base.dt_s
                    config = replace(base, eta_after=eta, noise_z_m=noise, noise_vz_m_s=3.0 * noise, fault_time_s=onset)
                    scenario_id = f"eta{eta!r}_noise{noise!r}_seed{seed}"
                    manifest["scenarios"].append({"scenario_id": scenario_id, "seed": seed, "config": asdict(config)})
                    for method in METHODS:
                        result = run_episode(config, method, seed)
                        result.summary["scenario_id"] = scenario_id
                        summaries.append(result.summary)
                        for row in result.rows:
                            record = {"scenario_id": scenario_id, "method": method, "seed": seed, "eta_after": eta, "noise_z_m": noise, **row}
                            if writer is None:
                                writer = csv.DictWriter(trace_file, fieldnames=list(record))
                                writer.writeheader()
                            writer.writerow(record)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(args.output, summaries, manifest)
    print(f"Completed {len(summaries)} episodes ({args.episodes} paired seeds per condition).")
    print(f"Read {args.output / 'RESULTS.md'}")
    print("Optional plots: python3 -m drone_adaptation.plot --input " + str(args.output))


if __name__ == "__main__":
    main()
