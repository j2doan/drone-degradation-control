"""Create static research figures from a completed switching-development run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics


LABELS = {"pid": "PID", "window_short": "Short-window LS", "window_long": "Long-window LS",
          "rls_fixed": "Fixed forgetting", "rls_variable": "Variable forgetting", "oracle": "Fault-aware reference",
          "ppo_h0": "PPO: no extended history", "ppo_h0.1": "PPO: 0.1 s history",
          "ppo_h0.5": "PPO: 0.5 s history", "ppo_h1": "PPO: 1 s history"}
COLORS = {"pid": "#708090", "window_short": "#0072B2", "window_long": "#56B4E9",
          "rls_fixed": "#CC79A7", "rls_variable": "#D55E00", "oracle": "#009E73",
          "ppo_h0": "#B79A00", "ppo_h1": "#332288"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean_by_training_seed(rows, metric):
    """Weight each trained model equally; classical rows have an empty seed."""
    values = {}
    for row in rows:
        if row.get(metric) not in (None, ""):
            values.setdefault(row.get("training_seed", ""), []).append(float(row[metric]))
    return statistics.mean(statistics.mean(v) for v in values.values()) if values else math.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--noise-z", type=float, default=0.01)
    parser.add_argument("--dwell", type=float, default=2.0)
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        parser.error("evaluation must be complete")
    os.environ.setdefault("MPLCONFIGDIR", str(args.input.resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    episodes = read_csv(args.input / "episodes.csv")
    traces = read_csv(args.input / "traces.csv")
    representative = [r for r in traces if r["scenario"] == "restoration"
                      and math.isclose(float(r["noise_z_m"]), args.noise_z)
                      and math.isclose(float(r["dwell_s"]), args.dwell)]
    if not representative:
        parser.error("no saved restoration trace at this dwell/noise; default traces use0.01m noise")
    methods = [m for m in LABELS if any(r["method"] == m for r in episodes)]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 8), constrained_layout=True)
    shown = [m for m in methods if m not in ("ppo_h0.1", "ppo_h0.5")]
    for method in shown:
        rows = [r for r in representative if r["method"] == method]
        t = [float(r["time_s"]) for r in rows]
        if not rows:
            continue
        axes[0].plot(t, [float(r["z_m"]) for r in rows], label=LABELS[method],
                     color=COLORS[method], linewidth=1.5, linestyle="--" if method == "oracle" else "-")
        estimates = [r for r in rows if r.get("eta_estimate")]
        if estimates and method != "oracle":
            axes[1].plot([float(r["time_s"]) for r in estimates], [float(r["eta_estimate"]) for r in estimates],
                         color=COLORS[method], linewidth=1.3, label=LABELS[method])
    reference = [r for r in representative if r["method"] == "pid"]
    axes[1].step([float(r["time_s"]) for r in reference], [float(r["eta_true"]) for r in reference],
                 where="post", color="#222222", linestyle="--", linewidth=2, label="True effectiveness")
    scenario = next(s for s in manifest["scenarios"] if s["id"] == representative[0]["scenario_id"])
    target = scenario["config"]["target_m"]
    axes[0].axhline(target, color="#333333", linestyle=":", linewidth=1)
    axes[0].axhspan(target - 0.1, target + 0.1, color="#009E73", alpha=0.06)
    for axis in axes:
        for onset, _ in scenario["events"]:
            axis.axvline(onset, color="#555555", linestyle=":", linewidth=1)
        axis.grid(alpha=0.2)
        axis.set_xlim(scenario["events"][0][0] - 1.0, scenario["events"][-1][0] + 5.0)
    axes[0].set_ylabel("True altitude (m)")
    axes[1].set_ylabel("Effectiveness estimate")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(ncol=2, fontsize=8, loc="best")
    axes[1].legend(ncol=3, fontsize=8, loc="best")
    fig.suptitle("Internal actuator loss and restoration — exploratory 1D screen\n"
                 "First evaluation seed; first training seed for PPO; no checkpoint selection", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(args.input / f"restoration_trace.{suffix}", dpi=160)
    plt.close(fig)

    scenarios = ["healthy", "sustained_loss", "restoration", "severity_change", "repeated_loss"]
    matrix = []
    for method in methods:
        matrix.append([mean_by_training_seed(
            [r for r in episodes if r["method"] == method and r["scenario"] == kind
             and math.isclose(float(r["noise_z_m"]), args.noise_z)
             and (kind in ("healthy", "sustained_loss") or math.isclose(float(r["dwell_s"]), args.dwell))],
            "failure_penalized_mae_m") for kind in scenarios])
    array = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(10, max(4.5, 0.43 * len(methods) + 2)), constrained_layout=True)
    mesh = axis.imshow(array, cmap="YlOrRd", aspect="auto", vmin=0)
    axis.set_xticks(range(len(scenarios)), [s.replace("_", "\n") for s in scenarios])
    axis.set_yticks(range(len(methods)), [LABELS[m] for m in methods])
    for i in range(len(methods)):
        for j in range(len(scenarios)):
            value = array[i, j]
            axis.text(j, i, f"{value:.3f}", ha="center", va="center",
                      color="white" if value > np.nanmax(array) * 0.6 else "#222222", fontsize=9)
    fig.colorbar(mesh, ax=axis, label="Failure-penalized mean absolute altitude error (m)")
    axis.set_title(f"Development screening at altitude noise SD {args.noise_z:g} m; switching dwell {args.dwell:g} s\n"
                   "PPO means weight each training seed equally; descriptive results, no novelty claim", fontsize=11, pad=15)
    for suffix in ("png", "svg"):
        fig.savefig(args.input / f"screening_error.{suffix}", dpi=160)
    plt.close(fig)
    print(f"Saved restoration_trace and screening_error PNG/SVG figures in {args.input}")


if __name__ == "__main__":
    main()
