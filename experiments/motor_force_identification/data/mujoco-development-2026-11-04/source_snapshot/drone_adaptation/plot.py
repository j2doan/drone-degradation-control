"""Optional Matplotlib figures for an already completed baseline run."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics


COLORS = {"pid": "#2563a5", "adaptive": "#d55e00", "oracle": "#00856a"}
LABELS = {"pid": "PID", "adaptive": "Explicit adaptation", "oracle": "Oracle (privileged)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/starter"))
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--noise-z", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    eta = args.eta if args.eta is not None else min(manifest["etas"])
    noise = args.noise_z if args.noise_z is not None else max(manifest["noise_z_levels_m"])
    seed = args.seed if args.seed is not None else manifest["first_seed"]
    with (args.input / "summary.csv").open(newline="", encoding="utf-8") as file:
        summaries = list(csv.DictReader(file))
    selection = [row for row in summaries if float(row["eta_after"]) == eta and float(row["noise_z_m"]) == noise and int(row["seed"]) == seed]
    if not selection:
        parser.error("No scenario matches that effectiveness, noise, and seed")
    scenario_id = selection[0]["scenario_id"]
    traces: dict[str, list[dict]] = {method: [] for method in manifest["methods"]}
    with (args.input / "trajectories.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["scenario_id"] == scenario_id:
                traces[row["method"]].append(row)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Plotting is optional. Install its dependency in an isolated environment:\n"
            "python3 -m venv .venv\n"
            ".venv/bin/python -m pip install -r requirements-plot.txt\n"
            ".venv/bin/python -m drone_adaptation.plot --input " + str(args.input)
        ) from error

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    onset = float(selection[0]["fault_time_s"])
    target = manifest["base_config"]["target_m"]
    weight = manifest["base_config"]["mass_kg"] * manifest["base_config"]["gravity_m_s2"]
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9.2), constrained_layout=True)
    for method, rows in traces.items():
        times = [float(row["time_s"]) - onset for row in rows]
        style = "--" if method == "oracle" else "-"
        axes[0].plot(times, [float(row["z_m"]) for row in rows], color=COLORS[method], linestyle=style, label=LABELS[method], linewidth=2)
        command_rows = [row for row in rows if row["command_n"]]
        axes[1].plot([float(row["time_s"]) - onset for row in command_rows], [float(row["command_n"]) / weight for row in command_rows], color=COLORS[method], linestyle=style, linewidth=1.4)
        estimates = [row for row in rows if row["eta_estimate"]]
        if method == "adaptive":
            axes[2].plot([float(row["time_s"]) - onset for row in estimates], [float(row["eta_estimate"]) for row in estimates], color=COLORS[method], linewidth=2, label="Estimated effectiveness")
    truth = traces[manifest["methods"][0]]
    axes[2].step([float(row["time_s"]) - onset for row in truth], [float(row["eta_true"]) for row in truth], where="post", color="#333333", linestyle="--", label="True effectiveness (diagnostic)")
    axes[0].axhline(target, color="#666666", linewidth=1, linestyle=":", label="Target")
    axes[0].axhspan(target - 0.1, target + 0.1, color="#00856a", alpha=0.06)
    axes[0].set_ylabel("True altitude (m)")
    axes[1].set_ylabel("Issued thrust / weight")
    axes[2].set_ylabel("Effectiveness")
    axes[2].set_xlabel("Time relative to thrust change (s)")
    for axis in axes:
        axis.axvline(0, color="#666666", linestyle=":", linewidth=1)
        axis.grid(alpha=0.18)
        axis.set_xlim(-2, 8)
    axes[0].legend(ncol=2, loc="lower right")
    axes[2].legend(loc="best")
    figure.suptitle(
        "Starter altitude experiment — 1D model, classical baselines\n"
        f"{100 * (1 - eta):.0f}% thrust loss; altitude noise SD {noise:g} m; evaluation seed {seed}",
        fontsize=13,
    )
    for suffix in ("png", "svg"):
        figure.savefig(args.input / f"baseline_recovery.{suffix}", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    etas = sorted(manifest["etas"], reverse=True)
    for method in manifest["methods"]:
        for axis, metric in zip(axes, ("peak_altitude_loss_m", "capped_recovery_time_s")):
            means, deviations = [], []
            for condition in etas:
                group = [float(row[metric]) for row in summaries if row["method"] == method and float(row["noise_z_m"]) == noise and float(row["eta_after"]) == condition]
                means.append(statistics.mean(group))
                deviations.append(statistics.stdev(group) if len(group) > 1 else 0.0)
            axis.errorbar([100 * (1 - value) for value in etas], means, yerr=deviations, marker="o", capsize=3, color=COLORS[method], linestyle="--" if method == "oracle" else "-", label=LABELS[method])
    axes[0].set_ylabel("Peak altitude loss (m)")
    axes[1].set_ylabel("Capped recovery time (s)")
    for axis in axes:
        axis.set_xlabel("Thrust loss (%)")
        axis.grid(alpha=0.18)
        axis.set_ylim(bottom=0)
    axes[0].legend(fontsize=9)
    figure.suptitle(f"Baseline checks at altitude noise SD {noise:g} m\nMean ± 1 SD across evaluation episodes; no learned policies", fontsize=12)
    for suffix in ("png", "svg"):
        figure.savefig(args.input / f"baseline_robustness.{suffix}", dpi=160)
    plt.close(figure)
    print(f"Saved baseline_recovery and baseline_robustness PNG/SVG figures in {args.input}")


if __name__ == "__main__":
    main()
