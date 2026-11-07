"""One figure per flight CSV: 3 stacked panels on a shared time axis.

  1. Distance to B        did the drone get there, and when? (band = within the arrival tolerance)
  2. Altitude             how much height the fault cost, did it come back, did it hit the ground? (shaded crash zone)
  3. Motor thrust         how the controller fought the fault: the faulty motor's asked vs delivered thrust in colour,
                          the other motors in gray
Same style everywhere: blue = drone, gray dashed = target, red dashed vertical line = fault onset, red solid vertical
line = crash. After a crash every line stops (the readings of a crashed drone mean nothing); the CSV keeps all rows.
The title line summarises the run. Old CSVs without fault columns are fine: there is just no fault line.

Run (from the isaac/ folder):
    python analysis/plots.py data/<name>.csv [--start X Y Z] [--goal X Y Z] [--out path.png]
Default output: data/<name>.png (next to the CSV). Start/goal and controller are read from data/<name>.json if present.
"""

import json
import math
import pathlib
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # isaac/ folder

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.metrics import ARRIVE_TOL, CRASH_MIN_Z, POS, compute_metrics, fault_onset_index
from analysis.route import distance_to, endpoints_for

DRONE_COLOR = "tab:blue"
TARGET_COLOR = "gray"
FAULT_COLOR = "tab:red"  # fault onset = dashed, crash = solid
FAULTY_MOTOR_COLOR = "tab:orange"
OTHER_MOTOR_COLOR = "0.7"  # light gray


def _title(df, metrics, controller):
    """e.g. 'PID  ·  motor 3 -21% @ 3.16 s  ·  final 0.30 m from B  ·  no crash'"""
    parts = [controller.upper()] if controller else []
    if math.isnan(metrics["fault_onset_t"]):
        parts.append("no fault")
    else:
        parts.append(f"motor {metrics['fault_motor']} -{metrics['fault_loss'] * 100:.0f}% thrust "
                     f"@ {metrics['fault_onset_t']:.2f} s")
    if metrics["crashed"]:
        parts.append(f"CRASHED @ {metrics['crash_t']:.1f} s")
    else:
        parts.append(f"final {metrics['final_dist_to_B']:.2f} m from B")
        parts.append("no crash")
    return "  ·  ".join(parts)


def make_plots(df, start, goal, out_path, controller=None):
    """Save the 3-panel figure to out_path and return the path."""
    metrics = compute_metrics(df, start, goal)
    onset = fault_onset_index(df)
    t_fault = df["t"].iloc[onset] if onset is not None else None
    t_crash = metrics["crash_t"] if metrics["crashed"] else None
    metrics_t_end = float(df["t"].iloc[-1])  # keep the full flight on the x-axis so a cut-off line is visible
    if t_crash is not None:
        df = df[df["t"] <= t_crash]  # plot only up to the crash
    t = df["t"].to_numpy()
    z = df[POS[2]].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    fig.suptitle(_title(df, metrics, controller), fontsize=11, fontweight="bold")

    # 1. distance to B
    ax = axes[0]
    ax.axhspan(0, ARRIVE_TOL, color=DRONE_COLOR, alpha=0.08, lw=0, label=f"arrived (within {ARRIVE_TOL:.2f} m)")
    ax.plot(t, distance_to(df[POS].to_numpy(), goal), lw=1.8, color=DRONE_COLOR, label="drone")
    ax.set_ylabel("distance to B [m]")
    ax.set_title("Distance to B", loc="left", fontsize=10)

    # 2. altitude
    ax = axes[1]
    ax.axhspan(0, CRASH_MIN_Z, color=FAULT_COLOR, alpha=0.08, lw=0, label=f"crash zone (below {CRASH_MIN_Z:.2f} m)")
    if "setpoint_z" in df:
        ax.plot(t, df["setpoint_z"], "--", lw=1.2, color=TARGET_COLOR, label="target altitude")
    ax.plot(t, z, lw=1.8, color=DRONE_COLOR, label="drone")
    ax.set_ylim(min(0.0, z.min() - 0.05), max(z.max(), df["setpoint_z"].max() if "setpoint_z" in df else 0) + 0.15)
    ax.set_ylabel("altitude [m]")
    ax.set_title("Altitude", loc="left", fontsize=10)

    # 3. motor thrust
    ax = axes[2]
    has_applied = all(f"thrust_applied_{i}" in df for i in range(4))
    faulty = int(df["fault_motor"].iloc[onset]) if onset is not None and "fault_motor" in df else None
    first_other = True
    for i in range(4):
        if i == faulty:
            continue
        col = f"thrust_applied_{i}" if has_applied else f"thrust_cmd_{i}"
        label = ("other motors" if faulty is not None else "all 4 motors") if first_other else None
        ax.plot(t, df[col], lw=1.0, color=OTHER_MOTOR_COLOR, label=label)
        first_other = False
    if faulty is not None:
        ax.plot(t, df[f"thrust_cmd_{faulty}"], "--", lw=1.6, color=FAULTY_MOTOR_COLOR,
                label=f"motor {faulty}: asked by controller")
        if has_applied:
            ax.plot(t, df[f"thrust_applied_{faulty}"], lw=1.8, color=FAULTY_MOTOR_COLOR,
                    label=f"motor {faulty}: actually delivered")
        ax.set_title("Motor thrust (faulty motor in colour)", loc="left", fontsize=10)
    else:
        ax.set_title("Motor thrust (no fault: all motors)", loc="left", fontsize=10)
    ax.set_ylabel("thrust [N]")
    ax.set_xlabel("time [s]")

    for i, ax in enumerate(axes):
        if t_fault is not None:
            ax.axvline(t_fault, color=FAULT_COLOR, ls="--", lw=1.3, label="fault onset" if i == 0 else None)
        if t_crash is not None:
            ax.axvline(t_crash, color=FAULT_COLOR, lw=1.8, label="crash (readings stop)" if i == 0 else None)
        ax.set_xlim(right=metrics_t_end)
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Flight CSV.")
    parser.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Point A [m].")
    parser.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Point B [m].")
    parser.add_argument("--out", default=None, help="Output PNG (default: next to the CSV, same stem).")
    args = parser.parse_args()

    start, goal = endpoints_for(args.csv, args.start, args.goal)
    summary = pathlib.Path(args.csv).with_suffix(".json")
    controller = json.loads(summary.read_text()).get("controller") if summary.exists() else None
    out = args.out or pathlib.Path(args.csv).with_suffix(".png")
    print(f"[RESULT] plot -> {make_plots(pd.read_csv(args.csv), start, goal, out, controller=controller)}")


if __name__ == "__main__":
    main()
