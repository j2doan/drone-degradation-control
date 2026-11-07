"""Step 3: fly a batch of seeded flights and save them into one folder per run name.

Run (from the drone folder):
    # seed 0 (healthy) + seeds 1-5 (random faults)
    ./run.sh 03_collect_data.py --name run1 --runs 5 --headless
    # only the healthy flight, with the viewer
    ./run.sh 03_collect_data.py --name test --runs 0
    # only seeds 3 and 7
    ./run.sh 03_collect_data.py --name s37 --seeds 3 7 --headless
    # also record the front camera (slower)
    ./run.sh 03_collect_data.py --name cam --runs 2 --camera --headless
    # the same batch in MuJoCo (no Isaac)
    ./run.sh 03_collect_data.py --sim mujoco --name mj1 --runs 5 --headless

Each seed has its own random curved path; seeds >= 1 also get a random fault (seed 0 is
the healthy reference). Every available drone (drones.py; today only d0 = PID) flies
every seed. Output in data/<name>/:
    manifest.json: receipt (simulator, GPU, versions, gains, settings, code hashes)
    summary.csv: one row per seed x drone (path, fault, feasibility, metrics)
    seed<k>/<drone>/flight.csv: one row per physics step (columns = STATE_COLUMNS)
    seed<k>/<drone>/flight.png: distance to B, altitude and motor thrust over time
    seed<k>/<drone>/flight.json: path, fault, feasibility, metrics
    (+ camera.npz and camera_preview.png with --camera)
An existing data/<name>/ is never overwritten unless --overwrite is given. Plots and
metrics can be redone later without a simulator:
    python analysis/plots.py data/<name>/seed1/d0/flight.csv
"""

import json
import pathlib
import shutil

import common

parser = common.build_parser(__doc__)
parser.add_argument(
    "--name", required=True, help="Run name: everything goes into data/<name>/."
)
parser.add_argument(
    "--runs",
    type=int,
    default=5,
    help="Number of faulted seeds (1..N) besides the healthy seed 0.",
)
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=None,
    help="Fly exactly these seeds instead of 0..runs.",
)
parser.add_argument(
    "--overwrite", action="store_true", help="Replace an existing data/<name>/."
)
args = common.parse_args(__doc__, parser)

RUN_DIR = pathlib.Path(__file__).resolve().parent / "data" / args.name
# no duplicates
SEEDS = (
    list(dict.fromkeys(args.seeds))
    if args.seeds is not None
    else list(range(args.runs + 1))
)
if args.runs < 0 or any(s < 0 for s in SEEDS):
    parser.error("--runs and --seeds must be >= 0")

# Refuse to overwrite an earlier run (checked before the simulator starts).
if RUN_DIR.exists():
    if not args.overwrite:
        raise SystemExit(
            f"[ERROR] {RUN_DIR} already exists. Pick a new --name, or add --overwrite."
        )
    shutil.rmtree(RUN_DIR)


def main():
    from drones import available

    # NOTE: with Isaac the whole process ends when the `with` block exits, so everything
    # (including saving) must happen inside the block.
    with common.session(args) as physics_cfg:
        world = common.make_world(args, physics_cfg)
        RUN_DIR.mkdir(parents=True)
        (RUN_DIR / "manifest.json").write_text(
            json.dumps(common.make_receipt(args), indent=2)
        )
        summary = []
        for seed in SEEDS:
            for drone in available():
                flight = common.prepare_flight(args, world, seed, drone)
                print(
                    f"[FLIGHT] seed {seed} {drone.name}: "
                    f"path {flight.path.length:.1f} m, "
                    f"fault {flight.fault.to_dict() if flight.fault else None}, "
                    f"{flight.duration:.1f} s",
                    flush=True,
                )
                rows = list(common.fly(args, world, flight))
                summary.append(
                    save_flight(RUN_DIR / f"seed{seed}" / drone.name, rows, flight)
                )
                # rewritten after every flight: a batch cut short keeps its rows
                save_summary(summary, final=False)
        save_summary(summary)


def save_flight(folder: pathlib.Path, rows, flight) -> dict:
    """Write flight.csv/.png/.json (+ camera files); return the summary row."""
    import numpy as np
    import pandas as pd
    from analysis.metrics import compute_metrics
    from analysis.plots import make_plots

    folder.mkdir(parents=True)
    df = pd.DataFrame(rows, columns=common.STATE_COLUMNS)
    df.to_csv(folder / "flight.csv", index=False)
    goal = [float(v) for v in flight.path.goal]
    start = [float(v) for v in flight.path.start]
    controller = args.controller if flight.drone.mode == "pid" else flight.drone.label
    make_plots(
        df,
        start,
        goal,
        folder / "flight.png",
        controller=controller,
        fault_feasible=flight.feasibility["feasible"],
    )
    metrics = compute_metrics(df, start, goal)
    info = {
        "seed": flight.seed,
        "drone": flight.drone.name,
        "drone_label": flight.drone.label,
        "controller": args.controller,
        "start": start,
        "goal": goal,
        "path": flight.path.to_dict(),
        "speed": args.speed,
        "hold": args.hold,
        "max_motor_thrust_n": args.max_thrust,
        "fault": flight.fault.to_dict() if flight.fault is not None else None,
        # could the damaged drone hover? if not, a crash is not the controller's fault
        "feasibility": flight.feasibility,
        "metrics": metrics,
    }
    if flight.frames:
        frames = np.stack(flight.frames).astype(np.uint8)
        np.savez_compressed(folder / "camera.npz", frames=frames)
        save_camera_preview(frames, folder / "camera_preview.png")
        info["camera"] = {
            "frames": len(frames),
            "size": list(frames.shape[1:3]),
            "rate_hz": 1 / common.CAMERA_PERIOD,
        }
    (folder / "flight.json").write_text(json.dumps(info, indent=2))

    print(
        f"[RESULT] seed {flight.seed} {flight.drone.name}: "
        + (
            f"CRASHED at {metrics['crash_t']:.2f} s"
            if metrics["crashed"]
            else f"final {metrics['final_dist_to_B']:.3f} m from B"
        )
        + f" | survivable={flight.feasibility['feasible']}",
        flush=True,
    )
    fault = flight.fault
    return {
        "seed": flight.seed,
        "drone": flight.drone.name,
        "route_length_m": round(flight.path.length, 3),
        "fault_motor": fault.motor if fault else -1,
        "fault_loss": round(fault.loss, 4) if fault else 0.0,
        "fault_onset_s": round(fault.onset_t, 3) if fault else None,
        "fault_feasible": flight.feasibility["feasible"],
        **metrics,
    }


def save_camera_preview(frames, path, count: int = 8):
    """A strip of `count` evenly spaced camera frames, to check the camera quickly."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    picks = np.linspace(0, len(frames) - 1, min(count, len(frames))).astype(int)
    fig, axes = plt.subplots(
        1, len(picks), figsize=(1.6 * len(picks), 2.0), squeeze=False
    )
    for ax, k in zip(axes[0], picks):
        ax.imshow(frames[k])
        ax.set_title(f"t={k * common.CAMERA_PERIOD:.1f}s", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_summary(summary, final: bool = True):
    import pandas as pd

    df = pd.DataFrame(summary)
    df.to_csv(RUN_DIR / "summary.csv", index=False)
    if not final:
        return
    faulted = df[df["seed"] > 0]
    print(f"[RESULT] {len(df)} flights -> {RUN_DIR}")
    if len(faulted):
        print(
            f"[RESULT] faulted flights: "
            f"{int(faulted['crashed'].sum())}/{len(faulted)} crashed "
            f"({int((~faulted['fault_feasible']).sum())} physically unsurvivable)"
        )


if __name__ == "__main__":
    main()
