"""Step 3: the same A->B flight as 02, logged, plotted and scored.

Every physics step goes to data/<name>.csv.

Run (from the drone folder):
    # PID, no fault, opens the viewer
    ./run.sh 03_collect_data.py --name pid_healthy
    ./run.sh 03_collect_data.py --headless --fault random --fault_seed 1 --name pid_s1
    # Isaac Lab's Lee controller
    ./run.sh 03_collect_data.py --headless --controller lee --name check_lee

Output in data/:
    <name>.csv: one row per physics step (columns = common.STATE_COLUMNS)
    <name>.png: distance to B, altitude and motor thrust over time, fault marked
    <name>.json: the run's settings, fault and metrics
Plots and metrics can be redone later without Isaac:
    python analysis/plots.py data/<name>.csv
"""

import json
import pathlib

import common

parser = common.build_parser(__doc__)
parser.add_argument(
    "--name", default="baseline_pid", help="Output file stem inside data/."
)
args = common.parse_args(__doc__, parser)

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"

# Refuse to overwrite an earlier run (checked before Isaac starts, so it costs nothing).
existing = [
    p.name
    for p in (DATA_DIR / f"{args.name}.{ext}" for ext in ("csv", "png", "json"))
    if p.exists()
]
if existing and not args.overwrite:
    raise SystemExit(
        f"[ERROR] data/ already has {', '.join(existing)}. "
        "Pick a new --name, or add --overwrite."
    )


def main():
    from isaaclab.app import launch_simulation
    from isaaclab.physics import PhysicsCfg

    # NOTE: launch_simulation() ends the whole process when its `with` block exits, so
    # everything (including saving the CSV) must happen inside the block.
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        world = common.make_world(args, physics_cfg)
        rows = list(common.fly_a_to_b(args, world))
        save(rows, world.fault, world.feasibility, common.make_receipt(args, world))


def save(rows, fault, feasibility, receipt):
    import pandas as pd
    from analysis.metrics import compute_metrics
    from analysis.plots import make_plots

    df = pd.DataFrame(rows, columns=common.STATE_COLUMNS)
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / f"{args.name}.csv"
    df.to_csv(csv_path, index=False)

    make_plots(
        df,
        args.start,
        args.goal,
        DATA_DIR / f"{args.name}.png",
        controller=args.controller,
        fault_feasible=feasibility["feasible"],
    )
    metrics = compute_metrics(df, args.start, args.goal)

    summary = {
        "name": args.name,
        "controller": args.controller,
        "start": args.start,
        "goal": args.goal,
        "speed": args.speed,
        "hold": args.hold,
        "max_motor_thrust_n": args.max_thrust,
        "fault_mode": args.fault,
        "fault_seed": args.fault_seed,
        "fault": fault.to_dict() if fault is not None else None,
        # could the damaged drone hover? if not, a crash is not the controller's fault
        "feasibility": feasibility,
        "metrics": metrics,
        # versions, code hashes, gains, settings: exactly how this run was produced
        "receipt": receipt,
    }
    with open(DATA_DIR / f"{args.name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[RESULT] {len(df)} rows -> {csv_path}")
    print(
        f"[RESULT] controller={args.controller}  fault={summary['fault']}  "
        f"physically survivable={feasibility['feasible']}"
    )
    for key, value in metrics.items():
        print(f"[RESULT]   {key:26s} {value}")


if __name__ == "__main__":
    main()
