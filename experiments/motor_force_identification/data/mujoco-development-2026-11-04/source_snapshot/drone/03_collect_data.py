"""Step 3: same A->B flight as 02, but log every physics step to data/<name>.csv, plot it and score it.

Run (from the drone folder):
    ./run.sh 03_collect_data.py --name pid_healthy                              # PID, no fault, opens the viewer
    ./run.sh 03_collect_data.py --headless --fault random --fault_seed 1 --name pid_s1
    ./run.sh 03_collect_data.py --headless --controller lee --name check_lee    # Isaac Lab's Lee controller

Output in data/:
    <name>.csv   one row per physics step (columns = common.STATE_COLUMNS)
    <name>.png   distance to B, altitude and motor thrust over time, with the fault marked
    <name>.json  the run's settings, fault and metrics
Plots and metrics can be redone later without Isaac: python analysis/plots.py data/<name>.csv
"""

import json
import pathlib

import common

parser = common.build_parser(__doc__)
parser.add_argument("--name", default="baseline_pid", help="Output file stem inside data/.")
args = common.parse_args(__doc__, parser)

from isaaclab.app import launch_simulation
from isaaclab.physics import PhysicsCfg

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


def main():
    # NOTE: launch_simulation() ends the whole process when its `with` block exits, so everything
    # (including saving the CSV) must happen inside the block.
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        world = common.make_world(args, physics_cfg)
        rows = list(common.fly_a_to_b(args, world))
        save(rows, world.fault)


def save(rows, fault):
    import pandas as pd

    from analysis.metrics import compute_metrics
    from analysis.plots import make_plots

    df = pd.DataFrame(rows, columns=common.STATE_COLUMNS)
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / f"{args.name}.csv"
    df.to_csv(csv_path, index=False)

    make_plots(df, args.start, args.goal, DATA_DIR / f"{args.name}.png", controller=args.controller)
    metrics = compute_metrics(df, args.start, args.goal)

    summary = {
        "name": args.name, "controller": args.controller, "start": args.start, "goal": args.goal,
        "speed": args.speed, "hold": args.hold, "fault_mode": args.fault, "fault_seed": args.fault_seed,
        "fault": fault.to_dict() if fault is not None else None, "metrics": metrics,
    }
    with open(DATA_DIR / f"{args.name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[RESULT] {len(df)} rows -> {csv_path}")
    print(f"[RESULT] controller={args.controller}  fault={summary['fault']}")
    for key, value in metrics.items():
        print(f"[RESULT]   {key:26s} {value}")


if __name__ == "__main__":
    main()
