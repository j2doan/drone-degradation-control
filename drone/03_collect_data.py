"""Step 3: same A->B flight as 02, but log every physics step to a CSV and plot it.

Run (from your IsaacLab folder; add --headless for no window):
    uv run --extra isaacsim python "<this folder>/03_collect_data.py" --name run1

Output: data/<name>.csv (one row per step, columns = common.STATE_COLUMNS) and
data/<name>.png. Load it with pandas.read_csv("data/<name>.csv").
"""

import pathlib

import common
import numpy as np

parser = common.build_parser(__doc__)
parser.add_argument(
    "--name", default="baseline_lee", help="Output file stem inside data/."
)
args = common.parse_args(__doc__, parser)


DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


def main():
    from isaaclab.app import launch_simulation
    from isaaclab.physics import PhysicsCfg

    # NOTE: launch_simulation() ends the whole process when its `with` block exits, so
    # everything (including saving the CSV) must happen inside the block.
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        sim, robot, controller, alloc_pinv = common.make_world(args, physics_cfg)
        rows = list(common.fly_a_to_b(args, sim, robot, controller, alloc_pinv))
        save(rows)


def save(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows, columns=common.STATE_COLUMNS)
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / f"{args.name}.csv"
    df.to_csv(csv_path, index=False)

    goal = np.asarray(args.goal)
    dist = np.linalg.norm(df[["pos_x", "pos_y", "pos_z"]].to_numpy() - goal, axis=1)
    print(f"[RESULT] {len(df)} rows -> {csv_path}")
    speed = np.linalg.norm(
        df[["lin_vel_x", "lin_vel_y", "lin_vel_z"]].to_numpy(), axis=1
    )
    print(
        f"[RESULT] final distance to B: {dist[-1]:.3f} m "
        f"| min height: {df['pos_z'].min():.2f} m | max speed: {speed.max():.2f} m/s"
    )

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for a in "xyz":
        ax[0].plot(df["t"], df[f"pos_{a}"], label=f"pos_{a}")
        ax[0].plot(df["t"], df[f"setpoint_{a}"], "--", alpha=0.5)
    ax[0].set_ylabel("position [m] (dashed = setpoint)")
    ax[0].legend()
    ax[1].plot(df["t"], dist)
    ax[1].set_ylabel("distance to B [m]")
    for i in range(4):
        ax[2].plot(df["t"], df[f"thrust_cmd_{i}"], label=f"prop {i}")
    ax[2].set_ylabel("thrust cmd [N]")
    ax[2].set_xlabel("time [s]")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(DATA_DIR / f"{args.name}.png", dpi=120)


if __name__ == "__main__":
    main()
