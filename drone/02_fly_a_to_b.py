"""Step 2: fly the drone in a straight line from A to B, then hover.

The Lee controller flies it (no RL).

Run (from your IsaacLab folder):
    # opens the viewer
    uv run --extra isaacsim python "<this folder>/02_fly_a_to_b.py"
    # no window
    uv run --extra isaacsim python "<this folder>/02_fly_a_to_b.py" --headless
    ... --start 0 0 1 --goal 3 2 1.5 --speed 0.5
"""

import common
import numpy as np

args = common.parse_args(__doc__)


def main():
    from isaaclab.app import launch_simulation
    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        sim, robot, controller, alloc_pinv = common.make_world(args, physics_cfg)
        goal = np.asarray(args.goal)
        print(
            f"[INFO] Flying A={args.start} -> B={args.goal} at {args.speed} m/s, "
            f"then hovering {args.hold}s"
        )

        last = None
        for row in common.fly_a_to_b(args, sim, robot, controller, alloc_pinv):
            last = row
            if row["step"] % int(1 / common.SIM_DT) == 0:  # once per simulated second
                pos = np.array([row["pos_x"], row["pos_y"], row["pos_z"]])
                print(
                    f"t={row['t']:5.1f}s  pos={np.round(pos, 2)}  "
                    f"dist_to_B={np.linalg.norm(goal - pos):.2f} m"
                )

        if last is not None:
            pos = np.array([last["pos_x"], last["pos_y"], last["pos_z"]])
            print(f"[RESULT] final distance to B: {np.linalg.norm(goal - pos):.3f} m")


if __name__ == "__main__":
    main()
