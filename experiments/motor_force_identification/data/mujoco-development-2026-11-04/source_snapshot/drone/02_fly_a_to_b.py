"""Step 2: fly the drone in a straight line from A to B with the Lee controller (no RL), then hover.

Run (from your IsaacLab folder):
    uv run --extra isaacsim python "<this folder>/02_fly_a_to_b.py"                    # opens the viewer
    uv run --extra isaacsim python "<this folder>/02_fly_a_to_b.py" --headless         # no window
    ... --start 0 0 1 --goal 3 2 1.5 --speed 0.5
"""

import numpy as np

import common

args = common.parse_args(__doc__)

from isaaclab.app import launch_simulation
from isaaclab.physics import PhysicsCfg


def main():
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        world = common.make_world(args, physics_cfg)
        goal = np.asarray(args.goal)
        print(f"[INFO] Flying A={args.start} -> B={args.goal} at {args.speed} m/s, then hovering {args.hold}s")

        print(f"[INFO] controller={args.controller}  fault={world.fault}")

        last = None
        for row in common.fly_a_to_b(args, world):
            if row["fault_active"] and (last is None or not last["fault_active"]):
                print(f"[FAULT] t={row['t']:.2f}s  motor {row['fault_motor']} lost {row['fault_loss']:.0%} of its thrust")
            last = row
            if row["step"] % int(1 / common.SIM_DT) == 0:  # once per simulated second
                pos = np.array([row["pos_x"], row["pos_y"], row["pos_z"]])
                print(f"t={row['t']:5.1f}s  pos={np.round(pos, 2)}  dist_to_B={np.linalg.norm(goal - pos):.2f} m")

        if last is not None:
            pos = np.array([last["pos_x"], last["pos_y"], last["pos_z"]])
            print(f"[RESULT] final distance to B: {np.linalg.norm(goal - pos):.3f} m")


if __name__ == "__main__":
    main()
