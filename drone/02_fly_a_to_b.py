"""Step 2: fly one seed's curved path from A to B and print progress.

The seed also picks the fault, if any. Saves nothing.

Run (from the drone folder):
    # seed 0 = healthy flight, opens the viewer
    ./run.sh 02_fly_a_to_b.py --seed 0
    # seed 3's path and fault, no window
    ./run.sh 02_fly_a_to_b.py --seed 3 --headless
    # the same flight in MuJoCo
    ./run.sh 02_fly_a_to_b.py --seed 3 --sim mujoco
Use 03_collect_data.py to save data.
"""

import common
import numpy as np

parser = common.build_parser(__doc__)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Which seed's path (and fault, for seeds >= 1) to fly.",
)
args = common.parse_args(__doc__, parser)


def main():
    from drones import DRONES

    with common.session(args) as physics_cfg:
        world = common.make_world(args, physics_cfg)
        flight = common.prepare_flight(args, world, args.seed, DRONES["d0"])
        goal = np.asarray(flight.path.goal)
        print(
            f"[INFO] seed {args.seed}: path {flight.path.length:.1f} m "
            f"from {np.round(flight.path.start, 2)} "
            f"to {np.round(goal, 2)} at {args.speed} m/s; fault={flight.fault}; "
            f"physically survivable={flight.feasibility['feasible']}"
        )

        last = None
        for row in common.fly(args, world, flight):
            if row["fault_active"] and (last is None or not last["fault_active"]):
                print(
                    f"[FAULT] t={row['t']:.2f}s  motor {row['fault_motor']} lost "
                    f"{row['fault_loss']:.0%} of its thrust"
                )
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
