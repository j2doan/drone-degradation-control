"""Step 1: spawn the ARL Robot 1 drone and print what we can measure and command.

Nothing flies.

Run (from the drone folder):
    ./run.sh 01_inspect_env.py --headless
    # the MuJoCo copy of the same drone
    ./run.sh 01_inspect_env.py --headless --sim mujoco
"""

import common

args = common.parse_args(__doc__)


def main():
    with common.session(args) as physics_cfg:
        world = common.make_world(args, physics_cfg)

        print("\n===== DRONE INSPECTION =====")
        print("simulator               :", args.sim)
        print("total mass [kg]         :", world.mass)
        print("inertia [kg m^2]        :", world.inertia)
        print("flying controller       :", args.controller)
        print("allocation matrix (rows Fx,Fy,Fz,Tx,Ty,Tz; cols = thrusters):")
        for row in world.allocation:
            print("   ", [float(v) for v in row])
        if args.sim == "isaac":
            robot, d = world.robot, world.robot.data
            print("thrusters (action order):", d.thruster_names)
            thr = robot.cfg.actuators["thrusters"]
            print("thrust range [N]        :", thr.thrust_range)
            print("thrust const range      :", thr.thrust_const_range)
            print("spin-up/down time const :", thr.tau_inc_range, thr.tau_dec_range)
            print("\nstate available at spawn (pos A):")
            for name in (
                "root_pos_w",
                "root_quat_w",
                "root_lin_vel_w",
                "root_ang_vel_b",
            ):
                print(
                    f"  {name:16s}",
                    common.backend(args)._t(getattr(d, name))[0].tolist(),
                )
        else:
            from motors import thruster

            print("thrust range [N]        :", thruster.THRUST_RANGE)
            print("thrust const range      :", thruster.THRUST_CONST_RANGE)
            print(
                "spin-up/down time const :",
                thruster.TAU_INC_RANGE,
                thruster.TAU_DEC_RANGE,
            )
            print("timestep [s]            :", world.model.opt.timestep)
        print("============================\n")


if __name__ == "__main__":
    main()
