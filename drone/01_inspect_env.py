"""Step 1: spawn the ARL Robot 1 drone and print what we can measure and command.

Nothing flies.

Run (from your IsaacLab folder):
    uv run --extra isaacsim python "<this folder>/01_inspect_env.py" --headless
"""

import common

args = common.parse_args(__doc__)


def main():
    from isaaclab.app import launch_simulation
    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        _, robot, controller, _ = common.make_world(args, physics_cfg)
        d = robot.data

        print("\n===== DRONE INSPECTION =====")
        print("thrusters (action order):", d.thruster_names)
        print("number of thrusters     :", robot.num_thrusters)
        print("total mass [kg]         :", controller.mass.flatten().tolist())
        print("allocation matrix (rows Fx,Fy,Fz,Tx,Ty,Tz; cols = thrusters):")
        print(robot.allocation_matrix)
        thr = robot.cfg.actuators["thrusters"]
        print("thrust range [N]        :", thr.thrust_range)
        print("thrust const range      :", thr.thrust_const_range)
        print("spin-up/down time const :", thr.tau_inc_range, thr.tau_dec_range)
        print("\nstate available at spawn (pos A):")
        for name in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_b"):
            print(f"  {name:16s}", common._t(getattr(d, name))[0].tolist())
        print("============================\n")


if __name__ == "__main__":
    main()
