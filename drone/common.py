"""Shared helpers for the drone scripts.

Pattern copied from IsaacLab/examples/arl_robot_1.py: one ARL Robot 1 quadrotor, flown
by the Lee position controller (a classical, non-learned controller), stepped by hand
with no RL environment.

IMPORTANT: Isaac Sim must be launched before most isaaclab modules can be imported, so
every import other than argparse/isaaclab.app happens inside a function.
"""

import argparse
import copy

from isaaclab.app import add_launcher_args

SIM_DT = 0.01  # physics step, seconds (same as the RL env)

# One row of the log = one physics step. Order matters: it is the CSV column order.
STATE_COLUMNS = (
    ["step", "t"]
    + [f"pos_{a}" for a in "xyz"]  # world position [m]
    + [f"quat_{a}" for a in "xyzw"]  # orientation (x, y, z, w)
    + [f"lin_vel_{a}" for a in "xyz"]  # world-frame linear velocity [m/s]
    + [f"ang_vel_{a}" for a in "xyz"]  # body-frame angular velocity [rad/s]
    + [f"setpoint_{a}" for a in "xyz"]  # where the controller was told to go
    + [f"thrust_cmd_{i}" for i in range(4)]  # commanded thrust per propeller
)


def build_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser: the flight options plus Isaac Lab's launcher flags."""
    parser = argparse.ArgumentParser(
        description=description, conflict_handler="resolve"
    )
    parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Point A in meters.",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=3,
        default=[3.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Point B in meters.",
    )
    parser.add_argument(
        "--speed", type=float, default=0.5, help="Speed of the moving setpoint, m/s."
    )
    parser.add_argument(
        "--hold", type=float, default=3.0, help="Seconds to hover at B after arriving."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Hard stop after this many steps (-1: no limit).",
    )
    parser.add_argument(
        "--physics",
        default="isaacsim_physx",
        choices=["isaacsim_physx"],
        help="Physics backend.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No window (faster). Default opens the Isaac Sim viewer.",
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    return parser


def parse_args(
    description: str, parser: argparse.ArgumentParser | None = None
) -> argparse.Namespace:
    args = (parser or build_parser(description)).parse_args()
    if args.headless:
        args.visualizer = []  # no visualizer = headless
    return args


def make_world(args, physics_cfg):
    """Create sim, ground, light, robot at A and the Lee controller.

    Call inside launch_simulation().
    """
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.utils import instantiate, replace
    from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG
    from isaaclab_contrib.controllers.lee_position_control import LeePosController
    from isaaclab_contrib.controllers.lee_position_control_cfg import (
        LeePosControllerCfg,
    )

    sim_cfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        device=args.device,
        physics=physics_cfg,
        use_newton_actuators=False,  # the thruster model only runs on PhysX
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.53, 0.81, 0.92))
    light_cfg.func("/World/DomeLight", light_cfg)
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    robot_cfg = replace(ARL_ROBOT_1_CFG, prim_path="/World/Robot")
    robot_cfg.init_state = copy.deepcopy(robot_cfg.init_state)
    robot_cfg.init_state.pos = tuple(args.start)  # spawn at A
    robot_cfg.actuators["thrusters"].dt = sim_cfg.dt
    robot = instantiate(robot_cfg)

    sim.reset()

    controller = LeePosController(
        LeePosControllerCfg(
            K_pos_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_vel_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_rot_range=((1.6, 1.6, 0.25), (1.85, 1.85, 0.4)),
            K_angvel_range=((0.4, 0.4, 0.075), (0.5, 0.5, 0.09)),
            max_inclination_angle_rad=1.0471975511965976,
            max_yaw_rate=1.0471975511965976,
        ),
        robot,
        num_envs=1,
        device=str(sim.device),
    )
    # The controller outputs a 6-D wrench [Fx,Fy,Fz,Tx,Ty,Tz]; the robot takes 4
    # propeller thrusts. The allocation matrix maps thrusts -> wrench, so its
    # pseudoinverse maps wrench -> thrusts.
    alloc = torch.tensor(
        robot_cfg.allocation_matrix, device=sim.device, dtype=torch.float32
    )
    alloc_pinv = torch.linalg.pinv(alloc)  # (4, 6)
    return sim, robot, controller, alloc_pinv


def _t(x):
    """Warp array or torch tensor -> torch tensor."""
    import torch
    import warp as wp

    return x if torch.is_tensor(x) else wp.to_torch(x)


def setpoint_at(t: float, start, goal, speed: float):
    """Point on the line A->B reached at time t by a 'carrot' moving at `speed`."""
    import numpy as np

    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    dist = float(np.linalg.norm(goal - start))
    frac = 1.0 if dist == 0.0 else min(t * speed / dist, 1.0)
    return start + frac * (goal - start)


class PathDrawer:
    """Draws the ideal and the flown path as lines in the Isaac Sim viewport.

    Green = the ideal straight path A->B, red = the path the drone has flown so far.
    Uses Isaac Sim's debug-draw extension; does nothing if unavailable.
    """

    GREEN, RED = (0.1, 1.0, 0.1, 1.0), (1.0, 0.1, 0.1, 1.0)

    def __init__(self, start, goal):
        self.start, self.goal, self.trail, self.draw = list(start), list(goal), [], None
        try:
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_manager.set_extension_enabled_immediate(
                "isaacsim.util.debug_draw", True
            )
            from isaacsim.util.debug_draw import _debug_draw

            self.draw = _debug_draw.acquire_debug_draw_interface()
        # drawing is cosmetic: never let it break a flight
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] path drawing unavailable: {e}")

    def update(self, pos):
        """Add the drone's current position to the red trail and redraw both lines."""
        if self.draw is None:
            return
        self.trail.append([float(p) for p in pos])
        self.draw.clear_lines()
        # true path A->B
        self.draw.draw_lines([self.start], [self.goal], [self.GREEN], [4.0])
        if len(self.trail) > 1:  # actual path: consecutive trail points joined
            n = len(self.trail) - 1
            self.draw.draw_lines(
                self.trail[:-1], self.trail[1:], [self.RED] * n, [4.0] * n
            )


def fly_a_to_b(args, sim, robot, controller, alloc_pinv):
    """Step the sim along the straight line A->B, then hover.

    Yields one dict per step (keys = STATE_COLUMNS).
    """
    import numpy as np
    import torch

    dist = float(np.linalg.norm(np.asarray(args.goal) - np.asarray(args.start)))
    total_steps = int((dist / args.speed + args.hold) / SIM_DT)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    # no lines when headless
    drawer = PathDrawer(args.start, args.goal) if args.visualizer else None

    for step in range(total_steps):
        if not sim.is_running():  # user closed the window
            return
        t = step * SIM_DT
        sp = setpoint_at(t, args.start, args.goal, args.speed)
        command = torch.zeros((1, 4), device=sim.device)  # [x, y, z, yaw]
        command[0, :3] = torch.tensor(sp, device=sim.device)

        wrench = controller.compute(command)
        # propellers can't pull down
        thrust_cmd = (wrench @ alloc_pinv.T).clamp(min=0.0)
        robot.set_thrust_target(thrust_cmd)

        # log the state the controller saw, before stepping
        row = {"step": step, "t": t}
        for name, vec in (
            ("pos", _t(robot.data.root_pos_w)),
            ("quat", _t(robot.data.root_quat_w)),
            ("lin_vel", _t(robot.data.root_lin_vel_w)),
            ("ang_vel", _t(robot.data.root_ang_vel_b)),
        ):
            labels = "xyzw" if name == "quat" else "xyz"
            for k, a in enumerate(labels):
                row[f"{name}_{a}"] = float(vec[0, k])
        for k, a in enumerate("xyz"):
            row[f"setpoint_{a}"] = float(sp[k])
        for i in range(4):
            row[f"thrust_cmd_{i}"] = float(thrust_cmd[0, i])
        yield row

        if (
            drawer is not None and step % 5 == 0
        ):  # redraw every 5 steps (20 Hz) to keep the viewer smooth
            drawer.update((row["pos_x"], row["pos_y"], row["pos_z"]))

        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
