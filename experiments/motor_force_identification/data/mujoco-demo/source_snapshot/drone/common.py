"""Shared helpers for the drone scripts. This is the only file that talks to Isaac Sim.

Pattern copied from IsaacLab/examples/arl_robot_1.py: one ARL Robot 1 quadrotor stepped by hand with no RL
environment, flown by a classical (non-learned) controller: our PID (controllers/pid.py, default) or Isaac Lab's
Lee position controller (--controller lee). Optional thrust-loss fault from faults/thrust_loss.py (--fault).
The simulator-independent parts (controllers/, faults/, analysis/) never import Isaac.

IMPORTANT: Isaac Sim must be launched before most isaaclab modules can be imported, so every import
other than argparse/isaaclab.app happens inside a function.
"""

import argparse
import copy
from types import SimpleNamespace

SIM_DT = 0.01  # physics step, seconds (same as the RL env)

# One row of the log = one physics step. Order matters: it is the CSV column order.
STATE_COLUMNS = (
    ["step", "t"]
    + [f"pos_{a}" for a in "xyz"]  # world position [m]
    + [f"quat_{a}" for a in "xyzw"]  # orientation (x, y, z, w)
    + [f"lin_vel_{a}" for a in "xyz"]  # world-frame linear velocity [m/s]
    + [f"ang_vel_{a}" for a in "xyz"]  # body-frame angular velocity [rad/s]
    + [f"setpoint_{a}" for a in "xyz"]  # where the controller was told to go
    + [f"thrust_cmd_{i}" for i in range(4)]  # thrust the controller asked for, per propeller [N]
    + [f"thrust_applied_{i}" for i in range(4)]  # thrust actually sent after the fault [N]
    + ["fault_active", "fault_motor", "fault_loss"]  # 0/1, motor index (-1 = none), lost fraction
    + [f"imu_lin_acc_{a}" for a in "xyz"]  # IMU proper acceleration, body frame (+9.81 on z at rest) [m/s^2]
    + [f"imu_ang_vel_{a}" for a in "xyz"]  # IMU angular velocity, body frame [rad/s]
)


def build_parser(description: str, *, include_launcher: bool = True) -> argparse.ArgumentParser:
    """Argument parser with the flight options plus Isaac Lab's launcher flags (--device, --headless...)."""
    parser = argparse.ArgumentParser(description=description, conflict_handler="resolve")
    parser.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 1.0], metavar=("X", "Y", "Z"),
                        help="Point A in meters.")
    parser.add_argument("--goal", type=float, nargs=3, default=[3.0, 0.0, 1.0], metavar=("X", "Y", "Z"),
                        help="Point B in meters.")
    parser.add_argument("--speed", type=float, default=0.5, help="Speed of the moving setpoint, m/s.")
    parser.add_argument("--hold", type=float, default=3.0, help="Seconds to hover at B after arriving.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Hard stop after this many steps (-1: no limit).")
    parser.add_argument("--controller", default="pid", choices=["pid", "lee"],
                        help="pid = our PID (default); lee = Isaac Lab's Lee controller (regression check).")
    parser.add_argument("--fault", default="none", choices=["none", "random", "fixed"],
                        help="none; random = seeded random thrust loss; fixed = use --fault_motor/_loss/_onset.")
    parser.add_argument("--fault_seed", type=int, default=0, help="Seed for --fault random (same seed = same fault).")
    parser.add_argument("--fault_motor", type=int, default=0, help="Motor index 0-3 for --fault fixed.")
    parser.add_argument("--fault_loss", type=float, default=0.25, help="Lost thrust fraction for --fault fixed.")
    parser.add_argument("--fault_onset", type=float, default=4.0, help="Fault start time [s] for --fault fixed.")
    parser.add_argument("--physics", default="isaacsim_physx", choices=["isaacsim_physx"], help="Physics backend.")
    parser.add_argument("--headless", action="store_true", help="No window (faster). Default opens the Isaac Sim viewer.")
    if include_launcher:
        from isaaclab.app import add_launcher_args
        add_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    return parser


def parse_args(description: str, parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    args = (parser or build_parser(description)).parse_args()
    if args.headless:
        args.visualizer = []  # no visualizer = headless
    return args


def travel_time(args) -> float:
    """Seconds the target point needs to go from A to B at --speed."""
    import numpy as np

    return float(np.linalg.norm(np.asarray(args.goal) - np.asarray(args.start))) / args.speed


def flight_duration(args, fault=None) -> float:
    """Seconds of the whole run: travel A->B, hover for --hold, and with a fault at least 10 s after it starts."""
    from faults.thrust_loss import run_duration

    return run_duration(travel_time(args) + args.hold, fault)


def make_fault(args):
    """The fault for this run (a FaultSpec), or None for --fault none."""
    from faults.thrust_loss import FaultSpec, sample_fault

    if args.fault == "random":
        return sample_fault(args.fault_seed, travel_time(args))  # onset while travelling; --hold does not matter
    if args.fault == "fixed":
        return FaultSpec(motor=args.fault_motor, loss=args.fault_loss, onset_t=args.fault_onset)
    return None


def make_world(args, physics_cfg):
    """Create sim, ground, light, robot at point A, IMU, controller and fault. Call inside launch_simulation().

    Returns a namespace with: sim, robot, imu, controller (the one flying), lee (always built: it also provides the
    drone's mass and inertia), alloc_pinv, fault.
    """
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import ImuCfg
    from isaaclab.utils import instantiate, replace
    from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG
    from isaaclab_contrib.controllers.lee_position_control import LeePosController
    from isaaclab_contrib.controllers.lee_position_control_cfg import LeePosControllerCfg

    from controllers.pid import CascadedPID

    sim_cfg = sim_utils.SimulationCfg(
        dt=SIM_DT, device=args.device, physics=physics_cfg,
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
    # IMU on the drone's main body. Must be created before sim.reset() so it initializes with the robot.
    imu = instantiate(ImuCfg(prim_path="/World/Robot/base_link"))

    sim.reset()

    lee = LeePosController(
        LeePosControllerCfg(
            K_pos_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_vel_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_rot_range=((1.6, 1.6, 0.25), (1.85, 1.85, 0.4)),
            K_angvel_range=((0.4, 0.4, 0.075), (0.5, 0.5, 0.09)),
            max_inclination_angle_rad=1.0471975511965976,
            max_yaw_rate=1.0471975511965976,
        ),
        robot, num_envs=1, device=str(sim.device),
    )
    if args.controller == "pid":
        controller = CascadedPID(num_envs=1, mass=float(lee.mass.flatten()[0]), inertia=lee.robot_inertia,
                                 dt=SIM_DT, device=str(sim.device))
    else:
        controller = lee
    # Both controllers output a 6-D wrench [Fx,Fy,Fz,Tx,Ty,Tz]; the robot takes 4 propeller thrusts.
    # The allocation matrix maps thrusts -> wrench, so its pseudoinverse maps wrench -> thrusts.
    alloc = torch.tensor(robot_cfg.allocation_matrix, device=sim.device, dtype=torch.float32)
    alloc_pinv = torch.linalg.pinv(alloc)  # (4, 6)
    return SimpleNamespace(sim=sim, robot=robot, imu=imu, controller=controller, lee=lee, alloc_pinv=alloc_pinv,
                           fault=make_fault(args))


def _t(x):
    """Warp array, Isaac ProxyArray or torch tensor -> torch tensor."""
    import torch
    import warp as wp

    if torch.is_tensor(x):
        return x
    if hasattr(x, "torch"):  # ProxyArray (e.g. sensor data) has a zero-copy .torch view
        return x.torch
    return wp.to_torch(x)


def setpoint_at(t: float, start, goal, speed: float):
    """Point on the straight line A->B that a 'carrot' moving at `speed` has reached at time t."""
    import numpy as np

    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    dist = float(np.linalg.norm(goal - start))
    frac = 1.0 if dist == 0.0 else min(t * speed / dist, 1.0)
    return start + frac * (goal - start)


def setpoint_vel_at(t: float, start, goal, speed: float):
    """Velocity of that 'carrot': `speed` along A->B while travelling, zero once it reaches B."""
    import numpy as np

    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    dist = float(np.linalg.norm(goal - start))
    if dist == 0.0 or t * speed >= dist:
        return np.zeros(3)
    return speed * (goal - start) / dist


class PathDrawer:
    """Draws two lines in the Isaac Sim viewport: the ideal straight path A->B (green) and the path the
    drone has actually flown so far (red). Uses Isaac Sim's debug-draw extension; does nothing if unavailable."""

    GREEN, RED = (0.1, 1.0, 0.1, 1.0), (1.0, 0.1, 0.1, 1.0)

    def __init__(self, start, goal):
        self.start, self.goal, self.trail, self.draw = list(start), list(goal), [], None
        try:
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
            from isaacsim.util.debug_draw import _debug_draw

            self.draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as e:  # drawing is cosmetic; never break the flight because of it
            print(f"[WARN] path drawing unavailable: {e}")

    def update(self, pos):
        """Add the drone's current position to the red trail and redraw both lines."""
        if self.draw is None:
            return
        self.trail.append([float(p) for p in pos])
        self.draw.clear_lines()
        self.draw.draw_lines([self.start], [self.goal], [self.GREEN], [4.0])  # true path A->B
        if len(self.trail) > 1:  # actual path: consecutive trail points joined
            n = len(self.trail) - 1
            self.draw.draw_lines(self.trail[:-1], self.trail[1:], [self.RED] * n, [4.0] * n)


def fly_a_to_b(args, world):
    """Step the sim along the straight line A->B, then hover. Yields one dict per step (keys = STATE_COLUMNS)."""
    import torch

    from faults.thrust_loss import apply as apply_fault
    from faults.thrust_loss import is_active

    sim, robot, imu, controller, fault = world.sim, world.robot, world.imu, world.controller, world.fault
    total_steps = int(flight_duration(args, fault) / SIM_DT)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    drawer = PathDrawer(args.start, args.goal) if args.visualizer else None  # no lines when headless

    for step in range(total_steps):
        if not sim.is_running():  # user closed the window
            return
        t = step * SIM_DT
        sp = setpoint_at(t, args.start, args.goal, args.speed)
        state = {
            "pos": _t(robot.data.root_pos_w),
            "quat": _t(robot.data.root_quat_w),
            "lin_vel": _t(robot.data.root_lin_vel_w),
            "ang_vel": _t(robot.data.root_ang_vel_b),
        }

        if args.controller == "pid":
            dev = sim.device
            sp_pos = torch.tensor(sp, device=dev, dtype=torch.float32).unsqueeze(0)
            sp_vel = torch.tensor(setpoint_vel_at(t, args.start, args.goal, args.speed), device=dev,
                                  dtype=torch.float32).unsqueeze(0)
            wrench = controller.compute(state["pos"], state["lin_vel"], state["quat"], state["ang_vel"], sp_pos, sp_vel)
        else:  # Lee reads the robot state itself and takes [x, y, z, yaw]
            command = torch.zeros((1, 4), device=sim.device)
            command[0, :3] = torch.tensor(sp, device=sim.device)
            wrench = controller.compute(command)
        thrust_cmd = (wrench @ world.alloc_pinv.T).clamp(min=0.0)  # propellers can't pull down
        thrust_applied = apply_fault(thrust_cmd, fault, t)  # the faulty motor delivers less than it is told
        robot.set_thrust_target(thrust_applied)

        # log the state the controller saw, before stepping
        row = {"step": step, "t": t}
        for name, vec in state.items():
            labels = "xyzw" if name == "quat" else "xyz"
            for k, a in enumerate(labels):
                row[f"{name}_{a}"] = float(vec[0, k])
        for k, a in enumerate("xyz"):
            row[f"setpoint_{a}"] = float(sp[k])
        for i in range(4):
            row[f"thrust_cmd_{i}"] = float(thrust_cmd[0, i])
            row[f"thrust_applied_{i}"] = float(thrust_applied[0, i])
        active = fault is not None and is_active(fault, t)
        row["fault_active"] = int(active)
        row["fault_motor"] = fault.motor if fault is not None else -1
        row["fault_loss"] = fault.loss if fault is not None else 0.0
        acc, gyro = _t(imu.data.lin_acc_b), _t(imu.data.ang_vel_b)
        for k, a in enumerate("xyz"):
            row[f"imu_lin_acc_{a}"] = float(acc[0, k])
            row[f"imu_ang_vel_{a}"] = float(gyro[0, k])
        yield row

        if drawer is not None and step % 5 == 0:  # redraw every 5 steps (20 Hz) to keep the viewer smooth
            drawer.update((row["pos_x"], row["pos_y"], row["pos_z"]))

        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
        imu.update(SIM_DT)
