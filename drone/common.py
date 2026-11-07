"""Shared helpers for the drone scripts. This is the only file that talks to Isaac Sim.

Pattern copied from IsaacLab/examples/arl_robot_1.py: one ARL Robot 1 quadrotor stepped
by hand with no RL environment. Each flight follows a seeded random curved path
(paths/curve.py); the same seed also picks the fault (faults/thrust_loss.py; seed 0 =
healthy) and the IMU noise (sensors/noise.py), so every drone flies the identical flight
for a seed. Drones come from drones.py: d0 = our PID (controllers/pid.py); d1-d3 add or
replace it with a PPO policy once one is plugged in. Isaac Lab's Lee controller stays
available as a regression check (--controller lee). The simulator-independent parts
(controllers/, faults/, paths/, sensors/, analysis/, drones.py) never import Isaac.

IMPORTANT: Isaac Sim must be launched before most isaaclab modules can be imported, so
every import other than argparse/isaaclab.app happens inside a function.
"""

import argparse
from types import SimpleNamespace

from isaaclab.app import add_launcher_args

SIM_DT = 0.01  # physics step, seconds (same as the RL env)
# Per-motor command cap [N], about 2x hover thrust. Same limit as Isaac Lab's drone RL
# env (actions 0-6 N), so the baseline and the future PPO drones play by the same rules.
# The motor model itself allows 0.1-10 N.
MAX_MOTOR_THRUST = 6.0
CAMERA_PERIOD = 0.1  # s, the front camera records at 10 Hz

# One row of the log = one physics step. Order matters: it is the CSV column order.
STATE_COLUMNS = (
    ["step", "t"]
    + [f"pos_{a}" for a in "xyz"]  # world position [m]
    + [f"quat_{a}" for a in "xyzw"]  # orientation (x, y, z, w)
    + [f"lin_vel_{a}" for a in "xyz"]  # world-frame linear velocity [m/s]
    + [f"ang_vel_{a}" for a in "xyz"]  # body-frame angular velocity [rad/s]
    # where the controller was told to go (traces the route)
    + [f"setpoint_{a}" for a in "xyz"]
    # thrust the controller asked for, per propeller, before the cap [N]
    + [f"thrust_cmd_{i}" for i in range(4)]
    # thrust the motor actually delivered: after the cap, the motor model (lag, 0.1-10
    # N) and the fault [N]
    + [f"thrust_applied_{i}" for i in range(4)]
    # 0/1, motor index (-1 = none), lost fraction
    + ["fault_active", "fault_motor", "fault_loss"]
    # IMU proper acceleration, body frame (+9.81 on z at rest) [m/s^2]
    + [f"imu_lin_acc_{a}" for a in "xyz"]
    + [f"imu_ang_vel_{a}" for a in "xyz"]  # IMU angular velocity, body frame [rad/s]
    # the same with realistic sensor noise + bias (what a policy sees)
    + [f"imu_lin_acc_noisy_{a}" for a in "xyz"]
    + [f"imu_ang_vel_noisy_{a}" for a in "xyz"]
    # index into the flight's camera.npz, or -1 (no new frame this step / camera off)
    + ["camera_frame"]
)


def build_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser: the flight options plus Isaac Lab's launcher flags."""
    parser = argparse.ArgumentParser(
        description=description, conflict_handler="resolve"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Speed of the moving target point along the path, m/s.",
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
        "--controller",
        default="pid",
        choices=["pid", "lee"],
        help="pid = our PID (default); "
        "lee = Isaac Lab's Lee controller (regression check).",
    )
    parser.add_argument(
        "--fault",
        default="random",
        choices=["none", "random", "fixed"],
        help="Fault for seeds >= 1 (seed 0 is always healthy): "
        "random from the seed (default); "
        "fixed = --fault_motor/_loss/_onset; none = all healthy.",
    )
    parser.add_argument(
        "--fault_motor", type=int, default=0, help="Motor index 0-3 for --fault fixed."
    )
    parser.add_argument(
        "--fault_loss",
        type=float,
        default=0.25,
        help="Lost thrust fraction for --fault fixed.",
    )
    parser.add_argument(
        "--fault_onset",
        type=float,
        default=4.0,
        help="Fault start time [s] for --fault fixed.",
    )
    parser.add_argument(
        "--max_thrust",
        type=float,
        default=MAX_MOTOR_THRUST,
        help="Per-motor command cap in newtons "
        "(default 6, like the Isaac Lab drone RL env).",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Record the front RGB camera (10 Hz). Slower runs.",
    )
    parser.add_argument(
        "--camera_res",
        type=int,
        default=64,
        choices=[64, 128],
        help="Camera image size (square).",
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
    parser = parser or build_parser(description)
    args = parser.parse_args()
    # catch bad values before Isaac starts (else a typo fails a minute later, mid-batch)
    if not 0 <= args.fault_motor <= 3:
        parser.error("--fault_motor must be 0-3")
    if not 0.0 <= args.fault_loss <= 1.0:
        parser.error("--fault_loss is a fraction: 0.0-1.0 (e.g. 0.4 = loses 40%)")
    if args.fault_onset < 0 or args.hold < 0:
        parser.error("--fault_onset and --hold must be >= 0")
    if args.speed <= 0 or args.max_thrust <= 0:
        parser.error("--speed and --max_thrust must be > 0")
    if args.headless:
        args.visualizer = []  # no visualizer = headless
    return args


def camera_cfg(args):
    """Front RGB camera on the drone body (world convention: +X forward, +Z up)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path="/World/Robot/base_link/front_cam",
        update_period=CAMERA_PERIOD,
        height=args.camera_res,
        width=args.camera_res,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, clipping_range=(0.05, 100.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
        ),
    )


def launch_cfg(args):
    """What to hand launch_simulation().

    With --camera the camera config is included, which makes the launcher switch camera
    rendering on (it scans the config for camera sensors).
    """
    from isaaclab.physics import PhysicsCfg

    return (
        SimpleNamespace(physics=PhysicsCfg(), camera=camera_cfg(args))
        if args.camera
        else PhysicsCfg()
    )


def install_fault_hook(robot):
    """Damage the faulty motor itself, not the order it receives.

    Wraps Isaac's thruster model so the fault scales the motor's OUTPUT, after its lag
    and its 0.1-10 N limit. The multirotor sends whatever `actuator.compute(...)`
    returns to the physics (Multirotor._apply_actuator_model in isaaclab_contrib,
    verified on the installed Isaac Lab), so a motor that lost 30% can never deliver
    more than 70% of its maximum. Returns (actuator, state); set state.fault for each
    flight and state.t to the current time before robot.write_data_to_sim().
    """
    from faults.thrust_loss import apply as apply_fault

    actuator = robot.actuators["thrusters"]
    state = SimpleNamespace(fault=None, t=0.0)
    nominal_compute = actuator.compute  # the healthy motor model

    def damaged_compute(control_action):
        out = nominal_compute(control_action)
        if state.fault is not None:
            # new tensor; motor model state untouched
            out.thrusts = apply_fault(out.thrusts, state.fault, state.t)
            # keep Isaac's logged value equal to what the physics receives
            actuator.applied_thrust = out.thrusts
        return out

    actuator.compute = damaged_compute
    return actuator, state


def make_world(args, physics_cfg):
    """Create sim, ground, light, robot, IMU, optional camera and controllers once.

    Call inside launch_simulation().

    Each flight then calls prepare_flight() to put the drone at its path's start.
    Returns a namespace with: sim, robot, imu, camera (or None), controller (PID or
    Lee), lee (always built: it also provides the drone's mass and inertia), mass,
    allocation, alloc_pinv, actuator + fault_state (the fault hook).
    """
    import isaaclab.sim as sim_utils
    import torch
    from controllers.pid import CascadedPID
    from isaaclab.sensors import ImuCfg
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
    # moved to each path's start later
    robot_cfg.init_state = replace(robot_cfg.init_state, pos=(0.0, 0.0, 1.0))
    robot_cfg.actuators["thrusters"].dt = sim_cfg.dt
    robot = instantiate(robot_cfg)
    # Sensors on the drone's main body. Must be created before sim.reset() so they
    # initialize with the robot.
    imu = instantiate(ImuCfg(prim_path="/World/Robot/base_link"))
    camera = instantiate(camera_cfg(args)) if args.camera else None

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
        robot,
        num_envs=1,
        device=str(sim.device),
    )
    mass = float(lee.mass.flatten()[0])
    if args.controller == "pid":
        controller = CascadedPID(
            num_envs=1,
            mass=mass,
            inertia=lee.robot_inertia,
            dt=SIM_DT,
            device=str(sim.device),
        )
    else:
        controller = lee
    # Both controllers output a 6-D wrench [Fx,Fy,Fz,Tx,Ty,Tz]; the robot takes 4
    # propeller thrusts. The allocation matrix maps thrusts -> wrench, so its
    # pseudoinverse maps wrench -> thrusts.
    alloc = torch.tensor(
        robot_cfg.allocation_matrix, device=sim.device, dtype=torch.float32
    )
    alloc_pinv = torch.linalg.pinv(alloc)  # (4, 6)
    actuator, fault_state = install_fault_hook(robot)
    return SimpleNamespace(
        sim=sim,
        robot=robot,
        imu=imu,
        camera=camera,
        controller=controller,
        lee=lee,
        mass=mass,
        allocation=robot_cfg.allocation_matrix,
        alloc_pinv=alloc_pinv,
        actuator=actuator,
        fault_state=fault_state,
    )


def make_fault(args, seed, travel_time):
    """The fault for this seed (a FaultSpec), or None; seed 0 is always healthy."""
    from faults.thrust_loss import FaultSpec, sample_fault

    if seed == 0 or args.fault == "none":
        return None
    if args.fault == "fixed":
        return FaultSpec(
            motor=args.fault_motor, loss=args.fault_loss, onset_t=args.fault_onset
        )
    # onset while travelling; --hold does not matter
    return sample_fault(seed, travel_time)


def prepare_flight(args, world, seed, drone):
    """Set up one flight and reset the drone to the path's start, at rest.

    The seed picks the path, the fault and the IMU noise.

    Everything is reset (pose, velocity, motor model, controller, policy, sensors) so a
    flight inside a batch behaves like a flight in a fresh launch.
    """
    import torch
    from faults.thrust_loss import hover_feasibility, run_duration
    from paths.curve import sample_path
    from sensors.noise import ImuNoise

    sim, robot, dev = world.sim, world.robot, world.sim.device
    path = sample_path(seed)
    travel = path.travel_time(args.speed)
    fault = make_fault(args, seed, travel)
    world.fault_state.fault, world.fault_state.t = fault, 0.0

    a = [float(v) for v in path.start]
    robot.write_root_pose_to_sim_index(
        root_pose=torch.tensor([[*a, 0.0, 0.0, 0.0, 1.0]], device=dev)
    )
    robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=dev)
    )
    # also resets the thruster model (motor lag state) to its initial spin. Pass the
    # index explicitly: on this Isaac Lab commit, Multirotor.reset(None) fails ("Item
    # indexing is not supported on wp.array objects").
    robot.reset(env_ids=torch.tensor([0], device=dev, dtype=torch.long))
    robot.set_thrust_target(torch.zeros((1, 4), device=dev))
    robot.write_data_to_sim()
    sim.forward()
    robot.update(0.0)
    world.controller.reset() if hasattr(world.controller, "reset") else None
    if drone.policy is not None:
        drone.policy.reset()
    world.imu.reset()
    if world.camera is not None:
        world.camera.reset()

    return SimpleNamespace(
        seed=seed,
        drone=drone,
        path=path,
        fault=fault,
        duration=run_duration(travel + args.hold, fault),
        feasibility=hover_feasibility(
            world.allocation, world.mass, fault, args.max_thrust
        ),
        noise=ImuNoise(seed, num_envs=1, device=str(dev)),
        frames=[],  # camera images (uint8 HxWx3), filled during the flight
    )


def _t(x):
    """Warp array, Isaac ProxyArray or torch tensor -> torch tensor."""
    import torch
    import warp as wp

    if torch.is_tensor(x):
        return x
    if hasattr(x, "torch"):  # ProxyArray (e.g. sensor data) has a zero-copy .torch view
        return x.torch
    return wp.to_torch(x)


class PathDrawer:
    """Draws the ideal and the flown path as lines in the Isaac Sim viewport.

    Green = the ideal path, red = the path the drone has flown so far. Uses Isaac Sim's
    debug-draw extension; does nothing if unavailable.
    """

    GREEN, RED = (0.1, 1.0, 0.1, 1.0), (1.0, 0.1, 0.1, 1.0)

    def __init__(self, route_points):
        self.route = [[float(v) for v in p] for p in route_points]
        self.trail, self.draw = [], None
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

    @staticmethod
    def _lines(draw, points, color):
        n = len(points) - 1
        if n > 0:
            draw.draw_lines(points[:-1], points[1:], [color] * n, [4.0] * n)

    def update(self, pos):
        """Add the drone's current position to the red trail and redraw both lines."""
        if self.draw is None:
            return
        self.trail.append([float(p) for p in pos])
        self.draw.clear_lines()
        self._lines(self.draw, self.route, self.GREEN)  # ideal path
        self._lines(self.draw, self.trail, self.RED)  # flown path


def fly(args, world, flight):
    """Fly one prepared flight along its path, then hover at B.

    Yields one dict per step (keys = STATE_COLUMNS).
    """
    import torch
    from drones import combine, make_observation
    from faults.thrust_loss import is_active

    sim, robot, imu, camera, controller = (
        world.sim,
        world.robot,
        world.imu,
        world.camera,
        world.controller,
    )
    fault, drone, dev = flight.fault, flight.drone, sim.device
    total_steps = int(flight.duration / SIM_DT)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    camera_every = round(CAMERA_PERIOD / SIM_DT)
    # no lines when headless
    drawer = PathDrawer(flight.path.sample(200)) if args.visualizer else None
    last_thrust = torch.zeros((1, 4), device=dev)

    for step in range(total_steps):
        if not sim.is_running():  # user closed the window
            return
        t = step * SIM_DT
        sp_np, sp_vel_np = flight.path.setpoint(t, args.speed)
        sp = torch.tensor(sp_np, device=dev, dtype=torch.float32).unsqueeze(0)
        sp_vel = torch.tensor(sp_vel_np, device=dev, dtype=torch.float32).unsqueeze(0)
        state = {
            "pos": _t(robot.data.root_pos_w),
            "quat": _t(robot.data.root_quat_w),
            "lin_vel": _t(robot.data.root_lin_vel_w),
            "ang_vel": _t(robot.data.root_ang_vel_b),
        }
        acc, gyro = _t(imu.data.lin_acc_b), _t(imu.data.ang_vel_b)
        acc_noisy, gyro_noisy = flight.noise.apply(acc, gyro)

        if args.controller == "pid":
            wrench = controller.compute(
                state["pos"],
                state["lin_vel"],
                state["quat"],
                state["ang_vel"],
                sp,
                sp_vel,
            )
        else:  # Lee reads the robot state itself and takes [x, y, z, yaw]
            command = torch.zeros((1, 4), device=dev)
            command[0, :3] = sp[0]
            wrench = controller.compute(command)
        # the PID's thrusts; propellers can't pull
        thrust_cmd = (wrench @ world.alloc_pinv.T).clamp(min=0.0)
        if (
            drone.mode != "pid"
        ):  # D1/D2 add a PPO correction, D3 replaces the PID (see drones.py)
            obs = make_observation(
                state["pos"],
                state["lin_vel"],
                state["quat"],
                state["ang_vel"],
                acc_noisy,
                gyro_noisy,
                sp,
                sp_vel,
                last_thrust,
            )
            thrust_cmd = combine(thrust_cmd, drone.policy.act(obs), drone.mode).clamp(
                min=0.0
            )
        capped = thrust_cmd.clamp(max=args.max_thrust)  # every motor command is capped
        if hasattr(
            controller, "set_saturated"
        ):  # anti-windup: no integral growth while a motor is maxed out
            controller.set_saturated((thrust_cmd > args.max_thrust).any(dim=1))
        robot.set_thrust_target(capped)
        world.fault_state.t = t
        # runs the motor model, then the fault hook damages the faulty motor's output
        robot.write_data_to_sim()
        # what each motor actually delivers this step
        thrust_applied = _t(world.actuator.applied_thrust)
        last_thrust = capped

        # log the state the controller saw, and the thrust that results, before stepping
        row = {"step": step, "t": t}
        for name, vec in state.items():
            labels = "xyzw" if name == "quat" else "xyz"
            for k, a in enumerate(labels):
                row[f"{name}_{a}"] = float(vec[0, k])
        for k, a in enumerate("xyz"):
            row[f"setpoint_{a}"] = float(sp_np[k])
        for i in range(4):
            row[f"thrust_cmd_{i}"] = float(thrust_cmd[0, i])
            row[f"thrust_applied_{i}"] = float(thrust_applied[0, i])
        row["fault_active"] = int(fault is not None and is_active(fault, t))
        row["fault_motor"] = fault.motor if fault is not None else -1
        row["fault_loss"] = fault.loss if fault is not None else 0.0
        for k, a in enumerate("xyz"):
            row[f"imu_lin_acc_{a}"] = float(acc[0, k])
            row[f"imu_ang_vel_{a}"] = float(gyro[0, k])
            row[f"imu_lin_acc_noisy_{a}"] = float(acc_noisy[0, k])
            row[f"imu_ang_vel_noisy_{a}"] = float(gyro_noisy[0, k])
        row["camera_frame"] = -1
        if camera is not None and step % camera_every == 0:
            rgb = _t(camera.data.output["rgb"])
            flight.frames.append(rgb[0, ..., :3].detach().to("cpu").numpy().copy())
            row["camera_frame"] = len(flight.frames) - 1
        yield row

        if (
            drawer is not None and step % 5 == 0
        ):  # redraw every 5 steps (20 Hz) to keep the viewer smooth
            drawer.update((row["pos_x"], row["pos_y"], row["pos_z"]))

        sim.step()
        robot.update(SIM_DT)
        imu.update(SIM_DT)
        if camera is not None:
            camera.update(SIM_DT)


def make_receipt(args) -> dict:
    """Everything needed to say later exactly how a batch was made (manifest.json)."""
    import dataclasses
    import datetime
    import hashlib
    import pathlib
    import platform
    import subprocess
    import sys

    import isaaclab
    import torch
    from controllers.pid import PIDGains

    drone_dir = pathlib.Path(__file__).resolve().parent
    code = {
        str(p.relative_to(drone_dir)).replace("\\", "/"): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()[:12]
        for p in sorted(drone_dir.rglob("*.py"))
        if "__pycache__" not in p.parts
    }
    # .../IsaacLab/source/isaaclab/isaaclab
    isaaclab_root = pathlib.Path(isaaclab.__file__).resolve().parents[3]
    try:
        revision = (
            subprocess.run(
                ["git", "-C", str(isaaclab_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv[1:]),
        "isaaclab_revision": revision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "controller": args.controller,
        "pid_gains": dataclasses.asdict(PIDGains()),
        "max_motor_thrust_n": args.max_thrust,
        "sim_dt_s": SIM_DT,
        "settings": {
            k: v
            for k, v in vars(args).items()
            if isinstance(v, (int, float, str, bool, list, type(None)))
        },
        "code_sha256": code,
    }
