"""Shared helpers for the drone scripts.

Flight options, CSV columns, faults, the run receipt, and the choice of simulator.

Two simulators fly the same ARL Robot 1 quadrotor with the same code around it:
  --sim isaac (default): Isaac Sim / Isaac Lab, sim_isaac.py
  --sim mujoco: MuJoCo with a model of the same drone (mass, inertia, motor layout,
      motor model), sim_mujoco.py
Only those two files talk to a simulator; this file forwards make_world / prepare_flight
/ fly to the chosen one.

Each flight follows a seeded random curved path (paths/curve.py); the same seed also
picks the fault (faults/thrust_loss.py; seed 0 = healthy) and the IMU noise
(sensors/noise.py), so every drone flies the identical flight for a seed. Drones come
from drones.py: d0 = our PID (controllers/pid.py); d1-d3 add or replace it with a PPO
policy once one is plugged in. The simulator-independent parts (controllers/, faults/,
motors/, paths/, sensors/, analysis/, drones.py) never import a simulator.

Isaac Sim must be launched before most isaaclab modules can be imported, so nothing
simulator-specific is imported at the top of this file.
"""

import argparse
import importlib
import sys

SIM_DT = 0.01  # physics step, seconds (same as the RL env)
# Per-motor command cap [N], about 2x hover thrust. Same limit as Isaac Lab's drone RL
# env (actions 0-6 N), so the baseline and the future PPO drones play by the same rules.
# The motor model itself allows 0.1-10 N.
MAX_MOTOR_THRUST = 6.0
CAMERA_PERIOD = 0.1  # s, the front camera records at 10 Hz
SIMS = ("isaac", "mujoco")

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


def requested_sim(argv=None) -> str:
    """The --sim value on the command line (default isaac), read before the parser."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sim", default="isaac", choices=SIMS)
    return pre.parse_known_args(sys.argv[1:] if argv is None else argv)[0].sim


def build_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the flight options.

    With --sim isaac it also has Isaac Lab's launcher flags (--device, ...).
    """
    sim = requested_sim()
    parser = argparse.ArgumentParser(
        description=description, conflict_handler="resolve"
    )
    parser.add_argument(
        "--sim",
        default="isaac",
        choices=SIMS,
        help="isaac = Isaac Sim (default); "
        "mujoco = MuJoCo (CPU, no Isaac needed; see README).",
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
        "lee = Isaac Lab's Lee controller (regression check, Isaac only).",
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
        "--headless",
        action="store_true",
        help="No window (faster). Default opens the viewer.",
    )
    if sim == "isaac":
        from isaaclab.app import add_launcher_args

        parser.add_argument(
            "--physics",
            default="isaacsim_physx",
            choices=["isaacsim_physx"],
            help="Physics backend.",
        )
        add_launcher_args(parser)
        parser.set_defaults(visualizer=["kit"])
    else:
        parser.set_defaults(visualizer=["mujoco"], device="cpu")
    return parser


def parse_args(
    description: str, parser: argparse.ArgumentParser | None = None
) -> argparse.Namespace:
    parser = parser or build_parser(description)
    args = parser.parse_args()
    # catch bad values here, before the simulator starts (a typo would otherwise fail a
    # minute later, mid-batch)
    if not 0 <= args.fault_motor <= 3:
        parser.error("--fault_motor must be 0-3")
    if not 0.0 <= args.fault_loss <= 1.0:
        parser.error("--fault_loss is a fraction: 0.0-1.0 (e.g. 0.4 = loses 40%)")
    if args.fault_onset < 0 or args.hold < 0:
        parser.error("--fault_onset and --hold must be >= 0")
    if args.speed <= 0 or args.max_thrust <= 0:
        parser.error("--speed and --max_thrust must be > 0")
    if args.sim == "mujoco" and args.controller == "lee":
        parser.error(
            "--controller lee is Isaac Lab's controller; "
            "it only exists with --sim isaac"
        )
    if args.headless:
        args.visualizer = []  # no visualizer = headless
    return args


def backend(args):
    """The simulator module for args.sim: sim_isaac or sim_mujoco (same functions)."""
    return importlib.import_module(f"sim_{args.sim}")


def session(args):
    """`with session(args) as physics_cfg:` starts the simulator.

    With Isaac the process ends when the block exits, so everything (including saving)
    must happen inside the block.
    """
    return backend(args).session(args)


def make_world(args, physics_cfg):
    """Build the simulator world once (drone, sensors, controller).

    See sim_isaac.make_world / sim_mujoco.make_world.
    """
    return backend(args).make_world(args, physics_cfg)


def prepare_flight(args, world, seed, drone):
    """Set up one flight (path, fault, noise) and put the drone at rest at its start."""
    return backend(args).prepare_flight(args, world, seed, drone)


def fly(args, world, flight):
    """Fly one prepared flight along its path, then hover at B.

    Yields one dict per step (keys = STATE_COLUMNS).
    """
    return backend(args).fly(args, world, flight)


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


def make_receipt(args) -> dict:
    """Everything needed to say later exactly how a batch was made (manifest.json)."""
    import dataclasses
    import datetime
    import hashlib
    import pathlib
    import platform

    import torch
    from controllers.pid import PIDGains

    drone_dir = pathlib.Path(__file__).resolve().parent
    code = {}
    for p in sorted(drone_dir.rglob("*.py")):
        rel = p.relative_to(drone_dir)
        if (
            "__pycache__" in rel.parts
            or rel.parts[0] == "data"
            or rel.parts[0].startswith(".")
        ):  # skip .venv-mujoco
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        code[str(rel).replace("\\", "/")] = digest[:12]
    return {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv[1:]),
        "sim": args.sim,
        **backend(args).versions(),
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
