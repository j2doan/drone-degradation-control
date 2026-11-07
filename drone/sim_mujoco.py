"""The MuJoCo backend (--sim mujoco).

The same ARL Robot 1 flights as sim_isaac.py, on the CPU, without Isaac.

The drone is built here from Isaac Lab's own numbers for ARL Robot 1, so results are
comparable with Isaac:
  - one rigid body: 1.24 kg, inertia diag(0.013404, 0.014404, 0.013804) kg m^2 (read
    from Isaac by 01_inspect_env.py)
  - four rotors placed so that MuJoCo's forces and torques equal Isaac's allocation
    matrix (rotor yaw torque 0.07 N m per N of thrust); test_sim_mujoco.py checks this
  - the motor model ported from Isaac (motors/thruster.py): 0.1-10 N, spin-up lag,
    faster spin-down
  - IMU = MuJoCo accelerometer + gyro at the body origin (proper acceleration: +9.81
    on z at rest, like Isaac's IMU)
  - optional front RGB camera, 0.15 m ahead of the body origin, looking forward
  - landing legs 0.25 m below the body origin, so the logged height on the ground is
    0.25 m as in Isaac
The flight loop, CSV row, fault (applied to the motor OUTPUT), cap, anti-windup and
drone slots are the same as Isaac's. What differs: PhysX vs MuJoCo contact/integration,
the drone's visual shape, and the motor parameters are drawn from the seed (Isaac draws
them unseeded), so a MuJoCo flight repeats exactly.
"""

import contextlib
import time
from itertools import pairwise
from types import SimpleNamespace

from common import CAMERA_PERIOD, SIM_DT, make_fault

# ARL Robot 1 (isaaclab_assets/robots/arl_robot_1.py and the values Isaac reports for
# its USD model)
MASS = 1.24  # kg
INERTIA = (0.013404, 0.014404, 0.013804)  # kg m^2, principal axes = body axes
# rows Fx, Fy, Fz, Tx, Ty, Tz; columns: back_left, back_right, front_left, front_right
ALLOCATION = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
    [-0.13, -0.13, 0.13, 0.13],
    [-0.13, 0.13, 0.13, -0.13],
    [-0.07, 0.07, -0.07, 0.07],
]
# m, body origin height when the drone stands on the ground (as measured in Isaac)
GROUND_CLEARANCE = 0.25
SKY = (0.53, 0.81, 0.92)  # Isaac's dome light color


def model_xml() -> str:
    """MJCF of the drone.

    A rotor's thrust F at (x, y) along body z gives torque (y F, -x F); its yaw reaction
    is the allocation's Tz entry. So x = -Ty entry, y = Tx entry.
    """
    rotors, motors = [], []
    for i in range(4):
        x, y, yaw = -ALLOCATION[4][i], ALLOCATION[3][i], ALLOCATION[5][i]
        rotors.append(
            f'<site name="rotor{i}" pos="{x} {y} 0.03" size="0.01"/>'
            f'<geom type="cylinder" size="0.07 0.005" pos="{x} {y} 0.03" '
            'rgba="0.2 0.2 0.2 0.5" '
            f'contype="0" conaffinity="0" mass="0"/>'
            f'<geom type="capsule" fromto="0 0 0 {x} {y} 0.02" size="0.01" '
            'rgba="0.3 0.3 0.3 1" '
            f'contype="0" conaffinity="0" mass="0"/>'
        )
        motors.append(
            f'<motor name="motor{i}" site="rotor{i}" gear="0 0 1 0 0 {yaw}" '
            'ctrllimited="true" '
            f'ctrlrange="0 10"/>'
        )
    sky = " ".join(str(c) for c in SKY)
    return f"""
<mujoco model="arl_robot_1">
  <option timestep="{SIM_DT}" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight ambient="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="{sky}" rgb2="{sky}"
             width="16" height="16"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.35 0.35 0.35"
             rgb2="0.45 0.45 0.45" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="20 20" texuniform="false"/>
  </asset>
  <worldbody>
    <light pos="0 0 10" dir="0 0 -1" directional="true"/>
    <geom name="ground" type="plane" size="50 50 0.1" material="grid"/>
    <body name="base_link" pos="0 0 1">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{MASS}"
                diaginertia="{INERTIA[0]} {INERTIA[1]} {INERTIA[2]}"/>
      <geom name="frame" type="box" size="0.08 0.08 0.03" rgba="0.1 0.1 0.6 1"
            mass="0"/>
      <geom name="legs" type="box" size="0.12 0.12 0.01"
            pos="0 0 {-GROUND_CLEARANCE + 0.01}" rgba="0.2 0.2 0.2 1" mass="0"/>
      <site name="imu" pos="0 0 0" size="0.01"/>
      <camera name="front" pos="0.15 0 0" xyaxes="0 -1 0 0 0 1" fovy="82"/>
      {"".join(rotors)}
    </body>
  </worldbody>
  <actuator>{"".join(motors)}</actuator>
  <sensor><accelerometer name="acc" site="imu"/><gyro name="gyro" site="imu"/></sensor>
</mujoco>
"""


@contextlib.contextmanager
def session(args):
    """MuJoCo needs no launcher.

    Yields a holder that make_world uses for the viewer, which is closed at the end.
    """
    holder = SimpleNamespace(viewer=None)
    try:
        yield holder
    finally:
        if holder.viewer is not None:
            holder.viewer.close()


def versions() -> dict:
    import mujoco

    return {
        "mujoco": mujoco.__version__,
        "drone_model": "ARL Robot 1 clone (sim_mujoco.model_xml)",
    }


def make_world(args, holder):
    """Build the model, motor model, PID and (unless --headless) the viewer once.

    Each flight then calls prepare_flight(). Returns a namespace with: model, data,
    controller, motors, mass, inertia, allocation, alloc_pinv, renderer (or None),
    viewer (or None).
    """
    import mujoco
    import torch
    from controllers.pid import CascadedPID
    from motors.thruster import ThrusterModel

    model = mujoco.MjModel.from_xml_string(model_xml())
    data = mujoco.MjData(model)
    body = model.body("base_link").id
    mass = float(model.body_subtreemass[body])
    inertia = [float(v) for v in model.body_inertia[body]]
    controller = CascadedPID(
        num_envs=1, mass=mass, inertia=torch.diag(torch.tensor(inertia)), dt=SIM_DT
    )
    # (4, 6): wrench -> thrusts
    alloc_pinv = torch.linalg.pinv(torch.tensor(ALLOCATION, dtype=torch.float32))
    renderer = (
        mujoco.Renderer(model, args.camera_res, args.camera_res)
        if args.camera
        else None
    )
    viewer = None
    if args.visualizer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = body
        viewer.cam.distance, viewer.cam.elevation = 4.0, -20.0
        holder.viewer = viewer
    return SimpleNamespace(
        model=model,
        data=data,
        body=body,
        controller=controller,
        motors=ThrusterModel(SIM_DT),
        mass=mass,
        inertia=inertia,
        allocation=ALLOCATION,
        alloc_pinv=alloc_pinv,
        renderer=renderer,
        viewer=viewer,
    )


def prepare_flight(args, world, seed, drone):
    """Set up one flight and reset the drone to the path's start, at rest.

    The seed picks the path, the fault, the IMU noise and the motor parameters.
    Everything is reset, so a flight inside a batch behaves like a flight in a fresh
    run.
    """
    import mujoco
    from faults.thrust_loss import hover_feasibility, run_duration
    from paths.curve import sample_path
    from sensors.noise import ImuNoise

    path = sample_path(seed)
    travel = path.travel_time(args.speed)
    fault = make_fault(args, seed, travel)

    model, data = world.model, world.data
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [float(v) for v in path.start]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # MuJoCo quaternions are (w, x, y, z)
    mujoco.mj_forward(model, data)
    world.motors.reset(seed)
    world.controller.reset()
    if drone.policy is not None:
        drone.policy.reset()

    return SimpleNamespace(
        seed=seed,
        drone=drone,
        path=path,
        fault=fault,
        duration=run_duration(travel + args.hold, fault),
        feasibility=hover_feasibility(
            world.allocation, world.mass, fault, args.max_thrust
        ),
        noise=ImuNoise(seed, num_envs=1),
        frames=[],  # camera images (uint8 HxWx3), filled during the flight
    )


class PathDrawer:
    """Draws the ideal path (green) and the path flown so far (red) in the viewer."""

    GREEN, RED = (0.1, 1.0, 0.1, 1.0), (1.0, 0.1, 0.1, 1.0)

    def __init__(self, viewer, route_points):
        self.viewer, self.route, self.trail = (
            viewer,
            [[float(v) for v in p] for p in route_points],
            [],
        )

    def _lines(self, scn, points, color):
        import mujoco
        import numpy as np

        for a, b in pairwise(points):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).flatten(),
                np.array(color, dtype=np.float32),
            )
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_LINE, 4.0, np.array(a), np.array(b)
            )
            scn.ngeom += 1

    def update(self, pos):
        self.trail.append([float(p) for p in pos])
        with self.viewer.lock():
            scn = self.viewer.user_scn
            scn.ngeom = 0
            self._lines(scn, self.route, self.GREEN)
            self._lines(scn, self.trail[-(scn.maxgeom - len(self.route)) :], self.RED)


def fly(args, world, flight):
    """Fly one prepared flight along its path, then hover at B.

    Yields one dict per step (keys = STATE_COLUMNS).
    """
    import mujoco
    import torch
    from drones import combine, make_observation
    from faults.thrust_loss import apply as apply_fault
    from faults.thrust_loss import is_active

    model, data, controller, viewer = (
        world.model,
        world.data,
        world.controller,
        world.viewer,
    )
    fault, drone = flight.fault, flight.drone
    total_steps = int(flight.duration / SIM_DT)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    camera_every = round(CAMERA_PERIOD / SIM_DT)
    drawer = PathDrawer(viewer, flight.path.sample(200)) if viewer is not None else None
    last_thrust = torch.zeros((1, 4))

    def vec(x):
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    wall_start = time.perf_counter()

    for step in range(total_steps):
        if viewer is not None and not viewer.is_running():  # user closed the window
            return
        t = step * SIM_DT
        sp_np, sp_vel_np = flight.path.setpoint(t, args.speed)
        sp, sp_vel = vec(sp_np), vec(sp_vel_np)
        w, x, y, z = data.qpos[3:7]
        state = {
            "pos": vec(data.qpos[:3]),
            "quat": vec([x, y, z, w]),  # stored (x, y, z, w) like Isaac
            "lin_vel": vec(data.qvel[:3]),  # free joint: world-frame linear velocity
            "ang_vel": vec(data.qvel[3:6]),  # free joint: body-frame angular velocity
        }
        acc, gyro = vec(data.sensor("acc").data), vec(data.sensor("gyro").data)
        acc_noisy, gyro_noisy = flight.noise.apply(acc, gyro)

        wrench = controller.compute(
            state["pos"], state["lin_vel"], state["quat"], state["ang_vel"], sp, sp_vel
        )
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
        # anti-windup
        controller.set_saturated((thrust_cmd > args.max_thrust).any(dim=1))
        # the motor model, then the fault damages the faulty motor's OUTPUT (same order
        # as the Isaac fault hook)
        thrust_applied = apply_fault(world.motors.step(capped), fault, t)
        data.ctrl[:] = thrust_applied[0].numpy()
        last_thrust = capped

        # log the state the controller saw, and the thrust that results, before stepping
        row = {"step": step, "t": t}
        for name, v in state.items():
            labels = "xyzw" if name == "quat" else "xyz"
            for k, a in enumerate(labels):
                row[f"{name}_{a}"] = float(v[0, k])
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
        if world.renderer is not None and step % camera_every == 0:
            world.renderer.update_scene(data, camera="front")
            flight.frames.append(world.renderer.render().copy())
            row["camera_frame"] = len(flight.frames) - 1
        yield row

        mujoco.mj_step(model, data)
        if viewer is not None:
            if step % 5 == 0:  # redraw the lines every 5 steps (20 Hz)
                drawer.update((row["pos_x"], row["pos_y"], row["pos_z"]))
            viewer.sync()
            ahead = (step + 1) * SIM_DT - (time.perf_counter() - wall_start)
            if (
                ahead > 0
            ):  # run at real time in the viewer, as fast as possible headless
                time.sleep(ahead)
