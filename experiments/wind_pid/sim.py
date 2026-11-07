"""Skydio X2 in MuJoCo, a position controller with an optional integral term, and a steady wind.

    position loop: error -> desired acceleration  a = kp*e + kd*(v_target - v) + I
                   I = ki * integral(e dt), clipped to +-i_limit (m/s^2), built up only when the integration rule allows
    force:         F = m * (a + g), tilt limited to MAX_TILT
    attitude loop: body z axis along F (yaw 0): PD on the rotation error (SO(3)) -> body torque
    mixer:         [thrust, torque] -> four rotor thrusts (inverse of the allocation matrix read from the model)

Wind is a constant horizontal force pushing on the drone's centre of mass (MuJoCo xfrc_applied), on from t = 0.

The three controllers of the study are the same code with different settings:
    PD                 ki = 0
    PID                integral always on (rule "always")
    PID, near-target   integral only while the drone is within `i_zone` metres of its waypoint (rule "near")
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLED_MODEL = HERE / "assets"  # holds skydio_x2/ (MuJoCo Menagerie, Apache-2.0)

GRAVITY = 9.81  # m/s^2
MAX_TILT = np.radians(45.0)  # rad, the position loop never asks for more tilt than this
KP_ATT = np.array([1.725, 1.725, 0.325])  # N m / rad
KD_ATT = np.array([0.45, 0.45, 0.0825])  # N m / (rad/s)

# Mission: hover at the first waypoint, then step the target through a 1.5 m square at 1.5 m height.
WAYPOINTS = np.array([[0.0, 0.0, 1.5], [1.5, 0.0, 1.5], [1.5, 1.5, 1.5], [0.0, 1.5, 1.5], [0.0, 0.0, 1.5]])
HOLD_T = 4.0  # s the target stays on each waypoint
WIND_DIR = np.array([1.0, 0.0, 0.0])  # the wind blows along +x
CRASH_MIN_Z = 0.3  # m, lower than this = crash


@dataclass(frozen=True)
class Gains:
    kp: float = 4.0  # 1/s^2: 1 m of error asks for 4 m/s^2
    kd: float = 3.0  # 1/s, on the velocity error
    ki: float = 0.0  # 1/s^3 (0 = PD)
    i_rule: str = "always"  # "always" or "near"
    i_zone: float = float("inf")  # m, for "near": build the integral only within this distance of the waypoint
    i_limit: float = float("inf")  # m/s^2, the largest push the integral term may give

    def label(self):
        if self.ki == 0:
            return "PD"
        zone = "" if self.i_rule == "always" else f", near {self.i_zone:g} m"
        return f"PID ki {self.ki:g}{zone}, limit {self.i_limit:g}"


def find_scene() -> Path:
    """skydio_x2/scene.xml from $MENAGERIE_DIR if set, else the copy bundled in assets/."""
    folders = ([Path(os.environ["MENAGERIE_DIR"])] if os.environ.get("MENAGERIE_DIR") else []) + [BUNDLED_MODEL]
    for folder in folders:
        scene = folder.expanduser().resolve() / "skydio_x2" / "scene.xml"
        if scene.is_file():
            return scene
    raise FileNotFoundError(f"Skydio X2 model not found in {folders}; restore assets/skydio_x2 "
                            "(https://github.com/google-deepmind/mujoco_menagerie) or set MENAGERIE_DIR.")


def load_model():
    return mujoco.MjModel.from_xml_path(str(find_scene()))


def drone_facts(model):
    """Mass, inertia (body frame, about the centre of mass) and the 4x4 allocation matrix, read from the model."""
    body = model.body("x2").id
    mass = float(model.body_subtreemass[body])
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[body])
    rot = rot.reshape(3, 3)
    inertia = rot @ np.diag(model.body_inertia[body]) @ rot.T
    com = model.body_ipos[body]
    alloc = np.zeros((4, model.nu))  # rows: Fz, Tx, Ty, Tz; columns: rotors 1-4
    for i in range(model.nu):
        r = model.site_pos[model.actuator_trnid[i, 0]] - com
        gear = model.actuator_gear[i]
        alloc[:, i] = [gear[2], r[1] * gear[2], -r[0] * gear[2], gear[5]]
    return mass, inertia, alloc


def _vee(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


class Controller:
    def __init__(self, gains: Gains, mass, inertia, alloc, dt, thrust_range):
        self.g, self.mass, self.inertia, self.dt = gains, mass, inertia, dt
        self.mixer = np.linalg.inv(alloc)
        self.lo, self.hi = thrust_range
        self.integral = np.zeros(3)  # integral of the position error, m s

    def integral_push(self):
        """The integral term's acceleration, m/s^2 (per axis, clipped to i_limit)."""
        return np.clip(self.g.ki * self.integral, -self.g.i_limit, self.g.i_limit)

    def act(self, pos, vel, rot, omega, target):
        g = self.g
        err = target - pos
        building = g.ki > 0 and (g.i_rule == "always" or np.linalg.norm(err) < g.i_zone)
        if building:
            self.integral += err * self.dt
            if np.isfinite(g.i_limit):  # stop building once the term is at its limit (no hidden wind-up)
                self.integral = np.clip(self.integral, -g.i_limit / g.ki, g.i_limit / g.ki)
        acc = g.kp * err - g.kd * vel + self.integral_push()
        force = self.mass * (acc + [0.0, 0.0, GRAVITY])
        fz = max(force[2], 0.1 * self.mass * GRAVITY)
        scale = min(1.0, fz * np.tan(MAX_TILT) / (np.linalg.norm(force[:2]) + 1e-12))
        force = np.array([force[0] * scale, force[1] * scale, fz])
        thrust = max(float(force @ rot[:, 2]), 0.0)

        b3 = force / np.linalg.norm(force)
        b2 = np.cross(b3, [1.0, 0.0, 0.0])
        b2 /= np.linalg.norm(b2)
        rot_d = np.column_stack([np.cross(b2, b3), b2, b3])
        e_rot = 0.5 * _vee(rot_d.T @ rot - rot.T @ rot_d)
        torque = -KP_ATT * e_rot - KD_ATT * omega + np.cross(omega, self.inertia @ omega)
        return np.clip(self.mixer @ np.concatenate([[thrust], torque]), self.lo, self.hi), building


def fly(gains: Gains, wind: float, model=None, log_every: int = 1):
    """Fly the mission in a steady wind of `wind` newtons. Returns (log dict of arrays, per-waypoint results)."""
    model = model or load_model()
    data = mujoco.MjData(model)
    mass, inertia, alloc = drone_facts(model)
    ctrl = Controller(gains, mass, inertia, alloc, model.opt.timestep, model.actuator_ctrlrange[0])
    body = model.body("x2").id
    data.qpos[:3] = WAYPOINTS[0]
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    steps_per_wp = int(round(HOLD_T / model.opt.timestep))
    n = steps_per_wp * (len(WAYPOINTS) - 1)
    keys = ["t", "wp", "target_x", "target_y", "target_z", "x", "y", "z", "vx", "vy", "vz", "err", "tilt_deg",
            "i_ax", "i_ay", "i_az", "building"]
    log = {k: [] for k in keys}
    end_err, crashed = [], False
    for k in range(n):
        wp = 1 + k // steps_per_wp
        target = WAYPOINTS[wp]
        pos, vel = data.qpos[:3].copy(), data.qvel[:3].copy()
        rot = data.xmat[body].reshape(3, 3).copy()
        motors, building = ctrl.act(pos, vel, rot, data.qvel[3:6].copy(), target)
        data.ctrl[:] = motors
        data.xfrc_applied[body, :3] = wind * WIND_DIR
        err = float(np.linalg.norm(target - pos))
        if k % log_every == 0:
            ia = ctrl.integral_push()
            row = [k * model.opt.timestep, wp, *target, *pos, *vel, err,
                   float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1)))), *ia, int(building)]
            for key, v in zip(keys, row):
                log[key].append(v)
        mujoco.mj_step(model, data)
        if (k + 1) % steps_per_wp == 0:  # end of this waypoint's hold: how far from it are we?
            end_err.append(float(np.linalg.norm(target - data.qpos[:3])))
        if data.qpos[2] < CRASH_MIN_Z:
            crashed = True
            break
    log = {k: np.asarray(v) for k, v in log.items()}
    return log, {"end_errors_m": end_err, "crashed": crashed}


def predicted_pd_offset(wind, mass, kp=Gains.kp):
    """Steady-state PD error: the controller pushes back kp*m*e, which balances the wind when e = F / (m kp)."""
    return wind / (mass * kp)


def gains_dict(g: Gains):
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in asdict(g).items()}
