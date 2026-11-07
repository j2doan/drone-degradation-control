"""Cascaded PID flight controller (numpy).

    outer loop: position error -> desired acceleration -> desired force (world) -> collective thrust
                                                       -> desired attitude (thrust axis along the force, yaw 0)
    inner loop: attitude error (on SO(3)) -> body torque
    mixer:      [thrust, torque] -> four motor thrusts (inverse of the allocation matrix)

Anti-windup: integrators only run while the drone is close to and moving slowly relative to the target, and they
freeze for one step whenever the last command hit a motor limit. The gains are fixed values tuned
for the X2.
"""

from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class Gains:
    kp_pos: np.ndarray = field(default_factory=lambda: np.array([3.0, 3.0, 1.75]))  # 1/s^2
    ki_pos: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 1.0]))  # 1/s^3
    kd_pos: np.ndarray = field(default_factory=lambda: np.array([3.0, 3.0, 1.75]))  # 1/s (on velocity error)
    i_pos_limit: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.3, 1.5]))  # m s
    i_vel_zone: float = 0.2  # m/s, integrate an axis only while its velocity error is below this ...
    i_pos_zone: float = 1.0  # m, ... and its position error below this
    kp_att: np.ndarray = field(default_factory=lambda: np.array([1.725, 1.725, 0.325]))  # N m / rad
    ki_att: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.2]))  # N m / (rad s)
    kd_att: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.45, 0.0825]))  # N m / (rad/s)
    i_att_limit: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 1.0]))  # rad s
    max_tilt: float = np.radians(60.0)

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


def _vee(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def desired_rotation(force, yaw: float = 0.0):
    """Rotation whose body z axis points along `force`, heading `yaw`."""
    b3 = force / (np.linalg.norm(force) + 1e-12)
    b2 = np.cross(b3, [np.cos(yaw), np.sin(yaw), 0.0])
    b2 /= np.linalg.norm(b2) + 1e-12
    return np.column_stack([np.cross(b2, b3), b2, b3])


class PID:
    def __init__(self, mass, inertia, allocation, dt=config.SIM_DT, cap=config.MAX_THRUST, gains=None):
        self.mass, self.inertia, self.dt, self.cap = float(mass), np.asarray(inertia), dt, cap
        self.mixer = np.linalg.inv(allocation)  # [Fz, Tx, Ty, Tz] -> motor thrusts
        self.g = gains or Gains()
        self.reset()

    def reset(self):
        self.int_pos = np.zeros(3)
        self.int_att = np.zeros(3)
        self.saturated = False

    def act(self, pos, vel, rot, omega_body, target_pos, target_vel):
        """Motor thrust request (4,) [N] from the state (world position/velocity, body->world rotation matrix,
        body angular velocity) and the target point (position, velocity)."""
        g = self.g
        integrate = 0.0 if self.saturated else 1.0

        # outer loop: position -> force
        e_pos, e_vel = target_pos - pos, target_vel - vel
        zone = ((np.abs(e_vel) < g.i_vel_zone) & (np.abs(e_pos) < g.i_pos_zone)) * integrate
        self.int_pos = np.clip(self.int_pos + zone * e_pos * self.dt, -g.i_pos_limit, g.i_pos_limit)
        acc = g.kp_pos * e_pos + g.ki_pos * self.int_pos + g.kd_pos * e_vel
        force = self.mass * (acc + np.array([0.0, 0.0, config.GRAVITY]))
        # A quadrotor can only push up: keep a small minimum vertical force, then limit the tilt.
        fz = max(force[2], 0.1 * self.mass * config.GRAVITY)
        fxy = force[:2]
        scale = min(1.0, fz * np.tan(g.max_tilt) / (np.linalg.norm(fxy) + 1e-12))
        force = np.array([fxy[0] * scale, fxy[1] * scale, fz])
        thrust = max(float(force @ rot[:, 2]), 0.0)  # the part of the force along the current thrust axis

        # inner loop: attitude -> torque
        rot_d = desired_rotation(force)
        e_rot = 0.5 * _vee(rot_d.T @ rot - rot.T @ rot_d)
        self.int_att = np.clip(self.int_att + integrate * e_rot * self.dt, -g.i_att_limit, g.i_att_limit)
        torque = (-g.kp_att * e_rot - g.ki_att * self.int_att - g.kd_att * omega_body
                  + np.cross(omega_body, self.inertia @ omega_body))

        request = self.mixer @ np.concatenate([[thrust], torque])
        self.saturated = bool(np.any((request > self.cap) | (request < 0.0)))  # freeze integrators next step
        return request
