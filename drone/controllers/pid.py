"""Cascaded PID position controller for a quadrotor, in pure PyTorch.

No Isaac Lab / Isaac Sim / warp imports, so the same code runs in Isaac Lab, MuJoCo, or
a toy simulator, and batched over many drones (every tensor has a leading num_envs
dimension N).

Structure (mirrors Isaac Lab's Lee position controller so the output drops into the same
pipeline):
  outer loop: position PID -> desired acceleration -> desired force F (world)
      -> collective thrust Fz and desired orientation R_d (thrust along F, yaw 0)
  inner loop: attitude PID -> body torque [Tx, Ty, Tz]
Output is the body-frame wrench [Fx, Fy, Fz, Tx, Ty, Tz] with Fx = Fy = 0; the caller
maps it to propeller thrusts (e.g. pseudoinverse of the allocation matrix), so nothing
here knows the motor layout.

Conventions: world z up, gravity along world -z, quaternions in (x, y, z, w) order,
linear velocity in the world frame, angular velocity in the body frame.
"""

from dataclasses import dataclass

import torch

# ---- tunable constants
GRAVITY = 9.81  # m/s^2, along world -z
MAX_TILT = 1.047  # rad (60 deg), same as the Lee config
YAW_SETPOINT = 0.0  # rad


@dataclass
class PIDGains:
    """Per-axis gains (x, y, z). Defaults start from Isaac Lab's Lee gains."""

    # Outer loop: position -> acceleration [1/s^2, 1/s^3, 1/s]
    kp_pos: tuple = (3.0, 3.0, 1.75)
    # z is strong enough to win back altitude after losing up to ~50% of one motor (a
    # steady thrust deficit); x, y only trim small offsets.
    ki_pos: tuple = (0.05, 0.05, 1.0)
    kd_pos: tuple = (3.0, 3.0, 1.75)  # acts on velocity error (sp_vel - vel)
    # anti-windup: clamp of the integrated position error per axis [m*s]
    i_pos_limit: tuple = (0.3, 0.3, 1.5)
    # Conditional integration (anti-windup): an axis only integrates while the drone
    # moves slowly on that axis (velocity error below i_vel_zone) and is not wildly off
    # (position error below i_pos_zone). This skips the fast start-up sag but still
    # catches a slow, stuck offset of any size.
    i_vel_zone: float = 0.2  # m/s
    i_pos_zone: float = 1.0  # m
    # Inner loop: attitude error -> torque. Like Lee, these are torques per rad (per
    # rad/s), not pre-multiplied by inertia.
    kp_att: tuple = (1.725, 1.725, 0.325)  # [N*m/rad]
    # The integral lets the controller hold a constant extra torque without a leftover
    # error. A weak motor needs exactly that; without it the drone settled ~0.26 m away
    # from B, level but offset, and turned ~9 deg.
    ki_att: tuple = (1.0, 1.0, 0.2)  # [N*m/(rad*s)]
    kd_att: tuple = (0.45, 0.45, 0.0825)  # on body rate [N*m/(rad/s)]
    # anti-windup clamp of the integrated attitude error per axis [rad*s]
    i_att_limit: tuple = (0.5, 0.5, 1.0)
    max_tilt: float = MAX_TILT


# ---- small rotation helpers (xyzw)
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) xyzw quaternion -> (N,3,3) matrix R (body->world; columns = body axes)."""
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a*b of xyzw quaternions."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def vee(S: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric (N,3,3) -> (N,3) vector."""
    return torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], dim=-1)


def desired_rotation(force_w: torch.Tensor, yaw: float) -> torch.Tensor:
    """Rotation whose body z points along force_w with heading `yaw`.

    Same as Lee's compute_desired_orientation.
    """
    b3 = force_w / (force_w.norm(dim=-1, keepdim=True) + 1e-12)
    heading = torch.zeros_like(force_w)
    heading[:, 0] = torch.cos(torch.tensor(yaw))
    heading[:, 1] = torch.sin(torch.tensor(yaw))
    b2 = torch.cross(b3, heading, dim=-1)
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-12)
    b1 = torch.cross(b2, b3, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


# ---- controller
class CascadedPID:
    """Batched cascaded PID: position -> thrust + desired attitude -> body torque."""

    def __init__(
        self,
        num_envs: int,
        mass: float,
        inertia,
        dt: float,
        device="cpu",
        gains: "PIDGains | None" = None,
    ):
        self.num_envs, self.mass, self.dt, self.device = (
            num_envs,
            float(mass),
            float(dt),
            device,
        )
        self.gains = gains if gains is not None else PIDGains()
        g = self.gains

        def vec(t):
            return torch.tensor(t, dtype=torch.float32, device=device).expand(
                num_envs, 3
            )

        self.kp_pos, self.ki_pos, self.kd_pos = (
            vec(g.kp_pos),
            vec(g.ki_pos),
            vec(g.kd_pos),
        )
        self.kp_att, self.ki_att, self.kd_att = (
            vec(g.kp_att),
            vec(g.ki_att),
            vec(g.kd_att),
        )
        self.i_pos_limit, self.i_att_limit = vec(g.i_pos_limit), vec(g.i_att_limit)

        inertia = torch.as_tensor(inertia, dtype=torch.float32, device=device)
        # (N,3,3)
        self.inertia = inertia.expand(num_envs, 3, 3) if inertia.dim() == 2 else inertia
        self.g_up = torch.tensor([0.0, 0.0, GRAVITY], device=device).expand(num_envs, 3)
        # tan(max_tilt) bounds horizontal force relative to vertical force
        self.tan_max_tilt = torch.tan(torch.tensor(g.max_tilt)).item()

        self.int_pos = torch.zeros(num_envs, 3, device=device)
        self.int_att = torch.zeros(num_envs, 3, device=device)
        # Anti-windup at the motor cap: True = the last command of this env was clipped
        # at a motor limit, so the next compute() freezes its integrators (see
        # set_saturated). Consumed by compute().
        self.saturated = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None) -> None:
        """Zero the integrators and saturation flag for env_ids (all envs when None)."""
        if env_ids is None:
            env_ids = slice(None)
        self.int_pos[env_ids] = 0.0
        self.int_att[env_ids] = 0.0
        self.saturated[env_ids] = False

    def set_saturated(self, mask) -> None:
        """Mark envs whose last command was clipped at a motor limit (bool (N,)).

        The caller clips the motor thrusts after this controller (e.g. at 6 N per
        motor); while a motor is clipped, integrating the error only winds the
        integrators up and causes overshoot later. On the next compute() the marked envs
        skip both integral updates (position and attitude); P and D act as usual. The
        mask is used once.
        """
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(
            self.num_envs
        )
        self.saturated = mask.clone()

    def compute(self, pos, vel, quat_xyzw, ang_vel_b, sp_pos, sp_vel) -> torch.Tensor:
        """Return the (N,6) body-frame wrench [Fx, Fy, Fz, Tx, Ty, Tz] (Fx = Fy = 0)."""
        g = self.gains
        R = quat_to_rotmat(quat_xyzw)  # body -> world
        body_z = R[:, :, 2]
        # 0 for envs flagged by set_saturated (integrators frozen this step), 1
        # otherwise; then consume the flag
        integrate = (~self.saturated).float().unsqueeze(-1)
        self.saturated = torch.zeros_like(self.saturated)

        # ---- outer loop: position PID -> desired acceleration -> desired world force
        pos_err = sp_pos - pos
        vel_err = sp_vel - vel
        in_zone = (
            (vel_err.abs() < g.i_vel_zone) & (pos_err.abs() < g.i_pos_zone)
        ).float()  # conditional integration
        in_zone = in_zone * integrate
        self.int_pos = torch.maximum(
            torch.minimum(self.int_pos + in_zone * pos_err * self.dt, self.i_pos_limit),
            -self.i_pos_limit,
        )
        a_des = (
            self.kp_pos * pos_err + self.ki_pos * self.int_pos + self.kd_pos * vel_err
        )
        F = self.mass * (a_des + self.g_up)  # thrust must also cancel gravity

        # Tilt limit: keep F within MAX_TILT of world up. Never ask for downward thrust
        # (a quadrotor cannot), keep a small minimum vertical part, then shrink the
        # horizontal part if needed.
        Fz_w = F[:, 2].clamp(min=0.1 * self.mass * GRAVITY)
        Fxy = F[:, :2]
        max_xy = Fz_w * self.tan_max_tilt
        scale = (max_xy / (Fxy.norm(dim=-1) + 1e-12)).clamp(max=1.0)
        F = torch.cat([Fxy * scale.unsqueeze(-1), Fz_w.unsqueeze(-1)], dim=-1)

        # Collective thrust = component of F along the current body z (as Lee does).
        # Rotors only push.
        thrust = (F * body_z).sum(-1).clamp(min=0.0)

        # ---- inner loop: attitude PID on the SO(3) error
        R_d = desired_rotation(F, YAW_SETPOINT)
        # e_R = 0.5 vee(R_d^T R - R^T R_d): how far R is rotated past R_d (Lee's sign)
        e_R = 0.5 * vee(R_d.transpose(1, 2) @ R - R.transpose(1, 2) @ R_d)
        self.int_att = torch.maximum(
            torch.minimum(self.int_att + integrate * e_R * self.dt, self.i_att_limit),
            -self.i_att_limit,
        )
        # desired body rate = 0 (hover / slow setpoint tracking), as in Lee's controller
        e_w = ang_vel_b
        # Coriolis feedforward w x (I w), exactly like Lee's compute_body_torque.
        Iw = torch.bmm(self.inertia, ang_vel_b.unsqueeze(-1)).squeeze(-1)
        torque = (
            -self.kp_att * e_R
            - self.ki_att * self.int_att
            - self.kd_att * e_w
            + torch.cross(ang_vel_b, Iw, dim=-1)
        )

        wrench = torch.zeros(self.num_envs, 6, device=self.device)
        wrench[:, 2] = thrust
        wrench[:, 3:] = torque
        return wrench
