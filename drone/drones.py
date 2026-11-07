"""The four drones of the study and the slot where a learned policy plugs in.

  d0: PID baseline, mode "pid": PID only (no policy needed; always available)
  d1: PID + residual PPO, mode "residual": PID thrust + a bounded PPO correction
  d2: PID + residual PPO + memory, mode "residual": same, plus recent history
  d3: PPO (no controller), mode "replace": PPO outputs the motor thrusts itself

The batch runner loops over available(): d0 now, d1-d3 as soon as register_policy()
gives them a policy. combine() merges the PID thrust and the policy output per mode;
make_observation() builds what a policy sees.

Simulator-independent: torch only (no isaaclab / isaacsim / omni / warp).

Usage (inside the flight loop):
    for spec in available():
        obs = make_observation(...)  # (N, len(OBS_NAMES))
        out = spec.policy.act(obs) if spec.policy is not None else None
        # the caller clamps
        thrust = combine(pid_thrust, out, spec.mode).clamp(0.0, MAX_MOTOR_THRUST)
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

NUM_MOTORS = 4
MODES = ("pid", "residual", "replace")
# Newtons per motor for a residual output of 1. Limit how much PPO can change, so it can
# nudge the PID but never wreck it: +-1 N is ~1/3 of the ~3.04 N hover thrust per motor,
# enough to compensate a weak motor, not enough to flip.
RESIDUAL_SCALE = 1.0


@runtime_checkable
class Policy(Protocol):
    """What a learned policy must offer to fly one of the drones.

    act(obs) takes the (N, len(OBS_NAMES)) observation and returns (N, NUM_MOTORS):
      residual policies (d1, d2): a correction in [-1, 1] per motor, scaled by
          RESIDUAL_SCALE and added to the PID;
      replacement policies (d3): the motor thrusts directly, in newtons.
    reset(env_ids=None) clears any internal state (e.g. memory) for those envs at the
    start of a flight.
    """

    def act(self, obs: torch.Tensor) -> torch.Tensor: ...

    def reset(self, env_ids=None) -> None: ...


@dataclass
class DroneSpec:
    name: str  # "d0".."d3"
    label: str  # human-readable, for plots and tables
    mode: str  # one of MODES
    policy: Any = None  # a Policy, or None (not trained yet)

    @property
    def ready(self) -> bool:
        """Can it fly now? The PID-only drone always can; the others need a policy."""
        return self.mode == "pid" or self.policy is not None


DRONES = {
    "d0": DroneSpec("d0", "PID baseline", "pid"),
    "d1": DroneSpec("d1", "PID + residual PPO", "residual"),
    "d2": DroneSpec("d2", "PID + residual PPO + memory", "residual"),
    "d3": DroneSpec("d3", "PPO (no controller)", "replace"),
}


def available() -> list:
    """DroneSpecs that can fly now, in name order: d0 plus every drone with a policy."""
    return [DRONES[k] for k in sorted(DRONES) if DRONES[k].ready]


def register_policy(name: str, policy) -> DroneSpec:
    """Plug a trained policy into drone `name` (e.g. "d1"); it joins available()."""
    if name not in DRONES:
        raise KeyError(f"unknown drone {name!r}; expected one of {sorted(DRONES)}")
    if DRONES[name].mode == "pid":
        raise ValueError(f"{name} is the PID-only baseline and takes no policy")
    if not isinstance(policy, Policy):
        raise TypeError("policy must have act(obs) and reset(env_ids=None)")
    DRONES[name].policy = policy
    return DRONES[name]


def combine(
    pid_thrust: torch.Tensor,
    policy_out,
    mode: str,
    residual_scale: float = RESIDUAL_SCALE,
) -> torch.Tensor:
    """Motor thrusts (N, NUM_MOTORS) for one drone mode.

    The caller still clamps the result to [0, cap].

    "pid": pid_thrust;  "residual": pid_thrust + residual_scale * policy_out (policy_out
    in [-1, 1]); "replace": policy_out (thrusts in newtons; pid_thrust is ignored and
    may be None).
    """
    if mode == "pid":
        return pid_thrust
    if mode == "residual":
        return pid_thrust + residual_scale * policy_out
    if mode == "replace":
        return policy_out
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


# Observation layout. Order matters: it is the column order of make_observation().
OBS_NAMES = (
    [f"target_rel_pos_{a}" for a in "xyz"]  # target_pos - pos, world frame [m]
    + [f"target_rel_vel_{a}" for a in "xyz"]  # target_vel - lin_vel, world frame [m/s]
    + [f"lin_vel_{a}" for a in "xyz"]  # world-frame linear velocity [m/s]
    + [f"quat_{a}" for a in "xyzw"]  # orientation (x, y, z, w)
    + [f"ang_vel_{a}" for a in "xyz"]  # body-frame angular velocity [rad/s]
    # IMU proper acceleration, body frame (noisy if noise is on) [m/s^2]
    + [f"imu_lin_acc_{a}" for a in "xyz"]
    # IMU angular velocity, body frame (noisy if noise is on) [rad/s]
    + [f"imu_ang_vel_{a}" for a in "xyz"]
    # thrust commanded on the previous step, after the cap [N]
    + [f"last_thrust_{i}" for i in range(NUM_MOTORS)]
)


def make_observation(
    pos,
    lin_vel,
    quat_xyzw,
    ang_vel_b,
    imu_acc,
    imu_gyro,
    target_pos,
    target_vel,
    last_thrust,
) -> torch.Tensor:
    """What a policy sees each step: an (N, len(OBS_NAMES)) float tensor.

    Columns are in OBS_NAMES order.

    The target is given RELATIVE to the drone (target_pos - pos, target_vel - lin_vel),
    so the same policy works on any route; the absolute position itself is not included.
    The observation NEVER includes the fault (which motor, how much loss, when it
    started): the policy must infer it from how the drone behaves (IMU, attitude,
    response to its own last thrust), like a real drone would. A memory policy (D2)
    stacks the last few of these observations itself.
    """
    parts = [
        target_pos - pos,
        target_vel - lin_vel,
        lin_vel,
        quat_xyzw,
        ang_vel_b,
        imu_acc,
        imu_gyro,
        last_thrust,
    ]
    obs = torch.cat([p.to(dtype=torch.float32) for p in parts], dim=-1)
    assert obs.shape[-1] == len(OBS_NAMES), (obs.shape, len(OBS_NAMES))
    return obs
