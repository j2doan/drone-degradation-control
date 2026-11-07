"""The ARL Robot 1 motor model, ported from Isaac Lab for MuJoCo.

MuJoCo's motors then behave like Isaac's.

A real propeller cannot change its thrust instantly: it has to spin up or down first.
Isaac Lab models this per motor (isaaclab_contrib/actuators/thruster.py,
Thruster.compute): the command is clamped to the motor's 0.1-10 N range, turned into a
spin rate (thrust = thrust_const * rpm^2), and the spin rate follows the command with a
time constant (tau_inc when speeding up, the much shorter tau_dec when slowing down),
integrated with RK4. The same equations are used here.

One difference: Isaac draws each motor's thrust_const and tau_inc at random without a
seed. Here they are drawn from the flight's seed (own RNG stream [seed, 13]), so a
MuJoCo flight is exactly repeatable.

Simulator-independent: numpy + torch only.

Usage:
    motors = ThrusterModel(dt=0.01)
    motors.reset(seed)  # new parameters + initial spin, once per flight
    # every step: (N, 4) commanded thrust -> (N, 4) delivered thrust
    thrust_out = motors.step(thrust_cmd)
"""

import numpy as np
import torch

# ARL Robot 1 values (isaaclab_assets/robots/arl_robot_1.py: thruster cfg, init_state)
THRUST_RANGE = (0.1, 10.0)  # N
THRUST_CONST_RANGE = (9.26312e-06, 1.826312e-05)  # N / rpm^2
TAU_INC_RANGE = (0.05, 0.08)  # s, spin-up time constant
TAU_DEC_RANGE = (0.005, 0.005)  # s, spin-down time constant
INIT_RPS = 200.0  # initial spin of every propeller after a reset
MAX_RATE = 100000.0  # ThrusterCfg.max_thrust_rate default
NUM_MOTORS = 4


class ThrusterModel:
    """Batched first-order motor lag in rpm space, as in Isaac Lab's Thruster.

    Discrete mixing factor 1 / (dt + tau), RK4 integration.
    """

    def __init__(self, dt: float, num_envs: int = 1, device="cpu"):
        self.dt, self.num_envs, self.device = float(dt), num_envs, device
        self.reset(0)

    def reset(self, seed: int) -> None:
        """Draw this flight's motor parameters from the seed.

        Every propeller is set back to its initial spin.
        """
        rng = np.random.default_rng([seed, 13])
        shape = (self.num_envs, NUM_MOTORS)

        def draw(lo_hi):
            values = rng.uniform(*lo_hi, size=shape)
            return torch.tensor(values, dtype=torch.float32, device=self.device)

        self.tau_inc = draw(TAU_INC_RANGE)
        self.tau_dec = draw(TAU_DEC_RANGE)
        self.thrust_const = draw(THRUST_CONST_RANGE).clamp(min=1e-6)
        self.thrust = self.thrust_const * INIT_RPS**2  # current thrust state [N]

    def _rate(self, error, mixing):
        return torch.clamp(mixing * error, -MAX_RATE, MAX_RATE)

    def step(self, thrust_cmd: torch.Tensor) -> torch.Tensor:
        """Advance one step towards the commanded thrust (N, 4) [N].

        Returns the delivered thrust (N, 4) [N].
        """
        des = torch.clamp(thrust_cmd.to(self.thrust), *THRUST_RANGE)
        slowing_down = torch.sign(self.thrust) * torch.sign(des - self.thrust) < 0
        tau = torch.where(slowing_down, self.tau_dec, self.tau_inc)
        mixing = 1.0 / (self.dt + tau)  # discrete approximation, Isaac's default
        rpm = torch.sqrt(torch.clamp(self.thrust / self.thrust_const, min=0.0))
        error = torch.sqrt(torch.clamp(des / self.thrust_const, min=0.0)) - rpm
        # RK4 as Isaac writes it (each stage re-evaluates the rate at a shifted error)
        k1 = self._rate(error, mixing)
        k2 = self._rate(error - 0.5 * self.dt * k1, mixing)
        k3 = self._rate(error - 0.5 * self.dt * k2, mixing)
        k4 = self._rate(error - self.dt * k3, mixing)
        rpm = rpm + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.thrust = self.thrust_const * rpm**2
        return torch.clamp(self.thrust, *THRUST_RANGE)
