"""IMU noise: a constant bias per flight plus fresh white noise every reading.

Seeded and batched.

Isaac's IMU is perfect. A real MEMS IMU is not: each axis has a small offset (bias) that
stays roughly constant over one flight, plus random jitter (white noise) on every
sample. ImuNoise adds both to the clean readings `lin_acc_b` (proper acceleration, body
frame, ~+9.81 on z at hover) [m/s^2] and `ang_vel_b` [rad/s], both (N,3).

Seeded with its own torch.Generator, so the same seed gives the same noise sequence,
independent of the other seeded streams (faults, PPO, torch's global RNG).

Simulator-independent: torch only (no isaaclab / isaacsim / omni / warp).

Usage:
    imu_noise = ImuNoise(seed, num_envs=N, device=dev)
    noisy_acc, noisy_gyro = imu_noise.apply(lin_acc_b, ang_vel_b)  # every step
    imu_noise.reset(env_ids)  # new flight -> new bias
"""

from dataclasses import dataclass

import torch

# Plausible values for a small drone's consumer MEMS IMU, after factory calibration and
# at a ~100 Hz sample rate. Illustrative orders of magnitude, not from a datasheet.
ACCEL_NOISE_STD = 0.05  # m/s^2, white noise per sample
GYRO_NOISE_STD = 0.005  # rad/s, white noise per sample
ACCEL_BIAS_STD = 0.02  # m/s^2, spread of the per-flight constant bias
GYRO_BIAS_STD = 0.002  # rad/s, spread of the per-flight constant bias
# The generator is seeded with seed * SEED_MULT + SEED_OFFSET, so seed k here never
# equals seed k of another stream.
SEED_MULT, SEED_OFFSET = 1000, 11


@dataclass
class ImuNoiseCfg:
    accel_noise_std: float = ACCEL_NOISE_STD
    gyro_noise_std: float = GYRO_NOISE_STD
    accel_bias_std: float = ACCEL_BIAS_STD
    gyro_bias_std: float = GYRO_BIAS_STD


class ImuNoise:
    """Batched, seeded IMU noise.

    noisy = clean + bias (constant per env until reset) + white noise (each call).
    """

    def __init__(
        self,
        seed: int,
        num_envs: int = 1,
        cfg: "ImuNoiseCfg | None" = None,
        device="cpu",
    ):
        self.num_envs, self.device = num_envs, device
        self.cfg = cfg if cfg is not None else ImuNoiseCfg()
        # a CPU torch.Generator works everywhere; samples are moved to `device` after
        self.gen = torch.Generator(device="cpu").manual_seed(
            int(seed) * SEED_MULT + SEED_OFFSET
        )
        self.accel_bias = torch.zeros(num_envs, 3, device=device)
        self.gyro_bias = torch.zeros(num_envs, 3, device=device)
        self.reset()

    def _randn(self, n: int) -> torch.Tensor:
        return torch.randn(n, 3, generator=self.gen).to(self.device)

    def reset(self, env_ids=None) -> None:
        """Draw a new constant bias for env_ids (all envs when None)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)
        n = len(env_ids)
        idx = env_ids.to(self.device)
        self.accel_bias[idx] = self._randn(n) * self.cfg.accel_bias_std
        self.gyro_bias[idx] = self._randn(n) * self.cfg.gyro_bias_std

    def apply(self, lin_acc: torch.Tensor, ang_vel: torch.Tensor):
        """Clean (N,3) accel [m/s^2] and gyro [rad/s] -> new noisy tensors.

        The inputs are not modified.
        """
        acc_noise = self._randn(self.num_envs) * self.cfg.accel_noise_std
        gyro_noise = self._randn(self.num_envs) * self.cfg.gyro_noise_std
        noisy_acc = lin_acc + self.accel_bias.to(lin_acc) + acc_noise.to(lin_acc)
        noisy_gyro = ang_vel + self.gyro_bias.to(ang_vel) + gyro_noise.to(ang_vel)
        return noisy_acc, noisy_gyro
