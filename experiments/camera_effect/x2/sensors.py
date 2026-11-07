"""IMU noise and the front camera.

IMU: MuJoCo gives perfect readings (the X2's own gyro and accelerometer at its "imu" site). A real MEMS IMU adds a
small constant offset per flight (bias) and fresh jitter on every sample (white noise); both are added here, seeded,
so the same flight always gets the same noise. Both clean and noisy readings are logged.

Camera: renders the model's "front" camera (mounted on the body, looking forward) at CAMERA_RES x CAMERA_RES.
Rendering only reads the simulation state; it never changes it.
"""

import mujoco
import numpy as np

from . import config


class ImuNoise:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng([seed, 2026])  # its own stream, independent of the route generator
        self.accel_bias = self.rng.normal(0.0, config.IMU_ACCEL_BIAS, 3)
        self.gyro_bias = self.rng.normal(0.0, config.IMU_GYRO_BIAS, 3)

    def apply(self, accel, gyro):
        return (accel + self.accel_bias + self.rng.normal(0.0, config.IMU_ACCEL_NOISE, 3),
                gyro + self.gyro_bias + self.rng.normal(0.0, config.IMU_GYRO_NOISE, 3))


class FrontCamera:
    def __init__(self, model, resolution: int = config.CAMERA_RES):
        self.renderer = mujoco.Renderer(model, resolution, resolution)

    def capture(self, data) -> np.ndarray:
        self.renderer.update_scene(data, camera="front")
        return self.renderer.render().copy()  # (H, W, 3) uint8

    def close(self):
        self.renderer.close()
