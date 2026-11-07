"""Sensors and state estimation for the camera on/off experiment.

The PID (x2/controller.py) normally reads the drone's true position and velocity.
Here it reads an *estimate* built from realistic sensors, so that a camera can add information:

    IMU            100 Hz   accelerometer (noisy, biased; from x2.sensors.ImuNoise) drives the prediction
    GPS             5 Hz    x, y, z with noise (GPS_SIGMA)
    Barometer      20 Hz    z with noise (BARO_SIGMA)
    Down camera    20 Hz    height from the floor's checker pattern, ground velocity from image shift   [camera on]

A linear Kalman filter per axis (state: position, velocity; input: world acceleration from the accelerometer)
fuses whatever measurements arrive. Attitude and body rates are taken as known (a good attitude estimator is
assumed in every condition), so the experiment isolates what the camera adds to position and velocity.
"""

import mujoco
import numpy as np

from x2 import config
from x2.model import find_scene

# ---------------------------------------------------------------- sensor settings
GPS_PERIOD, GPS_SIGMA = 0.2, np.array([0.30, 0.30, 0.40])  # s, m (x, y, z)
BARO_PERIOD, BARO_SIGMA = 0.05, 0.20  # s, m
CAM_PERIOD, CAM_RES, CAM_FOVY = 0.05, 160, 90.0  # s (20 Hz), px, deg
ACCEL_PROCESS_SIGMA = 0.15  # m/s^2: accelerometer noise + bias + model error, used as Kalman process noise
CAM_HEIGHT_RANGE = (0.25, 4.0)  # m, camera height readings outside this are rejected
GATE_SIGMAS = 5.0  # reject a measurement whose innovation exceeds this many standard deviations


def build_model():
    """The X2 scene plus a downward camera at the body origin (the drone's feet), looking straight down when level.
    A camera has no mass or collision: the physics is the same as without it.

    The model's own spotlight (mounted about 2 m up and aimed at the drone) is switched off: from above it, it
    blacks out part of the floor in the downward view. Lights do not affect the physics."""
    spec = mujoco.MjSpec.from_file(str(find_scene()))
    spec.body("x2").add_camera(name="down", pos=[0.0, 0.0, -0.005], quat=[1.0, 0.0, 0.0, 0.0], fovy=CAM_FOVY)
    model = spec.compile()
    model.light_active[model.light("spotlight").id] = 0
    return model


# ---------------------------------------------------------------- image processing
def _focal_px(res=CAM_RES, fovy=CAM_FOVY):
    return (res / 2) / np.tan(np.radians(fovy) / 2)


WINDOW = np.outer(np.hanning(CAM_RES), np.hanning(CAM_RES))
PAD = 512
MIN_FREQ_BINS = 5  # ignore frequencies below 5/512 cycles/px (slower than the floor pattern at any height >= 0.25 m)


def checker_frequency(gray):
    """Strongest spatial frequency of the image [cycles/pixel] (sub-bin accurate). The checker floor gives one
    clear peak; its frequency is inversely proportional to the distance from the camera to the floor."""
    g = (gray - gray.mean()) * WINDOW
    power = np.fft.fftshift(np.abs(np.fft.fft2(g, s=(PAD, PAD))))
    c = PAD // 2
    yy, xx = np.ogrid[:PAD, :PAD]
    power[(yy - c) ** 2 + (xx - c) ** 2 < MIN_FREQ_BINS ** 2] = 0.0  # drop DC and slow brightness gradients
    ky, kx = np.unravel_index(np.argmax(power), power.shape)

    def refine(m1, m0, p1):  # parabolic peak interpolation
        den = m1 - 2 * m0 + p1
        return 0.0 if den == 0 else 0.5 * (m1 - p1) / den

    ky = min(max(ky, 1), PAD - 2)
    kx = min(max(kx, 1), PAD - 2)
    dy = refine(power[ky - 1, kx], power[ky, kx], power[ky + 1, kx])
    dx = refine(power[ky, kx - 1], power[ky, kx], power[ky, kx + 1])
    return float(np.hypot(kx + dx - c, ky + dy - c) / PAD)


def correlation_surface(prev, cur):
    """Phase-correlation surface between two frames; its peak sits at the translation of `cur` relative to `prev`
    (index [row, col], wrapping around)."""
    a = np.fft.fft2((prev - prev.mean()) * WINDOW)
    b = np.fft.fft2((cur - cur.mean()) * WINDOW)
    cross = b * np.conj(a)
    return np.real(np.fft.ifft2(cross / (np.abs(cross) + 1e-9)))


def peak_near(corr, guess_col, guess_row, radius):
    """Sub-pixel peak of the correlation surface within `radius` px of a guessed shift (d_col, d_row).

    The floor is a checkerboard, so shifts that differ by one square look identical and the surface has a peak for
    each of them. Searching only near the shift predicted from the current velocity estimate picks the right one.
    """
    n = corr.shape[0]
    offsets = np.arange(-int(np.ceil(radius)), int(np.ceil(radius)) + 1)
    best, best_val = (0, 0), -np.inf
    gc, gr = int(round(guess_col)), int(round(guess_row))
    for dr in offsets:
        for dc in offsets:
            if (gc + dc - guess_col) ** 2 + (gr + dr - guess_row) ** 2 > radius ** 2:
                continue
            v = corr[(gr + dr) % n, (gc + dc) % n]
            if v > best_val:
                best, best_val = (gc + dc, gr + dr), v
    c, r = best

    def sub(m1, m0, p1):
        den = m1 - 2 * m0 + p1
        return 0.0 if den == 0 else 0.5 * (m1 - p1) / den

    dr_ = sub(corr[(r - 1) % n, c % n], corr[r % n, c % n], corr[(r + 1) % n, c % n])
    dc_ = sub(corr[r % n, (c - 1) % n], corr[r % n, c % n], corr[r % n, (c + 1) % n])
    return c + dc_, r + dr_


class DownCamera:
    """Renders the downward camera (no shadows or floor reflections) and turns frames into height and ground-velocity readings.

    Height: distance along the optical axis = image_frequency * focal / GROUND_FREQ (a farther floor packs more
    checker cycles into each pixel), then x cos(tilt).
    Velocity: the floor point at the image centre in the previous frame is found again in the current frame (image
    shift); intersecting both viewing rays with the floor, using the known camera attitude, gives the horizontal
    displacement with the drone's own rotation removed.
    """

    GROUND_FREQ = 2.5 * np.sqrt(2)  # cycles/m of the floor pattern's strongest component (checked in calibrate())

    def __init__(self, model):
        self.renderer = mujoco.Renderer(model, CAM_RES, CAM_RES)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False  # the drone's shadow would move with it
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False  # so would its mirror image in the floor
        self.cam_id = model.camera("down").id
        self.focal = _focal_px()
        self.prev = None  # (gray image, camera rotation, height) of the previous frame

    def render(self, data):
        self.renderer.update_scene(data, camera="down")
        rgb = self.renderer.render()
        return rgb.mean(axis=2).astype(np.float32), rgb

    def _ray(self, rot, col_off=0.0, row_off=0.0):
        """World direction of the ray through pixel (centre + offset). Image x = camera x, image rows go down
        (camera -y), the camera looks along its -z."""
        d = rot @ np.array([col_off / self.focal, -row_off / self.focal, -1.0])
        return d / np.linalg.norm(d)

    def _pixel_of(self, rot, rel):
        """Pixel offset (col, row) from the image centre at which a point at world offset `rel` from the camera
        appears."""
        c = rot.T @ rel
        return self.focal * c[0] / -c[2], -self.focal * c[1] / -c[2]

    def measure(self, data, predicted_vel_xy):
        """Returns (height or nan, horizontal velocity (2,) or nans, gray frame, rgb frame).

        `predicted_vel_xy` is the estimator's current horizontal velocity; it is only used to pick the right one of
        the checkerboard's look-alike shifts (see peak_near), not as a measurement.
        """
        gray, rgb = self.render(data)
        rot = data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        centre = self._ray(rot)
        height = np.nan
        if centre[2] < -0.5:  # camera points at least 60 deg below the horizon
            freq = checker_frequency(gray)
            if freq > 0:
                height = freq * self.focal / self.GROUND_FREQ * -centre[2]
        if not CAM_HEIGHT_RANGE[0] <= height <= CAM_HEIGHT_RANGE[1]:
            height = np.nan
        vel = np.full(2, np.nan)
        if self.prev is not None and np.isfinite(height) and np.isfinite(self.prev[2]):
            prev_gray, prev_rot, prev_h = self.prev
            d0 = self._ray(prev_rot)
            if d0[2] < -0.3:
                g0 = prev_h * d0[:2] / -d0[2]  # floor point under the previous image centre, relative to the drone
                # where that point should appear now if the drone moved at the predicted velocity
                moved = g0 - np.asarray(predicted_vel_xy) * CAM_PERIOD
                guess = self._pixel_of(rot, np.array([moved[0], moved[1], -height]))
                square_px = 0.2 * self.focal / height  # checker square size in pixels
                corr = correlation_surface(prev_gray, gray)
                dcol, drow = peak_near(corr, guess[0], guess[1], radius=max(1.5, 0.35 * square_px))
                d1 = self._ray(rot, dcol, drow)
                if d1[2] < -0.3:
                    g1 = height * d1[:2] / -d1[2]  # the same floor point, relative to the drone now
                    vel = (g0 - g1) / CAM_PERIOD
        self.prev = (gray, rot, height)
        return height, vel, gray, rgb

    def close(self):
        self.renderer.close()


# ---------------------------------------------------------------- Kalman filter
class Estimator:
    """Three independent 1-axis Kalman filters, state [position, velocity] per axis."""

    def __init__(self, pos0, dt=config.SIM_DT):
        self.x = np.zeros((3, 2))
        self.x[:, 0] = pos0
        self.P = np.tile(np.diag([0.05 ** 2, 0.05 ** 2]), (3, 1, 1))
        self.dt = dt
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.B = np.array([0.5 * dt * dt, dt])
        q = ACCEL_PROCESS_SIGMA ** 2
        self.Q = q * np.outer(self.B, self.B) + np.diag([0.0, 1e-6])
        self.rejected = 0

    def predict(self, accel_world):
        for i in range(3):
            self.x[i] = self.F @ self.x[i] + self.B * accel_world[i]
            self.P[i] = self.F @ self.P[i] @ self.F.T + self.Q

    def update(self, axis: int, value: float, sigma: float, kind: str):
        """kind = 'pos' or 'vel'. Measurements far outside the expected spread are rejected (gating)."""
        if not np.isfinite(value):
            return
        h = np.array([1.0, 0.0]) if kind == "pos" else np.array([0.0, 1.0])
        s = h @ self.P[axis] @ h + sigma ** 2
        innovation = value - h @ self.x[axis]
        if innovation ** 2 > GATE_SIGMAS ** 2 * s:
            self.rejected += 1
            return
        k = self.P[axis] @ h / s
        self.x[axis] = self.x[axis] + k * innovation
        self.P[axis] = (np.eye(2) - np.outer(k, h)) @ self.P[axis]

    @property
    def pos(self):
        return self.x[:, 0].copy()

    @property
    def vel(self):
        return self.x[:, 1].copy()
