"""One-dimensional (up/down only) drone with a sudden thrust loss, its PID, an oracle, and a Gymnasium environment
in which PPO learns a correction added to the PID's thrust command.

Physics (Skydio X2 numbers):
    mass 1.325 kg, total thrust command capped at 24 N (4 motors x 6 N), first-order motor lag 0.125 s,
    physics step 0.01 s. Thrust loss: the delivered thrust is multiplied by (1 - loss) from the onset time on.
    Hover needs 13.0 N, so a hover is impossible beyond a 45.8 % loss (24 N x 0.542 = 13.0 N).

Controllers:
    pid      altitude PID with gravity feedforward, using the 3D X2 PID's z gains (kp = kd = 1.75, ki = 1.0). The
             integrator may add up to 11 N (enough to hold a hover up to the 45.8 % limit) and freezes while the
             command is at a limit. (The 3D PID's tighter integrator limit allows only 2 N, about a 15 % total
             loss; that is plenty in 3D, where one motor losing 37.5 % costs only 9.4 % of total thrust.)
    oracle   the same PID, told the true loss at the instant it happens: command / (1 - loss)
    ppo      the same PID plus a learned correction of up to +-6 N, chosen 50 times a second

The PPO policy never sees the loss, the onset time or the delivered thrust directly. Its 5 inputs:
    height error, vertical velocity, PID command, its own previous correction, accelerometer (specific force)
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

MASS = 1.325  # kg
G = 9.81  # m/s^2
WEIGHT = MASS * G  # 13.0 N
MAX_THRUST = 24.0  # N, total
TAU = 0.125  # s, motor lag
DT = 0.01  # s, physics step
DECISION_EVERY = 2  # physics steps per PPO decision (50 Hz)
MAX_CORRECTION = 6.0  # N
TARGET_Z = 1.5  # m
CRASH_Z = 0.30  # m
EPISODE_T = 10.0  # s
RECOVER_TOL, RECOVER_HOLD = 0.05, 0.5  # m, s

# PID gains (the 3D X2 PID's z axis); integrator limit wide enough for 11 N of extra thrust
KP, KD, KI = 1.75, 1.75, 1.0
I_LIMIT = (MAX_THRUST - WEIGHT) / (MASS * KI)  # 8.3 m s


class Plant:
    def __init__(self, z0=TARGET_Z):
        self.z, self.vz = z0, 0.0
        self.thrust = WEIGHT  # motors start at hover thrust (spinning output, before any loss)
        self.alpha = 1.0 - np.exp(-DT / TAU)
        self.specific_force = G  # what an accelerometer reads (delivered thrust / mass); +g in hover

    def step(self, command, eta):
        command = float(np.clip(command, 0.0, MAX_THRUST))
        self.thrust += self.alpha * (command - self.thrust)
        delivered = eta * self.thrust
        acc = delivered / MASS - G
        self.z += self.vz * DT + 0.5 * acc * DT * DT
        self.vz += acc * DT
        if self.z < 0.0:  # the floor
            self.z, self.vz = 0.0, max(self.vz, 0.0)
        self.specific_force = delivered / MASS
        return command, delivered


class PID:
    def __init__(self):
        self.integral = 0.0
        self.saturated = False

    def act(self, z, vz, target=TARGET_Z):
        e, ev = target - z, -vz
        if not self.saturated:
            self.integral = float(np.clip(self.integral + e * DT, -I_LIMIT, I_LIMIT))
        return MASS * (G + KP * e + KD * ev + KI * self.integral)

    def note_command(self, final_command):
        """Anti-windup: freeze the integrator next step if the final command hit a limit."""
        self.saturated = final_command >= MAX_THRUST - 1e-9 or final_command <= 0.0


@dataclass
class Scenario:
    loss: float
    onset_t: float

    def eta(self, t):
        return 1.0 - self.loss if t >= self.onset_t - 1e-9 else 1.0


def observation(z, vz, pid_cmd, last_a, specific_force):
    """5 inputs, each scaled to roughly -1..1."""
    return np.array([(TARGET_Z - z) / 0.5, vz / 1.0, pid_cmd / WEIGHT - 1.0, last_a, specific_force / G - 1.0],
                    dtype=np.float32)


def run_episode(scenario, mode="pid", policy=None):
    """Fly one episode. mode = 'pid', 'oracle' or 'ppo' (policy = an SB3 model). Never stops early.
    Returns a dict of arrays (one entry per physics step)."""
    plant, pid = Plant(), PID()
    n = int(round(EPISODE_T / DT))
    log = {k: np.zeros(n) for k in ("t", "z", "vz", "pid_cmd", "correction", "command", "delivered", "eta")}
    a = 0.0
    for k in range(n):
        t = k * DT
        eta = scenario.eta(t)
        pid_cmd = pid.act(plant.z, plant.vz)
        if mode == "oracle":
            cmd = pid_cmd / eta
        elif mode == "ppo":
            if k % DECISION_EVERY == 0:
                obs = observation(plant.z, plant.vz, pid_cmd, a, plant.specific_force)
                a = float(np.clip(policy.predict(obs, deterministic=True)[0][0], -1.0, 1.0))
            cmd = pid_cmd + MAX_CORRECTION * a
        else:
            cmd = pid_cmd
        final, delivered = plant.step(cmd, eta)
        pid.note_command(final)
        for key, val in (("t", t), ("z", plant.z), ("vz", plant.vz), ("pid_cmd", pid_cmd),
                         ("correction", cmd - pid_cmd), ("command", final), ("delivered", delivered), ("eta", eta)):
            log[key][k] = val
    return log


def score(log, scenario):
    t, z = log["t"], log["z"]
    after = t >= scenario.onset_t
    crashed = bool((z < CRASH_Z).any())
    err = np.abs(z - TARGET_Z)
    rec = np.nan
    if not crashed and scenario.loss > 0:
        inside = err[after] <= RECOVER_TOL
        ta = t[after]
        if inside.all():
            rec = 0.0
        else:
            start = None
            for k in range(int(np.argmin(inside)), len(ta)):
                if not inside[k]:
                    start = None
                elif start is None:
                    start = k
                if start is not None and ta[k] - ta[start] >= RECOVER_HOLD - 1e-9:
                    rec = float(ta[start] - ta[0])
                    break
    return {"crashed": crashed, "crash_t": float(t[np.argmax(z < CRASH_Z)]) if crashed else np.nan,
            "max_drop": float(max(0.0, (TARGET_Z - z[after]).max())), "min_z": float(z.min()),
            "recovery_time": rec, "mean_abs_err_after": float(err[after].mean()),
            "final_err": float(err[-1]), "max_command": float(log["command"].max()),
            "mean_abs_correction": float(np.abs(log["correction"][after]).mean())}


class HoverEnv(gym.Env):
    """Training environment. Each episode: hover at 1.5 m for 10 s; with probability 0.8 the thrust drops by a random
    0-50 % at a random time between 2 and 5 s. Ends early only on a crash."""

    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        loss = 0.0 if rng.random() < 0.2 else float(rng.uniform(0.0, 0.5))
        self.scenario = Scenario(loss, float(rng.uniform(2.0, 5.0)))
        self.plant, self.pid = Plant(), PID()
        self.k, self.a = 0, 0.0
        self.pid_cmd = self.pid.act(self.plant.z, self.plant.vz)
        return self._obs(), {}

    def _obs(self):
        return observation(self.plant.z, self.plant.vz, self.pid_cmd, self.a, self.plant.specific_force)

    def step(self, action):
        self.a = float(np.clip(action[0], -1.0, 1.0))
        for _ in range(DECISION_EVERY):
            t = self.k * DT
            self.pid_cmd = self.pid.act(self.plant.z, self.plant.vz)
            final, _ = self.plant.step(self.pid_cmd + MAX_CORRECTION * self.a, self.scenario.eta(t))
            self.pid.note_command(final)
            self.k += 1
        e = TARGET_Z - self.plant.z
        crashed = self.plant.z < CRASH_Z
        # Linear height penalty: a squared one (first attempt) was so mild near the target that the policy learned
        # to hover ~0.27 m too high as a safety margin. |e| charges a steady offset much more.
        reward = 1.0 - 2.0 * abs(e) - 0.1 * self.plant.vz ** 2 - 0.01 * self.a ** 2
        if crashed:
            reward -= 50.0
        truncated = self.k * DT >= EPISODE_T - 1e-9
        return self._obs(), float(reward), bool(crashed), bool(truncated and not crashed), {}
