"""Gymnasium altitude task with a bounded correction around the baseline PID.

This is the same one-dimensional plant as the baseline experiment.  It is not
an Isaac task.  The policy sees noisy measurements and issued commands, never
true effectiveness, realized force, a fault label, or the episode clock.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import math
import random
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .core import AltitudePlant, ExperimentConfig, Observation, PIDController


class _ResidualAwarePID(PIDController):
    """Keep baseline behavior while also respecting final-command saturation.

    ``prepare`` advances the PID exactly once per control decision.  A later
    residual can push that command into saturation; ``combine`` then cancels
    an integral update that would push farther into the same limit.  The
    original PID's conservative anti-windup remains in force as well.
    """

    def prepare(self, observation: Observation, dt_s: float) -> float:
        self._integral_before_decision = self._integral_state
        self._decision_observation = observation
        self.base_command_n = super().act(observation, dt_s)
        return self.base_command_n

    def peek(self, observation: Observation) -> float:
        nominal = self.mass_kg * (
            self.gravity_m_s2
            + self.kp * (self.target_m - observation.z_m)
            - self.kd * observation.vz_m_s
            + self.ki * self._integral_state
        )
        return float(np.clip(nominal, 0.0, self.max_thrust_n))

    def combine(self, residual_n: float) -> tuple[float, float]:
        requested = self.base_command_n + residual_n
        error = self.target_m - self._decision_observation.z_m
        drives_saturation = (
            (requested > self.max_thrust_n and error > 0)
            or (requested < 0.0 and error < 0)
        )
        if drives_saturation:
            self._integral_state = self._integral_before_decision
            self.base_command_n = self.peek(self._decision_observation)
        final = float(np.clip(self.base_command_n + residual_n, 0.0, self.max_thrust_n))
        return self.base_command_n, final


class ResidualHoverEnv(gym.Env):
    """Learn a scalar residual using fixed-size, causally masked history.

    Observation dictionary (float32):

    * ``current``: eight values: measured altitude error / 0.3 m, measured
      vertical velocity / 0.5 m/s, previous issued command / weight - 1,
      previous normalized residual action, PID integral / 0.3 m*s, prepared
      PID command / weight - 1, immediate nominal velocity prediction error
      / 0.5 m/s, and a completed-transition validity flag.
    * ``history``: shape [6, 50].  The first five channels are the first five
      current features; channel six is a validity mask.  Frames include the
      current measurement paired with the *previous* command and action.
      The allowed newest frames are right-aligned and older entries are zero.

    Sampling is fixed at 50 Hz.  A zero-second window masks all history; the
    current input still contains a previous command and immediate discrepancy.
    Thus this tests additional temporal context beyond immediate feedback and
    the classical PID's integral state.
    """

    metadata = {"render_modes": []}
    MAX_HISTORY_STEPS = 50
    CURRENT_DIM = 8
    HISTORY_CHANNELS = 6
    DT_S = 0.02
    POSITION_SCALE_M = 0.3
    VELOCITY_SCALE_M_S = 0.5
    INTEGRAL_SCALE_M_S = 0.3
    MAX_STATE_COST = 10.0

    def __init__(
        self,
        history_s: float = 0.5,
        randomize: bool = True,
        scenario: ExperimentConfig | None = None,
        residual_scale: float = 0.6,
    ):
        super().__init__()
        if not math.isfinite(history_s) or not 0.0 <= history_s <= 1.0:
            raise ValueError("history_s must be finite and lie in [0, 1]")
        steps = history_s / self.DT_S
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise ValueError("history_s must be a multiple of the 0.02-second control interval")
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        self.history_s = float(history_s)
        self.history_steps = round(steps)
        self.randomize = bool(randomize)
        self.residual_scale = float(residual_scale)
        self._fixed_scenario = scenario or ExperimentConfig()
        self._validate_scenario(self._fixed_scenario)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "current": spaces.Box(-np.inf, np.inf, shape=(self.CURRENT_DIM,), dtype=np.float32),
                "history": spaces.Box(-np.inf, np.inf, shape=(self.HISTORY_CHANNELS, self.MAX_HISTORY_STEPS), dtype=np.float32),
            }
        )
        self.rows: list[dict[str, Any]] = []
        self._frames: deque[np.ndarray] = deque(maxlen=self.MAX_HISTORY_STEPS)
        self._done = True

    @staticmethod
    def _validate_scenario(scenario: ExperimentConfig) -> None:
        if not isinstance(scenario, ExperimentConfig):
            raise ValueError("scenario must be an ExperimentConfig")
        if not math.isclose(scenario.dt_s, ResidualHoverEnv.DT_S, abs_tol=1e-12):
            raise ValueError("ResidualHoverEnv uses a fixed 0.02-second control interval")

    def _sample_scenario(self) -> ExperimentConfig:
        eta = 1.0 if self.np_random.random() < 0.25 else float(self.np_random.uniform(0.85, 1.0))
        onset = round(float(self.np_random.uniform(3.0, 7.0)) / self.DT_S) * self.DT_S
        noise = float(self.np_random.uniform(0.0, 0.03))
        return replace(
            ExperimentConfig(), eta_after=eta, fault_time_s=onset,
            noise_z_m=noise, noise_vz_m_s=3.0 * noise,
        )

    def _measure(self) -> Observation:
        return self.plant.observe(
            noise_z=self._noise_rng.gauss(0.0, self.scenario.noise_z_m),
            noise_vz=self._noise_rng.gauss(0.0, self.scenario.noise_vz_m_s),
        )

    def _features(self) -> np.ndarray:
        return np.asarray(
            [
                (self.scenario.target_m - self.measurement.z_m) / self.POSITION_SCALE_M,
                self.measurement.vz_m_s / self.VELOCITY_SCALE_M_S,
                self._previous_command_n / self._weight_n - 1.0,
                self._previous_action,
                self.controller.integral_state / self.INTEGRAL_SCALE_M_S,
                self._base_command_n / self._weight_n - 1.0,
                self._prediction_error_m_s / self.VELOCITY_SCALE_M_S,
                float(self._has_transition),
            ], dtype=np.float32,
        )

    def _observation(self, *, append_frame: bool = True) -> dict[str, np.ndarray]:
        current = self._features()
        if append_frame:
            self._frames.append(current[:5].copy())
        history = np.zeros((self.HISTORY_CHANNELS, self.MAX_HISTORY_STEPS), dtype=np.float32)
        count = min(self.history_steps, len(self._frames))
        if count:
            frames = list(self._frames)[-count:]
            history[:5, -count:] = np.stack(frames, axis=1)
            history[5, -count:] = 1.0
        return {"current": current, "history": history}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        supplied = options.get("scenario") if options else None
        self.scenario = supplied if supplied is not None else (
            self._sample_scenario() if self.randomize else self._fixed_scenario
        )
        self._validate_scenario(self.scenario)
        # A separate generator preserves the baseline experiment's noise stream.
        noise_seed = seed if seed is not None else int(self.np_random.integers(0, 2**63 - 1))
        self._noise_rng = random.Random(noise_seed)
        self.plant = AltitudePlant(self.scenario)
        self.controller = _ResidualAwarePID(
            target_m=self.scenario.target_m,
            mass_kg=self.scenario.mass_kg,
            gravity_m_s2=self.scenario.gravity_m_s2,
            max_thrust_n=self.scenario.max_thrust_n,
        )
        self._weight_n = self.scenario.mass_kg * self.scenario.gravity_m_s2
        self._previous_command_n = self._weight_n
        self._previous_action = 0.0
        self._prediction_error_m_s = 0.0
        self._has_transition = False
        self.rows = []
        self._frames.clear()
        self.measurement = self._measure()
        self._base_command_n = self.controller.prepare(self.measurement, self.scenario.dt_s)
        self._done = False
        return self._observation(), {}

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a new or completed episode")
        array = np.asarray(action, dtype=np.float64)
        if array.size != 1 or not np.isfinite(array).all():
            raise ValueError("action must contain one finite scalar")
        chosen_action = float(np.clip(array.reshape(-1)[0], -1.0, 1.0))
        residual_n = self.residual_scale * self._weight_n * chosen_action
        base_command, command = self.controller.combine(residual_n)
        before = self.plant.state
        previous_measurement = self.measurement
        after = self.plant.step(command)
        self.rows.append(
            {
                "time_s": before.time_s,
                "z_m": before.z_m,
                "vz_m_s": before.vz_m_s,
                "observed_z_m": previous_measurement.z_m,
                "observed_vz_m_s": previous_measurement.vz_m_s,
                "command_n": self.plant.last_command_n,
                "applied_thrust_n": self.plant.last_applied_thrust_n,
                "eta_true": self.plant.eta_at(before.time_s),
                "eta_estimate": None,
                "integral_state": self.controller.integral_state,
                "base_command_n": base_command,
                "residual_action": chosen_action,
                "residual_n": residual_n,
            }
        )
        terminated = after.z_m <= self.scenario.min_altitude_m or after.z_m >= self.scenario.max_altitude_m
        truncated = after.time_s >= self.scenario.duration_s - 1e-9 and not terminated
        error = after.z_m - self.scenario.target_m
        state_cost = min(
            self.MAX_STATE_COST,
            4.0 * (error / self.POSITION_SCALE_M) ** 2
            + 0.25 * (after.vz_m_s / self.VELOCITY_SCALE_M_S) ** 2,
        )
        cost = state_cost + 0.01 * chosen_action**2 + 0.03 * (chosen_action - self._previous_action)**2
        reward = -float(cost * self.scenario.dt_s)
        if terminated:
            # Exceeds the maximum possible nonterminal cost of the whole horizon,
            # so ending early cannot improve return merely by avoiding future cost.
            maximum_rate = self.MAX_STATE_COST + 0.01 + 0.03 * 4.0
            reward -= max(200.0, maximum_rate * self.scenario.duration_s + 1.0)
        self._previous_command_n = self.plant.last_command_n
        self._previous_action = chosen_action
        self.measurement = self._measure()
        nominal_next_v = previous_measurement.vz_m_s + self.scenario.dt_s * (
            self._previous_command_n / self.scenario.mass_kg - self.scenario.gravity_m_s2
        )
        self._prediction_error_m_s = self.measurement.vz_m_s - nominal_next_v
        self._has_transition = True
        self._done = bool(terminated or truncated)
        if self._done:
            # Gymnasium still needs a terminal observation, but no next PID action
            # is taken, so do not advance its integrator beyond the last command.
            self._base_command_n = self.controller.peek(self.measurement)
        else:
            self._base_command_n = self.controller.prepare(self.measurement, self.scenario.dt_s)
        return self._observation(), reward, bool(terminated), bool(truncated), {}

    def episode_rows(self) -> list[dict[str, Any]]:
        """Return metric-compatible logs plus a clean final state frame.

        The endpoint has no next command or observation, matching the existing
        baseline logger. Calling this method does not mutate history or state.
        """
        state = self.plant.state
        terminal = {
            "time_s": state.time_s,
            "z_m": state.z_m,
            "vz_m_s": state.vz_m_s,
            "observed_z_m": None,
            "observed_vz_m_s": None,
            "command_n": None,
            "applied_thrust_n": None,
            "eta_true": self.plant.eta_at(state.time_s),
            "eta_estimate": None,
            "integral_state": self.controller.integral_state,
            "base_command_n": None,
            "residual_action": None,
            "residual_n": None,
        }
        return [*self.rows, terminal]
