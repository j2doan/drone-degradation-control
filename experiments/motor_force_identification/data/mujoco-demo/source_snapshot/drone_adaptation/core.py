"""Small, dependency-free altitude-control experiment.

This is a one-dimensional teaching model, not a flight simulator.  Its hidden
actuator effectiveness belongs to the plant; ordinary controllers receive only
observations and remember the commands they issued themselves.
"""

from dataclasses import dataclass
import math
from typing import Callable, Optional


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class ExperimentConfig:
    dt_s: float = 0.02
    duration_s: float = 15.0
    target_m: float = 1.5
    initial_z_m: float = 1.5
    initial_vz_m_s: float = 0.0
    mass_kg: float = 1.0
    gravity_m_s2: float = 9.81
    thrust_to_weight: float = 2.5
    fault_time_s: float = 5.0
    eta_after: float = 0.75
    noise_z_m: float = 0.01
    noise_vz_m_s: float = 0.03
    min_altitude_m: float = 0.05
    max_altitude_m: float = 4.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        for name in ("dt_s", "duration_s", "mass_kg", "gravity_m_s2",
                     "thrust_to_weight", "max_altitude_m"):
            _positive(name, getattr(self, name))
        if self.min_altitude_m < 0:
            raise ValueError("min_altitude_m must be nonnegative")
        if self.max_altitude_m <= self.min_altitude_m:
            raise ValueError("max_altitude_m must exceed min_altitude_m")
        for name in ("target_m", "initial_z_m"):
            if not self.min_altitude_m <= getattr(self, name) <= self.max_altitude_m:
                raise ValueError(f"{name} must lie within the altitude bounds")
        for name in ("noise_z_m", "noise_vz_m_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.fault_time_s <= self.duration_s:
            raise ValueError("fault_time_s must lie within the episode")
        if not 0 < self.eta_after <= 1:
            raise ValueError("eta_after must be in (0, 1]")
        ratio = self.duration_s / self.dt_s
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("duration_s must be an integer multiple of dt_s")

    @property
    def max_thrust_n(self) -> float:
        return self.mass_kg * self.gravity_m_s2 * self.thrust_to_weight

    @property
    def num_steps(self) -> int:
        return round(self.duration_s / self.dt_s)


@dataclass(frozen=True)
class State:
    time_s: float
    z_m: float
    vz_m_s: float


@dataclass(frozen=True)
class Observation:
    z_m: float
    vz_m_s: float


class AltitudePlant:
    """Integrate vertical motion with a hidden, abrupt thrust reduction.

    Altitude bounds are evaluation criteria, not hard walls.  The caller must
    stop a failed episode; this model never teleports a crashed drone upward.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._step_count = 0
        self._state = State(0.0, config.initial_z_m, config.initial_vz_m_s)
        self.last_command_n = 0.0
        self.last_applied_thrust_n = 0.0

    @property
    def state(self) -> State:
        return self._state

    def eta_at(self, time_s: float) -> float:
        time_s = _finite("time_s", time_s)
        return 1.0 if time_s < self.config.fault_time_s else self.config.eta_after

    def observe(self, noise_z: float = 0.0, noise_vz: float = 0.0) -> Observation:
        """Add caller-supplied noise offsets; no random generator is hidden here."""
        return Observation(
            self._state.z_m + _finite("noise_z", noise_z),
            self._state.vz_m_s + _finite("noise_vz", noise_vz),
        )

    def step(self, command_n: float) -> State:
        if self._step_count >= self.config.num_steps:
            raise RuntimeError("the configured episode horizon has been reached")
        command = _clip(_finite("command_n", command_n), 0.0,
                        self.config.max_thrust_n)
        t0 = self._step_count * self.config.dt_s
        t1 = (self._step_count + 1) * self.config.dt_s
        split = self.config.fault_time_s
        edges = (t0, split, t1) if t0 < split < t1 else (t0, t1)
        z, velocity = self._state.z_m, self._state.vz_m_s
        impulse = 0.0
        for start, end in zip(edges[:-1], edges[1:]):
            interval = end - start
            thrust = self.eta_at(start) * command
            acceleration = thrust / self.config.mass_kg - self.config.gravity_m_s2
            z += velocity * interval + 0.5 * acceleration * interval * interval
            velocity += acceleration * interval
            impulse += thrust * interval
        self._step_count += 1
        self.last_command_n = command
        self.last_applied_thrust_n = impulse / (t1 - t0)
        self._state = State(t1, z, velocity)
        return self._state


class PIDController:
    """Altitude PID with gravity feedforward and conditional anti-windup.

    Gains produce acceleration corrections.  Integration is inhibited when its
    next update would push the final actuator command farther into saturation.
    """

    def __init__(self, *, target_m: float, mass_kg: float,
                 gravity_m_s2: float, max_thrust_n: float,
                 kp: float = 5.0, kd: float = 3.5, ki: float = 2.0):
        self.target_m = _finite("target_m", target_m)
        self.mass_kg = _positive("mass_kg", mass_kg)
        self.gravity_m_s2 = _positive("gravity_m_s2", gravity_m_s2)
        self.max_thrust_n = _positive("max_thrust_n", max_thrust_n)
        self.kp = _finite("kp", kp)
        self.kd = _finite("kd", kd)
        self.ki = _finite("ki", ki)
        if min(self.kp, self.kd, self.ki) < 0:
            raise ValueError("controller gains must be nonnegative")
        self._integral_state = 0.0

    @property
    def eta_estimate(self) -> Optional[float]:
        return None

    @property
    def integral_state(self) -> float:
        return self._integral_state

    def reset(self) -> None:
        self._integral_state = 0.0

    def _before_act(self, observation: Observation, dt_s: float) -> None:
        pass

    def _eta_for_control(self) -> float:
        return 1.0

    def _record_command(self, command_n: float) -> None:
        pass

    def act(self, observation: Observation, dt_s: float) -> float:
        dt_s = _positive("dt_s", dt_s)
        z = _finite("observation.z_m", observation.z_m)
        velocity = _finite("observation.vz_m_s", observation.vz_m_s)
        self._before_act(observation, dt_s)
        eta = _positive("compensation effectiveness", self._eta_for_control())
        error = self.target_m - z
        proposed_integral = self._integral_state + error * dt_s if self.ki else 0.0

        def request(integral: float) -> float:
            nominal = self.mass_kg * (
                self.gravity_m_s2 + self.kp * error - self.kd * velocity
                + self.ki * integral
            )
            return nominal / eta

        proposed_command = request(proposed_integral)
        pushes_upper_limit = proposed_command > self.max_thrust_n and error > 0
        pushes_lower_limit = proposed_command < 0 and error < 0
        if not (pushes_upper_limit or pushes_lower_limit):
            self._integral_state = proposed_integral
        command = _clip(request(self._integral_state), 0.0, self.max_thrust_n)
        self._record_command(command)
        return command


class AdaptiveController(PIDController):
    """Estimate scalar effectiveness using nonoverlapping velocity windows.

    Over a window, delta-v + g*duration = eta * integral(command/mass).
    Recursive least squares uses this integrated relation instead of a noisy
    one-step acceleration estimate.  The model assumes known, fixed mass and
    no external vertical force; it cannot diagnose a physical cause.
    """

    def __init__(self, *, target_m: float, mass_kg: float,
                 gravity_m_s2: float, max_thrust_n: float,
                 kp: float = 5.0, kd: float = 3.5, ki: float = 2.0,
                 estimation_window_s: float = 0.2,
                 forgetting_factor: float = 0.9,
                 initial_covariance: float = 10.0,
                 eta_min: float = 0.4, eta_max: float = 1.2):
        super().__init__(target_m=target_m, mass_kg=mass_kg,
                         gravity_m_s2=gravity_m_s2, max_thrust_n=max_thrust_n,
                         kp=kp, kd=kd, ki=ki)
        self.estimation_window_s = _positive("estimation_window_s", estimation_window_s)
        self.forgetting_factor = _positive("forgetting_factor", forgetting_factor)
        if self.forgetting_factor > 1:
            raise ValueError("forgetting_factor must lie in (0, 1]")
        self.initial_covariance = _positive("initial_covariance", initial_covariance)
        self.eta_min = _positive("eta_min", eta_min)
        self.eta_max = _positive("eta_max", eta_max)
        if not self.eta_min <= 1.0 <= self.eta_max or self.eta_min >= self.eta_max:
            raise ValueError("eta bounds must be ordered and contain 1.0")
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._eta = 1.0
        self._covariance = self.initial_covariance
        self._previous_command: Optional[float] = None
        self._window_start_velocity: Optional[float] = None
        self._window_duration = 0.0
        self._window_input = 0.0

    @property
    def eta_estimate(self) -> float:
        return self._eta

    def _eta_for_control(self) -> float:
        return self._eta

    def _record_command(self, command_n: float) -> None:
        self._previous_command = command_n

    def _before_act(self, observation: Observation, dt_s: float) -> None:
        if self._window_start_velocity is None:
            self._window_start_velocity = observation.vz_m_s
            return
        if self._previous_command is None:
            return
        self._window_duration += dt_s
        self._window_input += self._previous_command * dt_s / self.mass_kg
        if self._window_duration + 1e-12 < self.estimation_window_s:
            return
        regressor = self._window_input
        response = (observation.vz_m_s - self._window_start_velocity
                    + self.gravity_m_s2 * self._window_duration)
        if abs(regressor) > 1e-9:
            denominator = (self.forgetting_factor
                           + self._covariance * regressor * regressor)
            gain = self._covariance * regressor / denominator
            self._eta = _clip(
                self._eta + gain * (response - regressor * self._eta),
                self.eta_min, self.eta_max,
            )
            self._covariance = self._covariance / denominator
        self._window_start_velocity = observation.vz_m_s
        self._window_duration = 0.0
        self._window_input = 0.0


class OracleController(PIDController):
    """Privileged diagnostic: compensate using the true effectiveness callable."""

    def __init__(self, *, true_eta: Callable[[], float], target_m: float,
                 mass_kg: float, gravity_m_s2: float, max_thrust_n: float,
                 kp: float = 5.0, kd: float = 3.5, ki: float = 2.0):
        if not callable(true_eta):
            raise ValueError("true_eta must be callable")
        super().__init__(target_m=target_m, mass_kg=mass_kg,
                         gravity_m_s2=gravity_m_s2, max_thrust_n=max_thrust_n,
                         kp=kp, kd=kd, ki=ki)
        self._true_eta = true_eta
        self._last_eta = 1.0

    @property
    def eta_estimate(self) -> float:
        return self._last_eta

    def reset(self) -> None:
        super().reset()
        self._last_eta = 1.0

    def _eta_for_control(self) -> float:
        self._last_eta = _positive("true_eta()", self._true_eta())
        return self._last_eta
