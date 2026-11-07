"""Internal screening tools for stale-context experiments in a scalar plant.

These references are not neural controllers or reproductions of published
fault-tolerant flight algorithms.  They isolate forgetting using a shared PID
and a known-mass, no-wind altitude model.  Effectiveness is hidden in the plant;
the estimator receives measurements and remembers its own commanded thrust.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .core import (
    AltitudePlant, ExperimentConfig, Observation, PIDController, State,
    _clip, _finite, _positive,
)


def _effectiveness(name: str, value: float) -> float:
    value = _finite(name, value)
    if not 0.4 <= value <= 1.2:
        raise ValueError(f"{name} must lie in [0.4, 1.2] for this screening model")
    return value


@dataclass(frozen=True)
class EffectivenessSchedule:
    """Immutable, right-continuous piecewise-constant effectiveness.

    ``events=()`` is healthy flight.  For example, ``((5, .75), (9, 1))``
    represents loss and restoration.  Further events express severity changes
    and repeated faults.  At an event time the new value applies immediately.
    Events must be strictly increasing; a time-zero event is permitted.
    """

    events: tuple[tuple[float, float], ...] = ()
    initial_eta: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_eta",
                           _effectiveness("initial_eta", self.initial_eta))
        normalized = []
        previous_time = -1.0
        try:
            events = tuple(self.events)
        except TypeError as exc:
            raise ValueError("events must contain (time_s, eta) pairs") from exc
        for event in events:
            try:
                time_s, eta = event
            except (TypeError, ValueError) as exc:
                raise ValueError("events must contain (time_s, eta) pairs") from exc
            time_s = _finite("event time_s", time_s)
            eta = _effectiveness("event eta", eta)
            if time_s < 0 or time_s <= previous_time:
                raise ValueError("event times must be nonnegative and strictly increasing")
            normalized.append((time_s, eta))
            previous_time = time_s
        object.__setattr__(self, "events", tuple(normalized))

    def eta_at(self, time_s: float) -> float:
        time_s = _finite("time_s", time_s)
        if time_s < 0:
            raise ValueError("time_s must be nonnegative")
        eta = self.initial_eta
        for event_time, new_eta in self.events:
            if event_time > time_s:
                break
            eta = new_eta
        return eta


class SwitchingAltitudePlant(AltitudePlant):
    """Core-compatible altitude plant with any number of scheduled switches.

    Each held-command step is integrated exactly across all event boundaries,
    including multiple events inside one control interval.  Bounds are failure
    criteria, not physical walls, as in :class:`AltitudePlant`.
    """

    def __init__(self, config: ExperimentConfig, schedule: EffectivenessSchedule):
        if not isinstance(schedule, EffectivenessSchedule):
            raise ValueError("schedule must be an EffectivenessSchedule")
        if any(time_s > config.duration_s for time_s, _ in schedule.events):
            raise ValueError("schedule events must lie within the episode horizon")
        self.schedule = schedule
        super().__init__(config)

    def reset(self) -> State:
        self._step_count = 0
        self._state = State(0.0, self.config.initial_z_m, self.config.initial_vz_m_s)
        self.last_command_n = 0.0
        self.last_applied_thrust_n = 0.0
        return self._state

    def eta_at(self, time_s: float) -> float:
        return self.schedule.eta_at(time_s)

    def step(self, command_n: float) -> State:
        if self._step_count >= self.config.num_steps:
            raise RuntimeError("the configured episode horizon has been reached")
        command = _clip(_finite("command_n", command_n), 0.0,
                        self.config.max_thrust_n)
        t0 = self._step_count * self.config.dt_s
        t1 = (self._step_count + 1) * self.config.dt_s
        edges = [t0]
        edges.extend(time_s for time_s, _ in self.schedule.events if t0 < time_s < t1)
        edges.append(t1)
        z, velocity = self._state.z_m, self._state.vz_m_s
        impulse = 0.0
        for start, end in zip(edges[:-1], edges[1:]):
            duration = end - start
            thrust = command * self.eta_at(start)
            acceleration = thrust / self.config.mass_kg - self.config.gravity_m_s2
            z += velocity * duration + 0.5 * acceleration * duration**2
            velocity += acceleration * duration
            impulse += thrust * duration
        self._step_count += 1
        self.last_command_n = command
        self.last_applied_thrust_n = impulse / (t1 - t0)
        self._state = State(t1, z, velocity)
        return self._state


class ForgettingController(PIDController):
    """PID compensation using completed velocity-transition blocks.

    Each nonoverlapping block uses ``y = delta_v + g * duration`` and
    ``x = integral(command / mass)``.  With constant effectiveness and no wind,
    ``y = eta * x``.  Windows close at observed sample boundaries, retaining
    their actual elapsed duration rather than inventing intermediate velocity
    measurements.  No time, schedule or fault label enters ``act``.

    ``window_short`` and ``window_long`` solve trailing least squares over
    0.2 s and 1 s of completed blocks.  RLS modes use all blocks with fixed
    exponential forgetting or a simple innovation-driven forgetting reference.
    The latter is explicitly not a reproduction of Han et al.'s algorithm.
    A velocity-noise burst or an external force can also change its estimate;
    this scalar estimator cannot identify the physical cause of a mismatch.
    """

    MODES = ("window_short", "window_long", "rls_fixed", "rls_variable")

    def __init__(self, *, target_m: float, mass_kg: float,
                 gravity_m_s2: float, max_thrust_n: float,
                 mode: str, kp: float = 5.0, kd: float = 3.5, ki: float = 2.0,
                 observation_window_s: float = 0.1,
                 history_window_s: Optional[float] = None,
                 forgetting_factor: float = 0.9,
                 variable_min_forgetting: float = 0.2,
                 variable_max_forgetting: float = 0.995,
                 innovation_tolerance_eta: float = 0.08,
                 innovation_scale_eta: float = 0.2,
                 initial_covariance: float = 10.0,
                 eta_min: float = 0.4, eta_max: float = 1.2):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        super().__init__(target_m=target_m, mass_kg=mass_kg,
                         gravity_m_s2=gravity_m_s2, max_thrust_n=max_thrust_n,
                         kp=kp, kd=kd, ki=ki)
        self.mode = mode
        self.observation_window_s = _positive("observation_window_s", observation_window_s)
        default_history = 0.2 if mode == "window_short" else 1.0
        self.history_window_s = _positive(
            "history_window_s", default_history if history_window_s is None else history_window_s)
        if self.history_window_s < self.observation_window_s:
            raise ValueError("history_window_s must be at least observation_window_s")
        self.forgetting_factor = _positive("forgetting_factor", forgetting_factor)
        self.variable_min_forgetting = _positive("variable_min_forgetting", variable_min_forgetting)
        self.variable_max_forgetting = _positive("variable_max_forgetting", variable_max_forgetting)
        if self.forgetting_factor > 1 or not (
            self.variable_min_forgetting <= self.variable_max_forgetting <= 1
        ):
            raise ValueError("forgetting factors must be ordered and lie in (0, 1]")
        self.innovation_tolerance_eta = _finite("innovation_tolerance_eta", innovation_tolerance_eta)
        if self.innovation_tolerance_eta < 0:
            raise ValueError("innovation_tolerance_eta must be nonnegative")
        self.innovation_scale_eta = _positive("innovation_scale_eta", innovation_scale_eta)
        self.initial_covariance = _positive("initial_covariance", initial_covariance)
        self.eta_min = _effectiveness("eta_min", eta_min)
        self.eta_max = _effectiveness("eta_max", eta_max)
        if not self.eta_min <= 1 <= self.eta_max or self.eta_min >= self.eta_max:
            raise ValueError("eta bounds must be ordered and contain 1")
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._eta = 1.0
        self._covariance = self.initial_covariance
        self._previous_command: Optional[float] = None
        self._previous_dt = 0.0
        self._next_dt = 0.0
        self._start_velocity: Optional[float] = None
        self._block_duration = 0.0
        self._block_input = 0.0
        self._elapsed = 0.0
        self._history = deque()
        self.num_estimator_updates = 0
        self.last_forgetting_factor: Optional[float] = None
        self.last_innovation_eta: Optional[float] = None

    @property
    def eta_estimate(self) -> float:
        return self._eta

    @property
    def covariance(self) -> float:
        return self._covariance

    def _eta_for_control(self) -> float:
        return self._eta

    def _record_command(self, command_n: float) -> None:
        self._previous_command = command_n
        self._previous_dt = self._next_dt

    def _before_act(self, observation: Observation, dt_s: float) -> None:
        self._next_dt = dt_s
        if self._start_velocity is None:
            self._start_velocity = observation.vz_m_s
            return
        if self._previous_command is None:
            return
        # The current observation follows the PREVIOUS command's interval.
        # The supplied dt describes the next command, which may differ.
        self._elapsed += self._previous_dt
        self._block_duration += self._previous_dt
        self._block_input += self._previous_command * self._previous_dt / self.mass_kg
        if self._block_duration + 1e-12 < self.observation_window_s:
            return
        x = self._block_input
        y = (observation.vz_m_s - self._start_velocity
             + self.gravity_m_s2 * self._block_duration)
        if abs(x) > 1e-9:
            self.last_innovation_eta = (y - x * self._eta) / x
            if self.mode.startswith("window_"):
                self._history.append((self._elapsed, self._block_duration, x, y))
                cutoff = self._elapsed - self.history_window_s
                # Whole blocks only: discard blocks extending before the window.
                # Always keep the newest block, even for coarse observation dt.
                while len(self._history) > 1 and self._history[0][0] - self._history[0][1] < cutoff - 1e-12:
                    self._history.popleft()
                denominator = sum(block_x**2 for _, _, block_x, _ in self._history)
                numerator = sum(block_x * block_y for _, _, block_x, block_y in self._history)
                self._eta = _clip(numerator / denominator, self.eta_min, self.eta_max)
            else:
                forgetting = self.forgetting_factor
                if self.mode == "rls_variable":
                    strength = _clip(
                        (abs(self.last_innovation_eta) - self.innovation_tolerance_eta)
                        / self.innovation_scale_eta, 0.0, 1.0)
                    forgetting = (self.variable_max_forgetting
                                  - strength * (self.variable_max_forgetting
                                                - self.variable_min_forgetting))
                self.last_forgetting_factor = forgetting
                denominator = forgetting + self._covariance * x * x
                gain = self._covariance * x / denominator
                self._eta = _clip(self._eta + gain * (y - x * self._eta),
                                  self.eta_min, self.eta_max)
                self._covariance = self._covariance / denominator
            self.num_estimator_updates += 1
        self._start_velocity = observation.vz_m_s
        self._block_duration = 0.0
        self._block_input = 0.0
