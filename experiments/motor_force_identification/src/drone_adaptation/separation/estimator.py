"""Causal, diagnostic motor/force regression with no fault-truth inputs.

Assumptions are intentionally narrow: mass and nominal actuation are calibrated,
only the selected candidate motor can lose effectiveness, the other motors are
healthy, and an unknown external force acts along world z.  Rotation must be
accounted for by projecting command-model thrusts using the *previous* body's z
axis.  The estimator uses vertical velocity, not realized thrust or event labels.

For a non-overlapping integration window of duration D::

    y = m * (v_end - v_start) + m*g*D - integral(sum(nominal_vertical))
    y = (eta - 1) * integral(selected_nominal_vertical) + Fz*D

A rolling least-squares fit assumes eta and Fz are constant within its retained
history.  It receives no change-point segmentation, so transitions bias that fit.
Motor-only regression omits Fz; joint regression adds an intercept.  A constant
selected input cannot identify both quantities: rank/excitation failure is
reported explicitly while the last accepted estimates are held.  The validity
flag describes regressor geometry, not proof that the physical model is correct.

The estimates are deliberately unconstrained diagnostics.  Values outside the
physical interval [0, 1] expose noise/model mismatch rather than being silently
clipped into apparently plausible motor health.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

GRAVITY_M_S2 = 9.81


def _finite(name: str, value) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite number') from error
    if not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


def _positive(name: str, value) -> float:
    value = _finite(name, value)
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def _four(name: str, values: Iterable[float]) -> tuple[float, ...]:
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(f'{name} must contain four finite numbers') from error
    if len(values) != 4:
        raise ValueError(f'{name} must contain four finite numbers')
    return tuple(_finite(name, v) for v in values)


def _four_or_scalar(name: str, values) -> tuple[float, ...]:
    try:
        iter(values)
    except TypeError:
        values = (values,) * 4
    return _four(name, values)


class NominalActuatorModel:
    """Four independent known exponential first-order motors.

    Targets are clipped to [0, max_thrust_n].  The nominal lag state approaches
    that target with exp(-dt/tau_inc) or exp(-dt/tau_dec); the returned output is
    then clipped to [min_thrust_n, max_thrust_n].  A zero time constant means an
    instantaneous response.  Initial state is explicit and need not exceed the
    minimum output.  Inputs are issued targets, never fault-scaled plant outputs.

    This is the portable pilot's specified model.  It is NOT a claim that an
    installed Isaac actuator uses this equation/order: the Isaac runner must
    check/calibrate its actual nominal model separately before using it here.
    """

    def __init__(self, tau_inc_s=0.05, tau_dec_s=0.05,
                 min_thrust_n=0.0, max_thrust_n=6.0,
                 initial_thrust_n=(0.0, 0.0, 0.0, 0.0)):
        self.tau_inc_s = _four_or_scalar('tau_inc_s', tau_inc_s)
        self.tau_dec_s = _four_or_scalar('tau_dec_s', tau_dec_s)
        if any(tau < 0 for tau in (*self.tau_inc_s, *self.tau_dec_s)):
            raise ValueError('motor time constants must be nonnegative')
        self.min_thrust_n = _finite('min_thrust_n', min_thrust_n)
        self.max_thrust_n = _positive('max_thrust_n', max_thrust_n)
        if not 0 <= self.min_thrust_n <= self.max_thrust_n:
            raise ValueError('thrust limits must satisfy 0 <= min <= max')
        self._state = _four('initial_thrust_n', initial_thrust_n)
        if any(not 0 <= value <= self.max_thrust_n for value in self._state):
            raise ValueError('initial thrust state must lie in [0, max_thrust_n]')

    @property
    def state_n(self) -> tuple[float, ...]:
        """Immutable snapshot of the nominal internal lag state."""
        return self._state

    def step(self, command_n, dt_s) -> tuple[float, ...]:
        command = _four('command_n', command_n)
        dt = _positive('dt_s', dt_s)
        updated = []
        for i, value in enumerate(command):
            target = min(self.max_thrust_n, max(0.0, value))
            previous = self._state[i]
            tau = self.tau_inc_s[i] if target > previous else self.tau_dec_s[i]
            alpha = 1.0 if tau == 0 else -math.expm1(-dt/tau)
            updated.append(previous + alpha*(target - previous))
        self._state = tuple(updated)
        return tuple(min(self.max_thrust_n, max(self.min_thrust_n, v)) for v in self._state)


@dataclass(frozen=True)
class _Window:
    start_s: float
    end_s: float
    input_ns: float
    response_ns: float

    @property
    def duration_s(self):
        return self.end_s - self.start_s


class MotorForceEstimator:
    """Compare motor-only and joint eta/Fz regression on the same causal data.

    ``observe(t, measured_vz, previous_nominal_vertical)`` consumes a velocity
    measured at t and four command-model force projections for the interval
    ending at t.  The first call has no prior interval and requires None.
    Subsequent calls require that prior interval, even before the first fit.

    A joint fit needs at least three complete windows, input standard deviation
    >= min_excitation_std_n, and a numerically full-rank normalized design.
    ``joint_identified`` is recomputed on every call; an invalid fit retains the
    last estimate but must not be counted as new identified evidence.  Report
    both held-estimate errors and validity coverage when evaluating the pilot.
    """

    def __init__(self, mass_kg, motor=3, history_s=3.0, window_s=0.1,
                 min_excitation_std_n=0.03):
        self.mass_kg = _positive('mass_kg', mass_kg)
        if isinstance(motor, bool) or not isinstance(motor, int) or not 0 <= motor < 4:
            raise ValueError('motor must be an integer in [0, 3]')
        self.motor = motor
        self.history_s = _positive('history_s', history_s)
        self.window_s = _positive('window_s', window_s)
        if self.history_s < 3*self.window_s - 1e-12:
            raise ValueError('history_s must contain at least three integration windows')
        self.min_excitation_std_n = _positive('min_excitation_std_n', min_excitation_std_n)
        self._last_t = None
        self._last_v = None
        self._window_start_t = None
        self._window_start_v = None
        self._selected_input = 0.0
        self._total_input = 0.0
        self._windows: deque[_Window] = deque()
        self._motor_eta = 1.0
        self._joint_eta = 1.0
        self._force = 0.0

    def observe(self, t_s, measured_vz_m_s, nominal_vertical_thrust_n=None) -> dict:
        t = _finite('t_s', t_s)
        velocity = _finite('measured_vz_m_s', measured_vz_m_s)
        if t < 0:
            raise ValueError('t_s must be nonnegative')
        if self._last_t is None:
            if nominal_vertical_thrust_n is not None:
                raise ValueError('first observation has no prior interval; pass None')
            self._last_t = self._window_start_t = t
            self._last_v = self._window_start_v = velocity
            return self._fit()
        if t <= self._last_t:
            raise ValueError('observation times must increase strictly')
        if nominal_vertical_thrust_n is None:
            raise ValueError('subsequent observations require previous-interval nominal thrusts')
        thrust = _four('nominal_vertical_thrust_n', nominal_vertical_thrust_n)
        dt = t-self._last_t
        # Nothing mutates until this call's time/measurement/input is validated.
        self._selected_input += thrust[self.motor]*dt
        self._total_input += math.fsum(thrust)*dt
        self._last_t, self._last_v = t, velocity
        window_duration = t-self._window_start_t
        if window_duration >= self.window_s - 1e-10:
            response = (self.mass_kg*(velocity-self._window_start_v)
                        + self.mass_kg*GRAVITY_M_S2*window_duration - self._total_input)
            self._windows.append(_Window(self._window_start_t, t, self._selected_input, response))
            self._window_start_t, self._window_start_v = t, velocity
            self._selected_input = self._total_input = 0.0
        cutoff = t-self.history_s
        while self._windows and self._windows[0].start_s < cutoff-1e-10:
            self._windows.popleft()
        return self._fit()

    def _fit(self):
        n = len(self._windows)
        excitation = 0.0
        condition = None
        motor_valid = False
        joint_valid = False
        status = 'waiting_for_windows'
        if n:
            xs = [window.input_ns for window in self._windows]
            ys = [window.response_ns for window in self._windows]
            durations = [window.duration_s for window in self._windows]
            xx = math.fsum(x*x for x in xs)
            if xx > 1e-16:
                self._motor_eta = 1.0 + math.fsum(x*y for x, y in zip(xs, ys))/xx
                motor_valid = True
            # Weighted means preserve the integrated-regression objective when
            # observation intervals do not divide window_s exactly.
            weights = [duration*duration for duration in durations]
            total_weight = math.fsum(weights)
            force_x = [x/duration for x, duration in zip(xs, durations)]
            force_y = [y/duration for y, duration in zip(ys, durations)]
            mean_x = math.fsum(w*x for w, x in zip(weights, force_x))/total_weight
            mean_y = math.fsum(w*y for w, y in zip(weights, force_y))/total_weight
            variance_x = math.fsum(w*(x-mean_x)**2 for w, x in zip(weights, force_x))/total_weight
            excitation = math.sqrt(max(0.0, variance_x))
            rms_x = math.sqrt(math.fsum(w*x*x for w, x in zip(weights, force_x))/total_weight)
            # Columns [selected input / RMS(selected input), 1] have equal
            # weighted norm. Their singular-value ratio measures collinearity.
            correlation = abs(mean_x)/rms_x if rms_x > 1e-12 else 1.0
            if correlation < 1.0-1e-14:
                condition = math.sqrt((1.0+correlation)/(1.0-correlation))
            if n < 3:
                status = 'waiting_for_windows'
            elif excitation < self.min_excitation_std_n or condition is None or condition > 1e6:
                status = 'insufficient_excitation'
            else:
                covariance = math.fsum(w*(x-mean_x)*(y-mean_y)
                                      for w, x, y in zip(weights, force_x, force_y))/total_weight
                slope = covariance/variance_x
                eta = 1.0+slope
                force = mean_y-slope*mean_x
                if math.isfinite(eta) and math.isfinite(force):
                    self._joint_eta, self._force = eta, force
                    joint_valid, status = True, 'identified'
                else:
                    status = 'numerical_failure'
        return {
            'motor_only_eta': self._motor_eta,
            'joint_eta': self._joint_eta,
            'joint_force_z_n': self._force,
            'joint_identified': joint_valid,
            'motor_only_identified': motor_valid,
            'excitation_std_n': excitation,
            'window_condition_number': condition,
            'sample_count': n,
            'joint_status': status,
        }


class NominalRPMActuatorModel:
    """Command-only replay of the inspected Isaac RPM-state motor equations.

    The official implementation was inspected at:
    https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_contrib/isaaclab_contrib/actuators/thruster.py

    Native thrust limits clip the command before dynamics. A separate rise/fall
    time constant selects a mixing factor; a rate-limited RPM state advances by
    Euler or RK4; squared RPM produces the next internal thrust state; native
    limits clip the returned output without replacing that internal state.
    Despite its source parameter name, ``max_rate`` limits RPM-state rate in
    these equations, rather than directly limiting thrust change in N/s.

    Parameters and initial *nominal* state must be known from healthy setup or
    calibration. Inputs after construction are bounded issued commands only;
    fault severity, actual force and post-fault actuator outputs are forbidden.
    Each parameter accepts a scalar or four values. One fixed integration dt is
    enforced because it also enters the discrete mixing factor.

    This is not a validation of the NVIDIA runtime. Check the installed actuator
    source, and check that the command-only predictions match the actuator's nominal
    output at every step within a stated numerical tolerance. If they disagree, stop
    the run rather than correcting this state from post-fault measurements.
    """

    def __init__(self, tau_inc_s, tau_dec_s, thrust_const, max_rate,
                 min_thrust_n, max_thrust_n, initial_thrust_n,
                 integration_scheme='rk4', use_discrete_approximation=True,
                 dt_s=0.01):
        if integration_scheme not in ('euler', 'rk4'):
            raise ValueError("integration_scheme must be 'euler' or 'rk4'")
        if not isinstance(use_discrete_approximation, bool):
            raise ValueError('use_discrete_approximation must be a bool')
        self.integration_scheme = integration_scheme
        self.use_discrete_approximation = use_discrete_approximation
        self.dt_s = _positive('dt_s', dt_s)
        self.tau_inc_s = _four_or_scalar('tau_inc_s', tau_inc_s)
        self.tau_dec_s = _four_or_scalar('tau_dec_s', tau_dec_s)
        self.thrust_const = _four_or_scalar('thrust_const', thrust_const)
        self.max_rate = _four_or_scalar('max_rate', max_rate)
        self.min_thrust_n = _four_or_scalar('min_thrust_n', min_thrust_n)
        self.max_thrust_n = _four_or_scalar('max_thrust_n', max_thrust_n)
        self._state = _four_or_scalar('initial_thrust_n', initial_thrust_n)
        if any(tau < 0 or (tau == 0 and not use_discrete_approximation)
               for tau in (*self.tau_inc_s, *self.tau_dec_s)):
            raise ValueError('time constants must be positive for continuous mixing and nonnegative for discrete mixing')
        if any(coefficient <= 0 for coefficient in self.thrust_const):
            raise ValueError('thrust_const must be positive')
        if any(rate < 0 for rate in self.max_rate):
            raise ValueError('max_rate must be nonnegative')
        if any(not 0 <= minimum <= maximum or maximum <= 0
               for minimum, maximum in zip(self.min_thrust_n, self.max_thrust_n)):
            raise ValueError('each native thrust range must satisfy 0 <= min <= max and max > 0')
        if any(value < 0 for value in self._state):
            raise ValueError('initial nominal thrust state must be nonnegative')

    @property
    def state_n(self) -> tuple[float, ...]:
        """Immutable internal nominal thrust-state snapshot, before output clipping."""
        return self._state

    def step(self, command_n, dt_s) -> tuple[float, ...]:
        command = _four('command_n', command_n)
        dt = _positive('dt_s', dt_s)
        if not math.isclose(dt, self.dt_s, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('step dt_s must match the fixed actuator-model dt_s')
        updated = []
        for i, issued in enumerate(command):
            target = min(self.max_thrust_n[i], max(self.min_thrust_n[i], issued))
            state = self._state[i]
            tau = self.tau_dec_s[i] if state > 0 and target < state else self.tau_inc_s[i]
            mixing = 1.0/(tau+dt) if self.use_discrete_approximation else 1.0/tau
            coefficient = self.thrust_const[i]
            rpm = math.sqrt(max(0.0, state/coefficient))
            desired_rpm = math.sqrt(max(0.0, target/coefficient))
            error = desired_rpm-rpm
            limit = self.max_rate[i]

            def rate(rpm_error):
                return min(limit, max(-limit, mixing*rpm_error))

            first = rate(error)
            if self.integration_scheme == 'euler':
                change = dt*first
            else:
                second = rate(error-0.5*dt*first)
                third = rate(error-0.5*dt*second)
                fourth = rate(error-dt*third)
                change = dt*(first+2.0*second+2.0*third+fourth)/6.0
            next_state = coefficient*(rpm+change)**2
            if not math.isfinite(next_state):
                raise ValueError('nominal RPM replay produced a nonfinite state')
            updated.append(next_state)
        # Commit only after all four outputs have passed the checks.
        self._state = tuple(updated)
        return tuple(min(maximum, max(minimum, value))
                     for value, minimum, maximum in zip(self._state, self.min_thrust_n, self.max_thrust_n))
