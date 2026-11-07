"""Pure specifications for a single-motor loss of thrust effectiveness.

Effectiveness multiplies THRUST, not RPM. A 0.75 factor means 25% less thrust;
using it as an RPM factor would instead produce 43.75% less thrust for a
quadratic thrust curve. This module does not modify an Isaac actuator model.
The caller must apply factors at its verified physical actuator interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


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


@dataclass(frozen=True)
class StepMotorFault:
    """A hidden step loss on one of four motors, with optional restoration.

    The new factor applies at ``onset_s`` and restores to one at
    ``onset_s + duration_s``. A missing duration means a permanent loss.
    Effectiveness one is a healthy control; zero specifies total loss but does
    not imply that a four-rotor controller can recover from that condition.
    Onset is an absolute simulation time; settling logic belongs to the runner.
    """

    motor: int
    effectiveness: float = 0.75
    onset_s: float = 5.0
    duration_s: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.motor, bool) or not isinstance(self.motor, int) or not 0 <= self.motor < 4:
            raise ValueError("motor must be an integer in [0, 3]")
        effectiveness = _finite("effectiveness", self.effectiveness)
        onset = _finite("onset_s", self.onset_s)
        if not 0.0 <= effectiveness <= 1.0:
            raise ValueError("effectiveness must lie in [0, 1]")
        if onset < 0.0:
            raise ValueError("onset_s must be nonnegative")
        object.__setattr__(self, "effectiveness", effectiveness)
        object.__setattr__(self, "onset_s", onset)
        if self.duration_s is not None:
            duration = _finite("duration_s", self.duration_s)
            if duration <= 0.0 or not math.isfinite(onset + duration):
                raise ValueError("duration_s must be positive with a finite restoration time")
            object.__setattr__(self, "duration_s", duration)

    def factors_at(self, t: float, num_motors: int = 4) -> tuple[float, ...]:
        """Return factors for commands applied starting at time ``t``.

        There is no early warning or transition smoothing. The runner should
        sample this at its chosen control/physics boundary and record that time.
        """
        t = _finite("t", t)
        if t < 0.0:
            raise ValueError("t must be nonnegative")
        if isinstance(num_motors, bool) or not isinstance(num_motors, int) or num_motors <= 0:
            raise ValueError("num_motors must be a positive integer")
        if self.motor >= num_motors:
            raise ValueError("fault motor is outside num_motors")
        active = t >= self.onset_s and (
            self.duration_s is None or t < self.onset_s + self.duration_s
        )
        return tuple(self.effectiveness if active and i == self.motor else 1.0 for i in range(num_motors))


def apply_effectiveness(commands: Iterable[float], factors: Iterable[float]) -> tuple[float, ...]:
    """Return scaled thrust values without modifying inputs or clipping limits.

    This numerical helper is not a simulator fault-injection adapter. Values
    must be nonnegative finite thrusts and matching factors in [0, 1].
    """
    commands = tuple(_finite("command", value) for value in commands)
    factors = tuple(_finite("factor", value) for value in factors)
    if not commands or len(commands) != len(factors):
        raise ValueError("commands and factors must be nonempty and have equal lengths")
    if any(command < 0.0 for command in commands):
        raise ValueError("commands must be nonnegative thrust values")
    if any(not 0.0 <= factor <= 1.0 for factor in factors):
        raise ValueError("factors must lie in [0, 1]")
    return tuple(command * factor for command, factor in zip(commands, factors))
