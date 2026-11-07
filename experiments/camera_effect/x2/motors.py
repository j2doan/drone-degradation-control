"""Motor model and thrust-loss fault.

Order inside one physics step (the same in every flight):
    controller request -> cap to [0, MAX_THRUST] -> first-order motor lag -> fault (damaged motor's output scaled
    by 1 - loss) -> MuJoCo.
The fault acts on the motor's output, after the lag: a damaged propeller or a weak motor produces less thrust than
the motor is spinning for. The controller never sees the fault; it only notices that the drone sags and tilts.
"""

from dataclasses import asdict, dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class Fault:
    motor: int  # 1-4
    loss: float  # fraction of thrust lost, 0-1
    onset_t: float  # s

    def to_dict(self):
        return asdict(self)


class Motors:
    def __init__(self, initial_thrust, fault: Fault | None, dt: float = config.SIM_DT,
                 tau: float = config.MOTOR_TAU, cap: float = config.MAX_THRUST):
        self.thrust = np.asarray(initial_thrust, dtype=float).copy()  # spinning thrust, before the fault
        self.fault, self.cap = fault, cap
        self.alpha = 1.0 - np.exp(-dt / tau)  # exact discretisation of a first-order lag
        if fault is not None and not (1 <= fault.motor <= 4 and 0.0 <= fault.loss <= 1.0):
            raise ValueError(f"bad fault {fault}")

    def step(self, request, t: float):
        """Advance one step. Returns (capped command, spinning thrust, applied thrust, fault active)."""
        command = np.clip(request, 0.0, self.cap)
        self.thrust = self.thrust + self.alpha * (command - self.thrust)
        applied = self.thrust.copy()
        active = self.fault is not None and t >= self.fault.onset_t - 1e-9
        if active:
            applied[self.fault.motor - 1] *= 1.0 - self.fault.loss
        return command, self.thrust.copy(), applied, active
