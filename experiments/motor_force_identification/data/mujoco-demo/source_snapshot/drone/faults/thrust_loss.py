"""Partial thrust loss on one motor, starting at a random time while the drone is travelling from A to B.

A FaultSpec says which motor fails, how much thrust it loses and when. sample_fault() draws one
deterministically from a seed, so every controller can be tested against the exact same fault.
apply() turns the controller's commanded thrusts into what the damaged drone actually produces.
run_duration() makes the run last long enough to watch the drone react to the fault.

Simulator-independent: numpy + torch only (no isaaclab / isaacsim / omni / warp).

Usage (inside the flight loop):
    spec = sample_fault(seed, travel_time)          # travel_time = |AB| / speed
    duration = run_duration(travel_time + hold, spec)
    thrust_applied = apply(thrust_cmd, spec, t)
"""

from dataclasses import asdict, dataclass

import numpy as np

LOSS_RANGE = (0.10, 0.50)  # fraction of thrust lost by the faulty motor
SETTLE_T = 3.0  # s; earliest onset: the drone sags in its first second and has settled by ~3 s
HOVER_MARGIN = 1.0  # s; latest onset is this long before the drone reaches B (no faults near the hover phase)
POST_FAULT_T = 10.0  # s; every run continues at least this long after the fault starts
NUM_MOTORS = 4


@dataclass
class FaultSpec:
    motor: int  # index 0..NUM_MOTORS-1
    loss: float  # fraction 0..1 of thrust lost
    onset_t: float  # s, time the fault starts

    def to_dict(self) -> dict:
        return asdict(self)


def sample_fault(seed: int, travel_time: float) -> FaultSpec:
    """Random fault, identical for identical seeds and routes.

    Onset is uniform in [SETTLE_T, travel_time - HOVER_MARGIN], i.e. while the drone is still on its way to B.
    `travel_time` is the time the target point needs to reach B (|AB| / speed); the hover time does not matter.
    """
    lo, hi = SETTLE_T, travel_time - HOVER_MARGIN
    if hi < lo:
        raise ValueError(
            f"Route too short for a fault: the drone reaches B after {travel_time:.2f} s, but the onset must lie in "
            f"[{SETTLE_T} s, {HOVER_MARGIN} s before reaching B]. Use a longer route or a lower --speed."
        )
    rng = np.random.default_rng(seed)
    motor = int(rng.integers(0, NUM_MOTORS))
    loss = float(rng.uniform(*LOSS_RANGE))
    onset_t = float(rng.uniform(lo, hi))
    return FaultSpec(motor=motor, loss=loss, onset_t=onset_t)


def run_duration(base_duration: float, spec: FaultSpec | None) -> float:
    """Seconds the run lasts: the normal flight, extended so it ends at least POST_FAULT_T after the fault starts."""
    if spec is None:
        return base_duration
    return max(base_duration, spec.onset_t + POST_FAULT_T)


def is_active(spec: FaultSpec | None, t: float) -> bool:
    return spec is not None and t >= spec.onset_t


def apply(thrust_cmd, spec: FaultSpec | None, t: float):
    """Thrusts (N, NUM_MOTORS) torch tensor -> new tensor with the faulty motor scaled by (1 - loss) after onset.

    Returns the input itself (no copy) if there is no fault.
    """
    if spec is None:
        return thrust_cmd
    out = thrust_cmd.clone()
    if is_active(spec, t):
        out[:, spec.motor] *= 1.0 - spec.loss
    return out
