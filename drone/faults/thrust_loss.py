"""Partial thrust loss on one motor, starting at a random time on the way to B.

A FaultSpec says which motor fails, how much thrust it loses and when. sample_fault()
draws one deterministically from a seed, so every controller can be tested against the
exact same fault. apply() scales the faulty motor's thrust. The simulator layer applies
it to the motor's OUTPUT (after the motor model and its limits), so the damaged motor's
maximum really drops: a motor that lost 30% can never deliver more than 70% of its
maximum, however hard it is asked (scaling the command instead would let a harder
request restore full power). run_duration() makes the run last long enough to watch the
drone react to the fault. hover_feasibility() says whether the damaged drone can
physically hover at all with its motor limits.

Simulator-independent: numpy + torch only (no isaaclab / isaacsim / omni / warp).

Usage (inside the flight loop):
    spec = sample_fault(seed, travel_time)  # travel_time = |AB| / speed
    duration = run_duration(travel_time + hold, spec)
    thrust_applied = apply(thrust_cmd, spec, t)
"""

from dataclasses import asdict, dataclass

import numpy as np

LOSS_RANGE = (0.10, 0.50)  # fraction of thrust lost by the faulty motor
# s; earliest onset: the drone sags in its first second and has settled by ~3 s
SETTLE_T = 3.0
# s; latest onset is this long before the drone reaches B (no faults while hovering)
HOVER_MARGIN = 1.0
POST_FAULT_T = 10.0  # s; every run continues at least this long after the fault starts
NUM_MOTORS = 4
GRAVITY = 9.81  # m/s^2


@dataclass
class FaultSpec:
    motor: int  # index 0..NUM_MOTORS-1
    loss: float  # fraction 0..1 of thrust lost
    onset_t: float  # s, time the fault starts

    def to_dict(self) -> dict:
        return asdict(self)


def sample_fault(seed: int, travel_time: float) -> FaultSpec:
    """Random fault, identical for identical seeds and routes.

    Onset is uniform in [SETTLE_T, travel_time - HOVER_MARGIN], i.e. while the drone is
    still on its way to B. `travel_time` is the time the target point needs to reach B
    (|AB| / speed); the hover time does not matter.
    """
    lo, hi = SETTLE_T, travel_time - HOVER_MARGIN
    if hi < lo:
        raise ValueError(
            "Route too short for a fault: "
            f"the drone reaches B after {travel_time:.2f} s, "
            "but the onset must lie in "
            f"[{SETTLE_T} s, {HOVER_MARGIN} s before reaching B]. "
            "Use a longer route or a lower --speed."
        )
    rng = np.random.default_rng(seed)
    motor = int(rng.integers(0, NUM_MOTORS))
    loss = float(rng.uniform(*LOSS_RANGE))
    onset_t = float(rng.uniform(lo, hi))
    return FaultSpec(motor=motor, loss=loss, onset_t=onset_t)


def run_duration(base_duration: float, spec: FaultSpec | None) -> float:
    """Seconds the run lasts.

    The normal flight, extended to end at least POST_FAULT_T after the fault starts.
    """
    if spec is None:
        return base_duration
    return max(base_duration, spec.onset_t + POST_FAULT_T)


def is_active(spec: FaultSpec | None, t: float) -> bool:
    return spec is not None and t >= spec.onset_t


def apply(thrust, spec: FaultSpec | None, t: float):
    """Thrusts (N, NUM_MOTORS) -> new tensor, the faulty motor scaled by (1 - loss).

    The scaling starts at the fault onset.

    Returns the input itself (no copy) if there is no fault; otherwise never modifies
    the input (the simulator's motor model keeps its own state in that tensor).
    """
    if spec is None:
        return thrust
    out = thrust.clone()
    if is_active(spec, t):
        out[:, spec.motor] *= 1.0 - spec.loss
    return out


def hover_feasibility(
    allocation,
    mass: float,
    spec: FaultSpec | None,
    max_thrust: float,
    gravity: float = GRAVITY,
) -> dict:
    """Can the (damaged) drone physically hold a hover, given each motor's limit?

    Pure numpy.

    `allocation` is the drone's 6 x NUM_MOTORS matrix (rows Fx, Fy, Fz, Tx, Ty, Tz: what
    each motor's thrust does to the body). Every motor can push between 0 and
    `max_thrust`; the faulty motor only up to max_thrust * (1 - loss).
      feasible: a level hover without spinning exists (lift = weight, no roll / pitch /
          yaw torque). If False, NO controller can keep this drone hovering level: a
          crash is not the controller's fault.
      feasible_if_spinning: a hover exists if the drone may spin about its vertical
          axis (yaw torque ignored). Quadrotors can survive even bigger losses this way,
          but none of our controllers try it; reported for reference.
    """
    A = np.asarray(allocation, dtype=float)[2:6]  # rows Fz, Tx, Ty, Tz
    n = A.shape[1]
    b = np.array([mass * gravity, 0.0, 0.0, 0.0])
    upper = np.full(n, float(max_thrust))
    if spec is not None:
        upper[spec.motor] *= 1.0 - spec.loss
    tol = 1e-9

    # level hover: 4 equations, 4 motors -> one exact set of motor thrusts (equal
    # thrusts on a symmetric quad)
    level = np.linalg.lstsq(A, b, rcond=None)[0]
    feasible = bool(
        np.allclose(A @ level, b, atol=1e-6)
        and (level >= -tol).all()
        and (level <= upper + tol).all()
    )

    # spinning allowed: drop the yaw equation -> a 1-D family level3 + s * direction;
    # find any s inside all limits
    level3 = np.linalg.lstsq(A[:3], b[:3], rcond=None)[0]
    # the motor combination that changes only yaw torque
    direction = np.linalg.svd(A[:3])[2][-1]
    lo, hi = -np.inf, np.inf
    for t0, d, ub in zip(level3, direction, upper):
        if abs(d) < 1e-12:
            if not (-tol <= t0 <= ub + tol):
                lo, hi = 1.0, -1.0  # this motor alone breaks the limits
            continue
        s1, s2 = (0.0 - t0) / d, (ub - t0) / d
        lo, hi = max(lo, min(s1, s2)), min(hi, max(s1, s2))
    return {
        "feasible": feasible,
        "feasible_if_spinning": bool(lo <= hi + tol),
        "hover_thrust_per_motor_n": [float(v) for v in level],
        "motor_max_thrust_n": [float(v) for v in upper],
    }
