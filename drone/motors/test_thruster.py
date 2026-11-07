"""Self-test for motors/thruster.py.

Run: python motors/test_thruster.py (or with pytest)
"""

import math
import pathlib
import sys
from itertools import pairwise

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # drone/

import torch

from motors.thruster import (
    INIT_RPS,
    TAU_DEC_RANGE,
    TAU_INC_RANGE,
    THRUST_CONST_RANGE,
    THRUST_RANGE,
    ThrusterModel,
)

DT = 0.01


def reference_step(thrust, des, k, tau_inc, tau_dec, dt=DT):
    """Isaac's Thruster.compute for one motor, written out with plain floats."""
    des = min(max(des, THRUST_RANGE[0]), THRUST_RANGE[1])
    tau = (
        tau_dec
        if math.copysign(1, thrust) * math.copysign(1, des - thrust) < 0
        and des != thrust
        else tau_inc
    )
    mix = 1.0 / (dt + tau)

    def rate(e):
        return max(-1e5, min(1e5, mix * e))

    rpm = math.sqrt(max(thrust / k, 0.0))
    e = math.sqrt(max(des / k, 0.0)) - rpm
    k1 = rate(e)
    k2 = rate(e - 0.5 * dt * k1)
    k3 = rate(e - 0.5 * dt * k2)
    k4 = rate(e - dt * k3)
    rpm += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return k * rpm**2


def test_matches_isaac_formula():
    m = ThrusterModel(DT)
    m.reset(3)
    state = [float(v) for v in m.thrust[0]]
    for cmd in [3.0] * 30 + [0.5] * 10 + [12.0] * 20 + [2.0] * 5:
        out = m.step(torch.full((1, 4), cmd))
        for i in range(4):
            state[i] = reference_step(
                state[i],
                cmd,
                float(m.thrust_const[0, i]),
                float(m.tau_inc[0, i]),
                float(m.tau_dec[0, i]),
            )
            assert abs(float(m.thrust[0, i]) - state[i]) < 1e-4 * max(1.0, state[i]), (
                cmd,
                i,
            )
            assert THRUST_RANGE[0] <= float(out[0, i]) <= THRUST_RANGE[1]


def test_seeded_parameters():
    a, b, c = ThrusterModel(DT), ThrusterModel(DT), ThrusterModel(DT)
    a.reset(5), b.reset(5), c.reset(6)
    assert torch.equal(a.thrust_const, b.thrust_const) and torch.equal(
        a.tau_inc, b.tau_inc
    )
    assert not torch.equal(a.thrust_const, c.thrust_const)
    assert ((a.tau_inc >= TAU_INC_RANGE[0]) & (a.tau_inc <= TAU_INC_RANGE[1])).all()
    assert torch.allclose(a.tau_dec, torch.full((1, 4), TAU_DEC_RANGE[0]))
    assert (
        (a.thrust_const >= THRUST_CONST_RANGE[0] * 0.999)
        & (a.thrust_const <= THRUST_CONST_RANGE[1] * 1.001)
    ).all()
    # starts at the initial spin (~0.4-0.7 N)
    assert torch.allclose(a.thrust, a.thrust_const * INIT_RPS**2)


def test_lag():
    m = ThrusterModel(DT)
    m.reset(0)
    up = [float(m.step(torch.full((1, 4), 3.0))[0, 0]) for _ in range(100)]
    # spins up over a few tenths of a second, then holds
    assert up[0] < 1.5 and up[-1] > 2.99
    assert all(b >= a - 1e-6 for a, b in pairwise(up))  # monotonic
    down = [float(m.step(torch.full((1, 4), 1.0))[0, 0]) for _ in range(6)]
    # slowing down is much faster (tau_dec = 5 ms): ~2 N gone in 6 steps
    assert down[-1] < 1.05


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"[OK] {fn.__name__}")
    print(f"[OK] {len(tests)} tests passed")
