"""Self-test for faults/thrust_loss.py. Run:  python faults/test_faults.py  (or with pytest)."""

import pathlib
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # experiment folder

import torch

from faults.thrust_loss import (HOVER_MARGIN, LOSS_RANGE, NUM_MOTORS, POST_FAULT_T, SETTLE_T, FaultSpec, apply,
                                is_active, run_duration, sample_fault)

DURATION = 6.0  # default travel time: 3 m at 0.5 m/s (the hover time does not affect the fault)


def test_deterministic():
    assert sample_fault(7, DURATION) == sample_fault(7, DURATION)


def test_different_seeds_differ():
    specs = {tuple(sample_fault(s, DURATION).to_dict().values()) for s in range(50)}
    assert len(specs) == 50


def test_ranges():
    motors = set()
    for s in range(2000):
        f = sample_fault(s, DURATION)
        motors.add(f.motor)
        assert 0 <= f.motor < NUM_MOTORS
        assert LOSS_RANGE[0] <= f.loss <= LOSS_RANGE[1]
        assert SETTLE_T <= f.onset_t <= DURATION - HOVER_MARGIN  # never near the hover phase
    assert motors == set(range(NUM_MOTORS))  # every motor gets picked eventually


def test_apply():
    spec = FaultSpec(motor=2, loss=0.3, onset_t=4.0)
    cmd = torch.full((5, 4), 3.0)
    before = apply(cmd, spec, 3.99)
    after = apply(cmd, spec, 4.0)
    assert not is_active(spec, 3.99) and is_active(spec, 4.0)
    assert torch.equal(before, cmd)
    assert torch.allclose(after[:, 2], torch.full((5,), 2.1))
    assert torch.equal(after[:, [0, 1, 3]], cmd[:, [0, 1, 3]])
    assert torch.equal(cmd, torch.full((5, 4), 3.0))  # input not modified
    assert after is not cmd
    assert apply(cmd, None, 100.0) is cmd


def test_run_duration():
    assert run_duration(9.0, None) == 9.0  # no fault: the normal flight
    assert run_duration(9.0, FaultSpec(0, 0.2, 4.0)) == 4.0 + POST_FAULT_T  # extended to 10 s after the fault
    assert run_duration(30.0, FaultSpec(0, 0.2, 4.0)) == 30.0  # already long enough


def test_too_short():
    try:
        sample_fault(0, SETTLE_T + HOVER_MARGIN - 0.01)
    except ValueError as e:
        assert "--speed" in str(e)
    else:
        raise AssertionError("expected ValueError for a too-short route")
    sample_fault(0, SETTLE_T + HOVER_MARGIN)  # zero-width window is still valid


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"[OK] {fn.__name__}")
    print(f"[OK] {len(tests)} tests passed; example spec seed=0: {sample_fault(0, DURATION).to_dict()}")
