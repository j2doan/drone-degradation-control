"""Self-test for drones.py.

Run: python test_drones.py (or with pytest)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # drone/

import torch
from drones import (
    DRONES,
    NUM_MOTORS,
    OBS_NAMES,
    RESIDUAL_SCALE,
    available,
    combine,
    make_observation,
    register_policy,
)


class DummyPolicy:
    """Always outputs the same value per motor."""

    def __init__(self, value: float):
        self.value, self.resets = value, 0

    def act(self, obs):
        return torch.full((obs.shape[0], NUM_MOTORS), self.value)

    def reset(self, env_ids=None):
        self.resets += 1


def test_registry():
    assert set(DRONES) == {"d0", "d1", "d2", "d3"}
    assert [d.mode for d in DRONES.values()] == [
        "pid",
        "residual",
        "residual",
        "replace",
    ]
    assert [d.name for d in available()] == ["d0"]
    spec = register_policy("d1", DummyPolicy(0.5))
    try:
        assert [d.name for d in available()] == ["d0", "d1"]
        assert spec.policy.act(torch.zeros(2, len(OBS_NAMES))).shape == (2, NUM_MOTORS)
        for bad in [
            ("d9", DummyPolicy(0.0)),
            ("d0", DummyPolicy(0.0)),
            ("d2", object()),
        ]:
            try:
                register_policy(*bad)
            except (KeyError, ValueError, TypeError):
                pass
            else:
                raise AssertionError(f"register_policy{bad} should fail")
    finally:
        DRONES["d1"].policy = None  # leave the registry as found
    assert [d.name for d in available()] == ["d0"]
    print(
        "registry: available() == [d0] by default; "
        "registering d1 makes it available; bad registrations refused"
    )


def test_combine():
    pid = torch.tensor([[3.0, 3.0, 3.0, 3.0], [2.0, 4.0, 2.0, 4.0]])
    out = torch.tensor([[1.0, -1.0, 0.5, 0.0], [0.0, 0.0, -0.5, 1.0]])
    assert torch.equal(combine(pid, None, "pid"), pid)
    assert torch.allclose(combine(pid, out, "residual"), pid + RESIDUAL_SCALE * out)
    assert torch.allclose(
        combine(pid, out, "residual", residual_scale=0.2), pid + 0.2 * out
    )
    assert torch.equal(combine(None, out, "replace"), out)
    try:
        combine(pid, out, "nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mode should fail")
    print(f"combine: pid / residual (scale {RESIDUAL_SCALE} N) / replace all correct")


def test_observation():
    n = 3
    pos = torch.tensor([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 0.2]])
    vel = torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.3]])
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * n)
    w = torch.randn(n, 3)
    acc, gyro = torch.randn(n, 3), torch.randn(n, 3)
    tpos = torch.tensor([[3.0, 0.0, 1.0]] * n)
    tvel = torch.tensor([[0.5, 0.0, 0.0]] * n)
    last = torch.rand(n, NUM_MOTORS) * 6
    obs = make_observation(pos, vel, quat, w, acc, gyro, tpos, tvel, last)
    assert obs.shape == (n, len(OBS_NAMES)) and obs.dtype == torch.float32
    assert len(set(OBS_NAMES)) == len(OBS_NAMES)
    col = {name: i for i, name in enumerate(OBS_NAMES)}
    rel_pos = obs[:, [col[f"target_rel_pos_{a}"] for a in "xyz"]]
    rel_vel = obs[:, [col[f"target_rel_vel_{a}"] for a in "xyz"]]
    assert torch.allclose(rel_pos, tpos - pos) and torch.allclose(rel_vel, tvel - vel)
    assert torch.allclose(
        obs[:, [col[f"last_thrust_{i}"] for i in range(NUM_MOTORS)]], last
    )
    assert torch.allclose(obs[:, [col[f"imu_lin_acc_{a}"] for a in "xyz"]], acc)
    assert not any(
        "fault" in name or "motor" in name or "loss" in name for name in OBS_NAMES
    )
    print(
        f"observation: shape {tuple(obs.shape)} "
        f"== (N, len(OBS_NAMES)={len(OBS_NAMES)}); "
        "relative target correct;"
        " no fault fields"
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    test_registry()
    test_combine()
    test_observation()
    print("ALL DRONES TESTS PASSED")
