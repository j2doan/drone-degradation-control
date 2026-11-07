"""Sim-free sanity tests for controllers/pid.py.

Run: python controllers/test_pid.py
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # drone/

import torch

from controllers.pid import GRAVITY, CascadedPID, quat_mul, quat_to_rotmat

MASS = 1.24
INERTIA = torch.diag(torch.tensor([0.01, 0.01, 0.02]))
DT = 0.01


def state(n=1, pos=(0.0, 0.0, 1.0)):
    """Level, at rest."""
    return (
        torch.tensor([pos] * n),
        torch.zeros(n, 3),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]] * n),
        torch.zeros(n, 3),
    )


def test_hover():
    pid = CascadedPID(1, MASS, INERTIA, DT)
    pos, vel, q, w = state()
    u = pid.compute(pos, vel, q, w, pos.clone(), torch.zeros(1, 3))
    assert u.shape == (1, 6)
    assert abs(u[0, 2].item() - MASS * GRAVITY) < 1e-4, u
    assert u[0, [0, 1, 3, 4, 5]].abs().max().item() < 1e-6, u
    print(
        f"hover: Fz={u[0, 2].item():.4f} (m*g={MASS * GRAVITY:.4f}), "
        f"torques={u[0, 3:].tolist()}"
    )


def test_tilt_sign():
    pid = CascadedPID(1, MASS, INERTIA, DT)
    pos, vel, q, w = state()
    u = pid.compute(
        pos, vel, q, w, pos + torch.tensor([[1.0, 0.0, 0.0]]), torch.zeros(1, 3)
    )
    # Desired thrust direction (world) is reconstructed via the torque: to tilt body z
    # toward +x, the drone must pitch positively about body y (right-hand rule: z
    # rotates toward x).
    assert u[0, 4].item() > 0, u
    assert abs(u[0, 3].item()) < 1e-6 and abs(u[0, 5].item()) < 1e-6, u
    print(
        f"tilt: setpoint +x -> Ty={u[0, 4].item():.4f} > 0 (pitch nose toward +x), "
        f"Tx={u[0, 3].item():.2e}"
    )


def simulate(pid, pos, vel, q, w, sp_pos, seconds, inertia=INERTIA):
    """Rigid-body quadrotor: thrust along body z, body torques; semi-implicit Euler."""
    n = pos.shape[0]
    J = inertia.expand(n, 3, 3)
    J_inv = torch.linalg.inv(J)
    g = torch.tensor([0.0, 0.0, -GRAVITY])
    max_tilt = 0.0
    for _ in range(round(seconds / DT)):
        u = pid.compute(pos, vel, q, w, sp_pos, torch.zeros_like(pos))
        R = quat_to_rotmat(q)
        acc = R[:, :, 2] * (u[:, 2:3] / MASS) + g
        Jw = torch.bmm(J, w.unsqueeze(-1)).squeeze(-1)
        w_dot = torch.bmm(
            J_inv, (u[:, 3:] - torch.cross(w, Jw, dim=-1)).unsqueeze(-1)
        ).squeeze(-1)
        vel = vel + acc * DT
        pos = pos + vel * DT
        w = w + w_dot * DT
        # q_dot = 0.5 q (x) [w, 0]
        q = q + 0.5 * quat_mul(q, torch.cat([w, torch.zeros(n, 1)], dim=-1)) * DT
        q = q / q.norm(dim=-1, keepdim=True)
        max_tilt = max(max_tilt, torch.acos(R[:, 2, 2].clamp(-1, 1)).max().item())
    return pos, vel, q, w, max_tilt


def test_closed_loop():
    pid = CascadedPID(1, MASS, INERTIA, DT)
    pos, vel, q, w = state()
    sp = torch.tensor([[1.0, 0.0, 1.0]])
    pos, vel, q, w, max_tilt = simulate(pid, pos, vel, q, w, sp, 6.0)
    err = (pos - sp).norm().item()
    up = quat_to_rotmat(q)[0, 2, 2].item()
    assert torch.isfinite(pos).all() and torch.isfinite(q).all()
    assert err < 0.05, err
    assert up > 0.99, up
    assert max_tilt < 1.047 + 0.05, max_tilt
    print(
        f"closed loop 6 s: final pos={[round(v, 4) for v in pos[0].tolist()]}, "
        f"err={err:.4f} m, "
        f"body_z.z={up:.5f}, max tilt={math.degrees(max_tilt):.1f} deg"
    )


def test_batched_and_reset():
    n = 8
    pid = CascadedPID(n, MASS, INERTIA.expand(n, 3, 3), DT)
    pos, vel, q, w = state(n)
    sp = pos + torch.randn(n, 3) * 0.5
    # 8 s: with the attitude integral, random setpoints up to ~1 m away take a little
    # longer to settle exactly; the integral is what removes the steady offsets a weak
    # motor causes in Isaac.
    pos, vel, q, w, _ = simulate(pid, pos, vel, q, w, sp, 8.0)
    err = (pos - sp).norm(dim=-1)
    assert torch.isfinite(pos).all() and (err < 0.05).all(), err
    assert pid.int_pos.abs().sum() > 0
    pid.reset(torch.tensor([0, 3]))
    assert pid.int_pos[[0, 3]].abs().sum() == 0 and pid.int_pos[1].abs().sum() > 0
    pid.reset()
    assert pid.int_pos.abs().sum() == 0 and pid.int_att.abs().sum() == 0
    print(
        f"batched N={n}: max final err={err.max().item():.4f} m; "
        "reset(env_ids) and reset() zero integrators"
    )


def test_saturation_freezes_integrators():
    n = 3
    pid = CascadedPID(n, MASS, INERTIA, DT)
    pos, vel, _, w = state(n)
    # small offset (inside the integration zone) and a slight tilt: both integrals move
    sp = pos + torch.tensor([[0.1, 0.0, 0.1]] * n)
    tilt = torch.tensor([[0.05, 0.0, 0.0, 1.0]] * n)
    tilt = tilt / tilt.norm(dim=-1, keepdim=True)
    for _ in range(5):
        pid.compute(pos, vel, tilt, w, sp, torch.zeros(n, 3))
    before_pos, before_att = pid.int_pos.clone(), pid.int_att.clone()
    pid.set_saturated(torch.tensor([False, True, False]))
    pid.compute(pos, vel, tilt, w, sp, torch.zeros(n, 3))
    assert torch.equal(pid.int_pos[1], before_pos[1]) and torch.equal(
        pid.int_att[1], before_att[1]
    )
    assert (pid.int_pos[[0, 2]] != before_pos[[0, 2]]).any(dim=-1).all()
    assert (pid.int_att[[0, 2]] != before_att[[0, 2]]).any(dim=-1).all()
    assert not pid.saturated.any()  # mask consumed
    # next step (no new mask): env 1 integrates again
    mid_pos = pid.int_pos.clone()
    pid.compute(pos, vel, tilt, w, sp, torch.zeros(n, 3))
    assert (pid.int_pos[1] != mid_pos[1]).any()
    # reset clears a pending mask
    pid.set_saturated(torch.ones(n, dtype=torch.bool))
    pid.reset(torch.tensor([0]))
    assert pid.saturated.tolist() == [False, True, True]
    print(
        "saturation: flagged env kept its integrators for one step, others updated; "
        "mask consumed; reset clears it"
    )


def test_saturation_never_set_is_unchanged():
    """set_saturated(all False) gives bit-identical output to never calling it."""
    pos, vel, q, w = state(2)
    sp = pos + torch.tensor([[0.3, -0.2, 0.1], [0.0, 0.1, -0.1]])
    a, b = CascadedPID(2, MASS, INERTIA, DT), CascadedPID(2, MASS, INERTIA, DT)
    for _ in range(20):
        b.set_saturated(torch.zeros(2, dtype=torch.bool))
        assert torch.equal(
            a.compute(pos, vel, q, w, sp, torch.zeros(2, 3)),
            b.compute(pos, vel, q, w, sp, torch.zeros(2, 3)),
        )
    print("saturation: an all-False mask is identical to never calling set_saturated")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_hover()
    test_tilt_sign()
    test_closed_loop()
    test_batched_and_reset()
    test_saturation_freezes_integrators()
    test_saturation_never_set_is_unchanged()
    print("ALL PID TESTS PASSED")
