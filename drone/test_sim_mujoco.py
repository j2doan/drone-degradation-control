"""Self-test for sim_mujoco.py (needs the MuJoCo venv).

Run: .venv-mujoco/Scripts/python test_sim_mujoco.py

Checks that the MuJoCo drone really is the ARL Robot 1 the controller expects, and that
flights repeat exactly.
"""

import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # drone/

import mujoco
import numpy as np
import sim_mujoco
from drones import DRONES


def make_args(**kw):
    a = {
        "sim": "mujoco",
        "speed": 0.5,
        "hold": 3.0,
        "max_steps": -1,
        "controller": "pid",
        "fault": "random",
        "fault_motor": 0,
        "fault_loss": 0.25,
        "fault_onset": 4.0,
        "max_thrust": 6.0,
        "camera": False,
        "camera_res": 64,
        "headless": True,
        "visualizer": [],
        "device": "cpu",
    }
    a.update(kw)
    return SimpleNamespace(**a)


def test_mass_and_inertia():
    m = mujoco.MjModel.from_xml_string(sim_mujoco.model_xml())
    body = m.body("base_link").id
    assert abs(m.body_subtreemass[body] - sim_mujoco.MASS) < 1e-9
    assert np.allclose(m.body_inertia[body], sim_mujoco.INERTIA)
    assert np.allclose(m.body_ipos[body], 0.0)  # center of mass at the body origin


def test_wrench_equals_allocation():
    """Each motor alone, at 1 N, must give that column of the allocation matrix.

    Checks MuJoCo's force and torque on the body.
    """
    m = mujoco.MjModel.from_xml_string(sim_mujoco.model_xml())
    d = mujoco.MjData(m)
    for i in range(4):
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 1.0
        d.ctrl[:] = 0.0
        d.ctrl[i] = 1.0
        mujoco.mj_forward(m, d)
        # free joint at identity pose: qfrc_actuator = [world force, body torque]
        assert np.allclose(
            d.qfrc_actuator, np.asarray(sim_mujoco.ALLOCATION)[:, i], atol=1e-9
        ), (i, d.qfrc_actuator)


def test_hover_and_imu():
    """Hover thrust on every motor: no acceleration.

    The IMU reads +9.81 m/s^2 on z (proper acceleration).
    """
    m = mujoco.MjModel.from_xml_string(sim_mujoco.model_xml())
    d = mujoco.MjData(m)
    d.qpos[2] = 1.0
    d.ctrl[:] = sim_mujoco.MASS * 9.81 / 4
    for _ in range(100):
        mujoco.mj_step(m, d)
    assert abs(d.qpos[2] - 1.0) < 1e-6 and np.allclose(d.qvel, 0.0, atol=1e-6)
    assert np.allclose(d.sensor("acc").data, [0, 0, 9.81], atol=1e-4)


def test_ground_clearance():
    m = mujoco.MjModel.from_xml_string(sim_mujoco.model_xml())
    d = mujoco.MjData(m)
    d.qpos[2] = 0.5
    for _ in range(300):  # motors off: fall and settle on the legs
        mujoco.mj_step(m, d)
    assert abs(d.qpos[2] - sim_mujoco.GROUND_CLEARANCE) < 0.01, d.qpos[2]


def _fly(seed, steps):
    args = make_args(max_steps=steps)
    with sim_mujoco.session(args) as holder:
        world = sim_mujoco.make_world(args, holder)
        flight = sim_mujoco.prepare_flight(args, world, seed, DRONES["d0"])
        return list(sim_mujoco.fly(args, world, flight)), world


def test_repeatable_and_resets():
    a, world = _fly(2, 300)
    b, _ = _fly(2, 300)
    assert a == b  # same seed -> identical log
    # a second flight in the same world starts from scratch
    args = make_args(max_steps=300)
    flight = sim_mujoco.prepare_flight(args, world, 2, DRONES["d0"])
    assert list(sim_mujoco.fly(args, world, flight)) == a


def test_short_healthy_flight():
    rows, _ = _fly(0, 600)  # 6 s on seed 0's path
    last = rows[-1]
    err = np.linalg.norm([last[f"pos_{k}"] - last[f"setpoint_{k}"] for k in "xyz"])
    assert err < 0.3, err  # following the moving target
    assert all(
        0.1 - 1e-6 <= r[f"thrust_applied_{i}"] <= 10.0 for r in rows for i in range(4)
    )


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"[OK] {fn.__name__}")
    print(f"[OK] {len(tests)} tests passed")
