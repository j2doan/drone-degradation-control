"""Skydio X2 MuJoCo physics used by the identification pilot.

The PID controller and the X2 model assets are bundled with this experiment.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MOTOR_TAU_S = 0.125
MIN_HEIGHT_M = 0.05
MAX_TILT_DEG = 60.0


def resolve_scene(scene_path: Path | None = None) -> Path:
    """Resolve an explicit scene or the bundled model inside this experiment."""
    if scene_path is None:
        menagerie = PROJECT_ROOT/'assets/mujoco_menagerie'
        scene_path = menagerie/'skydio_x2/scene.xml'
    scene = Path(scene_path).expanduser().resolve()
    if not scene.is_file():
        raise FileNotFoundError(
            f'Skydio X2 scene not found: {scene}. Keep this experiment’s assets directory '
            'together with its source files.')
    return scene


def load_controller():
    """Load the bundled PID controller.

    The import is lazy, so the configuration can be inspected without Torch.
    """
    from ..controllers import pid

    return pid, Path(pid.__file__).resolve()


def model_geometry(model):
    """Return (body_id, mass, body-frame inertia, 6x4 COM allocation)."""
    import mujoco
    import numpy as np
    if (model.nq, model.nv, model.nu) != (7, 6, 4):
        raise ValueError('The pilot requires one free rigid body with four site motors')
    body = model.body('x2').id
    if (model.njnt != 1 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
            or model.jnt_bodyid[0] != body or model.jnt_qposadr[0] != 0 or model.jnt_dofadr[0] != 0):
        raise ValueError('Require the X2 root free joint at qpos/dof address zero')
    if not np.all(model.actuator_ctrllimited):
        raise ValueError('Require explicit native motor control limits')
    mass = float(model.body_subtreemass[body])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
    rotation = rotation.reshape(3, 3)
    inertia = rotation@np.diag(model.body_inertia[body])@rotation.T
    allocation = np.zeros((6, 4))
    for i in range(4):
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_SITE:
            raise ValueError('Require site-transmission motors, not aggregate torque controls')
        site = int(model.actuator_trnid[i, 0])
        if model.site_bodyid[site] != body:
            raise ValueError('All rotor sites must belong to the X2 rigid body')
        site_rotation = np.empty(9)
        mujoco.mju_quat2Mat(site_rotation, model.site_quat[site])
        site_rotation = site_rotation.reshape(3, 3)
        force = site_rotation@model.actuator_gear[i, :3]
        reaction_torque = site_rotation@model.actuator_gear[i, 3:]
        arm = model.site_pos[site]-model.body_ipos[body]
        allocation[:3, i] = force
        allocation[3:, i] = np.cross(arm, force)+reaction_torque
        if model.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_NONE:
            raise ValueError('Unexpected hidden actuator dynamics; nominal motor replay would be invalid')
        if not np.allclose(model.actuator_gainprm[i, :3], [1., 0., 0.]) or np.any(model.actuator_biasprm[i]):
            raise ValueError('Expected direct unit-gain motor controls in newtons')
    if not np.allclose(allocation[:3], [[0.]*4, [0.]*4, [1.]*4], atol=1e-10):
        raise ValueError('Vertical estimator assumes all motors push along body +z')
    return body, mass, inertia, allocation


class EulerMotorReplay:
    """Command-only, known first-order Euler model, separate from plant state."""
    def __init__(self, initial, tau_s, maximum_n):
        self.state = tuple(float(v) for v in initial)
        self.tau_s = float(tau_s)
        self.maximum_n = float(maximum_n)
        if len(self.state) != 4 or self.tau_s < 0 or self.maximum_n <= 0:
            raise ValueError('Invalid nominal motor replay settings')

    def step(self, command, dt_s):
        commands = tuple(float(v) for v in command)
        if len(commands) != 4 or not all(math.isfinite(v) for v in commands) or dt_s <= 0:
            raise ValueError('Require four finite motor commands and positive dt')
        alpha = 1. if self.tau_s == 0 else min(1., dt_s/self.tau_s)
        self.state = tuple(min(self.maximum_n, max(0., state+alpha*(min(self.maximum_n, max(0., target))-state)))
                           for state, target in zip(self.state, commands))
        return self.state


def apply_outputs(model, data, body, realized_thrusts, force_world):
    """Set site motor outputs and a WORLD-frame force at COM, clearing old force."""
    import numpy as np
    thrust = np.asarray(realized_thrusts, dtype=float)
    force = np.asarray(force_world, dtype=float)
    if thrust.shape != (4,) or force.shape != (3,) or not np.isfinite(thrust).all() or not np.isfinite(force).all():
        raise ValueError('Require finite four-rotor thrust and three-axis external force')
    data.ctrl[:] = thrust  # site lever arm and reaction yaw gear are handled by MuJoCo exactly once
    data.xfrc_applied[:] = 0.
    data.xfrc_applied[body, :3] = force  # MuJoCo xfrc_applied is world force/torque at COM


def audit_actuation(model, data, body, allocation, realized_thrusts):
    """Check actual generalized actuator force against the expected site wrench.

    Free-joint translation uses world coordinates; its rotational generalized
    force is body-frame torque about the body origin, while allocation is about
    COM. Add the COM offset contribution for this coordinate comparison.
    """
    import mujoco
    import numpy as np
    mujoco.mj_fwdActuation(model, data)
    body_wrench = allocation@np.asarray(realized_thrusts, dtype=float)
    rotation = data.xmat[body].reshape(3, 3)
    expected = np.concatenate((rotation@body_wrench[:3],
                               body_wrench[3:]+np.cross(model.body_ipos[body], body_wrench[:3])))
    error = float(np.max(np.abs(data.qfrc_actuator-expected)))
    if not np.allclose(data.qfrc_actuator, expected, atol=2e-8, rtol=2e-8):
        raise RuntimeError(f'MuJoCo actuator wrench disagrees with site allocation (max error {error:g})')
    return body_wrench, error


def _model_provenance(scene):
    files = {}
    for path in sorted(scene.parent.rglob('*')):
        if path.is_file() and '.git' not in path.parts:
            files[str(path.relative_to(scene.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    revision = None  # Bundled asset hashes identify this copy; do not search parent repositories.
    return {'scene_path': str(scene), 'file_sha256': files, 'menagerie_revision': revision}


def _viewer_context(viewer, model, data):
    if viewer is None or viewer is False:
        return nullcontext(None)
    if viewer is True:
        import mujoco.viewer
        return mujoco.viewer.launch_passive(model, data)
    if callable(viewer):
        return viewer(model, data)
    return nullcontext(viewer)


