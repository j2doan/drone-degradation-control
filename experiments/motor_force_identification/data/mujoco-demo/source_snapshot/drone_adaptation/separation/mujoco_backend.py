"""MuJoCo Skydio X2 backend for the four-condition identification pilot.

The model, allocation conventions and actuator lag follow the team's MuJoCo X2 model;
the controller is loaded from the current shared pipeline without importing its
Isaac runtime. This is a separate hover/probe protocol: known healthy hover
initialization, a 6 N command cap, and loss applied AFTER nominal motor dynamics.
Neither shadow estimator controls the drone. The repository checkout and an
external MuJoCo Menagerie Skydio X2 asset are required.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from .config import PilotConfig
from .estimator import MotorForceEstimator

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project root
REPOSITORY_ROOT = PROJECT_ROOT.parent
MOTOR_TAU_S = 0.125
MIN_HEIGHT_M = 0.05
MAX_TILT_DEG = 60.0
_CONTROLLER_MODULE = '_capstone_current_pid'


def resolve_scene(scene_path: Path | None = None) -> Path:
    """Resolve explicit scene, MENAGERIE_DIR, or the shared ignored asset root."""
    if scene_path is None:
        menagerie = Path(os.environ.get('MENAGERIE_DIR', PROJECT_ROOT/'assets/mujoco_menagerie')).expanduser()
        scene_path = menagerie/'skydio_x2/scene.xml'
    scene = Path(scene_path).expanduser().resolve()
    if not scene.is_file():
        raise FileNotFoundError(
            f'Skydio X2 scene not found: {scene}. Set MENAGERIE_DIR to your MuJoCo Menagerie checkout '
            'or follow the asset setup instructions in the README.')
    return scene


def load_controller():
    """Import the canonical simulator-free PID with an isolated module name.

    Older tests import modules named controllers/faults/analysis. Loading
    the exact shared file avoids silently selecting those older modules from
    sys.modules or changing global sys.path search order.
    """
    path = REPOSITORY_ROOT/'drone/controllers/pid.py'
    if not path.is_file():
        raise FileNotFoundError(f'Current team PID not found at {path}; use the full source checkout')
    module = sys.modules.get(_CONTROLLER_MODULE)
    if module is None or Path(module.__file__).resolve() != path.resolve():
        spec = importlib.util.spec_from_file_location(_CONTROLLER_MODULE, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_CONTROLLER_MODULE] = module  # dataclasses needs its defining module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_CONTROLLER_MODULE, None)
            raise
    return module, path


def model_geometry(model):
    """Return (body_id, mass, body-frame inertia, 6x4 COM allocation)."""
    import mujoco
    import numpy as np
    if (model.nq, model.nv, model.nu) != (7, 6, 4):
        raise ValueError('The pilot requires one free rigid body with four site motors')
    body = model.body('x2').id
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
    try:
        revision = subprocess.check_output(['git','-C',str(scene.parent),'rev-parse','HEAD'],
                                           text=True,stderr=subprocess.DEVNULL,timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
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


def simulate(config: PilotConfig, *, scene_path: Path | None = None, viewer=None):
    """Run one real MuJoCo trajectory and return (rows, metadata, status).

    ``viewer=True`` opens a passive viewer (macOS requires mjpython). A viewer
    factory or handle is also accepted; None is genuinely headless. No rendering
    backend or GPU is initialized for headless simulation.
    """
    try:
        import mujoco
        import numpy as np
        import torch
    except ImportError as error:
        raise ImportError('MuJoCo pilot needs mujoco, numpy and torch; install requirements.txt in a virtual environment') from error
    scene = resolve_scene(scene_path)
    model = mujoco.MjModel.from_xml_path(str(scene))
    if not math.isclose(float(model.opt.timestep), config.dt_s, abs_tol=1e-12):
        raise ValueError('Model and pilot control/physics intervals must both be 0.01 s')
    if not np.allclose(model.opt.gravity, [0.,0.,-9.81]):
        raise ValueError('Pilot estimator assumes gravity [0,0,-9.81] m/s²')
    body, mass, inertia, allocation = model_geometry(model)
    native_min = model.actuator_ctrlrange[:, 0].copy()
    native_max = model.actuator_ctrlrange[:, 1].copy()
    if np.any(native_min > 0.) or np.any(native_max < config.max_thrust_n):
        raise ValueError('Command cap lies outside native X2 motor limits')
    hover = np.linalg.lstsq(allocation, [0.,0.,mass*9.81,0.,0.,0.], rcond=None)[0]
    if not np.allclose(allocation@hover, [0.,0.,mass*9.81,0.,0.,0.], atol=1e-8) or np.any(hover < 0.) or np.any(hover > config.max_thrust_n):
        raise ValueError('Healthy hover is outside the configured motor envelope')
    controller_module, controller_path = load_controller()
    controller = controller_module.CascadedPID(1, mass, inertia, config.dt_s)
    estimator = MotorForceEstimator(mass, motor=config.motor, history_s=config.history_s)
    data = mujoco.MjData(model)
    initial_rng = random.Random(config.seed+31_415_927)
    noise_rng = random.Random(config.seed+2_718_281)
    initial_jitter = initial_rng.uniform(-config.initial_z_jitter_m, config.initial_z_jitter_m)
    data.qpos[:3] = [0.,0.,config.target_z_m+initial_jitter]
    data.qpos[3:7] = [1.,0.,0.,0.]
    data.qvel[:] = 0.
    data.ctrl[:] = hover
    mujoco.mj_forward(model,data)
    plant_nominal = hover.copy()
    replay = EulerMotorReplay(hover, MOTOR_TAU_S, config.max_thrust_n)
    pinv = torch.tensor(np.linalg.pinv(allocation), dtype=torch.float32)
    as_tensor = lambda values: torch.tensor(np.asarray(values),dtype=torch.float32).unsqueeze(0)
    metadata = {'backend': 'mujoco_x2', 'simulator_version': mujoco.__version__,
        'versions': {'mujoco': mujoco.__version__, 'numpy': np.__version__, 'torch': torch.__version__},
        'physics': 'Full six-degree-of-freedom free rigid body, MuJoCo site motors, Skydio X2 Menagerie',
        'mass_kg': mass, 'inertia_body_kg_m2': inertia.tolist(), 'allocation_matrix': allocation.tolist(),
        'controller': 'Cascaded PID with motor-cap anti-windup; shadow estimates do not control it',
        'controller_path': str(controller_path), 'controller_sha256': hashlib.sha256(controller_path.read_bytes()).hexdigest(),
        'controller_gains': asdict(controller.gains),
        'actuator': {'model': 'First-order Euler thrust lag, calibrated known scalar constant', 'tau_s': MOTOR_TAU_S,
                     'command_cap_n': config.max_thrust_n, 'native_min_n': native_min.tolist(),
                     'native_max_n': native_max.tolist(), 'initial_nominal_thrust_n': hover.tolist(),
                     'order': 'requested -> bounded -> nominal lag/native clip -> effectiveness -> site force and reaction torque'},
        'initialization': 'Known healthy hover motor thrust, level attitude',
        'initial_z_jitter_m': initial_jitter, 'failure_thresholds': {'min_height_m': MIN_HEIGHT_M, 'max_tilt_deg': MAX_TILT_DEG},
        'nominal_replay_audit': {'maximum_absolute_error_n': 0., 'verified_intervals': 0, 'absolute_tolerance_n': 1e-10},
        'wrench_audit': {'maximum_generalized_force_error': 0., 'verified_intervals': 0},
        'model': _model_provenance(scene),
        'model_fluid_settings': {'density': float(model.opt.density), 'viscosity': float(model.opt.viscosity)},
        'estimator_inputs': ['time', 'noisy COM vertical velocity', 'previous command-only nominal replay projected by previous attitude'],
        'calibration_assumptions': ['Known candidate motor and mass', 'Known nominal lag and healthy initial motor state',
                                   'Only candidate motor degrades; world-vertical added force and no external moments',
                                   'Existing X2 aerodynamics/contact remain physics-model effects, not supplied to estimator'],
        'truth_fields_are_diagnostics_only': True, 'estimators_in_shadow': True,
        'velocity_convention': 'PID uses body-origin world linear velocity; estimator vz_m_s/measured_vz_m_s use COM velocity v_origin+R(omega_body x body_ipos). z_m tracks body-origin reference.',
        'protocol_difference': 'Hover/probe at 1m, 6N cap, post-output loss; do not pool with legacy straight-route target-loss sweeps',
        'status': 'running'}
    rows = []
    previous_model_vertical = None
    failed = False
    status = 'complete'
    termination = 'completed'
    row = None
    prefixes = ('thrust_requested', 'thrust_bounded', 'thrust_model', 'thrust_nominal_audit', 'thrust_realized_audit')
    try:
        with _viewer_context(viewer,model,data) as view:
            for step in range(round(config.duration_s/config.dt_s)+1):
                started = time.monotonic()
                t = step*config.dt_s
                # mj_step integrates qpos/qvel after its forward pass. Refresh
                # position-dependent fields at this new state before using xmat.
                mujoco.mj_forward(model,data)
                q = data.qpos.copy()
                v = data.qvel.copy()
                if not np.isfinite(q).all() or not np.isfinite(v).all():
                    raise RuntimeError('Nonfinite simulated state')
                rotation = data.xmat[body].reshape(3,3).copy()
                tilt = math.degrees(math.acos(max(-1.,min(1.,float(rotation[2,2])))))
                target,target_v = config.reference(t)
                com_velocity = v[:3]+rotation@np.cross(v[3:6],model.body_ipos[body])
                measured_vz = float(com_velocity[2])+noise_rng.gauss(0.,config.velocity_noise_std_m_s)
                estimates = estimator.observe(t,measured_vz,previous_model_vertical)
                failed = bool(q[2] <= MIN_HEIGHT_M or tilt >= MAX_TILT_DEG)
                terminal = failed or step == round(config.duration_s/config.dt_s)
                if view is not None and not view.is_running():
                    terminal,status,termination = True,'incomplete','viewer_closed'
                row = {'t_s': t, 'z_m':float(q[2]), 'vz_m_s':float(com_velocity[2]), 'measured_vz_m_s':measured_vz,
                       'body_origin_vz_m_s':float(v[2]), 'com_vz_m_s':float(com_velocity[2]),
                       'com_z_m':float(data.xipos[body,2]),
                       'target_z_m':target, 'target_vz_m_s':target_v,
                       'eta_true':config.effectiveness(t)[config.motor], 'force_true_z_n':config.force(t)[2],
                       'tilt_deg':tilt, 'body_z_world_z':float(rotation[2,2]), 'failed':failed,
                       'interval_end_s':None if terminal else (step+1)*config.dt_s, **estimates}
                row.update({f'pos_{axis}':float(value) for axis,value in zip('xyz',q[:3])})
                row.update({f'quat_{axis}':float(value) for axis,value in zip('xyzw',[q[4],q[5],q[6],q[3]])})
                row.update({f'lin_vel_{axis}':float(value) for axis,value in zip('xyz',v[:3])})
                row.update({f'ang_vel_{axis}':float(value) for axis,value in zip('xyz',v[3:6])})
                for prefix in prefixes:
                    row.update({f'{prefix}_{i}':None for i in range(4)})
                if terminal:
                    if failed:
                        status,termination='failed','height_limit' if q[2] <= MIN_HEIGHT_M else 'tilt_limit'
                    rows.append(row)
                    break
                quat_xyzw = [q[4],q[5],q[6],q[3]]
                desired = as_tensor([0.,0.,target])
                desired_velocity = as_tensor([0.,0.,target_v])
                wrench = controller.compute(as_tensor(q[:3]),as_tensor(v[:3]),as_tensor(quat_xyzw),
                                            as_tensor(v[3:6]),desired,desired_velocity)
                requested_tensor = wrench@pinv.T
                requested = requested_tensor[0].detach().numpy().astype(float)
                if not np.isfinite(requested).all():
                    raise RuntimeError('Nonfinite requested thrust')
                bounded = np.clip(requested,0.,config.max_thrust_n)
                controller.set_saturated([bool(np.any(np.abs(requested-bounded)>1e-9))])
                # Plant lag and command-only replay retain separate state.
                alpha=min(1.,config.dt_s/MOTOR_TAU_S)
                plant_nominal = np.clip(plant_nominal+alpha*(bounded-plant_nominal),native_min,native_max)
                prediction = replay.step(bounded,config.dt_s)
                replay_error=float(np.max(np.abs(np.asarray(prediction)-plant_nominal)))
                metadata['nominal_replay_audit']['maximum_absolute_error_n']=max(
                    metadata['nominal_replay_audit']['maximum_absolute_error_n'],replay_error)
                if replay_error > 1e-10:
                    raise RuntimeError('Independent command-only motor replay disagrees with nominal actuator')
                metadata['nominal_replay_audit']['verified_intervals']+=1
                realized=plant_nominal*np.asarray(config.effectiveness(t))
                force=config.force(t)
                apply_outputs(model,data,body,realized,force)
                actual_wrench,wrench_error=audit_actuation(model,data,body,allocation,realized)
                metadata['wrench_audit']['maximum_generalized_force_error']=max(
                    metadata['wrench_audit']['maximum_generalized_force_error'],wrench_error)
                metadata['wrench_audit']['verified_intervals']+=1
                for prefix,values in (('thrust_requested',requested),('thrust_bounded',bounded),
                                      ('thrust_model',prediction),('thrust_nominal_audit',plant_nominal),
                                      ('thrust_realized_audit',realized)):
                    row.update({f'{prefix}_{i}':float(value) for i,value in enumerate(values)})
                row.update({f'rotor_body_wrench_{axis}':float(value)
                            for axis,value in zip(('fx','fy','fz','tx','ty','tz'),actual_wrench)})
                row.update({f'external_force_world_{axis}':float(value) for axis,value in zip('xyz',force)})
                rows.append(row)
                previous_model_vertical=tuple(float(value)*float(rotation[2,2]) for value in prediction)
                mujoco.mj_step(model,data)
                if view is not None:
                    view.sync()
                    time.sleep(max(0.,config.dt_s-(time.monotonic()-started)))
    except Exception as error:
        status,termination='incomplete','backend_error'
        metadata['error']=f'{type(error).__name__}: {error}'
        if row is not None:
            row['interval_end_s']=None
            for prefix in prefixes:
                row.update({f'{prefix}_{i}':None for i in range(4)})
            if not rows or rows[-1] is not row:
                rows.append(row)
    metadata.update(status=status,failed=failed,termination_reason=termination,row_count=len(rows))
    return rows,metadata,status
