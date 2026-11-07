"""MuJoCo Skydio X2 backend for the four-condition identification pilot.

The model, allocation conventions and actuator lag follow the team's MuJoCo X2 model;
the controller is the bundled cascaded PID. No Isaac runtime is needed. This is a separate
hover/probe protocol: the drone starts in a healthy hover, motor commands are capped at 6 N,
and the loss is applied after the nominal motor dynamics.
Neither estimator controls the drone; both only watch the flight.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import random
import time

from .config import PilotConfig
from .estimator import MotorForceEstimator

from ..mujoco_pipeline.physics import (
    PROJECT_ROOT, MOTOR_TAU_S, MIN_HEIGHT_M, MAX_TILT_DEG,
    resolve_scene, load_controller, model_geometry, EulerMotorReplay,
    apply_outputs, audit_actuation, _model_provenance, _viewer_context,
)


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
