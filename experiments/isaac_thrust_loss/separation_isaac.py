"""Source-guarded Isaac backend for the motor/vertical-force pilot.

This module imports no simulator or Torch at module load. It is not a validated
Isaac flight implementation until run on the group's supported NVIDIA stack.
The force hook adds world-frame force at COM to the existing body-wrench buffers
BEFORE their single PhysX application; there is no competing second force call.

Source inspected during implementation (not a runtime version guarantee):
https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_contrib/isaaclab_contrib/assets/multirotor/multirotor.py
https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_contrib/isaaclab_contrib/actuators/thruster.py
"""
from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

LOCAL_SOURCE = Path(__file__).resolve().parent/'src'
if str(LOCAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(LOCAL_SOURCE))

from drone_adaptation.separation.config import PilotConfig


def _tensor(value):
    import torch
    if torch.is_tensor(value):
        return value
    if __package__:
        from . import common
    else:
        import common
    return common._t(value)


def _record(method):
    source = inspect.getsource(method)
    return {'qualified_name': method.__qualname__, 'source_file': inspect.getsourcefile(method),
            'source_sha256': hashlib.sha256(source.encode()).hexdigest()}


def validate_pipeline(robot):
    """Require the inspected allocation and COM-force path before installing hooks."""
    requirements = {
        '_apply_actuator_model': ('.compute(', '.thrusts', '_thrust_target_sim'),
        '_combine_thrusts': ('allocation_matrix', '_thrust_target_sim',
                            '_internal_force_target_sim[:,0,:]', '_internal_torque_target_sim[:,0,:]'),
        '_apply_combined_wrench': ('_combine_thrusts', 'apply_forces_and_torques_at_position', 'is_global=False'),
        'write_data_to_sim': ('_apply_actuator_model', '_apply_combined_wrench'),
    }
    records = {}
    for name, tokens in requirements.items():
        method = getattr(robot, name, None)
        if not callable(method):
            raise RuntimeError(f'Unsupported multirotor API: missing {name}')
        try:
            source = ''.join(inspect.getsource(method).split())
        except (TypeError, OSError) as exc:
            raise RuntimeError(f'Cannot inspect installed {name}; review the pipeline first') from exc
        if not all(token in source for token in tokens):
            raise RuntimeError(f'Installed {name} differs from the inspected allocation pipeline')
        if name == '_apply_combined_wrench' and not any(
                token in source for token in ('position_data=None', 'position=None')):
            raise RuntimeError('Cannot establish force application at the body center of mass')
        records[name] = _record(method)
    for name in ('_thrust_target_sim', '_internal_wrench_target_sim',
                 '_internal_force_target_sim', '_internal_torque_target_sim'):
        if not hasattr(robot, name):
            raise RuntimeError(f'Missing {name}; cannot check realized force and torque')
    return records


class ForceAtCOMAdapter:
    """Compose a world-frame force into the existing body-frame allocation path.

    The installed `_combine_thrusts` first refreshes all nominal buffers. This
    wrapper then adds R^T F to the base link's force (and zero extra torque).
    Each call starts from the freshly allocated rotor wrench, so force neither
    accumulates nor persists when the next interval requests zero.
    """
    _owner = '_capstone_force_at_com_adapter'

    def __init__(self, robot):
        self.robot = robot
        self.original = robot._combine_thrusts
        self.had_instance = '_combine_thrusts' in robot.__dict__
        self.original_instance = robot.__dict__.get('_combine_thrusts')
        self.force_body = None
        self.force_world = (0.0, 0.0, 0.0)
        self.calls = 0
        self.installed = False
        self.wrapped = None

    def set_force(self, force_world, quat_xyzw):
        import torch
        force = torch.as_tensor(force_world, dtype=quat_xyzw.dtype, device=quat_xyzw.device).reshape(1, 3)
        q = quat_xyzw.reshape(1, 4)
        if not torch.isfinite(force).all() or not torch.isfinite(q).all() or float(q.norm()) <= 0:
            raise ValueError('Finite force and a nonzero finite xyzw quaternion required')
        q = q / q.norm(dim=1, keepdim=True)
        xyz, w = q[:, :3], q[:, 3:]
        # R^T * v = v - 2 w (q_xyz x v) + 2 q_xyz x (q_xyz x v).
        cross = torch.cross(xyz, force, dim=1)
        self.force_body = force - 2*w*cross + 2*torch.cross(xyz, cross, dim=1)
        self.force_world = tuple(float(v) for v in force[0].tolist())

    def install(self):
        if self.installed:
            if getattr(self.robot, '_combine_thrusts') is not self.wrapped:
                raise RuntimeError('Force hook replaced by another writer')
            return self
        if getattr(self.robot, self._owner, None) is not None:
            raise RuntimeError('Only one force-at-COM adapter may be installed')
        def combine():
            if not self.installed or getattr(self.robot, '_combine_thrusts') is not self.wrapped:
                raise RuntimeError('Force adapter is no longer the active allocation hook')
            result = self.original()
            if self.force_body is None:
                raise RuntimeError('Set the world force and orientation before allocating')
            wrench = _tensor(self.robot._internal_wrench_target_sim)
            force = _tensor(self.robot._internal_force_target_sim)
            torque = _tensor(self.robot._internal_torque_target_sim)
            if tuple(wrench.shape) != (1, 6) or force.ndim != 3 or tuple(force.shape[:1]) != (1,) or force.shape[2] != 3:
                raise RuntimeError('Unsupported force allocation buffer shape')
            if torque.shape != force.shape:
                raise RuntimeError('Force/torque allocation shapes differ')
            # The installed SDK may expose aliases/views of one wrench buffer.
            # Compute once from a clone, then assign both targets: never add twice.
            combined_force = wrench[:, :3].clone() + self.force_body
            wrench[:, :3] = combined_force
            force[:, 0, :] = combined_force
            self.calls += 1
            return result
        self.wrapped = combine
        setattr(self.robot, self._owner, self)
        self.robot._combine_thrusts = combine
        self.installed = True
        return self

    def close(self):
        if not self.installed:
            return
        if getattr(self.robot, '_combine_thrusts') is not self.wrapped:
            raise RuntimeError('Refusing to overwrite an intervening force hook')
        if self.had_instance:
            self.robot._combine_thrusts = self.original_instance
        else:
            del self.robot._combine_thrusts
        delattr(self.robot, self._owner)
        self.installed = False


def check_buffers(robot, allocation, motor_adapter, force_adapter):
    """Verify the rotor output and total body-wrench buffers before physics."""
    import torch
    actual = motor_adapter.realized_thrust
    if actual is None or not torch.allclose(_tensor(robot._thrust_target_sim), actual, atol=1e-6, rtol=1e-6):
        raise RuntimeError('Post-lag motor output did not reach allocation')
    expected = (allocation @ actual[0]).reshape(1, 6)
    expected[:, :3] += force_adapter.force_body
    for observed, desired in (
        (_tensor(robot._internal_wrench_target_sim), expected),
        (_tensor(robot._internal_force_target_sim)[:, 0, :], expected[:, :3]),
        (_tensor(robot._internal_torque_target_sim)[:, 0, :], expected[:, 3:]),
    ):
        if not torch.allclose(observed, desired, atol=2e-5, rtol=1e-5):
            raise RuntimeError('Realized rotor force/torque plus external force disagree with application buffers')
    if bool((_tensor(robot._internal_force_target_sim)[:, 1:, :] != 0).any()) or bool(
            (_tensor(robot._internal_torque_target_sim)[:, 1:, :] != 0).any()):
        raise RuntimeError('Unexpected force/torque on bodies other than the guarded base link')
    return expected


def _json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'torch'):
        return value.torch.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _versions():
    result = {}
    for name in ('isaaclab', 'isaacsim', 'torch', 'numpy', 'warp-lang'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _revision(file):
    if file is None:
        return None
    try:
        return subprocess.check_output(['git', '-C', str(Path(file).parent), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _feasibility(allocation, mass, config, native_min):
    import torch
    matrix = allocation.detach().double().cpu().clone()
    matrix[:, config.motor] *= 1.0-config.motor_loss
    # Check the most demanding planned motor+down-force combination for ALL arms.
    target = torch.tensor([0., 0., mass*9.81-min(0., config.force_z_n), 0., 0., 0.], dtype=torch.float64)
    command = torch.linalg.lstsq(matrix, target).solution
    residual = float((matrix@command-target).norm())
    feasible = residual <= 1e-5*float(target.norm()) and bool(
        (command >= native_min-1e-6).all() and (command <= config.max_thrust_n+1e-6).all())
    record = {'feasible': feasible, 'required_nominal_thrust_n': command.tolist(),
              'residual_norm': residual, 'assumption': 'Static level hover, zero yaw torque, worst planned downward force; necessary not sufficient'}
    if not feasible:
        raise ValueError(f'Outside checked hover authority: {record}')
    return record


def _state(robot, t_s, noise):
    row = {'t_s': t_s}
    fields = (('pos', 'root_pos_w', 'xyz'), ('lin_vel', 'root_lin_vel_w', 'xyz'),
              ('ang_vel', 'root_ang_vel_b', 'xyz'), ('quat', 'root_quat_w', 'xyzw'))
    for prefix, field, labels in fields:
        for label, value in zip(labels, _tensor(getattr(robot.data, field))[0].tolist()):
            row[f'{prefix}_{label}'] = float(value)
    if not all(math.isfinite(value) for value in row.values()):
        raise RuntimeError('Nonfinite physical state')
    qx, qy, qz, qw = [row['quat_'+a] for a in 'xyzw']
    norm2 = qx*qx+qy*qy+qz*qz+qw*qw
    tilt = math.degrees(math.acos(max(-1., min(1., 1-2*(qx*qx+qy*qy)/norm2))))
    row.update(z_m=row['pos_z'], vz_m_s=row['lin_vel_z'],
               measured_vz_m_s=row['lin_vel_z']+noise(), tilt_deg=tilt,
               body_z_world_z=1-2*(qx*qx+qy*qy)/norm2)
    return row


def _write(output, rows, manifest):
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (output/'trajectory.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')


def run(config: PilotConfig, launcher_args, physics_cfg, output: Path, manifest: dict):
    """Run one full-PID episode; save inside launch_simulation's context.

    A separate RPM-state predictor starts from known healthy calibration and
    thereafter sees issued commands only. Every predicted nominal output must
    match the actuator's own nominal output before physics; if they disagree, the
    run stops. Only the independent prediction enters the next estimation observation.
    The actuator's nominal and realized outputs are used only for this check, never
    as estimator inputs.
    """
    import numpy as np
    import torch
    if __package__:
        from . import common
    else:
        import common
    if __package__:
        from .support.fault_actuator import MotorFaultAdapter
        from .support.faults import StepMotorFault
    else:
        from support.fault_actuator import MotorFaultAdapter
        from support.faults import StepMotorFault
    from drone_adaptation.separation.estimator import MotorForceEstimator, NominalRPMActuatorModel
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    jitter = random.Random(config.seed+2_000_003).uniform(-config.initial_z_jitter_m, config.initial_z_jitter_m)
    noise_rng = random.Random(config.seed+1_000_003)
    noise = lambda: noise_rng.gauss(0., config.velocity_noise_std_m_s)
    args = SimpleNamespace(**vars(launcher_args))
    args.start = [0., 0., config.target_z_m+jitter]
    args.goal = [0., 0., config.target_z_m]
    args.controller, args.fault, args.speed = 'pid', 'none', 0.5
    rows, motor_adapter, force_adapter = [], None, None
    termination = 'api_error'
    failure = None
    manifest['estimator_status'] = 'unassessed_setup'
    replay = {'verified_intervals': 0, 'maximum_absolute_error_n': 0.0,
              'absolute_tolerance_n': 1e-4, 'relative_tolerance': 1e-4,
              'status': 'not_started'}
    manifest['nominal_replay_audit'] = replay
    try:
        world = common.make_world(args, physics_cfg)
        robot, sim, controller = world.robot, world.sim, world.controller
        pipeline = validate_pipeline(robot)
        names = list(robot.data.thruster_names)
        actuator = robot.actuators['thrusters']
        if len(names) != 4 or list(actuator.thruster_names) != names:
            raise RuntimeError('Require one ordered group of exactly four thrusters')
        allocation = _tensor(robot.allocation_matrix)
        if tuple(allocation.shape) != (6, 4) or not torch.isfinite(allocation).all():
            raise RuntimeError('Require a finite 6x4 allocation matrix')
        if not torch.allclose(allocation@world.alloc_pinv@allocation, allocation, atol=1e-5, rtol=1e-5):
            raise RuntimeError('Controller and plant allocation disagree')
        if not torch.allclose(allocation[:3], torch.tensor([[0.]*4, [0.]*4, [1.]*4], device=allocation.device), atol=1e-6):
            raise RuntimeError('Vertical estimator pilot assumes all four rotors push along body +z')
        mass = float(controller.mass)
        native_min, native_max = float(actuator.min_thrust), float(actuator.max_thrust)
        if config.max_thrust_n > native_max:
            raise ValueError('Requested command cap exceeds native actuator maximum')
        authority = _feasibility(allocation, mass, config, native_min)
        actuator_values = {name: _json_value(_tensor(getattr(actuator, name)).detach().cpu())
                           for name in ('tau_inc_s', 'tau_dec_s', 'thrust_const', 'curr_thrust', 'max_rate')}
        actuator_values.update(native_min_thrust_n=native_min, native_max_thrust_n=native_max,
                               integration_scheme=actuator.cfg.integration_scheme,
                               use_discrete_approximation=actuator.cfg.use_discrete_approximation,
                               dt_s=actuator.cfg.dt)
        def calibrated_four(name):
            return tuple(float(v) for v in _tensor(getattr(actuator, name))[0].tolist())
        if not math.isclose(float(actuator.cfg.dt), config.dt_s, abs_tol=1e-10):
            raise RuntimeError('Actuator and physics intervals disagree')
        nominal_model = NominalRPMActuatorModel(
            tau_inc_s=calibrated_four('tau_inc_s'), tau_dec_s=calibrated_four('tau_dec_s'),
            thrust_const=calibrated_four('thrust_const'), max_rate=calibrated_four('max_rate'),
            min_thrust_n=native_min, max_thrust_n=native_max,
            initial_thrust_n=calibrated_four('curr_thrust'),
            integration_scheme=actuator.cfg.integration_scheme,
            use_discrete_approximation=actuator.cfg.use_discrete_approximation, dt_s=config.dt_s)
        estimator = MotorForceEstimator(mass_kg=mass, motor=config.motor, history_s=config.history_s)
        previous_nominal_vertical = None
        replay['status'] = 'checking_each_interval'
        eta = 1.-config.motor_loss if config.condition in ('motor', 'combined') else 1.
        motor_adapter = MotorFaultAdapter(actuator, StepMotorFault(config.motor, eta, config.motor_onset_s,
                                                                  config.motor_duration_s)).install()
        force_adapter = ForceAtCOMAdapter(robot).install()
        manifest.update(status='running', runtime_status='running',
                        plant={'mass_kg': mass, 'inertia_body_kg_m2': _json_value(controller.inertia.detach().cpu()),
                               'allocation_matrix': allocation.cpu().tolist(), 'motor_names': names,
                               'actuator_parameters': actuator_values, 'hover_authority': authority},
                        controller={'name': 'CascadedPID', 'gains': asdict(controller.gains),
                                    'state_input': 'ideal simulator pose and velocities; estimator-only synthetic vertical velocity noise'},
                        pipeline_sources=pipeline, actuator_compute_source=motor_adapter.source_info,
                        versions=_versions(), isaac_revision=_revision(motor_adapter.source_info.get('source_file')),
                        estimator_status='shadow_estimates_from_independent_command_replay',
                        estimator_calibration={'privilege': 'Known simulator mass, selected candidate motor, nominal actuator coefficients and healthy initial actuator state at setup',
                            'runtime_inputs': 'Issued bounded commands, previous attitude projection, measured vertical velocity; no audit thrust, fault or disturbance truth',
                            'restriction': 'Only selected motor may change; unknown force acts along world z; other motors nominal. Diagnostic estimates do not control PID.',
                            'future_work': 'Replace white-box coefficients/initial state with independent healthy calibration and validate resulting model error'},
                        initial_z_jitter_m=jitter, failure_thresholds={'min_height_m': 0.3, 'max_tilt_deg': 60., 'startup_grace_s': 0.5})
        _write(output, rows, manifest)
        count = int(round(config.duration_s/config.dt_s))
        for step in range(count+1):
            if not sim.is_running():
                termination = 'user_stopped'
                break
            t_s = step*config.dt_s
            row = _state(robot, t_s, noise)
            reference, ref_velocity = config.reference(t_s)
            # State at t observes only commands from the PREVIOUS interval.
            # No current action, eta label, force label or actuator audit enters this call.
            estimates = estimator.observe(t_s, row['measured_vz_m_s'], previous_nominal_vertical)
            row.update(estimates)
            row.update(target_z_m=reference, eta_true=config.effectiveness(t_s)[config.motor],
                       force_true_z_n=config.force(t_s)[2], interval_end_s=None,
                       estimator_status='command_replay_shadow_estimate')
            if t_s > 0.5 and row['z_m'] < 0.3:
                failure, termination = 'height_limit', 'failure'
            elif row['tilt_deg'] > 60.:
                failure, termination = 'tilt_limit', 'failure'
            if failure or step == count:
                for prefix in ('thrust_requested', 'thrust_bounded', 'thrust_predicted_nominal', 'thrust_nominal_audit', 'thrust_realized_audit'):
                    row.update({f'{prefix}_{i}': None for i in range(4)})
                row['failed'] = bool(failure)
                rows.append(row)
                if not failure:
                    termination = 'completed'
                break
            pos = _tensor(robot.data.root_pos_w)
            vel = _tensor(robot.data.root_lin_vel_w)
            quat = _tensor(robot.data.root_quat_w)
            ang = _tensor(robot.data.root_ang_vel_b)
            goal = torch.tensor([[0., 0., reference]], device=pos.device, dtype=pos.dtype)
            goal_vel = torch.tensor([[0., 0., ref_velocity]], device=vel.device, dtype=vel.dtype)
            requested = controller.compute(pos, vel, quat, ang, goal, goal_vel)@world.alloc_pinv.T
            if not torch.isfinite(requested).all():
                raise RuntimeError('Nonfinite requested motor command')
            bounded = requested.clamp(min=0., max=config.max_thrust_n)
            motor_adapter.set_time(t_s)
            force_adapter.set_force(config.force(t_s), quat)
            previous_motor_calls, previous_force_calls = motor_adapter.calls, force_adapter.calls
            robot.set_thrust_target(bounded)
            robot.write_data_to_sim()
            if motor_adapter.calls != previous_motor_calls+1 or force_adapter.calls != previous_force_calls+1:
                raise RuntimeError('Expected one actuator and one force-composition call per physics interval')
            if not torch.allclose(motor_adapter.last_requested, bounded, atol=1e-6, rtol=1e-6):
                raise RuntimeError('Controller command changed before the nominal actuator')
            # Predict from issued commands using an independent state. The actual
            # nominal output is comparison-only and cannot correct predictor state.
            predicted = nominal_model.step(tuple(float(v) for v in bounded[0].tolist()), config.dt_s)
            predicted_tensor = torch.tensor([predicted], dtype=motor_adapter.nominal_thrust.dtype,
                                            device=motor_adapter.nominal_thrust.device)
            error_n = float((predicted_tensor-motor_adapter.nominal_thrust).abs().max())
            replay['maximum_absolute_error_n'] = max(replay['maximum_absolute_error_n'], error_n)
            if not torch.allclose(predicted_tensor, motor_adapter.nominal_thrust,
                                  atol=replay['absolute_tolerance_n'], rtol=replay['relative_tolerance']):
                replay['status'] = 'mismatch'
                raise RuntimeError(f'Independent command-only nominal RPM replay mismatch ({error_n:.6g} N); no diagnostic results may be claimed')
            replay['verified_intervals'] += 1
            previous_nominal_vertical = tuple(value*row['body_z_world_z'] for value in predicted)
            total = check_buffers(robot, allocation, motor_adapter, force_adapter)
            row['interval_end_s'] = t_s+config.dt_s
            row['failed'] = False
            for prefix, values in (('thrust_requested', requested), ('thrust_bounded', bounded), ('thrust_predicted_nominal', predicted_tensor),
                                   ('thrust_nominal_audit', motor_adapter.nominal_thrust),
                                   ('thrust_realized_audit', motor_adapter.realized_thrust)):
                row.update({f'{prefix}_{i}': float(v) for i, v in enumerate(values[0].tolist())})
            row.update({f'total_body_wrench_{axis}': float(value) for axis, value in zip(('fx','fy','fz','tx','ty','tz'), total[0].tolist())})
            row.update({f'external_force_body_{axis}': float(value) for axis, value in zip('xyz', force_adapter.force_body[0].tolist())})
            rows.append(row)
            sim.step()
            robot.update(config.dt_s)
            world.imu.update(config.dt_s)
        manifest['status'] = 'complete' if termination in ('completed', 'failure') else 'incomplete'
        replay['status'] = 'verified_recorded_intervals' if manifest['status'] == 'complete' else 'verified_partial_intervals'
    except Exception as exc:
        manifest.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        print(f'[ERROR] {manifest["error"]}', file=sys.stderr, flush=True)
    finally:
        manifest.update(runtime_status=manifest.get('status', 'incomplete'), termination_reason=termination,
                        failure_reason=failure, failed=bool(failure), row_count=len(rows))
        for adapter in (force_adapter, motor_adapter):
            if adapter is not None:
                try:
                    adapter.close()
                except Exception as exc:
                    manifest['status'] = manifest['runtime_status'] = 'incomplete'
                    manifest['cleanup_error'] = f'{type(exc).__name__}: {exc}'
        _write(output, rows, manifest)
    return manifest['status'] == 'complete'
