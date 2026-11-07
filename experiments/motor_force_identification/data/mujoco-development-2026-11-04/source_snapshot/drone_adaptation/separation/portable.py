"""A vertical point-mass diagnostic with four nominal rotor command channels.

Attitude is perfectly stabilized. This model cannot validate an asymmetric rotor
fault's torque, six-degree-of-freedom recovery, Isaac integration, or sim-to-real.
The existing altitude PID flies the plant; both estimators run in shadow and do
not alter its commands. Model coefficients are known deliberately for this
identification check, not inferred from fault labels.
"""
from __future__ import annotations

import random
from ..core import Observation, PIDController
from .config import PilotConfig
from .estimator import MotorForceEstimator, NominalActuatorModel

MASS_KG = 1.24
GRAVITY_M_S2 = 9.81
TAU_S = 0.05


def simulate(config: PilotConfig):
    """Return (trajectory, runtime_metadata, status) without any filesystem I/O."""
    hover = MASS_KG * GRAVITY_M_S2 / 4
    if (4-config.motor_loss) * config.max_thrust_n <= MASS_KG*GRAVITY_M_S2 + abs(config.force_z_n):
        raise ValueError('Requested portable conditions leave insufficient vertical thrust reserve')
    actuator_params = dict(tau_inc_s=TAU_S, tau_dec_s=TAU_S, min_thrust_n=0.0,
                           max_thrust_n=config.max_thrust_n, initial_thrust_n=(hover,)*4)
    # The two objects share only known calibration coefficients. Plant output is
    # never supplied to the estimator: its nominal input is replayed from issued commands.
    plant_actuator = NominalActuatorModel(**actuator_params)
    command_model = NominalActuatorModel(**actuator_params)
    estimator = MotorForceEstimator(MASS_KG, motor=config.motor, history_s=config.history_s)
    pid = PIDController(target_m=config.target_z_m, mass_kg=MASS_KG, gravity_m_s2=GRAVITY_M_S2,
                        max_thrust_n=4*config.max_thrust_n)
    initial_rng = random.Random(config.seed + 31_415_927)
    noise_rng = random.Random(config.seed + 2_718_281)
    z = config.target_z_m + initial_rng.uniform(-config.initial_z_jitter_m, config.initial_z_jitter_m)
    vz = 0.0
    previous_model_thrust = None
    rows = []
    failed = False
    for step in range(round(config.duration_s/config.dt_s) + 1):
        t = step*config.dt_s
        measured_vz = vz + noise_rng.gauss(0, config.velocity_noise_std_m_s)
        estimates = estimator.observe(t, measured_vz, previous_model_thrust)
        target, target_velocity = config.reference(t)
        eta = config.effectiveness(t)
        force = config.force(t)[2]
        failed = z <= 0.3 or z >= 4.0
        terminal = failed or step == round(config.duration_s/config.dt_s)
        row = dict(t_s=t, z_m=z, vz_m_s=vz, measured_vz_m_s=measured_vz,
                   target_z_m=target, target_vz_m_s=target_velocity,
                   eta_true=eta[config.motor], force_true_z_n=force,
                   failed=failed, interval_end_s=None if terminal else (step+1)*config.dt_s,
                   **estimates)
        for prefix in ('thrust_requested', 'thrust_bounded', 'thrust_model', 'thrust_nominal', 'thrust_realized'):
            row.update({f'{prefix}_{i}': None for i in range(4)})
        if terminal:
            rows.append(row)
            break
        pid.target_m = target
        # Reference velocity is public task information, not hidden dynamics.
        total = pid.act(Observation(z, vz-target_velocity), config.dt_s)
        raw_total = MASS_KG*(GRAVITY_M_S2 + pid.kp*(target-z) - pid.kd*(vz-target_velocity)
                            + pid.ki*pid.integral_state)
        commands = (total/4,)*4
        model_nominal = command_model.step(commands, config.dt_s)
        plant_nominal = plant_actuator.step(commands, config.dt_s)
        realized = tuple(value*factor for value,factor in zip(plant_nominal, eta))
        for prefix, values in (('thrust_requested',(raw_total/4,)*4), ('thrust_bounded',commands),
                               ('thrust_model',model_nominal), ('thrust_nominal',plant_nominal),
                               ('thrust_realized',realized)):
            row.update({f'{prefix}_{i}': value for i,value in enumerate(values)})
        rows.append(row)
        acceleration = (sum(realized)+force)/MASS_KG-GRAVITY_M_S2
        z += vz*config.dt_s + .5*acceleration*config.dt_s**2
        vz += acceleration*config.dt_s
        previous_model_thrust = model_nominal  # perfect attitude -> world vertical projection is 1
    metadata = dict(backend='portable_vertical_model', mass_kg=MASS_KG, gravity_m_s2=GRAVITY_M_S2,
                    controller='Existing altitude PID; shadow estimates do not affect control',
                    controller_gains=dict(kp=pid.kp,kd=pid.kd,ki=pid.ki),
                    nominal_actuator_model='Known exponential first-order thrust lag; NOT the Isaac RPM actuator',
                    actuator_parameters=actuator_params,
                    assumptions=['Known candidate motor index, mass and nominal dynamics',
                                 'Perfectly stabilized attitude; identical nominal rotor allocation',
                                 'World-vertical external force only; no external moments',
                                 'Velocity from ideal state plus seeded Gaussian noise',
                                 'No learned policy, detection gate, controller adaptation, or physical flight validation'],
                    controller_observations='Ideal state, matching the current Isaac PID; synthetic velocity noise affects only shadow inference',
                    paired_nuisance_seed=config.seed,
                    estimator_inputs=['time','measured vertical velocity','previous command-model nominal vertical thrust'],
                    truth_fields_are_diagnostics_only=True,
                    status='failed' if failed else 'complete')
    return rows, metadata, metadata['status']
