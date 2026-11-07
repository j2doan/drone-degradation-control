"""Run one source-guarded PID motor/vertical-force pilot on the NVIDIA machine.

Dependency-free check (no flight):
    python 04_separation_pilot.py --condition healthy --check-config

Flight, using the existing Isaac launcher:
    ./run.sh 04_separation_pilot.py --condition healthy --motion probe --headless --output ABSOLUTE_NEW_DIRECTORY

All four conditions use the same 20 s vertical hover/probe task. Shadow estimates
use an independent nominal RPM replay initialized from known healthy calibration;
each interval is checked against the actuator's nominal output before physics.
The actuator's nominal and realized outputs are used only for this check, never as
estimator inputs. This integration has been checked against the source and with
mocks, but not yet validated on an NVIDIA runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT/"src") not in sys.path:
    sys.path.insert(0, str(ROOT/"src"))
from drone_adaptation.separation.config import CONDITIONS, MOTIONS, PilotConfig


def _optional_duration(value):
    return None if value.lower() in ('none', 'sustained') else float(value)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, conflict_handler='resolve')
    parser.add_argument('--condition', choices=CONDITIONS, default='healthy')
    parser.add_argument('--motion', choices=MOTIONS, default='probe')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--motor', type=int, default=3)
    defaults = PilotConfig()
    for item in fields(PilotConfig):
        if item.name in ('condition', 'motion', 'seed', 'motor'):
            continue
        value = getattr(defaults, item.name)
        kind = _optional_duration if item.name in ('motor_duration_s', 'force_duration_s') else float
        parser.add_argument('--'+item.name.replace('_','-'), type=kind, default=value)
    parser.add_argument('--sustained-force', action='store_true', help='Keep the force active until episode end')
    parser.add_argument('--output', type=Path, help='New output directory; absolute paths survive the Isaac launcher cwd change')
    parser.add_argument('--check-config', action='store_true', help='Validate only; no optional dependencies or simulation')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--physics', default='isaacsim_physx', choices=('isaacsim_physx',))
    return parser


def configuration(args):
    values = {item.name: getattr(args, item.name) for item in fields(PilotConfig)}
    if args.sustained_force:
        values['force_duration_s'] = None
    return PilotConfig(**values)


def _source_snapshot(output):
    """Archive the exact owned sources plus reused PID and post-lag fault code."""
    files = sorted({
        *(ROOT/'src/drone_adaptation').rglob('*.py'),
        *(path for path in HERE.rglob('*.py')
          if not {'data', 'src', 'source_snapshot', '__pycache__'}.intersection(path.relative_to(HERE).parts)),
    })
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        contents = source.read_bytes()
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        hashes[str(relative)] = hashlib.sha256(contents).hexdigest()
    return hashes


def main(argv=None):
    parser = build_parser()
    known, _ = parser.parse_known_args(argv)
    try:
        config = configuration(known)
    except ValueError as exc:
        parser.error(str(exc))
    if known.check_config:
        args = parser.parse_args(argv)
        print(json.dumps({'status': 'configuration_only_no_simulation', 'backend': 'isaac_unvalidated',
                          'config': configuration(args).to_dict(),
                          'estimator_status': 'configuration only; runtime command-replay checks have not run'}, indent=2, allow_nan=False))
        return 0
    # The parser's configuration/help paths deliberately work without Isaac.
    try:
        from isaaclab.app import add_launcher_args, launch_simulation
        from isaaclab.physics import PhysicsCfg
    except ImportError as exc:
        parser.error('Run flight with the group\'s Isaac Lab environment; --check-config needs no simulator. '+str(exc))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=['kit'])
    args = parser.parse_args(argv)
    if args.headless:
        args.visualizer = []
    try:
        config = configuration(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output is None:
        parser.error('--output must name a new directory for a flight')
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; choose a fresh directory')
    output.mkdir(parents=True)
    manifest = {'schema_version': 1, 'backend': 'isaac', 'status': 'starting', 'runtime_status': 'starting',
                'config': config.to_dict(), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'platform': platform.platform(),
                'source_sha256': _source_snapshot(output),
                'scope': 'Known selected motor; vertical force at COM; fixed PID; no learned or estimate-driven correction',
                'validation': 'New integration: source and mock checks only until this recorded runtime',
                'row_timing': 'State/measurement at t_s; command, actuator outputs and external force apply over [t_s, interval_end_s). Terminal has no next command.',
                'estimator_input_contract': 'Commands, measured vertical velocity and attitude only; actuator thrust, effectiveness and force labels are excluded. Known healthy calibration initializes an independent RPM replay; runtime mismatch aborts before physics.'}
    path = output/'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    from separation_isaac import run
    success = False
    try:
        with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
            success = run(config, args, physics_cfg, output, manifest)
            # Persist within this context: some Isaac launchers terminate on exit.
            path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
            print(f'[RESULT] {manifest["status"]}; {manifest.get("termination_reason")}; {output}', flush=True)
            if not success:
                raise SystemExit(1)
    except Exception as exc:
        manifest.update(status='incomplete', runtime_status='incomplete', error=f'{type(exc).__name__}: {exc}')
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        raise
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
