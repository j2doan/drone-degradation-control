"""Run or plan the four-condition motor/disturbance identification pilot.

python -m drone_adaptation.separation.run --backend mujoco --output results/mujoco-development --plot
python -m drone_adaptation.separation.run --backend isaac --plan-only --output results/separation-gpu-plan
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .config import CONDITIONS, MOTIONS, PilotConfig, matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def snapshot_sources(output):
    paths = [* (PROJECT_ROOT/'drone_adaptation').rglob('*.py'),
             * (PROJECT_ROOT/'drone').rglob('*.py'),
             PROJECT_ROOT/'experiments/isaac/fault_actuator.py',
             PROJECT_ROOT/'experiments/isaac/faults.py']
    hashes={}
    for source in sorted(paths):
        relative=source.relative_to(PROJECT_ROOT)
        dest=output/'source_snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    shared = PROJECT_ROOT.parent/'drone/controllers/pid.py'
    if shared.is_file():
        key = '_team/shared/drone/controllers/pid.py'
        destination = output/'source_snapshot'/key
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(shared,destination)
        hashes[key] = hashlib.sha256(shared.read_bytes()).hexdigest()
    return hashes


def config_arguments(config):
    values=config.to_dict()
    values.pop('dt_s')  # fixed by this pilot
    result=[]
    for key,value in values.items():
        if value is not None:
            result.extend(['--'+key.replace('_','-'),str(value)])
    if config.force_duration_s is None:
        result.append('--sustained-force')
    return result


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend',choices=('portable','mujoco','isaac'),default='portable')
    parser.add_argument('--output',type=Path,required=True,help='A new output directory; never overwrites')
    parser.add_argument('--conditions',choices=CONDITIONS,nargs='+',default=list(CONDITIONS))
    parser.add_argument('--scene',type=Path,help='Optional MuJoCo scene override; otherwise use the pinned X2 assets')
    parser.add_argument('--view',action='store_true',help='MuJoCo viewer for exactly one episode; use mjpython on macOS')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    parser.add_argument('--motions',choices=MOTIONS,nargs='+',default=list(MOTIONS))
    parser.add_argument('--plan-only',action='store_true',help='Write explicit commands/configs without launching a simulator')
    parser.add_argument('--isaaclab-dir',type=Path,help='Working SDK checkout containing pyproject.toml; otherwise ISAACLAB_DIR')
    parser.add_argument('--plot',action='store_true')
    defaults=PilotConfig()
    for name in ('motor','motor_loss','motor_onset_s','motor_duration_s','force_z_n','force_onset_s',
                 'force_duration_s','duration_s','history_s','velocity_noise_std_m_s','initial_z_jitter_m',
                 'probe_amplitude_m','probe_frequency_hz','probe_onset_s','target_z_m','max_thrust_n'):
        parser.add_argument('--'+name.replace('_','-'),type=int if name=='motor' else float,default=getattr(defaults,name))
    parser.add_argument('--sustained-force',action='store_true',help='Control condition: external force persists until episode end')
    return parser


def main(argv=None):
    parser=build_parser()
    args=parser.parse_args(argv)
    if args.plan_only and args.backend != 'isaac':
        parser.error('--plan-only prepares GPU commands; select --backend isaac')
    settings={name:getattr(args,name) for name in PilotConfig.__dataclass_fields__ if hasattr(args,name)}
    if args.sustained_force:
        settings['force_duration_s']=None
    try:
        if len(set(args.conditions)) != len(args.conditions):
            raise ValueError('conditions must be unique')
        configs=[c for c in matrix(PilotConfig(**settings),seeds=args.seeds,motions=args.motions) if c.condition in args.conditions]
    except (ValueError,TypeError) as error:
        parser.error(str(error))
    if args.view and (args.backend != 'mujoco' or len(configs) != 1):
        parser.error('--view requires --backend mujoco and exactly one condition, motion and seed')
    if args.scene is not None and args.backend != 'mujoco':
        parser.error('--scene is a MuJoCo-only option')
    output=args.output.expanduser().resolve()
    if output.exists():
        parser.error('Output already exists; choose a fresh directory')
    sdk=None
    if args.backend=='isaac' and not args.plan_only:
        import os
        raw=args.isaaclab_dir or os.environ.get('ISAACLAB_DIR')
        if not raw:
            parser.error('Set --isaaclab-dir or ISAACLAB_DIR for GPU execution; use --plan-only to prepare commands locally')
        sdk=Path(raw).expanduser().resolve()
        if not (sdk/'pyproject.toml').is_file() or not shutil.which('uv'):
            parser.error('Isaac execution requires the working SDK checkout and uv on PATH')
    output.mkdir(parents=True)
    manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  backend={'portable':'portable_vertical_model','mujoco':'mujoco_x2','isaac':'isaac'}[args.backend],
                  status='planned' if args.plan_only else 'running',
                  python=sys.version,platform=platform.platform(),seeds=args.seeds,motions=args.motions,conditions=args.conditions,
                  scene_override=str(args.scene.resolve()) if args.scene else None,
                  shared_controller_archive='_team/shared/drone/controllers/pid.py',
                  configs=[config.to_dict() for config in configs],
                  inference_only=True,estimators_in_shadow=True,
                  commands_file_role='Isaac child commands; portable/MuJoCo per-episode reproduction commands use fresh _rerun output directories',
                  paired_seeds='Initial-state jitter and synthetic observation noise, NOT independently trained policies',
                  selection='First listed seed per motion is the predeclared plot trace',
                  source_sha256=snapshot_sources(output))
    write_json(output/'manifest.json',manifest)
    commands=[]
    for config in configs:
        episode=output/f'{config.motion}_seed{config.seed}'/config.condition
        if args.backend=='isaac':
            command=['uv','run','--extra','isaacsim','python',str(PROJECT_ROOT/'drone/04_separation_pilot.py'),
                     *config_arguments(config),'--output',str(episode),'--headless']
            working_directory=str(sdk) if sdk else '<your IsaacLab checkout>'
        else:
            translated={'--condition':'--conditions','--motion':'--motions','--seed':'--seeds'}
            arguments=[translated.get(value,value) for value in config_arguments(config)]
            command=[sys.executable,'-m','drone_adaptation.separation.run','--backend',args.backend,
                     *arguments,'--output',str(episode.with_name(episode.name+'_rerun'))]
            if args.scene:
                command.extend(['--scene',str(args.scene.resolve())])
            working_directory=str(PROJECT_ROOT)
        commands.append(dict(config=config.to_dict(),output=str(episode),argv=command,
                             working_directory=working_directory))
    write_json(output/'commands.json',commands)
    if args.plan_only:
        print(f'Prepared {len(configs)} configurations; no simulations executed. Read {output / "commands.json"}')
        return 0
    from .analysis import summarize,write_report
    summaries=[]
    current_config=None
    current_episode=None
    try:
        for config,item in zip(configs,commands):
            episode=Path(item['output'])
            current_config,current_episode=config,episode
            if args.backend in ('portable','mujoco'):
                if args.backend=='mujoco':
                    from .mujoco_backend import simulate
                    rows,runtime,status=simulate(config,scene_path=args.scene,viewer=True if args.view else None)
                else:
                    from .portable import simulate
                    rows,runtime,status=simulate(config)
                episode.mkdir(parents=True)
                write_csv(episode/'trajectory.csv',rows)
                write_json(episode/'manifest.json',dict(config=config.to_dict(),**runtime))
            else:
                episode.parent.mkdir(parents=True,exist_ok=True)
                log_path=episode.parent/f'{config.condition}.log'
                with log_path.open('w',encoding='utf-8') as log:
                    result=subprocess.run(item['argv'],cwd=sdk,stdout=log,stderr=subprocess.STDOUT,check=False)
                if not (episode/'manifest.json').is_file():
                    raise RuntimeError(f'Isaac produced no manifest (exit {result.returncode}); inspect {log_path}')
                runtime=json.loads((episode/'manifest.json').read_text())
                status=runtime.get('status','incomplete')
                trace_path=episode/'trajectory.csv'
                if not trace_path.is_file():
                    raise RuntimeError(f'Isaac recorded no trajectory: {runtime.get("error", "no state samples")}; inspect {log_path}')
                with trace_path.open(newline='',encoding='utf-8') as stream:
                    rows=list(csv.DictReader(stream))
                if result.returncode and status=='complete':
                    status='incomplete'
            summary=summarize(rows,config,status=status)
            summaries.append(summary)
            write_json(episode/'summary.json',summary)
            write_csv(output/'episodes.csv',summaries)
            print(f'{config.motion} seed={config.seed} {config.condition}: {status}')
            if status not in ('complete','failed') or (args.backend=='isaac' and result.returncode):
                detail=runtime.get('error',runtime.get('termination_reason','uncompleted record'))
                raise RuntimeError(f'{args.backend} did not complete a valid record: {detail}. Matrix stopped; partial records retained.')
        manifest['status']='complete'
    except BaseException as error:
        manifest['status']='incomplete'
        manifest['error']=f'{type(error).__name__}: {error}'
        # Setup/API failures may produce no state samples. Retain an explicit
        # incomplete record rather than silently dropping the attempted trial.
        if current_config is not None and not any(
                row.get('condition')==current_config.condition and row.get('motion')==current_config.motion
                and row.get('seed')==current_config.seed for row in summaries):
            failed_attempt=dict(condition=current_config.condition,motion=current_config.motion,seed=current_config.seed,
                status='incomplete',failed=False,trace_complete=False,observed_s=0.0,
                common_horizon_s=current_config.duration_s,penalized_tracking_mae_m=1.0,error=manifest['error'])
            summaries.append(failed_attempt)
            current_episode.mkdir(parents=True,exist_ok=True)
            write_json(current_episode/'summary.json',failed_attempt)
            write_csv(output/'episodes.csv',summaries)
        raise
    finally:
        write_json(output/'manifest.json',manifest)
        write_report(output,summaries,manifest)
    if args.plot:
        from .analysis import plot_matrix
        plots=plot_matrix(output)
        print(f'Created {len(plots)} diagnostic figures.')
    print(f'Read {output / "RESULTS.md"}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
