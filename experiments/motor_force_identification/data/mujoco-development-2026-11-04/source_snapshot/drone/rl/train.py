"""Train YOUR PPO (ppo.py) on the Isaac Lab ARL Robot 1 drone env (hover / reach a target pose).

Run (from your IsaacLab folder), headless:
    uv run --extra isaacsim python "<this folder>/train.py" --num_envs 64 --iterations 20 --name smoke

Checkpoints go to rl/runs/<name>/model_<iter>.pt. Look at them with evaluate.py.

How it fits together:
  env (Isaac Lab ManagerBasedRLEnv, thousands of drones on the GPU)  <->  PPO (ppo.py)
  obs   = position, orientation, lin vel, ang vel, last action (with noise)   [see the env cfg]
  action= 4 propeller thrusts: thrust = 3.0 + 3.0 * a  newtons, clipped to [0, 6] (hover is ~3.04 N each)
  The reward is defined by the env (isaaclab_tasks/contrib/drone_arl/.../track_position_state_based_env_cfg.py).

STATUS: written against the Isaac Lab API but not yet run end to end. Run the --iterations 2 smoke test first.
"""

import argparse
import pathlib
import time

import torch

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 conflict_handler="resolve")
parser.add_argument("--task", default="IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--name", default="run")
parser.add_argument("--save_every", type=int, default=10)
parser.add_argument("--physics", default="isaacsim_physx", choices=["isaacsim_physx"])
add_launcher_args(parser)
args = parser.parse_args()  # headless unless --visualizer kit is passed

from isaaclab_tasks.utils import parse_env_cfg

RUN_DIR = pathlib.Path(__file__).resolve().parent / "runs"


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    with launch_simulation(env_cfg, args):
        import gymnasium as gym

        import isaaclab_tasks  # noqa: F401  (registers the envs)
        from ppo import PPO, ActorCritic, PPOConfig

        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        device = base.device
        obs_dim = base.observation_manager.group_obs_dim["policy"][0]
        act_dim = base.action_manager.total_action_dim
        print(f"[INFO] obs_dim={obs_dim} act_dim={act_dim} num_envs={base.num_envs} device={device}")

        model = ActorCritic(obs_dim, act_dim).to(device)
        ppo = PPO(model, PPOConfig())

        stats_acc = {"terminated": 0.0}

        def env_step(action):
            obs, rew, terminated, truncated, _ = env.step(action)
            stats_acc["terminated"] += float(terminated.float().mean())
            return obs["policy"], rew, (terminated | truncated)

        out_dir = RUN_DIR / args.name
        out_dir.mkdir(parents=True, exist_ok=True)
        obs = env.reset()[0]["policy"]
        t0 = time.time()
        for it in range(args.iterations):
            stats_acc["terminated"] = 0.0
            buf, obs = ppo.collect(env_step, obs)
            stats = ppo.update(buf)
            print(f"iter {it:4d} | reward/step {buf['rew'].mean():+.3f} | "
                  f"crash-rate/step {stats_acc['terminated'] / ppo.cfg.horizon:.4f} | "
                  f"pi {stats['pi_loss']:+.4f} v {stats['v_loss']:.3f} | {time.time() - t0:.0f}s")
            if (it + 1) % args.save_every == 0 or it == args.iterations - 1:
                torch.save({"model": model.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim},
                           out_dir / f"model_{it + 1}.pt")
        print(f"[RESULT] saved checkpoints in {out_dir}")
        env.close()


if __name__ == "__main__":
    main()
