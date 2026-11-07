"""Fly a trained checkpoint (from train.py) in 1 env, watch it, and log the flight to a CSV.

Run (from your IsaacLab folder):
    uv run --extra isaacsim python "<this folder>/evaluate.py" --checkpoint "<this folder>/runs/smoke/model_2.pt" --visualizer kit

Writes ../data/eval_<name>.csv (position, velocity, action per step).

STATUS: written against the Isaac Lab API but not yet run end to end.
"""

import argparse
import pathlib

import torch

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 conflict_handler="resolve")
parser.add_argument("--task", default="IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=500, help="Env steps (each 0.1 s of sim time).")
parser.add_argument("--name", default="policy")
parser.add_argument("--physics", default="isaacsim_physx", choices=["isaacsim_physx"])
add_launcher_args(parser)
args = parser.parse_args()

from isaaclab_tasks.utils import parse_env_cfg

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    with launch_simulation(env_cfg, args):
        import gymnasium as gym
        import pandas as pd

        import isaaclab_tasks  # noqa: F401
        from ppo import ActorCritic

        env = gym.make(args.task, cfg=env_cfg)
        device = env.unwrapped.device
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        robot = env.unwrapped.scene["robot"]
        obs = env.reset()[0]["policy"]
        rows = []
        for step in range(args.steps):
            with torch.no_grad():
                action = model.actor(obs)  # deterministic: use the mean, no sampling noise
            obs_dict, rew, terminated, truncated, _ = env.step(action)
            obs = obs_dict["policy"]
            pos = robot.data.root_pos_w[0].tolist()
            vel = robot.data.root_lin_vel_w[0].tolist()
            rows.append({"step": step, "reward": float(rew[0]), "terminated": bool(terminated[0]),
                         **{f"pos_{a}": pos[i] for i, a in enumerate("xyz")},
                         **{f"lin_vel_{a}": vel[i] for i, a in enumerate("xyz")},
                         **{f"action_{i}": float(action[0, i]) for i in range(action.shape[1])}})

        DATA_DIR.mkdir(exist_ok=True)
        path = DATA_DIR / f"eval_{args.name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        crashes = sum(r["terminated"] for r in rows)
        print(f"[RESULT] {len(rows)} steps -> {path} | episodes ended by crash: {crashes}")
        env.close()


if __name__ == "__main__":
    main()
