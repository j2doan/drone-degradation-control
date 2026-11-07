"""Sanity check for ppo.py with NO Isaac Sim: a batched 2-D point mass must learn to reach the origin.

Run (from your IsaacLab folder, or any python with torch):
    uv run --extra isaacsim python "<this folder>/test_ppo_toy.py"
Expected: mean reward/step rises from about -1.2 to about 0 (the point mass learns to reach the origin).
"""

import torch

from ppo import PPO, ActorCritic, PPOConfig

N, DT = 256, 0.05
torch.manual_seed(0)


class ToyEnv:
    """Point mass in 2-D. obs = (pos, vel), action = acceleration, reward = -distance. Episodes last 100 steps."""

    def reset(self):
        self.pos = torch.rand(N, 2) * 3 - 1.5
        self.vel = torch.zeros(N, 2)
        self.t = torch.zeros(N)
        return torch.cat([self.pos, self.vel], -1)

    def step(self, a):
        self.vel += a.clamp(-1, 1) * DT * 4
        self.pos += self.vel * DT
        self.t += 1
        rew = -self.pos.norm(dim=-1)
        done = self.t >= 100
        obs = torch.cat([self.pos, self.vel], -1)
        if done.any():  # auto-reset finished envs
            n = int(done.sum())
            self.pos[done] = torch.rand(n, 2) * 3 - 1.5
            self.vel[done] = 0
            self.t[done] = 0
            obs = torch.cat([self.pos, self.vel], -1)
        return obs, rew, done


env = ToyEnv()
ppo = PPO(ActorCritic(4, 2), PPOConfig(horizon=32, lr=1e-3))
obs = env.reset()
for it in range(150):
    buf, obs = ppo.collect(env.step, obs)
    stats = ppo.update(buf)
    if it % 25 == 0 or it == 149:
        print(f"iter {it:3d}  mean reward/step {buf['rew'].mean():+.3f}  {stats}")
