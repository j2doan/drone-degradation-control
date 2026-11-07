"""A small, readable PPO in plain PyTorch. This is YOUR file to modify (reward shaping lives in the env; the
learning algorithm lives here).

Works with any *vectorized, batched-on-GPU* env that gives obs of shape (N, obs_dim) and takes actions
(N, act_dim) - which is what Isaac Lab envs do. No Isaac imports here, so it is testable on its own
(see test_ppo_toy.py).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


def mlp(inp: int, out: int, hidden=(128, 128)) -> nn.Sequential:
    layers, last = [], inp
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    layers.append(nn.Linear(last, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian policy (mean from an MLP, learned state-independent log-std) + separate value MLP."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128), init_std: float = 0.5):
        super().__init__()
        self.actor = mlp(obs_dim, act_dim, hidden)
        self.critic = mlp(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.log(torch.tensor(init_std)))))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def act(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.critic(obs).squeeze(-1)


@dataclass
class PPOConfig:
    horizon: int = 24  # env steps collected per env per iteration
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 1.0
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0


class PPO:
    def __init__(self, model: ActorCritic, cfg: PPOConfig = PPOConfig()):
        self.model, self.cfg = model, cfg
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def collect(self, env_step, obs):
        """Roll out cfg.horizon steps. `env_step(action) -> (next_obs, reward, done)`. Returns (buffer, last_obs)."""
        T = self.cfg.horizon
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
        for _ in range(T):
            act, logp, val = self.model.act(obs)
            next_obs, rew, done = env_step(act)
            for k, v in zip(buf, (obs, act, logp, val, rew, done.float())):
                buf[k].append(v)
            obs = next_obs
        buf = {k: torch.stack(v) for k, v in buf.items()}  # each (T, N, ...)
        last_val = self.model.critic(obs).squeeze(-1)
        buf["adv"], buf["ret"] = self._gae(buf, last_val)
        return buf, obs

    def _gae(self, buf, last_val):
        T = buf["rew"].shape[0]
        adv = torch.zeros_like(buf["rew"])
        gae = torch.zeros_like(last_val)
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else buf["val"][t + 1]
            not_done = 1.0 - buf["done"][t]
            delta = buf["rew"][t] + self.cfg.gamma * next_val * not_done - buf["val"][t]
            gae = delta + self.cfg.gamma * self.cfg.lam * not_done * gae
            adv[t] = gae
        return adv, adv + buf["val"]

    def update(self, buf) -> dict:
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in buf.items()}
        adv = flat["adv"]
        flat["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = flat["obs"].shape[0]
        stats = {}
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(n, device=adv.device)
            for idx in perm.chunk(self.cfg.minibatches):
                d = self.model.dist(flat["obs"][idx])
                logp = d.log_prob(flat["act"][idx]).sum(-1)
                ratio = (logp - flat["logp"][idx]).exp()
                a = flat["adv"][idx]
                pi_loss = -torch.min(ratio * a, ratio.clamp(1 - self.cfg.clip, 1 + self.cfg.clip) * a).mean()
                v_loss = (self.model.critic(flat["obs"][idx]).squeeze(-1) - flat["ret"][idx]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                stats = {"pi_loss": pi_loss.item(), "v_loss": v_loss.item(), "entropy": ent.item()}
        return stats
