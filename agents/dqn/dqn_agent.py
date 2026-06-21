"""Double DQN agent with dueling network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.dqn.dqn_network import DuelingDQN
from agents.dqn.replay_buffer import ReplayBuffer


@dataclass
class DqnConfig:
    obs_dim: int = 18
    action_dim: int = 27
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 50_000
    tau: float = 0.001
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 50_000
    grad_clip: float = 1.0


class DqnAgent:
    def __init__(self, cfg: DqnConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        self.online_net = DuelingDQN(cfg.obs_dim, cfg.action_dim).to(device)
        self.target_net = DuelingDQN(cfg.obs_dim, cfg.action_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optim = torch.optim.Adam(self.online_net.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)

        self.total_steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        eps = self.epsilon() if explore else 0.0
        if np.random.rand() < eps:
            return int(np.random.randint(self.cfg.action_dim))

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        qvals = self.online_net(obs_t)
        return int(torch.argmax(qvals, dim=1).item())

    def store(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.buffer.add(obs, action, reward, next_obs, done)
        

    def train_step(self) -> Dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size:
            return {"loss": 0.0}
        self.total_steps += 1     

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.cfg.batch_size, self.device)

        q = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_net(next_obs).argmax(dim=1)
            next_q_target = self.target_net(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + (1.0 - dones) * self.cfg.gamma * next_q_target

        loss = F.smooth_l1_loss(q, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(self.online_net.parameters(), self.cfg.grad_clip)
        self.optim.step()

        self.soft_update()

        return {"loss": float(loss.item())}

    def soft_update(self) -> None:
        tau = self.cfg.tau
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optim": self.optim.state_dict(),
                "total_steps": self.total_steps,
                "cfg": self.cfg.__dict__,
            },
            path,
        )
        
        buf_path = path.with_suffix(".buf")
        with open(buf_path, "wb") as f:
            pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_buffer(self, path: str | Path) -> None:
        buf_path = Path(path).with_suffix(".buf")
        if buf_path.exists():
            with open(buf_path, "rb") as f:
                self.buffer = pickle.load(f)
            print(f"[DqnAgent] Replay buffer restored: {len(self.buffer)} transitions")
        else:
            print(f"[DqnAgent] No buffer sidecar found at {buf_path} — starting empty")
