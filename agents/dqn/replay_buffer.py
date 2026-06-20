"""Uniform replay buffer for DQN.

Array/list-backed ring buffer with O(1) random indexing.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import List , Tuple

import numpy as np
import torch


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = int(capacity)
        self .buffer: List [Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        
        t = Transition(
            obs=np.asarray(obs, dtype=np.float32),
            action = int (action),
            reward = float (reward),
            next_obs=np.asarray(next_obs, dtype=np.float32),
            done = bool (done),
        )
        if len ( self .buffer) < self .capacity:
 
            self .buffer.append(t)
        else :
            self.buffer [ self.pos ] = t
        self .pos = ( self .pos + 1 ) % self .capacity

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size > len(self.buffer):
            raise ValueError(
                f"batch_size ({batch_size}) cannot exceed buffer size ({len(self.buffer)})"
            )
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]

        obs = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        next_obs = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

        return obs, actions, rewards, next_obs, dones
