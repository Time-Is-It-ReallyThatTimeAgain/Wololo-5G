"""Dueling Double-DQN network for 18-dim observation and 27 actions."""

from __future__ import annotations

import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """Input(18) -> [256, 256, 128] with dueling heads -> Output(27)."""

    def __init__(self, obs_dim: int = 18, action_dim: int = 27) -> None:
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.adv_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature(x)
        value = self.value_stream(feats)
        adv = self.adv_stream(feats)
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q
