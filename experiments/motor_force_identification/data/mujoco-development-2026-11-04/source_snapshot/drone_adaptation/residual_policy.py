"""Small temporal feature extractor shared by every history condition."""

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TemporalResidualExtractor(BaseFeaturesExtractor):
    """All windows use the same parameters and 50-sample input layout.

    The environment zero-masks observations older than its permitted window.
    History=0 is a latest-information control: its temporal input is all zero,
    while the current vector still includes a completed command-response signal.
    Flattening the fixed-length convolution output gives the head access to the
    entire allowed window, rather than only the newest local receptive field.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 48):
        if features_dim != 48:
            raise ValueError("The shared extractor uses 48 features (16 current + 32 temporal)")
        super().__init__(observation_space, features_dim)
        current_size = observation_space["current"].shape[0]
        channels, length = observation_space["history"].shape
        self.current_net = nn.Sequential(nn.Linear(current_size, 16), nn.Tanh())
        self.history_net = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * length, 32),
            nn.Tanh(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            (self.current_net(observations["current"]), self.history_net(observations["history"])),
            dim=1,
        )
