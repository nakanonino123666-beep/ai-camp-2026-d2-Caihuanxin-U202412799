"""Models for real concrete crack image classification."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """A small two-class network for 3 by 64 by 64 image tensors."""

    def __init__(self) -> None:
        super().__init__()
        # TODO 1:
        # Build exactly this sequence:
        # Conv2d(3, 8, kernel_size=3, padding=1), ReLU, MaxPool2d(2),
        # Conv2d(8, 16, kernel_size=3, padding=1), ReLU, MaxPool2d(2),
        # Flatten, Linear(16 * 16 * 16, 2)
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # TODO 2: return the result of sending images through the stored network.
        return self.network(images)
