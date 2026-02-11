import torch
import torch.nn as nn


class SmokeCNN(nn.Module):
    """
    Minimal CNN for smoke testing.
    Binary output (presence / absence).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)
