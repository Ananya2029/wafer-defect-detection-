"""Compact CNN for 52x52 wafer maps with 8 independent (multi-label) outputs."""
import torch
import torch.nn as nn


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class WaferCNN(nn.Module):
    def __init__(self, n_labels=8, in_channels=2):
        super().__init__()
        self.features = nn.Sequential(
            block(in_channels, 32), nn.MaxPool2d(2),   # 52 -> 26
            block(32, 64), nn.MaxPool2d(2),            # 26 -> 13
            block(64, 128), nn.MaxPool2d(2),           # 13 -> 6
            block(128, 128),                           # last conv block: Grad-CAM target
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(128, n_labels))

    def forward(self, x):
        return self.head(self.features(x))


def random_dihedral(x: torch.Tensor) -> torch.Tensor:
    """Random 90-degree rotation + flip per batch. Defect patterns keep their meaning
    under rotation and mirroring (an edge defect on the left is still an edge defect)."""
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, dims=(2, 3))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(3,))
    return x
