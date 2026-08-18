"""Neural-network architectures for ECG classification."""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Two-layer residual block with optional temporal downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.convolutions(x) + self.skip(x))


class ECGCNN(nn.Module):
    """Compact 1D convolutional baseline for 12-lead ECG recordings."""

    def __init__(self, num_classes: int = 5):
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class ResidualECGCNN(nn.Module):
    """Regularized residual CNN for the first post-baseline experiment."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.features = nn.Sequential(
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
            ResidualBlock1D(128, 256, stride=2),
            ResidualBlock1D(256, 256),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(self.stem(x)))
