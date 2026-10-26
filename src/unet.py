from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.forensic_features import FORGERY_TYPES, INPUT_CHANNELS


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        out_channels: int = 1,
        features: Sequence[int] = (32, 64, 128, 256),
        num_classes: int = 2,
        num_forgery_types: int = len(FORGERY_TYPES),
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(current_channels, feature))
            current_channels = feature

        self.bottleneck = DoubleConv(current_channels, features[-1] * 2)
        reversed_features = list(reversed(features))

        for feature in reversed_features:
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.classification_head = nn.Linear(features[-1] * 2, num_classes)
        self.forgery_type_head = nn.Linear(features[-1] * 2, num_forgery_types)
        self.cam_features: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        current = x

        for down in self.downs:
            current = down(current)
            skip_connections.append(current)
            current = self.pool(current)

        current = self.bottleneck(current)
        self.cam_features = current
        if current.requires_grad:
            current.retain_grad()
        pooled = torch.mean(current, dim=(2, 3))
        classification_logits = self.classification_head(pooled)
        forgery_type_logits = self.forgery_type_head(pooled)

        for idx in range(len(self.ups) // 2):
            current = self.ups[idx * 2](current)
            skip = skip_connections[-(idx + 1)]
            if current.shape[2:] != skip.shape[2:]:
                current = nn.functional.interpolate(current, size=skip.shape[2:], mode="bilinear", align_corners=False)
            current = torch.cat([skip, current], dim=1)
            current = self.ups[idx * 2 + 1](current)

        return {
            "segmentation_logits": self.output_conv(current),
            "classification_logits": classification_logits,
            "forgery_type_logits": forgery_type_logits,
        }


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    cardinality = (pred + target).sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (cardinality + eps)).mean()
