"""Small ResNet-style policy/value network for Gomoku (15x15)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn(in_ch, out_ch, k=3, p=1, use_bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels) if use_batchnorm else None
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels) if use_batchnorm else None

    def forward(self, x):
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out += x
        return F.relu(out, inplace=True)


class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int = 15, channels: int = 64, num_blocks: int = 5, use_batchnorm: bool = True):
        super().__init__()
        self.board_size = board_size
        self.use_batchnorm = use_batchnorm
        self.stem = _conv_bn(3, channels, k=3, p=1, use_bn=use_batchnorm)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels, use_batchnorm) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2) if use_batchnorm else None
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1) if use_batchnorm else None
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [B, 3, H, W]
        out = self.stem(x)
        out = self.res_blocks(out)

        p = self.policy_conv(out)
        if self.policy_bn is not None:
            p = self.policy_bn(p)
        p = F.relu(p, inplace=True)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = self.value_conv(out)
        if self.value_bn is not None:
            v = self.value_bn(v)
        v = F.relu(v, inplace=True)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value.squeeze(-1)

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Value-only forward pass (skips policy head computation)."""
        out = self.stem(x)
        out = self.res_blocks(out)

        v = self.value_conv(out)
        if self.value_bn is not None:
            v = self.value_bn(v)
        v = F.relu(v, inplace=True)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        value = torch.tanh(self.value_fc2(v))
        return value.squeeze(-1)
