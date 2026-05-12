"""Small, local spectrogram encoders for short-event experiments.

These are intentionally compact reference implementations rather than copied
vendor code.  They share the encoder contract used by ``CustomAdapter``: a
``forward`` call returns a ``(B, C, F, T)`` feature map, so clip models pool it
and frame models retain its time axis.
"""

from __future__ import annotations


import torch
import torch.nn as nn


def _channels(base: int, multiplier: float) -> int:
    return max(8, int(round(base * float(multiplier) / 8)) * 8)


class _DWBlock(nn.Module):
    def __init__(self, channels: int, *, time_stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=(1, time_stride), padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DSCNN(nn.Module):
    """Depthwise-separable CNN baseline for low-latency short audio."""

    def __init__(self, in_chans: int = 1, width_mult: float = 1.0,
                 time_stride: int = 1, depth: int = 4, **_ignored):
        super().__init__()
        c = _channels(48, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, c, 3, stride=(2, max(1, int(time_stride))), padding=1, bias=False),
            nn.BatchNorm2d(c), nn.SiLU(),
        )
        self.blocks = nn.ModuleList([_DWBlock(c) for _ in range(max(1, int(depth)))])
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


class TCResNet(nn.Module):
    """Temporal-convolution residual encoder with deliberately low time stride."""

    def __init__(self, in_chans: int = 1, width_mult: float = 1.0,
                 time_stride: int = 1, depth: int = 4, **_ignored):
        super().__init__()
        c = _channels(64, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, c, 3, stride=(2, max(1, int(time_stride))), padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        for _ in range(max(1, int(depth))):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(c, c, (1, 5), padding=(0, 2), bias=False),
                nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, (1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(c),
            ))
        self.act = nn.ReLU(inplace=True)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = self.act(x + block(x))
        return x


class BCResNet(nn.Module):
    """Broadcast-residual inspired short-audio encoder.

    The frequency-mean branch broadcasts temporal context back over the
    feature map, retaining the useful KWS inductive bias without depending on
    an unpinned external implementation.
    """

    def __init__(self, in_chans: int = 1, width_mult: float = 1.0,
                 time_stride: int = 1, depth: int = 4, **_ignored):
        super().__init__()
        c = _channels(64, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, c, 3, stride=(2, max(1, int(time_stride))), padding=1, bias=False),
            nn.BatchNorm2d(c), nn.SiLU(),
        )
        self.blocks = nn.ModuleList()
        for _ in range(max(1, int(depth))):
            self.blocks.append(nn.ModuleDict({
                "local": nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                    nn.BatchNorm2d(c), nn.SiLU(),
                    nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c),
                ),
                "temporal": nn.Sequential(
                    nn.Conv1d(c, c, 5, padding=2, groups=c, bias=False),
                    nn.BatchNorm1d(c), nn.SiLU(),
                    nn.Conv1d(c, c, 1, bias=False), nn.BatchNorm1d(c),
                ),
            }))
        self.act = nn.SiLU()
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            local = block["local"](x)
            temporal = block["temporal"](x.mean(dim=2)).unsqueeze(2)
            x = self.act(x + local + temporal)
        return x


__all__ = ["BCResNet", "TCResNet", "DSCNN"]
