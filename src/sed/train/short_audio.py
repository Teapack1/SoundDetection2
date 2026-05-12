"""Short-event spectrogram encoders for glass/cheers experiments.

The classes here are dependency-free local models that follow the custom
backbone contract: ``in_chans``, ``n_bands`` and ``n_frames`` are accepted by
the constructor, and ``forward`` returns either a spatial map ``(B, C, F, T)``
or a pooled embedding ``(B, D)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _channels(base: int, multiplier: float) -> int:
    return max(8, int(round(base * float(multiplier) / 8.0)) * 8)


class _ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
        act: type[nn.Module] = nn.SiLU,
    ):
        padding = (
            kernel_size // 2
            if isinstance(kernel_size, int)
            else (kernel_size[0] // 2, kernel_size[1] // 2)
        )
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            act(inplace=True) if act is nn.ReLU else act(),
        )


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int] = (1, 1),
                 dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBNAct(in_ch, out_ch, 3, stride=stride, act=nn.ReLU),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Identity() if in_ch == out_ch and stride == (1, 1)
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class TransientCNNGAP(nn.Module):
    """Small CNN with global pooling-friendly feature maps."""

    def __init__(self, in_chans: int = 1, n_bands: int = 128, n_frames: int = 80,
                 width_mult: float = 1.0, depth: int = 4, time_stride: int = 1,
                 dropout: float = 0.1, **_ignored):
        super().__init__()
        _ = (n_bands, n_frames)
        c0 = _channels(48, width_mult)
        self.stem = nn.Sequential(
            _ConvBNAct(in_chans, c0, 5, stride=(2, max(1, int(time_stride)))),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
        )
        blocks: list[nn.Module] = []
        channels = c0
        for i in range(max(1, int(depth))):
            out = _channels(48 * (2 ** min(i // 2, 2)), width_mult)
            stride = (2, 1) if i > 0 and i % 2 == 0 else (1, 1)
            blocks.append(_ConvBNAct(channels, out, 3, stride=stride))
            channels = out
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


class TransientResNet(nn.Module):
    """Shallow residual CNN tuned for short transients."""

    def __init__(self, in_chans: int = 1, n_bands: int = 128, n_frames: int = 80,
                 width_mult: float = 1.0, depth: int = 4, time_stride: int = 1,
                 dropout: float = 0.1, **_ignored):
        super().__init__()
        _ = (n_bands, n_frames)
        c0 = _channels(64, width_mult)
        self.stem = _ConvBNAct(in_chans, c0, 5, stride=(2, max(1, int(time_stride))), act=nn.ReLU)
        blocks: list[nn.Module] = []
        channels = c0
        for i in range(max(1, int(depth))):
            out = _channels(64 * (2 ** min(i // 2, 2)), width_mult)
            stride = (2, 1) if i > 0 and i % 2 == 0 else (1, 1)
            blocks.append(_ResidualBlock(channels, out, stride=stride, dropout=dropout))
            channels = out
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


__all__ = [
    "TransientCNNGAP",
    "TransientResNet",
]
