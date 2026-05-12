"""Checkpoint-compatible EfficientSED Frame-MobileNet encoders.

This is a compact adaptation of the MIT-licensed fMN implementation from
https://github.com/theMoro/EfficientSED at revision
e3449bfd1f4922b81d82410ed842653f462bfca8.  Prediction heads are intentionally
omitted: SoundDetection2 supplies its own classifier while the official
AudioSet Weak/Strong encoder weights are loaded strictly.

Copyright (c) 2024 Florian Schmid. Licensed under the MIT License; the full
notice is distributed beside this module in ``EFFICIENTSED_LICENSE.txt``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.ops.misc import Conv2dNormActivation

EFFICIENTSED_RELEASE = "https://github.com/theMoro/EfficientSED/releases/download/v0.0.1"


def _make_divisible(value: float, divisor: int = 8) -> int:
    result = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return result + divisor if result < 0.9 * value else result


def _conv_out(size: int, padding: int, dilation: int, kernel: int, stride: int) -> int:
    return (int(size) + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class _SqueezeExcitation(nn.Module):
    def __init__(self, input_dim: int, squeeze_dim: int, se_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, squeeze_dim)
        self.fc2 = nn.Linear(squeeze_dim, input_dim)
        dimensions = [1, 2, 3]
        dimensions.remove(int(se_dim))
        self.se_dim = dimensions
        self.activation = nn.ReLU()
        self.scale_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(self.se_dim, keepdim=True)
        shape = scale.shape
        scale = scale.squeeze(2).squeeze(2)
        scale = self.fc2(self.activation(self.fc1(scale)))
        return self.scale_activation(scale).view(shape) * x


class _ConcurrentSE(nn.Module):
    def __init__(self, channels: int, bands: int, frames: int, *, reduction: int = 4):
        super().__init__()
        _ = (bands, frames)
        squeeze = _make_divisible(channels // int(reduction), 8)
        # Official fMN checkpoints use channel SE (se_dims='c'). Keep the
        # original attribute name so state-dict keys remain byte-compatible.
        self.conc_se_layers = nn.ModuleList([_SqueezeExcitation(channels, squeeze, 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([layer(x) for layer in self.conc_se_layers], dim=0).amax(dim=0)


class _BlockConfig:
    def __init__(self, in_channels: int, kernel: int, expanded_channels: int,
                 out_channels: int, use_se: bool, activation: str,
                 stride: tuple[int, int], dilation: tuple[int, int], width: float):
        self.input_channels = _make_divisible(in_channels * width)
        self.kernel = int(kernel)
        self.expanded_channels = _make_divisible(expanded_channels * width)
        self.out_channels = _make_divisible(out_channels * width)
        self.use_se = bool(use_se)
        self.use_hs = activation == "HS"
        self.stride = tuple(int(v) for v in stride)
        self.dilation = tuple(int(v) for v in dilation)
        self.f_dim = 0
        self.t_dim = 0

    def out_size(self, size: int, axis: int) -> int:
        dilation = self.dilation[axis]
        padding = (self.kernel - 1) // 2 * dilation
        return _conv_out(size, padding, dilation, self.kernel, self.stride[axis])


class _InvertedResidual(nn.Module):
    def __init__(self, config: _BlockConfig, norm_layer: Callable[..., nn.Module]):
        super().__init__()
        self.use_res_connect = (
            config.stride == (1, 1) and config.input_channels == config.out_channels
        )
        activation = nn.Hardswish if config.use_hs else nn.ReLU
        layers: list[nn.Module] = []
        if config.expanded_channels != config.input_channels:
            layers.append(Conv2dNormActivation(
                config.input_channels, config.expanded_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=activation,
            ))
        stride = list(config.stride)
        if config.dilation[1] > 1:
            stride[1] = 1
        layers.append(Conv2dNormActivation(
            config.expanded_channels, config.expanded_channels,
            kernel_size=config.kernel, stride=tuple(stride), dilation=config.dilation,
            groups=config.expanded_channels, norm_layer=norm_layer,
            activation_layer=activation,
        ))
        if config.use_se:
            layers.append(_ConcurrentSE(
                config.expanded_channels, config.f_dim, config.t_dim, reduction=4,
            ))
        layers.append(Conv2dNormActivation(
            config.expanded_channels, config.out_channels, kernel_size=1,
            norm_layer=norm_layer, activation_layer=None,
        ))
        self.block = nn.Sequential(*layers)
        self.out_channels = config.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        return result + x if self.use_res_connect else result


def _fmn_configs(width: float, n_bands: int, n_frames: int) -> list[_BlockConfig]:
    frequency_strides = (1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2)
    temporal_strides = (1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    definitions = (
        (16, 3, 16, 16, False, "RE"),
        (16, 3, 64, 24, False, "RE"),
        (24, 3, 72, 24, False, "RE"),
        (24, 5, 72, 40, True, "RE"),
        (40, 5, 120, 40, True, "RE"),
        (40, 5, 120, 40, True, "RE"),
        (40, 3, 240, 80, False, "HS"),
        (80, 3, 200, 80, False, "HS"),
        (80, 3, 184, 80, False, "HS"),
        (80, 3, 184, 80, False, "HS"),
        (80, 3, 480, 112, True, "HS"),
        (112, 3, 672, 112, True, "HS"),
        (112, 5, 672, 160, True, "HS"),
        (160, 5, 960, 160, True, "HS"),
        (160, 5, 960, 160, True, "HS"),
    )
    bands = _conv_out(n_bands, 1, 1, 3, 2)
    frames = _conv_out(n_frames, 1, 1, 3, 2)
    configs: list[_BlockConfig] = []
    for index, definition in enumerate(definitions):
        config = _BlockConfig(
            *definition,
            stride=(frequency_strides[index], temporal_strides[index]),
            dilation=(1, 1),
            width=width,
        )
        bands = config.out_size(bands, 0)
        frames = config.out_size(frames, 1)
        config.f_dim, config.t_dim = bands, frames
        configs.append(config)
    return configs


class EfficientSEDFrameMobileNet(nn.Module):
    """Official fMN encoder adapted to the local 4-D feature-map contract."""

    width_mult = 1.0
    checkpoint_name = "fmn10_strong.pt"
    checkpoint_sha256 = "92a5c1dc5aa21620e262420c4cc657d2f699bdfa20a748de506c354a2fe2770a"

    def __init__(self, in_chans: int = 1, n_bands: int = 128, n_frames: int = 41,
                 pretrained: bool = True, **_ignored):
        super().__init__()
        if int(in_chans) != 1:
            raise ValueError("EfficientSED fMN accepts one log-mel channel")
        if int(n_bands) != 128:
            raise ValueError(f"EfficientSED fMN expects 128 mel bands, got {int(n_bands)}")
        norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        configs = _fmn_configs(float(self.width_mult), int(n_bands), int(n_frames))
        layers: list[nn.Module] = [Conv2dNormActivation(
            1, configs[0].input_channels, kernel_size=3, stride=2,
            norm_layer=norm, activation_layer=nn.Hardswish,
        )]
        layers.extend(_InvertedResidual(config, norm) for config in configs)
        final_channels = 6 * configs[-1].out_channels
        layers.append(Conv2dNormActivation(
            configs[-1].out_channels, final_channels, kernel_size=1,
            norm_layer=norm, activation_layer=nn.Hardswish,
        ))
        self.features = nn.Sequential(*layers)
        self.out_channels = int(final_channels)
        self._initialize()
        if pretrained:
            self._load_official_checkpoint()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _load_official_checkpoint(self) -> None:
        url = f"{EFFICIENTSED_RELEASE}/{self.checkpoint_name}"
        state = load_state_dict_from_url(url, map_location="cpu", progress=True)
        cache_path = Path(torch.hub.get_dir()) / "checkpoints" / self.checkpoint_name
        if cache_path.exists():
            digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            if digest != self.checkpoint_sha256:
                raise RuntimeError(
                    "EfficientSED checkpoint checksum mismatch for "
                    f"{self.checkpoint_name}: {digest}"
                )
        prefix = "net.model."
        encoder = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not encoder:
            raise RuntimeError(
                f"EfficientSED checkpoint {self.checkpoint_name} "
                f"has no {prefix!r} encoder"
            )
        self.load_state_dict(encoder, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if x.shape[2] != 1:
            raise RuntimeError(
                "EfficientSED fMN frequency axis did not collapse to one bin; "
                "use its exact 128-mel contract"
            )
        return x


class EfficientSEDfMN10(EfficientSEDFrameMobileNet):
    width_mult = 1.0
    checkpoint_name = "fmn10_strong.pt"
    checkpoint_sha256 = "92a5c1dc5aa21620e262420c4cc657d2f699bdfa20a748de506c354a2fe2770a"


class EfficientSEDfMN20(EfficientSEDFrameMobileNet):
    width_mult = 2.0
    checkpoint_name = "fmn20_strong.pt"
    checkpoint_sha256 = "107afa1d1735f20591ed360eb7c1dfbc4b256c738339b039b281618c0955aaf8"


__all__ = ["EfficientSEDFrameMobileNet", "EfficientSEDfMN10", "EfficientSEDfMN20"]
