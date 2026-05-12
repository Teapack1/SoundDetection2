"""ClinkCNN-v2: compact, frequency-aware encoder for 0.4 s glass impacts.

The encoder consumes one mono spectrogram contract at a time.  The native
contract is ``128 x 81`` log-mel at 44.1 kHz, while the same learned topology
can be instantiated for validated higher-resolution mel or linear-frequency
contracts.  It derives an onset plane and a frequency-coordinate plane inside
the model, pools time before mixing and pyramidal pooling over frequency, and
returns a 512-dimensional clip embedding.

Unlike :mod:`sed.train.clink_cnn`, this implementation does not use activation
recomputation.  Its depthwise spatial mixing and 2x pointwise expansion keep it
small enough to train directly on memory-constrained GPUs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_BANDS = 128
N_FRAMES = 81
EMBEDDING_DIM = 512
MIN_BANDS = 32
MAX_BANDS = 1024
MIN_FRAMES = 25
MAX_FRAMES = 201


class DropPath(nn.Module):
    """Per-sample stochastic depth (an identity during evaluation/export)."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability <= 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class ChannelFirstGlobalResponseNorm(nn.Module):
    """ConvNeXt-v2 global response normalization for ``(B, C, F, T)``."""

    def __init__(self, channels: int):
        super().__init__()
        channels = int(channels)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(x, ord=2, dim=(2, 3), keepdim=True)
        response = response / (response.mean(dim=1, keepdim=True) + 1e-6)
        return x + self.gamma * (x * response) + self.beta


class GlassBlock(nn.Module):
    """Axis-separable residual block with a channels-first 2x MLP."""

    def __init__(self, channels: int, *, drop_path: float):
        super().__init__()
        channels = int(channels)
        expanded = 2 * channels
        self.frequency_branch = nn.Conv2d(
            channels,
            channels,
            kernel_size=(7, 1),
            padding=(3, 0),
            groups=channels,
            bias=False,
        )
        self.temporal_branch = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 7),
            padding=(0, 3),
            groups=channels,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.expand = nn.Conv2d(channels, expanded, kernel_size=1, bias=False)
        self.activation = nn.SiLU()
        self.grn = ChannelFirstGlobalResponseNorm(expanded)
        self.project = nn.Conv2d(expanded, channels, kernel_size=1, bias=False)
        # CaiT-style layer scale. The ConvNeXt 1e-6 default is tuned for 30+
        # block stacks; at this depth it leaves every residual branch
        # contributing ~1e-7 of the trunk signal (and receiving ~1e-6 gradient)
        # for the first epochs, so the trunk effectively starts as stem +
        # downsamples + pool. CaiT uses 0.1 for shallow stacks.
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = 0.5 * (self.frequency_branch(x) + self.temporal_branch(x))
        x = self.norm(x)
        x = self.expand(x)
        x = self.grn(self.activation(x))
        x = self.project(x) * self.layer_scale
        return residual + self.drop_path(x)


class DepthwisePointwiseDownsample(nn.Module):
    """Spatial downsample followed by a cheap channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.pointwise(self.depthwise(x))))


class PointwiseProjection(nn.Module):
    """Channel projection for stages that do not change spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.pointwise(x)))


def _stage_entry(in_channels: int, out_channels: int,
                 stride: tuple[int, int]) -> nn.Module:
    """Downsample when the stage strides, otherwise just project channels."""
    if tuple(stride) == (1, 1):
        return PointwiseProjection(in_channels, out_channels)
    return DepthwisePointwiseDownsample(in_channels, out_channels, stride)


class _ResidualGroup(nn.Sequential):
    """One independently-freezable block, including a preceding downsample."""


class FrequencyAwarePool(nn.Module):
    """Pool time per frequency row, then mix and pyramid-pool frequency.

    Temporal attention is normalized independently for every frequency row.
    Weighted mean and spread retain the decay envelope, while temporal maxima
    retain brief attacks.  A pointwise-plus-depthwise Conv1d mixer consumes the
    resulting row descriptors.  Average-pooling to 1, 2, and 4 frequency bins
    contributes ``7 * mixer`` values and global frequency max contributes
    ``mixer`` more, giving the fixed embedding without a large dense head.

    The mixer width is the whole information path from trunk to embedding, so
    it sets how much detail can transfer into a fine-tune.  64 channels keep the
    ``3 * C`` row descriptors at roughly 4.5x compression; the previous 32 threw
    away 12x and left too little for fine-grained glass-type distinctions.
    """

    pyramid_bins: tuple[int, ...] = (1, 2, 4)

    def __init__(
        self,
        channels: int = 96,
        *,
        dropout: float = 0.3,
        frequency_rows: int | None = None,
    ):
        super().__init__()
        channels = int(channels)
        self.frequency_rows = (
            int(frequency_rows) if frequency_rows is not None else None
        )
        attention_channels = 32
        mixer_channels = 64
        self.attention = nn.Sequential(
            nn.Conv2d(channels, attention_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(attention_channels, 1, kernel_size=1),
        )
        self.frequency_mixer = nn.Sequential(
            nn.Conv1d(3 * channels, mixer_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mixer_channels),
            nn.SiLU(),
            nn.Conv1d(
                mixer_channels,
                mixer_channels,
                kernel_size=3,
                padding=1,
                groups=mixer_channels,
                bias=False,
            ),
            nn.BatchNorm1d(mixer_channels),
            nn.SiLU(),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(EMBEDDING_DIM),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 1, F, T)`` weights summing to one along time."""
        return torch.softmax(self.attention(x), dim=-1)

    def temporal_statistics(self, x: torch.Tensor) -> torch.Tensor:
        """Return mean/spread/max descriptors ``(B, 3*C, F)``."""
        weights = self.attention_weights(x)
        mean = (weights * x).sum(dim=-1)
        centered = x - mean.unsqueeze(-1)
        spread = torch.sqrt((weights * centered.square()).sum(dim=-1) + 1e-5)
        maximum = x.amax(dim=-1)
        return torch.cat((mean, spread, maximum), dim=1)

    @staticmethod
    def _fixed_adaptive_average(
        x: torch.Tensor, bins: int, length: int | None = None,
    ) -> torch.Tensor:
        """Exportable equivalent of adaptive average pooling for a static axis.

        PyTorch's legacy ONNX exporter rejects ``adaptive_avg_pool1d`` when
        the input length is not divisible by the requested output (notably 58
        rows -> 4 bins for the FFT1024 linear profile).  Every exported bundle
        deliberately has a static frequency shape, so spelling out PyTorch's
        floor/ceil bin boundaries preserves the exact operation and exports as
        ordinary slices plus reductions.
        """
        length = int(length) if length is not None else int(x.shape[-1])
        outputs = []
        for index in range(int(bins)):
            start = (index * length) // int(bins)
            end = ((index + 1) * length + int(bins) - 1) // int(bins)
            outputs.append(x[..., start:end].mean(dim=-1))
        return torch.stack(outputs, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.frequency_mixer(self.temporal_statistics(x))
        pyramid = [
            self._fixed_adaptive_average(
                mixed, bins, length=self.frequency_rows,
            ).flatten(1)
            for bins in self.pyramid_bins
        ]
        # 64 * (1 + 2 + 4) + 64 = 512.  The extra max summary makes narrow,
        # high-Q resonance peaks survive average pooling.
        pyramid.append(mixed.amax(dim=-1).flatten(1))
        return self.projection(torch.cat(pyramid, dim=1))


# (channels, number of blocks, frequency stride, time stride).  Applied after
# the stride-(1, 2) stem, this yields 128x41 -> 64x41 -> 32x21 -> 16x11 -> 16x11.
#
# The final stage keeps 96 channels rather than widening to 128: it does not
# downsample, so the extra width bought little, and a 128-channel block held 23%
# of all parameters on its own, which unbalanced the last-N fine-tune policy.
_LAYOUT: tuple[tuple[int, int, int, int], ...] = (
    (32, 1, 2, 1),
    (64, 2, 2, 2),
    (96, 2, 2, 2),
    (96, 2, 1, 1),
)


class ClinkCNNv2(nn.Module):
    """One fixed-topology ClinkCNN-v2 encoder for a resolved input geometry."""

    def __init__(
        self,
        in_chans: int = 1,
        n_bands: int = N_BANDS,
        n_frames: int = N_FRAMES,
        pretrained: bool = False,
    ):
        super().__init__()
        if pretrained:
            raise ValueError(
                "ClinkCNN-v2 has no upstream weights; use random or parent initialization"
            )
        if int(in_chans) != 1:
            raise ValueError("ClinkCNN-v2 expects one mono log-spectrogram feature channel")
        self.n_bands = int(n_bands)
        self.n_frames = int(n_frames)
        if not MIN_BANDS <= self.n_bands <= MAX_BANDS:
            raise ValueError(
                f"ClinkCNN-v2 expects {MIN_BANDS}..{MAX_BANDS} frequency rows; "
                f"got {self.n_bands}"
            )
        if not MIN_FRAMES <= self.n_frames <= MAX_FRAMES:
            raise ValueError(
                f"ClinkCNN-v2 expects {MIN_FRAMES}..{MAX_FRAMES} time frames; "
                f"got {self.n_frames}"
            )

        coordinate = torch.linspace(-1.0, 1.0, self.n_bands).reshape(
            1, 1, self.n_bands, 1,
        )
        self.register_buffer("frequency_coordinate", coordinate, persistent=False)

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=(7, 3),
                stride=(1, 2),
                padding=(3, 1),
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )

        total_blocks = sum(depth for _, depth, _, _ in _LAYOUT)
        drop_rates = torch.linspace(0.0, 0.1, total_blocks).tolist()
        rate_index = 0
        previous = 32
        stages: list[nn.Module] = []
        for channels, depth, f_stride, t_stride in _LAYOUT:
            groups: list[nn.Module] = []
            for block_index in range(depth):
                modules: list[nn.Module] = []
                if block_index == 0:
                    modules.append(_stage_entry(
                        previous, channels, (f_stride, t_stride),
                    ))
                modules.append(GlassBlock(channels, drop_path=drop_rates[rate_index]))
                rate_index += 1
                groups.append(_ResidualGroup(*modules))
            stages.append(nn.Sequential(*groups))
            previous = channels
        self.stages = nn.ModuleList(stages)

        # Three stride-2 frequency reductions use SAME padding, so the static
        # final height is ceil(n_bands / 8). Persist it in the pool to keep the
        # ONNX graph's frequency pyramid entirely constant-foldable.
        final_frequency_rows = (self.n_bands + 7) // 8
        self.pool = FrequencyAwarePool(
            previous,
            dropout=0.3,
            frequency_rows=final_frequency_rows,
        )
        self.out_channels = EMBEDDING_DIM

    @staticmethod
    def positive_delta(x: torch.Tensor) -> torch.Tensor:
        first = torch.zeros_like(x[..., :1])
        change = torch.relu(x[..., 1:] - x[..., :-1])
        return torch.cat((first, change), dim=-1)

    def input_planes(self, x: torch.Tensor) -> torch.Tensor:
        """Create raw, positive-delta, and normalized-frequency planes."""
        # Every bundle has one fixed spatial feature contract. Skip the
        # Python-side guard only while tracing so ONNX can retain its dynamic
        # batch dimension; frequency/time remain static for that bundle.
        tracing = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
        expected = (1, self.n_bands, self.n_frames)
        if not tracing and (x.ndim != 4 or tuple(x.shape[1:]) != expected):
            got = "x".join(str(int(v)) for v in x.shape[1:]) if x.ndim == 4 else str(tuple(x.shape))
            raise ValueError(
                "ClinkCNN-v2 input must have shape "
                f"(B,1,{self.n_bands},{self.n_frames}); got {got}"
            )
        coordinate = self.frequency_coordinate.to(dtype=x.dtype).expand(
            x.shape[0], -1, -1, x.shape[-1],
        )
        return torch.cat((x, self.positive_delta(x), coordinate), dim=1)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(self.input_planes(x))
        for stage in self.stages:
            x = stage(x)
        return x

    def fine_tune_stages(self) -> list[tuple[str, nn.Module]]:
        """Nine ordered, non-overlapping groups for transfer learning."""
        groups: list[tuple[str, nn.Module]] = [("stem", self.stem)]
        for stage_index, stage in enumerate(self.stages):
            groups.extend(
                (f"stage{stage_index}.block{block_index}", block)
                for block_index, block in enumerate(stage)
            )
        groups.append(("pool", self.pool))
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.forward_map(x))


__all__ = [
    "EMBEDDING_DIM",
    "MAX_BANDS",
    "MAX_FRAMES",
    "MIN_BANDS",
    "MIN_FRAMES",
    "N_BANDS",
    "N_FRAMES",
    "ChannelFirstGlobalResponseNorm",
    "ClinkCNNv2",
    "DepthwisePointwiseDownsample",
    "DropPath",
    "FrequencyAwarePool",
    "GlassBlock",
    "PointwiseProjection",
]
