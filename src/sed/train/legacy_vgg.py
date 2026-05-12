"""Exact port of the legacy soundDetection Keras SmallerVGGNet.

The legacy system classified spectrogram *images*: a scipy PSD spectrogram was
dB-normalized to [0, 1], rendered through OpenCV's JET colormap into a 3-channel
image, transposed (time as rows), resized to 96×96, scaled to [0, 1], and fed to
a small VGG-style Keras net (pyimagesearch SmallerVGGNet).

In SoundDetection2 the colormap + transpose + resize now live in the **audio
feature** (``FeaturesCfg.colormap`` / ``image_size`` — see
``sed.audio.features.render_spectrogram_image``), so the feature *is* the
3-channel image the model trains and infers on, and the preview shows it. This
module is therefore just the SmallerVGGNet convolutional classifier that consumes
that image — a faithful 1:1 port of the Keras layers (same order Conv→ReLU→BN,
pool sizes, dropout rates, and the 1024-unit dense embedding). The trainer's
linear head reproduces the final softmax Dense (train with
``loss = "categorical_crossentropy"`` or ``"cross_entropy"`` for the legacy
objective).

It follows the ``CustomAdapter`` encoder contract: ``forward`` returns the
``(B, dense_units)`` embedding; ``forward_features`` exposes the pre-flatten conv
map for the Model tab's shape probes. The dense layer is sized from the input
``n_bands``/``n_frames`` (the image height/width) supplied by the adapter, so it
adapts to the configured ``image_size`` without a hardcoded shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallerVGGNet(nn.Module):
    """Legacy SmallerVGGNet classifier, ``(B, C, H, W)`` image → ``(B, dense_units)``.

    BatchNorm uses the Keras defaults (eps=1e-3, momentum 0.99 ⇒ torch 0.01) and
    every block keeps the legacy Conv → ReLU → BatchNorm order. The three pools
    (3×3 then 2×2 then 2×2, valid padding) reduce each spatial axis by 12×, so a
    96×96 image yields the legacy 128×8×8 = 8192-d flatten → Dense(1024).
    """

    def __init__(
        self,
        in_chans: int = 3,
        n_bands: int = 96,
        n_frames: int = 96,
        dense_units: int = 1024,
        conv_dropout: float = 0.25,
        dense_dropout: float = 0.5,
        pretrained: bool = False,  # no published weights; accepted per contract
        **_ignored,
    ):
        super().__init__()
        _ = pretrained
        self.in_chans = int(in_chans)

        def bn(ch: int) -> nn.BatchNorm2d:
            return nn.BatchNorm2d(ch, eps=1e-3, momentum=0.01)

        self.features = nn.Sequential(
            # CONV => RELU => POOL
            nn.Conv2d(self.in_chans, 32, 3, padding=1), nn.ReLU(inplace=True), bn(32),
            nn.MaxPool2d(3), nn.Dropout(conv_dropout),
            # (CONV => RELU) * 2 => POOL
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), bn(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), bn(64),
            nn.MaxPool2d(2), nn.Dropout(conv_dropout),
            # (CONV => RELU) * 2 => POOL
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), bn(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), bn(128),
            nn.MaxPool2d(2), nn.Dropout(conv_dropout),
        )
        # Valid pooling by 3, 2, 2 ⇒ each axis // 12 (floor). Sized from the input.
        h = int(n_bands) // 3 // 2 // 2
        w = int(n_frames) // 3 // 2 // 2
        if h < 1 or w < 1:
            raise ValueError(
                f"input {n_bands}×{n_frames} is too small for the three pools "
                f"(needs ≥12 on each axis); got feature map {h}×{w}"
            )
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * h * w, int(dense_units)),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(int(dense_units), eps=1e-3, momentum=0.01),
            nn.Dropout(dense_dropout),
        )
        self.out_channels = int(dense_units)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-flatten conv map (B, 128, h, w) — for shape probes/feature maps."""
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W) image input, got shape {tuple(x.shape)}")
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.forward_features(x))


__all__ = ["SmallerVGGNet"]
