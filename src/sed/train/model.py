"""SED model: a pluggable backbone + global pool + linear head.

The backbone is built by a :mod:`sed.train.backbones` adapter, so the trainer
supports EfficientAT (the vendored ``mn*`` AudioSet models), any ``timm``
architecture (ConvNeXt-Tiny, MobileNetV3, …), and user ``module:Class`` nets —
selected by ``cfg.model.backbone``.

EfficientAT is loaded from `vendor/EfficientAT/`, a git submodule of
`https://github.com/fschmid56/EfficientAT`::

    cd <repo-root>
    git submodule add https://github.com/fschmid56/EfficientAT vendor/EfficientAT
    git submodule update --init --recursive
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from sed.train.backbones import (
    EFFICIENTAT_PATH,
    _load_efficientat,  # re-exported for back-compat (tests import it from here)
    resolve_adapter,
)

log = logging.getLogger(__name__)

__all__ = ["EFFICIENTAT_PATH", "_load_efficientat", "SedModel", "AsymmetricFocalLoss"]


# ---------------------------------------------------------------------------
# SED model
# ---------------------------------------------------------------------------


class SedModel(nn.Module):
    """Backbone (via adapter) + global pool + linear head over our classes."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        n_bands: int = 128,
        n_frames: int = 100,
        params: dict | None = None,
        task: str = "clip",
        input_channels: int | None = None,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.adapter = resolve_adapter(backbone_name)
        self.task = str(task or "clip")
        self.input_channels = int(
            input_channels if input_channels is not None
            else getattr(self.adapter.spec, "input_channels", 1) or 1
        )
        self.backbone = self.adapter.build_backbone(
            pretrained=pretrained,
            n_bands=n_bands,
            n_frames=n_frames,
            input_channels=self.input_channels,
            **(params or {}),
        )

        # Determine feature dim by a dummy forward. Backbones are fully
        # convolutional with global pooling, so feat_dim is independent of the
        # band count (n_bands) — any feature height (mel n_mels or linear bins)
        # works without an architecture change. The probe runs in eval mode so a
        # single zeros batch neither pollutes BatchNorm running stats nor trips
        # BatchNorm's "expected >1 value per channel" guard for encoders whose
        # embedding ends in a BatchNorm1d (e.g. the legacy SmallerVGGNet).
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, self.input_channels, n_bands, n_frames)
                feat = self._extract_features(dummy)
        finally:
            self.backbone.train(was_training)
        feat_dim = feat.shape[-1]
        # Head dropout is a training knob (``training.head_dropout``), not part
        # of the model recipe, and is an identity at eval/export.
        self.head = nn.Sequential(
            nn.Dropout(float(head_dropout)),
            nn.Linear(feat_dim, num_classes),
        )
        self.num_classes = num_classes

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, n_bands, T) feature map → (B, feat_dim) embedding."""
        return self.adapter.forward_features(self.backbone, x)

    def forward_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Frame logits ``(B, T, C)`` for frame-capable SED recipes."""
        feat = self.adapter.forward_frame_features(self.backbone, x).transpose(1, 2)
        return self.head(feat)

    def backbone_stages(self) -> list[tuple[str, list[nn.Parameter]]]:
        """Ordered stem→…→last param groups of the backbone (for per-stage LR)."""
        return self.adapter.stages(self.backbone)

    def backbone_stage_modules(self) -> list[tuple[str, nn.Module]]:
        """Ordered stem→…→last stage modules of the backbone (for freezing/eval)."""
        return self.adapter.stage_modules(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task == "frame":
            # Clip-level reports keep working for frame experiments: a target
            # is present if any frame fires. Timestamp scoring uses
            # ``forward_frames`` directly.
            return self.forward_frames(x).amax(dim=1)
        feat = self._extract_features(x)
        return self.head(feat)


# ---------------------------------------------------------------------------
# Asymmetric focal loss
# ---------------------------------------------------------------------------


class AsymmetricFocalLoss(nn.Module):
    """Per-class focal loss with separate γ for the negative class.

    γ controls hard-example focusing, not class-frequency balance. Frequency
    imbalance is handled once by the shared training sampler. Defaults mirror
    ``TrainingCfg.focal_gamma_*``.
    """

    def __init__(
        self,
        num_classes: int,
        negative_index: int | None,
        gamma_pos: float = 1.0,
        gamma_neg: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.neg_idx = negative_index
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        # Retain an all-one buffer for compatibility with historical focal-loss
        # checkpoints, which persisted this state under the same key.
        self.register_buffer("class_weights", torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Multi-label: targets ∈ [0, 1]^{B×C}."""
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        prob = torch.sigmoid(logits)
        # focal modulation per class
        gammas = torch.full((self.num_classes,), self.gamma_pos,
                            device=logits.device, dtype=logits.dtype)
        if self.neg_idx is not None:
            gammas[self.neg_idx] = self.gamma_neg
        # for positives in target: focusing factor (1-p)^γ
        # for negatives in target: focusing factor p^γ
        pt = torch.where(targets >= 0.5, prob, 1.0 - prob)
        focal = (1 - pt).pow(gammas.unsqueeze(0))
        loss = focal * bce * self.class_weights.unsqueeze(0)
        return loss.mean()
