"""EfficientAT backbone wrapper with custom classification head.

We try to import EfficientAT from `vendor/EfficientAT/` (added as a git
submodule of `https://github.com/fschmid56/EfficientAT`). If that's not
present, we fall back to a TIMM MobileNetV3-Large with random init — usable
for smoke tests but you should set up the submodule for real training.

To install the submodule::

    cd <repo-root>
    git submodule add https://github.com/fschmid56/EfficientAT vendor/EfficientAT
    git submodule update --init --recursive
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

from sed.config import REPO_ROOT

log = logging.getLogger(__name__)

EFFICIENTAT_PATH = REPO_ROOT / "vendor" / "EfficientAT"


@contextmanager
def _chdir(path: Path):
    """Temporarily change CWD. Used to satisfy modules that open files via
    relative paths at import time (EfficientAT's helpers/utils.py loads
    'metadata/class_labels_indices.csv' that way)."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------------
# Backbone loader
# ---------------------------------------------------------------------------


def _load_efficientat(backbone_name: str, pretrained: bool) -> nn.Module:
    """Try to load from the EfficientAT vendor checkout; raise if not present."""
    if not EFFICIENTAT_PATH.exists():
        raise ImportError(
            f"EfficientAT not found at {EFFICIENTAT_PATH}. Run:\n"
            "  git submodule add https://github.com/fschmid56/EfficientAT "
            "vendor/EfficientAT && git submodule update --init"
        )
    metadata_csv = EFFICIENTAT_PATH / "metadata" / "class_labels_indices.csv"
    if not metadata_csv.exists():
        raise ImportError(
            f"EfficientAT submodule looks incomplete (missing {metadata_csv}). "
            "Try: git submodule update --init --recursive"
        )
    if str(EFFICIENTAT_PATH) not in sys.path:
        sys.path.insert(0, str(EFFICIENTAT_PATH))

    # EfficientAT's helpers/utils.py opens 'metadata/class_labels_indices.csv'
    # with a relative path at module-import time, so we must chdir into the
    # submodule directory while the import resolves. Once cached in
    # sys.modules, subsequent imports re-use the populated module globals
    # and the chdir is no longer needed.
    with _chdir(EFFICIENTAT_PATH):
        from models.mn.model import get_model  # type: ignore

    # Pretrained model name format used by EfficientAT (mn10_as etc.)
    model = get_model(
        width_mult={"mn01": 0.1, "mn04": 0.4, "mn05": 0.5, "mn10": 1.0,
                    "mn20": 2.0, "mn30": 3.0, "mn40": 4.0}.get(
                        backbone_name.split("_")[0].rstrip("_as"), 1.0
                    ),
        pretrained_name=backbone_name if pretrained else None,
    )
    return model


def _load_timm_fallback(backbone_name: str) -> nn.Module:
    log.warning("Using TIMM MobileNetV3 fallback — accuracy will be much lower.")
    try:
        import timm
    except ImportError as e:
        raise ImportError("Install `timm` for the fallback backbone, or set up "
                          "the EfficientAT submodule.") from e
    return timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=0,
                             global_pool="")


# ---------------------------------------------------------------------------
# SED model
# ---------------------------------------------------------------------------


class SedModel(nn.Module):
    """Backbone + global pool + linear head over our classes."""

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        try:
            self.backbone = _load_efficientat(backbone_name, pretrained)
            self._is_efficientat = True
        except ImportError as e:
            log.error("%s", e)
            self.backbone = _load_timm_fallback(backbone_name)
            self._is_efficientat = False

        # Determine feature dim by a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 100)
            feat = self._extract_features(dummy)
        feat_dim = feat.shape[-1]
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, num_classes),
        )
        self.num_classes = num_classes

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T) log-mel."""
        if self._is_efficientat:
            # EfficientAT returns (logits, embeddings); we discard logits.
            out = self.backbone(x)
            if isinstance(out, tuple):
                _logits, emb = out
            else:
                emb = out
            return emb
        # TIMM path: pool spatial dims
        x3 = x.repeat(1, 3, 1, 1)
        feat = self.backbone(x3)
        return feat.mean(dim=(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._extract_features(x)
        return self.head(feat)


# ---------------------------------------------------------------------------
# Asymmetric focal loss
# ---------------------------------------------------------------------------


class AsymmetricFocalLoss(nn.Module):
    """Per-class focal loss with separate γ for the negative class.

    With γ_neg > γ_pos and an extra weight on the `_negative` class, the
    optimiser prefers to be wrong on positives (miss a clink) over being wrong
    on negatives (trigger a false light cue) — exactly the behaviour we want.
    """

    def __init__(
        self,
        num_classes: int,
        negative_index: int,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        negative_weight: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.neg_idx = negative_index
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        weights = torch.ones(num_classes)
        weights[negative_index] = negative_weight
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Multi-label: targets ∈ [0, 1]^{B×C}."""
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        prob = torch.sigmoid(logits)
        # focal modulation per class
        gammas = torch.full((self.num_classes,), self.gamma_pos,
                            device=logits.device, dtype=logits.dtype)
        gammas[self.neg_idx] = self.gamma_neg
        # for positives in target: focusing factor (1-p)^γ
        # for negatives in target: focusing factor p^γ
        pt = torch.where(targets >= 0.5, prob, 1.0 - prob)
        focal = (1 - pt).pow(gammas.unsqueeze(0))
        loss = focal * bce * self.class_weights.unsqueeze(0)
        return loss.mean()
