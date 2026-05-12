"""Training objective helpers shared by training, export, and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - edge runtime can be torch-free
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


SINGLE_LABEL_LOSSES = {"cross_entropy", "ce", "softmax_ce"}
LEGACY_BCE_LOSSES = {"bce", "bce_logits", "binary_cross_entropy"}


def is_single_label_loss(loss_or_cfg: Any) -> bool:
    """Return whether the configured objective expects one class index per row."""
    if hasattr(loss_or_cfg, "training"):
        loss = getattr(loss_or_cfg.training, "loss", "")
    else:
        loss = loss_or_cfg
    return str(loss).strip().lower() in SINGLE_LABEL_LOSSES


def probability_mode(loss_or_cfg: Any) -> str:
    return "softmax" if is_single_label_loss(loss_or_cfg) else "sigmoid"


def objective_name(loss_or_cfg: Any) -> str:
    if hasattr(loss_or_cfg, "training"):
        loss = getattr(loss_or_cfg.training, "loss", "")
    else:
        loss = loss_or_cfg
    return str(loss).strip().lower()


def logits_to_probabilities(logits, loss_or_cfg: Any):
    """Convert logits to probabilities according to the training objective."""
    if _HAS_TORCH and isinstance(logits, torch.Tensor):
        if is_single_label_loss(loss_or_cfg):
            return torch.softmax(logits, dim=1)
        return torch.sigmoid(logits)
    arr = np.asarray(logits)
    if is_single_label_loss(loss_or_cfg):
        shifted = arr - np.max(arr, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)
    return 1.0 / (1.0 + np.exp(-arr))


def targets_to_indices(targets):
    """Normalize batch targets to integer class indices."""
    if not _HAS_TORCH:
        raise RuntimeError("targets_to_indices requires torch")
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets)
    if targets.ndim == 1:
        return targets.long()
    return torch.argmax(targets, dim=1).long()


def targets_to_one_hot_np(targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Normalize numpy targets to a float32 one-hot matrix."""
    arr = np.asarray(targets)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    idx = arr.astype(np.int64)
    out = np.zeros((idx.shape[0], num_classes), dtype=np.float32)
    if idx.shape[0]:
        out[np.arange(idx.shape[0]), idx] = 1.0
    return out
