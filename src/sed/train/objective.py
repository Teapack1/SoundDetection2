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


SINGLE_LABEL_LOSSES = {
    "cross_entropy",
    "ce",
    "softmax_ce",
    "categorical_crossentropy",
}
BCE_LOSSES = {"bce", "bce_logits", "binary_cross_entropy"}


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


def training_metric_descriptor(cfg: Any) -> dict[str, Any]:
    """Human-facing semantics for the active training loss and live metrics."""
    objective = str(getattr(cfg.training, "pretraining_objective", "supervised") or "supervised").lower()
    task = str(getattr(cfg.model, "task", "clip") or "clip").lower()
    loss = objective_name(cfg)
    if objective == "byol":
        return {
            "objective": "byol",
            "effective_loss": "byol_ssl",
            "effective_loss_label": "BYOL SSL",
            "val_loss_label": "val BYOL loss",
            "val_loss_description": "Self-supervised BYOL cosine representation loss; lower is better.",
            "probability_mode": "embedding",
            "val_accuracy_key": "",
            "val_accuracy_label": "",
            "val_accuracy_description": "BYOL does not produce classification accuracy during representation training.",
            "val_score_key": "val/byol_score",
            "val_score_label": "BYOL score",
            "val_score_description": "Tie-break score equal to negative BYOL loss; higher is better, not accuracy.",
            "show_val_accuracy": False,
            "primary_monitor": "val/loss",
            "primary_monitor_label": "validation BYOL loss",
            "val_f1_description": "F1 is unavailable during BYOL; run a supervised fine-tune or linear probe for class metrics.",
        }
    if task == "frame":
        return {
            "objective": "supervised",
            "effective_loss": "frame_bce_logits",
            "effective_loss_label": "Frame BCE",
            "val_loss_label": "val frame BCE",
            "val_loss_description": "BCE-with-logits over frame targets; clip scores are pooled for live AP/F1/accuracy.",
            "probability_mode": "sigmoid",
            "val_accuracy_key": "val/accuracy",
            "val_accuracy_label": "val acc",
            "val_accuracy_description": "Per-label clip-presence correctness at threshold 0.5 after pooling frame logits.",
            "val_score_key": "val/accuracy",
            "val_score_label": "val acc",
            "show_val_accuracy": True,
            "primary_monitor": "val/ap_macro_targets",
            "primary_monitor_label": "target AP",
            "val_f1_description": "Macro F1 over target classes, excluding _negative, at threshold 0.5.",
        }
    if loss in SINGLE_LABEL_LOSSES:
        return {
            "objective": "supervised",
            "effective_loss": "cross_entropy",
            "effective_loss_label": "CE",
            "val_loss_label": "val CE",
            "val_loss_description": "Cross-entropy over one class index per clip; lower is better.",
            "probability_mode": "softmax",
            "val_accuracy_key": "val/accuracy",
            "val_accuracy_label": "val acc",
            "val_accuracy_description": "Closed-set argmax class accuracy on the validation split.",
            "val_score_key": "val/accuracy",
            "val_score_label": "val acc",
            "show_val_accuracy": True,
            "primary_monitor": "val/ap_macro_targets",
            "primary_monitor_label": "target AP",
            "val_f1_description": "Macro F1 over target classes, excluding _negative.",
        }
    if loss == "asymmetric_focal":
        return {
            "objective": "supervised",
            "effective_loss": "asymmetric_focal",
            "effective_loss_label": "Asym focal",
            "val_loss_label": "val focal",
            "val_loss_description": "Asymmetric focal loss over independent sigmoid labels; lower is better.",
            "probability_mode": "sigmoid",
            "val_accuracy_key": "val/accuracy",
            "val_accuracy_label": "val acc",
            "val_accuracy_description": "Per-label correctness at threshold 0.5; can be high when negative labels dominate.",
            "val_score_key": "val/accuracy",
            "val_score_label": "val acc",
            "show_val_accuracy": True,
            "primary_monitor": "val/ap_macro_targets",
            "primary_monitor_label": "target AP",
            "val_f1_description": "Macro F1 over target classes, excluding _negative, at threshold 0.5.",
        }
    return {
        "objective": "supervised",
        "effective_loss": "bce_logits",
        "effective_loss_label": "BCE",
        "val_loss_label": "val BCE",
        "val_loss_description": "BCE-with-logits over independent sigmoid labels; lower is better.",
        "probability_mode": "sigmoid",
        "val_accuracy_key": "val/accuracy",
        "val_accuracy_label": "val acc",
        "val_accuracy_description": "Per-label correctness at threshold 0.5; can be high when negative labels dominate.",
        "val_score_key": "val/accuracy",
        "val_score_label": "val acc",
        "show_val_accuracy": True,
        "primary_monitor": "val/ap_macro_targets",
        "primary_monitor_label": "target AP",
        "val_f1_description": "Macro F1 over target classes, excluding _negative, at threshold 0.5.",
    }


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
