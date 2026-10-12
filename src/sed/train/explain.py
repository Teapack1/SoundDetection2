"""Prediction evidence localization for review-set evaluation.

The outputs here are intentionally framed as model evidence, not ground-truth
event localization. Clip classifiers can only tell us which input regions most
influenced a class score; frame models additionally expose class-vs-time logits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sed import config as config_mod
from sed.audio.features import kaldi_mel_center_freqs
from sed.config import Config, ProjectLayout
from sed.train.datamodule import SedDataset
from sed.train.lit_module import SedLitModule
from sed.train.objective import logits_to_probabilities, targets_to_one_hot_np

log = logging.getLogger(__name__)


EXPLANATION_VERSION = 1
VALID_SCOPES = {"frozen", "mined"}


@dataclass(frozen=True)
class ReviewSet:
    """Rows to score/explain plus the dataset root used for feature loading."""

    scope: str
    csv_path: Path
    dataset_dir: Path
    rows: list[dict[str, str]]
    temporary_csv: bool = False


def _safe_name(value: str) -> str:
    value = str(value or "").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in value)


def clip_key(path: str | Path) -> str:
    """Stable key used for per-clip explanation artifacts."""

    try:
        raw = str(Path(path).resolve())
    except Exception:  # noqa: BLE001
        raw = str(path)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def explanation_report_dir(model_dir: Path, scope: str) -> Path:
    scope = _normalize_scope(scope)
    return model_dir / "reports" / "explanations" / scope


def explanation_artifact_path(model_dir: Path, scope: str, path: str | Path, class_name: str) -> Path:
    return (
        explanation_report_dir(model_dir, scope)
        / "clips"
        / f"{clip_key(path)}--{_safe_name(class_name)}.json"
    )


def _normalize_scope(scope: str) -> str:
    value = str(scope or "").strip().lower()
    if value not in VALID_SCOPES:
        raise ValueError(f"scope must be one of {sorted(VALID_SCOPES)}, got {scope!r}")
    return value


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _load_sidecar(path: Path) -> dict[str, Any]:
    side = path.with_suffix(".json")
    if not side.exists():
        return {}
    try:
        with open(side, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.debug("failed to read sidecar %s: %s", side, exc)
        return {}


def _sidecar_probabilities(meta: dict[str, Any], classes: list[str]) -> dict[str, float]:
    for key in ("smoothed", "probs"):
        values = meta.get(key)
        if isinstance(values, dict):
            out: dict[str, float] = {}
            for cls in classes:
                try:
                    out[cls] = float(values.get(cls, 0.0))
                except (TypeError, ValueError):
                    out[cls] = 0.0
            return out
    for key in ("smoothed_percent", "probs_percent"):
        values = meta.get(key)
        if isinstance(values, dict):
            out = {}
            for cls in classes:
                try:
                    out[cls] = float(values.get(cls, 0.0)) / 100.0
                except (TypeError, ValueError):
                    out[cls] = 0.0
            return out
    return {}


def build_review_set(
    cfg: Config,
    project_id: str,
    model_id: str,
    scope: str,
    *,
    classes: list[str] | None = None,
    output_dir: str = "mined",
    work_dir: Path | None = None,
) -> ReviewSet:
    """Build the review rows shared by scoring and explanation.

    Frozen rows are the existing composed ``test.csv``. Mined rows are runtime
    mined clips under ``data_library/mined`` (or ``output_dir``), with the
    predicted class from the sidecar/class folder used as the weak review label.
    """

    _ = (cfg, model_id)
    scope = _normalize_scope(scope)
    layout = ProjectLayout(project_id)
    if scope == "frozen":
        csv_path = layout.dataset_dir / "test.csv"
        return ReviewSet(
            scope=scope,
            csv_path=csv_path,
            dataset_dir=layout.dataset_dir,
            rows=_read_csv_rows(csv_path),
            temporary_csv=False,
        )

    if not classes:
        raise ValueError("classes are required when building the mined review set")
    root = config_mod.LIBRARY_DIR / output_dir
    rows: list[dict[str, str]] = []
    if root.exists():
        for wav in sorted(root.rglob("*.wav")):
            meta = _load_sidecar(wav)
            try:
                rel_parts = wav.relative_to(root).parts
            except ValueError:
                rel_parts = ()
            folder_label = rel_parts[0] if rel_parts else wav.parent.name
            predicted = str(meta.get("predicted_class") or folder_label or "").strip()
            class_name = predicted if predicted in classes else (
                "_negative" if "_negative" in classes else classes[0]
            )
            probs = _sidecar_probabilities(meta, classes)
            row: dict[str, str] = {
                "path": str(wav),
                "class_name": class_name,
                "review_label": predicted or class_name,
                "predicted_class": predicted or class_name,
                "origin": "mined",
                "session_id": str(meta.get("session_id") or ""),
                "kind": str(meta.get("kind") or ""),
                "created_at": str(meta.get("created_at") or ""),
                "sidecar_confidence": str(float(meta.get("confidence") or 0.0)),
                "best_channel": str(meta.get("best_channel") or ""),
                "has_sidecar_probabilities": "true" if probs else "false",
            }
            for cls, prob in probs.items():
                row[f"sidecar_p:{cls}"] = f"{prob:.6f}"
            rows.append(row)

    base = Path(work_dir) if work_dir is not None else (
        ProjectLayout(project_id).model_dir(model_id)
        / "reports"
        / "explanations"
        / "mined"
    )
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / "mined_review.csv"
    fieldnames = [
        "path", "class_name", "review_label", "predicted_class", "origin",
        "session_id", "kind", "created_at", "sidecar_confidence",
        "best_channel", "has_sidecar_probabilities",
    ] + [f"sidecar_p:{cls}" for cls in classes]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return ReviewSet(
        scope=scope,
        csv_path=csv_path,
        dataset_dir=csv_path.parent,
        rows=rows,
        temporary_csv=True,
    )


def _load_lit_module(
    cfg: Config,
    project_id: str,
    model_id: str,
) -> tuple[SedLitModule, Config, list[str], Path, Path]:
    from sed.train.eval import _load_model_classes, _model_config, _overlay_runtime_settings
    from sed.train.registry import get_checkpoint

    layout = ProjectLayout(project_id)
    model_dir = layout.model_dir(model_id)
    run_cfg = _model_config(model_dir, cfg)
    run_cfg, _source = _overlay_runtime_settings(model_dir, run_cfg, enabled=True)
    classes = _load_model_classes(model_dir)
    ckpt_path = get_checkpoint(project_id, model_id)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"You are using `torch.load` with `weights_only=False`.*",
        )
        lit = SedLitModule.load_from_checkpoint(
            str(ckpt_path), cfg=run_cfg, classes=classes, map_location="cpu"
        )
    lit.eval()
    return lit, run_cfg, classes, model_dir, ckpt_path


def _linear_head(model: nn.Module) -> nn.Linear | None:
    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        return head
    if isinstance(head, nn.Sequential):
        for module in reversed(head):
            if isinstance(module, nn.Linear):
                return module
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return module
    return None


def class_activation_map(
    feature_map: torch.Tensor,
    classifier_weight: torch.Tensor,
    class_index: int,
    *,
    classifier_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact CAM for GAP + linear classifiers.

    Args:
        feature_map: ``(B, C, H, W)`` pre-pool feature map.
        classifier_weight: linear classifier weights ``(num_classes, C)``.
        class_index: target class index.
    Returns:
        ``(B, H, W)`` class activation map before resizing/normalization.
    """

    if feature_map.ndim != 4:
        raise ValueError(f"feature_map must be 4D, got shape {tuple(feature_map.shape)}")
    if classifier_weight.ndim != 2:
        raise ValueError("classifier_weight must be 2D")
    class_index = int(class_index)
    if class_index < 0 or class_index >= int(classifier_weight.shape[0]):
        raise IndexError(f"class_index {class_index} outside classifier weight")
    if int(feature_map.shape[1]) != int(classifier_weight.shape[1]):
        raise ValueError(
            "feature channels do not match classifier weights: "
            f"{feature_map.shape[1]} vs {classifier_weight.shape[1]}"
        )
    weights = classifier_weight[class_index].view(1, -1, 1, 1).to(feature_map.device)
    cam = torch.sum(feature_map * weights, dim=1)
    if classifier_bias is not None and class_index < int(classifier_bias.shape[0]):
        cam = cam + classifier_bias[class_index].to(feature_map.device)
    return cam


def _exact_cam(lit: SedLitModule, x: torch.Tensor, class_index: int) -> torch.Tensor | None:
    model = lit.model
    adapter = getattr(model, "adapter", None)
    backbone = getattr(model, "backbone", None)
    head = _linear_head(model)
    if adapter is None or backbone is None or head is None:
        return None
    feature_map = None
    with torch.no_grad():
        try:
            if hasattr(adapter, "forward_feature_map"):
                feature_map = adapter.forward_feature_map(backbone, x)
            elif hasattr(adapter, "_feature_map"):
                feature_map = adapter._feature_map(backbone, x)  # noqa: SLF001 - adapter hook.
        except Exception as exc:  # noqa: BLE001
            log.debug("exact CAM feature map unavailable: %s", exc)
            feature_map = None
    if not isinstance(feature_map, torch.Tensor) or feature_map.ndim != 4:
        return None
    if int(feature_map.shape[1]) != int(head.weight.shape[1]):
        return None
    with torch.no_grad():
        cam = class_activation_map(feature_map, head.weight, class_index, classifier_bias=head.bias)
        return _resize_to_input(cam, x)


def _last_conv_module(model: nn.Module) -> nn.Module | None:
    conv: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            conv = module
    return conv


def _grad_cam(lit: SedLitModule, x: torch.Tensor, class_index: int) -> torch.Tensor | None:
    target = _last_conv_module(lit.model)
    if target is None:
        return None
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            activations.append(output)

    def bwd_hook(_module, _grad_inputs, grad_output):
        if grad_output and isinstance(grad_output[0], torch.Tensor):
            gradients.append(grad_output[0])

    fwd = target.register_forward_hook(fwd_hook)
    bwd = target.register_full_backward_hook(bwd_hook)
    try:
        lit.zero_grad(set_to_none=True)
        logits = lit(x)
        score = logits[0, int(class_index)]
        score.backward(retain_graph=False)
        if not activations or not gradients:
            return None
        act = activations[-1]
        grad = gradients[-1]
        if act.ndim == 3:
            act = act.unsqueeze(2)
            grad = grad.unsqueeze(2)
        if act.ndim != 4 or grad.ndim != 4:
            return None
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1)
        return _resize_to_input(cam, x).detach()
    except Exception as exc:  # noqa: BLE001
        log.debug("grad-cam failed: %s", exc)
        return None
    finally:
        fwd.remove()
        bwd.remove()
        lit.zero_grad(set_to_none=True)


def _occlusion_heatmap(
    lit: SedLitModule,
    x: torch.Tensor,
    class_index: int,
    *,
    max_steps_per_axis: int = 12,
) -> torch.Tensor:
    """Black-box fallback: positive score drop after patch replacement."""

    with torch.no_grad():
        base_score = lit(x)[0, int(class_index)].detach()
    _, _channels, height, width = x.shape
    patch_h = max(1, math.ceil(height / max_steps_per_axis))
    patch_w = max(1, math.ceil(width / max_steps_per_axis))
    stride_h = patch_h
    stride_w = patch_w
    baseline = torch.zeros_like(x)
    heat = torch.zeros((1, height, width), dtype=x.dtype, device=x.device)
    counts = torch.zeros_like(heat)
    with torch.no_grad():
        for top in range(0, height, stride_h):
            bottom = min(height, top + patch_h)
            for left in range(0, width, stride_w):
                right = min(width, left + patch_w)
                masked = x.clone()
                masked[:, :, top:bottom, left:right] = baseline[:, :, top:bottom, left:right]
                score = lit(masked)[0, int(class_index)].detach()
                drop = torch.clamp(base_score - score, min=0.0)
                heat[:, top:bottom, left:right] += drop
                counts[:, top:bottom, left:right] += 1.0
    return heat / torch.clamp(counts, min=1.0)


def _frame_logit_heatmap(lit: SedLitModule, x: torch.Tensor, class_index: int) -> torch.Tensor | None:
    if not getattr(lit, "frame_task", False):
        return None
    try:
        with torch.no_grad():
            frame_logits = lit.model.forward_frames(x)
        values = frame_logits[0, :, int(class_index)]
        values = values.view(1, 1, 1, -1)
        values = F.interpolate(values, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return values[0]
    except Exception as exc:  # noqa: BLE001
        log.debug("frame-logit heatmap unavailable: %s", exc)
        return None


def _resize_to_input(cam: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if cam.ndim == 2:
        cam = cam.unsqueeze(0)
    if cam.ndim != 3:
        raise ValueError(f"CAM must have shape (B,H,W), got {tuple(cam.shape)}")
    return F.interpolate(
        cam.unsqueeze(1),
        size=x.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def _normalize_heatmap(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.maximum(arr, 0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(finite, 5))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        hi = float(np.max(finite))
        lo = float(np.min(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _time_edges(cfg: Config, width: int) -> list[float]:
    duration = float(getattr(cfg.features, "window_seconds", 0.0) or 0.0)
    if duration <= 0:
        duration = 1.0
    return [float(v) for v in np.linspace(0.0, duration, int(width) + 1)]


def _freq_edges(cfg: Config, height: int) -> list[float]:
    f = cfg.features
    fmin = float(getattr(f, "fmin", 0.0) or 0.0)
    fmax = float(getattr(f, "fmax", 0.0) or (int(cfg.model.sample_rate) / 2.0))
    feature_type = str(getattr(f, "feature_type", "mel") or "mel")
    if feature_type == "mel" and int(height) == int(getattr(f, "n_mels", height)):
        centers = kaldi_mel_center_freqs(height, fmin, fmax)
        if height == 1:
            return [fmin, fmax]
        mid = (centers[:-1] + centers[1:]) / 2.0
        edges = np.concatenate([[fmin], mid, [fmax]])
        return [float(v) for v in edges]
    return [float(v) for v in np.linspace(fmin, fmax, int(height) + 1)]


def _format_freq_range(low: float, high: float) -> str:
    if high >= 1000.0:
        return f"{low / 1000.0:.1f}-{high / 1000.0:.1f} kHz"
    return f"{low:.0f}-{high:.0f} Hz"


def _hotspot_boxes(
    heatmap: np.ndarray,
    time_edges: list[float],
    freq_edges: list[float],
    *,
    max_boxes: int = 3,
) -> list[dict[str, Any]]:
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.size == 0 or float(np.max(arr)) <= 0:
        return []
    height, width = arr.shape
    threshold = max(0.45, float(np.percentile(arr, 92)))
    mask = arr >= threshold
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[float, float, int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            ys: list[int] = []
            xs: list[int] = []
            mass = 0.0
            peak = 0.0
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                val = float(arr[cy, cx])
                mass += val
                peak = max(peak, val)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append((mass, peak, min(ys), max(ys), min(xs), max(xs)))
    if not components:
        flat_idx = np.argsort(arr.ravel())[-max_boxes:][::-1]
        components = []
        for idx in flat_idx:
            y, x = np.unravel_index(int(idx), arr.shape)
            components.append((float(arr[y, x]), float(arr[y, x]), y, y, x, x))
    components.sort(key=lambda item: (item[0], item[1]), reverse=True)
    total_mass = float(np.sum(arr)) or 1.0
    boxes: list[dict[str, Any]] = []
    for mass, peak, y0, y1, x0, x1 in components[:max_boxes]:
        time_start = float(time_edges[x0])
        time_end = float(time_edges[min(x1 + 1, len(time_edges) - 1)])
        freq_low = float(freq_edges[y0])
        freq_high = float(freq_edges[min(y1 + 1, len(freq_edges) - 1)])
        boxes.append({
            "time_start_s": time_start,
            "time_end_s": time_end,
            "freq_low_hz": freq_low,
            "freq_high_hz": freq_high,
            "label": f"{time_start:.2f}-{time_end:.2f} s, {_format_freq_range(freq_low, freq_high)}",
            "peak": float(peak),
            "mass": float(mass),
            "mass_fraction": float(mass / total_mass),
            "box": {
                "x0": float(x0 / max(1, width)),
                "x1": float((x1 + 1) / max(1, width)),
                "y0": float(y0 / max(1, height)),
                "y1": float((y1 + 1) / max(1, height)),
            },
        })
    return boxes


def _energy_overlap(heatmap: np.ndarray, x: torch.Tensor) -> float:
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.size == 0 or float(np.sum(arr)) <= 0:
        return 0.0
    energy = x.detach().cpu().float().abs().mean(dim=1)[0].numpy()
    if energy.shape != arr.shape:
        energy_t = torch.from_numpy(energy).view(1, 1, *energy.shape)
        energy = F.interpolate(
            energy_t, size=arr.shape, mode="bilinear", align_corners=False
        )[0, 0].numpy()
    cutoff = float(np.percentile(energy, 75))
    high_energy = energy >= cutoff
    return float(arr[high_energy].sum() / max(1e-8, arr.sum()))


def _heatmap_stats(heatmap: np.ndarray, x: torch.Tensor) -> dict[str, float]:
    arr = np.asarray(heatmap, dtype=np.float32)
    mass = float(np.sum(arr))
    if mass <= 0:
        return {
            "attribution_area": 0.0,
            "concentration": 0.0,
            "low_frequency_mass": 0.0,
            "high_frequency_mass": 0.0,
            "early_time_mass": 0.0,
            "late_time_mass": 0.0,
            "energy_overlap": 0.0,
        }
    active = arr >= max(0.35, float(np.percentile(arr, 85)))
    height, width = arr.shape
    top_mass = float(np.sort(arr.ravel())[-max(1, int(arr.size * 0.05)):].sum())
    low_mass = float(arr[: max(1, height // 3), :].sum() / mass)
    high_mass = float(arr[max(0, height - max(1, height // 3)):, :].sum() / mass)
    early_mass = float(arr[:, : max(1, width // 3)].sum() / mass)
    late_mass = float(arr[:, max(0, width - max(1, width // 3)):].sum() / mass)
    return {
        "attribution_area": float(active.mean()),
        "concentration": float(top_mass / mass),
        "low_frequency_mass": low_mass,
        "high_frequency_mass": high_mass,
        "early_time_mass": early_mass,
        "late_time_mass": late_mass,
        "energy_overlap": _energy_overlap(arr, x),
    }


def _target_class_indices(
    probs: np.ndarray,
    classes: list[str],
    rows: list[dict[str, str]],
    selection: str,
) -> list[list[int]]:
    value = str(selection or "top").strip().lower()
    if value in {"all", "*"}:
        return [list(range(len(classes))) for _ in range(probs.shape[0])]
    if value.startswith("class:"):
        wanted = value.split(":", 1)[1]
        if wanted not in classes:
            raise ValueError(f"unknown class for explanations: {wanted}")
        idx = classes.index(wanted)
        return [[idx] for _ in range(probs.shape[0])]
    out: list[list[int]] = []
    for i in range(probs.shape[0]):
        row = rows[i] if i < len(rows) else {}
        if value in {"predicted", "decision"}:
            cls = row.get("predicted_class") or classes[int(np.argmax(probs[i]))]
        elif value == "true":
            cls = row.get("class_name") or row.get("true_class") or classes[int(np.argmax(probs[i]))]
        else:
            cls = classes[int(np.argmax(probs[i]))]
        out.append([classes.index(cls)] if cls in classes else [int(np.argmax(probs[i]))])
    return out


def _explain_one(
    lit: SedLitModule,
    cfg: Config,
    x: torch.Tensor,
    class_index: int,
    method: str,
) -> tuple[np.ndarray, str]:
    method = str(method or "auto").lower()
    x = x.detach().cpu().float().unsqueeze(0)
    if method in {"auto", "frame"}:
        frame = _frame_logit_heatmap(lit, x, class_index)
        if frame is not None:
            return _normalize_heatmap(frame[0].detach().cpu().numpy()), "frame_logits"
        if method == "frame":
            raise RuntimeError("frame logits are unavailable for this model")
    if method in {"auto", "cam"}:
        cam = _exact_cam(lit, x, class_index)
        if cam is not None:
            return _normalize_heatmap(cam[0].detach().cpu().numpy()), "cam"
        if method == "cam":
            raise RuntimeError("exact CAM is unavailable for this model")
    if method in {"auto", "gradcam", "grad-cam"}:
        grad = _grad_cam(lit, x.clone().requires_grad_(True), class_index)
        if grad is not None:
            return _normalize_heatmap(grad[0].detach().cpu().numpy()), "grad_cam"
        if method in {"gradcam", "grad-cam"}:
            raise RuntimeError("Grad-CAM is unavailable for this model")
    if method in {"auto", "occlusion"}:
        occ = _occlusion_heatmap(lit, x, class_index)
        return _normalize_heatmap(occ[0].detach().cpu().numpy()), "occlusion"
    raise ValueError(f"unknown explanation method: {method}")


def _score_review_set(
    lit: SedLitModule,
    cfg: Config,
    project_id: str,
    review: ReviewSet,
    classes: list[str],
    *,
    predictions_out: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], list[dict[str, object]]]:
    from sed.train.eval import (
        _closed_set_decisions,
        _configured_thresholds,
        _firing_classes,
        _negative_role_classes,
        _prediction_records,
        _runtime_decisions,
        _write_prediction_table,
    )

    ds = SedDataset(review.csv_path, classes, cfg, layout=ProjectLayout(project_id))
    loader = DataLoader(ds, batch_size=max(1, int(cfg.training.batch_size)), shuffle=False, num_workers=0)
    probs_chunks: list[np.ndarray] = []
    targets_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            logits = lit(x)
            probs_chunks.append(logits_to_probabilities(logits, cfg).cpu().numpy())
            targets_chunks.append(y.cpu().numpy())
    if probs_chunks:
        probs = np.concatenate(probs_chunks, axis=0)
        targets = targets_to_one_hot_np(np.concatenate(targets_chunks, axis=0), len(classes))
    else:
        probs = np.zeros((0, len(classes)), dtype=np.float32)
        targets = np.zeros((0, len(classes)), dtype=np.float32)
    thresholds = _configured_thresholds(cfg, classes)
    negative_classes = _negative_role_classes(cfg, classes)
    if negative_classes:
        _idx, decisions = _runtime_decisions(
            probs,
            classes,
            thresholds,
            negative_max=cfg.inference.negative_max,
            margin_to_second_best=cfg.inference.margin_to_second_best,
            trigger_classes=_firing_classes(cfg, classes),
            negative_classes=negative_classes,
        )
    else:
        _idx, decisions = _closed_set_decisions(probs, classes)
    rows = list(ds.rows)
    records = _prediction_records(probs, targets, classes, rows, decisions)
    if predictions_out is not None:
        _write_prediction_table(records, classes, predictions_out)
    return probs, targets, rows, records


def _artifact_payload(
    *,
    scope: str,
    model_id: str,
    row: dict[str, str],
    record: dict[str, object],
    classes: list[str],
    class_index: int,
    probability: float,
    logit: float,
    method: str,
    heatmap: np.ndarray,
    cfg: Config,
    x: torch.Tensor,
) -> dict[str, Any]:
    height, width = heatmap.shape
    time_edges = _time_edges(cfg, width)
    freq_edges = _freq_edges(cfg, height)
    hotspots = _hotspot_boxes(heatmap, time_edges, freq_edges)
    stats = _heatmap_stats(heatmap, x.unsqueeze(0) if x.ndim == 3 else x)
    class_name = classes[class_index]
    true_class = str(record.get("true_class") or row.get("class_name") or "")
    predicted_class = str(record.get("predicted_class") or row.get("predicted_class") or "")
    decision_reason = str(record.get("decision_reason") or "")
    return {
        "ok": True,
        "version": EXPLANATION_VERSION,
        "scope": scope,
        "model_id": model_id,
        "path": str(row.get("path") or ""),
        "file": Path(str(row.get("path") or "")).name,
        "class_name": class_name,
        "class_index": int(class_index),
        "true_class": true_class,
        "predicted_class": predicted_class,
        "correct": bool(record.get("correct", true_class == predicted_class)),
        "decision_reason": decision_reason,
        "method": method,
        "probability": float(probability),
        "logit": float(logit),
        "heatmap_shape": [int(height), int(width)],
        "heatmap": np.round(heatmap.astype(np.float32), 4).tolist(),
        "time_edges_s": [round(float(v), 6) for v in time_edges],
        "freq_edges_hz": [round(float(v), 3) for v in freq_edges],
        "hotspots": hotspots,
        "stats": stats,
        "row_meta": {
            "origin": row.get("origin", ""),
            "session_id": row.get("session_id", ""),
            "kind": row.get("kind", ""),
            "best_channel": row.get("best_channel", ""),
            "created_at": row.get("created_at", ""),
        },
    }


def _summary_from_artifacts(
    *,
    model_id: str,
    scope: str,
    classes: list[str],
    artifacts: list[dict[str, Any]],
    n_clips: int,
    method_requested: str,
) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = {}
    by_pair: dict[str, list[dict[str, Any]]] = {}
    by_reason: dict[str, list[dict[str, Any]]] = {}
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for item in artifacts:
        by_class.setdefault(str(item.get("class_name") or ""), []).append(item)
        pair = f"{item.get('true_class') or '?'}→{item.get('predicted_class') or '?'}"
        by_pair.setdefault(pair, []).append(item)
        by_reason.setdefault(str(item.get("decision_reason") or ""), []).append(item)
        kind = str((item.get("row_meta") or {}).get("kind") or "")
        if kind:
            by_kind.setdefault(kind, []).append(item)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"count": 0}
        stats = [i.get("stats") or {} for i in items]
        hotspots = [h for item in items for h in (item.get("hotspots") or [])]
        return {
            "count": len(items),
            "mean_probability": float(np.mean([float(i.get("probability") or 0.0) for i in items])),
            "mean_concentration": float(np.mean([float(s.get("concentration") or 0.0) for s in stats])),
            "mean_attribution_area": float(np.mean([float(s.get("attribution_area") or 0.0) for s in stats])),
            "mean_energy_overlap": float(np.mean([float(s.get("energy_overlap") or 0.0) for s in stats])),
            "mean_low_frequency_mass": float(np.mean([float(s.get("low_frequency_mass") or 0.0) for s in stats])),
            "mean_high_frequency_mass": float(np.mean([float(s.get("high_frequency_mass") or 0.0) for s in stats])),
            "mean_early_time_mass": float(np.mean([float(s.get("early_time_mass") or 0.0) for s in stats])),
            "mean_late_time_mass": float(np.mean([float(s.get("late_time_mass") or 0.0) for s in stats])),
            "top_hotspots": sorted(
                hotspots,
                key=lambda h: (float(h.get("mass_fraction") or 0.0), float(h.get("peak") or 0.0)),
                reverse=True,
            )[:5],
            "representatives": [
                {
                    "path": i.get("path"),
                    "file": i.get("file"),
                    "class_name": i.get("class_name"),
                    "probability": i.get("probability"),
                    "method": i.get("method"),
                    "hotspot": (i.get("hotspots") or [{}])[0].get("label", ""),
                }
                for i in sorted(items, key=lambda v: float(v.get("probability") or 0.0), reverse=True)[:5]
            ],
        }

    return {
        "ok": True,
        "version": EXPLANATION_VERSION,
        "model_id": model_id,
        "scope": scope,
        "generated_at": datetime.now(UTC).isoformat(),
        "method_requested": method_requested,
        "classes": classes,
        "n_clips": int(n_clips),
        "n_maps": len(artifacts),
        "methods": {
            method: sum(1 for item in artifacts if item.get("method") == method)
            for method in sorted({str(item.get("method") or "") for item in artifacts})
        },
        "by_class": {key: summarize(items) for key, items in sorted(by_class.items())},
        "by_true_pred": {key: summarize(items) for key, items in sorted(by_pair.items())},
        "by_decision_reason": {key: summarize(items) for key, items in sorted(by_reason.items()) if key},
        "by_mined_kind": {key: summarize(items) for key, items in sorted(by_kind.items())},
        "representatives": summarize(artifacts).get("representatives", []),
    }


def generate_explanations(
    cfg: Config,
    project_id: str,
    model_id: str,
    *,
    scope: str = "frozen",
    method: str = "auto",
    classes: str = "top",
    force: bool = False,
    limit: int | None = None,
    output_dir: str = "mined",
) -> dict[str, Any]:
    """Score a review scope and write cached explanation artifacts."""

    scope = _normalize_scope(scope)
    lit, run_cfg, model_classes, model_dir, _ckpt = _load_lit_module(cfg, project_id, model_id)
    report_dir = explanation_report_dir(model_dir, scope)
    clips_dir = report_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    review = build_review_set(
        run_cfg,
        project_id,
        model_id,
        scope,
        classes=model_classes,
        output_dir=output_dir,
        work_dir=report_dir,
    )
    predictions_out = report_dir / "mined_predictions.tsv" if scope == "mined" else None
    probs, _targets, rows, records = _score_review_set(
        lit,
        run_cfg,
        project_id,
        review,
        model_classes,
        predictions_out=predictions_out,
    )
    if limit is not None:
        limit = max(0, int(limit))
        rows = rows[:limit]
        records = records[:limit]
        probs = probs[:limit]

    ds = SedDataset(review.csv_path, model_classes, run_cfg, layout=ProjectLayout(project_id))
    target_indices = _target_class_indices(probs, model_classes, records or rows, classes)
    artifacts: list[dict[str, Any]] = []
    with torch.no_grad():
        logits_all: list[np.ndarray] = []
        loader = DataLoader(ds, batch_size=max(1, int(run_cfg.training.batch_size)), shuffle=False, num_workers=0)
        for x_batch, _y in loader:
            logits_all.append(lit(x_batch).detach().cpu().numpy())
        logits = np.concatenate(logits_all, axis=0) if logits_all else np.zeros_like(probs)
    for i in range(min(len(rows), len(ds))):
        x, _y = ds[i]
        row = rows[i]
        record = records[i] if i < len(records) else {}
        for class_index in target_indices[i] if i < len(target_indices) else [int(np.argmax(probs[i]))]:
            class_name = model_classes[int(class_index)]
            out_path = explanation_artifact_path(model_dir, scope, row.get("path", ""), class_name)
            if out_path.exists() and not force:
                try:
                    payload = json.loads(out_path.read_text(encoding="utf-8"))
                    artifacts.append(payload)
                    continue
                except Exception:  # noqa: BLE001
                    pass
            heatmap, used_method = _explain_one(lit, run_cfg, x, int(class_index), method)
            probability = float(probs[i, int(class_index)]) if i < probs.shape[0] else 0.0
            logit = float(logits[i, int(class_index)]) if i < logits.shape[0] else 0.0
            payload = _artifact_payload(
                scope=scope,
                model_id=model_id,
                row=row,
                record=record,
                classes=model_classes,
                class_index=int(class_index),
                probability=probability,
                logit=logit,
                method=used_method,
                heatmap=heatmap,
                cfg=run_cfg,
                x=x,
            )
            npz_path = out_path.with_suffix(".npz")
            np.savez_compressed(npz_path, heatmap=heatmap.astype(np.float32))
            payload["heatmap_file"] = npz_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"), default=float)
            artifacts.append(payload)

    summary = _summary_from_artifacts(
        model_id=model_id,
        scope=scope,
        classes=model_classes,
        artifacts=artifacts,
        n_clips=len(rows),
        method_requested=method,
    )
    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def load_explanation_summary(
    project_id: str,
    model_id: str,
    scope: str,
) -> dict[str, Any] | None:
    model_dir = ProjectLayout(project_id).model_dir(model_id)
    path = explanation_report_dir(model_dir, scope) / "summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def load_explanation_for_clip(
    project_id: str,
    model_id: str,
    scope: str,
    path: str | Path,
    class_name: str,
) -> dict[str, Any] | None:
    model_dir = ProjectLayout(project_id).model_dir(model_id)
    artifact = explanation_artifact_path(model_dir, scope, path, class_name)
    if not artifact.exists():
        return None
    with open(artifact, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def explain_review_scope(
    cfg: Config,
    project_id: str,
    model_id: str,
    *,
    scope: str,
    method: str = "auto",
    classes: str = "top",
    force: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Small eval.py-facing wrapper for the shared review-set explainer."""

    return generate_explanations(
        cfg,
        project_id,
        model_id,
        scope=scope,
        method=method,
        classes=classes,
        force=force,
        limit=limit,
    )


__all__ = [
    "ReviewSet",
    "build_review_set",
    "class_activation_map",
    "clip_key",
    "explain_review_scope",
    "explanation_artifact_path",
    "explanation_report_dir",
    "generate_explanations",
    "load_explanation_for_clip",
    "load_explanation_summary",
]
