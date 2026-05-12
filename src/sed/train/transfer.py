"""Safe encoder-only transfer from any previous run to a new one.

Any completed run — in any project, including the current one — can seed the
next run's encoder ("parent" initialization). Chains compose: a scratch/BYOL
domain pretraining run can parent a mid-scale adaptation run, which can parent
a small venue fine-tune, recursively. Each run's manifest records its parent
plus one grandparent hop; the full ancestry is reconstructed by walking the
parents' manifests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sed.config import Config, ProjectLayout, load_config_snapshot
from sed.train.model_lab import recipe_from_config, recipe_signature
from sed.train.registry import get_checkpoint
from sed.util import sha256_file


def _feature_signature(cfg: Config) -> dict[str, Any]:
    from sed.audio.features import (
        feature_band_count,
        feature_channels,
        feature_frame_count_for_cfg,
        window_sample_count,
    )

    f = cfg.features
    frontend = str(f.frontend)
    feature_type = str(f.feature_type)
    stft_engine = str(f.stft_engine)
    pcen = bool(f.pcen)
    db_norm = bool(f.db_norm)
    standardize = bool(f.standardize)
    log_power = bool(f.log_power)
    scaling_method = str(getattr(f, "scaling_method", "none") or "none")
    scaling_scope = str(getattr(f, "scaling_scope", "global") or "global")
    scaling_stats = {
        str(k): [float(v) for v in (vals or [])]
        for k, vals in (getattr(f, "scaling_stats", {}) or {}).items()
    }
    if scaling_method == "none" and standardize:
        # Configs mutated without re-validation: the legacy bool still declares
        # the fitted global z-score (validation normally mirrors these).
        scaling_method, scaling_scope = "zscore", "global"

    if frontend == "ced":
        spectral_basis = "htk-mel"
        normalization = "power-db-ref-1-top-120"
        log_transform = "10*log10(power/ref=1),top_db=120"
        pre_emphasis = None
        stft_center = True
        stft_window = "hann-periodic"
        stft_pad_mode = "reflect"
        stft_detrend = None
        stft_scaling = "magnitude-squared"
        input_amplitude_scale = 1.0
    else:
        spectral_basis = "kaldi-htk-mel" if feature_type == "mel" else "linear-rfft"
        pre_emphasis = 0.97 if stft_engine == "librosa" else None
        stft_center = stft_engine == "librosa"
        stft_window = "hann-symmetric" if stft_engine == "librosa" else "tukey-0.25"
        stft_pad_mode = "reflect" if stft_engine == "librosa" else None
        stft_detrend = None if stft_engine == "librosa" else "constant"
        stft_scaling = "magnitude-squared" if stft_engine == "librosa" else "psd-density"
        input_amplitude_scale = 1.0 if stft_engine == "librosa" else 32767.0
        if pcen:
            normalization = "pcen" if scaling_method == "none" else f"pcen+{scaling_method}"
            log_transform = None
        elif db_norm:
            normalization = "legacy-fixed-db"
            log_transform = "10*log10(max(power,1e-12))"
        elif scaling_method == "minmax" and (feature_type == "mel" or log_power):
            normalization = "minmax-ln-power"
            log_transform = "ln(power+1e-5)"
        elif scaling_method == "zscore" and (feature_type == "mel" or log_power):
            normalization = "zscore-ln-power"
            log_transform = "ln(power+1e-5)"
        elif feature_type == "mel":
            normalization = "efficientat-fast-norm"
            log_transform = "ln(power+1e-5)"
        elif log_power:
            normalization = "ln-power"
            log_transform = "ln(power+1e-5)"
        else:
            normalization = "raw-power"
            log_transform = None

    return {
        # Version the derived semantics as well as storing every configurable
        # value. A future frontend implementation change must not silently make
        # an older encoder appear transfer-compatible.
        "contract_version": 3,
        "frontend": frontend,
        "feature_type": feature_type,
        "spectral_basis": spectral_basis,
        "sample_rate": int(cfg.model.sample_rate),
        "channels": int(cfg.model.channels),
        "rendered_feature_channels": int(feature_channels(f)),
        "window_seconds": float(f.window_seconds),
        "window_samples": int(window_sample_count(cfg.model.sample_rate, f.window_seconds)),
        "n_fft": int(f.n_fft),
        "win_length": int(f.win_length),
        "hop_length": int(f.hop_length),
        "n_mels": int(f.n_mels),
        "resolved_feature_bands": int(feature_band_count(f, cfg.model.sample_rate)),
        "resolved_feature_frames": int(feature_frame_count_for_cfg(cfg)),
        "fmin": int(f.fmin),
        "fmax": int(f.fmax),
        "stft_engine": stft_engine,
        "log_power": log_power,
        "db_norm": db_norm,
        "db_min": float(f.db_min),
        "db_max": float(f.db_max),
        "pcen": pcen,
        "pcen_time_constant": float(f.pcen_time_constant),
        "pcen_gain": float(f.pcen_gain),
        "pcen_bias": float(f.pcen_bias),
        "pcen_power": float(f.pcen_power),
        "pcen_eps": float(f.pcen_eps),
        "standardize": standardize,
        "standardize_mean": float(f.standardize_mean),
        "standardize_std": float(f.standardize_std),
        # Dataset-fitted scaling constants ARE the encoder's numeric contract:
        # two runs match only when method, scope, and every fitted value agree.
        "scaling_method": scaling_method,
        "scaling_scope": scaling_scope,
        "scaling_stats": scaling_stats,
        "colormap": str(f.colormap),
        "image_size": int(f.image_size),
        # Explicit fixed EfficientAT semantics. These are deliberately present
        # even though they are not user knobs: they define the value
        # distribution seen by a transferred encoder.
        "pre_emphasis": pre_emphasis,
        "stft_center": stft_center,
        "stft_window": stft_window,
        "stft_pad_mode": stft_pad_mode,
        "stft_detrend": stft_detrend,
        "stft_scaling": stft_scaling,
        "input_amplitude_scale": input_amplitude_scale,
        "mel_scale": "htk" if feature_type == "mel" else None,
        "mel_norm": "none" if feature_type == "mel" else None,
        "normalization": normalization,
        "log_transform": log_transform,
        "fast_norm_offset": 4.5 if normalization == "efficientat-fast-norm" else None,
        "fast_norm_scale": 5.0 if normalization == "efficientat-fast-norm" else None,
    }


def parent_feature_payload(parent: Config) -> dict[str, Any]:
    """Parent feature semantics plus descriptive fitted-stat provenance.

    The resolved mean/std are part of :func:`_feature_signature` and therefore
    must match for encoder transfer.  Dataset hash, split seed and sample count
    explain where those values came from, but intentionally do not participate
    in compatibility: two independently estimated contracts with identical
    resolved values feed the encoder exactly the same numbers.
    """
    f = parent.features
    return {
        **_feature_signature(parent),
        "standardize_stats_resolved": bool(
            getattr(f, "standardize_stats_resolved", True)
        ),
        "standardize_stats_dataset_sha256": str(
            getattr(f, "standardize_stats_dataset_sha256", "") or ""
        ),
        "standardize_stats_split_seed": int(
            getattr(f, "standardize_stats_split_seed", 0) or 0
        ),
        "standardize_stats_val_fraction": float(
            getattr(f, "standardize_stats_val_fraction", 0.0) or 0.0
        ),
        "standardize_stats_clips": int(
            getattr(f, "standardize_stats_clips", 0) or 0
        ),
    }


def _read_manifest(parent_project: str, parent_run: str) -> dict[str, Any]:
    manifest_path = ProjectLayout(parent_project).model_dir(parent_run) / "manifest.json"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def parent_compatibility(cfg: Config, parent_project: str, parent_run: str) -> dict[str, Any]:
    """Return an explainable strict compatibility decision for encoder transfer.

    Hard requirements: same backbone, same architecture params, same feature
    signature, checkpoint present. A task mismatch (clip vs frame) is only a
    warning — the classifier head is discarded and the encoder transfers; the
    strict state-dict load at init time remains the structural backstop.
    """
    layout = ProjectLayout(parent_project)
    config_path = layout.model_dir(parent_run) / "training_config.json"
    if not config_path.exists():
        return {"ok": False, "reason": f"parent config is missing: {config_path}",
                "warnings": [], "feature_matches": False, "hard_blocked": True}
    parent = load_config_snapshot(config_path)
    errors: list[str] = []
    warnings: list[str] = []
    if parent.model.backbone != cfg.model.backbone:
        errors.append(f"backbone differs ({parent.model.backbone} vs {cfg.model.backbone})")
    if dict(parent.model.params) != dict(cfg.model.params):
        errors.append("architecture parameters differ")
    if parent.model.task != cfg.model.task:
        warnings.append(
            f"task differs ({parent.model.task} vs {cfg.model.task}): "
            "the encoder transfers, and the head warm-starts per shared class name "
            "either way"
        )
    feature_matches = _feature_signature(parent) == _feature_signature(cfg)
    if not feature_matches:
        errors.append("feature geometry/contract differs")
    try:
        checkpoint = get_checkpoint(parent_project, parent_run)
    except FileNotFoundError as exc:
        errors.append(str(exc))
        checkpoint = None
    return {
        "ok": not errors,
        "reason": "; ".join(errors),
        "warnings": warnings,
        # A pure feature mismatch is fixable by adopting the parent's feature
        # config; hard_blocked marks the mismatches that are not.
        "feature_matches": feature_matches,
        "hard_blocked": bool([e for e in errors if e != "feature geometry/contract differs"]),
        "parent_project": parent_project,
        "parent_run": parent_run,
        "parent_recipe_signature": recipe_signature(recipe_from_config(parent)),
        "checkpoint": str(checkpoint) if checkpoint else "",
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else "",
    }


def parent_provenance_chain(parent_project: str, parent_run: str) -> dict[str, Any]:
    """One-hop ancestry read from the parent's manifest (best effort).

    Records how the parent itself was initialized so a chained run's manifest
    shows pretrain → adapt → fine-tune lineage without walking every bundle at
    train time. Deeper ancestry is reconstructable by following the
    grandparent fields into that run's own manifest.
    """
    manifest = _read_manifest(parent_project, parent_run)
    init = manifest.get("initialization")
    if not isinstance(init, dict):
        return {}
    out: dict[str, Any] = {"parent_initialization_mode": str(init.get("mode", "") or "")}
    if init.get("parent_project") or init.get("parent_run"):
        out["grandparent_project"] = str(init.get("parent_project", "") or "")
        out["grandparent_run"] = str(init.get("parent_run", "") or "")
        if init.get("parent_recipe_signature"):
            out["grandparent_signature"] = str(init.get("parent_recipe_signature", "") or "")
    elif init.get("weight_id"):
        out["parent_weight_id"] = str(init.get("weight_id", "") or "")
    return out


def parent_classes(parent_project: str, parent_run: str) -> list[str]:
    """Class list a parent run was trained on, or ``[]`` if it cannot be read."""
    layout = ProjectLayout(parent_project)
    for candidate in (
        layout.model_dir(parent_run) / "classes.json",
        layout.model_dir(parent_run) / "dataset" / "classes.json",
    ):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        names = doc if isinstance(doc, list) else doc.get("classes")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return []


def _warm_start_head(
    head: Any, parent_state: dict, parent_names: list[str], child_names: list[str],
) -> dict[str, Any]:
    """Copy the parent's classifier row/bias for every shared class name.

    The head is a fresh ``Linear`` over the *same* embedding the parent's head
    read (identical backbone, identical params — ``parent_compatibility`` already
    enforced both), so a shared class's fitted row is a far better starting point
    than a random one.  Classes the parent never saw keep their random
    initialization, and a shape mismatch aborts the warm start rather than
    silently corrupting the head.
    """
    import torch

    weight = parent_state.get("model.head.1.weight")
    bias = parent_state.get("model.head.1.bias")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        return {"transferred_classes": [], "skipped_reason": "parent has no linear head state"}
    linear = head[-1] if isinstance(head, torch.nn.Sequential) else head
    if not isinstance(linear, torch.nn.Linear):
        return {"transferred_classes": [], "skipped_reason": "child head is not a Linear"}
    if int(weight.shape[1]) != int(linear.weight.shape[1]):
        return {
            "transferred_classes": [],
            "skipped_reason": (
                f"embedding width differs (parent {int(weight.shape[1])}, "
                f"child {int(linear.weight.shape[1])})"
            ),
        }
    if int(weight.shape[0]) != len(parent_names):
        return {
            "transferred_classes": [],
            "skipped_reason": "parent head rows do not match its recorded class list",
        }
    parent_index = {name: i for i, name in enumerate(parent_names)}
    transferred: list[str] = []
    with torch.no_grad():
        for child_row, name in enumerate(child_names):
            source = parent_index.get(name)
            if source is None:
                continue
            linear.weight[child_row].copy_(weight[source].to(linear.weight.dtype))
            linear.bias[child_row].copy_(bias[source].to(linear.bias.dtype))
            transferred.append(name)
    return {
        "transferred_classes": transferred,
        "new_classes": [name for name in child_names if name not in parent_index],
        "skipped_reason": "" if transferred else "no class names in common",
    }


def initialize_encoder_from_parent(model: Any, cfg: Config) -> dict[str, Any]:
    """Load a compatible parent's backbone, and warm-start shared class heads.

    The encoder always transfers.  The classifier head transfers per class name
    unless ``training.transfer_head_init`` is ``"random"`` — see that field for
    why the default is to reuse what the parent already fitted.
    """
    info = parent_compatibility(cfg, cfg.model.parent_project, cfg.model.parent_run)
    if not info["ok"]:
        raise ValueError(f"incompatible parent initialization: {info['reason']}")
    import torch

    checkpoint = Path(info["checkpoint"])
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise ValueError(f"parent checkpoint has no state dict: {checkpoint}")
    prefix = "model.backbone."
    encoder_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not encoder_state:
        raise ValueError(f"parent checkpoint has no encoder state under {prefix!r}")
    missing, unexpected = model.model.backbone.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "parent encoder is structurally incompatible; "
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}"
        )
    info["mode"] = "parent"
    info["head_init"] = str(getattr(cfg.training, "transfer_head_init", "matching_classes"))
    if info["head_init"] == "matching_classes":
        parent_names = parent_classes(cfg.model.parent_project, cfg.model.parent_run)
        child_names = [str(name) for name in getattr(model, "classes", [])]
        if parent_names and child_names:
            info["head_transfer"] = _warm_start_head(
                model.model.head, state, parent_names, child_names,
            )
        else:
            info["head_transfer"] = {
                "transferred_classes": [],
                "skipped_reason": "parent or child class list unavailable",
            }
    else:
        info["head_transfer"] = {"transferred_classes": [], "skipped_reason": "disabled by config"}
    info.update(parent_provenance_chain(cfg.model.parent_project, cfg.model.parent_run))
    return info


def compatible_parent_runs(cfg: Config) -> list[dict[str, Any]]:
    """Enumerate candidate parent runs across every project (incl. this one).

    Each row carries the compatibility verdict plus manifest context (created,
    objective, selection score, task) and the parent's feature geometry so the
    picker can offer adopt-feature when geometries differ.
    """
    out: list[dict[str, Any]] = []
    from sed.config import list_projects

    for project_id in list_projects():
        layout = ProjectLayout(project_id)
        for run_id in layout.list_models():
            try:
                info = parent_compatibility(cfg, project_id, run_id)
            except Exception as exc:  # noqa: BLE001
                info = {"ok": False, "reason": str(exc), "warnings": [], "hard_blocked": True,
                        "feature_matches": False, "parent_project": project_id, "parent_run": run_id}
            info["selectable_with_feature_adoption"] = bool(
                not info.get("ok") and not info.get("hard_blocked")
                and info.get("feature_matches") is False
            )
            manifest = _read_manifest(project_id, run_id)
            training = manifest.get("training") if isinstance(manifest.get("training"), dict) else {}
            recipe = manifest.get("model_recipe") if isinstance(manifest.get("model_recipe"), dict) else {}
            info["created_at"] = str(manifest.get("created_at", "") or "")
            info["objective"] = str(training.get("objective", "") or "supervised")
            info["selected_score"] = training.get("selected_score")
            info["selection_metric"] = str(training.get("selection_metric", "") or "")
            info["task"] = str(recipe.get("task", "") or "")
            info["backbone"] = str(recipe.get("backbone", "") or "")
            config_path = layout.model_dir(run_id) / "training_config.json"
            try:
                info["feature"] = parent_feature_payload(load_config_snapshot(config_path))
            except Exception:  # noqa: BLE001
                info["feature"] = {}
            out.append(info)
    return sorted(out, key=lambda row: (not bool(row.get("ok")), str(row.get("parent_project")), str(row.get("parent_run"))))


__all__ = [
    "compatible_parent_runs",
    "initialize_encoder_from_parent",
    "parent_classes",
    "parent_compatibility",
    "parent_feature_payload",
    "parent_provenance_chain",
    "sha256_file",
]
