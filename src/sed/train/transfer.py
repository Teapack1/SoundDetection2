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
import logging
from pathlib import Path
from typing import Any

from sed.config import (
    EFFICIENTAT_FAST_NORM_MEAN,
    EFFICIENTAT_FAST_NORM_STD,
    Config,
    ProjectLayout,
    load_config_snapshot,
)
from sed.train.model_lab import recipe_from_config, recipe_signature
from sed.train.registry import get_checkpoint
from sed.util import sha256_file

log = logging.getLogger(__name__)


def _feature_signature(cfg: Config) -> dict[str, Any]:
    from sed.audio.features import (
        feature_band_count,
        feature_channels,
        feature_compression,
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
    compression = feature_compression(f)
    scaling_method = str(getattr(f, "scaling_method", "none") or "none")
    scaling_scope = str(getattr(f, "scaling_scope", "global") or "global")
    scaling_source = str(getattr(f, "scaling_source", "dataset") or "dataset")
    scaling_stats = {
        str(k): [float(v) for v in (vals or [])]
        for k, vals in (getattr(f, "scaling_stats", {}) or {}).items()
    }
    if scaling_method == "none" and standardize:
        # Configs mutated without re-validation: the legacy bool still declares
        # the fitted global z-score (validation normally mirrors these).
        scaling_method, scaling_scope, scaling_source = "zscore", "global", "dataset"
    is_fast_norm = (
        scaling_method == "zscore" and scaling_scope == "global"
        and scaling_source == "fixed"
        and scaling_stats.get("mean") == [EFFICIENTAT_FAST_NORM_MEAN]
        and scaling_stats.get("std") == [EFFICIENTAT_FAST_NORM_STD]
    )

    log_transform_for = {
        "none": None,
        "log": "ln(power+1e-5)",
        "db": "10*log10(max(power,1e-12))",
        "pcen": None,
    }
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
        log_transform = log_transform_for[compression]
        # The label names both stages, so two runs whose encoders see different
        # numbers can never read as the same contract.
        if db_norm:
            normalization = "legacy-fixed-db"
        elif is_fast_norm:
            normalization = "efficientat-fast-norm"
        elif scaling_method == "none":
            normalization = {"none": "raw-power", "log": "ln-power",
                             "db": "db-power", "pcen": "pcen"}[compression]
        else:
            normalization = f"{compression}+{scaling_method}-{scaling_scope}-{scaling_source}"

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
        # The two value axes. Both belong to the encoder's numeric contract:
        # a different compression or a different normalization means the
        # transferred weights see different numbers for the same sound.
        "compression": compression,
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
        # The normalization constants ARE part of that contract: two runs match
        # only when method, scope, source, and every constant agree.
        "scaling_method": scaling_method,
        "scaling_scope": scaling_scope,
        "scaling_source": scaling_source,
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


def _normalize_class_name(name: str) -> str:
    """Comparison form for class names: case, separators and spacing folded."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _name_tokens(name: str) -> list[str]:
    """Split a class name into its underscore/hyphen/space separated tokens."""
    out: list[str] = []
    current = ""
    for ch in str(name).lower():
        if ch.isalnum():
            current += ch
        elif current:
            out.append(current)
            current = ""
    if current:
        out.append(current)
    return out


def suggest_head_map(
    parent_names: list[str], child_names: list[str],
) -> dict[str, str]:
    """Propose ``{child class: parent class}`` rows for the classifier warm start.

    A *suggestion* for the Model tab's parent picker — nothing here is applied
    automatically, and the operator edits every row.  Exact matches are omitted
    because :func:`_warm_start_head` already handles them, so the result contains
    only the cross-name rows that would otherwise start random.

    Resolution, most to least confident:

    1. the same name after folding case and separators (``cheers_loud`` ↔
       ``cheers-loud``);
    2. an *umbrella* parent whose name is a token of the child's, which is the
       shape this pipeline actually produces — stage one labels ``cheers`` and
       stage two labels ``cheers_riedel_wine``;
    3. an umbrella parent whose normalized name is a prefix of the child's, for
       names written without a separator.

    A child that matches nothing is left out and keeps its random row.
    """
    parents = [str(name) for name in parent_names]
    exact = set(parents)
    by_normalized: dict[str, str] = {}
    for name in parents:
        by_normalized.setdefault(_normalize_class_name(name), name)

    out: dict[str, str] = {}
    for child in (str(name) for name in child_names):
        if child in exact:
            continue
        normalized = _normalize_class_name(child)
        match = by_normalized.get(normalized)
        if match is None:
            tokens = set(_name_tokens(child))
            # Prefer the longest umbrella name: with `cheers` and `cheers_thin`
            # both present, `cheers_thin_riedel` should follow the specific one.
            candidates = [
                name for name in parents
                if _normalize_class_name(name) in tokens
                or set(_name_tokens(name)) <= tokens
            ]
            if not candidates:
                candidates = [
                    name for name in parents
                    if len(_normalize_class_name(name)) >= 3
                    and normalized.startswith(_normalize_class_name(name))
                ]
            if candidates:
                match = max(candidates, key=lambda name: len(_normalize_class_name(name)))
        if match is not None and match != child:
            out[child] = match
    return out


def resolve_head_sources(
    parent_names: list[str],
    child_names: list[str],
    head_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Where each child class's classifier row comes from.

    Values are ``"exact"``, ``"mapped:<parent class>"`` or ``"random"``.  Shared
    by the warm start, the advice checks and the Model tab, so all three agree on
    what a configuration will actually do.
    """
    available = set(str(name) for name in parent_names)
    mapping = {str(k): str(v) for k, v in (head_map or {}).items()}
    out: dict[str, str] = {}
    for child in (str(name) for name in child_names):
        if child in available:
            out[child] = "exact"
            continue
        source = mapping.get(child)
        out[child] = f"mapped:{source}" if source in available else "random"
    return out


# Rows copied from one shared parent row would otherwise start bit-identical.
# Their gradients differ from the first step (the labels differ), so this is not
# required for learning — it just removes an exact tie that makes early argmax
# and margin behaviour degenerate. Scaled by the source row's own spread.
_HEAD_JITTER_SCALE = 0.02


def _warm_start_head(
    head: Any,
    parent_state: dict,
    parent_names: list[str],
    child_names: list[str],
    head_map: dict[str, str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Copy the parent's classifier row/bias into each child class's row.

    The head is a fresh ``Linear`` over the *same* embedding the parent's head
    read (identical backbone, identical params — ``parent_compatibility`` already
    enforced both), so a fitted row is a far better starting point than a random
    one.  A row is taken for a class the parent also had (exact name) or for a
    class ``head_map`` points at a parent class — the cross-stage case, where
    several vessel classes all start from the parent's umbrella ``cheers`` row.
    Classes with neither keep their random initialization, and a shape mismatch
    aborts the warm start rather than silently corrupting the head.
    """
    import torch

    weight = parent_state.get("model.head.1.weight")
    bias = parent_state.get("model.head.1.bias")
    sources = resolve_head_sources(parent_names, child_names, head_map)
    failed = lambda reason: {  # noqa: E731
        "sources": {name: "random" for name in child_names},
        "transferred_classes": [],
        "mapped_classes": {},
        "new_classes": list(child_names),
        "skipped_reason": reason,
    }
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        return failed("parent has no linear head state")
    linear = head[-1] if isinstance(head, torch.nn.Sequential) else head
    if not isinstance(linear, torch.nn.Linear):
        return failed("child head is not a Linear")
    if int(weight.shape[1]) != int(linear.weight.shape[1]):
        return failed(
            f"embedding width differs (parent {int(weight.shape[1])}, "
            f"child {int(linear.weight.shape[1])})"
        )
    if int(weight.shape[0]) != len(parent_names):
        return failed("parent head rows do not match its recorded class list")

    parent_index = {name: i for i, name in enumerate(parent_names)}
    generator = torch.Generator().manual_seed(int(seed))

    def _source_parent(name: str) -> str | None:
        source = sources.get(name, "random")
        if source == "exact":
            return name
        if source.startswith("mapped:"):
            return source.split(":", 1)[1]
        return None

    # How many child rows read each *parent row*: a row used once is copied
    # verbatim, and only genuinely duplicated rows are jittered.
    reuse: dict[str, int] = {}
    for name in child_names:
        parent_name = _source_parent(str(name))
        if parent_name is not None:
            reuse[parent_name] = reuse.get(parent_name, 0) + 1

    transferred: list[str] = []
    mapped: dict[str, str] = {}
    with torch.no_grad():
        for child_row, name in enumerate(child_names):
            source = sources.get(name, "random")
            parent_name = _source_parent(str(name))
            if parent_name is None:
                continue
            row = parent_index[parent_name]
            new_weight = weight[row].to(linear.weight.dtype).clone()
            new_bias = bias[row].to(linear.bias.dtype).clone()
            if reuse.get(parent_name, 0) > 1:
                spread = float(new_weight.std().item() or 0.0)
                if spread > 0:
                    noise = torch.empty(
                        new_weight.shape, dtype=torch.float32,
                    ).normal_(0.0, spread * _HEAD_JITTER_SCALE, generator=generator)
                    new_weight += noise.to(new_weight.dtype)
            linear.weight[child_row].copy_(new_weight)
            linear.bias[child_row].copy_(new_bias)
            if source == "exact":
                transferred.append(name)
            else:
                mapped[name] = parent_name

    new_classes = [name for name, src in sources.items() if src == "random"]
    return {
        "sources": sources,
        "transferred_classes": transferred,
        "mapped_classes": mapped,
        "new_classes": new_classes,
        "skipped_reason": (
            "" if (transferred or mapped) else "no class names in common and no head map"
        ),
    }


def _log_head_transfer(report: dict[str, Any]) -> None:
    """Name every classifier row's starting point in the training log.

    A cold target row is the difference between a fine-tune that converges in
    its epoch budget and one that collapses, so it belongs in the run log rather
    than only in the manifest.
    """
    if report.get("skipped_reason"):
        log.warning("head warm start skipped: %s", report["skipped_reason"])
        return
    for name, source in (report.get("sources") or {}).items():
        if source == "exact":
            log.info("head[%s] ← parent %r (same class name)", name, name)
        elif source.startswith("mapped:"):
            log.info("head[%s] ← parent %r (mapped)", name, source.split(":", 1)[1])
        else:
            log.info("head[%s] random — no parent row", name)
    cold = [n for n, s in (report.get("sources") or {}).items() if s == "random"]
    if cold:
        log.warning(
            "%d classifier row(s) start random on a transfer run: %s. "
            "Map them onto a parent class (training.transfer_head_map) if the "
            "parent already models the same event under another name.",
            len(cold), ", ".join(sorted(cold)),
        )


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
    head_map = dict(getattr(cfg.training, "transfer_head_map", {}) or {})
    info["head_map"] = head_map
    if info["head_init"] == "matching_classes":
        parent_names = parent_classes(cfg.model.parent_project, cfg.model.parent_run)
        child_names = [str(name) for name in getattr(model, "classes", [])]
        if parent_names and child_names:
            info["head_transfer"] = _warm_start_head(
                model.model.head,
                state,
                parent_names,
                child_names,
                head_map=head_map,
                seed=int(getattr(cfg.training, "seed", 0) or 0),
            )
            _log_head_transfer(info["head_transfer"])
        else:
            info["head_transfer"] = {
                "sources": {name: "random" for name in child_names},
                "transferred_classes": [],
                "mapped_classes": {},
                "new_classes": list(child_names),
                "skipped_reason": "parent or child class list unavailable",
            }
    else:
        info["head_transfer"] = {
            "sources": {
                str(name): "random" for name in getattr(model, "classes", [])
            },
            "transferred_classes": [],
            "mapped_classes": {},
            "new_classes": [str(name) for name in getattr(model, "classes", [])],
            "skipped_reason": "disabled by config",
        }
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

    child_classes = [str(name) for name in (cfg.dataset.target_classes or cfg.model.classes or [])]
    configured_head_map = dict(getattr(cfg.training, "transfer_head_map", {}) or {})
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
            # The parent's class list drives the Model tab's head-map editor: the
            # picker has to show which rows this run can warm-start from before
            # the run is configured, not after it has trained.
            names = parent_classes(project_id, run_id)
            info["classes"] = names
            info["head_map_suggestion"] = (
                suggest_head_map(names, child_classes) if names and child_classes else {}
            )
            info["head_sources"] = (
                resolve_head_sources(names, child_classes, configured_head_map)
                if names and child_classes else {}
            )
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
    "resolve_head_sources",
    "sha256_file",
    "suggest_head_map",
]
