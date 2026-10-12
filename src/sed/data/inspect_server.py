"""Local HTTP server for the interactive inspect UI.

`sed dataset inspect --serve` opens this on http://127.0.0.1:PORT. The
HTML report is rendered live and each clip carries a checkbox. Pressing
"Done" writes the ticked dataset/augmented clips to the project's soft
exclusion manifest for the next compose/train run. The "Delete selected"
button POSTs the chosen paths to /api/delete, which removes the WAV + sidecar
from disk.

No external deps — uses stdlib http.server. Single-process, localhost-only;
no auth needed because the server only binds to 127.0.0.1.
"""

from __future__ import annotations

import collections
import codecs
import csv
from functools import lru_cache
import json
import logging
import mimetypes
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
import threading
import uuid
import webbrowser
from concurrent.futures import FIRST_COMPLETED, wait
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from sed.config import (
    Config,
    ProjectLayout,
    effective_split_test_enabled,
    effective_split_test_fraction,
)
from sed.data.source_filters import (
    augmentation_settings_snapshot,
    clip_matches_filter_rule,
    library_paths_for_filters,
)

log = logging.getLogger(__name__)


_IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_AUDIO_SUFFIXES: frozenset[str] = frozenset({".wav"})
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CLINK_CNN_V2_GMAC_BUDGET = 0.25
_LINEAR_FRONTEND_SIGNATURE_FIELDS: tuple[str, ...] = (
    "feature_type",
    "frontend",
    "n_fft",
    "win_length",
    "hop_length",
    "n_mels",
    "fmin",
    "fmax",
    "window_seconds",
    "log_power",
    "stft_engine",
    "pcen",
    "pcen_time_constant",
    "pcen_gain",
    "pcen_bias",
    "pcen_power",
    "pcen_eps",
    "db_norm",
    "db_min",
    "db_max",
    "colormap",
    "image_size",
    "scaling_method",
    "scaling_scope",
)


@lru_cache(maxsize=32)
def _clink_cnn_v2_gmac(input_channels: int, n_bands: int, n_frames: int) -> float | None:
    """Measure encoder MACs for one Clink feature geometry, cached for UI probes."""
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode

        from sed.train.backbones import resolve_adapter

        adapter = resolve_adapter("clink-cnn-v2")
        backbone = adapter.build_backbone(
            pretrained=False,
            n_bands=int(n_bands),
            n_frames=int(n_frames),
            input_channels=int(input_channels),
        ).eval()
        sample = torch.zeros(1, int(input_channels), int(n_bands), int(n_frames))
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            backbone(sample)
        # PyTorch reports multiply and add as two FLOPs; the deployment budget
        # is expressed as multiply-accumulate operations.
        return float(counter.get_total_flops()) / 2e9
    except Exception as exc:  # noqa: BLE001 — compatibility report stays usable
        log.debug("ClinkCNN-v2 MAC estimate unavailable: %s", exc)
        return None


def _clink_cnn_v2_compute_verdict(gmac: float | None) -> dict | None:
    if gmac is None or float(gmac) <= _CLINK_CNN_V2_GMAC_BUDGET:
        return None
    return {
        "level": "warn",
        "kind": "compute_budget",
        "text": (
            f"Estimated {float(gmac):.3f} GMAC/clip exceeds the ClinkCNN-v2 "
            f"{_CLINK_CNN_V2_GMAC_BUDGET:.2f}-GMAC deployment budget for this "
            "feature geometry. Training remains allowed; restore the native "
            "128x81 input or benchmark deployment latency before promotion."
        ),
    }


def _preserve_fitted_linear_statistics(
    current_features: Any,
    candidate_features: Any,
    *,
    current_sample_rate: int,
    candidate_sample_rate: int,
) -> tuple[Any, bool]:
    """Keep fitted scaling statistics when a matching preset is only a template.

    Shipped scaled presets intentionally carry unresolved statistics.  The
    Audio Features estimator subsequently fits those values for the project,
    but a later Model-tab verification used to reapply the template and erase
    them.  Preserve the existing fitted values only when every feature-extractor
    input (including scaling method/scope) and sample rate is identical; a
    changed geometry, dataset, or seed is still rejected by the training
    preflight and must be estimated again.
    """
    if not (
        str(getattr(current_features, "scaling_method", "none") or "none") != "none"
        and bool(getattr(current_features, "standardize_stats_resolved", False))
        and str(getattr(candidate_features, "scaling_method", "none") or "none") != "none"
        and not bool(getattr(candidate_features, "standardize_stats_resolved", False))
        and int(current_sample_rate) == int(candidate_sample_rate)
    ):
        return candidate_features, False
    if any(
        getattr(current_features, field) != getattr(candidate_features, field)
        for field in _LINEAR_FRONTEND_SIGNATURE_FIELDS
    ):
        return candidate_features, False
    return current_features, True


def _safe_resolve_asset_path(
    layout: ProjectLayout, raw_path: str, allowed_suffixes: frozenset[str],
) -> Path | None:
    """Resolve a path requested via /audio?path=... or /image?path=... and
    only allow files inside the library dir, the project dataset report dir,
    or the legacy project's recordings dir. The suffix must be in
    `allowed_suffixes`. This keeps the server from being abused as a
    file-read primitive."""
    from sed.config import LIBRARY_DIR
    try:
        candidate = Path(raw_path).resolve()
    except Exception:  # noqa: BLE001
        return None
    allowed_roots = [
        (layout.dataset_inspect_dir / "augmented").resolve(),
        LIBRARY_DIR.resolve(),
        layout.recordings_dir.resolve(),
    ]
    for root in allowed_roots:
        try:
            candidate.relative_to(root)
            if candidate.is_file() and candidate.suffix.lower() in allowed_suffixes:
                return candidate
        except ValueError:
            continue
    return None


def _safe_resolve_audio_path(layout: ProjectLayout, raw_path: str) -> Path | None:
    return _safe_resolve_asset_path(layout, raw_path, _AUDIO_SUFFIXES)


_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")


def _query_readers(qs: dict) -> tuple[Callable, Callable, Callable, Callable]:
    """Typed readers over one parsed query string: ``(text, int, float, bool)``.

    The feature-lab GET handlers all take the same shape of query string, and
    each used to redefine these four closures itself. Besides the duplication
    the copies had drifted: one read an unrecognized boolean as ``False`` while
    the rest fell back to the caller's default, so the same malformed query
    meant different things depending on which endpoint received it. Unknown
    values now always fall back to the default, everywhere.
    """
    def text(key: str, default: str = "") -> str:
        return qs.get(key, [default])[0] or default

    def as_int(key: str, default: int) -> int:
        try:
            return int(float(text(key, str(default))))
        except (TypeError, ValueError):
            return default

    def as_float(key: str, default: float) -> float:
        try:
            return float(text(key, str(default)))
        except (TypeError, ValueError):
            return default

    def as_bool(key: str, default: bool) -> bool:
        value = text(key, "").strip().lower()
        if value in _TRUTHY:
            return True
        if value in _FALSY:
            return False
        return default

    return text, as_int, as_float, as_bool


def _job_status_payload(job: dict | None, *, missing: str) -> dict:
    """The shared background-job status envelope.

    Every long-running server action (compose, augment, delete, integrity scan,
    frozen eval, explanations, preview cache, review refresh) reports through
    this shape: elapsed seconds derived from the job's own clock, and no
    ``thread`` object — that is not JSON-serializable and must never reach the
    wire. Handlers that expose a rate or an ETA add those fields on top.
    """
    if not job:
        return {"ok": False, "error": missing}
    payload = dict(job)
    started = float(payload.get("started_at") or time.time())
    ended = payload.get("ended_at")
    payload["elapsed_s"] = max(0.0, float(ended or time.time()) - started)
    payload.pop("thread", None)
    return {"ok": True, "job": payload}


def _safe_resolve_model_dir(layout: ProjectLayout, model_id: str) -> Path | None:
    """Resolve a model bundle id without allowing path traversal."""
    model_id = str(model_id or "").strip()
    if not _MODEL_ID_RE.fullmatch(model_id):
        return None
    try:
        root = layout.models_dir.resolve()
        candidate = (layout.models_dir / model_id).resolve()
        candidate.relative_to(root)
    except Exception:  # noqa: BLE001
        return None
    return candidate


def _clip_delete_record(clip, library_root: Path) -> dict[str, Any]:
    path = clip.absolute_path(library_root)
    return {
        "path": os.path.normpath(str(path)),
        "relpath": str(getattr(clip, "relpath", "") or ""),
        "uuid": str(getattr(clip, "uuid", "") or ""),
        "category": str(getattr(clip, "category", "") or ""),
        "label": str(getattr(clip, "label", "") or ""),
        "origin_class": str(getattr(clip, "origin_class", "") or getattr(clip, "label", "") or ""),
        "project_id": str(getattr(clip, "project_id", "") or ""),
    }


def _deleted_ref_matches(ref: object, *, paths: set[str], relpaths: set[str], uuids: set[str]) -> bool:
    if isinstance(ref, str):
        try:
            return os.path.normpath(str(Path(ref).expanduser().resolve())) in paths
        except Exception:  # noqa: BLE001
            return False
    if not isinstance(ref, dict):
        return False
    if str(ref.get("uuid") or "") in uuids and str(ref.get("uuid") or ""):
        return True
    if str(ref.get("relpath") or "") in relpaths and str(ref.get("relpath") or ""):
        return True
    raw_path = str(ref.get("path") or "")
    if raw_path:
        try:
            return os.path.normpath(str(Path(raw_path).expanduser().resolve())) in paths
        except Exception:  # noqa: BLE001
            return False
    return False


def _apply_deleted_clips_to_selection(
    layout: ProjectLayout,
    clips: list,
    *,
    cfg: Config | None,
    added_clips: list | None = None,
    deleted_records: list[dict[str, Any]] | None = None,
) -> tuple[dict, bool]:
    """Prune moved/deleted references and adjust only matching rule counts."""
    from sed.data.compose import load_dataset_selection, write_dataset_selection

    selection = load_dataset_selection(layout)
    # ``absolute_path`` expects the shared library root; resolve it once.
    from sed.config import LIBRARY_DIR

    library_root = LIBRARY_DIR.resolve()
    records = deleted_records or [_clip_delete_record(clip, library_root) for clip in clips]
    paths = {str(record["path"]) for record in records}
    relpaths = {str(record["relpath"]) for record in records if record.get("relpath")}
    uuids = {str(record["uuid"]) for record in records if record.get("uuid")}
    changed = False

    assignments = selection.get("clip_assignments")
    if isinstance(assignments, dict):
        for class_name, raw_rows in list(assignments.items()):
            rows = raw_rows if isinstance(raw_rows, list) else [raw_rows]
            kept = [
                row for row in rows
                if not _deleted_ref_matches(row, paths=paths, relpaths=relpaths, uuids=uuids)
            ]
            if len(kept) != len(rows):
                changed = True
                if kept:
                    assignments[class_name] = kept
                else:
                    assignments.pop(class_name, None)

    overrides = selection.get("clip_overrides")
    if isinstance(overrides, dict):
        for key in list(overrides):
            if _deleted_ref_matches(key, paths=paths, relpaths=relpaths, uuids=uuids):
                overrides.pop(key, None)
                changed = True

    added_clips = added_clips or []

    def decrement_rule(rule: dict, *, frozen_test: bool = False) -> bool:
        label = _labels_summary_from_filters([rule])
        removed_hits = sum(
            1 for clip in clips
            if clip_matches_filter_rule(
                clip,
                rule,
                label=label,
                project_id=layout.id,
                frozen_test=frozen_test,
            )
        )
        added_hits = sum(
            1 for clip in added_clips
            if clip_matches_filter_rule(
                clip,
                rule,
                label=label,
                project_id=layout.id,
                frozen_test=frozen_test,
            )
        )
        delta = added_hits - removed_hits
        if not delta or "count" not in rule:
            return False
        try:
            old = int(rule.get("count") or 0)
        except (TypeError, ValueError):
            return False
        rule["count"] = max(0, old + delta)
        return rule["count"] != old

    source_rules = selection.get("assignments")
    if isinstance(source_rules, dict):
        for rows in source_rules.values():
            for rule in rows if isinstance(rows, list) else [rows]:
                if isinstance(rule, dict):
                    # Source-rule recounts use the rule's explicit project
                    # scope; ``split`` is only the compose destination.
                    changed |= decrement_rule(rule)

    recipes = selection.get("augmentation_recipes")
    if isinstance(recipes, list):
        for recipe in recipes:
            if not isinstance(recipe, dict):
                continue
            frozen = bool(recipe.get("frozen_test", False))
            recipe_changed = False
            for key in ("base_filters", "overlay_filters"):
                rows = recipe.get(key)
                if not isinstance(rows, list):
                    singular = recipe.get(key[:-1])
                    rows = [singular] if isinstance(singular, dict) else []
                for rule in rows:
                    if isinstance(rule, dict):
                        recipe_changed |= decrement_rule(rule, frozen_test=frozen)
            if recipe_changed:
                base_filters = recipe.get("base_filters") or (
                    [recipe["base_filter"]] if isinstance(recipe.get("base_filter"), dict) else []
                )
                overlay_filters = recipe.get("overlay_filters") or (
                    [recipe["overlay_filter"]] if isinstance(recipe.get("overlay_filter"), dict) else []
                )
                recipe["base_count"] = sum(int(rule.get("count") or 0) for rule in base_filters)
                recipe["overlay_count"] = sum(int(rule.get("count") or 0) for rule in overlay_filters)
                changed = True

    if changed:
        write_dataset_selection(layout, selection)
    return selection, changed


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _dedupe_strings(raw: object) -> list[str]:
    if isinstance(raw, str):
        parts = raw.replace(";", ",").split(",")
    elif isinstance(raw, list):
        parts = [str(item) for item in raw]
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        val = str(part).strip().lstrip("#")
        if not val or val in seen:
            continue
        out.append(val)
        seen.add(val)
    return out


def _selection_rule_id() -> str:
    return uuid.uuid4().hex[:12]


def _normalise_filter_rule(raw: object, *, default_split: str = "train") -> dict:
    if not isinstance(raw, dict):
        raw = {}
    row: dict = {}
    labels = _dedupe_strings(raw.get("labels", raw.get("label")))
    if labels:
        row["labels"] = labels
    project_ids = raw.get("project_ids", raw.get("project_id"))
    projects = _dedupe_strings(project_ids)
    if project_ids == "*":
        row["project_ids"] = "*"
    elif projects:
        row["project_ids"] = projects
    for key in (
        "density", "source_bucket", "session", "search",
        "clip_kind", "mic_filter",
    ):
        value = str(raw.get(key) or "").strip()
        if value:
            row[key] = value
    try:
        from sed.data.library import coerce_channel_list
        for key in ("triggered_channels", "recorded_channels", "mic_channels"):
            channels = coerce_channel_list(raw.get(key))
            if channels:
                row[key] = channels
    except Exception:  # noqa: BLE001
        pass
    if raw.get("channel_index") not in (None, ""):
        try:
            channel_index = int(raw.get("channel_index"))
            if channel_index >= 0:
                row["channel_index"] = channel_index
        except Exception:  # noqa: BLE001
            pass
    tags = _dedupe_strings(raw.get("tags_all", raw.get("tags")))
    if tags:
        row["tags_all"] = tags
    # Migrate legacy source-rule role fields once, but never persist them.
    split = str(raw.get("split") or raw.get("dataset_role") or default_split or "train").strip().lower()
    if split not in {"train", "test"}:
        split = "train"
    row["split"] = split
    if "count" in raw:
        try:
            row["count"] = int(raw.get("count") or 0)
        except Exception:  # noqa: BLE001
            pass
    row["id"] = str(raw.get("id") or _selection_rule_id())
    row["updated_at"] = _now_iso()
    return row


def _single_label_from_rule(rule: dict) -> str:
    labels = rule.get("labels")
    if isinstance(labels, list) and len(labels) == 1 and labels[0] != "*":
        return str(labels[0])
    label = rule.get("label")
    if isinstance(label, str) and label and label != "*":
        return label
    return "*"


def _normalise_filter_rules(raw: object, *, default_split: str = "train") -> list[dict]:
    rows = raw if isinstance(raw, list) else [raw]
    return [
        _normalise_filter_rule(row, default_split=default_split)
        for row in rows
        if isinstance(row, dict)
    ]


def _normalise_augmentation_recipe(raw: object) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    base_filters = _normalise_filter_rules(
        raw.get("base_filters", raw.get("base_filter") or {}),
        default_split="train",
    ) or [_normalise_filter_rule({}, default_split="train")]
    overlay_raw = raw.get("overlay_filters", raw.get("overlay_filter") or {})
    overlay_filters = _normalise_filter_rules(
        overlay_raw,
        default_split="train",
    )
    if not overlay_filters and not (isinstance(overlay_raw, list) and len(overlay_raw) == 0):
        overlay_filters = [_normalise_filter_rule({}, default_split="train")]
    recipe_id = str(raw.get("recipe_id") or raw.get("id") or _selection_rule_id())
    overlay_filter = overlay_filters[0] if overlay_filters else {}
    overlay_label = (
        str(raw.get("overlay"))
        if raw.get("overlay")
        else (_single_label_from_rule(overlay_filter) if overlay_filters else "")
    )
    # Recipes built from frozen-test buckets default to frozen-test output so
    # augmented test material lands in the test split without an extra click.
    frozen_raw = raw.get("frozen_test")
    if frozen_raw is None:
        # base_filters have already been normalised, so ``split`` is authoritative.
        frozen_test = bool(base_filters) and all(
            str(rule.get("split") or "").strip().lower() == "test"
            for rule in base_filters
        )
    else:
        frozen_test = _bool_value(frozen_raw)
    total_samples_raw = raw.get("total_samples")
    total_samples: int | None = None
    if total_samples_raw not in (None, ""):
        try:
            parsed_total = float(total_samples_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("total_samples must be a positive whole number") from exc
        if not parsed_total.is_integer() or parsed_total < 1:
            raise ValueError("total_samples must be a positive whole number")
        total_samples = int(parsed_total)
    recipe = {
        "id": recipe_id,
        "recipe_id": recipe_id,
        "base": str(raw.get("base") or _single_label_from_rule(base_filters[0]) or "*"),
        "overlay": overlay_label,
        "base_filter": base_filters[0],
        "overlay_filter": overlay_filter,
        "base_filters": base_filters,
        "overlay_filters": overlay_filters,
        "description": str(raw.get("description") or "").strip(),
        "tags": _dedupe_strings(raw.get("tags")),
        "base_count": int(raw.get("base_count") or sum(int(r.get("count") or 0) for r in base_filters)),
        "overlay_count": int(raw.get("overlay_count") or sum(int(r.get("count") or 0) for r in overlay_filters)),
        # An explicit recipe-level target. Unlike samples_per_source this is
        # independent of how many source buckets/clips happen to match.
        "total_samples": total_samples,
        # Per-recipe generation knobs that used to live at the page level.
        # `selected` defaults to True so existing recipes participate in
        # "Generate selected" until the user untoggles them.
        "prefix": str(raw.get("prefix") or "").strip(),
        "replace_existing": _bool_value(raw.get("replace_existing", False)),
        "frozen_test": frozen_test,
        "selected": _bool_value(raw.get("selected", True)),
        # Snapshot of the augmentation panel state at recipe-creation time.
        # When the recipe runs, these override the project defaults so two
        # recipes can use different pitch / SNR / mask thresholds. Empty dict
        # means "use whatever the project cfg has" (legacy recipes).
        "settings": _normalise_augmentation_settings(raw.get("settings") or {}, None)
                    if isinstance(raw.get("settings"), dict) else {},
        "updated_at": _now_iso(),
    }
    return recipe


def _labels_summary_from_filters(filters: list[dict]) -> str:
    labels: list[str] = []
    for rule in filters:
        raw = rule.get("labels", rule.get("label"))
        if isinstance(raw, str):
            labels.append(raw)
        elif isinstance(raw, list):
            labels.extend(str(item) for item in raw if str(item))
    labels = [label for label in labels if label and label != "*"]
    if not labels:
        return "*"
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    return ",".join(out)


def _count_library_filter(
    lib,
    rule: dict,
    *,
    project_id: str,
    frozen_test: bool = False,
) -> int:
    """How many library clips one source rule currently matches."""
    return len(library_paths_for_filters(
        lib,
        _labels_summary_from_filters([rule]),
        project_id=project_id,
        frozen_test=frozen_test,
        filters=[rule],
    ))


def _refresh_filter_rule_counts(
    lib,
    filters: list[dict],
    *,
    project_id: str,
    frozen_test: bool = False,
    trust_existing_counts: bool = False,
) -> bool:
    changed = False
    for rule in filters:
        if not isinstance(rule, dict):
            continue
        if trust_existing_counts and "count" in rule:
            continue
        count = _count_library_filter(
            lib,
            rule,
            project_id=project_id,
            frozen_test=frozen_test,
        )
        try:
            old_count = int(rule.get("count"))
        except Exception:  # noqa: BLE001
            old_count = None
        if old_count != count:
            rule["count"] = count
            changed = True
    return changed


def _refresh_recipe_counts(
    recipe: dict,
    *,
    selection: dict,
    cfg: Config | None,
    project_id: str,
    lib=None,
    trust_existing_counts: bool = False,
) -> dict:
    settings = _normalise_augmentation_settings(
        recipe.get("settings") if isinstance(recipe.get("settings"), dict)
        else selection.get("augmentation_settings", {}),
        cfg,
    )
    base_filters = recipe.get("base_filters") if isinstance(recipe.get("base_filters"), list) else []
    if not base_filters and isinstance(recipe.get("base_filter"), dict):
        base_filters = [recipe["base_filter"]]
        recipe["base_filters"] = base_filters
    overlay_filters = recipe.get("overlay_filters") if isinstance(recipe.get("overlay_filters"), list) else []
    if not overlay_filters and isinstance(recipe.get("overlay_filter"), dict) and recipe.get("overlay_filter"):
        overlay_filters = [recipe["overlay_filter"]]
        recipe["overlay_filters"] = overlay_filters
    # The recipe's overlay bucket is authoritative: a selected bucket means the
    # overlay stage runs, regardless of the panel toggle captured in settings.
    # Only a recipe without overlay filters can be pitch-only.
    overlay_enabled = bool(overlay_filters) or bool(settings.get("overlay_enabled", True))
    if overlay_enabled and not overlay_filters:
        raise ValueError("Overlay is enabled, but no overlay bucket is selected.")
    settings["overlay_enabled"] = overlay_enabled

    if lib is None:
        from sed.data.library import Library
        lib = Library()
    frozen = bool(recipe.get("frozen_test", False))
    _refresh_filter_rule_counts(
        lib,
        base_filters,
        project_id=project_id,
        frozen_test=frozen,
        trust_existing_counts=trust_existing_counts,
    )
    if overlay_enabled:
        _refresh_filter_rule_counts(
            lib,
            overlay_filters,
            project_id=project_id,
            frozen_test=frozen,
            trust_existing_counts=trust_existing_counts,
        )
    recipe["base"] = _labels_summary_from_filters(base_filters)
    recipe["overlay"] = _labels_summary_from_filters(overlay_filters) if overlay_enabled else "pitch"
    recipe["base_count"] = sum(int(rule.get("count") or 0) for rule in base_filters)
    recipe["overlay_count"] = (
        sum(int(rule.get("count") or 0) for rule in overlay_filters)
        if overlay_enabled
        else 0
    )
    recipe["settings"] = settings
    return recipe


def _overlay_label_from_sidecar(wav_path: Path) -> str:
    """Read just the overlay class from a derived clip's sidecar (cheap)."""
    sidecar = wav_path.with_suffix(".json")
    if not sidecar.exists():
        return ""
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    params = meta.get("augmentation_params") or {}
    return str(
        meta.get("overlay_label")
        or params.get("actual_overlay_label")
        or params.get("overlay_label")
        or ""
    )


def _overlay_labels_for_rule(lib, rule: dict, *, project_id: str) -> list[str]:
    """Distinct overlay classes among the derived clips a source rule matches."""
    try:
        paths = library_paths_for_filters(
            lib,
            _labels_summary_from_filters([rule]),
            project_id=project_id,
            frozen_test=False,
            filters=[rule],
        )
    except Exception:  # noqa: BLE001
        return []
    seen: set[str] = set()
    out: list[str] = []
    for path, _label in paths:
        overlay = _overlay_label_from_sidecar(path)
        if overlay and overlay not in seen:
            seen.add(overlay)
            out.append(overlay)
    return sorted(out)


def _attach_overlay_labels(selection: dict, *, project_id: str) -> None:
    """Annotate derived (augmented) source rules with their distinct overlay
    classes so the Inspect UI can show + search them. Display-only; not saved."""
    assignments = selection.get("assignments")
    if not isinstance(assignments, dict):
        return
    from sed.data.library import Library
    lib = Library()
    for rows in assignments.values():
        rule_list = [rows] if isinstance(rows, dict) else rows
        if not isinstance(rule_list, list):
            continue
        for rule in rule_list:
            if not isinstance(rule, dict):
                continue
            if str(rule.get("source_bucket") or "").lower() == "derived":
                rule["overlay_labels"] = _overlay_labels_for_rule(
                    lib, rule, project_id=project_id
                )
            else:
                rule.pop("overlay_labels", None)


def _refresh_selection_counts(selection: dict, *, cfg: Config | None, project_id: str) -> bool:
    """Update display counts on UI-managed source rules and recipes from disk."""
    from sed.data.library import Library

    changed = False
    lib = Library()
    assignments = selection.get("assignments", {})
    if isinstance(assignments, dict):
        for rows in assignments.values():
            if isinstance(rows, dict):
                rows = [rows]
            if isinstance(rows, list):
                changed |= _refresh_filter_rule_counts(
                    lib,
                    [row for row in rows if isinstance(row, dict)],
                    project_id=project_id,
                )

    recipes = selection.get("augmentation_recipes", [])
    if isinstance(recipes, list):
        for recipe in recipes:
            if not isinstance(recipe, dict):
                continue
            before = json.dumps(recipe, sort_keys=True, default=str)
            try:
                _refresh_recipe_counts(
                    recipe,
                    selection=selection,
                    cfg=cfg,
                    project_id=project_id,
                    lib=lib,
                )
            except ValueError:
                continue
            if json.dumps(recipe, sort_keys=True, default=str) != before:
                changed = True
    return changed


_AUG_BOOL_FIELDS = {
    "enabled",
    "pitch_shift_enabled",
    "overlay_enabled",
    "scale_noise_to_snr",
    "prevent_clipping",
    "mask_filter_enabled",
    "mix_map_enabled",
}
_AUG_INT_FIELDS = {
    "seed",
    "transient_window_ms",
    "mask_candidate_attempts",
    "mask_band_count",
    "mask_top_frames",
}
_AUG_FLOAT_FIELDS = {
    "samples_per_source",
}
_AUG_OPTIONAL_FLOAT_FIELDS = {
    "mask_min_mean_db",
    "mask_min_p25_db",
    "mask_min_band_db",
    "mask_min_effective_snr_db",
}
_AUG_RANGE_FIELDS = {"semitones_range", "snr_db_range"}


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _range_value(value: object) -> list[float] | None:
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return None
    if len(parts) != 2:
        return None
    try:
        return [float(parts[0]), float(parts[1])]
    except Exception:  # noqa: BLE001
        return None


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _normalise_augmentation_settings(raw: object, cfg: Config | None = None) -> dict:
    settings = augmentation_settings_snapshot(cfg)
    if not isinstance(raw, dict):
        return settings
    for key in _AUG_BOOL_FIELDS:
        if key in raw:
            settings[key] = _bool_value(raw[key])
    for key in _AUG_INT_FIELDS:
        if key in raw and raw[key] not in (None, ""):
            try:
                settings[key] = int(float(raw[key]))
            except Exception:  # noqa: BLE001
                pass
    for key in _AUG_FLOAT_FIELDS:
        if key in raw and raw[key] not in (None, ""):
            try:
                settings[key] = float(raw[key])
            except Exception:  # noqa: BLE001
                pass
    for key in _AUG_OPTIONAL_FLOAT_FIELDS:
        if key in raw:
            settings[key] = _optional_float(raw[key])
    for key in _AUG_RANGE_FIELDS:
        if key in raw:
            value = _range_value(raw[key])
            if value is not None:
                settings[key] = value
    return settings


def _apply_augmentation_settings(cfg: Config, settings: dict) -> None:
    clean = _normalise_augmentation_settings(settings, cfg)
    for key, value in clean.items():
        if not hasattr(cfg.augmentation, key):
            continue
        if key in _AUG_RANGE_FIELDS:
            setattr(cfg.augmentation, key, tuple(value))
        else:
            setattr(cfg.augmentation, key, value)


def _ensure_ui_selection(selection: dict) -> dict:
    selection["schema_version"] = 2
    selection["ui_managed"] = True
    selection.setdefault("assignments", {})
    selection.setdefault("clip_assignments", {})
    selection.setdefault("clip_overrides", {})
    selection.setdefault("augmentation_recipes", [])
    selection.setdefault("augmentation_settings", {})
    selection["updated_at"] = _now_iso()
    return selection


def _dataset_counts(layout: ProjectLayout) -> dict:
    """Row counts (minus header) for the composed train (pool) / test CSVs.

    ``train`` here is the whole train+val pool — val is not composed ahead
    of time, so it has no CSV of its own. Use ``_preview_train_val_counts``
    for a train/val estimate.
    """
    out: dict[str, int] = {}
    for name in ("train", "test"):
        p = layout.dataset_dir / f"{name}.csv"
        n = 0
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    n = max(0, sum(1 for _ in f) - 1)
            except Exception:  # noqa: BLE001
                n = 0
        out[name] = n
    return out


def _dataset_shape(layout: ProjectLayout) -> dict:
    """Cheap structural summary of the composed train pool.

    Sizes the Model tab's per-knob recommendations, which depend on more than
    the row count: ``min_class_groups`` (the smallest per-class number of
    independent recording sessions) is the ceiling on generalization, so a
    corpus with many clips but few groups must be advised like a smaller one.

    This reads only ``dataset/train.csv`` — one pass, no library scan — so it is
    safe to call from a page-load endpoint.
    """
    empty = {"rows": 0, "classes": [], "by_class": {}, "n_classes": 0,
             "negative_share": 0.0, "min_class_groups": 0, "min_class_rows": 0,
             "negative_origins": []}
    path = layout.dataset_dir / "train.csv"
    if not path.exists():
        return empty
    import csv as _csv
    from collections import Counter, defaultdict
    counts: Counter = Counter()
    groups: dict[str, set] = defaultdict(set)
    # Which library buckets `_negative` is made of. Some of them are room tone
    # and some are near-misses; the advice layer can only suggest splitting the
    # second kind out if it can see the names.
    negative_origins: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                name = str(row.get("class_name") or "")
                if not name:
                    continue
                counts[name] += 1
                session = str(row.get("session_id") or "")
                if session:
                    groups[name].add(session)
                if name == "_negative":
                    origin = str(row.get("origin_class") or "").strip()
                    if origin:
                        negative_origins.add(origin)
    except Exception:  # noqa: BLE001 — advisory input only
        return empty
    if not counts:
        return empty
    total = sum(counts.values())
    group_sizes = [len(s) for s in groups.values() if s]
    return {
        "rows": total,
        "classes": sorted(counts),
        "by_class": dict(counts),
        "n_classes": len(counts),
        "negative_share": round(counts.get("_negative", 0) / total, 4) if total else 0.0,
        "min_class_groups": min(group_sizes) if group_sizes else 0,
        "min_class_rows": min(counts.values()),
        "negative_origins": sorted(negative_origins),
    }


def _preview_train_val_split(layout: ProjectLayout, cfg: Config) -> dict:
    """Live train/val split preview matching what train time will do.

    Includes per-class counts and current split diagnostics so the Train tab
    can make grouped-split surprises visible (for example a class with zero
    validation rows because every source bucket has only one session).
    """
    configured_classes = [str(name) for name in cfg.model.classes]

    def _empty() -> dict:
        zeros = {name: 0 for name in configured_classes}
        return {
            "train": 0,
            "val": 0,
            "train_by_class": dict(zeros),
            "val_by_class": dict(zeros),
            "pool_source_counts": {"dataset": 0, "augmented": 0},
            "diagnostics": {},
        }

    path = layout.dataset_dir / "train.csv"
    if not path.exists():
        return _empty()
    import csv as _csv
    from collections import Counter
    from sed.data.splits import split_pool_rows_detailed
    with open(path, "r", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    val_fraction = float(getattr(cfg.dataset, "val_fraction", 0.15) or 0.15)
    train_rows, val_rows, diagnostics = split_pool_rows_detailed(
        rows, val_frac=val_fraction, seed=cfg.training.seed,
    )
    row_classes = sorted({
        str(row.get("class_name") or "") for row in rows
        if str(row.get("class_name") or "")
    })
    class_order = list(dict.fromkeys(configured_classes + row_classes))

    def _by_class(split_rows: list[dict]) -> dict[str, int]:
        counts = Counter(str(row.get("class_name") or "") for row in split_rows)
        return {name: int(counts.get(name, 0)) for name in class_order}

    pool_source_counts = {"dataset": 0, "augmented": 0}
    for row in rows:
        origin = str(row.get("origin") or "").strip().lower()
        key = "augmented" if origin in {"derived", "augmented"} else "dataset"
        pool_source_counts[key] += 1

    return {
        "train": len(train_rows),
        "val": len(val_rows),
        "train_by_class": _by_class(train_rows),
        "val_by_class": _by_class(val_rows),
        "pool_source_counts": pool_source_counts,
        "diagnostics": diagnostics,
    }


def _preview_train_val_counts(layout: ProjectLayout, cfg: Config) -> tuple[int, int]:
    """Backward-compatible total-count view of the live grouped split."""
    preview = _preview_train_val_split(layout, cfg)
    return int(preview["train"]), int(preview["val"])


def _parse_scaling_stats_param(raw: str) -> dict[str, list[float]]:
    """Parse the JSON-encoded ``scaling_stats`` query parameter (defensive)."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:  # noqa: BLE001 — a malformed preview param renders unscaled
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, list[float]] = {}
    for key, vals in parsed.items():
        try:
            out[str(key)] = [float(v) for v in (vals or [])]
        except (TypeError, ValueError):
            continue
    return out


def _fitted_scaling_training_readiness(
    cfg: Config, layout: ProjectLayout,
) -> dict[str, Any]:
    """Expose the exact fitted-scaling launch gate to the Inspect UI."""
    from sed.config import resolved_features
    from sed.train.train import _effective_scaling_method

    features = resolved_features(cfg.features)
    # Only constants that are meant to be fitted can be stale; fixed contract
    # constants (the AudioSet affine, the legacy dB window) have nothing to fit.
    enabled = (
        _effective_scaling_method(features) != "none"
        and str(getattr(features, "scaling_source", "dataset")) == "dataset"
    )
    if not enabled:
        return {"enabled": False, "ready": True, "problems": [], "auto_repairable": False}
    try:
        from sed.train.train import _validate_fitted_feature_scaling

        _validate_fitted_feature_scaling(cfg, layout.dataset_dir / "train.csv")
    except RuntimeError as exc:
        return {
            "enabled": True,
            "ready": False,
            "problems": [str(exc)],
            # A parent run's frontend is an encoder contract.  It must be
            # adopted exactly, never silently re-estimated on child data.
            "auto_repairable": str(cfg.model.initialization) != "parent",
        }
    return {"enabled": True, "ready": True, "problems": [], "auto_repairable": False}


def _refresh_stale_fitted_feature_scaling(
    cfg: Config, layout: ProjectLayout,
) -> dict[str, Any] | None:
    """Refresh stale scratch statistics before a Train-tab launch.

    The Train form deliberately saves its hyperparameters only when Start is
    pressed.  A changed seed or validation fraction therefore changes the
    grouped training subset at that same moment.  Refit here, after the new
    settings are durable and before spawning ``sed train``, so the child can
    never observe a half-updated training contract.
    """
    readiness = _fitted_scaling_training_readiness(cfg, layout)
    if not readiness["enabled"] or readiness["ready"]:
        return None
    if not readiness["auto_repairable"]:
        raise RuntimeError(" ".join(readiness["problems"]))

    from sed.data.feature_stats import fit_feature_scaling

    stats = fit_feature_scaling(
        cfg,
        layout,
        features=cfg.features,
        sample_rate=int(cfg.model.sample_rate),
    )
    refreshed_features = cfg.features.model_copy(update={
        "scaling_stats": stats["stats"],
        "standardize_stats_resolved": True,
        "standardize_stats_dataset_sha256": stats["dataset_sha256"],
        "standardize_stats_split_seed": stats["split_seed"],
        "standardize_stats_val_fraction": stats["val_fraction"],
        "standardize_stats_clips": stats["clips"],
    })
    from sed.config import update_config_toml_values

    update_config_toml_values(layout.config_toml, {
        "features": refreshed_features.model_dump(),
    })
    return stats


# A replicate sweep is bounded so a mistyped number cannot queue a week of GPU
# time. Ten is already well past the point where the spread stops moving.
MAX_REPLICATE_SEEDS = 20


def _replicate_seeds(body: dict[str, Any], base_seed: int) -> list[int]:
    """The seeds one Start press should train, in order.

    A single run is one draw from a distribution. Ranking two architectures on
    a 2-point frozen-test gap between two single runs is not a measurement —
    there is nothing to compare the gap against. Training the *same* config
    several times with only the seed changed gives that missing denominator:
    the spread across replicates is the noise floor, and a difference smaller
    than it is not a result.

    The first seed is always the one configured, so a 1-replicate sweep is
    byte-identical to the single run this used to be, and replicate 1 of a
    larger sweep is the run the user would have got anyway.
    """
    raw = body.get("seeds")
    if isinstance(raw, list) and raw:
        seeds: list[int] = []
        for value in raw:
            try:
                seed = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"seed must be an integer, got {value!r}") from None
            if seed not in seeds:
                seeds.append(seed)
        if len(seeds) > MAX_REPLICATE_SEEDS:
            raise ValueError(f"at most {MAX_REPLICATE_SEEDS} seeds per sweep (got {len(seeds)})")
        return seeds

    try:
        count = int(body.get("replicates") or 1)
    except (TypeError, ValueError):
        raise ValueError("replicates must be an integer") from None
    if count < 1:
        raise ValueError("replicates must be at least 1")
    if count > MAX_REPLICATE_SEEDS:
        raise ValueError(f"at most {MAX_REPLICATE_SEEDS} replicates per sweep (got {count})")
    # Consecutive from the configured seed: reproducible from the run history
    # alone, and it keeps replicate 1 identical to a plain single run.
    return [base_seed + i for i in range(count)]


def _train_recipe_signature(cfg) -> str:
    """recipe_signature for the live config (matches the runs-table signature)."""
    try:
        from sed.train.model_lab import recipe_from_config, recipe_signature
        return recipe_signature(recipe_from_config(cfg))
    except Exception:  # noqa: BLE001 — advisory only
        return ""


def _train_architecture_dropout(cfg) -> dict:
    """Complete effective dropout stack for the selected training policy."""
    try:
        from sed.train.fine_tuning import resolve_fine_tune_plan
        from sed.train.model_lab import architecture_dropout_info, effective_head_dropout

        byol = str(cfg.training.pretraining_objective or "supervised") == "byol"
        target_depth = "full" if byol else resolve_fine_tune_plan(cfg).target_depth
        info = architecture_dropout_info(
            str(cfg.model.recipe_id or cfg.model.backbone or ""),
            dict(cfg.model.params or {}),
            str(getattr(cfg.project, "id", "") or ""),
            train_depth=target_depth,
        )
        effective = effective_head_dropout(cfg)
        classifier = {
            "name": "head_dropout", "label": "classifier dropout",
            "mechanism": "dropout", "location": "classifier_head",
            "value": effective, "configured": float(cfg.training.head_dropout),
            "default": 0.2, "fixed": False, "owner": "train",
            "auto": bool(cfg.training.head_dropout_auto),
            "active": (not byol) and effective > 0.0,
            "help": (
                "Zeros values immediately before the shared classifier. "
                "It is unused by BYOL and off at evaluation/export."
            ),
        }
        info["target_train_depth"] = target_depth
        info["rows"] = [classifier, *(info.get("rows") or [])]
        info["classifier"] = classifier
        info["not_additive"] = True
        return info
    except Exception:  # noqa: BLE001 — advisory readout only
        return {
            "architecture_id": "", "label": "", "rows": [],
            "recommended_head_dropout": 0.2, "not_additive": True,
        }


def _backbone_has_pretrained_weights(backbone: str) -> bool:
    """Whether the selected backbone/recipe can actually load upstream weights."""
    try:
        from sed.config import backbone_pretrained_available
        if backbone_pretrained_available(backbone):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from sed.train.backbones import get_spec
        return bool(get_spec(str(backbone)).pretrained_available)
    except Exception:  # noqa: BLE001
        return False


def _train_has_transfer_weights(cfg: Config) -> bool:
    return (
        cfg.model.initialization == "parent"
        or (
            cfg.model.initialization != "random"
            and bool(cfg.model.pretrained)
            and _backbone_has_pretrained_weights(cfg.model.backbone)
        )
    )


def _model_selection_augmentation_defaults(
    cfg: Config,
    *,
    architecture_id: str,
    initialization: str,
) -> dict[str, bool]:
    """Non-destructive data-augmentation defaults for a model workflow.

    Pitch/overlay processing is offline, before composition, so the v2 venue
    fine-tune rule belongs with recipe selection rather than TrainingCfg. Only
    fill fields absent from the project TOML; an operator's explicit choice is
    never overwritten.
    """
    if str(architecture_id) != "clink-cnn-v2" or str(initialization) != "parent":
        return {}
    explicit = set(cfg.augmentation.model_fields_set)
    updates: dict[str, bool] = {}
    if "pitch_shift_enabled" not in explicit:
        updates["pitch_shift_enabled"] = False
    if "overlay_enabled" not in explicit:
        updates["overlay_enabled"] = True
    return updates


def _train_model_probe(cfg: Config) -> dict[str, Any] | None:
    """Probe the selected encoder once for Train-tab capability metadata."""
    try:
        from sed.train.backbones import get_spec, model_input_geometry, probe_spec

        geom = model_input_geometry(cfg)
        return probe_spec(
            get_spec(cfg.model.backbone),
            n_bands=int(geom["n_bands"]),
            n_frames=int(geom["n_frames"]),
            input_channels=int(geom["input_channels"]),
            params=dict(cfg.model.params),
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("fine-tune policy probe failed: %s", exc)
        return None


def _train_stage_count_from_probe(probe: dict[str, Any] | None) -> int | None:
    try:
        n_stages = int((probe or {}).get("n_stages") or 0)
    except (TypeError, ValueError):
        n_stages = 0
    return n_stages if n_stages > 0 else None


def _train_stage_count(cfg: Config) -> int | None:
    """Probe the exact encoder stage count for the current train recipe."""
    cached = cfg.__dict__.get("_inspect_train_model_probe", ...)
    return _train_stage_count_from_probe(
        _train_model_probe(cfg) if cached is ... else cached
    )


def _run_config_summary(layout: ProjectLayout, model_id: str) -> dict:
    """Backbone + key training/feature knobs a model was trained with, for the
    runs comparison table. Read from the frozen training_config.json."""
    try:
        with open(layout.model_dir(model_id) / "training_config.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return {"backbone": "", "pretrained": None, "train_depth": "", "feature": ""}
    model = data.get("model") or {}
    training = data.get("training") or {}
    feat = data.get("features") or {}
    ftype = str(feat.get("feature_type") or "mel")
    sample_rate = model.get("sample_rate", feat.get("sample_rate", "?"))
    try:
        from sed.audio.features import feature_band_count
        band_features = SimpleNamespace(
            feature_type=ftype,
            n_fft=int(feat.get("n_fft", 0) or 0),
            n_mels=int(feat.get("n_mels", 0) or 0),
            fmin=float(feat.get("fmin", 0) or 0),
            fmax=float(feat.get("fmax", 0) or 0),
            colormap=str(feat.get("colormap") or "none"),
            image_size=int(feat.get("image_size", 96) or 96),
        )
        bins = feature_band_count(band_features, int(sample_rate))
    except Exception:  # noqa: BLE001 — compact label only; keep endpoint robust
        bins = feat.get("n_mels") if ftype != "linear" else (
            (int(feat.get("n_fft", 0)) // 2 + 1) if feat.get("n_fft") else ""
        )
    feature = f"{ftype} {bins}b @ {sample_rate}Hz / {feat.get('window_seconds', '?')}s"
    signature = ""
    try:
        from sed.train.model_lab import recipe_signature_from_payload
        signature = recipe_signature_from_payload(model, layout.id)
    except Exception:  # noqa: BLE001 — signature is advisory for comparison
        signature = ""
    # Seed replication: runs sharing an experiment_key differ only by seed, so
    # the table can report mean ± spread instead of one draw per architecture.
    from sed.train.seed_groups import experiment_key, run_seed
    return {
        "backbone": str(model.get("backbone") or ""),
        "pretrained": bool(model.get("pretrained")) if "pretrained" in model else None,
        "train_depth": str(training.get("train_depth") or "full"),
        "loss": str(training.get("loss") or ""),
        "feature": feature,
        "recipe_signature": signature,
        "experiment_key": experiment_key(data),
        "seed": run_seed(data),
        # Trained class space — two runs on different classes are not comparable,
        # and a run cannot be scored on a test set holding classes it lacks.
        "classes": [str(c) for c in (model.get("classes") or [])],
        # Pretrain vs fine-tune, promoted from the detail view: the two stages
        # of this project's workflow are read very differently.
        "initialization": str(model.get("initialization") or ""),
        "parent_project": str(model.get("parent_project") or ""),
        "parent_run": str(model.get("parent_run") or ""),
    }


def _test_set_names(layout: ProjectLayout) -> dict[str, str]:
    """Map test-set hash -> the name it was saved under.

    Scores are stamped with a hash, which answers "same ground truth?" but not
    "which benchmark was that?". Naming the hashes lets the runs table say a run
    was scored on "benchmark v2 clean" instead of `sha256:9f3a…`.
    """
    try:
        from sed.data.frozen_test_presets import frozen_test_preset_state

        state = frozen_test_preset_state(layout)
    except Exception:  # noqa: BLE001 — a display nicety must not break the list
        return {}
    return {p["hash"]: p["name"] for p in state.get("presets", []) if p.get("hash")}


def _score_store_summary(model_dir: Path) -> list[dict]:
    """Stored score passes for one run, compacted for the runs list.

    Cheap: reads the JSON sidecars only, never the probability matrices.
    """
    try:
        from sed.train import score_store

        return [
            {
                "key": key,
                "split": side.get("split") or "test",
                "engine": side.get("engine") or "",
                "ir_precision": side.get("ir_precision"),
                "test_set_hash": side.get("test_set_hash") or "",
                "n": side.get("n"),
                "source": side.get("source") or "scored",
            }
            for key, side in score_store.available(model_dir).items()
        ]
    except Exception:  # noqa: BLE001 — a missing store is not an error
        return []


def _scored_test_sets(layout: ProjectLayout, runs: list[dict]) -> list[dict]:
    """Every frozen test set the loaded runs hold scores for, newest use first.

    This is what turns "these runs were scored on different test sets" from a
    warning into a choice: the table reports on one set, and says which runs are
    missing from it.
    """
    names = _test_set_names(layout)
    current = str((_current_frozen_test(layout) or {}).get("hash") or "")
    by_hash: dict[str, dict] = {}
    for run in runs:
        for entry in run.get("scores") or []:
            if entry.get("split") != "test":
                continue
            test_hash = str(entry.get("test_set_hash") or "")
            if not test_hash:
                continue
            row = by_hash.setdefault(test_hash, {
                "hash": test_hash,
                "name": names.get(test_hash, ""),
                "n_samples": entry.get("n"),
                "runs": 0,
                "is_current": test_hash == current,
            })
            row["runs"] += 1
    ordered = sorted(by_hash.values(), key=lambda r: (not r["is_current"], -r["runs"]))
    if current and current not in by_hash:
        # The current set exists but nothing has been scored on it yet — it must
        # still be selectable, or there is no way to ask for that scoring.
        ordered.insert(0, {
            "hash": current, "name": names.get(current, ""),
            "n_samples": int((_current_frozen_test(layout) or {}).get("total") or 0),
            "runs": 0, "is_current": True,
        })
    return ordered


def _current_test_health(layout: ProjectLayout) -> dict:
    """Whether the current frozen test can answer the question asked of it."""
    try:
        from sed.train.testset_health import health_for_test_csv

        return health_for_test_csv(layout.dataset_dir / "test.csv")
    except Exception:  # noqa: BLE001
        return {}


def _run_lineage(layout: ProjectLayout, model_id: str, *, max_depth: int = 12) -> list[dict]:
    """Ancestor chain of a run, nearest parent first.

    The two-stage workflow (domain pretrain -> venue fine-tune) means a run's
    result depends on a model trained elsewhere, possibly in another project.
    Without the chain a fine-tune's score is unattributable.
    """
    chain: list[dict] = []
    seen: set[tuple[str, str]] = set()
    project_id, run_id = layout.id, model_id
    for _ in range(max_depth):
        try:
            cfg_path = ProjectLayout(project_id).model_dir(run_id) / "training_config.json"
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a missing ancestor ends the chain
            break
        model = data.get("model") or {}
        parent_run = str(model.get("parent_run") or "")
        if not parent_run:
            break
        parent_project = str(model.get("parent_project") or project_id)
        key = (parent_project, parent_run)
        if key in seen:  # a cycle would otherwise spin to max_depth
            break
        seen.add(key)
        entry = {
            "project": parent_project, "run": parent_run,
            "exists": (ProjectLayout(parent_project).model_dir(parent_run)).exists(),
            "backbone": "", "classes": [], "initialization": "",
        }
        try:
            pdata = json.loads(
                (ProjectLayout(parent_project).model_dir(parent_run) / "training_config.json")
                .read_text(encoding="utf-8")
            )
            pmodel = pdata.get("model") or {}
            entry["backbone"] = str(pmodel.get("backbone") or "")
            entry["classes"] = [str(c) for c in (pmodel.get("classes") or [])]
            entry["initialization"] = str(pmodel.get("initialization") or "")
        except Exception:  # noqa: BLE001 — record the link even if unreadable
            pass
        chain.append(entry)
        if not entry["exists"]:
            break
        project_id, run_id = parent_project, parent_run
    return chain


def _current_frozen_test(layout: ProjectLayout) -> dict:
    """Hash + size of the project's CURRENT frozen test set (dataset snapshot).

    Lets the runs table flag metrics that were scored on an older frozen test —
    cheap JSON read, safe for page-load endpoints."""
    try:
        with open(layout.dataset_dir / "dataset_snapshot.json", encoding="utf-8") as f:
            snap = json.load(f)
        frozen = snap.get("frozen_test") or {}
        return frozen if isinstance(frozen, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _augmentation_truth(model_dir: Path, manifest: dict) -> dict:
    """Whether augmented (derived) clips were actually part of a run's dataset.

    The frozen training_config.json only records the offline recipe-builder
    settings (enabled flags default to true), so it cannot answer "was this
    run augmented?". The truth is the per-origin clip counts: fast path is
    manifest dataset.by_origin (newer runs), fallback is the run's copied
    dataset/dataset_snapshot.json (~1 MB — only parse on per-run endpoints).
    """
    by_origin = (manifest.get("dataset") or {}).get("by_origin") or {}
    if not by_origin:
        try:
            with open(model_dir / "dataset" / "dataset_snapshot.json", "r", encoding="utf-8") as f:
                by_origin = json.load(f).get("by_origin") or {}
        except Exception:  # noqa: BLE001
            by_origin = {}
    if not by_origin:
        return {"available": False}
    by_class = {}
    derived_total = 0
    clip_total = 0
    for cls, counts in by_origin.items():
        if not isinstance(counts, dict):
            continue
        derived = int(counts.get("derived") or 0)
        total = int(counts.get("total") or 0)
        by_class[cls] = {"derived": derived, "total": total}
        derived_total += derived
        clip_total += total
    return {
        "available": True,
        "applied": derived_total > 0,
        "derived_total": derived_total,
        "clip_total": clip_total,
        "by_class": by_class,
    }


def _confusion_asset_key(report_name: str) -> str:
    """`/api/train/asset?file=` value for one evaluation's confusion matrix."""
    name = str(report_name or "test")
    return "confusion_matrix" if name == "test" else f"confusion_matrix:{name}"


def _score_summary(block: dict) -> dict:
    """One scored block, compacted for the runs table.

    Per-class detail is deliberately dropped — the list endpoint is page-load
    latency for every run in the project; the detail endpoint serves the full
    blocks for one run. `read_*_test_evals` has already filled macro P/R and the
    false-fire rate from `per_class` for blocks written before those keys, so
    old runs populate the same columns as new ones.
    """
    return {
        "rule": block.get("rule"),
        "f1": block.get("f1_macro_targets"),
        "ap": block.get("ap_macro_targets"),
        "accuracy": block.get("accuracy"),
        "precision": block.get("precision_macro_targets"),
        "recall": block.get("recall_macro_targets"),
        "false_fire_rate": block.get("false_fire_rate"),
        "n_samples": block.get("n_samples"),
        # The prior every prior-bound metric in this block was computed at.
        # Two blocks are only comparable on precision/F1/accuracy/AP when these
        # match; recall and the per-negative rate are unaffected by it.
        "negative_prior": block.get("negative_prior"),
        "n_negatives": block.get("n_negatives"),
        "test_set_hash": block.get("test_set_hash") or "",
        "origin": block.get("origin") or "rule",
        "thresholds_source": block.get("thresholds_source") or "",
    }


def _frozen_test_summary(manifest: dict) -> dict[str, dict]:
    """Frozen-test checkpoint scores keyed by rule id, for the runs table."""
    from sed.train.test_rules import read_frozen_test_evals

    return {rid: _score_summary(b) for rid, b in read_frozen_test_evals(manifest).items()}


def _deployed_test_summary(manifest: dict) -> dict[str, dict]:
    """Exported-IR scores keyed "<precision>:<rule id>", for the runs table."""
    from sed.train.test_rules import read_deployed_test_evals

    out: dict[str, dict] = {}
    for key, block in read_deployed_test_evals(manifest).items():
        row = _score_summary(block)
        # `precision` in the compacted shape is macro-precision; the exported
        # variant travels as `ir_precision` so the two can never collide.
        row["ir_precision"] = block.get("ir_precision") or ""
        row["model_file"] = block.get("model_file") or ""
        out[key] = row
    return out


def _run_epoch_metrics(model_dir: Path) -> list[dict] | None:
    """Per-epoch metrics from the run's training report, for compare charts."""
    try:
        with open(model_dir / "reports" / "training_report.json", "r", encoding="utf-8") as f:
            metrics = json.load(f).get("epoch_metrics")
        return metrics if isinstance(metrics, list) else None
    except Exception:  # noqa: BLE001
        return None


def _parse_train_metrics(model_dir: Path) -> list[dict]:
    """Parse a run's Lightning CSV metrics into per-epoch rows for the live chart."""
    reports = model_dir / "reports"
    csv_path = reports / "training" / "metrics.csv"
    if not csv_path.exists():
        hits = list(reports.glob("**/metrics.csv")) if reports.exists() else []
        csv_path = hits[0] if hits else None
    if csv_path is None or not csv_path.exists():
        return []

    def _num(row: dict, *keys: str):
        for key in keys:
            v = row.get(key)
            if v not in (None, ""):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    by_epoch: dict[int, dict] = {}
    current_epoch = 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ep_raw = row.get("epoch")
                if ep_raw not in (None, ""):
                    try:
                        current_epoch = int(float(ep_raw))
                    except (TypeError, ValueError):
                        pass

                ep = current_epoch
                rec = by_epoch.setdefault(ep, {"epoch": ep, "values": {}})
                for key, value in row.items():
                    if key in {"epoch"} or value in (None, ""):
                        continue
                    try:
                        rec["values"][key] = float(value)
                    except (TypeError, ValueError):
                        continue
                tl = _num(row, "train/loss_epoch", "train/loss")
                if tl is not None:
                    rec["train_loss"] = tl
                vl = _num(row, "val/loss")
                if vl is not None:
                    rec["val_loss"] = vl
                va = _num(row, "val/accuracy")
                vap = _num(row, "val/ap_macro_targets")
                vf = _num(row, "val/f1_macro_targets")
                vbyol = _num(row, "val/byol_score")
                legacy_byol_score = vbyol is None and va is not None and vap is None and vf is None
                if va is not None and not legacy_byol_score:
                    rec["val_accuracy"] = va
                if vbyol is not None or legacy_byol_score:
                    rec["val_byol_score"] = vbyol if vbyol is not None else va
                if vap is not None:
                    rec["val_ap"] = vap
                if vf is not None:
                    rec["val_f1"] = vf
                for key, value in rec["values"].items():
                    if str(key).startswith(("lr-", "lr/")):
                        rec["lr"] = value
                        break
    except Exception:  # noqa: BLE001
        return []
    return [by_epoch[k] for k in sorted(by_epoch)]


def _find_new_model(layout: ProjectLayout, existing: set[str]) -> str:
    """Newest models/ subdir not present in `existing` (the just-started run)."""
    if not layout.models_dir.exists():
        return ""
    fresh = [p for p in layout.models_dir.iterdir() if p.is_dir() and p.name not in existing]
    if not fresh:
        return ""
    fresh.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return fresh[0].name


def _coerce_int_channels(raw: object, default: list[int]) -> list[int]:
    if raw in (None, ""):
        return list(default)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return [int(raw)]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        return [int(p) for p in parts] if parts else list(default)
    if isinstance(raw, list):
        return [int(p) for p in raw]
    return list(default)


def _validate_inference_channels(raw: object, default: list[int]) -> list[int]:
    channels = _coerce_int_channels(raw, default)
    if not channels:
        raise ValueError("select at least one input channel")
    if any(ch < 0 for ch in channels):
        raise ValueError("channels must be zero-based input indexes")
    if len(set(channels)) != len(channels):
        raise ValueError("channels must not contain duplicates")
    return channels


def _validate_multi_mic_mode(raw: object, channels: list[int], default: str = "max") -> str:
    mode = str(raw or default or "max").strip().lower()
    if mode not in {"max", "best", "single", "mean"}:
        raise ValueError("multi_mic_mode must be max, best, single, or mean")
    if mode == "single" and len(channels) != 1:
        raise ValueError("single microphone mode requires exactly one selected channel")
    return mode


def _coerce_optional_dbfs(raw: object, default: float | None = None) -> float | None:
    if raw in (None, ""):
        return default
    if isinstance(raw, str) and raw.strip().lower() in {"off", "none", "null", ""}:
        return None
    return float(raw)


def _coerce_bool(raw: object, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _coerce_targets(raw: object, *, known: set[str], default: set[str]) -> set[str]:
    if raw is None:
        return set(default)
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(part).strip() for part in raw if str(part).strip()]
    else:
        raise ValueError("targets must be a list or comma-separated string")
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError("unknown target class(es): " + ", ".join(unknown))
    return {c for c in values if c != "_negative"}


def _mic_label_for_channel(labels: list[str], channels: list[int], channel: int) -> str:
    try:
        order = channels.index(channel)
    except ValueError:
        order = -1
    if len(labels) == len(channels) and order >= 0:
        return labels[order]
    if 0 <= channel < len(labels):
        return labels[channel]
    if len(labels) == 1:
        return labels[0]
    return ""


def _model_openvino_summary(model_dir: Path) -> dict:
    try:
        from sed.runtime.manifest import resolve_model_path

        path = resolve_model_path(model_dir)
    except Exception as exc:  # noqa: BLE001
        return {
            "has_openvino": False,
            "openvino_precision": "",
            "openvino_file": "",
            "available_precisions": [p for p in ("fp16", "int8") if (model_dir / f"model_{p}.xml").exists()],
            "export_error": str(exc),
        }
    name = path.name
    precision = "int8" if "int8" in name else ("fp16" if "fp16" in name else ("fp32" if "fp32" in name else "ir"))
    available = [p for p in ("fp16", "int8") if (model_dir / f"model_{p}.xml").exists()]
    return {
        "has_openvino": True,
        "openvino_precision": precision,
        "openvino_file": name,
        "available_precisions": available,
        "export_error": "",
    }


def _export_run_ir(
    cfg, project_id: str, model_id: str,
    *,
    targets: set[str] | None = None,
    active_precision: str | None = None,
) -> dict:
    """Export ONNX + OpenVINO IR for an existing model bundle (batch job step).

    With default ``targets`` this skips bundles that already have an IR (the
    batch "export missing" behaviour). An explicit ``targets`` set (the detail
    export panel) always rebuilds so precision/format choices take effect. Uses
    the run's frozen config snapshot and its own copied dataset for INT8
    calibration, so the export is faithful to what was trained even if the
    project config moved on. Updates the manifest like the auto-export.
    """
    from sed.config import load_config_snapshot
    from sed.export.openvino import export_project
    from sed.train.registry import get_checkpoint, load_manifest, save_manifest

    layout = ProjectLayout(project_id)
    model_dir = layout.model_dir(model_id)
    if targets is None and (model_dir / "model_fp16.xml").exists():
        return {"skipped": "IR already exported"}
    snapshot = model_dir / "training_config.json"
    run_cfg = load_config_snapshot(snapshot) if snapshot.exists() else cfg
    ckpt_path = get_checkpoint(project_id, model_id)
    run_dataset = model_dir / "dataset"
    dataset_dir = run_dataset if (run_dataset / "train.csv").exists() else layout.dataset_dir
    export_info = export_project(
        run_cfg, project_id,
        ckpt_path=ckpt_path,
        out_dir=model_dir,
        dataset_dir=dataset_dir,
        targets=targets,
        active_precision=active_precision,
    )
    manifest = load_manifest(project_id, model_id)
    # Exported = at least one OpenVINO IR exists; an ONNX-only build still runs
    # live (ONNX/checkpoint fallback) but is not an "exported IR".
    has_ir = bool(export_info.get("active_openvino"))
    manifest["exported"] = has_ir
    # Merge, not replace: the block also carries `precision_decision` from
    # `sed models set-precision`, which a re-export must not wipe.
    manifest.setdefault("export", {}).update({
        "source_checkpoint": ckpt_path.name,
        "active_precision": export_info.get("active_precision"),
        "active_target": export_info.get("active_target"),
    })
    files = manifest.setdefault("files", {})
    if export_info.get("active_openvino"):
        files["active_openvino"] = export_info["active_openvino"]
    else:
        files.pop("active_openvino", None)
    if export_info.get("model_int8"):
        files["openvino_int8"] = export_info["model_int8"]
    else:
        files.pop("openvino_int8", None)
    if export_info.get("model_fp16"):
        files["openvino_fp16"] = export_info["model_fp16"]
    else:
        files.pop("openvino_fp16", None)
    if export_info.get("onnx"):
        files["onnx"] = export_info["onnx"]
    save_manifest(project_id, model_id, manifest)
    return {
        "active_precision": export_info.get("active_precision"),
        "int8": bool(export_info.get("model_int8")),
        "onnx_only": not has_ir,
    }


_LIBRARY_WAV_COUNT: tuple[float, int] | None = None  # (computed_at, count)


def _library_wav_count(ttl_s: float = 600.0) -> int:
    """Cached count of every WAV under the shared library sounds tree.

    The rglob over a 75k-clip library takes noticeable time — this is only ever
    called from the lazy /api/library/negatives-count endpoint (never from a
    page-load payload) and the result is cached for `ttl_s`.
    """
    global _LIBRARY_WAV_COUNT
    now = time.time()
    if _LIBRARY_WAV_COUNT is not None and now - _LIBRARY_WAV_COUNT[0] < ttl_s:
        return _LIBRARY_WAV_COUNT[1]
    from sed.config import LIBRARY_DIR

    root = LIBRARY_DIR / "sounds"
    count = sum(1 for _ in root.rglob("*.wav")) if root.exists() else 0
    _LIBRARY_WAV_COUNT = (now, count)
    return count


def _inference_models_payload(project_id: str, layout: ProjectLayout) -> dict:
    from sed.train import registry

    try:
        manifests = registry.load_manifests(project_id)
        active = registry.get_active_model(project_id) or ""
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "models": [], "active": "", "selected": ""}

    rows: list[dict] = []
    for man in manifests:
        model_id = str(man.get("model_id") or "")
        if not model_id:
            continue
        model_dir = layout.model_dir(model_id)
        ov = _model_openvino_summary(model_dir)
        summary = _run_config_summary(layout, model_id)
        files = man.get("files") or {}
        ckpt_name = str(files.get("checkpoint") or "")
        has_checkpoint = bool(
            (ckpt_name and (model_dir / ckpt_name).exists())
            or list(model_dir.glob("checkpoint_*.ckpt"))
        )
        has_onnx = (model_dir / "model.onnx").exists()
        has_config = (model_dir / "training_config.json").exists()
        calibrated = (model_dir / "runtime_settings.json").exists()
        # Engine chain: deployed OpenVINO IR → portable ONNX → raw checkpoint.
        # A run is live-runnable with ANY of them (sed.runtime.backends).
        engines = (
            [f"openvino:{p}" for p in ("int8", "fp16") if p in ov.get("available_precisions", [])]
            + (["onnx"] if has_onnx else [])
            + (["checkpoint"] if has_checkpoint else [])
        )
        can_run = bool(has_config and engines)
        reason = ""
        if not has_config:
            reason = "missing training_config.json"
        elif not engines:
            reason = "no runnable artifact (export or checkpoint required)"
        rows.append({
            "model_id": model_id,
            "description": man.get("description") or "",
            "created_at": man.get("created_at") or "",
            "active": model_id == active,
            "has_checkpoint": has_checkpoint,
            "has_onnx": has_onnx,
            "has_openvino": bool(ov["has_openvino"]),
            "openvino_precision": ov["openvino_precision"],
            "openvino_file": ov["openvino_file"],
            "available_precisions": ov.get("available_precisions", []),
            "engines": engines,
            "default_engine": engines[0] if engines else "",
            "calibrated": calibrated,
            "can_run": can_run,
            "disabled_reason": reason,
            "backbone": summary.get("backbone", ""),
            "feature": summary.get("feature", ""),
            "train_depth": summary.get("train_depth", ""),
            "pretrained": summary.get("pretrained"),
            "test_f1": (man.get("test_metrics") or {}).get("f1_macro_targets"),
            "test_ap": (man.get("test_metrics") or {}).get("ap_macro_targets"),
        })

    selected = ""
    active_row = next((r for r in rows if r["model_id"] == active and r["can_run"]), None)
    if active_row is not None:
        selected = active_row["model_id"]
    else:
        # No phantom fallback to a non-runnable row: an empty selection tells
        # the UI to disable Start with an actionable hint.
        first_run = next((r for r in rows if r["can_run"]), None)
        selected = first_run["model_id"] if first_run else ""
    return {"ok": True, "models": rows, "active": active, "selected": selected}


def _inference_device_payload(
    cfg: Config,
    requested_device: str | None = None,
    channels: list[int] | None = None,
) -> dict:
    from sed.audio.capture import list_input_devices, resolve_device

    import sounddevice as sd

    devices = list_input_devices()
    default_idx = None
    try:
        default_idx = sd.default.device[0]
    except Exception:  # noqa: BLE001
        default_idx = None
    selected_channels = (
        _validate_inference_channels(channels, list(cfg.inference.inference_channels))
        if channels is not None else list(cfg.inference.inference_channels)
    )
    required_channels = (max(selected_channels) + 1) if selected_channels else 1
    configured = str(cfg.deploy.record_device or "default")
    wanted = str(requested_device or configured or "default")
    resolved_idx = None
    resolved_error = ""
    try:
        resolved_idx = resolve_device(wanted)
    except Exception as exc:  # noqa: BLE001
        resolved_error = str(exc)

    rows = []
    resolved_device = None
    for d in devices:
        row = {
            "index": d.index,
            "name": d.name,
            "max_input_channels": d.max_input_channels,
            "default_samplerate": d.default_samplerate,
            "hostapi_name": d.hostapi_name,
            "is_default": d.index == default_idx,
            "is_resolved": d.index == resolved_idx,
            "enough_channels": d.max_input_channels >= required_channels,
        }
        if row["is_resolved"]:
            resolved_device = row
        rows.append(row)

    labels = list(cfg.audio.microphone_type)
    input_channels = []
    max_inputs = int((resolved_device or {}).get("max_input_channels") or required_channels or 1)
    for ch in range(max(0, max_inputs)):
        input_channels.append({
            "channel": ch,
            "label": _mic_label_for_channel(labels, selected_channels, ch),
            "selected": ch in selected_channels,
            "available": bool(resolved_device) and ch < int(resolved_device["max_input_channels"]),
        })

    warnings = []
    if resolved_error:
        warnings.append(f"configured device not found: {wanted}")
    elif resolved_device and not resolved_device["enough_channels"]:
        warnings.append(
            f"device has {resolved_device['max_input_channels']} input channel(s), "
            f"inference needs {required_channels}"
        )
    if selected_channels and len(set(selected_channels)) != len(selected_channels):
        warnings.append("selected input channels contain duplicates")

    return {
        "ok": True,
        "configured_device": configured,
        "requested_device": wanted,
        "resolved_index": resolved_idx,
        "resolved_device": resolved_device,
        "resolved_error": resolved_error,
        "devices": rows,
        "required_channels": required_channels,
        "record_sample_rate": cfg.audio.record_sample_rate,
        "blocksize": cfg.audio.blocksize,
        "record_channels": list(cfg.audio.record_channels),
        "inference_channels": list(selected_channels),
        "multi_mic_mode": cfg.inference.multi_mic_mode,
        "onset_gating_enabled": cfg.inference.onset_gating_enabled,
        "microphone_type": labels,
        "input_channels": input_channels,
        "warnings": warnings,
    }


def _active_inference_targets(cfg: Config) -> set[str]:
    """Classes that may fire a trigger — role == "target" only.

    Routed through `cfg.class_roles()` rather than reading `[targets].enabled`
    directly: that keeps this in sync with the real arbiter (`build_arbiter`),
    which already applies the suppressor name-heuristic default and the
    negative-role exclusion. Reading `[targets]` directly here used to miss
    both — an unconfigured `hardnegs`-named class showed as "active" even
    though it never fires, and an explicitly `enabled=false` class could
    still show as "active" via the empty-`[targets]` fallback branch.
    """
    known = set(cfg.model.classes)
    roles = cfg.class_roles(list(known))
    return {cls for cls, role in roles.items() if role == "target"}


def _inference_mining_classes(cfg: Config, active_targets: set[str], enabled: bool) -> set[str]:
    if not enabled:
        return set()
    known = set(cfg.model.classes)
    flagged = {c for c in cfg.mining_classes() if c in known}
    return flagged or {c for c in active_targets if c in known}


def _inference_settings_payload(
    project_id: str,
    layout: ProjectLayout,
    edge_cfg: Config,
    *,
    model_id: str = "",
    requested_device: str | None = None,
    channels: list[int] | None = None,
    multi_mic_mode: str | None = None,
) -> dict:
    models = _inference_models_payload(project_id, layout)
    selected_model = str(model_id or models.get("selected") or "").strip()
    model = next((m for m in models.get("models", []) if m.get("model_id") == selected_model), None)
    runtime_cfg = edge_cfg
    config_error = ""
    if selected_model and (layout.model_dir(selected_model) / "training_config.json").exists():
        try:
            from sed.config import load_config_for_inference_model

            runtime_cfg = load_config_for_inference_model(project_id, selected_model)
        except Exception as exc:  # noqa: BLE001
            config_error = str(exc)
            runtime_cfg = edge_cfg
    selected_channels = _validate_inference_channels(
        channels if channels is not None else runtime_cfg.inference.inference_channels,
        list(runtime_cfg.inference.inference_channels),
    )
    mode = _validate_multi_mic_mode(
        multi_mic_mode or runtime_cfg.inference.multi_mic_mode,
        selected_channels,
        default=edge_cfg.inference.multi_mic_mode,
    )
    runtime_cfg.inference.inference_channels = selected_channels
    runtime_cfg.inference.multi_mic_mode = mode
    device = requested_device or runtime_cfg.deploy.record_device or "default"
    device_payload = _inference_device_payload(runtime_cfg, device, selected_channels)
    active_targets = _active_inference_targets(runtime_cfg)
    mining_default = bool(runtime_cfg.mining.enabled)
    mining_classes = _inference_mining_classes(runtime_cfg, active_targets, mining_default)
    classes = list(runtime_cfg.model.classes)
    thresholds = runtime_cfg.effective_thresholds(classes)
    roles = runtime_cfg.class_roles(classes)
    vetoes = runtime_cfg.class_vetoes(classes)
    return {
        "ok": True,
        "model": model,
        "models": models.get("models", []),
        "active_model": models.get("active", ""),
        "selected_model": selected_model,
        "config_error": config_error,
        "classes": classes,
        "thresholds": thresholds,
        "active_targets": sorted(active_targets),
        "mining_classes": sorted(mining_classes),
        "target_rows": [
            {
                "class": cls,
                "enabled": cls in active_targets,
                "mine": cls in runtime_cfg.mining_classes(),
                "mine_threshold": runtime_cfg.mine_threshold(cls),
                "threshold": thresholds.get(cls),
                # Kept for older frontend builds; prefer "role".
                "negative": roles.get(cls) == "negative",
                "role": roles.get(cls),
                "veto_max": vetoes.get(cls),
            }
            for cls in classes
        ],
        "inference": {
            "channels": selected_channels,
            "multi_mic_mode": mode,
            "onset_gating_enabled": bool(runtime_cfg.inference.onset_gating_enabled),
            "min_dbfs": runtime_cfg.inference.min_dbfs,
            "hop_ms": runtime_cfg.inference.hop_ms,
            "window_seconds": runtime_cfg.features.window_seconds,
            "negative_max": runtime_cfg.inference.negative_max,
            "smoothing_enabled": runtime_cfg.inference.smoothing_enabled,
            "consecutive_frames": runtime_cfg.inference.consecutive_frames,
        },
        "audio": {
            "record_sample_rate": runtime_cfg.audio.record_sample_rate,
            "blocksize": runtime_cfg.audio.blocksize,
            "microphone_type": list(runtime_cfg.audio.microphone_type),
        },
        "device": device_payload,
        "artnet": {"configured": bool(runtime_cfg.artnet.enabled), "default_enabled": False},
        "mining": {"configured": bool(runtime_cfg.mining.enabled), "default_enabled": False},
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, collections.deque)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:  # noqa: BLE001
        pass
    return str(value)


def make_handler(
    cfg: Config | None,
    project_id: str,
    render_html_callable,
    layout: ProjectLayout,
    shutdown_event: threading.Event | None = None,
    max_thumbs_per_class: int = 0,
):
    """Factory that closes over the project/config so we can build a stdlib
    BaseHTTPRequestHandler subclass without globals."""
    shutdown_event = shutdown_event or threading.Event()
    # Guards in-place mutation of the shared cfg (ThreadingHTTPServer runs
    # handlers concurrently; readers tolerate slightly stale values, but a
    # half-applied input setup must never be observable).
    cfg_write_lock = threading.Lock()
    augment_lock = threading.Lock()
    augment_job: dict | None = None
    augment_cancel_event = threading.Event()
    delete_lock = threading.Lock()
    delete_job: dict | None = None
    integrity_lock = threading.Lock()
    integrity_job: dict | None = None
    refresh_lock = threading.Lock()
    refresh_job: dict | None = None
    pretraining_rescan_lock = threading.Lock()
    pretraining_rescan_job: dict | None = None
    compose_lock = threading.Lock()
    compose_job: dict | None = None
    preview_cache_lock = threading.Lock()
    preview_cache_job: dict | None = None
    preview_cache_cancel_event = threading.Event()
    preview_cache = None
    preview_workers = None
    train_lock = threading.Lock()
    train_job: dict | None = None
    feature_stats_lock = threading.Lock()
    feature_stats_job: dict | None = None
    # Serializes GPU resource-estimate probes (subprocesses); never held
    # while a training run is active — the endpoint refuses with 409 instead.
    estimate_lock = threading.Lock()
    batch_eval_lock = threading.Lock()
    batch_eval_job: dict | None = None

    def reap_batch_eval_job() -> None:
        """Fail a batch job whose worker thread is gone. Caller holds the lock.

        A worker that dies before it can report (import error, an exception
        raised outside its own try, a killed thread) would otherwise pin the
        job in "starting"/"running" forever: the UI polls 0/N until the page
        is closed and every later batch is refused with "already running".
        """
        job = batch_eval_job
        if not job or str(job.get("state")) not in {"starting", "running"}:
            return
        thread = job.get("thread")
        if thread is None:
            # Still inside start() — unless start() itself died before it could
            # attach the thread, which this age check catches.
            if time.time() - float(job.get("started_at") or 0.0) < 30.0:
                return
        elif thread.is_alive():
            return
        job["state"] = "error"
        job["error"] = str(job.get("error") or "") or (
            "batch worker stopped unexpectedly — see the inspect server log "
            "for the traceback"
        )
        job["ended_at"] = job.get("ended_at") or time.time()
        log.error("batch evaluation worker died without reporting (job %s)",
                  job.get("job_id"))
    explain_lock = threading.Lock()
    explain_job: dict | None = None
    frozen_eval_lock = threading.Lock()
    frozen_eval_job: dict | None = None
    meter_lock = threading.Lock()
    meter_job: dict | None = None
    inference_lock = threading.Lock()
    inference_job: dict | None = None
    activity_proc_sample = {"wall": time.monotonic(), "cpu": time.process_time()}
    activity_events_lock = threading.Lock()
    activity_events: list[dict] = []
    clip_activity_lock = threading.Lock()
    clip_activity_jobs: dict[str, dict] = {}

    def start_clip_activity(title: str, detail: str, *, tab: str | None = None) -> dict:
        """Register synchronous clip/search work for the shared activity bar.

        Clip pages are intentionally served synchronously so the response can
        contain the requested cards.  They still need to be visible while a
        large inventory is being scanned; otherwise the UI incorrectly looks
        idle even though the request is doing the expensive work.
        """
        job = {
            "job_id": uuid.uuid4().hex[:12], "type": "clip_load",
            "state": "running", "title": title, "detail": detail,
            "progress": None, "started_at": time.time(), "tab": tab,
            "section": None,
        }
        with clip_activity_lock:
            clip_activity_jobs[job["job_id"]] = job
        return job

    def finish_clip_activity(job: dict) -> None:
        with clip_activity_lock:
            clip_activity_jobs.pop(str(job.get("job_id") or ""), None)

    def record_activity_event(title: str, *, event_type: str = "action", detail: str = "", tab: str | None = "library", section: str | None = "source_rules") -> None:
        """Keep lightweight successful actions visible in the unified bar."""
        now = time.time()
        with activity_events_lock:
            activity_events.append({
                "job_id": uuid.uuid4().hex[:12], "type": event_type,
                "state": "finished", "title": title, "detail": detail,
                "progress": 1.0, "started_at": now, "ended_at": now,
                "tab": tab, "section": section,
            })
            del activity_events[:-40]

    def stop_meter() -> dict:
        nonlocal meter_job
        with meter_lock:
            job = meter_job
            meter_job = None
        if not job:
            return {"state": "idle"}
        stream = job.get("stream")
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        job["stream"] = None
        job["state"] = "stopped"
        job["ended_at"] = time.time()
        return job

    def stop_inference() -> dict:
        nonlocal inference_job
        with inference_lock:
            job = inference_job
            if not job:
                return {"state": "idle"}
            job["state"] = "stopping"
            runtime = job.get("runtime")
            artnet = job.get("artnet")
        if runtime is not None:
            try:
                runtime.stop()
            except Exception as exc:  # noqa: BLE001
                log.warning("inspect inference stop failed: %s", exc)
        if artnet is not None:
            try:
                artnet.stop()
            except Exception as exc:  # noqa: BLE001
                log.warning("inspect inference ArtNet stop failed: %s", exc)
        with inference_lock:
            if inference_job is job:
                job["runtime"] = None
                job["artnet"] = None
                job["state"] = "stopped"
                job["ended_at"] = time.time()
        return job

    def cleanup_runtime_resources() -> None:
        stop_meter()
        stop_inference()

    def stop_all_jobs() -> None:
        """Best-effort cancellation of every background job before shutdown.

        Pressing Done must never leave a training subprocess or worker thread
        running past the server. Meter + inference are stopped by
        cleanup_runtime_resources(); the cancel events are cooperatively
        observed by the augment / preview-cache workers; the training
        subprocess is terminated the same way /api/train/cancel does."""
        cleanup_runtime_resources()
        augment_cancel_event.set()
        preview_cache_cancel_event.set()
        with batch_eval_lock:
            if batch_eval_job:
                batch_eval_job["cancel_requested"] = True
        with train_lock:
            job = train_job
            proc = job.get("proc") if job else None
            if job:
                job["cancel"] = True
                try:
                    Path(str(job.get("stop_file") or "")).unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
        if proc is not None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass

    def preview_service():
        """Lazily create the shared, bounded preview worker service."""
        nonlocal preview_cache, preview_workers
        with preview_cache_lock:
            if preview_cache is None:
                from sed.data.preview_cache import PreviewCache

                preview_cache = PreviewCache()
            if preview_workers is None:
                from sed.data.preview_cache import PreviewWorkerPool

                preview_workers = PreviewWorkerPool(preview_cache, workers=2)
            return preview_cache, preview_workers

    class Handler(BaseHTTPRequestHandler):

        # Silence the default access logging to stderr per request — too
        # noisy when 100 audio clips load on one page.
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            _ = (format, args)
            if self.path.startswith("/api/"):
                log.info("%s %s", self.command, self.path)

        # ---- GET --------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                qs = parse_qs(url.query)
                run_id = (qs.get("run", [""])[0] or "").strip() or None
                try:
                    html = render_html_callable(run_id=run_id)
                except TypeError:
                    html = render_html_callable()
                self._send_html(html)
            elif url.path == "/audio":
                qs = parse_qs(url.query)
                paths = qs.get("path", [])
                if not paths:
                    self._send(400, b"missing ?path"); return
                resolved = _safe_resolve_audio_path(layout, paths[0])
                if resolved is None:
                    self._send(403, b"forbidden"); return
                self._send_file(resolved)
            elif url.path == "/thumbnail":
                qs = parse_qs(url.query)
                paths = qs.get("path", [])
                t_type = qs.get("type", ["spec"])[0]
                if not paths:
                    self._send(400, b"missing ?path"); return
                resolved = _safe_resolve_audio_path(layout, paths[0])
                if resolved is None:
                    self._send(403, b"forbidden"); return
                if t_type != "spec" or cfg is None:
                    self._send(404, b"thumbnail unavailable"); return
                activity_job = start_clip_activity(
                    "Loading spectrogram", resolved.name, tab=None,
                )
                try:
                    from sed.data.presets import preview_features_dict

                    cache, workers = preview_service()
                    features = preview_features_dict(cfg, layout)
                    migrated = cache.migrate_legacy(
                        resolved, features, project_id=project_id,
                        label=str(getattr(cfg.features, "preset", "") or ""),
                    )
                    if migrated is not None:
                        self._send_file(migrated.path)
                        return
                    hit = cache.lookup(resolved, features, project_id=project_id)
                    if hit is not None:
                        self._send_file(hit.path)
                        return
                    future = workers.submit(
                        resolved, features, project_id=project_id,
                        label=str(getattr(cfg.features, "preset", "") or ""), priority=0,
                    )
                    # Do not occupy one request thread per costly spectrogram.
                    # The browser preview loader understands 202 and retries
                    # within its four-request queue.
                    try:
                        result = future.result(timeout=0.01)
                    except TimeoutError:
                        self._send_json({"ok": True, "state": "queued", "retry_after_ms": 250}, status=202)
                        return
                    self._send_file(result.path)
                except Exception as exc:  # noqa: BLE001
                    log.warning("thumbnail queue failed for %s: %s", resolved, exc)
                    self._send(500, b"failed to queue thumbnail")
                finally:
                    finish_clip_activity(activity_job)
            elif url.path == "/image":
                qs = parse_qs(url.query)
                paths = qs.get("path", [])
                if not paths:
                    self._send(400, b"missing ?path"); return
                resolved = _safe_resolve_asset_path(layout, paths[0], _IMAGE_SUFFIXES)
                if resolved is None:
                    self._send(403, b"forbidden"); return
                self._send_file(resolved)
            elif url.path == "/api/frozen-run/image":
                qs = parse_qs(url.query)
                run_id = (qs.get("run", [""])[0] or "").strip() or None
                kind = (qs.get("kind", [""])[0] or "").strip()
                from sed.data.inspect import frozen_run_report_image_path
                resolved = frozen_run_report_image_path(layout, run_id, kind)
                if resolved is None:
                    self._send(404, b"not found"); return
                self._send_file(resolved)
            elif url.path == "/api/frozen-run":
                qs = parse_qs(url.query)
                run_id = (qs.get("run", [""])[0] or "").strip() or None
                rule_id = (qs.get("rule", [""])[0] or "").strip() or None
                self._handle_frozen_run(run_id, rule_id)
            elif url.path == "/api/evaluate-frozen/status":
                qs = parse_qs(url.query)
                self._handle_evaluate_frozen_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/explanations/status":
                qs = parse_qs(url.query)
                self._handle_explain_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/explanations/summary":
                self._handle_explain_summary(parse_qs(url.query))
            elif url.path == "/api/explanations/map":
                self._handle_explain_map(parse_qs(url.query))
            elif url.path == "/api/selection":
                self._handle_selection_get()
            elif url.path == "/api/augmentation-generated-counts":
                self._handle_augmentation_generated_counts()
            elif url.path == "/api/panel-totals":
                self._handle_panel_totals()
            elif url.path == "/api/clips":
                # keep_blank_values, because "origins=" (every subbucket
                # unchecked) has to stay distinguishable from no origins= at
                # all (no subbucket filter). Every other param here already
                # falls back on an empty string.
                qs = parse_qs(url.query, keep_blank_values=True)
                self._handle_clips_page(
                    tab=(qs.get("tab", [""])[0] or "").strip(),
                    class_name=(qs.get("class", [""])[0] or ""),
                    offset=(qs.get("offset", ["0"])[0] or "0"),
                    limit=(qs.get("limit", ["100"])[0] or "100"),
                    run_id=(qs.get("run_id", qs.get("run", [""]))[0] or "").strip() or None,
                    origin_class=(qs.get("origin_class", [""])[0] or "").strip(),
                    # None when absent (every bucket) vs "" when present and
                    # empty (the user unchecked them all) — different answers.
                    origins=(qs["origins"][0] if "origins" in qs else None),
                    group_axis=(qs.get("group_axis", [""])[0] or "").strip(),
                    group_key=(qs.get("group_key", [""])[0] or ""),
                    query=(qs.get("query", qs.get("q", [""]))[0] or "").strip(),
                )
            elif url.path == "/api/activity":
                self._handle_activity()
            elif url.path == "/api/review-data/status":
                qs = parse_qs(url.query)
                self._handle_review_data_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/augment/status":
                qs = parse_qs(url.query)
                self._handle_augment_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/delete/status":
                qs = parse_qs(url.query)
                self._handle_delete_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/dataset/integrity/status":
                qs = parse_qs(url.query)
                self._handle_integrity_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/compose/status":
                qs = parse_qs(url.query)
                self._handle_compose_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/preview-cache/status":
                qs = parse_qs(url.query)
                self._handle_preview_cache_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/preview-cache":
                self._handle_preview_cache_list()
            elif url.path == "/api/feature-presets":
                self._handle_feature_presets_get()
            elif url.path == "/api/feature-preview":
                self._handle_feature_preview(parse_qs(url.query))
            elif url.path == "/api/feature-audio":
                self._handle_feature_audio(parse_qs(url.query))
            elif url.path == "/api/feature-analysis":
                self._handle_feature_analysis(parse_qs(url.query))
            elif url.path == "/api/feature-model-fit":
                self._handle_feature_model_fit(parse_qs(url.query))
            elif url.path == "/api/feature-stats-estimate":
                self._handle_feature_stats_estimate(parse_qs(url.query))
            elif url.path == "/api/model-catalog":
                self._handle_model_catalog_get()
            elif url.path == "/api/model-user-specs":
                self._handle_model_user_specs_get()
            elif url.path == "/api/library/negatives-count":
                self._handle_library_negatives_count()
            elif url.path == "/api/model-parent-candidates":
                self._handle_model_parent_candidates_get(parse_qs(url.query))
            elif url.path == "/api/model-probe":
                self._handle_model_probe_get(parse_qs(url.query))
            elif url.path == "/api/model-task-readiness":
                self._handle_model_task_readiness_get(parse_qs(url.query))
            elif url.path == "/api/annotation-clips":
                self._handle_annotation_clips_get()
            elif url.path == "/api/clip-annotations":
                self._handle_clip_annotations_get(parse_qs(url.query))
            elif url.path == "/api/target-classes":
                self._handle_target_classes_get()
            elif url.path == "/api/pretraining/counts":
                self._handle_pretraining_counts()
            elif url.path == "/api/pretraining/rescan/status":
                qs = parse_qs(url.query)
                self._handle_pretraining_rescan_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/pretraining/subbuckets":
                self._handle_pretraining_subbuckets()
            elif url.path == "/api/dataset/test-config":
                self._handle_dataset_test_config_get()
            elif url.path == "/api/train/config":
                self._handle_train_config()
            elif url.path == "/api/train/status":
                self._handle_train_status()
            elif url.path == "/api/train/batch-evaluate/status":
                qs = parse_qs(url.query)
                self._handle_batch_eval_status((qs.get("job_id", [""])[0] or "").strip())
            elif url.path == "/api/advice":
                self._handle_advice_get()
            elif url.path == "/api/feature-scaling-drift":
                self._handle_feature_scaling_drift_get()
            elif url.path == "/api/frozen-test-presets":
                self._handle_frozen_test_presets_get()
            elif url.path == "/api/train/runs":
                self._handle_train_runs()
            elif url.path == "/api/train/run":
                qs = parse_qs(url.query)
                self._handle_train_run((qs.get("model_id", [""])[0] or "").strip())
            elif url.path == "/api/train/compare":
                qs = parse_qs(url.query)
                raw = (qs.get("model_ids", [""])[0] or "").strip()
                self._handle_train_compare([m.strip() for m in raw.split(",") if m.strip()])
            elif url.path == "/api/train/asset":
                qs = parse_qs(url.query)
                self._handle_train_asset(
                    (qs.get("model_id", [""])[0] or "").strip(),
                    (qs.get("file", [""])[0] or "").strip(),
                )
            elif url.path == "/api/inference/models":
                self._handle_inference_models()
            elif url.path == "/api/inference/settings":
                self._handle_inference_settings(parse_qs(url.query))
            elif url.path == "/api/inference/meter/status":
                self._handle_inference_meter_status()
            elif url.path == "/api/inference/status":
                self._handle_inference_status()
            else:
                self._send(404, b"not found")

        # ---- POST -------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/api/done":
                stop_all_jobs()
                self._handle_done()
                threading.Thread(target=shutdown_event.set, daemon=True).start()
                return
            if url.path == "/api/delete/start":
                self._handle_delete_start(); return
            if url.path == "/api/dataset/integrity/start":
                self._handle_integrity_start(); return
            if url.path == "/api/review-data/start":
                self._handle_review_data_start(); return
            if url.path == "/api/mined/triage":
                self._handle_mined_triage()
                return
            if url.path == "/api/selection":
                self._handle_selection()
                return
            if url.path == "/api/selection-rule":
                self._handle_selection_rule()
                return
            if url.path == "/api/dataset-preset/save":
                self._handle_dataset_preset("save"); return
            if url.path == "/api/dataset-preset/apply":
                self._handle_dataset_preset("apply"); return
            if url.path == "/api/dataset-preset/delete":
                self._handle_dataset_preset("delete"); return
            if url.path == "/api/dataset-preset/new":
                self._handle_dataset_preset("new"); return
            if url.path == "/api/dataset-preset/rename":
                self._handle_dataset_preset("rename"); return
            if url.path == "/api/augmentation-rule":
                self._handle_augmentation_rule()
                return
            if url.path == "/api/augmentation-settings":
                self._handle_augmentation_settings()
                return
            if url.path == "/api/augment/start":
                self._handle_augment_start()
                return
            if url.path == "/api/augment/cancel":
                self._handle_augment_cancel()
                return
            if url.path == "/api/compose/start":
                self._handle_compose_start(); return
            if url.path == "/api/preview-cache/start":
                self._handle_preview_cache_start(); return
            if url.path == "/api/preview-cache/cancel":
                self._handle_preview_cache_cancel(); return
            if url.path == "/api/preview-cache/delete":
                self._handle_preview_cache_delete(); return
            if url.path == "/api/pretraining/rescan":
                self._handle_pretraining_rescan(); return
            if url.path == "/api/pretraining/assign":
                self._handle_pretraining_assign(); return
            if url.path == "/api/feature-preset/save":
                self._handle_feature_preset_save(); return
            if url.path == "/api/feature-preset/delete":
                self._handle_feature_preset_delete(); return
            if url.path == "/api/feature-preset/apply":
                self._handle_feature_preset_apply(); return
            if url.path == "/api/feature-training/apply":
                self._handle_feature_training_apply(); return
            if url.path == "/api/evaluate-frozen":
                self._handle_evaluate_frozen()
                return
            if url.path == "/api/explanations/start":
                self._handle_explain_start()
                return
            if url.path == "/api/target-classes":
                self._handle_target_classes_post(); return
            if url.path == "/api/dataset/split-config":
                self._handle_dataset_split_config_post(); return
            if url.path == "/api/model-recipe/select":
                self._handle_model_recipe_select(); return
            if url.path == "/api/clip-annotations/propose":
                self._handle_clip_annotations_propose(); return
            if url.path == "/api/clip-annotations/save":
                self._handle_clip_annotations_save(); return
            if url.path == "/api/train/config-preview":
                self._handle_train_config_preview(); return
            if url.path == "/api/train/resource-estimate":
                self._handle_train_resource_estimate(); return
            if url.path == "/api/train/start":
                self._handle_train_start(); return
            if url.path == "/api/train/cancel":
                self._handle_train_cancel(); return
            if url.path == "/api/train/stop-save":
                self._handle_train_stop_save(); return
            if url.path == "/api/train/evaluate":
                self._handle_train_evaluate(); return
            if url.path == "/api/train/evaluate-exported":
                self._handle_train_evaluate_exported(); return
            if url.path == "/api/train/calibrate":
                self._handle_train_calibrate(); return
            if url.path == "/api/train/batch-evaluate/check":
                self._handle_batch_eval_check(); return
            if url.path.startswith("/api/frozen-test-preset/"):
                self._handle_frozen_test_preset(url.path.rsplit("/", 1)[-1]); return
            if url.path == "/api/train/frozen-metrics":
                self._handle_train_frozen_metrics(); return
            if url.path == "/api/train/batch-evaluate/start":
                self._handle_batch_eval_start(); return
            if url.path == "/api/train/batch-evaluate/cancel":
                self._handle_batch_eval_cancel(); return
            if url.path == "/api/train/set-active":
                self._handle_train_set_active(); return
            if url.path == "/api/model-user-specs/delete":
                self._handle_model_user_specs_delete(); return
            if url.path == "/api/train/delete":
                self._handle_train_delete(); return
            if url.path == "/api/inference/meter/start":
                self._handle_inference_meter_start(); return
            if url.path == "/api/inference/meter/stop":
                self._handle_inference_meter_stop(); return
            if url.path == "/api/inference/start":
                self._handle_inference_start(); return
            if url.path == "/api/inference/stop":
                self._handle_inference_stop(); return
            if url.path == "/api/inference/set-active":
                self._handle_inference_set_active(); return
            if url.path == "/api/inference/save-input-setup":
                self._handle_inference_save_input_setup(); return
            self._send(404, b"not found")

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

        def _delete_paths(self, raw_paths: list, job: dict | None = None) -> dict:
            deleted: list[str] = []
            errors: list[dict] = []
            sidecar_metadata: dict[str, dict] = {}
            phase_timings: dict[str, float] = {}
            total = len(raw_paths)
            phase_started = time.perf_counter()

            def set_phase(name: str) -> None:
                if job is not None:
                    with delete_lock:
                        job["phase"] = name
                        job["current_name"] = ""

            for i, rp in enumerate(raw_paths, 1):
                resolved = _safe_resolve_audio_path(layout, rp)
                if job is not None:
                    with delete_lock:
                        job["completed"] = i - 1
                        job["current_name"] = Path(str(rp)).name
                if resolved is None:
                    errors.append({"path": str(rp), "error": "forbidden or missing"})
                    if job is not None:
                        with delete_lock:
                            job["completed"] = i
                            job["failed_count"] = len(errors)
                            job["errors"] = errors[-10:]
                    continue
                sidecar = resolved.with_suffix(".json")
                thumb = resolved.with_suffix(".spec.png")
                mix_map = resolved.with_suffix(".mix.png")
                try:
                    try:
                        raw_meta = json.loads(sidecar.read_text(encoding="utf-8"))
                        if isinstance(raw_meta, dict):
                            sidecar_metadata[os.path.normpath(str(resolved))] = raw_meta
                    except (OSError, ValueError):
                        pass
                    resolved.unlink(missing_ok=True)
                    sidecar.unlink(missing_ok=True)
                    thumb.unlink(missing_ok=True)
                    mix_map.unlink(missing_ok=True)
                    deleted.append(str(resolved))
                    log.info("deleted %s", resolved.name)
                except Exception as e:  # noqa: BLE001
                    errors.append({"path": str(resolved), "error": str(e)})
                    log.warning("delete failed %s: %s", resolved, e)
                if job is not None:
                    with delete_lock:
                        job["completed"] = i
                        job["deleted"] = list(deleted)
                        job["deleted_count"] = len(deleted)
                        job["failed_count"] = len(errors)
                        job["errors"] = errors[-10:]
            phase_timings["unlink"] = round(time.perf_counter() - phase_started, 6)
            deleted_records: list[dict[str, Any]] = []
            affected_scopes: set[str] = set()
            selection: dict = {}
            selection_changed = False
            cache_removed: dict[str, int] = {"metadata": 0, "integrity": 0}
            compose_state: dict = {}
            if deleted:
                set_phase("indexes")
                phase_started = time.perf_counter()
                from sed.config import LIBRARY_DIR
                from sed.data.library import Library
                from sed.data.pretraining import shared_index

                lib = Library()
                removed_clips = lib.remove_paths_from_manifest(deleted)
                pretrain_index = shared_index()
                removed_pretraining = pretrain_index.remove_paths(deleted)
                removed_clips.extend(
                    pretrain_index.clip_for_row(row, probe=False)
                    for row in removed_pretraining
                )
                records_by_path = {
                    record["path"]: record
                    for record in (
                        _clip_delete_record(clip, LIBRARY_DIR.resolve())
                        for clip in removed_clips
                    )
                }
                library_root = LIBRARY_DIR.resolve()
                for raw_path in deleted:
                    normalized = os.path.normpath(str(raw_path))
                    record = records_by_path.get(normalized)
                    if record is None:
                        try:
                            relpath = str(Path(normalized).relative_to(library_root))
                            category = Path(relpath).parts[0] if Path(relpath).parts else ""
                        except ValueError:
                            relpath, category = "", "recordings"
                        meta = sidecar_metadata.get(normalized, {})
                        record = {
                            "path": normalized,
                            "relpath": relpath,
                            "uuid": str(meta.get("uuid") or ""),
                            "category": category,
                            "label": str(meta.get("label") or Path(normalized).parent.name),
                            "origin_class": str(meta.get("origin_class") or meta.get("label") or Path(normalized).parent.name),
                            "project_id": str(meta.get("project_id") or ""),
                        }
                    meta = sidecar_metadata.get(normalized, {})
                    for key in ("overlay_label", "base_label", "recipe_id", "assigned_target_class"):
                        if meta.get(key) not in (None, ""):
                            record[key] = meta[key]
                    deleted_records.append(record)
                    category = str(record.get("category") or "")
                    if category in {"sounds", "raw"}:
                        affected_scopes.update({"library", "dataset", "frozen", "recipes"})
                    elif category == "derived":
                        affected_scopes.update({"library", "augmented", "frozen", "recipes"})
                    elif category == "mined":
                        affected_scopes.update({"library", "mined", "dataset", "frozen", "recipes"})
                    elif category == "pretraining":
                        affected_scopes.update({"library", "pretraining", "dataset", "frozen", "recipes"})
                    else:
                        affected_scopes.update({"dataset", "frozen"})
                phase_timings["indexes"] = round(time.perf_counter() - phase_started, 6)

                set_phase("project metadata")
                phase_started = time.perf_counter()
                selection, selection_changed = _apply_deleted_clips_to_selection(
                    layout, removed_clips, cfg=cfg, deleted_records=deleted_records,
                )
                from sed.data.compose import (
                    dataset_compose_state,
                    load_dataset_selection,
                    load_excluded_clip_paths,
                    mark_composed_dataset_dirty_for_paths,
                    write_excluded_clip_paths,
                )
                excluded = load_excluded_clip_paths(layout)
                deleted_keys = {os.path.normpath(str(Path(path).resolve())) for path in deleted}
                kept_excluded = sorted(path for path in excluded if os.path.normpath(path) not in deleted_keys)
                if len(kept_excluded) != len(excluded):
                    write_excluded_clip_paths(layout, kept_excluded)
                dirty = mark_composed_dataset_dirty_for_paths(layout, deleted)
                if dirty.get("dirty"):
                    affected_scopes.update({"dataset", "frozen"})
                    dirty_splits = dirty.get("splits") or {}
                    for record in deleted_records:
                        splits = dirty_splits.get(str(record.get("path") or ""))
                        if isinstance(splits, list):
                            record["composed_splits"] = list(splits)
                if not selection:
                    selection = load_dataset_selection(layout)
                if cfg is not None:
                    compose_state = dataset_compose_state(layout, selection, cfg)
                phase_timings["project_metadata"] = round(time.perf_counter() - phase_started, 6)

                set_phase("cache cleanup")
                phase_started = time.perf_counter()
                try:
                    with preview_cache_lock:
                        cache = preview_cache
                    if cache is not None:
                        cache.remove_sources([Path(path) for path in deleted])
                except Exception as exc:  # noqa: BLE001
                    log.warning("preview cache source cleanup failed: %s", exc)
                try:
                    from sed.data.inspect import remove_inspect_cache_paths

                    cache_removed = remove_inspect_cache_paths(layout, deleted)
                except Exception as exc:  # noqa: BLE001
                    log.warning("inspect cache cleanup failed: %s", exc)
                phase_timings["cache_cleanup"] = round(time.perf_counter() - phase_started, 6)
            return {
                "deleted": deleted,
                "count": len(deleted),
                "errors": errors,
                "failed_count": len(errors),
                "total": total,
                "deleted_records": deleted_records,
                "affected_scopes": sorted(affected_scopes),
                "selection": selection,
                "selection_changed": selection_changed,
                "compose_state": compose_state,
                "cache_removed": cache_removed,
                "phase_timings": phase_timings,
            }

        def _handle_delete_status(self, job_id: str = "") -> None:
            self._send_job_status(
                delete_lock, lambda: delete_job, job_id, lambda job: _job_status_payload(job, missing="no delete job"),
            )

        def _handle_mined_triage(self) -> None:
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON"); return

            paths = body.get("paths", [])
            action = body.get("action")

            if not isinstance(paths, list) or not action:
                self._send(400, b"missing paths or action"); return

            target_class = body.get("target_class")
            if action == "accept_negative":
                target_class = "_negative"
            elif action == "reclassify" and not target_class:
                self._send(400, b"missing target_class for reclassify"); return

            from sed.data.library import Library, LibraryClip
            lib = Library()

            errors = []
            moved = []
            old_paths: list[Path] = []
            old_clips = []

            for p in paths:
                try:
                    wav_path = _safe_resolve_audio_path(layout, str(p))
                    if wav_path is None or not wav_path.exists():
                        continue
                    try:
                        wav_path.resolve().relative_to(lib.root.resolve() / "mined")
                    except ValueError:
                        errors.append({"path": str(p), "error": "only mined clips can be triaged"})
                        continue
                    old_paths.append(wav_path)
                    old_clips.append(LibraryClip.from_sidecar(wav_path, lib.root))

                    sidecar_path = wav_path.with_suffix(".json")
                    if sidecar_path.exists():
                        with open(sidecar_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    else:
                        meta = {}

                    meta["review_status"] = "accepted"
                    meta["label"] = target_class
                    meta["category"] = "sounds"
                    if "origin_class" not in meta:
                        meta["origin_class"] = meta.get("predicted_class") or wav_path.parent.name

                    with open(sidecar_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)

                    target_dir = lib.raw_dir(target_class)
                    target_dir.mkdir(parents=True, exist_ok=True)

                    for ext in [".wav", ".json", ".spec.png", ".mix.png"]:
                        src_file = wav_path.with_suffix(ext)
                        if src_file.exists():
                            # rename may fail across devices, but usually library is on same mount
                            src_file.rename(target_dir / src_file.name)

                    moved.append(str(target_dir / wav_path.name))
                except Exception as e:  # noqa: BLE001
                    log.warning("failed to triage %s: %s", p, e)
                    errors.append({"path": str(p), "error": str(e)})

            if moved:
                try:
                    with preview_cache_lock:
                        cache = preview_cache
                    if cache is not None:
                        cache.remove_sources([Path(p) for p in paths])
                except Exception as exc:  # noqa: BLE001
                    log.warning("preview cache source cleanup failed: %s", exc)
                lib.remove_paths_from_manifest(old_paths)
                added_clips = []
                for path in moved:
                    try:
                        added_clips.append(lib.append_sidecar(Path(path)))
                    except Exception as exc:  # noqa: BLE001
                        log.warning("failed to append moved clip to manifest %s: %s", path, exc)
                _apply_deleted_clips_to_selection(
                    layout, old_clips, cfg=cfg, added_clips=added_clips,
                )
                try:
                    from sed.data.inspect import remove_inspect_cache_paths
                    remove_inspect_cache_paths(layout, old_paths)
                except Exception as exc:  # noqa: BLE001
                    log.warning("inspect cache cleanup failed: %s", exc)

            self._send_json({
                "ok": True, "moved": len(moved), "paths": moved,
                "affected_scopes": ["mined", "dataset", "library"], "errors": errors,
            })

        def _handle_delete_start(self) -> None:
            nonlocal delete_job
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON"); return
            raw_paths = body.get("paths", [])
            if not isinstance(raw_paths, list):
                self._send(400, b"paths must be an array"); return
            with delete_lock:
                if delete_job and str(delete_job.get("state")) in {"starting", "running", "cleanup"}:
                    self._send_json({"ok": False, "error": "delete job already running"}, status=409)
                    return
                job_id = uuid.uuid4().hex[:12]
                delete_job = {
                    "job_id": job_id,
                    "state": "starting",
                    "phase": "delete",
                    "total": len(raw_paths),
                    "completed": 0,
                    "deleted_count": 0,
                    "failed_count": 0,
                    "current_name": "",
                    "deleted": [],
                    "errors": [],
                    "started_at": time.time(),
                    "ended_at": None,
                }

            def run_job() -> None:
                with delete_lock:
                    if delete_job and delete_job.get("job_id") == job_id:
                        delete_job["state"] = "running"
                try:
                    result = self._delete_paths(raw_paths, delete_job)
                    with delete_lock:
                        if delete_job and delete_job.get("job_id") == job_id:
                            delete_job.update(result)
                            delete_job["state"] = "finished"
                            delete_job["phase"] = "finished"
                            delete_job["current_name"] = ""
                            delete_job["ended_at"] = time.time()
                except Exception as exc:  # noqa: BLE001
                    log.exception("background delete failed")
                    with delete_lock:
                        if delete_job and delete_job.get("job_id") == job_id:
                            delete_job["state"] = "error"
                            delete_job["phase"] = "error"
                            delete_job["error"] = str(exc)
                            delete_job["ended_at"] = time.time()

            thread = threading.Thread(target=run_job, daemon=True)
            with delete_lock:
                delete_job["thread"] = thread
                payload = _job_status_payload(delete_job, missing="no delete job")
                thread.start()
            self._send_json(payload)

        def _handle_integrity_status(self, job_id: str = "") -> None:
            self._send_job_status(
                integrity_lock, lambda: integrity_job, job_id, lambda job: _job_status_payload(job, missing="no audio-integrity scan"),
            )

        def _handle_integrity_start(self) -> None:
            """Start one cached, training-equivalent Dataset audio scan."""
            nonlocal integrity_job
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            with integrity_lock:
                if integrity_job and str(integrity_job.get("state")) in {"starting", "running"}:
                    self._send_json({"ok": False, "error": "audio-integrity scan already running"}, status=409)
                    return
                job_id = uuid.uuid4().hex[:12]
                integrity_job = {
                    "job_id": job_id, "state": "starting", "total": 0,
                    "completed": 0, "current_name": "", "corrupt_count": 0,
                    "checked": 0, "reused": 0, "error": "",
                    "started_at": time.time(), "ended_at": None,
                }

            def progress(update: dict) -> None:
                with integrity_lock:
                    if not integrity_job or integrity_job.get("job_id") != job_id:
                        return
                    integrity_job["state"] = "running"
                    integrity_job["completed"] = int(update.get("completed") or 0)
                    integrity_job["total"] = int(update.get("total") or 0)
                    integrity_job["current_name"] = str(update.get("current_name") or "")

            def run_job() -> None:
                try:
                    with integrity_lock:
                        if integrity_job and integrity_job.get("job_id") == job_id:
                            integrity_job["state"] = "running"
                    from sed.data.inspect import scan_dataset_audio_integrity

                    result = scan_dataset_audio_integrity(cfg, layout, progress=progress)
                    with integrity_lock:
                        if integrity_job and integrity_job.get("job_id") == job_id:
                            integrity_job.update(result)
                            integrity_job["state"] = "finished"
                            integrity_job["current_name"] = ""
                            integrity_job["ended_at"] = time.time()
                except Exception as exc:  # noqa: BLE001 - return a usable UI error
                    log.exception("audio-integrity scan failed")
                    with integrity_lock:
                        if integrity_job and integrity_job.get("job_id") == job_id:
                            integrity_job["state"] = "error"
                            integrity_job["error"] = str(exc)
                            integrity_job["ended_at"] = time.time()

            thread = threading.Thread(target=run_job, daemon=True)
            with integrity_lock:
                integrity_job["thread"] = thread
                payload = _job_status_payload(integrity_job, missing="no audio-integrity scan")
                thread.start()
            self._send_json(payload)

        def _handle_frozen_run(self, run_id: str | None, rule_id: str | None = None) -> None:
            from sed.data.inspect import frozen_run_is_completed, frozen_run_prediction_payload
            if run_id and not frozen_run_is_completed(layout, run_id):
                self._send_json({"ok": False, "error": f"unknown completed model: {run_id}"}, status=404)
                return
            log.info(
                "switching frozen-test inspection to run %s rule %s",
                run_id or "selected/latest",
                rule_id or "preferred/current",
            )
            payload = frozen_run_prediction_payload(layout, run_id=run_id, rule_id=rule_id)
            self._send_json(payload)

        def _handle_evaluate_frozen_status(self, job_id: str = "") -> None:
            self._send_job_status(
                frozen_eval_lock, lambda: frozen_eval_job, job_id, lambda job: _job_status_payload(job, missing="no frozen-test evaluation job"),
            )

        def _handle_evaluate_frozen(self) -> None:
            """Start one rule-specific score on the current composed frozen set."""
            nonlocal frozen_eval_job
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            run_id = str(body.get("run_id") or "").strip()
            from sed.train import registry
            from sed.data.inspect import frozen_run_is_completed
            from sed.train.test_rules import TestRule, parse_rule_id
            project_id = layout.id
            model_id = run_id or (registry.get_active_model(project_id) or "")
            if not model_id:
                self._send_json(
                    {"ok": False, "error": "no model selected and no active model"},
                    status=400,
                )
                return
            if not frozen_run_is_completed(layout, model_id):
                self._send_json(
                    {"ok": False, "error": f"unknown completed model: {model_id}"},
                    status=404,
                )
                return

            raw_rule_id = str(body.get("rule_id") or "").strip()
            if not raw_rule_id:
                # Compatibility with the first Frozen-tab endpoint contract.
                decision_rule = str(body.get("decision_rule") or "fixed").strip().lower()
                try:
                    threshold = float(body.get("threshold", 0.90))
                    rule = TestRule(decision_rule, threshold if decision_rule == "fixed" else None)
                except (TypeError, ValueError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
            else:
                try:
                    rule = parse_rule_id(raw_rule_id)
                except (TypeError, ValueError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return

            with frozen_eval_lock:
                if frozen_eval_job and str(frozen_eval_job.get("state")) in {"starting", "running"}:
                    self._send_json(
                        {"ok": False, "error": "a frozen-test evaluation is already running"},
                        status=409,
                    )
                    return
                job_id = uuid.uuid4().hex[:12]
                frozen_eval_job = {
                    "job_id": job_id,
                    "state": "starting",
                    "model_id": model_id,
                    "rule_id": rule.id,
                    "rule_label": rule.label,
                    "started_at": time.time(),
                    "ended_at": None,
                    "result": {},
                    "error": "",
                    "phase": "queued",
                    "completed": 0,
                    "total": 0,
                    "current_name": "",
                    "current_class": "",
                }

            log.info("evaluating model %s on current frozen test under %s", model_id, rule.id)

            def run_job() -> None:
                from sed.train.eval import evaluate_run_under_rule

                def progress(update: dict[str, object]) -> None:
                    with frozen_eval_lock:
                        if not frozen_eval_job or frozen_eval_job.get("job_id") != job_id:
                            return
                        for key in (
                            "phase", "completed", "total", "current_name",
                            "current_class",
                        ):
                            if key in update:
                                frozen_eval_job[key] = update[key]

                with frozen_eval_lock:
                    if frozen_eval_job and frozen_eval_job.get("job_id") == job_id:
                        frozen_eval_job["state"] = "running"
                        frozen_eval_job["phase"] = "loading model checkpoint"
                try:
                    metrics = evaluate_run_under_rule(
                        cfg, project_id, model_id, rule, progress=progress,
                    )
                    result = {
                        "model_id": model_id,
                        "rule_id": rule.id,
                        "rule_label": rule.label,
                        "accuracy": metrics.get("accuracy"),
                        "f1_macro_targets": metrics.get("f1_macro_targets"),
                        "n_samples": metrics.get("n_samples"),
                        "score_profile": metrics.get("score_profile"),
                        "prediction_mode": metrics.get("prediction_mode"),
                        "thresholds_source": metrics.get("thresholds_source"),
                        "test_set_hash": metrics.get("test_set_hash"),
                    }
                    with frozen_eval_lock:
                        if frozen_eval_job and frozen_eval_job.get("job_id") == job_id:
                            frozen_eval_job["result"] = result
                            frozen_eval_job["state"] = "finished"
                            frozen_eval_job["ended_at"] = time.time()
                except Exception as exc:  # noqa: BLE001 — surface through job status.
                    log.exception("frozen-test evaluation failed")
                    with frozen_eval_lock:
                        if frozen_eval_job and frozen_eval_job.get("job_id") == job_id:
                            frozen_eval_job["error"] = str(exc)
                            frozen_eval_job["state"] = "error"
                            frozen_eval_job["ended_at"] = time.time()

            thread = threading.Thread(target=run_job, daemon=True)
            with frozen_eval_lock:
                if frozen_eval_job and frozen_eval_job.get("job_id") == job_id:
                    frozen_eval_job["thread"] = thread
                    payload = _job_status_payload(frozen_eval_job, missing="no frozen-test evaluation job")
                    thread.start()
                else:
                    payload = {"ok": False, "error": "failed to start frozen-test evaluation"}
            self._send_json(payload, status=202 if payload.get("ok") else 500)

        def _resolve_explain_model_id(self, run_id: str = "") -> tuple[str, str | None]:
            from sed.data.inspect import frozen_run_is_completed
            from sed.train import registry

            requested = str(run_id or "").strip()
            model_id = requested or (registry.get_active_model(layout.id) or "")
            if not model_id:
                return "", "no model selected and no active model"
            if not frozen_run_is_completed(layout, model_id):
                return "", f"unknown completed model: {model_id}"
            return model_id, None

        def _handle_explain_status(self, job_id: str = "") -> None:
            self._send_job_status(
                explain_lock, lambda: explain_job, job_id, lambda job: _job_status_payload(job, missing="no explanation job"),
            )

        def _handle_explain_summary(self, qs: dict) -> None:
            scope = (qs.get("scope", ["frozen"])[0] or "frozen").strip().lower()
            run_id = (qs.get("run", qs.get("model_id", [""]))[0] or "").strip()
            model_id, error = self._resolve_explain_model_id(run_id)
            if error:
                self._send_json({"ok": False, "error": error}, status=404)
                return
            try:
                from sed.train.explain import load_explanation_summary

                summary = load_explanation_summary(layout.id, model_id, scope)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc), "model_id": model_id}, status=400)
                return
            if summary is None:
                self._send_json(
                    {"ok": False, "error": "explanation summary not found", "model_id": model_id},
                    status=404,
                )
                return
            self._send_json({"ok": True, "model_id": model_id, "scope": scope, "summary": summary})

        def _handle_explain_map(self, qs: dict) -> None:
            scope = (qs.get("scope", ["frozen"])[0] or "frozen").strip().lower()
            run_id = (qs.get("run", qs.get("model_id", [""]))[0] or "").strip()
            raw_path = (qs.get("path", [""])[0] or "").strip()
            class_name = (qs.get("class", qs.get("class_name", [""]))[0] or "").strip()
            if not raw_path or not class_name:
                self._send_json({"ok": False, "error": "missing path or class"}, status=400)
                return
            model_id, error = self._resolve_explain_model_id(run_id)
            if error:
                self._send_json({"ok": False, "error": error}, status=404)
                return
            try:
                from sed.train.explain import load_explanation_for_clip

                payload = load_explanation_for_clip(
                    layout.id, model_id, scope, raw_path, class_name
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc), "model_id": model_id}, status=400)
                return
            if payload is None:
                self._send_json(
                    {
                        "ok": False,
                        "error": "explanation map not found",
                        "model_id": model_id,
                        "scope": scope,
                    },
                    status=404,
                )
                return
            self._send_json({"ok": True, "model_id": model_id, "scope": scope, "explanation": payload})

        def _handle_explain_start(self) -> None:
            nonlocal explain_job
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            run_id = str(body.get("run_id") or body.get("model_id") or "").strip()
            model_id, error = self._resolve_explain_model_id(run_id)
            if error:
                self._send_json({"ok": False, "error": error}, status=404)
                return
            raw_scopes = body.get("scopes", body.get("scope", ["frozen", "mined"]))
            if isinstance(raw_scopes, str):
                scopes = [s.strip().lower() for s in raw_scopes.split(",") if s.strip()]
            elif isinstance(raw_scopes, list):
                scopes = [str(s).strip().lower() for s in raw_scopes if str(s).strip()]
            else:
                scopes = ["frozen", "mined"]
            if not scopes:
                self._send_json({"ok": False, "error": "no scopes requested"}, status=400)
                return
            method = str(body.get("method") or "auto").strip() or "auto"
            class_selection = str(body.get("classes") or "top").strip() or "top"
            force = bool(body.get("force", False))
            limit_value = body.get("limit")
            limit = None
            if limit_value not in (None, ""):
                try:
                    limit = int(limit_value)
                except (TypeError, ValueError):
                    self._send_json({"ok": False, "error": "limit must be an integer"}, status=400)
                    return

            with explain_lock:
                if explain_job and str(explain_job.get("state")) in {"starting", "running"}:
                    self._send_json({"ok": False, "error": "explanation job already running"}, status=409)
                    return
                job_id = uuid.uuid4().hex[:12]
                explain_job = {
                    "job_id": job_id,
                    "state": "starting",
                    "model_id": model_id,
                    "scopes": scopes,
                    "method": method,
                    "classes": class_selection,
                    "force": force,
                    "limit": limit,
                    "completed": 0,
                    "total": len(scopes),
                    "current_scope": "",
                    "results": [],
                    "errors": [],
                    "started_at": time.time(),
                    "ended_at": None,
                }

            def run_job() -> None:
                from sed.train.explain import generate_explanations

                with explain_lock:
                    if explain_job and explain_job.get("job_id") == job_id:
                        explain_job["state"] = "running"
                for idx, scope in enumerate(scopes, start=1):
                    with explain_lock:
                        if explain_job and explain_job.get("job_id") == job_id:
                            explain_job["current_scope"] = scope
                            explain_job["completed"] = idx - 1
                    try:
                        summary = generate_explanations(
                            cfg,
                            layout.id,
                            model_id,
                            scope=scope,
                            method=method,
                            classes=class_selection,
                            force=force,
                            limit=limit,
                        )
                        result = {
                            "scope": scope,
                            "n_clips": summary.get("n_clips"),
                            "n_maps": summary.get("n_maps"),
                            "methods": summary.get("methods", {}),
                        }
                        with explain_lock:
                            if explain_job and explain_job.get("job_id") == job_id:
                                explain_job["results"].append(result)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("explanation analysis failed for %s", scope)
                        with explain_lock:
                            if explain_job and explain_job.get("job_id") == job_id:
                                explain_job["errors"].append({"scope": scope, "error": str(exc)})
                    finally:
                        with explain_lock:
                            if explain_job and explain_job.get("job_id") == job_id:
                                explain_job["completed"] = idx
                with explain_lock:
                    if explain_job and explain_job.get("job_id") == job_id:
                        explain_job["state"] = "finished" if not explain_job["errors"] else "error"
                        explain_job["current_scope"] = ""
                        explain_job["ended_at"] = time.time()

            thread = threading.Thread(target=run_job, daemon=True)
            with explain_lock:
                if explain_job and explain_job.get("job_id") == job_id:
                    explain_job["thread"] = thread
                    payload = _job_status_payload(explain_job, missing="no explanation job")
                    thread.start()
                else:
                    payload = {"ok": False, "error": "failed to start explanation job"}
            self._send_json(payload, status=200 if payload.get("ok") else 500)

        # ---- Train lab --------------------------------------------------
        # The Train tab wraps the exact `sed train` pipeline as the CLI: hyper-
        # parameters are written to config.toml (so they freeze into the bundle),
        # training runs as a subprocess, and eval/calibrate/set-active reuse the
        # same registry + eval + calibrate code paths the CLI uses.

        _TRAIN_INT = ("epochs", "batch_size", "accumulate_grad_batches",
                      "warmup_epochs", "patience", "seed",
                      "num_workers", "specaugment_freq_mask", "specaugment_time_mask",
                      "fine_tune_warmup_epochs")
        _TRAIN_FLOAT = ("lr_backbone", "lr_head", "weight_decay", "head_dropout",
                        "focal_gamma_pos", "focal_gamma_neg", "lr_layer_decay",
                        "ssl_ema_decay", "mixup_alpha", "calibrate_per_1000",
                        "auto_test_threshold", "int8_max_regression")
        _TRAIN_STR = (
            "precision", "loss", "train_depth", "fine_tune_policy",
            "pretraining_objective", "class_balance", "calibrate_negsrc",
            "transfer_head_init",
        )
        _TRAIN_BOOL = ("specaugment_enabled", "auto_export", "freeze_batchnorm",
                       "calibrate_thresholds", "bn_recalibrate", "head_dropout_auto",
                       "calibrate_after", "auto_test", "export_verify_int8")
        # Frozen-test rule ids and export artifact names; both validated by
        # TrainingCfg before they reach config.toml.
        _TRAIN_STR_LIST = ("auto_test_rules", "export_targets")

        def _train_config_payload(self, cfg: Config) -> dict:
            from sed.audio.features import feature_band_count
            from sed.config import backbone_registry_payload, is_pretraining_project
            from sed.train.fine_tuning import (
                available_fine_tune_policies,
                coerce_fine_tune_policy,
                is_transfer_initialization,
            )
            t, f = cfg.training, cfg.features
            training = {
                k: getattr(t, k)
                for k in (self._TRAIN_INT + self._TRAIN_FLOAT + self._TRAIN_STR + self._TRAIN_BOOL)
                if hasattr(t, k)
            }
            # Resolved (never None), so the form always shows what the next run
            # will actually score — including for pre-rule config.toml files.
            training["auto_test_rules"] = t.resolved_auto_test_rules()
            # Same for the export artifact list: a config.toml written before the
            # key existed must still show the form the default it will get.
            training["export_targets"] = list(t.export_targets or [])
            try:
                from sed.train.model_lab import effective_head_dropout

                training["head_dropout_configured"] = float(t.head_dropout)
                training["head_dropout_effective"] = effective_head_dropout(cfg)
                # The number input always shows what the next run will use;
                # auto/manual ownership is carried separately.
                training["head_dropout"] = training["head_dropout_effective"]
            except Exception:  # noqa: BLE001 - keep config endpoint available
                training["head_dropout_effective"] = float(t.head_dropout)
            from sed.train.batching import training_batch_sizes

            batching = training_batch_sizes(t)
            training.setdefault(
                "accumulate_grad_batches", batching["accumulate_grad_batches"],
            )
            counts = _dataset_counts(layout)
            split_preview = _preview_train_val_split(layout, cfg)
            counts["train"] = int(split_preview["train"])
            counts["val"] = int(split_preview["val"])
            model_probe = _train_model_probe(cfg)
            # Keep the historic stage-count seam (including test/extension
            # overrides) while the normal implementation reuses this exact
            # probe for BatchNorm availability rather than building again.
            cfg.__dict__["_inspect_train_model_probe"] = model_probe
            try:
                n_stages = _train_stage_count(cfg)
            finally:
                cfg.__dict__.pop("_inspect_train_model_probe", None)
            policies = [policy.to_payload() for policy in available_fine_tune_policies(cfg, n_stages=n_stages)]
            selected_policy = coerce_fine_tune_policy(cfg, n_stages=n_stages)
            training["fine_tune_policy"] = selected_policy
            if cfg.training.pretraining_objective == "byol":
                policy_message = "BYOL trains the whole encoder; supervised fine-tune policy is skipped."
            elif not is_transfer_initialization(cfg):
                policy_message = "Random/scratch initialization trains the full encoder; partial freezing is not useful."
            elif n_stages is not None:
                policy_message = f"Policy options are filtered for {cfg.model.backbone} ({n_stages} encoder stages)."
            else:
                policy_message = "Policy options are filtered for the selected model when stage metadata is available."
            # Class balance + frozen-test provenance from the last composed
            # snapshot (best-effort — absent until the first compose).
            by_class: dict = {}
            frozen_test_snap: dict = {}
            composed_split_config: dict = {}
            split_diagnostics: dict = {}
            try:
                with open(layout.dataset_dir / "dataset_snapshot.json", "r", encoding="utf-8") as sf:
                    snap_doc = json.load(sf)
                    by_class = snap_doc.get("by_class") or {}
                    frozen_test_snap = snap_doc.get("frozen_test") or {}
                    composed_split_config = snap_doc.get("split_config") or {}
                    split_diagnostics = snap_doc.get("split_diagnostics") or {}
            except Exception:  # noqa: BLE001
                by_class = {}
                frozen_test_snap = {}
                composed_split_config = {}
                split_diagnostics = {}
            split_test_enabled = effective_split_test_enabled(cfg)
            split_test_fraction = effective_split_test_fraction(cfg)
            composed_split_enabled = bool(
                composed_split_config.get(
                    "split_test_enabled",
                    composed_split_config.get("pretraining_test_enabled", split_test_enabled),
                )
            )
            composed_split_fraction = float(
                composed_split_config.get(
                    "split_test_fraction",
                    composed_split_config.get("pretraining_test_fraction", split_test_fraction),
                )
            )
            has_transfer_weights = _train_has_transfer_weights(cfg)
            pretraining_project = is_pretraining_project(cfg)
            export_supported = True
            export_message = ""
            try:
                from sed.train.backbones import resolve_adapter

                export_supported = bool(resolve_adapter(cfg.model.backbone).spec.export_supported)
            except Exception:  # noqa: BLE001 - training/export will surface details later
                export_supported = True
            if not export_supported:
                export_message = (
                    f"{cfg.model.backbone} does not support OpenVINO export; live "
                    "inference will run its PyTorch checkpoint (slower, still fully "
                    "functional)."
                )
            # Cheap CSV scan, same helper the runs tab uses — so the launch form
            # and the results table agree on what the benchmark can measure.
            test_health = _current_test_health(layout)
            by_class_i = {str(k): int(v) for k, v in by_class.items()}
            n_train = int(counts.get("train", 0))
            frozen_quality = str(frozen_test_snap.get("source_quality") or "")
            frozen_warning = (
                "Frozen test uses pinned holdouts; build a randomized bucket-aware frozen test for comparable evaluation."
                if frozen_quality == "pinned_holdouts" else ""
            )
            # Smallest per-class count of independent recording groups. This is
            # the ceiling on generalization — a corpus with many clips but few
            # groups is redundant — so the recommender keys its regularization
            # advice off it rather than off the backbone name.
            n_groups = 0
            try:
                by_class_diag = (split_preview.get("diagnostics") or {}).get("by_class") or {}
                group_counts = [
                    int(row.get("pool_groups") or 0)
                    for row in by_class_diag.values()
                    if int(row.get("pool_groups") or 0) > 0
                ]
                n_groups = min(group_counts) if group_counts else 0
            except Exception:  # noqa: BLE001 — advisory input only
                n_groups = 0
            recommended = None
            try:
                from sed.train.recommend import recommend_hparams
                recommended = recommend_hparams(
                    backbone=cfg.model.backbone,
                    pretrained=has_transfer_weights,
                    train_depth=selected_policy,
                    fine_tune_policy="",
                    by_class=by_class_i,
                    n_train=n_train,
                    classes=list(cfg.model.classes),
                    n_stages=n_stages,
                    n_groups=n_groups,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("hparam recommend failed: %s", exc)
            # Named strategy presets — each resolved against the current dataset so
            # the filled values stay adapted while the strategy is forced.
            presets: list = []
            try:
                from sed.train.recommend import (
                    TRAINING_PRESETS,
                    preset_hparams,
                    preset_is_applicable,
                )
                for preset in TRAINING_PRESETS:
                    resolved = preset_hparams(
                        preset["id"],
                        backbone=cfg.model.backbone,
                        pretrained=has_transfer_weights,
                        by_class=by_class_i,
                        n_train=n_train,
                        classes=list(cfg.model.classes),
                        n_stages=n_stages,
                        n_groups=n_groups,
                    )
                    presets.append({
                        "id": preset["id"], "label": preset["label"], "when": preset["when"],
                        "requires_init": preset.get("requires_init", "any"),
                        "requires_pretraining_project": bool(
                            preset.get("requires_pretraining_project", False)
                        ),
                        "applicable": preset_is_applicable(
                            preset,
                            has_transfer_weights=has_transfer_weights,
                            is_pretraining_project=pretraining_project,
                        ),
                        "values": resolved["values"], "reasons": resolved["reasons"],
                    })
            except Exception as exc:  # noqa: BLE001
                log.debug("training presets failed: %s", exc)
            try:
                from sed.train.objective import training_metric_descriptor
                metric_descriptor = training_metric_descriptor(cfg)
            except Exception as exc:  # noqa: BLE001
                log.debug("training metric descriptor failed: %s", exc)
                metric_descriptor = {}
            feature_readiness = _fitted_scaling_training_readiness(cfg, layout)
            return {
                "ok": True,
                "backbones": backbone_registry_payload(),
                "model": {
                    "backbone": cfg.model.backbone, "pretrained": bool(cfg.model.pretrained),
                    "recipe_id": cfg.model.recipe_id, "weight_id": cfg.model.weight_id,
                    "params": cfg.model.params, "task": cfg.model.task,
                    "initialization": cfg.model.initialization,
                    "parent_project": cfg.model.parent_project, "parent_run": cfg.model.parent_run,
                    "recipe_signature": _train_recipe_signature(cfg),
                    "dropout": _train_architecture_dropout(cfg),
                },
                "training": training,
                "batching": batching,
                "class_counts": by_class_i,
                "metric_descriptor": metric_descriptor,
                "fine_tune_options": {
                    "selected": selected_policy,
                    "policies": policies,
                    "n_stages": n_stages,
                    "has_transfer_weights": has_transfer_weights,
                    "applies": cfg.training.pretraining_objective != "byol",
                    "message": policy_message,
                },
                "recommended": recommended,
                "presets": presets,
                "after_training": {
                    "has_frozen_test": int(counts.get("test", 0)) > 0,
                    "has_val_split": int(counts.get("val", 0)) > 0,
                    "has_negative_class": "_negative" in cfg.model.classes,
                    "venue_negatives": int(by_class_i.get("_negative", 0)),
                    "export_supported": export_supported,
                    "export_message": export_message,
                    "batchnorm_layers": model_probe.get("batchnorm_layers") if model_probe else None,
                    # Probe failures should not falsely remove an otherwise
                    # valid action; a known zero is the only disabled state.
                    "batchnorm_recalibration_supported": (
                        model_probe is None or "batchnorm_layers" not in model_probe
                        or int(model_probe.get("batchnorm_layers") or 0) > 0
                    ),
                    # What the frozen test can actually answer. A set with no
                    # background clips cannot measure a false positive at all,
                    # and that is worth knowing BEFORE spending the GPU time
                    # rather than after reading a blank column.
                    "frozen_test_clips": int(test_health.get("n_samples") or 0),
                    "frozen_test_negatives": test_health.get("n_negatives"),
                },
                "dataset_config": {
                    "val_fraction": float(cfg.dataset.val_fraction),
                    "split_test_enabled": split_test_enabled,
                    "split_test_fraction": split_test_fraction,
                    "pretraining_test_enabled": split_test_enabled,
                    "pretraining_test_fraction": split_test_fraction,
                    "total_composed_rows": int(sum(counts.values())),
                    "frozen_test_present": bool(frozen_test_snap.get("present")),
                    "frozen_test_source": frozen_test_snap.get("source", ""),
                    "frozen_test_total": int(frozen_test_snap.get("total", 0)),
                    "frozen_test_source_counts": frozen_test_snap.get("source_counts", {}),
                    "frozen_test_source_quality": frozen_quality,
                    "frozen_test_warning": frozen_warning,
                    "training_class_counts": split_preview["train_by_class"],
                    "validation_class_counts": split_preview["val_by_class"],
                    "training_pool_source_counts": split_preview["pool_source_counts"],
                    # Prefer the live split at the saved val fraction. The
                    # snapshot diagnostic may have been composed with an older
                    # fraction and is only a fallback when there is no pool.
                    "split_diagnostics": (
                        split_preview["diagnostics"] or split_diagnostics
                    ),
                    "composed_with": composed_split_config,
                    # Only about the automatic test-set settings — val_fraction
                    # is no longer a compose-time concern (split live at train
                    # start), so it never makes this stale.
                    "stale": bool(composed_split_config) and (
                        composed_split_enabled != split_test_enabled
                        or abs(composed_split_fraction - split_test_fraction) > 1e-9
                    ),
                },
                "features": {
                    **f.model_dump(),
                    "sample_rate": cfg.model.sample_rate,
                    # Authoritative input height (mel bands / linear bins / image rows).
                    "n_bands": feature_band_count(f, cfg.model.sample_rate),
                },
                "feature_readiness": feature_readiness,
                "counts": counts,
            }

        def _handle_train_config(self) -> None:
            nonlocal cfg
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            if layout.config_toml.exists():
                try:
                    from sed.config import load_config

                    cfg = load_config(project_id)
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": "invalid saved config: " + str(exc)}, status=400)
                    return
            self._send_json(self._train_config_payload(cfg))

        def _handle_train_config_preview(self) -> None:
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            raw = cfg.model_dump(mode="json")
            model_over = body.get("model") or {}
            if isinstance(model_over, dict):
                target = raw.setdefault("model", {})
                if isinstance(target, dict):
                    for key in (
                        "backbone", "pretrained", "recipe_id", "weight_id", "task",
                        "initialization", "parent_project", "parent_run", "params",
                    ):
                        if key in model_over:
                            target[key] = model_over[key]
            try:
                preview_cfg = Config.model_validate(raw)
                from sed.train.model_lab import validate_config_recipe_params

                preview_cfg = validate_config_recipe_params(preview_cfg)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": "invalid preview settings: " + str(exc)}, status=400)
                return
            self._send_json(self._train_config_payload(preview_cfg))

        def _handle_dataset_test_config_get(self) -> None:
            """Frozen-test settings + current state for the Library tab's
            Test set panel. Val is not here — it's a Train-tab, train-time
            concern (see /api/train/config's dataset_config).

            Deliberately cheap: only reads config + dataset_snapshot.json.
            Staleness ("would compose change the frozen test?") is NOT
            recomputed here — that check re-scans the library per class
            (including on-demand pretraining-corpus header probes) and can
            take a minute-plus on large corpora. It's already computed once
            per render/rescan as CLIPS_DATA.frozen_context; the client reads
            that instead of triggering a second expensive pass on every
            Library tab load.
            """
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            frozen_test_snap: dict = {}
            try:
                with open(layout.dataset_dir / "dataset_snapshot.json", "r", encoding="utf-8") as sf:
                    snap_doc = json.load(sf)
                    frozen_test_snap = snap_doc.get("frozen_test") or {}
            except Exception:  # noqa: BLE001
                frozen_test_snap = {}
            from sed.data.compose import split_frozen_test_state

            try:
                split_state = split_frozen_test_state(layout)
            except Exception:  # noqa: BLE001 — display-only
                split_state = {"built": False, "clips": 0, "fraction": 0.0}
            self._send_json({
                "ok": True,
                "split_test_enabled": effective_split_test_enabled(cfg),
                "split_test_fraction": effective_split_test_fraction(cfg),
                # What the automatic holdout actually is right now, which is what
                # decides whether the panel's one button builds, rebuilds, or
                # returns the held-out clips.
                "split_test_state": split_state,
                "frozen_test_present": bool(frozen_test_snap.get("present")),
                "frozen_test_source": frozen_test_snap.get("source", ""),
                "frozen_test_total": int(frozen_test_snap.get("total", 0)),
                "frozen_test_source_counts": frozen_test_snap.get("source_counts", {}),
                "frozen_test_source_quality": frozen_test_snap.get("source_quality", ""),
                "frozen_test_split_diagnostics": frozen_test_snap.get("split_diagnostics", {}),
            })

        def _handle_dataset_split_config_post(self) -> None:
            """Persist val_fraction (Train tab) and/or the automatic test-set
            settings (Library tab). Writes config.toml only — it never
            recomposes or retrains; val applies at the next training run,
            the test-set settings apply the next time it's (re)built.
            """
            nonlocal cfg
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            updates: dict = {}
            if "val_fraction" in body:
                try:
                    v = float(body["val_fraction"])
                except (TypeError, ValueError):
                    self._send_json({"ok": False, "error": "val_fraction must be a number"}, status=400)
                    return
                if not (0.0 <= v < 0.9):
                    self._send_json({"ok": False, "error": "val_fraction must be between 0 and 0.9"}, status=400)
                    return
                updates["val_fraction"] = v
            raw_split_fraction = (
                body["split_test_fraction"]
                if "split_test_fraction" in body
                else body.get("pretraining_test_fraction")
            )
            if raw_split_fraction is not None:
                try:
                    v = float(raw_split_fraction)
                except (TypeError, ValueError):
                    self._send_json({"ok": False, "error": "split_test_fraction must be a number"}, status=400)
                    return
                if not (0.0 <= v < 0.9):
                    self._send_json({"ok": False, "error": "split_test_fraction must be between 0 and 0.9"}, status=400)
                    return
                updates["split_test_fraction"] = v
                updates["pretraining_test_fraction"] = v
            if "split_test_enabled" in body or "pretraining_test_enabled" in body:
                raw_enabled = (
                    body["split_test_enabled"]
                    if "split_test_enabled" in body
                    else body.get("pretraining_test_enabled")
                )
                updates["split_test_enabled"] = bool(raw_enabled)
                updates["pretraining_test_enabled"] = bool(raw_enabled)
            if not updates:
                self._send_json({"ok": False, "error": "no fields to update"}, status=400)
                return
            try:
                from sed.config import load_config, update_config_toml_values
                update_config_toml_values(layout.config_toml, {"dataset": updates})
                cfg = load_config(project_id)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": f"failed to write config: {exc}"}, status=500)
                return
            self._send_json({"ok": True, "dataset_config": {
                "val_fraction": float(cfg.dataset.val_fraction),
                "split_test_enabled": effective_split_test_enabled(cfg),
                "split_test_fraction": effective_split_test_fraction(cfg),
                "pretraining_test_enabled": effective_split_test_enabled(cfg),
                "pretraining_test_fraction": effective_split_test_fraction(cfg),
            }})

        # ---- Target-class editing ---------------------------------------

        def _handle_target_classes_get(self) -> None:
            from sed.data.compose import (
                ensure_class_visuals,
                load_dataset_selection,
                write_dataset_selection,
            )

            classes = list(cfg.model.classes) if cfg else []
            selection = load_dataset_selection(layout)
            class_visuals, changed = ensure_class_visuals(selection, classes)
            if changed:
                write_dataset_selection(layout, selection)
            self._send_json({
                "ok": True,
                "classes": classes,
                "class_visuals": class_visuals,
            })

        def _handle_target_classes_post(self) -> None:
            nonlocal cfg
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400); return
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON"); return
            action = str(body.get("action") or "").strip().lower()
            name = str(body.get("name") or "").strip()
            new_name = str(body.get("new_name") or "").strip()
            classes = list(cfg.model.classes)

            def _valid(n: str) -> bool:
                return bool(re.match(r"^[A-Za-z0-9_\-]+$", n))

            if action == "set_target":
                self._handle_set_target_fields(name, classes, body.get("fields"))
                return

            if action == "add":
                if not _valid(name):
                    self._send_json({"ok": False, "error": "invalid class name (use letters, digits, _ or -)"}, status=400); return
                if name in classes:
                    self._send_json({"ok": False, "error": f"class {name!r} already exists"}, status=400); return
                classes.append(name)
            elif action == "remove":
                if name not in classes:
                    self._send_json({"ok": False, "error": f"unknown class {name!r}"}, status=400); return
                if len(classes) <= 1:
                    self._send_json({"ok": False, "error": "cannot remove the last class"}, status=400); return
                classes = [c for c in classes if c != name]
            elif action == "rename":
                if name not in classes:
                    self._send_json({"ok": False, "error": f"unknown class {name!r}"}, status=400); return
                if not _valid(new_name):
                    self._send_json({"ok": False, "error": "invalid new name"}, status=400); return
                if new_name in classes:
                    self._send_json({"ok": False, "error": f"class {new_name!r} already exists"}, status=400); return
                classes = [new_name if c == name else c for c in classes]
            else:
                self._send_json({"ok": False, "error": "action must be add, remove or rename"}, status=400); return

            try:
                from sed.config import load_config, update_config_toml_values
                config_updates: dict[str, dict] = {"dataset": {"target_classes": classes}}
                config_removals: dict[str, list[str]] = {}
                # `[targets]` (role/enabled/threshold/veto_max/mine/mine_threshold)
                # must follow a rename and disappear on remove, or every one of
                # those settings silently desyncs from the renamed/removed class.
                existing_target = cfg.targets.get(name)
                if existing_target is not None:
                    if action == "rename":
                        # None fields must not be written — `_format_toml_value`
                        # has no null representation and would serialise one as
                        # the literal string "None", corrupting the entry.
                        dumped = {
                            k: v for k, v in existing_target.model_dump().items()
                            if v is not None
                        }
                        config_updates["targets"] = {new_name: dumped}
                        config_removals["targets"] = [name]
                    elif action == "remove":
                        config_removals["targets"] = [name]
                update_config_toml_values(layout.config_toml, config_updates, removals=config_removals)
                # Keep the dataset selection's per-class assignments in step.
                from sed.data.compose import (
                    CLASS_VISUALS_KEY,
                    ensure_class_visuals,
                    load_dataset_selection,
                    write_dataset_selection,
                )
                sel = load_dataset_selection(layout)
                assigns = sel.get("assignments") or {}
                if action == "rename" and name in assigns:
                    assigns[new_name] = assigns.pop(name)
                if action == "remove":
                    assigns.pop(name, None)
                sel["assignments"] = assigns
                visuals = dict(sel.get(CLASS_VISUALS_KEY) or {})
                if action == "rename" and name in visuals:
                    visuals[new_name] = visuals.pop(name)
                if action == "remove":
                    visuals.pop(name, None)
                sel[CLASS_VISUALS_KEY] = visuals
                class_visuals, _ = ensure_class_visuals(sel, classes)
                write_dataset_selection(layout, sel)
                cfg = load_config(project_id)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": f"failed to update classes: {exc}"}, status=500); return
            self._send_json({
                "ok": True,
                "classes": list(cfg.model.classes),
                "class_visuals": class_visuals,
            })

        def _handle_set_target_fields(self, name: str, known_classes: list[str], fields: object) -> None:
            """POST `/api/target-classes {action: "set_target", name, fields}`.

            Merges `fields` onto the class's existing `[targets]` entry (a
            partial update — TOML writing replaces a key's whole value, so the
            existing entry is read first and only the given fields change) and
            validates the result before writing. This is the persistence side
            of the role/mine/mine_threshold controls: nothing else in the UI
            writes `[targets]`.
            """
            nonlocal cfg
            if name not in known_classes:
                self._send_json({"ok": False, "error": f"unknown class {name!r}"}, status=400); return
            if not isinstance(fields, dict) or not fields:
                self._send_json({"ok": False, "error": "fields must be a non-empty object"}, status=400); return
            allowed = {"role", "enabled", "threshold", "veto_max", "mine", "mine_threshold", "channel", "value"}
            unknown = sorted(set(fields) - allowed)
            if unknown:
                self._send_json({"ok": False, "error": f"unknown target field(s): {unknown}"}, status=400); return
            try:
                from sed.config import TargetCfg, load_config, update_config_toml_values

                existing = cfg.targets.get(name)
                merged = existing.model_dump() if existing is not None else {}
                merged.update(fields)
                merged = {k: v for k, v in merged.items() if v is not None}
                TargetCfg.model_validate(merged)  # fail fast with a clear error before writing
                update_config_toml_values(layout.config_toml, {"targets": {name: merged}})
                cfg = load_config(project_id)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": f"failed to update target {name!r}: {exc}"}, status=400); return
            updated = cfg.targets.get(name)
            self._send_json({
                "ok": True,
                "class": name,
                "target": updated.model_dump() if updated is not None else None,
                "role": cfg.class_roles([name]).get(name),
            })

        # ---- Model tab: pluggable architectures -------------------------

        def _handle_model_catalog_get(self) -> None:
            """The richer, source-aware catalog used by pretraining projects."""
            nonlocal cfg
            from sed.train.model_lab import catalog_payload, recipe_from_config
            from sed.data.presets import feature_source_for_config
            if layout.config_toml.exists():
                try:
                    from sed.config import load_config

                    cfg = load_config(project_id)
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": "invalid saved config: " + str(exc)}, status=400)
                    return
            keep = {cfg.model.recipe_id} if cfg and cfg.model.recipe_id else set()
            # Structure of the composed train pool — rows, classes and the
            # smallest per-class recording-group count. One CSV pass, cheap
            # enough for a page-load endpoint. Sizes the per-knob
            # recommendations, which key off redundancy as well as size.
            shape = _dataset_shape(layout)
            train_rows = shape.get("rows") or _dataset_counts(layout).get("train", 0)
            payload = catalog_payload(project_id, keep_ids=keep,
                                      train_rows=train_rows, dataset=shape)
            payload.update({
                "ok": True,
                "active_recipe": recipe_from_config(cfg).to_payload() if cfg else None,
                # Not part of the model recipe (it seeds weights, it does not
                # change the architecture), but the Model tab's parent picker
                # needs the saved rows to render the head-map editor pre-filled.
                "transfer_head_map": (
                    dict(getattr(cfg.training, "transfer_head_map", {}) or {}) if cfg else {}
                ),
                "target_classes": (
                    [str(c) for c in (cfg.dataset.target_classes or cfg.model.classes or [])]
                    if cfg else []
                ),
                "feature": ({
                    **cfg.features.model_dump(),
                    "sample_rate": cfg.model.sample_rate,
                } if cfg else {}),
                "feature_source": (
                    feature_source_for_config(cfg, project_id=project_id) if cfg else {}
                ),
            })
            self._send_json(payload)

        def _handle_library_negatives_count(self) -> None:
            """Clip counts behind the FP/h-calibration NEG source dropdown.

            venue = this project's `_negative` train clips (dataset snapshot);
            library = every WAV under the shared library sounds tree (cached
            rglob — this endpoint is fetched lazily after first paint)."""
            venue = 0
            try:
                with open(layout.dataset_dir / "dataset_snapshot.json", "r", encoding="utf-8") as f:
                    snap = json.load(f)
                venue = int((snap.get("by_class") or {}).get("_negative", 0))
            except Exception:  # noqa: BLE001 — no snapshot yet → 0
                pass
            try:
                library = _library_wav_count()
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json({"ok": True, "venue": venue, "library": library})

        def _handle_model_user_specs_get(self) -> None:
            """User-registered backbone specs (presets/models/*.json) for the
            Model tab's 'Registered custom models' block."""
            try:
                from sed.train.backbones import all_specs
                rows = [spec.to_payload() for spec in all_specs().values() if not spec.builtin]
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json({"ok": True, "specs": rows})

        def _handle_model_user_specs_delete(self) -> None:
            try:
                body = self._read_json_body()
                name = str(body.get("name") or "").strip()
                if not name:
                    raise ValueError("missing spec name")
                from sed.train.backbones import delete_user_spec
                removed = delete_user_spec(name)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, "removed": removed, "name": name})

        def _handle_model_parent_candidates_get(self, qs: dict | None = None) -> None:
            """Candidate parent runs from every project, including this one.

            Optional ?backbone=/&task=/&params= (JSON) evaluate compatibility
            against the Model-tab form state instead of the persisted config,
            so the picker updates live while the user changes the recipe.
            """
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            from sed.train.transfer import compatible_parent_runs
            probe_cfg = cfg
            qs = qs or {}
            backbone = (qs.get("backbone", [""])[0] or "").strip()
            task = (qs.get("task", [""])[0] or "").strip()
            params_raw = (qs.get("params", [""])[0] or "").strip()
            model_updates: dict = {}
            if backbone:
                model_updates["backbone"] = backbone
            if task:
                model_updates["task"] = task
            if params_raw:
                try:
                    parsed = json.loads(params_raw)
                    if isinstance(parsed, dict):
                        model_updates["params"] = parsed
                except json.JSONDecodeError:
                    pass
            if model_updates:
                try:
                    probe_cfg = cfg.model_copy(
                        update={"model": cfg.model.model_copy(update=model_updates)}
                    )
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
            self._send_json({"ok": True, "candidates": compatible_parent_runs(probe_cfg)})

        def _handle_model_probe_get(self, qs: dict) -> None:
            """Real build (and optional checkpoint download) of the selected model.

            Doubles as the Model tab's "Verify model" check: with pretrained=1 it
            downloads + loads the AudioSet checkpoint and reports the true feat_dim,
            pre-pool feature map, stage count and parameter count.
            """
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            backbone = (qs.get("backbone", ["mn10_as"])[0] or "mn10_as").strip()
            pretrained = (qs.get("pretrained", [""])[0] or "").strip().lower() in ("1", "true", "yes", "on")
            try:
                from sed.audio.features import (
                    feature_band_count,
                    feature_channels,
                    feature_frame_count_for_cfg,
                )
                from sed.train.backbones import measure_model_fit
                n_bands = feature_band_count(cfg.features, cfg.model.sample_rate)
                n_frames = feature_frame_count_for_cfg(cfg)
                in_ch = feature_channels(cfg.features)
                fit = measure_model_fit(backbone, n_bands=n_bands, n_frames=n_frames,
                                        params=dict(cfg.model.params), pretrained=pretrained,
                                        input_channels=in_ch)
            except ImportError as exc:
                self._send_json({
                    "ok": False,
                    "error_kind": "missing_dependency",
                    "install": "pip install transformers accelerate",
                    "error": str(exc),
                }, status=400)
                return
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, "pretrained": pretrained, "probe": fit,
                             "input_shape": [1, int(fit.get("input_channels") or 1), n_bands, n_frames]})

        def _handle_model_task_readiness_get(self, qs: dict) -> None:
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            task = (qs.get("task", ["clip"])[0] or "clip").strip().lower()
            try:
                from sed.data.annotations import task_dataset_readiness
                report = task_dataset_readiness(cfg, task, layout.dataset_dir)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, **report})

        def _annotation_path(self, raw_path: str) -> Path | None:
            return _safe_resolve_audio_path(layout, raw_path)

        def _handle_annotation_clips_get(self) -> None:
            from sed.data.library import Library
            rows = []
            for clip in Library().all_clips(rebuild_if_empty=True):
                path = clip.absolute_path()
                if not path.exists():
                    continue
                rows.append({
                    "path": str(path), "label": clip.label, "duration_s": clip.duration_s,
                    "intervals": clip.event_intervals,
                })
            rows.sort(key=lambda row: (row["label"], row["path"]))
            self._send_json({"ok": True, "clips": rows[:1000], "truncated": len(rows) > 1000})

        def _handle_clip_annotations_get(self, qs: dict) -> None:
            raw_path = (qs.get("path", [""])[0] or "").strip()
            path = self._annotation_path(raw_path)
            if path is None or not path.exists():
                self._send_json({"ok": False, "error": "unknown clip path"}, status=404)
                return
            from sed.data.library import LibraryClip
            clip = LibraryClip.from_sidecar(path)
            self._send_json({"ok": True, "path": str(path), "label": clip.label,
                             "duration_s": clip.duration_s, "intervals": clip.event_intervals})

        def _handle_clip_annotations_propose(self) -> None:
            try:
                body = self._read_json_body()
                path = self._annotation_path(str(body.get("path") or ""))
                if path is None or not path.exists():
                    raise ValueError("unknown clip path")
                from sed.data.annotations import propose_event_intervals
                from sed.data.library import LibraryClip
                clip = LibraryClip.from_sidecar(path)
                intervals = propose_event_intervals(path, cfg, str(body.get("label") or clip.label))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, "path": str(path), "intervals": intervals,
                             "message": "Proposal only — confirm and save it before frame training."})

        def _handle_clip_annotations_save(self) -> None:
            try:
                body = self._read_json_body()
                path = self._annotation_path(str(body.get("path") or ""))
                if path is None or not path.exists():
                    raise ValueError("unknown clip path")
                intervals = body.get("intervals") or []
                from sed.data.library import Library
                clip = Library().set_event_intervals(path, intervals)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, "intervals": clip.event_intervals,
                             "message": "Intervals saved. Confirmed labels are eligible for frame SED."})

        def _handle_model_recipe_select(self) -> None:
            """Validate and persist one catalog recipe atomically into config.toml."""
            nonlocal cfg
            try:
                body = self._read_json_body()
                from sed.train.model_lab import architecture_by_id, validate_recipe_params, weight_by_id
                architecture = architecture_by_id(str(body.get("architecture_id") or ""), project_id)
                if not architecture.dependencies_available():
                    missing = " ".join(architecture.requires)
                    self._send_json({
                        "ok": False,
                        "error_kind": "missing_dependency",
                        "install": f"pip install {missing} accelerate",
                        "error": (
                            f"{architecture.label} needs the optional package(s) "
                            f"{missing!r} — install them and reload."
                        ),
                    }, status=400)
                    return
                task = str(body.get("task") or "clip").strip().lower()
                if task not in architecture.capabilities:
                    raise ValueError(f"{architecture.label} does not support {task} task")
                params = validate_recipe_params(architecture, body.get("params") or {})
                weight_id = str(body.get("weight_id") or "")
                weight = weight_by_id(architecture.id, weight_id, project_id) if weight_id else None
                initialization = str(body.get("initialization") or ("upstream" if weight else "random")).lower()
                if initialization == "upstream" and weight is None:
                    raise ValueError("select a concrete weight variant or use random initialization")
                parent_project_sel = str(body.get("parent_project") or "")
                parent_run_sel = str(body.get("parent_run") or "")
                adopt_parent_feature = bool(body.get("adopt_parent_feature"))
                # Head map rows are part of the recipe decision: which parent
                # class each of this run's classes warm-starts from. Only kept
                # for parent initialization; any other mode clears them so a
                # stale map cannot survive a switch back to scratch/upstream.
                head_map_body = body.get("transfer_head_map")
                head_map_sel: dict[str, str] = {}
                if initialization == "parent" and isinstance(head_map_body, dict):
                    head_map_sel = {
                        str(child).strip(): str(parent).strip()
                        for child, parent in head_map_body.items()
                        if str(child).strip() and str(parent).strip()
                        and str(child).strip() != str(parent).strip()
                    }
                parent_snapshot = None
                if initialization == "parent":
                    # "From previous run": any completed run (any project, incl.
                    # this one) seeds the encoder. No catalog weight is loaded.
                    if not parent_project_sel or not parent_run_sel:
                        raise ValueError("parent initialization needs parent_project and parent_run")
                    from sed.config import load_config_snapshot
                    parent_config_path = (
                        ProjectLayout(parent_project_sel).model_dir(parent_run_sel) / "training_config.json"
                    )
                    if not parent_config_path.exists():
                        raise ValueError(
                            f"parent run has no training_config.json: {parent_project_sel}/{parent_run_sel}"
                        )
                    parent_snapshot = load_config_snapshot(parent_config_path)
                    weight = None
                    weight_id = ""
                    if head_map_sel:
                        from sed.train.transfer import parent_classes as _parent_classes

                        known = set(_parent_classes(parent_project_sel, parent_run_sel))
                        unknown = sorted(set(head_map_sel.values()) - known) if known else []
                        if unknown:
                            raise ValueError(
                                "classifier head map points at class(es) the parent run does "
                                f"not have: {', '.join(unknown)}"
                            )
                if architecture.provider == "efficientat" and initialization == "upstream" and params:
                    raise ValueError(
                        "EfficientAT pretrained checkpoints use their published architecture defaults; "
                        "constructor parameters require scratch initialization"
                    )
                backbone = (weight.label if architecture.provider == "efficientat" and weight
                            else architecture.source if architecture.provider == "timm"
                            else architecture.id)
                pretrained = initialization == "upstream"
                if initialization == "random":
                    # Architecture-only recipe: no checkpoint is loaded, so no
                    # weight id is recorded, and an EfficientAT variant reduces
                    # to its base width — the hop/mels suffixes describe the
                    # checkpoint's training feature, not the network.
                    from sed.train.model_lab import efficientat_base_backbone
                    weight = None
                    weight_id = ""
                    if architecture.provider == "efficientat":
                        # Scratch-only widths (mn01…) have no checkpoint variant to
                        # derive the name from, so the UI sends the backbone itself.
                        requested_backbone = str(body.get("backbone") or "")
                        backbone = efficientat_base_backbone(requested_backbone or backbone)
                if parent_snapshot is not None:
                    # The encoder must be structurally identical to the parent's,
                    # so the parent's backbone name is adopted verbatim.
                    backbone = parent_snapshot.model.backbone
                # Advanced kwargs are tested with the real adapter before we
                # write a recipe users cannot train.
                from sed.audio.features import feature_band_count, feature_channels, feature_frame_count
                from sed.train.backbones import BackboneSpec, ParamSpec, get_spec, probe_spec, save_user_spec
                if architecture.source.startswith("project:"):
                    save_user_spec(BackboneSpec(
                        name=architecture.id, family="custom", source=architecture.source,
                        input_channels=architecture.input_channels,
                        params=[ParamSpec(**dict(item)) for item in architecture.params],
                        description=architecture.description, provider="local",
                        task_capabilities=list(architecture.capabilities),
                    ))
                spec = get_spec(backbone)
                # Optional: assign an audio feature to this recipe (the Model tab's
                # feature dropdown). Resolve + validate it so the probe and the
                # persisted config use the feature the recipe will actually train on.
                feature_body = body.get("feature")
                if parent_snapshot is not None and adopt_parent_feature:
                    # The parent's frozen training_config IS the feature contract:
                    # adopt its geometry so the transferred encoder sees the same
                    # input it was trained on.
                    from sed.data.presets import feature_cfg_to_preset
                    feature_body = feature_cfg_to_preset(parent_snapshot)
                    feature_body["name"] = ""  # contract payload, not a saved preset
                applied_feature = None
                preserved_fitted_stats = False
                if isinstance(feature_body, dict) and feature_body:
                    from sed.data.presets import apply_preset_to_features_cfg
                    target_features = apply_preset_to_features_cfg(cfg.features, feature_body)
                    target_sr = int(feature_body.get("sample_rate") or cfg.model.sample_rate)
                    target_features, preserved_fitted_stats = _preserve_fitted_linear_statistics(
                        cfg.features,
                        target_features,
                        current_sample_rate=int(cfg.model.sample_rate),
                        candidate_sample_rate=target_sr,
                    )
                    applied_feature = feature_body
                else:
                    target_features = cfg.features
                    target_sr = int(cfg.model.sample_rate)
                candidate_cfg = cfg.model_copy(update={
                    "model": cfg.model.model_copy(update={
                        "backbone": backbone, "params": params, "task": task,
                        "sample_rate": target_sr,
                        "channels": int(feature_channels(target_features)),
                    }),
                    "features": target_features,
                })
                from sed.config import validate_clink_cnn_v2_feature_contract
                validate_clink_cnn_v2_feature_contract(
                    candidate_cfg, phase="Model Lab selection",
                )
                probe = probe_spec(
                    spec,
                    n_bands=feature_band_count(target_features, target_sr),
                    n_frames=feature_frame_count(sample_rate=target_sr,
                                                 window_seconds=target_features.window_seconds,
                                                 hop_length=target_features.hop_length,
                                                 n_fft=target_features.n_fft,
                                                 frontend=target_features.frontend,
                                                 win_length=int(getattr(target_features, "win_length", 0) or 0) or None,
                                                 stft_engine=str(getattr(target_features, "stft_engine", "librosa"))),
                    params=params,
                    input_channels=int(feature_channels(target_features)),
                )
                compatibility = None
                if parent_snapshot is not None:
                    # Validate strict encoder compatibility against the recipe as
                    # it will be persisted (backbone/params/task + target feature),
                    # so an incompatible selection never reaches config.toml.
                    from sed.train.transfer import parent_compatibility
                    compatibility = parent_compatibility(candidate_cfg, parent_project_sel, parent_run_sel)
                    if not compatibility.get("ok"):
                        raise ValueError(
                            "incompatible parent run: " + str(compatibility.get("reason") or "unknown")
                        )
                from sed.config import load_config, update_config_toml_values
                model_updates = {
                    "backbone": backbone, "pretrained": pretrained, "recipe_id": architecture.id,
                    "weight_id": weight_id, "params": params, "params_contract_version": 2,
                    "task": task,
                    "initialization": initialization,
                    "parent_project": parent_project_sel if initialization == "parent" else "",
                    "parent_run": parent_run_sel if initialization == "parent" else "",
                }
                updates: dict = {
                    "model": model_updates,
                    # Always written, so leaving parent initialization clears a
                    # map that no longer applies.
                    "training": {"transfer_head_map": head_map_sel},
                }
                augmentation_defaults = _model_selection_augmentation_defaults(
                    cfg,
                    architecture_id=architecture.id,
                    initialization=initialization,
                )
                if augmentation_defaults:
                    updates["augmentation"] = augmentation_defaults
                if applied_feature is not None:
                    # Persist the assigned feature into [features] + [model].sample_rate
                    # / channels so the Train tab, export, deploy, and inference
                    # use the exact selected input contract.
                    model_updates["sample_rate"] = target_sr
                    model_updates["channels"] = int(feature_channels(target_features))
                    updates["features"] = target_features.model_dump()
                update_config_toml_values(layout.config_toml, updates)
                preset_name = str(applied_feature.get("name") or "") if applied_feature else ""
                if applied_feature is not None:
                    from sed.data.compose import load_dataset_selection, write_dataset_selection
                    from sed.data.presets import DATASET_PRESET_KEY
                    selection = load_dataset_selection(layout)
                    if preset_name:
                        selection[DATASET_PRESET_KEY] = preset_name
                    else:
                        selection.pop(DATASET_PRESET_KEY, None)
                    write_dataset_selection(layout, selection)
                cfg = load_config(project_id)
            except ImportError as exc:
                # Optional-dependency constructors (transformers encoders) raise
                # an actionable ImportError from the probe — surface it as a
                # structured install hint instead of a generic validation error.
                self._send_json({
                    "ok": False,
                    "error_kind": "missing_dependency",
                    "install": "pip install transformers accelerate",
                    "error": str(exc),
                }, status=400)
                return
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            feature_summary = None
            if cfg is not None:
                from sed.audio.features import feature_channels
                from sed.data.presets import feature_source_for_config
                f = cfg.features
                feature_summary = {
                    **f.model_dump(),
                    "sample_rate": cfg.model.sample_rate,
                    "channels": int(feature_channels(f)), "preset": f.preset,
                    "source": feature_source_for_config(cfg, project_id=project_id),
                }
            head_sources: dict = {}
            if initialization == "parent" and cfg is not None:
                from sed.train.transfer import parent_classes as _parent_classes
                from sed.train.transfer import resolve_head_sources

                head_sources = resolve_head_sources(
                    _parent_classes(parent_project_sel, parent_run_sel),
                    [str(c) for c in (cfg.dataset.target_classes or cfg.model.classes or [])],
                    head_map_sel,
                )
            self._send_json({"ok": True, "recipe": {
                "architecture_id": architecture.id, "backbone": backbone, "weight_id": weight_id,
                "params": params, "task": task, "initialization": initialization,
                "parent_project": parent_project_sel if initialization == "parent" else "",
                "parent_run": parent_run_sel if initialization == "parent" else "",
                "transfer_head_map": head_map_sel,
            }, "probe": probe, "feature": feature_summary,
                "head_sources": head_sources,
                "compatibility": compatibility,
                "preserved_fitted_linear_statistics": preserved_fitted_stats,
                "augmentation_defaults": augmentation_defaults,
                "adopted_feature": applied_feature if (parent_snapshot is not None and adopt_parent_feature) else None})

        # ---- unified activity + resource snapshot -----------------------

        def _activity_snapshot(self) -> dict:
            """One cheap, normalized view of every background job + machine load.

            The redesigned top bar polls this single endpoint (~1 Hz) instead of
            the eight per-job /status routes. Each job dict is read under its own
            lock and reduced to JSON-safe summary fields only — never the deque /
            proc / stream / runtime handles the jobs also carry."""
            terminal = {
                "finished", "error", "done", "saved", "cancelled",
                "canceled", "stopped", "idle", "",
            }
            grace = 8.0  # keep just-finished jobs briefly so the bar can announce them

            def gate(job: dict) -> tuple[bool, bool]:
                """(include, finished) — running jobs are always included; a
                terminal job lingers for `grace` seconds so the frontend can fire
                exactly one completion toast for it."""
                state = str(job.get("state") or "").strip().lower()
                if state not in terminal:
                    return True, False
                ended = job.get("ended_at")
                if ended is not None and (time.time() - float(ended)) <= grace:
                    return True, True
                return False, False

            def elapsed_of(job: dict) -> float:
                started = float(job.get("started_at") or time.time())
                ended = job.get("ended_at")
                return max(0.0, float(ended or time.time()) - started)

            def eta_from_progress(job: dict, progress) -> float | None:
                try:
                    p = float(progress)
                except (TypeError, ValueError):
                    return None
                el = elapsed_of(job)
                if 0.0 < p < 1.0 and el > 0:
                    return max(0.0, el * (1.0 - p) / p)
                return None

            jobs: list[dict] = []
            active_count = 0
            train_active = False
            train_workers = 0

            def emit(entry: dict, finished: bool) -> None:
                nonlocal active_count
                entry["finished"] = finished
                jobs.append(entry)
                if not finished:
                    active_count += 1

            # -- training (subprocess: the machine's heaviest consumer) --
            with train_lock:
                tj = dict(train_job) if train_job else None
            if tj:
                inc, fin = gate(tj)
                if inc:
                    model_id = tj.get("model_id") or ""
                    metrics = _parse_train_metrics(layout.model_dir(model_id)) if model_id else []
                    epoch = (max((m.get("epoch") or 0) for m in metrics) + 1) if metrics else 0
                    max_epochs = int(tj.get("max_epochs") or 0)
                    progress = (epoch / max_epochs) if max_epochs else None
                    detail = f"epoch {epoch}/{max_epochs}" if max_epochs else "starting"
                    emit({
                        "type": "train", "job_id": tj.get("job_id"), "state": tj.get("state"),
                        "title": (f"Training {model_id}".strip() if model_id else "Training"),
                        "detail": detail, "progress": progress,
                        "elapsed_s": elapsed_of(tj), "eta_s": None,
                        "tab": "train", "section": "live",
                    }, fin)
                    if not fin:
                        train_active = True
                        train_workers = int(getattr(getattr(cfg, "training", None), "num_workers", 0) or 0)

            # -- dataset compose / frozen-test build --
            with compose_lock:
                cj = dict(compose_job) if compose_job else None
            if cj:
                inc, fin = gate(cj)
                if inc:
                    is_frozen = bool(cj.get("build_split_test"))
                    progress = cj.get("progress")
                    emit({
                        "type": "compose", "job_id": cj.get("job_id"), "state": cj.get("state"),
                        "title": ("Building frozen test" if is_frozen else "Composing dataset"),
                        "detail": str(cj.get("phase_detail") or cj.get("phase") or ""),
                        "progress": progress, "elapsed_s": elapsed_of(cj),
                        "eta_s": eta_from_progress(cj, progress),
                        "tab": ("frozen" if is_frozen else "dataset"), "section": None,
                    }, fin)

            # -- augmentation generation --
            with augment_lock:
                aj = dict(augment_job) if augment_job else None
            if aj:
                inc, fin = gate(aj)
                if inc:
                    total = int(aj.get("total") or 0)
                    completed = int(aj.get("completed") or 0)
                    progress = (completed / total) if total else None
                    emit({
                        "type": "augment", "job_id": aj.get("job_id"), "state": aj.get("state"),
                        "title": "Augmenting clips",
                        "detail": (str(aj.get("current_recipe") or "") or (f"{completed}/{total}" if total else "")),
                        "progress": progress, "elapsed_s": elapsed_of(aj),
                        "eta_s": eta_from_progress(aj, progress),
                        "tab": "augmented", "section": None,
                    }, fin)

            # -- assigned preview-cache build --
            with preview_cache_lock:
                pj = dict(preview_cache_job) if preview_cache_job else None
            if pj:
                inc, fin = gate(pj)
                if inc:
                    total = int(pj.get("total") or 0)
                    completed = int(pj.get("completed") or 0)
                    progress = (completed / total) if total else None
                    emit({
                        "type": "preview_cache", "job_id": pj.get("job_id"), "state": pj.get("state"),
                        "title": "Building preview cache",
                        "detail": str(pj.get("phase") or (f"{completed}/{total}" if total else "")),
                        "progress": progress, "elapsed_s": elapsed_of(pj),
                        "eta_s": eta_from_progress(pj, progress),
                        "tab": "audiofeatures", "section": None,
                    }, fin)

            # -- train-only dataset normalization fit --
            with feature_stats_lock:
                fsj = dict(feature_stats_job) if feature_stats_job else None
            if fsj:
                inc, fin = gate(fsj)
                if inc:
                    total = int(fsj.get("total") or 0)
                    completed = int(fsj.get("completed") or 0)
                    progress = (completed / total) if total else None
                    detail = f"{completed:,}/{total:,} clips"
                    groups = int(fsj.get("groups") or 0)
                    if groups:
                        detail += f" · {groups:,} groups"
                    emit({
                        "type": "feature_stats", "job_id": fsj.get("job_id"),
                        "state": fsj.get("state"), "title": "Fitting dataset normalization",
                        "detail": detail, "progress": progress,
                        "elapsed_s": elapsed_of(fsj), "eta_s": eta_from_progress(fsj, progress),
                        "tab": "audiofeatures", "section": None,
                    }, fin)

            # -- review-data / metadata refresh (no nav target) --
            with refresh_lock:
                rj = dict(refresh_job) if refresh_job else None
            if rj:
                inc, fin = gate(rj)
                if inc:
                    progress = rj.get("progress")
                    emit({
                        "type": "review_data", "job_id": rj.get("job_id"), "state": rj.get("state"),
                        "title": "Refreshing metadata",
                        "detail": str(rj.get("phase_detail") or rj.get("phase") or ""),
                        "progress": progress, "elapsed_s": elapsed_of(rj),
                        "eta_s": eta_from_progress(rj, progress),
                        "tab": None, "section": None,
                    }, fin)

            # -- synchronous clip pages and spectrogram requests ------------
            # These are not background jobs with a status endpoint, but they
            # can scan thousands of files or wait for a preview worker. Keep
            # them in the same bar so "Idle · ready" is never shown during
            # visible clip loading/search work.
            with clip_activity_lock:
                clip_jobs = [dict(job) for job in clip_activity_jobs.values()]
            for clip_job in clip_jobs:
                emit({
                    "type": "clip_load", "job_id": clip_job.get("job_id"),
                    "state": clip_job.get("state", "running"),
                    "title": clip_job.get("title") or "Loading clips",
                    "detail": clip_job.get("detail") or "",
                    "progress": None, "elapsed_s": elapsed_of(clip_job),
                    "eta_s": None, "tab": clip_job.get("tab"),
                    "section": None,
                }, False)

            # -- cached audio-integrity scan --
            with integrity_lock:
                ij = dict(integrity_job) if integrity_job else None
            if ij:
                inc, fin = gate(ij)
                if inc:
                    total = int(ij.get("total") or 0)
                    completed = int(ij.get("completed") or 0)
                    progress = (completed / total) if total else ij.get("progress")
                    emit({
                        "type": "integrity", "job_id": ij.get("job_id"), "state": ij.get("state"),
                        "title": "Scanning audio integrity",
                        "detail": str(ij.get("phase_detail") or ij.get("current_name") or ""),
                        "progress": progress, "elapsed_s": elapsed_of(ij),
                        "eta_s": eta_from_progress(ij, progress),
                        "tab": "library", "section": "integrity",
                    }, fin)

            # -- pretraining corpus header rescan --
            with pretraining_rescan_lock:
                prj = dict(pretraining_rescan_job) if pretraining_rescan_job else None
            if prj:
                inc, fin = gate(prj)
                if inc:
                    emit({
                        "type": "pretraining_rescan", "job_id": prj.get("job_id"), "state": prj.get("state"),
                        "title": "Rescanning corpus headers",
                        "detail": str(prj.get("phase_detail") or prj.get("phase") or ""),
                        "progress": prj.get("progress"), "elapsed_s": elapsed_of(prj),
                        "eta_s": eta_from_progress(prj, prj.get("progress")),
                        "tab": "library", "section": "pretraining",
                    }, fin)

            # -- lightweight completed actions (preset/rule edits, etc.) --
            with activity_events_lock:
                recent_events = list(activity_events)
            for event in recent_events:
                inc, fin = gate(event)
                if inc:
                    emit({
                        "type": event.get("type") or "action", "job_id": event.get("job_id"),
                        "state": event.get("state"), "title": event.get("title") or "Action",
                        "detail": event.get("detail") or "", "progress": 1.0,
                        "elapsed_s": elapsed_of(event), "eta_s": None,
                        "tab": event.get("tab"), "section": event.get("section"),
                    }, fin)

            # -- delete (no nav target) --
            with delete_lock:
                dj = dict(delete_job) if delete_job else None
            if dj:
                inc, fin = gate(dj)
                if inc:
                    total = int(dj.get("total") or 0)
                    completed = int(dj.get("completed") or 0)
                    progress = (completed / total) if total else None
                    emit({
                        "type": "delete", "job_id": dj.get("job_id"), "state": dj.get("state"),
                        "title": "Deleting clips",
                        "detail": (str(dj.get("current_name") or "") or (f"{completed}/{total}" if total else "")),
                        "progress": progress, "elapsed_s": elapsed_of(dj), "eta_s": None,
                        "tab": None, "section": None,
                    }, fin)

            # -- batch model evaluation / calibration --
            with batch_eval_lock:
                reap_batch_eval_job()
                bj = {k: v for k, v in batch_eval_job.items() if k != "thread"} if batch_eval_job else None
            if bj:
                inc, fin = gate(bj)
                if inc:
                    total = int(bj.get("total") or 0)
                    completed = int(bj.get("completed") or 0)
                    progress = (completed / total) if total else None
                    current = str(bj.get("current_model") or "")
                    step = str(bj.get("current_step") or "")
                    detail = (f"{current} · {step}" if current and step else current) or (
                        f"{completed}/{total}" if total else ""
                    )
                    emit({
                        "type": "batch_eval", "job_id": bj.get("job_id"), "state": bj.get("state"),
                        "title": "Batch evaluating models",
                        "detail": detail,
                        "progress": progress, "elapsed_s": elapsed_of(bj),
                        "eta_s": eta_from_progress(bj, progress),
                        "tab": "train", "section": "runs",
                    }, fin)

            # -- one-run Frozen Test evaluation --
            with frozen_eval_lock:
                fej = {k: v for k, v in frozen_eval_job.items() if k != "thread"} if frozen_eval_job else None
            if fej:
                inc, fin = gate(fej)
                if inc:
                    total = int(fej.get("total") or 0)
                    completed = int(fej.get("completed") or 0)
                    progress = (completed / total) if total else None
                    phase = str(fej.get("phase") or "evaluating")
                    current_class = str(fej.get("current_class") or "")
                    current_name = str(fej.get("current_name") or "")
                    count = f"{completed:,}/{total:,}" if total else ""
                    detail = " · ".join(
                        value for value in (phase, count, current_class, current_name) if value
                    )
                    emit({
                        "type": "frozen_eval", "job_id": fej.get("job_id"),
                        "state": fej.get("state"),
                        "title": f"Evaluating {fej.get('model_id') or 'frozen test'}",
                        "detail": detail, "progress": progress,
                        "elapsed_s": elapsed_of(fej),
                        "eta_s": eta_from_progress(fej, progress),
                        "tab": "frozen", "section": None,
                    }, fin)

            # -- model-evidence generation for Frozen/Mined review --
            with explain_lock:
                ej = {k: v for k, v in explain_job.items() if k != "thread"} if explain_job else None
            if ej:
                inc, fin = gate(ej)
                if inc:
                    total = int(ej.get("total") or 0)
                    completed = int(ej.get("completed") or 0)
                    progress = (completed / total) if total else None
                    scope = str(ej.get("current_scope") or "")
                    count = f"{completed}/{total} scopes" if total else ""
                    detail = " · ".join(
                        value for value in (str(ej.get("model_id") or ""), scope, count) if value
                    )
                    emit({
                        "type": "explain", "job_id": ej.get("job_id"),
                        "state": ej.get("state"), "title": "Explaining predictions",
                        "detail": detail, "progress": progress,
                        "elapsed_s": elapsed_of(ej),
                        "eta_s": eta_from_progress(ej, progress),
                        "tab": scope if scope in {"frozen", "mined"} else "frozen",
                        "section": None,
                    }, fin)

            # -- live inference (indeterminate) --
            with inference_lock:
                ij = dict(inference_job) if inference_job else None
            if ij:
                inc, fin = gate(ij)
                if inc:
                    emit({
                        "type": "inference", "job_id": ij.get("job_id"), "state": ij.get("state"),
                        "title": "Live inference",
                        "detail": str(ij.get("model_id") or "listening"),
                        "progress": None, "elapsed_s": elapsed_of(ij), "eta_s": None,
                        "tab": "train", "section": "inference",
                    }, fin)

            # -- machine load (dependency-free, real values on Linux) --
            try:
                cpu_count = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                cpu_count = os.cpu_count() or 1
            cpu_count = max(1, cpu_count)
            try:
                load1, load5, load15 = os.getloadavg()
            except (OSError, AttributeError):
                load1 = load5 = load15 = 0.0
            load_frac = min(1.0, load1 / cpu_count)
            now_mono = time.monotonic()
            now_cpu = time.process_time()
            wall_delta = max(0.001, now_mono - float(activity_proc_sample.get("wall") or now_mono))
            cpu_delta = max(0.0, now_cpu - float(activity_proc_sample.get("cpu") or now_cpu))
            activity_proc_sample.update({"wall": now_mono, "cpu": now_cpu})
            process_cpu_pct = min(1000.0, 100.0 * cpu_delta / wall_delta)
            process_rss_mb = None
            try:
                import psutil
                process_rss_mb = float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
            except Exception:  # noqa: BLE001 — telemetry must never break activity.
                try:
                    import resource
                    process_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
                except Exception:  # noqa: BLE001
                    pass
            if train_active or load_frac > 0.85:
                level = "heavy"
            elif active_count or load_frac > 0.5:
                level = "busy"
            else:
                level = "free"
            resources = {
                "cpu_count": cpu_count,
                "loadavg": [round(load1, 2), round(load5, 2), round(load15, 2)],
                "load_frac": round(load_frac, 3),
                "active_threads": threading.active_count(),
                "process_cpu_percent": round(process_cpu_pct, 1),
                "process_rss_mb": round(process_rss_mb, 1) if process_rss_mb is not None else None,
                "process_threads": threading.active_count(),
                "cpu_percent": round(process_cpu_pct, 1),
                "rss_mb": round(process_rss_mb, 1) if process_rss_mb is not None else None,
                "thread_count": threading.active_count(),
                "measurement_scope": "Inspect server process-wide; shared by concurrent jobs",
                "training_active": train_active,
                "train_workers": train_workers,
                "level": level,
            }
            for entry in jobs:
                entry["resources"] = {
                    "process_cpu_percent": resources["process_cpu_percent"],
                    "process_rss_mb": resources["process_rss_mb"],
                    "process_threads": resources["process_threads"],
                    "cpu_percent": resources["cpu_percent"],
                    "rss_mb": resources["rss_mb"],
                    "thread_count": resources["thread_count"],
                    "measurement_scope": resources["measurement_scope"],
                }
            return {"ok": True, "jobs": jobs, "resources": resources}

        def _handle_activity(self) -> None:
            self._send_json(self._activity_snapshot())

        def _handle_train_status(self) -> None:
            with train_lock:
                if not train_job:
                    self._send_json({"ok": True, "job": {"state": "idle"}, "metrics": []})
                    return
                state = train_job.get("state", "idle")
                model_id = train_job.get("model_id") or ""
                max_epochs = train_job.get("max_epochs") or 0
                started = float(train_job.get("started_at") or time.time())
                ended = train_job.get("ended_at")
                error = train_job.get("error", "")
                log_tail = "\n".join(list(train_job.get("log") or []))
                # Replicate sweep progress. Always present — an ordinary launch
                # is a sweep of one, so the UI has no second shape to handle.
                seed = train_job.get("seed")
                seed_index = int(train_job.get("seed_index") or 0)
                seed_total = int(train_job.get("seed_total") or 1)
                seeds = list(train_job.get("seeds") or [])
                replicates = [dict(r) for r in (train_job.get("replicates") or [])]
            metrics = _parse_train_metrics(layout.model_dir(model_id)) if model_id else []
            epoch = (max((m.get("epoch") or 0) for m in metrics) + 1) if metrics else 0
            self._send_json({
                "ok": True,
                "job": {
                    "state": state, "model_id": model_id, "epoch": epoch,
                    "max_epochs": max_epochs,
                    "elapsed_s": max(0.0, float(ended or time.time()) - started),
                    "error": error, "log_tail": log_tail,
                    "seed": seed, "seed_index": seed_index, "seed_total": seed_total,
                    "seeds": seeds, "replicates": replicates,
                },
                "metrics": metrics,
            })

        def _handle_train_cancel(self) -> None:
            with train_lock:
                if not train_job or str(train_job.get("state")) not in ("starting", "running", "stopping"):
                    self._send_json({"ok": False, "error": "no running training job"}, status=404)
                    return
                train_job["cancel"] = True
                train_job["state"] = "canceling"
                try:
                    Path(str(train_job.get("stop_file") or "")).unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
                proc = train_job.get("proc")
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
            self._send_json({"ok": True})

        def _handle_train_stop_save(self) -> None:
            with train_lock:
                if not train_job or str(train_job.get("state")) not in ("starting", "running"):
                    self._send_json({"ok": False, "error": "no running training job"}, status=404)
                    return
                stop_file = Path(str(train_job.get("stop_file") or ""))
                train_job["stop_requested"] = True
                train_job["state"] = "stopping"
                train_job["log"].append(
                    "[inspect] Stop & save requested; training will finish the current epoch and finalize the model."
                )
            try:
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.write_text(_now_iso(), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                with train_lock:
                    if train_job:
                        train_job["state"] = "running"
                        train_job["stop_requested"] = False
                self._send_json({"ok": False, "error": "failed to request graceful stop: " + str(exc)}, status=500)
                return
            self._send_json({"ok": True})

        def _handle_train_resource_estimate(self) -> None:
            """Measure what the configured run would cost before launching it.

            Builds the same effective config train-start would freeze (without
            persisting it) and runs sed.train.resource_estimate as a subprocess
            on the current form values: real optimizer steps on synthetic
            batches, so the CUDA context and every allocation are released when
            it exits and cannot crowd out an actual training run.
            """
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            with train_lock:
                if train_job and str(train_job.get("state")) in ("starting", "running", "stopping", "canceling"):
                    self._send_json({
                        "ok": False,
                        "error": "a training run is in progress — the GPU is busy",
                    }, status=409)
                    return
            if not estimate_lock.acquire(blocking=False):
                self._send_json({"ok": False, "error": "an estimate is already running"}, status=409)
                return
            try:
                try:
                    effective_cfg, _updates = self._train_effective_updates(body, cfg)
                except ValueError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
                split_preview = _preview_train_val_split(layout, effective_cfg)

                import tempfile
                from sed.config import REPO_ROOT
                tmp = tempfile.NamedTemporaryFile(
                    "w", suffix=".json", prefix="sed_estimate_", delete=False,
                )
                try:
                    json.dump(effective_cfg.model_dump(mode="json"), tmp)
                    tmp.close()
                    cmd = [
                        sys.executable, "-m", "sed.train.resource_estimate", tmp.name,
                        "--train-rows", str(int(split_preview.get("train", 0))),
                        "--val-rows", str(int(split_preview.get("val", 0))),
                        "--dataset-dir", str(layout.dataset_dir),
                    ]
                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    result = subprocess.run(
                        cmd, cwd=str(REPO_ROOT), capture_output=True,
                        text=True, timeout=300, env=env,
                    )
                    payload = None
                    for line in reversed((result.stdout or "").strip().splitlines()):
                        try:
                            payload = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                    if not isinstance(payload, dict):
                        tail = " | ".join((result.stderr or "").strip().splitlines()[-3:])
                        self._send_json({
                            "ok": False,
                            "error": "estimate failed: " + (tail or f"exit code {result.returncode}"),
                        }, status=500)
                        return
                    self._send_json(payload)
                except subprocess.TimeoutExpired:
                    self._send_json({"ok": False, "error": "estimate timed out after 300 s"}, status=500)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
            finally:
                estimate_lock.release()

        def _train_effective_updates(self, body: dict, base_cfg: Config) -> tuple[Config, dict]:
            """Clean a Train-tab override payload into config.toml ``updates``
            plus the validated effective Config (fine-tune policy coerced the
            same way the saved run config will be). Shared by train-start,
            which persists ``updates``, and resource-estimate, which must see
            exactly the config a run would use without persisting anything.
            Raises ValueError on invalid settings.
            """
            updates: dict[str, dict] = {}
            model_over = body.get("model") or {}
            m: dict = {}
            if "backbone" in model_over:
                m["backbone"] = str(model_over["backbone"])
            if "pretrained" in model_over:
                m["pretrained"] = bool(model_over["pretrained"])
            for key in ("recipe_id", "weight_id", "task", "initialization", "parent_project", "parent_run"):
                if key in model_over:
                    m[key] = str(model_over[key] or "")
            if "params_contract_version" in model_over:
                m["params_contract_version"] = int(model_over["params_contract_version"] or 1)
            if "params" in model_over and isinstance(model_over["params"], dict):
                m["params"] = model_over["params"]
            effective_initialization = str(
                m.get("initialization", base_cfg.model.initialization) or "upstream"
            )
            if effective_initialization == "random":
                # Keep the frozen recipe honest for scratch runs: no checkpoint
                # loads, so no weight id is recorded, and an EfficientAT variant
                # name reduces to its base width architecture.
                from sed.train.model_lab import efficientat_base_backbone
                m["pretrained"] = False
                if str(m.get("weight_id", base_cfg.model.weight_id) or ""):
                    m["weight_id"] = ""
                backbone_name = str(m.get("backbone", base_cfg.model.backbone) or "")
                if backbone_name.startswith(("mn", "dymn")):
                    m["backbone"] = efficientat_base_backbone(backbone_name)
            if m:
                updates["model"] = m
            training_over = body.get("training") or {}
            clean: dict = {}
            for k in self._TRAIN_INT:
                if k in training_over:
                    try:
                        clean[k] = int(training_over[k])
                    except (TypeError, ValueError):
                        pass
            for k in self._TRAIN_FLOAT:
                if k in training_over:
                    try:
                        clean[k] = float(training_over[k])
                    except (TypeError, ValueError):
                        pass
            for k in self._TRAIN_STR:
                if k in training_over:
                    clean[k] = str(training_over[k])
            for k in self._TRAIN_BOOL:
                if k in training_over:
                    clean[k] = bool(training_over[k])
            for k in self._TRAIN_STR_LIST:
                if k in training_over and isinstance(training_over[k], list):
                    clean[k] = [str(v) for v in training_over[k]]
            if "fine_tune_policy" in clean:
                from sed.train.fine_tuning import normalize_fine_tune_policy
                clean["fine_tune_policy"] = normalize_fine_tune_policy(clean["fine_tune_policy"])
                effective_pretrained = bool(m.get("pretrained", base_cfg.model.pretrained))
                effective_initialization = str(m.get("initialization", base_cfg.model.initialization) or "upstream")
                if effective_initialization == "random" or (
                    not effective_pretrained and effective_initialization != "parent"
                ):
                    clean["fine_tune_policy"] = "full"
            if clean:
                updates["training"] = clean

            effective_cfg = base_cfg
            if updates:
                raw = base_cfg.model_dump(mode="json")
                for section, section_updates in updates.items():
                    target = raw.setdefault(section, {})
                    if isinstance(target, dict):
                        target.update(section_updates)
                    else:
                        raw[section] = dict(section_updates)
                try:
                    effective_cfg = Config.model_validate(raw)
                    policy_model_keys = ("backbone", "pretrained", "initialization", "params")
                    should_coerce_policy = (
                        effective_cfg.training.fine_tune_policy != "manual"
                        and (
                            "fine_tune_policy" in clean
                            or any(k in m for k in policy_model_keys)
                        )
                    )
                    if should_coerce_policy:
                        from sed.train.fine_tuning import coerce_fine_tune_policy

                        # Use the same stage probe as the config endpoint so
                        # the saved run config matches the depth shown in the UI.
                        coerced_policy = coerce_fine_tune_policy(
                            effective_cfg,
                            n_stages=_train_stage_count(effective_cfg),
                        )
                        if coerced_policy != effective_cfg.training.fine_tune_policy:
                            updates.setdefault("training", {})["fine_tune_policy"] = coerced_policy
                            raw.setdefault("training", {})["fine_tune_policy"] = coerced_policy
                            effective_cfg = Config.model_validate(raw)
                except Exception as exc:  # noqa: BLE001
                    raise ValueError("invalid training settings: " + str(exc)) from exc
            try:
                from sed.train.model_lab import validate_config_recipe_params

                effective_cfg = validate_config_recipe_params(effective_cfg)
            except Exception as exc:  # noqa: BLE001
                raise ValueError("invalid model constructor parameters: " + str(exc)) from exc
            return effective_cfg, updates

        def _handle_train_start(self) -> None:
            nonlocal cfg, train_job
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            with train_lock:
                if train_job and str(train_job.get("state")) in ("starting", "running", "stopping", "canceling"):
                    self._send_json({"ok": False, "error": "a training run is already in progress"}, status=409)
                    return

            # Persist hyperparameter overrides into config.toml (so they freeze
            # into the bundle exactly like a CLI run).
            from sed.config import load_config, update_config_toml_values
            try:
                # Hyperparameters in the Train form are intentionally staged
                # until Start.  Reload first so a long-lived Inspect page never
                # overwrites a setting saved by another UI action.
                cfg = load_config(project_id)
                effective_cfg, updates = self._train_effective_updates(body, cfg)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            try:
                if updates:
                    update_config_toml_values(layout.config_toml, updates)
                    cfg = load_config(project_id)
                    effective_cfg = cfg
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": "failed to write config: " + str(exc)}, status=500)
                return

            counts = _dataset_counts(layout)
            if counts.get("train", 0) < 1:
                self._send_json({"ok": False, "error": "no training data — compose a dataset first"}, status=400)
                return
            from sed.data.compose import load_dataset_dirty

            dirty = load_dataset_dirty(layout)
            if dirty:
                count = int(dirty.get("count") or len(dirty.get("paths") or []))
                self._send_json({
                    "ok": False,
                    "error": (
                        f"dataset references {count} deleted source clip"
                        f"{'s' if count != 1 else ''} — run Compose before training"
                    ),
                    "compose_required": True,
                    "dataset_dirty": dirty,
                }, status=409)
                return

            # A seed/validation split change changes the exact train-only rows
            # that dataset-fitted feature scaling sees.  Refresh those
            # moments after saving the staged Train values and before launching
            # the CLI child, so changing a Train control cannot create the
            # familiar stale-statistics failure.
            fitted_stats = None
            try:
                fitted_stats = _refresh_stale_fitted_feature_scaling(
                    effective_cfg, layout,
                )
                if fitted_stats is not None:
                    cfg = load_config(project_id)
                    effective_cfg = cfg
            except Exception as exc:  # noqa: BLE001 — surface a safe launch gate
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return

            description = str(body.get("description") or "").strip()
            # FP/h calibration lives in config.toml ([training] calibrate_after)
            # and is executed by `sed train` itself, so the CLI and this panel
            # produce identical bundles and the checkbox survives a page reload.
            max_epochs = int(effective_cfg.training.epochs)

            base_seed = int(effective_cfg.training.seed)
            try:
                seeds = _replicate_seeds(body, base_seed)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return

            with train_lock:
                job_id = uuid.uuid4().hex[:12]
                stop_file = layout.root / f".train_stop_{job_id}"
                stop_file.unlink(missing_ok=True)
                train_job = {
                    "job_id": job_id, "state": "starting", "model_id": "",
                    "max_epochs": max_epochs, "started_at": time.time(), "ended_at": None,
                    "error": "", "cancel": False, "stop_requested": False,
                    "stop_file": str(stop_file),
                    "log": collections.deque(maxlen=400), "proc": None,
                    # Replicate sweep. `seeds` is [base] for an ordinary launch,
                    # so every consumer can read these unconditionally.
                    "seeds": list(seeds), "seed": seeds[0], "base_seed": base_seed,
                    "seed_index": 0, "seed_total": len(seeds), "replicates": [],
                }

            def run_training() -> None:
                nonlocal train_job, cfg
                from sed.config import REPO_ROOT, load_config, update_config_toml_values
                script = Path(sys.executable).with_name("sed")
                if script.exists():
                    cmd = [str(script), "train", "--project", project_id]
                else:
                    cmd = [sys.executable, "-c", "from sed.cli import app; app()", "train", "--project", project_id]
                if description:
                    cmd += ["--description", description]

                def append(line: str) -> None:
                    with train_lock:
                        if train_job and train_job.get("job_id") == job_id:
                            train_job["log"].append(line.rstrip("\n"))

                def replace_last(line: str) -> None:
                    with train_lock:
                        if not train_job or train_job.get("job_id") != job_id:
                            return
                        log_rows = train_job["log"]
                        if log_rows:
                            log_rows[-1] = line.rstrip("\r\n")
                        else:
                            log_rows.append(line.rstrip("\r\n"))

                def run_one() -> tuple[str, str]:
                    """One `sed train` child, start to finish → (state, error).

                    Everything seed-specific has already been written to
                    config.toml by the caller, so the child is an ordinary CLI
                    run and its bundle freezes the same config a manual
                    `sed train` would.
                    """
                    # Recomputed per replicate: `_find_new_model` identifies the
                    # bundle this child created by diffing against what existed
                    # before it started, and the previous replicate's bundle is
                    # now part of "before".
                    existing = (
                        {p.name for p in layout.models_dir.iterdir() if p.is_dir()}
                        if layout.models_dir.exists() else set()
                    )
                    try:
                        append("[inspect] launching: " + " ".join(shlex.quote(part) for part in cmd))
                        env = os.environ.copy()
                        env["PYTHONUNBUFFERED"] = "1"
                        env.setdefault("PYTHONIOENCODING", "utf-8")
                        with train_lock:
                            stop_file_env = str(train_job.get("stop_file") or "") if train_job else ""
                        if stop_file_env:
                            env["SED_TRAIN_STOP_FILE"] = stop_file_env
                        proc = subprocess.Popen(
                            cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=0, env=env,
                        )
                    except Exception as exc:  # noqa: BLE001
                        return "error", str(exc)

                    with train_lock:
                        if train_job and train_job.get("job_id") == job_id:
                            train_job["proc"] = proc
                            train_job["state"] = "stopping" if train_job.get("stop_requested") else "running"

                    assert proc.stdout is not None
                    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                    current_line = ""
                    inline_active = False

                    def commit_current() -> None:
                        nonlocal current_line, inline_active
                        if inline_active:
                            replace_last(current_line)
                        elif current_line:
                            append(current_line)
                        current_line = ""
                        inline_active = False

                    def process_text(text: str) -> None:
                        nonlocal current_line, inline_active
                        for ch in text:
                            if ch == "\r":
                                if current_line and not inline_active:
                                    append(current_line)
                                current_line = ""
                                if inline_active:
                                    replace_last("")
                                else:
                                    append("")
                                inline_active = True
                                continue
                            if ch == "\n":
                                commit_current()
                                continue
                            current_line += ch
                            if inline_active:
                                replace_last(current_line)

                    while True:
                        chunk = proc.stdout.read(128)
                        if not chunk:
                            break
                        process_text(decoder.decode(chunk))
                        with train_lock:
                            known = bool(train_job and train_job.get("model_id"))
                        if not known:
                            mid = _find_new_model(layout, existing)
                            if mid:
                                with train_lock:
                                    if train_job and train_job.get("job_id") == job_id:
                                        train_job["model_id"] = mid
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        process_text(tail)
                    commit_current()
                    rc = proc.wait()

                    with train_lock:
                        cancelled = bool(train_job and train_job.get("cancel"))
                        stop_requested = bool(train_job and train_job.get("stop_requested"))
                        if train_job and train_job.get("job_id") == job_id and not train_job.get("model_id"):
                            train_job["model_id"] = _find_new_model(layout, existing)
                        final_model_id = str(train_job.get("model_id") or "") if train_job else ""
                        if train_job and train_job.get("job_id") == job_id:
                            train_job["proc"] = None

                    if cancelled:
                        if final_model_id:
                            model_dir = layout.model_dir(final_model_id)
                            if not (model_dir / "manifest.json").exists():
                                try:
                                    shutil.rmtree(model_dir, ignore_errors=True)
                                    append(f"[inspect] discarded partial model bundle {final_model_id}")
                                except Exception as exc:  # noqa: BLE001
                                    append(f"[inspect] partial bundle cleanup failed: {exc}")
                        return "cancelled", ""
                    if rc == 0:
                        # The FP/h sweep runs inside `sed train` now (train.py,
                        # from [training] calibrate_after), so a CLI run gets it
                        # too and it lands before the frozen-test scoring rather
                        # than after. Running it again here would sweep a second
                        # time and, worse, against the newly-active model
                        # instead of the trained one.
                        return ("saved" if stop_requested else "done"), ""
                    return "error", f"training exited with code {rc}"

                def set_seed(seed: int) -> None:
                    """Point the project at one replicate's seed and re-fit for it.

                    The seed selects the train/val split, so dataset-fitted
                    feature scaling has to be re-measured on the rows that split
                    actually leaves in train — the launch gate in train.py
                    rejects statistics whose split seed disagrees. Refitting per
                    replicate is what makes each one internally consistent
                    rather than sharing one replicate's normalization.
                    """
                    update_config_toml_values(layout.config_toml, {"training": {"seed": int(seed)}})
                    seed_cfg = load_config(project_id)
                    refit = _refresh_stale_fitted_feature_scaling(seed_cfg, layout)
                    if refit is not None:
                        append(
                            f"[inspect] refitted feature scaling for seed {seed} "
                            f"({refit.get('clips')} train clips)"
                        )

                total = len(seeds)
                final_state, err = "done", ""
                for index, seed in enumerate(seeds):
                    with train_lock:
                        if not train_job or train_job.get("job_id") != job_id:
                            return
                        # Cancel and Stop both end the sweep: the user asked for
                        # this configuration to stop training, not for the next
                        # seed of it to start.
                        if train_job.get("cancel") or train_job.get("stop_requested"):
                            break
                        train_job["seed"] = int(seed)
                        train_job["seed_index"] = index
                        train_job["model_id"] = ""
                    if total > 1:
                        append(f"[inspect] ── replicate {index + 1}/{total} · seed {seed} ──")
                        try:
                            set_seed(int(seed))
                        except Exception as exc:  # noqa: BLE001
                            final_state, err = "error", f"could not prepare seed {seed}: {exc}"
                            break
                    final_state, err = run_one()
                    with train_lock:
                        if train_job and train_job.get("job_id") == job_id:
                            train_job["replicates"].append({
                                "seed": int(seed),
                                "model_id": str(train_job.get("model_id") or ""),
                                "state": final_state,
                            })
                    # A failed replicate ends the sweep rather than burning
                    # hours on seeds that will fail the same way.
                    if final_state in ("error", "cancelled"):
                        break

                if total > 1:
                    # Leave the project on the seed the user configured, so the
                    # sweep does not silently become the new default.
                    try:
                        set_seed(base_seed)
                        cfg = load_config(project_id)
                    except Exception as exc:  # noqa: BLE001
                        append(f"[inspect] could not restore seed {base_seed}: {exc}")

                with train_lock:
                    if train_job and train_job.get("job_id") == job_id:
                        train_job["state"] = final_state
                        train_job["error"] = err
                        train_job["ended_at"] = time.time()
                        train_job["proc"] = None
                        try:
                            Path(str(train_job.get("stop_file") or "")).unlink(missing_ok=True)
                        except Exception:  # noqa: BLE001
                            pass

            threading.Thread(target=run_training, daemon=True).start()
            self._send_json({
                "ok": True,
                "job_id": job_id,
                "normalization_refit": fitted_stats,
            })

        def _handle_train_runs(self) -> None:
            from sed.train import registry
            try:
                manifests = registry.load_manifests(project_id)
                active = registry.get_active_model(project_id) or ""
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            runs = []
            for man in manifests:
                mid = man.get("model_id") or ""
                tme = man.get("test_metrics_exported") or {}
                tr = man.get("training") or {}
                exp = ""
                if man.get("exported"):
                    exp = (man.get("export") or {}).get("active_precision") or "yes"
                summary = _run_config_summary(layout, mid)
                mdir = layout.model_dir(mid)
                ov = _model_openvino_summary(mdir)
                # Engine chain the runtime resolves (OpenVINO active precision →
                # ONNX → checkpoint) — what production actually runs, and the
                # fallback path for VINO-incompatible or never-exported models.
                has_onnx = (mdir / "model.onnx").exists()
                has_ckpt = bool(list(mdir.glob("checkpoint_*.ckpt")))
                engines = (
                    [f"openvino:{p}" for p in ("int8", "fp16")
                     if p in ov.get("available_precisions", [])]
                    + (["onnx"] if has_onnx else [])
                    + (["checkpoint"] if has_ckpt else [])
                )
                if ov.get("has_openvino"):
                    default_engine = "openvino:" + str(ov.get("openvino_precision") or "fp16")
                elif has_onnx:
                    default_engine = "onnx"
                elif has_ckpt:
                    default_engine = "checkpoint"
                else:
                    default_engine = ""
                stats = man.get("model_stats") or {}
                # Manifest-only augmentation signal — no snapshot parsing on the
                # list endpoint; older manifests lack by_origin and show null.
                by_origin = (man.get("dataset") or {}).get("by_origin") or {}
                augmented = None
                if by_origin:
                    augmented = sum(
                        int(c.get("derived") or 0) for c in by_origin.values() if isinstance(c, dict)
                    )
                # Calibration provenance from the bundle's deployed decision
                # layer — cheap JSON read, same cost class as the config summary.
                calibration = {}
                try:
                    from sed.config import load_runtime_settings
                    calibration = load_runtime_settings(
                        layout.model_dir(mid) / "runtime_settings.json"
                    ).get("calibration") or {}
                except Exception:  # noqa: BLE001
                    calibration = {}
                runs.append({
                    "model_id": mid,
                    "description": man.get("description") or "",
                    "created_at": man.get("created_at") or "",
                    "backbone": summary.get("backbone", ""),
                    "pretrained": summary.get("pretrained"),
                    "train_depth": summary.get("train_depth", ""),
                    "loss": summary.get("loss", ""),
                    "feature": summary.get("feature", ""),
                    "recipe_signature": summary.get("recipe_signature", ""),
                    # Seed replication + trained class space (see _run_config_summary).
                    "experiment_key": summary.get("experiment_key", ""),
                    "seed": summary.get("seed"),
                    "classes": summary.get("classes", []),
                    "initialization": summary.get("initialization", ""),
                    "parent_project": summary.get("parent_project", ""),
                    "parent_run": summary.get("parent_run", ""),
                    # Which named dataset definition produced this run.
                    "dataset_preset": (man.get("dataset") or {}).get("preset_name", ""),
                    "dataset_preset_modified": bool(
                        (man.get("dataset") or {}).get("preset_modified", False)),
                    "total_params": stats.get("total_params"),
                    "trainable_params": stats.get("trainable_params"),
                    "n_stages": stats.get("n_stages"),
                    "trained_stages": stats.get("trained_stages"),
                    # Training run — outcome of the fit itself.
                    "epochs_ran": tr.get("epochs_ran"),
                    "epochs_max": tr.get("epochs_max"),
                    "selected_epoch": tr.get("selected_epoch"),
                    "stopped_early": tr.get("stopped_early"),
                    "selection_metric": tr.get("selection_metric", ""),
                    "train_loss": tr.get("selected_train_loss"),
                    # Validation — each run's own split, during training.
                    "selected_val_loss": tr.get("selected_val_loss"),
                    "val_f1": tr.get("best_val_f1_macro_targets"),
                    "val_ap": tr.get("best_val_ap_macro_targets"),
                    "val_accuracy": tr.get("best_val_accuracy"),
                    # The val twins of the frozen-test false-fire columns, at the
                    # same fixed 0.9 gate that drives checkpoint selection.
                    "val_recall_at_0_9": tr.get("selected_val_recall_macro_targets_at_0_9"),
                    "val_fp_per_1k_at_0_9": tr.get(
                        "selected_val_positive_triggers_per_1000_negatives_at_0_9"),
                    # Frozen test, keyed by decision rule; deployed IR, keyed by
                    # "<precision>:<rule id>". The table discovers which column
                    # bands exist from these keys.
                    "frozen_test": _frozen_test_summary(man),
                    "deployed_test": _deployed_test_summary(man),
                    "thresholds_source": (man.get("test_metrics") or {}).get("thresholds_source") or "",
                    "calibration_mode": str(calibration.get("mode") or ""),
                    "calibration_target_precision": calibration.get("target_precision"),
                    # False-alarm budget calibration (sed.runtime.calibrate) —
                    # the peak per-class FP/h the sweep left, if it was run.
                    "calibration_max_per_1000": calibration.get("max_per_1000"),
                    "calibration_system_per_1000": calibration.get("system_per_1000"),
                    "calibration_target_per_1000": calibration.get("target_per_1000"),
                    "calibration_negative_source": calibration.get("negative_source", ""),
                    "deployed_model_file": tme.get("model_file") or "",
                    "deployed_test_set_hash": tme.get("test_set_hash") or "",
                    "export": exp,
                    "has_openvino": bool(ov.get("has_openvino")),
                    "active": mid == active,
                    "available_precisions": ov.get("available_precisions", []),
                    "engines": engines,
                    "default_engine": default_engine,
                    "augmented": augmented,
                    # Which (engine, test set) pairs this run has stored scores
                    # for. Every dynamic metric is derived from these, so the
                    # table uses them to decide what it can compute and what
                    # still needs a scoring pass.
                    "scores": _score_store_summary(mdir),
                })
            frozen = _current_frozen_test(layout)
            from sed.train.test_rules import rule_catalog
            self._send_json({
                "ok": True, "runs": runs, "active": active,
                "current_test_hash": str(frozen.get("hash") or ""),
                "current_test_total": int(frozen.get("total") or 0),
                # Every frozen test set any loaded run holds scores for, so the
                # table can be pointed at one benchmark at a time instead of
                # mixing test sets in one column.
                "scored_test_sets": _scored_test_sets(layout, runs),
                # Health of the CURRENT test set (negatives present? enough
                # clips per class?) — a false-alarm metric on a set with no
                # negatives is not a small number, it is no number at all.
                "current_test_health": _current_test_health(layout),
                # hash -> saved name, so a score can name the benchmark it came
                # from rather than showing a bare digest.
                "test_set_names": _test_set_names(layout),
                # The one description of every decision rule — the browser renders
                # the batch panel, the after-training panel, the column headers and
                # the metric guide from this, so no rule wording is written twice.
                "rule_catalog": rule_catalog(),
            })

        def _handle_train_run(self, model_id: str) -> None:
            if not model_id:
                self._send_json({"ok": False, "error": "missing model_id"}, status=400)
                return
            from sed.train import registry
            try:
                man = registry.load_manifest(project_id, model_id)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=404)
                return
            from sed.train.test_rules import read_deployed_test_evals, read_frozen_test_evals

            # Full per-rule blocks (with per_class) — one run, so the cost is fine.
            frozen_test = {
                rule_id: {**block, "confusion_key": _confusion_asset_key(block.get("report_name"))}
                for rule_id, block in read_frozen_test_evals(man).items()
            }
            # Exported IR, keyed "<precision>:<rule id>" — same shape as
            # frozen_test so the detail panel renders both with one code path.
            deployed_test = read_deployed_test_evals(man)
            training_config = None
            try:
                tc_path = layout.model_dir(model_id) / "training_config.json"
                if tc_path.exists():
                    with open(tc_path, "r", encoding="utf-8") as f:
                        training_config = json.load(f)
            except Exception as exc:
                log.debug("failed to load training_config.json: %s", exc)

            frozen = _current_frozen_test(layout)
            mdir = layout.model_dir(model_id)
            ov = _model_openvino_summary(mdir)
            has_onnx = (mdir / "model.onnx").exists()
            has_ckpt = bool(list(mdir.glob("checkpoint_*.ckpt")))
            engines = (
                [f"openvino:{p}" for p in ("int8", "fp16")
                 if p in ov.get("available_precisions", [])]
                + (["onnx"] if has_onnx else [])
                + (["checkpoint"] if has_ckpt else [])
            )
            if ov.get("has_openvino"):
                default_engine = "openvino:" + str(ov.get("openvino_precision") or "fp16")
            elif has_onnx:
                default_engine = "onnx"
            elif has_ckpt:
                default_engine = "checkpoint"
            else:
                default_engine = ""
            calibration = {}
            try:
                from sed.config import load_runtime_settings
                calibration = load_runtime_settings(
                    mdir / "runtime_settings.json"
                ).get("calibration") or {}
            except Exception:  # noqa: BLE001
                calibration = {}
            # Whether OpenVINO export is even possible for this architecture, so
            # the export panel offers ONNX/checkpoint only for VINO-incompatible
            # backbones instead of dead buttons.
            export_supported = True
            try:
                from sed.train.backbones import resolve_adapter
                bb = (_run_config_summary(layout, model_id) or {}).get("backbone") or ""
                if bb:
                    export_supported = bool(resolve_adapter(bb).spec.export_supported)
            except Exception:  # noqa: BLE001
                export_supported = True
            self._send_json({
                "ok": True, "model_id": model_id,
                "frozen_test": frozen_test, "deployed_test": deployed_test,
                "current_test_hash": str(frozen.get("hash") or ""),
                "exported": bool(man.get("exported")),
                "export": man.get("export") or {},
                "available_precisions": ov.get("available_precisions", []),
                "engines": engines,
                "default_engine": default_engine,
                "export_supported": export_supported,
                "calibration": calibration,
                "training_config": training_config,
                "run_dataset": man.get("dataset"),
                "run_training_stats": man.get("training"),
                "run_recipe": man.get("model_recipe"),
                "run_initialization": man.get("initialization"),
                "run_model_stats": man.get("model_stats"),
                "git_commit": man.get("git_commit"),
                "augmentation_applied": _augmentation_truth(mdir, man),
                # Where this model came from: the domain-pretrain -> venue
                # fine-tune chain, possibly crossing projects.
                "lineage": _run_lineage(layout, model_id),
                "test_set_names": _test_set_names(layout),
            })

        def _handle_train_compare(self, model_ids: list[str]) -> None:
            """Side-by-side run comparison: full config + metrics + epoch curves."""
            ids = [mid for mid in model_ids if mid][:8]
            if len(ids) < 2:
                self._send_json({"ok": False, "error": "need at least 2 model_ids"}, status=400)
                return
            from sed.train import registry
            runs = []
            for mid in ids:
                try:
                    man = registry.load_manifest(project_id, mid)
                except Exception as exc:  # noqa: BLE001
                    runs.append({"model_id": mid, "error": str(exc)})
                    continue
                model_dir = layout.model_dir(mid)
                training_config = None
                try:
                    with open(model_dir / "training_config.json", "r", encoding="utf-8") as f:
                        training_config = json.load(f)
                except Exception as exc:  # noqa: BLE001
                    log.debug("compare: no training_config.json for %s: %s", mid, exc)
                tm = man.get("test_metrics") or {}
                tme = man.get("test_metrics_exported") or {}
                tr = man.get("training") or {}
                # Calibration provenance from the bundle's deployed decision
                # layer — how this run's thresholds were fitted, if at all.
                calibration = {}
                try:
                    from sed.config import load_runtime_settings
                    calibration = load_runtime_settings(
                        model_dir / "runtime_settings.json"
                    ).get("calibration") or {}
                except Exception:  # noqa: BLE001
                    calibration = {}
                runs.append({
                    "model_id": mid,
                    "description": man.get("description") or "",
                    "created_at": man.get("created_at") or "",
                    "git_commit": man.get("git_commit") or "",
                    "training_config": training_config,
                    "run_dataset": man.get("dataset"),
                    "run_training_stats": tr,
                    "run_recipe": man.get("model_recipe"),
                    "run_initialization": man.get("initialization"),
                    "run_model_stats": man.get("model_stats"),
                    "augmentation_applied": _augmentation_truth(model_dir, man),
                    "frozen_test": _frozen_test_summary(man),
                    "deployed_test": _deployed_test_summary(man),
                    "metrics": {
                        "val_f1": tr.get("best_val_f1_macro_targets"),
                        "val_ap": tr.get("best_val_ap_macro_targets"),
                        "val_accuracy": tr.get("best_val_accuracy"),
                    },
                    "evaluation": {
                        "test_set_hash": tm.get("test_set_hash") or "",
                        "test_n_samples": tm.get("n_samples"),
                        "thresholds_source": tm.get("thresholds_source") or "",
                        "thresholds": tm.get("thresholds") or {},
                        "calibration_mode": str(calibration.get("mode") or ""),
                        "calibration_target_precision": calibration.get("target_precision"),
                        "deployed_model_file": tme.get("model_file") or "",
                        "deployed_test_set_hash": tme.get("test_set_hash") or "",
                    },
                    "epoch_metrics": _run_epoch_metrics(model_dir),
                })
            frozen = _current_frozen_test(layout)
            self._send_json({
                "ok": True, "runs": runs,
                "current_test_hash": str(frozen.get("hash") or ""),
            })

        def _handle_train_delete(self) -> None:
            body = self._read_json_body()
            raw_ids = body.get("model_ids")
            if raw_ids is None and body.get("model_id") is not None:
                raw_ids = [body.get("model_id")]
            if not isinstance(raw_ids, list):
                self._send_json({"ok": False, "error": "model_ids must be an array"}, status=400)
                return
            model_ids: list[str] = []
            seen: set[str] = set()
            for raw in raw_ids:
                mid = str(raw or "").strip()
                if not mid or mid in seen:
                    continue
                model_ids.append(mid)
                seen.add(mid)
            if not model_ids:
                self._send_json({"ok": False, "error": "select at least one model"}, status=400)
                return

            from sed.train import registry
            try:
                manifests = registry.load_manifests(project_id)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            known = {str(man.get("model_id") or "") for man in manifests if man.get("model_id")}
            invalid: list[str] = []
            for mid in model_ids:
                model_dir = _safe_resolve_model_dir(layout, mid)
                if (
                    mid not in known
                    or model_dir is None
                    or not model_dir.is_dir()
                    or not (model_dir / "manifest.json").exists()
                ):
                    invalid.append(mid)
            if invalid:
                self._send_json({
                    "ok": False,
                    "error": "unknown model(s): " + ", ".join(invalid),
                    "model_ids": invalid,
                }, status=404)
                return

            with train_lock:
                active_train = (
                    train_job
                    if train_job
                    and str(train_job.get("state")) in {
                        "starting", "running", "stopping", "canceling",
                    }
                    else None
                )
                active_train_model = str((active_train or {}).get("model_id") or "")
            if active_train_model and active_train_model in seen:
                self._send_json({
                    "ok": False,
                    "error": f"model is currently training: {active_train_model}",
                    "model_id": active_train_model,
                }, status=409)
                return

            with inference_lock:
                active_inference = (
                    inference_job
                    if inference_job
                    and str(inference_job.get("state")) in {
                        "starting", "running", "stopping",
                    }
                    else None
                )
                active_inference_model = str((active_inference or {}).get("model_id") or "")
            if active_inference_model and active_inference_model in seen:
                self._send_json({
                    "ok": False,
                    "error": f"model is currently running inference: {active_inference_model}",
                    "model_id": active_inference_model,
                }, status=409)
                return

            deleted: list[str] = []
            errors: list[dict] = []
            for mid in model_ids:
                try:
                    registry.delete_model(project_id, mid)
                    deleted.append(mid)
                    log.info("deleted model run %s", mid)
                except Exception as exc:  # noqa: BLE001
                    errors.append({"model_id": mid, "error": str(exc)})
                    log.warning("delete model run failed %s: %s", mid, exc)
            try:
                active = registry.get_active_model(project_id) or ""
            except Exception:  # noqa: BLE001
                active = ""
            self._send_json({
                "ok": len(errors) == 0,
                "deleted": deleted,
                "deleted_count": len(deleted),
                "errors": errors,
                "failed_count": len(errors),
                "active": active,
            }, status=200 if not errors else 500)

        def _handle_train_asset(self, model_id: str, file: str) -> None:
            allowed = {
                "training_curves": "reports/training_curves.png",
                "confusion_matrix": "reports/confusion_matrix.png",
            }
            rel = allowed.get(file)
            # Per-rule confusion matrices: "confusion_matrix:<rule slug>". The
            # slug charset is restricted to what TestRule.slug can produce, so
            # no traversal is reachable through this branch.
            if rel is None and file.startswith("confusion_matrix:"):
                slug = file.split(":", 1)[1]
                if slug and all(ch.isalnum() or ch in ("_", "-") for ch in slug):
                    rel = f"reports/{slug}_confusion_matrix.png"
            if not model_id or rel is None:
                self._send(404, b"not found")
                return
            md = layout.model_dir(model_id).resolve()
            target = (md / rel).resolve()
            try:
                target.relative_to(md)
            except ValueError:
                self._send(403, b"forbidden")
                return
            if not target.exists():
                self._send(404, b"not found")
                return
            self._send_file(target)

        def _handle_train_frozen_metrics(self) -> None:
            """Re-derive frozen-test metrics for many runs at one operating point.

            No model runs here. Every number comes from the probability matrices
            already on disk, re-cut through the same `sed.train.metrics` code
            that wrote the manifest — which is what lets the Runs & compare
            controls move a confidence, a margin or a false-alarm budget and
            have the table, the per-class rows and the confusion matrix all
            follow without a scoring job.
            """
            from sed.train import rescore, score_store

            try:
                body = self._read_json_body()
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": f"bad request: {exc}"}, status=400)
                return

            model_ids = [str(m).strip() for m in (body.get("model_ids") or []) if str(m).strip()]
            if not model_ids:
                self._send_json({"ok": False, "error": "no model_ids"}, status=400)
                return
            test_hash = str(body.get("test_hash") or "").strip()
            if not test_hash:
                test_hash = str((_current_frozen_test(layout) or {}).get("hash") or "")
            decision = rescore.Decision.from_payload(body.get("decision"))
            want_curve = bool(body.get("include_curve"))

            results: dict[str, dict] = {}
            missing: list[str] = []
            for model_id in model_ids:
                run_dir = layout.model_dir(model_id)
                key = score_store.score_key("checkpoint", test_hash) if test_hash else ""
                block = rescore.rescore(run_dir, key, decision) if key else None
                if block is None:
                    missing.append(model_id)
                    continue
                if not want_curve:
                    # The curve is 256 points per run; only the compare view
                    # plots it, and sending it for 60 runs dwarfs everything else.
                    block.pop("operating_curve", None)
                results[model_id] = block

            self._send_json({
                "ok": True,
                "test_hash": test_hash,
                "decision": {
                    "confidence": decision.confidence,
                    "negative_max": decision.negative_max,
                    "margin": decision.margin,
                    "mode": decision.mode,
                    "target_precision": decision.target_precision,
                    "budget_per_1000": decision.budget_per_1000,
                    "recall_floor": decision.recall_floor,
                },
                "metrics": results,
                # Runs with no stored scores for this test set. They are not an
                # error — they simply have not been scored on it yet, and the UI
                # offers that as an action.
                "unscored": missing,
            })

        def _handle_train_evaluate(self) -> None:
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            model_id = str(body.get("model_id") or "").strip()
            if not model_id:
                self._send_json({"ok": False, "error": "missing model_id"}, status=400)
                return
            from sed.train.eval import TestRule, evaluate_run_under_rule

            # Default rule = the shared 0.90 confidence gate, i.e. what the UI
            # offers first and what makes runs comparable.
            raw = body.get("rule")
            raw = raw if isinstance(raw, dict) else {"kind": "fixed"}
            try:
                rule = TestRule(
                    str(raw.get("kind") or "fixed").strip(),
                    None if raw.get("param") in (None, "") else float(raw["param"]),
                )
            except (TypeError, ValueError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            try:
                metrics = evaluate_run_under_rule(cfg, project_id, model_id, rule)
            except Exception as exc:  # noqa: BLE001
                log.exception("train-tab evaluate failed")
                self._send_json({"ok": False, "error": str(exc), "model_id": model_id})
                return
            self._send_json({
                "ok": True, "model_id": model_id, "rule_id": rule.id,
                "metrics": {
                    "rule": rule.as_dict(),
                    "per_class": metrics.get("per_class"),
                    "f1_macro_targets": metrics.get("f1_macro_targets"),
                    "ap_macro_targets": metrics.get("ap_macro_targets"),
                    "accuracy": metrics.get("accuracy"),
                },
            })

        def _handle_train_evaluate_exported(self) -> None:
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            model_id = str(body.get("model_id") or "").strip()
            if not model_id:
                self._send_json({"ok": False, "error": "missing model_id"}, status=400)
                return
            precision = str(body.get("precision") or "auto").strip().lower()
            if precision not in ("auto", "fp16", "int8"):
                self._send_json({"ok": False, "error": f"unknown precision: {precision}"}, status=400)
                return
            model_dir = layout.model_dir(model_id)
            ov = _model_openvino_summary(model_dir)
            if precision != "auto" and precision not in ov.get("available_precisions", []):
                self._send_json({"ok": False, "error": f"no {precision} export for this model"}, status=400)
                return
            from sed.train.eval import EXPORTED_RULE_KINDS, TestRule, evaluate_exported_run
            raw_rule = body.get("rule") or {}
            try:
                kind = str(raw_rule.get("kind") or "deployed")
                if kind not in EXPORTED_RULE_KINDS:
                    raise ValueError(
                        f"rule {kind} cannot score an exported IR "
                        f"(use {', '.join(EXPORTED_RULE_KINDS)})"
                    )
                param = raw_rule.get("param")
                rule = TestRule(kind, float(param) if param is not None else None)
            except (TypeError, ValueError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            try:
                metrics = evaluate_exported_run(
                    cfg, project_id, model_id, precision=precision, rule=rule,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("train-tab evaluate-exported failed")
                self._send_json({"ok": False, "error": str(exc), "model_id": model_id})
                return
            self._send_json({
                "ok": True, "model_id": model_id,
                # The detail panel opens this tab; same key the runs table bands use.
                "key": f"{metrics.get('precision') or precision}:{rule.id}",
            })

        def _handle_train_calibrate(self) -> None:
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            model_id = str(body.get("model_id") or "").strip()
            from sed.config import load_config
            from sed.runtime.calibrate import calibrate
            from sed.train import registry
            # calibrate targets the active model — make the requested model active first.
            if model_id:
                try:
                    registry.set_active_model(project_id, model_id)
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": str(exc)})
                    return
            try:
                result = calibrate(
                    load_config(project_id), project_id,
                    target_per_1000=float(body.get("target_per_1000") or 1.0),
                    negative_source=str(body.get("negative_source") or "venue"),
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("train-tab calibrate failed")
                self._send_json({"ok": False, "error": str(exc), "model_id": model_id})
                return
            self._send_json({
                "ok": True, "model_id": model_id,
                "thresholds": result["thresholds"],
                "achieved_per_1000": result.get("achieved_per_1000") or {},
                "max_per_1000": max(
                    (v for v in (result.get("achieved_per_1000") or {}).values() if v is not None),
                    default=None,
                ),
                # What the venue actually gets: every class armed at once.
                "system_per_1000": result.get("system_per_1000"),
                "target_per_1000": result.get("target_per_1000"),
                "clips_scored": result.get("clips_scored"),
                "negative_files": result.get("negative_files"),
                "negative_source": result.get("negative_source"),
                "simulation_mode": result.get("simulation_mode"),
            })

        # ---- Batch evaluation -------------------------------------------
        # One background job that applies the SAME conditions to every selected
        # run: score the CURRENT frozen test under one or more decision rules
        # (checkpoint), score the exported IR, export missing IR, and/or refit
        # each run's deployed thresholds. Only the last of these mutates a
        # bundle — the rules never touch runtime_settings.json.

        def _batch_eval_payload(self, job: dict | None) -> dict:
            if not job:
                return {"ok": False, "error": "no batch evaluation job"}
            payload = {k: v for k, v in job.items() if k != "thread"}
            started = float(payload.get("started_at") or time.time())
            ended = payload.get("ended_at")
            payload["elapsed_s"] = max(0.0, float(ended or time.time()) - started)
            return {"ok": True, "job": payload}

        def _handle_batch_eval_status(self, job_id: str = "") -> None:
            with batch_eval_lock:
                reap_batch_eval_job()
                job = batch_eval_job
                if job_id and job and str(job.get("job_id")) != job_id:
                    job = None
                payload = self._batch_eval_payload(job)
            self._send_json(payload, status=200 if payload.get("ok") else 404)

        def _handle_batch_eval_cancel(self) -> None:
            with batch_eval_lock:
                reap_batch_eval_job()
                job = batch_eval_job
                if not job or str(job.get("state")) not in {"starting", "running"}:
                    self._send_json({"ok": False, "error": "no batch evaluation job running"}, status=409)
                    return
                job["cancel_requested"] = True
                payload = self._batch_eval_payload(job)
            self._send_json(payload)

        @staticmethod
        def _batch_metrics_summary(metrics: dict) -> dict:
            return {
                "f1_macro_targets": metrics.get("f1_macro_targets"),
                "ap_macro_targets": metrics.get("ap_macro_targets"),
                "precision_macro_targets": metrics.get("precision_macro_targets"),
                "recall_macro_targets": metrics.get("recall_macro_targets"),
                "accuracy": metrics.get("accuracy"),
                "n_samples": metrics.get("n_samples"),
                "thresholds_source": metrics.get("thresholds_source"),
                "test_set_hash": metrics.get("test_set_hash"),
            }

        def _handle_advice_get(self) -> None:
            """Feature/model recommendations for the current config.

            Advisory only: the payload never gates anything, it explains what a
            choice costs so an unusual one is deliberate.
            """
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            from sed.train.advice import config_advice

            # Training advice is about the schedule *for this dataset*, so it
            # needs the composed train pool's shape. _dataset_shape is a single
            # train.csv pass, safe on a page-load endpoint.
            try:
                shape = _dataset_shape(layout)
            except Exception:  # noqa: BLE001 — advisory input only
                shape = {}
            # Whether the clips still fit the active feature scale needs audio,
            # which is too slow here — report the last explicit measurement, and
            # nothing when the dataset or the scale has moved since.
            from sed.data.feature_stats import load_cached_scaling_drift

            cached_drift = load_cached_scaling_drift(cfg, layout)
            if cached_drift:
                shape["scaling_drift"] = cached_drift
            self._send_json({"ok": True, **config_advice(cfg, shape)})

        def _handle_feature_scaling_drift_get(self) -> None:
            """Measure this project's clips against the active scaling constants.

            The counterpart to "Estimate from dataset": a fine-tune must NOT
            refit its parent's constants, so the actionable question there is
            whether the venue material still lands inside them.
            """
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            from sed.data.feature_stats import measure_scaling_drift

            try:
                result = measure_scaling_drift(cfg, layout)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, "drift": result})

        def _handle_frozen_test_presets_get(self) -> None:
            """Named test sets plus the live one's fitness, for the workbench.

            Both halves answer one question — "can this benchmark support the
            comparison I am about to make" — so they travel together rather than
            forcing the UI to correlate two endpoints.
            """
            from sed.data.frozen_test_presets import frozen_test_preset_state
            from sed.train.testset_health import health_for_test_csv

            self._send_json({
                "ok": True,
                "state": frozen_test_preset_state(layout),
                "health": health_for_test_csv(layout.dataset_dir / "test.csv"),
            })

        def _handle_frozen_test_preset(self, action: str) -> None:
            """Named frozen-test sets: save / apply / derive-without-leaks / delete.

            `derive` is the leakage remedy. It is deliberately additive — it
            writes a NEW named test set and does not touch the live one, so
            scores already computed against the old benchmark stay reproducible.
            Making it live is a separate, explicit `apply`.
            """
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send_json({"ok": False, "error": "invalid JSON"}, status=400)
                return
            from sed.data.frozen_test_presets import (
                apply_frozen_test_preset,
                delete_frozen_test_preset,
                derive_preset_excluding,
                frozen_test_preset_state,
                save_frozen_test_preset,
            )

            name = str(body.get("name") or "").strip()
            try:
                if action == "save":
                    record = save_frozen_test_preset(layout, name, notes=str(body.get("notes") or ""))
                elif action == "apply":
                    record = apply_frozen_test_preset(layout, name)
                    record_activity_event(
                        "Frozen test applied", detail=record.get("name", ""), section="train")
                elif action == "derive":
                    paths = [str(p) for p in (body.get("excluded_paths") or [])]
                    if not paths:
                        self._send_json(
                            {"ok": False, "error": "excluded_paths is required"}, status=400)
                        return
                    record = derive_preset_excluding(
                        layout, name, paths,
                        source=str(body.get("source") or ""),
                        notes=str(body.get("notes") or ""),
                    )
                    record_activity_event(
                        "Frozen test derived",
                        detail=f"{record.get('name', '')} — {len(paths)} clip(s) removed",
                        section="train")
                elif action == "delete":
                    if not delete_frozen_test_preset(layout, name):
                        self._send_json({"ok": False, "error": "preset not found"}, status=404)
                        return
                    record = {}
                else:
                    self._send_json({"ok": False, "error": f"unknown action: {action}"}, status=404)
                    return
            except KeyError:
                self._send_json({"ok": False, "error": "preset not found"}, status=404)
                return
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._send_json({
                "ok": True, "preset": record, "state": frozen_test_preset_state(layout),
            })

        def _handle_batch_eval_check(self) -> None:
            """Pre-flight the selected runs against the CURRENT frozen test.

            Answers, before a job spends GPU time: can each run be scored at all
            (class space, checkpoint), and would the score be fair (leakage).
            Read-only — it proposes remedies, it never edits the test set.
            """
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send_json({"ok": False, "error": "invalid JSON"}, status=400)
                return
            model_ids = [str(m or "").strip() for m in (body.get("model_ids") or [])]
            model_ids = [m for m in dict.fromkeys(model_ids) if m]
            if not model_ids:
                self._send_json({"ok": False, "error": "select at least one model"}, status=400)
                return

            from sed.train.test_compat import check_runs, parse_class_mapping
            from sed.train.testset_health import health_for_test_csv

            mapping = parse_class_mapping(body.get("class_mapping"))
            test_csv = layout.dataset_dir / "test.csv"
            runs = []
            for mid in model_ids:
                mdir = layout.model_dir(mid)
                runs.append({
                    "id": mid,
                    "model_dir": mdir,
                    "classes": _run_config_summary(layout, mid).get("classes") or [],
                })
            try:
                report = check_runs(test_csv, runs, mapping=mapping)
            except Exception as exc:  # noqa: BLE001 — a pre-flight must not 500
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            # The test set's own fitness (negatives present? enough support?) is
            # part of "can this benchmark answer the question", so it travels
            # with the per-run verdicts rather than living in a separate tab.
            report["test_health"] = health_for_test_csv(test_csv)
            # Named test sets, so the UI can say whether the live benchmark has
            # a name to restore it by before offering to derive a new one.
            from sed.data.frozen_test_presets import frozen_test_preset_state
            report["preset_state"] = frozen_test_preset_state(layout)
            self._send_json({"ok": True, **report})

        def _handle_batch_eval_start(self) -> None:
            nonlocal batch_eval_job
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send_json({"ok": False, "error": "invalid JSON"}, status=400)
                return

            raw_ids = body.get("model_ids")
            if not isinstance(raw_ids, list):
                self._send_json({"ok": False, "error": "model_ids must be an array"}, status=400)
                return
            model_ids: list[str] = []
            seen: set[str] = set()
            for raw in raw_ids:
                mid = str(raw or "").strip()
                if mid and mid not in seen:
                    model_ids.append(mid)
                    seen.add(mid)
            if not model_ids:
                self._send_json({"ok": False, "error": "select at least one model"}, status=400)
                return

            calibration = str(body.get("calibration") or "none").strip()
            if calibration not in ("none", "val_f1", "val_precision"):
                self._send_json({"ok": False, "error": f"unknown calibration mode: {calibration}"}, status=400)
                return
            evaluate_exported = bool(body.get("evaluate_exported", False))
            # Folds the test set's labels onto the space each run was trained on
            # (loud|medium|quiet -> cheers), so a taxonomy change does not
            # orphan a project's run history. Merges only; never invents a class.
            batch_class_mapping = {
                str(k): str(v) for k, v in (body.get("class_mapping") or {}).items() if k and v
            }
            # Each IR precision is scored and stored separately, so FP16 and
            # INT8 can be compared instead of overwriting each other. The old
            # single-valued `exported_precision` is still accepted.
            raw_precisions = body.get("exported_precisions")
            if isinstance(raw_precisions, list):
                exported_precisions = [str(p).strip().lower() for p in raw_precisions]
            else:
                exported_precisions = [str(body.get("exported_precision") or "auto").strip().lower()]
            exported_precisions = list(dict.fromkeys(p for p in exported_precisions if p))
            bad_precisions = [p for p in exported_precisions if p not in ("auto", "fp16", "int8")]
            if bad_precisions:
                self._send_json({"ok": False, "error": f"unknown precision: {bad_precisions[0]}"}, status=400)
                return
            if evaluate_exported and not exported_precisions:
                self._send_json({"ok": False, "error": "select at least one IR precision"}, status=400)
                return
            raw_ir_rule = body.get("exported_rule") or {}
            try:
                from sed.train.test_rules import EXPORTED_RULE_KINDS, TestRule
                ir_kind = str((raw_ir_rule or {}).get("kind") or "deployed")
                if ir_kind not in EXPORTED_RULE_KINDS:
                    raise ValueError(
                        f"rule {ir_kind} cannot score an exported IR "
                        f"(use {', '.join(EXPORTED_RULE_KINDS)})"
                    )
                ir_param = (raw_ir_rule or {}).get("param")
                exported_rule = TestRule(ir_kind, float(ir_param) if ir_param is not None else None)
            except (TypeError, ValueError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            export_missing = bool(body.get("export_missing", False))
            # Optional export-panel controls: which artifacts to (re)build and
            # which precision to make active. `export_targets` present forces a
            # rebuild (vs the default "export missing" skip-if-present).
            raw_targets = body.get("export_targets")
            export_targets: set[str] | None = None
            if raw_targets is not None:
                if not isinstance(raw_targets, list):
                    self._send_json({"ok": False, "error": "export_targets must be an array"}, status=400)
                    return
                export_targets = {str(t).lower() for t in raw_targets}
                bad = export_targets - {"onnx", "fp16", "int8"}
                if bad:
                    self._send_json({"ok": False, "error": f"unknown export target(s): {sorted(bad)}"}, status=400)
                    return
                if not export_targets:
                    self._send_json({"ok": False, "error": "select at least one export target"}, status=400)
                    return
                export_missing = True   # explicit target set always exports
            export_active_precision = str(body.get("export_active_precision") or "auto").strip().lower()
            if export_active_precision not in ("auto", "fp16", "int8"):
                self._send_json({"ok": False, "error": f"unknown precision: {export_active_precision}"}, status=400)
                return

            def _bounded(name: str, default: float) -> float | None:
                try:
                    v = float(body.get(name, default))
                except (TypeError, ValueError):
                    return None
                return v if 0.0 < v < 1.0 else None

            target_precision = _bounded("target_precision", 0.90)
            if calibration == "val_precision" and target_precision is None:
                self._send_json({"ok": False, "error": "target_precision must be in (0, 1)"}, status=400)
                return

            # The worker's heavy imports are resolved here, in the request
            # thread: an import failure has to come back as a visible HTTP
            # error. Deferred into the background thread it would kill the
            # worker before it could report and pin the job at "starting".
            try:
                from sed.config import load_config
                from sed.train import registry
                from sed.train.eval import (
                    TestRule,
                    calibrate_run_thresholds,
                    evaluate_exported_run,
                    evaluate_run_under_rule,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("batch evaluation could not import the model code")
                self._send_json({
                    "ok": False,
                    "error": f"could not load the evaluation code: "
                             f"{type(exc).__name__}: {exc}",
                }, status=500)
                return

            # Frozen-test decision rules to score, e.g.
            # [{"kind": "fixed", "param": 0.9}, {"kind": "argmax"}].
            raw_rules = body.get("rules")
            if raw_rules is not None and not isinstance(raw_rules, list):
                self._send_json({"ok": False, "error": "rules must be an array"}, status=400)
                return
            rules: list = []
            for raw in raw_rules or []:
                if not isinstance(raw, dict):
                    self._send_json({"ok": False, "error": "each rule must be an object"}, status=400)
                    return
                try:
                    rule = TestRule(
                        str(raw.get("kind") or "").strip(),
                        None if raw.get("param") in (None, "") else float(raw["param"]),
                    )
                except (TypeError, ValueError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
                if not any(r.id == rule.id for r in rules):
                    rules.append(rule)

            try:
                manifests = registry.load_manifests(project_id)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            known = {str(man.get("model_id") or "") for man in manifests if man.get("model_id")}
            unknown = [mid for mid in model_ids if mid not in known]
            if unknown:
                self._send_json({
                    "ok": False,
                    "error": "unknown model(s): " + ", ".join(unknown),
                    "model_ids": unknown,
                }, status=404)
                return

            if calibration == "none" and not (rules or evaluate_exported or export_missing):
                self._send_json({"ok": False, "error": "select at least one action"}, status=400)
                return

            scores_frozen_test = bool(rules or evaluate_exported)
            if scores_frozen_test:
                test_csv = layout.dataset_dir / "test.csv"
                try:
                    with open(test_csv, encoding="utf-8") as f:
                        test_rows = max(0, sum(1 for _ in f) - 1)
                except OSError:
                    test_rows = 0
                if test_rows == 0:
                    self._send_json({
                        "ok": False,
                        "error": "current frozen test set is empty — build it in the "
                                 "Library tab (or record test material) and re-run "
                                 "dataset compose first",
                    }, status=400)
                    return
            frozen_warning = ""
            if scores_frozen_test:
                frozen = _current_frozen_test(layout)
                if str(frozen.get("source_quality") or "") == "pinned_holdouts":
                    frozen_warning = (
                        "current frozen test uses pinned holdouts; randomized split-derived "
                        "test is recommended for comparable batch evaluation"
                    )

            with train_lock:
                training_active = bool(
                    train_job
                    and str(train_job.get("state")) in {"starting", "running", "stopping", "canceling"}
                )
            if training_active:
                self._send_json({
                    "ok": False,
                    "error": "a training run is in progress — batch evaluation would "
                             "fight it for CPU and read half-written bundles; wait for "
                             "it to finish",
                }, status=409)
                return

            options = {
                "calibration": calibration,
                "target_precision": target_precision,
                "rules": [r.as_dict() for r in rules],
                "evaluate_exported": evaluate_exported,
                "exported_precisions": exported_precisions,
                "exported_rule": exported_rule.as_dict(),
                "export_missing": export_missing,
                "export_targets": sorted(export_targets) if export_targets else None,
                "export_active_precision": export_active_precision,
                "frozen_test_warning": frozen_warning,
            }
            with batch_eval_lock:
                reap_batch_eval_job()
                if batch_eval_job and str(batch_eval_job.get("state")) in {"starting", "running"}:
                    self._send_json({"ok": False, "error": "batch evaluation already running"}, status=409)
                    return
                job_id = uuid.uuid4().hex[:12]
                batch_eval_job = {
                    "job_id": job_id,
                    "state": "starting",
                    "total": len(model_ids),
                    "completed": 0,
                    "failed_count": 0,
                    "current_model": "",
                    "current_step": "",
                    "options": options,
                    "results": [],
                    "cancel_requested": False,
                    "started_at": time.time(),
                    "ended_at": None,
                }

            def set_progress(**kw) -> None:
                with batch_eval_lock:
                    if batch_eval_job and batch_eval_job.get("job_id") == job_id:
                        batch_eval_job.update(kw)

            def canceled() -> bool:
                with batch_eval_lock:
                    return bool(
                        batch_eval_job
                        and batch_eval_job.get("job_id") == job_id
                        and batch_eval_job.get("cancel_requested")
                    )

            def run_job() -> None:
                # Everything the worker touches is already imported (above) —
                # and everything it does is inside this guard, so no failure
                # can leave the job pinned at "starting".
                try:
                    # Fresh config like the CLI — picks up current [targets] /
                    # [inference] decision keys rather than the boot-time view.
                    job_cfg = load_config(project_id)
                    set_progress(state="running")
                    for mid in model_ids:
                        if canceled():
                            break
                        entry: dict = {"model_id": mid, "steps": {}, "errors": []}
                        if calibration != "none":
                            set_progress(current_model=mid, current_step="calibrate")
                            try:
                                fitted = calibrate_run_thresholds(
                                    job_cfg, project_id, mid,
                                    mode=calibration,
                                    target_precision=float(target_precision or 0.90),
                                )
                                if fitted is None:
                                    entry["steps"]["calibrate"] = {
                                        "skipped": "not applicable (no _negative class or no val split)",
                                    }
                                else:
                                    entry["steps"]["calibrate"] = {
                                        "thresholds": {k: round(float(v), 4) for k, v in fitted.items()},
                                    }
                            except Exception as exc:  # noqa: BLE001
                                entry["errors"].append(f"calibrate: {exc}")
                        if export_missing:
                            set_progress(current_model=mid, current_step="export")
                            try:
                                entry["steps"]["export"] = _export_run_ir(
                                    job_cfg, project_id, mid,
                                    targets=export_targets,
                                    active_precision=(
                                        None if export_active_precision == "auto"
                                        else export_active_precision
                                    ),
                                )
                            except Exception as exc:  # noqa: BLE001
                                entry["errors"].append(f"export: {exc}")
                        for rule in rules:
                            if canceled():
                                break
                            set_progress(current_model=mid, current_step=rule.label)
                            try:
                                metrics = evaluate_run_under_rule(
                                    job_cfg, project_id, mid, rule,
                                    class_mapping=batch_class_mapping or None,
                                )
                                entry["steps"][rule.id] = self._batch_metrics_summary(metrics)
                            except Exception as exc:  # noqa: BLE001
                                entry["errors"].append(f"{rule.label}: {exc}")
                        # One scoring pass per requested IR precision; each lands
                        # in its own manifest block, so FP16 and INT8 accumulate.
                        for prec in (exported_precisions if evaluate_exported else []):
                            if canceled():
                                break
                            step = f"deployed {prec}"
                            set_progress(current_model=mid, current_step=step)
                            try:
                                metrics = evaluate_exported_run(
                                    job_cfg, project_id, mid,
                                    use_model_dataset=False,
                                    precision=prec,
                                    rule=exported_rule,
                                )
                                summary = self._batch_metrics_summary(metrics)
                                summary["model_file"] = metrics.get("model_file")
                                summary["precision"] = metrics.get("precision") or prec
                                entry["steps"][f"deployed:{prec}"] = summary
                            except Exception as exc:  # noqa: BLE001
                                entry["errors"].append(f"{step}: {exc}")
                        with batch_eval_lock:
                            if batch_eval_job and batch_eval_job.get("job_id") == job_id:
                                batch_eval_job["results"].append(entry)
                                batch_eval_job["completed"] += 1
                                if entry["errors"]:
                                    batch_eval_job["failed_count"] += 1
                    set_progress(
                        state="canceled" if canceled() else "finished",
                        current_model="", current_step="",
                        ended_at=time.time(),
                    )
                except BaseException as exc:  # noqa: BLE001
                    # BaseException, not Exception: whatever stops this worker
                    # has to reach the UI, or the job hangs at "starting".
                    log.exception("batch evaluation job failed")
                    set_progress(
                        state="error",
                        error=f"{type(exc).__name__}: {exc}",
                        ended_at=time.time(),
                    )

            thread = threading.Thread(target=run_job, daemon=True)
            with batch_eval_lock:
                batch_eval_job["thread"] = thread
                payload = self._batch_eval_payload(batch_eval_job)
                thread.start()
            self._send_json(payload)

        def _handle_train_set_active(self) -> None:
            model_id = str(self._read_json_body().get("model_id") or "").strip()
            if not model_id:
                self._send_json({"ok": False, "error": "missing model_id"}, status=400)
                return
            from sed.train import registry
            try:
                registry.set_active_model(project_id, model_id)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)})
                return
            self._send_json({"ok": True, "model_id": model_id})

        # ---- Inference lab ---------------------------------------------

        def _handle_inference_models(self) -> None:
            self._send_json(_inference_models_payload(project_id, layout))

        def _handle_inference_settings(self, qs: dict) -> None:
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            model_id = (qs.get("model_id", [""])[0] or "").strip()
            device = (qs.get("device", [""])[0] or "").strip() or None
            raw_channels = (qs.get("channels", [""])[0] or "").strip()
            mode = (qs.get("multi_mic_mode", [""])[0] or "").strip() or None
            try:
                channels = (
                    _validate_inference_channels(raw_channels, list(cfg.inference.inference_channels))
                    if raw_channels else None
                )
                payload = _inference_settings_payload(
                    project_id,
                    layout,
                    cfg,
                    model_id=model_id,
                    requested_device=device,
                    channels=channels,
                    multi_mic_mode=mode,
                )
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json(payload)

        def _meter_snapshot(self) -> dict:
            with meter_lock:
                job = meter_job
                if not job:
                    return {"state": "idle"}
                return {
                    "state": job.get("state", "idle"),
                    "device": job.get("device", ""),
                    "device_index": job.get("device_index"),
                    "device_name": job.get("device_name", ""),
                    "sample_rate": job.get("sample_rate"),
                    "channels": list(job.get("channels") or []),
                    "levels": _json_safe(job.get("levels") or []),
                    "xruns": int(job.get("xruns") or 0),
                    "blocks": int(job.get("blocks") or 0),
                    "last_status": str(job.get("last_status") or ""),
                    "error": str(job.get("error") or ""),
                    "elapsed_s": max(0.0, time.time() - float(job.get("started_at") or time.time())),
                }

        def _handle_inference_meter_status(self) -> None:
            self._send_json({"ok": True, "meter": self._meter_snapshot()})

        def _handle_inference_meter_stop(self) -> None:
            self._send_json({"ok": True, "meter": _json_safe(stop_meter())})

        def _handle_inference_meter_start(self) -> None:
            nonlocal meter_job
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            with inference_lock:
                active = inference_job and str(inference_job.get("state")) in ("starting", "running", "stopping")
            if active:
                self._send_json({"ok": False, "error": "stop inference before starting the meter"}, status=409)
                return
            with meter_lock:
                if meter_job and str(meter_job.get("state")) == "running":
                    self._send_json({"ok": False, "error": "meter is already running"}, status=409)
                    return
            stream = None
            try:
                import numpy as np
                import sounddevice as sd
                from sed.audio.capture import resolve_device

                device = str(body.get("device") or cfg.deploy.record_device or "default")
                channels = _validate_inference_channels(
                    body.get("channels"), list(cfg.inference.inference_channels)
                )
                n_open = max(channels) + 1
                sample_rate = int(body.get("sample_rate") or cfg.audio.record_sample_rate)
                blocksize = int(body.get("blocksize") or cfg.audio.blocksize)
                idx = resolve_device(device)
                info = sd.query_devices(idx)
                if n_open > int(info["max_input_channels"]):
                    raise ValueError(
                        f"channels {channels} need {n_open} input channel(s), "
                        f"but device has {info['max_input_channels']}"
                    )
                levels = [
                    {"channel": ch, "peak": 0.0, "rms": 0.0, "peak_dbfs": -120.0, "rms_dbfs": -120.0}
                    for ch in channels
                ]
                job = {
                    "state": "running",
                    "device": device,
                    "device_index": idx,
                    "device_name": str(info.get("name") or ""),
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "levels": levels,
                    "xruns": 0,
                    "blocks": 0,
                    "last_status": "",
                    "error": "",
                    "started_at": time.time(),
                    "stream": None,
                }

                def callback(indata, frames, time_info, status):
                    _ = (frames, time_info)
                    block = indata.astype("float32", copy=False)
                    next_levels = []
                    for ch in channels:
                        x = block[:, ch]
                        peak = float(np.max(np.abs(x))) if x.size else 0.0
                        rms = float(np.sqrt(np.mean(x ** 2) + 1e-12)) if x.size else 0.0
                        next_levels.append({
                            "channel": ch,
                            "peak": peak,
                            "rms": rms,
                            "peak_dbfs": float(20.0 * np.log10(peak + 1e-12)),
                            "rms_dbfs": float(20.0 * np.log10(rms + 1e-12)),
                        })
                    with meter_lock:
                        if meter_job is job:
                            job["levels"] = next_levels
                            job["blocks"] = int(job.get("blocks") or 0) + 1
                            if status:
                                job["xruns"] = int(job.get("xruns") or 0) + 1
                                job["last_status"] = str(status)

                stream = sd.InputStream(
                    device=idx,
                    channels=n_open,
                    samplerate=sample_rate,
                    blocksize=blocksize,
                    dtype="float32",
                    callback=callback,
                )
                with meter_lock:
                    meter_job = job
                    job["stream"] = stream
                stream.start()
            except Exception as exc:  # noqa: BLE001
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:  # noqa: BLE001
                        pass
                with meter_lock:
                    if meter_job is not None and str(meter_job.get("device")) == str(body.get("device") or cfg.deploy.record_device or "default"):
                        meter_job = None
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, "meter": self._meter_snapshot()})

        def _inference_snapshot(self) -> dict:
            with inference_lock:
                job = inference_job
                if not job:
                    return {"state": "idle", "safe_test": True, "artnet": False, "mining": False}
                runtime = job.get("runtime")
                snapshot = {
                    "job_id": job.get("job_id", ""),
                    "state": job.get("state", "idle"),
                    "model_id": job.get("model_id", ""),
                    "device": job.get("device", ""),
                    "error": job.get("error", ""),
                    "started_at": job.get("started_at"),
                    "elapsed_s": max(0.0, time.time() - float(job.get("started_at") or time.time())),
                    "classes": list(job.get("classes") or []),
                    "active_targets": list(job.get("active_targets") or []),
                    "detections": list(job.get("detections") or []),
                    "settings": dict(job.get("settings") or {}),
                    "safe_test": not bool(job.get("artnet_active")) and not bool(job.get("mining_enabled")),
                    "artnet": bool(job.get("artnet_active")),
                    "artnet_requested": bool(job.get("artnet_requested")),
                    "mining": bool(job.get("mining_enabled")),
                    "mining_classes": list(job.get("mining_classes") or []),
                    "stats": dict(job.get("stats") or {}),
                }
            if runtime is not None:
                try:
                    stats = dict(runtime.stats)
                    if getattr(runtime, "capture", None) is not None:
                        stats["capture"] = dict(runtime.capture.stats)
                    snapshot["stats"] = stats
                    snapshot["classes"] = list(getattr(runtime, "classes", snapshot["classes"]) or [])
                except Exception as exc:  # noqa: BLE001
                    snapshot["stats_error"] = str(exc)
            return _json_safe(snapshot)

        def _handle_inference_status(self) -> None:
            self._send_json({"ok": True, "inference": self._inference_snapshot()})

        def _handle_inference_stop(self) -> None:
            self._send_json({"ok": True, "inference": _json_safe(stop_inference())})

        def _handle_inference_start(self) -> None:
            nonlocal inference_job
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            with meter_lock:
                meter_active = meter_job and str(meter_job.get("state")) == "running"
            if meter_active:
                self._send_json({"ok": False, "error": "stop the meter before starting inference"}, status=409)
                return
            with inference_lock:
                if inference_job and str(inference_job.get("state")) in ("starting", "running", "stopping"):
                    self._send_json({"ok": False, "error": "inference is already running"}, status=409)
                    return

            model_id = str(body.get("model_id") or "").strip()
            if not model_id:
                model_id = str(_inference_models_payload(project_id, layout).get("selected") or "")
            if not model_id:
                self._send_json({"ok": False, "error": "no model selected"}, status=400)
                return
            model_dir = layout.model_dir(model_id)
            if not (model_dir / "training_config.json").exists():
                self._send_json({"ok": False, "error": f"unknown model: {model_id}"}, status=404)
                return
            ov = _model_openvino_summary(model_dir)
            # Engine choice: auto walks OpenVINO IR → ONNX → checkpoint; the
            # explicit values pin one artifact (sed.runtime.backends).
            precision = str(body.get("precision") or "auto").strip().lower()
            if precision not in ("auto", "fp16", "int8", "onnx", "checkpoint"):
                self._send_json({"ok": False, "error": f"unknown engine: {precision}"}, status=400)
                return
            if precision in ("fp16", "int8") and precision not in ov.get("available_precisions", []):
                self._send_json({"ok": False, "error": f"no {precision} export for this model"}, status=400)
                return
            if precision == "onnx" and not (model_dir / "model.onnx").exists():
                self._send_json({"ok": False, "error": "no model.onnx export for this model"}, status=400)
                return

            try:
                from sed.config import load_config_for_inference_model

                runtime_cfg = load_config_for_inference_model(project_id, model_id)
                device = str(body.get("device") or runtime_cfg.deploy.record_device or "default")
                channels = _validate_inference_channels(
                    body.get("channels"), list(runtime_cfg.inference.inference_channels)
                )
                mode = _validate_multi_mic_mode(
                    body.get("multi_mic_mode"),
                    channels,
                    default=runtime_cfg.inference.multi_mic_mode,
                )
                onset_enabled = _coerce_bool(
                    body.get("onset_gating_enabled"),
                    runtime_cfg.inference.onset_gating_enabled,
                )
                min_dbfs = _coerce_optional_dbfs(
                    body.get("min_dbfs"),
                    runtime_cfg.inference.min_dbfs,
                )
                default_targets = _active_inference_targets(runtime_cfg)
                active_targets = _coerce_targets(
                    body.get("targets"),
                    known=set(runtime_cfg.model.classes),
                    default=default_targets,
                )
                mining_enabled = _coerce_bool(body.get("mine"), False)
                artnet_requested = _coerce_bool(body.get("with_artnet"), False)

                runtime_cfg.deploy.record_device = device
                runtime_cfg.inference.inference_channels = channels
                runtime_cfg.inference.multi_mic_mode = mode
                runtime_cfg.inference.onset_gating_enabled = onset_enabled
                runtime_cfg.inference.min_dbfs = min_dbfs
                runtime_cfg.mining.enabled = mining_enabled
                runtime_cfg.artnet.enabled = bool(artnet_requested and runtime_cfg.artnet.enabled)
                runtime_cfg = Config.model_validate(runtime_cfg.model_dump(mode="json"))
                mining_classes = _inference_mining_classes(
                    runtime_cfg, active_targets, mining_enabled
                )
                device_payload = _inference_device_payload(runtime_cfg, device, channels)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return

            if device_payload.get("resolved_error"):
                self._send_json({"ok": False, "error": device_payload["resolved_error"], "device": device_payload}, status=400)
                return
            if device_payload.get("resolved_device") and not device_payload["resolved_device"].get("enough_channels"):
                self._send_json({
                    "ok": False,
                    "error": "device does not expose enough input channels",
                    "device": device_payload,
                }, status=400)
                return

            from sed.runtime.manifest import resolve_runtime_artifact

            model_precision = None if precision == "auto" else precision
            try:
                resolved_engine, resolved_path = resolve_runtime_artifact(
                    model_dir, prefer=model_precision
                )
            except FileNotFoundError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            if resolved_engine == "openvino":
                name = resolved_path.name
                resolved_precision = "int8" if "int8" in name else ("fp16" if "fp16" in name else "ir")
            else:
                resolved_precision = resolved_engine

            job_id = uuid.uuid4().hex[:12]
            settings = {
                "model_id": model_id,
                "device": device,
                "channels": channels,
                "multi_mic_mode": mode,
                "onset_gating_enabled": onset_enabled,
                "min_dbfs": min_dbfs,
                "targets": sorted(active_targets),
                "mine": mining_enabled,
                "mining_classes": sorted(mining_classes),
                "with_artnet": artnet_requested,
                "artnet_active": bool(runtime_cfg.artnet.enabled),
                "hop_ms": runtime_cfg.inference.hop_ms,
                "precision_requested": precision,
                "precision": resolved_precision,
                "engine": resolved_engine,
                "artifact": resolved_path.name,
            }
            job = {
                "job_id": job_id,
                "state": "starting",
                "model_id": model_id,
                "device": device,
                "started_at": time.time(),
                "ended_at": None,
                "error": "",
                "runtime": None,
                "artnet": None,
                "classes": [],
                "active_targets": sorted(active_targets),
                "mining_classes": sorted(mining_classes),
                "mining_enabled": mining_enabled,
                "artnet_requested": artnet_requested,
                "artnet_active": bool(runtime_cfg.artnet.enabled),
                "settings": settings,
                "detections": collections.deque(maxlen=24),
                "stats": {},
            }
            with inference_lock:
                inference_job = job

            def run_inference() -> None:
                nonlocal inference_job
                runtime = None
                artnet = None
                try:
                    from sed.lighting.artnet import ArtnetController, trigger_to_target
                    from sed.runtime.mining import DetectionMiner
                    from sed.runtime.infer import RealtimeInference

                    miner = (
                        DetectionMiner(cfg=runtime_cfg, project_id=project_id, classes=mining_classes)
                        if mining_enabled and mining_classes else None
                    )
                    if runtime_cfg.artnet.enabled:
                        artnet = ArtnetController(runtime_cfg)
                        with inference_lock:
                            if inference_job is job:
                                job["artnet"] = artnet

                    def on_trigger(ev) -> None:
                        if artnet is not None:
                            target = trigger_to_target(runtime_cfg, ev.class_name)
                            if target is not None:
                                ch, value = target
                                artnet.fire(ch, peak=value)

                    def on_frame(probs: dict[str, float]) -> None:
                        with inference_lock:
                            if inference_job is job:
                                job["last_frame"] = dict(probs)

                    def on_detection(det):
                        saved = miner.maybe_save(det) if miner is not None else None
                        with inference_lock:
                            if inference_job is job:
                                job["detections"].append({
                                    "ts": time.time(),
                                    "runtime_ts": det.timestamp,
                                    "kind": det.kind,
                                    "class": det.class_name,
                                    "score": float(det.confidence),
                                    "mic": det.best_channel,
                                    "peak_dbfs": det.peak_dbfs,
                                    "rms_dbfs": det.rms_dbfs,
                                    "min_dbfs": det.min_dbfs,
                                    "saved": str(saved) if saved is not None else "",
                                    "active_targets": list(det.active_targets),
                                    "probs": dict(det.probs),
                                    "smoothed": dict(det.smoothed),
                                })
                        return saved

                    runtime = RealtimeInference(
                        cfg=runtime_cfg,
                        project_id=project_id,
                        on_trigger=on_trigger,
                        on_frame=on_frame,
                        on_detection=on_detection,
                        active_trigger_classes=active_targets,
                        detection_classes=mining_classes,
                        live_monitor=True,
                        model_precision=model_precision,
                    )
                    with inference_lock:
                        if inference_job is not job or str(job.get("state")) == "stopping":
                            should_stop = True
                        else:
                            should_stop = False
                            job["runtime"] = runtime
                            job["classes"] = list(runtime.classes)
                            job["active_targets"] = sorted(active_targets)
                    if should_stop:
                        runtime.stop()
                        if artnet is not None:
                            artnet.stop()
                        return
                    runtime.start()
                    with inference_lock:
                        if inference_job is job and str(job.get("state")) not in ("stopping", "stopped"):
                            job["state"] = "running"
                except Exception as exc:  # noqa: BLE001
                    if runtime is not None:
                        try:
                            runtime.stop()
                        except Exception:  # noqa: BLE001
                            pass
                    if artnet is not None:
                        try:
                            artnet.stop()
                        except Exception:  # noqa: BLE001
                            pass
                    with inference_lock:
                        if inference_job is job:
                            job["runtime"] = None
                            job["artnet"] = None
                            job["state"] = "error"
                            job["error"] = str(exc)
                            job["ended_at"] = time.time()

            threading.Thread(target=run_inference, daemon=True).start()
            self._send_json({"ok": True, "inference": self._inference_snapshot()})

        def _handle_inference_set_active(self) -> None:
            model_id = str(self._read_json_body().get("model_id") or "").strip()
            if not model_id:
                self._send_json({"ok": False, "error": "missing model_id"}, status=400)
                return
            if not (layout.model_dir(model_id) / "training_config.json").exists():
                self._send_json({"ok": False, "error": f"unknown model: {model_id}"}, status=404)
                return
            from sed.train import registry
            try:
                registry.set_active_model(project_id, model_id)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json({"ok": True, "model_id": model_id})

        def _handle_inference_save_input_setup(self) -> None:
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            body = self._read_json_body()
            try:
                device = str(body.get("device") or cfg.deploy.record_device or "default")
                channels = _validate_inference_channels(
                    body.get("channels"), list(cfg.inference.inference_channels)
                )
                mode = _validate_multi_mic_mode(
                    body.get("multi_mic_mode"),
                    channels,
                    default=cfg.inference.multi_mic_mode,
                )
                onset_enabled = _coerce_bool(
                    body.get("onset_gating_enabled"),
                    cfg.inference.onset_gating_enabled,
                )
                device_payload = _inference_device_payload(cfg, device, channels)
                if device_payload.get("resolved_error"):
                    raise ValueError(str(device_payload["resolved_error"]))
                if device_payload.get("resolved_device") and not device_payload["resolved_device"].get("enough_channels"):
                    raise ValueError("device does not expose enough input channels")
                from sed.config import update_config_toml_values

                with cfg_write_lock:
                    update_config_toml_values(layout.config_toml, {
                        "deploy": {"record_device": device},
                        "inference": {
                            "inference_channels": channels,
                            "multi_mic_mode": mode,
                            "onset_gating_enabled": onset_enabled,
                        },
                    })
                    cfg.deploy.record_device = device
                    cfg.inference.inference_channels = channels
                    cfg.inference.multi_mic_mode = mode
                    cfg.inference.onset_gating_enabled = onset_enabled
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json({
                "ok": True,
                "device": device,
                "channels": channels,
                "multi_mic_mode": mode,
                "onset_gating_enabled": onset_enabled,
            })

        def _handle_selection(self) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            from sed.data.compose import load_dataset_selection, write_dataset_selection
            if body.get("action") == "assign_clips":
                class_name = str(body.get("class_name") or "").strip()
                clips = body.get("clips", [])
                if not class_name or not isinstance(clips, list):
                    self._send(400, b"invalid assignment payload")
                    return
                selection = load_dataset_selection(layout)
                by_class = selection.setdefault("clip_assignments", {})
                rows = by_class.setdefault(class_name, [])
                existing_keys = {
                    (str(r.get("uuid") or ""), str(r.get("relpath") or ""), str(r.get("path") or ""))
                    for r in rows
                    if isinstance(r, dict)
                }
                added = 0
                for clip in clips:
                    if not isinstance(clip, dict):
                        continue
                    row = {
                        k: str(clip.get(k))
                        for k in ("uuid", "relpath", "path", "label", "project_id")
                        if clip.get(k)
                    }
                    if not row:
                        continue
                    key = (row.get("uuid", ""), row.get("relpath", ""), row.get("path", ""))
                    if key in existing_keys:
                        continue
                    rows.append(row)
                    existing_keys.add(key)
                    added += 1
                selection["updated_at"] = datetime.now(UTC).isoformat()
                path = write_dataset_selection(layout, selection)
                classes = self._adopt_definition_classes(selection, [class_name])
                self._send_json({
                    "ok": True, "selection": str(path), "added": added,
                    "target_classes": classes,
                    **self._selection_meta(selection),
                })
                return

            path = write_dataset_selection(layout, body)
            self._send_json({"ok": True, "selection": str(path)})

        def _selection_meta(self, selection: dict) -> dict:
            """Cheap dataset-state annotations for selection-shaped responses:
            compose staleness (recompose-needed top-bar flag) + source-rule
            preset state. File reads and hashing only — no library scans."""
            from sed.data.compose import dataset_compose_state, dataset_preset_state
            meta: dict = {"compose_state": {}, "preset_state": {}}
            try:
                meta["compose_state"] = dataset_compose_state(layout, selection, cfg)
            except Exception:  # noqa: BLE001 — annotations are display-only.
                log.debug("compose-state annotation failed", exc_info=True)
            try:
                meta["preset_state"] = dataset_preset_state(layout, selection, cfg)
            except Exception:  # noqa: BLE001 — annotations are display-only.
                log.debug("preset-state annotation failed", exc_info=True)
            return meta

        def _handle_dataset_preset(self, action: str) -> None:
            nonlocal cfg
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            from sed.data.compose import (
                apply_dataset_preset,
                delete_dataset_preset,
                load_dataset_selection,
                new_dataset_selection,
                rename_dataset_preset,
                save_dataset_preset,
            )
            name = str(body.get("name") or "").strip()
            if action not in {"new"} and not name:
                self._send_json({"ok": False, "error": "name is required"}, status=400)
                return
            if action == "new":
                from sed.data.compose import BLANK_DATASET_CLASSES
                from sed.config import load_config

                selection = new_dataset_selection(layout)
                # new_dataset_selection() just rewrote config.toml's class
                # list; the server's cached cfg would otherwise keep reporting
                # the previous definition's classes (same reload apply() does).
                cfg = load_config(project_id)
                record_activity_event("New blank dataset", detail="source rules and classes cleared", section="source_rules")
                self._send_json({
                    "ok": True,
                    "selection": selection,
                    "preset": {"classes": list(BLANK_DATASET_CLASSES)},
                    **self._selection_meta(selection),
                })
                return
            if action == "save":
                selection = load_dataset_selection(layout)
                try:
                    # cfg carries the class list, so the preset is a complete
                    # dataset definition rather than source rules alone.
                    record = save_dataset_preset(layout, name, selection, cfg)
                except ValueError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
                summary = {k: record[k] for k in ("name", "slug", "rule_count", "clip_count")}
                summary["classes"] = list(record.get("classes") or [])
                record_activity_event("Dataset preset saved", detail=record["name"], section="source_rules")
                self._send_json({
                    "ok": True, "preset": summary, **self._selection_meta(selection),
                })
                return
            if action == "apply":
                try:
                    selection, record = apply_dataset_preset(layout, name)
                except KeyError:
                    self._send_json({"ok": False, "error": "preset not found"}, status=404)
                    return
                if record.get("classes"):
                    # A v2 preset rewrites config.toml's target_classes; the
                    # server's cached cfg would otherwise report the old ones.
                    from sed.config import load_config

                    cfg = load_config(project_id)
                summary = {k: record[k] for k in ("name", "slug", "rule_count", "clip_count")}
                summary["classes"] = list(record.get("classes") or [])
                record_activity_event("Dataset preset loaded", detail=record["name"], section="source_rules")
                self._send_json({
                    "ok": True,
                    "preset": summary,
                    "selection": selection,
                    **self._selection_meta(selection),
                })
                return
            if action == "delete":
                removed = delete_dataset_preset(layout, name)
                if not removed:
                    self._send_json({"ok": False, "error": "preset not found"}, status=404)
                    return
                selection = load_dataset_selection(layout)
                record_activity_event("Dataset preset deleted", detail=name, section="source_rules")
                self._send_json({
                    "ok": True, "removed": True, **self._selection_meta(selection),
                })
                return
            if action == "rename":
                try:
                    record = rename_dataset_preset(layout, name, str(body.get("new_name") or ""))
                except KeyError:
                    self._send_json({"ok": False, "error": "preset not found"}, status=404)
                    return
                except ValueError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
                selection = load_dataset_selection(layout)
                record_activity_event("Dataset preset renamed", detail=record["name"], section="source_rules")
                self._send_json({
                    "ok": True,
                    "preset": {k: record[k] for k in ("name", "slug", "rule_count", "clip_count")},
                    **self._selection_meta(selection),
                })
                return
            self._send(400, b"unknown action")

        def _handle_selection_get(self) -> None:
            from sed.data.compose import (
                ensure_auto_mined_negative_rules,
                load_dataset_selection,
                selection_manifest_path,
                write_dataset_selection,
            )
            selection = load_dataset_selection(layout)
            selection, auto_changed = ensure_auto_mined_negative_rules(
                cfg, layout, selection, write=False
            )
            from sed.data.compose import ensure_class_visuals
            class_visuals, visuals_changed = ensure_class_visuals(
                selection, list(cfg.model.classes) if cfg else []
            )
            settings = selection.get("augmentation_settings")
            if not isinstance(settings, dict) or not settings:
                settings = augmentation_settings_snapshot(cfg)
                selection = dict(selection)
                selection["augmentation_settings"] = settings
            if auto_changed or visuals_changed:
                write_dataset_selection(layout, selection)
            # Recompute display counts from disk so deletes/imports show through.
            try:
                _refresh_selection_counts(selection, cfg=cfg, project_id=layout.id)
            except Exception:  # noqa: BLE001 — counts are display-only.
                log.debug("selection count refresh failed", exc_info=True)
            # Annotate derived source rules with their overlay class(es) for the UI.
            try:
                _attach_overlay_labels(selection, project_id=layout.id)
            except Exception:  # noqa: BLE001 — overlay labels are display-only.
                log.debug("overlay-label annotation failed", exc_info=True)
            self._send_json({
                "ok": True,
                "selection": selection,
                "class_visuals": class_visuals,
                "path": str(selection_manifest_path(layout)),
                "augmentation_settings": settings,
                **self._selection_meta(selection),
            })

        def _handle_augmentation_generated_counts(self) -> None:
            from sed.data.compose import load_dataset_selection
            from sed.data.inspect import augmentation_recipe_counts
            selection = load_dataset_selection(layout)
            settings = selection.get("augmentation_settings")
            if not isinstance(settings, dict) or not settings:
                settings = augmentation_settings_snapshot(cfg)
            recipes = selection.get("augmentation_recipes")
            if not isinstance(recipes, list):
                recipes = []
            self._send_json({
                "ok": True,
                "augmentation_generated_counts": augmentation_recipe_counts(
                    layout,
                    recipes=recipes,
                    # Baseline mirrors the run-time cfg snapshot; each recipe's
                    # own settings dict is layered on top inside the counter.
                    settings=_normalise_augmentation_settings(settings, cfg),
                ),
            })

        def _handle_panel_totals(self) -> None:
            if cfg is None:
                self._send(400, b"missing config")
                return
            from sed.data.inspect import panel_totals
            self._send_json({"ok": True, "panel_totals": panel_totals(cfg, layout)})

        def _handle_clips_page(
            self, tab: str, class_name: str, offset: str = "0",
            limit: str = "100", run_id: str | None = None,
            origin_class: str = "", query: str = "",
            origins: str | None = None, group_axis: str = "", group_key: str = "",
        ) -> None:
            if cfg is None:
                self._send(400, b"missing config")
                return
            if tab not in {"dataset", "augmented", "frozen", "mined", "pretraining"}:
                self._send_json({"ok": False, "error": f"unknown tab: {tab}"}, status=400)
                return
            try:
                off = max(0, int(offset))
                lim = max(1, min(1000, int(limit)))
            except (TypeError, ValueError):
                self._send_json({"ok": False, "error": "offset/limit must be integers"}, status=400)
                return
            # Subbucket / sub-group narrowing is resolved from the gathered
            # source items, which only the three review tabs have.  Mined and
            # pretraining build their cards straight from disk.
            filterable = tab in {"dataset", "augmented", "frozen"}
            selected_origins: set[str] | None = None
            if filterable and origins is not None:
                # An empty set here is not "no filter": it is the user having
                # unchecked every subbucket, which must show nothing.
                selected_origins = {part for part in origins.split("|") if part}
            try:
                from sed.data.inspect import clips_page
                query_label = f" · searching {query!r}" if query else ""
                activity_job = start_clip_activity(
                    "Searching clips" if query else "Loading clips",
                    f"{tab} · {class_name}{query_label}", tab=tab,
                )
                try:
                    cards, total = clips_page(
                        cfg, layout, tab, class_name, off, lim,
                        run_id=run_id,
                        origin_class=origin_class if filterable else "",
                        origins=selected_origins,
                        group_axis=group_axis if filterable else "",
                        group_key=group_key if filterable else "",
                        query=query,
                        return_total=True,
                    )
                finally:
                    finish_clip_activity(activity_job)
            except Exception as exc:  # noqa: BLE001
                log.exception("clips page failed")
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json({
                "ok": True, "tab": tab, "class": class_name,
                "offset": off, "limit": lim, "origin_class": origin_class,
                "origins": origins or "", "group_axis": group_axis, "group_key": group_key,
                "query": query,
                "total": total, "clips": cards,
            })

        def _handle_selection_rule(self) -> None:
            nonlocal cfg
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            from sed.data.compose import (
                load_dataset_selection,
                mark_auto_mined_rules_dismissed,
                write_dataset_selection,
            )

            action = str(body.get("action") or "add").strip().lower()
            class_name = str(body.get("class_name") or "").strip()
            if action == "add" and not class_name:
                self._send(400, b"class_name is required")
                return
            # A rule's class is part of the dataset definition, and the
            # definition owns the class space: assigning to a class the project
            # does not list yet adds it rather than being rejected.
            adopt = [
                name for name in (
                    class_name,
                    str(body.get("new_class") or "").strip(),
                ) if name and action in {"add", "update"}
            ]

            selection = _ensure_ui_selection(load_dataset_selection(layout))
            assignments = selection.setdefault("assignments", {})
            removed = 0
            added = 0

            if action == "add":
                rule = _normalise_filter_rule(
                    body.get("rule") or {},
                    default_split=str(body.get("split") or "train"),
                )
                rows = assignments.setdefault(class_name, [])
                if not isinstance(rows, list):
                    rows = [rows] if isinstance(rows, dict) else []
                    assignments[class_name] = rows
                rows.append(rule)
                added = 1
            elif action == "update":
                old_class = str(body.get("old_class") or class_name).strip()
                new_class = str(body.get("new_class") or class_name).strip()
                rule_id = str(body.get("rule_id") or "").strip()
                if not old_class or not new_class or not rule_id:
                    self._send_json({"ok": False, "error": "old_class, new_class and rule_id are required"}, status=400)
                    return
                source_rows = assignments.get(old_class, [])
                if isinstance(source_rows, dict):
                    source_rows = [source_rows]
                if not isinstance(source_rows, list):
                    source_rows = []
                found = None
                kept = []
                for row in source_rows:
                    if isinstance(row, dict) and str(row.get("id") or "") == rule_id and found is None:
                        found = _normalise_filter_rule(body.get("rule") or row, default_split="train")
                    else:
                        kept.append(row)
                if found is None:
                    self._send_json({"ok": False, "error": "source rule not found"}, status=404)
                    return
                assignments[old_class] = kept
                if not kept:
                    assignments.pop(old_class, None)
                assignments.setdefault(new_class, []).append(found)
                added = 1
            elif action == "remove":
                rule_id = str(body.get("rule_id") or "").strip()
                targets = [class_name] if class_name else list(assignments.keys())
                for target in targets:
                    rows = assignments.get(target, [])
                    if not isinstance(rows, list):
                        continue
                    kept = []
                    for row in rows:
                        if isinstance(row, dict) and rule_id and str(row.get("id") or "") == rule_id:
                            mark_auto_mined_rules_dismissed(selection, [row])
                            removed += 1
                            continue
                        kept.append(row)
                    if kept:
                        assignments[target] = kept
                    else:
                        assignments.pop(target, None)
            elif action == "clear":
                if class_name:
                    rows = assignments.get(class_name, []) or []
                    if isinstance(rows, dict):
                        rows = [rows]
                    mark_auto_mined_rules_dismissed(
                        selection, [row for row in rows if isinstance(row, dict)]
                    )
                    removed = len(rows)
                    assignments[class_name] = []
                else:
                    all_rows = []
                    for rows in assignments.values():
                        if isinstance(rows, dict):
                            all_rows.append(rows)
                        elif isinstance(rows, list):
                            all_rows.extend(row for row in rows if isinstance(row, dict))
                    mark_auto_mined_rules_dismissed(selection, all_rows)
                    removed = len(all_rows)
                    selection["assignments"] = {}
            else:
                self._send(400, b"unknown action")
                return

            selection["updated_at"] = _now_iso()
            path = write_dataset_selection(layout, selection)
            classes = self._adopt_definition_classes(selection, adopt)
            # A rule can create a previously unseen target class.  Give it a
            # durable display token in the same write path, without changing
            # any compose-relevant selection content.
            from sed.data.compose import ensure_class_visuals
            class_visuals, visuals_changed = ensure_class_visuals(selection, classes)
            if visuals_changed:
                path = write_dataset_selection(layout, selection)
            record_activity_event(
                f"Source rule {action}",
                detail=f"{added or removed} rule(s)" if action in {"add", "remove", "clear", "update"} else "",
                section="source_rules",
            )
            self._send_json({
                "ok": True,
                "selection": selection,
                "path": str(path),
                "added": added,
                "removed": removed,
                "target_classes": classes,
                "class_visuals": class_visuals,
                **self._selection_meta(selection),
            })

        def _adopt_definition_classes(
            self, selection: dict, extra: list[str] | None = None
        ) -> list[str]:
            """Keep ``[dataset] target_classes`` equal to the definition's classes.

            Returns the effective class list so a response can push it straight
            into the UI's class chips and assign dropdown.
            """
            nonlocal cfg
            from sed.data.compose import ensure_dataset_definition_classes

            try:
                added = ensure_dataset_definition_classes(layout, cfg, selection, extra)
            except Exception:  # noqa: BLE001 — class adoption must not fail a rule edit
                log.warning("could not adopt dataset-definition classes", exc_info=True)
                added = None
            if added:
                from sed.config import load_config

                cfg = load_config(project_id)
            return [str(c) for c in (cfg.model.classes if cfg else [])]

        def _handle_augmentation_settings(self) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            from sed.data.compose import load_dataset_selection, write_dataset_selection

            raw_settings = body.get("settings", body)
            settings = _normalise_augmentation_settings(raw_settings, cfg)
            selection = _ensure_ui_selection(load_dataset_selection(layout))
            selection["augmentation_settings"] = settings
            selection["updated_at"] = _now_iso()
            path = write_dataset_selection(layout, selection)
            self._send_json({
                "ok": True,
                "settings": settings,
                "selection": selection,
                "path": str(path),
            })

        def _handle_augmentation_rule(self) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            from sed.data.compose import load_dataset_selection, write_dataset_selection

            action = str(body.get("action") or "add").strip().lower()

            selection = _ensure_ui_selection(load_dataset_selection(layout))
            recipes = selection.setdefault("augmentation_recipes", [])
            if not isinstance(recipes, list):
                recipes = []
                selection["augmentation_recipes"] = recipes
            added = 0
            removed = 0
            deleted_generated = 0

            if action == "add":
                try:
	                    recipe = _normalise_augmentation_recipe(body.get("recipe") or {})
	                    recipe = _refresh_recipe_counts(
	                        recipe, selection=selection, cfg=cfg, project_id=project_id,
	                        trust_existing_counts=True,
	                    ) if cfg is not None else recipe
                except ValueError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                    return
                recipes.append(recipe)
                added = 1
            elif action == "update":
                raw_patch = body.get("recipe") or {}
                target_id = str(raw_patch.get("recipe_id") or raw_patch.get("id") or "").strip()
                replaced = False
                for i, row in enumerate(recipes):
                    if (
                        isinstance(row, dict)
                        and str(row.get("recipe_id") or row.get("id") or "") == target_id
                    ):
                        # Merge the raw patch onto the existing row BEFORE
                        # normalising, so unspecified fields (filters, mask
                        # thresholds, description, …) keep their saved values
                        # instead of being defaulted away by the normaliser.
                        merged_raw = {**row, **raw_patch}
                        try:
	                            recipe = _normalise_augmentation_recipe(merged_raw)
	                            recipe = _refresh_recipe_counts(
	                                recipe, selection=selection, cfg=cfg, project_id=project_id,
	                                trust_existing_counts=True,
	                            ) if cfg is not None else recipe
                        except ValueError as exc:
                            self._send_json({"ok": False, "error": str(exc)}, status=400)
                            return
                        recipes[i] = recipe
                        replaced = True
                        break
                if not replaced:
                    self._send_json(
                        {"ok": False, "error": "recipe not found"}, status=404,
                    )
                    return
                added = 1
            elif action == "delete_generated":
                rule_id = str(body.get("rule_id") or body.get("recipe_id") or "").strip()
                if not rule_id:
                    self._send_json(
                        {"ok": False, "error": "rule_id required"}, status=400,
                    )
                    return
                from sed.data.augment import delete_generated_for_recipes
                deleted_generated = delete_generated_for_recipes(
                    project_id,
                    [rule_id],
                    frozen_test=None,
                )
                # No selection mutation — just respond with current state.
                selection["updated_at"] = _now_iso()
                self._send_json({
                    "ok": True,
                    "selection": selection,
                    "added": 0,
                    "removed": 0,
                    "deleted_generated": deleted_generated,
                })
                return
            elif action == "remove":
                rule_id = str(body.get("rule_id") or "").strip()
                kept = []
                removed_ids: list[str] = []
                for row in recipes:
                    if isinstance(row, dict) and rule_id and str(row.get("id") or row.get("recipe_id") or "") == rule_id:
                        removed += 1
                        removed_ids.append(str(row.get("recipe_id") or row.get("id") or rule_id))
                        continue
                    kept.append(row)
                selection["augmentation_recipes"] = kept
                if body.get("delete_generated") and removed_ids:
                    from sed.data.augment import delete_generated_for_recipes
                    deleted_generated = delete_generated_for_recipes(
                        project_id,
                        removed_ids,
                        frozen_test=None,
                    )
            elif action == "clear":
                removed_ids = [
                    str(row.get("recipe_id") or row.get("id") or "")
                    for row in recipes
                    if isinstance(row, dict)
                ]
                removed = len(recipes)
                selection["augmentation_recipes"] = []
                if body.get("delete_generated") and removed_ids:
                    from sed.data.augment import delete_generated_for_recipes
                    deleted_generated = delete_generated_for_recipes(
                        project_id,
                        removed_ids,
                        frozen_test=None,
                    )
            else:
                self._send(400, b"unknown action")
                return

            selection["updated_at"] = _now_iso()
            path = write_dataset_selection(layout, selection)
            self._send_json({
                "ok": True,
                "selection": selection,
                "path": str(path),
                "added": added,
                "removed": removed,
                "deleted_generated": deleted_generated,
            })

        def _augment_job_payload(self, job: dict | None) -> dict:
            """Shared envelope plus the generation rate the UI shows."""
            envelope = _job_status_payload(job, missing="no augmentation job")
            if not envelope["ok"]:
                return envelope
            payload = envelope["job"]
            elapsed = float(payload["elapsed_s"])
            completed = int(payload.get("completed") or 0)
            payload["clips_per_sec"] = (completed / elapsed) if elapsed > 0 else 0.0
            return envelope

        def _handle_augment_status(self, job_id: str = "") -> None:
            self._send_job_status(
                augment_lock, lambda: augment_job, job_id, lambda job: self._augment_job_payload(job),
            )

        def _handle_augment_cancel(self) -> None:
            with augment_lock:
                job = augment_job
                if not job or str(job.get("state")) not in {"running", "starting"}:
                    self._send_json({"ok": False, "error": "no running augmentation job"}, status=404)
                    return
                augment_cancel_event.set()
                job["state"] = "canceling"
                payload = self._augment_job_payload(job)
            self._send_json(payload)

        def _handle_augment_start(self) -> None:
            nonlocal augment_job
            if cfg is None:
                self._send(400, b"missing config")
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            from sed.data.compose import load_dataset_selection
            from sed.data.augment import (
                AugmentationCancelled,
                AugmentationDisabled,
                materialize_library_recipes,
            )

            selection = load_dataset_selection(layout)
            all_recipes = selection.get("augmentation_recipes", [])
            if not isinstance(all_recipes, list) or not all_recipes:
                self._send_json({"ok": False, "error": "No UI augmentation recipes saved."}, status=400)
                return

            # Select recipes by id. New shape: recipe_ids: [...]; legacy single
            # recipe_id is still accepted. If neither is provided, fall back to
            # the persisted per-recipe "selected" flag (default true) so old
            # clients keep working.
            raw_ids = body.get("recipe_ids")
            recipe_ids: list[str] = []
            if isinstance(raw_ids, list):
                recipe_ids = [str(rid).strip() for rid in raw_ids if str(rid).strip()]
            legacy_id = str(body.get("recipe_id") or "").strip()
            if not recipe_ids and legacy_id:
                recipe_ids = [legacy_id]

            def _row_id(row: dict) -> str:
                return str(row.get("recipe_id") or row.get("id") or "").strip()

            if recipe_ids:
                wanted = set(recipe_ids)
                recipes = [row for row in all_recipes if isinstance(row, dict) and _row_id(row) in wanted]
            else:
                # Honor the per-recipe `selected` flag when the client didn't
                # explicitly select. Default True keeps prior behavior of
                # "generate everything" when the field is absent.
                recipes = [
                    row for row in all_recipes
                    if isinstance(row, dict) and bool(row.get("selected", True))
                ]

            if not recipes:
                self._send_json(
                    {"ok": False, "error": "No recipes selected for generation."}, status=400,
                )
                return
            with augment_lock:
                if augment_job and str(augment_job.get("state")) in {"starting", "running", "canceling"}:
                    self._send_json({"ok": False, "error": "augmentation job already running"}, status=409)
                    return
                augment_cancel_event.clear()
                job_id = uuid.uuid4().hex[:12]
                generation_id = uuid.uuid4().hex[:12]
                augment_job = {
                    "job_id": job_id,
                    "generation_id": generation_id,
                    "state": "starting",
                    "total": 0,
                    "completed": 0,
                    "generated": 0,
                    "skipped": 0,
                    "failed": 0,
                    "current_recipe": "",
                    "counts": {},
                    "warnings": [],
                    "error": "",
                    "started_at": time.time(),
                    "ended_at": None,
                }

            def progress(update: dict) -> None:
                with augment_lock:
                    if not augment_job or augment_job.get("job_id") != job_id:
                        return
                    for key in (
                        "state", "total", "completed", "generated", "skipped",
                        "failed", "current_recipe", "generation_id", "warnings",
                    ):
                        if key in update:
                            augment_job[key] = update[key]

            def run_job() -> None:
                nonlocal augment_job
                run_cfg = cfg.model_copy(deep=True)
                _apply_augmentation_settings(run_cfg, selection.get("augmentation_settings", {}))
                try:
                    counts = materialize_library_recipes(
                        run_cfg,
                        project_id,
                        recipes,
                        seed=int(body["seed"]) if body.get("seed") not in (None, "") else None,
                        generation_id=generation_id,
                        progress=progress,
                        should_cancel=augment_cancel_event.is_set,
                    )
                    with augment_lock:
                        if augment_job and augment_job.get("job_id") == job_id:
                            augment_job["counts"] = counts
                            augment_job["state"] = "finished"
                            augment_job["ended_at"] = time.time()
                except AugmentationCancelled:
                    with augment_lock:
                        if augment_job and augment_job.get("job_id") == job_id:
                            augment_job["state"] = "canceled"
                            augment_job["ended_at"] = time.time()
                except AugmentationDisabled as exc:
                    log.warning("augmentation disabled: %s", exc)
                    with augment_lock:
                        if augment_job and augment_job.get("job_id") == job_id:
                            augment_job["state"] = "error"
                            augment_job["error"] = str(exc)
                            augment_job["ended_at"] = time.time()
                except Exception as exc:  # noqa: BLE001
                    log.exception("background UI augmentation failed")
                    with augment_lock:
                        if augment_job and augment_job.get("job_id") == job_id:
                            augment_job["state"] = "error"
                            augment_job["error"] = str(exc)
                            augment_job["ended_at"] = time.time()

            thread = threading.Thread(target=run_job, daemon=True)
            with augment_lock:
                # augment_job was set unconditionally above inside this same
                # critical section, so it's always truthy here. Keep the lock
                # held across thread.start() so the "running" state is visible
                # before the worker can change it.
                augment_job["thread"] = thread
                augment_job["state"] = "running"
                payload = self._augment_job_payload(augment_job)
                thread.start()
            self._send_json(payload)

        # ---- shared assigned-preview cache ---------------------------------

        def _preview_cache_job_payload(self, job: dict | None) -> dict:
            """Shared envelope plus rate and an ETA extrapolated from clip counts."""
            envelope = _job_status_payload(job, missing="no preview cache job")
            if not envelope["ok"]:
                return envelope
            payload = envelope["job"]
            elapsed = float(payload["elapsed_s"])
            completed = int(payload.get("completed") or 0)
            total = int(payload.get("total") or 0)
            payload["clips_per_sec"] = completed / elapsed if elapsed > 0 else 0.0
            if 0 < completed < total and payload["clips_per_sec"] > 0:
                payload["eta_s"] = max(0.0, (total - completed) / payload["clips_per_sec"])
            elif str(payload.get("state") or "") == "finished":
                payload["eta_s"] = 0.0
            else:
                payload["eta_s"] = None
            return envelope

        def _handle_preview_cache_status(self, job_id: str = "") -> None:
            self._send_job_status(
                preview_cache_lock, lambda: preview_cache_job, job_id, lambda job: self._preview_cache_job_payload(job),
            )

        def _handle_preview_cache_list(self) -> None:
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            try:
                from sed.data.presets import preview_features_dict
                from sed.data.preview_cache import preview_signature

                cache, _workers = preview_service()
                features = preview_features_dict(cfg, layout)
                active_signature = preview_signature(features)
                variants = cache.variants_for_project(project_id)
                with preview_cache_lock:
                    assigned_total = int((preview_cache_job or {}).get("total") or 0)
                self._send_json({
                    "ok": True,
                    "active_signature": active_signature,
                    "assigned_total": assigned_total,
                    "variants": variants,
                })
            except Exception as exc:  # noqa: BLE001
                log.exception("preview cache list failed")
                self._send_json({"ok": False, "error": str(exc)}, status=500)

        def _handle_preview_cache_cancel(self) -> None:
            with preview_cache_lock:
                job = preview_cache_job
                if not job or str(job.get("state")) not in {"starting", "running", "canceling"}:
                    self._send_json({"ok": False, "error": "no running preview cache build"}, status=404)
                    return
                preview_cache_cancel_event.set()
                job["state"] = "canceling"
                payload = self._preview_cache_job_payload(job)
            self._send_json(payload)

        def _handle_preview_cache_delete(self) -> None:
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            signature = str(body.get("signature") or "").strip()
            if not re.fullmatch(r"[0-9a-f]{20}", signature):
                self._send_json({"ok": False, "error": "invalid preview variant"}, status=400)
                return
            with preview_cache_lock:
                if preview_cache_job and str(preview_cache_job.get("state")) in {"starting", "running", "canceling"}:
                    self._send_json({"ok": False, "error": "cancel the active preview build first"}, status=409)
                    return
            try:
                removed = preview_service()[0].delete_project_variant(project_id, signature)
                self._send_json({"ok": True, "removed": removed})
            except Exception as exc:  # noqa: BLE001
                log.exception("preview cache variant delete failed")
                self._send_json({"ok": False, "error": str(exc)}, status=500)

        def _handle_preview_cache_start(self) -> None:
            nonlocal preview_cache_job
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400)
                return
            try:
                # Accept a body for future scope options, while keeping the
                # initial API intentionally single-purpose and bounded.
                if int(self.headers.get("Content-Length", 0) or 0):
                    self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            try:
                from sed.data.presets import preview_feature_parts, preview_features_dict
                from sed.data.preview_cache import preview_signature

                features = preview_features_dict(cfg, layout)
                preview_features_cfg, _sr = preview_feature_parts(cfg, layout)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            label = str(getattr(preview_features_cfg, "preset", "") or "current feature")
            with preview_cache_lock:
                if preview_cache_job and str(preview_cache_job.get("state")) in {"starting", "running", "canceling"}:
                    self._send_json({"ok": False, "error": "preview cache build already running"}, status=409)
                    return
                preview_cache_cancel_event.clear()
                job_id = uuid.uuid4().hex[:12]
                preview_cache_job = {
                    "job_id": job_id,
                    "state": "starting",
                    "scope": "assigned",
                    "signature": preview_signature(features),
                    "label": label,
                    "phase": "scanning assigned clips",
                    "total": 0,
                    "completed": 0,
                    "generated": 0,
                    "migrated": 0,
                    "skipped": 0,
                    "failed": 0,
                    "current_name": "",
                    "error": "",
                    "started_at": time.time(),
                    "ended_at": None,
                }

            def update(**values) -> bool:
                with preview_cache_lock:
                    if not preview_cache_job or preview_cache_job.get("job_id") != job_id:
                        return False
                    preview_cache_job.update(values)
                    return True

            def run_job() -> None:
                try:
                    cache, workers = preview_service()
                    update(phase="cleaning orphaned cache entries")
                    cache.clean_orphans()
                    update(phase="scanning assigned clips")
                    from sed.data.inspect import assigned_preview_paths

                    paths = assigned_preview_paths(cfg, layout)
                    if not paths:
                        update(
                            state="finished", phase="finished", total=0,
                            current_name="", error="no assigned Dataset, Augmented, or Frozen-test clips",
                            ended_at=time.time(),
                        )
                        return
                    update(total=len(paths), phase="building previews")
                    pending: dict[object, Path] = {}
                    next_index = 0
                    # A small producer buffer keeps two workers busy without
                    # placing tens of thousands of queued closures in memory.
                    pending_limit = 12
                    while next_index < len(paths) or pending:
                        if preview_cache_cancel_event.is_set():
                            for future in pending:
                                future.cancel()
                            update(state="canceled", phase="canceled", ended_at=time.time(), current_name="")
                            return
                        while next_index < len(paths) and len(pending) < pending_limit:
                            wav = paths[next_index]
                            next_index += 1
                            update(state="running", current_name=wav.name)
                            migrated = cache.migrate_legacy(
                                wav, features, project_id=project_id, label=label,
                            )
                            if migrated is not None:
                                with preview_cache_lock:
                                    if preview_cache_job and preview_cache_job.get("job_id") == job_id:
                                        preview_cache_job["completed"] += 1
                                        preview_cache_job["migrated"] += 1
                                continue
                            future = workers.submit(
                                wav, features, project_id=project_id, label=label, priority=10,
                            )
                            pending[future] = wav
                        if not pending:
                            continue
                        done, _waiting = wait(
                            list(pending), timeout=0.25, return_when=FIRST_COMPLETED,
                        )
                        for future in done:
                            wav = pending.pop(future)
                            try:
                                result = future.result()
                                with preview_cache_lock:
                                    if preview_cache_job and preview_cache_job.get("job_id") == job_id:
                                        preview_cache_job["completed"] += 1
                                        preview_cache_job["generated" if result.created else "skipped"] += 1
                                        preview_cache_job["current_name"] = wav.name
                            except Exception as exc:  # noqa: BLE001
                                log.warning("preview cache failed for %s: %s", wav, exc)
                                with preview_cache_lock:
                                    if preview_cache_job and preview_cache_job.get("job_id") == job_id:
                                        preview_cache_job["completed"] += 1
                                        preview_cache_job["failed"] += 1
                                        preview_cache_job["current_name"] = wav.name
                    update(state="finished", phase="finished", ended_at=time.time(), current_name="")
                except Exception as exc:  # noqa: BLE001
                    log.exception("assigned preview cache build failed")
                    update(state="error", phase="error", error=str(exc), ended_at=time.time(), current_name="")

            thread = threading.Thread(target=run_job, name="sed-preview-cache-build", daemon=True)
            with preview_cache_lock:
                preview_cache_job["thread"] = thread
                preview_cache_job["state"] = "running"
                payload = self._preview_cache_job_payload(preview_cache_job)
                thread.start()
            self._send_json(payload)

        def _pretraining_rescan_payload(self, job: dict | None) -> dict:
            if not job:
                return {"ok": False, "error": "no pretraining rescan"}
            payload = {k: v for k, v in job.items() if k not in {"thread", "done_event"}}
            started = float(payload.get("started_at") or time.time())
            ended = payload.get("ended_at")
            payload["elapsed_s"] = max(0.0, float(ended or time.time()) - started)
            return {"ok": True, "job": payload, **{
                k: payload[k] for k in ("pretraining_counts", "subbuckets", "total", "classes") if k in payload
            }}

        def _handle_pretraining_rescan_status(self, job_id: str = "") -> None:
            self._send_job_status(
                pretraining_rescan_lock, lambda: pretraining_rescan_job, job_id, lambda job: self._pretraining_rescan_payload(job),
            )

        def _handle_pretraining_rescan(self) -> None:
            """Track the corpus header walk as a cancellable-style activity job.

            The index implementation currently exposes one rebuild operation, so
            progress is phase-based rather than per-file. Small rescans are
            allowed to finish before the response for compatibility with older
            clients; larger walks return immediately and are polled by the UI.
            """
            nonlocal pretraining_rescan_job
            with pretraining_rescan_lock:
                if pretraining_rescan_job and str(pretraining_rescan_job.get("state")) in {"starting", "running"}:
                    self._send_json({"ok": False, "error": "pretraining rescan already running"}, status=409)
                    return
                job_id = uuid.uuid4().hex[:12]
                done_event = threading.Event()
                pretraining_rescan_job = {
                    "job_id": job_id, "state": "starting", "phase": "starting",
                    "phase_detail": "preparing corpus index", "progress": 0.0,
                    "started_at": time.time(), "ended_at": None, "done_event": done_event,
                }

            def run_job() -> None:
                try:
                    with pretraining_rescan_lock:
                        if pretraining_rescan_job and pretraining_rescan_job.get("job_id") == job_id:
                            pretraining_rescan_job.update({"state": "running", "phase": "headers", "phase_detail": "reading corpus WAV headers", "progress": 0.1})
                    from sed.data.pretraining import PretrainIndex
                    index = PretrainIndex()
                    index.rebuild()
                    counts = index.labels_with_counts()
                    subbuckets = self._pretraining_subbucket_rows(index)
                    with pretraining_rescan_lock:
                        if pretraining_rescan_job and pretraining_rescan_job.get("job_id") == job_id:
                            pretraining_rescan_job.update({
                                "state": "finished", "phase": "finished", "phase_detail": "index refreshed", "progress": 1.0,
                                "pretraining_counts": counts, "subbuckets": subbuckets,
                                "total": sum(counts.values()), "classes": len(counts), "ended_at": time.time(),
                            })
                except Exception as exc:  # noqa: BLE001
                    log.exception("pretraining rescan failed")
                    with pretraining_rescan_lock:
                        if pretraining_rescan_job and pretraining_rescan_job.get("job_id") == job_id:
                            pretraining_rescan_job.update({"state": "error", "phase": "error", "phase_detail": str(exc), "error": str(exc), "ended_at": time.time()})
                finally:
                    done_event.set()

            thread = threading.Thread(target=run_job, name="sed-pretraining-rescan", daemon=True)
            with pretraining_rescan_lock:
                pretraining_rescan_job["thread"] = thread
                thread.start()
            # Preserve the old immediate response for quick index walks while
            # never holding the request open for a large corpus.
            done_event.wait(0.05)
            with pretraining_rescan_lock:
                payload = self._pretraining_rescan_payload(pretraining_rescan_job)
            self._send_json(payload)

        @staticmethod
        def _pretraining_subbucket_rows(index, *, rebuild_if_missing: bool = True) -> list:
            stats = index.subbuckets_with_stats(rebuild_if_missing=rebuild_if_missing)
            rows = sorted(stats.values(), key=lambda s: (s["origin_class"], -int(s["sample_rate"] or 0)))
            for s in rows:
                s.pop("duration_total", None)
            return rows

        def _handle_pretraining_counts(self) -> None:
            """Corpus label counts from the on-disk index only (page-load safe).

            Never walks/probes the corpus — with a 50k+ clip library that scan
            belongs exclusively to the explicit POST /api/pretraining/rescan.
            """
            try:
                from sed.data.pretraining import PretrainIndex
                index = PretrainIndex()
                counts = index.labels_with_counts(rebuild_if_missing=False)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json({
                "ok": True,
                "pretraining_counts": counts,
                "total": sum(counts.values()),
                "index_present": index.index_path.exists(),
            })

        def _handle_pretraining_subbuckets(self) -> None:
            """Per-subbucket corpus stats + which target class each is assigned to.

            Reads the on-disk index only (no corpus walk) — this GET fires on
            page load for every project in the unified UI.
            """
            try:
                from sed.data.pretraining import PretrainIndex
                rows = self._pretraining_subbucket_rows(PretrainIndex(), rebuild_if_missing=False)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500); return
            # Map (origin_class, sample_rate) -> assigned target class(es).
            assigned: dict[str, list[str]] = {}
            try:
                from sed.data.compose import load_dataset_selection
                sel = load_dataset_selection(layout)
                for cls, rules in (sel.get("assignments") or {}).items():
                    for rule in rules:
                        if str(rule.get("source_bucket")) != "pretraining":
                            continue
                        for lbl in (rule.get("labels") or []):
                            key = f"{lbl}@{int(rule.get('sample_rate') or 0)}"
                            assigned.setdefault(key, []).append(cls)
            except Exception:  # noqa: BLE001
                pass
            for s in rows:
                key = f"{s['origin_class']}@{int(s['sample_rate'] or 0)}"
                key_any = f"{s['origin_class']}@0"  # rules not pinned to a rate
                names = list(assigned.get(key, []))
                if key_any != key:
                    names += assigned.get(key_any, [])
                s["assigned_to"] = sorted(dict.fromkeys(names))
            self._send_json({
                "ok": True,
                "subbuckets": rows,
                "classes": list(cfg.model.classes) if cfg else [],
            })

        def _handle_pretraining_assign(self) -> None:
            """Assign (or unassign) one corpus subbucket to a target class as a
            pretraining selection rule (source_bucket=pretraining, pinned to the
            subbucket's sample rate)."""
            if cfg is None:
                self._send_json({"ok": False, "error": "missing config"}, status=400); return
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON"); return
            action = str(body.get("action") or "add").strip().lower()
            class_name = str(body.get("class_name") or "").strip()
            origin_class = str(body.get("origin_class") or "").strip()
            try:
                sample_rate = int(body.get("sample_rate") or 0)
            except (TypeError, ValueError):
                sample_rate = 0
            if class_name not in (cfg.model.classes or []):
                self._send_json({"ok": False, "error": f"unknown target class {class_name!r}"}, status=400); return
            if not origin_class:
                self._send_json({"ok": False, "error": "missing origin_class"}, status=400); return
            try:
                from sed.data.compose import load_dataset_selection, write_dataset_selection
                sel = load_dataset_selection(layout)
                assigns = sel.setdefault("assignments", {})
                rules = assigns.setdefault(class_name, [])

                def _is_match(r: dict) -> bool:
                    return (str(r.get("source_bucket")) == "pretraining"
                            and origin_class in (r.get("labels") or [])
                            and int(r.get("sample_rate") or 0) == sample_rate)

                if action == "remove":
                    assigns[class_name] = [r for r in rules if not _is_match(r)]
                else:
                    if not any(_is_match(r) for r in rules):
                        rules.append({
                            "id": uuid.uuid4().hex[:12],
                            "source_bucket": "pretraining",
                            "labels": [origin_class],
                            "sample_rate": sample_rate,
                            "subbucket": str(body.get("subbucket") or ""),
                            "updated_at": datetime.now(UTC).isoformat(),
                        })
                write_dataset_selection(layout, sel)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": f"assign failed: {exc}"}, status=500); return
            self._send_json({"ok": True, "class_name": class_name,
                             "origin_class": origin_class, "sample_rate": sample_rate})

        # ---- audio-feature presets / lab ------------------------------------

        def _handle_feature_presets_get(self) -> None:
            """List saved feature presets + backbone registry for the lab."""
            try:
                from sed.config import backbone_registry_payload
                from sed.data.compose import load_dataset_selection
                from sed.data.inspect import default_feature_preview_clip
                from sed.data.presets import (
                    DATASET_PRESET_KEY,
                    DATASET_PREVIEW_KEY,
                    feature_source_for_config,
                    list_feature_presets,
                )
                presets = list_feature_presets()
                active = ""
                preview_active = ""
                try:
                    selection = load_dataset_selection(layout)
                    active = str(selection.get(DATASET_PRESET_KEY) or "")
                    preview_active = str(selection.get(DATASET_PREVIEW_KEY) or "")
                except Exception:  # noqa: BLE001 — active marker is advisory only
                    active = ""
                current = None
                current_source = {"kind": "custom", "name": "", "label": "Custom audio feature",
                                  "base_kind": "", "base_name": ""}
                contract = None
                if cfg is not None:
                    from sed.data.presets import feature_cfg_to_preset, model_default_feature_preset
                    current = feature_cfg_to_preset(cfg, name=cfg.features.preset)
                    current_source = feature_source_for_config(cfg, project_id=project_id)
                    # A virtual default is always a complete payload.  This
                    # keeps the visible Model-tab handoff loadable even before
                    # the Audio Features endpoint has refreshed.
                    contract = model_default_feature_preset(cfg, project_id=project_id)
            except Exception as exc:  # noqa: BLE001
                log.exception("feature-presets list failed")
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json({
                "ok": True,
                "presets": presets,
                "backbones": backbone_registry_payload(),
                "feature_types": ["mel", "linear"],
                "active_preset": active,
                "preview_preset": preview_active,
                "current_source": current_source,
                "current": current,
                "contract": contract,
                "model_default": contract,
                "default_clip": str(default_feature_preview_clip()),
            })

        def _handle_feature_preview(self, qs: dict) -> None:
            """Render a large live spectrogram PNG for the requested settings."""
            _first, _int, _float, _bool = _query_readers(qs)

            raw_path = _first("path").strip()
            if raw_path:
                resolved = _safe_resolve_audio_path(layout, raw_path)
                if resolved is None:
                    self._send(403, b"forbidden"); return
            else:
                from sed.data.inspect import default_feature_preview_clip
                resolved = default_feature_preview_clip()
            if not resolved.exists():
                self._send(404, b"sample clip not found"); return
            try:
                from sed.data.inspect import render_feature_preview_png
                png = render_feature_preview_png(
                    resolved,
                    feature_type=_first("feature_type", "mel").strip().lower() or "mel",
                    frontend=_first("frontend", "efficientat").strip().lower() or "efficientat",
                    sample_rate=_int("sample_rate", 32000),
                    n_fft=_int("n_fft", 1024),
                    win_length=_int("win_length", 800),
                    hop_length=_int("hop_length", 320),
                    n_mels=_int("n_mels", 128),
                    fmin=_int("fmin", 0),
                    fmax=_int("fmax", 15000),
                    window_seconds=float(_first("window_seconds", "1.0") or 1.0),
                    log_power=_bool("log_power", True),
                    stft_engine=_first("stft_engine", "librosa").strip().lower() or "librosa",
                    db_norm=_bool("db_norm", False),
                    db_min=_float("db_min", -20.0),
                    db_max=_float("db_max", 60.0),
                    colormap=_first("colormap", "none").strip().lower() or "none",
                    image_size=_int("image_size", 96),
                    pcen=_bool("pcen", False),
                    pcen_params={
                        "time_constant": _float("pcen_time_constant", 0.4),
                        "gain": _float("pcen_gain", 0.98),
                        "bias": _float("pcen_bias", 2.0),
                        "power": _float("pcen_power", 0.5),
                        "eps": _float("pcen_eps", 1e-6),
                    },
                    compression=_first("compression", "").strip().lower() or None,
                    standardize=_bool("standardize", False),
                    standardize_mean=_float("standardize_mean", 0.0),
                    standardize_std=_float("standardize_std", 1.0),
                    scaling_method=_first("scaling_method", "none").strip().lower() or "none",
                    scaling_scope=_first("scaling_scope", "global").strip().lower() or "global",
                    scaling_source=_first("scaling_source", "dataset").strip().lower() or "dataset",
                    scaling_stats=_parse_scaling_stats_param(_first("scaling_stats", "")),
                )
            except Exception as exc:  # noqa: BLE001 — bad params shouldn't 500 the lab
                log.warning("feature preview render failed: %s", exc)
                self._send(400, f"preview failed: {exc}".encode("utf-8")); return
            self._send(200, png, "image/png")

        def _handle_feature_audio(self, qs: dict) -> None:
            """Render the lab's current listening configuration as WAV: the
            sample window, optionally mixed with the overlay at the given SNR,
            optionally band-limited to the selected Hz ranges."""
            _first, _int, _float, _bool = _query_readers(qs)

            raw_path = _first("path").strip()
            if raw_path:
                resolved = _safe_resolve_audio_path(layout, raw_path)
                if resolved is None:
                    self._send(403, b"forbidden"); return
            else:
                from sed.data.inspect import default_feature_preview_clip
                resolved = default_feature_preview_clip()
            if not resolved.exists():
                self._send(404, b"sample clip not found"); return
            # A bad/forbidden/missing overlay must not fail playback — the clean
            # sample still plays (mirrors _handle_feature_analysis).
            overlay_path = None
            raw_overlay = _first("overlay").strip()
            if raw_overlay:
                ro = _safe_resolve_audio_path(layout, raw_overlay)
                if ro is not None and ro.exists():
                    overlay_path = str(ro)
            sample_rate = _int("sample_rate", 32000)
            bands: list[tuple[float, float]] = []
            for part in _first("bands").split(","):
                lo, sep, hi = part.strip().partition("-")
                try:
                    lo_hz, hi_hz = float(lo), float(hi)
                except (TypeError, ValueError):
                    continue
                if not sep or hi_hz <= lo_hz:
                    continue
                bands.append((max(0.0, lo_hz), min(float(sample_rate) / 2.0, hi_hz)))
            try:
                from sed.data.inspect import render_feature_audio
                wav = render_feature_audio(
                    resolved,
                    sample_rate=sample_rate,
                    window_seconds=_float("window_seconds", 1.0),
                    n_fft=_int("n_fft", 1024),
                    win_length=_int("win_length", 800),
                    hop_length=_int("hop_length", 320),
                    overlay_path=overlay_path,
                    snr_db=_float("snr", 0.0),
                    bands=bands or None,
                )
            except Exception as exc:  # noqa: BLE001 — bad params shouldn't 500 the lab
                log.warning("feature audio render failed: %s", exc)
                self._send(400, f"audio failed: {exc}".encode("utf-8")); return
            self._send(200, wav, "audio/wav")

        def _handle_feature_analysis(self, qs: dict) -> None:
            """Interactive analysis (matrix + metrics + transient) for 1..N clips."""
            _first, _int, _float, _bool = _query_readers(qs)

            from sed.data.inspect import analyze_feature_clips, default_feature_preview_clip

            raw_paths = [p for p in qs.get("path", []) if p and p.strip()]
            resolved: list = []
            for raw in raw_paths:
                r = _safe_resolve_audio_path(layout, raw.strip())
                if r is None:
                    self._send(403, b"forbidden"); return
                if r.exists():
                    resolved.append(r)
            if not resolved:
                default = default_feature_preview_clip()
                if not default.exists():
                    self._send(404, b"no clips to analyze"); return
                resolved = [default]
            # Optional noise-overlay clip → per-band survival at the given SNR.
            # A bad/forbidden/missing overlay must NOT fail the whole analysis —
            # just skip the overlay so the target still renders (the UI then shows
            # "overlay clip not found" rather than freezing).
            overlay_path = None
            raw_overlay = _first("overlay").strip()
            if raw_overlay:
                ro = _safe_resolve_audio_path(layout, raw_overlay)
                if ro is not None and ro.exists():
                    overlay_path = str(ro)
            try:
                result = analyze_feature_clips(
                    resolved,
                    feature_type=_first("feature_type", "mel").strip().lower() or "mel",
                    frontend=_first("frontend", "efficientat").strip().lower() or "efficientat",
                    sample_rate=_int("sample_rate", 32000),
                    n_fft=_int("n_fft", 1024),
                    win_length=_int("win_length", 800),
                    hop_length=_int("hop_length", 320),
                    n_mels=_int("n_mels", 128),
                    fmin=_int("fmin", 0),
                    fmax=_int("fmax", 15000),
                    window_seconds=_float("window_seconds", 1.0),
                    log_power=_bool("log_power", True),
                    stft_engine=_first("stft_engine", "librosa").strip().lower() or "librosa",
                    db_norm=_bool("db_norm", False),
                    db_min=_float("db_min", -20.0),
                    db_max=_float("db_max", 60.0),
                    pcen=_bool("pcen", False),
                    pcen_params={
                        "time_constant": _float("pcen_time_constant", 0.4),
                        "gain": _float("pcen_gain", 0.98),
                        "bias": _float("pcen_bias", 2.0),
                        "power": _float("pcen_power", 0.5),
                        "eps": _float("pcen_eps", 1e-6),
                    },
                    standardize=_bool("standardize", False),
                    standardize_mean=_float("standardize_mean", 0.0),
                    standardize_std=_float("standardize_std", 1.0),
                    compression=_first("compression", "").strip().lower() or None,
                    scaling_method=_first("scaling_method", "none").strip().lower() or "none",
                    scaling_scope=_first("scaling_scope", "global").strip().lower() or "global",
                    scaling_source=_first("scaling_source", "dataset").strip().lower() or "dataset",
                    scaling_stats=_parse_scaling_stats_param(_first("scaling_stats", "")),
                    overlay_path=overlay_path,
                    snr_db=_float("snr", 0.0),
                    transient_window_ms=_int("transient_window_ms", 100),
                    transient_bands=_int("transient_bands", 10),
                    fingerprint_peaks=_int("fingerprint_peaks", 6),
                    fingerprint_prominence_db=_float("fingerprint_prominence_db", 6.0),
                    onset_start_ms=(_float("onset_start_ms", 0.0)
                                    if _first("onset_start_ms").strip() else None),
                    onset_end_ms=(_float("onset_end_ms", 0.0)
                                  if _first("onset_end_ms").strip() else None),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("feature analysis failed: %s", exc)
                self._send(400, f"analysis failed: {exc}".encode("utf-8")); return
            self._send_json(result)

        def _handle_feature_stats_estimate(self, qs: dict) -> None:
            """Fit dataset scaling statistics on training-only groups.

            Every readable row on the training side of the live grouped split
            contributes. Validation and frozen-test rows never do. Batched
            extractors and streaming Welford moments avoid retaining the full
            feature matrix; progress is exposed through the shared activity bar.
            """
            nonlocal feature_stats_job
            _first, _int, _float, _bool = _query_readers(qs)

            from types import SimpleNamespace

            from sed.data.feature_stats import fit_feature_scaling

            features = SimpleNamespace(
                feature_type=_first("feature_type", "mel").strip().lower() or "mel",
                frontend=_first("frontend", "efficientat").strip().lower() or "efficientat",
                n_fft=_int("n_fft", 1024),
                win_length=_int("win_length", 800),
                hop_length=_int("hop_length", 320),
                n_mels=_int("n_mels", 128),
                fmin=_int("fmin", 0),
                fmax=_int("fmax", 15000),
                window_seconds=_float("window_seconds", 1.0),
                log_power=_first("log_power", "1") in {"1", "true", "yes", "on"},
                stft_engine=_first("stft_engine", "librosa").strip().lower() or "librosa",
                db_norm=_first("db_norm", "0") in {"1", "true", "yes", "on"},
                pcen=_first("pcen", "0") in {"1", "true", "yes", "on"},
                pcen_time_constant=_float("pcen_time_constant", 0.4),
                pcen_gain=_float("pcen_gain", 0.98),
                pcen_bias=_float("pcen_bias", 2.0),
                pcen_power=_float("pcen_power", 0.5),
                pcen_eps=_float("pcen_eps", 1e-6),
                compression=_first("compression", "").strip().lower(),
                scaling_method=_first("scaling_method", "zscore").strip().lower() or "zscore",
                scaling_scope=_first("scaling_scope", "global").strip().lower() or "global",
                scaling_source=_first("scaling_source", "dataset").strip().lower() or "dataset",
                colormap=_first("colormap", "none").strip().lower() or "none",
            )
            with feature_stats_lock:
                if feature_stats_job and str(feature_stats_job.get("state")) in {"starting", "running"}:
                    self._send_json({"ok": False, "error": "feature statistics fit already running"}, status=409)
                    return
                feature_stats_job = {
                    "job_id": uuid.uuid4().hex[:12], "state": "starting",
                    "phase": "preparing", "completed": 0, "total": 0,
                    "groups": 0, "started_at": time.time(), "ended_at": None,
                }

            def progress(update: dict) -> None:
                with feature_stats_lock:
                    if not feature_stats_job:
                        return
                    feature_stats_job.update({
                        "state": "running",
                        "phase": str(update.get("phase") or "fitting"),
                        "completed": int(update.get("completed") or 0),
                        "total": int(update.get("total") or 0),
                        "groups": int(update.get("groups") or 0),
                    })
            try:
                result = fit_feature_scaling(
                    cfg,
                    layout,
                    features=features,
                    sample_rate=_int("sample_rate", 32000),
                    progress=progress,
                )
            except Exception as exc:  # noqa: BLE001
                with feature_stats_lock:
                    if feature_stats_job:
                        feature_stats_job.update({
                            "state": "error", "error": str(exc), "ended_at": time.time(),
                        })
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            with feature_stats_lock:
                if feature_stats_job:
                    feature_stats_job.update({
                        "state": "finished", "phase": "finished",
                        "completed": int(result.get("clips") or 0),
                        "total": int(result.get("training_rows") or result.get("clips") or 0),
                        "groups": int(result.get("training_groups") or 0),
                        "cache_hit": bool(result.get("cache_hit")),
                        "ended_at": time.time(),
                    })
            self._send_json({"ok": True, **result})

        def _handle_feature_model_fit(self, qs: dict) -> None:
            """Analytical 'how your feature fits the model' report for the Align panel."""
            _first, _int, _float, _bool = _query_readers(qs)

            from types import SimpleNamespace

            from sed.audio.features import model_fit_report

            sample_rate = _int("sample_rate", 32000)
            feats = SimpleNamespace(
                feature_type=_first("feature_type", "mel").strip().lower() or "mel",
                frontend=_first("frontend", "efficientat").strip().lower() or "efficientat",
                n_fft=_int("n_fft", 1024),
                win_length=_int("win_length", 800),
                hop_length=_int("hop_length", 320),
                n_mels=_int("n_mels", 128),
                fmin=_int("fmin", 0),
                fmax=_int("fmax", 15000),
                window_seconds=float(_first("window_seconds", "1.0") or 1.0),
                pcen=_bool("pcen", False),
                stft_engine=_first("stft_engine", "librosa").strip().lower() or "librosa",
                colormap=_first("colormap", "none").strip().lower() or "none",
                image_size=_int("image_size", 96),
            )
            model_params: dict = {}
            raw_params = _first("params", "").strip()
            if raw_params:
                try:
                    parsed = json.loads(raw_params)
                    if not isinstance(parsed, dict):
                        raise ValueError("model params must be a JSON object")
                    model_params = parsed
                except Exception as exc:  # noqa: BLE001
                    self._send(
                        400,
                        f"model-fit failed: invalid params JSON: {exc}".encode("utf-8"),
                    )
                    return
            try:
                backbone = _first("backbone", "mn10_as").strip() or "mn10_as"
                from sed.train.model_lab import (
                    _infer_architecture_id,
                    architecture_by_id,
                    validate_recipe_params,
                )

                architecture_id = _infer_architecture_id(backbone, "", project_id)
                if not architecture_id.startswith("legacy:"):
                    architecture = architecture_by_id(architecture_id, project_id)
                    model_params = validate_recipe_params(architecture, model_params)
                    if (
                        architecture.provider == "efficientat"
                        and _bool("pretrained", False)
                        and model_params
                    ):
                        raise ValueError(
                            "EfficientAT pretrained checkpoints use fixed constructor defaults; "
                            "switch to scratch initialization to customize"
                        )
                report = model_fit_report(
                    feats,
                    backbone,
                    _bool("pretrained", False),
                    sample_rate,
                    _first("task", "clip").strip().lower() or "clip",
                    params=model_params,
                    weight_id=_first("weight_id", "").strip(),
                )
                if backbone == "clink-cnn-v2":
                    shape = report.get("input_shape") or []
                    if len(shape) == 4:
                        gmac = _clink_cnn_v2_gmac(
                            int(shape[1]), int(shape[2]), int(shape[3]),
                        )
                        if gmac is not None:
                            report["gmac_per_clip"] = gmac
                            report["gmac_budget_per_clip"] = _CLINK_CNN_V2_GMAC_BUDGET
                        compute_verdict = _clink_cnn_v2_compute_verdict(gmac)
                        if compute_verdict is not None:
                            report.setdefault("verdicts", []).append(compute_verdict)
            except Exception as exc:  # noqa: BLE001
                log.warning("model-fit report failed: %s", exc)
                self._send(400, f"model-fit failed: {exc}".encode("utf-8")); return
            self._send_json({"ok": True, "fit": report})

        def _handle_feature_preset_save(self) -> None:
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON"); return
            name = str(body.get("name") or body.get("preset", {}).get("name") or "").strip()
            payload = body.get("preset") if isinstance(body.get("preset"), dict) else body
            if not name and isinstance(payload, dict):
                name = str(payload.get("name") or "").strip()
            if not name:
                self._send(400, b"missing preset name"); return
            if name.lower().startswith("contract:"):
                self._send_json({
                    "ok": False,
                    "error": "'contract:' names are reserved for the unsaved model-default "
                             "feature — save it under a custom name to make it a preset",
                }, status=400)
                return
            try:
                from sed.data.presets import save_feature_preset
                preset = save_feature_preset(name, payload)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._send_json({"ok": True, "preset": preset})

        def _handle_feature_preset_delete(self) -> None:
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON"); return
            name = str(body.get("name") or "").strip()
            if not name:
                self._send(400, b"missing preset name"); return
            from sed.data.presets import delete_feature_preset
            removed = delete_feature_preset(name)
            self._send_json({"ok": removed, "deleted": removed, "name": name})

        def _handle_feature_preset_apply(self) -> None:
            """Set the Library tab's preview-only feature for dataset thumbnails.

            This is unrelated to the model/training feature: it never touches
            config.toml ([features]/[model]) and emits no model-compatibility
            warning. It only chooses how dataset clips are rendered for browsing.
            """
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON"); return
            name = str(body.get("name") or "").strip()
            if not name:
                self._send(400, b"missing preset name"); return
            try:
                from sed.data.inspect import set_preview_features
                from sed.data.presets import set_preview_preset
                preset = set_preview_preset(layout.id, name)
            except Exception as exc:  # noqa: BLE001
                log.exception("preview feature apply failed")
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            # Refresh the preview signature so clip thumbnails re-render in the
            # selected preview feature on the next request.
            try:
                set_preview_features(cfg, layout)
            except Exception:  # noqa: BLE001 — preview refresh is best-effort
                log.debug("preview refresh after apply failed", exc_info=True)
            self._send_json({
                "ok": True,
                "preset": preset.get("name") or name,
                "label": preset.get("label"),
            })

        def _handle_feature_training_apply(self) -> None:
            """Validate and persist the Audio Features form as this project's input.

            Saved presets remain reusable global templates, while this endpoint
            is intentionally project-scoped: it updates ``[features]`` and the
            model sample rate/channels together, validates the selected recipe's
            real input tensor, and leaves a refreshable readiness result for the
            Train tab.  No browser-only feature state can reach a training run.
            """
            nonlocal cfg
            try:
                body = self._read_json_body()
                feature_body = body.get("feature") if isinstance(body.get("feature"), dict) else body
                if not isinstance(feature_body, dict) or not feature_body:
                    raise ValueError("missing audio feature configuration")
                from sed.config import (
                    load_config,
                    update_config_toml_values,
                    validate_clink_cnn_v2_feature_contract,
                )
                from sed.data.presets import (
                    apply_preset_to_features_cfg,
                    feature_cfg_to_preset,
                    feature_source_for_config,
                )
                from sed.audio.features import (
                    feature_band_count,
                    feature_channels,
                    feature_frame_count,
                )
                from sed.train.backbones import get_spec, probe_spec
                from sed.train.model_lab import validate_config_recipe_params

                # The running Inspect server can outlive another UI action, so
                # always begin from the latest saved project configuration.
                cfg = load_config(project_id)
                target_features = apply_preset_to_features_cfg(cfg.features, feature_body)
                target_sr = int(feature_body.get("sample_rate") or cfg.model.sample_rate)
                candidate = cfg.model_copy(update={
                    "model": cfg.model.model_copy(update={
                        "sample_rate": target_sr,
                        "channels": int(feature_channels(target_features)),
                    }),
                    "features": target_features,
                })
                validate_clink_cnn_v2_feature_contract(
                    candidate, phase="Audio Features apply",
                )
                candidate = validate_config_recipe_params(candidate)

                # Build against the exact selected tensor before persisting it.
                # This catches a feature/model combination which is syntactically
                # valid but structurally unsupported by a selected architecture.
                probe = probe_spec(
                    get_spec(candidate.model.backbone),
                    n_bands=feature_band_count(target_features, target_sr),
                    n_frames=feature_frame_count(
                        sample_rate=target_sr,
                        window_seconds=target_features.window_seconds,
                        hop_length=target_features.hop_length,
                        n_fft=target_features.n_fft,
                        frontend=target_features.frontend,
                        win_length=int(target_features.win_length),
                        stft_engine=str(target_features.stft_engine),
                    ),
                    params=dict(candidate.model.params),
                    input_channels=int(feature_channels(target_features)),
                )
                compatibility = None
                if str(candidate.model.initialization) == "parent":
                    from sed.train.transfer import parent_compatibility

                    compatibility = parent_compatibility(
                        candidate,
                        str(candidate.model.parent_project),
                        str(candidate.model.parent_run),
                    )
                    if not compatibility.get("ok"):
                        raise ValueError(
                            "incompatible parent run: "
                            + str(compatibility.get("reason") or "unknown")
                        )

                update_config_toml_values(layout.config_toml, {
                    "features": target_features.model_dump(),
                    "model": {
                        "sample_rate": target_sr,
                        "channels": int(feature_channels(target_features)),
                    },
                })
                cfg = load_config(project_id)
                self._send_json({
                    "ok": True,
                    "feature": feature_cfg_to_preset(cfg, name=cfg.features.preset),
                    "feature_source": feature_source_for_config(cfg, project_id=project_id),
                    "readiness": _fitted_scaling_training_readiness(cfg, layout),
                    "probe": probe,
                    "compatibility": compatibility,
                })
            except Exception as exc:  # noqa: BLE001 — user configuration feedback
                self._send_json({"ok": False, "error": str(exc)}, status=400)

        def _refresh_job_payload(self, job: dict | None) -> dict:
            """Shared envelope plus an ETA extrapolated from the progress fraction."""
            envelope = _job_status_payload(job, missing="no refresh job")
            if not envelope["ok"]:
                return envelope
            payload = envelope["job"]
            elapsed = float(payload["elapsed_s"])
            progress_value = float(payload.get("progress") or 0.0)
            if 0 < progress_value < 1 and elapsed > 0:
                payload["eta_s"] = max(0.0, elapsed * (1.0 - progress_value) / progress_value)
            elif str(payload.get("state") or "") == "finished":
                payload["eta_s"] = 0.0
            else:
                payload["eta_s"] = None
            return envelope

        def _handle_review_data_status(self, job_id: str = "") -> None:
            self._send_job_status(
                refresh_lock, lambda: refresh_job, job_id, lambda job: self._refresh_job_payload(job),
            )

        def _handle_review_data_start(self) -> None:
            nonlocal refresh_job
            if cfg is None:
                self._send(400, b"missing config")
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}") if length else {}
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON")
                return
            try:
                from sed.data.inspect import normalise_review_refresh_scopes
                scopes, full_refresh = normalise_review_refresh_scopes(body.get("scopes"))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            run_id = str(body.get("run_id") or body.get("run") or "").strip() or None
            with refresh_lock:
                if refresh_job and str(refresh_job.get("state")) in {"starting", "running"}:
                    self._send_json({"ok": False, "error": "refresh job already running"}, status=409)
                    return
                job_id = uuid.uuid4().hex[:12]
                refresh_job = {
                    "job_id": job_id,
                    "state": "starting",
                    "phase": "starting",
                    "phase_detail": "starting",
                    "phase_count": 0,
                    "progress": 0.0,
                    "eta_s": None,
                    "scopes": scopes,
                    "full_refresh": full_refresh,
                    "summary_counts": {},
                    "payload": None,
                    "error": "",
                    "started_at": time.time(),
                    "ended_at": None,
                }

            def progress(update: dict) -> None:
                with refresh_lock:
                    if not refresh_job or refresh_job.get("job_id") != job_id:
                        return
                    refresh_job["state"] = "running"
                    if "phase" in update:
                        refresh_job["phase"] = update["phase"]
                    if "phase_detail" in update:
                        refresh_job["phase_detail"] = update["phase_detail"]
                    if "count" in update:
                        refresh_job["phase_count"] = update["count"]
                    if "progress" in update:
                        refresh_job["progress"] = update["progress"]

            def run_job() -> None:
                try:
                    from sed.data.inspect import review_data_payload
                    payload = review_data_payload(
                        cfg,
                        layout,
                        max_per_class=max_thumbs_per_class,
                        run_id=run_id,
                        scopes=None if full_refresh else scopes,
                        progress=progress,
                    )
                    with refresh_lock:
                        if refresh_job and refresh_job.get("job_id") == job_id:
                            refresh_job["payload"] = payload
                            refresh_job["summary_counts"] = payload.get("summary_counts", {})
                            refresh_job["state"] = "finished"
                            refresh_job["phase"] = "finished"
                            refresh_job["phase_detail"] = "finished"
                            refresh_job["progress"] = 1.0
                            refresh_job["eta_s"] = 0.0
                            refresh_job["ended_at"] = time.time()
                except Exception as exc:  # noqa: BLE001
                    log.exception("background review refresh failed")
                    with refresh_lock:
                        if refresh_job and refresh_job.get("job_id") == job_id:
                            refresh_job["state"] = "error"
                            refresh_job["phase"] = "error"
                            refresh_job["phase_detail"] = "error"
                            refresh_job["error"] = str(exc)
                            refresh_job["ended_at"] = time.time()

            thread = threading.Thread(target=run_job, daemon=True)
            with refresh_lock:
                refresh_job["thread"] = thread
                payload = self._refresh_job_payload(refresh_job)
                thread.start()
            self._send_json(payload)

        def _handle_compose_status(self, job_id: str = "") -> None:
            self._send_job_status(
                compose_lock, lambda: compose_job, job_id, lambda job: _job_status_payload(job, missing="no compose job"),
            )

        def _handle_compose_start(self) -> None:
            nonlocal compose_job
            if cfg is None:
                self._send(400, b"missing config")
                return
            try:
                body = self._read_json_body()
            except Exception:  # noqa: BLE001
                body = {}
            confirmation = body.get("frozen_test_confirmation")
            if confirmation is None:
                confirmation = body.get("confirmation_token")
            build_split_test = bool(body.get("build_split_test", False))
            with compose_lock:
                if compose_job and str(compose_job.get("state")) in {"starting", "running"}:
                    self._send_json({"ok": False, "error": "compose job already running"}, status=409)
                    return
                job_id = uuid.uuid4().hex[:12]
                compose_job = {
                    "job_id": job_id,
                    "state": "starting",
                    "phase": "starting",
                    "phase_detail": "starting",
                    "phase_count": 0,
                    "phase_total": 0,
                    "progress": 0.0,
                    "build_split_test": build_split_test,
                    "summary": {},
                    "error": "",
                    "started_at": time.time(),
                    "ended_at": None,
                }

            def progress(update: dict) -> None:
                with compose_lock:
                    if not compose_job or compose_job.get("job_id") != job_id:
                        return
                    compose_job["state"] = "running"
                    if "phase" in update:
                        compose_job["phase"] = update["phase"]
                    if "phase_detail" in update:
                        compose_job["phase_detail"] = update["phase_detail"]
                    if "count" in update:
                        compose_job["phase_count"] = update["count"]
                    if "total" in update:
                        compose_job["phase_total"] = update["total"]
                    if "progress" in update:
                        compose_job["progress"] = update["progress"]

            def run_job() -> None:
                nonlocal compose_job
                try:
                    from sed.data.compose import compose_dataset
                    with compose_lock:
                        if compose_job and compose_job.get("job_id") == job_id:
                            compose_job["state"] = "running"
                            compose_job["phase"] = "composing"
                    summary = compose_dataset(
                        cfg,
                        project_id,
                        confirm_frozen_test_change=confirmation,
                        build_split_test=build_split_test,
                        progress=progress,
                    )
                    with compose_lock:
                        if compose_job and compose_job.get("job_id") == job_id:
                            compose_job["summary"] = summary
                            compose_job["state"] = "finished"
                            compose_job["phase"] = "finished"
                            compose_job["phase_detail"] = "finished"
                            compose_job["progress"] = 1.0
                            compose_job["ended_at"] = time.time()
                except Exception as exc:  # noqa: BLE001
                    log.exception("background compose failed")
                    with compose_lock:
                        if compose_job and compose_job.get("job_id") == job_id:
                            compose_job["state"] = "error"
                            compose_job["phase"] = "error"
                            compose_job["phase_detail"] = "error"
                            compose_job["error"] = str(exc)
                            compose_job["ended_at"] = time.time()

            thread = threading.Thread(target=run_job, daemon=True)
            with compose_lock:
                compose_job["thread"] = thread
                payload = _job_status_payload(compose_job, missing="no compose job")
                thread.start()
            self._send_json(payload)

        def _handle_done(self) -> None:
            """Persist the currently ticked clips as the next-compose exclusion set."""
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw_paths = []
            if length:
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                    raw_paths = body.get("exclude_paths", body.get("paths", []))
                except Exception:  # noqa: BLE001
                    self._send(400, b"invalid JSON")
                    return
            if not isinstance(raw_paths, list):
                self._send(400, b"exclude_paths must be an array")
                return
            excluded = []
            for rp in raw_paths:
                resolved = _safe_resolve_audio_path(layout, rp)
                if resolved is not None:
                    excluded.append(resolved)
            from sed.data.compose import write_excluded_clip_paths
            manifest = write_excluded_clip_paths(layout, excluded)
            self._send_json({
                "ok": True,
                "excluded": [str(p) for p in excluded],
                "count": len(excluded),
                "manifest": str(manifest),
            })

        # ---- helpers ----------------------------------------------------

        def _send(self, status: int, body: bytes, ctype: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def _send_json(self, payload, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _send_job_status(self, lock, read_job, job_id: str, build) -> None:
            """Answer one polled background-job status request.

            ``job_id`` scopes the answer to a single run: a browser tab still
            polling a finished operation must get a 404, not the progress of
            whatever started after it — otherwise two overlapping operations
            report into each other's activity bar. The job is read and rendered
            under its own lock so a worker thread cannot swap it mid-render.
            """
            with lock:
                job = read_job()
                if job_id and job and str(job.get("job_id")) != job_id:
                    job = None
                payload = build(job)
            self._send_json(payload, status=200 if payload.get("ok") else 404)

        def _send_html(self, html: str) -> None:
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

        def _send_file(self, path: Path) -> None:
            ctype, _ = mimetypes.guess_type(str(path))
            ctype = ctype or "application/octet-stream"
            try:
                size = path.stat().st_size
            except OSError:
                self._send(404, b"not found"); return
            # Honour Range requests so the browser can seek long audio
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            status = 200
            if rng and rng.startswith("bytes="):
                try:
                    spec = rng[6:]
                    sp, ep = spec.split("-")
                    if sp: start = int(sp)
                    if ep: end = int(ep)
                    end = min(end, size - 1)
                    status = 206
                except Exception:  # noqa: BLE001
                    status = 200; start, end = 0, size - 1
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = f.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    Handler.cleanup = staticmethod(cleanup_runtime_resources)
    return Handler


def serve(
    cfg: Config,
    project_id: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    max_thumbs_per_class: int = 0,
    run_id: str | None = None,
) -> None:
    """Start the interactive inspect server. Blocks until Ctrl+C."""
    from sed.data.inspect import inspect_project
    # Re-render the HTML on every GET / so newly recorded clips appear
    # after a browser refresh.
    layout = ProjectLayout(project_id)

    log.info("rendering initial report ...")
    out_path = inspect_project(
        cfg, project_id,
        max_thumbs_per_class=max_thumbs_per_class,
        interactive=True,
        run_id=run_id,
    )
    log.info("static report also written to %s", out_path)

    def render_html_now(requested_run_id: str | None = None) -> str:
        from sed.data.inspect import _quick_rerender
        return _quick_rerender(cfg, project_id,
                               max_thumbs_per_class=max_thumbs_per_class,
                               run_id=requested_run_id or run_id)

    shutdown_event = threading.Event()
    Handler = make_handler(
        cfg,
        project_id,
        render_html_now,
        layout,
        shutdown_event,
        max_thumbs_per_class=max_thumbs_per_class,
    )

    # Bind, possibly auto-incrementing the port if busy
    for try_port in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, try_port), Handler)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"could not bind any port in [{port},{port+19}]")

    # Background jobs are threads; an uncaught one only prints to stderr, which
    # is easy to miss in a scrolled terminal. Route them through the log so a
    # dead worker always leaves a timestamped traceback behind.
    def _log_thread_exception(args) -> None:
        if args.exc_type is SystemExit:
            return
        log.error(
            "unhandled exception in background thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _log_thread_exception

    url = f"http://{host}:{httpd.server_port}/"
    log.info("inspect server listening on %s", url)
    log.info("Ctrl+C, or press 'Done' in the browser, to stop.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    # Serve on a worker thread so the main thread can also watch for the
    # in-browser "Done" button (which sets shutdown_event).
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    try:
        while not shutdown_event.is_set():
            if shutdown_event.wait(timeout=0.5):
                break
        log.info("Done received from browser — shutting down.")
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        cleanup = getattr(Handler, "cleanup", None)
        if callable(cleanup):
            cleanup()
        httpd.shutdown()
        server_thread.join(timeout=2.0)
