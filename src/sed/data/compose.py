"""Compose a project dataset from the unified shared audio library.

Physical recordings in ``projects/<id>/recordings`` are intentionally ignored.
Projects map library labels to model classes through config and the optional
dataset selection file; clips themselves stay in ``data_library``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from datetime import UTC, datetime

import soundfile as sf

from sed.config import (
    Config,
    ProjectLayout,
    effective_split_test_enabled,
    effective_split_test_fraction,
)
from sed.data.library import (
    DERIVED_CATEGORY,
    MINED_CATEGORY,
    PRETRAIN_CATEGORY,
    RAW_CATEGORY,
    Library,
    LibraryClip,
    _current_library_dir,
    clip_matches_mic_rule,
    clip_matches_session,
    clip_mic_channels,
)
from sed.data.splits import split_group_key, split_train_val_detailed
from sed.util import PhasedProgress, sha256_file

log = logging.getLogger(__name__)


EXCLUSIONS_FILENAME = "excluded_clips.json"
SELECTION_FILENAME = "dataset_selection.json"
DATASET_DIRTY_FILENAME = "dataset_dirty.json"
AUTO_MINED_DISMISSALS_KEY = "auto_mined_dismissed_bucket_keys"

def _pretrain_index():
    from sed.data.pretraining import shared_index
    return shared_index()


def _empty_dataset_selection() -> dict:
    return {
        "schema_version": 1,
        "assignments": {},
        "clip_assignments": {},
        "clip_overrides": {},
        "augmentation_combinations": {},
        "augmentation_recipes": [],
        "augmentation_settings": {},
        AUTO_MINED_DISMISSALS_KEY: [],
    }


def exclusion_manifest_path(layout: ProjectLayout) -> Path:
    return layout.dataset_dir / EXCLUSIONS_FILENAME


def selection_manifest_path(layout: ProjectLayout) -> Path:
    return layout.dataset_dir / SELECTION_FILENAME


def _migrate_selection_roles(payload: dict) -> bool:
    """Convert legacy rule roles to composition destinations in place."""
    changed = False
    assignments = payload.get("assignments")
    if isinstance(assignments, dict):
        for rows in assignments.values():
            if isinstance(rows, dict):
                rows = [rows]
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if "split" not in row and str(row.get("dataset_role") or "").lower() in {"train", "test"}:
                    row["split"] = str(row["dataset_role"]).lower()
                    changed = True
                if "dataset_role" in row:
                    row.pop("dataset_role", None)
                    changed = True
    for key in ("clip_assignments", "clip_overrides"):
        groups = payload.get(key)
        if isinstance(groups, dict):
            for rows in groups.values():
                if isinstance(rows, dict):
                    rows = [rows]
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if isinstance(row, dict) and "dataset_role" in row:
                        row.pop("dataset_role", None)
                        changed = True
    recipes = payload.get("augmentation_recipes")
    if isinstance(recipes, list):
        for recipe in recipes:
            if not isinstance(recipe, dict):
                continue
            for key in ("base_filter", "overlay_filter"):
                rule = recipe.get(key)
                if isinstance(rule, dict):
                    if "split" not in rule and str(rule.get("dataset_role") or "").lower() in {"train", "test"}:
                        rule["split"] = str(rule["dataset_role"]).lower()
                        changed = True
                    if "dataset_role" in rule:
                        rule.pop("dataset_role", None)
                        changed = True
            for key in ("base_filters", "overlay_filters"):
                rows = recipe.get(key)
                if isinstance(rows, list):
                    for rule in rows:
                        if isinstance(rule, dict):
                            if "split" not in rule and str(rule.get("dataset_role") or "").lower() in {"train", "test"}:
                                rule["split"] = str(rule["dataset_role"]).lower()
                                changed = True
                            if "dataset_role" in rule:
                                rule.pop("dataset_role", None)
                                changed = True
    return changed


def load_dataset_selection(layout: ProjectLayout) -> dict:
    path = selection_manifest_path(layout)
    if not path.exists():
        return _empty_dataset_selection()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to read dataset selection %s: %s", path, exc)
        return _empty_dataset_selection()
    if not isinstance(payload, dict):
        return _empty_dataset_selection()
    payload.setdefault("schema_version", 1)
    payload.setdefault("assignments", {})
    payload.setdefault("clip_assignments", {})
    payload.setdefault("clip_overrides", {})
    payload.setdefault("augmentation_combinations", {})
    payload.setdefault("augmentation_recipes", [])
    payload.setdefault("augmentation_settings", {})
    payload.setdefault(AUTO_MINED_DISMISSALS_KEY, [])
    _migrate_selection_roles(payload)
    return payload


def write_dataset_selection(layout: ProjectLayout, selection: dict) -> Path:
    out = selection_manifest_path(layout)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(selection or {})
    payload.setdefault("schema_version", 1)
    payload.setdefault("assignments", {})
    payload.setdefault("clip_assignments", {})
    payload.setdefault("clip_overrides", {})
    payload.setdefault("augmentation_combinations", {})
    payload.setdefault("augmentation_recipes", [])
    payload.setdefault("augmentation_settings", {})
    payload.setdefault(AUTO_MINED_DISMISSALS_KEY, [])
    _migrate_selection_roles(payload)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Dataset source-rule presets + compose-staleness fingerprints
# ---------------------------------------------------------------------------

DATASET_PRESETS_FILENAME = "dataset_presets.json"

# The dataset_selection.json keys compose_dataset() actually reads. Everything
# else in the selection (feature/preview preset names, augmentation recipes and
# settings, mined-rule dismissals) has no effect on the composed CSVs.
COMPOSE_SELECTION_KEYS = ("assignments", "clip_assignments", "clip_overrides")

# Rule bookkeeping that changes without changing what compose selects: display
# counts, random ids, timestamps, and server-side overlay annotations.
_VOLATILE_RULE_KEYS = frozenset({"id", "count", "updated_at", "created_at", "overlay_labels"})


def _strip_volatile(value):
    if isinstance(value, dict):
        return {
            k: _strip_volatile(v)
            for k, v in sorted(value.items())
            if k not in _VOLATILE_RULE_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def selection_rules_fingerprint(selection: dict) -> str:
    """Stable hash of the compose-relevant selection content.

    Volatile bookkeeping (ids, display counts, timestamps) is stripped so
    re-saving identical rules — e.g. applying a preset that matches the current
    state — does not change the fingerprint.
    """
    content = {
        key: _strip_volatile((selection or {}).get(key) or {})
        for key in COMPOSE_SELECTION_KEYS
    }
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def selection_compose_fingerprint(
    selection: dict, cfg: Config | None, *, layout: ProjectLayout | None = None
) -> str:
    """Fingerprint of every cheap input that shapes train.csv/test.csv.

    Covers the selection rules, the target classes, the split-test settings,
    and the soft-exclusion set. Deliberately excludes on-disk library content —
    this must stay file-read + hash cheap for page-load endpoints.
    """
    content = {
        "rules": selection_rules_fingerprint(selection),
        "classes": list(cfg.model.classes) if cfg is not None else [],
        "split_test": [
            bool(effective_split_test_enabled(cfg)) if cfg is not None else False,
            float(effective_split_test_fraction(cfg)) if cfg is not None else 0.0,
        ],
    }
    if layout is not None:
        content["excluded"] = sorted(load_excluded_clip_paths(layout))
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def dataset_snapshot_path(layout: ProjectLayout) -> Path:
    return layout.dataset_dir / "dataset_snapshot.json"


def dataset_dirty_path(layout: ProjectLayout) -> Path:
    return layout.dataset_dir / DATASET_DIRTY_FILENAME


def load_dataset_dirty(layout: ProjectLayout) -> dict:
    path = dataset_dirty_path(layout)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable marker must still block training
        return {
            "schema_version": 1,
            "reason": "The composed dataset dirty marker is unreadable.",
            "paths": [],
        }
    return payload if isinstance(payload, dict) else {}


def clear_dataset_dirty(layout: ProjectLayout) -> None:
    dataset_dirty_path(layout).unlink(missing_ok=True)


def mark_composed_dataset_dirty_for_paths(
    layout: ProjectLayout,
    paths: list[str | Path] | set[str | Path],
    *,
    reason: str = "source clip deleted",
) -> dict:
    """Mark composed CSVs stale when they reference one of ``paths``.

    A targeted source deletion is intentionally cheap: it does not silently
    rewrite the authoritative train/test CSVs or change a frozen test.  The
    marker makes that mismatch explicit and blocks training until the operator
    runs Compose, while deletions unrelated to this project remain invisible.
    """
    wanted = {_normalize_clip_path(path) for path in paths}
    affected: dict[str, set[str]] = {}
    if wanted:
        for split in ("train", "test"):
            csv_path = layout.dataset_dir / f"{split}.csv"
            if not csv_path.exists():
                continue
            try:
                with open(csv_path, encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        raw = str(row.get("path") or "")
                        if raw and _normalize_clip_path(raw) in wanted:
                            affected.setdefault(_normalize_clip_path(raw), set()).add(split)
            except Exception as exc:  # noqa: BLE001
                log.warning("failed to inspect %s for deleted paths: %s", csv_path, exc)
    if not affected:
        return {"dirty": False, "paths": [], "splits": {}}

    old = load_dataset_dirty(layout)
    old_paths = {
        _normalize_clip_path(path)
        for path in old.get("paths", [])
        if isinstance(path, str) and path
    }
    merged_paths = sorted(old_paths | set(affected))
    split_map: dict[str, list[str]] = {}
    for path, splits in (old.get("splits") or {}).items():
        if isinstance(splits, list):
            split_map[_normalize_clip_path(path)] = sorted({str(v) for v in splits})
    for path, splits in affected.items():
        split_map[path] = sorted(set(split_map.get(path, [])) | splits)
    payload = {
        "schema_version": 1,
        "created_at": str(old.get("created_at") or _utc_now_iso()),
        "updated_at": _utc_now_iso(),
        "reason": reason,
        "paths": merged_paths,
        "count": len(merged_paths),
        "splits": split_map,
    }
    out = dataset_dirty_path(layout)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    return {"dirty": True, **payload}


# dataset_snapshot.json can be ~1 MB (per-clip source hashes); the staleness
# check runs on every selection read/edit, so cache the recorded fingerprint
# per (mtime, size). Value None means the snapshot exists but is unreadable.
_snapshot_fingerprint_cache: dict[str, tuple[float, int, str | None]] = {}


def _snapshot_recorded_fingerprint(path: Path) -> str | None:
    try:
        st = path.stat()
    except OSError:
        return None
    key = str(path)
    cached = _snapshot_fingerprint_cache.get(key)
    if cached is not None and cached[0] == st.st_mtime and cached[1] == st.st_size:
        return cached[2]
    value: str | None
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        value = str((snapshot or {}).get("selection_fingerprint") or "")
    except Exception:  # noqa: BLE001 — unreadable snapshot ⇒ recompose
        value = None
    _snapshot_fingerprint_cache[key] = (st.st_mtime, st.st_size, value)
    return value


def dataset_compose_state(
    layout: ProjectLayout, selection: dict, cfg: Config | None
) -> dict:
    """Does the composed dataset still reflect the current selection?

    Compares the current selection fingerprint with the one recorded in
    dataset_snapshot.json at compose time. File reads and hashing only — safe
    for page-load endpoints on large corpora.
    """
    current = selection_compose_fingerprint(selection, cfg, layout=layout)
    path = dataset_snapshot_path(layout)
    state: dict = {
        "composed": False,
        "stale": False,
        "reason": "",
        "fingerprint": current,
    }
    if not path.exists():
        if any((selection or {}).get(key) for key in COMPOSE_SELECTION_KEYS):
            state["stale"] = True
            state["reason"] = (
                "Dataset has never been composed — run Compose to build train/test.csv."
            )
        dirty = load_dataset_dirty(layout)
        if dirty:
            state["stale"] = True
            state["dirty"] = True
            state["dirty_count"] = int(dirty.get("count") or len(dirty.get("paths") or []))
            state["reason"] = (
                f"{state['dirty_count']} composed source clip"
                f"{'s were' if state['dirty_count'] != 1 else ' was'} deleted — run Compose before training."
            )
        return state
    state["composed"] = True
    try:
        snapshot_stat = path.stat()
    except OSError:
        snapshot_stat = None
    if snapshot_stat is not None:
        state["composed_at"] = datetime.fromtimestamp(
            snapshot_stat.st_mtime, tz=UTC
        ).isoformat()
    recorded = _snapshot_recorded_fingerprint(path)
    if recorded is None:
        state["stale"] = True
        state["reason"] = "dataset_snapshot.json is unreadable — recompose the dataset."
    elif not recorded:
        state["stale"] = True
        state["reason"] = (
            "Dataset was composed before selection tracking — recompose once to sync."
        )
    elif recorded != current:
        state["stale"] = True
        state["reason"] = "Dataset selection changed since the last compose."
    dirty = load_dataset_dirty(layout)
    if dirty:
        state["stale"] = True
        state["dirty"] = True
        state["dirty_count"] = int(dirty.get("count") or len(dirty.get("paths") or []))
        state["reason"] = (
            f"{state['dirty_count']} composed source clip"
            f"{'s were' if state['dirty_count'] != 1 else ' was'} deleted — run Compose before training."
        )
    return state


def dataset_presets_path(layout: ProjectLayout) -> Path:
    return layout.dataset_dir / DATASET_PRESETS_FILENAME


def _empty_dataset_presets() -> dict:
    return {"schema_version": 1, "active": "", "presets": []}


def load_dataset_presets(layout: ProjectLayout) -> dict:
    path = dataset_presets_path(layout)
    if not path.exists():
        return _empty_dataset_presets()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to read dataset presets %s: %s", path, exc)
        return _empty_dataset_presets()
    if not isinstance(payload, dict) or not isinstance(payload.get("presets"), list):
        return _empty_dataset_presets()
    payload.setdefault("schema_version", 1)
    payload.setdefault("active", "")
    return payload


def write_dataset_presets(layout: ProjectLayout, payload: dict) -> Path:
    out = dataset_presets_path(layout)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _preset_slug(name: str) -> str:
    from sed.data.presets import slugify_preset_name

    return slugify_preset_name(name)


def _find_dataset_preset(payload: dict, name_or_slug: str) -> dict | None:
    wanted = str(name_or_slug or "").strip()
    slug = _preset_slug(wanted)
    for record in payload.get("presets", []):
        if isinstance(record, dict) and str(record.get("slug") or "") in (wanted, slug):
            return record
    return None


def _preset_rule_counts(rules: dict) -> tuple[int, int]:
    """(source-rule count, per-clip assignment count) for a preset summary."""
    rule_count = sum(
        len(rows) for rows in (rules.get("assignments") or {}).values()
        if isinstance(rows, list)
    )
    clip_count = sum(
        len(rows) for rows in (rules.get("clip_assignments") or {}).values()
        if isinstance(rows, list)
    )
    return rule_count, clip_count


DATASET_PRESET_SCHEMA_VERSION = 2


def classes_fingerprint(classes: list[str] | None) -> str:
    """Stable hash of a class list. Order-insensitive: reordering classes does
    not change what the dataset selects, only the head's index assignment."""
    blob = json.dumps(sorted(str(c) for c in (classes or [])), separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _preset_split_from_cfg(cfg: Config | None) -> dict:
    """The split knobs that shape train/val/test alongside the source rules."""
    if cfg is None:
        return {}
    return {
        "val_fraction": float(cfg.dataset.val_fraction),
        "split_test_enabled": effective_split_test_enabled(cfg),
        "split_test_fraction": effective_split_test_fraction(cfg),
    }


def save_dataset_preset(
    layout: ProjectLayout,
    name: str,
    selection: dict,
    cfg: Config | None = None,
) -> dict:
    """Snapshot the current dataset definition as a named per-project preset.

    A preset is source rules **plus the class list and split settings** — the
    complete answer to "what dataset is this", so switching between a
    pretraining corpus definition and a venue fine-tune definition is one
    action. Passing ``cfg`` records the classes; omitting it writes a v1-shaped
    rules-only preset, which is what pre-v2 callers get.

    Saving under an existing name (same slug) updates that preset in place.
    The saved preset becomes the active one.
    """
    display = str(name or "").strip()
    if not display:
        raise ValueError("preset name is required")
    slug = _preset_slug(display)
    rules = {
        key: json.loads(json.dumps((selection or {}).get(key) or {}))
        for key in COMPOSE_SELECTION_KEYS
    }
    rule_count, clip_count = _preset_rule_counts(rules)
    payload = load_dataset_presets(layout)
    existing = _find_dataset_preset(payload, slug)
    classes = list(cfg.dataset.target_classes) if cfg is not None else None
    record = {
        "schema_version": DATASET_PRESET_SCHEMA_VERSION if cfg is not None else 1,
        "name": display,
        "slug": slug,
        "rules": rules,
        "rules_fingerprint": selection_rules_fingerprint(selection),
        "rule_count": rule_count,
        "clip_count": clip_count,
        "created_at": str((existing or {}).get("created_at") or "") or _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }
    if classes is not None:
        record["classes"] = classes
        record["classes_fingerprint"] = classes_fingerprint(classes)
        record["split"] = _preset_split_from_cfg(cfg)
    if existing is not None:
        payload["presets"] = [
            record if r is existing else r for r in payload.get("presets", [])
        ]
    else:
        payload.setdefault("presets", []).append(record)
    payload["active"] = slug
    write_dataset_presets(layout, payload)
    return record


def apply_dataset_preset(
    layout: ProjectLayout,
    name_or_slug: str,
    *,
    apply_classes: bool = True,
) -> tuple[dict, dict]:
    """Restore a preset's dataset definition: source rules, and for v2 presets
    the class list and split settings too.

    Returns ``(selection, preset_record)``. Everything outside the compose keys
    (augmentation recipes/settings, feature/preview presets, dismissals) is
    left untouched, so switching datasets never loses generated audio or lab
    state. Raises ``KeyError`` when the preset does not exist.

    Classes are written to ``config.toml`` because they live in config, not in
    the selection. A v1 preset carries none, so it leaves the project's classes
    alone rather than clearing them — `dataset_preset_state` flags those as
    "no classes saved" so the difference is visible instead of silent.
    """
    payload = load_dataset_presets(layout)
    record = _find_dataset_preset(payload, name_or_slug)
    if record is None:
        raise KeyError(f"dataset preset '{name_or_slug}' not found")
    selection = load_dataset_selection(layout)
    rules = record.get("rules") or {}
    for key in COMPOSE_SELECTION_KEYS:
        selection[key] = json.loads(json.dumps(rules.get(key) or {}))
    selection["ui_managed"] = True
    selection["updated_at"] = _utc_now_iso()
    write_dataset_selection(layout, selection)

    classes = record.get("classes")
    if apply_classes and isinstance(classes, list) and classes:
        from sed.config import update_config_toml_values

        updates: dict[str, dict] = {"dataset": {"target_classes": [str(c) for c in classes]}}
        split = record.get("split")
        if isinstance(split, dict):
            for key in ("val_fraction", "split_test_enabled", "split_test_fraction"):
                if split.get(key) is not None:
                    updates["dataset"][key] = split[key]
        update_config_toml_values(layout.config_toml, updates)

    payload["active"] = str(record.get("slug") or "")
    write_dataset_presets(layout, payload)
    return selection, record


def new_dataset_selection(layout: ProjectLayout) -> dict:
    """Start an unnamed blank dataset while preserving augmentation/lab state."""
    selection = load_dataset_selection(layout)
    for key in COMPOSE_SELECTION_KEYS:
        selection[key] = {}
    selection["ui_managed"] = True
    selection["updated_at"] = _utc_now_iso()
    write_dataset_selection(layout, selection)
    payload = load_dataset_presets(layout)
    payload["active"] = ""
    write_dataset_presets(layout, payload)
    return selection


def rename_dataset_preset(layout: ProjectLayout, name_or_slug: str, new_name: str) -> dict:
    """Rename a preset without changing its rules or active state."""
    display = str(new_name or "").strip()
    if not display:
        raise ValueError("new preset name is required")
    payload = load_dataset_presets(layout)
    record = _find_dataset_preset(payload, name_or_slug)
    if record is None:
        raise KeyError(f"dataset preset '{name_or_slug}' not found")
    slug = _preset_slug(display)
    clash = _find_dataset_preset(payload, slug)
    if clash is not None and clash is not record:
        raise ValueError("a preset with that name already exists")
    old_slug = str(record.get("slug") or "")
    record["name"] = display
    record["slug"] = slug
    record["updated_at"] = _utc_now_iso()
    if payload.get("active") == old_slug:
        payload["active"] = slug
    write_dataset_presets(layout, payload)
    return record


def delete_dataset_preset(layout: ProjectLayout, name_or_slug: str) -> bool:
    """Delete a preset (the current selection rules are kept as they are)."""
    payload = load_dataset_presets(layout)
    record = _find_dataset_preset(payload, name_or_slug)
    if record is None:
        return False
    payload["presets"] = [r for r in payload.get("presets", []) if r is not record]
    if payload.get("active") == record.get("slug"):
        payload["active"] = ""
    write_dataset_presets(layout, payload)
    return True


def dataset_preset_state(
    layout: ProjectLayout, selection: dict, cfg: Config | None = None
) -> dict:
    """Preset summary for the inspect UI: list, active preset, modified flag.

    With ``cfg`` the summary also reports whether the active preset's classes
    still match the project's — a preset can be rule-identical yet describe a
    different class space, which changes the dataset just as much.
    """
    payload = load_dataset_presets(layout)
    current = selection_rules_fingerprint(selection)
    current_classes = (
        classes_fingerprint(list(cfg.dataset.target_classes)) if cfg is not None else None
    )
    presets = []
    active_slug = str(payload.get("active") or "")
    active_record: dict | None = None
    for record in payload.get("presets", []):
        if not isinstance(record, dict):
            continue
        classes = record.get("classes")
        has_classes = isinstance(classes, list) and bool(classes)
        presets.append({
            "name": str(record.get("name") or record.get("slug") or ""),
            "slug": str(record.get("slug") or ""),
            "rule_count": int(record.get("rule_count") or 0),
            "clip_count": int(record.get("clip_count") or 0),
            "updated_at": str(record.get("updated_at") or ""),
            "matches_current": str(record.get("rules_fingerprint") or "") == current,
            "schema_version": int(record.get("schema_version") or 1),
            # v1 presets predate class capture: applying one leaves the project's
            # classes as they are. Surfaced so the UI can say so rather than
            # letting a switch look complete when it only restored rules.
            "has_classes": has_classes,
            "classes": [str(c) for c in classes] if has_classes else [],
            "classes_match_current": (
                None if (current_classes is None or not has_classes)
                else str(record.get("classes_fingerprint") or "") == current_classes
            ),
        })
        if record.get("slug") == active_slug:
            active_record = record
    presets.sort(key=lambda p: p["name"].lower())
    if active_record is None:
        active_slug = ""
    modified = bool(
        active_record is not None
        and str(active_record.get("rules_fingerprint") or "") != current
    )
    active_classes = active_record.get("classes") if active_record else None
    if (
        not modified
        and current_classes is not None
        and isinstance(active_classes, list)
        and active_classes
    ):
        # Rule-identical but a different class space is still a modified dataset.
        modified = str(active_record.get("classes_fingerprint") or "") != current_classes
    return {
        "active": active_slug,
        "active_name": str((active_record or {}).get("name") or ""),
        "modified": modified,
        "presets": presets,
    }


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@lru_cache(maxsize=None)
def _realdir(directory: str) -> str:
    return os.path.realpath(directory)


def _normalize_clip_path(path: str | Path) -> str:
    # Canonical key for clip-path comparisons. Equivalent to
    # ``Path(path).expanduser().resolve()`` for the generated library layout,
    # but resolves each unique parent directory only once — per-file resolve()
    # walks every path component (an lstat each) and dominated compose/inspect
    # profiles at 75k clips.
    p = os.path.expanduser(os.fspath(path))
    directory, name = os.path.split(os.path.normpath(os.path.join(os.getcwd(), p)))
    return os.path.join(_realdir(directory), name)


def load_excluded_clip_paths(layout: ProjectLayout) -> set[str]:
    path = exclusion_manifest_path(layout)
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to read exclusion manifest %s: %s", path, exc)
        return set()
    raw_paths = payload.get("paths", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_paths, list):
        return set()
    return {
        _normalize_clip_path(p)
        for p in raw_paths
        if isinstance(p, str) and p
    }


def write_excluded_clip_paths(layout: ProjectLayout, paths: list[str | Path]) -> Path:
    normalized = sorted({_normalize_clip_path(p) for p in paths})
    out = exclusion_manifest_path(layout)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "paths": normalized,
                "count": len(normalized),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def _path_exists(path: str | Path) -> bool:
    try:
        return Path(path).expanduser().resolve().exists()
    except Exception:  # noqa: BLE001
        return False


def _prune_excluded_clip_paths(layout: ProjectLayout) -> set[str]:
    paths = load_excluded_clip_paths(layout)
    kept = sorted(p for p in paths if _path_exists(p))
    if len(kept) != len(paths):
        write_excluded_clip_paths(layout, kept)
    return set(kept)


def _clip_ref_is_live(
    ref: dict,
    *,
    uuids: set[str],
    relpaths: set[str],
    paths: set[str],
) -> bool:
    uuid = str(ref.get("uuid") or "")
    if uuid:
        return uuid in uuids
    relpath = str(ref.get("relpath") or "")
    if relpath:
        return relpath in relpaths
    raw_path = str(ref.get("path") or "")
    if raw_path:
        try:
            return str(Path(raw_path).expanduser().resolve()) in paths
        except Exception:  # noqa: BLE001
            return False
    return True


def prune_dataset_selection(layout: ProjectLayout, selection: dict | None = None) -> tuple[dict, bool]:
    """Remove clip assignments that point to files no longer present.

    Users may delete WAV/JSON pairs directly in a file browser. The library
    manifest is rebuildable, but the project selection file can still contain
    explicit clip references. This keeps that metadata in sync with disk.
    """
    selection = load_dataset_selection(layout) if selection is None else selection
    lib = Library()
    clips = lib.rebuild_manifest()
    uuids = {clip.uuid for clip in clips if clip.uuid}
    relpaths = {clip.relpath for clip in clips if clip.relpath}
    base = Path(os.path.realpath(lib.root))
    paths = {os.path.normpath(str(clip.absolute_path(base))) for clip in clips}

    changed = False
    assignments = selection.get("clip_assignments", {})
    if isinstance(assignments, dict):
        for class_name, rows in list(assignments.items()):
            normalized = rows if isinstance(rows, list) else [rows]
            kept = []
            for row in normalized:
                ref = {"path": row} if isinstance(row, str) else row
                if not isinstance(ref, dict):
                    continue
                if _clip_ref_is_live(ref, uuids=uuids, relpaths=relpaths, paths=paths):
                    kept.append(row)
            if kept:
                assignments[class_name] = kept
            else:
                assignments.pop(class_name, None)
            if len(kept) != len(normalized):
                changed = True

    overrides = selection.get("clip_overrides", {})
    if isinstance(overrides, dict):
        for key in list(overrides):
            if not _path_exists(key):
                overrides.pop(key, None)
                changed = True

    return selection, changed


def prune_project_metadata(layout: ProjectLayout) -> None:
    selection, changed = prune_dataset_selection(layout)
    if changed:
        write_dataset_selection(layout, selection)
    _prune_excluded_clip_paths(layout)


def _filter_excluded(
    clips: list["SourceClip"],
    excluded_paths: set[str],
) -> tuple[list["SourceClip"], int]:
    if not excluded_paths:
        return clips, 0
    kept = [clip for clip in clips if _normalize_clip_path(clip.path) not in excluded_paths]
    return kept, len(clips) - len(kept)


def _rule_dict(mapping: object) -> dict:
    if hasattr(mapping, "model_dump"):
        return mapping.model_dump(exclude_none=True, exclude_unset=True)  # type: ignore[no-any-return]
    return dict(mapping) if isinstance(mapping, dict) else {}


def _rule_labels(mapping: dict | object) -> list[str | None]:
    """Return selected labels for one source mapping.

    `labels = []` means select none. `labels = ["*"]`, `label = "*"`, or an
    omitted label means select all labels in that category.
    """
    row = _rule_dict(mapping)
    if "labels" in row:
        labels = list(row.get("labels") or [])
        if not labels:
            return []
        return [None] if "*" in labels else [str(label) for label in labels]
    label = row.get("label")
    if label in (None, "*"):
        return [None]
    return [str(label)]


def _expand_source_rules(rules: dict) -> dict[str, list[dict]]:
    """Expand `{labels=[...]}` rows into the older one-row-per-label shape."""
    expanded: dict[str, list[dict]] = {}
    for class_name, mapping in rules.items():
        rows = mapping if isinstance(mapping, list) else [mapping]
        out_rows: list[dict] = []
        for row in rows:
            row = _rule_dict(row)
            if not row:
                continue
            for label in _rule_labels(row):
                expanded_row = {
                    k: v for k, v in row.items()
                    if k not in ("label", "labels")
                }
                if label is not None:
                    expanded_row["label"] = label
                out_rows.append(expanded_row)
        expanded[class_name] = out_rows
    return expanded


@dataclass
class SourceClip:
    path: str
    class_name: str
    session_id: str
    sample_rate: int
    duration_s: float
    is_target: bool                # whether this class is a positive target
    origin: str                    # "project" | "external" | "augmented"
    # For derived/augmented clips: filename stem of the source recording, so
    # the split can keep variants out of val when their source trains (and
    # vice versa). Empty for raw clips or unlinkable legacy derived clips.
    source_stem: str = ""
    # The literal library category ("raw", "derived", "mined")
    category: str = ""
    # Original class/category this clip came from (import folder or recorded
    # class) — preserved through assignment so the Dataset tab can group a
    # target class by its constituent origin classes and the CSV keeps it.
    origin_class: str = ""
    # Stable source bucket + leakage group written into dataset CSVs. The
    # splitter falls back for older CSVs, but new composition writes these
    # explicitly so diagnostics can explain which buckets reached each split.
    split_bucket: str = ""
    split_group: str = ""
    original_source: str = ""
    augmentation_family: str = ""
    source_split_group: str = ""


def _read_sidecar(wav_path: Path) -> dict | None:
    side = wav_path.with_suffix(".json")
    if not side.exists():
        return None
    try:
        with open(side, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _snapshot_source_hashes(splits: dict[str, list["SourceClip"]]) -> dict[str, str]:
    source_hashes: dict[str, str] = {}
    for clips in splits.values():
        for clip in clips:
            path = Path(clip.path)
            if clip.path in source_hashes:
                continue
            try:
                source_hashes[clip.path] = sha256_file(path)
            except OSError:
                source_hashes[clip.path] = ""
    return dict(sorted(source_hashes.items()))


def _scan_project_class(
    class_dir: Path, class_name: str, is_target: bool, origin: str = "project"
) -> list[SourceClip]:
    out: list[SourceClip] = []
    if not class_dir.exists():
        return out
    for wav in sorted(class_dir.rglob("*.wav")):
        meta = _read_sidecar(wav)
        try:
            info = sf.info(str(wav))
        except Exception as e:  # noqa: BLE001
            log.warning("skipping unreadable %s: %s", wav, e)
            continue
        meta = meta or {}
        session_id = meta.get("session_id", wav.stem)
        source_group = str(meta.get("source_split_group") or session_id)
        out.append(
            SourceClip(
                path=str(wav.resolve()),
                class_name=class_name,
                session_id=session_id,
                sample_rate=info.samplerate,
                duration_s=info.frames / info.samplerate if info.samplerate else 0.0,
                is_target=is_target,
                origin=origin,
                origin_class=class_name,
                split_bucket=f"{origin}:{class_name}",
                split_group=source_group,
                original_source=str(
                    meta.get("original_source")
                    or meta.get("source_path")
                    or meta.get("source")
                    or wav.resolve()
                ),
                augmentation_family=str(
                    meta.get("augmentation_family") or meta.get("augmentation") or ""
                ),
                source_split_group=source_group,
            )
        )
    return out


def _scan_augmented_class(
    class_dir: Path, class_name: str, is_target: bool
) -> list[SourceClip]:
    """Pick up offline-augmented WAVs under an ``augmented/<class>/`` folder.

    Synthetic session_id is derived from the original ``source`` filename in
    the sidecar for traceability."""
    out: list[SourceClip] = []
    if not class_dir.exists():
        return out
    for wav in sorted(class_dir.rglob("*.wav")):
        meta = _read_sidecar(wav)
        try:
            info = sf.info(str(wav))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping unreadable augmented %s: %s", wav, exc)
            continue
        meta = meta or {}
        source_name = str(meta.get("original_source") or meta.get("source") or "")
        source_group = str(
            meta.get("source_split_group")
            or meta.get("group_id")
            or meta.get("session_id")
            or meta.get("parent_uuid")
            or (Path(source_name).stem if source_name else "")
            or f"aug-{wav.stem}"
        )
        session_id = source_group
        out.append(
            SourceClip(
                path=str(wav.resolve()),
                class_name=class_name,
                session_id=session_id,
                sample_rate=info.samplerate,
                duration_s=info.frames / info.samplerate if info.samplerate else 0.0,
                is_target=is_target,
                origin="augmented",
                source_stem=Path(source_name).stem if source_name else "",
                origin_class=class_name,
                split_bucket=f"augmented:{class_name}",
                split_group=source_group,
                original_source=source_name,
                augmentation_family=str(
                    meta.get("augmentation_family") or meta.get("augmentation") or "derived"
                ),
                source_split_group=source_group,
            )
        )
    return out


def _positive_test_labels_for_class(
    class_name: str,
    positive_rules: dict[str, list[dict]] | None,
) -> list[str]:
    """Return frozen-test positive folder names that map to ``class_name``.

    Legacy helper retained for older tests/imports. New composition selects
    frozen-test clips with source rules whose ``split`` is ``"test"``.
    """
    labels = [class_name]
    for rule in (positive_rules or {}).get(class_name, []):
        if rule.get("category") != "positives":
            continue
        label = rule.get("label")
        if label is None:
            continue
        labels.append(str(label))

    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def _scan_frozen_test(
    layout: ProjectLayout,
    classes: list[str],
    positive_rules: dict[str, list[dict]] | None = None,
) -> list[SourceClip]:
    """Legacy scanner for project-local frozen test folders.

    New dataset composition ignores project-local WAVs and selects holdout
    material with source rules whose ``split`` is ``"test"`` instead.
    """
    out: list[SourceClip] = []
    test_root = layout.test_dir
    if not test_root.exists():
        return out

    for class_name in classes:
        if class_name == "_negative":
            continue
        for label in _positive_test_labels_for_class(class_name, positive_rules):
            cls_dir = layout.test_positives_dir(label)
            out.extend(_scan_project_class(cls_dir, class_name, is_target=True))

    if "_negative" in classes:
        neg_root = test_root / "negatives"
        if neg_root.exists():
            for label_dir in sorted(p for p in neg_root.iterdir() if p.is_dir()):
                out.extend(_scan_project_class(label_dir, "_negative", is_target=False))

    aug_root = test_root / "augmented"
    if aug_root.exists():
        for cls_dir in sorted(p for p in aug_root.iterdir() if p.is_dir()):
            class_name = cls_dir.name
            if class_name not in classes:
                continue
            out.extend(
                _scan_augmented_class(
                    cls_dir,
                    class_name,
                    is_target=class_name != "_negative",
                )
            )

    return out


def _test_set_hash(test_clips: list[SourceClip]) -> str:
    """Stable hash of the frozen test set (sorted paths + class names).

    Different model iterations can compare hashes to confirm they were scored
    on the same held-out ground truth. Adding a new test clip changes the
    hash; nothing else does.
    """
    paths = sorted((c.class_name, c.path) for c in test_clips)
    h = hashlib.sha256()
    for cls, path in paths:
        h.update(cls.encode("utf-8"))
        h.update(b"\0")
        h.update(path.encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


def _selection_is_ui_managed(selection: dict | None) -> bool:
    return bool(isinstance(selection, dict) and selection.get("ui_managed") is True)


def _selection_rules(
    selection: dict,
    class_name: str,
    *,
    split: str | None = None,
) -> list[dict]:
    assignments = selection.get("assignments", {}) if isinstance(selection, dict) else {}
    rows = assignments.get(class_name, [])
    if isinstance(rows, dict):
        rows = [rows]
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_split = str(row.get("split") or "train").strip().lower()
        if split is not None and row_split and row_split != split:
            continue
        out.append(row)
    return out


def _selection_clip_refs(selection: dict, class_name: str) -> list[dict]:
    assignments = selection.get("clip_assignments", {}) if isinstance(selection, dict) else {}
    rows = assignments.get(class_name, [])
    if isinstance(rows, dict):
        rows = [rows]
    normalized: list[dict] = []
    for row in rows:
        if isinstance(row, str):
            normalized.append({"path": row})
        elif isinstance(row, dict):
            normalized.append(row)
    return normalized


def _clip_ref_sets(refs: list[dict]) -> tuple[set[str], set[str], set[str]]:
    """Pre-resolve explicit clip references once so matching a candidate clip
    is a set lookup instead of a per-(clip, ref) Path.resolve() pair."""
    uuids: set[str] = set()
    relpaths: set[str] = set()
    paths: set[str] = set()
    for ref in refs:
        uuid = str(ref.get("uuid") or "")
        if uuid:
            uuids.add(uuid)
        relpath = str(ref.get("relpath") or "")
        if relpath:
            relpaths.add(relpath)
        raw_path = str(ref.get("path") or "")
        if raw_path:
            try:
                paths.add(str(Path(raw_path).expanduser().resolve()))
            except Exception:  # noqa: BLE001
                continue
    return uuids, relpaths, paths


def _clip_matches_ref_sets(
    clip: LibraryClip,
    base: Path,
    uuids: set[str],
    relpaths: set[str],
    paths: set[str],
) -> bool:
    if clip.uuid and clip.uuid in uuids:
        return True
    if clip.relpath and clip.relpath in relpaths:
        return True
    if paths:
        return os.path.normpath(str(clip.absolute_path(base))) in paths
    return False


def _config_rules_for_class(cfg: Config, class_name: str) -> list[dict]:
    if cfg.dataset.sources:
        return _expand_source_rules(cfg.dataset.sources).get(class_name, [])
    rows: list[dict] = []
    for block in (cfg.dataset.project or {}, cfg.dataset.external or {}):
        expanded = _expand_source_rules(block)
        rows.extend(expanded.get(class_name, []))
    return rows


def _rules_for_class(
    cfg: Config,
    selection: dict,
    class_name: str,
    *,
    split: str,
) -> list[dict]:
    selected = _selection_rules(selection, class_name, split=split)
    if selected or _selection_is_ui_managed(selection):
        return selected
    # Config/legacy source mappings are training-pool compatibility inputs.
    # Frozen-test membership must be explicit in composition rules (or carved
    # by the percentage split); never mirror ordinary training mappings into
    # the holdout merely because the test scan is running.
    if split == "test":
        return []
    configured = _config_rules_for_class(cfg, class_name)
    if configured:
        return configured
    if class_name == "_negative":
        return []
    return [{"label": class_name}]


def _project_ids_for_rule(rule: dict, current_project: str, *, test: bool) -> set[str] | None:
    raw = rule.get("project_ids", rule.get("project_id"))
    if raw == "*":
        return None
    if isinstance(raw, str) and raw:
        return {raw}
    if isinstance(raw, list):
        return {str(item) for item in raw}
    if test:
        return {current_project}
    return {"_library", current_project}


def _tags_for_rule(rule: dict) -> list[str] | None:
    raw = rule.get("tags_all", rule.get("tags"))
    if isinstance(raw, str):
        tags = [part.strip().lstrip("#") for part in raw.replace(";", ",").split(",")]
    elif isinstance(raw, list):
        tags = [str(part).strip().lstrip("#") for part in raw]
    else:
        return None
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if not tag or tag in seen:
            continue
        deduped.append(tag)
        seen.add(tag)
    return deduped or None


def _category_for_rule(rule: dict) -> str:
    raw = str(rule.get("source_bucket") or rule.get("category") or "raw").strip().lower()
    if raw in {"raw", "sound", "sounds", "positive", "positives", "negative", "negatives"}:
        return RAW_CATEGORY
    if raw in {"derived", "augmentation", "augmented"}:
        return DERIVED_CATEGORY
    if raw in {"mined", "false_positive", "false-positives"}:
        return MINED_CATEGORY
    if raw in {"pretraining", "pretrain", "pretraining_corpus"}:
        return PRETRAIN_CATEGORY
    return RAW_CATEGORY


def _clip_kind_for_rule(rule: dict) -> str | None:
    """Resolve an optional clip_kind filter (mono | multichannel | channel_split).

    Lets a dataset rule target *only* the per-channel mono splits or *only* the
    multichannel originals, even though both share the same label and raw bucket.
    """
    raw = str(rule.get("clip_kind") or rule.get("kind") or "").strip().lower()
    if raw in {"", "*", "all", "any"}:
        return None
    if raw in {"split", "splits", "channel_split", "channel-split", "channels", "mono_split"}:
        return "channel_split"
    if raw in {"multi", "multichannel", "multi_channel", "multi-channel", "original"}:
        return "multichannel"
    if raw in {"mono", "single"}:
        return "mono"
    return raw


def _channel_index_for_rule(rule: dict) -> int | None:
    raw = rule.get("channel_index")
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _review_status_for_rule(rule: dict) -> str | None:
    raw = str(rule.get("review_status") or rule.get("status") or "accepted").strip()
    if raw in {"", "*", "all"}:
        return None
    return raw


def _clip_text_matches_rule(clip: LibraryClip, rule: dict) -> bool:
    q = str(rule.get("search") or "").strip().lower()
    session_q = str(rule.get("session") or rule.get("session_or_date") or "").strip().lower()
    if not q and not session_q:
        return True
    parts = [
        Path(clip.relpath).name,
        clip.relpath,
        clip.label,
        clip.project_id,
        clip.density,
        clip.review_status,
        clip.category,
        clip.session_id,
        clip.recorded_at,
        ",".join(str(ch) for ch in clip_mic_channels(clip)),
        " ".join(clip.tags or []),
    ]
    haystack = " ".join(str(part or "").lower() for part in parts)
    if q and q not in haystack:
        return False
    if session_q and not clip_matches_session(clip, session_q):
        return False
    return True


def _clip_bucket_session_value(clip: LibraryClip) -> str:
    return clip.session_id or (clip.recorded_at[:10] if clip.recorded_at else "")


def _mic_rule_fields_for_clip(clip: LibraryClip) -> dict:
    channels = sorted(set(int(ch) for ch in clip_mic_channels(clip)))
    if channels:
        return {"triggered_channels": channels}
    if clip.clip_kind == "channel_split" and clip.channel_index is not None:
        return {"channel_index": int(clip.channel_index)}
    return {"mic_filter": "unknown"}


def _auto_mined_bucket_key(clip: LibraryClip) -> str:
    payload = {
        "label": clip.label,
        "project_id": clip.project_id,
        "density": clip.density,
        "tags": list(clip.tags or []),
        "source_bucket": "mined",
        "clip_kind": clip.clip_kind,
        "mic": _mic_rule_fields_for_clip(clip),
        "session": _clip_bucket_session_value(clip),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _auto_mined_rule_for_bucket(key: str, clips: list[LibraryClip]) -> dict:
    first = clips[0]
    rule = {
        "id": f"auto-mined-{key[:12]}",
        "labels": [first.label],
        "project_ids": [first.project_id],
        "split": "train",
        "density": first.density,
        "source_bucket": "mined",
        "clip_kind": first.clip_kind,
        "auto_source": "mined",
        "auto_bucket_key": key,
        "count": len(clips),
    }
    tags = [str(tag) for tag in (first.tags or []) if str(tag)]
    if tags:
        rule["tags_all"] = tags
    session = _clip_bucket_session_value(first)
    if session:
        rule["session"] = session
    rule.update(_mic_rule_fields_for_clip(first))
    return rule


def mark_auto_mined_rules_dismissed(selection: dict, rows: list[dict]) -> bool:
    """Remember removed auto-mined bucket rules so refresh does not recreate them."""
    dismissed = selection.setdefault(AUTO_MINED_DISMISSALS_KEY, [])
    if not isinstance(dismissed, list):
        dismissed = []
        selection[AUTO_MINED_DISMISSALS_KEY] = dismissed
    existing = {str(key) for key in dismissed if str(key)}
    changed = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("auto_source") or "") != "mined":
            continue
        key = str(row.get("auto_bucket_key") or "")
        if not key or key in existing:
            continue
        dismissed.append(key)
        existing.add(key)
        changed = True
    if changed:
        selection[AUTO_MINED_DISMISSALS_KEY] = sorted(existing)
    return changed


def ensure_auto_mined_negative_rules(
    cfg: Config | None,
    layout: ProjectLayout,
    selection: dict | None = None,
    *,
    write: bool = False,
) -> tuple[dict, bool]:
    """Persist one auto `_negative` source rule per mined library bucket."""
    selection = load_dataset_selection(layout) if selection is None else selection
    if cfg is None or "_negative" not in set(cfg.model.classes):
        return selection, False

    lib = Library()
    buckets: dict[str, list[LibraryClip]] = {}
    for clip in lib.query(
        category=MINED_CATEGORY,
        project_ids={layout.id},
        split="train",
        review_status="accepted",
    ):
        key = _auto_mined_bucket_key(clip)
        buckets.setdefault(key, []).append(clip)

    dismissed_raw = selection.setdefault(AUTO_MINED_DISMISSALS_KEY, [])
    dismissed = {str(key) for key in dismissed_raw if str(key)} if isinstance(dismissed_raw, list) else set()
    if not isinstance(dismissed_raw, list):
        selection[AUTO_MINED_DISMISSALS_KEY] = sorted(dismissed)

    assignments = selection.setdefault("assignments", {})
    if not isinstance(assignments, dict):
        assignments = {}
        selection["assignments"] = assignments

    rows_raw = assignments.get("_negative", [])
    if isinstance(rows_raw, dict):
        rows = [rows_raw]
    elif isinstance(rows_raw, list):
        rows = [row for row in rows_raw if isinstance(row, dict)]
    else:
        rows = []

    changed = rows != rows_raw
    kept: list[dict] = []
    live_auto_keys: set[str] = set()
    for row in rows:
        if str(row.get("auto_source") or "") != "mined":
            kept.append(row)
            continue
        key = str(row.get("auto_bucket_key") or "")
        if not key or key not in buckets or key in dismissed:
            changed = True
            continue
        generated = _auto_mined_rule_for_bucket(key, buckets[key])
        kept.append(generated)
        live_auto_keys.add(key)
        if row != generated:
            changed = True

    for key in sorted(buckets):
        if key in live_auto_keys or key in dismissed:
            continue
        kept.append(_auto_mined_rule_for_bucket(key, buckets[key]))
        changed = True

    if kept:
        assignments["_negative"] = kept
    elif "_negative" in assignments:
        assignments.pop("_negative", None)
        changed = True

    if changed:
        selection["updated_at"] = datetime_now_utc_iso()
        if write:
            write_dataset_selection(layout, selection)
    return selection, changed


def datetime_now_utc_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _append_clip_once(
    out: list[SourceClip],
    seen: set[str],
    clip: LibraryClip,
    class_name: str,
    *,
    origin: str,
    base: Path,
) -> None:
    # ``base`` must already be resolved (see _library_clips_for_class): relpaths
    # from the manifest/index are clean, so joining them onto a resolved root
    # gives a canonical path without a per-clip resolve() — Path.resolve() does
    # one lstat per path component and dominated page-load profiles at 75k clips.
    path = os.path.normpath(str(clip.absolute_path(base)))
    if path in seen:
        return
    seen.add(path)
    out.append(_clip_to_source(clip, class_name, origin=origin, base=base))


def _clip_to_source(
    clip: LibraryClip,
    class_name: str,
    *,
    origin: str,
    base: Path | None = None,
) -> SourceClip:
    if base is None:
        base = Path(os.path.realpath(_current_library_dir()))
    origin_class = getattr(clip, "origin_class", "") or clip.label
    source_split_group = (
        clip.source_split_group
        or clip.group_id
        or clip.session_id
        or clip.parent_uuid
        or clip.uuid
    )
    return SourceClip(
        path=os.path.normpath(str(clip.absolute_path(base))),
        class_name=class_name,
        session_id=clip.session_id or clip.uuid,
        sample_rate=clip.sample_rate,
        duration_s=clip.duration_s,
        is_target=class_name not in {"_negative", "_unknown", "_background"},
        origin=origin,
        category=clip.category,
        origin_class=origin_class,
        source_stem=(
            Path(clip.original_source or clip.source_path).stem
            if (clip.original_source or clip.source_path)
            else ""
        ),
        split_bucket=f"{origin}:{origin_class}",
        split_group=source_split_group,
        original_source=clip.original_source or clip.source_path or clip.relpath,
        augmentation_family=clip.augmentation_family,
        source_split_group=source_split_group,
    )


def _library_clips_for_class(
    cfg: Config,
    project_id: str,
    class_name: str,
    *,
    split: str,
    selection: dict | None = None,
    include_derived: bool = True,
) -> list[SourceClip]:
    selection = selection or {}
    lib = Library()
    # Resolve the library root once; every per-clip path below is then a plain
    # join + normpath instead of a syscall-heavy Path.resolve() per clip.
    base = Path(os.path.realpath(lib.root))
    rules = _rules_for_class(
        cfg,
        selection,
        class_name,
        split=split,
    )
    selected_clip_refs = _selection_clip_refs(selection, class_name)
    out: list[SourceClip] = []
    seen: set[str] = set()

    if selected_clip_refs:
        categories = [RAW_CATEGORY, MINED_CATEGORY]
        if include_derived:
            categories.append(DERIVED_CATEGORY)
        candidates = [
            clip
            for category in categories
            for clip in lib.query(
                category=category,
                review_status="accepted",
            )
        ]
        ref_uuids, ref_relpaths, ref_paths = _clip_ref_sets(selected_clip_refs)
        for clip in candidates:
            if not _clip_matches_ref_sets(clip, base, ref_uuids, ref_relpaths, ref_paths):
                continue
            _append_clip_once(
                out,
                seen,
                clip,
                class_name,
                origin=clip.category,
                base=base,
            )

    for rule in rules:
        labels = _rule_labels(rule)
        if not labels:
            continue
        category = _category_for_rule(rule)
        project_ids = _project_ids_for_rule(rule, project_id, test=split == "test")
        density = rule.get("density")
        if category == PRETRAIN_CATEGORY:
            # Pretraining clips live in a dedicated index, not the manifest. Pull
            # the assigned class folder(s) from it and synthesize library clips
            # (sample_rate / duration_s probed on demand) so they flow through
            # the same filter + append path as normal sources.
            label_filter = [] if labels == [None] else [lbl for lbl in labels if lbl]
            candidates = _pretrain_index().clips_for_labels(label_filter, probe=True)
            # A rule may pin one subbucket (one sample rate) of a folder.
            want_sr = rule.get("sample_rate")
            if want_sr:
                try:
                    want_sr = int(want_sr)
                    candidates = [c for c in candidates if int(c.sample_rate) == want_sr]
                except (TypeError, ValueError):
                    pass
        else:
            candidates = lib.query(
                category=category,
                project_ids=project_ids,
                review_status=_review_status_for_rule(rule),
                density=str(density) if density else None,
                tags_all=_tags_for_rule(rule),
            )
        clip_kind = _clip_kind_for_rule(rule)
        channel_index = _channel_index_for_rule(rule)
        for clip in candidates:
            if labels != [None] and clip.label not in labels:
                continue
            if clip_kind is None:
                # Per-channel splits are opt-in: a rule must explicitly request
                # clip_kind=channel_split to include them, otherwise the mono
                # split would double-count with the original. The explicit
                # mono audio filter is the exception: it means every stored
                # one-channel clip, including channel_split provenance.
                mic_filter = str(rule.get("mic_filter") or "").strip().lower()
                if clip.clip_kind == "channel_split" and mic_filter not in {"mono", "one", "one-channel", "one_channel"}:
                    continue
            elif clip.clip_kind != clip_kind:
                continue
            if channel_index is not None and clip.channel_index != channel_index:
                continue
            if not clip_matches_mic_rule(clip, rule):
                continue
            if not _clip_text_matches_rule(clip, rule):
                continue
            _append_clip_once(
                out,
                seen,
                clip,
                class_name,
                origin=category,
                base=base,
            )

    should_include_target_derived = include_derived and not _selection_is_ui_managed(selection)
    if should_include_target_derived:
        for clip in lib.query(
            category=DERIVED_CATEGORY,
            label=class_name,
            project_ids={project_id},
            review_status="accepted",
        ):
            _append_clip_once(
                out,
                seen,
                clip,
                class_name,
                origin=DERIVED_CATEGORY,
                base=base,
            )
    return out


def pretraining_test_path(layout: ProjectLayout) -> Path:
    """Legacy pretraining split-test manifest path."""
    return layout.dataset_dir / "pretraining_test.json"


def split_frozen_test_path(layout: ProjectLayout) -> Path:
    return layout.dataset_dir / "frozen_test_split.json"


def frozen_test_manifest_path(layout: ProjectLayout) -> Path:
    return layout.dataset_dir / "frozen_test_manifest.json"


BUCKET_FROZEN_SOURCE = "selected_buckets"
SPLIT_FROZEN_SOURCE = "split_derived"


def _utc_now_iso() -> str:
    import time as _time

    return _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())


def _read_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _load_split_frozen_manifest(layout: ProjectLayout) -> tuple[dict, Path | None]:
    path = split_frozen_test_path(layout)
    data = _read_json_dict(path)
    if data:
        return data, path
    legacy = pretraining_test_path(layout)
    data = _read_json_dict(legacy)
    if data:
        data = dict(data)
        data.setdefault("legacy_source", "pretraining_test")
        return data, legacy
    return {}, None


def _split_manifest_record(
    *,
    project_id: str,
    classes: list[str],
    fraction: float,
    seed: int,
    clips: list[SourceClip],
    carve_modes: dict[str, str],
    split_diagnostics: dict | None = None,
    previous: dict | None = None,
) -> dict:
    previous = previous or {}
    chosen_by_class: dict[str, list[str]] = {c: [] for c in classes}
    sessions_by_class: dict[str, list[str]] = {c: [] for c in classes}
    buckets_by_class: dict[str, list[str]] = {c: [] for c in classes}
    for clip in clips:
        chosen_by_class.setdefault(clip.class_name, []).append(_normalize_clip_path(clip.path))
        sessions_by_class.setdefault(clip.class_name, []).append(
            split_group_key(clip)
        )
        buckets_by_class.setdefault(clip.class_name, []).append(clip.split_bucket or clip.origin_class or clip.origin)
    now = _utc_now_iso()
    created_at = str(previous.get("created_at") or now)
    return {
        "schema_version": 3,
        "kind": "split_derived_frozen_test",
        "project_id": project_id,
        "carve": "bucket-session-grouped",
        "carve_modes": carve_modes,
        "split_diagnostics": split_diagnostics or previous.get("split_diagnostics", {}),
        "fraction": float(fraction),
        "seed": seed,
        "created_at": created_at,
        "changed_at": now if previous else created_at,
        "keys": sorted({_normalize_clip_path(c.path) for c in clips}),
        "by_class": {c: sorted(set(keys)) for c, keys in chosen_by_class.items()},
        "sessions_by_class": {
            c: sorted(set(sessions)) for c, sessions in sessions_by_class.items()
        },
        "buckets_by_class": {
            c: sorted(set(buckets)) for c, buckets in buckets_by_class.items()
        },
    }


def _write_split_frozen_manifest(layout: ProjectLayout, record: dict) -> None:
    path = split_frozen_test_path(layout)
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    tmp.replace(path)


def _partition_split_frozen_keys(
    layout: ProjectLayout,
    by_class_train: dict[str, list[SourceClip]],
    classes: list[str],
    keys: set[str],
    carve_modes: dict[str, str],
) -> tuple[list[SourceClip], dict[str, list[SourceClip]]]:
    test_clips: list[SourceClip] = []
    test_stems: set[str] = set()
    for class_name in classes:
        for clip in by_class_train.get(class_name, []):
            if _normalize_clip_path(clip.path) in keys:
                test_clips.append(clip)
                test_stems.add(Path(clip.path).stem)

    remaining: dict[str, list[SourceClip]] = {c: [] for c in classes}
    derived_origins = {DERIVED_CATEGORY, "augmented"}
    dropped_derived = 0
    for class_name in classes:
        for clip in by_class_train.get(class_name, []):
            if _normalize_clip_path(clip.path) in keys:
                continue
            if clip.origin in derived_origins and clip.source_stem in test_stems:
                dropped_derived += 1
                continue
            remaining[class_name].append(clip)
    if dropped_derived:
        log.info(
            "dropped %d derived clip(s) whose source landed in split-derived "
            "frozen test (train/test leakage guard)",
            dropped_derived,
        )

    _warn_on_frozen_test_session_leak(
        test_clips, remaining, carve_modes, split_frozen_test_path(layout)
    )
    return test_clips, remaining


def _carve_split_frozen_test(
    project_id: str,
    by_class_train: dict[str, list[SourceClip]],
    classes: list[str],
    fraction: float,
    seed: int,
    previous: dict | None,
) -> tuple[list[SourceClip], dict[str, list[SourceClip]], dict, bool]:
    derived_origins = {DERIVED_CATEGORY, "augmented"}
    pool = [
        clip
        for class_name in classes
        for clip in by_class_train.get(class_name, [])
    ]
    raw_pool = [c for c in pool if c.origin not in derived_origins]
    derived_pool = [c for c in pool if c.origin in derived_origins]
    remaining_flat, picked, split_diagnostics = split_train_val_detailed(
        raw_pool, derived_pool, val_frac=fraction, seed=seed,
    )
    carve_modes = dict(split_diagnostics.get("modes") or {})
    remaining: dict[str, list[SourceClip]] = {c: [] for c in classes}
    for clip in remaining_flat:
        remaining.setdefault(clip.class_name, []).append(clip)
    record = _split_manifest_record(
        project_id=project_id,
        classes=classes,
        fraction=fraction,
        seed=seed,
        clips=picked,
        carve_modes=carve_modes,
        split_diagnostics=split_diagnostics,
        previous=previous,
    )
    return picked, remaining, record, True


def _resolve_split_frozen_test(
    layout: ProjectLayout,
    project_id: str,
    by_class_train: dict[str, list[SourceClip]],
    classes: list[str],
    fraction: float,
    *,
    allow_carve: bool = True,
) -> tuple[list[SourceClip], dict[str, list[SourceClip]], dict | None, bool]:
    """Resolve the persisted split-derived frozen test without writing it.

    Returns ``(test_clips, remaining_train_by_class, manifest_record,
    manifest_needs_write)``. The record is written only after compose passes the
    frozen-test confirmation gate.

    ``allow_carve=False`` (ordinary dataset composition) never creates or
    resizes this slice — it only re-applies an already-persisted one against
    the current pool. Creating or resizing it requires an explicit build
    (``allow_carve=True``), fired from the Library tab's "Build frozen test
    set" action, so the frozen test never changes as a side effect of routine
    source-rule, class, or sample edits.
    """
    persisted, persisted_path = _load_split_frozen_manifest(layout)
    seed = int.from_bytes(hashlib.sha1(project_id.encode()).digest()[:4], "big")
    persisted_fraction = persisted.get("fraction") if persisted else None
    try:
        persisted_fraction_f = float(persisted_fraction)
    except (TypeError, ValueError):
        persisted_fraction_f = None

    if not persisted:
        if not allow_carve:
            return [], by_class_train, None, False
        return _carve_split_frozen_test(
            project_id, by_class_train, classes, fraction, seed, None,
        )

    # ``allow_carve`` is only true for the explicit Build frozen test action.
    # Re-run the deterministic carve even when the fraction is unchanged so a
    # splitter improvement (for example class-representation fallback) can be
    # applied to an existing project. Routine compose passes False and keeps
    # the persisted benchmark fixed.
    if allow_carve:
        return _carve_split_frozen_test(
            project_id, by_class_train, classes, fraction, seed, persisted,
        )

    # Reuse the persisted slice as-is. A fraction edit that hasn't been
    # applied via an explicit rebuild does not resize it.
    effective_fraction = persisted_fraction_f if persisted_fraction_f is not None else fraction
    keys = {_normalize_clip_path(k) for k in (persisted.get("keys") or [])}
    carve_modes = dict(persisted.get("carve_modes") or {})
    split_diagnostics = dict(persisted.get("split_diagnostics") or {})
    test_clips, remaining = _partition_split_frozen_keys(
        layout, by_class_train, classes, keys, carve_modes
    )
    record = _split_manifest_record(
        project_id=project_id,
        classes=classes,
        fraction=float(effective_fraction),
        seed=int(persisted.get("seed") or seed),
        clips=test_clips,
        carve_modes=carve_modes,
        split_diagnostics=split_diagnostics,
        previous=persisted,
    )
    needs_migration = persisted_path != split_frozen_test_path(layout)
    return test_clips, remaining, record, bool(needs_migration)


def _resolve_pretraining_frozen_test(
    layout: ProjectLayout,
    project_id: str,
    by_class_train: dict[str, list[SourceClip]],
    classes: list[str],
    fraction: float,
) -> tuple[list[SourceClip], dict[str, list[SourceClip]]]:
    """Compatibility wrapper for older tests/imports.

    New composition uses the project-agnostic split-derived frozen test manifest
    and passes its explicit build intent directly. This legacy wrapper preserves
    its historical contract: create a carve when none exists, then reuse it.
    """
    persisted, _ = _load_split_frozen_manifest(layout)
    test_clips, remaining, record, needs_write = _resolve_split_frozen_test(
        layout, project_id, by_class_train, classes, fraction,
        allow_carve=not bool(persisted),
    )
    if needs_write:
        _write_split_frozen_manifest(layout, record)
    return test_clips, remaining


def _warn_on_frozen_test_session_leak(
    test_clips: list[SourceClip],
    remaining: dict[str, list[SourceClip]],
    carve_modes: dict[str, str],
    path: Path,
) -> None:
    """Flag frozen test sets whose sessions also feed train/val.

    Legacy carves (schema 1) sampled random clips per class, so test clips sit
    seconds away from train clips of the same recording — their metrics are
    optimistic. Classes the session-grouped carve itself degraded to a
    contiguous fallback are expected to share sessions and are not reported.
    """
    if not test_clips:
        return

    def _sid(clip: SourceClip) -> str:
        return split_group_key(clip)

    fallback = {c for c, mode in carve_modes.items() if "fallback" in str(mode)}
    test_sessions: dict[str, set[str]] = {}
    for clip in test_clips:
        test_sessions.setdefault(clip.class_name, set()).add(_sid(clip))
    leaky: dict[str, int] = {}
    for class_name, sessions in sorted(test_sessions.items()):
        if class_name in fallback:
            continue
        train_sessions = {_sid(c) for c in remaining.get(class_name, [])}
        overlap = sessions & train_sessions
        if overlap:
            leaky[class_name] = len(overlap)
    if leaky:
        log.warning(
            "frozen pretraining test shares recording sessions with train/val "
            "(%s). Test metrics are optimistic: near-duplicate clips from the "
            "same session sit on both sides. Delete %s and recompose to carve "
            "a session-grouped test set (this changes the test set, so scores "
            "are not comparable with older runs).",
            ", ".join(f"{cls}: {n} session(s)" for cls, n in leaky.items()),
            path,
        )


def _frozen_source_counts(source_by_path: dict[str, str]) -> dict[str, int]:
    counts = {BUCKET_FROZEN_SOURCE: 0, SPLIT_FROZEN_SOURCE: 0}
    for source in source_by_path.values():
        counts[source] = counts.get(source, 0) + 1
    return counts


def _frozen_source_by_class(
    test: list[SourceClip], source_by_path: dict[str, str], classes: list[str]
) -> dict[str, dict[str, int]]:
    out = {
        BUCKET_FROZEN_SOURCE: {c: 0 for c in classes},
        SPLIT_FROZEN_SOURCE: {c: 0 for c in classes},
    }
    for clip in test:
        source = source_by_path.get(_normalize_clip_path(clip.path), BUCKET_FROZEN_SOURCE)
        out.setdefault(source, {c: 0 for c in classes})
        out[source][clip.class_name] = out[source].get(clip.class_name, 0) + 1
    return out


def _frozen_row_record(clip: SourceClip, source: str) -> dict:
    return {
        "path": clip.path,
        "class_name": clip.class_name,
        "session_id": clip.session_id,
        "sample_rate": clip.sample_rate,
        "duration_s": clip.duration_s,
        "is_target": bool(clip.is_target),
        "origin": clip.origin,
        "origin_class": clip.origin_class or clip.class_name,
        "source_stem": clip.source_stem,
        "split_bucket": clip.split_bucket or clip.origin_class or clip.origin,
        "split_group": clip.split_group or split_group_key(clip),
        "original_source": clip.original_source or clip.path,
        "augmentation_family": clip.augmentation_family,
        "source_split_group": (
            clip.source_split_group or clip.split_group or split_group_key(clip)
        ),
        "frozen_source": source,
    }


def _build_frozen_manifest(
    *,
    project_id: str,
    test: list[SourceClip],
    source_by_path: dict[str, str],
    classes: list[str],
    test_hash: str,
    split_record: dict | None,
) -> dict:
    source_counts = _frozen_source_counts(source_by_path)
    split_diagnostics = (
        split_record.get("split_diagnostics", {})
        if isinstance(split_record, dict) else {}
    )
    return {
        "schema_version": 1,
        "project_id": project_id,
        "created_at": _utc_now_iso(),
        "hash": test_hash,
        "total": len(test),
        "by_class": {
            class_name: sum(1 for clip in test if clip.class_name == class_name)
            for class_name in classes
        },
        "source_counts": source_counts,
        "source_by_class": _frozen_source_by_class(test, source_by_path, classes),
        "split_manifest": str(split_frozen_test_path(ProjectLayout(project_id)))
        if split_record else "",
        "split_diagnostics": split_diagnostics,
        "rows": [
            _frozen_row_record(
                clip,
                source_by_path.get(_normalize_clip_path(clip.path), BUCKET_FROZEN_SOURCE),
            )
            for clip in sorted(test, key=lambda c: (c.class_name, c.path))
        ],
    }


def _write_frozen_manifest(layout: ProjectLayout, manifest: dict) -> None:
    path = frozen_test_manifest_path(layout)
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(path)


def _existing_frozen_snapshot(layout: ProjectLayout) -> dict:
    snap = _read_json_dict(layout.dataset_dir / "dataset_snapshot.json")
    frozen = snap.get("frozen_test") if isinstance(snap.get("frozen_test"), dict) else {}
    return frozen if isinstance(frozen, dict) else {}


def _frozen_confirmation_token(
    project_id: str,
    old_hash: str,
    new_hash: str,
    old_total: int,
    new_total: int,
) -> str:
    payload = "\0".join([project_id, old_hash, new_hash, str(old_total), str(new_total)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _confirmation_ok(
    confirmation: bool | str | None,
    *,
    project_id: str,
    old_hash: str,
    new_hash: str,
    old_total: int,
    new_total: int,
) -> bool:
    if confirmation is True:
        return True
    if isinstance(confirmation, str) and confirmation:
        return confirmation == _frozen_confirmation_token(
            project_id, old_hash, new_hash, old_total, new_total
        )
    return False


def _dedupe_frozen_sources(
    bucket_clips: list[SourceClip],
    split_clips: list[SourceClip],
) -> tuple[list[SourceClip], dict[str, str]]:
    out: list[SourceClip] = []
    source_by_path: dict[str, str] = {}
    seen: set[str] = set()
    for source, clips in (
        (BUCKET_FROZEN_SOURCE, bucket_clips),
        (SPLIT_FROZEN_SOURCE, split_clips),
    ):
        for clip in clips:
            key = _normalize_clip_path(clip.path)
            if key in seen:
                continue
            seen.add(key)
            source_by_path[key] = source
            out.append(clip)
    return out, source_by_path


def _filter_train_paths(
    by_class_train: dict[str, list[SourceClip]],
    classes: list[str],
    frozen_keys: set[str],
) -> dict[str, list[SourceClip]]:
    if not frozen_keys:
        return by_class_train
    filtered: dict[str, list[SourceClip]] = {c: [] for c in classes}
    for class_name in classes:
        for clip in by_class_train.get(class_name, []):
            if _normalize_clip_path(clip.path) in frozen_keys:
                continue
            filtered[class_name].append(clip)
    return filtered


def compose_dataset(
    cfg: Config,
    project_id: str,
    *,
    confirm_frozen_test_change: bool | str | None = None,
    build_split_test: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Build train.csv (the full train+val pool) / test.csv from data_library.

    Val is not composed here — sed.data.splits.split_pool_rows carves it out
    of train.csv live, at train time, seeded by cfg.training.seed.

    ``build_split_test`` gates the split-derived frozen test carve (see
    ``_resolve_split_frozen_test``): routine composition (the default) only
    reuses an already-built split, never creates or resizes one. Pass True
    only for an explicit, user-initiated "build frozen test set" action.

    ``progress``, if given, is called with ``{"phase", "phase_detail",
    "progress", ...}`` updates so a caller (e.g. the inspect server's compose
    job) can surface something better than a static "composing" spinner —
    class-by-class library scans and the frozen-test carve are the slow
    steps on large corpora.
    """
    layout = ProjectLayout(project_id)
    prune_project_metadata(layout)
    classes = list(cfg.model.classes)
    if not classes:
        raise ValueError(
            "model.classes is empty. Add at least one class to "
            "[dataset] target_classes in projects/<id>/config.toml."
        )

    split_test_enabled = effective_split_test_enabled(cfg)
    split_test_fraction = effective_split_test_fraction(cfg)
    will_resolve_split = split_test_enabled and split_test_fraction > 0.0

    base_weights = {
        "scan train": 0.40, "scan test": 0.28,
        "resolve frozen test": 0.22, "write dataset": 0.10,
    }
    selected_phases = ["scan train", "scan test"]
    if will_resolve_split:
        selected_phases.append("resolve frozen test")
    selected_phases.append("write dataset")
    notify = PhasedProgress(
        [(phase, base_weights[phase]) for phase in selected_phases], progress
    )

    selection = load_dataset_selection(layout)
    selection, auto_changed = ensure_auto_mined_negative_rules(
        cfg, layout, selection, write=True
    )
    if auto_changed:
        log.info("updated automatic mined _negative source rules")
    n_classes = len(classes)
    by_class_train: dict[str, list[SourceClip]] = {}
    for i, c in enumerate(classes):
        notify("scan train", fraction=i / n_classes, detail=f"scanning train: {c}", count=i, total=n_classes)
        by_class_train[c] = _library_clips_for_class(
            cfg, project_id, c, split="train", selection=selection
        )
    notify("scan train", fraction=1.0, detail="train scan complete", count=n_classes, total=n_classes)
    by_class_test: dict[str, list[SourceClip]] = {}
    for i, c in enumerate(classes):
        notify("scan test", fraction=i / n_classes, detail=f"scanning test: {c}", count=i, total=n_classes)
        by_class_test[c] = _library_clips_for_class(
            cfg, project_id, c, split="test", selection=selection
        )
    notify("scan test", fraction=1.0, detail="test scan complete", count=n_classes, total=n_classes)

    excluded_paths = load_excluded_clip_paths(layout)
    excluded_by_origin: dict[str, dict[str, int]] = {
        c: {"sounds": 0, "derived": 0, "mined": 0} for c in classes
    }
    if excluded_paths:
        for class_name in classes:
            original = list(by_class_train[class_name])
            by_class_train[class_name], _ = _filter_excluded(original, excluded_paths)
            for clip in original:
                if _normalize_clip_path(clip.path) in excluded_paths:
                    if clip.category == DERIVED_CATEGORY:
                        bucket = "derived"
                    elif clip.category == MINED_CATEGORY:
                        bucket = "mined"
                    else:
                        bucket = "sounds"
                    excluded_by_origin[class_name][bucket] += 1

    bucket_frozen_clips = [clip for clips in by_class_test.values() for clip in clips]
    bucket_frozen_keys = {_normalize_clip_path(clip.path) for clip in bucket_frozen_clips}
    # Explicit frozen-test buckets win when a clip is reachable through both
    # selected test rules and train rules, including pretraining corpus rules.
    by_class_train = _filter_train_paths(by_class_train, classes, bucket_frozen_keys)

    split_frozen_clips: list[SourceClip] = []
    split_record: dict | None = None
    split_record_needs_write = False
    if will_resolve_split:
        notify(
            "resolve frozen test", fraction=0.0,
            detail="carving split-derived frozen test" if build_split_test
            else "resolving split-derived frozen test",
        )
        split_frozen_clips, by_class_train, split_record, split_record_needs_write = (
            _resolve_split_frozen_test(
                layout, project_id, by_class_train, classes, split_test_fraction,
                allow_carve=build_split_test,
            )
        )
        notify("resolve frozen test", fraction=1.0, detail="frozen test split resolved")

    frozen_test_clips, frozen_source_by_path = _dedupe_frozen_sources(
        bucket_frozen_clips, split_frozen_clips
    )
    frozen_test_by_class: dict[str, int] = {c: 0 for c in classes}
    for c in frozen_test_clips:
        frozen_test_by_class[c.class_name] = frozen_test_by_class.get(c.class_name, 0) + 1
    has_frozen_test = len(frozen_test_clips) > 0

    pool_clips: list[SourceClip] = []
    counts_by_origin: dict[str, dict[str, int]] = {}
    for class_name in classes:
        clips = by_class_train[class_name]
        pool_clips.extend(clips)
        sounds = sum(1 for c in clips if c.category == RAW_CATEGORY)
        derived = sum(1 for c in clips if c.category == DERIVED_CATEGORY)
        mined = sum(1 for c in clips if c.category == MINED_CATEGORY)
        counts_by_origin[class_name] = {
            "sounds": sounds,
            "derived": derived,
            "mined": mined,
            "total": len(clips),
        }

    log.info("--- composed dataset summary ---")
    for class_name, c in counts_by_origin.items():
        log.info(
            "  %-22s  sounds=%3d  derived=%3d  mined=%3d  total=%4d",
            class_name, c["sounds"], c["derived"], c["mined"], c["total"],
        )
        excluded_total = sum(excluded_by_origin[class_name].values())
        if excluded_total:
            log.info("  %-22s  excluded=%3d", class_name, excluded_total)
    if has_frozen_test:
        source_counts = _frozen_source_counts(frozen_source_by_path)
        source_bits = [
            f"selected buckets={source_counts.get(BUCKET_FROZEN_SOURCE, 0)}",
            f"split-derived={source_counts.get(SPLIT_FROZEN_SOURCE, 0)}",
        ]
        src = ", ".join(source_bits)
        log.info("--- frozen test set (%s) ---", src)
        for class_name in classes:
            log.info("  %-22s  %d", class_name, frozen_test_by_class.get(class_name, 0))
    if "_negative" in classes and counts_by_origin.get("_negative", {}).get("total", 0) == 0:
        log.warning(
            "_negative is empty. Assign one or more library labels to "
            "_negative in the Inspect Library tab or compatibility "
            "[dataset.sources]."
        )
    for class_name in classes:
        if class_name == "_negative":
            continue
        if counts_by_origin[class_name]["total"] == 0:
            log.warning(
                "class '%s' has ZERO clips. Either record some "
                "(`sed record --class %s`) or assign a library bucket "
                "in the Inspect Library tab.",
                class_name, class_name,
            )

    # Short raw clips (e.g. imported legacy 0.4 s recordings) get zero-padded
    # to the model window at load time — the pad is digital silence live
    # windows never contain, so flag them. Derived clips are exempt: the
    # overlay pipeline already fills the full window with background.
    min_train_duration_s = 0.95 * float(cfg.features.window_seconds)
    for class_name in classes:
        short_raw = sum(
            1
            for c in by_class_train[class_name]
            if c.category == RAW_CATEGORY and 0.0 < c.duration_s < min_train_duration_s
        )
        if short_raw:
            log.warning(
                "class '%s' has %d raw clip(s) shorter than the %.2fs model "
                "window — they will be zero-padded in training. Use short "
                "clips as [augmentation.combinations] bases (overlay fills "
                "the window with background) instead of assigning their "
                "label to a class directly.",
                class_name, short_raw, float(cfg.features.window_seconds),
            )

    # No train/val split here: train.csv is the whole non-test pool. Val is
    # carved out live at train time (see sed.data.splits.split_pool_rows),
    # seeded by cfg.training.seed — an ordinary, on-the-fly training-pipeline
    # split, recomputed every run instead of pinned to compose time.
    train = pool_clips
    test = list(frozen_test_clips)
    derived_origins = {DERIVED_CATEGORY, "augmented"}
    _preview_train, _preview_val, split_diagnostics = split_train_val_detailed(
        [c for c in train if c.origin not in derived_origins],
        [c for c in train if c.origin in derived_origins],
        val_frac=float(getattr(cfg.dataset, "val_fraction", 0.15) or 0.15),
        seed=int(getattr(cfg.training, "seed", 42) or 42),
    )
    if has_frozen_test:
        log.info("  test: frozen clips — %d clips", len(test))
    else:
        log.info("  test: empty — post-training eval will be skipped")

    test_hash = _test_set_hash(test) if has_frozen_test else ""
    frozen_manifest = _build_frozen_manifest(
        project_id=project_id,
        test=test,
        source_by_path=frozen_source_by_path,
        classes=classes,
        test_hash=test_hash,
        split_record=split_record,
    )
    source_counts = frozen_manifest["source_counts"]
    source_by_class = frozen_manifest["source_by_class"]
    if source_counts.get(BUCKET_FROZEN_SOURCE, 0) and source_counts.get(SPLIT_FROZEN_SOURCE, 0):
        frozen_source = "combined"
    elif source_counts.get(SPLIT_FROZEN_SOURCE, 0):
        frozen_source = SPLIT_FROZEN_SOURCE
    elif source_counts.get(BUCKET_FROZEN_SOURCE, 0):
        frozen_source = BUCKET_FROZEN_SOURCE
    else:
        frozen_source = ""

    frozen_summary = {
        "present": has_frozen_test,
        "source": frozen_source,
        "by_class": frozen_test_by_class,
        "total": len(frozen_test_clips),
        "hash": test_hash,
        "source_counts": source_counts,
        "source_by_class": source_by_class,
        "manifest": str(frozen_test_manifest_path(layout)),
        "split_manifest": str(split_frozen_test_path(layout)) if split_record else "",
        "split_diagnostics": frozen_manifest.get("split_diagnostics", {}),
        "source_quality": (
            "combined"
            if source_counts.get(BUCKET_FROZEN_SOURCE, 0) and source_counts.get(SPLIT_FROZEN_SOURCE, 0)
            else (
                "randomized_split"
                if source_counts.get(SPLIT_FROZEN_SOURCE, 0)
                else ("pinned_holdouts" if source_counts.get(BUCKET_FROZEN_SOURCE, 0) else "")
            )
        ),
    }

    existing_frozen = _existing_frozen_snapshot(layout)
    old_hash = str(existing_frozen.get("hash") or "")
    old_total = int(existing_frozen.get("total") or 0)
    if (
        old_total > 0
        and old_hash != test_hash
        and not _confirmation_ok(
            confirm_frozen_test_change,
            project_id=project_id,
            old_hash=old_hash,
            new_hash=test_hash,
            old_total=old_total,
            new_total=len(test),
        )
    ):
        token = _frozen_confirmation_token(
            project_id, old_hash, test_hash, old_total, len(test)
        )
        return {
            "schema_version": 2,
            "needs_confirmation": True,
            "confirmation_token": token,
            "confirmation_reason": "Frozen test set would change.",
            "old_frozen_test": {
                "hash": old_hash,
                "total": old_total,
                "source_counts": existing_frozen.get("source_counts", {}),
            },
            "new_frozen_test": frozen_summary,
            "frozen_test": frozen_summary,
            "split_config": {
                "split_test_enabled": split_test_enabled,
                "split_test_fraction": split_test_fraction,
                "pretraining_test_enabled": split_test_enabled,
                "pretraining_test_fraction": split_test_fraction,
            },
            "split_diagnostics": split_diagnostics,
            "splits": {"train": len(train), "test": len(test)},
        }

    # ---- Write CSVs ----
    # train.csv is the whole non-test pool (train+val undivided); val is
    # carved out live at train time, so it has no CSV of its own here.
    notify("write dataset", fraction=0.0, detail="writing train/test CSVs")
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    stale_val = layout.dataset_dir / "val.csv"
    if stale_val.exists():
        stale_val.unlink()
    for name, items in [("train", train), ("test", test)]:
        out_csv = layout.dataset_dir / f"{name}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["path", "class_name", "session_id", "sample_rate", "duration_s",
                 "is_target", "origin", "origin_class", "source_stem",
                 "split_bucket", "split_group", "original_source",
                 "augmentation_family", "source_split_group", "frozen_source"]
            )
            for c in items:
                frozen_source_row = (
                    frozen_source_by_path.get(_normalize_clip_path(c.path), "")
                    if name == "test" else ""
                )
                w.writerow([c.path, c.class_name, c.session_id, c.sample_rate,
                            f"{c.duration_s:.4f}", int(c.is_target), c.origin,
                            c.origin_class or c.class_name, c.source_stem,
                            c.split_bucket or c.origin_class or c.origin,
                            c.split_group or split_group_key(c),
                            c.original_source or c.path,
                            c.augmentation_family,
                            c.source_split_group or c.split_group or split_group_key(c),
                            frozen_source_row])

    class_index = {name: i for i, name in enumerate(classes)}
    with open(layout.dataset_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "class_index": class_index}, f, indent=2)

    snapshot = {
        "schema_version": 2,
        "total_clips": len(pool_clips) + len(test),
        "by_class": {c: counts_by_origin[c]["total"] for c in classes},
        "by_origin": counts_by_origin,
        "excluded": {
            "manifest": str(exclusion_manifest_path(layout)),
            "total": sum(sum(v.values()) for v in excluded_by_origin.values()),
            "by_origin": excluded_by_origin,
        },
        "frozen_test": frozen_summary,
        "split_config": {
            "split_test_enabled": split_test_enabled,
            "split_test_fraction": split_test_fraction,
            "pretraining_test_enabled": split_test_enabled,
            "pretraining_test_fraction": split_test_fraction,
        },
        "split_diagnostics": split_diagnostics,
        "splits": {"train": len(train), "test": len(test)},
        "source_hashes": _snapshot_source_hashes({"train": train, "test": test}),
        # Staleness marker for the inspect UI: compares against the current
        # selection fingerprint to flag "recompose needed" after rule edits.
        "selection_fingerprint": selection_compose_fingerprint(
            selection, cfg, layout=layout
        ),
    }
    with open(layout.dataset_dir / "dataset_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    _write_frozen_manifest(layout, frozen_manifest)
    if split_record is not None and split_record_needs_write:
        _write_split_frozen_manifest(layout, split_record)
    clear_dataset_dirty(layout)
    notify("write dataset", fraction=1.0, detail="dataset written")

    return snapshot
