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
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from sed.config import Config, ProjectLayout
from sed.data.library import (
    DERIVED_CATEGORY,
    MINED_CATEGORY,
    RAW_CATEGORY,
    Library,
    LibraryClip,
    clip_matches_mic_rule,
    clip_mic_channels,
)
from sed.data.splits import split_train_val_no_leakage

log = logging.getLogger(__name__)


EXCLUSIONS_FILENAME = "excluded_clips.json"
SELECTION_FILENAME = "dataset_selection.json"


def _empty_dataset_selection() -> dict:
    return {
        "schema_version": 1,
        "assignments": {},
        "clip_assignments": {},
        "clip_overrides": {},
        "augmentation_combinations": {},
        "augmentation_recipes": [],
        "augmentation_settings": {},
    }


def exclusion_manifest_path(layout: ProjectLayout) -> Path:
    return layout.dataset_dir / EXCLUSIONS_FILENAME


def selection_manifest_path(layout: ProjectLayout) -> Path:
    return layout.dataset_dir / SELECTION_FILENAME


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
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _normalize_clip_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


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
    paths = {str(clip.absolute_path(lib.root).resolve()) for clip in clips}

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


def _read_sidecar(wav_path: Path) -> dict | None:
    side = wav_path.with_suffix(".json")
    if not side.exists():
        return None
    try:
        with open(side, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_source_hashes(splits: dict[str, list["SourceClip"]]) -> dict[str, str]:
    source_hashes: dict[str, str] = {}
    for clips in splits.values():
        for clip in clips:
            path = Path(clip.path)
            if clip.path in source_hashes:
                continue
            try:
                source_hashes[clip.path] = _file_sha256(path)
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
        session_id = meta.get("session_id", wav.stem) if meta else wav.stem
        out.append(
            SourceClip(
                path=str(wav.resolve()),
                class_name=class_name,
                session_id=session_id,
                sample_rate=info.samplerate,
                duration_s=info.frames / info.samplerate if info.samplerate else 0.0,
                is_target=is_target,
                origin=origin,
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
        source_name = (meta or {}).get("source", "")
        session_id = f"aug-of-{source_name}" if source_name else f"aug-{wav.stem}"
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
            )
        )
    return out


def _positive_test_labels_for_class(
    class_name: str,
    positive_rules: dict[str, list[dict]] | None,
) -> list[str]:
    """Return frozen-test positive folder names that map to ``class_name``.

    Legacy helper retained for older tests/imports. New composition reads
    frozen test clips from library sidecars with ``dataset_role="test"``.
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

    New dataset composition ignores project-local WAVs and uses library
    sidecars with ``dataset_role="test"`` instead.
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
    dataset_role: str | None = None,
) -> list[dict]:
    assignments = selection.get("assignments", {}) if isinstance(selection, dict) else {}
    rows = assignments.get(class_name, [])
    if isinstance(rows, dict):
        rows = [rows]
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        split = str(row.get("split") or "").strip().lower()
        if dataset_role is not None and split and split != dataset_role:
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


def _clip_matches_ref(clip: LibraryClip, ref: dict, base: Path) -> bool:
    uuid = str(ref.get("uuid") or "")
    if uuid and clip.uuid == uuid:
        return True
    relpath = str(ref.get("relpath") or "")
    if relpath and clip.relpath == relpath:
        return True
    raw_path = str(ref.get("path") or "")
    if raw_path:
        try:
            return Path(raw_path).expanduser().resolve() == clip.absolute_path(base).resolve()
        except Exception:  # noqa: BLE001
            return False
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
    dataset_role: str,
) -> list[dict]:
    selected = _selection_rules(selection, class_name, dataset_role=dataset_role)
    if selected or _selection_is_ui_managed(selection):
        return selected
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


def _dataset_role_for_rule(rule: dict, dataset_role: str) -> str:
    raw = str(rule.get("dataset_role") or "").strip()
    if raw in {"train", "test", "ignore"}:
        return raw
    return dataset_role


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
        clip.dataset_role,
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
    session_text = f"{clip.session_id or ''} {clip.recorded_at or ''}".lower()
    if session_q and session_q not in session_text:
        return False
    return True


def _append_clip_once(
    out: list[SourceClip],
    seen: set[str],
    clip: LibraryClip,
    class_name: str,
    *,
    origin: str,
    base: Path,
) -> None:
    path = str(clip.absolute_path(base).resolve())
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
    return SourceClip(
        path=str(clip.absolute_path(base).resolve()),
        class_name=class_name,
        session_id=clip.session_id or clip.uuid,
        sample_rate=clip.sample_rate,
        duration_s=clip.duration_s,
        is_target=class_name != "_negative",
        origin=origin,
        source_stem=Path(clip.source_path).stem if clip.source_path else "",
    )


def _library_clips_for_class(
    cfg: Config,
    project_id: str,
    class_name: str,
    *,
    dataset_role: str,
    selection: dict | None = None,
    include_derived: bool = True,
) -> list[SourceClip]:
    selection = selection or {}
    lib = Library()
    rules = _rules_for_class(
        cfg,
        selection,
        class_name,
        dataset_role=dataset_role,
    )
    selected_clip_refs = _selection_clip_refs(selection, class_name)
    out: list[SourceClip] = []
    seen: set[str] = set()

    if selected_clip_refs:
        categories = [RAW_CATEGORY]
        if include_derived:
            categories.append(DERIVED_CATEGORY)
        candidates = [
            clip
            for category in categories
            for clip in lib.query(
                category=category,
                dataset_role=dataset_role,
                review_status="accepted",
            )
        ]
        for clip in candidates:
            if not any(_clip_matches_ref(clip, ref, lib.root) for ref in selected_clip_refs):
                continue
            _append_clip_once(
                out,
                seen,
                clip,
                class_name,
                origin=clip.category,
                base=lib.root,
            )

    for rule in rules:
        labels = _rule_labels(rule)
        if not labels:
            continue
        category = _category_for_rule(rule)
        if category == MINED_CATEGORY:
            continue
        project_ids = _project_ids_for_rule(rule, project_id, test=dataset_role == "test")
        density = rule.get("density")
        role = _dataset_role_for_rule(rule, dataset_role)
        candidates = lib.query(
            category=category,
            project_ids=project_ids,
            dataset_role=role,
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
                # split would double-count with channel 0 of the original.
                if clip.clip_kind == "channel_split":
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
                base=lib.root,
            )

    should_include_target_derived = include_derived and not _selection_is_ui_managed(selection)
    if should_include_target_derived:
        for clip in lib.query(
            category=DERIVED_CATEGORY,
            label=class_name,
            project_ids={project_id},
            dataset_role=dataset_role,
            review_status="accepted",
        ):
            _append_clip_once(
                out,
                seen,
                clip,
                class_name,
                origin=DERIVED_CATEGORY,
                base=lib.root,
            )
    return out


def compose_dataset(cfg: Config, project_id: str, split_seed: int = 42) -> dict:
    """Build train.csv / val.csv / test.csv from data_library only."""
    layout = ProjectLayout(project_id)
    prune_project_metadata(layout)
    classes = list(cfg.model.classes)
    if not classes:
        raise ValueError(
            "model.classes is empty. Add at least one class to "
            "[dataset] target_classes in projects/<id>/config.toml."
        )

    selection = load_dataset_selection(layout)
    by_class_train: dict[str, list[SourceClip]] = {
        c: _library_clips_for_class(
            cfg, project_id, c, dataset_role="train", selection=selection
        )
        for c in classes
    }
    by_class_test: dict[str, list[SourceClip]] = {
        c: _library_clips_for_class(
            cfg, project_id, c, dataset_role="test", selection=selection
        )
        for c in classes
    }

    excluded_paths = load_excluded_clip_paths(layout)
    excluded_by_origin: dict[str, dict[str, int]] = {
        c: {"sounds": 0, "derived": 0} for c in classes
    }
    if excluded_paths:
        for class_name in classes:
            original = list(by_class_train[class_name])
            by_class_train[class_name], _ = _filter_excluded(original, excluded_paths)
            for clip in original:
                if _normalize_clip_path(clip.path) in excluded_paths:
                    bucket = "derived" if clip.origin == DERIVED_CATEGORY else "sounds"
                    excluded_by_origin[class_name][bucket] += 1

    frozen_test_clips = [clip for clips in by_class_test.values() for clip in clips]
    frozen_test_by_class: dict[str, int] = {c: 0 for c in classes}
    for c in frozen_test_clips:
        frozen_test_by_class[c.class_name] = frozen_test_by_class.get(c.class_name, 0) + 1
    has_frozen_test = len(frozen_test_clips) > 0

    pool_clips: list[SourceClip] = []
    counts_by_origin: dict[str, dict[str, int]] = {}
    for class_name in classes:
        clips = by_class_train[class_name]
        pool_clips.extend(clips)
        sounds = sum(1 for c in clips if c.origin == RAW_CATEGORY)
        derived = sum(1 for c in clips if c.origin == DERIVED_CATEGORY)
        counts_by_origin[class_name] = {
            "sounds": sounds,
            "derived": derived,
            "total": len(clips),
        }

    log.info("--- composed dataset summary ---")
    for class_name, c in counts_by_origin.items():
        log.info(
            "  %-22s  sounds=%3d  derived=%3d  total=%4d",
            class_name, c["sounds"], c["derived"], c["total"],
        )
        excluded_total = sum(excluded_by_origin[class_name].values())
        if excluded_total:
            log.info("  %-22s  excluded=%3d", class_name, excluded_total)
    if has_frozen_test:
        log.info("--- frozen test set (data_library dataset_role=test) ---")
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
            if c.origin == RAW_CATEGORY and 0.0 < c.duration_s < min_train_duration_s
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

    # Leakage-free split: raw clips split by recording session; derived/
    # augmented clips train-only, and dropped when their source is in val.
    derived_origins = {DERIVED_CATEGORY, "augmented"}
    raw_pool = [c for c in pool_clips if c.origin not in derived_origins]
    derived_pool = [c for c in pool_clips if c.origin in derived_origins]
    train, val, split_modes = split_train_val_no_leakage(
        raw_pool, derived_pool, val_frac=0.15, seed=split_seed,
    )
    test = list(frozen_test_clips)
    log.info("--- split modes per class ---")
    for class_name, mode in split_modes.items():
        log.info("  %-22s  %s", class_name, mode)
    if has_frozen_test:
        log.info("  test: frozen library test clips — %d clips", len(test))
    else:
        log.info("  test: empty (no library clips with dataset_role=test) — post-training eval will be skipped")

    # ---- Write CSVs ----
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    for name, items in [("train", train), ("val", val), ("test", test)]:
        out_csv = layout.dataset_dir / f"{name}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["path", "class_name", "session_id", "sample_rate", "duration_s",
                 "is_target", "origin"]
            )
            for c in items:
                w.writerow([c.path, c.class_name, c.session_id, c.sample_rate,
                            f"{c.duration_s:.4f}", int(c.is_target), c.origin])

    class_index = {name: i for i, name in enumerate(classes)}
    with open(layout.dataset_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "class_index": class_index}, f, indent=2)

    test_hash = _test_set_hash(test) if has_frozen_test else ""
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
        "frozen_test": {
            "present": has_frozen_test,
            "by_class": frozen_test_by_class,
            "total": len(frozen_test_clips),
            "hash": test_hash,
        },
        "split_modes": split_modes,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "source_hashes": _snapshot_source_hashes({"train": train, "val": val, "test": test}),
    }
    with open(layout.dataset_dir / "dataset_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    return snapshot
