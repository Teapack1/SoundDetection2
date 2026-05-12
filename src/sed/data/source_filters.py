"""Shared augmentation-recipe helpers: source-filter resolution + settings snapshot.

Both the inspect UI (recipe counts while editing) and the augmentation
materializer (clip selection at generation time) resolve recipe filter rules
through this module, so the two can never disagree. Imports must stay light —
library/pretraining only, no DSP stack — because the inspect server calls this
on request threads during page loads.
"""

from __future__ import annotations

import os
from pathlib import Path

from sed.config import Config
from sed.data.library import (
    DERIVED_CATEGORY,
    MINED_CATEGORY,
    PRETRAIN_CATEGORY,
    RAW_CATEGORY,
    Library,
    clip_matches_mic_rule,
    clip_matches_session,
    clip_mic_channels,
)
from sed.data.pretraining import shared_index

_RAW_BUCKETS = {"", "raw", "sound", "sounds", "positive", "positives", "negative", "negatives"}
_DERIVED_BUCKETS = {"derived", "augmentation", "augmented"}
_MINED_BUCKETS = {"mined", "false_positive", "false-positives"}
_PRETRAIN_BUCKETS = {"pretraining", "pretrain"}


def category_for_bucket(rule: dict) -> str | None:
    bucket = str(rule.get("source_bucket") or rule.get("category") or "raw").strip().lower()
    if bucket in _RAW_BUCKETS:
        return RAW_CATEGORY
    if bucket in _DERIVED_BUCKETS:
        return DERIVED_CATEGORY
    if bucket in _MINED_BUCKETS:
        return MINED_CATEGORY
    if bucket in _PRETRAIN_BUCKETS:
        return PRETRAIN_CATEGORY
    return None


def clip_matches_filter_rule(
    clip,
    rule: dict,
    *,
    label: str,
    project_id: str,
    frozen_test: bool = False,
) -> bool:
    """Whether one already-resolved library clip matches a source rule.

    Keeping this predicate separate from inventory enumeration lets targeted
    mutations adjust only the rules/recipes touched by a deleted clip while
    using exactly the same semantics as augmentation generation and recounts.
    """
    rule = rule or {}
    category = category_for_bucket(rule)
    if category is None or str(getattr(clip, "category", "")) != category:
        return False

    raw_project_ids = rule.get("project_ids", rule.get("project_id"))
    if raw_project_ids == "*":
        project_ids = None
    elif isinstance(raw_project_ids, str) and raw_project_ids:
        project_ids = {raw_project_ids}
    elif isinstance(raw_project_ids, list) and raw_project_ids:
        project_ids = {str(item) for item in raw_project_ids if str(item)}
    else:
        project_ids = (
            {project_id}
            if category in {DERIVED_CATEGORY, MINED_CATEGORY} or frozen_test
            else {"_library", project_id}
        )
    if project_ids is not None and str(getattr(clip, "project_id", "")) not in project_ids:
        return False

    review_status = str(rule.get("review_status") or rule.get("status") or "accepted").strip()
    if review_status in {"", "*", "all"}:
        review_status = "accepted"
    if str(getattr(clip, "review_status", "")) != review_status:
        return False
    density = str(rule.get("density") or "").strip()
    if density and str(getattr(clip, "density", "")) != density:
        return False

    raw_tags = rule.get("tags_all", rule.get("tags"))
    if isinstance(raw_tags, str):
        tags_all = [
            part.strip().lstrip("#")
            for part in raw_tags.replace(";", ",").split(",")
            if part.strip()
        ]
    elif isinstance(raw_tags, list):
        tags_all = [str(part).strip().lstrip("#") for part in raw_tags if str(part).strip()]
    else:
        tags_all = []
    if tags_all and not set(tags_all).issubset(set(getattr(clip, "tags", None) or [])):
        return False

    raw_labels = rule.get("labels", rule.get("label"))
    if isinstance(raw_labels, str):
        labels = [raw_labels]
    elif isinstance(raw_labels, list):
        labels = [str(item) for item in raw_labels if str(item)]
    else:
        labels = [label]
    if labels and "*" not in labels and str(getattr(clip, "label", "")) not in labels:
        return False

    if category == PRETRAIN_CATEGORY and rule.get("sample_rate") not in (None, ""):
        try:
            if int(getattr(clip, "sample_rate", 0)) != int(rule["sample_rate"]):
                return False
        except (TypeError, ValueError):
            pass
    channel_index = None
    if rule.get("channel_index") not in (None, ""):
        try:
            channel_index = int(rule.get("channel_index"))
        except (TypeError, ValueError):
            channel_index = None
    if channel_index is not None and getattr(clip, "channel_index", None) != channel_index:
        return False
    if not clip_matches_mic_rule(clip, rule):
        return False

    search = str(rule.get("search") or "").strip().lower()
    if search:
        haystack = " ".join(
            str(part or "").lower()
            for part in (
                Path(str(getattr(clip, "relpath", ""))).name,
                getattr(clip, "relpath", ""),
                getattr(clip, "label", ""),
                getattr(clip, "project_id", ""),
                getattr(clip, "density", ""),
                getattr(clip, "review_status", ""),
                getattr(clip, "session_id", ""),
                getattr(clip, "recorded_at", ""),
                ",".join(str(ch) for ch in clip_mic_channels(clip)),
                " ".join(getattr(clip, "tags", None) or []),
            )
        )
        if search not in haystack:
            return False
    session = str(rule.get("session") or rule.get("session_or_date") or "").strip().lower()
    return not session or clip_matches_session(clip, session)


def library_paths_for_filter(
    lib: Library,
    rule: dict,
    *,
    label: str,
    project_id: str,
    frozen_test: bool = False,
) -> list[tuple[Path, str]]:
    """Resolve one recipe filter rule to ``(wav_path, label)`` source clips.

    A rule resolves to the same audio whichever split the caller is building —
    a rule's ``split`` chooses the composition destination, not the material.
    ``frozen_test`` only narrows project scope: a holdout batch draws from the
    current project alone, never from the shared ``_library`` pool.
    """
    rule = rule or {}
    category = category_for_bucket(rule)
    if category is None:
        return []

    raw_project_ids = rule.get("project_ids", rule.get("project_id"))
    if raw_project_ids == "*":
        project_ids = None
    elif isinstance(raw_project_ids, str) and raw_project_ids:
        project_ids = {raw_project_ids}
    elif isinstance(raw_project_ids, list) and raw_project_ids:
        project_ids = {str(item) for item in raw_project_ids if str(item)}
    else:
        project_ids = (
            {project_id}
            if category in {DERIVED_CATEGORY, MINED_CATEGORY} or frozen_test
            else {"_library", project_id}
        )

    review_status = str(rule.get("review_status") or rule.get("status") or "accepted").strip()
    if review_status in {"", "*", "all"}:
        review_status = "accepted"
    density = str(rule.get("density") or "").strip() or None

    raw_tags = rule.get("tags_all", rule.get("tags"))
    if isinstance(raw_tags, str):
        tags_all = [
            part.strip().lstrip("#")
            for part in raw_tags.replace(";", ",").split(",")
            if part.strip()
        ]
    elif isinstance(raw_tags, list):
        tags_all = [str(part).strip().lstrip("#") for part in raw_tags if str(part).strip()]
    else:
        tags_all = None

    raw_labels = rule.get("labels", rule.get("label"))
    if isinstance(raw_labels, str):
        labels = [raw_labels]
    elif isinstance(raw_labels, list):
        labels = [str(item) for item in raw_labels if str(item)]
    else:
        labels = [label]
    all_labels = not labels or "*" in labels

    if category == PRETRAIN_CATEGORY:
        # Pretraining clips live in a dedicated index, not the manifest (see
        # sed.data.pretraining). Synthesized clips are train/accepted/_library,
        # so the query-equivalent predicates below make e.g. frozen-test rules
        # resolve to nothing instead of silently pulling corpus clips.
        label_filter = [] if all_labels else labels
        candidates = shared_index().clips_for_labels(label_filter, probe=True)
        want_sr = rule.get("sample_rate")
        if want_sr not in (None, ""):
            try:
                want_sr = int(want_sr)
                candidates = [c for c in candidates if int(c.sample_rate) == want_sr]
            except (TypeError, ValueError):
                pass
        candidates = [
            c
            for c in candidates
            if c.review_status == review_status
            and (project_ids is None or c.project_id in project_ids)
            and (density is None or c.density == density)
            and (not tags_all or set(tags_all).issubset(set(c.tags or [])))
        ]
    else:
        candidates = lib.query(
            category=category,
            review_status=review_status,
            project_ids=project_ids,
            density=density,
            tags_all=tags_all,
        )

    out: list[tuple[Path, str]] = []
    for clip in candidates:
        if not clip_matches_filter_rule(
            clip,
            rule,
            label=label,
            project_id=project_id,
            frozen_test=frozen_test,
        ):
            continue
        out.append((clip.absolute_path(lib.root), clip.label))
    return out


def library_paths_for_filters(
    lib: Library,
    label: str,
    *,
    project_id: str,
    frozen_test: bool,
    filters: list[dict] | None,
) -> list[tuple[Path, str]]:
    """Union of :func:`library_paths_for_filter` over several rules, deduped."""
    rows = filters or [{}]
    out: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for rule in rows:
        for path, actual_label in library_paths_for_filter(
            lib,
            rule,
            label=label,
            project_id=project_id,
            frozen_test=frozen_test,
        ):
            # All candidate paths are joined from the same library roots, so a
            # string normpath is a sufficient dedupe key; Path.resolve() here
            # cost ~18s per call on a 75k-clip corpus (syscalls per component).
            key = os.path.normpath(str(path))
            if key in seen:
                continue
            seen.add(key)
            out.append((path, actual_label))
    return out


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def augmentation_settings_snapshot(cfg: Config | None) -> dict:
    """Per-recipe snapshot of every augmentation knob the generator honors."""
    if cfg is None:
        return {}
    aug = cfg.augmentation
    return {
        "enabled": bool(aug.enabled),
        "seed": int(aug.seed),
        "samples_per_source": float(aug.samples_per_source),
        "pitch_shift_enabled": bool(aug.pitch_shift_enabled),
        "semitones_range": [float(aug.semitones_range[0]), float(aug.semitones_range[1])],
        "overlay_enabled": bool(aug.overlay_enabled),
        "snr_db_range": [float(aug.snr_db_range[0]), float(aug.snr_db_range[1])],
        "scale_noise_to_snr": bool(aug.scale_noise_to_snr),
        "prevent_clipping": bool(aug.prevent_clipping),
        "transient_window_ms": int(aug.transient_window_ms),
        "mask_filter_enabled": bool(aug.mask_filter_enabled),
        "mask_candidate_attempts": int(aug.mask_candidate_attempts),
        "mask_min_mean_db": _optional_float(aug.mask_min_mean_db),
        "mask_min_p25_db": _optional_float(aug.mask_min_p25_db),
        "mask_min_band_db": _optional_float(aug.mask_min_band_db),
        "mask_min_effective_snr_db": _optional_float(aug.mask_min_effective_snr_db),
        "mask_band_count": int(aug.mask_band_count),
        "mask_top_frames": int(aug.mask_top_frames),
        "mix_map_enabled": bool(aug.mix_map_enabled),
    }
