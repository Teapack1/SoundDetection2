"""Dataset inspection: visualize and quality-check recorded clips.

Generates an HTML report with:
  - Class distribution (counts, durations)
  - Per-clip thumbnails: log-mel spectrogram + waveform + audio playback

The report is a single self-contained HTML file (with inline base64 PNGs for
spectrograms and a copy of audio files served from a relative path).

Interactive mode (``--serve``) lets reviewers tick dataset/augmented clips for
soft exclusion from the next compose/train run. The "Delete selected" button
still permanently removes WAVs and sidecars from disk.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn

from sed.audio.features import EFFICIENTAT_DISPLAY_RANGE, waveform_to_logmel
from sed.config import Config, ProjectLayout

log = logging.getLogger(__name__)


def _load_exclusions_for_render(layout: ProjectLayout) -> set[str]:
    from sed.data.compose import load_excluded_clip_paths
    return load_excluded_clip_paths(layout)


# ---------------------------------------------------------------------------
# Plotting helpers — return base64 PNG strings
# ---------------------------------------------------------------------------


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Preview spectrograms must show what the model consumes. The active project's
# feature contract is registered here at render/serve time; thumbnail helpers
# read it instead of the historical waveform_to_logmel defaults (which were the
# legacy frontend at the file's native sample rate — NOT the model view).
_PREVIEW_FEATURES: dict = {}

# Fixed display range so brightness is comparable across clips instead of
# per-image autoscale. Quiet clips genuinely render darker than loud ones:
# that is the model's view (absolute level is signal), but nothing above the
# digital-silence floor is clipped. Shared with the augmentation mix maps.
_EFFICIENTAT_PREVIEW_RANGE = EFFICIENTAT_DISPLAY_RANGE


def set_preview_features(cfg: Config | None) -> None:
    global _PREVIEW_FEATURES
    if cfg is None:
        _PREVIEW_FEATURES = {}
        return
    f = cfg.features
    _PREVIEW_FEATURES = {
        "sample_rate": int(cfg.model.sample_rate),
        "frontend": str(f.frontend),
        "n_fft": int(f.n_fft),
        "win_length": int(f.win_length),
        "hop_length": int(f.hop_length),
        "n_mels": int(f.n_mels),
        "fmin": int(f.fmin),
        "fmax": int(f.fmax),
        # Display range is part of the rendered art — changing it must also
        # invalidate cached thumbnails.
        "display_range": list(_EFFICIENTAT_PREVIEW_RANGE),
    }
    _ensure_thumbnail_cache_matches_features()


def _ensure_thumbnail_cache_matches_features() -> None:
    """Spec thumbnails are a cache of the preview pipeline. When the feature
    contract changes (e.g. legacy → efficientat), drop them once so every
    preview regenerates in the model's view."""
    from sed.config import LIBRARY_DIR

    if not _PREVIEW_FEATURES:
        return
    signature = json.dumps(_PREVIEW_FEATURES, sort_keys=True)
    marker = LIBRARY_DIR / ".spec_thumbs_signature"
    try:
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == signature:
            return
        count = 0
        if LIBRARY_DIR.exists():
            for thumb in LIBRARY_DIR.rglob("*.spec.png"):
                thumb.unlink(missing_ok=True)
                count += 1
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(signature, encoding="utf-8")
        if count:
            log.info(
                "preview feature contract changed — cleared %d cached spectrogram thumbnail(s)",
                count,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("thumbnail cache signature check failed: %s", exc)


def _preview_logmel(wave: np.ndarray, sr: int) -> tuple[np.ndarray, float | None, float | None]:
    """Model-view log-mel for previews: resampled to the model rate and run
    through the same frontend the net consumes. Returns (spec, vmin, vmax)."""
    p = _PREVIEW_FEATURES
    if not p:
        return waveform_to_logmel(wave, sample_rate=sr)[0], None, None
    target_sr = int(p["sample_rate"])
    if int(sr) != target_sr:
        from sed.audio.resample import resample_audio
        wave = resample_audio(wave, orig_sr=int(sr), target_sr=target_sr)
    spec = waveform_to_logmel(
        wave,
        sample_rate=target_sr,
        frontend=p["frontend"],
        n_fft=p["n_fft"],
        win_length=p["win_length"],
        hop_length=p["hop_length"],
        n_mels=p["n_mels"],
        fmin=p["fmin"],
        fmax=p["fmax"],
    )[0]
    if p["frontend"] == "efficientat":
        return spec, *_EFFICIENTAT_PREVIEW_RANGE
    return spec, None, None


def _spectrogram_thumb(wave: np.ndarray, sr: int, height: int = 96) -> str:
    spec, vmin, vmax = _preview_logmel(wave, sr)   # (n_mels, T)
    fig, ax = plt.subplots(figsize=(3.0, max(0.5, height / 96.0)))
    ax.imshow(spec, aspect="auto", origin="lower", cmap="magma",
              interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.axis("off")
    return _png_b64(fig)


def _clean_description(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _normalize_tags(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [part.strip() for part in value.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(part).strip() for part in value]
    else:
        raw = [str(value).strip()]
    seen: set[str] = set()
    tags: list[str] = []
    for tag in raw:
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def _description_tags_html(card: dict) -> str:
    lines: list[str] = []
    description = _clean_description(card.get("description"))
    if description:
        lines.append(
            "<div class='meta-line meta-note'>"
            f"<span class='meta-label'>description:</span> {html.escape(description)}"
            "</div>"
        )
    tags = _normalize_tags(card.get("tags"))
    if tags:
        tag_text = " ".join(html.escape(f"#{tag}") for tag in tags)
        lines.append(
            "<div class='meta-line meta-note'>"
            f"<span class='meta-label'>tags:</span> {tag_text}"
            "</div>"
        )
    return "".join(lines)


def _description_tags_search_text(card: dict) -> str:
    parts = [_clean_description(card.get("description"))]
    parts.extend(_normalize_tags(card.get("tags")))
    return " ".join(part for part in parts if part).lower()


def _channel_list_label(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        raw = [part.strip() for part in value.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(part).strip() for part in value]
    else:
        raw = [str(value).strip()]
    return ",".join(part for part in raw if part and part != "None")


def _microphone_label(card: dict) -> str:
    label = str(card.get("microphone_label") or "").strip()
    if label:
        return label
    mic = card.get("microphone")
    if mic not in (None, ""):
        return f"mic {mic}"
    recorded = card.get("recorded_channels") or card.get("source_recorded_channels") or []
    rec_label = _channel_list_label(recorded)
    if rec_label:
        return f"mic {rec_label}"
    if card.get("clip_kind") == "channel_split" and card.get("channel_index") is not None:
        return f"split ch{card.get('channel_index')}"
    return ""


def _microphone_meta_parts(card: dict) -> list[str]:
    parts: list[str] = []
    mic_label = _microphone_label(card)
    if mic_label:
        parts.append(mic_label)
    if card.get("clip_kind"):
        if card.get("clip_kind") == "channel_split" and card.get("channel_count"):
            parts.append(f"split {card.get('channel_index')}/{card.get('channel_count')}")
        elif card.get("clip_kind") != "mono":
            parts.append(str(card.get("clip_kind")))
    triggered = _channel_list_label(card.get("triggered_channels") or card.get("source_triggered_channels") or [])
    if triggered:
        parts.append(f"triggered {triggered}")
    recorded = _channel_list_label(card.get("recorded_channels") or card.get("source_recorded_channels") or [])
    if recorded and recorded != triggered:
        parts.append(f"recorded {recorded}")
    mic_type = card.get("microphone_type") or card.get("source_microphone_type")
    if isinstance(mic_type, list):
        mic_type = ", ".join(str(v) for v in mic_type if v)
    if mic_type and str(mic_type).strip() and str(mic_type).strip() != "generic":
        parts.append(str(mic_type).strip())
    if card.get("mic_position"):
        parts.append(f"pos {card.get('mic_position')}")
    if card.get("device"):
        parts.append(f"device {card.get('device')}")
    return parts


def _map_values_db(raw: object, *, flux: bool = False) -> list[str]:
    if not isinstance(raw, dict):
        return []
    out: list[str] = []
    for key in sorted(raw, key=lambda v: str(v)):
        try:
            value = float(raw[key])
        except Exception:  # noqa: BLE001
            continue
        if flux:
            import math
            value = 20 * math.log10(value + 1e-12)
        suffix = "dB" if flux else "dBFS"
        out.append(f"{key}:{value:+.0f}{suffix}")
    return out


def _get_cached_metadata(wav_path: Path, cache: dict) -> tuple[int, float]:
    try:
        stat = wav_path.stat()
        mtime = stat.st_mtime
        size = stat.st_size
    except Exception:
        try:
            info = sf.info(str(wav_path))
            return int(info.samplerate), float(info.duration)
        except Exception:
            return 16000, 0.0

    key = str(wav_path.resolve())
    if key in cache:
        entry = cache[key]
        if entry.get("mtime") == mtime and entry.get("size") == size:
            return int(entry["sample_rate"]), float(entry["duration_s"])

    try:
        info = sf.info(str(wav_path))
        sr = int(info.samplerate)
        dur = float(info.duration)
    except Exception:
        sr, dur = 16000, 0.0

    cache[key] = {
        "mtime": mtime,
        "size": size,
        "sample_rate": sr,
        "duration_s": dur,
    }
    return sr, dur


def _get_or_generate_thumbnail_file(wav_path: Path, t_type: str) -> Path | None:
    if t_type != "spec":
        return None
    try:
        wav_path = Path(wav_path).resolve()
        if not wav_path.exists():
            return None

        png_path = wav_path.with_suffix(".spec.png")

        if png_path.exists():
            if png_path.stat().st_mtime >= wav_path.stat().st_mtime:
                return png_path

        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=-1)

        spec, vmin, vmax = _preview_logmel(data, sr)
        fig, ax = plt.subplots(figsize=(3.0, 1.0))
        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma",
                  interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.axis("off")

        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(png_path), format="png", dpi=72, bbox_inches="tight")
        plt.close(fig)
        return png_path
    except Exception as e:
        log.error("Failed to generate spec thumbnail for %s: %s", wav_path, e)
        return None


def _init_metadata_cache(layout: ProjectLayout) -> dict:
    cache_path = layout.dataset_inspect_dir / "inspect_metadata_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Failed to load metadata cache: %s", e)
    return {}


def _save_metadata_cache(layout: ProjectLayout, cache: dict) -> None:
    cache_path = layout.dataset_inspect_dir / "inspect_metadata_cache.json"
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        log.warning("Failed to save metadata cache: %s", e)



# ---------------------------------------------------------------------------
# Per-class reference images (optional drop-in convention)
# ---------------------------------------------------------------------------
#
# A user can place a single photo per target class at either:
#   projects/<id>/references/<class>/_reference.{jpg,jpeg,png,webp}
#   data_library/sounds/<class>/_reference.{jpg,jpeg,png,webp}
# The project-scoped file wins; otherwise the library file is used as a
# shared fallback across projects. The leading underscore keeps the file at
# the top of directory listings and is naturally skipped by every "*.wav"
# glob in the codebase (recorder, compose, server, etc.).

_REFERENCE_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


def _find_class_reference_image(class_name: str, layout: ProjectLayout) -> Path | None:
    """Return the first existing `_reference.{ext}` for this class, probing
    project references dir then the shared library. Returns None for the
    `_negative` class or when nothing is found."""
    if class_name == "_negative":
        return None
    from sed.config import LIBRARY_DIR
    candidate_dirs = [
        layout.root / "references" / class_name,
        LIBRARY_DIR / "sounds" / class_name,
    ]
    for d in candidate_dirs:
        for ext in _REFERENCE_IMAGE_EXTS:
            p = d / f"_reference{ext}"
            if p.is_file():
                return p
    return None


def _image_data_uri(path: Path) -> str:
    """Encode a small image as a base64 data URI for inline embedding in the
    static HTML report. Suffix determines the MIME type."""
    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_class_images(
    classes: list[str], layout: ProjectLayout, interactive: bool,
) -> dict[str, str]:
    """Map each class to the URL/data-URI to render in the inspector. Classes
    without a reference image are simply omitted from the dict."""
    from urllib.parse import quote
    out: dict[str, str] = {}
    for cls in classes:
        path = _find_class_reference_image(cls, layout)
        if path is None:
            continue
        if interactive:
            out[cls] = f"/image?path={quote(str(path))}"
        else:
            out[cls] = _image_data_uri(path)
    return out


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _gather_clips(
    layout: ProjectLayout, cfg: Config | None = None
) -> list[tuple[Path, str, dict]]:
    """Return (wav_path, class_name, sidecar_meta_or_empty) tuples for the
    DATASET tab.

    This mirrors dataset composition: selected data_library/sounds clips only.
    """
    items: list[tuple[Path, str, dict]] = []
    if cfg is None:
        return items
    try:
        from sed.data.compose import (
            _library_clips_for_class,
            load_dataset_selection,
        )
        selection = load_dataset_selection(layout)
        for class_name in cfg.model.classes:
            for clip in _library_clips_for_class(
                cfg, layout.id, class_name, dataset_role="train", selection=selection
            ):
                wav = Path(clip.path)
                meta = _load_sidecar(wav)
                meta.setdefault("origin", clip.origin)
                items.append((wav, class_name, meta))
    except Exception as exc:  # noqa: BLE001
        log.warning("inspect: failed to gather library clips: %s", exc)
    return items


def _augmentation_group_label(meta: dict, fallback: str = "derived") -> str:
    params = meta.get("augmentation_params") if isinstance(meta.get("augmentation_params"), dict) else {}
    base = str(
        meta.get("base_label")
        or params.get("actual_base_label")
        or params.get("base_label")
        or meta.get("label")
        or meta.get("class_name")
        or fallback
    ).strip() or fallback
    overlay = str(
        meta.get("overlay_label")
        or params.get("actual_overlay_label")
        or params.get("overlay_label")
        or ""
    ).strip()
    technique = str(meta.get("augmentation") or params.get("augmentation_technique") or "").strip().lower()
    if overlay:
        return f"{base} × {overlay}"
    if technique in {"pitch", "pitch_shift"} or params.get("pitch_shift_enabled"):
        return f"{base} × pitch"
    return base


def _gather_review_clips(
    layout: ProjectLayout, cfg: Config | None = None
) -> tuple[list[tuple[Path, str, dict]], list[tuple[Path, str, dict]]]:
    """Return assigned raw and assigned derived clips for review tabs.

    The Dataset and Augmented tabs are views of the current project selection,
    not a recursive scan of everything on disk. Generated derived clips remain
    in the Library tab until explicitly assigned to a target class.
    """
    raw_items: list[tuple[Path, str, dict]] = []
    derived_items: list[tuple[Path, str, dict]] = []
    if cfg is None:
        return raw_items, derived_items
    try:
        from sed.data.compose import _library_clips_for_class, load_dataset_selection
        from sed.data.library import DERIVED_CATEGORY

        selection = load_dataset_selection(layout)
        for target_class in cfg.model.classes:
            for clip in _library_clips_for_class(
                cfg,
                layout.id,
                target_class,
                dataset_role="train",
                selection=selection,
                include_derived=True,
            ):
                wav = Path(clip.path)
                meta = _load_sidecar(wav)
                meta.setdefault("origin", clip.origin)
                meta["assigned_target_class"] = target_class
                meta["library_label"] = meta.get("label") or wav.parent.name
                if clip.origin == DERIVED_CATEGORY:
                    derived_items.append((wav, _augmentation_group_label(meta, wav.parent.name), meta))
                else:
                    raw_items.append((wav, target_class, meta))
    except Exception as exc:  # noqa: BLE001
        log.warning("inspect: failed to gather selected review clips: %s", exc)
    return raw_items, derived_items


def _gather_augmented(layout: ProjectLayout) -> list[tuple[Path, str, dict]]:
    """Discover augmented WAVs materialised by `sd dataset augment`.

    Returns (wav_path, class_name, sidecar_meta_or_empty) tuples.
    """
    items: list[tuple[Path, str, dict]] = []
    from sed.config import LIBRARY_DIR
    aug_root = LIBRARY_DIR / "derived" / layout.id / "trainval"
    if not aug_root.exists():
        aug_root = LIBRARY_DIR / "derived" / layout.id / "train"
    if not aug_root.exists():
        return items
    for class_dir in sorted(aug_root.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            wavs = sorted(class_dir.rglob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            wavs = sorted(class_dir.rglob("*.wav"), key=lambda p: p.name, reverse=True)
        for wav in wavs:
            meta = _load_sidecar(wav)
            items.append((wav, _augmentation_group_label(meta, class_dir.name), meta))
    return items


def _library_browser_items(layout: ProjectLayout) -> list[dict]:
    """Return compact metadata rows for the library-builder tab."""
    try:
        from sed.data.library import (
            DERIVED_CATEGORY,
            MINED_CATEGORY,
            RAW_CATEGORY,
            Library,
            clip_mic_channels,
        )
        lib = Library()
        items: list[dict] = []
        for clip in lib.all_clips(rebuild_if_empty=True):
            wav = clip.absolute_path(lib.root)
            if not wav.exists():
                continue
            meta = _load_sidecar(wav)
            category = clip.category
            source_bucket = (
                "raw" if category == RAW_CATEGORY
                else "mined" if category == MINED_CATEGORY
                else "derived" if category == DERIVED_CATEGORY
                else category
            )
            recorded_at = str(meta.get("recorded_at") or meta.get("created_at") or "")
            tags = list(clip.tags or meta.get("tags") or [])
            params = meta.get("augmentation_params") if isinstance(meta.get("augmentation_params"), dict) else {}
            mic_channels = sorted(set(int(ch) for ch in clip_mic_channels(clip)))
            mic_key = ",".join(str(ch) for ch in mic_channels)
            mic_label = (
                "unknown"
                if not mic_channels
                else ("mic " if len(mic_channels) == 1 else "mics ")
                + ",".join(str(ch) for ch in mic_channels)
            )
            mic_kind = (
                "unknown"
                if not mic_channels
                else "single"
                if len(mic_channels) == 1
                else "multi"
            )
            row = {
                "path": str(wav.resolve()),
                "relpath": clip.relpath,
                "uuid": clip.uuid,
                "name": wav.name,
                "label": clip.label,
                "project_id": clip.project_id,
                "dataset_role": clip.dataset_role,
                "density": clip.density,
                "review_status": clip.review_status,
                "category": category,
                "source_bucket": source_bucket,
                "clip_kind": clip.clip_kind,
                "channel_index": clip.channel_index,
                "channel_count": clip.channel_count,
                "parent_uuid": clip.parent_uuid,
                "recorded_channels": list(clip.recorded_channels),
                "triggered_channels": list(clip.triggered_channels),
                "mic_channels": mic_channels,
                "mic_key": mic_key,
                "mic_label": mic_label,
                "mic_kind": mic_kind,
                "session_id": clip.session_id,
                "recorded_at": recorded_at,
                "date": recorded_at[:10],
                "duration_s": clip.duration_s,
                "sample_rate": clip.sample_rate,
                "tags": tags,
                "tags_text": " ".join(str(tag) for tag in tags),
                "recipe_id": str(meta.get("recipe_id") or params.get("recipe_id") or ""),
                "recipe_signature": str(meta.get("recipe_signature") or params.get("recipe_signature") or ""),
                "generation_id": str(meta.get("generation_id") or params.get("generation_id") or ""),
                "base_label": str(meta.get("base_label") or params.get("actual_base_label") or params.get("base_label") or ""),
                "overlay_label": str(meta.get("overlay_label") or params.get("actual_overlay_label") or params.get("overlay_label") or ""),
                "base_source": str(meta.get("base_source") or params.get("base_source") or ""),
                "overlay_source": str(meta.get("overlay_source") or params.get("overlay_source") or ""),
                "selected": False,
            }
            row["search_text"] = " ".join(
                str(row.get(k) or "").lower()
                for k in (
                    "name", "label", "project_id", "dataset_role", "density",
                    "review_status", "category", "source_bucket", "clip_kind",
                    "parent_uuid", "mic_key", "mic_label", "session_id",
                    "recorded_at", "tags_text", "recipe_id", "generation_id",
                    "base_label", "overlay_label", "base_source", "overlay_source",
                )
            )
            items.append(row)
        return items
    except Exception as exc:  # noqa: BLE001
        log.warning("inspect: failed to gather library browser rows: %s", exc)
        return []


def _read_dataset_snapshot_hash(dataset_dir: Path) -> str:
    path = dataset_dir / "dataset_snapshot.json"
    if not path.exists():
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            snap = json.load(f)
        return snap.get("frozen_test", {}).get("hash", "")
    except Exception:  # noqa: BLE001
        return ""


def _current_ui_frozen_test_hash(layout: ProjectLayout, cfg: Config | None) -> str:
    if cfg is None:
        return ""
    try:
        from sed.data.compose import _library_clips_for_class, _test_set_hash, load_dataset_selection
        selection = load_dataset_selection(layout)
        if not selection.get("ui_managed"):
            return ""
        clips = [
            clip
            for class_name in cfg.model.classes
            for clip in _library_clips_for_class(
                cfg,
                layout.id,
                class_name,
                dataset_role="test",
                selection=selection,
                include_derived=True,
            )
        ]
        return _test_set_hash(clips) if clips else ""
    except Exception as exc:  # noqa: BLE001
        log.warning("inspect: failed to hash current frozen-test selection: %s", exc)
        return ""


def _frozen_test_stale_context(layout: ProjectLayout, cfg: Config | None) -> dict | None:
    if cfg is None:
        return None
    try:
        from sed.data.compose import load_dataset_selection
        selection = load_dataset_selection(layout)
        if not selection.get("ui_managed"):
            return None
    except Exception:  # noqa: BLE001
        return None
    current_hash = _current_ui_frozen_test_hash(layout, cfg)
    snapshot_hash = _read_dataset_snapshot_hash(layout.dataset_dir)
    if current_hash and snapshot_hash == current_hash:
        return None
    if current_hash and not snapshot_hash and not (layout.dataset_dir / "test.csv").exists():
        return None
    if not current_hash and not snapshot_hash:
        return None
    reason = (
        "Frozen test selection changed. Compose the dataset again to review the current frozen test set."
        if current_hash
        else "Frozen test rules are empty now. Compose the dataset again to clear the previous frozen test set."
    )
    return {
        "stale": True,
        "stale_reason": reason,
        "current_hash": current_hash,
        "snapshot_hash": snapshot_hash,
    }


def _selected_run_id(layout: ProjectLayout) -> str:
    from sed.train.registry import get_active_model
    val = get_active_model(layout.id)
    if val:
        return val
    selected = layout.models_dir / "selected_run.txt"
    if selected.exists():
        return selected.read_text(encoding="utf-8").strip()
    return ""


def _read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _run_metric_label(run_dir: Path, manifest: dict) -> str:
    metrics = manifest.get("test_metrics") if isinstance(manifest.get("test_metrics"), dict) else {}
    if not metrics:
        metrics = _read_json_file(run_dir / "reports" / "test_metrics.json")
    value = metrics.get("f1_macro_targets")
    suffix = "F1"
    if value is None:
        value = metrics.get("accuracy")
        suffix = "acc"
    try:
        return f"{float(value) * 100.0:.1f}% {suffix}"
    except Exception:  # noqa: BLE001
        return ""


def _run_loss_label(run_dir: Path) -> str:
    cfg = _read_json_file(run_dir / "training_config.json")
    training = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    loss = str(training.get("loss") or "").strip().lower()
    if loss == "cross_entropy":
        return "CE"
    if loss == "asymmetric_focal":
        return "focal"
    if loss == "bce_logits":
        return "BCE"
    return loss or ""


def frozen_run_report_image_path(layout: ProjectLayout, run_id: str | None, kind: str) -> Path | None:
    """Resolve one allowed frozen-run report image for the preview modal."""
    run = str(run_id or "").strip() or _selected_run_id(layout)
    if not run:
        return None
    kind_map = {
        "confusion": "confusion_matrix.png",
        "training": "training_curves.png",
    }
    filename = kind_map.get(str(kind or "").strip().lower())
    if not filename:
        return None
    try:
        models_root = layout.models_dir.resolve()
        run_dir = (layout.models_dir / run).resolve()
        run_dir.relative_to(models_root)
    except Exception:  # noqa: BLE001
        return None
    path = run_dir / "reports" / filename
    if path.is_file() and path.suffix.lower() == ".png":
        return path
    return None


def _frozen_run_options(layout: ProjectLayout, active_run_id: str = "") -> list[dict]:
    """Runs available for frozen-test inspection, newest first."""
    selected = _selected_run_id(layout)
    if not layout.models_dir.exists():
        return []
    options: list[dict] = []
    for run_dir in sorted(layout.models_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        report = run_dir / "reports" / "test_predictions.tsv"
        test_csv = run_dir / "dataset" / "test.csv"
        manifest_path = run_dir / "manifest.json"
        if not (report.exists() or test_csv.exists() or manifest_path.exists()):
            continue
        manifest = _read_json_file(manifest_path)
        tag = str(manifest.get("tag") or "").strip()
        label = run_dir.name
        if tag:
            label = f"{label} · {tag}"
        metric_label = _run_metric_label(run_dir, manifest)
        loss_label = _run_loss_label(run_dir)
        info_bits = [bit for bit in (metric_label, loss_label) if bit]
        if info_bits:
            label = f"{label} · {' · '.join(info_bits)}"
        badges = []
        if run_dir.name == selected:
            badges.append("selected")
        if report.exists():
            badges.append("predictions")
        elif test_csv.exists():
            badges.append("dataset")
        if badges:
            label = f"{label} ({', '.join(badges)})"
        sort_key = str(manifest.get("created_at") or "")
        if not sort_key:
            try:
                sort_key = str(max(report.stat().st_mtime if report.exists() else 0.0,
                                   test_csv.stat().st_mtime if test_csv.exists() else 0.0))
            except OSError:
                sort_key = ""
        options.append({
            "run_id": run_dir.name,
            "label": label,
            "active": run_dir.name == active_run_id,
            "sort_key": sort_key,
            "metric_label": metric_label,
            "loss_label": loss_label,
            "has_confusion": (run_dir / "reports" / "confusion_matrix.png").exists(),
            "has_training_plot": (run_dir / "reports" / "training_curves.png").exists(),
        })
    options.sort(key=lambda row: row["sort_key"], reverse=True)
    return options


def _pending_frozen_test_items(
    layout: ProjectLayout,
    cfg: Config | None,
) -> list[tuple[Path, str, dict]]:
    """Preview current UI-managed split=test rules before compose writes test.csv."""
    if cfg is None:
        return []
    try:
        from sed.data.compose import _library_clips_for_class, load_dataset_selection
        selection = load_dataset_selection(layout)
        items: list[tuple[Path, str, dict]] = []
        for class_name in cfg.model.classes:
            for clip in _library_clips_for_class(
                cfg,
                layout.id,
                class_name,
                dataset_role="test",
                selection=selection,
            ):
                wav = Path(clip.path)
                meta = _load_sidecar(wav)
                meta.update({
                    "test_class_name": clip.class_name,
                    "test_session_id": clip.session_id,
                    "test_origin": clip.origin,
                    "pending_compose": True,
                })
                items.append((wav, clip.class_name, meta))
        return sorted(items, key=lambda item: (item[1], item[2].get("test_session_id", ""), str(item[0])))
    except Exception as exc:  # noqa: BLE001
        log.warning("inspect: failed to build pending frozen-test preview: %s", exc)
        return []


def _gather_frozen_test(
    layout: ProjectLayout,
    run_id: str | None = None,
    cfg: Config | None = None,
) -> tuple[list[tuple[Path, str, dict]], dict]:
    """Read composed dataset/test.csv rows, if present.

    The frozen-test tab intentionally depends on composition: this guarantees
    labels reflect the exact held-out set used by evaluation, including source
    label aliases folded into target classes by dataset compose.
    """
    report, prediction_run_id = _latest_prediction_report(layout, run_id=run_id)
    selected = _selected_run_id(layout)
    effective_run_id = run_id or prediction_run_id or selected

    context = {
        "requested_run_id": run_id or "",
        "selected_run_id": selected,
        "viewed_run_id": effective_run_id or "",
        "run_options": [],
        "prediction_report": str(report) if report is not None else "",
        "source": "current_project_dataset",
        "display_test_hash": "",
        "current_test_hash": _read_dataset_snapshot_hash(layout.dataset_dir),
        "hash_matches_current": True,
        "pending_compose": False,
    }

    if report is not None:
        items: list[tuple[Path, str, dict]] = []
        with open(report, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                raw_path = row.get("path", "")
                if not raw_path:
                    continue
                path = Path(raw_path)
                meta = _load_sidecar(path)
                meta.update({
                    "test_class_name": row.get("true_class", ""),
                    "test_session_id": row.get("session_id", ""),
                    "test_origin": row.get("origin", ""),
                })
                items.append((path, row.get("true_class", ""), meta))
        display_hash = ""
        if effective_run_id:
            display_hash = _read_dataset_snapshot_hash(layout.model_dir(effective_run_id) / "dataset")
        context.update({
            "source": "test_predictions.tsv",
            "display_test_hash": display_hash,
            "run_options": _frozen_run_options(layout, effective_run_id or ""),
            "hash_matches_current": (
                not display_hash
                or not context["current_test_hash"]
                or display_hash == context["current_test_hash"]
            ),
        })
        return items, context

    if effective_run_id:
        run_test_csv = layout.model_dir(effective_run_id) / "dataset" / "test.csv"
        if run_test_csv.exists():
            items = _read_frozen_test_csv(run_test_csv)
            display_hash = _read_dataset_snapshot_hash(layout.model_dir(effective_run_id) / "dataset")
            context.update({
                "source": "run_dataset",
                "display_test_hash": display_hash,
                "run_options": _frozen_run_options(layout, effective_run_id or ""),
                "hash_matches_current": (
                    not display_hash
                    or not context["current_test_hash"]
                    or display_hash == context["current_test_hash"]
                ),
            })
            return items, context

    test_csv = layout.dataset_dir / "test.csv"
    if not test_csv.exists():
        pending_items = _pending_frozen_test_items(layout, cfg)
        if pending_items:
            context.update({
                "source": "pending_dataset_selection",
                "pending_compose": True,
                "pending_count": len(pending_items),
            })
            context["run_options"] = _frozen_run_options(layout, effective_run_id or "")
            return pending_items, context
        context["run_options"] = _frozen_run_options(layout, effective_run_id or "")
        return [], context
    items = _read_frozen_test_csv(test_csv)
    context["display_test_hash"] = context["current_test_hash"]
    context["run_options"] = _frozen_run_options(layout, effective_run_id or "")
    return items, context


def _read_frozen_test_csv(test_csv: Path) -> list[tuple[Path, str, dict]]:
    items: list[tuple[Path, str, dict]] = []
    with open(test_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_path = row.get("path", "")
            if not raw_path:
                continue
            path = Path(raw_path)
            meta = _load_sidecar(path)
            meta.update({
                "test_class_name": row.get("class_name", ""),
                "test_session_id": row.get("session_id", ""),
                "test_sample_rate": row.get("sample_rate", ""),
                "test_duration_s": row.get("duration_s", ""),
                "test_is_target": row.get("is_target", ""),
                "test_origin": row.get("origin", ""),
            })
            items.append((path, row.get("class_name", ""), meta))
    return items


def _load_sidecar(wav: Path) -> dict:
    side = wav.with_suffix(".json")
    if not side.exists():
        return {}
    try:
        with open(side, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _find_source_sidecar(layout: ProjectLayout, source_name: str) -> dict:
    """Resolve `source_name` against recorded clips and return its sidecar."""
    if not source_name:
        return {}
    for sub in ("positives", "negatives"):
        root = layout.recordings_dir / sub
        if not root.exists():
            continue
        for wav in root.rglob(source_name):
            return _load_sidecar(wav)
    return {}


def _tip_attr(text: str) -> str:
    return html.escape(text, quote=True)


def _augmentation_search_text(card: dict) -> str:
    """Text indexed by augmented-tab search."""
    params = card.get("augmentation_params") or {}
    pieces = [
        card.get("name", ""),
        card.get("source", ""),
        params.get("source", ""),
        params.get("negative_source", ""),
        params.get("base_role", ""),
        params.get("base_label", ""),
        params.get("overlay_label", ""),
        params.get("actual_base_label", ""),
        params.get("actual_overlay_label", ""),
        params.get("combination_description", ""),
        " ".join(_normalize_tags(params.get("combination_tags", []))),
        card.get("base_label", ""),
        card.get("overlay_label", ""),
        card.get("recipe_id", ""),
        card.get("assigned_target_class", ""),
        params.get("snr_db", ""),
        params.get("semitones", ""),
        params.get("transient_window_ms", ""),
        card.get("masking_score_db", ""),
        card.get("masking_score_p25_db", ""),
        card.get("masking_score_min_db", ""),
        card.get("effective_snr_db", ""),
        card.get("audit_pred", ""),
        "model-miss" if card.get("audit_correct") is False else "",
        _description_tags_search_text(card),
    ]
    return " ".join(str(p) for p in pieces if p).lower()


def _format_meta_text(card: dict) -> str:
    """One dense line per card. Order: src · bg · technique · key-params · SR
    · peak. Missing fields are simply omitted."""
    parts: list[str] = []
    origin = card.get("origin")
    if origin:
        parts.append(f"origin {origin}")
    src = card.get("source") or card.get("name", "")
    if src:
        parts.append(f"src {src}")
    bg = (card.get("augmentation_params") or {}).get("negative_source")
    if bg:
        parts.append(f"neg {bg}")
    technique = card.get("augmentation", "")
    if technique and technique != "none":
        parts.append(technique)
    params = card.get("augmentation_params") or {}
    if technique == "overlay" and "snr_db" in params:
        parts.append(f"snr={float(params['snr_db']):+.1f} dB")
        if params.get("pitch_shift_enabled") and "semitones" in params:
            parts.append(f"{float(params['semitones']):+.1f} st")
        if "transient_window_ms" in params:
            parts.append(f"win={int(params['transient_window_ms'])} ms")
    elif "semitones" in params:
        parts.append(f"{float(params['semitones']):+.1f} st")
    sr = card.get("sample_rate")
    if sr:
        parts.append(f"{int(sr) // 1000} kHz")
    if card.get("augmentation") == "none":
        flux_strength = card.get("source_onset_strength")
        if flux_strength and flux_strength > 0:
            import math
            parts.append(f"flux {20 * math.log10(flux_strength + 1e-12):+.0f} dB")
        flux_by_mic = _map_values_db(
            card.get("triggered_flux") or card.get("source_triggered_flux"),
            flux=True,
        )
        if flux_by_mic:
            parts.append(f"mic flux {'/'.join(flux_by_mic)}")
        baseline = card.get("source_onset_baseline")
        if baseline and baseline > 0:
            import math
            parts.append(f"base {20 * math.log10(float(baseline) + 1e-12):+.0f} dB")

    peak = card.get("peak_dbfs")
    if peak is None: peak = card.get("source_peak_dbfs")
    if peak is not None:
        parts.append(f"peak {peak:+.1f} dBFS")
    peak_by_mic = _map_values_db(card.get("triggered_peak_dbfs") or card.get("source_triggered_peak_dbfs"))
    if peak_by_mic:
        parts.append(f"mic peak {'/'.join(peak_by_mic)}")

    tmel = card.get("transient_mel_db")
    if tmel is None: tmel = card.get("source_transient_mel_db")
    if tmel is not None:
        parts.append(f"tmel {tmel:+.1f} dB")

    rms = card.get("rms_dbfs")
    if rms is None: rms = card.get("source_rms_dbfs")
    if rms is not None:
        parts.append(f"rms {rms:+.1f} dBFS")
    note = _description_tags_search_text(card)
    if note:
        parts.append(note)
    mic_parts = _microphone_meta_parts(card)
    if mic_parts:
        parts.append(" · ".join(mic_parts))

    return " · ".join(parts)


def _format_meta_html(card: dict) -> str:
    lines = []

    # 1. Sources (long)
    src = card.get("source")
    if src and src != card.get("name"):
        lines.append(f"<div class='meta-line'><span class='meta-label'>src:</span> {src}</div>")
    bg = (card.get("augmentation_params") or {}).get("negative_source")
    if bg:
        lines.append(f"<div class='meta-line'><span class='meta-label'>neg:</span> {bg}</div>")

    # 2. Audio sample properties
    audio_parts = []
    assigned_target = card.get("assigned_target_class")
    if assigned_target:
        audio_parts.append(f"target {assigned_target}")
    origin = card.get("origin")
    if origin:
        audio_parts.append(f"origin {origin}")
    sr = card.get("sample_rate")
    if sr:
        audio_parts.append(f"{int(sr) // 1000}kHz")
    dur = card.get("duration_s")
    if dur is not None:
        audio_parts.append(f"{dur:.2f}s")
    if card.get("augmentation") == "none":
        flux = card.get("source_onset_strength")
        if flux and flux > 0:
            import math
            audio_parts.append(f"flux {20 * math.log10(flux + 1e-12):+.0f}dB")
        flux_by_mic = _map_values_db(
            card.get("triggered_flux") or card.get("source_triggered_flux"),
            flux=True,
        )
        if flux_by_mic:
            audio_parts.append(f"mic flux {'/'.join(flux_by_mic)}")
        baseline = card.get("source_onset_baseline")
        if baseline and baseline > 0:
            import math
            audio_parts.append(f"base {20 * math.log10(float(baseline) + 1e-12):+.0f}dB")

    peak = card.get("peak_dbfs")
    if peak is None: peak = card.get("source_peak_dbfs")
    if peak is not None:
        audio_parts.append(f"peak {peak:+.1f}dBFS")
    peak_by_mic = _map_values_db(card.get("triggered_peak_dbfs") or card.get("source_triggered_peak_dbfs"))
    if peak_by_mic:
        audio_parts.append(f"mic peak {'/'.join(peak_by_mic)}")

    tmel = card.get("transient_mel_db")
    if tmel is None: tmel = card.get("source_transient_mel_db")
    if tmel is not None:
        audio_parts.append(f"tmel {tmel:+.1f}dB")

    rms = card.get("rms_dbfs")
    if rms is None: rms = card.get("source_rms_dbfs")
    if rms is not None:
        audio_parts.append(f"rms {rms:+.1f}dBFS")

    if audio_parts:
        lines.append(f"<div class='meta-line'>{' · '.join(audio_parts)}</div>")

    mic_parts = _microphone_meta_parts(card)
    if mic_parts:
        lines.append(
            "<div class='meta-line'><span class='meta-label'>mic:</span> "
            + html.escape(" · ".join(mic_parts))
            + "</div>"
        )

    # 3. Augmentation
    technique = card.get("augmentation", "")
    if technique and technique != "none":
        aug_parts = [f"<span style='color:#444;font-weight:600'>{technique}</span>"]
        params = card.get("augmentation_params") or {}
        if technique == "overlay" and "snr_db" in params:
            aug_parts.append(f"snr {float(params['snr_db']):+.1f}dB")
            if params.get("pitch_shift_enabled") and "semitones" in params:
                aug_parts.append(f"{float(params['semitones']):+.1f}st")
            if "transient_window_ms" in params:
                aug_parts.append(f"window {int(params['transient_window_ms'])}ms")
        elif technique in {"pitch", "pitch_shift"} and "semitones" in params:
            aug_parts.append(f"{float(params['semitones']):+.1f}st")

        if card.get("base_label"):
            aug_parts.append(f"base {html.escape(str(card.get('base_label')))}")
        if card.get("overlay_label"):
            aug_parts.append(f"overlay {html.escape(str(card.get('overlay_label')))}")
        if card.get("recipe_id"):
            aug_parts.append(f"recipe {html.escape(str(card.get('recipe_id'))[:8])}")

        # Masking score (if applicable)
        m = card.get("masking_score_db")
        if m is not None:
            m = float(m)
            cls_mask = "mask-good" if m >= 6 else "mask-warn" if m >= 0 else "mask-bad"
            mt = card.get("mask_target_db")
            mb = card.get("mask_background_db")
            p25 = card.get("masking_score_p25_db")
            worst = card.get("masking_score_min_db")
            if mt is not None and mb is not None:
                mask_tip = (
                    f"target transient: {float(mt):+.1f} dB\n"
                    f"overlay same bands: {float(mb):+.1f} dB\n"
                    f"mean mask: {m:+.1f} dB"
                )
                if p25 is not None:
                    mask_tip += f"\np25 mask: {float(p25):+.1f} dB"
                if worst is not None:
                    mask_tip += f"\nworst-band mask: {float(worst):+.1f} dB"
                aug_parts.append(
                    f"<span class='tip {cls_mask}' data-tip='{_tip_attr(mask_tip)}'>"
                    f"mask {m:+0.1f} dB</span>"
                )
            else:
                aug_parts.append(f"<span class='{cls_mask}'>mask {m:+0.1f} dB</span>")
            if p25 is not None:
                aug_parts.append(f"p25 {float(p25):+0.1f}dB")
            if worst is not None:
                aug_parts.append(f"worst {float(worst):+0.1f}dB")

        eff = card.get("effective_snr_db")
        if eff is not None:
            aug_parts.append(f"eff snr {float(eff):+0.1f}dB")
        if card.get("audit_pred"):
            ok = bool(card.get("audit_correct"))
            conf = card.get("audit_conf")
            conf_txt = f" {float(conf):.2f}" if conf is not None else ""
            seen_txt = " · seen in training" if card.get("audit_seen") else ""
            aug_parts.append(
                f"<span class='{'mask-good' if ok else 'mask-bad'}'>model "
                f"{html.escape(str(card['audit_pred']))} {'✓' if ok else '✗'}{conf_txt}</span>{seen_txt}"
            )

        lines.append(f"<div class='meta-line'>{' · '.join(aug_parts)}</div>")

    lines.append(_description_tags_html(card))
    return "".join(lines)


def inspect_project(
    cfg: Config, project_id: str, max_thumbs_per_class: int = 30,
    interactive: bool = False,
    run_id: str | None = None,
) -> Path:
    """Generate the HTML inspection report. Returns its path."""
    layout = ProjectLayout(project_id)
    set_preview_features(cfg)
    try:
        from sed.data.compose import prune_project_metadata
        prune_project_metadata(layout)
    except Exception as exc:  # noqa: BLE001
        log.warning("inspect cleanup skipped: %s", exc)
    layout.dataset_inspect_dir.mkdir(parents=True, exist_ok=True)
    items, derived_items = _gather_review_clips(layout, cfg)
    library_items = _library_browser_items(layout)
    if not items:
        log.info("No composed clips found yet; rendering library browser only.")

    log.info("Inspecting %d clips ...", len(items))

    labels: list[str] = []
    durations: list[float] = []
    paths: list[Path] = []
    metas: list[dict] = []
    metadata_cache = _init_metadata_cache(layout)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[cyan]Analyzing clips...", total=len(items))
        for wav, label, meta in items:
            try:
                _sr, duration = _get_cached_metadata(wav, metadata_cache)
                labels.append(label)
                durations.append(duration)
                paths.append(wav)
                metas.append(meta)
            except Exception as e:  # noqa: BLE001
                log.warning("skipping %s: %s", wav, e)
            progress.advance(task_id)

    classes = sorted(set(labels))

    class_summary = {
        cls: {
            "count": int(sum(1 for l in labels if l == cls)),
            "total_seconds": float(sum(d for l, d in zip(labels, durations) if l == cls)),
        }
        for cls in classes
    }

    panels = _build_panels(
        items=list(zip(paths, labels, metas, durations)),
        classes=classes,
        max_per_class=max_thumbs_per_class,
        is_rerender=False,
        excluded_paths=_load_exclusions_for_render(layout),
        interactive=interactive,
        metadata_cache=metadata_cache,
    )

    aug_panels = _build_panels_from_augmented(
        layout,
        items=derived_items,
        max_per_class=max_thumbs_per_class,
        excluded_paths=_load_exclusions_for_render(layout),
        interactive=interactive,
        metadata_cache=metadata_cache,
    )
    stale_frozen = _frozen_test_stale_context(layout, cfg)
    if stale_frozen:
        frozen_panels, frozen_predictions_run_id, frozen_has_predictions, frozen_context = {}, "", False, stale_frozen
    else:
        frozen_panels, frozen_predictions_run_id, frozen_has_predictions, frozen_context = (
            _build_panels_from_frozen_test(
                layout, max_per_class=max_thumbs_per_class, run_id=run_id,
                cfg=cfg,
                interactive=interactive, metadata_cache=metadata_cache,
            )
        )
    mined_panels = _build_panels_from_mined(
        layout, output_dir=cfg.mining.output_dir, max_per_class=max_thumbs_per_class,
        interactive=interactive, metadata_cache=metadata_cache,
    )

    _save_metadata_cache(layout, metadata_cache)
    class_images = _build_class_images(classes, layout, interactive=interactive)

    html = _render_html(
        project_id=project_id,
        class_summary=class_summary,
        panels=panels,
        aug_panels=aug_panels,
        frozen_panels=frozen_panels,
        frozen_predictions_run_id=frozen_predictions_run_id,
        frozen_has_predictions=frozen_has_predictions,
        frozen_context=frozen_context,
        mined_panels=mined_panels,
        library_items=library_items,
        model_classes=list(cfg.model.classes),
        interactive=interactive,
        class_images=class_images,
    )
    out = layout.dataset_inspect_dir / "inspection.html"
    out.write_text(html, encoding="utf-8")
    log.info("Wrote %s", out)
    return out



def _quick_rerender(
    cfg: Config, project_id: str, max_thumbs_per_class: int = 0,
    run_id: str | None = None,
) -> str:
    """Re-render the HTML for the inspect server. Re-scans the recordings dir
    so newly recorded clips are picked up on browser refresh."""
    layout = ProjectLayout(project_id)
    set_preview_features(cfg)
    try:
        from sed.data.compose import prune_project_metadata
        prune_project_metadata(layout)
    except Exception as exc:  # noqa: BLE001
        log.warning("inspect cleanup skipped: %s", exc)
    items, derived_items = _gather_review_clips(layout, cfg)
    library_items = _library_browser_items(layout)
    excluded_paths = _load_exclusions_for_render(layout)
    metadata_cache = _init_metadata_cache(layout)
    if not items:
        aug_panels = _build_panels_from_augmented(
            layout, max_per_class=max_thumbs_per_class, excluded_paths=excluded_paths,
            interactive=True, metadata_cache=metadata_cache, items=derived_items,
        )
        stale_frozen = _frozen_test_stale_context(layout, cfg)
        if stale_frozen:
            frozen_panels, frozen_predictions_run_id, frozen_has_predictions, frozen_context = {}, "", False, stale_frozen
        else:
            frozen_panels, frozen_predictions_run_id, frozen_has_predictions, frozen_context = (
                _build_panels_from_frozen_test(
                    layout, max_per_class=max_thumbs_per_class, run_id=run_id,
                    cfg=cfg,
                    interactive=True, metadata_cache=metadata_cache,
                )
            )
        mined_panels = _build_panels_from_mined(
            layout, output_dir=cfg.mining.output_dir, max_per_class=max_thumbs_per_class,
            interactive=True, metadata_cache=metadata_cache,
        )
        _save_metadata_cache(layout, metadata_cache)
        return _render_html(
            project_id=project_id, class_summary={},
            panels={}, interactive=True,
            aug_panels=aug_panels,
            frozen_panels=frozen_panels,
            frozen_predictions_run_id=frozen_predictions_run_id,
            frozen_has_predictions=frozen_has_predictions,
            frozen_context=frozen_context,
            mined_panels=mined_panels,
            library_items=library_items,
            model_classes=list(cfg.model.classes),
            class_images={},
        )

    paths = [w for w, _l, _m in items]
    labels = [l for _w, l, _m in items]
    metas = [m for _w, _l, m in items]
    durations: list[float] = []
    for wav in paths:
        try:
            _sr, dur = _get_cached_metadata(wav, metadata_cache)
            durations.append(dur)
        except Exception:  # noqa: BLE001
            durations.append(0.0)
    classes = sorted({l for l in labels})
    class_summary = {
        cls: {
            "count": int(sum(1 for l in labels if l == cls)),
            "total_seconds": float(sum(d for l, d in zip(labels, durations) if l == cls)),
        }
        for cls in classes
    }
    panels = _build_panels(
        items=list(zip(paths, labels, metas, durations)),
        classes=classes,
        max_per_class=max_thumbs_per_class,
        is_rerender=True,
        excluded_paths=excluded_paths,
        interactive=True,
        metadata_cache=metadata_cache,
    )

    aug_panels = _build_panels_from_augmented(
        layout,
        items=derived_items,
        max_per_class=max_thumbs_per_class,
        excluded_paths=excluded_paths,
        interactive=True,
        metadata_cache=metadata_cache,
    )
    stale_frozen = _frozen_test_stale_context(layout, cfg)
    if stale_frozen:
        frozen_panels, frozen_predictions_run_id, frozen_has_predictions, frozen_context = {}, "", False, stale_frozen
    else:
        frozen_panels, frozen_predictions_run_id, frozen_has_predictions, frozen_context = (
            _build_panels_from_frozen_test(
                layout, max_per_class=max_thumbs_per_class, run_id=run_id,
                cfg=cfg,
                interactive=True, metadata_cache=metadata_cache,
            )
        )
    mined_panels = _build_panels_from_mined(
        layout, output_dir=cfg.mining.output_dir, max_per_class=max_thumbs_per_class,
        interactive=True, metadata_cache=metadata_cache,
    )

    _save_metadata_cache(layout, metadata_cache)
    class_images = _build_class_images(classes, layout, interactive=True)

    return _render_html(
        project_id=project_id,
        class_summary=class_summary,
        panels=panels,
        aug_panels=aug_panels,
        frozen_panels=frozen_panels,
        frozen_predictions_run_id=frozen_predictions_run_id,
        frozen_has_predictions=frozen_has_predictions,
        frozen_context=frozen_context,
        mined_panels=mined_panels,
        library_items=library_items,
        model_classes=list(cfg.model.classes),
        interactive=True,
        class_images=class_images,
    )


def augmentation_recipe_counts(
    layout: ProjectLayout,
    recipes: list[dict] | None = None,
    settings: dict | None = None,
    frozen_test: bool | None = None,
) -> dict[str, dict[str, int]]:
    """Count generated derived clips by UI recipe id.

    Recipe ids are written into new sidecars at the top level and inside
    augmentation_params. The nested fallback keeps older generated clips visible
    after a metadata-only schema adjustment. Generations whose recipe was
    removed from the selection are still returned (marked ``orphan``) so the
    UI can list and delete them.
    """
    from sed.config import LIBRARY_DIR

    root = LIBRARY_DIR / "derived" / layout.id
    counts: dict[str, dict[str, int]] = {}
    recipe_info: dict[str, dict] = {}
    expected: dict[str, int] = {}
    saved_ids: set[str] = {
        str(recipe.get("recipe_id") or recipe.get("id") or "").strip()
        for recipe in (recipes or [])
        if isinstance(recipe, dict)
    }
    if recipes:
        try:
            from sed.data.augment import (
                _library_paths_for_filters,
                generated_clip_matches_recipe,
                source_names_for_recipe_paths,
            )
            from sed.data.library import Library
            lib = Library()
            for recipe in recipes:
                if not isinstance(recipe, dict):
                    continue
                recipe_id = str(recipe.get("recipe_id") or recipe.get("id") or "").strip()
                if not recipe_id:
                    continue
                # Mirror materialize_library_recipes: each recipe is signed with
                # its own settings snapshot layered over the baseline, plus its
                # own frozen_test flag. Otherwise panel tweaks after generation
                # would mark every clip stale even though a re-run would skip them.
                recipe_settings = recipe.get("settings") if isinstance(recipe.get("settings"), dict) else {}
                effective_settings = {**(settings or {}), **(recipe_settings or {})}
                recipe_frozen = bool(recipe.get("frozen_test", frozen_test))
                source_role = "test" if recipe_frozen else "train"
                base_filters = recipe.get("base_filters") or ([recipe.get("base_filter")] if recipe.get("base_filter") else [])
                overlay_filters = recipe.get("overlay_filters") or ([recipe.get("overlay_filter")] if recipe.get("overlay_filter") else [])
                recipe_overlay = bool(effective_settings.get("overlay_enabled", True))
                base_paths = _library_paths_for_filters(
                    lib,
                    str(recipe.get("base") or ""),
                    dataset_role=source_role,
                    project_id=layout.id,
                    frozen_test=recipe_frozen,
                    filters=base_filters,
                )
                overlay_paths = (
                    _library_paths_for_filters(
                        lib,
                        str(recipe.get("overlay") or ""),
                        dataset_role=source_role,
                        project_id=layout.id,
                        frozen_test=recipe_frozen,
                        filters=overlay_filters,
                    )
                    if recipe_overlay and overlay_filters
                    else []
                )
                recipe_info[recipe_id] = {
                    "recipe": recipe,
                    "settings": effective_settings,
                    "frozen_test": recipe_frozen,
                    "base_sources": source_names_for_recipe_paths(base_paths),
                    "overlay_sources": source_names_for_recipe_paths(overlay_paths),
                    "matches": generated_clip_matches_recipe,
                }
                expected[recipe_id] = int(recipe.get("expected_count") or recipe.get("base_count") or 0)
        except Exception:  # noqa: BLE001
            recipe_info = {}
    for split_name in ("trainval", "test"):
        split_root = root / split_name
        if not split_root.exists():
            continue
        for sidecar in split_root.rglob("*.json"):
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            params = meta.get("augmentation_params") if isinstance(meta.get("augmentation_params"), dict) else {}
            recipe_id = str(meta.get("recipe_id") or params.get("recipe_id") or "").strip()
            if not recipe_id:
                continue
            row = counts.setdefault(recipe_id, {
                "total": 0,
                "current": 0,
                "stale": 0,
                "trainval": 0,
                "test": 0,
                "expected": expected.get(recipe_id, 0),
                "orphan": recipe_id not in saved_ids,
                "base_label": "",
                "overlay_label": "",
            })
            row["total"] += 1
            row[split_name] += 1
            if not row["base_label"]:
                row["base_label"] = str(meta.get("base_label") or params.get("base_label") or meta.get("label") or "")
            if not row["overlay_label"]:
                row["overlay_label"] = str(meta.get("overlay_label") or params.get("overlay_label") or "")
            info = recipe_info.get(recipe_id)
            if info and not info["matches"](
                meta,
                info["recipe"],
                info["settings"],
                frozen_test=info["frozen_test"],
                base_sources=info["base_sources"],
                overlay_sources=info["overlay_sources"],
            ):
                row["stale"] += 1
            else:
                row["current"] += 1
    return counts


REVIEW_REFRESH_SCOPE_ORDER = ("library", "recipes", "dataset", "augmented", "frozen", "mined")
_REVIEW_REFRESH_SCOPE_SET = set(REVIEW_REFRESH_SCOPE_ORDER)


def normalise_review_refresh_scopes(raw: object | None = None) -> tuple[list[str], bool]:
    """Return requested inspect refresh scopes and whether this is a full apply."""
    if raw is None:
        return list(REVIEW_REFRESH_SCOPE_ORDER), True
    if isinstance(raw, str):
        parts = [part.strip().lower() for part in raw.replace(";", ",").split(",")]
    elif isinstance(raw, (list, tuple, set)):
        parts = [str(part).strip().lower() for part in raw]
    else:
        parts = [str(raw).strip().lower()]
    parts = [part for part in parts if part]
    if not parts or "all" in parts or "*" in parts:
        return list(REVIEW_REFRESH_SCOPE_ORDER), True
    invalid = sorted(set(parts) - _REVIEW_REFRESH_SCOPE_SET)
    if invalid:
        raise ValueError(f"unknown refresh scope(s): {', '.join(invalid)}")
    wanted = set(parts)
    return [scope for scope in REVIEW_REFRESH_SCOPE_ORDER if scope in wanted], False


def _mined_panel_totals(output_dir: str = "mined") -> dict[str, int]:
    """True file count per mined class dir (ignores the display card cap)."""
    from sed.config import LIBRARY_DIR
    root = LIBRARY_DIR / output_dir
    if not root.exists():
        return {}
    out: dict[str, int] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            n = sum(1 for _ in class_dir.rglob("*.wav"))
        except Exception:  # noqa: BLE001
            n = 0
        if n:
            out[class_dir.name] = n
    return out


def panel_totals(
    cfg: Config,
    layout: ProjectLayout,
    scopes: object | None = None,
    *,
    items: list[tuple[Path, str, dict]] | None = None,
    derived_items: list[tuple[Path, str, dict]] | None = None,
) -> dict[str, dict[str, int]]:
    """True per-class clip counts per tab, ignoring the display card cap.

    Cheap by design: counts items/files only (no audio decode or thumbnail
    generation) so the UI can show the real dataset size even when it renders
    only the first ``max_thumbs_per_class`` cards. Callers in the refresh path
    may pass already-gathered ``items``/``derived_items`` to avoid re-scanning.
    """
    from collections import Counter

    scope_list, _ = normalise_review_refresh_scopes(scopes)
    requested = set(scope_list)
    totals: dict[str, dict[str, int]] = {}

    if {"dataset", "augmented"} & requested:
        if items is None and derived_items is None:
            items, derived_items = _gather_review_clips(layout, cfg)
        items = items or []
        derived_items = derived_items or []
        if "dataset" in requested:
            totals["dataset"] = dict(Counter(label for _w, label, _m in items))
        if "augmented" in requested:
            totals["augmented"] = dict(Counter(label for _w, label, _m in derived_items))

    if "frozen" in requested:
        # Mirror the frozen builder: a stale/uncomposed test set has no clips.
        if _frozen_test_stale_context(layout, cfg):
            totals["frozen"] = {}
        else:
            frozen_items, _ctx = _gather_frozen_test(layout, run_id=None, cfg=cfg)
            totals["frozen"] = dict(Counter(cls for _w, cls, _m in frozen_items if cls))

    if "mined" in requested:
        totals["mined"] = _mined_panel_totals(cfg.mining.output_dir)

    return totals


def review_data_payload(
    cfg: Config,
    layout: ProjectLayout,
    max_per_class: int = 0,
    run_id: str | None = None,
    progress=None,
    scopes: object | None = None,
) -> dict:
    """Fresh inspect-server panel data for the manual Review refresh button."""
    set_preview_features(cfg)
    scope_list, full_refresh = normalise_review_refresh_scopes(scopes)
    requested = set(scope_list)
    needs_selection = bool({"dataset", "augmented"} & requested)
    needs_metadata = bool({"dataset", "augmented", "frozen", "mined"} & requested)
    needs_excluded = bool({"dataset", "augmented"} & requested)

    base_weights = {
        "cleanup": 0.04,
        "scan selection": 0.10,
        "scan library": 0.14,
        "build dataset cards": 0.18,
        "build augmented": 0.18,
        "build frozen test": 0.14,
        "build mined": 0.14,
        "load recipe counts": 0.08,
    }
    selected_phases: list[str] = []
    if full_refresh:
        selected_phases.append("cleanup")
    if needs_selection:
        selected_phases.append("scan selection")
    if "library" in requested:
        selected_phases.append("scan library")
    if "dataset" in requested:
        selected_phases.append("build dataset cards")
    if "augmented" in requested:
        selected_phases.append("build augmented")
    if "frozen" in requested:
        selected_phases.append("build frozen test")
    if "mined" in requested:
        selected_phases.append("build mined")
    if "recipes" in requested:
        selected_phases.append("load recipe counts")
    total_weight = sum(base_weights[phase] for phase in selected_phases) or 1.0
    phase_weights = [(phase, base_weights[phase] / total_weight) for phase in selected_phases]
    starts: dict[str, float] = {}
    acc = 0.0
    for phase, weight in phase_weights:
        starts[phase] = acc
        acc += weight
    spans = dict(phase_weights)

    def notify(phase: str, *, fraction: float = 0.0, detail: str = "", **extra) -> None:
        if progress is None:
            return
        try:
            start = starts.get(phase, 0.0)
            span = spans.get(phase, 0.0)
            value = min(0.99, max(0.0, start + span * max(0.0, min(1.0, fraction))))
            progress({
                "phase": phase,
                "phase_detail": detail or phase,
                "progress": value,
                **extra,
            })
        except Exception:  # noqa: BLE001
            pass

    payload: dict[str, object] = {
        "ok": True,
        "scopes": scope_list,
        "full_refresh": full_refresh,
    }
    summary_counts: dict[str, int] = {}

    if full_refresh:
        notify("cleanup", fraction=0.0)
        try:
            from sed.data.compose import prune_project_metadata
            prune_project_metadata(layout)
        except Exception as exc:  # noqa: BLE001
            log.warning("inspect refresh cleanup skipped: %s", exc)
        notify("cleanup", fraction=1.0)

    items: list[tuple[Path, str, dict]] = []
    derived_items: list[tuple[Path, str, dict]] = []
    if needs_selection:
        notify("scan selection", fraction=0.0)
        items, derived_items = _gather_review_clips(layout, cfg)
        notify("scan selection", fraction=1.0, count=len(items) + len(derived_items))

    if "library" in requested:
        notify("scan library", fraction=0.0)
        library_items = _library_browser_items(layout)
        payload["library_items"] = library_items
        summary_counts["library"] = len(library_items)
        notify("scan library", fraction=1.0, count=len(library_items))

    metadata_cache = _init_metadata_cache(layout) if needs_metadata else {}
    excluded_paths = _load_exclusions_for_render(layout) if needs_excluded else set()

    if "dataset" in requested:
        paths = [w for w, _label, _meta in items]
        labels = [label for _w, label, _meta in items]
        metas = [meta for _w, _label, meta in items]
        durations: list[float] = []
        for wav in paths:
            try:
                _sr, dur = _get_cached_metadata(wav, metadata_cache)
                durations.append(dur)
            except Exception:  # noqa: BLE001
                durations.append(0.0)
        classes = sorted(set(labels))
        class_summary = {
            cls: {
                "count": int(sum(1 for label in labels if label == cls)),
                "total_seconds": float(sum(d for label, d in zip(labels, durations) if label == cls)),
            }
            for cls in classes
        }
        notify("build dataset cards", fraction=0.0, count=len(paths))
        panels = _build_panels(
            items=list(zip(paths, labels, metas, durations)),
            classes=classes,
            max_per_class=max_per_class,
            is_rerender=True,
            excluded_paths=excluded_paths,
            interactive=True,
            metadata_cache=metadata_cache,
        )
        payload["class_summary"] = class_summary
        payload["panels"] = panels
        summary_counts["dataset"] = sum(len(v) for v in panels.values())
        notify("build dataset cards", fraction=1.0, count=len(paths))

    if "augmented" in requested:
        notify("build augmented", fraction=0.0, count=len(derived_items))
        aug_panels = _build_panels_from_augmented(
            layout,
            items=derived_items,
            max_per_class=max_per_class,
            excluded_paths=excluded_paths,
            interactive=True,
            metadata_cache=metadata_cache,
        )
        payload["aug_panels"] = aug_panels
        summary_counts["augmented"] = sum(len(v) for v in aug_panels.values())
        notify("build augmented", fraction=1.0, count=len(derived_items))

    if "frozen" in requested:
        notify("build frozen test", fraction=0.0)
        stale_frozen = _frozen_test_stale_context(layout, cfg)
        if stale_frozen:
            frozen_panels, frozen_predictions_run_id, frozen_has_predictions, frozen_context = {}, "", False, stale_frozen
        else:
            frozen_panels, frozen_predictions_run_id, frozen_has_predictions, frozen_context = (
                _build_panels_from_frozen_test(
                    layout,
                    max_per_class=max_per_class,
                    run_id=run_id,
                    cfg=cfg,
                    interactive=True,
                    metadata_cache=metadata_cache,
                )
            )
        payload["frozen_panels"] = frozen_panels
        payload["frozen_predictions_run_id"] = frozen_predictions_run_id
        payload["frozen_has_predictions"] = frozen_has_predictions
        payload["frozen_context"] = frozen_context
        summary_counts["frozen"] = sum(len(v) for v in frozen_panels.values())
        notify("build frozen test", fraction=1.0, count=summary_counts["frozen"])

    if "mined" in requested:
        notify("build mined", fraction=0.0)
        mined_panels = _build_panels_from_mined(
            layout,
            output_dir=cfg.mining.output_dir,
            max_per_class=max_per_class,
            interactive=True,
            metadata_cache=metadata_cache,
        )
        payload["mined_panels"] = mined_panels
        summary_counts["mined"] = sum(len(v) for v in mined_panels.values())
        notify("build mined", fraction=1.0, count=summary_counts["mined"])

    if needs_metadata:
        _save_metadata_cache(layout, metadata_cache)

    if "recipes" in requested:
        notify("load recipe counts", fraction=0.0)
        counts = augmentation_recipe_counts(layout)
        payload["augmentation_generated_counts"] = counts
        summary_counts["recipes"] = len(counts)
        notify("load recipe counts", fraction=1.0, count=len(counts))

    # True per-class totals (uncapped) for the requested tabs so the UI counts
    # reflect the real dataset size. Reuse the already-gathered clip lists to
    # avoid a second scan for the dataset/augmented tabs.
    payload["panel_totals"] = panel_totals(
        cfg, layout, scopes=scope_list, items=items, derived_items=derived_items
    )

    payload["summary_counts"] = summary_counts
    return payload


def clips_page(
    cfg: Config,
    layout: ProjectLayout,
    tab: str,
    class_name: str,
    offset: int = 0,
    limit: int = 100,
    run_id: str | None = None,
) -> list[dict]:
    """One page of clip cards for a single class/tab beyond the display cap.

    Lets the UI keep an initial bounded payload (``max_thumbs_per_class`` cards
    per class) and fetch the rest on demand when the user clicks "Load more".
    """
    set_preview_features(cfg)
    offset = max(0, int(offset))
    limit = max(1, int(limit))
    end = offset + limit
    excluded_paths = _load_exclusions_for_render(layout)
    metadata_cache = _init_metadata_cache(layout)
    cards: list[dict] = []
    try:
        if tab in ("dataset", "augmented"):
            items, derived_items = _gather_review_clips(layout, cfg)
            if tab == "dataset":
                selected = [(w, l, m) for (w, l, m) in items if l == class_name]
                durations: list[float] = []
                for wav, _l, _m in selected:
                    try:
                        _sr, dur = _get_cached_metadata(wav, metadata_cache)
                    except Exception:  # noqa: BLE001
                        dur = 0.0
                    durations.append(dur)
                panels = _build_panels(
                    items=[(w, l, m, d) for (w, l, m), d in zip(selected, durations)],
                    classes=[class_name],
                    max_per_class=end,
                    is_rerender=True,
                    excluded_paths=excluded_paths,
                    interactive=True,
                    metadata_cache=metadata_cache,
                )
            else:
                selected = [(w, l, m) for (w, l, m) in derived_items if l == class_name]
                panels = _build_panels_from_augmented(
                    layout,
                    items=selected,
                    max_per_class=end,
                    excluded_paths=excluded_paths,
                    interactive=True,
                    metadata_cache=metadata_cache,
                )
            cards = panels.get(class_name, [])
        elif tab == "frozen":
            if not _frozen_test_stale_context(layout, cfg):
                panels, _run, _has, _ctx = _build_panels_from_frozen_test(
                    layout, max_per_class=end, run_id=run_id, cfg=cfg,
                    interactive=True, metadata_cache=metadata_cache,
                )
                cards = panels.get(class_name, [])
        elif tab == "mined":
            panels = _build_panels_from_mined(
                layout, output_dir=cfg.mining.output_dir, max_per_class=end,
                interactive=True, metadata_cache=metadata_cache,
            )
            cards = panels.get(class_name, [])
    finally:
        _save_metadata_cache(layout, metadata_cache)
    return cards[offset:end]



# ---------------------------------------------------------------------------
# Read augmented samples from disk into clip-card panels
# ---------------------------------------------------------------------------


def _audit_card_fields(entry: object) -> dict:
    """Card fields from one augment-audit per-clip record (empty when unaudited)."""
    if not isinstance(entry, dict):
        return {
            "audit_pred": "",
            "audit_conf": None,
            "audit_correct": None,
            "audit_seen": False,
        }
    return {
        "audit_pred": str(entry.get("predicted_class") or ""),
        "audit_conf": entry.get("predicted_confidence"),
        "audit_correct": bool(entry.get("correct")),
        "audit_seen": bool(entry.get("seen_in_training")),
    }


def _load_augment_audit(layout: ProjectLayout) -> dict:
    """Latest `sed evaluate augmented` report for the project, or {}."""
    path = layout.dataset_inspect_dir / "augment_audit.json"
    if not path.exists():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return report if isinstance(report, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to read augment audit report %s: %s", path, exc)
        return {}


def _build_panels_from_augmented(
    layout: ProjectLayout,
    items: list[tuple[Path, str, dict]] | None = None,
    max_per_class: int = 200,
    excluded_paths: set[str] | None = None,
    interactive: bool = False,
    metadata_cache: dict | None = None,
) -> dict[str, list[dict]]:
    """Build panels for explicitly assigned derived clips."""
    items = items if items is not None else []
    if not items:
        return {}
    excluded_paths = excluded_paths or set()
    audit = _load_augment_audit(layout)
    audit_clips = audit.get("per_clip", {}) if isinstance(audit, dict) else {}
    panels: dict[str, list[dict]] = {}
    for wav, cls, meta in items:
        bucket = panels.setdefault(cls, [])
        if max_per_class > 0 and len(bucket) >= max_per_class:
            continue
        if interactive:
            sr, duration = _get_cached_metadata(wav, metadata_cache or {})
            spec_b64 = ""
        else:
            try:
                data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=-1)
                duration = len(data) / sr
            except Exception:  # noqa: BLE001
                continue
            spec_b64 = _spectrogram_thumb(data, sr)
        technique = meta.get("augmentation", "unknown")
        if not isinstance(technique, str):
            # legacy dict-shaped value — fall back to a generic label
            technique = "unknown"
        params = meta.get("augmentation_params") if isinstance(meta.get("augmentation_params"), dict) else {}
        source_meta = _find_source_sidecar(layout, meta.get("source", ""))
        card = {
            "path": str(wav),
            "name": wav.name,
            "uuid": meta.get("uuid", ""),
            "session_id": "",
            "duration_s": float(duration),
            "spec_b64": spec_b64,
            "source": meta.get("source", ""),
            "augmentation": technique,
            "augmentation_params": meta.get("augmentation_params", {}),
            "origin": meta.get("source_origin") or meta.get("origin", ""),
            "description": _clean_description(
                meta.get("description") or source_meta.get("description", "")
            ),
            "tags": _normalize_tags(meta.get("tags") or source_meta.get("tags", [])),
            "masking_score_db": meta.get("masking_score_db"),
            "masking_score_p25_db": meta.get("masking_score_p25_db"),
            "masking_score_min_db": meta.get("masking_score_min_db"),
            "mask_target_db": meta.get("mask_target_db"),
            "mask_background_db": meta.get("mask_background_db"),
            "effective_snr_db": meta.get("effective_snr_db", params.get("effective_snr_db")),
            "mix_png": str(wav.with_suffix(".mix.png")) if wav.with_suffix(".mix.png").exists() else "",
            **_audit_card_fields(audit_clips.get(str(wav))),
            "base_label": meta.get("base_label") or params.get("actual_base_label") or params.get("base_label") or "",
            "overlay_label": meta.get("overlay_label") or params.get("actual_overlay_label") or params.get("overlay_label") or "",
            "recipe_id": meta.get("recipe_id") or params.get("recipe_id") or "",
            "assigned_target_class": meta.get("assigned_target_class", ""),
            "sample_rate": int(sr),
            "transient_mel_db": meta.get("transient_mel_db"),
            "source_peak_dbfs": source_meta.get("peak_dbfs"),
            "source_rms_dbfs": source_meta.get("rms_dbfs"),
            "source_onset_strength": source_meta.get("onset_strength"),
            "source_onset_baseline": source_meta.get("onset_baseline"),
            "source_transient_mel_db": source_meta.get("transient_mel_db"),
            "source_recorded_channels": source_meta.get("recorded_channels", []),
            "source_triggered_channels": source_meta.get("triggered_channels", []),
            "source_triggered_flux": source_meta.get("triggered_flux", {}),
            "source_triggered_peak_dbfs": source_meta.get("triggered_peak_dbfs", {}),
            "source_microphone_type": source_meta.get("microphone_type", ""),
            "recorded_channels": meta.get("recorded_channels", source_meta.get("recorded_channels", [])),
            "triggered_channels": meta.get("triggered_channels", source_meta.get("triggered_channels", [])),
            "triggered_flux": meta.get("triggered_flux", source_meta.get("triggered_flux", {})),
            "triggered_peak_dbfs": meta.get("triggered_peak_dbfs", source_meta.get("triggered_peak_dbfs", {})),
            "clip_kind": meta.get("clip_kind") or source_meta.get("clip_kind", ""),
            "channel_index": meta.get("channel_index", source_meta.get("channel_index")),
            "channel_count": meta.get("channel_count", source_meta.get("channel_count")),
            "microphone": meta.get("microphone", source_meta.get("microphone")),
            "microphone_label": meta.get("microphone_label", source_meta.get("microphone_label", "")),
            "microphone_type": meta.get("microphone_type", source_meta.get("microphone_type", "")),
            "device": meta.get("device", source_meta.get("device", "")),
            "mic_position": meta.get("mic_position", source_meta.get("mic_position", "")),
            "soft_excluded": str(wav.resolve()) in excluded_paths,
        }
        card["search_text"] = _augmentation_search_text(card)
        card["meta_text"] = _format_meta_text(card)
        card["meta_html"] = _format_meta_html(card)
        bucket.append(card)
    return panels



def _build_panels_from_mined(
    layout: ProjectLayout,
    output_dir: str = "mined",
    max_per_class: int = 200,
    interactive: bool = False,
    metadata_cache: dict | None = None,
) -> dict[str, list[dict]]:
    """Build panels for runtime-mined false-positive candidates."""
    from sed.config import LIBRARY_DIR
    root = LIBRARY_DIR / output_dir
    if not root.exists():
        return {}
    panels: dict[str, list[dict]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        bucket = panels.setdefault(class_dir.name, [])
        try:
            wavs = sorted(class_dir.rglob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            wavs = sorted(class_dir.rglob("*.wav"), key=lambda p: p.name, reverse=True)
        for wav in wavs:
            if max_per_class > 0 and len(bucket) >= max_per_class:
                break
            meta = _load_sidecar(wav)
            if interactive:
                sr, duration = _get_cached_metadata(wav, metadata_cache or {})
                spec_b64 = ""
            else:
                try:
                    data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
                    if data.ndim > 1:
                        data = data.mean(axis=-1)
                    duration = len(data) / sr
                except Exception:  # noqa: BLE001
                    continue
                spec_b64 = _spectrogram_thumb(data, sr)
            predicted = str(meta.get("predicted_class") or class_dir.name)
            confidence = float(meta.get("confidence") or 0.0)
            probs_percent = _prob_percent_map(meta, "probs_percent", "probs")
            smoothed_percent = _prob_percent_map(meta, "smoothed_percent", "smoothed")
            card = {
                "path": str(wav),
                "name": wav.name,
                "uuid": meta.get("uuid", ""),
                "session_id": meta.get("session_id", ""),
                "duration_s": float(duration),
                "spec_b64": spec_b64,
                "augmentation": "none",
                "augmentation_params": {},
                "origin": meta.get("origin", ""),
                "description": meta.get("description", ""),
                "tags": meta.get("tags", []),
                "sample_rate": int(sr),
                "predicted_class": predicted,
                "confidence": confidence,
                "kind": meta.get("kind", ""),
                "created_at": meta.get("created_at", ""),
                "peak_dbfs": meta.get("peak_dbfs"),
                "rms_dbfs": meta.get("rms_dbfs"),
                "best_channel": meta.get("best_channel"),
                "probs_percent": probs_percent,
                "smoothed_percent": smoothed_percent,
                "prob_html": _format_prob_html(smoothed_percent or probs_percent),
                "search_text": " ".join([
                    wav.name.lower(),
                    predicted.lower(),
                    str(meta.get("kind", "")).lower(),
                    str(meta.get("created_at", "")).lower(),
                ]),
            }
            card["meta_html"] = _format_mined_meta_html(card)
            bucket.append(card)
    return {cls: clips for cls, clips in panels.items() if clips}



def _prob_percent_map(meta: dict, percent_key: str, raw_key: str) -> dict[str, float]:
    percent = meta.get(percent_key)
    if isinstance(percent, dict):
        return {str(k): float(v) for k, v in percent.items()}
    raw = meta.get(raw_key)
    if isinstance(raw, dict):
        return {str(k): float(v) * 100.0 for k, v in raw.items()}
    return {}


def _format_prob_html(values: dict[str, float]) -> str:
    if not values:
        return "<div class='prob-empty'>no class probabilities in sidecar</div>"
    rows = []
    for cls, value in sorted(values.items(), key=lambda item: item[1], reverse=True):
        width = max(0.0, min(100.0, float(value)))
        rows.append(
            "<div class='prob-row'>"
            f"<span class='prob-name'>{cls}</span>"
            f"<span class='prob-bar'><span style='width:{width:.1f}%'></span></span>"
            f"<span class='prob-val'>{value:.1f}%</span>"
            "</div>"
        )
    return "".join(rows)


def _format_mined_meta_html(card: dict) -> str:
    parts = [
        f"<b>{card.get('predicted_class', '')}</b>",
        f"conf {float(card.get('confidence') or 0.0) * 100.0:.1f}%",
    ]
    peak = card.get("peak_dbfs")
    if peak is not None:
        parts.append(f"peak {float(peak):.1f} dBFS")
    rms = card.get("rms_dbfs")
    if rms is not None:
        parts.append(f"rms {float(rms):.1f} dBFS")
    best = card.get("best_channel")
    if best is not None:
        parts.append(f"mic {best}")
    created = card.get("created_at")
    if created:
        parts.append(str(created))
    return "<div class='meta-line'>" + " · ".join(parts) + "</div>"


def _latest_prediction_report(
    layout: ProjectLayout,
    run_id: str | None = None,
) -> tuple[Path | None, str]:
    """Return the preferred test_predictions.tsv and the run id it came from."""
    if run_id:
        candidate = layout.model_dir(run_id) / "reports" / "test_predictions.tsv"
        if candidate.exists():
            return candidate, run_id
        return None, run_id

    selected = _selected_run_id(layout)
    if selected:
        candidate = layout.model_dir(selected) / "reports" / "test_predictions.tsv"
        if candidate.exists():
            return candidate, selected

    candidates = sorted(
        layout.models_dir.glob("*/reports/test_predictions.tsv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if layout.models_dir.exists() else []
    if candidates:
        run_id = candidates[0].parents[1].name
        return candidates[0], run_id
    return None, ""


def _load_test_predictions(
    layout: ProjectLayout,
    run_id: str | None = None,
) -> tuple[dict[str, dict], str]:
    """Load per-clip frozen-test predictions keyed by absolute/resolved path."""
    report, run_id = _latest_prediction_report(layout, run_id=run_id)
    if report is None:
        return {}, ""
    out: dict[str, dict] = {}
    try:
        with open(report, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                raw_path = row.get("path", "")
                if not raw_path:
                    continue
                out[raw_path] = row
                try:
                    out[str(Path(raw_path).resolve())] = row
                except OSError:
                    pass
    except Exception as exc:  # noqa: BLE001
        log.warning("inspect: failed to read frozen-test predictions %s: %s", report, exc)
        return {}, ""
    return out, run_id


def _prediction_for_path(predictions: dict[str, dict], path: Path) -> dict:
    return predictions.get(str(path)) or predictions.get(str(path.resolve())) or {}


def _float_or_none(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _boolish(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _prediction_prob_percent(pred: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in pred.items():
        if not key.startswith("p:"):
            continue
        prob = _float_or_none(value)
        if prob is None:
            continue
        values[key[2:]] = prob * 100.0
    return values


def _frozen_prediction_card_fields(
    wav: Path,
    cls: str,
    meta: dict,
    pred: dict,
) -> dict:
    true_class = pred.get("true_class") or cls
    predicted_class = pred.get("predicted_class") or ""
    correct = _boolish(pred.get("correct"))
    triggered = _boolish(pred.get("triggered"))
    if correct is None and predicted_class:
        correct = predicted_class == true_class
    if triggered is None and predicted_class:
        triggered = predicted_class != "_negative"
    probs_percent = _prediction_prob_percent(pred)
    return {
        "path": str(wav),
        "true_class": true_class,
        "predicted_class": predicted_class or "not_evaluated",
        "correct": bool(correct) if correct is not None else False,
        "triggered": bool(triggered) if triggered is not None else False,
        "decision_reason": pred.get("decision_reason", "not_evaluated"),
        "decision_threshold": _float_or_none(pred.get("decision_threshold")),
        "predicted_confidence": _float_or_none(pred.get("predicted_confidence")),
        "argmax_class": pred.get("argmax_class") or "",
        "argmax_confidence": _float_or_none(pred.get("argmax_confidence")),
        "second_class": pred.get("second_class", ""),
        "second_confidence": _float_or_none(pred.get("second_confidence")),
        "margin": _float_or_none(pred.get("margin")),
        "probs_percent": probs_percent,
        "prob_html": _format_prob_html(probs_percent),
        "has_prediction": bool(pred),
        "origin": meta.get("test_origin") or meta.get("origin", ""),
        "augmentation": meta.get("augmentation", "none")
        if isinstance(meta.get("augmentation", "none"), str)
        else "unknown",
        "description": meta.get("description", ""),
        "tags": meta.get("tags", []),
    }


def _format_frozen_prediction_html(card: dict) -> str:
    lines: list[str] = []
    status = "correct" if card.get("correct") else "wrong"
    triggered = "trigger" if card.get("triggered") else "no trigger"
    lines.append(
        "<div class='meta-line'>"
        f"<b>true {html.escape(str(card.get('true_class', '')))}</b>"
        f" · predicted <b>{html.escape(str(card.get('predicted_class', '')))}</b>"
        f" · {status} · {triggered}"
        "</div>"
    )
    decision_parts = []
    reason = card.get("decision_reason")
    if reason:
        decision_parts.append(f"reason {reason}")
    threshold = card.get("decision_threshold")
    if threshold is not None:
        decision_parts.append(f"threshold {float(threshold):.3f}")
    pred_conf = card.get("predicted_confidence")
    if pred_conf is not None:
        decision_parts.append(f"pred conf {float(pred_conf) * 100.0:.1f}%")
    margin = card.get("margin")
    if margin is not None:
        decision_parts.append(f"margin {float(margin) * 100.0:.1f} pp")
    if decision_parts:
        lines.append("<div class='meta-line'>" + " · ".join(decision_parts) + "</div>")
    return "<div class='frozen-pred-meta'>" + "".join(lines) + "</div>"


def _format_frozen_sample_html(card: dict) -> str:
    lines: list[str] = []
    audio_parts = []
    origin = card.get("origin")
    if origin:
        audio_parts.append(f"origin {origin}")
    source = card.get("source")
    if source:
        audio_parts.append(f"src {source}")
    augmentation = card.get("augmentation")
    if augmentation and augmentation != "none":
        audio_parts.append(f"aug {augmentation}")
    dur = card.get("duration_s")
    if dur is not None:
        audio_parts.append(f"{float(dur):.2f}s")
    sr = card.get("sample_rate")
    if sr:
        audio_parts.append(f"{int(sr) // 1000}kHz")
    peak = card.get("peak_dbfs")
    if peak is not None:
        audio_parts.append(f"peak {float(peak):+.1f}dBFS")
    rms = card.get("rms_dbfs")
    if rms is not None:
        audio_parts.append(f"rms {float(rms):+.1f}dBFS")
    flux = card.get("source_onset_strength")
    if flux and flux > 0:
        import math
        audio_parts.append(f"flux {20 * math.log10(float(flux) + 1e-12):+.0f}dB")
    flux_by_mic = _map_values_db(card.get("triggered_flux") or card.get("source_triggered_flux"), flux=True)
    if flux_by_mic:
        audio_parts.append(f"mic flux {'/'.join(flux_by_mic)}")
    peak_by_mic = _map_values_db(card.get("triggered_peak_dbfs") or card.get("source_triggered_peak_dbfs"))
    if peak_by_mic:
        audio_parts.append(f"mic peak {'/'.join(peak_by_mic)}")
    tmel = card.get("transient_mel_db")
    if tmel is not None:
        audio_parts.append(f"tmel {float(tmel):+.1f}dB")
    if audio_parts:
        lines.append("<div class='meta-line'>" + " · ".join(str(p) for p in audio_parts) + "</div>")
    mic_parts = _microphone_meta_parts(card)
    if mic_parts:
        lines.append(
            "<div class='meta-line'><span class='meta-label'>mic:</span> "
            + html.escape(" · ".join(mic_parts))
            + "</div>"
        )
    lines.append(_description_tags_html(card))
    return "<div class='frozen-sample-meta'>" + "".join(lines) + "</div>"


def _format_frozen_meta_html(card: dict) -> str:
    return _format_frozen_prediction_html(card) + _format_frozen_sample_html(card)


def _frozen_prediction_payload_for_card(card: dict) -> dict:
    return {
        "path": card["path"],
        "true_class": card["true_class"],
        "predicted_class": card["predicted_class"],
        "correct": bool(card["correct"]),
        "triggered": bool(card["triggered"]),
        "decision_reason": card["decision_reason"],
        "predicted_confidence": card["predicted_confidence"],
        "argmax_confidence": card["argmax_confidence"],
        "margin": card["margin"],
        "has_prediction": bool(card["has_prediction"]),
        "prediction_html": _format_frozen_prediction_html(card),
        "prob_html": card["prob_html"],
        "search_text": _frozen_search_text(card),
    }


def _frozen_search_text(card: dict) -> str:
    pieces = [
        card.get("name", ""),
        card.get("true_class", ""),
        card.get("predicted_class", ""),
        card.get("decision_reason", ""),
        card.get("argmax_class", ""),
        card.get("origin", ""),
        card.get("augmentation", ""),
        _description_tags_search_text(card),
    ]
    return " ".join(str(p) for p in pieces if p).lower()


def _build_panels_from_frozen_test(
    layout: ProjectLayout,
    max_per_class: int = 200,
    run_id: str | None = None,
    cfg: Config | None = None,
    interactive: bool = False,
    metadata_cache: dict | None = None,
) -> tuple[dict[str, list[dict]], str, bool, dict]:
    """Build panels for composed frozen-test clips plus latest predictions."""
    items, context = _gather_frozen_test(layout, run_id=run_id, cfg=cfg)
    if not items:
        return {}, context.get("viewed_run_id", ""), False, context
    predictions, predictions_run_id = _load_test_predictions(layout, run_id=run_id)
    effective_run_id = predictions_run_id or context.get("viewed_run_id", "")
    context["viewed_run_id"] = effective_run_id
    panels: dict[str, list[dict]] = {}
    for wav, cls, meta in items:
        if not cls:
            continue
        bucket = panels.setdefault(cls, [])
        if max_per_class > 0 and len(bucket) >= max_per_class:
            continue
        if interactive:
            sr, duration = _get_cached_metadata(wav, metadata_cache or {})
            spec_b64 = ""
        else:
            try:
                data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=-1)
                duration = len(data) / sr
            except Exception:  # noqa: BLE001
                continue
            spec_b64 = _spectrogram_thumb(data, sr)
        augmentation = meta.get("augmentation", "none")
        if not isinstance(augmentation, str):
            augmentation = "unknown"
        pred = _prediction_for_path(predictions, wav)
        card = {
            "path": str(wav),
            "name": wav.name,
            "uuid": meta.get("uuid", ""),
            "session_id": meta.get("test_session_id") or meta.get("session_id", ""),
            "duration_s": float(duration),
            "spec_b64": spec_b64,
            "sample_rate": int(sr),
            "origin": meta.get("test_origin") or meta.get("origin", ""),
            "source": meta.get("source", ""),
            "augmentation": augmentation,
            "augmentation_params": meta.get("augmentation_params", {}),
            "peak_dbfs": meta.get("peak_dbfs"),
            "rms_dbfs": meta.get("rms_dbfs"),
            "transient_mel_db": meta.get("transient_mel_db"),
            "source_onset_strength": meta.get("onset_strength"),
            "source_onset_baseline": meta.get("onset_baseline"),
            "recorded_channels": meta.get("recorded_channels", []),
            "triggered_channels": meta.get("triggered_channels", []),
            "triggered_flux": meta.get("triggered_flux", {}),
            "triggered_peak_dbfs": meta.get("triggered_peak_dbfs", {}),
            "clip_kind": meta.get("clip_kind", ""),
            "channel_index": meta.get("channel_index"),
            "channel_count": meta.get("channel_count"),
            "microphone": meta.get("microphone"),
            "microphone_label": meta.get("microphone_label", ""),
            "microphone_type": meta.get("microphone_type", ""),
            "device": meta.get("device", ""),
            "mic_position": meta.get("mic_position", ""),
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
        }
        card.update(_frozen_prediction_card_fields(wav, cls, card, pred))
        card["meta_html"] = _format_frozen_meta_html(card)
        card["search_text"] = _frozen_search_text(card)
        bucket.append(card)
    return dict(sorted(panels.items())), effective_run_id, bool(predictions), context



def frozen_run_prediction_payload(layout: ProjectLayout, run_id: str | None = None) -> dict:
    """Return only frozen-test prediction metadata for in-browser run switching.

    This intentionally avoids thumbnail generation and audio decoding; the
    sample cards already exist in the browser and only the prediction rows,
    probability bars, and filter attributes need to change.
    """
    items, context = _gather_frozen_test(layout, run_id=run_id)
    predictions, predictions_run_id = _load_test_predictions(layout, run_id=run_id)
    effective_run_id = predictions_run_id or context.get("viewed_run_id", "")
    context["viewed_run_id"] = effective_run_id
    by_path: dict[str, dict] = {}
    for wav, cls, meta in items:
        pred = _prediction_for_path(predictions, wav)
        card = _frozen_prediction_card_fields(wav, cls, meta, pred)
        payload = _frozen_prediction_payload_for_card(card)
        by_path[str(wav)] = payload
        try:
            by_path[str(wav.resolve())] = payload
        except OSError:
            pass
    return {
        "ok": True,
        "run_id": effective_run_id,
        "has_predictions": bool(predictions),
        "context": context,
        "cards": by_path,
    }


def _build_panels(
    items: list[tuple[Path, str, dict, float]],
    classes: list[str],
    max_per_class: int,
    is_rerender: bool = False,
    excluded_paths: set[str] | None = None,
    interactive: bool = False,
    metadata_cache: dict | None = None,
) -> dict[str, list[dict]]:
    panels: dict[str, list[dict]] = {c: [] for c in classes}
    excluded_paths = excluded_paths or set()
    # Within a "recorded together" split bucket, order cards by channel index
    # (ch0, ch1, …). Stable, so non-split buckets keep their existing order.
    items = sorted(
        items,
        key=lambda it: it[2].get("channel_index") if it[2].get("channel_index") is not None else -1,
    )
    desc = "[cyan]Refreshing thumbnails..." if is_rerender else "[cyan]Generating thumbnails..."
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        transient=is_rerender,
    ) as progress:
        task_id = progress.add_task(desc, total=len(items))
        for path, label, meta, duration in items:
            if max_per_class > 0 and len(panels[label]) >= max_per_class:
                progress.advance(task_id)
                continue
            if interactive:
                sr, dur = _get_cached_metadata(path, metadata_cache or {})
                spec_b64 = ""
                duration = dur
            else:
                try:
                    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
                    if data.ndim > 1:
                        data = data.mean(axis=-1)
                except Exception:  # noqa: BLE001
                    continue
                spec_b64 = _spectrogram_thumb(data, sr)
            card = {
                "path": str(path),
                "name": path.name,
                "uuid": meta.get("uuid", ""),
                "session_id": meta.get("session_id", ""),
                "duration_s": float(duration),
                "spec_b64": spec_b64,
                "augmentation": "none",
                "augmentation_params": {},
                "sample_rate": int(sr),
                "source_onset_strength": meta.get("onset_strength"),
                "source_onset_baseline": meta.get("onset_baseline"),
                "source_peak_dbfs": meta.get("peak_dbfs"),
                "source_rms_dbfs": meta.get("rms_dbfs"),
                "source_transient_mel_db": meta.get("transient_mel_db"),
                "origin": meta.get("origin", ""),
                "description": meta.get("description", ""),
                "tags": meta.get("tags", []),
                "recorded_channels": meta.get("recorded_channels", []),
                "triggered_channels": meta.get("triggered_channels", []),
                "triggered_flux": meta.get("triggered_flux", {}),
                "triggered_peak_dbfs": meta.get("triggered_peak_dbfs", {}),
                "clip_kind": meta.get("clip_kind", ""),
                "channel_index": meta.get("channel_index"),
                "channel_count": meta.get("channel_count"),
                "microphone": meta.get("microphone"),
                "microphone_label": meta.get("microphone_label", ""),
                "microphone_type": meta.get("microphone_type", ""),
                "device": meta.get("device", ""),
                "mic_position": meta.get("mic_position", ""),
                "assigned_target_class": meta.get("assigned_target_class", ""),
                "library_label": meta.get("library_label") or meta.get("label") or label,
                "soft_excluded": str(path.resolve()) in excluded_paths,
            }
            card["meta_text"] = _format_meta_text(card)
            card["meta_html"] = _format_meta_html(card)
            card["search_text"] = " ".join([
                card["name"].lower(),
                str(card.get("assigned_target_class") or "").lower(),
                str(card.get("library_label") or "").lower(),
                _description_tags_search_text(card),
            ]).strip()
            panels[label].append(card)

            progress.advance(task_id)
    return panels


def _render_html(
    project_id: str,
    class_summary: dict,
    panels: dict[str, list[dict]],
    aug_panels: dict[str, list[dict]] | None = None,
    frozen_panels: dict[str, list[dict]] | None = None,
    frozen_predictions_run_id: str = "",
    frozen_has_predictions: bool = False,
    frozen_context: dict | None = None,
    mined_panels: dict[str, list[dict]] | None = None,
    library_items: list[dict] | None = None,
    model_classes: list[str] | None = None,
    interactive: bool = False,
    class_images: dict[str, str] | None = None,
) -> str:
    from jinja2 import Template
    aug_panels = aug_panels or {}
    frozen_panels = frozen_panels or {}
    frozen_context = frozen_context or {}
    mined_panels = mined_panels or {}
    library_items = library_items or []
    model_classes = model_classes or sorted(class_summary)
    class_images = class_images or {}
    aug_total = sum(len(clips) for clips in aug_panels.values())
    frozen_total = sum(len(clips) for clips in frozen_panels.values())
    mined_total = sum(len(clips) for clips in mined_panels.values())

    if interactive:
        panels_json = json.dumps(panels)
        aug_panels_json = json.dumps(aug_panels)
        frozen_panels_json = json.dumps(frozen_panels)
        frozen_context_json = json.dumps(frozen_context)
        mined_panels_json = json.dumps(mined_panels)
        library_items_json = json.dumps(library_items)
    else:
        panels_json = "{}"
        aug_panels_json = "{}"
        frozen_panels_json = "{}"
        frozen_context_json = "{}"
        mined_panels_json = "{}"
        library_items_json = "[]"

    tpl = Template(_HTML_TEMPLATE)
    return tpl.render(
        project_id=project_id,
        class_summary=class_summary,
        panels=panels,
        panels_json=panels_json,
        aug_panels=aug_panels,
        aug_panels_json=aug_panels_json,
        aug_total=aug_total,
        frozen_panels=frozen_panels,
        frozen_panels_json=frozen_panels_json,
        frozen_context_json=frozen_context_json,
        frozen_total=frozen_total,
        frozen_predictions_run_id=frozen_predictions_run_id,
        frozen_has_predictions=frozen_has_predictions,
        frozen_context=frozen_context,
        mined_panels=mined_panels,
        mined_panels_json=mined_panels_json,
        mined_total=mined_total,
        library_items=library_items,
        library_items_json=library_items_json,
        library_total=len(library_items),
        model_classes=model_classes,
        interactive=interactive,
        class_images=class_images,
    )



_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SED Inspection — {{ project_id }}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap">
<style>
body { font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #fafafa; color: #1e293b; max-width: 1300px; margin: 0 auto;
       padding: 0 1.5rem 2rem 1.5rem; -webkit-font-smoothing: antialiased; }
h1 { font-size: 2.25rem; font-weight: 700; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; margin-top: 2rem; color: #0f172a; }
h2 { margin-top: 1.8rem; font-size: 1.5rem; font-weight: 600; color: #1e293b; }
h3.section { background: #1e293b; color: #fff; padding: 0.6rem 1rem; border-radius: 8px; font-weight: 500; font-size: 1.1rem; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
           gap: 0.75rem; margin-bottom: 1.5rem; }
.card { background: #fff; padding: 1rem 1.25rem; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.02);
        border: 1px solid #f1f5f9; }
.clips { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
          gap: 0.75rem; }
.clip { background: #fff; padding: 0.75rem; border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04); position: relative;
        border: 1px solid #f1f5f9;
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s, background-color 0.2s; }
.clip.warn { border-left: 4px solid #f59e0b; }
.warn-text { color: #b45309; font-weight: 600; }
.clip.selected { background: #fef2f2; opacity: 0.65; border-color: #fca5a5; }
.clip.selected .clip-body { filter: grayscale(50%); }
/* Whole-card click toggles selection (interactive mode). Excludes audio + checkbox. */
.interactive .clip .clip-body { cursor: pointer; }
.interactive .clip:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-color: #cbd5e1; }
.interactive .clip:hover:not(.selected) { background: #faf5ff; }
.clip h3 { font-size: 0.88rem; font-weight: 600; margin: 0 0 0.35rem 0; color: #0f172a; display: flex; align-items: center; gap: 0.4rem; min-width: 0; }
/* Reserve room for the absolutely-positioned select checkbox so long names
   and pills never slide underneath it. */
.interactive .clip h3 { padding-right: 30px; }
.clip-name { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.clip-pills { flex: 0 0 auto; display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
.clip .meta { font-size: 0.78rem; color: #64748b; margin-bottom: 0.5rem; }
.clip .meta-line { margin-bottom: 3px; line-height: 1.45; overflow-wrap: anywhere; }
.clip .meta-label { color: #94a3b8; font-weight: 600; margin-right: 3px; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.02em; }
.tip { cursor: pointer; border-bottom: 1px dotted #94a3b8; }
.hover-tip {
  position: fixed; z-index: 2000; max-width: min(420px, calc(100vw - 24px));
  background: rgba(15,23,42,0.98); color: #fff; padding: 10px 12px;
  border-radius: 8px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
  font-size: 0.75rem; line-height: 1.5; white-space: pre-line; pointer-events: none;
  opacity: 0; transform: translateY(3px); transition: opacity 120ms ease, transform 120ms ease;
  border: 1px solid #334155;
}
.hover-tip.show { opacity: 1; transform: translateY(0); }
.clip img { display: block; width: 100%; border-radius: 6px; }
.clip img.spec {
  transition: box-shadow 0.15s ease-in-out;
}
.clip.playing img.spec {
  box-shadow: 0 0 0 3px #10b981, 0 0 12px rgba(16, 185, 129, 0.6);
}
.play-spec {
  cursor: pointer;
}
.clip .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.clip .pair-label { font-size: 0.7rem; text-align: center; color: #94a3b8; margin-top: 2px; }
audio { width: 100%; height: 36px; margin-top: 0.5rem; border-radius: 8px; }
.search-filter { display: inline-flex; align-items: center; gap: 0.5rem; margin: 0 0.5rem 0.5rem 0; }
.search-filter input {
  width: 18rem; max-width: 72vw; padding: 0.45rem 0.75rem; border: 1px solid #cbd5e1;
  border-radius: 8px; font: inherit; font-size: 0.875rem; background: #fff; outline: none; transition: border-color 0.2s;
}
.search-filter input:focus { border-color: #6366f1; box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1); }
.search-filter button {
  background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 8px;
  padding: 0.45rem 0.9rem; cursor: pointer; font-size: 0.875rem; font-weight: 500; transition: background 0.2s;
}
.search-filter button:hover { background: #e2e8f0; }
.notice { background: #fef3c7; padding: 0.75rem 1.25rem; border-left: 4px solid #f59e0b;
          margin: 1.5rem 0; border-radius: 0 8px 8px 0; font-size: 0.9rem; color: #78350f; }
.notice.danger { background: #fee2e2; border-left-color: #ef4444; color: #991b1b; }
.mask-good { color: #10b981; font-weight: 600; }
.mask-warn { color: #f59e0b; font-weight: 600; }
.mask-bad  { color: #ef4444; font-weight: 600; }

/* Interactive mode: checkbox, sticky toolbar */
.select-box { position: absolute; top: 8px; right: 10px;
              width: 22px; height: 22px; cursor: pointer; z-index: 2;
              accent-color: #ef4444; border-radius: 4px; transition: transform 0.1s; }
.select-box:hover { transform: scale(1.15); }
.class-select-all {
  width: 18px; height: 18px; cursor: pointer; vertical-align: middle;
  margin-right: 0.75rem; accent-color: #ef4444; border-radius: 4px; transition: transform 0.1s;
}
.class-select-all:hover { transform: scale(1.15); }
.toolbar { position: sticky; top: 0; z-index: 100; background: #0f172a; color: #fff;
           padding: 0.75rem 1.5rem; margin: 0 -1.5rem 1.5rem -1.5rem;
           display: flex; gap: 1.5rem; align-items: center; justify-content: space-between;
           box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }
.toolbar .count { font-size: 1.15rem; font-weight: 600; }
.toolbar .count .num { color: #fca5a5; font-size: 1.25rem; }
.toolbar .hint { font-size: 0.85rem; color: #94a3b8; }
.toolbar .selection-summary { display: flex; gap: 1.5rem; align-items: center; min-width: 0; }
.toolbar .actions { display: flex; gap: 0.5em; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
.toolbar .status, .toolbar .refresh-status { font-size: 0.85rem; color: #94a3b8; min-width: 100px; text-align: right; font-weight: 500; }
.toolbar .job-status {
    display: inline-flex; align-items: center; max-width: min(520px, 42vw);
    padding: 0.32rem 0.55rem; border: 1px solid rgba(251,191,36,0.4);
    border-radius: 999px; background: rgba(30,41,59,0.92); color: #fde68a;
    font-size: 0.78rem; font-weight: 650; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
.toolbar .job-status[hidden] { display: none; }
.toolbar .job-status.err { color: #fecaca; border-color: rgba(248,113,113,0.5); }
.toolbar .status.ok { color: #4ade80; }
.toolbar .status.err { color: #f87171; }
.toolbar button.done {
    background: #10b981; color: #fff; border: none; padding: 0.5rem 1.25rem;
    border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 0.95rem; transition: background 0.2s;
}
.toolbar button.done:hover { background: #059669; }
.toolbar button.done:disabled { background: #475569; cursor: default; }
.toolbar button.del {
    background: #ef4444; color: #fff; border: none; padding: 0.5rem 1.25rem;
    border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 0.95rem; transition: background 0.2s;
}
.toolbar button.del:hover { background: #dc2626; }
.toolbar button.del:disabled { background: #475569; cursor: default; }
.toolbar button.refresh {
    background: #2563eb; color: #fff; border: none; padding: 0.5rem 1.25rem;
    border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 0.95rem; transition: background 0.2s;
}
.toolbar button.refresh:hover { background: #1d4ed8; }
.toolbar button.refresh:disabled { background: #475569; cursor: wait; }
.toolbar button.primary-action {
    background: #7c3aed; color: #fff; border: none; padding: 0.5rem 1.25rem;
    border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 0.95rem; transition: background 0.2s;
}
.toolbar button.primary-action:hover { background: #6d28d9; }
.toolbar button.primary-action:disabled { background: #475569; cursor: wait; }
body.ui-busy, body.ui-busy * { cursor: wait !important; }
.library-active .toolbar { justify-content: flex-end; }
.library-active .toolbar .selection-only { display: none; }
.clip.deleted { display: none; }
.kbd { background: #334155; padding: 2px 6px; border-radius: 4px; font-family: monospace;
       font-size: 0.8rem; color: #f1f5f9; border: 1px solid #475569; }
.done-overlay {
    position: fixed; inset: 0; background: rgba(15,23,42,0.96); color: #fff;
    display: none; align-items: center; justify-content: center; flex-direction: column;
    z-index: 1000; font-size: 1.5rem; text-align: center; padding: 2rem;
}
.done-overlay.show { display: flex; }
.done-overlay .big { font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 700; color: #10b981; }
/* ---- Tabs (Dataset / Augmented) ----------------------------------------- */
.tabs { display: flex; gap: 0; margin: 1.5rem 0 1rem; border-bottom: 2px solid #1e293b; }
.tabs button {
  background: #f1f5f9; border: none; padding: 0.75rem 1.75rem; cursor: pointer;
  font-size: 1rem; font-weight: 600; color: #475569;
  border-radius: 8px 8px 0 0; margin-right: 4px; border: 1px solid transparent; border-bottom: none; transition: all 0.2s;
}
.tabs button:hover { background: #e2e8f0; color: #0f172a; }
.tabs button.active { background: #1e293b; color: #fff; border-color: #1e293b; }
.tabs button .count { opacity: 0.65; font-weight: 400; margin-left: 0.5rem; font-size: 0.85rem; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
/* ---- Collapsible per-class sections (<details>) ------------------------- */
.class-block {
  background: #fff; border-radius: 10px; margin: 0.5rem 0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.02); border: 1px solid #e2e8f0;
}
.class-block > summary {
  cursor: pointer; padding: 0.8rem 1.2rem; font-weight: 600; font-size: 1.1rem;
  user-select: none; list-style: none; display: flex; align-items: center; transition: background 0.15s;
  border-radius: 10px;
}
.class-block > summary::-webkit-details-marker { display: none; }
.class-block > summary::before {
  content: "▶"; display: inline-block; margin-right: 0.75rem;
  transition: transform 0.2s; font-size: 0.75rem; color: #94a3b8;
}
.class-block[open] > summary { border-radius: 10px 10px 0 0; border-bottom: 1px solid #f1f5f9; }
.class-block[open] > summary::before { transform: rotate(90deg); }
.class-block > summary:hover { background: #f8fafc; }
.class-block > summary .count { opacity: 0.6; font-weight: 400; margin-left: 0.5rem; font-size: 0.85rem; }
.class-block > summary .flagct { color: #ef4444; font-weight: 600; margin-left: 1rem; font-size: 0.85rem; }
.class-block .clips { padding: 1rem; }
.class-ref { display: block; max-height: 200px; max-width: 100%;
             object-fit: contain; margin: 0.5rem 1.2rem 1rem;
             border-radius: 8px; background: #fafafa;
             box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
.expand-all {
  background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.4rem 1rem;
  border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 500; margin-bottom: 1rem; margin-right: 0.25rem; transition: background 0.2s;
}
.expand-all:hover { background: #e2e8f0; }

.load-more-btn {
  grid-column: 1 / -1; background: #f8fafc; border: 2px dashed #cbd5e1;
  padding: 0.85rem; border-radius: 10px; cursor: pointer; font-size: 0.95rem; font-weight: 600;
  color: #475569; text-align: center; transition: all 0.2s; margin-top: 1rem;
}
.load-more-btn:hover { background: #f1f5f9; border-color: #94a3b8; color: #0f172a; }

/* Predictions pills and margins style */
.pill { display: inline-flex; align-items: center; justify-content: center;
         padding: 1px 7px; border-radius: 10px; font-size: 0.66rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
.pill.good { background: #d1fae5; color: #065f46; }
.pill.bad { background: #fee2e2; color: #991b1b; }
.pill.neg { background: #f1f5f9; color: #475569; }
.prob-box { background: #f8fafc; border-radius: 8px; padding: 6px 10px; font-size: 0.74rem; border: 1px solid #e2e8f0; margin-top: 0.5rem; display: grid; gap: 3px; }
.prob-row { display: grid; grid-template-columns: minmax(72px, max-content) 1fr 46px; align-items: center; gap: 6px; min-width: 0; }
.prob-name { color: #475569; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.prob-bar { display: block; height: 7px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
.prob-bar span { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #818cf8, #6366f1); min-width: 1px; }
.prob-val { text-align: right; font-variant-numeric: tabular-nums; color: #334155; }
/* First row is the highest probability — emphasize it, tinted by verdict. */
.prob-row:first-child .prob-name, .prob-row:first-child .prob-val { font-weight: 700; color: #0f172a; }
.frozen-correct .prob-row:first-child .prob-bar span { background: linear-gradient(90deg, #34d399, #10b981); }
.frozen-wrong .prob-row:first-child .prob-bar span { background: linear-gradient(90deg, #f87171, #ef4444); }
.prob-empty { color: #94a3b8; }

.aug-filters { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-bottom: 1rem; background: #f8fafc; padding: 0.5rem; border-radius: 8px; border: 1px solid #e2e8f0; }
.aug-filters button { background: #fff; border: 1px solid #cbd5e1; padding: 0.4rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 500; transition: all 0.2s; }
.aug-filters button:hover { background: #f1f5f9; }
.aug-filters button.active { background: #1e293b; color: #fff; border-color: #1e293b; }
.aug-filters .bar-end { margin-left: auto; display: inline-flex; align-items: center; gap: 0.4rem; }
.aug-filters .expand-all { margin: 0; padding: 0.35rem 0.75rem; font-size: 0.78rem; }
.mask-filter { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; color: #475569; }
.mask-filter input, .mask-filter select { padding: 0.35rem 0.5rem; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.85rem; background: #fff; }

.run-select { background: #f8fafc; padding: 0.5rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 0.75rem; display: inline-flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; }
.run-select select { padding: 0.35rem 0.5rem; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.85rem; background: #fff; font-weight: 500; }
.run-preview-btn { border: 1px solid #cbd5e1; background: #fff; color: #334155; padding: 0.32rem 0.48rem; border-radius: 6px; cursor: pointer; font-size: 0.78rem; font-weight: 700; line-height: 1; }
.run-preview-btn:hover { background: #eef2ff; border-color: #a5b4fc; color: #3730a3; }
#frozen-run-status { font-size: 0.8rem; font-weight: 500; }
#frozen-run-status.busy { color: #f59e0b; }
#frozen-run-status.ok { color: #10b981; }
#frozen-run-status.err { color: #ef4444; }
.run-preview-modal[hidden] { display: none; }
.run-preview-modal { position: fixed; inset: 0; z-index: 1200; display: grid; place-items: center; padding: 1.5rem; }
.run-preview-backdrop { position: absolute; inset: 0; background: rgba(15,23,42,0.72); }
.run-preview-dialog { position: relative; width: min(980px, 96vw); max-height: 92vh; display: grid; grid-template-rows: auto minmax(0, 1fr); background: #fff; border-radius: 8px; box-shadow: 0 24px 60px rgba(15,23,42,0.35); border: 1px solid #cbd5e1; overflow: hidden; }
.run-preview-head { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; padding: 0.55rem 0.7rem; border-bottom: 1px solid #e2e8f0; background: #f8fafc; }
.run-preview-title { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.9rem; font-weight: 700; color: #0f172a; }
.run-preview-actions { display: inline-flex; align-items: center; gap: 0.35rem; }
.run-preview-actions button { border: 1px solid #cbd5e1; background: #fff; color: #334155; padding: 0.32rem 0.5rem; border-radius: 6px; cursor: pointer; font-size: 0.78rem; font-weight: 700; }
.run-preview-actions button.active { background: #1e293b; color: #fff; border-color: #1e293b; }
.run-preview-body { min-height: 260px; max-height: calc(92vh - 52px); overflow: auto; background: #fff; display: grid; place-items: center; padding: 0.75rem; }
.run-preview-body img { max-width: 100%; height: auto; display: block; }
.run-preview-body .preview-error { color: #991b1b; font-weight: 600; }

.mined-actions { display: inline-flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; background: #f8fafc; padding: 0.5rem; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 0.85rem; }
.mined-actions input { padding: 0.35rem 0.5rem; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.85rem; }
.mined-actions button { background: #6366f1; color: #fff; border: none; padding: 0.45rem 1rem; border-radius: 6px; cursor: pointer; font-weight: 600; transition: background 0.2s; }
.mined-actions button:hover { background: #4f46e5; }

.library-builder { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.85rem; }
.library-workbar { display: flex; flex-wrap: wrap; align-items: end; gap: 0.65rem; padding-bottom: 0.75rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 0.75rem; }
.library-workbar label, .library-recipe-fields label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.72rem; color: #64748b; font-weight: 700; text-transform: uppercase; }
.library-workbar select, .library-workbar input, .library-recipe-fields input, .library-recipe-fields select { min-width: 0; padding: 0.4rem 0.5rem; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.85rem; background: #fff; }
.library-workbar button, .library-rule-table button, .recipe-actions button { border: 1px solid #cbd5e1; background: #f8fafc; padding: 0.42rem 0.7rem; border-radius: 6px; cursor: pointer; font-weight: 600; color: #334155; }
.library-workbar button.primary, .recipe-actions button.primary { background: #1e293b; color: #fff; border-color: #1e293b; }
.library-workbar button.warn { background: #fef3c7; color: #92400e; border-color: #f59e0b; }
.library-workbar button.danger { background: #fff1f2; color: #991b1b; border-color: #fecaca; }
.library-workbar button:disabled { opacity: 0.55; cursor: wait; }
.library-workbar button.refresh { background: #eef2ff; color: #3730a3; border-color: #c7d2fe; }
.library-grid { display: grid; grid-template-columns: minmax(280px, 0.9fr) minmax(0, 1.25fr); gap: 0.85rem; align-items: start; }
.library-grid > *, .library-designer, .library-section { min-width: 0; }
.library-designer { display: grid; gap: 0.85rem; min-width: 0; }
.library-section { border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem; background: #fff; }
.library-section h3 { margin: 0 0 0.6rem 0; font-size: 0.95rem; color: #0f172a; display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
.section-action { border: 1px solid #cbd5e1; background: #f8fafc; color: #475569; font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.55rem; border-radius: 6px; cursor: pointer; }
.section-action:hover { background: #e2e8f0; }
.section-action:disabled { opacity: 0.55; cursor: wait; }
.section-refresh { display: inline-flex; align-items: center; justify-content: center; width: 1.75rem; height: 1.75rem; border: 1px solid #cbd5e1; background: #fff; color: #475569; border-radius: 6px; cursor: pointer; font-size: 0.95rem; line-height: 1; font-weight: 700; }
.section-refresh:hover { background: #eef2ff; border-color: #a5b4fc; color: #3730a3; }
.section-refresh:disabled { opacity: 0.55; cursor: wait; }
.section-refresh.spinning { animation: sed-spin 0.8s linear infinite; color: #3730a3; }
@keyframes sed-spin { to { transform: rotate(360deg); } }
.section-status { color: #64748b; font-size: 0.78rem; font-weight: 600; margin-left: 0.35rem; }
.section-status.err { color: #ef4444; }
.section-status.ok { color: #10b981; }
.section-head { display: inline-flex; align-items: center; gap: 0.35rem; }
.library-section h4 { margin: 0.85rem 0 0.45rem; font-size: 0.82rem; color: #475569; text-transform: uppercase; letter-spacing: 0.02em; }
.library-filters { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 0.5rem; margin-bottom: 0.75rem; }
.library-filters label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.72rem; color: #64748b; font-weight: 700; text-transform: uppercase; }
.library-filters input, .library-filters select { min-width: 0; padding: 0.4rem 0.5rem; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.85rem; background: #fff; }
.library-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem; margin-bottom: 0.75rem; }
.library-actions button { border: 1px solid #cbd5e1; background: #f8fafc; padding: 0.42rem 0.7rem; border-radius: 6px; cursor: pointer; font-weight: 600; color: #334155; }
.library-actions button.primary { background: #1e293b; color: #fff; border-color: #1e293b; }
.library-actions button.warn { background: #fef3c7; color: #92400e; border-color: #f59e0b; }
.library-actions select { padding: 0.42rem 0.5rem; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; }
.library-count { color: #64748b; font-size: 0.85rem; margin-left: auto; }
.library-table-wrap { overflow: auto; max-height: 58vh; border: 1px solid #e2e8f0; border-radius: 8px; }
.library-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.library-table th { position: sticky; top: 0; z-index: 1; background: #f8fafc; color: #475569; text-align: left; padding: 0.5rem; border-bottom: 1px solid #e2e8f0; white-space: nowrap; }
.library-table td { padding: 0.45rem 0.5rem; border-bottom: 1px solid #f1f5f9; white-space: nowrap; }
.library-table tbody tr { cursor: pointer; }
.library-table tr.library-group-row td { background: #f1f5f9; border-top: 1px solid #e2e8f0; font-size: 0.86rem; padding: 0.45rem 0.6rem; }
.library-table tr.library-group-row:hover td { background: #e2e8f0; }
.group-toggle { display: inline-block; width: 1rem; color: #64748b; }
.group-label { font-weight: 700; color: #0f172a; }
.group-stats { color: #64748b; font-size: 0.76rem; margin-left: 0.55rem; }
.group-selected { color: #4f46e5; font-weight: 600; font-size: 0.76rem; margin-left: 0.55rem; }
.library-table tr.bucket-row td:first-child { padding-left: 1.5rem; }
.library-table tr.derived { background: #eceff4; }
.library-table tr.derived:hover { background: #e2e8f0; }
.aug-hint { color: #7c3aed; font-size: 0.72rem; font-weight: 600; white-space: nowrap; }
.library-table tr.selected { background: #eef2ff; outline: 2px solid #6366f1; outline-offset: -2px; }
.library-table tr.assigned { box-shadow: inset 3px 0 0 #22c55e; }
.library-table tr:hover { background: #f8fafc; }
.library-table .time-cell { font-variant-numeric: tabular-nums; }
.library-label-cell .label-wrap { display: inline-flex; align-items: center; gap: 0.35rem; }
.assigned-dot { width: 8px; height: 8px; border-radius: 999px; background: #16a34a; box-shadow: 0 0 0 2px #dcfce7; flex: 0 0 auto; }
.library-table .action-cell { text-align: right; }
.library-row-delete { border: 1px solid #fecaca; background: #fff1f2; color: #991b1b; border-radius: 6px; padding: 0.28rem 0.55rem; cursor: pointer; font-weight: 700; font-size: 0.76rem; }
.library-row-delete:hover { background: #ffe4e6; }
.library-row-delete:disabled { opacity: 0.55; cursor: wait; }
.rule-search { font-size: 0.78rem; padding: 0.2rem 0.45rem; border: 1px solid #cbd5e1; border-radius: 6px; min-width: 13rem; margin-left: 0.5rem; }
.rule-overlay { color: #7c3aed; cursor: help; }
.library-rule-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.library-rule-table th { color: #64748b; text-align: left; padding: 0.35rem 0.45rem; border-bottom: 1px solid #e2e8f0; white-space: nowrap; }
.library-rule-table td { padding: 0.38rem 0.45rem; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
.rule-summary { color: #475569; line-height: 1.35; max-width: 360px; white-space: normal; }
.recipe-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem; margin-top: 0.55rem; }
.library-recipe-fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 0.5rem; }
.bucket-pill { display: block; min-height: 28px; padding: 0.35rem 0.5rem; border-radius: 6px; border: 1px solid #cbd5e1; background: #f8fafc; color: #475569; font-size: 0.78rem; line-height: 1.35; white-space: normal; overflow-wrap: anywhere; }
.bucket-pickers { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.65rem; margin-top: 0.55rem; }
.bucket-picker { display: grid; gap: 0.35rem; align-content: start; }
.augment-settings-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.65rem; }
.augment-settings-panel { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.65rem; }
.augment-group { border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.55rem; background: #fff; display: grid; gap: 0.45rem; min-width: 0; }
.augment-group.disabled { background: #f8fafc; color: #94a3b8; opacity: 0.72; }
.augment-group.disabled input:not([type="checkbox"]) { background: #e2e8f0; color: #94a3b8; cursor: not-allowed; }
.augment-group-title { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; font-size: 0.75rem; font-weight: 800; color: #475569; text-transform: uppercase; position: relative; z-index: 1; }
.augment-group-title label.check { pointer-events: auto; cursor: pointer; }
.augment-group-title label.check input[type="checkbox"] { pointer-events: auto; }
.augment-settings-grid label, .recipe-actions label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.72rem; color: #64748b; font-weight: 700; text-transform: uppercase; }
.augment-settings-grid label.check, .recipe-actions label.check { flex-direction: row; align-items: center; text-transform: none; font-size: 0.82rem; font-weight: 600; color: #475569; }
.augment-settings-grid input { min-width: 0; padding: 0.4rem 0.5rem; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.85rem; background: #fff; }
.setting-field { min-width: 0; }
.setting-field .setting-head { display: flex; align-items: center; justify-content: space-between; gap: 0.35rem; line-height: 1.2; }
.setting-field .tip { flex: 0 0 auto; }
.setting-note { color: #64748b; font-size: 0.75rem; line-height: 1.35; text-transform: none; font-weight: 500; }
.setting-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 0.35rem; }
.source-rule-cards, .recipe-card-list, .existing-aug-list { display: grid; gap: 0.55rem; max-height: 32vh; overflow: auto; padding-right: 0.15rem; }
.source-rule-card, .recipe-card, .existing-aug-card { border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc; padding: 0.55rem; display: grid; gap: 0.35rem; min-width: 0; }
.rule-card-head, .recipe-card-head { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; font-size: 0.84rem; font-weight: 700; color: #0f172a; }
.rule-card-meta, .recipe-card-meta { display: flex; flex-wrap: wrap; gap: 0.25rem; font-size: 0.72rem; color: #64748b; }
.rule-chip { display: inline-flex; border: 1px solid #cbd5e1; border-radius: 999px; padding: 0.12rem 0.42rem; background: #fff; max-width: 100%; overflow-wrap: anywhere; }
.recipe-card-actions { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.recipe-card details { font-size: 0.75rem; color: #475569; }
.recipe-card pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.35rem 0 0; max-height: 180px; overflow: auto; }
/* --- Compact row layouts for Source Rules, Existing Augmentations -------- */
.lib-rows { display: grid; gap: 0; max-height: 32vh; overflow: auto; padding-right: 0.15rem; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; }
.lib-rows .lib-empty { padding: 0.6rem 0.75rem; color: #94a3b8; font-size: 0.82rem; }
.lib-rows .lib-empty.busy { color: #475569; background: #f8fafc; }
.lib-rows .lib-empty.err { color: #991b1b; background: #fee2e2; }
.lib-row { display: grid; grid-template-columns: minmax(140px, max-content) 1fr 26px; align-items: center; gap: 0.65rem; padding: 0.4rem 0.6rem; border-bottom: 1px solid #f1f5f9; font-size: 0.82rem; min-width: 0; }
.lib-row:last-child { border-bottom: none; }
.lib-row.source-rule-row { background: var(--class-bg, #fff); border-left: 4px solid var(--class-border, #e2e8f0); }
.lib-row.source-rule-row.derived { filter: brightness(0.94); box-shadow: inset 0 0 0 1px #c4b5fd; }
.lib-row-name { font-weight: 600; color: #1e293b; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lib-row-kv { color: #64748b; min-width: 0; overflow-wrap: anywhere; }
.lib-row-split-pill { display: inline-flex; align-items: center; border: 1px solid #cbd5e1; border-radius: 999px; padding: 0.02rem 0.34rem; background: #f8fafc; color: #334155; font-size: 0.68rem; font-weight: 800; font-style: italic; text-transform: uppercase; letter-spacing: 0.02em; }
.lib-row-x { border: 1px solid #e2e8f0; background: #fff; color: #94a3b8; width: 24px; height: 24px; line-height: 1; padding: 0; border-radius: 6px; cursor: pointer; font-size: 0.95rem; }
.lib-row-x:hover { background: #fee2e2; color: #991b1b; border-color: #fecaca; }
/* --- Augmentation subsection rhythm -------------------------------------- */
.aug-subsection { border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; padding: 0.7rem 0.85rem; margin-bottom: 0.85rem; }
.aug-subsection > h4 { margin: 0 0 0.55rem; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.02em; color: #475569; font-weight: 700; }
/* --- Saved Recipes: per-row redesign with expandable details ------------- */
.recipe-rows { display: grid; gap: 0; max-height: 36vh; overflow: auto; padding-right: 0.15rem; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; }
.recipe-rows .lib-empty { padding: 0.6rem 0.75rem; color: #94a3b8; font-size: 0.82rem; }
.recipe-row { display: grid; gap: 0; border-bottom: 1px solid #f1f5f9; font-size: 0.82rem; }
.recipe-row:last-child { border-bottom: none; }
.recipe-row-head { display: grid; grid-template-columns: 22px minmax(0, 1fr) auto auto auto; align-items: center; gap: 0.65rem; padding: 0.4rem 0.6rem 0.15rem; min-width: 0; }
.recipe-row-head .recipe-select { display: flex; align-items: center; justify-content: center; }
.recipe-row-title { font-weight: 600; color: #0f172a; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.86rem; }
.recipe-row-title .x { color: #94a3b8; margin: 0 0.25rem; font-weight: 400; }
.recipe-row-meta { color: #64748b; font-size: 0.74rem; padding: 0 0.6rem 0.4rem 2.1rem; overflow-wrap: anywhere; }
.recipe-row-stats { color: #475569; font-size: 0.76rem; white-space: nowrap; }
.recipe-row-toggle { border: 1px solid #e2e8f0; background: #fff; color: #475569; width: 22px; height: 22px; line-height: 1; padding: 0; border-radius: 6px; cursor: pointer; font-size: 0.8rem; }
.recipe-row-toggle:hover { background: #f1f5f9; }
.recipe-row-remove { border: 1px solid #e2e8f0; background: #fff; color: #64748b; padding: 0.15rem 0.55rem; border-radius: 6px; cursor: pointer; font-size: 0.78rem; }
.recipe-row-remove:hover { background: #fee2e2; color: #991b1b; border-color: #fecaca; }
.recipe-row-body { display: flex; flex-wrap: wrap; gap: 0.6rem 0.9rem; align-items: center; padding: 0.4rem 0.75rem 0.55rem 2.1rem; background: #f8fafc; border-top: 1px solid #f1f5f9; }
/* HTML `hidden` would normally collapse this row, but the display:flex above
   wins on equal specificity. This selector — placed after — restores
   `display:none` when the toggle button sets `body.hidden = true`. */
.recipe-row-body[hidden] { display: none; }
.recipe-row-body label { display: inline-flex; flex-direction: column; gap: 0.2rem; font-size: 0.68rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
.recipe-row-body label.check { flex-direction: row; align-items: center; gap: 0.35rem; text-transform: none; font-size: 0.8rem; font-weight: 500; color: #475569; }
.recipe-row-body input[type="text"] { padding: 0.25rem 0.4rem; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.8rem; background: #fff; width: 110px; }
.recipe-row-body details { flex-basis: 100%; font-size: 0.74rem; color: #475569; margin-top: 0.2rem; }
.recipe-row-body details summary { cursor: pointer; color: #475569; font-weight: 500; }
.recipe-row-body details .kv-table { display: grid; grid-template-columns: max-content 1fr; gap: 0.15rem 0.8rem; margin: 0.35rem 0 0; font-size: 0.74rem; }
.recipe-row-body details .kv-table .k { color: #94a3b8; text-transform: uppercase; font-size: 0.66rem; letter-spacing: 0.02em; }
.recipe-row-body details .kv-table .v { color: #334155; font-variant-numeric: tabular-nums; }
.recipe-row-body pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.3rem 0 0; max-height: 160px; overflow: auto; }
.recipe-progress { display: inline-block; min-width: 60px; height: 4px; background: #e2e8f0; border-radius: 999px; overflow: hidden; vertical-align: middle; margin-left: 0.4rem; }
.recipe-progress span { display: block; height: 100%; background: #10b981; width: 0%; }
/* Mask filter needs room: the labels are explanatory and each field has a tip. */
.augment-group[data-aug-group="mask"] {
    grid-column: span 2;
}
.augment-group[data-aug-group="mask"] .augment-settings-grid {
    grid-template-columns: 1fr 1fr;
    gap: 0.65rem;
}
.augment-group[data-aug-group="mask"] .augment-settings-grid input { padding: 0.42rem 0.5rem; font-size: 0.85rem; }
/* Recipe Builder bucket buttons: visible enough to read as actions, still quiet. */
.bucket-picker button { padding: 0.35rem 0.6rem; border: 1px solid #a5b4fc; background: #eef2ff; color: #1e293b; border-radius: 6px; font-size: 0.82rem; cursor: pointer; font-weight: 650; }
.bucket-picker button:hover { background: #e0e7ff; border-color: #818cf8; }
.augment-progress { display: grid; gap: 0.3rem; min-width: min(420px, 100%); flex: 1 1 280px; }
.augment-progress .bar { height: 8px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
.augment-progress .bar span { display: block; height: 100%; width: 0%; background: #2563eb; }
.generated-meter { min-width: 92px; }
.generated-meter .bar { height: 6px; background: #e2e8f0; border-radius: 999px; overflow: hidden; margin-top: 4px; }
.generated-meter .bar span { display: block; height: 100%; background: #10b981; width: 0%; }
@media (max-width: 1180px) { .library-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body class="{% if interactive %}interactive library-active{% endif %}">

{% if interactive %}
<div class="toolbar">
  <div class="selection-summary selection-only">
    <div class="count">
      Selected <span class="num" id="select-count">0</span> clips
    </div>
    <div class="hint">
      Click cards to mark. Hold <kbd class="kbd">Shift</kbd> + click to range-select.
    </div>
  </div>
  <div class="status" id="status"></div>
  <div class="actions">
    <span class="job-status" id="background-job-status" hidden><!-- backgroundJobHint --></span>
    <span class="refresh-status" id="review-refresh-status"></span>
    <button class="del selection-only" id="del-btn" type="button" title="Permanently delete marked files from disk">
      Delete selected
    </button>
    <button type="button" class="refresh" id="library-refresh-review" title="Rescan all tabs: re-read the library, dataset, augmented, frozen-test and mined views from disk. Use after recording/generating/deleting files, or to refresh every count at once. Does NOT rebuild the training dataset — use Compose dataset for that.">
      Rescan all ⤓
    </button>
    <button type="button" class="primary-action" id="compose-dataset-btn" title="Compose dataset: rebuild train/val/test.csv (incl. the frozen test) from the current library selection + recipes. Required before training and to make frozen-test / dataset changes take effect. Runs here, then refreshes the affected tabs.">
      Compose dataset ⚙
    </button>
    <button class="done" id="done-btn" type="button" title="Write exclusions to composer manifest and close">
      Done
    </button>
  </div>
</div>
{% endif %}

<div class="done-overlay" id="done-overlay">
  <div class="big">All Done!</div>
  <div>The inspect server has recorded your exclusions.</div>
  <div style="font-size:0.75em; margin-top:1em; opacity:0.8;">You can now close this tab.</div>
</div>

<h1>Sound Event Detection Dataset Inspection</h1>

<div class="summary">
  <div class="card">
    <div style="color: #64748b; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Project ID</div>
    <div style="font-size: 1.5rem; font-weight: 700; margin-top: 4px;">{{ project_id }}</div>
  </div>
  {% for cls, stat in class_summary.items() %}
  <div class="card">
    <div style="color: #64748b; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">{{ cls }}</div>
    <div style="font-size: 1.5rem; font-weight: 700; margin-top: 4px;">
      {{ stat.count }} clips
    </div>
  </div>
  {% endfor %}
</div>

<div class="tabs">
  <button class="{% if interactive %}active{% endif %}" data-tab="library">Library <span class="count"></span></button>
  <button class="{% if not interactive %}active{% endif %}" data-tab="dataset">Dataset <span class="count"></span></button>
  <button data-tab="augmented">Augmented <span class="count"></span></button>
  <button data-tab="frozen">Frozen test <span class="count"></span></button>
  <button data-tab="mined">Mined False Positives <span class="count"></span></button>
</div>

{# ---- Library builder tab -------------------------------------------- #}
<div class="tab-panel {% if interactive %}active{% endif %}" id="tab-library">
  <div class="library-builder">
    <div class="library-workbar">
      <label>target class
        <select id="library-target-class">
          {% for cls in model_classes %}
          <option value="{{ cls|e }}">{{ cls }}</option>
          {% endfor %}
        </select>
      </label>
      <label>split
        <select id="library-target-split">
          <option value="native">bucket role</option>
          <option value="train">train</option>
          <option value="test">frozen test</option>
        </select>
      </label>
      <button type="button" class="primary" id="library-assign-class">Assign bucket to class</button>
      <button type="button" class="danger" id="library-delete-selected" title="Delete files in the selected Library bucket rows" disabled>Delete selected</button>
      <button type="button" class="danger" id="library-delete-small" title="Delete every bucket in the current filtered view that has fewer clips than a threshold you enter">Delete small buckets</button>
      <span class="library-count" id="library-count"></span>
    </div>

    <div class="library-grid">
      <div>
        <div class="library-section">
          <h3>
            <span>Library Filters</span>
            <span class="section-head">
              <button type="button" class="section-refresh" id="library-scope-refresh" data-refresh-scopes="library" data-refresh-status="library-scope-status" data-refresh-label="library" title="Rescan this view: re-read the Library buckets from disk. Fast, does not rebuild the dataset." aria-label="Rescan library buckets">↻</button>
              <span class="section-status" id="library-scope-status"></span>
              <button type="button" class="section-action" id="library-clear-filters" title="Reset all filters and list every bucket">clear filters</button>
            </span>
          </h3>
          <div class="library-filters">
            <label>search <input id="library-search" type="search" placeholder="label, tag, filename"></label>
            <label>label <select id="library-label-filter"><option value="">all</option></select></label>
            <label>density <select id="library-density-filter"><option value="">all</option></select></label>
            <label>project <select id="library-project-filter"><option value="">all</option></select></label>
            <label>role <select id="library-role-filter"><option value="">all</option></select></label>
            <label>tag <select id="library-tag-filter"><option value="">all</option></select></label>
            <label>bucket <select id="library-bucket-filter"><option value="">all</option></select></label>
            <label>mic <select id="library-mic-filter"><option value="">all</option></select></label>
            <label>session/time <input id="library-session-filter" type="search" placeholder="session, date, or time"></label>
          </div>
        </div>

        <div class="library-table-wrap">
          <table class="library-table">
            <thead>
              <tr>
                <th>Label</th><th>Count</th><th>Project</th><th>Role</th><th>Density</th>
                <th>Tags</th><th>Bucket</th><th>Kind/Mic</th><th>Session/Time</th><th></th>
              </tr>
            </thead>
            <tbody id="library-table-body"></tbody>
          </table>
        </div>
      </div>

      <div class="library-designer">
        <div class="library-section">
          <h3>
            <span class="section-head">Dataset Source Rules <span class="section-status" id="source-rules-status">loading...</span></span>
            <input type="search" id="source-rules-search" class="rule-search" placeholder="filter rules… (class, label, tags, overlay)" title="Filter source rules by any displayed text — class, audio label, project, density, tags, bucket, mic, or overlay class.">
            <button type="button" class="section-action" id="source-rules-clear" title="Remove every source rule from all classes">clear all</button>
          </h3>
          <div id="library-rule-table-body" class="lib-rows"><div class="lib-empty busy">Loading saved source rules from dataset_selection.json...</div></div>
        </div>

        <div class="library-section">
          <h3>Augmentation</h3>

          <div class="aug-subsection">
            <h4>Augmentation Settings</h4>
            <div class="augment-settings-panel">
              <div class="augment-group" data-aug-group="core">
                <div class="augment-group-title">Core <label class="check"><input id="aug-setting-enabled" type="checkbox"> enabled</label></div>
                <div class="augment-settings-grid">
                  <label>seed <input id="aug-setting-seed" type="number" step="1"></label>
                  <label>samples/source <input id="aug-setting-samples-per-source" type="number" step="1" min="0"></label>
                  <label class="setting-field">
                    <span class="setting-head">transient window ms <span class="tip" data-tip="Shared by Overlay and Mask Filter. Overlay uses this window to align/measure SNR around the base transient; Mask Filter uses the same window to choose active transient frames and compare overlay masking.">?</span></span>
                    <input id="aug-setting-transient-window-ms" type="number" step="1" min="0" data-disabled-by="aug-setting-enabled">
                  </label>
                </div>
              </div>
              <div class="augment-group" data-aug-group="pitch">
                <div class="augment-group-title">Pitch <span class="tip" data-tip="Randomly shifts pitch while preserving duration. Small ranges keep the same sound identity.">?</span> <label class="check"><input id="aug-setting-pitch-shift-enabled" type="checkbox"> enabled</label></div>
                <div class="augment-settings-grid">
                  <label>semitones <span class="tip" data-tip="Minimum and maximum pitch shift in musical semitones. Example: -1 to 1 is subtle.">?</span>
                    <span class="setting-pair">
                      <input id="aug-setting-semitones-min" type="number" step="0.1" data-disabled-by="aug-setting-pitch-shift-enabled">
                      <input id="aug-setting-semitones-max" type="number" step="0.1" data-disabled-by="aug-setting-pitch-shift-enabled">
                    </span>
                  </label>
                </div>
              </div>
              <div class="augment-group" data-aug-group="overlay">
                <div class="augment-group-title">Overlay <label class="check"><input id="aug-setting-overlay-enabled" type="checkbox"> enabled</label></div>
                <div class="augment-settings-grid">
                  <label class="setting-field">
                    <span class="setting-head">SNR dB <span class="tip" data-tip="Target-to-overlay loudness. Higher dB keeps the base sound clearer; lower dB makes harder examples.">?</span></span>
                    <span class="setting-pair">
                      <input id="aug-setting-snr-min" type="number" step="0.5" data-disabled-by="aug-setting-overlay-enabled">
                      <input id="aug-setting-snr-max" type="number" step="0.5" data-disabled-by="aug-setting-overlay-enabled">
                    </span>
                  </label>
                  <label class="check"><input id="aug-setting-scale-noise-to-snr" type="checkbox" data-disabled-by="aug-setting-overlay-enabled"> scale SNR <span class="tip" data-tip="Scale the overlay so the sampled SNR is measured against the target transient, not the whole clip average.">?</span></label>
                  <label class="check"><input id="aug-setting-prevent-clipping" type="checkbox" data-disabled-by="aug-setting-overlay-enabled"> prevent clipping <span class="tip" data-tip="If the mix would clip, reduce gain to keep the waveform inside the valid range.">?</span></label>
                </div>
              </div>
              <div class="augment-group" data-aug-group="mask">
                <div class="augment-group-title">Mask filter <label class="check"><input id="aug-setting-mask-filter-enabled" type="checkbox"> enabled</label></div>
                <div class="augment-settings-grid">
                  <label class="setting-field">
                    <span class="setting-head">attempts <span class="tip" data-tip="How many overlay candidates to try for each base clip before keeping the best acceptable one or skipping. Higher is slower but safer.">?</span></span>
                    <input id="aug-setting-mask-candidate-attempts" type="number" step="1" min="1" data-disabled-by="aug-setting-mask-filter-enabled">
                  </label>
                  <label class="setting-field">
                    <span class="setting-head">mean mask dB <span class="tip" data-tip="Primary gate: average dB the base transient sits above the overlay in its active mel bands. Positive = audible margin, 0 = equal energy, negative = buried. 6–10 keeps mixes clean and easy, 3–6 is realistic, lower admits hard mixes. Blank disables this gate.">?</span></span>
                    <input id="aug-setting-mask-min-mean-db" type="number" step="0.5" placeholder="optional" data-disabled-by="aug-setting-mask-filter-enabled">
                  </label>
                  <label class="setting-field">
                    <span class="setting-head">p25 mask dB <span class="tip" data-tip="Consistency gate: the lower-quartile band margin must clear this, so a few strong bands cannot hide several masked ones. Set it 2–4 dB below mean mask (e.g. mean 6 → p25 3). Blank disables.">?</span></span>
                    <input id="aug-setting-mask-min-p25-db" type="number" step="0.5" placeholder="optional" data-disabled-by="aug-setting-mask-filter-enabled">
                  </label>
                  <label class="setting-field">
                    <span class="setting-head">worst-band dB <span class="tip" data-tip="Strictest gate: the single weakest transient band must clear this. Negative values are normal — −3 lets one band be buried by 3 dB while the event stays audible overall; −6 admits harder mixes; 0 or higher is very conservative. Blank disables.">?</span></span>
                    <input id="aug-setting-mask-min-band-db" type="number" step="0.5" placeholder="optional" data-disabled-by="aug-setting-mask-filter-enabled">
                  </label>
                  <label class="setting-field">
                    <span class="setting-head">min eff. SNR dB <span class="tip" data-tip="Broadband floor: base-vs-overlay loudness in the transient window. The band gates can stay positive while the overlay swamps the whole mix — this catches those. −3 trims truly buried mixes, +3 keeps mixes clean. Blank disables.">?</span></span>
                    <input id="aug-setting-mask-min-effective-snr-db" type="number" step="0.5" placeholder="optional" data-disabled-by="aug-setting-mask-filter-enabled">
                  </label>
                  <label class="setting-field">
                    <span class="setting-head">bands <span class="tip" data-tip="How many mel bands from the base transient are compared against the overlay. More bands are broader; fewer focus on the strongest transient bands.">?</span></span>
                    <input id="aug-setting-mask-band-count" type="number" step="1" min="1" data-disabled-by="aug-setting-mask-filter-enabled">
                  </label>
                  <label class="setting-field">
                    <span class="setting-head">top frames <span class="tip" data-tip="How many loud transient frames inside the shared window are used for the mask score. Smaller values focus on the attack; larger values include more decay.">?</span></span>
                    <input id="aug-setting-mask-top-frames" type="number" step="1" min="1" data-disabled-by="aug-setting-mask-filter-enabled">
                  </label>
                </div>
              </div>
            </div>
            <div class="recipe-actions">
              <span class="library-count" id="augment-settings-status" title="Panel state is auto-saved; each Add recipe snapshots these knobs onto the recipe.">draft &mdash; baked into the recipe on Add</span>
            </div>
          </div>

          <div class="aug-subsection">
            <h4>Recipe Builder</h4>
            <div class="library-recipe-fields">
              <label>description <input id="recipe-description" placeholder="shown in metadata"></label>
              <label>tags <input id="recipe-tags" placeholder="cutlery,hard"></label>
            </div>
            <div class="bucket-pickers">
              <div class="bucket-picker">
                <button type="button" id="recipe-use-base">Use selected as base</button>
                <span class="bucket-pill" id="recipe-base-summary">base: not set</span>
              </div>
              <div class="bucket-picker">
                <button type="button" id="recipe-use-overlay">Use selected as overlay</button>
                <span class="bucket-pill" id="recipe-overlay-summary">overlay: not set</span>
              </div>
            </div>
            <div class="recipe-actions">
              <button type="button" class="primary" id="recipe-add">Add recipe</button>
            </div>
          </div>

          <div class="aug-subsection">
            <h4 class="section-head">
              Saved Recipes
              <button type="button" class="section-refresh" id="recipes-scope-refresh" data-refresh-scopes="recipes" data-refresh-status="recipes-scope-status" data-refresh-label="recipes" title="Rescan generated recipe counts from disk. Fast, does not rebuild the dataset." aria-label="Rescan recipe counts">↻</button>
              <span class="section-status" id="recipes-scope-status"></span>
            </h4>
            <div id="augmentation-rule-table-body" class="recipe-rows"></div>
          </div>

          <div class="aug-subsection">
            <h4>Generation</h4>
            <div class="recipe-actions">
              <button type="button" class="primary" id="augment-generate">Generate selected</button>
              <button type="button" id="augment-cancel" disabled>Cancel</button>
              <div class="augment-progress">
                <span class="library-count" id="augment-status"></span>
                <div class="bar"><span id="augment-progress-bar"></span></div>
              </div>
            </div>
          </div>

          <div class="aug-subsection">
            <h4>Recent Augmentations</h4>
            <div id="existing-augmentation-list" class="lib-rows"><div class="lib-empty busy">Loading generated augmentation counts...</div></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

{# ---- Dataset tab ----------------------------------------------------- #}
<div class="tab-panel {% if not interactive %}active{% endif %}" id="tab-dataset">
  <div class="aug-filters">
    {% if interactive %}
    <button type="button" class="section-refresh" id="dataset-scope-refresh" data-refresh-scopes="dataset" data-refresh-status="dataset-scope-status" data-refresh-label="dataset" title="Rescan this view: re-read the assigned Dataset cards + true counts from disk. Fast — to rebuild train/val/test, use Compose dataset." aria-label="Rescan dataset cards">↻</button>
    <span class="section-status" id="dataset-scope-status"></span>
    {% endif %}
    <span class="mask-filter">
      <label for="dataset-search">search</label>
      <input id="dataset-search" type="search" placeholder="cheers_medium">
      <button type="button" id="dataset-search-clear">clear</button>
    </span>
    <span class="bar-end">
      <button class="expand-all" data-target="tab-dataset" data-action="open">Expand all</button>
      <button class="expand-all" data-target="tab-dataset" data-action="close">Collapse all</button>
    </span>
  </div>
  {% for cls, clips in panels.items() %}
  <details class="class-block" data-class="{{ cls | e }}">
    <summary>
      {% if interactive %}
      <input class="class-select-all" type="checkbox" aria-label="Select all in class">
      {% endif %}
      {{ cls }}
      <span class="count" data-label="clip">{{ clips|length }} clip{{ 's' if clips|length != 1 else '' }}</span>
    </summary>
    {% if class_images.get(cls) %}
    <a href="{{ class_images[cls] }}" target="_blank" rel="noopener">
      <img class="class-ref" src="{{ class_images[cls] }}" alt="reference: {{ cls }}">
    </a>
    {% endif %}
    <div class="clips" data-class-clips="{{ cls | e }}">
    {% if not interactive or clips|length <= 100 %}
      {% for c in clips %}
      <div class="clip{% if c.soft_excluded %} selected{% endif %}"
           data-path="{{ c.path|e }}"
           data-name="{{ c.search_text|default(c.name|lower)|e }}">
        <div class="clip-body">
          <h3><span class="clip-name" title="{{ c.name|e }}">{{ c.name }}</span></h3>
          <div class="meta">{{ c.meta_html|safe }}</div>
          <img class="spec play-spec"
               src="{% if interactive %}/thumbnail?path={{ c.path|urlencode }}&type=spec{% else %}data:image/png;base64,{{ c.spec_b64 }}{% endif %}"
               alt="mel">
          <audio class="hidden-audio" preload="none" src="file://{{ c.path }}"></audio>
        </div>
      </div>
      {% endfor %}
    {% endif %}
    </div>
  </details>
  {% endfor %}
</div>

{# ---- Augmented tab --------------------------------------------------- #}
<div class="tab-panel" id="tab-augmented">
  <div class="aug-filters">
    {% if interactive %}
    <button type="button" class="section-refresh" id="augmented-scope-refresh" data-refresh-scopes="augmented" data-refresh-status="augmented-scope-status" data-refresh-label="augmented" title="Rescan this view: re-read the assigned Augmented cards from disk. Fast, does not rebuild the dataset." aria-label="Rescan augmented cards">↻</button>
    <span class="section-status" id="augmented-scope-status"></span>
    {% endif %}
    <button class="aug-filter active" data-technique="all">All</button>
    <button class="aug-filter" data-technique="noise">Overlay noise</button>
    <button class="aug-filter" data-technique="shift">Pitch shift</button>
    <button class="aug-filter" data-technique="other">Other/Mix</button>
    <button class="aug-filter" data-technique="misses" title="Clips the audited model did not recognize as their base label (run `sed evaluate augmented` first)">Model misses</button>
    <span class="mask-filter">
      <label for="aug-search">search</label>
      <input id="aug-search" type="search" placeholder="cutlery">
      <label for="aug-tag-filter">tag</label>
      <select id="aug-tag-filter"><option value="">all</option></select>
      <label for="mask-metric">mask</label>
      <select id="mask-metric">
        <option value="masking_score_db">mean dB</option>
        <option value="masking_score_p25_db">p25 dB</option>
        <option value="masking_score_min_db">worst-band dB</option>
        <option value="effective_snr_db">effective SNR dB</option>
      </select>
      <label for="mask-min">min</label>
      <input id="mask-min" type="number" step="0.5" placeholder="dB">
      <label for="mask-max">max</label>
      <input id="mask-max" type="number" step="0.5" placeholder="dB">
      <button type="button" id="mask-clear">clear</button>
      <button type="button" id="mix-view-toggle" title="Two-color mel view: orange = base sound, blue = overlay, whitish = both. Only clips generated with mix maps switch.">mix view</button>
    </span>
    <span class="bar-end">
      <button class="expand-all" data-target="tab-augmented" data-action="open">Expand all</button>
      <button class="expand-all" data-target="tab-augmented" data-action="close">Collapse all</button>
    </span>
  </div>
  {% if not aug_panels %}
  <div class="notice">
    No assigned augmented samples yet. Generated derived clips first appear in
    Library filters as <code>bucket=derived</code>; assign those buckets to a
    target class to review them here.
  </div>
  {% else %}
  <div class="notice">
    {{ aug_total }} assigned augmented sample{{ 's' if aug_total != 1 else '' }}.
  </div>
  {% for cls, clips in aug_panels.items() %}
  <details class="class-block" data-class="{{ cls | e }}">
    <summary>
      {% if interactive %}
      <input class="class-select-all" type="checkbox" aria-label="Select all in class">
      {% endif %}
      {{ cls }}
      <span class="count" data-label="augmented clip">{{ clips|length }} augmented clip{{ 's' if clips|length != 1 else '' }}</span>
    </summary>
    {% if class_images.get(cls) %}
    <a href="{{ class_images[cls] }}" target="_blank" rel="noopener">
      <img class="class-ref" src="{{ class_images[cls] }}" alt="reference: {{ cls }}">
    </a>
    {% endif %}
    <div class="clips" data-class-clips="{{ cls | e }}">
    {% if not interactive or clips|length <= 100 %}
      {% for c in clips %}
      <div class="clip{% if c.soft_excluded %} selected{% endif %}"
           data-path="{{ c.path|e }}"
           data-name="{{ c.search_text|e }}"
           data-technique="{{ c.augmentation|e }}"
           data-mask="{{ c.masking_score_db|default('') }}">
        <div class="clip-body">
          <h3><span class="clip-name" title="{{ c.name|e }}">{{ c.name }}</span></h3>
          <div class="meta">{{ c.meta_html|safe }}</div>
          <img class="spec play-spec"
               src="{% if interactive %}/thumbnail?path={{ c.path|urlencode }}&type=spec{% else %}data:image/png;base64,{{ c.spec_b64 }}{% endif %}"
               alt="mel">
          <audio class="hidden-audio" preload="none" src="file://{{ c.path }}"></audio>
        </div>
      </div>
      {% endfor %}
    {% endif %}
    </div>
  </details>
  {% endfor %}
  {% endif %}
</div>

{# ---- Frozen test tab ------------------------------------------------- #}
<div class="tab-panel" id="tab-frozen">
  {% if interactive %}
  <div class="run-select">
    <button type="button" class="section-refresh" id="frozen-scope-refresh" data-refresh-scopes="frozen" data-refresh-status="frozen-scope-status" data-refresh-label="frozen" title="Rescan this view: re-read the composed Frozen-test set from disk. Fast — to CHANGE which clips are in the frozen test, use Compose dataset." aria-label="Rescan frozen-test cards">↻</button>
    <span class="section-status" id="frozen-scope-status"></span>
  </div>
  {% endif %}
  {% if not frozen_panels %}
  <div class="notice">
    {% if frozen_context.stale %}
    {{ frozen_context.stale_reason }}<br>
    <code>uv run sed dataset compose --project {{ project_id }}</code>
    {% else %}
    No composed frozen test set was found yet. This tab appears after dataset compose writes
    <code>projects/{{ project_id }}/dataset/test.csv</code>. Assign source rules with split <code>test</code>,
    then run:<br>
    <code>uv run sed dataset compose --project {{ project_id }}</code>
    {% endif %}
  </div>
  {% else %}
  <div class="notice">
    {% if frozen_context.pending_compose %}
    Pending frozen test preview from current source rules. Run
    <code>uv run sed dataset compose --project {{ project_id }}</code>
    to write <code>test.csv</code> before evaluation.<br>
    {% endif %}
    {{ frozen_total }} frozen test sample{{ 's' if frozen_total != 1 else '' }} across
    {{ frozen_panels|length }} dataset class bucket{{ 's' if frozen_panels|length != 1 else '' }}.
    {% if frozen_has_predictions %}
    Prediction metadata is loaded from run <code>{{ frozen_predictions_run_id }}</code>
    using <code>{{ frozen_context.source }}</code>.
    {% else %}
    No <code>test_predictions.tsv</code> was found yet; run
    <code>uv run sed evaluate run --project {{ project_id }}</code>
    to add probabilities and correctness.
    {% endif %}
  </div>
  {% if interactive and frozen_context.run_options %}
  <div class="run-select">
    <label for="frozen-run-select">run</label>
    <select id="frozen-run-select">
      <option value="" {% if not frozen_context.requested_run_id %}selected{% endif %}>
        selected/latest{% if frozen_context.viewed_run_id %}: {{ frozen_context.viewed_run_id }}{% endif %}
      </option>
      {% for run in frozen_context.run_options %}
      <option value="{{ run.run_id|e }}"
              {% if frozen_context.requested_run_id == run.run_id %}selected{% endif %}>
        {{ run.label }}
      </option>
      {% endfor %}
    </select>
    <button type="button" class="run-preview-btn" data-open-run-preview="confusion" title="Preview confusion matrix for this run">CM</button>
    <button type="button" class="run-preview-btn" data-open-run-preview="training" title="Preview training curves for this run">Plot</button>
    <button type="button" class="run-preview-btn" id="frozen-evaluate-btn" title="Re-score this model on the CURRENT frozen test set (test.csv) and refresh the confusion matrix. Run Compose dataset first if you changed the frozen set.">Evaluate</button>
    <span id="frozen-run-status"></span>
  </div>
  <div class="run-preview-modal" id="frozen-run-preview-modal" hidden>
    <div class="run-preview-backdrop" data-close-run-preview></div>
    <div class="run-preview-dialog" role="dialog" aria-modal="true" aria-labelledby="frozen-run-preview-title">
      <div class="run-preview-head">
        <div class="run-preview-title" id="frozen-run-preview-title">Run preview</div>
        <div class="run-preview-actions">
          <button type="button" data-switch-run-preview="confusion">CM</button>
          <button type="button" data-switch-run-preview="training">Plot</button>
          <button type="button" data-close-run-preview title="Close preview">x</button>
        </div>
      </div>
      <div class="run-preview-body" id="frozen-run-preview-body">
        <img id="frozen-run-preview-image" alt="run report preview">
      </div>
    </div>
  </div>
  {% endif %}
  <div class="aug-filters">
    <button class="frozen-filter active" data-frozen="all">All</button>
    <button class="frozen-filter" data-frozen="incorrect">Incorrect label</button>
    <button class="frozen-filter" data-frozen="pred-negative">Predicted _negative</button>
    <button class="frozen-filter" data-frozen="pred-label">Predicted Label</button>
    <span class="mask-filter">
      <label for="frozen-search">search</label>
      <input id="frozen-search" type="search" placeholder="cheers_loud">
      <label for="frozen-metric">metric</label>
      <select id="frozen-metric">
        <option value="predicted-confidence">predicted confidence</option>
        <option value="argmax-confidence">argmax confidence</option>
        <option value="margin">margin</option>
      </select>
      <label for="frozen-min">min</label>
      <input id="frozen-min" type="number" min="0" max="1" step="0.01" placeholder="0.00">
      <label for="frozen-max">max</label>
      <input id="frozen-max" type="number" min="0" max="1" step="0.01" placeholder="1.00">
      <button type="button" id="frozen-clear">clear</button>
    </span>
    <span class="bar-end">
      <button class="expand-all" data-target="tab-frozen" data-action="open">Expand all</button>
      <button class="expand-all" data-target="tab-frozen" data-action="close">Collapse all</button>
    </span>
  </div>
  {% for cls, clips in frozen_panels.items() %}
  <details class="class-block" data-class="{{ cls | e }}">
    <summary>
      {% if interactive %}
      <input class="class-select-all" type="checkbox" aria-label="Select all in class">
      {% endif %}
      {{ cls }}
      <span class="count" data-label="test clip">{{ clips|length }} test clip{{ 's' if clips|length != 1 else '' }}</span>
    </summary>
    {% if class_images.get(cls) %}
    <a href="{{ class_images[cls] }}" target="_blank" rel="noopener">
      <img class="class-ref" src="{{ class_images[cls] }}" alt="reference: {{ cls }}">
    </a>
    {% endif %}
    <div class="clips" data-class-clips="{{ cls | e }}">
    {% if not interactive or clips|length <= 100 %}
      {% for c in clips %}
      <div class="clip frozen-clip
                  {{ 'frozen-correct' if c.correct else 'frozen-wrong' }}
                  {{ 'frozen-negative' if c.predicted_class == '_negative' else 'frozen-trigger' if c.triggered else '' }}
                  {{ 'frozen-missing' if not c.has_prediction else '' }}"
           data-path="{{ c.path|e }}"
           data-name="{{ c.search_text|e }}"
           data-frozen-true="{{ c.true_class|e }}"
           data-frozen-pred="{{ c.predicted_class|e }}"
           data-frozen-correct="{{ 'true' if c.correct else 'false' }}"
           data-frozen-triggered="{{ 'true' if c.triggered else 'false' }}"
           data-frozen-reason="{{ c.decision_reason|e }}"
           data-frozen-has-prediction="{{ 'true' if c.has_prediction else 'false' }}"
           data-frozen-predicted-confidence="{% if c.predicted_confidence is not none %}{{ '%.6f'|format(c.predicted_confidence) }}{% endif %}"
           data-frozen-argmax-confidence="{% if c.argmax_confidence is not none %}{{ '%.6f'|format(c.argmax_confidence) }}{% endif %}"
           data-frozen-margin="{% if c.margin is not none %}{{ '%.6f'|format(c.margin) }}{% endif %}">
        <div class="clip-body">
          <h3>
            <span class="clip-name" title="{{ c.name|e }}">{{ c.name }}</span>
            <span class="clip-pills">
              <span class="pill result-pill {% if c.correct %}good{% else %}bad{% endif %}">
                {% if c.correct %}correct{% else %}review{% endif %}
              </span>
              <span class="pill neg pred-neg-pill"
                    {% if c.predicted_class != '_negative' %}style="display:none"{% endif %}>
                _negative
              </span>
            </span>
          </h3>
          <div class="meta">{{ c.meta_html|safe }}</div>
          <img class="spec play-spec"
               src="{% if interactive %}/thumbnail?path={{ c.path|urlencode }}&type=spec{% else %}data:image/png;base64,{{ c.spec_b64 }}{% endif %}"
               alt="mel">
          <div class="prob-box">{{ c.prob_html|safe }}</div>
          <audio class="hidden-audio" preload="none" src="file://{{ c.path }}"></audio>
        </div>
      </div>
      {% endfor %}
    {% endif %}
    </div>
  </details>
  {% endfor %}
  {% endif %}
</div>

{# ---- Mined hardnegs tab ---------------------------------------------- #}
<div class="tab-panel" id="tab-mined">
  {% if not mined_panels %}
  <div class="notice">
    No mined detections yet. Run live inference with:<br>
    <code>uv run sed inference --project {{ project_id }} --target &lt;class&gt; --mine-detections</code>
  </div>
  {% else %}
  <div class="notice">
    {{ mined_total }} mined detection{{ 's' if mined_total != 1 else '' }} across
    {{ mined_panels|length }} predicted class bucket{{ 's' if mined_panels|length != 1 else '' }}.
  </div>
  {% if interactive %}
  <div class="mined-actions">
    <label for="mined-label">negative label</label>
    <input id="mined-label" value="mined_false_positive">
    <button type="button" id="promote-mined-btn">Promote selected to negatives</button>
  </div>
  {% endif %}
  <div class="aug-filters">
    {% if interactive %}
    <button type="button" class="section-refresh" id="mined-scope-refresh" data-refresh-scopes="mined" data-refresh-status="mined-scope-status" data-refresh-label="mined" title="Rescan this view: re-read mined false-positive candidates from disk. Fast, does not rebuild the dataset." aria-label="Rescan mined candidates">↻</button>
    <span class="section-status" id="mined-scope-status"></span>
    {% endif %}
    <button class="mined-filter active" data-mined="all">All</button>
    {% for cls in mined_panels.keys() %}
    <button class="mined-filter" data-mined="{{ cls }}">{{ cls }}</button>
    {% endfor %}
    <span class="mask-filter">
      <label for="mined-search">search</label>
      <input id="mined-search" type="search" placeholder="cheers_loud">
      <label for="mined-conf-min">min conf %</label>
      <input id="mined-conf-min" type="number" step="1" placeholder="%">
      <button type="button" id="mined-clear">clear</button>
    </span>
    <span class="bar-end">
      <button class="expand-all" data-target="tab-mined" data-action="open">Expand all</button>
      <button class="expand-all" data-target="tab-mined" data-action="close">Collapse all</button>
    </span>
  </div>
  {% for cls, clips in mined_panels.items() %}
  <details class="class-block" data-class="{{ cls | e }}">
    <summary>
      {% if interactive %}
      <input class="class-select-all" type="checkbox" aria-label="Select all in class">
      {% endif %}
      {{ cls }}
      <span class="count" data-label="mined clip">{{ clips|length }} mined clip{{ 's' if clips|length != 1 else '' }}</span>
    </summary>
    <div class="clips" data-class-clips="{{ cls | e }}">
    {% if not interactive or clips|length <= 100 %}
      {% for c in clips %}
      <div class="clip mined-clip"
           data-path="{{ c.path|e }}"
           data-name="{{ c.search_text|e }}"
           data-mined="{{ c.predicted_class|e }}"
           data-confidence="{{ '%.4f'|format(c.confidence * 100.0) }}">
        <div class="clip-body">
          <h3><span class="clip-name" title="{{ c.name|e }}">{{ c.name }}</span></h3>
          <div class="meta">{{ c.meta_html|safe }}</div>
          <img class="spec play-spec"
               src="{% if interactive %}/thumbnail?path={{ c.path|urlencode }}&type=spec{% else %}data:image/png;base64,{{ c.spec_b64 }}{% endif %}"
               alt="mel">
          <div class="prob-box">{{ c.prob_html|safe }}</div>
          <audio class="hidden-audio" preload="none" src="file://{{ c.path }}"></audio>
        </div>
      </div>
      {% endfor %}
    {% endif %}
    </div>
  </details>
  {% endfor %}
  {% endif %}
</div>

<div class="hover-tip" id="hover-tip"></div>

<script id="clips-data-json" type="application/json">
{
  "panels": {{ panels_json | safe }},
  "aug_panels": {{ aug_panels_json | safe }},
  "frozen_panels": {{ frozen_panels_json | safe }},
  "frozen_context": {{ frozen_context_json | safe }},
  "mined_panels": {{ mined_panels_json | safe }},
  "library_items": {{ library_items_json | safe }}
}
</script>

<script>
// --- Interactive Mode Frontend Script --------------------------------------
const isInteractive = {{ 'true' if interactive else 'false' }};
const projectId = {{ project_id | tojson }};
let CLIPS_DATA = { panels: {}, aug_panels: {}, frozen_panels: {}, frozen_context: {}, mined_panels: {}, library_items: [] };
// True per-class counts per tab (dataset/augmented/frozen/mined), independent
// of the display card cap, so headers/tab counts show the real dataset size.
let panelTotals = {};
try {
    const dataEl = document.getElementById("clips-data-json");
    if (dataEl) {
        CLIPS_DATA = JSON.parse(dataEl.textContent);
    }
} catch (e) {
    console.error("Failed to parse clips data:", e);
}

// Pagination state: tracks limit for each key "tab:className"
const PAGINATION_STATE = {};
let lastClickedCard = null;

// Track active filters
let activeAugFilter = "all";
let activeFrozenFilter = "all";
let activeMinedFilter = "all";
let selectionState = { assignments: {}, augmentation_recipes: [], augmentation_settings: {} };
let selectionLoaded = false;
let selectionLoadError = "";
let sourceRuleQuery = "";   // live filter for the Dataset Source Rules list
let augmentationGeneratedCounts = {};
let augmentationGeneratedCountsLoaded = false;
let augmentationGeneratedCountsLoading = false;
let augmentationGeneratedCountsError = "";
let libraryRows = [];
let libraryAllRows = [];
let selectedLibraryRowKeys = new Set();
let lastLibraryRowIndex = null;
const expandedLibraryLabels = new Set();
let mixViewActive = false;
let recipeBaseRules = [];
let recipeOverlayRules = [];
let reviewDirty = false;
let augmentJobId = "";
let augmentPollTimer = null;
let augmentationRunning = false;
let augmentationCanceling = false;
let augmentationSettingsDirty = false;
let refreshRequestSeq = 0;
let deleteJobId = "";
let deleteRunning = false;
let refreshJobId = "";
let refreshRunning = false;
let composeRunning = false;
let activeRefreshPromise = null;
let currentFrozenRunId = (CLIPS_DATA.frozen_context || {}).viewed_run_id || "";
// Which Saved-Recipe rows are expanded; survives re-renders so editing
// `prefix` while typing doesn't snap the row closed.
const expandedRecipes = new Set();
const recipeUpdateTimers = {};

// Search and threshold DOM references
const datasetSearchInput = document.getElementById("dataset-search");
const augSearchInput = document.getElementById("aug-search");
const augTagSelect = document.getElementById("aug-tag-filter");
const maskMetricSelect = document.getElementById("mask-metric");
const maskMinInput = document.getElementById("mask-min");
const maskMaxInput = document.getElementById("mask-max");
const frozenRunSelect = document.getElementById("frozen-run-select");
const frozenSearchInput = document.getElementById("frozen-search");
const frozenMetricSelect = document.getElementById("frozen-metric");
const frozenMinInput = document.getElementById("frozen-min");
const frozenMaxInput = document.getElementById("frozen-max");
const minedSearchInput = document.getElementById("mined-search");
const minedConfMinInput = document.getElementById("mined-conf-min");
const librarySearchInput = document.getElementById("library-search");
const librarySessionInput = document.getElementById("library-session-filter");
const libraryFilterIds = [
    "library-label-filter", "library-density-filter", "library-project-filter",
    "library-role-filter", "library-bucket-filter",
    "library-tag-filter", "library-mic-filter"
];

// Help text & status
const statusEl = document.getElementById("status");
const countEl  = document.getElementById("select-count");

function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#039;");
}

function cssEscape(value) {
    if (window.CSS && typeof window.CSS.escape === "function") return window.CSS.escape(value);
    return String(value).replace(/["\\\\]/g, "\\\\$&");
}

function compactUtcToMs(value) {
    const raw = String(value || "");
    const match = raw.match(/(\d{8})T(\d{6})(?:\.(\d{1,6}))?Z/);
    if (!match) return null;
    const [, ymd, hms, frac] = match;
    const msText = (frac || "").slice(0, 3).padEnd(3, "0");
    const ms = Date.UTC(
        Number(ymd.slice(0, 4)),
        Number(ymd.slice(4, 6)) - 1,
        Number(ymd.slice(6, 8)),
        Number(hms.slice(0, 2)),
        Number(hms.slice(2, 4)),
        Number(hms.slice(4, 6)),
        Number(msText || 0),
    );
    return Number.isFinite(ms) ? ms : null;
}

function timestampToMs(value) {
    if (!value) return null;
    const compact = compactUtcToMs(value);
    if (compact !== null) return compact;
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function clipTimestampMs(c) {
    return timestampToMs(c.recorded_at) || timestampToMs(c.name) || timestampToMs(c.session_id) || null;
}

function sessionTimestampMs(c) {
    return timestampToMs(c.session_id) || timestampToMs(c.recorded_at) || timestampToMs(c.name) || null;
}

function formatLocalDateTime(ms) {
    if (ms === null || ms === undefined) return "";
    const d = new Date(ms);
    if (!Number.isFinite(d.getTime())) return "";
    const pad = n => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function libraryTimeLabel(row) {
    const c = row.first || {};
    const session = formatLocalDateTime(row.session_ms);
    const start = formatLocalDateTime(row.first_ms);
    const end = formatLocalDateTime(row.last_ms);
    if (start && end && start !== end) return `${start} - ${end}`;
    if (session && start && session !== start) return `${session} / ${start}`;
    return start || session || c.session_id || c.date || "";
}

function libraryTimeTitle(row) {
    const c = row.first || {};
    const parts = [];
    if (c.session_id) parts.push(`raw session: ${c.session_id}`);
    if (c.recorded_at) parts.push(`raw recorded: ${c.recorded_at}`);
    if (row.first_ms !== null && row.first_ms !== undefined) parts.push(`first local: ${formatLocalDateTime(row.first_ms)}`);
    if (row.last_ms !== null && row.last_ms !== undefined) parts.push(`last local: ${formatLocalDateTime(row.last_ms)}`);
    return parts.join("\\n");
}

function stableHue(text) {
    let hash = 0;
    const value = String(text || "");
    for (let i = 0; i < value.length; i++) hash = ((hash * 31) + value.charCodeAt(i)) >>> 0;
    return hash % 360;
}

const CLASS_TINT_PALETTE = [
    { h: 210, s: 58, l: 93, b: 47 },
    { h: 150, s: 46, l: 91, b: 42 },
    { h: 42,  s: 56, l: 91, b: 46 },
    { h: 270, s: 42, l: 93, b: 52 },
    { h: 184, s: 48, l: 90, b: 40 },
    { h: 16,  s: 48, l: 92, b: 49 },
    { h: 226, s: 48, l: 92, b: 49 },
    { h: 96,  s: 40, l: 90, b: 39 },
    { h: 318, s: 36, l: 93, b: 50 },
    { h: 52,  s: 46, l: 89, b: 43 },
    { h: 198, s: 44, l: 91, b: 42 },
    { h: 6,   s: 38, l: 93, b: 48 },
];

function classTintStyle(className) {
    const hue = stableHue(className);
    const p = CLASS_TINT_PALETTE[hue % CLASS_TINT_PALETTE.length];
    return `--class-hue:${p.h};--class-bg:hsla(${p.h}, ${p.s}%, ${p.l}%, 0.86);--class-border:hsla(${p.h}, ${p.s}%, ${p.b}%, 0.58);`;
}

function micChannels(c) {
    const raw = (c.triggered_channels && c.triggered_channels.length)
        ? c.triggered_channels
        : ((c.recorded_channels && c.recorded_channels.length) ? c.recorded_channels : (c.mic_channels || []));
    return [...new Set((raw || [])
        .map(v => Number(v))
        .filter(v => Number.isInteger(v) && v >= 0))]
        .sort((a, b) => a - b);
}

function micKey(c) {
    return micChannels(c).join(",");
}

function micLabelFromKey(key) {
    if (!key) return "unknown";
    const n = key.split(",").filter(Boolean).length;
    return `${n === 1 ? "mic" : "mics"} ${key}`;
}

function micLabelFromValue(value) {
    if (!value) return "";
    if (value === "unknown") return "unknown";
    if (value.startsWith("split:")) return `mic ${value.slice(6)}`;
    return micLabelFromKey(value.replace(/^mic:/, ""));
}

function bucketDatasetRole(c) {
    const role = String((c && c.dataset_role) || "train").trim().toLowerCase();
    return role === "test" ? "test" : "train";
}

function libraryMicBucketValue(c) {
    const key = c.mic_key || micKey(c);
    if (key) return `mic:${key}`;
    if ((c.clip_kind || "") === "channel_split" && c.channel_index !== undefined && c.channel_index !== null) {
        return `split:${c.channel_index}`;
    }
    return "unknown";
}

function libraryMicBucketLabel(c) {
    return micLabelFromValue(libraryMicBucketValue(c));
}

function micMatchesValue(c, value) {
    if (!value) return true;
    const chans = micChannels(c);
    const key = chans.join(",");
    if (value === "unknown") return chans.length === 0;
    if (value === "single") return chans.length === 1;
    if (value === "multi") return chans.length > 1;
    if (value.startsWith("mic:")) return key === value.slice(4);
    if (value.startsWith("split:")) {
        return (c.clip_kind || "") === "channel_split" && String(c.channel_index) === value.slice(6);
    }
    return true;
}

function ruleMicValue(rule) {
    if (!rule) return "";
    if (rule.mic_filter) return String(rule.mic_filter);
    const raw = rule.triggered_channels || rule.recorded_channels || rule.mic_channels || rule.mics || [];
    const values = Array.isArray(raw)
        ? raw
        : String(raw).replace(/^mic:/, "").split(",");
    if (!values.length) return "";
    const key = [...new Set(values.map(v => Number(v)).filter(v => Number.isInteger(v) && v >= 0))]
        .sort((a, b) => a - b)
        .join(",");
    return key ? `mic:${key}` : "";
}

function ruleSplitChannelValue(rule) {
    if (!rule) return "";
    if (rule.channel_index === undefined || rule.channel_index === null || rule.channel_index === "") return "";
    const n = Number(rule.channel_index);
    return Number.isInteger(n) && n >= 0 ? `split:${n}` : "";
}

function findClipObject(path, tabType) {
    if (tabType === "dataset") {
        return Object.values(CLIPS_DATA.panels).flat().find(c => c.path === path);
    } else if (tabType === "augmented") {
        return Object.values(CLIPS_DATA.aug_panels).flat().find(c => c.path === path);
    } else if (tabType === "frozen") {
        return Object.values(CLIPS_DATA.frozen_panels).flat().find(c => c.path === path);
    } else if (tabType === "mined") {
        return Object.values(CLIPS_DATA.mined_panels).flat().find(c => c.path === path);
    } else if (tabType === "library") {
        return (CLIPS_DATA.library_items || []).find(c => c.path === path);
    }
    return null;
}

// Reads a select's effective value: a pending restoreValue (set while the
// options are being rebuilt after a refresh) wins over the DOM value.
function selectValueOrPending(id) {
    const el = document.getElementById(id);
    if (!el) return "";
    return el.dataset.restoreValue !== undefined ? el.dataset.restoreValue : (el.value || "");
}

function libraryActiveFilters() {
    return {
        q: (librarySearchInput && librarySearchInput.value || "").trim().toLowerCase(),
        session: (librarySessionInput && librarySessionInput.value || "").trim().toLowerCase(),
        label: selectValueOrPending("library-label-filter"),
        density: selectValueOrPending("library-density-filter"),
        project: selectValueOrPending("library-project-filter"),
        role: selectValueOrPending("library-role-filter"),
        bucket: selectValueOrPending("library-bucket-filter"),
        tag: selectValueOrPending("library-tag-filter"),
        mic: selectValueOrPending("library-mic-filter"),
    };
}

// Faceted matcher: `skipKey` leaves one dimension unconstrained so each
// dropdown can offer the values still reachable under all *other* filters.
function libraryItemsMatchingFilters(f, skipKey = "") {
    return (CLIPS_DATA.library_items || []).filter(c => {
        if (c.deleted) return false;
        if ((c.review_status || "accepted") !== "accepted") return false;
        if (skipKey !== "label" && f.label && c.label !== f.label) return false;
        if (skipKey !== "density" && f.density && c.density !== f.density) return false;
        if (skipKey !== "project" && f.project && c.project_id !== f.project) return false;
        if (skipKey !== "role" && f.role && c.dataset_role !== f.role) return false;
        if (skipKey !== "bucket" && f.bucket && c.source_bucket !== f.bucket) return false;
        if (skipKey !== "tag" && f.tag && !(c.tags || []).includes(f.tag)) return false;
        if (skipKey !== "mic" && f.mic && !micMatchesValue(c, f.mic)) return false;
        if (f.q && !(c.search_text || "").includes(f.q)) return false;
        const sessionText = `${c.session_id || ""} ${c.recorded_at || ""} ${c.date || ""} ${formatLocalDateTime(sessionTimestampMs(c))} ${formatLocalDateTime(clipTimestampMs(c))}`.toLowerCase();
        if (f.session && !sessionText.includes(f.session)) return false;
        return true;
    });
}

function libraryFilteredItems() {
    return libraryItemsMatchingFilters(libraryActiveFilters());
}

function filterClipsList(clips, tabType) {
    if (tabType === "library") return libraryFilteredItems();
    if (tabType === "dataset") {
        const q = (datasetSearchInput && datasetSearchInput.value || "").trim().toLowerCase();
        return clips.filter(c => !c.deleted && (!q || (c.search_text || "").includes(q)));
    }
    if (tabType === "augmented") {
        const q = (augSearchInput && augSearchInput.value || "").trim().toLowerCase();
        const tag = (augTagSelect && augTagSelect.value || "").trim();
        const maskMetric = (maskMetricSelect && maskMetricSelect.value) || "masking_score_db";
        const minMask = maskMinInput && maskMinInput.value !== "" ? Number(maskMinInput.value) : null;
        const maxMask = maskMaxInput && maskMaxInput.value !== "" ? Number(maskMaxInput.value) : null;
        const technique = activeAugFilter;

        return clips.filter(c => {
            if (c.deleted) return false;
            if (q && !(c.search_text || "").includes(q)) return false;
            if (tag && !(c.tags || []).includes(tag)) return false;

            if (technique === "noise" && c.augmentation !== "overlay") return false;
            if (technique === "shift" && !["pitch", "pitch_shift"].includes(c.augmentation)) return false;
            if (technique === "other" && (c.augmentation === "overlay" || c.augmentation === "pitch" || c.augmentation === "pitch_shift")) return false;
            // "misses" = audited by `sed evaluate augmented` AND not recognized.
            if (technique === "misses" && c.audit_correct !== false) return false;

            const rawMask = c[maskMetric];
            const maskValue = rawMask === null || rawMask === undefined || rawMask === "" ? null : Number(rawMask);
            if (minMask !== null && (maskValue === null || maskValue < minMask)) return false;
            if (maxMask !== null && (maskValue === null || maskValue > maxMask)) return false;
            return true;
        });
    }
    if (tabType === "frozen") {
        const q = (frozenSearchInput && frozenSearchInput.value || "").trim().toLowerCase();
        const filterType = activeFrozenFilter;
        const metric = frozenMetricSelect ? frozenMetricSelect.value : "predicted-confidence";
        const minVal = frozenMinInput && frozenMinInput.value !== "" ? Number(frozenMinInput.value) : null;
        const maxVal = frozenMaxInput && frozenMaxInput.value !== "" ? Number(frozenMaxInput.value) : null;

        return clips.filter(c => {
            if (c.deleted) return false;
            if (q && !(c.search_text || "").includes(q)) return false;

            if (filterType === "incorrect" && c.correct) return false;
            if (filterType === "pred-negative" && c.predicted_class !== "_negative") return false;
            if (filterType === "pred-label" && (c.predicted_class === "_negative" || c.predicted_class === "not_evaluated")) return false;

            let val = null;
            if (metric === "predicted-confidence") val = c.predicted_confidence;
            else if (metric === "argmax-confidence") val = c.argmax_confidence;
            else if (metric === "margin") val = c.margin;

            if (val !== null && val !== undefined) {
                if (minVal !== null && val < minVal) return false;
                if (maxVal !== null && val > maxVal) return false;
            }
            return true;
        });
    }
    if (tabType === "mined") {
        const q = (minedSearchInput && minedSearchInput.value || "").trim().toLowerCase();
        const minConf = minedConfMinInput && minedConfMinInput.value !== "" ? Number(minedConfMinInput.value) : null;
        const activeFilter = activeMinedFilter;

        return clips.filter(c => {
            if (c.deleted) return false;
            if (activeFilter !== "all" && c.predicted_class !== activeFilter) return false;
            if (q && !(c.search_text || "").includes(q)) return false;

            const conf = c.confidence * 100.0;
            if (minConf !== null && conf < minConf) return false;
            return true;
        });
    }
    return clips;
}

function renderCard(c, tabType) {
    const isSelected = c.soft_excluded || false;
    let cardClasses = "clip";
    if (isSelected) cardClasses += " selected";

    let selectBoxHtml = "";
    if (isInteractive) {
        let label = "Mark for next-train exclusion";
        if (tabType === "frozen") label = "Mark frozen-test clip for deletion";
        else if (tabType === "mined") label = "Mark for promotion to negatives";

        selectBoxHtml = `<input class="select-box" type="checkbox" aria-label="${label}" ${isSelected ? "checked" : ""}>`;
    }

    let pillHtml = "";
    let probHtml = "";

    if (tabType === "frozen") {
        cardClasses += " frozen-clip";
        if (c.correct) cardClasses += " frozen-correct";
        else cardClasses += " frozen-wrong";

        if (c.predicted_class === "_negative") cardClasses += " frozen-negative";
        else if (c.triggered) cardClasses += " frozen-trigger";

        if (!c.has_prediction) cardClasses += " frozen-missing";

        pillHtml = `
            <span class="pill result-pill ${c.correct ? 'good' : 'bad'}">
                ${c.correct ? 'correct' : 'review'}
            </span>
            <span class="pill neg pred-neg-pill" ${c.predicted_class !== '_negative' ? 'style="display:none"' : ''}>
                _negative
            </span>
        `;
        probHtml = `<div class="prob-box">${c.prob_html || ''}</div>`;
    } else if (tabType === "mined") {
        cardClasses += " mined-clip";
        probHtml = `<div class="prob-box">${c.prob_html || ''}</div>`;
    }

    // Mix view: two-color base/overlay mel rendered at generation time —
    // only overlay-augmented clips have one; others keep the regular spec.
    const useMixMap = tabType === "augmented" && mixViewActive && c.mix_png;
    const specSrc = useMixMap
        ? `/image?path=${encodeURIComponent(c.mix_png)}`
        : `/thumbnail?path=${encodeURIComponent(c.path)}&type=spec`;
    const audioSrc = `/audio?path=${encodeURIComponent(c.path)}`;

    return `
        <div class="${cardClasses}"
             data-path="${escapeHtml(c.path)}"
             data-name="${escapeHtml(c.search_text || c.name || '')}"
             ${tabType === 'frozen' ? `
                 data-frozen-true="${escapeHtml(c.true_class || '')}"
                 data-frozen-pred="${escapeHtml(c.predicted_class || '')}"
                 data-frozen-correct="${c.correct ? 'true' : 'false'}"
                 data-frozen-triggered="${c.triggered ? 'true' : 'false'}"
                 data-frozen-reason="${escapeHtml(c.decision_reason || '')}"
                 data-frozen-has-prediction="${c.has_prediction ? 'true' : 'false'}"
                 data-frozen-predicted-confidence="${c.predicted_confidence !== null && c.predicted_confidence !== undefined ? c.predicted_confidence : ''}"
                 data-frozen-argmax-confidence="${c.argmax_confidence !== null && c.argmax_confidence !== undefined ? c.argmax_confidence : ''}"
                 data-frozen-margin="${c.margin !== null && c.margin !== undefined ? c.margin : ''}"
             ` : ''}
             ${tabType === 'mined' ? `
                 data-mined="${escapeHtml(c.predicted_class || '')}"
                 data-confidence="${c.confidence !== null && c.confidence !== undefined ? (c.confidence * 100.0).toFixed(4) : ''}"
             ` : ''}>
            ${selectBoxHtml}
            <div class="clip-body">
                <h3>
                    <span class="clip-name" title="${escapeHtml(c.name || '')}">${escapeHtml(c.name || '')}</span>
                    ${pillHtml ? `<span class="clip-pills">${pillHtml}</span>` : ''}
                </h3>
                <div class="meta">${c.meta_html || ''}</div>
                <img class="spec play-spec" src="${specSrc}" alt="mel" loading="lazy">
                ${probHtml}
                <audio class="hidden-audio" preload="none" src="${audioSrc}"></audio>
            </div>
        </div>
    `;
}

// Rebuild a select's options to ["all", ...pairs]. The active selection (or a
// pending restoreValue from a refresh) is kept even when the facet no longer
// offers it, so an applied filter never silently disappears — the user clears
// it explicitly.
function rebuildSelectOptions(el, pairs) {
    if (!el) return;
    const current = el.dataset.restoreValue !== undefined ? el.dataset.restoreValue : (el.value || "");
    delete el.dataset.restoreValue;
    const list = [...pairs];
    if (current && !list.some(([value]) => value === current)) {
        list.unshift([current, current]);
    }
    el.innerHTML = "";
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "all";
    el.appendChild(all);
    list.forEach(([value, label]) => {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = label;
        el.appendChild(opt);
    });
    el.value = current;
}

// Cascading facets: each dropdown only offers values that remain reachable
// under all the *other* active filters. Re-run after every filter change,
// refresh, or delete.
function populateLibraryFilters() {
    const f = libraryActiveFilters();
    const specs = [
        ["library-label-filter", "label", c => c.label],
        ["library-density-filter", "density", c => c.density],
        ["library-project-filter", "project", c => c.project_id],
        ["library-role-filter", "role", c => c.dataset_role],
        ["library-bucket-filter", "bucket", c => c.source_bucket],
    ];
    specs.forEach(([id, key, getter]) => {
        const el = document.getElementById(id);
        if (!el) return;
        const items = libraryItemsMatchingFilters(f, key);
        const values = [...new Set(items.map(getter).filter(Boolean))].sort();
        rebuildSelectOptions(el, values.map(v => [v, v]));
    });
    const tagEl = document.getElementById("library-tag-filter");
    if (tagEl) {
        const items = libraryItemsMatchingFilters(f, "tag");
        const values = [...new Set(items.flatMap(c => c.tags || []).filter(Boolean))].sort();
        rebuildSelectOptions(tagEl, values.map(v => [v, v]));
    }
    const micEl = document.getElementById("library-mic-filter");
    if (micEl) {
        const items = libraryItemsMatchingFilters(f, "mic");
        const exactValues = [...new Set(items
            .map(c => libraryMicBucketValue(c))
            .filter(v => v && v !== "unknown"))]
            .sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
        const hasUnknown = items.some(c => libraryMicBucketValue(c) === "unknown");
        rebuildSelectOptions(micEl, [
            ["single", "single mic"],
            ["multi", "multi mic"],
            ...(hasUnknown ? [["unknown", "unknown"]] : []),
            ...exactValues.map(value => [value, micLabelFromValue(value)]),
        ]);
    }
}

function clearLibraryFilters() {
    if (librarySearchInput) librarySearchInput.value = "";
    if (librarySessionInput) librarySessionInput.value = "";
    libraryFilterIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.value = "";
            delete el.dataset.restoreValue;
        }
    });
    applyFilters("library");
}

function populateAugmentedTagFilter() {
    if (!augTagSelect || augTagSelect.dataset.populated === "true") return;
    const values = [...new Set(
        Object.values(CLIPS_DATA.aug_panels || {})
            .flat()
            .flatMap(c => c.tags || [])
            .filter(Boolean)
    )].sort();
    values.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        augTagSelect.appendChild(opt);
    });
    restoreSelectValue(augTagSelect);
    augTagSelect.dataset.populated = "true";
}

function renderLibraryTable() {
    const tbody = document.getElementById("library-table-body");
    if (!tbody) return;
    const filtered = libraryFilteredItems();
    const groups = new Map();
    filtered.forEach(c => {
        const micBucket = libraryMicBucketValue(c);
        const parentBucket = (c.clip_kind || "") === "channel_split" ? "" : (c.parent_uuid || "");
        const key = JSON.stringify([
            c.label || "", c.project_id || "", c.dataset_role || "", c.density || "",
            (c.tags || []).join("|"), c.source_bucket || "",
            c.clip_kind || "", micBucket,
            parentBucket, c.session_id || c.date || "",
            c.recipe_id || "", c.overlay_label || ""
        ]);
        if (!groups.has(key)) groups.set(key, {
            key, first: c, count: 0, paths: [],
            first_ms: null, last_ms: null, session_ms: sessionTimestampMs(c),
        });
        const row = groups.get(key);
        row.count += 1;
        if (c.path) row.paths.push(c.path);
        const ms = clipTimestampMs(c) || sessionTimestampMs(c);
        if (ms !== null && ms !== undefined) {
            if (row.first_ms === null || ms < row.first_ms) {
                row.first_ms = ms;
                row.first = c;
            }
            if (row.last_ms === null || ms > row.last_ms) row.last_ms = ms;
        }
        const sessionMs = sessionTimestampMs(c);
        if (sessionMs !== null && sessionMs !== undefined) {
            if (row.session_ms === null || sessionMs < row.session_ms) row.session_ms = sessionMs;
        }
    });
    libraryAllRows = [...groups.values()].sort((a, b) => {
        const at = a.first_ms ?? a.session_ms ?? Number.POSITIVE_INFINITY;
        const bt = b.first_ms ?? b.session_ms ?? Number.POSITIVE_INFINITY;
        if (at !== bt) return at - bt;
        const ak = `${a.first.label || ""} ${a.first.project_id || ""} ${a.first.density || ""} ${a.first.session_id || ""}`;
        const bk = `${b.first.label || ""} ${b.first.project_id || ""} ${b.first.density || ""} ${b.first.session_id || ""}`;
        return ak.localeCompare(bk, undefined, {numeric: true});
    });
    libraryRows = libraryAllRows.slice(0, 600);
    const visibleKeys = new Set(libraryRows.map(row => row.key));
    selectedLibraryRowKeys = new Set([...selectedLibraryRowKeys].filter(key => visibleKeys.has(key)));
    libraryRows.forEach(row => {
        const c = row.first;
        const role = bucketDatasetRole(c);
        const rule = {
            labels: c.label ? [c.label] : [],
            split: role,
            dataset_role: role,
            count: row.count,
        };
        if (c.project_id) rule.project_ids = [c.project_id];
        if (c.density) rule.density = c.density;
        if (c.tags && c.tags.length) rule.tags_all = [...c.tags];
        if (c.source_bucket || c.category) rule.source_bucket = c.source_bucket || c.category;
        if (c.clip_kind) rule.clip_kind = c.clip_kind;
        const exactMicKey = c.mic_key || micKey(c);
        if (exactMicKey) {
            rule.triggered_channels = exactMicKey.split(",").map(v => Number(v)).filter(v => Number.isInteger(v));
        } else if ((c.clip_kind || "") === "channel_split" && c.channel_index !== undefined && c.channel_index !== null) {
            rule.channel_index = Number(c.channel_index);
        } else if ((c.mic_kind || "unknown") === "unknown") {
            rule.mic_filter = "unknown";
        }
        if (c.session_id || c.date) rule.session = c.session_id || c.date;
        row.rule = rule;
        row.assignedClasses = assignedClassesForRule(rule);
    });
    // Group bucket rows under their label so the table lists one collapsible
    // main row per label; expand the labels you work with to pick buckets.
    const labelGroups = [];
    const groupsByLabel = new Map();
    libraryRows.forEach((row, index) => {
        const label = row.first.label || "(no label)";
        let group = groupsByLabel.get(label);
        if (!group) {
            group = { label, rows: [], clipCount: 0, assignedClasses: new Set(), selectedCount: 0 };
            groupsByLabel.set(label, group);
            labelGroups.push(group);
        }
        group.rows.push({ row, index });
        group.clipCount += row.count;
        (row.assignedClasses || []).forEach(cls => group.assignedClasses.add(cls));
        if (selectedLibraryRowKeys.has(row.key)) group.selectedCount += 1;
    });
    // With a single matching label there is nothing to fold — show its buckets.
    if (labelGroups.length === 1) expandedLibraryLabels.add(labelGroups[0].label);

    const bucketRowHtml = (row, index, renderIndex) => {
        const c = row.first;
        const selected = selectedLibraryRowKeys.has(row.key);
        const assignedClasses = row.assignedClasses || [];
        const assignedDot = assignedClasses.length
            ? `<span class="assigned-dot" title="Assigned to ${escapeHtml(assignedClasses.join(', '))}"></span>`
            : "";
        let kindLabel = c.clip_kind || "mono";
        if (c.clip_kind === "channel_split") {
            kindLabel = "split";
        } else if (c.clip_kind === "multichannel") {
            kindLabel = `multi · ${c.channel_count || '?'}ch`;
        }
        const micLabel = (c.mic_label && c.mic_label !== "unknown")
            ? c.mic_label
            : libraryMicBucketLabel(c);
        const kindMicLabel = `${kindLabel} · ${micLabel}`;
        const timeLabel = libraryTimeLabel(row);
        const timeTitle = libraryTimeTitle(row);
        const isDerived = (c.source_bucket || c.category) === "derived";
        return `
        <tr data-key="${escapeHtml(row.key)}" data-index="${index}" data-render-index="${renderIndex}" class="bucket-row ${selected ? 'selected' : ''} ${assignedClasses.length ? 'assigned' : ''} ${isDerived ? 'derived' : ''}">
            <td class="library-label-cell"><span class="label-wrap">${assignedDot}<span>${escapeHtml(c.label || '')}</span></span></td>
            <td>${row.count}</td>
            <td>${escapeHtml(c.project_id || '')}</td>
            <td>${escapeHtml(c.dataset_role || '')}</td>
            <td>${escapeHtml(c.density || '')}</td>
            <td>${escapeHtml((c.tags || []).join(', '))}</td>
            <td>${escapeHtml(c.source_bucket || c.category || '')}${isDerived ? ` <span class="aug-hint" title="augmented${c.overlay_label ? ` · overlay: ${escapeHtml(c.overlay_label)}` : ''}">⨯${escapeHtml(c.overlay_label || 'pitch')}</span>` : ''}</td>
            <td>${escapeHtml(kindMicLabel)}</td>
            <td class="time-cell" title="${escapeHtml(timeTitle)}">${escapeHtml(timeLabel)}</td>
            <td class="action-cell"><button type="button" class="library-row-delete" data-delete-library-row="${index}" title="Delete all WAV and sidecar files in this bucket">Delete</button></td>
        </tr>
    `;};

    let renderIndex = 0;
    tbody.innerHTML = labelGroups.map(group => {
        const isOpen = expandedLibraryLabels.has(group.label);
        const assignedDot = group.assignedClasses.size
            ? `<span class="assigned-dot" title="Assigned to ${escapeHtml([...group.assignedClasses].sort().join(', '))}"></span>`
            : "";
        const header = `
        <tr class="library-group-row" data-group-label="${escapeHtml(group.label)}">
            <td colspan="10">
                <span class="group-toggle">${isOpen ? '▾' : '▸'}</span>
                ${assignedDot}<span class="group-label">${escapeHtml(group.label)}</span>
                <span class="group-stats">${group.rows.length} bucket${group.rows.length !== 1 ? 's' : ''} · ${group.clipCount} clips</span>
                <span class="group-selected" data-group-selected="${escapeHtml(group.label)}">${group.selectedCount ? `${group.selectedCount} selected` : ''}</span>
            </td>
        </tr>`;
        if (!isOpen) return header;
        return header + group.rows.map(({ row, index }) => bucketRowHtml(row, index, renderIndex++)).join("");
    }).join("");
    tbody.querySelectorAll("tr.library-group-row").forEach(rowEl => {
        rowEl.addEventListener("click", () => {
            const label = rowEl.dataset.groupLabel;
            if (expandedLibraryLabels.has(label)) expandedLibraryLabels.delete(label);
            else expandedLibraryLabels.add(label);
            renderLibraryTable();
        });
    });
    tbody.querySelectorAll("tr[data-key]").forEach(rowEl => {
        rowEl.addEventListener("click", ev => handleLibraryRowClick(rowEl, ev));
    });
    tbody.querySelectorAll("[data-delete-library-row]").forEach(btn => {
        btn.addEventListener("click", ev => {
            ev.stopPropagation();
            deleteLibraryRowsForIndex(Number(btn.dataset.deleteLibraryRow || -1), btn);
        });
    });
    const count = document.getElementById("library-count");
    if (count) {
        const selected = selectedLibraryRowKeys.size;
        count.textContent = `${filtered.length} clips · ${labelGroups.length} label(s) · ${libraryRows.length} bucket(s)${selected ? ` · ${selected} selected` : " · all buckets active"}`;
    }
    updateMutatingBusy();
}

function handleLibraryRowClick(rowEl, ev) {
    const key = rowEl.dataset.key;
    const renderIndex = Number(rowEl.dataset.renderIndex || 0);
    if (!key) return;
    if (ev.shiftKey && lastLibraryRowIndex !== null) {
        // Range-select over the rows the user can actually see — buckets
        // hidden inside collapsed label groups never join a shift range.
        const start = Math.min(lastLibraryRowIndex, renderIndex);
        const end = Math.max(lastLibraryRowIndex, renderIndex);
        selectedLibraryRowKeys.clear();
        document.querySelectorAll("#library-table-body tr[data-key]").forEach(el => {
            const ri = Number(el.dataset.renderIndex ?? -1);
            if (ri >= start && ri <= end) selectedLibraryRowKeys.add(el.dataset.key);
        });
    } else if (ev.ctrlKey || ev.metaKey) {
        if (selectedLibraryRowKeys.has(key)) selectedLibraryRowKeys.delete(key);
        else selectedLibraryRowKeys.add(key);
        lastLibraryRowIndex = renderIndex;
    } else {
        selectedLibraryRowKeys.clear();
        selectedLibraryRowKeys.add(key);
        lastLibraryRowIndex = renderIndex;
    }
    updateLibraryRowSelection();
}

function updateLibraryRowSelection() {
    document.querySelectorAll("#library-table-body tr[data-key]").forEach(row => {
        row.classList.toggle("selected", selectedLibraryRowKeys.has(row.dataset.key));
    });
    const selectedByLabel = {};
    libraryRows.forEach(row => {
        if (!selectedLibraryRowKeys.has(row.key)) return;
        const label = row.first.label || "(no label)";
        selectedByLabel[label] = (selectedByLabel[label] || 0) + 1;
    });
    document.querySelectorAll("#library-table-body [data-group-selected]").forEach(span => {
        const n = selectedByLabel[span.dataset.groupSelected] || 0;
        span.textContent = n ? `${n} selected` : "";
    });
    const filtered = libraryFilteredItems();
    const count = document.getElementById("library-count");
    if (count) {
        const selected = selectedLibraryRowKeys.size;
        count.textContent = `${filtered.length} clips · ${libraryRows.length} bucket(s)${selected ? ` · ${selected} selected` : " · all buckets active"}`;
    }
    updateLibraryBulkDeleteButton();
}

function selectedLibraryRows() {
    if (!selectedLibraryRowKeys.size) return [];
    return libraryRows.filter(row => selectedLibraryRowKeys.has(row.key));
}

function updateLibraryBulkDeleteButton() {
    const btn = document.getElementById("library-delete-selected");
    if (!btn) return;
    const busy = deleteRunning || refreshRunning || augmentationRunning;
    const selected = selectedLibraryRows();
    const pathCount = new Set(selected.flatMap(row => row.paths || []).filter(Boolean)).size;
    btn.disabled = busy || pathCount === 0;
    btn.textContent = selected.length > 1 ? `Delete ${selected.length} buckets` : "Delete selected";
    btn.title = busy
        ? "Background job is running."
        : pathCount
        ? `Delete ${pathCount} clip(s) from the selected Library bucket row(s)`
        : "Select one or more Library bucket rows to delete";
}

function libraryRowsForDeleteIndex(index) {
    const row = libraryRows[index];
    if (!row) return [];
    if (selectedLibraryRowKeys.size > 1 && selectedLibraryRowKeys.has(row.key)) {
        return selectedLibraryRows();
    }
    return [row];
}

async function deleteLibraryRowsForIndex(index, button) {
    return deleteLibraryRows(libraryRowsForDeleteIndex(index), button);
}

async function deleteSelectedLibraryRows() {
    return deleteLibraryRows(selectedLibraryRows(), document.getElementById("library-delete-selected"));
}

async function deleteSmallLibraryBuckets() {
    const raw = prompt(
        "Delete all buckets in the current filtered view with fewer clips than:",
        "20",
    );
    if (raw === null) return;
    const min = parseInt(raw, 10);
    if (!Number.isFinite(min) || min < 1) {
        flash("enter a whole number of clips, e.g. 20", "err");
        return;
    }
    const small = libraryAllRows.filter(row => Number(row.count || 0) < min);
    if (!small.length) {
        flash(`no buckets with fewer than ${min} clips`, "ok");
        return;
    }
    await deleteLibraryRows(small, document.getElementById("library-delete-small"));
}

async function deleteLibraryRows(rows, button) {
    rows = (rows || []).filter(Boolean);
    if (!rows.length) return;
    const paths = [...new Set(rows.flatMap(row => row.paths || []).filter(Boolean))];
    if (!paths.length) {
        alert("No files are available for the selected Library bucket rows.");
        return;
    }
    const detailRows = rows.slice(0, 4).map(row => {
        const c = row.first || {};
        return [
            c.label ? `label=${c.label}` : "",
            c.project_id ? `project=${c.project_id}` : "",
            c.dataset_role ? `role=${c.dataset_role}` : "",
            c.density ? `density=${c.density}` : "",
            c.source_bucket || c.category ? `bucket=${c.source_bucket || c.category}` : "",
            c.session_id || c.date ? `session=${c.session_id || c.date}` : "",
        ].filter(Boolean).join(" · ");
    }).filter(Boolean);
    if (rows.length > detailRows.length) detailRows.push(`...and ${rows.length - detailRows.length} more bucket(s)`);
    const detail = detailRows.length ? `\\n\\n${detailRows.join("\\n")}` : "";
    if (!confirm(`Permanently delete ${paths.length} clip(s) from ${rows.length} Library bucket row(s)?${detail}\n\nThis deletes the WAV files and JSON sidecars from disk. This cannot be undone.`)) return;

    if (button) button.disabled = true;
    try {
        await deletePathsInBackground(paths, rows.length === 1 ? "library bucket" : "library buckets");
        rows.forEach(row => selectedLibraryRowKeys.delete(row.key));
        updateLibraryRowSelection();
    } catch (e) {
        flash(`bucket delete failed: ${e.message || e}`, "err");
        if (button) button.disabled = false;
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function setBackgroundJobStatus(text, cls = "busy") {
    const el = document.getElementById("background-job-status");
    if (!el) return;
    if (!text) {
        el.hidden = true;
        el.textContent = "";
        el.className = "job-status";
        return;
    }
    el.hidden = false;
    el.className = `job-status ${cls || ""}`.trim();
    el.textContent = text;
}

function deleteProgressText(job) {
    const total = Number(job.total || 0);
    const completed = Number(job.completed || 0);
    if (job.phase === "cleanup") return "delete cleanup";
    return `deleting ${completed}/${total}${job.current_name ? ` - ${job.current_name}` : ""}`;
}

async function deletePathsInBackground(paths, label = "clips") {
    if (deleteRunning) throw new Error("delete already running");
    deleteRunning = true;
    updateMutatingBusy();
    setBackgroundJobStatus("starting delete");
    flash(`starting delete for ${paths.length} ${label}...`);
    try {
        const start = await fetch("/api/delete/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paths }),
        });
        const started = await start.json();
        if (!start.ok || !started.ok) throw new Error(started.error || `HTTP ${start.status}`);
        deleteJobId = (started.job || {}).job_id || "";
        let finalJob = started.job || {};
        while (deleteJobId) {
            await sleep(500);
            const r = await fetch(`/api/delete/status?job_id=${encodeURIComponent(deleteJobId)}`, { cache: "no-store" });
            const data = await r.json();
            if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
            const job = data.job || {};
            finalJob = job;
            const text = deleteProgressText(job);
            setBackgroundJobStatus(text);
            flash(text);
            if (["finished", "error"].includes(job.state)) break;
        }
        if (finalJob.state === "error") throw new Error(finalJob.error || "delete failed");
        const deletedSet = new Set(finalJob.deleted || []);
        removeDeletedPathsFromReviewData(deletedSet);
        renderReviewData();
        setBackgroundJobStatus(`deleted ${finalJob.deleted_count || deletedSet.size} clip(s); applying views`);
        flash(`deleted ${finalJob.deleted_count || deletedSet.size} clip(s); applying...`);
        deleteJobId = "";
        deleteRunning = false;
        updateMutatingBusy();
        await refreshReviewData({ silent: true });
        flash(`deleted ${finalJob.deleted_count || deletedSet.size} clip(s) from ${label}`, "ok");
        setBackgroundJobStatus("");
        refreshCount();
    } finally {
        deleteJobId = "";
        deleteRunning = false;
        updateMutatingBusy();
        if (!refreshRunning) setBackgroundJobStatus("");
    }
}

function asArray(value) {
    if (Array.isArray(value)) return value;
    if (value === undefined || value === null || value === "") return [];
    return [value];
}

function sortedStrings(value) {
    return asArray(value).map(v => String(v)).filter(Boolean).sort();
}

function ruleComparable(rule) {
    const r = rule || {};
    return JSON.stringify({
        labels: sortedStrings(r.labels || r.label),
        project_ids: r.project_ids === "*" ? ["*"] : sortedStrings(r.project_ids || r.project_id),
        density: String(r.density || ""),
        dataset_role: String(r.dataset_role || r.split || "train"),
        split: String(r.split || r.dataset_role || "train"),
        tags_all: sortedStrings(r.tags_all || r.tags),
        source_bucket: String(r.source_bucket || r.category || ""),
        clip_kind: String(r.clip_kind || ""),
        mic: String(ruleMicValue(r) || ruleSplitChannelValue(r) || ""),
        session: String(r.session || r.session_or_date || ""),
        search: String(r.search || ""),
    });
}

function assignedClassesForRule(rule) {
    const target = ruleComparable(rule);
    const assignments = selectionState.assignments || {};
    const classes = [];
    Object.keys(assignments).sort().forEach(className => {
        const rows = Array.isArray(assignments[className]) ? assignments[className] : [];
        if (rows.some(row => ruleComparable(row) === target)) classes.push(className);
    });
    return classes;
}

function libraryItemsMatchingRule(rule) {
    const labels = new Set(asArray(rule.labels || rule.label));
    const rawProjects = rule.project_ids || [];
    const projects = rawProjects === "*" ? null : new Set(asArray(rawProjects));
    const tags = new Set(asArray(rule.tags_all || rule.tags));
    const search = (rule.search || "").toLowerCase();
    const session = (rule.session || "").toLowerCase();
    return (CLIPS_DATA.library_items || []).filter(c => {
        if (c.deleted) return false;
        if ((c.review_status || "accepted") !== "accepted") return false;
        if (labels.size && !labels.has(c.label)) return false;
        if (projects && projects.size && !projects.has(c.project_id)) return false;
        if (rule.density && c.density !== rule.density) return false;
        if (rule.dataset_role && c.dataset_role !== rule.dataset_role) return false;
        if (rule.source_bucket && c.source_bucket !== rule.source_bucket) return false;
        // Mirror compose: per-channel splits are opt-in. A rule must explicitly
        // ask for clip_kind=channel_split, otherwise splits are excluded so they
        // don't double-count with channel 0 of the multichannel original.
        if (rule.clip_kind) {
            if ((c.clip_kind || "mono") !== rule.clip_kind) return false;
        } else if ((c.clip_kind || "mono") === "channel_split") {
            return false;
        }
        if (rule.channel_index !== undefined && rule.channel_index !== null && rule.channel_index !== "") {
            if (String(c.channel_index) !== String(rule.channel_index)) return false;
        }
        const micRuleValue = ruleMicValue(rule) || ruleSplitChannelValue(rule);
        if (micRuleValue && !micMatchesValue(c, micRuleValue)) return false;
        if (tags.size) {
            const clipTags = new Set(c.tags || []);
            for (const tag of tags) {
                if (!clipTags.has(tag)) return false;
            }
        }
        if (search && !(c.search_text || "").includes(search)) return false;
        const sessionText = `${c.session_id || ""} ${c.recorded_at || ""} ${c.date || ""} ${formatLocalDateTime(sessionTimestampMs(c))} ${formatLocalDateTime(clipTimestampMs(c))}`.toLowerCase();
        if (session && !sessionText.includes(session)) return false;
        return true;
    });
}

function cloneRule(rule) {
    return JSON.parse(JSON.stringify(rule || {}));
}

function selectedOrVisibleLibraryRules(split = "native") {
    const selectedKeys = selectedLibraryRowKeys.size ? selectedLibraryRowKeys : null;
    const sourceRows = libraryRows.filter(row => !selectedKeys || selectedKeys.has(row.key));
    return sourceRows.map(row => {
        const rule = cloneRule(row.rule);
        if (split !== "native") {
            rule.split = split;
            rule.dataset_role = split;
        } else {
            const role = rule.dataset_role === "test" ? "test" : "train";
            rule.split = role;
            rule.dataset_role = role;
        }
        rule.count = libraryItemsMatchingRule(rule).length;
        return rule;
    });
}

function ruleSummary(rule) {
    if (!rule) return "not set";
    const parts = [];
    if (rule.labels && rule.labels.length) parts.push(`label=${rule.labels.join("+")}`);
    if (rule.project_ids && rule.project_ids.length) parts.push(`project=${rule.project_ids.join("+")}`);
    if (rule.density) parts.push(`density=${rule.density}`);
    if (rule.tags_all && rule.tags_all.length) parts.push(`tags=${rule.tags_all.join("+")}`);
    if (rule.source_bucket) parts.push(`bucket=${rule.source_bucket}`);
    if (rule.clip_kind) parts.push(`kind=${rule.clip_kind}`);
    const micRule = ruleMicValue(rule) || ruleSplitChannelValue(rule);
    if (micRule) parts.push(`mic=${micLabelFromValue(micRule)}`);
    if (rule.session) parts.push(`session=${rule.session}`);
    if (rule.search) parts.push(`search=${rule.search}`);
    return parts.length ? parts.join(" · ") : "all accepted raw clips";
}

function ruleLabels(rule) {
    return escapeHtml((rule && rule.labels || []).join(", ") || "*");
}

function ruleProjects(rule) {
    const projects = rule && rule.project_ids;
    if (projects === "*") return "*";
    return escapeHtml((projects || []).join(", ") || "");
}

function ruleTags(rule) {
    return escapeHtml((rule && rule.tags_all || []).join(", ") || "");
}

function rulesSummary(rules) {
    if (!rules || !rules.length) return "not set";
    if (rules.length === 1) return ruleSummary(rules[0]);
    const labels = [...new Set(rules.flatMap(rule => rule.labels || []))].filter(Boolean);
    const count = rules.reduce((acc, rule) => acc + Number(rule.count || 0), 0);
    return `${rules.length} buckets${labels.length ? `: ${labels.join(", ")}` : ""} (${count})`;
}

function ruleBriefSummary(rule) {
    if (!rule) return "not set";
    const parts = [];
    if (rule.labels && rule.labels.length) parts.push(`label=${rule.labels.join(",")}`);
    if (rule.project_ids && rule.project_ids.length) parts.push(`project=${(rule.project_ids === "*" ? ["*"] : rule.project_ids).join(",")}`);
    if (rule.density) parts.push(`density=${rule.density}`);
    if (rule.tags_all && rule.tags_all.length) parts.push(`tags=${rule.tags_all.join(",")}`);
    if (rule.source_bucket) parts.push(`bucket=${rule.source_bucket}`);
    if (rule.clip_kind) parts.push(`kind=${rule.clip_kind}`);
    const micRule = ruleMicValue(rule) || ruleSplitChannelValue(rule);
    if (micRule) parts.push(`mic=${micLabelFromValue(micRule)}`);
    if (rule.count !== undefined) parts.push(`${Number(rule.count)} clips`);
    return parts.length ? parts.join(" · ") : "all accepted raw clips";
}

function sourceRuleAudioLabel(rule) {
    const labels = rule && rule.labels;
    return labels && labels.length ? labels.join(",") : "*";
}

function sourceRuleBriefSummary(rule) {
    if (!rule) return "all accepted raw clips";
    const parts = [];
    if (rule.project_ids && rule.project_ids.length) parts.push(`project=${(rule.project_ids === "*" ? ["*"] : rule.project_ids).join(",")}`);
    if (rule.density) parts.push(`density=${rule.density}`);
    if (rule.tags_all && rule.tags_all.length) parts.push(`tags=${rule.tags_all.join(",")}`);
    if (rule.source_bucket) parts.push(`bucket=${rule.source_bucket}`);
    if (rule.clip_kind) parts.push(`kind=${rule.clip_kind}`);
    const micRule = ruleMicValue(rule) || ruleSplitChannelValue(rule);
    if (micRule) parts.push(`mic=${micLabelFromValue(micRule)}`);
    if (rule.count !== undefined) parts.push(`${Number(rule.count)} clips`);
    return parts.length ? parts.join(" · ") : "all accepted raw clips";
}

function sourceRuleDetailsHtml(rule) {
    const split = rule && (rule.split || rule.dataset_role || "train");
    const summary = sourceRuleBriefSummary(rule);
    return `<span class="lib-row-split-pill">${escapeHtml(split)}</span> · ${escapeHtml(summary)}`;
}

function rulesBriefSummary(rules) {
    if (!rules || !rules.length) return "not set";
    if (rules.length === 1) return ruleBriefSummary(rules[0]);
    const labels = [...new Set(rules.flatMap(rule => rule.labels || []))].filter(Boolean);
    const count = rules.reduce((acc, rule) => acc + Number(rule.count || 0), 0);
    return `${rules.length} buckets${labels.length ? ` · ${labels.join(", ")}` : ""} · ${count} clips`;
}

// Just the label name(s) — used as the headline of a saved recipe row so the
// row reads "riedel_veloce ⨯ cutlery" rather than "label=riedel_veloce · …".
function labelsOnlySummary(rules) {
    if (!rules || !rules.length) return "";
    const labels = [...new Set(rules.flatMap(rule => rule.labels || []))].filter(Boolean);
    return labels.length ? labels.join(",") : "";
}

// Muted meta line under the recipe head — project / density / tags / counts,
// merged across base + overlay filter sets without duplicating values.
function mergedRuleMeta(baseFilters, overlayFilters) {
    const collect = (rules, key) => [...new Set((rules || []).flatMap(r => {
        const v = r && r[key];
        if (!v) return [];
        if (Array.isArray(v)) return v;
        if (v === "*") return ["*"];
        return [v];
    }))].filter(Boolean);
    const baseCount = (baseFilters || []).reduce((acc, r) => acc + Number(r.count || 0), 0);
    const overlayCount = (overlayFilters || []).reduce((acc, r) => acc + Number(r.count || 0), 0);
    const projects = collect([...(baseFilters || []), ...(overlayFilters || [])], "project_ids");
    const densities = collect([...(baseFilters || []), ...(overlayFilters || [])], "density");
    const tags = collect([...(baseFilters || []), ...(overlayFilters || [])], "tags_all");
    const buckets = collect([...(baseFilters || []), ...(overlayFilters || [])], "source_bucket");
    const parts = [];
    if (projects.length) parts.push(`project=${projects.join(",")}`);
    if (densities.length) parts.push(`density=${densities.join(",")}`);
    if (tags.length) parts.push(`tags=${tags.join(",")}`);
    if (buckets.length) parts.push(`bucket=${buckets.join(",")}`);
    parts.push(`${baseCount} base`);
    parts.push(`${overlayCount} overlay`);
    return parts.join(" · ");
}

// Compact kv pairs for the recipe row's details panel — only includes
// settings that are actually on (e.g. semitones only when pitch is enabled).
function recipeDetailRows(recipe) {
    const s = (recipe && recipe.settings && typeof recipe.settings === "object") ? recipe.settings : {};
    const rows = [];
    if (recipe.description) rows.push(["description", String(recipe.description)]);
    if (recipe.tags && recipe.tags.length) rows.push(["tags", recipe.tags.join(", ")]);
    // Always-on knobs
    if (s.samples_per_source !== undefined) rows.push(["samples/source", String(s.samples_per_source)]);
    if (s.seed !== undefined) rows.push(["seed", String(s.seed)]);
    // Pitch
    if (s.pitch_shift_enabled) {
        const r = s.semitones_range || [];
        rows.push(["pitch", `${r[0] ?? "?"} … ${r[1] ?? "?"} semitones`]);
    } else if (s.pitch_shift_enabled === false) {
        rows.push(["pitch", "off"]);
    }
    // Overlay
    if (s.overlay_enabled) {
        const r = s.snr_db_range || [];
        rows.push(["snr (dB)", `${r[0] ?? "?"} … ${r[1] ?? "?"}`]);
        if (s.scale_noise_to_snr !== undefined) rows.push(["scale snr", s.scale_noise_to_snr ? "yes" : "no"]);
        if (s.prevent_clipping !== undefined) rows.push(["prevent clipping", s.prevent_clipping ? "yes" : "no"]);
        if (s.transient_window_ms !== undefined) rows.push(["transient win (ms)", String(s.transient_window_ms)]);
    }
    // Mask filter
    if (s.mask_filter_enabled) {
        if (s.mask_candidate_attempts !== undefined) rows.push(["mask attempts", String(s.mask_candidate_attempts)]);
        if (s.mask_min_mean_db !== undefined && s.mask_min_mean_db !== null) rows.push(["mask mean dB", String(s.mask_min_mean_db)]);
        if (s.mask_min_p25_db !== undefined && s.mask_min_p25_db !== null) rows.push(["mask p25 dB", String(s.mask_min_p25_db)]);
        if (s.mask_min_band_db !== undefined && s.mask_min_band_db !== null) rows.push(["mask worst-band dB", String(s.mask_min_band_db)]);
        if (s.mask_min_effective_snr_db !== undefined && s.mask_min_effective_snr_db !== null) rows.push(["min eff. SNR dB", String(s.mask_min_effective_snr_db)]);
        if (s.mask_band_count !== undefined) rows.push(["mask bands", String(s.mask_band_count)]);
        if (s.mask_top_frames !== undefined) rows.push(["mask top frames", String(s.mask_top_frames)]);
    } else if (s.mask_filter_enabled === false) {
        rows.push(["mask filter", "off"]);
    }
    // Bucket filter strings live in the muted meta line under the recipe head —
    // repeating them here was redundant clutter.
    return rows;
}

function setInputValue(id, value) {
    const el = document.getElementById(id);
    if (!el || value === undefined) return;
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = value === null ? "" : value;
}

function populateAugmentationSettings(settings) {
    if (!settings) return;
    setInputValue("aug-setting-enabled", settings.enabled);
    setInputValue("aug-setting-seed", settings.seed);
    setInputValue("aug-setting-samples-per-source", settings.samples_per_source);
    setInputValue("aug-setting-pitch-shift-enabled", settings.pitch_shift_enabled);
    setInputValue("aug-setting-semitones-min", (settings.semitones_range || [])[0]);
    setInputValue("aug-setting-semitones-max", (settings.semitones_range || [])[1]);
    setInputValue("aug-setting-overlay-enabled", settings.overlay_enabled);
    setInputValue("aug-setting-snr-min", (settings.snr_db_range || [])[0]);
    setInputValue("aug-setting-snr-max", (settings.snr_db_range || [])[1]);
    setInputValue("aug-setting-scale-noise-to-snr", settings.scale_noise_to_snr);
    setInputValue("aug-setting-prevent-clipping", settings.prevent_clipping);
    setInputValue("aug-setting-transient-window-ms", settings.transient_window_ms);
    setInputValue("aug-setting-mask-filter-enabled", settings.mask_filter_enabled);
    setInputValue("aug-setting-mask-candidate-attempts", settings.mask_candidate_attempts);
    setInputValue("aug-setting-mask-min-mean-db", settings.mask_min_mean_db);
    setInputValue("aug-setting-mask-min-p25-db", settings.mask_min_p25_db);
    setInputValue("aug-setting-mask-min-band-db", settings.mask_min_band_db);
    setInputValue("aug-setting-mask-min-effective-snr-db", settings.mask_min_effective_snr_db);
    setInputValue("aug-setting-mask-band-count", settings.mask_band_count);
    setInputValue("aug-setting-mask-top-frames", settings.mask_top_frames);
    augmentationSettingsDirty = false;
    updateAugmentationSettingState();
    updateAugmentationSettingsStatus("saved defaults");
}

function updateAugmentationSettingsStatus(text) {
    const status = document.getElementById("augment-settings-status");
    if (status) status.textContent = text || "";
}

let augmentationSettingsSaveTimer = null;
function markAugmentationSettingsDirty() {
    augmentationSettingsDirty = true;
    updateAugmentationSettingsStatus("saving…");
    if (augmentationSettingsSaveTimer) clearTimeout(augmentationSettingsSaveTimer);
    // Debounce so dragging a number input doesn't fire a request per pixel.
    augmentationSettingsSaveTimer = setTimeout(() => {
        saveAugmentationSettings(false)
            .then(() => updateAugmentationSettingsStatus("draft — baked into the recipe on Add"))
            .catch(() => updateAugmentationSettingsStatus("save failed"));
    }, 350);
}

function updateAugmentationSettingState() {
    const globalEnabled = inputChecked("aug-setting-enabled");
    document.querySelectorAll("[data-disabled-by]").forEach(input => {
        const toggle = document.getElementById(input.dataset.disabledBy);
        const enabled = globalEnabled && (!toggle || toggle.checked);
        input.disabled = !enabled;
    });
    document.querySelectorAll(".augment-group").forEach(group => {
        const toggle = group.querySelector(".augment-group-title input[type='checkbox']");
        const enabled = globalEnabled && (!toggle || toggle.checked || toggle.id === "aug-setting-enabled");
        group.classList.toggle("disabled", !enabled);
    });
}

function inputNumber(id, fallback = 0) {
    const el = document.getElementById(id);
    const value = el ? el.value : "";
    if (value === "" || value === undefined) return fallback;
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

function inputOptionalNumber(id) {
    const el = document.getElementById(id);
    const value = el ? el.value : "";
    if (value === "" || value === undefined || value === null) return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

function inputChecked(id) {
    const el = document.getElementById(id);
    return !!(el && el.checked);
}

function collectAugmentationSettings() {
    return {
        enabled: inputChecked("aug-setting-enabled"),
        seed: inputNumber("aug-setting-seed", 42),
        samples_per_source: inputNumber("aug-setting-samples-per-source", 1),
        pitch_shift_enabled: inputChecked("aug-setting-pitch-shift-enabled"),
        semitones_range: [
            inputNumber("aug-setting-semitones-min", -1),
            inputNumber("aug-setting-semitones-max", 1),
        ],
        overlay_enabled: inputChecked("aug-setting-overlay-enabled"),
        snr_db_range: [
            inputNumber("aug-setting-snr-min", 5),
            inputNumber("aug-setting-snr-max", 30),
        ],
        scale_noise_to_snr: inputChecked("aug-setting-scale-noise-to-snr"),
        prevent_clipping: inputChecked("aug-setting-prevent-clipping"),
        transient_window_ms: inputNumber("aug-setting-transient-window-ms", 200),
        mask_filter_enabled: inputChecked("aug-setting-mask-filter-enabled"),
        mask_candidate_attempts: inputNumber("aug-setting-mask-candidate-attempts", 8),
        mask_min_mean_db: inputOptionalNumber("aug-setting-mask-min-mean-db"),
        mask_min_p25_db: inputOptionalNumber("aug-setting-mask-min-p25-db"),
        mask_min_band_db: inputOptionalNumber("aug-setting-mask-min-band-db"),
        mask_min_effective_snr_db: inputOptionalNumber("aug-setting-mask-min-effective-snr-db"),
        mask_band_count: inputNumber("aug-setting-mask-band-count", 10),
        mask_top_frames: inputNumber("aug-setting-mask-top-frames", 5),
    };
}

async function saveAugmentationSettings(showStatus = true) {
    const status = document.getElementById("augment-settings-status");
    if (showStatus && status) status.textContent = "saving...";
    const settings = collectAugmentationSettings();
    const r = await fetch("/api/augmentation-settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings }),
    });
    const data = await r.json();
    if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
    selectionState = data.selection || selectionState;
    selectionState.augmentation_settings = data.settings || settings;
    augmentationSettingsDirty = false;
    updateAugmentationSettingsStatus("draft — baked into the recipe on Add");
    if (showStatus) flash("augmentation panel state saved", "ok");
    return data.settings || settings;
}

function markReviewDirty() {
    reviewDirty = true;
    const btn = document.getElementById("library-refresh-review");
    const status = document.getElementById("review-refresh-status");
    if (btn) btn.disabled = !!(augmentationRunning || deleteRunning || refreshRunning);
    if (status) status.textContent = "review tabs need apply";
}

function setRefreshHeld(held) {
    const btn = document.getElementById("library-refresh-review");
    if (btn) btn.disabled = !!held || !!(augmentationRunning || deleteRunning || refreshRunning);
}

function sourceRuleCount(assignments) {
    return Object.values(assignments || {}).reduce((acc, rows) => acc + (Array.isArray(rows) ? rows.length : 0), 0);
}

function setSourceRulesStatus(text, cls = "") {
    const status = document.getElementById("source-rules-status");
    if (!status) return;
    status.textContent = text || "";
    status.className = "section-status" + (cls ? ` ${cls}` : "");
}

function renderSourceRulesLoading() {
    setSourceRulesStatus("loading...");
    const clearBtn = document.getElementById("source-rules-clear");
    if (clearBtn) clearBtn.disabled = true;
    const box = document.getElementById("library-rule-table-body");
    if (box) box.innerHTML = `<div class="lib-empty busy">Loading saved source rules from dataset_selection.json...</div>`;
}

function renderSourceRulesFailure(error) {
    selectionLoadError = error ? String(error) : "unknown error";
    setSourceRulesStatus("load failed", "err");
    const clearBtn = document.getElementById("source-rules-clear");
    if (clearBtn) clearBtn.disabled = true;
    const box = document.getElementById("library-rule-table-body");
    if (box) {
        box.innerHTML = `<div class="lib-empty err">Could not load saved source rules: ${escapeHtml(selectionLoadError)}</div>`;
    }
}

async function loadSelectionState() {
    if (!selectionLoaded) renderSourceRulesLoading();
    try {
        const r = await fetch("/api/selection", { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        selectionState = data.selection || { assignments: {}, augmentation_recipes: [], augmentation_settings: {} };
        selectionLoaded = true;
        selectionLoadError = "";
        populateAugmentationSettings(data.augmentation_settings || selectionState.augmentation_settings || {});
        renderSelectionRules();
        renderAugmentationRules();
        renderExistingAugmentations();
        loadAugmentationGeneratedCounts();
    } catch (e) {
        renderSourceRulesFailure(e && e.message ? e.message : e);
        console.error("selection load failed:", e);
        flash("selection load failed", "err");
    }
}

async function addSelectionRule() {
    const assignBtn = document.getElementById("library-assign-class");
    if (assignBtn) assignBtn.disabled = true;
    setRefreshHeld(true);
    const className = (document.getElementById("library-target-class") || {}).value || "";
    const split = (document.getElementById("library-target-split") || {}).value || "native";
    if (!className) {
        alert("Choose a target class first.");
        if (assignBtn) assignBtn.disabled = false;
        setRefreshHeld(false);
        return;
    }
    const rules = selectedOrVisibleLibraryRules(split);
    if (!rules.length) {
        alert("No bucket rows match the current filters.");
        if (assignBtn) assignBtn.disabled = false;
        setRefreshHeld(false);
        return;
    }
    if (rules.some(rule => rule.source_bucket === "mined")) {
        alert("Promote mined clips first, then assign the promoted raw label.");
        if (assignBtn) assignBtn.disabled = false;
        setRefreshHeld(false);
        return;
    }
    if (rules.some(rule => !rule.count)) {
        if (!confirm("At least one selected bucket currently has 0 clips. Save anyway?")) {
            if (assignBtn) assignBtn.disabled = false;
            setRefreshHeld(false);
            return;
        }
    }
    try {
        let saved = 0;
        for (const rule of rules) {
            const r = await fetch("/api/selection-rule", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "add", class_name: className, split: rule.split || "train", rule }),
            });
            const data = await r.json();
            if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
            selectionState = data.selection;
            saved += Number(data.added || 0);
        }
        renderSelectionRules();
        markReviewDirty();
        flash(`saved ${saved} rule(s) for ${className}`, "ok");
    } catch (e) {
        flash("rule save failed", "err");
    } finally {
        if (assignBtn) assignBtn.disabled = false;
        setRefreshHeld(false);
    }
}

async function removeSelectionRule(className, ruleId) {
    setRefreshHeld(true);
    try {
        const r = await fetch("/api/selection-rule", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "remove", class_name: className, rule_id: ruleId }),
        });
        const data = await r.json();
        if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
        selectionState = data.selection;
        renderSelectionRules();
        markReviewDirty();
        flash("rule removed", "ok");
    } catch (e) {
        flash("rule remove failed", "err");
    } finally {
        setRefreshHeld(false);
    }
}

// Overlay class(es) of a derived source rule — backend annotates derived rules
// with the distinct overlays among their matched clips (see _attach_overlay_labels).
function sourceRuleOverlayLabels(rule) {
    return Array.isArray(rule && rule.overlay_labels) ? rule.overlay_labels.filter(Boolean) : [];
}

// "overlay=a,b,c" (first 3) with "…" + a hover tooltip listing the full set.
function sourceRuleOverlayHtml(rule) {
    const all = sourceRuleOverlayLabels(rule);
    if (!all.length) return "";
    const shown = all.slice(0, 3).join(",");
    const more = all.length > 3 ? "…" : "";
    return ` · <span class="rule-overlay" title="overlay: ${escapeHtml(all.join(", "))}">overlay=${escapeHtml(shown)}${more}</span>`;
}

// Everything shown on a row (plus overlay) joined for the search filter.
function sourceRuleHaystack(className, rule) {
    return [
        className,
        sourceRuleAudioLabel(rule),
        sourceRuleBriefSummary(rule),
        rule && rule.split,
        sourceRuleOverlayLabels(rule).join(" "),
    ].filter(Boolean).join(" ").toLowerCase();
}

function renderSelectionRules() {
    const box = document.getElementById("library-rule-table-body");
    if (!box) return;
    const assignments = selectionState.assignments || {};
    const rows = [];
    Object.keys(assignments).sort().forEach(className => {
        (assignments[className] || []).forEach(rule => rows.push({ className, rule }));
    });
    if (!selectionLoaded && selectionLoadError) {
        renderSourceRulesFailure(selectionLoadError);
        return;
    }
    if (!selectionLoaded) {
        renderSourceRulesLoading();
        return;
    }
    const clearBtn = document.getElementById("source-rules-clear");
    if (clearBtn) clearBtn.disabled = false;
    const q = (sourceRuleQuery || "").trim().toLowerCase();
    const filtered = q
        ? rows.filter(({ className, rule }) => sourceRuleHaystack(className, rule).includes(q))
        : rows;
    if (!rows.length) setSourceRulesStatus("0 rules", "");
    else if (q) setSourceRulesStatus(`showing ${filtered.length} of ${rows.length}`, filtered.length ? "ok" : "");
    else setSourceRulesStatus(`loaded ${rows.length}`, "ok");
    let emptyMsg = `<div class="lib-empty">No source rules yet — pick clips in the Library table, then click Assign to class.</div>`;
    if (rows.length && !filtered.length) emptyMsg = `<div class="lib-empty">No source rules match “${escapeHtml(q)}”.</div>`;
    box.innerHTML = filtered.length ? filtered.map(({ className, rule }) => {
        const isDerived = (rule && rule.source_bucket) === "derived";
        return `
        <div class="lib-row source-rule-row${isDerived ? ' derived' : ''}" style="${escapeHtml(classTintStyle(className))}">
            <span class="lib-row-name">${escapeHtml(className)} · ${escapeHtml(sourceRuleAudioLabel(rule))}${isDerived ? ' <span class="aug-hint" title="augmented bucket">⨯aug</span>' : ''}</span>
            <span class="lib-row-kv">${sourceRuleDetailsHtml(rule)}${isDerived ? sourceRuleOverlayHtml(rule) : ''}</span>
            <button type="button" class="lib-row-x" aria-label="Remove" title="Remove rule" data-remove-rule="${escapeHtml(rule.id || '')}" data-class="${escapeHtml(className)}">×</button>
        </div>
    `;}).join("") : emptyMsg;
    box.querySelectorAll("[data-remove-rule]").forEach(btn => {
        btn.addEventListener("click", () => removeSelectionRule(btn.dataset.class, btn.dataset.removeRule));
    });
    renderLibraryTable();
    updateMutatingBusy();
}

async function clearAllSelectionRules() {
    if (!selectionLoaded) {
        flash("source rules are still loading", "err");
        return;
    }
    const assignments = selectionState.assignments || {};
    const total = sourceRuleCount(assignments);
    if (!total) {
        flash("no source rules to clear", "ok");
        return;
    }
    if (!confirm(`Remove all ${total} source rule(s) from every class?`)) return;
    setRefreshHeld(true);
    try {
        const r = await fetch("/api/selection-rule", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "clear" }),
        });
        const data = await r.json();
        if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
        selectionState = data.selection;
        renderSelectionRules();
        markReviewDirty();
        flash(`removed ${data.removed || total} rule(s)`, "ok");
    } catch (e) {
        flash("clear rules failed", "err");
    } finally {
        setRefreshHeld(false);
    }
}

function updateRecipeBucketSummary() {
    const base = document.getElementById("recipe-base-summary");
    const overlay = document.getElementById("recipe-overlay-summary");
    if (base) base.textContent = `base: ${rulesBriefSummary(recipeBaseRules)}`;
    if (overlay) overlay.textContent = `overlay: ${rulesBriefSummary(recipeOverlayRules)}`;
}

// Prefill description ("base x overlay") and tags (union of both buckets'
// tags) from the picked buckets. Fields stay editable; a field the user has
// typed into (dataset.userEdited) is never overwritten.
function applyRecipeFieldDefaults() {
    const desc = document.getElementById("recipe-description");
    const tags = document.getElementById("recipe-tags");
    const baseLabels = labelsOnlySummary(recipeBaseRules);
    if (desc && !desc.dataset.userEdited && baseLabels) {
        const overlayLabels = labelsOnlySummary(recipeOverlayRules);
        desc.value = overlayLabels ? `${baseLabels} x ${overlayLabels}` : baseLabels;
    }
    if (tags && !tags.dataset.userEdited) {
        const merged = [...new Set(
            [...recipeBaseRules, ...recipeOverlayRules].flatMap(r => (r && r.tags_all) || [])
        )].filter(Boolean);
        if (merged.length) tags.value = merged.join(",");
    }
}

function resetRecipeFields() {
    ["recipe-description", "recipe-tags"].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.value = "";
        delete el.dataset.userEdited;
    });
    applyRecipeFieldDefaults();
}

function setRecipeBucket(which) {
    // Keep each bucket's own dataset role — forcing "train" here made test
    // buckets count 0 clips and impossible to augment.
    const rules = selectedOrVisibleLibraryRules("native");
    if (!rules.length) {
        alert("No bucket rows match the current filters.");
        return;
    }
    if (which === "base") {
        recipeBaseRules = rules;
    } else {
        recipeOverlayRules = rules;
        // Picking an overlay bucket means the user wants the overlay stage —
        // turn it on so the recipe doesn't silently degrade to pitch-only.
        const overlayToggle = document.getElementById("aug-setting-overlay-enabled");
        if (overlayToggle && !overlayToggle.checked) {
            overlayToggle.checked = true;
            overlayToggle.dispatchEvent(new Event("change", { bubbles: true }));
        }
    }
    updateRecipeBucketSummary();
    applyRecipeFieldDefaults();
}

function parseTagInput(value) {
    const out = [];
    const seen = new Set();
    (value || "").split(",").forEach(raw => {
        const tag = raw.trim().replace(/^#/, "");
        if (tag && !seen.has(tag)) {
            out.push(tag);
            seen.add(tag);
        }
    });
    return out;
}

async function addAugmentationRule() {
    const settings = collectAugmentationSettings();
    if (!settings.enabled) {
        alert("Enable augmentation before adding a recipe.");
        return;
    }
    if (!recipeBaseRules.length) {
        alert("Set the base bucket first.");
        return;
    }
    // An explicitly chosen overlay bucket always wins over the panel toggle —
    // never silently drop it (that produced pitch-only recipes with 0 overlay).
    if (recipeOverlayRules.length) {
        settings.overlay_enabled = true;
    } else if (settings.overlay_enabled) {
        alert("Set the overlay bucket, or disable overlay for a pitch-only recipe.");
        return;
    }
    if (!settings.pitch_shift_enabled && !settings.overlay_enabled) {
        alert("Enable pitch shift, overlay, or both before adding a recipe.");
        return;
    }
    // The recipe's output split follows the base buckets: test bases generate
    // frozen-test clips. Mixed train+test bases would make the destination
    // ambiguous, so refuse those.
    const baseRoles = new Set(recipeBaseRules.map(rule => (rule.dataset_role === "test" ? "test" : "train")));
    if (baseRoles.size > 1) {
        alert("Base buckets mix train and test clips. Pick buckets from one split per recipe.");
        return;
    }
    const frozenTest = baseRoles.has("test");
    // Snapshot the current Augmentation Settings panel onto the recipe so it
    // runs with the same pitch / SNR / mask thresholds that were on screen
    // when the user clicked Add — independent of any later panel edits.
    const recipe = {
        base_filters: recipeBaseRules,
        overlay_filters: recipeOverlayRules,
        description: (document.getElementById("recipe-description") || {}).value || "",
        tags: parseTagInput((document.getElementById("recipe-tags") || {}).value || ""),
        frozen_test: frozenTest,
        settings,
    };
    try {
        const r = await fetch("/api/augmentation-rule", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                action: "add",
                recipe,
            }),
        });
        const data = await r.json();
        if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
        selectionState = data.selection;
        renderAugmentationRules();
        markReviewDirty();
        resetRecipeFields();
        flash("saved augmentation recipe", "ok");
    } catch (e) {
        flash("recipe save failed", "err");
    }
}

async function removeAugmentationRule(ruleId) {
    const deleteGenerated = confirm(`Remove generated clips for this recipe too?

OK = remove recipe and generated clips.
Cancel = remove only the saved recipe.`);
    try {
        const r = await fetch("/api/augmentation-rule", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "remove", rule_id: ruleId, delete_generated: deleteGenerated }),
        });
        const data = await r.json();
        if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
        selectionState = data.selection;
        renderAugmentationRules();
        renderExistingAugmentations();
        markReviewDirty();
        flash(deleteGenerated ? `recipe removed, deleted ${data.deleted_generated || 0} generated clip(s)` : "recipe removed", "ok");
    } catch (e) {
        flash("recipe remove failed", "err");
    }
}

function renderAugmentationRules() {
    const box = document.getElementById("augmentation-rule-table-body");
    if (!box) return;
    const rows = Array.isArray(selectionState.augmentation_recipes) ? selectionState.augmentation_recipes : [];
    if (!rows.length) {
        box.innerHTML = `<div class="lib-empty">No augmentation recipes saved yet. Build one with the Recipe Builder above.</div>`;
        updateMutatingBusy();
        return;
    }
    const panelSamples = Math.max(1, inputNumber("aug-setting-samples-per-source", 1));
    box.innerHTML = rows.map(recipe => {
        const baseFilters = recipe.base_filters || (recipe.base_filter ? [recipe.base_filter] : []);
        const overlayFilters = recipe.overlay_filters || (recipe.overlay_filter ? [recipe.overlay_filter] : []);
        const baseCount = recipe.base_count !== undefined ? Number(recipe.base_count) : baseFilters.reduce((acc, rule) => acc + Number(rule.count || 0), 0);
        const overlayCount = recipe.overlay_count !== undefined ? Number(recipe.overlay_count) : overlayFilters.reduce((acc, rule) => acc + Number(rule.count || 0), 0);
        const rid = recipe.recipe_id || recipe.id || "";
        const countRow = augmentationGeneratedCounts[rid] || {};
        const generated = Number(countRow.current !== undefined ? countRow.current : countRow.total !== undefined ? countRow.total : 0);
        const stale = Number(countRow.stale || 0);
        // Use the recipe's own snapshotted samples_per_source for the expected
        // math — newer recipes can override the current panel value.
        const settings = (recipe.settings && typeof recipe.settings === "object") ? recipe.settings : {};
        const sps = Number(settings.samples_per_source ?? panelSamples) || 1;
        const expected = Math.max(1, baseCount * sps);
        const pct = Math.max(0, Math.min(100, (generated / expected) * 100));
        const selected = recipe.selected !== false;
        const isOpen = expandedRecipes.has(rid);
        const prefixVal = escapeHtml(recipe.prefix || "");
        // Collapse rule sets into name-only labels in the head, and push the
        // structured key=value text to the muted meta line below the head.
        const baseLabel = labelsOnlySummary(baseFilters) || (recipe.base || "?");
        // The overlay bucket is authoritative: only a recipe without overlay
        // filters is pitch-only.
        const overlayLabel = overlayFilters.length
            ? (labelsOnlySummary(overlayFilters) || recipe.overlay || "?")
            : "pitch";
        const metaParts = mergedRuleMeta(baseFilters, overlayFilters);
        const detailsRows = recipeDetailRows(recipe);
        return `
        <div class="recipe-row" data-recipe-id="${escapeHtml(rid)}">
          <div class="recipe-row-head">
            <label class="recipe-select" title="Include in Generate selected"><input type="checkbox" data-recipe-select="${escapeHtml(rid)}" ${selected ? "checked" : ""}></label>
            <span class="recipe-row-title">${escapeHtml(baseLabel)}<span class="x">⨯</span>${escapeHtml(overlayLabel)}</span>
            <span class="recipe-row-stats">${generated}/${expected}${stale ? ` · ${stale} stale` : ""}<span class="recipe-progress"><span style="width:${pct.toFixed(1)}%"></span></span></span>
            <button type="button" class="recipe-row-toggle" data-toggle-recipe="${escapeHtml(rid)}" aria-expanded="${isOpen ? 'true' : 'false'}" title="Per-recipe settings">${isOpen ? '▾' : '▸'}</button>
            <button type="button" class="recipe-row-remove" data-remove-recipe="${escapeHtml(rid)}">Remove</button>
          </div>
          <div class="recipe-row-meta">${escapeHtml(metaParts)}</div>
          <div class="recipe-row-body"${isOpen ? '' : ' hidden'}>
            <label>prefix <input type="text" data-recipe-prefix="${escapeHtml(rid)}" value="${prefixVal}" placeholder="ui-pass"></label>
            <label class="check"><input type="checkbox" data-recipe-replace="${escapeHtml(rid)}" ${recipe.replace_existing ? "checked" : ""}> replace existing</label>
            <label class="check"><input type="checkbox" data-recipe-frozen="${escapeHtml(rid)}" ${recipe.frozen_test ? "checked" : ""}> frozen test</label>
            <details>
              <summary>details</summary>
              <div class="kv-table">
                ${detailsRows.map(([k, v]) => `<span class="k">${escapeHtml(k)}</span><span class="v">${escapeHtml(v)}</span>`).join("")}
              </div>
            </details>
          </div>
        </div>
        `;
    }).join("");
    // Toggle expand/collapse
    box.querySelectorAll("[data-toggle-recipe]").forEach(btn => {
        btn.addEventListener("click", () => {
            const rid = btn.dataset.toggleRecipe;
            const row = btn.closest(".recipe-row");
            const body = row && row.querySelector(".recipe-row-body");
            if (!body) return;
            if (body.hidden) { body.hidden = false; expandedRecipes.add(rid); btn.textContent = "▾"; btn.setAttribute("aria-expanded", "true"); }
            else              { body.hidden = true;  expandedRecipes.delete(rid); btn.textContent = "▸"; btn.setAttribute("aria-expanded", "false"); }
        });
    });
    box.querySelectorAll("[data-remove-recipe]").forEach(btn => {
        btn.addEventListener("click", () => removeAugmentationRule(btn.dataset.removeRecipe));
    });
    // Selection / per-recipe controls — debounce text input to avoid hammering
    // the API on each keystroke; checkboxes update immediately.
    box.querySelectorAll("[data-recipe-select]").forEach(input => {
        input.addEventListener("change", () => updateAugmentationRecipe(input.dataset.recipeSelect, { selected: input.checked }));
    });
    box.querySelectorAll("[data-recipe-replace]").forEach(input => {
        input.addEventListener("change", () => updateAugmentationRecipe(input.dataset.recipeReplace, { replace_existing: input.checked }));
    });
    box.querySelectorAll("[data-recipe-frozen]").forEach(input => {
        input.addEventListener("change", () => updateAugmentationRecipe(input.dataset.recipeFrozen, { frozen_test: input.checked }));
    });
    box.querySelectorAll("[data-recipe-prefix]").forEach(input => {
        const rid = input.dataset.recipePrefix;
        const fire = () => updateAugmentationRecipe(rid, { prefix: input.value });
        input.addEventListener("input", () => {
            clearTimeout(recipeUpdateTimers[rid]);
            recipeUpdateTimers[rid] = setTimeout(fire, 250);
        });
        input.addEventListener("change", () => { clearTimeout(recipeUpdateTimers[rid]); fire(); });
    });
    updateMutatingBusy();
}

async function updateAugmentationRecipe(rid, patch) {
    const rows = Array.isArray(selectionState.augmentation_recipes) ? selectionState.augmentation_recipes : [];
    const current = rows.find(r => (r.recipe_id || r.id) === rid);
    if (!current) return;
    const recipe = { ...current, ...patch };
    try {
        const r = await fetch("/api/augmentation-rule", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "update", recipe }),
        });
        const data = await r.json();
        if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
        selectionState = data.selection;
        // Mutate in-place so the DOM (and the input the user is currently
        // typing into) stays stable; no re-render.
        const idx = rows.findIndex(r => (r.recipe_id || r.id) === rid);
        if (idx >= 0) {
            const fresh = (selectionState.augmentation_recipes || []).find(r => (r.recipe_id || r.id) === rid);
            if (fresh) rows[idx] = fresh;
        }
    } catch (e) {
        flash(`recipe update failed: ${e.message || e}`, "err");
    }
}

function renderExistingAugmentations() {
    const box = document.getElementById("existing-augmentation-list");
    if (!box) return;
    if (!augmentationGeneratedCountsLoaded && !augmentationGeneratedCountsError) {
        box.innerHTML = `<div class="lib-empty busy">Loading generated augmentation counts...</div>`;
        return;
    }
    if (augmentationGeneratedCountsError && !augmentationGeneratedCountsLoaded) {
        box.innerHTML = `<div class="lib-empty err">Could not load generated augmentation counts: ${escapeHtml(augmentationGeneratedCountsError)}</div>`;
        return;
    }
    const rows = [];
    const recipes = Array.isArray(selectionState.augmentation_recipes) ? selectionState.augmentation_recipes : [];
    const recipesById = {};
    recipes.forEach(recipe => {
        const rid = recipe.recipe_id || recipe.id || "";
        if (rid) recipesById[rid] = recipe;
    });
    // Every generation found on disk for this project — including orphans whose
    // recipe was removed — so leftover files are always visible and deletable.
    Object.keys(augmentationGeneratedCounts).sort().forEach(rid => {
        const counts = augmentationGeneratedCounts[rid] || {};
        const current = Number(counts.current || counts.total || 0);
        const stale = Number(counts.stale || 0);
        if (!current && !stale) return;
        const recipe = recipesById[rid];
        const overlayFilters = recipe ? (recipe.overlay_filters || (recipe.overlay_filter ? [recipe.overlay_filter] : [])) : [];
        const base = (recipe ? recipe.base : counts.base_label) || "?";
        const overlay = recipe
            ? (overlayFilters.length ? (recipe.overlay || "?") : "pitch")
            : (counts.overlay_label || "pitch");
        const orphan = !recipe;
        const kv = `${current} current${stale ? ` · ${stale} stale` : ""}${orphan ? " · recipe removed" : ""}`;
        rows.push(`
          <div class="lib-row">
            <span class="lib-row-name">${escapeHtml(base)} ⨯ ${escapeHtml(overlay)}</span>
            <span class="lib-row-kv">${escapeHtml(kv)}</span>
            <button type="button" class="lib-row-x" aria-label="Delete generated files" title="Delete all generated WAV/JSON files of this augmentation run" data-delete-existing="${escapeHtml(rid)}">×</button>
          </div>
        `);
    });
    box.innerHTML = rows.length ? rows.join("") : `<div class="lib-empty">No generated augmentations yet.</div>`;
    box.querySelectorAll("[data-delete-existing]").forEach(btn => {
        btn.addEventListener("click", () => deleteExistingAugmentation(btn.dataset.deleteExisting));
    });
    updateMutatingBusy();
}

async function loadAugmentationGeneratedCounts() {
    if (augmentationGeneratedCountsLoading) return;
    augmentationGeneratedCountsLoading = true;
    augmentationGeneratedCountsError = "";
    renderExistingAugmentations();
    try {
        const r = await fetch("/api/augmentation-generated-counts", { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (!data.ok) throw new Error(data.error || "count load failed");
        augmentationGeneratedCounts = data.augmentation_generated_counts || {};
        augmentationGeneratedCountsLoaded = true;
        renderExistingAugmentations();
        // The Saved Recipes table is drawn once before counts arrive, so refresh
        // it here — otherwise every recipe is stuck showing "0/expected".
        renderAugmentationRules();
    } catch (e) {
        augmentationGeneratedCountsError = e && e.message ? e.message : String(e);
        console.error("generated augmentation count load failed:", e);
        augmentationGeneratedCountsLoading = false;
        renderExistingAugmentations();
        renderAugmentationRules();
    } finally {
        augmentationGeneratedCountsLoading = false;
    }
}

// True uncapped per-class counts for the dataset/augmented/frozen/mined tabs.
// Fetched once on load so headers/tab counts show the real size even though
// only the first cards are rendered; rescans refresh these via panel_totals.
async function loadPanelTotals() {
    try {
        const r = await fetch("/api/panel-totals", { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (!data.ok) throw new Error(data.error || "panel totals load failed");
        panelTotals = data.panel_totals || {};
        // Re-render so headers/tab counts and "Load more (on disk)" reflect totals.
        renderReviewData();
    } catch (e) {
        console.error("panel totals load failed:", e);
    }
}

async function deleteExistingAugmentation(rid) {
    if (!rid) return;
    if (!confirm("Delete all generated clip files for this augmentation run? (A saved recipe stays and can regenerate them.)")) return;
    try {
        const r = await fetch("/api/augmentation-rule", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "delete_generated", rule_id: rid }),
        });
        const data = await r.json();
        if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
        flash(`deleted ${data.deleted_generated || 0} generated clip(s)`, "ok");
        await loadSelectionState();
        markReviewDirty();
    } catch (e) {
        flash(`delete failed: ${e.message || e}`, "err");
    }
}

function setAugmentationBusy(busy, canceling = false) {
    augmentationRunning = busy || canceling;
    augmentationCanceling = canceling;
    updateMutatingBusy();
}

function updateMutatingBusy() {
    const busy = augmentationRunning || deleteRunning || refreshRunning || composeRunning;
    if (document.body) document.body.classList.toggle("ui-busy", busy);
    const selectors = [
        "#library-assign-class",
        "#library-delete-selected",
        "#library-delete-small",
        "#library-refresh-review",
        "#compose-dataset-btn",
        "#del-btn",
        "#promote-mined-btn",
        "#done-btn",
        "#source-rules-clear",
        "#recipe-add",
        "#recipe-use-base",
        "#recipe-use-overlay",
        "#augment-generate",
        ".library-row-delete",
        "[data-remove-rule]",
        "[data-remove-recipe]",
        "[data-delete-existing]",
    ];
    document.querySelectorAll(selectors.join(",")).forEach(el => {
        if (busy) {
            if (el.dataset.idleTitle === undefined) el.dataset.idleTitle = el.getAttribute("title") || "";
            el.setAttribute("title", "Background job is running.");
        } else if (el.dataset.idleTitle !== undefined) {
            if (el.dataset.idleTitle) el.setAttribute("title", el.dataset.idleTitle);
            else el.removeAttribute("title");
            delete el.dataset.idleTitle;
        }
        el.disabled = busy;
    });
    updateLibraryBulkDeleteButton();
    const cancel = document.getElementById("augment-cancel");
    if (cancel) cancel.disabled = !augmentationRunning || augmentationCanceling || deleteRunning || refreshRunning;
}

function updateAugmentProgress(job) {
    const status = document.getElementById("augment-status");
    const bar = document.getElementById("augment-progress-bar");
    const total = Number(job.total || 0);
    const completed = Number(job.completed || 0);
    const pct = total > 0 ? Math.max(0, Math.min(100, completed / total * 100)) : 0;
    if (bar) bar.style.width = `${pct.toFixed(1)}%`;
    if (status) {
        const speed = Number(job.clips_per_sec || 0);
        status.textContent = `${job.state || "running"} · ${completed}/${total} · generated ${job.generated || 0} · skipped ${job.skipped || 0} · ${speed.toFixed(2)} clips/s`;
    }
    if (!["finished", "canceled", "error"].includes(job.state || "")) {
        setBackgroundJobStatus(`augmenting ${completed}/${total}`);
    }
}

async function pollAugmentationJob() {
    if (!augmentJobId) return;
    try {
        const r = await fetch(`/api/augment/status?job_id=${encodeURIComponent(augmentJobId)}`, { cache: "no-store" });
        const data = await r.json();
        if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
        const job = data.job || {};
        updateAugmentProgress(job);
        if (["finished", "canceled", "error"].includes(job.state)) {
            clearInterval(augmentPollTimer);
            augmentPollTimer = null;
            setAugmentationBusy(false, false);
            augmentJobId = "";
            await loadSelectionState();
            markReviewDirty();
            const warnings = Array.isArray(job.warnings) ? job.warnings : [];
            // Persistent status line (the flash fades after a second; this stays
            // until the next run) + auto-refresh so the new derived buckets show
            // up in the Library table without a manual Apply click.
            const status = document.getElementById("augment-status");
            if (job.state === "finished") {
                const summary = `finished — ${job.generated || 0} clip(s) generated, ${job.skipped || 0} skipped → Library bucket=derived`;
                if (status) status.textContent = warnings.length ? `${summary} · ${warnings.join('; ')}` : summary;
                if (warnings.length) flash(`augmentation finished — ${warnings.join('; ')}`, "ok");
                else flash(`augmentation finished — generated ${job.generated || 0} clip(s)`, "ok");
                refreshReviewDataJob({ silent: true });
            } else if (job.state === "canceled") {
                if (status) status.textContent = `canceled — ${job.generated || 0} clip(s) kept`;
                flash("augmentation canceled", "ok");
                refreshReviewDataJob({ silent: true });
            } else {
                if (status) status.textContent = `failed: ${job.error || 'unknown error'}`;
                flash(`augmentation failed: ${job.error || 'unknown error'}`, "err");
            }
            setBackgroundJobStatus("");
            return;
        }
        setAugmentationBusy(job.state === "running", job.state === "canceling");
    } catch (e) {
        clearInterval(augmentPollTimer);
        augmentPollTimer = null;
        setAugmentationBusy(false, false);
        setBackgroundJobStatus("");
        flash("augmentation status failed", "err");
    }
}

async function generateAugmentations() {
    const status = document.getElementById("augment-status");
    if (augmentationRunning) return;
    const currentSettings = collectAugmentationSettings();
    if (!currentSettings.enabled) {
        flash("enable augmentation before generating", "err");
        return;
    }
    if (!currentSettings.pitch_shift_enabled && !currentSettings.overlay_enabled) {
        flash("enable pitch shift, overlay, or both before generating", "err");
        return;
    }
    const recipes = Array.isArray(selectionState.augmentation_recipes) ? selectionState.augmentation_recipes : [];
    const recipeIds = recipes
        .filter(r => r.selected !== false)
        .map(r => r.recipe_id || r.id)
        .filter(Boolean);
    if (!recipeIds.length) {
        flash("no recipes selected — tick at least one Saved Recipe row", "err");
        return;
    }
    setAugmentationBusy(true, false);
    setBackgroundJobStatus("starting augmentation");
    if (status) status.textContent = "starting...";
    try {
        const savedSettings = await saveAugmentationSettings(false);
        const r = await fetch("/api/augment/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ recipe_ids: recipeIds, settings: savedSettings }),
        });
        const data = await r.json();
        if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
        augmentJobId = (data.job || {}).job_id || "";
        updateAugmentProgress(data.job || {});
        if (augmentPollTimer) clearInterval(augmentPollTimer);
        augmentPollTimer = setInterval(pollAugmentationJob, 700);
    } catch (e) {
        setAugmentationBusy(false, false);
        setBackgroundJobStatus("");
        if (status) status.textContent = "start failed";
        flash(`augmentation start failed: ${e.message || e}`, "err");
    }
}

async function cancelAugmentation() {
    if (!augmentJobId) return;
    setAugmentationBusy(false, true);
    setBackgroundJobStatus("stopping augmentation");
    const status = document.getElementById("augment-status");
    if (status) status.textContent = "stopping...";
    try {
        await fetch("/api/augment/cancel", { method: "POST" });
    } catch (e) {
        flash("cancel failed", "err");
    }
}

function panelDataForTab(tabType) {
    if (tabType === "dataset") return CLIPS_DATA.panels || {};
    if (tabType === "augmented") return CLIPS_DATA.aug_panels || {};
    if (tabType === "frozen") return CLIPS_DATA.frozen_panels || {};
    if (tabType === "mined") return CLIPS_DATA.mined_panels || {};
    return {};
}

function classBlockHtml(tabType, className, clips) {
    const label = tabType === "augmented" ? "augmented clip" : tabType === "frozen" ? "test clip" : tabType === "mined" ? "mined clip" : "clip";
    const selectAll = isInteractive ? `<input class="class-select-all" type="checkbox" aria-label="Select all in class">` : "";
    return `
      <details class="class-block" data-class="${escapeHtml(className)}">
        <summary>
          ${selectAll}
          ${escapeHtml(className)}
          <span class="count" data-label="${label}">${clips.length} ${label}${clips.length === 1 ? "" : "s"}</span>
        </summary>
        <div class="clips" data-class-clips="${escapeHtml(className)}"></div>
      </details>
    `;
}

function rebuildReviewTab(tabType) {
    const panel = document.getElementById("tab-" + tabType);
    if (!panel) return;
    const openClasses = new Set(
        Array.from(panel.querySelectorAll("details.class-block[open]"))
            .map(el => el.dataset.class || "")
            .filter(Boolean)
    );
    panel.querySelectorAll("details.class-block, .dynamic-empty").forEach(el => el.remove());
    panel.querySelectorAll(":scope > .notice").forEach(el => el.remove());
    const data = panelDataForTab(tabType);
    const classes = Object.keys(data).sort();
    if (!classes.length) {
        if (tabType === "frozen") {
            const ctx = CLIPS_DATA.frozen_context || {};
            if (ctx.stale) {
                panel.insertAdjacentHTML("beforeend", `<div class="notice dynamic-empty">${escapeHtml(ctx.stale_reason || "Frozen test selection changed.")}<br>Click <b>Compose dataset</b> (top right) to rebuild the frozen test now.</div>`);
            } else {
                panel.insertAdjacentHTML("beforeend", `<div class="notice dynamic-empty">No composed frozen test set yet. Assign source rules with split <code>test</code>, then click <b>Compose dataset</b> (top right) — it writes <code>projects/${escapeHtml(projectId)}/dataset/test.csv</code> and this tab fills in.</div>`);
            }
        } else {
            panel.insertAdjacentHTML("beforeend", `<div class="notice dynamic-empty">No ${tabType} clips are available yet.</div>`);
        }
        return;
    }
    if (tabType === "frozen" && (CLIPS_DATA.frozen_context || {}).pending_compose) {
        panel.insertAdjacentHTML("beforeend", `<div class="notice dynamic-empty">Pending frozen test preview from the current source rules. Click <b>Compose dataset</b> (top right) to write <code>test.csv</code> before evaluation.</div>`);
    }
    panel.insertAdjacentHTML("beforeend", classes.map(cls => classBlockHtml(tabType, cls, data[cls] || [])).join(""));
    panel.querySelectorAll("details.class-block").forEach(details => {
        if (openClasses.has(details.dataset.class || "")) details.open = true;
    });
    classes.forEach(cls => renderClassClips(tabType, cls));
}

function resetSelectOptions(selectEl) {
    if (!selectEl) return;
    const previousValue = selectEl.value || "";
    const first = selectEl.querySelector("option[value='']");
    selectEl.innerHTML = "";
    selectEl.appendChild(first || Object.assign(document.createElement("option"), { value: "", textContent: "all" }));
    if (previousValue) selectEl.dataset.restoreValue = previousValue;
    delete selectEl.dataset.populated;
}

function restoreSelectValue(selectEl) {
    if (!selectEl || selectEl.dataset.restoreValue === undefined) return;
    const previousValue = selectEl.dataset.restoreValue;
    const exists = Array.from(selectEl.options).some(opt => opt.value === previousValue);
    selectEl.value = exists ? previousValue : "";
    delete selectEl.dataset.restoreValue;
}

function resetReviewFilterOptions() {
    resetSelectOptions(document.getElementById("library-label-filter"));
    resetSelectOptions(document.getElementById("library-density-filter"));
    resetSelectOptions(document.getElementById("library-project-filter"));
    resetSelectOptions(document.getElementById("library-role-filter"));
    resetSelectOptions(document.getElementById("library-bucket-filter"));
    resetSelectOptions(document.getElementById("library-mic-filter"));
    resetSelectOptions(document.getElementById("library-tag-filter"));
    resetSelectOptions(augTagSelect);
}

function refreshRuleCountsFromLibrary(rules) {
    asArray(rules).forEach(rule => {
        if (rule && typeof rule === "object") {
            rule.count = libraryItemsMatchingRule(rule).length;
        }
    });
}

function countUniqueLibraryPathsForRules(rules) {
    const paths = new Set();
    asArray(rules).forEach(rule => {
        libraryItemsMatchingRule(rule || {}).forEach(c => paths.add(c.path));
    });
    return paths.size;
}

function refreshSelectionCountsFromLibrary() {
    const assignments = selectionState.assignments || {};
    Object.keys(assignments).forEach(className => {
        refreshRuleCountsFromLibrary(assignments[className]);
    });
    refreshRuleCountsFromLibrary(recipeBaseRules);
    refreshRuleCountsFromLibrary(recipeOverlayRules);
    const recipes = Array.isArray(selectionState.augmentation_recipes) ? selectionState.augmentation_recipes : [];
    recipes.forEach(recipe => {
        const baseFilters = recipe.base_filters || (recipe.base_filter ? [recipe.base_filter] : []);
        const overlayFilters = recipe.overlay_filters || (recipe.overlay_filter ? [recipe.overlay_filter] : []);
        const settings = (recipe.settings && typeof recipe.settings === "object") ? recipe.settings : {};
        refreshRuleCountsFromLibrary(baseFilters);
        recipe.base_count = countUniqueLibraryPathsForRules(baseFilters);
        if (settings.overlay_enabled === false) {
            recipe.overlay_count = 0;
        } else {
            refreshRuleCountsFromLibrary(overlayFilters);
            recipe.overlay_count = countUniqueLibraryPathsForRules(overlayFilters);
        }
    });
    updateRecipeBucketSummary();
}

function clearMissingPaginationState() {
    const valid = new Set();
    ["dataset", "augmented", "frozen", "mined"].forEach(tabType => {
        Object.keys(panelDataForTab(tabType)).forEach(className => valid.add(`${tabType}:${className}`));
    });
    Object.keys(PAGINATION_STATE).forEach(key => {
        if (!valid.has(key)) delete PAGINATION_STATE[key];
    });
}

function removeDeletedPathsFromPanelMap(panelMap, deletedSet) {
    Object.keys(panelMap || {}).forEach(className => {
        panelMap[className] = (panelMap[className] || []).filter(c => !deletedSet.has(c.path));
        if (!panelMap[className].length) delete panelMap[className];
    });
}

function removeDeletedPathsFromReviewData(deletedSet) {
    CLIPS_DATA.library_items = (CLIPS_DATA.library_items || []).filter(c => !deletedSet.has(c.path));
    removeDeletedPathsFromPanelMap(CLIPS_DATA.panels || {}, deletedSet);
    removeDeletedPathsFromPanelMap(CLIPS_DATA.aug_panels || {}, deletedSet);
    removeDeletedPathsFromPanelMap(CLIPS_DATA.frozen_panels || {}, deletedSet);
    removeDeletedPathsFromPanelMap(CLIPS_DATA.mined_panels || {}, deletedSet);
}

function renderReviewData() {
    resetReviewFilterOptions();
    populateLibraryFilters();
    populateAugmentedTagFilter();
    ["dataset", "augmented", "frozen", "mined"].forEach(rebuildReviewTab);
    renderLibraryTable();
    refreshSelectionCountsFromLibrary();
    renderSelectionRules();
    renderAugmentationRules();
    renderExistingAugmentations();
    updateLiveCounts();
    refreshCount();
    updateMutatingBusy();
}

function applyReviewDataPayload(data, options = {}) {
    // Merge, don't overwrite: a scoped rescan returns only the requested keys,
    // so only replace what the payload actually carries — otherwise refreshing
    // one tab would wipe the others.
    if (data.panels !== undefined) CLIPS_DATA.panels = data.panels || {};
    if (data.aug_panels !== undefined) CLIPS_DATA.aug_panels = data.aug_panels || {};
    if (data.frozen_panels !== undefined) CLIPS_DATA.frozen_panels = data.frozen_panels || {};
    if (data.frozen_context !== undefined) CLIPS_DATA.frozen_context = data.frozen_context || {};
    if (data.mined_panels !== undefined) CLIPS_DATA.mined_panels = data.mined_panels || {};
    if (data.library_items !== undefined) CLIPS_DATA.library_items = data.library_items || [];
    if (data.augmentation_generated_counts !== undefined) {
        augmentationGeneratedCounts = data.augmentation_generated_counts || {};
        augmentationGeneratedCountsLoaded = true;
        augmentationGeneratedCountsError = "";
    }
    if (data.panel_totals && typeof data.panel_totals === "object") {
        Object.keys(data.panel_totals).forEach(tab => {
            panelTotals[tab] = data.panel_totals[tab] || {};
        });
    }
    clearMissingPaginationState();
    if (options.render !== false) renderReviewData();
}

async function refreshReviewData(options = {}) {
    if (activeRefreshPromise) return activeRefreshPromise;
    activeRefreshPromise = refreshReviewDataJob(options).finally(() => {
        activeRefreshPromise = null;
    });
    return activeRefreshPromise;
}

function refreshSummaryText(counts) {
    const c = counts || {};
    return `applied: ${Number(c.dataset || 0)} dataset, ${Number(c.augmented || 0)} augmented, ${Number(c.frozen || 0)} frozen, ${Number(c.mined || 0)} mined, ${Number(c.library || 0)} library`;
}

function refreshStatusText(job) {
    if ((job.state || "") === "finished") return "applying changes...";
    const phase = job.phase_detail || job.phase || job.state || "applying";
    const count = Number(job.phase_count || 0);
    const pct = Number(job.progress || 0);
    const pctText = pct > 0 ? ` · ${Math.max(0, Math.min(99, Math.round(pct * 100)))}%` : "";
    const eta = Number(job.eta_s);
    const etaText = Number.isFinite(eta) && eta > 0 ? ` · ~${Math.max(1, Math.round(eta))}s left` : "";
    const countText = count ? ` · ${count}` : "";
    return `Applying · ${phase}${pctText}${etaText}${countText}`;
}

// Convenience: rescan only the named scope(s), e.g. "frozen" or "dataset,recipes".
function rescanScopes(scopes, opts = {}) {
    return refreshReviewData(Object.assign({ scopes }, opts));
}

function composeSummaryText(summary) {
    const s = summary || {};
    const splits = s.splits || {};
    const train = Number(splits.train || 0);
    const val = Number(splits.val || 0);
    const test = Number((s.frozen_test || {}).total ?? splits.test ?? 0);
    if (train || val || test) return `composed: ${train} train, ${val} val, ${test} frozen test`;
    return "dataset composed";
}

// Rebuild train/val/test.csv from the current selection, then refresh the tabs
// it affects so new dataset rows + frozen-test clips show without leaving the page.
async function composeDataset() {
    if (composeRunning || refreshRunning) return;
    const btn = document.getElementById("compose-dataset-btn");
    composeRunning = true;
    updateMutatingBusy();
    if (btn) { if (!btn.dataset.originalText) btn.dataset.originalText = btn.textContent; btn.disabled = true; btn.textContent = "Composing..."; }
    setBackgroundJobStatus("Composing dataset...");
    try {
        const start = await fetch("/api/compose/start", { method: "POST", cache: "no-store" });
        const started = await start.json();
        if (!start.ok || !started.ok) throw new Error(started.error || `HTTP ${start.status}`);
        let jobId = (started.job || {}).job_id || "";
        if (!jobId) throw new Error("compose job did not start");
        let finalJob = started.job || {};
        while (jobId) {
            await sleep(700);
            const r = await fetch(`/api/compose/status?job_id=${encodeURIComponent(jobId)}`, { cache: "no-store" });
            const data = await r.json();
            if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
            finalJob = data.job || {};
            if (["finished", "error"].includes(finalJob.state)) break;
        }
        if (finalJob.state === "error") throw new Error(finalJob.error || "compose failed");
        flash(composeSummaryText(finalJob.summary), "ok");
        setBackgroundJobStatus("Refreshing dataset + frozen test...");
        // The composed CSVs changed dataset/frozen membership and recipe counts.
        await rescanScopes("dataset,frozen,recipes", { silent: true, label: "dataset + frozen" });
    } catch (e) {
        setBackgroundJobStatus(`compose failed: ${e.message || e}`, "err");
        flash(`compose failed: ${e.message || e}`, "err");
    } finally {
        composeRunning = false;
        if (btn) { btn.disabled = false; btn.textContent = btn.dataset.originalText || "Compose dataset ⚙"; }
        updateMutatingBusy();
        if (!refreshRunning && !deleteRunning && !augmentationRunning) setTimeout(() => setBackgroundJobStatus(""), 1500);
    }
}

function setRefreshBtnBusy(btn, busy, busyText) {
    if (!btn) return;
    if (busy) {
        if (!btn.dataset.originalText) btn.dataset.originalText = btn.textContent || "";
        btn.disabled = true;
        if (btn.classList.contains("section-refresh")) btn.classList.add("spinning");
        else btn.textContent = busyText || "Rescanning...";
    } else {
        btn.disabled = false;
        btn.classList.remove("spinning");
        if (!btn.classList.contains("section-refresh")) {
            btn.textContent = btn.dataset.originalText || "Rescan all";
        }
    }
}

async function refreshReviewDataJob(options = {}) {
    const silent = !!(options && options.silent);
    const scopes = options.scopes || "";              // "" => full rescan of every tab
    const scopeLabel = options.label || (scopes ? String(scopes) : "all");
    const seq = ++refreshRequestSeq;
    const btn = options.button || document.getElementById("library-refresh-review");
    const status = options.statusEl || document.getElementById("review-refresh-status");
    refreshRunning = true;
    updateMutatingBusy();
    setRefreshBtnBusy(btn, true);
    setBackgroundJobStatus(`Rescanning ${scopeLabel}...`);
    if (status) status.textContent = "scanning...";
    try {
        const start = await fetch("/api/review-data/start", {
            method: "POST", cache: "no-store",
            headers: { "Content-Type": "application/json" },
            body: scopes ? JSON.stringify({ scopes }) : "{}",
        });
        const started = await start.json();
        if (!start.ok || !started.ok) throw new Error(started.error || `HTTP ${start.status}`);
        refreshJobId = (started.job || {}).job_id || "";
        if (!refreshJobId) throw new Error("rescan job did not start");
        let finalJob = started.job || {};
        while (refreshJobId) {
            await sleep(650);
            const r = await fetch(`/api/review-data/status?job_id=${encodeURIComponent(refreshJobId)}`, { cache: "no-store" });
            const data = await r.json();
            if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
            const job = data.job || {};
            finalJob = job;
            if (seq !== refreshRequestSeq) return;
            // Full "Applying · …" progress lives only on the top job bar; the
            // inline refresh status keeps a short state so the line isn't shown twice.
            setBackgroundJobStatus(refreshStatusText(job));
            if (status) status.textContent = "scanning...";
            if (["finished", "error"].includes(job.state)) break;
        }
        if (finalJob.state === "error") throw new Error(finalJob.error || "rescan failed");
        const data = finalJob.payload || {};
        if (!data.ok) throw new Error(data.error || "rescan returned no payload");
        if (seq !== refreshRequestSeq) return;
        setBackgroundJobStatus("Rendering changes...");
        if (status) status.textContent = "rendering...";
        applyReviewDataPayload(data, { render: false });
        await loadSelectionState();
        if (seq !== refreshRequestSeq) return;
        renderReviewData();
        reviewDirty = false;
        const summary = refreshSummaryText(data.summary_counts || finalJob.summary_counts || {});
        if (status) status.textContent = scopes ? "updated" : summary;
        setBackgroundJobStatus("");
        if (!silent) flash(scopes ? `rescanned ${scopeLabel}` : summary, "ok");
    } catch (e) {
        if (seq === refreshRequestSeq) {
            if (status) status.textContent = "failed";
            setBackgroundJobStatus(`rescan failed: ${e.message || e}`, "err");
            if (!silent) flash(`rescan failed: ${e.message || e}`, "err");
        }
    } finally {
        refreshJobId = "";
        refreshRunning = false;
        if (seq === refreshRequestSeq) setRefreshBtnBusy(btn, false);
        updateMutatingBusy();
        if (seq === refreshRequestSeq && !deleteRunning && !augmentationRunning) setTimeout(() => setBackgroundJobStatus(""), 1200);
    }
}

function renderClassClips(tabType, className, append = false) {
    const panelId = "tab-" + tabType;
    const panel = document.getElementById(panelId);
    if (!panel) return;

    const container = panel.querySelector(`.clips[data-class-clips="${cssEscape(className)}"]`);
    if (!container) return;

    let clips = [];
    if (tabType === "dataset") clips = CLIPS_DATA.panels[className] || [];
    else if (tabType === "augmented") clips = CLIPS_DATA.aug_panels[className] || [];
    else if (tabType === "frozen") clips = CLIPS_DATA.frozen_panels[className] || [];
    else if (tabType === "mined") clips = CLIPS_DATA.mined_panels[className] || [];

    const filtered = filterClipsList(clips, tabType);
    const key = `${tabType}:${className}`;
    if (!append) {
        PAGINATION_STATE[key] = 100;
        container.innerHTML = "";
    }

    const limit = PAGINATION_STATE[key];
    const pageClips = filtered.slice(append ? limit - 100 : 0, limit);

    let html = pageClips.map(c => renderCard(c, tabType)).join("");

    const existingButton = container.querySelector(".load-more-btn");
    if (existingButton) existingButton.remove();

    if (append) {
        const temp = document.createElement("div");
        temp.innerHTML = html;
        while (temp.firstChild) {
            container.appendChild(temp.firstChild);
        }
    } else {
        container.innerHTML = html;
    }

    const trueTotal = classTrueTotal(tabType, className);
    const moreInMemory = filtered.length > limit;
    const moreOnServer = clips.length < trueTotal && tabType !== "library";
    if (moreInMemory || moreOnServer) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "load-more-btn";
        if (moreInMemory) {
            btn.textContent = `Load More (${filtered.length - limit} more loaded)`;
            btn.addEventListener("click", (ev) => {
                ev.stopPropagation();
                PAGINATION_STATE[key] += 100;
                renderClassClips(tabType, className, true);
            });
        } else {
            // All loaded cards are shown but more exist on disk beyond the cap.
            btn.textContent = `Load More (${trueTotal - clips.length} more on disk)`;
            btn.addEventListener("click", async (ev) => {
                ev.stopPropagation();
                btn.disabled = true;
                btn.textContent = "Loading...";
                const added = await fetchMoreClips(tabType, className);
                if (added > 0) {
                    PAGINATION_STATE[key] += 100;
                    renderClassClips(tabType, className, true);
                } else {
                    btn.remove();
                }
            });
        }
        container.appendChild(btn);
    }

    bindCardEvents(container, tabType);
    const details = container.closest(".class-block");
    if (details) updateClassSelectAllStatus(details);
}

// Pull the next page of cards for one class from the server (beyond the cap),
// append them to the in-memory panel, and return how many new clips were added.
async function fetchMoreClips(tabType, className) {
    const data = panelDataForTab(tabType);
    const arr = data[className] || (data[className] = []);
    const offset = arr.length;
    try {
        const params = new URLSearchParams({
            tab: tabType, "class": className, offset: String(offset), limit: "200",
        });
        const r = await fetch(`/api/clips?${params.toString()}`, { cache: "no-store" });
        const j = await r.json();
        if (!r.ok || !j.ok) throw new Error(j.error || `HTTP ${r.status}`);
        const incoming = Array.isArray(j.clips) ? j.clips : [];
        const seen = new Set(arr.map(c => c.path));
        let added = 0;
        incoming.forEach(c => { if (!seen.has(c.path)) { arr.push(c); added++; } });
        return added;
    } catch (e) {
        flash(`load more failed: ${e.message || e}`, "err");
        return 0;
    }
}

function bindCardEvents(container, tabType) {
    container.querySelectorAll(".clip").forEach(card => {
        if (card.dataset.eventsBound === tabType) return;
        const path = card.dataset.path;
        let clipObj = findClipObject(path, tabType);
        if (!clipObj) return;
        card.dataset.eventsBound = tabType;

        const box = card.querySelector(".select-box");
        if (box) {
            box.addEventListener("click", ev => ev.stopPropagation());
            box.addEventListener("change", ev => {
                const checked = ev.target.checked;
                clipObj.soft_excluded = checked;
                card.classList.toggle("selected", checked);
                refreshCount();
                const details = card.closest(".class-block");
                if (details) updateClassSelectAllStatus(details);
                lastClickedCard = card;
            });
        }

        const img = card.querySelector(".play-spec");
        if (img) {
            img.addEventListener("click", ev => {
                ev.stopPropagation();
                const audio = card.querySelector(".hidden-audio");
                if (!audio) return;

                document.querySelectorAll(".clip.playing").forEach(other => {
                    if (other === card) return;
                    const a = other.querySelector(".hidden-audio");
                    if (a) a.pause();
                    other.classList.remove("playing");
                });

                if (audio.paused) {
                    audio.currentTime = 0;
                    audio.play().then(() => card.classList.add("playing"))
                                .catch(() => {});
                } else {
                    audio.pause();
                    card.classList.remove("playing");
                }
            });
        }

        const audio = card.querySelector(".hidden-audio");
        if (audio) {
            audio.addEventListener("ended", () => card.classList.remove("playing"));
        }

        const body = card.querySelector(".clip-body");
        if (body) {
            body.addEventListener("click", ev => {
                if (ev.target.closest("audio") || ev.target.closest(".play-spec") || ev.target.closest("a")) return;

                if (ev.shiftKey && lastClickedCard && lastClickedCard !== card) {
                    const parentBlock = card.closest(".class-block");
                    if (parentBlock && lastClickedCard.closest(".class-block") === parentBlock) {
                        const allRenderedCards = Array.from(parentBlock.querySelectorAll(".clip"));
                        const idx1 = allRenderedCards.indexOf(lastClickedCard);
                        const idx2 = allRenderedCards.indexOf(card);

                        if (idx1 !== -1 && idx2 !== -1) {
                            const start = Math.min(idx1, idx2);
                            const end = Math.max(idx1, idx2);
                            const targetState = !clipObj.soft_excluded;

                            for (let i = start; i <= end; i++) {
                                const targetCard = allRenderedCards[i];
                                if (!targetCard.classList.contains("deleted")) {
                                    const targetPath = targetCard.dataset.path;
                                    const targetObj = findClipObject(targetPath, tabType);
                                    if (targetObj) {
                                        targetObj.soft_excluded = targetState;
                                        targetCard.classList.toggle("selected", targetState);
                                        const subBox = targetCard.querySelector(".select-box");
                                        if (subBox) subBox.checked = targetState;
                                    }
                                }
                            }
                            refreshCount();
                            updateClassSelectAllStatus(parentBlock);
                            lastClickedCard = card;
                            return;
                        }
                    }
                }

                const checked = !clipObj.soft_excluded;
                clipObj.soft_excluded = checked;
                card.classList.toggle("selected", checked);
                if (box) box.checked = checked;
                refreshCount();
                const details = card.closest(".class-block");
                if (details) updateClassSelectAllStatus(details);

                lastClickedCard = card;
            });
        }
    });
}

function refreshCount() {
    let count = 0;
    (CLIPS_DATA.library_items || []).forEach(c => { if (c.selected && !c.deleted) count++; });
    Object.values(CLIPS_DATA.panels).flat().forEach(c => { if (c.soft_excluded && !c.deleted) count++; });
    Object.values(CLIPS_DATA.aug_panels).flat().forEach(c => { if (c.soft_excluded && !c.deleted) count++; });
    Object.values(CLIPS_DATA.frozen_panels).flat().forEach(c => { if (c.soft_excluded && !c.deleted) count++; });
    Object.values(CLIPS_DATA.mined_panels).flat().forEach(c => { if (c.soft_excluded && !c.deleted) count++; });
    if (countEl) countEl.textContent = count;
}

// True per-class total (uncapped) when known, else the loaded card count.
function classTrueTotal(tabType, className) {
    const loaded = (panelDataForTab(tabType)[className] || []).length;
    const t = panelTotals[tabType];
    if (t && t[className] != null) return Math.max(Number(t[className]) || 0, loaded);
    return loaded;
}

function updateLiveCounts() {
    document.querySelectorAll(".tabs button[data-tab]").forEach(btn => {
        const tab = btn.dataset.tab;
        const countSpan = btn.querySelector(".count");
        if (!countSpan) return;
        if (tab === "library") {
            countSpan.textContent = `(${libraryFilteredItems().length})`;
            return;
        }
        const data = panelDataForTab(tab);
        let loaded = 0, filtered = 0, trueTotal = 0;
        Object.keys(data).forEach(cls => {
            const clips = data[cls] || [];
            loaded += clips.length;
            filtered += filterClipsList(clips, tab).length;
            trueTotal += classTrueTotal(tab, cls);
        });
        // Under an active filter the uncapped total is unknown, so show the
        // filtered (loaded) count; otherwise show the real, uncapped total.
        const isFiltered = filtered !== loaded;
        countSpan.textContent = `(${isFiltered ? filtered : trueTotal})`;
    });

    document.querySelectorAll("details.class-block").forEach(details => {
        const tabType = details.closest(".tab-panel").id.replace("tab-", "");
        const className = details.dataset.class;
        const countSpan = details.querySelector("summary .count");
        if (!countSpan) return;

        const clips = panelDataForTab(tabType)[className] || [];
        const loaded = clips.length;
        const total = classTrueTotal(tabType, className);
        const filtered = filterClipsList(clips, tabType).length;
        const label = countSpan.dataset.label || "clip";
        const plural = (n) => `${label}${n === 1 ? "" : "s"}`;
        let text;
        if (filtered !== loaded) {
            text = `${filtered} of ${total} ${plural(total)}`;        // filter/search active
        } else if (total > loaded) {
            text = `${loaded} of ${total} ${plural(total)}`;          // more on disk than loaded
        } else {
            text = `${total} ${plural(total)}`;                       // fully loaded, unfiltered
        }
        countSpan.textContent = text;

        details.style.display = filtered === 0 ? "none" : "";
    });
}

function updateClassSelectAllStatus(details) {
    const mainBox = details.querySelector(".class-select-all");
    if (!mainBox) return;
    const tabType = details.closest(".tab-panel").id.replace("tab-", "");
    const className = details.dataset.class;

    let clips = [];
    if (tabType === "dataset") clips = CLIPS_DATA.panels[className] || [];
    else if (tabType === "augmented") clips = CLIPS_DATA.aug_panels[className] || [];
    else if (tabType === "frozen") clips = CLIPS_DATA.frozen_panels[className] || [];
    else if (tabType === "mined") clips = CLIPS_DATA.mined_panels[className] || [];

    const filtered = filterClipsList(clips, tabType);
    if (filtered.length === 0) {
        mainBox.checked = false;
        mainBox.indeterminate = false;
        return;
    }

    const selectedCount = filtered.filter(c => c.soft_excluded).length;
    if (selectedCount === 0) {
        mainBox.checked = false;
        mainBox.indeterminate = false;
    } else if (selectedCount === filtered.length) {
        mainBox.checked = true;
        mainBox.indeterminate = false;
    } else {
        mainBox.checked = false;
        mainBox.indeterminate = true;
    }
}

function updateAllClassSelectAllStatus() {
    document.querySelectorAll("details.class-block").forEach(d => {
        updateClassSelectAllStatus(d);
    });
}

function applyFilters(tabType) {
    if (tabType === "library") {
        populateLibraryFilters();
        renderLibraryTable();
    } else if (tabType === "dataset") {
        Object.keys(CLIPS_DATA.panels).forEach(cls => renderClassClips("dataset", cls));
    } else if (tabType === "augmented") {
        Object.keys(CLIPS_DATA.aug_panels).forEach(cls => renderClassClips("augmented", cls));
    } else if (tabType === "frozen") {
        Object.keys(CLIPS_DATA.frozen_panels).forEach(cls => renderClassClips("frozen", cls));
    } else if (tabType === "mined") {
        Object.keys(CLIPS_DATA.mined_panels).forEach(cls => renderClassClips("mined", cls));
    }
    updateLiveCounts();
}

function flash(msg, cls) {
    if (!statusEl) return;
    statusEl.textContent = msg;
    statusEl.className = "status " + (cls || "");
    if (cls === "ok") setTimeout(() => { statusEl.textContent = ""; statusEl.className = "status"; }, 1500);
}

// Expand / collapse all class blocks in a tab
document.querySelectorAll(".expand-all").forEach(btn => {
    btn.addEventListener("click", () => {
        const panel = document.getElementById(btn.dataset.target);
        const open = btn.dataset.action === "open";
        panel.querySelectorAll("details.class-block").forEach(d => d.open = open);
    });
});

// Single smooth hover tooltip
const hoverTip = document.getElementById("hover-tip");
let tipTimer = null;
function placeTip(ev) {
    if (!hoverTip) return;
    const pad = 12;
    hoverTip.style.left = "0px";
    hoverTip.style.top = "0px";
    const rect = hoverTip.getBoundingClientRect();
    let x = ev.clientX + 14;
    let y = ev.clientY + 14;
    if (x + rect.width + pad > window.innerWidth) x = ev.clientX - rect.width - 14;
    if (y + rect.height + pad > window.innerHeight) y = ev.clientY - rect.height - 14;
    hoverTip.style.left = Math.max(pad, x) + "px";
    hoverTip.style.top = Math.max(pad, y) + "px";
}
document.addEventListener("mouseover", ev => {
    const el = ev.target.closest("[data-tip]");
    if (!el) return;
    clearTimeout(tipTimer);
    hoverTip.textContent = el.dataset.tip || "";
    placeTip(ev);
    tipTimer = setTimeout(() => hoverTip.classList.add("show"), 120);
});
document.addEventListener("mousemove", ev => {
    const el = ev.target.closest("[data-tip]");
    if (el) placeTip(ev);
});
document.addEventListener("mouseout", ev => {
    const el = ev.target.closest("[data-tip]");
    if (el) {
        clearTimeout(tipTimer);
        hoverTip.classList.remove("show");
    }
});

// --- Tabs Switch -----------------------------------------------------------
document.querySelectorAll(".tabs button").forEach(btn => {
    btn.addEventListener("click", () => {
        const id = btn.dataset.tab;
        document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("active", b === btn));
        document.querySelectorAll(".tab-panel").forEach(p => {
            p.classList.toggle("active", p.id === "tab-" + id);
        });
        document.body.classList.toggle("library-active", id === "library");
    });
});

// --- Dataset search event --------------------------------------------------
populateLibraryFilters();
populateAugmentedTagFilter();
if (librarySearchInput) librarySearchInput.addEventListener("input", () => applyFilters("library"));
if (librarySessionInput) librarySessionInput.addEventListener("input", () => applyFilters("library"));
const libraryClearFilters = document.getElementById("library-clear-filters");
if (libraryClearFilters) libraryClearFilters.addEventListener("click", clearLibraryFilters);
const sourceRulesClear = document.getElementById("source-rules-clear");
if (sourceRulesClear) sourceRulesClear.addEventListener("click", clearAllSelectionRules);
const sourceRulesSearch = document.getElementById("source-rules-search");
if (sourceRulesSearch) sourceRulesSearch.addEventListener("input", () => {
    sourceRuleQuery = sourceRulesSearch.value || "";
    renderSelectionRules();
});
libraryFilterIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", () => applyFilters("library"));
});
const libraryAssignClass = document.getElementById("library-assign-class");
if (libraryAssignClass) {
    libraryAssignClass.addEventListener("click", () => addSelectionRule());
}
const libraryDeleteSelected = document.getElementById("library-delete-selected");
if (libraryDeleteSelected) {
    libraryDeleteSelected.addEventListener("click", () => deleteSelectedLibraryRows());
}
const libraryDeleteSmall = document.getElementById("library-delete-small");
if (libraryDeleteSmall) {
    libraryDeleteSmall.addEventListener("click", () => deleteSmallLibraryBuckets());
}
const libraryRefreshReview = document.getElementById("library-refresh-review");
if (libraryRefreshReview) {
    libraryRefreshReview.addEventListener("click", () => refreshReviewData());
}
// Per-section "Rescan this view" (↻) buttons: scoped, fast refresh of one tab.
document.querySelectorAll(".section-refresh[data-refresh-scopes]").forEach(btn => {
    btn.addEventListener("click", () => {
        if (refreshRunning || composeRunning) return;
        const scopes = btn.dataset.refreshScopes || "";
        const statusEl = btn.dataset.refreshStatus ? document.getElementById(btn.dataset.refreshStatus) : null;
        rescanScopes(scopes, { button: btn, statusEl, label: btn.dataset.refreshLabel || scopes });
    });
});
const composeDatasetBtn = document.getElementById("compose-dataset-btn");
if (composeDatasetBtn) {
    composeDatasetBtn.addEventListener("click", () => composeDataset());
}
// "Save as project defaults" button is gone — panel state autosaves on every
// edit (debounced) and each Add recipe takes a fresh snapshot of these knobs.
document.querySelectorAll(".augment-settings-panel input").forEach(input => {
    input.addEventListener("input", () => {
        markAugmentationSettingsDirty();
        updateAugmentationSettingState();
        renderAugmentationRules();
    });
    input.addEventListener("change", () => {
        markAugmentationSettingsDirty();
        updateAugmentationSettingState();
        renderAugmentationRules();
    });
});
const recipeUseBase = document.getElementById("recipe-use-base");
if (recipeUseBase) recipeUseBase.addEventListener("click", () => setRecipeBucket("base"));
const recipeUseOverlay = document.getElementById("recipe-use-overlay");
if (recipeUseOverlay) recipeUseOverlay.addEventListener("click", () => setRecipeBucket("overlay"));
const recipeAdd = document.getElementById("recipe-add");
if (recipeAdd) recipeAdd.addEventListener("click", addAugmentationRule);
["recipe-description", "recipe-tags"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", () => {
        if (el.value) el.dataset.userEdited = "1";
        else delete el.dataset.userEdited;
    });
});
const augmentGenerate = document.getElementById("augment-generate");
if (augmentGenerate) {
    augmentGenerate.addEventListener("click", generateAugmentations);
}
const augmentCancel = document.getElementById("augment-cancel");
if (augmentCancel) {
    augmentCancel.addEventListener("click", cancelAugmentation);
}

if (datasetSearchInput) {
    datasetSearchInput.addEventListener("input", () => applyFilters("dataset"));
}
const datasetSearchClear = document.getElementById("dataset-search-clear");
if (datasetSearchClear && datasetSearchInput) {
    datasetSearchClear.addEventListener("click", () => {
        datasetSearchInput.value = "";
        applyFilters("dataset");
    });
}

// --- Augmented filters events ----------------------------------------------
document.querySelectorAll(".aug-filter").forEach(btn => {
    btn.addEventListener("click", () => {
        activeAugFilter = btn.dataset.technique;
        document.querySelectorAll(".aug-filter").forEach(b => b.classList.toggle("active", b === btn));
        applyFilters("augmented");
    });
});
if (augSearchInput) augSearchInput.addEventListener("input", () => applyFilters("augmented"));
if (augTagSelect) augTagSelect.addEventListener("change", () => applyFilters("augmented"));
if (maskMetricSelect) maskMetricSelect.addEventListener("change", () => applyFilters("augmented"));
if (maskMinInput) maskMinInput.addEventListener("input", () => applyFilters("augmented"));
if (maskMaxInput) maskMaxInput.addEventListener("input", () => applyFilters("augmented"));
const maskClear = document.getElementById("mask-clear");
if (maskClear) {
    maskClear.addEventListener("click", () => {
        if (augSearchInput) augSearchInput.value = "";
        if (augTagSelect) augTagSelect.value = "";
        if (maskMetricSelect) maskMetricSelect.value = "masking_score_db";
        if (maskMinInput) maskMinInput.value = "";
        if (maskMaxInput) maskMaxInput.value = "";
        applyFilters("augmented");
    });
}
const mixViewToggle = document.getElementById("mix-view-toggle");
if (mixViewToggle) {
    mixViewToggle.addEventListener("click", () => {
        mixViewActive = !mixViewActive;
        mixViewToggle.classList.toggle("active", mixViewActive);
        applyFilters("augmented");
    });
}

function selectedFrozenRunId() {
    if (frozenRunSelect && frozenRunSelect.value) return frozenRunSelect.value;
    return currentFrozenRunId || (CLIPS_DATA.frozen_context || {}).viewed_run_id || "";
}

function runPreviewKindLabel(kind) {
    return kind === "training" ? "training curves" : "confusion matrix";
}

function setRunPreviewKind(kind) {
    document.querySelectorAll("[data-switch-run-preview]").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.switchRunPreview === kind);
    });
}

function openFrozenRunPreview(kind) {
    const runId = selectedFrozenRunId();
    const modal = document.getElementById("frozen-run-preview-modal");
    const img = document.getElementById("frozen-run-preview-image");
    const body = document.getElementById("frozen-run-preview-body");
    const title = document.getElementById("frozen-run-preview-title");
    if (!modal || !img || !body || !title) return;
    if (!runId) {
        flash("no frozen-test run selected", "err");
        return;
    }
    const cleanKind = kind === "training" ? "training" : "confusion";
    setRunPreviewKind(cleanKind);
    title.textContent = `${runId} · ${runPreviewKindLabel(cleanKind)}`;
    body.querySelectorAll(".preview-error").forEach(el => el.remove());
    img.hidden = false;
    img.alt = `${runPreviewKindLabel(cleanKind)} for ${runId}`;
    img.src = `/api/frozen-run/image?run=${encodeURIComponent(runId)}&kind=${encodeURIComponent(cleanKind)}&t=${Date.now()}`;
    img.onerror = () => {
        img.hidden = true;
        body.querySelectorAll(".preview-error").forEach(el => el.remove());
        const err = document.createElement("div");
        err.className = "preview-error";
        err.textContent = `No ${runPreviewKindLabel(cleanKind)} image found for ${runId}.`;
        body.appendChild(err);
    };
    modal.hidden = false;
}

function closeFrozenRunPreview() {
    const modal = document.getElementById("frozen-run-preview-modal");
    const img = document.getElementById("frozen-run-preview-image");
    if (img) img.removeAttribute("src");
    if (modal) modal.hidden = true;
}

document.querySelectorAll("[data-open-run-preview]").forEach(btn => {
    btn.addEventListener("click", () => openFrozenRunPreview(btn.dataset.openRunPreview));
});
document.querySelectorAll("[data-switch-run-preview]").forEach(btn => {
    btn.addEventListener("click", () => openFrozenRunPreview(btn.dataset.switchRunPreview));
});
document.querySelectorAll("[data-close-run-preview]").forEach(el => {
    el.addEventListener("click", closeFrozenRunPreview);
});
document.addEventListener("keydown", ev => {
    if (ev.key === "Escape") closeFrozenRunPreview();
});

// --- Frozen filters events ------------------------------------------------ -
// Fetch a run's frozen-test predictions, merge them into the panels, and
// re-render. Shared by the run picker and the Evaluate button.
async function refreshFrozenRun(selectedRun, { statusBusyText = "loading predictions..." } = {}) {
    const statusSpan = document.getElementById("frozen-run-status");
    if (statusSpan) { statusSpan.textContent = statusBusyText; statusSpan.className = "busy"; }
    try {
        const url = new URL("/api/frozen-run", window.location.href);
        if (selectedRun) url.searchParams.set("run", selectedRun);
        const r = await fetch(url.toString(), { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const payload = await r.json();
        const predictions = payload.cards || {};
        currentFrozenRunId = payload.run_id || selectedRun || currentFrozenRunId;
        CLIPS_DATA.frozen_context = payload.context || CLIPS_DATA.frozen_context || {};

        // Merge into CLIPS_DATA.frozen_panels
        Object.keys(CLIPS_DATA.frozen_panels).forEach(cls => {
            CLIPS_DATA.frozen_panels[cls].forEach(c => {
                const pred = predictions[c.path] || {};
                c.correct = pred.correct !== undefined ? pred.correct : false;
                c.predicted_class = pred.predicted_class || "not_evaluated";
                c.triggered = pred.triggered !== undefined ? pred.triggered : false;
                c.predicted_confidence = pred.predicted_confidence;
                c.argmax_confidence = pred.argmax_confidence;
                c.margin = pred.margin;
                c.prob_html = pred.prob_html || "";
            });
        });

        const pageUrl = new URL(window.location.href);
        if (selectedRun) pageUrl.searchParams.set("run", selectedRun);
        else pageUrl.searchParams.delete("run");
        history.replaceState(null, "", pageUrl.toString());
        if (statusSpan) {
            statusSpan.textContent = `loaded ${payload.run_id || "selected/latest"}`;
            statusSpan.className = "ok";
        }
        applyFilters("frozen");
        return payload;
    } catch (e) {
        if (statusSpan) {
            statusSpan.textContent = "run load failed";
            statusSpan.className = "err";
        }
        throw e;
    }
}

if (frozenRunSelect) {
    frozenRunSelect.addEventListener("change", async () => {
        if (frozenRunSelect.disabled) return;
        frozenRunSelect.disabled = true;
        try { await refreshFrozenRun(frozenRunSelect.value || ""); }
        catch (e) { /* status already shown */ }
        finally { frozenRunSelect.disabled = false; }
    });
}

const frozenEvaluateBtn = document.getElementById("frozen-evaluate-btn");
if (frozenEvaluateBtn) {
    frozenEvaluateBtn.addEventListener("click", async () => {
        const statusSpan = document.getElementById("frozen-run-status");
        const runId = (frozenRunSelect && frozenRunSelect.value) || "";
        frozenEvaluateBtn.disabled = true;
        if (frozenRunSelect) frozenRunSelect.disabled = true;
        if (statusSpan) { statusSpan.textContent = "evaluating… (up to ~1 min)"; statusSpan.className = "busy"; }
        try {
            const r = await fetch("/api/evaluate-frozen", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ run_id: runId }),
                cache: "no-store",
            });
            const data = await r.json();
            if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
            const acc = data.accuracy != null ? `${(data.accuracy * 100).toFixed(1)}%` : "n/a";
            const f1 = data.f1_macro_targets != null ? Number(data.f1_macro_targets).toFixed(3) : "n/a";
            flash(`evaluated ${data.model_id}: acc ${acc} · F1(targets) ${f1} · ${data.n_samples ?? "?"} clips`, "ok");
            // Reload predictions for the evaluated run; the confusion matrix image is
            // re-fetched (cache-busted) next time the CM preview is opened.
            await refreshFrozenRun(runId, { statusBusyText: "loading new results..." });
        } catch (e) {
            if (statusSpan) { statusSpan.textContent = `evaluate failed: ${e.message || e}`; statusSpan.className = "err"; }
            flash(`evaluate failed: ${e.message || e}`, "err");
        } finally {
            frozenEvaluateBtn.disabled = false;
            if (frozenRunSelect) frozenRunSelect.disabled = false;
        }
    });
}
document.querySelectorAll(".frozen-filter").forEach(btn => {
    btn.addEventListener("click", () => {
        activeFrozenFilter = btn.dataset.frozen;
        document.querySelectorAll(".frozen-filter").forEach(b => b.classList.toggle("active", b === btn));
        applyFilters("frozen");
    });
});
if (frozenSearchInput) frozenSearchInput.addEventListener("input", () => applyFilters("frozen"));
if (frozenMetricSelect) frozenMetricSelect.addEventListener("change", () => applyFilters("frozen"));
if (frozenMinInput) frozenMinInput.addEventListener("input", () => applyFilters("frozen"));
if (frozenMaxInput) frozenMaxInput.addEventListener("input", () => applyFilters("frozen"));
const frozenClear = document.getElementById("frozen-clear");
if (frozenClear) {
    frozenClear.addEventListener("click", () => {
        if (frozenSearchInput) frozenSearchInput.value = "";
        if (frozenMetricSelect) frozenMetricSelect.value = "predicted-confidence";
        if (frozenMinInput) frozenMinInput.value = "";
        if (frozenMaxInput) frozenMaxInput.value = "";
        applyFilters("frozen");
    });
}

// --- Mined filters events --------------------------------------------------
document.querySelectorAll(".mined-filter").forEach(btn => {
    btn.addEventListener("click", () => {
        activeMinedFilter = btn.dataset.mined;
        document.querySelectorAll(".mined-filter").forEach(b => b.classList.toggle("active", b === btn));
        applyFilters("mined");
    });
});
if (minedSearchInput) minedSearchInput.addEventListener("input", () => applyFilters("mined"));
if (minedConfMinInput) minedConfMinInput.addEventListener("input", () => applyFilters("mined"));
const minedClear = document.getElementById("mined-clear");
if (minedClear) {
    minedClear.addEventListener("click", () => {
        if (minedSearchInput) minedSearchInput.value = "";
        if (minedConfMinInput) minedConfMinInput.value = "";
        applyFilters("mined");
    });
}

// --- Class Select All checkbox ---------------------------------------------
document.addEventListener("click", ev => {
    if (ev.target.closest(".class-select-all")) ev.stopPropagation();
});
document.addEventListener("change", ev => {
    if (!ev.target.matches(".class-select-all")) return;
    const details = ev.target.closest(".class-block");
    if (!details) return;
    const tabType = details.closest(".tab-panel").id.replace("tab-", "");
    const className = details.dataset.class;
    const checked = ev.target.checked;

    let clips = [];
    if (tabType === "dataset") clips = CLIPS_DATA.panels[className] || [];
    else if (tabType === "augmented") clips = CLIPS_DATA.aug_panels[className] || [];
    else if (tabType === "frozen") clips = CLIPS_DATA.frozen_panels[className] || [];
    else if (tabType === "mined") clips = CLIPS_DATA.mined_panels[className] || [];

    const filtered = filterClipsList(clips, tabType);
    filtered.forEach(c => {
        c.soft_excluded = checked;
    });

    details.querySelectorAll(".clip").forEach(card => {
        if (!card.classList.contains("deleted")) {
            card.classList.toggle("selected", checked);
            const subBox = card.querySelector(".select-box");
            if (subBox) subBox.checked = checked;
        }
    });

    refreshCount();
    updateClassSelectAllStatus(details);
});

// --- Delete selected button ------------------------------------------------
const delBtn = document.getElementById("del-btn");
if (delBtn) {
    delBtn.addEventListener("click", async () => {
        const selectedPaths = [];
        (CLIPS_DATA.library_items || []).forEach(c => { if (c.selected && !c.deleted) selectedPaths.push(c.path); });
        Object.values(CLIPS_DATA.panels).flat().forEach(c => { if (c.soft_excluded && !c.deleted) selectedPaths.push(c.path); });
        Object.values(CLIPS_DATA.aug_panels).flat().forEach(c => { if (c.soft_excluded && !c.deleted) selectedPaths.push(c.path); });
        Object.values(CLIPS_DATA.frozen_panels).flat().forEach(c => { if (c.soft_excluded && !c.deleted) selectedPaths.push(c.path); });
        Object.values(CLIPS_DATA.mined_panels).flat().forEach(c => { if (c.soft_excluded && !c.deleted) selectedPaths.push(c.path); });

        if (selectedPaths.length === 0) {
            alert("No clips are selected. Click clips to select them first.");
            return;
        }
        if (!confirm(`Permanently delete ${selectedPaths.length} clip(s) from disk?\n\nThis cannot be undone.`)) return;
        try {
            await deletePathsInBackground(selectedPaths, "selected clips");
        } catch (e) {
            flash(`delete failed: ${e.message || e}`, "err");
        }
    });
}

// --- Promote selected mined clips ------------------------------------------
const promoteMinedBtn = document.getElementById("promote-mined-btn");
if (promoteMinedBtn) {
    promoteMinedBtn.addEventListener("click", async () => {
        const selectedMined = [];
        Object.values(CLIPS_DATA.mined_panels).flat().forEach(c => { if (c.soft_excluded && !c.deleted) selectedMined.push(c.path); });

        if (selectedMined.length === 0) {
            alert("No mined clips are selected. Select mined clips first.");
            return;
        }
        const labelEl = document.getElementById("mined-label");
        const label = (labelEl && labelEl.value || "mined_false_positive").trim();
        if (!label) {
            alert("Choose a negative label first.");
            return;
        }
        if (!confirm(`Promote ${selectedMined.length} mined clip(s) to accepted library label ${label}?`)) return;
        promoteMinedBtn.disabled = true;
        flash("promoting…");
        try {
            const r = await fetch("/api/promote-mined", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ paths: selectedMined, label }),
            });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const data = await r.json();
            const movedSet = new Set((data.moved || []).map(x => x.src));

            selectedMined.forEach(path => {
                if (movedSet.has(path)) {
                    const c = findClipObject(path, "mined");
                    if (c) {
                        c.deleted = true;
                        c.soft_excluded = false;
                    }
                }
            });

            flash(`promoted ${data.count} clip(s)`, "ok");
            applyFilters("mined");
            markReviewDirty();
            refreshCount();
        } catch (e) {
            flash("promotion failed", "err");
        } finally {
            promoteMinedBtn.disabled = false;
        }
    });
}

// --- Done button -----------------------------------------------------------
const doneBtn = document.getElementById("done-btn");
const doneOverlay = document.getElementById("done-overlay");
if (doneBtn) {
    doneBtn.addEventListener("click", async () => {
        doneBtn.disabled = true;
        flash("finishing…");
        const excludePaths = [];
        Object.values(CLIPS_DATA.panels).flat().forEach(c => { if (c.soft_excluded && !c.deleted) excludePaths.push(c.path); });
        Object.values(CLIPS_DATA.aug_panels).flat().forEach(c => { if (c.soft_excluded && !c.deleted) excludePaths.push(c.path); });

        try {
            const r = await fetch("/api/done", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ exclude_paths: excludePaths }),
            });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
        } catch { /* server may close before responding */ }
        flash("done", "ok");
        doneOverlay.classList.add("show");
    });
}

// Initial Boot Rendering
if (isInteractive) {
    applyFilters("library");
    loadSelectionState();
    loadPanelTotals();
    updateAugmentationSettingState();
    updateRecipeBucketSummary();
    renderExistingAugmentations();
    applyFilters("dataset");
    applyFilters("augmented");
    applyFilters("frozen");
    applyFilters("mined");
    refreshCount();
} else {
    // Non-interactive fallback: just bind events for existing DOM cards
    document.querySelectorAll(".clip").forEach(card => {
        const img = card.querySelector(".play-spec");
        if (img) {
            img.addEventListener("click", ev => {
                ev.stopPropagation();
                const audio = card.querySelector(".hidden-audio");
                if (!audio) return;
                document.querySelectorAll(".clip.playing").forEach(other => {
                    if (other === card) return;
                    const a = other.querySelector(".hidden-audio");
                    if (a) a.pause();
                    other.classList.remove("playing");
                });
                if (audio.paused) {
                    audio.currentTime = 0;
                    audio.play().then(() => card.classList.add("playing")).catch(() => {});
                } else {
                    audio.pause();
                    card.classList.remove("playing");
                }
            });
        }
        const audio = card.querySelector(".hidden-audio");
        if (audio) {
            audio.addEventListener("ended", () => card.classList.remove("playing"));
        }
    });
}
</script>

</body>
</html>
"""
