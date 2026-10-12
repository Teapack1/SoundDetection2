"""Unified audio library shared across projects.

Raw recordings live under ``data_library/sounds/<label>/``. Runtime-mined
clips live under ``data_library/mined/<predicted_class>/`` and generated
augmentation artifacts live under ``data_library/derived/<project>/<role>/``.

Sidecars are the source of truth. ``manifest.jsonl`` is a rebuildable index
for fast listing and filtering.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid as _uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from sed import config as config_mod
from sed.config import LIBRARY_DIR
from sed.util import unique_path

log = logging.getLogger(__name__)

SCHEMA_VERSION = 3
RAW_CATEGORY = "sounds"
MINED_CATEGORY = "mined"
DERIVED_CATEGORY = "derived"
# Large external corpora imported for model pretraining/finetuning. These live
# under ``data_library/pretraining/<class>/`` and are deliberately *not* part of
# CATEGORIES: they carry no per-clip sidecars and must never be walked by
# ``rebuild_manifest`` / ``_prune_incomplete_file_pairs`` (which delete WAVs that
# lack a sidecar). They are tracked by a dedicated lightweight index instead
# (see ``sed.data.pretraining.PretrainIndex``).
PRETRAIN_CATEGORY = "pretraining"
CATEGORIES: tuple[str, ...] = (RAW_CATEGORY, MINED_CATEGORY, DERIVED_CATEGORY)
LEGACY_CATEGORIES: tuple[str, ...] = ("positives", "negatives")
# Conventional tag for material recorded as holdout (``sed record --test``).
# A clip carries no split of its own, so this tag is simply what a frozen-test
# source rule filters on (``tags_all``) to pull that material into test.csv.
HOLDOUT_TAG = "holdout"
_SIDECAR_SUFFIX = ".json"
_THUMB_SUFFIX = ".spec.png"
_INITIAL_LIBRARY_DIR = LIBRARY_DIR


def _current_library_dir() -> Path:
    # Tests and small scripts historically patched either sed.config.LIBRARY_DIR
    # or sed.data.library.LIBRARY_DIR. Honor both without making import order
    # decide which library tree gets scanned.
    if LIBRARY_DIR != _INITIAL_LIBRARY_DIR and LIBRARY_DIR != config_mod.LIBRARY_DIR:
        return LIBRARY_DIR
    if config_mod.LIBRARY_DIR != _INITIAL_LIBRARY_DIR:
        return config_mod.LIBRARY_DIR
    if LIBRARY_DIR != _INITIAL_LIBRARY_DIR:
        return LIBRARY_DIR
    return config_mod.LIBRARY_DIR


def _remove_preview_cache_sources(paths: list[Path]) -> None:
    """Reclaim shared preview assets after Library-owned audio is removed."""
    if not paths:
        return
    try:
        from sed.data.preview_cache import PreviewCache

        PreviewCache(root=_current_library_dir()).remove_sources(paths)
    except Exception as exc:  # noqa: BLE001
        log.warning("preview cache cleanup failed: %s", exc)


def safe_label(label: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label.strip())
    return cleaned or "unlabeled"


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"


def coerce_channel_list(value: Any) -> list[int]:
    """Return a de-duplicated list of non-negative channel indexes."""
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = (
            value.strip().lower()
            .replace("mic:", "")
            .replace("mics", "")
            .replace("mic", "")
            .replace(";", ",")
        )
        raw_items: list[Any] = [part.strip() for part in cleaned.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    out: list[int] = []
    seen: set[int] = set()
    for item in raw_items:
        try:
            channel = int(item)
        except (TypeError, ValueError):
            continue
        if channel < 0 or channel in seen:
            continue
        out.append(channel)
        seen.add(channel)
    return out


def _coerce_float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(int(key))] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def normalize_event_intervals(value: Any, *, duration_s: float | None = None) -> list[dict[str, Any]]:
    """Validate portable onset/offset annotations stored in clip sidecars.

    Intervals are intentionally simple JSON records so they survive imports,
    derived datasets, and frozen model bundles.  A proposal is never strong
    ground truth until a human marks it ``confirmed``.
    """
    rows = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    maximum = float(duration_s) if duration_s is not None and duration_s > 0 else None
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        try:
            onset = max(0.0, float(raw.get("onset_s", raw.get("onset", 0.0))))
            offset = float(raw.get("offset_s", raw.get("offset", onset)))
        except (TypeError, ValueError):
            continue
        if maximum is not None:
            onset, offset = min(onset, maximum), min(offset, maximum)
        if offset <= onset:
            continue
        out.append({
            "label": str(raw.get("label") or raw.get("class_name") or "").strip(),
            "onset_s": onset,
            "offset_s": offset,
            "confirmed": bool(raw.get("confirmed", False)),
            "proposed_by": str(raw.get("proposed_by") or ""),
        })
    return out


def _normal_channel_set(channels: list[int]) -> list[int]:
    return sorted(set(int(c) for c in channels if int(c) >= 0))


def clip_mic_channels(clip: "LibraryClip") -> list[int]:
    """Physical mic set used for library filtering; empty means legacy/unknown."""
    return list(clip.triggered_channels or clip.recorded_channels or [])


def clip_matches_session(clip: "LibraryClip", query: object) -> bool:
    """Match raw recording sessions and generated augmentation run prefixes."""
    wanted = str(query or "").strip().lower()
    if not wanted:
        return True
    candidates = " ".join(
        str(value or "").lower()
        for value in (
            clip.session_id,
            clip.recorded_at,
            clip.relpath,
            Path(clip.relpath).stem,
        )
    )
    if wanted in candidates:
        return True
    normalized_wanted = "".join(ch for ch in wanted if ch.isalnum())
    normalized_candidates = "".join(ch for ch in candidates if ch.isalnum())
    return bool(normalized_wanted) and normalized_wanted in normalized_candidates


def clip_audio_channels(clip: "LibraryClip") -> int:
    """Return the channels in the stored audio file.

    Microphone provenance is optional and is deliberately not used here.  A
    legacy mono clip can have no recorded/triggered channel metadata at all,
    while a channel-split clip has one stored channel but retains a parent
    multichannel count for provenance.
    """
    try:
        channels = int(clip.channels or 0)
    except (TypeError, ValueError):
        channels = 0
    if channels > 0:
        return channels
    if str(getattr(clip, "clip_kind", "") or "").lower() == "multichannel":
        try:
            return max(1, int(clip.channel_count or 1))
        except (TypeError, ValueError):
            return 1
    return 1


def clip_matches_mic_rule(clip: "LibraryClip", rule: dict[str, Any]) -> bool:
    """Match UI/recipe mic filters against a library clip.

    Supported rule forms:
      - mic_filter: "mono" | "single" | "multi" | "unknown" | "0,2" | "mic:0,2"
      - triggered_channels / recorded_channels / mic_channels / mics: list or CSV
    """
    raw: Any = None
    for key in (
        "mic_filter", "triggered_channels", "recorded_channels",
        "mic_channels", "mics",
    ):
        if rule.get(key) not in (None, ""):
            raw = rule.get(key)
            break
    if raw in (None, "", "*", "all", "any"):
        return True

    clip_channels = _normal_channel_set(clip_mic_channels(clip))
    audio_channels = clip_audio_channels(clip)
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"", "*", "all", "any"}:
            return True
        if value in {"mono", "one-channel", "one_channel", "audio-mono"}:
            # "mono" means every stored audio file that is not multichannel;
            # missing microphone metadata must not exclude it.
            return audio_channels <= 1
        if value in {"multichannel", "multi-channel", "audio-multi"}:
            return audio_channels > 1
        if value in {"unknown", "legacy", "none"}:
            return not clip_channels
        if value in {"single", "mono", "one"}:
            return len(clip_channels) == 1
        if value in {"multi", "multichannel", "multiple"}:
            return len(clip_channels) > 1
        raw = value

    wanted = _normal_channel_set(coerce_channel_list(raw))
    return bool(wanted) and clip_channels == wanted


@dataclass
class LibraryClip:
    uuid: str
    relpath: str
    label: str
    category: str = RAW_CATEGORY
    project_id: str = "_library"
    # NOTE: a clip carries no train/test membership. Which split a recording
    # lands in is a composition decision expressed by a source rule's ``split``
    # (see sed.data.compose), so the same audio can serve either side without
    # rewriting sidecars.
    density: str = "normal"
    review_status: str = "accepted"
    recording_counter: int = 0
    duration_s: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    # Multichannel split provenance. ``clip_kind`` distinguishes as-recorded
    # mono ("mono"), as-recorded multichannel originals ("multichannel"), and
    # per-channel mono clips extracted from a multichannel original
    # ("channel_split"). The grouping fields tie split siblings back together.
    clip_kind: str = "mono"
    channel_index: int | None = None
    channel_count: int = 1
    parent_uuid: str = ""
    group_id: str = ""
    recorded_channels: list[int] = field(default_factory=list)
    triggered_channels: list[int] = field(default_factory=list)
    microphone: int | None = None
    microphone_label: str = ""
    triggered_flux: dict[str, float] = field(default_factory=dict)
    triggered_peak_dbfs: dict[str, float] = field(default_factory=dict)
    recorded_at: str = ""
    session_id: str = ""
    device: str = ""
    mic_position: str = ""
    location: str = ""
    microphone_type: str | list[str] = "generic"
    origin: str = "recorded"
    source_path: str = ""
    # Stable provenance for derived clips. ``original_source`` names the root
    # recording even through repeated augmentation; ``source_split_group`` is
    # the root recording/session group that every variant must inherit.
    original_source: str = ""
    augmentation_family: str = ""
    source_split_group: str = ""
    # The original class/category this clip came from in its source data
    # structure (the import folder name, or the recorded class). Preserved
    # through reassignment so cards/filters can always show provenance, and the
    # Dataset tab can group a target class by its constituent origin classes.
    # Defaults to ``label`` when absent from the sidecar.
    origin_class: str = ""
    # For derived (augmented) clips: the overlay sound class mixed onto the base.
    # Empty for raw recordings. Surfaced so the Inspect source-rule rows can show
    # which overlay(s) a derived bucket contains.
    overlay_label: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    # Strong-label support for optional frame-level SED. Empty means this clip
    # remains a normal weak/clip-labelled training example.
    event_intervals: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def absolute_path(self, base: Path | None = None) -> Path:
        return (base or _current_library_dir()) / self.relpath

    @classmethod
    def from_sidecar(cls, wav_path: Path, root: Path | None = None) -> "LibraryClip":
        root = root or _current_library_dir()
        sidecar = wav_path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if sidecar.exists():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))

        try:
            info = sf.info(str(wav_path))
            sample_rate = int(info.samplerate)
            channels = int(info.channels)
            duration_s = info.frames / info.samplerate if info.samplerate else 0.0
        except Exception:  # noqa: BLE001
            sample_rate = int(meta.get("sample_rate") or 0)
            channels = int(meta.get("channels") or 1)
            duration_s = float(meta.get("duration_s") or 0.0)

        try:
            relpath = str(wav_path.resolve().relative_to(root.resolve()))
        except ValueError:
            relpath = str(wav_path)

        parts = Path(relpath).parts
        inferred_category = parts[0] if parts else RAW_CATEGORY
        if inferred_category in LEGACY_CATEGORIES:
            inferred_category = RAW_CATEGORY

        label = (
            meta.get("label")
            or meta.get("class_name")
            or (parts[1] if len(parts) > 1 else wav_path.parent.name)
            or wav_path.stem
        )

        category = str(meta.get("category") or inferred_category or RAW_CATEGORY)
        if category in LEGACY_CATEGORIES:
            category = RAW_CATEGORY

        raw_mic_type = meta.get("microphone_type") or "generic"
        if isinstance(raw_mic_type, list):
            mic_type = [str(m) for m in raw_mic_type]
        else:
            mic_type = str(raw_mic_type)

        review_status = str(meta.get("review_status") or _status_for_category(category))
        if category == MINED_CATEGORY and review_status == "mined":
            review_status = "accepted"

        params = (
            meta.get("augmentation_params")
            if isinstance(meta.get("augmentation_params"), dict)
            else {}
        )
        source_path_value = str(meta.get("source_path") or meta.get("source") or "")
        original_source = str(meta.get("original_source") or source_path_value)
        augmentation_family = str(
            meta.get("augmentation_family")
            or meta.get("augmentation")
            or params.get("augmentation_technique")
            or ""
        )
        source_split_group = str(meta.get("source_split_group") or "")
        if not source_split_group and category == DERIVED_CATEGORY:
            source_split_group = str(
                meta.get("group_id")
                or meta.get("session_id")
                or meta.get("parent_uuid")
                or (Path(original_source).stem if original_source else "")
            )

        return cls(
            uuid=str(meta.get("uuid") or _uuid.uuid4().hex),
            relpath=relpath,
            category=category,
            label=str(label),
            project_id=str(meta.get("project_id") or "_library"),
            density=str(meta.get("density") or _density_from_tags(meta.get("tags")) or "normal"),
            review_status=review_status,
            recording_counter=int(meta.get("recording_counter") or 0),
            duration_s=float(meta.get("duration_s") or duration_s),
            sample_rate=int(meta.get("sample_rate") or sample_rate),
            channels=int(meta.get("channels") or channels),
            clip_kind=str(
                meta.get("clip_kind")
                or ("multichannel" if int(meta.get("channels") or channels) > 1 else "mono")
            ),
            channel_index=(
                int(meta["channel_index"])
                if meta.get("channel_index") is not None
                else None
            ),
            channel_count=int(meta.get("channel_count") or meta.get("channels") or channels or 1),
            parent_uuid=str(meta.get("parent_uuid") or ""),
            group_id=str(meta.get("group_id") or meta.get("parent_uuid") or ""),
            recorded_channels=coerce_channel_list(meta.get("recorded_channels")),
            triggered_channels=coerce_channel_list(meta.get("triggered_channels")),
            microphone=(
                int(meta["microphone"])
                if meta.get("microphone") is not None
                else None
            ),
            microphone_label=str(meta.get("microphone_label") or ""),
            triggered_flux=_coerce_float_map(meta.get("triggered_flux")),
            triggered_peak_dbfs=_coerce_float_map(meta.get("triggered_peak_dbfs")),
            recorded_at=str(meta.get("recorded_at") or meta.get("created_at") or ""),
            session_id=str(meta.get("session_id") or ""),
            device=str(meta.get("device") or ""),
            mic_position=str(meta.get("mic_position") or ""),
            location=str(meta.get("location") or ""),
            microphone_type=mic_type,
            origin=str(meta.get("origin") or "recorded"),
            source_path=source_path_value,
            original_source=original_source,
            augmentation_family=augmentation_family,
            source_split_group=source_split_group,
            origin_class=str(meta.get("origin_class") or label),
            overlay_label=str(
                meta.get("overlay_label")
                or (meta.get("augmentation_params") or {}).get("actual_overlay_label")
                or (meta.get("augmentation_params") or {}).get("overlay_label")
                or ""
            ),
            tags=list(meta.get("tags") or []),
            notes=str(meta.get("notes") or meta.get("description") or ""),
            event_intervals=normalize_event_intervals(
                meta.get("event_intervals"), duration_s=float(meta.get("duration_s") or duration_s)
            ),
            schema_version=int(meta.get("schema_version") or SCHEMA_VERSION),
        )

    def to_sidecar(self, wav_path: Path | None = None) -> dict[str, Any]:
        data = asdict(self)
        if wav_path is not None:
            data["path"] = str(wav_path.resolve())
        data["class_name"] = self.label
        # Preserve the original class even when unset on construction (older
        # imports / direct recordings) so cards and filters always have it.
        data["origin_class"] = self.origin_class or self.label
        if self.category == DERIVED_CATEGORY:
            data["original_source"] = self.original_source or self.source_path
            data["augmentation_family"] = self.augmentation_family or "derived"
            data["source_split_group"] = (
                self.source_split_group
                or self.group_id
                or self.session_id
                or self.parent_uuid
                or Path(self.original_source or self.source_path).stem
            )
        return data


def _status_for_category(category: str) -> str:
    return "accepted"


def _density_from_tags(tags: Any) -> str | None:
    if not isinstance(tags, list):
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("density:"):
            return tag.split(":", 1)[1]
    return None


class Library:
    def __init__(self, root: Path | None = None):
        self.root = root or _current_library_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.touch(exist_ok=True)
        self._clips_cache: list[LibraryClip] | None = None
        self._clips_cache_signature: tuple[int, int] | None = None

    def _manifest_signature(self) -> tuple[int, int]:
        stat = self.manifest.stat()
        return (int(stat.st_mtime_ns), int(stat.st_size))

    def _invalidate_cache(self) -> None:
        self._clips_cache = None
        self._clips_cache_signature = None

    # ---- paths ------------------------------------------------------------

    def raw_dir(self, label: str) -> Path:
        out = self.root / RAW_CATEGORY / safe_label(label)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def mined_dir(self, predicted_class: str) -> Path:
        out = self.root / MINED_CATEGORY / safe_label(predicted_class)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def derived_dir(self, project_id: str, split: str, class_name: str) -> Path:
        """Output directory for generated clips of one composition ``split``.

        Augmented material is kept physically separate per split so a
        frozen-test batch can never be picked up by a training scan.
        """
        out = self.root / DERIVED_CATEGORY / safe_label(project_id) / split / safe_label(class_name)
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ---- read -------------------------------------------------------------

    def all_clips(self, rebuild_if_empty: bool = False) -> list[LibraryClip]:
        if rebuild_if_empty and self.manifest.stat().st_size == 0:
            return self.rebuild_manifest()

        signature = self._manifest_signature()
        if self._clips_cache is not None and self._clips_cache_signature == signature:
            return list(self._clips_cache)

        clips: list[LibraryClip] = []
        stale = False
        with open(self.manifest, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning(
                        "library manifest parse failed at %s:%d: %s; rebuilding",
                        self.manifest,
                        lineno,
                        exc,
                    )
                    return self.rebuild_manifest()
                row = self._normalize_manifest_row(row)
                try:
                    clip = LibraryClip(**row)
                except TypeError:
                    clip = self._clip_from_legacy_row(row)
                if not self._clip_files_complete(clip):
                    stale = True
                    continue
                clips.append(clip)
        if stale:
            return self.rebuild_manifest()
        self._clips_cache = clips
        self._clips_cache_signature = signature
        return list(clips)

    def query(
        self,
        *,
        category: str | None = None,
        label: str | None = None,
        project_ids: set[str] | None = None,
        review_status: str | None = None,
        density: str | None = None,
        tags_all: list[str] | None = None,
        **_legacy_membership: object,
    ) -> list[LibraryClip]:
        clips = self.all_clips(rebuild_if_empty=True)
        if category is not None:
            clips = [c for c in clips if c.category == category]
        if label is not None:
            clips = [c for c in clips if c.label == label]
        if project_ids is not None:
            clips = [c for c in clips if c.project_id in project_ids]
        # ``**_legacy_membership`` absorbs the retired ``dataset_role`` argument.
        # Dataset membership comes exclusively from composition rules; a
        # recording is never filtered into train/test here.
        if review_status is not None:
            clips = [c for c in clips if c.review_status == review_status]
        if density is not None:
            clips = [c for c in clips if c.density == density]
        if tags_all:
            required = set(tags_all)
            clips = [c for c in clips if required.issubset(set(c.tags))]
        return clips

    def labels(self, category: str = RAW_CATEGORY) -> list[str]:
        return sorted({c.label for c in self.all_clips(rebuild_if_empty=True) if c.category == category})

    def stats(self) -> dict:
        out: dict[str, Any] = {"total": 0, "categories": {}}
        for clip in self.all_clips(rebuild_if_empty=True):
            out["total"] += 1
            cat = out["categories"].setdefault(clip.category, {"count": 0, "labels": {}})
            cat["count"] += 1
            cat["labels"][clip.label] = cat["labels"].get(clip.label, 0) + 1
        return out

    # ---- write ------------------------------------------------------------

    def append_clip(self, clip: LibraryClip) -> LibraryClip:
        with open(self.manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(clip), sort_keys=True) + "\n")
        self._invalidate_cache()
        return clip

    def append_sidecar(self, wav_path: Path) -> LibraryClip:
        return self.append_clip(LibraryClip.from_sidecar(wav_path, self.root))

    def write_sidecar(self, wav_path: Path, clip: LibraryClip) -> None:
        wav_path.with_suffix(".json").write_text(
            json.dumps(clip.to_sidecar(wav_path), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def set_event_intervals(self, wav_path: Path, intervals: list[dict[str, Any]]) -> LibraryClip:
        """Persist confirmed/proposed SED intervals and rebuild the shared index."""
        wav_path = Path(wav_path).resolve()
        clip = LibraryClip.from_sidecar(wav_path, self.root)
        clip.event_intervals = normalize_event_intervals(intervals, duration_s=clip.duration_s)
        self.write_sidecar(wav_path, clip)
        self.rebuild_manifest()
        return clip

    def import_clip(
        self,
        source_path: Path,
        *,
        label: str,
        project_id: str = "_library",
        density: str = "normal",
        review_status: str = "accepted",
        move: bool = False,
        notes: str = "",
        tags: list[str] | None = None,
        **_legacy_membership: object,
    ) -> LibraryClip:
        """Copy (or move) a clip into the library.

        ``**_legacy_membership`` absorbs the retired ``dataset_role`` argument:
        an imported recording has no train/test membership, so a project's
        source rules decide where it composes.
        """
        source_path = Path(source_path)
        safe = safe_label(label)
        clip_uuid = _uuid.uuid4().hex
        target = self.raw_dir(label) / f"{utc_stamp()}_{safe}_0001_{clip_uuid[:8]}.wav"
        target = unique_path(target)
        if move:
            shutil.move(str(source_path), str(target))
        else:
            shutil.copy2(str(source_path), str(target))
        clip = LibraryClip.from_sidecar(target, self.root)
        clip.uuid = clip_uuid
        clip.category = RAW_CATEGORY
        clip.label = label
        clip.project_id = project_id
        clip.density = density
        clip.review_status = review_status
        clip.recorded_at = clip.recorded_at or utc_stamp()
        clip.origin = "imported"
        clip.source_path = str(source_path.resolve())
        clip.tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
        clip.notes = notes
        self.write_sidecar(target, clip)
        return self.append_clip(clip)

    def slice_file(
        self,
        source_path: Path,
        *,
        label: str,
        seconds: float,
        hop_seconds: float | None = None,
        project_id: str = "_library",
        density: str = "normal",
        review_status: str = "accepted",
        notes: str = "",
        tags: list[str] | None = None,
        **_legacy_membership: object,
    ) -> list[LibraryClip]:
        """Cut a long recording into consecutive window-length library clips.

        Continuous ambience/background beds are the preferred negative
        material: every slice is a full model window, so training never has
        to zero-pad them the way it pads short imported clips. The trailing
        partial window is dropped and the source file is left untouched (keep
        it around — threshold calibration consumes long files directly).
        """
        source_path = Path(source_path)
        if seconds <= 0:
            raise ValueError("slice seconds must be positive")
        hop = float(hop_seconds) if hop_seconds else float(seconds)
        if hop <= 0:
            raise ValueError("slice hop_seconds must be positive")
        data, sr = sf.read(str(source_path), dtype="float32", always_2d=True)
        win_n = int(round(seconds * sr))
        hop_n = int(round(hop * sr))
        if win_n <= 0 or hop_n <= 0:
            raise ValueError("slice window is empty at this sample rate")

        safe = safe_label(label)
        out_dir = self.raw_dir(label)
        session_id = f"slice-{source_path.stem}-{_uuid.uuid4().hex[:8]}"
        clean_tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
        clips: list[LibraryClip] = []
        stamp = utc_stamp()
        for idx, start in enumerate(range(0, len(data) - win_n + 1, hop_n), start=1):
            clip_uuid = _uuid.uuid4().hex
            target = unique_path(
                out_dir / f"{stamp}_{safe}_{idx:04d}_{clip_uuid[:8]}.wav"
            )
            sf.write(str(target), data[start : start + win_n], sr, subtype="PCM_16")
            clip = LibraryClip.from_sidecar(target, self.root)
            clip.uuid = clip_uuid
            clip.category = RAW_CATEGORY
            clip.label = label
            clip.project_id = project_id
            clip.density = density
            clip.review_status = review_status
            clip.recording_counter = idx
            clip.recorded_at = stamp
            clip.session_id = session_id
            clip.origin = "sliced"
            clip.source_path = str(source_path.resolve())
            clip.tags = list(clean_tags)
            offset_note = (
                f"slice {start / sr:.2f}-{(start + win_n) / sr:.2f}s of {source_path.name}"
            )
            clip.notes = f"{notes} | {offset_note}" if notes else offset_note
            self.write_sidecar(target, clip)
            clips.append(self.append_clip(clip))
        return clips

    def add_clip(
        self,
        source_path: Path,
        category: str,
        label: str,
        sample_rate: int,
        channels: int,
        duration_s: float,
        session_id: str = "",
        device: str = "",
        mic_position: str = "",
        tags: list[str] | None = None,
        notes: str = "",
        move: bool = True,
    ) -> LibraryClip:
        """Compatibility wrapper for older imports.

        Legacy callers may still pass ``positives`` or ``negatives`` as the
        category. New storage is always ``sounds/<label>/``; target/negative
        meaning is assigned by project dataset config.
        """
        clip = self.import_clip(
            source_path,
            label=label,
            project_id="_library",
            density=_density_from_tags(tags) or "normal",
            review_status="accepted",
            move=move,
            notes=notes,
        )
        clip.sample_rate = sample_rate or clip.sample_rate
        clip.channels = channels or clip.channels
        clip.duration_s = duration_s or clip.duration_s
        clip.session_id = session_id or clip.session_id
        clip.device = device or clip.device
        clip.mic_position = mic_position or clip.mic_position
        clip.tags = list(tags or [])
        self.write_sidecar(self.root / clip.relpath, clip)
        self.rebuild_manifest()
        return clip

    def split_clip(self, wav_path: Path, *, force: bool = False) -> list[Path]:
        """Split a multichannel recording into per-channel mono library clips.

        The multichannel original is left untouched (it stays the archive). Each
        channel becomes a mono WAV under ``sounds/<label>/<label>_ch<c>/`` with a
        sidecar carrying ``clip_kind="channel_split"`` plus the grouping fields
        (``parent_uuid``/``group_id``/``channel_index``/``channel_count``) so the
        siblings can be shown together and selected independently of the original.

        Returns the list of created mono WAV paths (empty if the source is mono
        or already split and ``force`` is False).
        """
        wav_path = Path(wav_path)
        sidecar = wav_path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = {}

        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        n_ch = int(data.shape[1])
        if n_ch <= 1:
            return []

        parent_uuid = str(meta.get("uuid") or "")
        label = str(meta.get("label") or meta.get("class_name") or wav_path.parent.name)
        safe = safe_label(label)
        parent8 = (parent_uuid or wav_path.stem)[:8]
        label_dir = self.raw_dir(label)

        existing = list(label_dir.glob(f"{safe}_ch*/*_{parent8}_*.wav"))
        if existing:
            if not force:
                return []
            # Clean redo: drop the previous splits for this parent first so we
            # don't accumulate duplicates.
            for old in existing:
                old.unlink(missing_ok=True)
                old.with_suffix(_SIDECAR_SUFFIX).unlink(missing_ok=True)
            _remove_preview_cache_sources(existing)

        created: list[Path] = []
        ts = utc_stamp()
        parent_recorded = coerce_channel_list(meta.get("recorded_channels"))
        parent_triggered = set(coerce_channel_list(meta.get("triggered_channels")))
        parent_flux = _coerce_float_map(meta.get("triggered_flux"))
        parent_peak = _coerce_float_map(meta.get("triggered_peak_dbfs"))
        for c in range(n_ch):
            physical_channel = parent_recorded[c] if c < len(parent_recorded) else None
            ch_dir = label_dir / f"{safe}_ch{c}"
            ch_dir.mkdir(parents=True, exist_ok=True)
            split_uuid = _uuid.uuid4().hex
            dst = unique_path(ch_dir / f"{ts}_{safe}_ch{c}_{parent8}_{split_uuid[:8]}.wav")
            sf.write(str(dst), data[:, c], sr, subtype="PCM_16")

            clip = LibraryClip.from_sidecar(dst, self.root)
            clip.uuid = split_uuid
            clip.category = RAW_CATEGORY
            clip.label = label
            clip.channels = 1
            clip.clip_kind = "channel_split"
            clip.channel_index = c
            clip.channel_count = n_ch
            clip.parent_uuid = parent_uuid
            clip.group_id = parent_uuid or parent8
            clip.recorded_channels = (
                [physical_channel] if physical_channel is not None else []
            )
            clip.microphone = physical_channel
            clip.microphone_label = (
                f"mic {physical_channel}"
                if physical_channel is not None
                else f"split ch{c}"
            )
            clip.triggered_channels = (
                [physical_channel]
                if physical_channel is not None and physical_channel in parent_triggered
                else []
            )
            flux_key = str(physical_channel)
            clip.triggered_flux = (
                {flux_key: parent_flux[flux_key]}
                if flux_key in parent_flux
                else {}
            )
            clip.triggered_peak_dbfs = (
                {flux_key: parent_peak[flux_key]}
                if flux_key in parent_peak
                else {}
            )
            clip.origin = "channel_split"
            clip.source_path = str(wav_path.resolve())
            clip.project_id = str(meta.get("project_id") or clip.project_id)
            clip.density = str(meta.get("density") or clip.density)
            clip.review_status = str(meta.get("review_status") or clip.review_status)
            clip.session_id = str(meta.get("session_id") or clip.session_id)
            clip.device = str(meta.get("device") or clip.device)
            clip.mic_position = str(meta.get("mic_position") or clip.mic_position)
            parent_mic = meta.get("microphone_type") or clip.microphone_type
            if isinstance(parent_mic, list):
                if c < len(parent_mic):
                    clip.microphone_type = str(parent_mic[c])
                elif parent_mic:
                    clip.microphone_type = str(parent_mic[-1])
                else:
                    clip.microphone_type = "generic"
            else:
                clip.microphone_type = str(parent_mic)
            clip.recorded_at = str(meta.get("recorded_at") or clip.recorded_at)
            clip.tags = list(meta.get("tags") or [])
            self.write_sidecar(dst, clip)
            self.append_clip(clip)
            created.append(dst)
        return created

    def remove_clip(self, clip_uuid: str) -> bool:
        clips = self.all_clips()
        matched = [c for c in clips if c.uuid == clip_uuid]
        if not matched:
            return False
        for clip in matched:
            path = self.root / clip.relpath
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            _remove_preview_cache_sources([path])
        self.rebuild_manifest()
        return True

    def remove_clips(self, paths: list[str | Path] | set[str | Path]) -> list[LibraryClip]:
        """Delete these clips' files and drop their manifest rows in one pass.

        The bulk counterpart to :meth:`remove_clip`, which rebuilds the whole
        manifest per clip — fine for undoing one take, but quadratic when a
        recording run of several hundred clips is discarded at once.

        Returns the clip records that were removed from the index.
        """
        wavs = [Path(p) for p in paths]
        for wav in wavs:
            self._delete_file_pair(wav)
        return self.remove_paths_from_manifest([str(w) for w in wavs])

    def remove_paths_from_manifest(
        self, paths: list[str | Path] | set[str | Path],
    ) -> list[LibraryClip]:
        """Atomically remove exact library paths from ``manifest.jsonl``.

        The sidecars remain the rebuildable source of truth, but a targeted
        delete must not call :meth:`rebuild_manifest`: that walks every library
        directory and reparses every sidecar for a one-row mutation.  This
        method filters the existing index once, preserves unrelated rows, and
        returns the removed clip records so callers can update project counts
        without another inventory scan.
        """
        wanted: set[str] = set()
        root = self.root.resolve()
        for raw in paths:
            try:
                rel = Path(raw).expanduser().resolve().relative_to(root)
            except (OSError, ValueError):
                continue
            if not rel.parts or rel.parts[0] not in {*CATEGORIES, *LEGACY_CATEGORIES}:
                continue
            wanted.add(str(rel))
        if not wanted or not self.manifest.exists():
            return []

        kept_lines: list[str] = []
        removed: list[LibraryClip] = []
        changed = False
        with open(self.manifest, encoding="utf-8") as src:
            for line in src:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    # Preserve a malformed row.  ``all_clips`` retains its
                    # existing repair-on-read behavior for unrelated damage.
                    kept_lines.append(stripped)
                    continue
                relpath = str(row.get("relpath") or "")
                if relpath not in wanted:
                    kept_lines.append(stripped)
                    continue
                changed = True
                normalized = self._normalize_manifest_row(row)
                try:
                    removed.append(LibraryClip(**{
                        key: value for key, value in normalized.items()
                        if key in LibraryClip.__dataclass_fields__
                    }))
                except TypeError:
                    try:
                        removed.append(self._clip_from_legacy_row(normalized))
                    except Exception:  # noqa: BLE001 - deletion still succeeds
                        pass

        if not changed:
            return []
        tmp = self.manifest.with_name(f"{self.manifest.name}.{_uuid.uuid4().hex}.tmp")
        with open(tmp, "w", encoding="utf-8") as dst:
            for line in kept_lines:
                dst.write(line + "\n")
            dst.flush()
        tmp.replace(self.manifest)
        self._clips_cache = None
        self._clips_cache_signature = None
        return removed

    def rebuild_manifest(self) -> list[LibraryClip]:
        clips: list[LibraryClip] = []
        roots = [self.root / c for c in CATEGORIES]
        roots.extend(self.root / c for c in LEGACY_CATEGORIES)
        self._prune_incomplete_file_pairs(roots)
        seen_uuids: set[str] = set()
        seen_paths: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            for wav in sorted(root.rglob("*.wav")):
                if not wav.with_suffix(_SIDECAR_SUFFIX).exists():
                    self._delete_file_pair(wav)
                    continue
                try:
                    sidecar_path = wav.with_suffix(_SIDECAR_SUFFIX)
                    legacy_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    clip = LibraryClip.from_sidecar(wav, self.root)
                    if isinstance(legacy_sidecar, dict) and "dataset_role" in legacy_sidecar:
                        # One-time, lossless metadata migration: dataset
                        # membership is now selected by project rules.
                        legacy_sidecar.pop("dataset_role", None)
                        sidecar_path.write_text(
                            json.dumps(legacy_sidecar, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
                    log.warning("skipping invalid library sidecar for %s: %s", wav, exc)
                    continue
                row = self._normalize_manifest_row(asdict(clip))
                clip = LibraryClip(**{
                    k: v for k, v in row.items()
                    if k in LibraryClip.__dataclass_fields__
                })
                if clip.uuid and clip.uuid in seen_uuids:
                    continue
                if clip.relpath in seen_paths:
                    continue
                if clip.uuid:
                    seen_uuids.add(clip.uuid)
                seen_paths.add(clip.relpath)
                clips.append(clip)
        tmp = self.manifest.with_name(f"{self.manifest.name}.{_uuid.uuid4().hex}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for clip in clips:
                f.write(json.dumps(asdict(clip), sort_keys=True) + "\n")
            f.flush()
        tmp.replace(self.manifest)
        self._clips_cache = clips
        self._clips_cache_signature = self._manifest_signature()
        return list(clips)

    def _clip_files_complete(self, clip: LibraryClip) -> bool:
        wav = self.root / clip.relpath
        return wav.exists() and wav.with_suffix(_SIDECAR_SUFFIX).exists()

    def _delete_file_pair(self, wav_path: Path) -> None:
        wav_path.unlink(missing_ok=True)
        wav_path.with_suffix(_SIDECAR_SUFFIX).unlink(missing_ok=True)
        _thumbnail_path(wav_path).unlink(missing_ok=True)
        _remove_preview_cache_sources([wav_path])

    def _prune_incomplete_file_pairs(self, roots: list[Path]) -> None:
        for root in roots:
            if not root.exists():
                continue
            for wav in sorted(root.rglob("*.wav")):
                if not wav.with_suffix(_SIDECAR_SUFFIX).exists():
                    self._delete_file_pair(wav)
            for sidecar in sorted(root.rglob(f"*{_SIDECAR_SUFFIX}")):
                if not sidecar.with_suffix(".wav").exists():
                    sidecar.unlink(missing_ok=True)
            for thumb in sorted(root.rglob(f"*{_THUMB_SUFFIX}")):
                wav = _wav_for_thumbnail(thumb)
                if wav is None or not wav.exists():
                    thumb.unlink(missing_ok=True)

    def legacy_audio_paths(self) -> list[Path]:
        paths: list[Path] = []
        for category in LEGACY_CATEGORIES:
            root = self.root / category
            if root.exists():
                paths.extend(sorted(root.rglob("*.wav")))
        return paths

    def migrate_legacy(self, *, mode: str = "copy", dry_run: bool = False) -> dict[str, Any]:
        """Move or copy legacy positives/negatives folders into sounds/.

        The old folder name no longer carries target-vs-negative meaning. Each
        clip becomes a normal raw library sound with the same label and sidecar
        provenance, ready to be selected by project config.
        """
        if mode not in {"copy", "move"}:
            raise ValueError("mode must be 'copy' or 'move'")

        items: list[dict[str, str]] = []
        for wav in self.legacy_audio_paths():
            clip = LibraryClip.from_sidecar(wav, self.root)
            label = safe_label(clip.label)
            target_dir = self.root / RAW_CATEGORY / label
            target = unique_path(target_dir / wav.name)
            items.append({
                "source": str(wav),
                "dest": str(target),
                "label": clip.label,
                "mode": mode,
            })
            if dry_run:
                continue

            target_dir.mkdir(parents=True, exist_ok=True)
            sidecar = wav.with_suffix(".json")
            target_sidecar = target.with_suffix(".json")
            if mode == "move":
                shutil.move(str(wav), str(target))
                if sidecar.exists():
                    shutil.move(str(sidecar), str(target_sidecar))
            else:
                shutil.copy2(str(wav), str(target))
                if sidecar.exists():
                    shutil.copy2(str(sidecar), str(target_sidecar))

            migrated = LibraryClip.from_sidecar(target, self.root)
            migrated.category = RAW_CATEGORY
            migrated.label = clip.label
            migrated.relpath = str(target.relative_to(self.root))
            migrated.project_id = clip.project_id
            migrated.review_status = clip.review_status
            migrated.density = clip.density
            migrated.source_path = clip.source_path or str(wav.resolve())
            self.write_sidecar(target, migrated)

        if not dry_run:
            self.rebuild_manifest()
        return {
            "dry_run": dry_run,
            "mode": mode,
            "count": len(items),
            "items": items,
        }

    # ---- legacy helpers ---------------------------------------------------

    def _normalize_manifest_row(self, row: dict[str, Any]) -> dict[str, Any]:
        # Dataset membership is composition-only. Strip role metadata from
        # legacy manifests while normalising them into ordinary library clips.
        row = dict(row)
        row.pop("dataset_role", None)
        if row.get("category") in LEGACY_CATEGORIES:
            row["category"] = RAW_CATEGORY
            row.setdefault("project_id", "_library")
            row.setdefault("density", _density_from_tags(row.get("tags")) or "normal")
            row.setdefault("review_status", "accepted")
            row.setdefault("recording_counter", 0)
            row.setdefault("microphone_type", "generic")
            row.setdefault("origin", "legacy_manifest")
            row.setdefault("source_path", "")
            row.setdefault("schema_version", SCHEMA_VERSION)
            row["label"] = row.get("label") or row.get("class_name") or "unlabeled"
        elif row.get("category") == MINED_CATEGORY and row.get("review_status") == "mined":
            row["review_status"] = "accepted"
        return row

    def _clip_from_legacy_row(self, row: dict[str, Any]) -> LibraryClip:
        normalized = self._normalize_manifest_row(row)
        allowed = set(LibraryClip.__dataclass_fields__)
        return LibraryClip(**{k: v for k, v in normalized.items() if k in allowed})




def _thumbnail_path(wav_path: Path) -> Path:
    return wav_path.with_suffix(_THUMB_SUFFIX)


def _wav_for_thumbnail(path: Path) -> Path | None:
    name = path.name
    if not name.endswith(_THUMB_SUFFIX):
        return None
    return path.with_name(name[: -len(_THUMB_SUFFIX)] + ".wav")
