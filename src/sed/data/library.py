"""Unified audio library shared across projects.

Raw recordings live under ``data_library/sounds/<label>/``. Runtime-mined
review clips live under ``data_library/mined/<predicted_class>/`` and generated
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

from sed.config import LIBRARY_DIR

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2
RAW_CATEGORY = "sounds"
MINED_CATEGORY = "mined"
DERIVED_CATEGORY = "derived"
CATEGORIES: tuple[str, ...] = (RAW_CATEGORY, MINED_CATEGORY, DERIVED_CATEGORY)
LEGACY_CATEGORIES: tuple[str, ...] = ("positives", "negatives")
_SIDECAR_SUFFIX = ".json"
_THUMB_SUFFIX = ".spec.png"


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


def _normal_channel_set(channels: list[int]) -> list[int]:
    return sorted(set(int(c) for c in channels if int(c) >= 0))


def clip_mic_channels(clip: "LibraryClip") -> list[int]:
    """Physical mic set used for library filtering; empty means legacy/unknown."""
    return list(clip.triggered_channels or clip.recorded_channels or [])


def clip_matches_mic_rule(clip: "LibraryClip", rule: dict[str, Any]) -> bool:
    """Match UI/recipe mic filters against a library clip.

    Supported rule forms:
      - mic_filter: "single" | "multi" | "unknown" | "0,2" | "mic:0,2"
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
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"", "*", "all", "any"}:
            return True
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
    dataset_role: str = "train"
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
    microphone_type: str | list[str] = "generic"
    origin: str = "recorded"
    source_path: str = ""
    # For derived (augmented) clips: the overlay sound class mixed onto the base.
    # Empty for raw recordings. Surfaced so the Inspect source-rule rows can show
    # which overlay(s) a derived bucket contains.
    overlay_label: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    def absolute_path(self, base: Path | None = None) -> Path:
        return (base or LIBRARY_DIR) / self.relpath

    @classmethod
    def from_sidecar(cls, wav_path: Path, root: Path | None = None) -> "LibraryClip":
        root = root or LIBRARY_DIR
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

        return cls(
            uuid=str(meta.get("uuid") or _uuid.uuid4().hex),
            relpath=relpath,
            category=category,
            label=str(label),
            project_id=str(meta.get("project_id") or "_library"),
            dataset_role=str(meta.get("dataset_role") or "train"),
            density=str(meta.get("density") or _density_from_tags(meta.get("tags")) or "normal"),
            review_status=str(meta.get("review_status") or _status_for_category(inferred_category)),
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
            microphone_type=mic_type,
            origin=str(meta.get("origin") or "recorded"),
            source_path=str(meta.get("source_path") or meta.get("source") or ""),
            overlay_label=str(
                meta.get("overlay_label")
                or (meta.get("augmentation_params") or {}).get("actual_overlay_label")
                or (meta.get("augmentation_params") or {}).get("overlay_label")
                or ""
            ),
            tags=list(meta.get("tags") or []),
            notes=str(meta.get("notes") or meta.get("description") or ""),
            schema_version=int(meta.get("schema_version") or SCHEMA_VERSION),
        )

    def to_sidecar(self, wav_path: Path | None = None) -> dict[str, Any]:
        data = asdict(self)
        if wav_path is not None:
            data["path"] = str(wav_path.resolve())
        data["class_name"] = self.label
        return data


def _status_for_category(category: str) -> str:
    return "mined" if category == MINED_CATEGORY else "accepted"


def _density_from_tags(tags: Any) -> str | None:
    if not isinstance(tags, list):
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("density:"):
            return tag.split(":", 1)[1]
    return None


class Library:
    def __init__(self, root: Path | None = None):
        self.root = root or LIBRARY_DIR
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

    def derived_dir(self, project_id: str, dataset_role: str, class_name: str) -> Path:
        out = self.root / DERIVED_CATEGORY / safe_label(project_id) / dataset_role / safe_label(class_name)
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
        dataset_role: str | None = None,
        review_status: str | None = None,
        density: str | None = None,
        tags_all: list[str] | None = None,
    ) -> list[LibraryClip]:
        clips = self.all_clips(rebuild_if_empty=True)
        if category is not None:
            clips = [c for c in clips if c.category == category]
        if label is not None:
            clips = [c for c in clips if c.label == label]
        if project_ids is not None:
            clips = [c for c in clips if c.project_id in project_ids]
        if dataset_role is not None:
            clips = [c for c in clips if c.dataset_role == dataset_role]
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

    def import_clip(
        self,
        source_path: Path,
        *,
        label: str,
        project_id: str = "_library",
        dataset_role: str = "train",
        density: str = "normal",
        review_status: str = "accepted",
        move: bool = False,
        notes: str = "",
        tags: list[str] | None = None,
    ) -> LibraryClip:
        source_path = Path(source_path)
        safe = safe_label(label)
        clip_uuid = _uuid.uuid4().hex
        target = self.raw_dir(label) / f"{utc_stamp()}_{safe}_0001_{clip_uuid[:8]}.wav"
        target = _unique_path(target)
        if move:
            shutil.move(str(source_path), str(target))
        else:
            shutil.copy2(str(source_path), str(target))
        clip = LibraryClip.from_sidecar(target, self.root)
        clip.uuid = clip_uuid
        clip.category = RAW_CATEGORY
        clip.label = label
        clip.project_id = project_id
        clip.dataset_role = dataset_role
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
        dataset_role: str = "train",
        density: str = "normal",
        review_status: str = "accepted",
        notes: str = "",
        tags: list[str] | None = None,
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
            target = _unique_path(
                out_dir / f"{stamp}_{safe}_{idx:04d}_{clip_uuid[:8]}.wav"
            )
            sf.write(str(target), data[start : start + win_n], sr, subtype="PCM_16")
            clip = LibraryClip.from_sidecar(target, self.root)
            clip.uuid = clip_uuid
            clip.category = RAW_CATEGORY
            clip.label = label
            clip.project_id = project_id
            clip.dataset_role = dataset_role
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
            dataset_role="train",
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
            dst = _unique_path(ch_dir / f"{ts}_{safe}_ch{c}_{parent8}_{split_uuid[:8]}.wav")
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
            clip.dataset_role = str(meta.get("dataset_role") or clip.dataset_role)
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

    def promote_mined(self, wav_path: Path, *, label: str, project_id: str) -> Path | None:
        wav_path = Path(wav_path).resolve()
        try:
            wav_path.relative_to((self.root / MINED_CATEGORY).resolve())
        except ValueError:
            return None

        sidecar = wav_path.with_suffix(".json")
        original_meta: dict[str, Any] = {}
        if sidecar.exists():
            try:
                original_meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                original_meta = {}
        clip = LibraryClip.from_sidecar(wav_path, self.root)
        clip.category = RAW_CATEGORY
        clip.label = label
        clip.project_id = project_id
        clip.dataset_role = "train"
        clip.review_status = "accepted"
        clip.origin = "runtime_mining"
        clip.source_path = str(wav_path)
        dest = _unique_path(self.raw_dir(label) / wav_path.name)
        shutil.move(str(wav_path), str(dest))
        if sidecar.exists():
            sidecar.unlink(missing_ok=True)
        clip.relpath = str(dest.relative_to(self.root))
        merged = dict(original_meta)
        merged.update(clip.to_sidecar(dest))
        dest.with_suffix(".json").write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.rebuild_manifest()
        return dest

    def remove_clip(self, clip_uuid: str) -> bool:
        clips = self.all_clips()
        matched = [c for c in clips if c.uuid == clip_uuid]
        if not matched:
            return False
        for clip in matched:
            path = self.root / clip.relpath
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
        self.rebuild_manifest()
        return True

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
                    clip = LibraryClip.from_sidecar(wav, self.root)
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
            target = _unique_path(target_dir / wav.name)
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
            migrated.dataset_role = clip.dataset_role
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
        if row.get("category") in LEGACY_CATEGORIES:
            row = dict(row)
            row["category"] = RAW_CATEGORY
            row.setdefault("project_id", "_library")
            row.setdefault("dataset_role", "train")
            row.setdefault("density", _density_from_tags(row.get("tags")) or "normal")
            row.setdefault("review_status", "accepted")
            row.setdefault("recording_counter", 0)
            row.setdefault("microphone_type", "generic")
            row.setdefault("origin", "legacy_manifest")
            row.setdefault("source_path", "")
            row.setdefault("schema_version", SCHEMA_VERSION)
            row["label"] = row.get("label") or row.get("class_name") or "unlabeled"
        return row

    def _clip_from_legacy_row(self, row: dict[str, Any]) -> LibraryClip:
        normalized = self._normalize_manifest_row(row)
        allowed = set(LibraryClip.__dataclass_fields__)
        return LibraryClip(**{k: v for k, v in normalized.items() if k in allowed})


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    n = 2
    while True:
        candidate = path.with_name(f"{stem}_{n}{suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _thumbnail_path(wav_path: Path) -> Path:
    return wav_path.with_suffix(_THUMB_SUFFIX)


def _wav_for_thumbnail(path: Path) -> Path | None:
    name = path.name
    if not name.endswith(_THUMB_SUFFIX):
        return None
    return path.with_name(name[: -len(_THUMB_SUFFIX)] + ".wav")
