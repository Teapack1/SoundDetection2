"""Lightweight index for a large pretraining corpus.

Pretraining clips live under ``data_library/pretraining/<class>/.../*.wav`` and
carry no per-clip ``.json`` sidecars. They are deliberately kept out of the main
``manifest.jsonl`` so the existing library's rebuild/prune logic — which deletes
any WAV lacking a sidecar (see ``Library.rebuild_manifest`` /
``_prune_incomplete_file_pairs``) — never touches them.

Instead they are tracked by a single ``_index.jsonl`` file under the pretraining
root: one JSON row per clip. The index is intentionally cheap to build — a
directory walk only, no audio decoding. ``sample_rate`` / ``duration_s`` are
filled lazily (and cached on the in-memory row) only when a clip is actually
composed into a dataset; the inspect previews tab never needs them.

Each row::

    {
      "relpath": "pretraining/<class>/<subdir...>/<file>.wav",  # relative to LIBRARY_DIR
      "label": "<class>",       # top-level folder under pretraining/
      "session": "<subdir>"     # remaining sub-path, or the class if flat
    }

``relpath`` is stored relative to ``LIBRARY_DIR`` (it keeps the ``pretraining/``
prefix) so a synthesized ``LibraryClip.absolute_path(LIBRARY_DIR)`` resolves
correctly through the same code path that ``compose`` uses for normal clips.
"""

from __future__ import annotations

import json
import logging
import uuid as _uuid
from pathlib import Path
from typing import Any

import soundfile as sf

from sed import config as config_mod
from sed.data.library import PRETRAIN_CATEGORY, LibraryClip

log = logging.getLogger(__name__)

_INDEX_NAME = "_index.jsonl"


def pretraining_root() -> Path:
    """Absolute path to ``data_library/pretraining`` (honors patched configs)."""
    return config_mod.LIBRARY_DIR / PRETRAIN_CATEGORY


_SHARED_INDEX: "PretrainIndex | None" = None


def shared_index() -> "PretrainIndex":
    """Process-wide PretrainIndex (its signature guard reloads on file change)."""
    global _SHARED_INDEX
    if _SHARED_INDEX is None:
        _SHARED_INDEX = PretrainIndex()
    return _SHARED_INDEX


def subbucket_key(label: str, sample_rate: int, channels: int) -> str:
    """A subbucket groups one origin folder by its audio parameters, so clips
    with a different sample rate (or channel count) split into separate
    subbuckets even when they live in the same folder."""
    key = f"{label} @ {int(sample_rate)} Hz"
    if int(channels or 1) > 1:
        key += f" · {int(channels)}ch"
    return key


class PretrainIndex:
    """Reads/writes the lightweight ``_index.jsonl`` for the pretraining corpus."""

    def __init__(self, root: Path | None = None):
        # Paths are resolved live (not cached) so a patched/changed
        # ``config.LIBRARY_DIR`` is always honored — important for tests and for
        # a long-lived singleton that must not pin a stale library root.
        self._explicit_root = Path(root) if root is not None else None
        self._rows_cache: list[dict[str, Any]] | None = None
        self._rows_cache_signature: tuple[int, int] | None = None
        # Synthesized LibraryClip list (uuid5 + dataclass per row is ~4s per
        # call at 75k rows). Guarded by the same index signature as the rows.
        self._clips_cache: "list[LibraryClip] | None" = None
        self._clips_cache_signature: tuple[int, int] | None = None

    @property
    def library_dir(self) -> Path:
        if self._explicit_root is not None:
            return self._explicit_root.parent
        return config_mod.LIBRARY_DIR

    @property
    def root(self) -> Path:
        if self._explicit_root is not None:
            return self._explicit_root
        return config_mod.LIBRARY_DIR / PRETRAIN_CATEGORY

    @property
    def index_path(self) -> Path:
        return self.root / _INDEX_NAME

    # ---- helpers ----------------------------------------------------------

    def _signature(self) -> tuple[int, int] | None:
        try:
            stat = self.index_path.stat()
        except OSError:
            return None
        return (int(stat.st_mtime_ns), int(stat.st_size))

    @staticmethod
    def _row_from_wav(
        wav: Path, library_dir: Path, *, probe: bool = True
    ) -> dict[str, Any] | None:
        try:
            relpath = wav.resolve().relative_to(library_dir.resolve())
        except ValueError:
            return None
        parts = relpath.parts  # ("pretraining", "<class>", ..., "<file>.wav")
        if len(parts) < 3:
            # File sitting directly in pretraining/ with no class folder — skip.
            return None
        label = parts[1]
        session = "/".join(parts[2:-1]) or label
        row: dict[str, Any] = {
            "relpath": str(relpath), "label": label, "session": session,
            "origin_class": label,
        }
        if probe:
            # Header read only (no decode) so we know each clip's sample rate,
            # length and channels — and can split a folder into per-parameter
            # subbuckets.
            try:
                info = sf.info(str(wav))
                sr = int(info.samplerate)
                ch = int(info.channels)
                dur = info.frames / info.samplerate if info.samplerate else 0.0
                row["sample_rate"] = sr
                row["channels"] = ch
                row["duration_s"] = round(dur, 4)
                row["subbucket"] = subbucket_key(label, sr, ch)
            except Exception as exc:  # noqa: BLE001
                log.debug("pretraining: failed to probe %s: %s", relpath, exc)
        return row

    # ---- build ------------------------------------------------------------

    def rebuild(self, *, probe: bool = True) -> list[dict[str, Any]]:
        """Walk the corpus once and (re)write ``_index.jsonl``.

        With ``probe=True`` each clip's WAV header is read (cheap, no decode) so
        ``sample_rate`` / ``duration_s`` / ``channels`` / ``subbucket`` are stored
        up front — the Library/Dataset views then show consistent per-subbucket
        parameters without re-probing.
        """
        rows: list[dict[str, Any]] = []
        if self.root.exists():
            for wav in sorted(self.root.rglob("*.wav")):
                row = self._row_from_wav(wav, self.library_dir, probe=probe)
                if row is not None:
                    rows.append(row)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_name(f"{_INDEX_NAME}.{_uuid.uuid4().hex}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        tmp.replace(self.index_path)
        self._rows_cache = rows
        self._rows_cache_signature = self._signature()
        self._clips_cache = None
        self._clips_cache_signature = None
        return list(rows)

    # ---- read -------------------------------------------------------------

    def all_rows(self, rebuild_if_missing: bool = True) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            if rebuild_if_missing and self.root.exists():
                return self.rebuild()
            return []
        signature = self._signature()
        if self._rows_cache is not None and self._rows_cache_signature == signature:
            return self._rows_cache
        rows: list[dict[str, Any]] = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("relpath") and row.get("label"):
                    rows.append(row)
        self._rows_cache = rows
        self._rows_cache_signature = signature
        return rows

    def labels_with_counts(self, rebuild_if_missing: bool = True) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.all_rows(rebuild_if_missing=rebuild_if_missing):
            label = str(row.get("label") or "")
            if label:
                counts[label] = counts.get(label, 0) + 1
        return counts

    def subbuckets_with_stats(self, rebuild_if_missing: bool = True) -> dict[str, dict[str, Any]]:
        """Per-subbucket stats (count + sample rate + duration min/max/mean).

        A subbucket is one origin folder at one set of audio parameters, so a
        folder with mixed sample rates yields multiple subbuckets — exactly what
        the Library filter lists.
        """
        stats: dict[str, dict[str, Any]] = {}
        for row in self.all_rows(rebuild_if_missing=rebuild_if_missing):
            label = str(row.get("label") or "")
            sr = int(row.get("sample_rate") or 0)
            ch = int(row.get("channels") or 1)
            key = str(row.get("subbucket") or subbucket_key(label, sr, ch))
            dur = float(row.get("duration_s") or 0.0)
            s = stats.get(key)
            if s is None:
                s = stats[key] = {
                    "subbucket": key, "label": label,
                    "origin_class": str(row.get("origin_class") or label),
                    "sample_rate": sr, "channels": ch, "count": 0,
                    "duration_total": 0.0, "duration_min": None, "duration_max": None,
                }
            s["count"] += 1
            s["duration_total"] += dur
            if dur > 0:
                s["duration_min"] = dur if s["duration_min"] is None else min(s["duration_min"], dur)
                s["duration_max"] = dur if s["duration_max"] is None else max(s["duration_max"], dur)
        for s in stats.values():
            s["duration_mean"] = (s["duration_total"] / s["count"]) if s["count"] else 0.0
        return stats

    def rows_for_label(
        self, label: str, *, offset: int = 0, limit: int | None = None
    ) -> list[dict[str, Any]]:
        matched = [row for row in self.all_rows() if str(row.get("label")) == label]
        if offset:
            matched = matched[offset:]
        if limit is not None:
            matched = matched[:limit]
        return matched

    def count_for_label(self, label: str) -> int:
        return sum(1 for row in self.all_rows() if str(row.get("label")) == label)

    def remove_paths(
        self, paths: list[str | Path] | set[str | Path],
    ) -> list[dict[str, Any]]:
        """Atomically remove exact corpus paths from the lightweight index.

        Deleting a corpus WAV without updating ``_index.jsonl`` made the next
        Dataset refresh synthesize the missing clip again.  Filtering the
        in-memory/on-disk rows is linear in the compact index and avoids the
        expensive header-probing corpus rebuild.
        """
        wanted: set[str] = set()
        library_dir = self.library_dir.resolve()
        for raw in paths:
            try:
                rel = Path(raw).expanduser().resolve().relative_to(library_dir)
            except (OSError, ValueError):
                continue
            if rel.parts and rel.parts[0] == PRETRAIN_CATEGORY:
                wanted.add(str(rel))
        if not wanted:
            return []
        if not self.index_path.exists():
            return []

        # Preserve unrelated JSONL bytes and parse only candidate lines. At
        # 75k rows, decoding and re-encoding the entire index cost more than
        # the file removal itself; the exact relpath token narrows the common
        # one-clip case to one JSON decode.
        tokens = {
            relpath: f'"relpath": {json.dumps(relpath)}'
            for relpath in wanted
        }
        removed: list[dict[str, Any]] = []
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_name(f"{_INDEX_NAME}.{_uuid.uuid4().hex}.tmp")
        with open(self.index_path, encoding="utf-8") as src, open(
            tmp, "w", encoding="utf-8",
        ) as dst:
            for line in src:
                candidates = [relpath for relpath, token in tokens.items() if token in line]
                if not candidates:
                    dst.write(line)
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    dst.write(line)
                    continue
                if str(row.get("relpath") or "") not in wanted:
                    dst.write(line)
                    continue
                removed.append(row)
            dst.flush()
        if not removed:
            tmp.unlink(missing_ok=True)
            return []
        tmp.replace(self.index_path)
        if self._rows_cache is not None:
            self._rows_cache = [
                row for row in self._rows_cache
                if str(row.get("relpath") or "") not in wanted
            ]
        self._rows_cache_signature = self._signature()
        self._clips_cache = None
        self._clips_cache_signature = None
        return removed

    def abs_path(self, row: dict[str, Any]) -> Path:
        return self.library_dir / str(row.get("relpath") or "")

    # ---- clip synthesis ---------------------------------------------------

    def clip_for_row(self, row: dict[str, Any], *, probe: bool = True) -> LibraryClip:
        """Synthesize a ``LibraryClip`` from an index row.

        With ``probe=True`` the audio header is read (once) to fill
        ``sample_rate`` / ``duration_s`` and the result is cached back onto the
        in-memory ``row`` so repeated lookups within a run don't re-decode.
        """
        relpath = str(row.get("relpath") or "")
        label = str(row.get("label") or "")
        session = str(row.get("session") or label)
        sample_rate = int(row.get("sample_rate") or 0)
        duration_s = float(row.get("duration_s") or 0.0)
        if probe and (sample_rate <= 0 or duration_s <= 0.0):
            wav = self.library_dir / relpath
            try:
                info = sf.info(str(wav))
                sample_rate = int(info.samplerate)
                duration_s = info.frames / info.samplerate if info.samplerate else 0.0
                row["sample_rate"] = sample_rate
                row["duration_s"] = duration_s
            except Exception as exc:  # noqa: BLE001
                log.debug("pretraining: failed to probe %s: %s", relpath, exc)
        return LibraryClip(
            uuid=_uuid.uuid5(_uuid.NAMESPACE_URL, relpath).hex,
            relpath=relpath,
            label=label,
            category=PRETRAIN_CATEGORY,
            project_id="_library",
            review_status="accepted",
            duration_s=duration_s,
            sample_rate=sample_rate,
            channels=1,
            clip_kind="mono",
            session_id=session,
            origin="pretraining",
            origin_class=str(row.get("origin_class") or label),
        )

    def all_clips(self, *, probe: bool = True) -> list[LibraryClip]:
        """All corpus rows synthesized to clips, cached per index signature."""
        rows = self.all_rows()
        signature = self._rows_cache_signature
        if self._clips_cache is not None and self._clips_cache_signature == signature:
            return self._clips_cache
        clips = [self.clip_for_row(row, probe=probe) for row in rows]
        self._clips_cache = clips
        self._clips_cache_signature = signature
        return clips

    def clips_for_labels(self, labels: list[str], *, probe: bool = True) -> list[LibraryClip]:
        wanted = {str(lbl) for lbl in labels if lbl}
        clips = self.all_clips(probe=probe)
        if not wanted:
            return list(clips)
        return [clip for clip in clips if clip.label in wanted]
