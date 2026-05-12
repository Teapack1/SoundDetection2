"""Shared, signature-aware spectrogram preview cache.

The inspect UI can contain thousands of cards.  Preview generation therefore
lives outside the request/render path: one cache entry is shared for an audio
file + feature/render signature, and a small priority worker pool serialises
the expensive work without making the browser stampede the server.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image

from sed.audio.features import (
    display_range_for_spec,
    waveform_to_feature,
)
log = logging.getLogger(__name__)

CACHE_DIRNAME = ".preview_cache"
CACHE_LAYOUT_VERSION = "palette-png-v3-jet-rgb-freq-up"


def preview_features_from_parts(features_cfg, sample_rate) -> dict[str, Any]:
    """Canonical preview contract from explicit feature parts.

    Lets dataset previews render from a preview-only feature (chosen in the
    Library tab) that is independent of the model/training ``[features]`` config.
    """
    from sed.audio.features import pcen_params_from_features

    f = features_cfg
    return {
        "sample_rate": int(sample_rate),
        "frontend": str(f.frontend),
        "feature_type": str(f.feature_type),
        "n_fft": int(f.n_fft),
        "win_length": int(f.win_length),
        "hop_length": int(f.hop_length),
        "n_mels": int(f.n_mels),
        "fmin": int(f.fmin),
        "fmax": int(f.fmax),
        "log_power": bool(f.log_power),
        "stft_engine": str(getattr(f, "stft_engine", "librosa")),
        "db_norm": bool(getattr(f, "db_norm", False)),
        "db_min": float(getattr(f, "db_min", -20.0)),
        "db_max": float(getattr(f, "db_max", 60.0)),
        "colormap": str(getattr(f, "colormap", "none")),
        "image_size": int(getattr(f, "image_size", 96)),
        "pcen": bool(f.pcen),
        "pcen_params": pcen_params_from_features(f),
        "layout": CACHE_LAYOUT_VERSION,
        "palette_colors": 128,
    }


def preview_features_from_config(cfg) -> dict[str, Any]:
    """Canonical preview contract for a loaded project config."""
    return preview_features_from_parts(cfg.features, cfg.model.sample_rate)


def preview_signature(features: dict[str, Any]) -> str:
    raw = json.dumps(features, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _cache_pixels(spec: np.ndarray) -> np.ndarray:
    """Keep time horizontal and widen short time axes for card previews."""
    arr = np.asarray(spec)
    if arr.ndim != 2:
        return arr
    bands, frames = arr.shape
    if bands <= 0 or frames <= 0 or frames >= bands:
        return arr
    idx = np.rint(np.linspace(0, frames - 1, bands)).astype(np.intp)
    return arr[:, idx]


def _display_pixels(spec: np.ndarray) -> np.ndarray:
    """Image pixels for previews: top row is highest frequency."""
    pixels = _cache_pixels(spec)
    if np.asarray(pixels).ndim != 2:
        return pixels
    return np.flipud(pixels)


def _feature_kwargs(features: dict[str, Any]) -> dict[str, Any]:
    common: dict[str, Any] = {
        "n_fft": int(features["n_fft"]),
        "win_length": int(features["win_length"]),
        "hop_length": int(features["hop_length"]),
        "fmin": int(features["fmin"]),
        "fmax": int(features["fmax"]),
        "pcen": bool(features.get("pcen", False)),
        "pcen_params": features.get("pcen_params"),
    }
    if str(features.get("feature_type")) == "linear":
        return {
            **common,
            "log_power": bool(features.get("log_power", True)),
            "stft_engine": str(features.get("stft_engine", "librosa")),
            "db_norm": bool(features.get("db_norm", False)),
            "db_min": float(features.get("db_min", -20.0)),
            "db_max": float(features.get("db_max", 60.0)),
        }
    return {
        **common,
        "frontend": str(features["frontend"]),
        "n_mels": int(features["n_mels"]),
    }


def _render_palette_png(wav_path: Path, features: dict[str, Any], output: Path) -> int:
    """Render a compact indexed PNG and replace the destination atomically."""
    from matplotlib import colormaps

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    target_sr = int(features["sample_rate"])
    if int(sr) != target_sr:
        from sed.audio.resample import resample_audio

        data = resample_audio(data, orig_sr=int(sr), target_sr=target_sr)
    feature_type = str(features.get("feature_type") or "mel")
    spec = waveform_to_feature(
        data,
        target_sr,
        feature_type=feature_type,
        **_feature_kwargs(features),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    colormap = str(features.get("colormap") or "none").lower()
    if colormap != "none":
        # A colormap image feature: the thumbnail IS the 3-channel image the
        # model consumes (colormap + transpose + resize), rendered directly.
        from sed.audio.features import render_spectrogram_image

        img = render_spectrogram_image(spec, colormap, int(features.get("image_size", 96)))
        rgb = (np.clip(np.transpose(img, (1, 2, 0)), 0.0, 1.0) * 255).astype(np.uint8)
        image = Image.fromarray(rgb, "RGB").quantize(
            colors=int(features.get("palette_colors") or 128),
            method=Image.Quantize.MEDIANCUT,
        )
        try:
            image.save(temporary, format="PNG", optimize=True)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return output.stat().st_size
    vmin, vmax = display_range_for_spec(
        spec,
        feature_type,
        bool(features.get("log_power", True)),
        bool(features.get("pcen", False)),
        db_norm=bool(features.get("db_norm", False)),
    )
    pixels = _display_pixels(spec)
    lo = float(vmin) if vmin is not None else float(np.nanpercentile(pixels, 1.0))
    hi = float(vmax) if vmax is not None else float(np.nanpercentile(pixels, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    normal = np.clip((pixels - lo) / (hi - lo), 0.0, 1.0)
    rgba = colormaps["magma"](normal, bytes=True)
    image = Image.fromarray(rgba, "RGBA").convert("RGB").quantize(
        colors=int(features.get("palette_colors") or 128),
        method=Image.Quantize.MEDIANCUT,
    )
    try:
        image.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output.stat().st_size


@dataclass(frozen=True)
class PreviewResult:
    path: Path
    signature: str
    created: bool
    size_bytes: int


class PreviewCache:
    """Persistent cache/index shared by all local projects."""

    def __init__(self, root: Path | None = None):
        # Resolve this lazily.  Tests and the inspect server can change the
        # configured library directory after this module has been imported.
        if root is None:
            from sed.config import LIBRARY_DIR

            root = LIBRARY_DIR
        self.root = Path(root) / CACHE_DIRNAME
        self.db_path = self.root / "index.sqlite3"
        self._db_lock = threading.RLock()
        self._key_lock_guard = threading.Lock()
        self._key_locks: dict[tuple[str, str], tuple[threading.Lock, int]] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._db_lock, self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS variants (
                    signature TEXT PRIMARY KEY,
                    features_json TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assets (
                    signature TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_mtime_ns INTEGER NOT NULL,
                    source_size INTEGER NOT NULL,
                    cache_path TEXT NOT NULL,
                    bytes INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (signature, source_path)
                );
                CREATE TABLE IF NOT EXISTS project_assets (
                    project_id TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    PRIMARY KEY (project_id, signature, source_path)
                );
                CREATE INDEX IF NOT EXISTS idx_project_assets_variant
                    ON project_assets(project_id, signature);
                """
            )

    @staticmethod
    def _source_key(wav_path: Path) -> str:
        return str(wav_path.resolve())

    @staticmethod
    def _source_stat(wav_path: Path) -> tuple[int, int]:
        stat = wav_path.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)

    @staticmethod
    def _asset_name(source_key: str) -> str:
        return hashlib.sha256(source_key.encode("utf-8")).hexdigest() + ".png"

    def asset_path(self, wav_path: Path, signature: str) -> Path:
        return self.root / signature / self._asset_name(self._source_key(wav_path))

    def _key_lock(self, key: tuple[str, str]) -> threading.Lock:
        with self._key_lock_guard:
            entry = self._key_locks.get(key)
            if entry is None:
                lock = threading.Lock()
                self._key_locks[key] = (lock, 1)
                return lock
            lock, users = entry
            self._key_locks[key] = (lock, users + 1)
            return lock

    def _release_key_lock(self, key: tuple[str, str], lock: threading.Lock) -> None:
        with self._key_lock_guard:
            current = self._key_locks.get(key)
            if current is None or current[0] is not lock:
                return
            users = current[1] - 1
            if users <= 0:
                self._key_locks.pop(key, None)
            else:
                self._key_locks[key] = (lock, users)

    def register_variant(self, features: dict[str, Any], label: str = "") -> str:
        signature = preview_signature(features)
        with self._db_lock, self._connect() as con:
            con.execute(
                """
                INSERT INTO variants(signature, features_json, label, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    label=CASE WHEN excluded.label <> '' THEN excluded.label ELSE variants.label END
                """,
                (signature, json.dumps(features, sort_keys=True), label, time.time()),
            )
        return signature

    def _is_current(self, con: sqlite3.Connection, signature: str, source: str, mtime: int, size: int) -> PreviewResult | None:
        row = con.execute(
            "SELECT cache_path, source_mtime_ns, source_size, bytes FROM assets WHERE signature=? AND source_path=?",
            (signature, source),
        ).fetchone()
        if row is None or int(row["source_mtime_ns"]) != mtime or int(row["source_size"]) != size:
            return None
        path = Path(str(row["cache_path"]))
        if not path.exists():
            return None
        return PreviewResult(path, signature, False, int(row["bytes"]))

    def lookup(self, wav_path: Path, features: dict[str, Any], project_id: str = "") -> PreviewResult | None:
        try:
            source = self._source_key(wav_path)
            mtime, size = self._source_stat(wav_path)
        except OSError:
            return None
        signature = self.register_variant(features)
        with self._db_lock, self._connect() as con:
            hit = self._is_current(con, signature, source, mtime, size)
            if hit is not None and project_id:
                con.execute(
                    "INSERT OR IGNORE INTO project_assets(project_id, signature, source_path) VALUES (?, ?, ?)",
                    (project_id, signature, source),
                )
            return hit

    def ensure(self, wav_path: Path, features: dict[str, Any], project_id: str = "", label: str = "") -> PreviewResult:
        wav_path = Path(wav_path).resolve()
        source = self._source_key(wav_path)
        signature = self.register_variant(features, label=label)
        key = (signature, source)
        lock = self._key_lock(key)
        try:
            with lock:
                hit = self.lookup(wav_path, features, project_id=project_id)
                if hit is not None:
                    return hit
                mtime, size = self._source_stat(wav_path)
                output = self.asset_path(wav_path, signature)
                bytes_written = _render_palette_png(wav_path, features, output)
                with self._db_lock, self._connect() as con:
                    con.execute(
                        """
                        INSERT INTO assets(signature, source_path, source_mtime_ns, source_size, cache_path, bytes, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(signature, source_path) DO UPDATE SET
                            source_mtime_ns=excluded.source_mtime_ns,
                            source_size=excluded.source_size,
                            cache_path=excluded.cache_path,
                            bytes=excluded.bytes,
                            updated_at=excluded.updated_at
                        """,
                        (signature, source, mtime, size, str(output), bytes_written, time.time()),
                    )
                    if project_id:
                        con.execute(
                            "INSERT OR IGNORE INTO project_assets(project_id, signature, source_path) VALUES (?, ?, ?)",
                            (project_id, signature, source),
                        )
                return PreviewResult(output, signature, True, bytes_written)
        finally:
            self._release_key_lock(key, lock)

    def migrate_legacy(self, wav_path: Path, features: dict[str, Any], *, project_id: str = "", label: str = "") -> PreviewResult | None:
        """Move one valid legacy sibling ``.spec.png`` into this variant.

        Older inspect versions placed previews beside source audio.  We only
        adopt a thumbnail at least as new as its source, re-pack it as an
        indexed PNG, and remove the sibling after the cache/index write.  A
        stale legacy preview is left alone: normal generation will replace it
        with the feature-signature-correct result.
        """
        wav_path = Path(wav_path).resolve()
        legacy = wav_path.with_suffix(".spec.png")
        try:
            if not legacy.is_file():
                return None
            if legacy.stat().st_mtime_ns < wav_path.stat().st_mtime_ns:
                # It cannot describe this source revision.  The caller will
                # render a current cache entry; remove the obsolete duplicate.
                legacy.unlink(missing_ok=True)
                return None
            source = self._source_key(wav_path)
            mtime, size = self._source_stat(wav_path)
        except OSError:
            return None
        signature = self.register_variant(features, label=label)
        key = (signature, source)
        lock = self._key_lock(key)
        try:
            with lock:
                hit = self.lookup(wav_path, features, project_id=project_id)
                if hit is not None:
                    legacy.unlink(missing_ok=True)
                    return hit
                output = self.asset_path(wav_path, signature)
                output.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(legacy) as old:
                    image = old.convert("RGB").quantize(
                        colors=int(features.get("palette_colors") or 128),
                        method=Image.Quantize.MEDIANCUT,
                    )
                    temporary = output.with_name(
                        f".{output.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                    )
                    try:
                        image.save(temporary, format="PNG", optimize=True)
                        os.replace(temporary, output)
                    finally:
                        temporary.unlink(missing_ok=True)
                bytes_written = output.stat().st_size
                with self._db_lock, self._connect() as con:
                    con.execute(
                        """
                        INSERT INTO assets(signature, source_path, source_mtime_ns, source_size, cache_path, bytes, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(signature, source_path) DO UPDATE SET
                            source_mtime_ns=excluded.source_mtime_ns,
                            source_size=excluded.source_size,
                            cache_path=excluded.cache_path,
                            bytes=excluded.bytes,
                            updated_at=excluded.updated_at
                        """,
                        (signature, source, mtime, size, str(output), bytes_written, time.time()),
                    )
                    if project_id:
                        con.execute(
                            "INSERT OR IGNORE INTO project_assets(project_id, signature, source_path) VALUES (?, ?, ?)",
                            (project_id, signature, source),
                        )
                legacy.unlink(missing_ok=True)
                return PreviewResult(output, signature, True, bytes_written)
        except Exception as exc:  # noqa: BLE001
            log.warning("legacy preview migration failed for %s: %s", wav_path, exc)
            return None
        finally:
            self._release_key_lock(key, lock)

    def variants_for_project(self, project_id: str) -> list[dict[str, Any]]:
        with self._db_lock, self._connect() as con:
            rows = con.execute(
                """
                SELECT v.signature, v.label, v.features_json, v.created_at,
                       COUNT(pa.source_path) AS clips, COALESCE(SUM(a.bytes), 0) AS bytes
                FROM variants v
                JOIN project_assets pa ON pa.signature=v.signature
                LEFT JOIN assets a ON a.signature=pa.signature AND a.source_path=pa.source_path
                WHERE pa.project_id=?
                GROUP BY v.signature
                ORDER BY v.created_at DESC
                """,
                (project_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                features = json.loads(str(row["features_json"]))
            except json.JSONDecodeError:
                features = {}
            out.append({
                "signature": str(row["signature"]), "label": str(row["label"] or ""),
                "features": features, "clips": int(row["clips"]), "bytes": int(row["bytes"]),
            })
        return out

    def delete_project_variant(self, project_id: str, signature: str) -> dict[str, int]:
        with self._db_lock, self._connect() as con:
            rows = con.execute(
                "SELECT source_path FROM project_assets WHERE project_id=? AND signature=?",
                (project_id, signature),
            ).fetchall()
            con.execute("DELETE FROM project_assets WHERE project_id=? AND signature=?", (project_id, signature))
            removed_files = 0
            freed = 0
            for row in rows:
                source = str(row["source_path"])
                shared = con.execute(
                    "SELECT 1 FROM project_assets WHERE signature=? AND source_path=? LIMIT 1",
                    (signature, source),
                ).fetchone()
                if shared:
                    continue
                asset = con.execute(
                    "SELECT cache_path, bytes FROM assets WHERE signature=? AND source_path=?",
                    (signature, source),
                ).fetchone()
                if asset:
                    Path(str(asset["cache_path"])).unlink(missing_ok=True)
                    freed += int(asset["bytes"])
                    removed_files += 1
                con.execute("DELETE FROM assets WHERE signature=? AND source_path=?", (signature, source))
        return {"removed": len(rows), "files": removed_files, "bytes": freed}

    def remove_sources(self, paths: list[Path]) -> int:
        source_paths = [self._source_key(path) for path in paths]
        removed = 0
        with self._db_lock, self._connect() as con:
            for source in source_paths:
                rows = con.execute("SELECT cache_path FROM assets WHERE source_path=?", (source,)).fetchall()
                for row in rows:
                    Path(str(row["cache_path"])).unlink(missing_ok=True)
                    removed += 1
                con.execute("DELETE FROM project_assets WHERE source_path=?", (source,))
                con.execute("DELETE FROM assets WHERE source_path=?", (source,))
        return removed

    def clean_orphans(self) -> dict[str, int]:
        removed = 0
        freed = 0
        with self._db_lock, self._connect() as con:
            rows = con.execute("SELECT signature, source_path, cache_path, bytes FROM assets").fetchall()
            for row in rows:
                source = Path(str(row["source_path"]))
                cache = Path(str(row["cache_path"]))
                if source.exists() and cache.exists():
                    continue
                cache.unlink(missing_ok=True)
                freed += int(row["bytes"])
                removed += 1
                con.execute(
                    "DELETE FROM project_assets WHERE signature=? AND source_path=?",
                    (str(row["signature"]), str(row["source_path"])),
                )
                con.execute(
                    "DELETE FROM assets WHERE signature=? AND source_path=?",
                    (str(row["signature"]), str(row["source_path"])),
                )
        return {"removed": removed, "bytes": freed}


@dataclass(order=True)
class _QueuedPreview:
    priority: int
    order: int
    wav_path: Path
    features: dict[str, Any]
    project_id: str
    label: str
    future: Future


class PreviewWorkerPool:
    """Two-worker priority queue shared by visible cards and cache builds."""

    def __init__(self, cache: PreviewCache, workers: int = 2):
        self.cache = cache
        self._queue: queue.PriorityQueue[_QueuedPreview] = queue.PriorityQueue()
        self._lock = threading.Lock()
        self._counter = 0
        self._inflight: dict[tuple[str, str], Future] = {}
        self._inflight_priorities: dict[tuple[str, str], int] = {}
        self._threads = [
            threading.Thread(target=self._run, name=f"sed-preview-{idx}", daemon=True)
            for idx in range(max(1, workers))
        ]
        for thread in self._threads:
            thread.start()

    def submit(
        self, wav_path: Path, features: dict[str, Any], *, project_id: str = "",
        label: str = "", priority: int = 10,
    ) -> Future:
        source = str(Path(wav_path).resolve())
        signature = preview_signature(features)
        key = (signature, source)
        hit = self.cache.lookup(Path(wav_path), features, project_id=project_id)
        if hit is not None:
            ready: Future = Future()
            ready.set_result(hit)
            return ready
        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                # A visible card may arrive while this item is waiting in a
                # background build.  Put the same Future back into the queue
                # at the higher priority; set_running_or_notify_cancel in the
                # worker makes the losing queue entry a no-op, so decoding is
                # still performed once.
                old_priority = self._inflight_priorities.get(key, priority)
                if priority < old_priority and not existing.running():
                    self._inflight_priorities[key] = priority
                    self._counter += 1
                    self._queue.put(_QueuedPreview(
                        priority, self._counter, Path(wav_path), dict(features),
                        project_id, label, existing,
                    ))
                if project_id:
                    def associate(_done: Future) -> None:
                        if not _done.cancelled() and _done.exception() is None:
                            self.cache.lookup(Path(wav_path), features, project_id=project_id)
                    existing.add_done_callback(associate)
                return existing
            future: Future = Future()
            self._inflight[key] = future
            self._inflight_priorities[key] = priority
            self._counter += 1
            self._queue.put(_QueuedPreview(
                priority, self._counter, Path(wav_path), dict(features), project_id, label, future,
            ))
            return future

    def _run(self) -> None:
        while True:
            work = self._queue.get()
            source = str(work.wav_path.resolve())
            key = (preview_signature(work.features), source)
            try:
                # The same future may appear twice when a visible request
                # promotes background work.  Only the first worker owns it.
                try:
                    owns_future = work.future.set_running_or_notify_cancel()
                except RuntimeError:
                    # Another worker won the promoted duplicate entry.
                    owns_future = False
                if not owns_future:
                    continue
                work.future.set_result(self.cache.ensure(
                    work.wav_path, work.features,
                    project_id=work.project_id, label=work.label,
                ))
            except Exception as exc:  # noqa: BLE001
                log.warning("preview generation failed for %s: %s", work.wav_path, exc)
                if work.future.running():
                    work.future.set_exception(exc)
            finally:
                with self._lock:
                    if self._inflight.get(key) is work.future and work.future.done():
                        self._inflight.pop(key, None)
                        self._inflight_priorities.pop(key, None)
                self._queue.task_done()
