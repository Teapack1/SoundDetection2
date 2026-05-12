"""Offline audio augmentation pipeline.

Reads source recordings from project-selected library clips,
applies the canonical overlay transformation, writes each result to
``data_library/derived/<project>/<role>/<class>/`` with a sidecar JSON naming the technique
used and its parameters.

Designed for human review via ``sed dataset inspect --serve`` *before* training.

Technique implemented:
    overlay          Optional pitch shift, then transient-aware negative overlay

Temporal jitter is intentionally NOT done here — it lives at recording time
via [onset] randomize_preroll. Post-processing rolls/pads would wrap or
zero-pad the transient and corrupt the spectrogram around the attack.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable

import numpy as np
import soundfile as sf
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn

from sed import config as config_mod
from sed.audio.features import (
    _mel_power_np,
    kaldi_mel_center_freqs,
    window_sample_count,
)
from sed.audio.resample import resample_audio
from sed.config import Config, ProjectLayout
from sed.data.library import (
    DERIVED_CATEGORY,
    Library,
    SCHEMA_VERSION,
    safe_label,
)
from sed.data.source_filters import (
    augmentation_settings_snapshot,
    library_paths_for_filters,
)
from sed.util import unique_path



log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _slug_part(value: str, max_len: int = 32) -> str:
    """Make a short filename component that still reads well in the inspector."""
    value = Path(value).stem if value else ""
    cleaned = "".join(c.lower() if c.isalnum() else "_" for c in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return (cleaned[:max_len].strip("_") or "clip")


def _default_run_prefix(seed: int) -> str:
    return f"aug-{datetime.now().strftime('%Y%m%d-%H%M%S')}-s{seed}"


_AUGMENTATION_START_CACHE: dict[str, str] = {}


def _augmentation_started_at(run_prefix: str) -> str:
    """Return one stable start time for every sample in an augmentation run."""
    raw = str(run_prefix or "")
    if raw in _AUGMENTATION_START_CACHE:
        return _AUGMENTATION_START_CACHE[raw]
    try:
        stamp = raw.split("aug-", 1)[1].split("-s", 1)[0]
        value = datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=UTC).isoformat()
    except (IndexError, ValueError):
        value = datetime.now(UTC).isoformat()
    _AUGMENTATION_START_CACHE[raw] = value
    return value


class AugmentationCancelled(RuntimeError):
    """Raised when a UI augmentation job is cooperatively stopped."""


class AugmentationDisabled(RuntimeError):
    """Raised when augmentation is requested but disabled in the config.

    Surfaces a clear error in the UI rather than silently returning zero clips.
    """


_AUG_TUPLE_FIELDS = {"semitones_range", "snr_db_range"}


def _apply_settings_dict(cfg: Config, settings: dict) -> None:
    """Apply a recipe settings snapshot dict to cfg.augmentation in place.

    Used by `materialize_library_recipes` to swap each recipe's snapshotted
    knobs into the cfg before running, so two recipes with different SNR or
    mask thresholds produce different clips in the same job. Quietly skips
    keys that don't exist on the dataclass (forward-compat for new fields).
    """
    if not isinstance(settings, dict) or not settings:
        return
    for key, value in settings.items():
        if not hasattr(cfg.augmentation, key):
            continue
        if key in _AUG_TUPLE_FIELDS and isinstance(value, list):
            try:
                value = tuple(value)
            except Exception:  # noqa: BLE001
                continue
        try:
            setattr(cfg.augmentation, key, value)
        except Exception:  # noqa: BLE001
            continue


_RECIPE_FILTER_VOLATILE_KEYS = {
    "id",
    "count",
    "updated_at",
    "created_at",
    "base_count",
    "overlay_count",
    "expected_count",
}


def _canonical_signature_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _canonical_signature_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            if val not in (None, "", [], {})
        }
    if isinstance(value, (list, tuple, set)):
        values = [_canonical_signature_value(item) for item in value]
        if all(not isinstance(item, dict) for item in values):
            return sorted(values, key=lambda item: json.dumps(item, sort_keys=True, default=str))
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return value


def _canonical_filter_rule(rule: dict) -> dict:
    return {
        str(key): _canonical_signature_value(value)
        for key, value in sorted(rule.items(), key=lambda item: str(item[0]))
        if (
            not str(key).startswith("_")
            and str(key) not in _RECIPE_FILTER_VOLATILE_KEYS
            and value not in (None, "", [], {})
        )
    }


def _canonical_filter_rules(filters: list[dict]) -> list[dict]:
    rows = [_canonical_filter_rule(rule) for rule in filters if isinstance(rule, dict)]
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"), default=str))


def _canonical_recipe_settings(settings: dict) -> dict:
    if not isinstance(settings, dict):
        return {}
    return {
        str(key): _canonical_signature_value(value)
        for key, value in sorted(settings.items(), key=lambda item: str(item[0]))
        if not str(key).startswith("_") and value not in (None, "", [], {})
    }


def recipe_signature(recipe: dict, settings: dict, *, frozen_test: bool = False) -> str:
    payload = {
        "recipe": {
            "base_filters": _canonical_filter_rules(
                recipe.get("base_filters") or ([recipe.get("base_filter")] if recipe.get("base_filter") else [])
            ),
            "overlay_filters": _canonical_filter_rules(
                recipe.get("overlay_filters") or ([recipe.get("overlay_filter")] if recipe.get("overlay_filter") else [])
            ),
            "description": recipe.get("description") or "",
            "tags": _canonical_signature_value(recipe.get("tags") or []),
        },
        "settings": _canonical_recipe_settings(settings),
        "frozen_test": bool(frozen_test),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _recipe_labels_from_filters(recipe: dict, key: str) -> set[str]:
    labels: set[str] = set()
    for rule in _recipe_filters(recipe, key):
        raw = rule.get("labels", rule.get("label"))
        if isinstance(raw, str):
            if raw and raw != "*":
                labels.add(raw)
        elif isinstance(raw, list):
            labels.update(str(item) for item in raw if str(item) and str(item) != "*")
    fallback = str(recipe.get(key) or "").strip()
    if fallback and fallback not in {"*", "pitch"}:
        labels.add(fallback)
    return labels


def source_names_for_recipe_paths(paths: list[tuple[Path, str]]) -> set[str]:
    names: set[str] = set()
    for path, _label in paths:
        for value in (path, path.resolve()):
            names.add(str(value))
            names.add(value.name)
        names.add(_source_display_name(path))
    return {name for name in names if name}


def _settings_compatible_with_sidecar(meta: dict, current_settings: dict) -> bool:
    stored = meta.get("augmentation_settings")
    if not isinstance(stored, dict) or not stored:
        return False
    current = _canonical_recipe_settings(current_settings)
    for key, value in _canonical_recipe_settings(stored).items():
        if key in current and current[key] != value:
            return False
    return True


def generated_clip_matches_recipe(
    meta: dict,
    recipe: dict,
    settings: dict,
    *,
    frozen_test: bool = False,
    base_sources: set[str] | None = None,
    overlay_sources: set[str] | None = None,
) -> bool:
    """Return whether a generated sidecar still belongs to this recipe.

    New clips use the stable ``recipe_signature``. Older clips may have been
    signed with volatile UI fields like rule counts; for those, fall back to
    checking the stored augmentation settings, labels, and source filenames.
    """
    params = meta.get("augmentation_params") if isinstance(meta.get("augmentation_params"), dict) else {}
    expected_sig = recipe_signature(recipe, settings, frozen_test=frozen_test)
    sidecar_sig = str(meta.get("recipe_signature") or params.get("recipe_signature") or "")
    if sidecar_sig and sidecar_sig == expected_sig:
        return True
    if not _settings_compatible_with_sidecar(meta, settings):
        return False

    base_label = str(meta.get("base_label") or params.get("base_label") or meta.get("label") or "")
    overlay_label = str(meta.get("overlay_label") or params.get("overlay_label") or "")
    base_labels = _recipe_labels_from_filters(recipe, "base")
    overlay_labels = _recipe_labels_from_filters(recipe, "overlay")
    if base_labels and base_label and base_label not in base_labels:
        return False
    if overlay_labels and overlay_label and overlay_label not in overlay_labels:
        return False

    base_source = str(meta.get("base_source") or params.get("base_source") or meta.get("source") or params.get("source") or "")
    overlay_source = str(meta.get("overlay_source") or params.get("overlay_source") or params.get("negative_source") or "")
    if base_sources and base_source and base_source not in base_sources:
        return False
    overlay_enabled = bool(settings.get("overlay_enabled", True))
    if overlay_enabled and overlay_sources and overlay_source and overlay_source not in overlay_sources:
        return False
    return True


# Generated stems already end in a zero-padded index, so a name collision must
# not append a bare ``_2`` — `_next_augmentation_index` below strips the prefix
# and reads everything up to the first ``__`` as that index, and "0001_2" is not
# a number. A doubled separator keeps the index parseable.
_AUG_COLLISION_SEPARATOR = "__"


def _augmentation_filename(
    run_prefix: str,
    class_name: str,
    idx: int,
) -> str:
    prefix = _slug_part(run_prefix, max_len=40)
    cls = _slug_part(class_name, max_len=36)
    return f"{prefix}_aug_{cls}_{idx + 1:04d}.wav"


def _next_augmentation_index(out_dir: Path, run_prefix: str, class_name: str) -> int:
    prefix = f"{_slug_part(run_prefix, max_len=40)}_aug_{_slug_part(class_name, max_len=36)}_"
    next_idx = 0
    if not out_dir.exists():
        return next_idx
    for wav in out_dir.glob(f"{prefix}*.wav"):
        number = wav.stem[len(prefix):].split("__", 1)[0]
        if number.isdigit():
            next_idx = max(next_idx, int(number))
    return next_idx




def _source_origin(path: Path, layout: ProjectLayout) -> str:
    try:
        path.resolve().relative_to(config_mod.LIBRARY_DIR.resolve())
        return "library"
    except ValueError:
        pass
    try:
        path.resolve().relative_to(layout.recordings_dir.resolve())
        return "project"
    except ValueError:
        return "external"


def _source_display_name(path: Path) -> str:
    """Prefer the original filename preserved in a library sidecar."""
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            source = meta.get("source_path") or meta.get("source")
            if source:
                return Path(str(source)).name
        except Exception:  # noqa: BLE001
            pass
    return path.name


def _derived_provenance(source: Path, augmentation_family: str) -> dict[str, str]:
    """Provenance inherited by every variant made from ``source``.

    ``source_split_group`` is deliberately inherited recursively: generating a
    variant from an already-derived clip must not create a new independent
    recording group.  Legacy sidecars did not carry these explicit fields, so
    their stable recording/group ids and finally their source stem provide a
    compatible fallback.
    """
    meta: dict = {}
    try:
        sidecar = source.with_suffix(".json")
        if sidecar.exists():
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
    except Exception:  # noqa: BLE001 - provenance must not abort generation
        meta = {}

    prior_source = str(
        meta.get("original_source")
        or meta.get("source_path")
        or meta.get("source")
        or source.resolve()
    )
    source_group = str(
        meta.get("source_split_group")
        or meta.get("group_id")
        or meta.get("session_id")
        or meta.get("parent_uuid")
        or meta.get("uuid")
        or Path(prior_source).stem
        or source.stem
    )
    return {
        "original_source": prior_source,
        "augmentation_family": str(augmentation_family or "derived"),
        "source_split_group": source_group,
    }


def _get_n_samples(rate: float, rng: random.Random) -> int:
    """Handle fractional samples_per_source (e.g. 0.5 = 50% chance per source)."""
    n_whole = int(rate)
    remainder = rate - n_whole
    if rng.random() < remainder:
        return n_whole + 1
    return n_whole


def _draw_snr_db(
    snr_db_range: tuple[float, float],
    rng: random.Random,
    min_snr_db: float | None = None,
) -> float:
    """Draw an SNR while honoring an optional per-class floor."""
    lo, hi = snr_db_range
    if min_snr_db is not None:
        lo = max(lo, float(min_snr_db))
        hi = max(hi, lo)
    return float(rng.uniform(lo, hi))


def _load_native(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float32 at its native sample rate."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return wave.astype(np.float32), sr


def _ensure_length(
    x: np.ndarray, target_n: int, rng: random.Random | None = None
) -> np.ndarray:
    """Crop if too long, zero-pad if too short.

    With ``rng`` both the crop start (long clips) and the pad offset (short
    clips) are drawn from it, so augmentation runs stay seed-reproducible —
    short bases land at a random offset inside the window and the overlay
    fills the rest with background, so neither the pad position nor the
    transient position is a constant cue. Without ``rng`` the result is fully
    deterministic — center crop / pad at the start — which keeps
    train/val/test dataset loading stable across epochs and evaluations.
    """
    if len(x) >= target_n:
        max_start = len(x) - target_n
        start = rng.randint(0, max_start) if rng is not None else max_start // 2
        return x[start : start + target_n]
    out = np.zeros(target_n, dtype=np.float32)
    offset = rng.randint(0, target_n - len(x)) if rng is not None else 0
    out[offset : offset + len(x)] = x
    return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def _mono(wave: np.ndarray) -> np.ndarray:
    if wave.ndim > 1:
        wave = wave.mean(axis=-1)
    return wave.astype(np.float32, copy=False)


# Defaults match the repository's EfficientAT feature contract.  Augmentation
# diagnostics must see the same value distribution as training: 0.97
# pre-emphasis, symmetric Hann STFT, Kaldi/HTK triangular banks, and the exact
# configured frequency limits and FFT geometry.  ``_mel_power_np`` is the
# feature module's shared no-log implementation of that contract.
_MEL_DEFAULTS: dict = {
    "sample_rate": 32000,
    "n_mels": 128,
    "n_fft": 1024,
    "win_length": 800,
    "hop_length": 320,
    "fmin": 0.0,
    "fmax": 15000.0,
}
_TRANSIENT_LOCATOR_WINDOW_MS = 100

_active_mel: dict = dict(_MEL_DEFAULTS)


def set_augment_features(cfg: Config | None) -> None:
    """Route augmentation analysis onto the active EfficientAT geometry.

    Materialization resamples every input to ``model.sample_rate`` before this
    analysis runs, so keeping the configured sample counts (rather than a
    rounded millisecond approximation) gives exact training/analysis parity.
    """
    global _active_mel
    if cfg is None:
        _active_mel = dict(_MEL_DEFAULTS)
        return
    try:
        f = cfg.features
        _active_mel = {
            "sample_rate": int(cfg.model.sample_rate),
            "n_mels": int(f.n_mels),
            "n_fft": int(f.n_fft),
            "win_length": int(f.win_length),
            "hop_length": int(f.hop_length),
            "fmin": float(f.fmin),
            "fmax": float(f.fmax),
        }
    except Exception:  # noqa: BLE001 — fall back to the EfficientAT defaults
        _active_mel = dict(_MEL_DEFAULTS)


def _mel_hop_length(sr: int) -> int:
    # Frame/sample reporting is expressed in the caller's original sample
    # coordinates even when analysis resamples to the model rate below.
    model_sr = max(1, int(_active_mel["sample_rate"]))
    return max(1, round(int(_active_mel["hop_length"]) * int(sr) / model_sr))


def _mel_win_length(sr: int) -> int:
    model_sr = max(1, int(_active_mel["sample_rate"]))
    return max(1, round(int(_active_mel["win_length"]) * int(sr) / model_sr))


def _frames_for_window(sr: int, window_ms: int) -> int:
    hop = _mel_hop_length(sr)
    return max(1, round((window_ms / 1000.0) * sr / hop))


def _frame_to_sample(frame: int, sr: int) -> int:
    return max(0, int(frame) * _mel_hop_length(sr))


def _frame_window_bounds(
    n_samples: int,
    start_frame: int,
    sr: int,
    window_frames: int,
) -> tuple[int, int]:
    lo = _frame_to_sample(start_frame, sr)
    hi = _frame_to_sample(start_frame + window_frames, sr)
    lo = max(0, min(n_samples, lo))
    hi = max(0, min(n_samples, hi))
    return lo, hi


def _mel_magnitude(wave: np.ndarray, sr: int) -> np.ndarray:
    """Mel power using the exact EfficientAT training frontend."""
    wave = _mono(wave)
    model_sr = int(_active_mel["sample_rate"])
    if int(sr) != model_sr:
        wave = resample_audio(wave, orig_sr=int(sr), target_sr=model_sr)
    n_fft = int(_active_mel["n_fft"])
    if len(wave) < n_fft:
        wave = np.pad(wave, (0, n_fft - len(wave)))
    return _mel_power_np(
        wave,
        model_sr,
        n_fft,
        int(_active_mel["win_length"]),
        int(_active_mel["hop_length"]),
        int(_active_mel["n_mels"]),
        float(_active_mel["fmin"]),
        min(float(_active_mel["fmax"]), float(model_sr) / 2.0),
    )


def _locate_transient_frame(wave: np.ndarray, sr: int) -> int:
    """Locate the main onset by spectral flux, not absolute loudness.

    This intentionally mirrors the recorder's Superflux idea: compare each
    log-mel frame against the preceding few frames and keep only positive
    energy changes. A steady loud hum therefore contributes very little, while
    a fine quiet glass onset can still win if it creates a sudden spectral
    change.

    Uses a sliding 100 ms sum window to accumulate onset flux energy,
    returning the starting frame index of the maximum energy window.
    """
    mel = _mel_magnitude(wave, sr)
    log_mel = np.log1p(mel)
    if log_mel.shape[1] <= 1:
        return 0

    # Favor the glass-cling region without making low/mid-band clinks vanish.
    centers = kaldi_mel_center_freqs(
        log_mel.shape[0],
        float(_active_mel["fmin"]),
        min(
            float(_active_mel["fmax"]),
            float(_active_mel["sample_rate"]) / 2.0,
        ),
    )
    weights = np.clip((centers - 700.0) / 2500.0, 0.0, 1.0)
    weighted_flux = np.zeros(log_mel.shape[1], dtype=np.float32)
    for frame in range(1, log_mel.shape[1]):
        lo = max(0, frame - 3)
        ref = log_mel[:, lo:frame].max(axis=1)
        diff = np.maximum(log_mel[:, frame] - ref, 0.0)
        weighted_flux[frame] = float(np.mean(diff * weights))
    weighted_flux[:2] = 0.0
    weighted_flux[-2:] = 0.0

    window_frames = min(_frames_for_window(sr, _TRANSIENT_LOCATOR_WINDOW_MS), len(weighted_flux))
    if window_frames <= 0:
        return 0
    rolling_flux = np.zeros(len(weighted_flux) - window_frames + 1, dtype=np.float32)
    for i in range(len(rolling_flux)):
        rolling_flux[i] = float(np.sum(weighted_flux[i : i + window_frames]))

    return int(rolling_flux.argmax())


# ---------------------------------------------------------------------------
# Masking score — used to flag mixes where the target transient gets buried.
# ---------------------------------------------------------------------------


def _active_transient_bands(
    target_mel: np.ndarray,
    frame: int,
    sr: int,
    window_ms: int,
    max_bands: int = 10,
    top_frames: int = 5,
) -> tuple[np.ndarray, slice]:
    """Select bands whose energy rises inside the transient window.

    Absolute energy alone lets steady hum/room tone steal the selected bands.
    Ranking by excess over the immediately preceding baseline keeps the score
    focused on the event footprint.
    """
    window_frames = _frames_for_window(sr, window_ms)
    lo = max(0, min(target_mel.shape[1], frame))
    hi = max(0, min(target_mel.shape[1], frame + window_frames))
    if hi <= lo:
        hi = min(target_mel.shape[1], lo + 1)
        lo = max(0, hi - 1)

    frame_slice = slice(lo, hi)
    event_power = _mean_of_top_frames(target_mel[:, frame_slice], top_frames)
    base_hi = lo
    base_lo = max(0, base_hi - window_frames)
    if base_hi > base_lo:
        baseline = np.median(target_mel[:, base_lo:base_hi], axis=1)
    elif hi - lo > 1:
        baseline = target_mel[:, lo]
    else:
        baseline = np.zeros_like(event_power)
    event_db = 10.0 * np.log10(event_power + 1e-10)
    baseline_db = 10.0 * np.log10(baseline + 1e-10)
    excess = np.maximum(event_db - baseline_db, 0.0)

    if hi - lo > 1:
        cur_db = 10.0 * np.log10(target_mel[:, lo + 1 : hi] + 1e-10)
        prev_db = 10.0 * np.log10(target_mel[:, lo : hi - 1] + 1e-10)
        local_flux = np.maximum(cur_db - prev_db, 0.0).mean(axis=1)
    else:
        local_flux = np.zeros_like(event_power)
    band_score = excess + local_flux
    if not np.any(band_score > 0.0):
        band_score = event_power

    ordered = np.argsort(band_score)[::-1]
    top_n = min(max_bands, len(event_power))
    bands = ordered[:top_n]

    return np.asarray(bands, dtype=np.int64), frame_slice


def _mean_of_top_frames(values: np.ndarray, top_frames: int) -> np.ndarray:
    """Mean of the strongest frames along the last axis.

    For a 100 ms masking window this lets a 50 ms soft attack define the
    energy comparison instead of being diluted by quiet release frames.
    """
    arr = np.asarray(values)
    if arr.shape[-1] == 0:
        return np.mean(arr, axis=-1)
    k = max(1, min(int(top_frames), arr.shape[-1]))
    if k == arr.shape[-1]:
        return np.mean(arr, axis=-1)
    partitioned = np.partition(arr, arr.shape[-1] - k, axis=-1)
    return np.mean(partitioned[..., -k:], axis=-1)


def _top_frame_offsets(values: np.ndarray, top_frames: int) -> np.ndarray:
    """Return frame offsets for the strongest target frames in one band."""
    arr = np.asarray(values)
    if arr.size == 0:
        return np.asarray([0], dtype=np.int64)
    k = max(1, min(int(top_frames), arr.size))
    if k == arr.size:
        return np.arange(arr.size, dtype=np.int64)
    offsets = np.argpartition(arr, arr.size - k)[-k:]
    return np.asarray(sorted(int(i) for i in offsets), dtype=np.int64)


def masking_score_db(
    target: np.ndarray,
    mixed: np.ndarray,
    sr: int,
    *,
    background: np.ndarray | None = None,
    transient_window_ms: int = 100,
    mask_band_count: int = 10,
    mask_top_frames: int = 5,
) -> float:
    """How much louder is the clean target transient than the local background?"""
    return masking_score_details(
        target,
        mixed,
        sr,
        background=background,
        transient_window_ms=transient_window_ms,
        mask_band_count=mask_band_count,
        mask_top_frames=mask_top_frames,
    )["masking_score_db"]


def masking_score_details(
    target: np.ndarray,
    mixed: np.ndarray,
    sr: int,
    *,
    background: np.ndarray | None = None,
    transient_window_ms: int = 100,
    mask_band_count: int = 10,
    mask_top_frames: int = 5,
) -> dict[str, object]:
    """Return target/background mel dB values plus their mask difference.

    The transient time is located on the clean target with spectral flux. The
    score then compares the target component against the background component
    in the target's active mel bands and in the same small time window.

    Performs a band-by-band dB subtraction and averages the differences.
    """
    target = _mono(target)
    mixed = _mono(mixed)
    if background is None:
        n = min(len(target), len(mixed))
        background = mixed[:n] - target[:n]
    else:
        background = _mono(background)

    target_mel = _mel_magnitude(target, sr)
    bg_mel = _mel_magnitude(background, sr)
    frame = min(_locate_transient_frame(target, sr), target_mel.shape[1] - 1)
    bands, frame_slice = _active_transient_bands(
        target_mel,
        frame,
        sr,
        transient_window_ms,
        max_bands=mask_band_count,
        top_frames=mask_top_frames,
    )

    # Ensure background slice has identical length and mapping
    hi = min(frame_slice.stop, bg_mel.shape[1])
    lo = min(frame_slice.start, hi - 1) if hi > 0 else 0
    frame_slice = slice(max(0, lo), max(lo + 1, hi))

    # Compute target and background dB for each band over the same slice,
    # then calculate the band-by-band dB difference and average them.
    target_dbs = []
    bg_dbs = []
    diffs = []
    for b in bands:
        target_values = target_mel[b, frame_slice]
        bg_values = bg_mel[b, frame_slice]
        offsets = _top_frame_offsets(target_values, mask_top_frames)
        offsets = offsets[offsets < bg_values.shape[0]]
        if offsets.size == 0:
            offsets = np.asarray([0], dtype=np.int64)
        target_power_b = float(np.mean(target_values[offsets]))
        bg_power_b = float(np.mean(bg_values[offsets]))
        t_db = 10.0 * np.log10(target_power_b + 1e-10)
        b_db = 10.0 * np.log10(bg_power_b + 1e-10)
        target_dbs.append(t_db)
        bg_dbs.append(b_db)
        diffs.append(t_db - b_db)

    target_db = float(np.mean(target_dbs))
    background_db = float(np.mean(bg_dbs))
    masking_score = float(np.mean(diffs))
    masking_p25 = float(np.percentile(diffs, 25))
    masking_min = float(np.min(diffs))

    centers = kaldi_mel_center_freqs(
        target_mel.shape[0],
        float(_active_mel["fmin"]),
        min(
            float(_active_mel["fmax"]),
            float(_active_mel["sample_rate"]) / 2.0,
        ),
    )
    band_indices = [int(b) for b in bands]

    return {
        "mask_target_db": target_db,
        "mask_background_db": background_db,
        "masking_score_db": masking_score,
        "masking_score_p25_db": masking_p25,
        "masking_score_min_db": masking_min,
        "mask_transient_frame": int(frame),
        "mask_transient_sample": int(_frame_to_sample(frame, sr)),
        "mask_window_frames": int(frame_slice.stop - frame_slice.start),
        "mask_window_start_frame": int(frame_slice.start),
        "mask_window_end_frame": int(frame_slice.stop),
        "mask_window_start_sample": int(_frame_to_sample(frame_slice.start, sr)),
        "mask_window_end_sample": int(_frame_to_sample(frame_slice.stop, sr)),
        "mask_band_count": int(len(band_indices)),
        "mask_top_frames": int(max(1, min(int(mask_top_frames), frame_slice.stop - frame_slice.start))),
        "mask_selected_bands": band_indices,
        "mask_compared_background_bands": band_indices,
        "mask_selected_band_hz": [float(centers[b]) for b in band_indices],
        "mask_band_margins_db": [float(d) for d in diffs],
    }


def transient_mel_db(
    wave: np.ndarray,
    sr: int,
    *,
    transient_window_ms: int = 100,
    mask_band_count: int = 10,
    mask_top_frames: int = 5,
) -> float:
    """Dominant-mel intensity around the flux-located transient.

    Unlike waveform peak dBFS, this measures the local mel-band footprint of
    the transient itself, so unrelated bumps elsewhere in the clip do not win.
    """
    wave = _mono(wave)
    mel = _mel_magnitude(wave, sr)
    frame = min(_locate_transient_frame(wave, sr), mel.shape[1] - 1)
    bands, frame_slice = _active_transient_bands(
        mel,
        frame,
        sr,
        transient_window_ms,
        max_bands=mask_band_count,
        top_frames=mask_top_frames,
    )

    # Compute db for each band and take the mean
    band_dbs = [
        10.0 * np.log10(float(_mean_of_top_frames(mel[b, frame_slice], mask_top_frames)) + 1e-10)
        for b in bands
    ]
    return float(np.mean(band_dbs))


def render_mix_map(
    target: np.ndarray,
    background: np.ndarray,
    sr: int,
    out_path: Path,
    masking: dict | None = None,
    features: dict | None = None,
) -> bool:
    """Write a two-color mel PNG: base transient in orange, overlay in blue.

    Both components are rendered in the model's own mel view (same frontend,
    mel layout, and fixed display range as the spectrogram previews), so the
    mix map lines up with the normal card spectrograms. Overlapping energy
    blends towards white; a base that is visually faint next to the overlay is
    also acoustically buried. The mask transient window is marked with two
    vertical lines and the selected mask bands with ticks on the left edge.
    """
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        log.warning("mix map skipped — Pillow unavailable: %s", exc)
        return False
    try:
        from sed.audio.features import EFFICIENTAT_DISPLAY_RANGE, waveform_to_logmel

        feature_kwargs = dict(features or {})
        t_spec = waveform_to_logmel(_mono(target), sample_rate=sr, **feature_kwargs)[0]
        b_spec = waveform_to_logmel(_mono(background), sample_rate=sr, **feature_kwargs)[0]
        frames = min(t_spec.shape[1], b_spec.shape[1])
        vmin, vmax = EFFICIENTAT_DISPLAY_RANGE
        t_n = np.clip((t_spec[:, :frames] - vmin) / (vmax - vmin), 0.0, 1.0)
        b_n = np.clip((b_spec[:, :frames] - vmin) / (vmax - vmin), 0.0, 1.0)
        bands = t_n.shape[0]
        rgb = np.zeros((bands, frames, 3), dtype=np.float32)
        rgb[..., 0] += t_n          # base -> orange (R + 0.6 G)
        rgb[..., 1] += 0.6 * t_n
        rgb[..., 2] += b_n          # overlay -> blue (B + 0.2 G)
        rgb[..., 1] += 0.2 * b_n
        np.clip(rgb, 0.0, 1.0, out=rgb)
        img = (rgb * 255.0).astype(np.uint8)[::-1]  # low mels at the bottom
        if isinstance(masking, dict):
            for key in ("mask_window_start_frame", "mask_window_end_frame"):
                frame = masking.get(key)
                if frame is None:
                    continue
                col = min(max(int(frame), 0), frames - 1)
                img[:, col] = np.maximum(img[:, col], 160)
            for band in masking.get("mask_selected_bands") or []:
                row = bands - 1 - int(band)
                if 0 <= row < bands:
                    img[row, : min(4, frames)] = 255
        image = Image.fromarray(img, mode="RGB")
        # Slightly taller than the 3:1 card spectrograms (~2.3:1), and much
        # shorter than the old 1.2:1 maps — same card width, comparable look.
        image = image.resize((max(frames * 3, 240), bands), Image.NEAREST)
        image.save(str(out_path), format="PNG")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("mix map render failed for %s: %s", out_path, exc)
        return False


def _mix_with_scaled_noise(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    *,
    scale_noise_to_snr: bool = True,
    prevent_clipping: bool = True,
    transient_window_ms: int = 0,
    sample_rate: int | None = None,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return mixed wave, post-gain signal component, noise component, params."""
    signal = _mono(signal)
    noise = _mono(noise)
    reference = signal if reference is None else _mono(reference)
    scaled_noise = noise.copy()
    transient_frame: int | None = None
    transient_sample: int | None = None
    transient_window_frames: int | None = None
    reference_mode = "full_clip"

    if scale_noise_to_snr:
        if transient_window_ms > 0 and sample_rate is not None:
            start_frame = _locate_transient_frame(reference, sample_rate)
            transient_frame = int(start_frame)
            transient_sample = _frame_to_sample(start_frame, sample_rate)
            transient_window_frames = _frames_for_window(sample_rate, transient_window_ms)

            s_lo, s_hi = _frame_window_bounds(
                len(reference),
                start_frame,
                sample_rate,
                transient_window_frames,
            )
            s_rms = _rms(reference[s_lo:s_hi]) if s_hi > s_lo else _rms(reference)

            n_lo, n_hi = _frame_window_bounds(
                len(noise),
                start_frame,
                sample_rate,
                transient_window_frames,
            )
            n_rms = _rms(noise[n_lo:n_hi]) if n_hi > n_lo else _rms(noise)

            reference_mode = "transient_window_aligned"
            if n_rms < 1e-8:
                n_rms = _rms(noise)
                reference_mode = "transient_window_signal_full_noise"
        else:
            s_rms = _rms(reference)
            n_rms = _rms(noise)

        if n_rms >= 1e-9:
            target_noise_rms = s_rms / (10 ** (snr_db / 20.0))
            scaled_noise = noise * (target_noise_rms / n_rms)

    out = (signal + scaled_noise).astype(np.float32)
    clipped = False
    post_gain = 1.0
    if prevent_clipping and out.size:
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            post_gain = 1.0 / peak
            out = (out * post_gain).astype(np.float32)
            clipped = True

    # Achieved broadband SNR of the final mix, measured in the base transient
    # window when available. Unlike the drawn `snr_db`, this is also defined
    # with scale_noise_to_snr off (native-loudness overlays), where it captures
    # mixes whose overlay audibly swamps the base even though the band-limited
    # mask margins stay positive.
    effective_snr_db: float | None = None
    if signal.size and scaled_noise.size:
        if transient_window_ms > 0 and sample_rate is not None:
            eff_frame = _locate_transient_frame(reference, sample_rate)
            eff_window_frames = _frames_for_window(sample_rate, transient_window_ms)
            e_lo, e_hi = _frame_window_bounds(len(signal), eff_frame, sample_rate, eff_window_frames)
            sig_rms = _rms(signal[e_lo:e_hi]) if e_hi > e_lo else _rms(signal)
            ovl_rms = _rms(scaled_noise[e_lo:e_hi]) if e_hi > e_lo else _rms(scaled_noise)
        else:
            sig_rms = _rms(signal)
            ovl_rms = _rms(scaled_noise)
        if sig_rms > 1e-9 and ovl_rms > 1e-9:
            effective_snr_db = float(20.0 * np.log10(sig_rms / ovl_rms))

    params = {
        "snr_db": float(snr_db),
        "effective_snr_db": effective_snr_db,
        "scale_noise_to_snr": bool(scale_noise_to_snr),
        "prevent_clipping": bool(prevent_clipping),
        "transient_window_ms": int(transient_window_ms),
        "snr_reference": reference_mode,
        "transient_frame": transient_frame,
        "transient_sample": transient_sample,
        "transient_window_frames": transient_window_frames,
        "post_gain": float(post_gain),
        "clipped_in_post": bool(clipped),
    }
    return (
        out,
        (signal * post_gain).astype(np.float32),
        (scaled_noise * post_gain).astype(np.float32),
        params,
    )


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Simple fixed-SNR overlay helper used by masking tests and utilities."""
    out, _, _, _ = _mix_with_scaled_noise(
        signal, noise, snr_db,
        scale_noise_to_snr=True,
        prevent_clipping=True,
    )
    return out


# ---------------------------------------------------------------------------
# Lazy WAV pool — generic utility for negative/overlay directories.
# ---------------------------------------------------------------------------


class AudioPool:
    """Lazy pool of audio files at a fixed sample rate.

    Accepts a single root or a list of roots; pools every ``*.wav`` underneath
    them. Loads on first access and caches in memory. ``.filtered(labels)``
    returns a view restricted to clips whose path matches any of the given
    label substrings.
    """

    def __init__(self, roots: Path | list[Path] | str | list[str], sample_rate: int):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.roots = [Path(r) for r in roots]
        self.sr = sample_rate
        paths: list[Path] = []
        for r in self.roots:
            if r.exists():
                paths.extend(sorted(r.rglob("*.wav")))
        seen: set[Path] = set()
        uniq: list[Path] = []
        for p in paths:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                uniq.append(p)
        self.paths = uniq
        self._cache: dict[Path, tuple[np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def filtered(self, labels: list[str]) -> "AudioPool":
        if not labels:
            return self
        wanted = {lab.lower() for lab in labels}
        sub = AudioPool.__new__(AudioPool)
        sub.roots = self.roots
        sub.sr = self.sr
        sub._cache = self._cache
        sub.paths = [p for p in self.paths if any(seg.lower() in wanted for seg in p.parts)]
        return sub

    def excluded(self, labels: list[str]) -> "AudioPool":
        if not labels:
            return self
        unwanted = {lab.lower() for lab in labels}
        sub = AudioPool.__new__(AudioPool)
        sub.roots = self.roots
        sub.sr = self.sr
        sub._cache = self._cache
        sub.paths = [p for p in self.paths if not any(seg.lower() in unwanted for seg in p.parts)]
        return sub

    def random(self, n_samples: int, rng: random.Random | None = None) -> tuple[np.ndarray, Path] | None:
        if not self.paths:
            return None
        rng = rng or random
        path = rng.choice(self.paths)
        if path not in self._cache:
            try:
                self._cache[path] = _load_native(path)
            except Exception as exc:  # noqa: BLE001
                log.warning("audio pool: failed to load %s: %s", path, exc)
                self.paths.remove(path)
                return self.random(n_samples, rng)
        wave, sr = self._cache[path]
        if sr != self.sr:
            wave = resample_audio(wave, orig_sr=sr, target_sr=self.sr)
        return _ensure_length(wave, n_samples, rng=rng), path

    def random_native(self, rng: random.Random | None = None) -> tuple[np.ndarray, int, Path] | None:
        if not self.paths:
            return None
        rng = rng or random
        path = rng.choice(self.paths)
        if path not in self._cache:
            try:
                self._cache[path] = _load_native(path)
            except Exception as exc:  # noqa: BLE001
                log.warning("audio pool: failed to load %s: %s", path, exc)
                self.paths.remove(path)
                return self.random_native(rng)
        wave, sr = self._cache[path]
        return wave, sr, path


def _audio_pool_from_paths(paths: list[Path], sample_rate: int) -> AudioPool:
    """Create an AudioPool view from exact paths instead of scanning roots."""
    pool = AudioPool.__new__(AudioPool)
    pool.roots = sorted({p.parent for p in paths})
    pool.sr = sample_rate
    seen: set[Path] = set()
    pool.paths = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        pool.paths.append(path)
    pool._cache = {}
    return pool


# ---------------------------------------------------------------------------
# Transform helpers.
# ---------------------------------------------------------------------------


def apply_pitch_shift(wave: np.ndarray, sr: int, semitones: float) -> tuple[np.ndarray, dict]:
    """Shift fundamental frequency by ``semitones`` while preserving duration."""
    import librosa
    out = librosa.effects.pitch_shift(wave, sr=sr, n_steps=float(semitones))
    if len(out) != len(wave):
        out = _ensure_length(out, len(wave))
    return out.astype(np.float32), {"semitones": float(semitones)}


def _draw_pitch(cfg: Config, rng: random.Random, wave: np.ndarray,
                sample_rate: int) -> tuple[np.ndarray, float]:
    """One seeded pitch draw, or a pass-through when pitch shift is off.

    Shared by all three materialize pipelines. Shifts below 0.2 semitones are
    snapped outward: a draw that rounds to "no shift" costs a full resynthesis
    and yields a near-duplicate of the base clip, which inflates the variant
    count without adding acoustic variety.
    """
    if not cfg.augmentation.pitch_shift_enabled:
        return wave, 0.0
    semis = rng.uniform(*cfg.augmentation.semitones_range)
    if abs(semis) < 0.2:
        semis = 0.2 if semis >= 0 else -0.2
    shifted, _params = apply_pitch_shift(wave, sample_rate, semis)
    return shifted, float(semis)


def _candidate_rank(candidate: dict[str, object]) -> tuple[float, float, float]:
    """Order overlay candidates by how well the target transient survives.

    Worst-band-first: the p25 margin leads, because a mix that buries a few
    bands is worse for a thin glass clink than one that shaves the mean evenly.
    Used to keep the best attempt when no candidate clears the mask gate.
    """
    masking = candidate["masking"]
    assert isinstance(masking, dict)
    return (
        float(masking["masking_score_p25_db"]),
        float(masking["masking_score_db"]),
        float(masking["masking_score_min_db"]),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_sidecar(path: Path, sidecar: dict) -> None:
    if "augmented_at" not in sidecar:
        sidecar["augmented_at"] = datetime.now(UTC).isoformat()
    if sidecar.get("category") == DERIVED_CATEGORY and sidecar.get("augmentation_run_prefix"):
        sidecar.setdefault(
            "augmentation_started_at",
            _augmentation_started_at(str(sidecar["augmentation_run_prefix"])),
        )
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(sidecar), f, indent=2)


def _load_source(path: Path) -> tuple[np.ndarray, int] | None:
    try:
        return _load_native(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("augment: source load failed %s: %s", path, exc)
        return None


def _materialize_target_augmented(
    cfg: Config,
    project_id: str,
    *,
    seed: int,
    run_prefix: str,
    replace: bool,
    frozen_test: bool,
) -> dict[str, dict[str, int]]:
    """Generate explicit [augmentation.combinations] recipes."""
    layout = ProjectLayout(project_id)
    sr = cfg.model.sample_rate
    rng = random.Random(seed)

    lib = Library()
    # The destination directory is an output staging location only; the
    # generated recording sidecar carries no train/test identity.
    derived_path_role = "test" if frozen_test else "trainval"
    aug_root = lib.root / DERIVED_CATEGORY / project_id / derived_path_role
    if replace and aug_root.exists():
        shutil.rmtree(aug_root)
    aug_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = {}
    settings_snapshot = augmentation_settings_snapshot(cfg)

    def draw_overlay(
        pool: AudioPool,
        *,
        avoid_path: Path | None = None,
    ) -> tuple[np.ndarray, int, Path] | None:
        if len(pool) == 0:
            return None
        for _ in range(5):
            drawn = pool.random_native(rng)
            if drawn is None:
                return None
            wave, native_sr, path = drawn
            if avoid_path is None or path.resolve() != avoid_path.resolve() or len(pool) <= 1:
                return wave, native_sr, path
        return pool.random_native(rng)

    def write_sample(
        *,
        target_class: str,
        wave: np.ndarray,
        final_sr: int,
        source: Path,
        idx: int,
        params: dict,
        masking: dict[str, object] | None,
    ) -> None:
        out_dir = aug_root / target_class
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = unique_path(
            out_dir / _augmentation_filename(run_prefix, target_class, idx),
            separator=_AUG_COLLISION_SEPARATOR,
        )
        sf.write(str(out_path), wave, final_sr, subtype="PCM_16")

        peak = float(np.max(np.abs(wave))) if wave.size else 0.0
        peak_dbfs = 20.0 * float(np.log10(peak + 1e-10)) if peak > 0 else -120.0
        rms = float(np.sqrt(np.mean(wave**2))) if wave.size else 0.0
        rms_dbfs = 20.0 * float(np.log10(rms + 1e-10)) if rms > 0 else -120.0
        description = str(params.get("combination_description") or "").strip()
        raw_tags = params.get("combination_tags") or []
        raw_tags = [raw_tags] if isinstance(raw_tags, str) else raw_tags
        tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
        tmel_db = transient_mel_db(
            wave,
            final_sr,
            mask_band_count=cfg.augmentation.mask_band_count,
            mask_top_frames=cfg.augmentation.mask_top_frames,
        )
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "category": DERIVED_CATEGORY,
            **_derived_provenance(source, "overlay"),
            "label": target_class,
            "project_id": project_id,
            "density": "normal",
            "review_status": "accepted",
            "recording_counter": idx,
            "recipe_id": str(params.get("recipe_id") or ""),
            "description": description,
            "tags": tags,
            "source": _source_display_name(source),
            "class_name": target_class,
            "target_class": target_class,
            "augmentation": "overlay",
            "source_origin": _source_origin(source, layout),
            "augmentation_run_prefix": run_prefix,
            "augmentation_index": idx,
            "augmentation_params": params,
            "augmentation_settings": settings_snapshot,
            "seed": seed,
            "peak_dbfs": peak_dbfs,
            "rms_dbfs": rms_dbfs,
            "transient_mel_db": tmel_db,
        }
        if masking:
            sidecar.update(masking)
        _write_sidecar(out_path, sidecar)
        lib.append_sidecar(out_path)

    def choose_candidate(
        *,
        target_class: str,
        clean_wave: np.ndarray,
        base_sr: int,
        base_path: Path,
        base_label: str,
        combo_base: str,
        combo_overlay: str,
        combo_description: str,
        combo_tags: list[str],
        combo_recipe_id: str,
        overlay_pool: AudioPool,
        overlay_labels: dict[Path, str],
        mask_min_mean_db: float,
        mask_min_p25_db: float,
        mask_min_band_db: float,
    ) -> dict[str, object] | None:
        attempts = (
            max(1, int(cfg.augmentation.mask_candidate_attempts))
            if cfg.augmentation.mask_filter_enabled and target_class != "_negative"
            else 1
        )
        best: dict[str, object] | None = None

        def mask_passes(masking: dict[str, object]) -> bool:
            return (
                float(masking["masking_score_db"]) >= mask_min_mean_db
                and float(masking["masking_score_p25_db"]) >= mask_min_p25_db
                and float(masking["masking_score_min_db"]) >= mask_min_band_db
            )

        def fallback_allowed(masking: dict[str, object]) -> bool:
            return (
                float(masking["masking_score_p25_db"])
                >= min(0.0, mask_min_p25_db)
                and float(masking["masking_score_min_db"])
                >= mask_min_band_db - 6.0
            )

        for attempt in range(1, attempts + 1):
            drawn = draw_overlay(overlay_pool, avoid_path=base_path)
            if drawn is None:
                return best
            overlay_wave, overlay_sr, overlay_path = drawn
            final_sr = min(base_sr, overlay_sr)

            if base_sr != final_sr:
                base_wave_for_mix = resample_audio(clean_wave, base_sr, final_sr)
            else:
                base_wave_for_mix = clean_wave

            if overlay_sr != final_sr:
                overlay_wave_for_mix = resample_audio(overlay_wave, overlay_sr, final_sr)
            else:
                overlay_wave_for_mix = overlay_wave

            target_n = len(base_wave_for_mix)
            overlay_wave_for_mix = _ensure_length(overlay_wave_for_mix, target_n, rng=rng)

            base_wave, semitones = _draw_pitch(cfg, rng, base_wave_for_mix, final_sr)
            snr_db = _draw_snr_db(cfg.augmentation.snr_db_range, rng)
            out, target_component, background_component, mix_params = _mix_with_scaled_noise(
                base_wave,
                overlay_wave_for_mix,
                snr_db,
                scale_noise_to_snr=cfg.augmentation.scale_noise_to_snr,
                prevent_clipping=cfg.augmentation.prevent_clipping,
                transient_window_ms=cfg.augmentation.transient_window_ms,
                sample_rate=final_sr,
            )
            overlay_label = overlay_labels.get(overlay_path.resolve(), combo_overlay)
            params = {
                **mix_params,
                "base_role": "negative" if target_class == "_negative" else "target",
                "source": _source_display_name(base_path),
                "negative_source": _source_display_name(overlay_path),
                "base_source": _source_display_name(base_path),
                "overlay_source": _source_display_name(overlay_path),
                "base_label": combo_base,
                "overlay_label": combo_overlay,
                "actual_base_label": base_label,
                "actual_overlay_label": overlay_label,
                "combination_description": combo_description,
                "combination_tags": list(combo_tags),
                "recipe_id": combo_recipe_id,
                "target_class": target_class,
                "mask_min_mean_db": float(mask_min_mean_db),
                "mask_min_p25_db": float(mask_min_p25_db),
                "mask_min_band_db": float(mask_min_band_db),
                "pitch_shift_enabled": bool(cfg.augmentation.pitch_shift_enabled),
                "semitones": float(semitones),
                "overlay_enabled": bool(cfg.augmentation.overlay_enabled),
                "snr_db": float(snr_db),
                "mask_filter_enabled": bool(cfg.augmentation.mask_filter_enabled),
                "mask_attempts_used": attempt,
                "mask_filter_passed": target_class == "_negative",
                "mask_fallback_used": False,
            }
            masking: dict[str, object] | None = None
            if target_class != "_negative":
                masking = masking_score_details(
                    target_component,
                    out,
                    final_sr,
                    background=background_component,
                    transient_window_ms=cfg.augmentation.transient_window_ms,
                    mask_band_count=cfg.augmentation.mask_band_count,
                    mask_top_frames=cfg.augmentation.mask_top_frames,
                )
            candidate = {
                "wave": out,
                "final_sr": final_sr,
                "params": params,
                "source": base_path,
                "masking": masking,
            }
            if masking is None:
                return candidate
            if best is None or _candidate_rank(candidate) > _candidate_rank(best):
                best = candidate
            if not cfg.augmentation.mask_filter_enabled or mask_passes(masking):
                params["mask_filter_passed"] = True
                return candidate

        if best is None:
            return None
        best_masking = best["masking"]
        best_params = best["params"]
        assert isinstance(best_masking, dict)
        assert isinstance(best_params, dict)
        if fallback_allowed(best_masking):
            best_params["mask_attempts_used"] = attempts
            best_params["mask_filter_passed"] = False
            best_params["mask_fallback_used"] = True
            return best
        log.info(
            "augment: skipped %s because best explicit overlay was masked "
            "(mean=%.1f p25=%.1f min=%.1f dB after %d attempts)",
            base_path.name,
            float(best_masking["masking_score_db"]),
            float(best_masking["masking_score_p25_db"]),
            float(best_masking["masking_score_min_db"]),
            attempts,
        )
        return None

    for target_class, recipes in sorted(cfg.augmentation.combinations.items()):
        out_dir = aug_root / target_class
        idx = _next_augmentation_index(out_dir, run_prefix, target_class)
        class_counts: dict[str, int] = {}
        for combo in recipes:
            base_filters = combo.base_filters or ([combo.base_filter] if combo.base_filter else [{}])
            overlay_filters = combo.overlay_filters or ([combo.overlay_filter] if combo.overlay_filter else [{}])
            base_paths = library_paths_for_filters(
                lib,
                combo.base,
                project_id=project_id,
                frozen_test=frozen_test,
                filters=base_filters,
            )
            overlay_paths = library_paths_for_filters(
                lib,
                combo.overlay,
                project_id=project_id,
                frozen_test=frozen_test,
                filters=overlay_filters,
            )
            if not base_paths:
                log.warning("augment: no base clips for label %s", combo.base)
                continue
            if not overlay_paths:
                log.warning("augment: no overlay clips for label %s", combo.overlay)
                continue
            overlay_pool = _audio_pool_from_paths([p for p, _label in overlay_paths], sr)
            overlay_labels = {p.resolve(): label for p, label in overlay_paths}
            mask_min_mean_db = (
                combo.mask_min_mean_db
                if combo.mask_min_mean_db is not None
                else cfg.augmentation.mask_min_mean_db
            )
            mask_min_p25_db = (
                combo.mask_min_p25_db
                if combo.mask_min_p25_db is not None
                else cfg.augmentation.mask_min_p25_db
            )
            mask_min_band_db = (
                combo.mask_min_band_db
                if combo.mask_min_band_db is not None
                else cfg.augmentation.mask_min_band_db
            )
            for base_path, base_label in base_paths:
                loaded = _load_source(base_path)
                if loaded is None:
                    continue
                clean, base_sr = loaded
                for _ in range(_get_n_samples(cfg.augmentation.samples_per_source, rng)):
                    candidate = choose_candidate(
                        target_class=target_class,
                        clean_wave=clean,
                        base_sr=base_sr,
                        base_path=base_path,
                        base_label=base_label,
                        combo_base=combo.base,
                        combo_overlay=combo.overlay,
                        combo_description=combo.description,
                        combo_tags=combo.tags,
                        combo_recipe_id=combo.recipe_id,
                        overlay_pool=overlay_pool,
                        overlay_labels=overlay_labels,
                        mask_min_mean_db=float(mask_min_mean_db),
                        mask_min_p25_db=float(mask_min_p25_db),
                        mask_min_band_db=float(mask_min_band_db),
                    )
                    if candidate is None:
                        break
                    write_sample(
                        target_class=target_class,
                        wave=candidate["wave"],
                        final_sr=candidate["final_sr"],
                        source=base_path,
                        idx=idx,
                        params=candidate["params"],
                        masking=candidate["masking"],
                    )
                    idx += 1
                    class_counts["overlay"] = class_counts.get("overlay", 0) + 1
        if class_counts:
            counts[target_class] = class_counts

    lib.rebuild_manifest()
    return counts


def _expected_n_samples(rate: float) -> int:
    if rate <= 0:
        return 0
    whole = int(rate)
    return whole + (1 if rate > whole else 0)


def _recipe_filters(recipe: dict, key: str) -> list[dict]:
    plural = recipe.get(f"{key}_filters")
    if isinstance(plural, list):
        return [row for row in plural if isinstance(row, dict)]
    single = recipe.get(f"{key}_filter")
    if isinstance(single, dict):
        return [single]
    return [{}]


def _recipe_source_buckets(
    lib: Library,
    recipe: dict,
    key: str,
    *,
    project_id: str,
    frozen_test: bool,
) -> list[list[tuple[Path, str]]]:
    """Resolve each selected source rule as one independently sampled bucket.

    Flattening all matching paths makes a large bucket more likely to be
    selected than a small one.  Inspect recipes instead first choose one of
    their non-empty selected buckets uniformly, then choose a clip from it.
    """
    label = str(recipe.get(key) or _labels_summary(_recipe_filters(recipe, key)))
    buckets: list[list[tuple[Path, str]]] = []
    for rule in _recipe_filters(recipe, key):
        paths = library_paths_for_filters(
            lib,
            label,
            project_id=project_id,
            frozen_test=frozen_test,
            filters=[rule],
        )
        if paths:
            buckets.append(paths)
    return buckets


def _draw_bucket_path(
    buckets: list[list[tuple[Path, str]]],
    rng: random.Random,
) -> tuple[Path, str] | None:
    """Draw a random clip from a random non-empty selected source bucket."""
    if not buckets:
        return None
    return rng.choice(rng.choice(buckets))


def _labels_summary(filters: list[dict]) -> str:
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


def delete_generated_for_recipes(
    project_id: str,
    recipe_ids: list[str],
    *,
    frozen_test: bool | None = None,
) -> int:
    """Delete generated WAV/JSON pairs with matching recipe ids."""
    wanted = {str(item) for item in recipe_ids if str(item)}
    if not wanted:
        return 0
    lib = Library()
    roles = ["test" if frozen_test else "trainval"] if frozen_test is not None else ["trainval", "test"]
    deleted = 0
    deleted_paths: list[Path] = []
    for role in roles:
        root = lib.root / DERIVED_CATEGORY / project_id / role
        if not root.exists():
            continue
        for sidecar in sorted(root.rglob("*.json")):
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            params = meta.get("augmentation_params") if isinstance(meta.get("augmentation_params"), dict) else {}
            recipe_id = str(meta.get("recipe_id") or params.get("recipe_id") or "").strip()
            if recipe_id not in wanted:
                continue
            wav = sidecar.with_suffix(".wav")
            thumb = sidecar.with_suffix(".spec.png")
            mix_map = sidecar.with_suffix(".mix.png")
            try:
                wav.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
                thumb.unlink(missing_ok=True)
                mix_map.unlink(missing_ok=True)
                deleted += 1
                deleted_paths.append(wav)
            except Exception as exc:  # noqa: BLE001
                log.warning("failed to delete generated recipe clip %s: %s", sidecar, exc)
    if deleted:
        try:
            cache_root = lib.root / ".preview_cache"
            if cache_root.exists():
                from sed.data.preview_cache import PreviewCache

                PreviewCache(root=lib.root).remove_sources(deleted_paths)
        except Exception as exc:  # noqa: BLE001
            log.warning("preview cache cleanup for generated clips failed: %s", exc)
        lib.rebuild_manifest()
    return deleted


def materialize_library_recipes(
    cfg: Config,
    project_id: str,
    recipes: list[dict],
    *,
    seed: int | None = None,
    run_prefix: str | None = None,
    replace: bool = False,
    frozen_test: bool = False,
    generation_id: str | None = None,
    progress: Callable[[dict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, dict[str, int]]:
    """Generate targetless UI recipes as reusable derived library clips."""
    set_augment_features(cfg)
    layout = ProjectLayout(project_id)
    sr = cfg.model.sample_rate
    seed = seed if seed is not None else cfg.augmentation.seed
    run_prefix = run_prefix or _default_run_prefix(seed)
    generation_id = generation_id or uuid.uuid4().hex[:12]
    rng = random.Random(seed)
    # `settings_snapshot` is recomputed per recipe inside the execution loop
    # below — write_sample reads it via closure after each per-recipe apply.
    settings_snapshot: dict = {}

    lib = Library()

    if not cfg.augmentation.enabled:
        raise AugmentationDisabled(
            "augmentation.enabled is false — enable augmentation in project config to generate clips"
        )
    # Baseline = the cfg state at function entry (project defaults + the panel
    # state already layered in by the caller). Each recipe is applied on top
    # of this baseline so two recipes can't leak settings into one another.
    baseline_settings = augmentation_settings_snapshot(cfg)
    warnings: list[str] = []

    normalized: list[dict] = []
    for raw in recipes:
        if not isinstance(raw, dict):
            continue
        recipe = dict(raw)
        recipe_id = str(recipe.get("recipe_id") or recipe.get("id") or uuid.uuid4().hex[:12])
        recipe["recipe_id"] = recipe_id
        recipe["id"] = recipe_id
        recipe["base_filters"] = _recipe_filters(recipe, "base")
        recipe["overlay_filters"] = _recipe_filters(recipe, "overlay")
        recipe["base"] = str(recipe.get("base") or _labels_summary(recipe["base_filters"]))
        recipe["overlay"] = str(recipe.get("overlay") or _labels_summary(recipe["overlay_filters"]))
        # Per-recipe flags fall back to top-level kwargs so old call sites stay correct.
        recipe["_replace_existing"] = bool(recipe.get("replace_existing", replace))
        recipe["_frozen_test"] = bool(recipe.get("frozen_test", frozen_test))
        recipe["_run_prefix"] = str(recipe.get("prefix") or "").strip() or run_prefix
        # The recipe's settings snapshot (or empty for legacy recipes). The
        # signature folds these in so existing clips become "stale" when the
        # recipe's settings change.
        recipe_settings = recipe.get("settings") if isinstance(recipe.get("settings"), dict) else {}
        effective_settings = {**baseline_settings, **(recipe_settings or {})}
        recipe["_settings"] = recipe_settings or {}
        recipe["_effective_settings"] = effective_settings
        recipe["recipe_signature"] = recipe_signature(recipe, effective_settings, frozen_test=recipe["_frozen_test"])
        normalized.append(recipe)

    # Replace-existing recipes: delete their on-disk clips up front so the
    # plan-the-deficit math below sees zero. We split per recipe to honor each
    # recipe's frozen_test flag.
    for recipe in normalized:
        if recipe["_replace_existing"]:
            delete_generated_for_recipes(
                project_id,
                [str(recipe["recipe_id"])],
                frozen_test=recipe["_frozen_test"],
            )

    def count_existing_current(recipe: dict, split_name: str) -> int:
        split_root = lib.root / DERIVED_CATEGORY / project_id / split_name
        if not split_root.exists():
            return 0
        base_sources = source_names_for_recipe_paths(recipe.get("base_paths") or [])
        overlay_sources = source_names_for_recipe_paths(recipe.get("overlay_paths") or [])
        total_current = 0
        for sidecar in split_root.rglob("*.json"):
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            params = meta.get("augmentation_params") if isinstance(meta.get("augmentation_params"), dict) else {}
            rid = str(meta.get("recipe_id") or params.get("recipe_id") or "").strip()
            if rid != str(recipe["recipe_id"]):
                continue
            if generated_clip_matches_recipe(
                meta,
                recipe,
                recipe["_effective_settings"],
                frozen_test=recipe["_frozen_test"],
                base_sources=base_sources,
                overlay_sources=overlay_sources,
            ):
                total_current += 1
        return total_current

    resolved: list[dict] = []
    total = 0
    blocking_warnings: list[str] = []
    for recipe in normalized:
        recipe_frozen = recipe["_frozen_test"]
        derived_path_role_r = "test" if recipe_frozen else "trainval"
        aug_root_r = lib.root / DERIVED_CATEGORY / project_id / derived_path_role_r
        aug_root_r.mkdir(parents=True, exist_ok=True)
        # Legacy recipes retain their per-base count.  New Inspect recipes may
        # instead set one explicit output total, independent of source count.
        effective_settings = recipe["_effective_settings"]
        recipe_samples = float(effective_settings.get("samples_per_source", cfg.augmentation.samples_per_source))
        recipe_expected_per_base = _expected_n_samples(recipe_samples)
        recipe["_expected_per_base"] = recipe_expected_per_base
        recipe_enabled = bool(effective_settings.get("enabled", True))
        recipe_pitch = bool(effective_settings.get("pitch_shift_enabled", False))
        recipe_overlay = bool(effective_settings.get("overlay_enabled", True))
        base_paths = library_paths_for_filters(
            lib,
            recipe["base"],
            project_id=project_id,
            frozen_test=recipe_frozen,
            filters=recipe["base_filters"],
        )
        overlay_paths = (
            library_paths_for_filters(
                lib,
                recipe["overlay"],
                project_id=project_id,
                frozen_test=recipe_frozen,
                filters=recipe["overlay_filters"],
            )
            if recipe_overlay and recipe["overlay_filters"]
            else []
        )
        base_buckets = _recipe_source_buckets(
            lib, recipe, "base", project_id=project_id, frozen_test=recipe_frozen,
        )
        overlay_buckets = (
            _recipe_source_buckets(
                lib, recipe, "overlay", project_id=project_id, frozen_test=recipe_frozen,
            )
            if recipe_overlay and recipe["overlay_filters"] else []
        )
        recipe_total = recipe.get("total_samples")
        expected_total = int(recipe_total) if recipe_total is not None else len(base_paths) * recipe_expected_per_base
        recipe["base_paths"] = base_paths
        recipe["overlay_paths"] = overlay_paths
        existing = count_existing_current(recipe, derived_path_role_r)
        to_generate = max(0, expected_total - existing)
        if not recipe_enabled:
            msg = f"recipe {recipe['recipe_id']}: augmentation disabled in recipe settings"
            warnings.append(msg)
            blocking_warnings.append(msg)
            to_generate = 0
        elif not recipe_pitch and not recipe_overlay:
            msg = f"recipe {recipe['recipe_id']}: enable pitch shift, overlay, or both"
            warnings.append(msg)
            blocking_warnings.append(msg)
            to_generate = 0
        elif recipe_total is None and recipe_expected_per_base <= 0:
            msg = f"recipe {recipe['recipe_id']}: samples_per_source must be greater than 0"
            warnings.append(msg)
            blocking_warnings.append(msg)
        elif not base_paths:
            msg = f"recipe {recipe['recipe_id']}: no base clips matched"
            warnings.append(msg)
            blocking_warnings.append(msg)
        elif recipe_overlay and not overlay_paths:
            msg = f"recipe {recipe['recipe_id']}: no overlay clips matched"
            warnings.append(msg)
            blocking_warnings.append(msg)
        elif expected_total > 0 and to_generate == 0:
            warnings.append(
                f"recipe {recipe['recipe_id']}: already at target ({existing}/{expected_total}) — nothing to fill"
            )
        log.info(
            "augment plan recipe=%s base=%d overlay=%d expected=%d existing=%d to_generate=%d frozen_test=%s prefix=%r",
            recipe["recipe_id"], len(base_paths), len(overlay_paths),
            expected_total, existing, to_generate, recipe_frozen, recipe["_run_prefix"],
        )
        resolved.append({
            **recipe,
            "base_paths": base_paths,
            "overlay_paths": overlay_paths,
            "base_buckets": base_buckets,
            "overlay_buckets": overlay_buckets,
            "_aug_root": aug_root_r,
            "_to_generate": to_generate,
        })
        total += to_generate

    if total == 0 and blocking_warnings:
        raise AugmentationDisabled("; ".join(blocking_warnings[:4]))

    status = {
        "state": "running",
        "total": total,
        "completed": 0,
        "generated": 0,
        "skipped": 0,
        "failed": 0,
        "current_recipe": "",
        "generation_id": generation_id,
        "warnings": list(warnings),
    }

    def emit(extra: dict | None = None) -> None:
        if extra:
            status.update(extra)
        if progress:
            progress(dict(status))

    def check_cancel() -> None:
        if should_cancel and should_cancel():
            emit({"state": "canceling"})
            raise AugmentationCancelled("augmentation canceled")

    def draw_overlay(
        pools: list[AudioPool], *, avoid_path: Path | None = None,
    ) -> tuple[np.ndarray, int, Path] | None:
        available = [pool for pool in pools if len(pool)]
        if not available:
            return None
        for _ in range(5):
            check_cancel()
            pool = rng.choice(available)
            drawn = pool.random_native(rng)
            if drawn is None:
                return None
            wave, native_sr, path = drawn
            if avoid_path is None or path.resolve() != avoid_path.resolve() or len(pool) <= 1:
                return wave, native_sr, path
        return rng.choice(available).random_native(rng)

    def choose_candidate(
        *,
        clean_wave: np.ndarray,
        base_sr: int,
        base_path: Path,
        base_label: str,
        recipe: dict,
        overlay_pools: list[AudioPool],
        overlay_labels: dict[Path, str],
    ) -> dict[str, object] | None:
        attempts = max(1, int(cfg.augmentation.mask_candidate_attempts)) if cfg.augmentation.mask_filter_enabled else 1
        best: dict[str, object] | None = None
        def optional_threshold(value: object) -> float | None:
            if value in (None, ""):
                return None
            return float(value)

        mask_min_mean_db = optional_threshold(cfg.augmentation.mask_min_mean_db)
        mask_min_p25_db = optional_threshold(cfg.augmentation.mask_min_p25_db)
        mask_min_band_db = optional_threshold(cfg.augmentation.mask_min_band_db)
        mask_min_effective_snr_db = optional_threshold(cfg.augmentation.mask_min_effective_snr_db)
        active_thresholds = [
            (key, threshold)
            for key, threshold in (
                ("masking_score_db", mask_min_mean_db),
                ("masking_score_p25_db", mask_min_p25_db),
                ("masking_score_min_db", mask_min_band_db),
            )
            if threshold is not None
        ]
        has_active_gates = bool(active_thresholds) or mask_min_effective_snr_db is not None

        def mask_passes(masking: dict[str, object]) -> bool:
            if not active_thresholds:
                return True
            return all(float(masking[key]) >= float(threshold) for key, threshold in active_thresholds)

        def effective_snr_passes(candidate_params: dict) -> bool:
            """Broadband floor: the band gates can stay positive while the
            overlay swamps the whole mix — this catches those candidates."""
            if mask_min_effective_snr_db is None:
                return True
            value = candidate_params.get("effective_snr_db")
            return value is not None and float(value) >= mask_min_effective_snr_db

        def fallback_allowed() -> bool:
            return not has_active_gates

        for attempt in range(1, attempts + 1):
            check_cancel()
            drawn = draw_overlay(overlay_pools, avoid_path=base_path)
            if drawn is None:
                return best
            overlay_wave, overlay_sr, overlay_path = drawn
            final_sr = min(base_sr, overlay_sr)

            if base_sr != final_sr:
                base_wave_for_mix = resample_audio(clean_wave, base_sr, final_sr)
            else:
                base_wave_for_mix = clean_wave

            if overlay_sr != final_sr:
                overlay_wave_for_mix = resample_audio(overlay_wave, overlay_sr, final_sr)
            else:
                overlay_wave_for_mix = overlay_wave

            target_n = len(base_wave_for_mix)
            overlay_wave_for_mix = _ensure_length(overlay_wave_for_mix, target_n, rng=rng)

            base_wave, semitones = _draw_pitch(cfg, rng, base_wave_for_mix, final_sr)
            snr_db = _draw_snr_db(cfg.augmentation.snr_db_range, rng)
            out, target_component, background_component, mix_params = _mix_with_scaled_noise(
                base_wave,
                overlay_wave_for_mix,
                snr_db,
                scale_noise_to_snr=cfg.augmentation.scale_noise_to_snr,
                prevent_clipping=cfg.augmentation.prevent_clipping,
                transient_window_ms=cfg.augmentation.transient_window_ms,
                sample_rate=final_sr,
            )
            overlay_label = overlay_labels.get(overlay_path.resolve(), recipe["overlay"])
            params = {
                **mix_params,
                "base_role": "base",
                "source": _source_display_name(base_path),
                "negative_source": _source_display_name(overlay_path),
                "base_source": _source_display_name(base_path),
                "overlay_source": _source_display_name(overlay_path),
                "base_label": base_label,
                "overlay_label": overlay_label,
                "actual_base_label": base_label,
                "actual_overlay_label": overlay_label,
                "combination_description": str(recipe.get("description") or ""),
                "combination_tags": list(recipe.get("tags") or []),
                "recipe_id": recipe["recipe_id"],
                "recipe_signature": recipe["recipe_signature"],
                "generation_id": generation_id,
                "mask_min_mean_db": mask_min_mean_db,
                "mask_min_p25_db": mask_min_p25_db,
                "mask_min_band_db": mask_min_band_db,
                "mask_min_effective_snr_db": mask_min_effective_snr_db,
                "pitch_shift_enabled": bool(cfg.augmentation.pitch_shift_enabled),
                "semitones": float(semitones),
                "overlay_enabled": bool(cfg.augmentation.overlay_enabled),
                "snr_db": float(snr_db),
                "mask_filter_enabled": bool(cfg.augmentation.mask_filter_enabled),
                "mask_attempts_used": attempt,
                "mask_filter_passed": False,
                "mask_fallback_used": False,
            }
            masking = masking_score_details(
                target_component,
                out,
                final_sr,
                background=background_component,
                transient_window_ms=cfg.augmentation.transient_window_ms,
                mask_band_count=cfg.augmentation.mask_band_count,
                mask_top_frames=cfg.augmentation.mask_top_frames,
            )
            candidate = {
                "wave": out,
                "final_sr": final_sr,
                "params": params,
                "source": base_path,
                "masking": masking,
                "target_component": target_component,
                "background_component": background_component,
            }
            if best is None or _candidate_rank(candidate) > _candidate_rank(best):
                best = candidate
            if not cfg.augmentation.mask_filter_enabled or (
                mask_passes(masking) and effective_snr_passes(params)
            ):
                params["mask_filter_passed"] = True
                return candidate

        if best is None:
            return None
        best_params = best["params"]
        assert isinstance(best_params, dict)
        if fallback_allowed():
            best_params["mask_attempts_used"] = attempts
            best_params["mask_filter_passed"] = False
            best_params["mask_fallback_used"] = True
            return best
        return None

    def base_only_candidate(
        *,
        clean_wave: np.ndarray,
        base_sr: int,
        base_path: Path,
        base_label: str,
        recipe: dict,
    ) -> dict[str, object]:
        base_wave, semitones = _draw_pitch(cfg, rng, clean_wave, base_sr)
        params = {
            "base_role": "base",
            "source": _source_display_name(base_path),
            "negative_source": "",
            "base_source": _source_display_name(base_path),
            "overlay_source": "",
            "base_label": base_label,
            "overlay_label": "",
            "actual_base_label": base_label,
            "actual_overlay_label": "",
            "combination_description": str(recipe.get("description") or ""),
            "combination_tags": list(recipe.get("tags") or []),
            "recipe_id": recipe["recipe_id"],
            "recipe_signature": recipe["recipe_signature"],
            "generation_id": generation_id,
            "augmentation_technique": "pitch" if cfg.augmentation.pitch_shift_enabled else "base",
            "pitch_shift_enabled": bool(cfg.augmentation.pitch_shift_enabled),
            "semitones": float(semitones),
            "overlay_enabled": False,
            "snr_db": None,
            "mask_filter_enabled": False,
            "mask_filter_passed": None,
            "mask_fallback_used": False,
        }
        return {
            "wave": base_wave,
            "final_sr": base_sr,
            "params": params,
            "source": base_path,
            "masking": None,
            "target_component": None,
            "background_component": None,
        }

    def write_sample(
        *,
        output_label: str,
        wave: np.ndarray,
        final_sr: int,
        source: Path,
        idx: int,
        params: dict,
        masking: dict[str, object] | None,
        aug_root_r: Path,
        run_prefix_r: str,
        base_density: str = "",
        base_tags: list[str] | None = None,
        target_component: np.ndarray | None = None,
        background_component: np.ndarray | None = None,
    ) -> None:
        out_dir = aug_root_r / safe_label(output_label)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = unique_path(
            out_dir / _augmentation_filename(run_prefix_r, output_label, idx),
            separator=_AUG_COLLISION_SEPARATOR,
        )
        sf.write(str(out_path), wave, final_sr, subtype="PCM_16")
        mix_map_name = ""
        if (
            bool(cfg.augmentation.mix_map_enabled)
            and target_component is not None
            and background_component is not None
        ):
            mix_png = out_path.with_suffix(".mix.png")
            # Render in the model's mel view so the map matches the previews.
            mix_features = {
                "frontend": cfg.features.frontend,
                "n_fft": cfg.features.n_fft,
                "win_length": cfg.features.win_length,
                "hop_length": cfg.features.hop_length,
                "n_mels": cfg.features.n_mels,
                "fmin": cfg.features.fmin,
                "fmax": cfg.features.fmax,
            }
            if render_mix_map(
                target_component, background_component, final_sr, mix_png,
                masking=masking, features=mix_features,
            ):
                mix_map_name = mix_png.name
        peak = float(np.max(np.abs(wave))) if wave.size else 0.0
        peak_dbfs = 20.0 * float(np.log10(peak + 1e-10)) if peak > 0 else -120.0
        rms = float(np.sqrt(np.mean(wave**2))) if wave.size else 0.0
        rms_dbfs = 20.0 * float(np.log10(rms + 1e-10)) if rms > 0 else -120.0
        raw_tags = params.get("combination_tags") or []
        raw_tags = [raw_tags] if isinstance(raw_tags, str) else raw_tags
        # Recipe tags first, then the base clip's own tags — dedup keeps order.
        tags: list[str] = []
        for tag in [*raw_tags, *(base_tags or [])]:
            cleaned = str(tag).strip()
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
        tmel_db = transient_mel_db(
            wave,
            sr,
            mask_band_count=cfg.augmentation.mask_band_count,
            mask_top_frames=cfg.augmentation.mask_top_frames,
        )
        technique = str(params.get("augmentation_technique") or "overlay")
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "category": DERIVED_CATEGORY,
            **_derived_provenance(source, technique),
            "label": output_label,
            "class_name": output_label,
            "project_id": project_id,
            # Inherit the base clip's density so derived buckets keep filtering
            # like their source material.
            "density": str(base_density or "").strip() or "normal",
            "review_status": "accepted",
            "recording_counter": idx,
            "recipe_id": str(params.get("recipe_id") or ""),
            "recipe_signature": str(params.get("recipe_signature") or ""),
            "generation_id": generation_id,
            "description": str(params.get("combination_description") or ""),
            "tags": tags,
            "source": _source_display_name(source),
            "base_source": str(params.get("base_source") or ""),
            "overlay_source": str(params.get("overlay_source") or ""),
            "base_label": output_label,
            "overlay_label": str(params.get("overlay_label") or ""),
            "augmentation": technique,
            "source_origin": _source_origin(source, layout),
            "augmentation_run_prefix": run_prefix_r,
            "augmentation_index": idx,
            "augmentation_params": params,
            "augmentation_settings": settings_snapshot,
            "seed": seed,
            "peak_dbfs": peak_dbfs,
            "rms_dbfs": rms_dbfs,
            "transient_mel_db": tmel_db,
            "effective_snr_db": params.get("effective_snr_db"),
            "mix_map": mix_map_name,
        }
        if masking:
            sidecar.update(masking)
        _write_sidecar(out_path, sidecar)
        lib.append_sidecar(out_path)

    counts: dict[str, dict[str, int]] = {}
    emit({"state": "running"})
    try:
        for recipe in resolved:
            check_cancel()
            recipe_id = str(recipe["recipe_id"])
            status["current_recipe"] = recipe_id
            base_paths = recipe["base_paths"]
            overlay_paths = recipe["overlay_paths"]
            base_buckets = recipe["base_buckets"]
            overlay_buckets = recipe["overlay_buckets"]
            to_generate = int(recipe.get("_to_generate") or 0)
            aug_root_r: Path = recipe["_aug_root"]
            run_prefix_r = recipe["_run_prefix"]
            recipe_expected_per_base = int(recipe.get("_expected_per_base") or 0)
            # Re-apply this recipe's snapshot to cfg so the closures
            # (draw_pitch / choose_candidate / write_sample) read the recipe's
            # own pitch / SNR / mask thresholds rather than the previous
            # recipe's. Restore the baseline first so partial recipe
            # snapshots don't leak fields from the previous iteration.
            _apply_settings_dict(cfg, baseline_settings)
            _apply_settings_dict(cfg, recipe.get("_settings") or {})
            settings_snapshot = augmentation_settings_snapshot(cfg)
            # Fill-up policy: nothing missing → skip without bumping status
            # (warnings already captured above for empty matches).
            recipe_overlay = bool(cfg.augmentation.overlay_enabled)
            if (
                not base_paths
                or (recipe_overlay and not overlay_paths)
                or (recipe.get("total_samples") is None and recipe_expected_per_base <= 0)
                or to_generate <= 0
            ):
                emit()
                continue
            overlay_pools = [
                _audio_pool_from_paths([p for p, _label in bucket], sr)
                for bucket in overlay_buckets
            ] if recipe_overlay else []
            overlay_labels = {p.resolve(): label for p, label in overlay_paths}
            indices: dict[str, int] = {}
            remaining = to_generate
            rejected = 0
            max_rejected = max(20, len(base_buckets) * max(1, cfg.augmentation.mask_candidate_attempts) * 4)
            while remaining > 0:
                check_cancel()
                drawn_base = _draw_bucket_path(base_buckets, rng)
                if drawn_base is None:
                    raise AugmentationDisabled(f"recipe {recipe_id}: no base clips matched")
                base_path, base_label = drawn_base
                loaded = _load_source(base_path)
                if loaded is None:
                    rejected += 1
                    status["failed"] += 1
                    if rejected >= max_rejected:
                        raise AugmentationDisabled(
                            f"recipe {recipe_id}: could not load a usable base clip after {rejected} random draws"
                        )
                    emit()
                    continue
                clean, base_sr = loaded
                try:
                    base_meta = json.loads(
                        base_path.with_suffix(".json").read_text(encoding="utf-8")
                    )
                except Exception:  # noqa: BLE001
                    base_meta = {}
                base_density = str(base_meta.get("density") or "").strip()
                base_tags = [
                    str(tag).strip()
                    for tag in (base_meta.get("tags") or [])
                    if str(tag).strip()
                ]
                idx = indices.setdefault(
                    base_label,
                    _next_augmentation_index(aug_root_r / safe_label(base_label), run_prefix_r, base_label),
                )
                if recipe_overlay:
                    candidate = choose_candidate(
                        clean_wave=clean,
                        base_sr=base_sr,
                        base_path=base_path,
                        base_label=base_label,
                        recipe=recipe,
                        overlay_pools=overlay_pools,
                        overlay_labels=overlay_labels,
                    )
                else:
                    candidate = base_only_candidate(
                        clean_wave=clean,
                        base_sr=base_sr,
                        base_path=base_path,
                        base_label=base_label,
                        recipe=recipe,
                    )
                if candidate is None:
                    rejected += 1
                    status["skipped"] += 1
                    if rejected >= max_rejected:
                        raise AugmentationDisabled(
                            f"recipe {recipe_id}: smart mask found no acceptable random overlay after {rejected} base draws"
                        )
                    emit()
                    continue
                write_sample(
                    output_label=base_label,
                    wave=candidate["wave"],
                    final_sr=candidate["final_sr"],
                    source=base_path,
                    idx=idx,
                    params=candidate["params"],
                    masking=candidate["masking"],
                    aug_root_r=aug_root_r,
                    run_prefix_r=run_prefix_r,
                    base_density=base_density,
                    base_tags=base_tags,
                    target_component=candidate.get("target_component"),
                    background_component=candidate.get("background_component"),
                )
                indices[base_label] = idx + 1
                label_counts = counts.setdefault(base_label, {})
                technique = str(candidate["params"].get("augmentation_technique") or "overlay")
                label_counts[technique] = label_counts.get(technique, 0) + 1
                status["completed"] += 1
                status["generated"] += 1
                remaining -= 1
                rejected = 0
                emit()
    finally:
        lib.rebuild_manifest()
    emit({"state": "finished"})
    return counts


def materialize_augmented(
    cfg: Config,
    project_id: str,
    seed: int | None = None,
    run_prefix: str | None = None,
    replace: bool = False,
    frozen_test: bool = False,
) -> dict[str, dict[str, int]]:
    """Generate the configured augmented samples to disk.

    Returns a nested dict ``{class_name: {technique: count}}``. By default,
    new samples are appended under a timestamped run prefix so multiple review
    batches can coexist. Pass ``replace=True`` to clear existing generated
    samples in the selected destination before writing the new batch. When
    ``frozen_test`` is true, bases come from library clips with
    composition rules and the selected frozen-test output destination decide
    whether the generated files enter train.csv or test.csv; sidecars remain
    composition-agnostic.
    """
    set_augment_features(cfg)
    layout = ProjectLayout(project_id)
    sr = cfg.model.sample_rate
    target_n = window_sample_count(sr, cfg.features.window_seconds)
    seed = seed if seed is not None else cfg.augmentation.seed
    run_prefix = run_prefix or _default_run_prefix(seed)
    rng = random.Random(seed)

    lib = Library()
    source_split = "test" if frozen_test else "train"
    derived_path_role = "test" if frozen_test else "trainval"
    aug_root = lib.root / DERIVED_CATEGORY / project_id / derived_path_role
    if replace and aug_root.exists():
        shutil.rmtree(aug_root)
    aug_root.mkdir(parents=True, exist_ok=True)

    if not cfg.augmentation.enabled:
        log.info("augmentation.enabled = false; nothing generated.")
        return {}
    if not cfg.augmentation.overlay_enabled:
        log.info("augmentation.overlay_enabled = false; nothing generated.")
        return {}
    if cfg.augmentation.combinations:
        return _materialize_target_augmented(
            cfg,
            project_id,
            seed=seed,
            run_prefix=run_prefix,
            replace=replace,
            frozen_test=frozen_test,
        )
    counts: dict[str, dict[str, int]] = {}

    # ── Per-positive-class techniques ───────────────────────────────────────
    pos_by_class: dict[str, list[Path]] = {
        c: [] for c in cfg.model.classes if c != "_negative"
    }
    from sed.data.compose import (
        _library_clips_for_class,
        load_dataset_selection,
    )

    selection = load_dataset_selection(layout)
    for class_name in pos_by_class:
        pos_by_class[class_name] = [
            Path(clip.path)
            for clip in _library_clips_for_class(
                cfg,
                project_id,
                class_name,
                split=source_split,
                selection=selection,
                include_derived=False,
            )
        ]

    negative_paths = [
        Path(clip.path)
        for clip in _library_clips_for_class(
            cfg,
            project_id,
            "_negative",
            split=source_split,
            selection=selection,
            include_derived=False,
        )
    ]
    negative_pool = _audio_pool_from_paths(negative_paths, sr)
    frozen_negative_paths = negative_paths if frozen_test else []
    frozen_negative_pool = negative_pool if frozen_test else _audio_pool_from_paths([], sr)
    overlay_negative_pool = negative_pool
    if frozen_test and len(overlay_negative_pool) == 0 and len(frozen_negative_pool) > 0:
        overlay_negative_pool = frozen_negative_pool
    if len(overlay_negative_pool) == 0:
        log.info("no negatives available for overlay augmentation.")
        return {}

    pos_classes = [
        (class_name, srcs)
        for class_name, srcs in sorted(pos_by_class.items())
        if srcs
    ]

    total_sources = sum(len(srcs) for _, srcs in pos_classes)
    if cfg.augmentation.generate_negative_overlay and "_negative" in cfg.model.classes:
        negative_base_pool = frozen_negative_pool if frozen_test else negative_pool
        if cfg.augmentation.negative_overlay_base_labels:
            total_sources += len(
                negative_base_pool.filtered(
                    cfg.augmentation.negative_overlay_base_labels
                ).paths
            )
        else:
            total_sources += len(
                frozen_negative_paths if frozen_test else negative_paths
            )

    def draw_negative(
        pool: AudioPool,
        *,
        avoid_path: Path | None = None,
    ) -> tuple[np.ndarray, int, Path] | None:
        if len(pool) == 0:
            return None
        for _ in range(5):
            drawn = pool.random_native(rng)
            if drawn is None:
                return None
            wave, native_sr, path = drawn
            if avoid_path is None or path.resolve() != avoid_path.resolve() or len(pool) <= 1:
                return wave, native_sr, path
        return pool.random_native(rng)

    def overlay_params(
        base_role: str,
        source: Path,
        negative_path: Path,
        snr_db: float,
        semitones: float,
        mix_params: dict,
    ) -> dict:
        return {
            **mix_params,
            "base_role": base_role,
            "source": _source_display_name(source),
            "negative_source": _source_display_name(negative_path),
            "pitch_shift_enabled": bool(cfg.augmentation.pitch_shift_enabled),
            "semitones": float(semitones),
            "overlay_enabled": bool(cfg.augmentation.overlay_enabled),
            "snr_db": float(snr_db),
        }

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[cyan]Augmenting...", total=total_sources)

        for cls, sources in pos_classes:
            cls_out = aug_root / cls
            cls_out.mkdir(parents=True, exist_ok=True)
            cls_counts: dict[str, int] = {}

            def write(wave: np.ndarray, final_sr: int, technique: str, params: dict, source: Path,
                      masking: float | dict | None, idx: int) -> None:
                out_path = unique_path(
                    cls_out / _augmentation_filename(run_prefix, cls, idx),
                    separator=_AUG_COLLISION_SEPARATOR,
                )
                sf.write(str(out_path), wave, final_sr, subtype="PCM_16")

                peak = float(np.max(np.abs(wave))) if wave.size else 0.0
                peak_dbfs = 20.0 * float(np.log10(peak + 1e-10)) if peak > 0 else -120.0
                rms = float(np.sqrt(np.mean(wave**2))) if wave.size else 0.0
                rms_dbfs = 20.0 * float(np.log10(rms + 1e-10)) if rms > 0 else -120.0
                tmel_db = transient_mel_db(
                    wave,
                    final_sr,
                    mask_band_count=cfg.augmentation.mask_band_count,
                    mask_top_frames=cfg.augmentation.mask_top_frames,
                )

                sidecar = {
                    "schema_version": SCHEMA_VERSION,
                    "category": DERIVED_CATEGORY,
                    **_derived_provenance(source, technique),
                    "label": cls,
                    "project_id": project_id,
                    "density": "normal",
                    "review_status": "accepted",
                    "recording_counter": idx,
                    "source": _source_display_name(source),
                    "class_name": cls,
                    "augmentation": technique,
                    "source_origin": _source_origin(source, layout),
                    "augmentation_run_prefix": run_prefix,
                    "augmentation_index": idx,
                    "augmentation_params": params,
                    "seed": seed,
                    "peak_dbfs": peak_dbfs,
                    "rms_dbfs": rms_dbfs,
                    "transient_mel_db": tmel_db,
                }
                if isinstance(masking, dict):
                    sidecar.update(masking)
                elif masking is not None:
                    sidecar["masking_score_db"] = float(masking)
                _write_sidecar(out_path, sidecar)
                lib.append_sidecar(out_path)
                cls_counts[technique] = cls_counts.get(technique, 0) + 1

            idx_counter: dict[str, int] = {
                "overlay": _next_augmentation_index(cls_out, run_prefix, cls)
            }

            def next_idx(technique: str) -> int:
                i = idx_counter.get(technique, 0)
                idx_counter[technique] = i + 1
                return i

            def mask_passes(masking: dict[str, object]) -> bool:
                return (
                    float(masking["masking_score_db"]) >= cfg.augmentation.mask_min_mean_db
                    and float(masking["masking_score_p25_db"]) >= cfg.augmentation.mask_min_p25_db
                    and float(masking["masking_score_min_db"]) >= cfg.augmentation.mask_min_band_db
                )

            def fallback_allowed(masking: dict[str, object]) -> bool:
                # Allow a near-miss, but not a mix where the lower quartile or
                # an important individual band is clearly buried.
                return (
                    float(masking["masking_score_p25_db"]) >= min(0.0, cfg.augmentation.mask_min_p25_db)
                    and float(masking["masking_score_min_db"]) >= cfg.augmentation.mask_min_band_db - 6.0
                )

            def build_candidate(
                clean_wave: np.ndarray,
                base_sr: int,
                source_path: Path,
                pool: AudioPool,
                min_snr_db: float | None,
            ) -> dict[str, object] | None:
                drawn = draw_negative(pool)
                if drawn is None:
                    return None
                negative_wave, negative_sr, negative_path = drawn
                final_sr = min(base_sr, negative_sr)

                if base_sr != final_sr:
                    base_wave_for_mix = resample_audio(clean_wave, base_sr, final_sr)
                else:
                    base_wave_for_mix = clean_wave

                if negative_sr != final_sr:
                    negative_wave_for_mix = resample_audio(negative_wave, negative_sr, final_sr)
                else:
                    negative_wave_for_mix = negative_wave

                target_n = len(base_wave_for_mix)
                negative_wave_for_mix = _ensure_length(negative_wave_for_mix, target_n, rng=rng)

                base_wave, semitones = _draw_pitch(cfg, rng, base_wave_for_mix, final_sr)
                snr_db = _draw_snr_db(cfg.augmentation.snr_db_range, rng, min_snr_db)
                out, target_component, background_component, mix_params = _mix_with_scaled_noise(
                    base_wave,
                    negative_wave_for_mix,
                    snr_db,
                    scale_noise_to_snr=cfg.augmentation.scale_noise_to_snr,
                    prevent_clipping=cfg.augmentation.prevent_clipping,
                    transient_window_ms=cfg.augmentation.transient_window_ms,
                    sample_rate=final_sr,
                )
                params = overlay_params(
                    "positive",
                    source_path,
                    negative_path,
                    snr_db,
                    semitones,
                    mix_params,
                )
                masking = masking_score_details(
                    target_component,
                    out,
                    final_sr,
                    background=background_component,
                    transient_window_ms=cfg.augmentation.transient_window_ms,
                    mask_band_count=cfg.augmentation.mask_band_count,
                    mask_top_frames=cfg.augmentation.mask_top_frames,
                )
                return {
                    "wave": out,
                    "final_sr": final_sr,
                    "params": params,
                    "source": source_path,
                    "masking": masking,
                }

            def choose_candidate(
                clean_wave: np.ndarray,
                base_sr: int,
                source_path: Path,
                pool: AudioPool,
                min_snr_db: float | None,
            ) -> dict[str, object] | None:
                attempts = (
                    max(1, int(cfg.augmentation.mask_candidate_attempts))
                    if cfg.augmentation.mask_filter_enabled
                    else 1
                )
                best: dict[str, object] | None = None
                for attempt in range(1, attempts + 1):
                    candidate = build_candidate(clean_wave, base_sr, source_path, pool, min_snr_db)
                    if candidate is None:
                        return best
                    masking = candidate["masking"]
                    params = candidate["params"]
                    assert isinstance(masking, dict)
                    assert isinstance(params, dict)
                    params.update({
                        "mask_filter_enabled": bool(cfg.augmentation.mask_filter_enabled),
                        "mask_attempts_used": attempt,
                        "mask_filter_passed": False,
                        "mask_fallback_used": False,
                    })
                    if best is None or _candidate_rank(candidate) > _candidate_rank(best):
                        best = candidate
                    if not cfg.augmentation.mask_filter_enabled or mask_passes(masking):
                        params["mask_filter_passed"] = True
                        return candidate

                if best is None:
                    return None
                best_masking = best["masking"]
                best_params = best["params"]
                assert isinstance(best_masking, dict)
                assert isinstance(best_params, dict)
                if fallback_allowed(best_masking):
                    best_params["mask_attempts_used"] = attempts
                    best_params["mask_filter_passed"] = False
                    best_params["mask_fallback_used"] = True
                    return best
                log.info(
                    "augment: skipped %s because best overlay was masked "
                    "(mean=%.1f p25=%.1f min=%.1f dB after %d attempts)",
                    source_path.name,
                    float(best_masking["masking_score_db"]),
                    float(best_masking["masking_score_p25_db"]),
                    float(best_masking["masking_score_min_db"]),
                    attempts,
                )
                return None

            for src in sources:
                progress.update(task_id, description=f"[cyan]Augmenting {cls}: {src.name[:20]}...")
                loaded = _load_source(src)
                if loaded is None or not cfg.augmentation.generate_positive_overlay:
                    progress.advance(task_id)
                    continue
                clean, base_sr = loaded
                per_class_pool = overlay_negative_pool.filtered(
                    cfg.augmentation.per_class_negative_label.get(cls, [])
                )
                if len(per_class_pool) == 0:
                    per_class_pool = overlay_negative_pool
                min_snr = cfg.augmentation.per_class_min_snr_db.get(cls)
                for _ in range(_get_n_samples(cfg.augmentation.samples_per_source, rng)):
                    candidate = choose_candidate(clean, base_sr, src, per_class_pool, min_snr)
                    if candidate is None:
                        break
                    write(
                        candidate["wave"],
                        candidate["final_sr"],
                        "overlay",
                        candidate["params"],
                        src,
                        candidate["masking"],
                        next_idx("overlay"),
                    )

                progress.advance(task_id)

            if cls_counts:
                counts[cls] = cls_counts

    # ── negative + negative → _negative bucket ─────────────────────────────
    if (
        cfg.augmentation.generate_negative_overlay
        and "_negative" in cfg.model.classes
        and (frozen_negative_paths if frozen_test else negative_paths)
    ):
        neg_out = aug_root / "_negative"
        neg_out.mkdir(parents=True, exist_ok=True)
        neg_counts: dict[str, int] = {}

        if cfg.augmentation.negative_overlay_base_labels:
            base_source_pool = frozen_negative_pool if frozen_test else negative_pool
            base_pool = base_source_pool.filtered(cfg.augmentation.negative_overlay_base_labels)
            overlay_pool = overlay_negative_pool.excluded(cfg.augmentation.negative_overlay_base_labels)
        else:
            base_pool = frozen_negative_pool if frozen_test else negative_pool
            overlay_pool = overlay_negative_pool
        if len(overlay_pool) == 0:
            overlay_pool = overlay_negative_pool

        base_paths = list(base_pool.paths)
        if not base_paths:
            log.warning("No negative base clips found matching %s", cfg.augmentation.negative_overlay_base_labels)
            return counts

        per_base_counts = [
            _get_n_samples(cfg.augmentation.samples_per_source, rng)
            for _ in base_paths
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ) as neg_progress:
            neg_task = neg_progress.add_task(
                "[cyan]Generating _negative...",
                total=sum(per_base_counts),
            )

            idx = _next_augmentation_index(neg_out, run_prefix, "_negative")
            for base_path, n_samples in zip(base_paths, per_base_counts, strict=True):
                loaded = _load_native(base_path)
                if loaded is None:
                    continue
                base_clean, base_sr = loaded
                for _ in range(n_samples):
                    drawn = draw_negative(overlay_pool, avoid_path=base_path)
                    if drawn is None:
                        break
                    negative_wave, negative_sr, negative_path = drawn
                    final_sr = min(base_sr, negative_sr)
                    
                    if base_sr != final_sr:
                        base_wave_for_mix = resample_audio(base_clean, base_sr, final_sr)
                    else:
                        base_wave_for_mix = base_clean
                        
                    if negative_sr != final_sr:
                        negative_wave_for_mix = resample_audio(negative_wave, negative_sr, final_sr)
                    else:
                        negative_wave_for_mix = negative_wave
                        
                    target_n = len(base_wave_for_mix)
                    negative_wave_for_mix = _ensure_length(negative_wave_for_mix, target_n, rng=rng)
                    
                    base_wave, semitones = _draw_pitch(cfg, rng, base_wave_for_mix, final_sr)
                    snr_db = rng.uniform(*cfg.augmentation.snr_db_range)
                    out, _, _, mix_params = _mix_with_scaled_noise(
                        base_wave,
                        negative_wave_for_mix,
                        snr_db,
                        scale_noise_to_snr=cfg.augmentation.scale_noise_to_snr,
                        prevent_clipping=cfg.augmentation.prevent_clipping,
                        transient_window_ms=cfg.augmentation.transient_window_ms,
                        sample_rate=final_sr,
                    )
                    params = overlay_params(
                        "negative",
                        base_path,
                        negative_path,
                        snr_db,
                        semitones,
                        mix_params,
                    )
                    out_path = unique_path(
                        neg_out / _augmentation_filename(run_prefix, "_negative", idx),
                        separator=_AUG_COLLISION_SEPARATOR,
                    )
                    sf.write(str(out_path), out, final_sr, subtype="PCM_16")

                    peak = float(np.max(np.abs(out))) if out.size else 0.0
                    peak_dbfs = 20.0 * float(np.log10(peak + 1e-10)) if peak > 0 else -120.0
                    rms = float(np.sqrt(np.mean(out**2))) if out.size else 0.0
                    rms_dbfs = 20.0 * float(np.log10(rms + 1e-10)) if rms > 0 else -120.0
                    tmel_db = transient_mel_db(
                        out,
                        final_sr,
                        mask_band_count=cfg.augmentation.mask_band_count,
                        mask_top_frames=cfg.augmentation.mask_top_frames,
                    )

                    sidecar = {
                        "schema_version": SCHEMA_VERSION,
                        "category": DERIVED_CATEGORY,
                        **_derived_provenance(base_path, "overlay"),
                        "label": "_negative",
                        "project_id": project_id,
                        "density": "normal",
                        "review_status": "accepted",
                        "recording_counter": idx,
                        "source": _source_display_name(base_path),
                        "class_name": "_negative",
                        "augmentation": "overlay",
                        "source_origin": _source_origin(base_path, layout),
                        "augmentation_run_prefix": run_prefix,
                        "augmentation_index": idx,
                        "augmentation_params": params,
                        "seed": seed,
                        "peak_dbfs": peak_dbfs,
                        "rms_dbfs": rms_dbfs,
                        "transient_mel_db": tmel_db,
                    }
                    _write_sidecar(out_path, sidecar)
                    lib.append_sidecar(out_path)
                    idx += 1
                    neg_counts["overlay"] = neg_counts.get("overlay", 0) + 1
                    neg_progress.advance(neg_task)
        if neg_counts:
            counts["_negative"] = neg_counts

    lib.rebuild_manifest()
    return counts
