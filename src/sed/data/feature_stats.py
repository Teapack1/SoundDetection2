"""Dataset-fitted feature scaling statistics (the generalized normalization fit).

Single implementation shared by the Audio Features "Estimate from dataset"
action (``/api/feature-stats-estimate``), the Train-launch preflight refit, and
``sed train``'s validation. Fits constants for ``features.scaling_method``
(``zscore`` | ``minmax``) at ``features.scaling_scope`` (``global`` |
``per_band``) on the SAME live grouped training split that training will use —
validation clips never contribute — with clips loaded exactly the way
``SedDataset._load`` feeds training (first channel, deterministic center crop).

The fit runs in the compression-only domain (natural-log power or PCEN output,
before any scaling), so the fitted constants land on the values the model will
actually see once scaling is applied.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

# Legacy callers may still request a bounded diagnostic fit. Canonical fits
# pass no limit and consume every row in the grouped training side.
MAX_STATS_CLIPS = 2_048
_BAND_SAMPLE_CAP = 4_096
_STATS_CACHE_FILENAME = "feature_scaling_cache.json"

# Robust min-max edges: p1/p99 of the compressed values, so a handful of
# outlier frames cannot stretch the scale; values outside clip to [0, 1].
MINMAX_PERCENTILES: tuple[float, float] = (1.0, 99.0)


def load_training_window(
    path: str | Path, sample_rate: int, window_seconds: float
) -> np.ndarray:
    """Load one clip exactly as ``SedDataset._load`` feeds training.

    First channel (not a downmix), resampled to the model rate, deterministic
    center-crop / end-pad to one window. The feature-lab preview loader
    (``load_feature_window``) head-crops and mean-downmixes, which diverges
    from training whenever the window is shorter than the clip — fitted
    statistics must see the training distribution.
    """
    from sed.audio.features import window_sample_count
    from sed.audio.resample import resample_audio
    from sed.data.augment import _ensure_length

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[0] == 0:
        raise RuntimeError(f"failed to read audio {str(path)!r}: empty audio")
    wave = data[:, 0]
    if int(sr) != int(sample_rate):
        wave = resample_audio(wave, orig_sr=int(sr), target_sr=int(sample_rate))
    return _ensure_length(wave, window_sample_count(int(sample_rate), float(window_seconds)))


def balanced_stats_rows(rows: list[dict], *, limit: int, seed: int) -> list[dict]:
    """Choose statistics clips evenly across classes and source groups.

    The caller passes the *training side* of the live grouped split.  A seeded
    round-robin first balances classes and then source groups within each
    class; rows within a group are shuffled deterministically.  This prevents
    one large recording session from dominating fitted frontend statistics.
    """
    from sed.data.splits import split_group_key

    maximum = max(0, int(limit))
    if maximum == 0:
        return []
    rng = random.Random(int(seed))
    grouped: dict[str, dict[str, list[dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list),
    )
    for row in rows:
        class_name = str(row.get("class_name") or "").strip() or "_unknown"
        group = str(split_group_key(row) or row.get("path") or "").strip()
        if not group:
            continue
        grouped[class_name][group].append(row)

    class_groups: dict[str, list[list[dict]]] = {}
    for class_name in sorted(grouped):
        queues: list[list[dict]] = []
        group_names = sorted(grouped[class_name])
        rng.shuffle(group_names)
        for group_name in group_names:
            queue = list(grouped[class_name][group_name])
            rng.shuffle(queue)
            queues.append(queue)
        if queues:
            class_groups[class_name] = queues

    selected: list[dict] = []
    group_cursor = {class_name: 0 for class_name in class_groups}
    active = sorted(class_groups)
    while active and len(selected) < maximum:
        progressed = False
        for class_name in list(active):
            queues = class_groups[class_name]
            start = group_cursor[class_name] % len(queues)
            chosen: list[dict] | None = None
            chosen_index = start
            for offset in range(len(queues)):
                index = (start + offset) % len(queues)
                if queues[index]:
                    chosen = queues[index]
                    chosen_index = index
                    break
            if chosen is None:
                active.remove(class_name)
                continue
            selected.append(chosen.pop())
            group_cursor[class_name] = (chosen_index + 1) % len(queues)
            progressed = True
            if len(selected) >= maximum:
                break
        if not progressed:
            break
    return selected


def merge_value_moments(
    count: int, mean: float, m2: float, values: np.ndarray
) -> tuple[int, float, float]:
    """Merge one feature batch into global Welford population moments."""
    array = np.asarray(values, dtype=np.float64)
    batch_count = int(array.size)
    if batch_count == 0:
        return count, mean, m2
    batch_mean = float(array.mean(dtype=np.float64))
    batch_m2 = float(np.square(array - batch_mean, dtype=np.float64).sum())
    if count == 0:
        return batch_count, batch_mean, batch_m2
    total = int(count) + batch_count
    delta = batch_mean - float(mean)
    merged_mean = float(mean) + delta * batch_count / total
    merged_m2 = float(m2) + batch_m2 + delta * delta * count * batch_count / total
    return total, merged_mean, merged_m2


def _merge_band_moments(
    count: int, mean: np.ndarray | None, m2: np.ndarray | None, values: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    """Merge one ``(B, F, T)`` batch into per-band Welford moments ``(F,)``."""
    array = np.asarray(values, dtype=np.float64)
    batch_count = int(array.shape[0] * array.shape[2])
    batch_mean = array.mean(axis=(0, 2))
    batch_m2 = np.square(array - batch_mean[None, :, None]).sum(axis=(0, 2))
    if count == 0 or mean is None or m2 is None:
        return batch_count, batch_mean, batch_m2
    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / total)
    merged_m2 = m2 + batch_m2 + np.square(delta) * (count * batch_count / total)
    return total, merged_mean, merged_m2


class _BandSample:
    """Bounded deterministic per-band value subsample for percentiles.

    Keeps roughly ``cap`` values per band by doubling the keep-stride whenever
    the buffer overflows — deterministic (no RNG) and order-stable, so repeated
    fits over the same sample produce identical percentiles.
    """

    def __init__(self, cap: int = _BAND_SAMPLE_CAP):
        self.cap = int(cap)
        self.parts: list[np.ndarray] = []
        self.columns = 0
        self.stride = 1

    def add(self, values: np.ndarray) -> None:
        """values: (B, F, T) — flattened per band to (F, B·T) before striding."""
        array = np.asarray(values, dtype=np.float32)
        flat = np.transpose(array, (1, 0, 2)).reshape(array.shape[1], -1)
        take = flat[:, :: self.stride]
        self.parts.append(take)
        self.columns += take.shape[1]
        while self.columns > 2 * self.cap:
            self.parts = [part[:, ::2] for part in self.parts]
            self.columns = sum(part.shape[1] for part in self.parts)
            self.stride *= 2

    def matrix(self) -> np.ndarray:
        """(F, N) matrix of retained values."""
        if not self.parts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(self.parts, axis=1)


def _bare_extractor(features: Any, sample_rate: int):
    """Extractor for the compression-only domain — no normalization applied.

    Statistics have to be fitted on the values the normalization stage will
    consume, so every extractor here is built with an explicitly empty
    :class:`~sed.audio.features.ScalingSpec`. Any compression works (log, dB,
    PCEN, or none): what the constants mean is simply the domain they were
    measured in, and the same pair is replayed at train, eval and deploy time.
    """
    from sed.audio.features import (
        CedFeatureExtractor,
        FeatureExtractor,
        LinearExtractor,
        MelExtractor,
        ScalingSpec,
        feature_compression,
        pcen_params_from_features,
    )

    feature_type = str(getattr(features, "feature_type", "mel") or "mel").lower()
    compression = feature_compression(features)
    common = {
        "sample_rate": int(sample_rate),
        "n_fft": int(features.n_fft),
        "win_length": int(features.win_length),
        "hop_length": int(features.hop_length),
        "fmin": int(features.fmin),
        "fmax": int(features.fmax),
    }
    identity = ScalingSpec()
    if str(getattr(features, "frontend", "efficientat") or "efficientat").lower() == "ced":
        # CED's power→dB mapping IS its compression stage.
        return CedFeatureExtractor(
            **common, n_mels=int(features.n_mels), scaling=identity,
        )
    if str(getattr(features, "colormap", "none") or "none").lower() != "none":
        raise ValueError(
            "a colormap image render is already a [0, 1] picture — there is no "
            "value domain left to fit scaling constants on"
        )
    if feature_type == "linear":
        if str(getattr(features, "stft_engine", "librosa") or "librosa").lower() != "librosa":
            raise ValueError(
                "fitted scaling on a linear feature requires the librosa/Torch STFT contract"
            )
        if compression == "pcen":
            return FeatureExtractor(
                **common, n_mels=int(features.n_mels), feature_type="linear",
                compression="pcen", pcen_params=pcen_params_from_features(features),
                frontend="efficientat", scaling=identity,
            )
        return LinearExtractor(**common, compression=compression, scaling=identity)
    if compression == "pcen":
        return FeatureExtractor(
            **common, n_mels=int(features.n_mels), feature_type="mel",
            compression="pcen", pcen_params=pcen_params_from_features(features),
            frontend="efficientat", scaling=identity,
        )
    return MelExtractor(
        **common, n_mels=int(features.n_mels), frontend="efficientat",
        compression=compression, scaling=identity,
    )


def _finalize_stats(
    method: str,
    scope: str,
    *,
    global_mean: float,
    global_std: float,
    band_mean: np.ndarray,
    band_std: np.ndarray,
    band_values: np.ndarray,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Fitted stats for the requested method x scope, plus the health report.

    Degenerate bands (empty mel filters sit at a constant floor, so their std
    or percentile span is ~0) are clamped rather than allowed to explode the
    scale; the health report counts them.
    """
    round6 = lambda arr: [round(float(v), 6) for v in np.asarray(arr).ravel()]  # noqa: E731

    if method == "zscore":
        floor = max(1e-6, 1e-3 * global_std)
        if scope == "per_band":
            clamped = int(np.count_nonzero(band_std < floor))
            std_vec = np.maximum(band_std, floor)
            stats = {"mean": round6(band_mean), "std": round6(std_vec)}
            scaled = (band_values - band_mean[:, None]) / std_vec[:, None]
        else:
            clamped = 0
            stats = {"mean": [round(global_mean, 8)], "std": [round(max(global_std, floor), 8)]}
            scaled = (band_values - global_mean) / max(global_std, floor)
        per_band_mean = scaled.mean(axis=1) if scaled.size else np.zeros(0)
        per_band_std = scaled.std(axis=1) if scaled.size else np.zeros(0)
        health = {
            "post_mean": round(float(scaled.mean()), 4) if scaled.size else 0.0,
            "post_std": round(float(scaled.std()), 4) if scaled.size else 0.0,
            "worst_band_mean_abs": (
                round(float(np.abs(per_band_mean).max()), 4) if per_band_mean.size else 0.0
            ),
            "worst_band_std_dev": (
                round(float(np.abs(per_band_std - 1.0).max()), 4) if per_band_std.size else 0.0
            ),
            "clamped_bands": clamped,
        }
        return stats, health

    lo_p, hi_p = MINMAX_PERCENTILES
    if scope == "per_band":
        lo = np.percentile(band_values, lo_p, axis=1)
        hi = np.percentile(band_values, hi_p, axis=1)
        span_floor = max(1e-3, 1e-3 * float(hi.max() - lo.min() or 1.0))
        degenerate = int(np.count_nonzero(hi - lo < span_floor))
        hi = np.where(hi - lo < span_floor, lo + span_floor, hi)
        stats = {"lo": round6(lo), "hi": round6(hi)}
        scaled = (band_values - lo[:, None]) / (hi - lo)[:, None]
    else:
        lo_v = float(np.percentile(band_values, lo_p))
        hi_v = float(np.percentile(band_values, hi_p))
        if hi_v - lo_v < 1e-3:
            hi_v = lo_v + 1e-3
            degenerate = 1
        else:
            degenerate = 0
        stats = {"lo": [round(lo_v, 8)], "hi": [round(hi_v, 8)]}
        scaled = (band_values - lo_v) / (hi_v - lo_v)
    clipped = (scaled < 0.0) | (scaled > 1.0)
    per_band_clipped = clipped.mean(axis=1) if clipped.size else np.zeros(0)
    health = {
        "clipped_fraction": round(float(clipped.mean()), 4) if clipped.size else 0.0,
        "worst_band_clipped_fraction": (
            round(float(per_band_clipped.max()), 4) if per_band_clipped.size else 0.0
        ),
        "degenerate_bands": degenerate,
    }
    return stats, health


def fit_feature_scaling(
    cfg: Any,
    layout: Any,
    *,
    features: Any,
    sample_rate: int,
    limit: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit balanced, train-only scaling statistics for the active live split.

    ``features.scaling_method`` selects what is fitted (``zscore`` → mean/std,
    ``minmax`` → robust p1/p99 lo/hi) and ``features.scaling_scope`` at which
    granularity (one global pair, or one pair per feature band). Everything is
    accumulated in one pass: global + per-band Welford moments and a bounded
    per-band value subsample for percentiles and the health report.
    """
    from sed.data.splits import split_group_key, split_pool_rows_detailed
    from sed.util import sha256_file

    method = str(getattr(features, "scaling_method", "") or "").lower()
    scope = str(getattr(features, "scaling_scope", "global") or "global").lower()
    if method not in ("zscore", "minmax"):
        raise ValueError(
            "fit_feature_scaling needs features.scaling_method of 'zscore' or 'minmax'"
        )
    if scope not in ("global", "per_band"):
        raise ValueError("features.scaling_scope must be 'global' or 'per_band'")

    train_csv = layout.dataset_dir / "train.csv"
    rows: list[dict] = []
    if train_csv.exists():
        with open(train_csv, encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if row.get("path")]
    if not rows:
        raise ValueError("no composed train.csv — compose the dataset first")

    split_seed = int(cfg.training.seed)
    val_fraction = float(cfg.dataset.val_fraction)
    train_rows, _val_rows, _diagnostics = split_pool_rows_detailed(
        rows, val_frac=val_fraction, seed=split_seed,
    )
    sample = (
        balanced_stats_rows(
            train_rows, limit=max(1, int(limit)), seed=split_seed,
        )
        if limit is not None
        else list(train_rows)
    )
    if not sample:
        raise ValueError("the grouped split has no training clips")

    dataset_sha256 = sha256_file(train_csv)
    feature_fields = (
        "feature_type", "frontend", "compression", "n_fft", "win_length",
        "hop_length", "n_mels", "fmin", "fmax", "window_seconds", "log_power",
        "stft_engine", "db_norm", "db_min", "db_max", "pcen",
        "pcen_time_constant", "pcen_gain", "pcen_bias", "pcen_power", "pcen_eps",
        "scaling_method", "scaling_scope", "scaling_source",
    )
    feature_signature = {
        key: getattr(features, key, None) for key in feature_fields
    }
    cache_key_payload = {
        "dataset_sha256": dataset_sha256,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "sample_rate": int(sample_rate),
        "feature_signature": feature_signature,
        "limit": int(limit) if limit is not None else None,
    }
    cache_key = hashlib.sha256(
        json.dumps(
            cache_key_payload, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
    ).hexdigest()
    cache_path = layout.dataset_dir / _STATS_CACHE_FILENAME
    try:
        cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache_payload = {}
    cached = (
        (cache_payload.get("entries") or {}).get(cache_key)
        if isinstance(cache_payload, dict)
        else None
    )
    if isinstance(cached, dict):
        result = dict(cached)
        result["cache_hit"] = True
        if progress is not None:
            progress({
                "phase": "cache hit", "completed": len(sample), "total": len(sample),
                "groups": result.get("training_groups", 0),
            })
        return result

    if progress is not None:
        progress({
            "phase": "fitting", "completed": 0, "total": len(sample), "groups": 0,
        })

    extractor = _bare_extractor(features, int(sample_rate))
    window_seconds = float(getattr(features, "window_seconds", 1.0))

    total = 0
    mean = 0.0
    m2 = 0.0
    band_count = 0
    band_mean: np.ndarray | None = None
    band_m2: np.ndarray | None = None
    band_sample = _BandSample()
    used = 0
    batch: list[np.ndarray] = []
    batch_paths: list[str] = []
    batch_groups: list[str] = []
    used_groups: set[str] = set()

    def consume_batch() -> None:
        nonlocal total, mean, m2, band_count, band_mean, band_m2
        nonlocal used, batch, batch_paths, batch_groups
        if not batch:
            return
        try:
            values = extractor(np.stack(batch, axis=0))
        except Exception as exc:
            raise RuntimeError(
                "feature extraction failed for training batch containing: "
                + ", ".join(batch_paths[:8])
            ) from exc
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float32)
        total, mean, m2 = merge_value_moments(total, mean, m2, values)
        band_count, band_mean, band_m2 = _merge_band_moments(
            band_count, band_mean, band_m2, values
        )
        band_sample.add(values)
        used += len(batch)
        used_groups.update(batch_groups)
        batch = []
        batch_paths = []
        batch_groups = []
        if progress is not None:
            progress({
                "phase": "fitting", "completed": used, "total": len(sample),
                "current_name": "", "groups": len(used_groups),
            })

    for row in sample:
        path = str(row["path"])
        try:
            batch.append(load_training_window(
                path, int(sample_rate), window_seconds,
            ))
            batch_paths.append(path)
            batch_groups.append(str(split_group_key(row)))
        except Exception as exc:
            raise RuntimeError(
                f"dataset-wide scaling fit could not read training clip: {path}: {exc}"
            ) from exc
        if len(batch) >= 32:
            consume_batch()
    consume_batch()
    if not total or band_mean is None or band_m2 is None:
        raise RuntimeError("no readable clips in the sample")

    global_std = max(m2 / total, 1e-12) ** 0.5
    band_std = np.sqrt(np.maximum(band_m2 / max(band_count, 1), 0.0))
    stats, health = _finalize_stats(
        method,
        scope,
        global_mean=mean,
        global_std=global_std,
        band_mean=band_mean,
        band_std=band_std,
        band_values=band_sample.matrix().astype(np.float64),
    )
    result = {
        "method": method,
        "scope": scope,
        "stats": stats,
        "health": health,
        "bands": int(band_mean.shape[0]),
        # Legacy scalars kept for the global z-score consumers/UI fields.
        "mean": round(mean, 8),
        "std": round(global_std, 8),
        "clips": used,
        "dataset_sha256": dataset_sha256,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "training_rows": len(train_rows),
        "training_groups": len({split_group_key(row) for row in train_rows}),
        "feature_signature": feature_signature,
        "cache_key": cache_key,
        "cache_hit": False,
    }
    entries = cache_payload.get("entries") if isinstance(cache_payload, dict) else None
    if not isinstance(entries, dict):
        entries = {}
    entries[cache_key] = result
    # Keep the cache compact while retaining the most recent signatures.
    entries = dict(list(entries.items())[-16:])
    payload = {"schema_version": 1, "entries": entries}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    tmp.replace(cache_path)
    return result


# How far a per-band mean may sit from the fitted one, measured in fitted
# standard deviations, before the encoder is being fed a distribution it was not
# fitted on. A z-scored input is supposed to land at ~0 ± 1, so a whole sigma of
# offset is already a large systematic shift.
DRIFT_WARN_SIGMA = 0.5
DRIFT_ERROR_SIGMA = 1.0
# Same idea for min-max: values outside [lo, hi] are clipped, so a large clipped
# fraction means real dynamic range is being flattened away.
DRIFT_WARN_CLIPPED = 0.05
DRIFT_ERROR_CLIPPED = 0.15
# Bounded sample — this runs on a page-load path, not as part of a fit.
DRIFT_SAMPLE_CLIPS = 256


_DRIFT_CACHE_FILENAME = "feature_scaling_drift.json"


def _drift_cache_key(cfg: Any, layout: Any) -> str:
    """Identity of a drift measurement: this feature scale on this train pool."""
    from sed.util import sha256_file

    features = cfg.features
    train_csv = layout.dataset_dir / "train.csv"
    payload = {
        "dataset_sha256": sha256_file(train_csv) if train_csv.exists() else "",
        "sample_rate": int(cfg.model.sample_rate),
        "seed": int(cfg.training.seed),
        "scaling_method": str(getattr(features, "scaling_method", "none")),
        "scaling_scope": str(getattr(features, "scaling_scope", "global")),
        "scaling_stats": {
            str(k): [round(float(v), 6) for v in (vals or [])]
            for k, vals in (getattr(features, "scaling_stats", {}) or {}).items()
        },
        "geometry": [
            str(getattr(features, "feature_type", "mel")),
            str(getattr(features, "frontend", "efficientat")),
            int(features.n_fft), int(features.win_length), int(features.hop_length),
            int(features.n_mels), int(features.fmin), int(features.fmax),
            float(features.window_seconds), bool(getattr(features, "pcen", False)),
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def load_cached_scaling_drift(cfg: Any, layout: Any) -> dict[str, Any] | None:
    """A previously measured drift result, if it still describes this config.

    Reading audio is too slow for a page-load endpoint, so the advice payload
    reports whatever the last explicit measurement found and nothing when the
    dataset or the scale has changed since.
    """
    try:
        payload = json.loads(
            (layout.dataset_dir / _DRIFT_CACHE_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        key = _drift_cache_key(cfg, layout)
    except Exception:
        return None
    entry = payload.get("entry")
    return entry if isinstance(entry, dict) and payload.get("key") == key else None


def measure_scaling_drift(
    cfg: Any,
    layout: Any,
    *,
    limit: int = DRIFT_SAMPLE_CLIPS,
) -> dict[str, Any]:
    """How well this project's clips fit the scaling constants currently active.

    A fine-tune inherits its parent's fitted constants on purpose — they are part
    of the encoder's numeric contract, and refitting them on venue data would
    silently move the input distribution the transferred weights expect. That
    makes the opposite question the important one: are the venue clips still
    *inside* the scale the parent fitted? A microphone change, a quieter room or
    a different gain staging shows up here as a systematic offset long before it
    shows up as a confusing validation curve.

    Measured in the same compression-only domain the fit uses
    (:func:`_bare_extractor`), on a bounded class/group-balanced sample of the
    composed training pool. Returns ``{"status": "ok"|"warn"|"error"|"skipped"}``
    plus the numbers behind it.
    """
    from sed.audio.features import feature_band_count, resolve_feature_scaling
    from sed.data.splits import split_group_key

    features = cfg.features
    sample_rate = int(cfg.model.sample_rate)
    spec = resolve_feature_scaling(features, n_bands=feature_band_count(features, sample_rate))
    if not spec.active:
        return {"status": "skipped", "reason": "no scaling is active"}
    if spec.source != "dataset":
        # Fixed contract constants (the AudioSet fast-norm, the legacy dB window)
        # are not fitted on anything, so "did this dataset drift out of the fit"
        # is not a question they can answer. The Model tab's contract report is
        # what says whether a feature suits those constants.
        return {
            "status": "skipped",
            "reason": "the active scaling uses fixed contract constants, not values fitted on a dataset",
        }

    train_csv = layout.dataset_dir / "train.csv"
    if not train_csv.exists():
        return {"status": "skipped", "reason": "no composed train.csv"}
    with open(train_csv, encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("path")]
    if not rows:
        return {"status": "skipped", "reason": "no composed train.csv"}

    sample = balanced_stats_rows(rows, limit=max(1, int(limit)), seed=int(cfg.training.seed))
    if not sample:
        return {"status": "skipped", "reason": "no clips to measure"}

    extractor = _bare_extractor(features, sample_rate)
    window_seconds = float(getattr(features, "window_seconds", 1.0))
    band_count = 0
    band_mean: np.ndarray | None = None
    band_m2: np.ndarray | None = None
    clipped_total = 0
    value_total = 0
    batch: list[np.ndarray] = []
    used = 0

    def consume() -> None:
        nonlocal band_count, band_mean, band_m2, batch, used, clipped_total, value_total
        if not batch:
            return
        values = extractor(np.stack(batch, axis=0))
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float32)
        band_count, band_mean, band_m2 = _merge_band_moments(
            band_count, band_mean, band_m2, values,
        )
        if spec.method == "minmax":
            lo = spec.a.reshape((-1, 1)) if spec.a.size > 1 else float(spec.a[0])
            hi = spec.b.reshape((-1, 1)) if spec.b.size > 1 else float(spec.b[0])
            outside = (values < lo) | (values > hi)
            clipped_total += int(outside.sum())
            value_total += int(outside.size)
        used += len(batch)
        batch = []

    for row in sample:
        try:
            batch.append(load_training_window(str(row["path"]), sample_rate, window_seconds))
        except Exception as exc:  # one unreadable clip is not a verdict
            log.warning("scaling drift: skipping unreadable clip %s: %s", row["path"], exc)
            continue
        if len(batch) >= 32:
            consume()
    consume()
    if band_mean is None or band_m2 is None or not used:
        return {"status": "skipped", "reason": "no readable clips"}

    observed_mean = band_mean
    observed_std = np.sqrt(np.maximum(band_m2 / max(band_count, 1), 0.0))
    out: dict[str, Any] = {
        "method": spec.method,
        "scope": "per_band" if spec.a.size > 1 else "global",
        "clips": used,
        "groups": len({split_group_key(row) for row in sample}),
        "bands": int(observed_mean.shape[0]),
    }

    if spec.method == "zscore":
        fitted_mean = spec.a.astype(np.float64)
        fitted_std = spec.b.astype(np.float64)
        if fitted_mean.size == 1:
            fitted_mean = np.full(observed_mean.shape, float(fitted_mean[0]))
            fitted_std = np.full(observed_std.shape, float(fitted_std[0]))
        sigma = np.abs(observed_mean - fitted_mean) / np.maximum(fitted_std, 1e-6)
        ratio = observed_std / np.maximum(fitted_std, 1e-6)
        worst = int(np.argmax(sigma))
        out.update({
            "max_mean_shift_sigma": round(float(sigma.max()), 3),
            "median_mean_shift_sigma": round(float(np.median(sigma)), 3),
            "worst_band": worst,
            "worst_band_std_ratio": round(float(ratio[worst]), 3),
            "min_std_ratio": round(float(ratio.min()), 3),
            "max_std_ratio": round(float(ratio.max()), 3),
        })
        peak = float(sigma.max())
        out["status"] = (
            "error" if peak >= DRIFT_ERROR_SIGMA
            else "warn" if peak >= DRIFT_WARN_SIGMA
            else "ok"
        )
    else:
        fraction = (clipped_total / value_total) if value_total else 0.0
        out["clipped_fraction"] = round(float(fraction), 4)
        out["status"] = (
            "error" if fraction >= DRIFT_ERROR_CLIPPED
            else "warn" if fraction >= DRIFT_WARN_CLIPPED
            else "ok"
        )

    # Cached so the Recommendations panel can report the last measurement
    # without re-reading audio on every page load.
    try:
        cache_path = layout.dataset_dir / _DRIFT_CACHE_FILENAME
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"key": _drift_cache_key(cfg, layout), "entry": out},
                       sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(cache_path)
    except OSError as exc:
        log.warning("could not cache the scaling drift measurement: %s", exc)
    return out
