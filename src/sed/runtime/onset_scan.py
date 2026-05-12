"""Measure the onset gate on recorded audio.

Production with ``[inference] onset_gating_enabled`` scores exactly one window
per onset the detector emits, so **the onset detector is a hard ceiling on
system recall** — a clink it misses is one the model never sees — and it sets
the false-alarm denominator, since FP/hour is ``onsets_per_hour × P(fire|onset)``.

Neither number has ever been measured. This module runs just the gate (no model,
no annotation) over recorded audio and reports both, plus the flux distribution
against the configured trigger threshold so ``[onset] mad_k`` / ``min_flux_db``
can be tuned against evidence rather than guessed.

It reuses the production detector and merge rule (`StreamingSuperflux` +
`OnsetMerger`), so what it counts is what the runtime would have scored.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from sed.audio.onset import OnsetMerger, StreamingSuperflux
from sed.audio.resample import resample_audio
from sed.config import Config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnsetScanResult:
    """Gate behaviour on one recording."""

    path: str
    duration_s: float
    sample_rate: int
    channels: list[int]
    # Per-mic onset times in seconds, before the cross-channel merge.
    per_channel_onsets_s: dict[int, list[float]] = field(default_factory=dict)
    # The merged stream — what onset-gated inference would actually have scored.
    merged_onsets_s: list[float] = field(default_factory=list)
    # Per-block (flux, adaptive threshold) samples for the histogram.
    flux_samples: list[float] = field(default_factory=list)
    gate_samples: list[float] = field(default_factory=list)
    min_flux: float = 0.0

    @property
    def raw_onset_count(self) -> int:
        return sum(len(v) for v in self.per_channel_onsets_s.values())

    @property
    def merged_onset_count(self) -> int:
        return len(self.merged_onsets_s)

    @property
    def onsets_per_hour(self) -> float:
        """The conversion factor between per-onset and per-hour false-alarm rates."""
        if self.duration_s <= 0:
            return 0.0
        return self.merged_onset_count / self.duration_s * 3600.0

    @property
    def merge_ratio(self) -> float:
        """Raw onsets per merged onset — how much the cross-mic merge collapses."""
        if self.merged_onset_count == 0:
            return 0.0
        return self.raw_onset_count / self.merged_onset_count

    @property
    def median_interval_s(self) -> float | None:
        """Median gap between scored onsets — compare against the refractory."""
        if len(self.merged_onsets_s) < 2:
            return None
        return float(np.median(np.diff(np.asarray(self.merged_onsets_s))))


def _detector_for(cfg: Config, sample_rate: int) -> StreamingSuperflux:
    """A detector configured exactly as the runtime builds it (infer.py)."""
    o = cfg.onset
    return StreamingSuperflux(
        sample_rate=sample_rate,
        hop_ms=o.hop_ms,
        n_mels=o.trigger_n_mels,
        median_window_s=o.median_window_s,
        mad_k=o.mad_k,
        refractory_ms=min(150, int(o.recording_cooldown_ms)),
        min_flux=o.min_flux,
    )


def scan_file(
    cfg: Config,
    path: Path,
    *,
    channels: list[int] | None = None,
    blocksize: int | None = None,
) -> OnsetScanResult:
    """Run the production onset gate over one recording.

    Everything happens at ``model.sample_rate`` — the rate the runtime detects
    at — so onset indices convert to seconds directly. The file is resampled
    once up front rather than per block: block-wise resampling accumulates the
    phase drift that `model_rate_index` exists to avoid, and here we have the
    whole file anyway.
    """
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    model_sr = int(cfg.model.sample_rate)
    duration_s = audio.shape[0] / float(file_sr)

    wanted = list(channels if channels is not None else cfg.inference.inference_channels)
    available = [c for c in wanted if 0 <= c < audio.shape[1]]
    if not available:
        # A mono capture of a multi-mic venue is still worth scanning.
        available = [0]
    if len(available) < len(wanted):
        log.warning(
            "%s has %d channel(s); scanning %s instead of %s",
            path.name, audio.shape[1], available, wanted,
        )

    block_n = int(blocksize or cfg.audio.blocksize or 1024)
    merger = OnsetMerger(int(round(cfg.onset.cross_channel_merge_ms / 1000.0 * model_sr)))

    detectors = {ch: _detector_for(cfg, model_sr) for ch in available}
    signals = {
        ch: (
            audio[:, ch] if file_sr == model_sr
            else resample_audio(audio[:, ch], orig_sr=file_sr, target_sr=model_sr)
        )
        for ch in available
    }
    n_samples = min(len(s) for s in signals.values())

    per_channel: dict[int, list[float]] = {ch: [] for ch in available}
    merged: list[float] = []
    flux_samples: list[float] = []
    gate_samples: list[float] = []

    for start in range(0, n_samples, block_n):
        end = min(start + block_n, n_samples)
        block_onsets: list[int] = []
        for ch in available:
            events = detectors[ch].process_block(signals[ch][start:end], end)
            for ev in events:
                per_channel[ch].append(ev.sample_idx / model_sr)
                block_onsets.append(int(ev.sample_idx))
            _dbfs, flux, threshold = detectors[ch].get_status()
            # The effective gate is the adaptive threshold floored by min_flux.
            flux_samples.append(float(flux))
            gate_samples.append(max(float(threshold), float(cfg.onset.min_flux)))
        if block_onsets:
            merged.extend(idx / model_sr for idx in merger.push(block_onsets))

    return OnsetScanResult(
        path=str(path),
        duration_s=duration_s,
        sample_rate=model_sr,
        channels=available,
        per_channel_onsets_s={ch: sorted(v) for ch, v in per_channel.items()},
        merged_onsets_s=sorted(merged),
        flux_samples=flux_samples,
        gate_samples=gate_samples,
        min_flux=float(cfg.onset.min_flux),
    )


def scan_files(
    cfg: Config,
    paths: list[Path],
    *,
    channels: list[int] | None = None,
    on_file: object = None,
) -> list[OnsetScanResult]:
    results: list[OnsetScanResult] = []
    for p in paths:
        try:
            result = scan_file(cfg, p, channels=channels)
        except Exception as exc:  # noqa: BLE001 — one bad file must not end the scan
            log.warning("onset scan failed for %s: %s", p, exc)
            continue
        results.append(result)
        if callable(on_file):
            on_file(result)
    return results


def aggregate(results: list[OnsetScanResult]) -> dict:
    """Pool per-file scans into the numbers that matter downstream.

    ``onsets_per_hour`` is the headline: it converts a
    false-alarms-per-1000-onsets figure (which has an observable denominator on
    a clip corpus) into FP/hour (which does not, until this is measured).
    """
    total_s = sum(r.duration_s for r in results)
    merged = sum(r.merged_onset_count for r in results)
    raw = sum(r.raw_onset_count for r in results)
    intervals = [
        d for r in results
        for d in np.diff(np.asarray(r.merged_onsets_s)).tolist()
        if len(r.merged_onsets_s) >= 2
    ]
    return {
        "files": len(results),
        "duration_s": total_s,
        "raw_onsets": raw,
        "merged_onsets": merged,
        "onsets_per_hour": (merged / total_s * 3600.0) if total_s > 0 else 0.0,
        "merge_ratio": (raw / merged) if merged else 0.0,
        "median_interval_s": float(np.median(intervals)) if intervals else None,
    }


def flux_histogram(results: list[OnsetScanResult], bins: int = 12) -> dict:
    """Distribution of spectral flux against the trigger gate, in dB.

    Shows the headroom the gate is operating on: if the flux mass sits far below
    the gate, quiet clinks are being rejected before the model ever sees them,
    and `[onset] mad_k` / `min_flux_db` are the knobs — not the classifier.
    """
    flux = np.asarray([f for r in results for f in r.flux_samples], dtype=np.float64)
    gate = np.asarray([g for r in results for g in r.gate_samples], dtype=np.float64)
    if flux.size == 0:
        return {"bins": [], "n": 0, "above_gate_fraction": None}

    def _db(x: np.ndarray) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(x, 1e-12))

    flux_db = _db(flux)
    finite = flux_db[np.isfinite(flux_db)]
    if finite.size == 0:
        return {"bins": [], "n": 0, "above_gate_fraction": None}

    lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.min()) + 1.0
    counts, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    return {
        "n": int(finite.size),
        "bins": [
            {"lo_db": float(edges[i]), "hi_db": float(edges[i + 1]), "count": int(c)}
            for i, c in enumerate(counts)
        ],
        "median_flux_db": float(np.median(finite)),
        "median_gate_db": float(np.median(_db(gate[np.isfinite(_db(gate))]))) if gate.size else None,
        "above_gate_fraction": float(np.mean(flux > gate)) if gate.size == flux.size else None,
    }
