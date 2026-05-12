"""Streaming onset detection for auto-triggered recording.

Implements a Superflux-style spectral-flux onset detector with an adaptive
baseline (median + MAD). Designed to run online, in the audio callback thread
or just downstream of it.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class OnsetEvent:
    """Detected onset event with absolute sample index in the ring buffer."""
    sample_idx: int          # absolute sample index of the trigger frame
    strength: float
    baseline: float
    channel: int | None = None  # physical input channel when known


class StreamingSuperflux:
    """Online superflux onset detector.

    Feed it raw audio blocks via `process_block`. It emits an OnsetEvent
    whenever the spectral-flux envelope rises N MAD-deviations above the
    rolling median AND above the absolute min_flux floor.

    The detector is mono — pass it a single channel.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        hop_ms: int = 32,
        n_mels: int = 64,
        max_filter_size: int = 3,
        median_window_s: float = 5.0,
        mad_k: float = 4.0,
        refractory_ms: int = 150,
        min_flux: float = 0.005,
    ):
        self.sample_rate = sample_rate
        self.hop = int(sample_rate * hop_ms / 1000)
        self.n_mels = n_mels
        self.max_size = max_filter_size
        self.mad_k = mad_k
        self.min_flux = min_flux
        self.refractory_samples = int(sample_rate * refractory_ms / 1000)

        # n_fft must be >= hop; round up to the next power-of-two.
        min_nfft = max(1024, self.hop)
        self._n_fft = 1 << (min_nfft - 1).bit_length()

        # Mel filterbank
        import librosa
        self._mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=self._n_fft, n_mels=n_mels, fmin=0,
            fmax=sample_rate // 2,
        ).astype(np.float32)
        self._win = np.hanning(self._n_fft).astype(np.float32)

        # State
        self._tail = np.zeros(self._n_fft - self.hop, dtype=np.float32)
        self._mel_hist = collections.deque(maxlen=self.max_size + 1)
        median_n = int(median_window_s * sample_rate / self.hop)
        self._env_hist = collections.deque(maxlen=median_n)
        self._last_trigger_idx = -10_000_000

        # Live status — polled by recorder display, never logged to terminal
        self._peak_rms: float = 0.0
        self._current_threshold: float = 0.0
        self._last_flux: float = 0.0

        log.debug(
            "onset: sr=%d hop=%d n_fft=%d mad_k=%.1f min_flux=%.4f refractory=%dms",
            sample_rate, self.hop, self._n_fft, self.mad_k, self.min_flux, refractory_ms,
        )

    def reset(self) -> None:
        self._tail[:] = 0.0
        self._mel_hist.clear()
        self._env_hist.clear()
        self._last_trigger_idx = -10_000_000
        self._peak_rms = 0.0

    def get_status(self) -> tuple[float, float, float]:
        """Return (peak_dbfs, last_flux, current_threshold) for the live display."""
        dbfs = 20.0 * float(np.log10(self._peak_rms + 1e-10))
        return dbfs, self._last_flux, self._current_threshold

    def _frame(self, x: np.ndarray) -> np.ndarray:
        """Concatenate tail + new audio, frame at `hop` step, return (n_frames, n_fft)."""
        full = np.concatenate([self._tail, x])
        nframes = (len(full) - self._n_fft) // self.hop + 1
        if nframes <= 0:
            self._tail = full
            return np.zeros((0, self._n_fft), dtype=np.float32)
        frames = np.lib.stride_tricks.as_strided(
            full,
            shape=(nframes, self._n_fft),
            strides=(full.strides[0] * self.hop, full.strides[0]),
            writeable=False,
        ).copy()
        consumed = (nframes - 1) * self.hop + self._n_fft
        self._tail = full[consumed - (self._n_fft - self.hop):].copy()
        return frames

    def process_block(self, block: np.ndarray, write_idx_at_end: int) -> list[OnsetEvent]:
        """Process one audio block (1-d float32) and return any new onsets."""
        if block.ndim > 1:
            block = block.mean(axis=-1).astype(np.float32)

        rms = float(np.sqrt(np.mean(block ** 2)))
        if rms > self._peak_rms:
            self._peak_rms = rms

        frames = self._frame(block)
        if frames.shape[0] == 0:
            return []

        # STFT magnitude → mel
        windowed = frames * self._win
        mag = np.abs(np.fft.rfft(windowed, n=self._n_fft, axis=-1)).astype(np.float32)
        mel = mag @ self._mel_fb.T                  # (nframes, n_mels)
        mel_log = np.log1p(mel)

        events: list[OnsetEvent] = []
        sample_idx_of_first_frame = (
            write_idx_at_end - block.shape[0] + (self._n_fft // 2)
        )

        for f_i in range(mel_log.shape[0]):
            cur = mel_log[f_i]
            self._mel_hist.append(cur)
            if len(self._mel_hist) <= self.max_size:
                continue
            # superflux: max over recent frames, half-wave rectified diff
            stack = np.stack(list(self._mel_hist)[: self.max_size], axis=0)
            ref = stack.max(axis=0)
            flux = np.maximum(cur - ref, 0.0).mean()
            self._env_hist.append(flux)
            self._last_flux = flux

            if len(self._env_hist) < 32:           # warm-up period
                continue
            arr = np.fromiter(self._env_hist, dtype=np.float32)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) + 1e-8
            threshold = med + self.mad_k * mad
            self._current_threshold = threshold

            frame_sample_idx = sample_idx_of_first_frame + f_i * self.hop
            if flux > threshold:
                if flux < self.min_flux:
                    log.debug("onset below min_flux: %.4f < %.4f", flux, self.min_flux)
                elif frame_sample_idx - self._last_trigger_idx <= self.refractory_samples:
                    log.debug("onset in refractory period")
                else:
                    events.append(OnsetEvent(
                        sample_idx=frame_sample_idx,
                        strength=float(flux),
                        baseline=med,
                    ))
                    self._last_trigger_idx = frame_sample_idx
                    log.debug("onset: flux=%.4f threshold=%.4f", flux, threshold)

        return events
