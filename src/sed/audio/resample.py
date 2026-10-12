"""Shared audio resampling helpers.

Hardware interfaces often run at 48 kHz while EfficientAT-style models use
32 kHz features. Resample complete waveforms at the boundary; callers that
need fixed-size model windows can pad/trim after this step.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import firwin, resample_poly, upfirdn


def resample_audio(wave: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a full 1D or 2D waveform, preserving duration.

    The returned length is round(len(wave) * target_sr / orig_sr). Multi-channel
    arrays are expected as (samples, channels) and keep the same channel count.
    """
    if orig_sr == target_sr:
        return wave.astype(np.float32, copy=False)
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"sample rates must be positive, got {orig_sr} -> {target_sr}")
    if wave.ndim not in (1, 2):
        raise ValueError(f"expected 1D or 2D audio, got shape {wave.shape}")

    expected = int(round(wave.shape[0] * target_sr / orig_sr))
    if expected <= 0:
        return np.zeros((0,) + wave.shape[1:], dtype=np.float32)

    divisor = gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    out = resample_poly(wave.astype(np.float32, copy=False), up, down, axis=0)

    # resample_poly can differ by one sample for non-integer durations. Normalize
    # to the mathematically expected duration; this corrects converter boundary
    # rounding after the full input has been processed.
    if out.shape[0] > expected:
        out = out[:expected]
    elif out.shape[0] < expected:
        pad_shape = ((0, expected - out.shape[0]),)
        if out.ndim == 2:
            pad_shape += ((0, 0),)
        out = np.pad(out, pad_shape)
    return out.astype(np.float32, copy=False)


def model_rate_index(input_idx: int, input_sr: int, model_sr: int) -> int:
    """Map a ring index measured at the input sample rate to the model rate.

    Derived fresh from the absolute ring index each block — never accumulated
    from per-block resampled lengths, whose rounding (e.g. 1024 input samples ->
    683 vs the true 682.67 model samples) would drift the mapping by ~29 ms per
    minute of session. Used by both the recorder (onset->clip) and the runtime
    (onset->scoring window) so the two stay in lockstep.
    """
    return int(round(input_idx * model_sr / input_sr))


class StreamingResampler:
    """Resample a continuous stream block by block, without seams.

    :func:`resample_audio` is stateless: it zero-pads whatever array it is
    handed, so calling it once per audio block filters each block as an
    isolated signal and the concatenated result carries a filter ramp at every
    block boundary. Feeding that to an onset detector is close to the worst
    case — measured on 30 s of a steady 440 Hz tone with no transients in it at
    all, the default gate fired 124 times at 48 kHz and 19 times at 44.1 kHz,
    while the same audio resampled in one piece fired none. A card configured
    at the model's own rate skips resampling entirely and behaves perfectly, so
    the artifact also made the detector's sensitivity depend on the interface.

    This class keeps the filter's input history across calls, so the output is
    identical to resampling the whole stream at once and does not depend on how
    the audio happens to be chunked. The filter, its padding and the leading
    trim are the ones :func:`scipy.signal.resample_poly` uses, so streaming and
    whole-file paths agree sample for sample.

    Output lags input by the filter's group delay — under a millisecond at the
    rates used here — because a sample is only emitted once every tap that
    contributes to it has arrived.

    Not thread-safe: give each channel its own instance, and call
    :meth:`reset` after any discontinuity in the stream.
    """

    def __init__(self, orig_sr: int, target_sr: int) -> None:
        self.orig_sr = int(orig_sr)
        self.target_sr = int(target_sr)
        if self.orig_sr <= 0 or self.target_sr <= 0:
            raise ValueError(
                f"sample rates must be positive, got {orig_sr} -> {target_sr}"
            )
        self.passthrough = self.orig_sr == self.target_sr
        if self.passthrough:
            self.up = self.down = 1
            self._hist_n = 0
            self.reset()
            return

        divisor = gcd(self.orig_sr, self.target_sr)
        self.up = self.target_sr // divisor
        self.down = self.orig_sr // divisor
        max_rate = max(self.up, self.down)
        half_len = 10 * max_rate
        taps = firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * self.up
        # resample_poly pads the FILTER (not the signal) so its group delay
        # falls on a whole output sample; mirroring that is what makes the
        # streaming output land on the same grid.
        n_pre_pad = self.down - half_len % self.down
        self._h = np.concatenate([np.zeros(n_pre_pad), taps]).astype(np.float64)
        self._n_pre_remove = (half_len + n_pre_pad) // self.down
        # Input samples the filter reaches back over, rounded up to a whole
        # number of `down` so the decimation phase can never shift between
        # calls — that shift is what would reintroduce a seam.
        reach = int(np.ceil(self._h.shape[0] / self.up))
        self._hist_n = int(np.ceil(reach / self.down)) * self.down
        self.reset()

    @property
    def latency(self) -> int:
        """Output samples held back, waiting for taps that have not arrived.

        The steady-state gap between this resampler and
        :func:`resample_audio` over the same input — a fixed group delay, under
        a millisecond at the rates used here, not a growing backlog.
        """
        if self.passthrough:
            return 0
        return int(self._n_pre_remove)

    @property
    def output_index(self) -> int:
        """Absolute index, at the target rate, one past the last emitted sample.

        The detector timebase. Counted from what the resampler has actually
        emitted rather than converted from the input index, so the two clocks
        cannot drift apart over a long session.
        """
        return self._emitted

    def reset(self) -> None:
        """Forget the stream. Call after a gap, a device change or a restart."""
        self._hist = np.zeros(self._hist_n, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._consumed = 0
        self._emitted = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        """Resample one block, returning every output sample now final.

        The returned length varies from call to call — a block may produce
        nothing while the resampler waits for a whole decimation step — and
        that is normal. Concatenating the returns reproduces
        ``resample_audio`` over the concatenated input.
        """
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if self.passthrough:
            # Still count: `output_index` is the caller's timebase, and a card
            # already running at the target rate must report the same clock as
            # one that needs converting.
            self._consumed += block.shape[0]
            self._emitted += block.shape[0]
            return block
        self._pending = (
            block if self._pending.size == 0
            else np.concatenate([self._pending, block])
        )
        # Consume whole decimation steps only; a partial step would move the
        # phase and put the seam back.
        n = (self._pending.shape[0] // self.down) * self.down
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        take, self._pending = self._pending[:n], self._pending[n:]

        buf = np.concatenate([self._hist, take])
        base = self._consumed - self._hist_n          # absolute index of buf[0]
        y = upfirdn(self._h, buf, self.up, self.down)

        # Global index of y[0], on the same grid resample_audio produces.
        k0 = base * self.up // self.down - self._n_pre_remove
        # An output is final once every tap feeding it lies inside buf.
        first_final = int(np.ceil((self._h.shape[0] - 1) / self.down))
        last_final = ((buf.shape[0] - 1) * self.up) // self.down
        start = max(first_final, self._emitted - k0)
        end = min(last_final + 1, y.shape[0])

        self._consumed += n
        self._hist = buf[-self._hist_n:].copy() if self._hist_n else buf[:0]
        if end <= start:
            return np.zeros(0, dtype=np.float32)
        self._emitted = k0 + end
        return y[start:end].astype(np.float32, copy=False)
