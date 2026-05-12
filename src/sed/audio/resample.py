"""Shared audio resampling helpers.

Hardware interfaces often run at 48 kHz while EfficientAT-style models use
32 kHz features. Resample complete waveforms at the boundary; callers that
need fixed-size model windows can pad/trim after this step.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


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
