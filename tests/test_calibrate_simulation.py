"""Tests for the calibration ambience-source machinery.

We can't run the full calibrate() without a trained OpenVINO model, but we
can unit-test the window-generation logic that addresses the empty-venue
workflow: combining a quiet venue tone with library busy-ambience.
"""

import random

import numpy as np
import pytest

from sed.runtime.calibrate import _generate_simulated_window
from sed.data.augment import _rms


def _wave(amp: float, n: int = 32000, freq_hz: float = 440.0) -> np.ndarray:
    t = np.arange(n) / 32000.0
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_simulated_window_is_louder_than_room_tone_at_high_snr():
    """At SNR=+12 dB, the busy overlay should dominate the empty-venue tone."""
    rng = random.Random(0)
    win_n = 32000
    primary = _wave(0.01, win_n * 3)                    # quiet room tone
    overlays = [_wave(0.5, win_n * 3, freq_hz=200.0)]   # loud crowd surrogate

    out = _generate_simulated_window(
        primary, overlays, win_n, rng, snr_range_db=(12.0, 12.0)
    )
    assert out.shape == (win_n,)
    assert _rms(out) > _rms(primary[:win_n]) * 5, \
        "high-SNR overlay should make the window much louder than empty room"


def test_simulated_window_resembles_room_tone_at_negative_snr():
    """At SNR=-12 dB, the empty-venue tone should dominate the overlay."""
    rng = random.Random(0)
    win_n = 32000
    primary = _wave(0.3, win_n * 3, freq_hz=100.0)       # loud-ish room tone
    overlays = [_wave(0.3, win_n * 3, freq_hz=2000.0)]   # crowd at very low SNR

    out = _generate_simulated_window(
        primary, overlays, win_n, rng, snr_range_db=(-12.0, -12.0)
    )
    # When overlay is 12 dB below room tone, the room tone's spectrum should
    # dominate. Check by FFT power at the room-tone freq vs. overlay freq.
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(win_n, d=1 / 32000)
    room_power = spec[np.argmin(np.abs(freqs - 100.0))]
    overlay_power = spec[np.argmin(np.abs(freqs - 2000.0))]
    assert room_power > overlay_power, \
        "at SNR=-12 dB, room tone band should beat overlay band"


def test_simulated_window_falls_back_to_primary_when_no_overlays():
    """No library ambience available → just yield the primary."""
    rng = random.Random(0)
    win_n = 32000
    primary = _wave(0.1, win_n * 3)
    out = _generate_simulated_window(primary, [], win_n, rng)
    assert out.shape == (win_n,)
    # Should be a slice of the primary, not zero or modified
    assert _rms(out) == pytest.approx(_rms(primary[:win_n]), rel=0.5)


def test_simulated_window_handles_short_inputs():
    """Inputs shorter than win_n should be tiled, not crash."""
    rng = random.Random(0)
    win_n = 32000
    primary = _wave(0.1, win_n // 4)        # 8000 samples — shorter than window
    overlays = [_wave(0.3, win_n // 8)]     # 4000 samples — shorter still
    out = _generate_simulated_window(primary, overlays, win_n, rng,
                                     snr_range_db=(0.0, 0.0))
    assert out.shape == (win_n,)
    assert np.all(np.isfinite(out))


def test_simulated_window_reproducible_with_same_seed():
    win_n = 32000
    primary = _wave(0.1, win_n * 3)
    overlays = [_wave(0.3, win_n * 3, freq_hz=300.0),
                _wave(0.3, win_n * 3, freq_hz=600.0)]
    a = _generate_simulated_window(primary, overlays, win_n, random.Random(42),
                                   snr_range_db=(0.0, 12.0))
    b = _generate_simulated_window(primary, overlays, win_n, random.Random(42),
                                   snr_range_db=(0.0, 12.0))
    np.testing.assert_array_equal(a, b)
