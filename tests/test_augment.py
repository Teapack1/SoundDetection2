"""Tests for the augmentation pipeline — especially the masking-aware mix."""

import numpy as np
import pytest

from sed.data.augment import mix_at_snr, masking_score_db


def _impulse(n: int, position: int = 100, amp: float = 0.5) -> np.ndarray:
    """A short transient burst at `position`, decaying like a clink."""
    x = np.zeros(n, dtype=np.float32)
    decay_n = 800
    end = min(n, position + decay_n)
    decay = np.exp(-np.arange(end - position) / 200.0)
    x[position:end] = amp * decay * np.sin(
        2 * np.pi * 4000 * np.arange(end - position) / 32000
    )
    return x


def _noise(n: int, amp: float = 0.1) -> np.ndarray:
    return amp * np.random.RandomState(0).randn(n).astype(np.float32)


def test_mix_at_snr_keeps_target_audible_at_high_snr():
    sig = _impulse(32000)
    noise = _noise(32000, amp=0.1)
    mixed = mix_at_snr(sig, noise, snr_db=20.0)
    assert mixed.shape == sig.shape
    # at +20 dB SNR the mix should be very close to the signal in peak
    assert abs(mixed.max() - sig.max()) < 0.2


def test_masking_score_high_when_target_dominant():
    """When the target is much louder than the noise, its transient should
    dominate the mix → high transient-dominance dB."""
    sig = _impulse(32000, amp=0.5)
    noise = _noise(32000, amp=0.05)
    mixed = mix_at_snr(sig, noise, snr_db=20.0)
    score = masking_score_db(sig, mixed, sr=32000)
    # the 4 kHz transient should poke well above noise floor at its peak frame
    assert score > 6.0, f"expected target to dominate, got {score:.2f} dB"


def test_masking_score_decreases_with_lower_snr():
    """The transient-dominance score should drop monotonically as the SNR
    drops — a clean target dominates much more than the same target buried
    under heavy in-band noise."""
    sig = _impulse(32000, amp=0.1, position=100)
    rng = np.random.RandomState(0)
    # Co-band tonal noise (simulates speech formants overlapping the clink band)
    t = np.arange(32000) / 32000.0
    co_band = (np.sin(2*np.pi*3800*t) + np.sin(2*np.pi*4200*t)
               + 0.3 * rng.randn(32000)).astype(np.float32)

    scores = []
    for snr_db in (30.0, 10.0, -5.0, -20.0):
        mixed = mix_at_snr(sig, co_band, snr_db=snr_db)
        scores.append(masking_score_db(sig, mixed, sr=32000))

    # Each subsequent (lower) SNR should yield ≤ the previous score (within
    # small tolerance for noise variance)
    for i in range(1, len(scores)):
        assert scores[i] <= scores[i-1] + 1.0, \
            f"non-monotonic: SNR step {i} score {scores[i]:.2f} > prev {scores[i-1]:.2f}"
    # And the dynamic range should be substantial — at least a few dB
    # between best and worst case
    assert scores[0] - scores[-1] > 3.0, \
        f"metric not responsive enough: range={scores[0]-scores[-1]:.2f} dB"


def test_per_class_min_snr_clamps_low_snr():
    # The augmenter should clamp SNR to per_class_min for the matching class
    from sed.data.augment import AudioPool, WaveformAugmenter

    class _StaticPool:
        """Always returns the same (loud) noise — easier to reason about."""
        def __init__(self, n):
            self._noise = _noise(n, amp=0.5)
        def __len__(self): return 1
        def random(self, n, rng=None): return self._noise[:n].copy()

    sig = _impulse(32000, amp=0.05)
    pool = _StaticPool(32000)
    aug = WaveformAugmenter(
        sample_rate=32000,
        ambience_pool=pool,
        external_neg_pool=_StaticPool(32000),
        domain_mix_prob=1.0,
        snr_db_range=(-20.0, -19.9),               # would normally murder the signal
        per_class_min_snr_db={"clink_wine_thin": 15.0},  # but this class is protected
        external_neg_mix_prob=0.0,
        masking_check=False,                       # isolate the SNR-clamp behaviour
        time_shift_ms=0,
        pitch_semitones=(0.0, 0.0),
        time_stretch=(1.0, 1.0),
        gaussian_noise_amp=(0.0, 0.0),
        seed=1,
    )
    mixed = aug(sig.copy(), is_target=True, class_name="clink_wine_thin")
    # With +15 dB SNR clamp, the target's transient should still dominate
    # its TF cell in the mix.
    score = masking_score_db(sig, mixed, sr=32000)
    assert score > 3.0, f"expected target preserved, got dominance={score:.2f} dB"
