"""Audio augmentation for noise-robust training.

The single most important augmentation is **domain mixing**: clean positives
mixed with venue ambience at varying SNR. This is what makes the model
robust to real on-site conditions when you only have a few hundred clean
positives.

Augmentation chain (waveform domain, applied during DataLoader):
  1. Time shift  ±200 ms   (positives only)
  2. Pitch shift ±2 semis  (positives only; preserves transient feel)
  3. Time stretch ±5%      (cheap, sparingly)
  4. Domain mixing         (mix with random ambience clip at random SNR)
  5. External-noise mix    (mix with a random AudioSet/library negative ~20%)
  6. Gaussian micro-noise  (mic noise floor)

Then in spec domain (post-mel):
  7. SpecAugment: 2 freq + 2 time masks
  8. Mixup at the spectrogram level (α=0.3)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf

from sed.audio.resample import resample_audio
from sed.config import REPO_ROOT


def _load_wav_resampled(path: Path, target_sr: int, channels: int = 1) -> np.ndarray:
    """Load an audio file as 1-d float32 at target_sr."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=False)
    else:
        data = data[:, 0]
    if sr != target_sr:
        data = resample_audio(data, orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32)


def _ensure_length(x: np.ndarray, target_n: int) -> np.ndarray:
    if len(x) >= target_n:
        start = random.randint(0, len(x) - target_n)
        return x[start : start + target_n]
    out = np.zeros(target_n, dtype=np.float32)
    out[: len(x)] = x
    return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix `signal` and `noise` so that signal-to-noise ratio is `snr_db`."""
    s = _rms(signal)
    n = _rms(noise)
    if n < 1e-9:
        return signal.copy()
    target_noise_rms = s / (10 ** (snr_db / 20.0))
    scaled = noise * (target_noise_rms / n)
    out = signal + scaled
    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        out /= peak
    return out.astype(np.float32)


def _peak_mel_cell(wave: np.ndarray, sr: int, n_mels: int = 64) -> tuple[int, int, np.ndarray]:
    """Returns (peak_band, peak_frame, mel_log_spectrogram) for the loudest
    time-frequency cell. Used to locate where the target's transient lives."""
    import librosa
    if wave.ndim > 1:
        wave = wave.mean(axis=-1)
    if len(wave) < sr // 4:
        wave = np.pad(wave, (0, sr // 4 - len(wave)))
    mel = librosa.feature.melspectrogram(
        y=wave.astype(np.float32), sr=sr, n_mels=n_mels,
        n_fft=1024, hop_length=256, power=2.0,
    )
    log_mel = 10.0 * np.log10(mel + 1e-10)
    band, frame = np.unravel_index(log_mel.argmax(), log_mel.shape)
    return int(band), int(frame), log_mel


def masking_score_db(target: np.ndarray, mixed: np.ndarray, sr: int) -> float:
    """How well did the target's transient survive the mix? Higher = better.

    Find the target's brightest time-frequency cell (its "transient peak").
    In the MIXED audio at that same time frame, measure how much the same
    mel band sticks up above the median band energy.

    Interpretation:
      ≥ 6 dB   → target dominates the mixed frame; clinkable
       3-6 dB  → target still visible but partially masked
       0-3 dB  → roughly equal to noise floor; iffy
      < 0 dB   → target band is BELOW noise median; buried
    """
    import librosa
    band, frame, _ = _peak_mel_cell(target, sr)

    if mixed.ndim > 1:
        mixed = mixed.mean(axis=-1)
    if len(mixed) < sr // 4:
        mixed = np.pad(mixed, (0, sr // 4 - len(mixed)))
    mixed_mel = librosa.feature.melspectrogram(
        y=mixed.astype(np.float32), sr=sr, n_mels=64,
        n_fft=1024, hop_length=256, power=2.0,
    )
    if frame >= mixed_mel.shape[1]:
        frame = mixed_mel.shape[1] - 1

    frame_powers = mixed_mel[:, frame]
    peak_band_power = float(frame_powers[band])
    median_power    = float(np.median(frame_powers))
    return 10.0 * np.log10((peak_band_power + 1e-10) / (median_power + 1e-10))


# ---------------------------------------------------------------------------
# Pool of background / external clips, lazily loaded
# ---------------------------------------------------------------------------


class AudioPool:
    """Lazy pool of audio files at a fixed sample rate.

    Accepts either a single path or a list of paths. When multiple roots are
    given, all *.wav under any of them are pooled together — letting us
    combine the shared library ambience with a project's own venue-empty
    recordings, for example.
    """

    def __init__(self, roots: Path | list[Path] | str | list[str], sample_rate: int):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.roots = [Path(r) for r in roots]
        self.sr = sample_rate
        self.paths: list[Path] = []
        for r in self.roots:
            if r.exists():
                self.paths.extend(sorted(r.rglob("*.wav")))
        # de-dupe in case the same dir was passed twice
        seen = set(); uniq = []
        for p in self.paths:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp); uniq.append(p)
        self.paths = uniq
        self._cache: dict[Path, np.ndarray] = {}

    # Back-compat: some older callers expect `.root` (singular).
    @property
    def root(self) -> Path:
        return self.roots[0] if self.roots else Path()

    def __len__(self) -> int:
        return len(self.paths)

    def random(self, n_samples: int, rng: random.Random | None = None) -> np.ndarray | None:
        if not self.paths:
            return None
        rng = rng or random
        path = rng.choice(self.paths)
        if path not in self._cache:
            try:
                self._cache[path] = _load_wav_resampled(path, self.sr)
            except Exception:  # noqa: BLE001
                # remove from pool; try again recursively (max 3 attempts)
                self.paths.remove(path)
                return self.random(n_samples, rng)
        clip = self._cache[path]
        return _ensure_length(clip, n_samples)


# ---------------------------------------------------------------------------
# Waveform-level augmenter
# ---------------------------------------------------------------------------


class WaveformAugmenter:
    """Stateless-per-call augmenter; pass it the clip + label info."""

    def __init__(
        self,
        sample_rate: int,
        ambience_pool: AudioPool,
        external_neg_pool: AudioPool,
        domain_mix_prob: float = 0.95,
        snr_db_range: tuple[float, float] = (-2.0, 25.0),
        per_class_min_snr_db: dict[str, float] | None = None,
        external_neg_mix_prob: float = 0.20,
        masking_check: bool = True,
        masking_min_db: float = 3.0,
        time_shift_ms: int = 200,
        pitch_shift_prob: float = 0.4,
        pitch_semitones: tuple[float, float] = (-2.0, 2.0),
        time_stretch_prob: float = 0.2,
        time_stretch: tuple[float, float] = (0.95, 1.05),
        gaussian_noise_amp: tuple[float, float] = (0.001, 0.01),
        seed: int | None = None,
    ):
        self.sr = sample_rate
        self.amb = ambience_pool
        self.ext = external_neg_pool
        self.domain_mix_prob = domain_mix_prob
        self.snr = snr_db_range
        self.per_class_min_snr = per_class_min_snr_db or {}
        self.ext_prob = external_neg_mix_prob
        self.masking_check = masking_check
        self.masking_min_db = masking_min_db
        self.shift_n = int(time_shift_ms / 1000 * sample_rate)
        self.pitch_prob = pitch_shift_prob
        self.pitch = pitch_semitones
        self.stretch_prob = time_stretch_prob
        self.stretch = time_stretch
        self.gnoise = gaussian_noise_amp
        self.rng = random.Random(seed)
        self.stats = {"mixes_attempted": 0, "mixes_skipped_masking": 0}

    def __call__(
        self,
        wave: np.ndarray,
        is_target: bool,
        class_name: str | None = None,
    ) -> np.ndarray:
        n = len(wave)

        # Time shift (only positives — moves transient within window)
        if is_target and self.shift_n > 0:
            shift = self.rng.randint(-self.shift_n, self.shift_n)
            wave = np.roll(wave, shift)
            if shift > 0:
                wave[:shift] = 0.0
            elif shift < 0:
                wave[shift:] = 0.0

        # Pitch & time stretch (cheap, sometimes) — disable by setting prob=0
        if is_target and self.pitch_prob > 0 and self.rng.random() < self.pitch_prob:
            try:
                import librosa
                semis = self.rng.uniform(*self.pitch)
                wave = librosa.effects.pitch_shift(wave, sr=self.sr, n_steps=semis)
            except Exception:  # noqa: BLE001
                pass

        if self.stretch_prob > 0 and self.rng.random() < self.stretch_prob:
            try:
                import librosa
                rate = self.rng.uniform(*self.stretch)
                wave = librosa.effects.time_stretch(wave, rate=rate)
                wave = _ensure_length(wave, n)
            except Exception:  # noqa: BLE001
                pass

        # Domain mixing — clean positive ⊗ ambience at random SNR.
        # Per-class min SNR clamps the lower bound so fragile classes (fine
        # quiet glasses) aren't buried.
        if self.rng.random() < self.domain_mix_prob:
            amb = self.amb.random(n, self.rng)
            if amb is not None:
                lo, hi = self.snr
                if class_name and class_name in self.per_class_min_snr:
                    lo = max(lo, self.per_class_min_snr[class_name])
                snr = self.rng.uniform(lo, hi)
                target_only = wave.copy()
                mixed = mix_at_snr(wave, amb, snr)
                self.stats["mixes_attempted"] += 1
                # Masking check: if the target's transient is no longer
                # dominant in the mixed frame (score < masking_min_db dB above
                # the median band), skip this mix and fall back to the clean
                # target (otherwise we'd train on noise-as-target).
                if is_target and self.masking_check:
                    score = masking_score_db(target_only, mixed, self.sr)
                    if score < self.masking_min_db:
                        self.stats["mixes_skipped_masking"] += 1
                        wave = target_only
                    else:
                        wave = mixed
                else:
                    wave = mixed

        # External-negative mix (small probability)
        if self.rng.random() < self.ext_prob:
            ext = self.ext.random(n, self.rng)
            if ext is not None:
                snr = self.rng.uniform(0.0, 20.0)
                wave = mix_at_snr(wave, ext, snr)

        # Gaussian noise floor — disabled when both bounds are 0
        if self.gnoise[1] > 0:
            amp = self.rng.uniform(*self.gnoise)
            wave = wave + np.random.randn(len(wave)).astype(np.float32) * amp

        # Clip / normalize
        wave = _ensure_length(wave, n)
        peak = float(np.max(np.abs(wave))) or 1.0
        if peak > 1.0:
            wave = wave / peak
        return wave.astype(np.float32)


# ---------------------------------------------------------------------------
# Spectrogram-level augmentation
# ---------------------------------------------------------------------------


def spec_augment(
    spec: np.ndarray,
    freq_masks: int = 2,
    freq_width: int = 24,
    time_masks: int = 2,
    time_width: int = 25,
    rng: random.Random | None = None,
) -> np.ndarray:
    """spec: (n_mels, T) — modified in place."""
    rng = rng or random
    out = spec.copy()
    n_mels, T = out.shape
    for _ in range(freq_masks):
        f = rng.randint(0, freq_width)
        f0 = rng.randint(0, max(1, n_mels - f))
        out[f0:f0 + f, :] = out.mean()
    for _ in range(time_masks):
        t = rng.randint(0, time_width)
        t0 = rng.randint(0, max(1, T - t))
        out[:, t0:t0 + t] = out.mean()
    return out


def mixup_pair(
    a_spec: np.ndarray,
    a_target: np.ndarray,
    b_spec: np.ndarray,
    b_target: np.ndarray,
    alpha: float = 0.3,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    spec = lam * a_spec + (1 - lam) * b_spec
    target = lam * a_target + (1 - lam) * b_target
    return spec.astype(np.float32), target.astype(np.float32)


def resolve_pool_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return REPO_ROOT / rel_or_abs
