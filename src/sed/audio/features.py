"""Log-mel spectrogram feature extraction.

This module is the single source of truth for training, export calibration,
calibration, inspection, and live inference. The only supported frontend is
``efficientat``, which mirrors the preprocessing the vendored EfficientAT
AudioSet-pretrained MobileNet checkpoints were trained with: pre-emphasis,
STFT, Kaldi mel banks, natural log, and the (x + 4.5) / 5 fast normalization.
The old ``legacy_torchaudio_db`` frontend was removed — bundles trained with
it need an older repo revision or a retrain.
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import hann

try:
    import torch
    import torchaudio
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# Shared display range for visualizing efficientat fast-normalized features
# ((ln(mel) + 4.5) / 5): -1.40 is digital silence (the true floor) and +1.6
# covers p99.9 of the loudest library transients (~65 dB span). Used by the
# inspect previews and the augmentation mix maps so both render comparably.
EFFICIENTAT_DISPLAY_RANGE: tuple[float, float] = (-1.4, 1.6)


def _require_efficientat(frontend: str) -> str:
    value = str(frontend or "efficientat").lower()
    if value != "efficientat":
        raise ValueError(
            f"unsupported features.frontend {frontend!r} — only 'efficientat' is "
            "supported (the legacy_torchaudio_db frontend was removed; retrain "
            "the model or pin an older repo revision)"
        )
    return value


def feature_frame_count(
    *,
    sample_rate: int = 32000,
    window_seconds: float = 1.0,
    hop_length: int = 320,
    n_fft: int = 1024,
    frontend: str = "efficientat",
) -> int:
    """Return the number of time frames produced for one model window."""
    _require_efficientat(frontend)
    _ = n_fft
    n = int(round(window_seconds * sample_rate))
    # EfficientAT first applies a two-tap pre-emphasis conv without padding,
    # shortening the waveform by one sample. torch.stft(center=True) then
    # yields 1 + floor(len / hop) frames.
    n = max(0, n - 1)
    return 1 + max(0, n // hop_length)


def _kaldi_mel_banks_np(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Numpy equivalent of torchaudio.compliance.kaldi.get_mel_banks.

    EfficientAT pads Kaldi's ``n_fft/2`` filters with one zero Nyquist column
    before multiplying by the power spectrum. VTLN is unused by EfficientAT
    inference, so this implements the straight triangular mel-bank path.
    """
    num_fft_bins = n_fft // 2
    nyquist = 0.5 * sample_rate
    if fmax <= 0.0:
        fmax += nyquist
    if not (0.0 <= fmin < nyquist and 0.0 < fmax <= nyquist and fmin < fmax):
        raise ValueError(f"bad mel limits: fmin={fmin}, fmax={fmax}, nyquist={nyquist}")

    def mel_scale(freq):
        return 1127.0 * np.log1p(freq / 700.0)

    mel_low = mel_scale(fmin)
    mel_high = mel_scale(fmax)
    mel_delta = (mel_high - mel_low) / (n_mels + 1)

    bins = np.arange(n_mels, dtype=np.float32)[:, None]
    left = mel_low + bins * mel_delta
    center = mel_low + (bins + 1.0) * mel_delta
    right = mel_low + (bins + 2.0) * mel_delta

    fft_bin_width = sample_rate / n_fft
    mel = mel_scale(fft_bin_width * np.arange(num_fft_bins, dtype=np.float32))[None, :]
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    bank = np.maximum(0.0, np.minimum(up, down))
    return np.pad(bank, ((0, 0), (0, 1)), mode="constant").astype(np.float32)


class MelExtractor:
    """Waveform → normalized mel features.

    Follows the vendored EfficientAT preprocessing: pre-emphasis, STFT, Kaldi
    mel banks, natural log, and the (x + 4.5) / 5 fast normalization.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 1024,
        win_length: int = 800,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: int = 0,
        fmax: int = 15000,
        frontend: str = "efficientat",
        device: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.frontend = _require_efficientat(frontend)

        if _HAS_TORCH:
            self.window = torch.hann_window(win_length, periodic=False).to(device)
            mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
                n_mels,
                n_fft,
                sample_rate,
                float(fmin),
                float(fmax),
                vtln_low=100.0,
                vtln_high=-500.0,
                vtln_warp_factor=1.0,
            )
            mel_basis = torch.nn.functional.pad(
                mel_basis, (0, 1), mode="constant", value=0
            )
            self.mel_basis = mel_basis.to(device)

    def __call__(self, waveform):
        """waveform: 1-d numpy array or torch tensor (samples,) or (B, samples).

        Returns fast-normalized log-mel of shape (B, n_mels, T).
        """
        if not _HAS_TORCH:
            return self._efficientat_numpy(np.asarray(waveform))
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return self._efficientat_torch(waveform)

    def _efficientat_torch(self, waveform):
        waveform = waveform[:, 1:] - 0.97 * waveform[:, :-1]
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        spec = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=window,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel_basis = self.mel_basis.to(device=waveform.device, dtype=power.dtype)
        melspec = torch.matmul(mel_basis, power)
        melspec = (melspec + 0.00001).log()
        return (melspec + 4.5) / 5.0

    # --- numpy fallback for inference-only (torch-free) deployments ------

    def _efficientat_numpy(self, waveform: np.ndarray) -> np.ndarray:
        import librosa
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        window = hann(self.win_length, sym=True).astype(np.float32)
        mel_basis = _kaldi_mel_banks_np(
            self.n_mels,
            self.n_fft,
            self.sample_rate,
            float(self.fmin),
            float(self.fmax),
        )
        out = []
        for w in waveform:
            w = w.astype(np.float32, copy=False)
            if len(w) >= 2:
                w = w[1:] - 0.97 * w[:-1]
            else:
                w = np.zeros((0,), dtype=np.float32)
            stft = librosa.stft(
                w,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                pad_mode="reflect",
            )
            power = np.abs(stft).astype(np.float32) ** 2
            mel = mel_basis @ power
            out.append((np.log(mel + 0.00001) + 4.5) / 5.0)
        return np.stack(out, axis=0).astype(np.float32)


def waveform_to_logmel(
    wave: np.ndarray,
    sample_rate: int = 32000,
    **kwargs,
) -> np.ndarray:
    """Convenience for one-shot feature extraction in pure numpy land."""
    extractor = MelExtractor(sample_rate=sample_rate, **kwargs)
    feats = extractor(wave)
    if _HAS_TORCH and isinstance(feats, torch.Tensor):
        feats = feats.numpy()
    return feats
