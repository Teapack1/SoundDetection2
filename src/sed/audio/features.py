"""Log-mel spectrogram feature extraction matched to EfficientAT mn10_as.

A single source of truth for both training and inference. Implemented with
torchaudio (training) and a pure-numpy fallback (inference if torch is absent).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torchaudio
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# Mean / std over the AudioSet training set used by EfficientAT.
# Source: fschmid56/EfficientAT pretrained checkpoint normalization.
EFFICIENTAT_MEAN = -4.2677393
EFFICIENTAT_STD = 4.5689974


class MelExtractor:
    """Wraps torchaudio.transforms.MelSpectrogram with consistent normalization."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 1024,
        win_length: int = 800,
        hop_length: int = 320,
        n_mels: int = 128,
        fmin: int = 0,
        fmax: int = 16000,
        device: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        if _HAS_TORCH:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=fmin,
                f_max=fmax,
                power=2.0,
                center=True,
                norm="slaney",
                mel_scale="slaney",
            ).to(device)

    @staticmethod
    def amp_to_db(x, eps: float = 1e-10):
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            return 10.0 * torch.log10(x + eps)
        return 10.0 * np.log10(x + eps)

    def __call__(self, waveform):
        """waveform: 1-d numpy array or torch tensor (samples,) or (B, samples).

        Returns log-mel of shape (B, n_mels, T), normalized to AudioSet stats.
        """
        if not _HAS_TORCH:
            return self._numpy(waveform)
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        spec = self.mel(waveform)              # (B, n_mels, T)
        log = self.amp_to_db(spec)
        return (log - EFFICIENTAT_MEAN) / EFFICIENTAT_STD

    # --- numpy fallback for inference-only deployments -----------------

    def _numpy(self, waveform: np.ndarray) -> np.ndarray:
        # librosa import deferred; we want the inference image to be torch-free
        import librosa
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        out = []
        for w in waveform:
            mel = librosa.feature.melspectrogram(
                y=w.astype(np.float32),
                sr=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                power=2.0,
                center=True,
                htk=False,
                norm="slaney",
            )
            log = 10.0 * np.log10(mel + 1e-10)
            out.append((log - EFFICIENTAT_MEAN) / EFFICIENTAT_STD)
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
