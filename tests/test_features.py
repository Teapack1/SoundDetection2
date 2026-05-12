from __future__ import annotations

import numpy as np
import pytest
import torch

from sed.config import REPO_ROOT
from sed.audio.features import MelExtractor, feature_frame_count


def test_feature_frame_count_tracks_frontend_shape():
    assert feature_frame_count(
        sample_rate=32000,
        window_seconds=1.0,
        hop_length=320,
        n_fft=1024,
        frontend="efficientat",
    ) == 100
    # The legacy frontend was removed — asking for it must fail loudly
    # instead of silently producing differently-shaped features.
    with pytest.raises(ValueError, match="legacy_torchaudio_db frontend was removed"):
        feature_frame_count(
            sample_rate=32000,
            window_seconds=1.0,
            hop_length=320,
            n_fft=1024,
            frontend="legacy_torchaudio_db",
        )
    with pytest.raises(ValueError, match="only 'efficientat'"):
        MelExtractor(frontend="legacy_torchaudio_db")


def test_efficientat_frontend_matches_vendored_preprocess_eval():
    import sys

    efficientat = REPO_ROOT / "vendor" / "EfficientAT"
    if str(efficientat) not in sys.path:
        sys.path.insert(0, str(efficientat))
    from models.preprocess import AugmentMelSTFT

    rng = np.random.default_rng(0)
    wave = (0.01 * rng.standard_normal(32000)).astype("float32")
    wave[8000:8010] += np.linspace(0.0, 0.3, 10, dtype="float32")

    ours = MelExtractor(
        sample_rate=32000,
        n_fft=1024,
        win_length=800,
        hop_length=320,
        n_mels=128,
        fmin=0,
        fmax=15000,
        frontend="efficientat",
    )(wave)

    ref = AugmentMelSTFT(
        n_mels=128,
        sr=32000,
        win_length=800,
        hopsize=320,
        n_fft=1024,
        freqm=0,
        timem=0,
        fmin=0,
        fmax=15000,
    )
    ref.eval()
    with torch.no_grad():
        expected = ref(torch.from_numpy(wave[None, :]))

    assert ours.shape == expected.shape == (1, 128, 100)
    assert torch.allclose(ours, expected, atol=1e-5, rtol=1e-5)


def test_efficientat_numpy_fallback_matches_torch_path():
    rng = np.random.default_rng(1)
    wave = (0.01 * rng.standard_normal(32000)).astype("float32")
    extractor = MelExtractor(fmax=15000, frontend="efficientat")

    torch_path = extractor(wave).detach().numpy()
    numpy_path = extractor._efficientat_numpy(wave)

    assert numpy_path.shape == torch_path.shape == (1, 128, 100)
    assert np.max(np.abs(numpy_path - torch_path)) < 1e-4
