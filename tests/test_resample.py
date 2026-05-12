import numpy as np

from sed.audio.resample import resample_audio


def test_resample_48k_to_32k_preserves_duration():
    wave = np.linspace(-1.0, 1.0, 48000, dtype=np.float32)
    out = resample_audio(wave, orig_sr=48000, target_sr=32000)
    assert out.shape == (32000,)


def test_resample_44100_to_32k_preserves_duration():
    wave = np.linspace(-1.0, 1.0, 44100, dtype=np.float32)
    out = resample_audio(wave, orig_sr=44100, target_sr=32000)
    assert out.shape == (32000,)


def test_resample_multichannel_preserves_channels_and_duration():
    wave = np.zeros((48000, 4), dtype=np.float32)
    wave[:, 0] = 1.0
    wave[:, 3] = -1.0
    out = resample_audio(wave, orig_sr=48000, target_sr=32000)
    assert out.shape == (32000, 4)
    assert out.dtype == np.float32
