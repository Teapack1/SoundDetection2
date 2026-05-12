"""Parity tests: the legacy soundDetection feature + model reproduced exactly.

The legacy system (see /home/om/work/projects/soundDetection) computed a scipy
PSD spectrogram, mapped it to [0, 1] over a fixed dB range, rendered it through
OpenCV's JET colormap, transposed + resized to 96×96, and classified it with a
Keras SmallerVGGNet. In SoundDetection2 the scipy PSD + dB map + JET colormap +
transpose + resize all live in the *audio feature* (a colormap image feature),
so the feature IS the 3-channel image the model trains/infers on. These tests
reproduce the reference pipeline inline (scipy only, no cv2/keras dependency) and
assert the ported feature + encoder match it.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal as sps

from sed.audio.features import (
    LEGACY_INT16_SCALE,
    _scipy_psd_power_np,
    build_feature_extractor,
    feature_band_count,
    feature_channels,
    feature_frame_count,
    feature_frame_count_for_cfg,
    jet_lut,
    render_spectrogram_image,
    waveform_to_feature,
)
from sed.config import Config, FeaturesCfg, ModelCfg
from sed.train.legacy_vgg import SmallerVGGNet


def _legacy_wave(seconds: float = 0.4, sr: int = 44100) -> np.ndarray:
    """A deterministic int16-scale test clip as float32 (int16/32768), like sf.read."""
    rng = np.random.default_rng(1234)
    t = np.arange(int(round(seconds * sr)))
    tone = 0.3 * np.sin(2 * np.pi * 3000 * t / sr)
    noise = 0.02 * rng.standard_normal(t.shape)
    x = (tone + noise) * 32768.0
    return np.clip(x, -32768, 32767).astype(np.int16).astype(np.float32) / 32768.0


def _legacy_cfg() -> Config:
    cfg = Config()
    cfg.features = FeaturesCfg(
        feature_type="linear", stft_engine="scipy", db_norm=True,
        n_fft=190, win_length=190, hop_length=167, fmin=0, fmax=22050,
        window_seconds=0.4, colormap="jet", image_size=96,
    )
    cfg.model = ModelCfg(
        backbone="legacy-smallervggnet", pretrained=False, sample_rate=44100,
        task="clip", initialization="random", classes=["_negative", "cheers"],
        channels=3,
    )
    return cfg


def _legacy_reference_spectrogram(int16_wave, sr, spec=96, db_min=-20.0, db_max=60.0):
    """The exact legacy convertToSpec.py math (scipy defaults, dB norm)."""
    _, _, sxx = sps.spectrogram(int16_wave, sr, nperseg=(spec - 1) * 2)
    with np.errstate(divide="ignore"):
        dbs = 10 * np.log10(sxx)
    dbs = (dbs - db_min) / (db_max - db_min)
    return np.clip(dbs, 0, 1)


def test_scipy_engine_matches_legacy_psd():
    sr = 44100
    wave = _legacy_wave(sr=sr)
    int16_wave = np.rint(wave * LEGACY_INT16_SCALE)  # what the legacy code read
    _, _, legacy = sps.spectrogram(int16_wave, sr, nperseg=190)
    ours = _scipy_psd_power_np(wave, sr, n_fft=190, win_length=190, hop_length=167)
    assert ours.shape == legacy.shape == (96, legacy.shape[1])
    assert np.allclose(ours, legacy, rtol=1e-3, atol=1e-2)


def test_db_norm_feature_matches_legacy_image_intensities():
    sr = 44100
    wave = _legacy_wave(sr=sr)
    int16_wave = np.rint(wave * LEGACY_INT16_SCALE)
    legacy = _legacy_reference_spectrogram(int16_wave, sr)
    ours = waveform_to_feature(
        wave, sr, feature_type="linear", stft_engine="scipy",
        db_norm=True, db_min=-20.0, db_max=60.0,
        n_fft=190, win_length=190, hop_length=167, fmin=0, fmax=sr / 2,
    )
    assert ours.shape == legacy.shape
    assert np.allclose(ours, legacy, atol=1e-4)
    assert ours.min() >= 0.0 and ours.max() <= 1.0


def test_scipy_engine_frame_count_is_uncentered():
    n = int(round(0.4 * 44100))
    expected = 1 + (n - 190) // 167
    assert feature_frame_count(
        sample_rate=44100, window_seconds=0.4, hop_length=167,
        n_fft=190, win_length=190, stft_engine="scipy",
    ) == expected


def test_jet_lut_endpoints():
    """The JET table (RGB) runs dark blue (low) → dark red (high)."""
    lut = jet_lut()  # (256, 3) float in [0, 1], RGB order
    assert lut.shape == (256, 3)
    r0, _, b0 = lut[0]
    r1, _, b1 = lut[255]
    assert b0 > 0.4 and r0 == 0.0     # low value → blue
    assert r1 > 0.4 and b1 == 0.0     # high value → red


def test_colormap_feature_is_three_channel_image():
    """A colormap image feature reports 3 channels and image_size geometry."""
    cfg = _legacy_cfg()
    assert feature_channels(cfg.features) == 3
    assert feature_band_count(cfg.features, cfg.model.sample_rate) == 96
    assert feature_frame_count_for_cfg(cfg) == 96
    ext = build_feature_extractor(cfg)
    out = ext(_legacy_wave())
    assert out.shape == (1, 3, 96, 96)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_colormap_render_matches_jet_flip_resize():
    """render_spectrogram_image = JET(RGB) → flipud (freq up) → half-pixel resize,
    matching an independent torch bilinear reference to sub-1/255."""
    sr = 44100
    wave = _legacy_wave(sr=sr)
    feat01 = waveform_to_feature(
        wave, sr, feature_type="linear", stft_engine="scipy",
        db_norm=True, n_fft=190, win_length=190, hop_length=167, fmin=0, fmax=sr / 2,
    )
    lut = jet_lut()
    idx = np.floor(feat01 * 255.0).astype(np.int64).clip(0, 255)
    img = lut[idx]                              # (F, T, 3), RGB
    img = np.flipud(img)                        # highest frequency on the top row
    ref = torch.nn.functional.interpolate(
        torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float(),
        size=(96, 96), mode="bilinear", align_corners=False,
    )[0].numpy()                                # (3, 96, 96)

    ours = render_spectrogram_image(feat01, "jet", 96)
    assert ours.shape == (3, 96, 96)
    assert np.allclose(ours, ref, atol=1e-5)


def test_colormap_render_is_blue_for_quiet_high_for_red():
    """Low energy → blue-dominant, high energy → red-dominant (RGB JET)."""
    quiet = np.full((96, 100), 0.05, dtype=np.float32)
    loud = np.full((96, 100), 0.95, dtype=np.float32)
    q = render_spectrogram_image(quiet, "jet", 96)   # (3, H, W) = (R, G, B)
    r = render_spectrogram_image(loud, "jet", 96)
    assert q[2].mean() > q[0].mean() + 0.3           # blue >> red when quiet
    assert r[0].mean() > r[2].mean() + 0.3           # red >> blue when loud


def test_smallervggnet_consumes_three_channel_image():
    """SmallerVGGNet is a plain 3-channel VGG: (B,3,96,96) → (B,1024), (B,128,8,8)."""
    net = SmallerVGGNet(in_chans=3, n_bands=96, n_frames=96).eval()
    x = torch.rand(3, 3, 96, 96)
    with torch.no_grad():
        emb = net(x)
        fmap = net.forward_features(x)
    assert emb.shape == (3, 1024)
    assert fmap.shape == (3, 128, 8, 8)
    assert torch.isfinite(emb).all()


def test_legacy_preset_and_backbone_registered():
    from sed.data import presets
    from sed.train.backbones import get_spec

    presets._seed_builtin_presets()
    p = presets.load_feature_preset("legacy-sounddetection")
    assert p["feature_type"] == "linear"
    assert p["stft_engine"] == "scipy"
    assert p["db_norm"] is True
    assert p["colormap"] == "jet"
    assert int(p["image_size"]) == 96
    assert p["backbone"] == "legacy-smallervggnet"
    assert p["pretrained"] is False

    spec = get_spec("legacy-smallervggnet")
    assert spec.family == "custom"
    assert spec.source == "sed.train.legacy_vgg:SmallerVGGNet"
    assert spec.input_channels == 3
