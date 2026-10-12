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
    assert feature_frame_count(
        sample_rate=16000,
        window_seconds=0.4,
        hop_length=160,
        n_fft=512,
        frontend="ced",
    ) == 41
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


def test_window_sample_count_is_shared_and_rounds():
    """Train crop, export tracing, calibration, and runtime ring buffer all
    size the window via this helper; round() (not int()) so a float-noise
    product like 22050 * 0.9999999 doesn't drop a sample on one side only."""
    from sed.audio.features import window_sample_count

    assert window_sample_count(32000, 1.0) == 32000
    assert window_sample_count(22050, 0.9999999) == 22050  # int() would give 22049
    assert window_sample_count(16000, 1.5) == 24000


def test_mel_filter_diagnostics_detects_empty_low_filters():
    """High n_mels vs n_fft leaves the narrowest low HTK triangles between FFT
    bin centers — those rows capture zero power and render as black strips."""
    from sed.audio.features import mel_filter_diagnostics

    bad = mel_filter_diagnostics(256, 2048, 48000, 0, 23000)
    assert bad["empty_rows"] > 0
    assert bad["max_usable_n_mels"] < 256
    # Empty filters concentrate at the low end of the bank.
    assert min(bad["empty_row_indices"]) == 0

    worse = mel_filter_diagnostics(224, 1024, 48000, 0, 23000)
    assert worse["empty_rows"] > bad["empty_rows"]

    # The EfficientAT pretrained contract is clean.
    good = mel_filter_diagnostics(128, 1024, 32000, 0, 15000)
    assert good["empty_rows"] == 0
    assert good["max_usable_n_mels"] == 128

    # max_usable is self-consistent: that many mels produce no empty rows.
    check = mel_filter_diagnostics(bad["max_usable_n_mels"], 2048, 48000, 0, 23000)
    assert check["empty_rows"] == 0


def test_mel_geometry_warning_rides_model_fit_report():
    from sed.audio.features import mel_geometry_warning, model_fit_report
    from sed.config import FeaturesCfg

    degenerate = FeaturesCfg(n_mels=256, n_fft=1024, fmin=0, fmax=23000)
    verdict, diag = mel_geometry_warning(degenerate, 48000)
    assert verdict is not None and verdict["kind"] == "mel_empty_filters"
    assert str(diag["max_usable_n_mels"]) in verdict["text"]

    report = model_fit_report(degenerate, "mn10_as", False, 48000)
    assert any(v.get("kind") == "mel_empty_filters" for v in report["verdicts"])
    assert report["mel_diagnostics"]["empty_rows"] > 0

    clean = model_fit_report(FeaturesCfg(), "mn10_as", True, 32000)
    assert not any(v.get("kind") == "mel_empty_filters" for v in clean["verdicts"])

    # Linear features have no mel bank to go empty.
    linear = FeaturesCfg(feature_type="linear")
    assert mel_geometry_warning(linear, 32000) == (None, None)


def test_shipped_geometry_presets_have_no_empty_filters():
    """The built-in geometry presets must never demo the black-strip failure."""
    from sed.audio.features import mel_filter_diagnostics
    from sed.data.presets import _BUILTIN_PRESETS

    for preset in _BUILTIN_PRESETS:
        if preset.get("feature_type") != "mel" or preset.get("frontend") != "efficientat":
            continue
        diag = mel_filter_diagnostics(
            preset["n_mels"], preset["n_fft"], preset["sample_rate"],
            preset["fmin"], preset["fmax"],
        )
        assert diag["empty_rows"] == 0, (preset["name"], diag)


def test_standardized_logmel_matches_fastnorm_up_to_affine():
    """standardize replaces only the final affine: (fast*5 - 4.5 - mean)/std."""
    import numpy as np

    from sed.audio.features import MelExtractor

    rng = np.random.default_rng(7)
    wave = (0.1 * rng.standard_normal(32000)).astype("float32")
    fast = MelExtractor(sample_rate=32000)(wave)
    std = MelExtractor(
        sample_rate=32000, standardize=True, standardize_mean=-4.5, standardize_std=5.0
    )(wave)
    fast = np.asarray(fast.numpy() if hasattr(fast, "numpy") else fast)
    std = np.asarray(std.numpy() if hasattr(std, "numpy") else std)
    # With mean=-4.5, std=5 the two normalizations are identical by construction.
    assert np.allclose(fast, std, atol=1e-5)

    zs = MelExtractor(
        sample_rate=32000, standardize=True, standardize_mean=-4.2677393,
        standardize_std=9.1379948,
    )(wave)
    zs = np.asarray(zs.numpy() if hasattr(zs, "numpy") else zs)
    log_mel = fast * 5.0 - 4.5
    assert np.allclose(zs, (log_mel + 4.2677393) / 9.1379948, atol=1e-4)


def test_standardize_config_validation_and_routing():
    import pytest as _pytest

    from sed.audio.features import LinearExtractor, MelExtractor, build_feature_extractor
    from sed.config import Config

    cfg = Config()
    cfg.features.standardize = True
    cfg.features.standardize_mean = -4.27
    cfg.features.standardize_std = 9.14
    cfg = Config.model_validate(cfg.model_dump(mode="json"))
    assert cfg.features.matches_pretrained_normalization() is False
    ext = build_feature_extractor(cfg)
    # The fitted z-score reaches the extractor as its one normalization stage.
    assert isinstance(ext, MelExtractor)
    assert ext.compression == "log"
    assert (ext.scaling.method, ext.scaling.scope, ext.scaling.source) == (
        "zscore", "global", "dataset",
    )
    assert ext.scaling.a.tolist() == [pytest.approx(-4.27)]
    assert ext.scaling.b.tolist() == [pytest.approx(9.14)]

    with _pytest.raises(ValueError, match="standardize_std"):
        Config.model_validate({"features": {"standardize": True, "standardize_std": 0.0}})
    # db_norm IS a normalization (a fixed min-max over [db_min, db_max]), so
    # asking for a fitted z-score at the same time is two normalizations.
    with _pytest.raises(ValueError, match="two different"):
        Config.model_validate({"features": {
            "feature_type": "linear", "standardize": True, "db_norm": True,
        }})

    # Compression and normalization are independent axes, so combinations that
    # the old per-feature-type flags forbade are now ordinary configurations.
    with_pcen = Config.model_validate({"features": {"standardize": True, "pcen": True}})
    assert with_pcen.features.compression == "pcen"
    assert (with_pcen.features.scaling_method, with_pcen.features.scaling_source) == (
        "zscore", "dataset",
    )
    raw_power = Config.model_validate({"features": {
        "feature_type": "linear", "standardize": True, "log_power": False,
    }})
    assert raw_power.features.compression == "none"
    assert raw_power.features.scaling_method == "zscore"

    linear = Config.model_validate({
        "features": {
            "feature_type": "linear", "standardize": True,
            "standardize_mean": -7.0, "standardize_std": 3.0,
        },
    })
    linear_ext = build_feature_extractor(linear)
    assert isinstance(linear_ext, LinearExtractor)
    assert linear_ext.compression == "log"
    assert linear_ext.scaling.method == "zscore"
    assert linear_ext.scaling.a.tolist() == [pytest.approx(-7.0)]
    assert linear_ext.scaling.b.tolist() == [pytest.approx(3.0)]


@pytest.mark.parametrize(
    ("n_fft", "win_length", "expected_bands"),
    [(1024, 882, 463), (2048, 1764, 925)],
)
def test_batched_torch_linear_extractor_matches_numpy_fallback(
    n_fft, win_length, expected_bands,
):
    from sed.audio.features import LinearExtractor

    rng = np.random.default_rng(n_fft)
    waves = (0.03 * rng.standard_normal((2, 17_640))).astype(np.float32)
    waves[:, 4_000:4_008] += np.linspace(0.0, 0.5, 8, dtype=np.float32)
    extractor = LinearExtractor(
        sample_rate=44_100,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=220,
        fmin=80,
        fmax=20_000,
    )

    torch_output = extractor(torch.from_numpy(waves)).detach().cpu().numpy()
    numpy_output = extractor._numpy(waves)

    assert torch_output.shape == numpy_output.shape == (2, expected_bands, 81)
    np.testing.assert_allclose(torch_output, numpy_output, rtol=2e-5, atol=4e-5)


def test_standardized_linear_extractor_applies_one_global_affine():
    from sed.audio.features import LinearExtractor, waveform_to_linear

    rng = np.random.default_rng(41)
    waves = (0.02 * rng.standard_normal((2, 17_640))).astype(np.float32)
    raw_extractor = LinearExtractor(
        sample_rate=44_100, n_fft=1024, win_length=882, hop_length=220,
        fmin=80, fmax=20_000,
    )
    raw = raw_extractor(torch.from_numpy(waves)).detach().cpu().numpy()
    mean = float(raw.mean(dtype=np.float64))
    std = float(raw.std(dtype=np.float64))
    standardized_extractor = LinearExtractor(
        sample_rate=44_100, n_fft=1024, win_length=882, hop_length=220,
        fmin=80, fmax=20_000, standardize=True,
        standardize_mean=mean, standardize_std=std,
    )

    standardized = standardized_extractor(torch.from_numpy(waves)).detach().cpu().numpy()
    np.testing.assert_allclose(standardized, (raw - mean) / std, rtol=1e-6, atol=2e-6)
    assert float(standardized.mean()) == pytest.approx(0.0, abs=2e-6)
    assert float(standardized.std()) == pytest.approx(1.0, abs=2e-6)

    numpy_single = waveform_to_linear(
        waves[0], 44_100, n_fft=1024, win_length=882, hop_length=220,
        fmin=80, fmax=20_000, standardize=True,
        standardize_mean=mean, standardize_std=std,
    )
    np.testing.assert_allclose(standardized[0], numpy_single, rtol=2e-5, atol=4e-5)


def test_db_norm_floor_keeps_legacy_output_and_finite_values():
    import numpy as np

    from sed.audio.features import waveform_to_linear

    silent = np.zeros(32000, dtype="float32")
    spec = waveform_to_linear(silent, 32000, db_norm=True, log_power=False)
    assert np.isfinite(spec).all()
    assert float(spec.max()) == 0.0  # true silence still maps to the floor


def test_ced_numpy_fallback_matches_the_torch_frontend():
    """The torch-free edge path must produce the same dB-mel as training.

    torchaudio pads its STFT with "reflect" and librosa defaults to "constant";
    unmatched, the fallback drifted ~20 dB on the frames touching the clip
    edges — exactly where an onset-centred transient sits.
    """
    from sed.audio.features import CedFeatureExtractor

    rng = np.random.default_rng(0)
    extractor = CedFeatureExtractor(16000, 512, 512, 160, 64, 0, 8000)
    for _ in range(3):
        wave = (rng.standard_normal(16000) * 0.05).astype(np.float32)
        torch_db = extractor(wave)
        torch_db = torch_db.numpy() if hasattr(torch_db, "numpy") else torch_db
        numpy_db = extractor._numpy(wave)
        assert torch_db.shape == numpy_db.shape
        np.testing.assert_allclose(torch_db, numpy_db, rtol=1e-4, atol=2e-3)


def test_scipy_engine_frame_count_matches_the_padded_extractor():
    """A window shorter than one scipy segment is zero-padded to exactly one
    frame, so the predicted model-input width must not be 0."""
    from sed.audio.features import waveform_to_feature, window_sample_count

    kwargs = dict(
        sample_rate=32000, hop_length=320, n_fft=1024, win_length=800,
        frontend="efficientat", stft_engine="scipy",
    )
    for window_seconds in (0.02, 0.4, 1.0):
        n = window_sample_count(32000, window_seconds)
        wave = np.zeros(n, dtype=np.float32)
        predicted = feature_frame_count(window_seconds=window_seconds, **kwargs)
        actual = waveform_to_feature(
            wave, 32000, feature_type="linear", n_fft=1024, win_length=800,
            hop_length=320, fmin=0, fmax=15000, db_norm=True, stft_engine="scipy",
        ).shape[-1]
        assert predicted == actual, f"{window_seconds}s: {predicted} != {actual}"
