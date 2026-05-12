"""Dataset-fitted feature scaling: config schema, extractors, fitting, loader parity."""

from __future__ import annotations

import csv
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sed.config import Config, FeaturesCfg


def _fit_env(tmp_path, *, seconds=1.0, sr=32000, clips_per_group=3):
    rng = np.random.default_rng(5)
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in ("glass", "_negative"):
        for g in range(3):
            for i in range(clips_per_group):
                p = dataset / f"{cls}_{g}_{i}.wav"
                wave = (rng.standard_normal(int(seconds * sr)) * (0.02 + 0.04 * g))
                sf.write(str(p), wave.astype(np.float32), sr)
                rows.append({"path": str(p), "class_name": cls, "split_group": f"{cls}-g{g}"})
    with open(dataset / "train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class_name", "split_group"])
        w.writeheader()
        w.writerows(rows)
    layout = types.SimpleNamespace(dataset_dir=dataset)
    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(seed=42),
        dataset=types.SimpleNamespace(val_fraction=0.15),
    )
    return cfg, layout


_BASE_FEATURES = dict(
    feature_type="mel", frontend="efficientat", n_fft=1024, win_length=800,
    hop_length=320, n_mels=64, fmin=0, fmax=15000, window_seconds=1.0,
    log_power=True, stft_engine="librosa", db_norm=False, pcen=False,
    pcen_time_constant=0.4, pcen_gain=0.98, pcen_bias=2.0,
    pcen_power=0.5, pcen_eps=1e-6,
)


# --- config schema -----------------------------------------------------------


def test_legacy_standardize_migrates_to_global_zscore():
    f = FeaturesCfg(standardize=True, standardize_mean=-4.2, standardize_std=2.1)
    assert f.scaling_method == "zscore"
    assert f.scaling_scope == "global"
    assert f.scaling_stats == {"mean": [-4.2], "std": [2.1]}
    # round-trips unchanged
    again = FeaturesCfg.model_validate(f.model_dump())
    assert again.scaling_stats == f.scaling_stats and again.standardize


def test_scaling_fields_mirror_back_into_legacy_scalars():
    f = FeaturesCfg(scaling_method="zscore", scaling_stats={"mean": [-3.0], "std": [1.5]})
    assert f.standardize and f.standardize_mean == -3.0 and f.standardize_std == 1.5


def test_per_band_and_minmax_do_not_claim_the_legacy_bool():
    per_band = FeaturesCfg(
        scaling_method="zscore", scaling_scope="per_band",
        scaling_stats={"mean": [0.0, 1.0], "std": [1.0, 2.0]},
    )
    assert not per_band.standardize
    minmax = FeaturesCfg(scaling_method="minmax", scaling_stats={"lo": [-11.0], "hi": [2.0]})
    assert not minmax.standardize
    # both survive a snapshot round-trip
    for f in (per_band, minmax):
        again = FeaturesCfg.model_validate(f.model_dump())
        assert again.scaling_stats == f.scaling_stats


def test_legacy_jet_mode_requires_exact_scipy_db_contract():
    valid = FeaturesCfg(
        feature_type="linear", stft_engine="scipy", db_norm=True,
        db_min=-20.0, db_max=60.0, colormap="jet",
    )
    assert valid.colormap == "jet"
    with pytest.raises(ValueError, match="legacy reproduction mode"):
        FeaturesCfg(
            feature_type="linear", stft_engine="librosa", db_norm=True,
            db_min=-20.0, db_max=60.0, colormap="jet",
        )


def test_pcen_plus_fitted_scaling_is_legal_but_legacy_bool_stays_off():
    f = FeaturesCfg(pcen=True, scaling_method="zscore", scaling_stats={"mean": [0.1], "std": [1.0]})
    assert f.pcen and f.scaling_method == "zscore" and not f.standardize
    FeaturesCfg.model_validate(f.model_dump())


@pytest.mark.parametrize("bad", [
    dict(scaling_method="minmax", db_norm=True, feature_type="linear"),
    dict(scaling_method="zscore", scaling_stats={"mean": [0.0], "std": [0.0]}),
    dict(scaling_method="minmax", scaling_stats={"lo": [2.0], "hi": [1.0]}),
    dict(scaling_method="zscore", scaling_scope="per_band",
         scaling_stats={"mean": [0.0, 1.0], "std": [1.0]}),
    dict(scaling_method="none", scaling_stats={"mean": [0.0], "std": [1.0]}),
    dict(scaling_method="zscore", scaling_scope="global",
         scaling_stats={"mean": [0.0, 1.0], "std": [1.0, 1.0]}),
])
def test_invalid_scaling_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        FeaturesCfg(**bad)


def test_pretrained_normalization_requires_no_scaling():
    assert FeaturesCfg().matches_pretrained_normalization()
    scaled = FeaturesCfg(scaling_method="zscore", scaling_stats={"mean": [0.0], "std": [1.0]})
    assert not scaled.matches_pretrained_normalization()


# --- extractors --------------------------------------------------------------


def _wave(n=32000):
    return (np.random.default_rng(0).standard_normal(n) * 0.05).astype(np.float32)


def test_mel_torch_and_numpy_paths_agree_for_every_method_and_scope():
    from sed.audio.features import MelExtractor, scaling_spec_from_stats

    rng = np.random.default_rng(1)
    wave = _wave()
    cases = {
        ("zscore", "global"): {"mean": [-4.5], "std": [2.0]},
        ("zscore", "per_band"): {"mean": list(rng.normal(-5, 1, 128)),
                                 "std": list(rng.uniform(0.5, 2.0, 128))},
        ("minmax", "global"): {"lo": [-11.0], "hi": [2.0]},
        ("minmax", "per_band"): {"lo": list(rng.uniform(-12, -8, 128)),
                                 "hi": list(rng.uniform(0, 2, 128))},
    }
    for (method, _scope), stats in cases.items():
        spec = scaling_spec_from_stats(method, stats)
        ex = MelExtractor(scaling=spec)
        torch_out = ex(wave).numpy()
        numpy_out = ex._efficientat_numpy(wave[None, :])
        np.testing.assert_allclose(torch_out, numpy_out, atol=2e-3)
        if method == "minmax":
            assert torch_out.min() >= 0.0 and torch_out.max() <= 1.0


def test_global_zscore_spec_equals_legacy_scalar_standardize():
    from sed.audio.features import MelExtractor, scaling_spec_from_stats

    wave = _wave()
    legacy = MelExtractor(standardize=True, standardize_mean=-4.5, standardize_std=2.0)
    spec = MelExtractor(scaling=scaling_spec_from_stats("zscore", {"mean": [-4.5], "std": [2.0]}))
    assert float((legacy(wave) - spec(wave)).abs().max()) == 0.0


def test_declared_unfitted_scaling_suppresses_fast_norm():
    from sed.audio.features import MelExtractor, ScalingSpec

    wave = _wave()
    identity = MelExtractor(scaling=ScalingSpec(method="zscore"))
    raw_log = MelExtractor(standardize=True, standardize_mean=0.0, standardize_std=1.0)
    assert float((identity(wave) - raw_log(wave)).abs().max()) == 0.0


def test_pcen_output_is_scaled_by_the_spec():
    from sed.audio.features import FeatureExtractor, scaling_spec_from_stats

    wave = _wave()
    spec = scaling_spec_from_stats("zscore", {"mean": [0.5], "std": [0.25]})
    scaled = FeatureExtractor(feature_type="mel", pcen=True, scaling=spec)(wave)[0]
    plain = FeatureExtractor(feature_type="mel", pcen=True)(wave)[0]
    np.testing.assert_allclose(scaled, (plain - 0.5) / 0.25, atol=1e-5)


def test_build_feature_extractor_rejects_stale_band_vectors():
    from sed.audio.features import build_feature_extractor

    cfg = Config()
    cfg.features.scaling_method = "zscore"
    cfg.features.scaling_scope = "per_band"
    cfg.features.scaling_stats = {"mean": [0.0] * 64, "std": [1.0] * 64}
    cfg = Config.model_validate(cfg.model_dump(mode="json"))  # n_mels stays 128
    with pytest.raises(ValueError, match="re-estimate"):
        build_feature_extractor(cfg)


# --- loader parity (fix C) ---------------------------------------------------


def test_stats_loader_matches_training_loader_center_crop_and_channel(tmp_path):
    """The fit must see exactly what SedDataset._load feeds training: first
    channel and a deterministic center crop — not the preview loader's
    head-crop + channel mean."""
    from sed.data.feature_stats import load_training_window

    sr = 32000
    rng = np.random.default_rng(9)
    stereo = np.stack([
        rng.standard_normal(sr).astype(np.float32) * 0.1,
        rng.standard_normal(sr).astype(np.float32) * 0.9,
    ], axis=1)
    p = tmp_path / "stereo.wav"
    sf.write(str(p), stereo, sr)

    window_seconds = 0.4
    got = load_training_window(p, sr, window_seconds)

    from sed.data.augment import _ensure_length
    expected = _ensure_length(stereo[:, 0], int(round(window_seconds * sr)))
    # atol covers the int16 quantization of the written WAV
    np.testing.assert_allclose(got, expected, atol=1e-4)
    # and it is NOT the head crop the preview loader takes
    assert not np.allclose(got, stereo[: int(window_seconds * sr), 0], atol=1e-4)


# --- fitting -----------------------------------------------------------------


def test_fit_feature_scaling_all_methods_shapes_health_and_determinism(tmp_path):
    from sed.data.feature_stats import fit_feature_scaling

    cfg, layout = _fit_env(tmp_path)
    for method, scope in [("zscore", "global"), ("zscore", "per_band"),
                          ("minmax", "global"), ("minmax", "per_band")]:
        feats = types.SimpleNamespace(**_BASE_FEATURES, scaling_method=method,
                                      scaling_scope=scope)
        result = fit_feature_scaling(cfg, layout, features=feats, sample_rate=32000, limit=64)
        keys = ("mean", "std") if method == "zscore" else ("lo", "hi")
        expected_len = 1 if scope == "global" else 64
        assert len(result["stats"][keys[0]]) == expected_len
        assert len(result["stats"][keys[1]]) == expected_len
        assert result["clips"] > 0 and result["dataset_sha256"]
        assert result["split_seed"] == 42
        again = fit_feature_scaling(cfg, layout, features=feats, sample_rate=32000, limit=64)
        assert again["stats"] == result["stats"]
        if method == "zscore":
            assert abs(result["health"]["post_mean"]) < 0.01
            assert abs(result["health"]["post_std"] - 1.0) < 0.01
            if scope == "per_band":
                # per-band standardization zeroes every band's residual
                assert result["health"]["worst_band_mean_abs"] < 0.01
                assert result["health"]["worst_band_std_dev"] < 0.01
        else:
            # robust p1/p99 edges clip ≈2% of values by construction
            assert 0.0 < result["health"]["clipped_fraction"] < 0.1


def test_fitted_stats_load_into_config_and_extractor(tmp_path):
    from sed.audio.features import build_feature_extractor
    from sed.data.feature_stats import fit_feature_scaling

    cfg, layout = _fit_env(tmp_path)
    feats = types.SimpleNamespace(**_BASE_FEATURES, scaling_method="zscore",
                                  scaling_scope="per_band")
    result = fit_feature_scaling(cfg, layout, features=feats, sample_rate=32000, limit=64)

    full = Config()
    full.features.n_mels = 64
    full.features.scaling_method = "zscore"
    full.features.scaling_scope = "per_band"
    full.features.scaling_stats = result["stats"]
    full = Config.model_validate(full.model_dump(mode="json"))
    extractor = build_feature_extractor(full)
    out = extractor(_wave())
    out = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
    assert out.shape[1] == 64


def test_canonical_fit_uses_all_training_rows_and_reuses_signature_cache(tmp_path):
    from sed.data.feature_stats import fit_feature_scaling

    cfg, layout = _fit_env(tmp_path)
    feats = types.SimpleNamespace(**_BASE_FEATURES, scaling_method="zscore",
                                  scaling_scope="per_band")
    first = fit_feature_scaling(cfg, layout, features=feats, sample_rate=32000)
    second = fit_feature_scaling(cfg, layout, features=feats, sample_rate=32000)

    assert first["clips"] == first["training_rows"]
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["cache_key"] == first["cache_key"]
    changed = types.SimpleNamespace(**{**_BASE_FEATURES, "n_mels": 32},
                                    scaling_method="zscore", scaling_scope="per_band")
    third = fit_feature_scaling(cfg, layout, features=changed, sample_rate=32000)
    assert third["cache_hit"] is False
    assert third["cache_key"] != first["cache_key"]


def test_canonical_fit_fails_with_unreadable_training_path(tmp_path):
    from sed.data.feature_stats import fit_feature_scaling
    from sed.data.splits import split_pool_rows_detailed

    cfg, layout = _fit_env(tmp_path)
    with open(layout.dataset_dir / "train.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train_rows, _val_rows, _diagnostics = split_pool_rows_detailed(
        rows, val_frac=cfg.dataset.val_fraction, seed=cfg.training.seed,
    )
    missing = train_rows[0]["path"]
    Path(missing).unlink()
    feats = types.SimpleNamespace(**_BASE_FEATURES, scaling_method="zscore",
                                  scaling_scope="per_band")

    with pytest.raises(RuntimeError, match="could not read training clip") as exc:
        fit_feature_scaling(cfg, layout, features=feats, sample_rate=32000)
    assert missing in str(exc.value)


def test_per_band_welford_and_percentiles_match_direct_numpy(tmp_path):
    """The streamed fit must agree with a direct whole-sample computation."""
    from sed.audio.features import MelExtractor
    from sed.data.feature_stats import (
        balanced_stats_rows, fit_feature_scaling, load_training_window,
    )
    from sed.data.splits import split_pool_rows_detailed

    cfg, layout = _fit_env(tmp_path)
    feats = types.SimpleNamespace(**_BASE_FEATURES, scaling_method="zscore",
                                  scaling_scope="per_band")
    result = fit_feature_scaling(cfg, layout, features=feats, sample_rate=32000, limit=64)

    with open(layout.dataset_dir / "train.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train_rows, _va, _di = split_pool_rows_detailed(rows, val_frac=0.15, seed=42)
    sample = balanced_stats_rows(train_rows, limit=64, seed=42)
    ex = MelExtractor(n_mels=64, standardize=True, standardize_mean=0.0, standardize_std=1.0)
    values = ex(np.stack([
        load_training_window(r["path"], 32000, 1.0) for r in sample
    ], axis=0))
    values = values.numpy() if hasattr(values, "numpy") else np.asarray(values)
    direct_mean = values.transpose(1, 0, 2).reshape(64, -1).mean(axis=1)
    direct_std = values.transpose(1, 0, 2).reshape(64, -1).std(axis=1)
    np.testing.assert_allclose(result["stats"]["mean"], direct_mean, atol=1e-4)
    np.testing.assert_allclose(result["stats"]["std"], direct_std, atol=1e-4)
