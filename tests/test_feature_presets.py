"""Audio-feature presets, the warn-not-block contract, and the linear preview."""

from __future__ import annotations

import numpy as np
import pytest


_CLINK_V2_PROFILE_CASES = {
    "clink-v2-native": {
        "feature_type": "mel", "n_fft": 1024, "win_length": 882,
        "hop_length": 220, "n_mels": 128, "shape": (128, 81),
    },
    "clink-v2-dense-mel256": {
        "feature_type": "mel", "n_fft": 2048, "win_length": 882,
        "hop_length": 220, "n_mels": 256, "shape": (256, 81),
    },
    "clink-v2-resolved-mel256": {
        "feature_type": "mel", "n_fft": 2048, "win_length": 1764,
        "hop_length": 220, "n_mels": 256, "shape": (256, 81),
    },
    "clink-v2-temporal-mel128": {
        "feature_type": "mel", "n_fft": 1024, "win_length": 882,
        "hop_length": 110, "n_mels": 128, "shape": (128, 161),
    },
    "clink-v2-linear-fft1024": {
        "feature_type": "linear", "n_fft": 1024, "win_length": 882,
        "hop_length": 220, "n_mels": 128, "shape": (463, 81),
    },
    "clink-v2-linear-fft2048": {
        "feature_type": "linear", "n_fft": 2048, "win_length": 1764,
        "hop_length": 220, "n_mels": 128, "shape": (925, 81),
    },
}


def _patch_presets_dir(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PRESETS_DIR", tmp_path / "presets")
    monkeypatch.setattr(cfg_mod, "FEATURE_PRESETS_DIR", tmp_path / "presets" / "audio_features")


def _patch_project_dirs(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")


# --- preset CRUD -----------------------------------------------------------


def test_presets_seed_and_round_trip(tmp_path, monkeypatch):
    _patch_presets_dir(tmp_path, monkeypatch)
    from sed.data import presets

    names = {p["name"] for p in presets.list_feature_presets()}
    assert {"geometry-a-efficient-short", "linear-demo", "ced-context-assisted-1s"} <= names
    # Retired built-ins (superseded by the virtual model-contract entry) are
    # neither seeded nor listed.
    assert not names & {"efficientat-standard", "ced-tiny-contract"}

    context = presets.load_feature_preset("ced-context-assisted-1s")
    assert context["frontend"] == "ced"
    assert context["window_seconds"] == 1.0
    assert context["backbone"] == "ced-mini"

    saved = presets.save_feature_preset(
        "My Custom!", {"feature_type": "mel", "n_mels": 64, "sample_rate": 16000}
    )
    assert saved["name"] == "my-custom"  # slugified
    assert saved["n_mels"] == 64 and saved["sample_rate"] == 16000

    loaded = presets.load_feature_preset("my-custom")
    assert loaded["n_mels"] == 64

    assert presets.delete_feature_preset("my-custom") is True
    assert "my-custom" not in {p["name"] for p in presets.list_feature_presets()}


# ClinkCNN-v2's 128x81 default geometry is asserted in
# tests/test_clink_cnn_v2.py::test_catalog_default_extracts_exact_17640_to_128_by_81_contract.


def test_clink_v2_native_preset_locks_exact_frontend_contract(tmp_path, monkeypatch):
    _patch_presets_dir(tmp_path, monkeypatch)
    from sed.data import presets
    from sed.train.model_lab import architecture_by_id, recommended_feature_for

    expected = {
        "sample_rate": 44100,
        "n_fft": 1024,
        "win_length": 882,
        "hop_length": 220,
        "n_mels": 128,
        "fmin": 80,
        "fmax": 20000,
        "window_seconds": 0.4,
        "pcen": False,
        "standardize": False,
    }
    built_in = presets.load_feature_preset("clink-v2-native")
    contract = recommended_feature_for(architecture_by_id("clink-cnn-v2"))

    assert built_in is not None
    assert contract is not None
    for key, value in expected.items():
        assert built_in[key] == value
        assert contract[key] == value
    assert built_in["frontend"] == contract["frontend"] == "efficientat"
    assert built_in["feature_type"] == contract["feature_type"] == "mel"


@pytest.mark.parametrize(
    ("preset_name", "expected"),
    list(_CLINK_V2_PROFILE_CASES.items()),
    ids=list(_CLINK_V2_PROFILE_CASES),
)
def test_clink_v2_profile_waveform_contracts_are_exact_and_extractable(
    preset_name, expected, tmp_path, monkeypatch,
):
    _patch_presets_dir(tmp_path, monkeypatch)

    from sed.audio.features import (
        build_feature_extractor,
        feature_band_count,
        feature_frame_count_for_cfg,
        mel_filter_diagnostics,
        window_sample_count,
    )
    from sed.config import Config, FeaturesCfg
    from sed.data import presets

    preset = presets.load_feature_preset(preset_name)
    assert preset is not None
    assert preset["backbone"] == "clink-cnn-v2"
    assert preset["sample_rate"] == 44_100
    assert preset["window_seconds"] == 0.4
    assert preset["frontend"] == "efficientat"
    assert preset["fmin"] == 80
    assert preset["fmax"] == 20_000
    assert preset["pcen"] is False
    assert preset["log_power"] is True
    assert preset["db_norm"] is False
    assert preset["colormap"] == "none"
    for field in ("feature_type", "n_fft", "win_length", "hop_length", "n_mels"):
        assert preset[field] == expected[field]
    if preset_name == "clink-v2-native":
        assert not preset["label"].startswith("Experimental")
        assert preset["standardize"] is False
    else:
        assert preset["label"].startswith("Experimental")
    if expected["feature_type"] == "linear":
        assert preset["standardize"] is True
        assert preset["standardize_stats_resolved"] is False

    cfg = Config.model_validate({
        "model": {
            "backbone": "clink-cnn-v2",
            "pretrained": False,
            "initialization": "random",
            "sample_rate": preset["sample_rate"],
        },
        "features": {
            key: value
            for key, value in preset.items()
            if key in FeaturesCfg.model_fields
        },
    })
    assert feature_band_count(cfg.features, cfg.model.sample_rate) == expected["shape"][0]
    assert feature_frame_count_for_cfg(cfg) == expected["shape"][1]
    assert window_sample_count(cfg.model.sample_rate, cfg.features.window_seconds) == 17_640

    output = build_feature_extractor(cfg)(np.zeros(17_640, dtype=np.float32))
    output = np.asarray(output.detach().cpu() if hasattr(output, "detach") else output)
    assert output.shape == (1, *expected["shape"])
    assert np.isfinite(output).all()

    if expected["feature_type"] == "mel":
        diagnostics = mel_filter_diagnostics(
            preset["n_mels"], preset["n_fft"], preset["sample_rate"],
            preset["fmin"], preset["fmax"],
        )
        assert diagnostics["empty_rows"] == 0


def test_deleting_builtin_preset_stays_deleted(tmp_path, monkeypatch):
    # Built-ins are seeded once; deleting one must not be resurrected by the next
    # list() call (the user clicked Delete — it should be gone).
    _patch_presets_dir(tmp_path, monkeypatch)
    from sed.data import presets

    assert "pcen-mel" in {p["name"] for p in presets.list_feature_presets()}
    assert presets.delete_feature_preset("pcen-mel") is True
    after = {p["name"] for p in presets.list_feature_presets()}
    assert "pcen-mel" not in after
    # Other built-ins remain (only the one deleted is gone).
    assert "linear-demo" in after


def test_linear_preset_allows_zero_n_mels(tmp_path, monkeypatch):
    _patch_presets_dir(tmp_path, monkeypatch)
    from sed.data import presets

    # n_mels is irrelevant to a linear feature, so it must not block the save.
    saved = presets.save_feature_preset("lin", {"feature_type": "linear", "n_mels": 0})
    assert saved["feature_type"] == "linear"


def test_apply_preset_to_features_cfg_validates(tmp_path, monkeypatch):
    _patch_presets_dir(tmp_path, monkeypatch)
    from sed.config import FeaturesCfg
    from sed.data import presets

    preset = presets.load_feature_preset("geometry-a-efficient-short")
    fc = presets.apply_preset_to_features_cfg(FeaturesCfg(), preset)
    assert fc.feature_type == "mel" and fc.n_mels == 128
    assert fc.preset == "geometry-a-efficient-short"


def test_apply_unsaved_feature_payload_clears_stale_preset_marker():
    from sed.config import FeaturesCfg
    from sed.data import presets

    cfg = FeaturesCfg()
    cfg.preset = "efficientat-standard"
    updated = presets.apply_preset_to_features_cfg(
        cfg,
        {"name": "", "feature_type": "mel", "n_mels": 40, "hop_length": 320},
    )

    assert updated.preset == ""
    assert updated.n_mels == 40


def test_contract_feature_source_detects_weight_specific_variant():
    from sed.config import Config
    from sed.data import presets

    cfg = Config()
    cfg.model.backbone = "mn10_as_mels_40"
    cfg.model.pretrained = True
    contract = presets.contract_feature_preset("mn10_as_mels_40")
    assert contract is not None
    assert contract["n_mels"] == 40
    assert contract["window_seconds"] == 0.4
    assert contract["upstream_window_seconds"] == 1.0

    cfg.model.sample_rate = int(contract["sample_rate"])
    for field in (
        "feature_type", "frontend", "n_fft", "win_length", "hop_length",
        "n_mels", "fmin", "fmax", "window_seconds", "log_power", "pcen",
        "stft_engine", "db_norm", "colormap",
    ):
        setattr(cfg.features, field, contract[field])
    cfg.features.preset = ""

    source = presets.feature_source_for_config(cfg)
    assert source["kind"] == "contract"
    assert source["name"] == "contract:mn10_as_mels_40"

    cfg.features.n_mels = 128
    custom = presets.feature_source_for_config(cfg)
    assert custom["kind"] == "custom"
    assert custom["base_kind"] == ""


def test_model_default_feature_resolver_distinguishes_checkpoint_and_scratch_mn():
    """Every visible MN default has a full payload, including mn01 scratch."""
    from sed.config import Config
    from sed.data import presets

    pretrained = Config()
    pretrained.model.backbone = "mn10_as_mels_40"
    pretrained.model.pretrained = True
    pretrained.model.initialization = "upstream"
    checkpoint = presets.model_default_feature_preset(pretrained)
    assert checkpoint is not None
    assert checkpoint["source_kind"] == "checkpoint_contract"
    assert checkpoint["name"] == "contract:mn10_as_mels_40"
    assert checkpoint["n_mels"] == 40

    scratch = Config()
    scratch.model.backbone = "mn01"
    scratch.model.pretrained = False
    scratch.model.initialization = "random"
    scratch.model.recipe_id = "efficientat-mn"
    architecture_default = presets.model_default_feature_preset(scratch)
    assert architecture_default is not None
    assert architecture_default["source_kind"] == "architecture_default"
    assert architecture_default["name"] == "contract:mn01"
    assert architecture_default["frontend"] == "efficientat"
    assert architecture_default["n_mels"] == 128

    scratch.model.sample_rate = int(architecture_default["sample_rate"])
    for field in (
        "feature_type", "frontend", "n_fft", "win_length", "hop_length", "n_mels",
        "fmin", "fmax", "window_seconds", "log_power", "pcen", "stft_engine",
        "db_norm", "colormap",
    ):
        setattr(scratch.features, field, architecture_default[field])
    source = presets.feature_source_for_config(scratch)
    assert source["kind"] == "contract"
    assert source["source_kind"] == "architecture_default"
    assert source["label"].startswith("Model default: mn01")


def test_saved_preset_source_becomes_custom_after_edit(tmp_path, monkeypatch):
    _patch_presets_dir(tmp_path, monkeypatch)
    from sed.config import Config
    from sed.data import presets

    preset = presets.save_feature_preset(
        "legacy-copy",
        {
            "feature_type": "linear",
            "sample_rate": 44100,
            "n_fft": 190,
            "win_length": 190,
            "hop_length": 167,
            "fmax": 22050,
            "window_seconds": 0.4,
            "stft_engine": "scipy",
            "db_norm": True,
            "colormap": "jet",
            "image_size": 96,
        },
    )
    cfg = Config()
    cfg.model.sample_rate = int(preset["sample_rate"])
    cfg.features = presets.apply_preset_to_features_cfg(cfg.features, preset)

    source = presets.feature_source_for_config(cfg)
    assert source["kind"] == "saved"
    assert source["name"] == "legacy-copy"

    cfg.features.hop_length += 1
    edited = presets.feature_source_for_config(cfg)
    assert edited["kind"] == "custom"
    assert edited["base_kind"] == "saved"
    assert edited["base_name"] == "legacy-copy"


# --- apply to a project ----------------------------------------------------


def test_apply_feature_preset_writes_config_and_selection(tmp_path, monkeypatch):
    _patch_presets_dir(tmp_path, monkeypatch)
    _patch_project_dirs(tmp_path, monkeypatch)
    from sed.config import ProjectLayout, load_config, project_config_template
    from sed.data import presets
    from sed.data.compose import load_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p"), encoding="utf-8")

    presets.save_feature_preset(
        "custom-mel",
        {
            "feature_type": "mel",
            "n_mels": 96,
            "sample_rate": 16000,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "fmax": 8000,
            "backbone": "mn10_as",
            "pretrained": True,
        },
    )

    result = presets.apply_feature_preset_to_project("p", "custom-mel")
    assert result["ok"] is True

    cfg = load_config("p")
    assert cfg.features.n_mels == 96
    assert cfg.features.preset == "custom-mel"
    assert cfg.model.sample_rate == 16000
    assert cfg.model.backbone == "mn10_as"

    selection = load_dataset_selection(layout)
    assert selection[presets.DATASET_PRESET_KEY] == "custom-mel"

    # Off-contract (96 mels @ 16k while pretrained) → warning, but apply succeeds.
    assert result["warning"]


# --- warn-not-block contract ----------------------------------------------


def test_off_contract_pretrained_warns_not_raises():
    from sed.config import (
        Config,
        pretrained_contract_warning,
        validate_pretrained_feature_contract,
    )

    cfg = Config()
    cfg.model.pretrained = True
    cfg.features.n_mels = 64  # off-contract
    # Must not raise.
    validate_pretrained_feature_contract(cfg, phase="training")
    assert "n_mels" in pretrained_contract_warning(cfg)


def test_linear_and_pcen_train_with_warning_not_block():
    # v2: any feature is trainable. Linear / PCEN with a pretrained backbone must
    # NOT block training — they only warn (and the warning names the cause).
    from sed.config import (
        Config,
        pretrained_contract_warning,
        validate_pretrained_feature_contract,
    )

    for mutate in (
        lambda c: setattr(c.features, "feature_type", "linear"),
        lambda c: setattr(c.features, "pcen", True),
    ):
        cfg = Config()
        cfg.model.pretrained = True
        mutate(cfg)
        # Never raises (warn-only).
        validate_pretrained_feature_contract(cfg, phase="training")
        warning = pretrained_contract_warning(cfg)
        assert warning  # incompatibility is surfaced
        assert ("feature_type" in warning) or ("pcen" in warning.lower())

    # The removed hard block must be gone.
    import sed.config as config_mod
    assert not hasattr(config_mod, "require_trainable_feature_type")


def test_on_contract_has_no_warning():
    from sed.config import Config, pretrained_contract_warning

    cfg = Config()
    cfg.model.pretrained = True
    assert pretrained_contract_warning(cfg) == ""


def test_hop_variant_backbone_expects_its_native_hop():
    from sed.config import Config, pretrained_feature_mismatches

    cfg = Config()
    cfg.model.pretrained = True
    cfg.model.backbone = "mn10_as_hop_5"
    cfg.features.hop_length = 160          # 5 ms @ 32 kHz — this variant's native hop
    cfg.features.window_seconds = 0.4      # deployment choice, never a weight contract
    assert pretrained_feature_mismatches(cfg) == []

    cfg.features.hop_length = 320          # the generic default is off-contract here
    assert any("hop_length" in m for m in pretrained_feature_mismatches(cfg))


def test_mels_variant_backbone_expects_its_native_band_count():
    """Regression: the check used to compare every backbone against the global
    128-mel contract, so mn10_as_mels_40 at its correct n_mels=40 warned."""
    from sed.config import Config, pretrained_feature_mismatches

    cfg = Config()
    cfg.model.pretrained = True
    cfg.model.backbone = "mn10_as_mels_40"
    cfg.features.n_mels = 40
    assert pretrained_feature_mismatches(cfg) == []

    cfg.features.n_mels = 128              # off-contract for this checkpoint
    assert any("n_mels=128 (expected 40)" in m for m in pretrained_feature_mismatches(cfg))


def test_unknown_mn_backbone_falls_back_to_global_contract():
    from sed.config import Config, pretrained_feature_mismatches

    cfg = Config()
    cfg.model.pretrained = True
    cfg.model.backbone = "mn10_custom_variant"
    cfg.features.n_mels = 40
    assert any("n_mels=40 (expected 128)" in m for m in pretrained_feature_mismatches(cfg))


# --- linear preview dispatcher --------------------------------------------


def test_waveform_to_feature_linear_is_2d():
    from sed.audio.features import waveform_to_feature

    wave = (np.random.randn(32000) * 0.1).astype(np.float32)
    mel = waveform_to_feature(wave, 32000, feature_type="mel", n_mels=128)
    lin = waveform_to_feature(wave, 32000, feature_type="linear", n_fft=1024, fmax=16000)
    assert mel.ndim == 2 and mel.shape[0] == 128
    assert lin.ndim == 2 and lin.shape[0] > 128  # full linear resolution


# --- generalized extractor + band count (v2) ------------------------------


def test_feature_extractor_mel_matches_mel_extractor():
    # build_feature_extractor must keep the verified mel+efficientat path byte-for-byte.
    from sed.audio.features import MelExtractor, build_feature_extractor
    from sed.config import Config

    cfg = Config()
    ext = build_feature_extractor(cfg)
    assert isinstance(ext, MelExtractor)
    wave = (np.random.randn(32000) * 0.1).astype(np.float32)
    ref = MelExtractor(sample_rate=32000)(wave)
    out = ext(wave)
    ref = ref.numpy() if hasattr(ref, "numpy") else ref
    out = out.numpy() if hasattr(out, "numpy") else out
    assert np.allclose(np.asarray(out), np.asarray(ref), atol=1e-6)


def test_feature_extractor_linear_and_pcen_finite_and_band_count():
    from sed.audio.features import (
        FeatureExtractor,
        LinearExtractor,
        build_feature_extractor,
        feature_band_count,
    )
    from sed.config import Config

    wave = (np.random.randn(32000) * 0.1).astype(np.float32)

    cfg_lin = Config()
    cfg_lin.features.feature_type = "linear"
    ext_lin = build_feature_extractor(cfg_lin)
    assert isinstance(ext_lin, LinearExtractor)
    out_lin = ext_lin(wave)
    assert out_lin.shape[1] == feature_band_count(cfg_lin.features, 32000)
    out_lin = out_lin.detach().cpu().numpy()
    assert np.isfinite(out_lin).all()

    cfg_pcen = Config()
    cfg_pcen.features.pcen = True
    ext_pcen = build_feature_extractor(cfg_pcen)
    assert isinstance(ext_pcen, FeatureExtractor)
    out_pcen = ext_pcen(wave)
    assert out_pcen.shape[1] == 128 == feature_band_count(cfg_pcen.features, 32000)
    assert np.isfinite(out_pcen).all()


def test_apply_preset_writes_pretrained_flag(tmp_path, monkeypatch):
    _patch_presets_dir(tmp_path, monkeypatch)
    _patch_project_dirs(tmp_path, monkeypatch)
    from sed.config import ProjectLayout, load_config, project_config_template
    from sed.data import presets

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p"), encoding="utf-8")

    # An off-contract feature ships with pretrained=false; applying writes that
    # flag directly (initialization is owned by the Model tab, not auto-flipped).
    presets.save_feature_preset(
        "lin-scratch",
        {"feature_type": "linear", "fmax": 16000, "backbone": "mn10_as",
         "pretrained": False},
    )
    result = presets.apply_feature_preset_to_project("p", "lin-scratch")
    assert result["ok"] is True

    cfg = load_config("p")
    assert cfg.features.feature_type == "linear"
    assert cfg.model.pretrained is False


# --- interactive analysis --------------------------------------------------


def test_feature_analysis_shapes_and_metrics():
    from sed.data.inspect import (
        analyze_feature_clips,
        default_feature_preview_clip,
        feature_analysis,
    )

    clip = default_feature_preview_clip()
    if not clip.exists():
        import pytest
        pytest.skip("default sample clip not present")

    a = feature_analysis(clip, feature_type="mel", sample_rate=32000)
    assert a["n_rows"] == 128 and a["n_cols"] > 0
    assert len(a["matrix"]) == len(a["freqs_hz"]) == min(128, 192)
    assert len(a["matrix"][0]) == len(a["times_ms"])
    assert set(a["region_split"]) == {"low", "mid", "high"}
    assert a["mel_squash"]["ratio"] >= 1
    assert isinstance(a["verdicts"], list)
    assert a["transient"] is None or "start_ms" in a["transient"]

    fp = a["fingerprint"]
    assert fp is not None and fp["source"] in ("onset", "loudest")
    assert fp["end_ms"] > fp["start_ms"] >= 0
    assert 0.0 <= fp["flatness"] <= 1.0 and fp["tonality"] in ("tonal", "mixed", "broadband")
    for p in fp["peaks"]:
        assert p["lo_hz"] <= p["center_hz"] <= p["hi_hz"]
        assert 0.0 <= p["energy_frac"] <= 1.0
        assert isinstance(p["salience_db"], float)
        assert p["tier"] in ("narrow", "broad")
        if "mel_band_hz" in p:
            assert p["mel_band_hz"] > 0 and isinstance(p["mel_resolved"], bool)
    # Narrow salient peaks are listed before broad regions, never after.
    tiers = [p["tier"] for p in fp["peaks"]]
    assert tiers == sorted(tiers, key=lambda t: 0 if t == "narrow" else 1)
    if fp["mel_fit"] is not None:
        assert isinstance(fp["mel_fit"]["ok"], bool) and fp["mel_fit"]["note"]

    # The resolved analysis window + per-band onset energy (what reconciles the
    # fingerprint pills with the band readout) are always present.
    assert a["onset"]["source"] in ("manual", "onset", "loudest")
    assert a["onset"]["end_ms"] > a["onset"]["start_ms"] >= 0
    assert a["onset"]["start_ms"] == fp["start_ms"] and a["onset"]["end_ms"] == fp["end_ms"]
    assert len(a["band_energy_onset"]) == len(a["freqs_hz"])
    assert a["band_energy_onset_total"] > 0

    multi = analyze_feature_clips([clip, clip], feature_type="mel")
    assert len(multi["clips"]) == 2
    assert len(multi["avg_band_mean"]) == 128
    # Same clip twice → every consensus range must be shared by both clips.
    consensus = multi["fingerprint_consensus"]
    if fp["peaks"]:
        assert consensus and all(m["clips"] == 2 for m in consensus)
        assert all(m["lo_hz"] <= m["center_hz"] <= m["hi_hz"] for m in consensus)


def test_fingerprint_hz_labels_use_kaldi_mel_centers():
    # Mel row labels must match the Kaldi/HTK banks the frontend applies —
    # librosa's Slaney centers drift hundreds of Hz off in the upper bands.
    from sed.audio.features import kaldi_mel_center_freqs
    from sed.data.inspect import default_feature_preview_clip, feature_analysis

    clip = default_feature_preview_clip()
    if not clip.exists():
        import pytest
        pytest.skip("default sample clip not present")
    a = feature_analysis(clip, feature_type="mel", n_mels=128, fmin=0, fmax=15000)
    centers = kaldi_mel_center_freqs(128, 0.0, 15000.0)
    assert abs(a["freqs_hz"][-1] - centers[-1]) < 1.0
    assert abs(a["freqs_hz"][0] - centers[0]) < 1.0


# --- v3: band alignment, robust range, model fit, overlay survival --------


def test_band_arrays_align_with_matrix_rows_linear():
    # The v2 bug: linear band_mean was full-res (513) while matrix/freqs were 192.
    from sed.data.inspect import default_feature_preview_clip, feature_analysis

    clip = default_feature_preview_clip()
    if not clip.exists():
        import pytest
        pytest.skip("default sample clip not present")
    a = feature_analysis(clip, feature_type="linear", fmax=16000)
    n = len(a["freqs_hz"])
    assert n == len(a["matrix"]) == len(a["band_mean"]) == len(a["band_energy"]) == n
    assert len(a["band_energy_onset"]) == n
    assert a["band_energy_total"] > 0


def test_robust_display_range_monotonic_and_mel_absolute():
    from sed.audio.features import display_range_for_spec, robust_display_range
    import numpy as np

    spec = (np.random.randn(40, 40) * 5).astype("float32")
    lo, hi = robust_display_range(spec)
    assert lo is not None and hi is not None and hi > lo
    # mel keeps its calibrated absolute range; linear/pcen go robust per-image.
    assert display_range_for_spec(spec, "mel") == (-1.4, 1.6)
    assert display_range_for_spec(spec, "linear") != (-1.4, 1.6)
    assert display_range_for_spec(spec, "mel", pcen=True) != (-1.4, 1.6)


def test_mn_reduce_matches_architecture():
    from sed.audio.features import _mn_reduce

    # SAME-padding stride-2 layers; verified against a real forward through the
    # vendored MN/DyMN backbones (128x1000 -> 4x32, 128x100 -> 4x4).
    assert _mn_reduce(128) == 4       # AudioSet mel bands
    assert _mn_reduce(1000) == 32     # 10 s of frames
    assert _mn_reduce(100) == 4       # 1 s window
    assert _mn_reduce(64) == 2        # half-resolution mel
    assert _mn_reduce(32) == 1        # thin but valid (one frequency cell)


def test_model_fit_report_default_and_thin():
    from sed.audio.features import model_fit_report
    from sed.config import Config

    cfg = Config()
    r = model_fit_report(cfg.features, "mn10_as", True, 32000)
    # The project default is a 0.4 s window: 40 frames, which the backbone's
    # 32x reduction takes to 2 temporal cells. Still on the AudioSet contract —
    # window length is a deployment choice, not part of the weight contract.
    assert r["input_shape"] == [1, 1, 128, 40]
    assert r["feature_map"] == [4, 2]
    assert r["feat_dim"] == 960
    assert r["on_contract"] is True
    assert all(v["level"] != "error" for v in r["verdicts"])
    # The analytic EfficientAT report carries a real (scratch-built) param
    # count; checkpoint variants share the base width's architecture.
    assert isinstance(r["total_params"], int) and r["total_params"] > 1_000_000
    r256 = model_fit_report(cfg.features, "mn10_as_mels_256", True, 32000)
    assert r256["total_params"] == r["total_params"]
    r20 = model_fit_report(cfg.features, "mn20_as", False, 32000)
    assert r20["total_params"] > r["total_params"]

    # 32 mel bands collapse the frequency axis to a single (thin) cell: still
    # trainable through the conv + global pool, so it warns rather than errors.
    tiny = Config()
    tiny.features.n_mels = 32
    rt = model_fit_report(tiny.features, "mn01", False, 32000)
    assert rt["feature_map"][0] == 1
    assert any(v["level"] == "warn" for v in rt["verdicts"])


def test_model_fit_report_uses_feature_channels_for_colormap_inputs():
    from sed.audio.features import model_fit_report
    from sed.config import Config

    one_channel = Config()
    one_channel.features.feature_type = "linear"
    legacy_spec = model_fit_report(
        one_channel.features, "legacy-smallervggnet", False, 32000
    )
    assert legacy_spec["input_shape"][1] == 1
    assert legacy_spec["input_channels"] == 1

    cfg = Config()
    cfg.features.feature_type = "linear"
    cfg.features.colormap = "jet"
    cfg.features.image_size = 96
    legacy = model_fit_report(cfg.features, "legacy-smallervggnet", False, 32000)
    assert legacy["input_shape"] == [1, 3, 96, 96]
    assert legacy["input_channels"] == 3

    mn_scratch = model_fit_report(cfg.features, "mn01", False, 32000)
    assert mn_scratch["input_shape"] == [1, 3, 96, 96]
    assert any("3 input channels" in v["text"] for v in mn_scratch["verdicts"])

    mn_pretrained = model_fit_report(cfg.features, "mn10_as", True, 32000)
    assert mn_pretrained["input_shape"] == [1, 3, 96, 96]
    assert any(v["level"] == "error" and "1-channel" in v["text"]
               for v in mn_pretrained["verdicts"])


def test_ced_tiny_feature_contract_and_model_fit_shape():
    from sed.audio.features import feature_frame_count, model_fit_report
    from sed.config import Config
    from sed.data import presets

    # The CED contract is the virtual (unsaved) model-default entry now — the
    # old ced-tiny-contract built-in preset was retired in its favor.
    preset = presets.contract_feature_preset("ced-tiny")
    assert preset is not None
    assert preset["frontend"] == "ced"
    assert preset["sample_rate"] == 16000
    assert preset["n_mels"] == 64
    assert preset["n_fft"] == preset["win_length"] == 512
    assert preset["hop_length"] == 160

    cfg = Config.model_validate({
        "model": {"sample_rate": 16000, "backbone": "ced-tiny", "pretrained": True},
        "features": {k: preset[k] for k in (
            "feature_type", "frontend", "n_fft", "win_length", "hop_length",
            "n_mels", "fmin", "fmax", "window_seconds", "log_power",
        )},
    })
    assert feature_frame_count(
        sample_rate=16000,
        window_seconds=0.4,
        hop_length=160,
        n_fft=512,
        frontend="ced",
        win_length=512,
    ) == 41
    report = model_fit_report(cfg.features, "ced-tiny", True, 16000)
    assert report["input_shape"] == [1, 1, 64, 41]
    assert report["input_channels"] == 1


def test_ced_feature_extractor_shape():
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from sed.audio.features import build_feature_extractor
    from sed.config import Config

    cfg = Config.model_validate({
        "model": {"sample_rate": 16000},
        "features": {
            "frontend": "ced", "feature_type": "mel", "n_fft": 512,
            "win_length": 512, "hop_length": 160, "n_mels": 64,
            "fmin": 0, "fmax": 8000, "window_seconds": 0.4,
        },
    })
    ext = build_feature_extractor(cfg)
    out = ext(np.zeros(6400, dtype=np.float32))
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    assert out.shape == (1, 64, 41)


def test_ced_frontend_rejects_off_contract_geometry():
    from pydantic import ValidationError
    from sed.config import Config

    with pytest.raises(ValidationError, match="n_mels=64"):
        Config.model_validate({
            "model": {"sample_rate": 16000},
            "features": {
                "frontend": "ced", "feature_type": "mel", "n_fft": 512,
                "win_length": 512, "hop_length": 160, "n_mels": 128,
                "fmin": 0, "fmax": 8000,
            },
        })
    with pytest.raises(ValidationError, match="model.sample_rate=16000"):
        Config.model_validate({
            "model": {"sample_rate": 32000},
            "features": {
                "frontend": "ced", "feature_type": "mel", "n_fft": 512,
                "win_length": 512, "hop_length": 160, "n_mels": 64,
                "fmin": 0, "fmax": 8000,
            },
        })


def test_model_fit_report_dymn_and_frame_task():
    from sed.audio.features import model_fit_report
    from sed.config import Config

    # DyMN is analytic (same 32x reduction as mn) and on the AudioSet contract.
    # Pinned to 1 s so the thin-window contrast below stays a real contrast.
    cfg = Config()
    cfg.features.window_seconds = 1.0
    r = model_fit_report(cfg.features, "dymn10_as", True, 32000, task="clip")
    assert r["feature_map"] == [4, 4]
    assert r["feat_dim"] == 960
    assert r["on_contract"] is True

    # A 0.4 s window leaves only ~2 temporal cells: fine for clip, warned for frame.
    short = Config()
    short.features.window_seconds = 0.4
    rc = model_fit_report(short.features, "dymn10_as", True, 32000, task="clip")
    assert not any("temporal cell" in v["text"] for v in rc["verdicts"])
    rf = model_fit_report(short.features, "dymn10_as", True, 32000, task="frame")
    assert any(v["level"] == "warn" and "temporal cell" in v["text"] for v in rf["verdicts"])


def test_model_fit_report_contract_compatibility_severity():
    from sed.audio.features import model_fit_report
    from sed.config import Config

    # On-contract pretrained: one OK compatibility verdict, no severity tags.
    on = model_fit_report(Config().features, "mn10_as", True, 32000)
    assert any("transfer directly" in v["text"] for v in on["verdicts"])
    assert not any(v.get("severity") for v in on["verdicts"])

    # Off-contract pretrained: a HIGH feature_type mismatch plus a LOW hop one,
    # each tagged with a severity (the Model tab's compatibility report).
    off = Config()
    off.features.feature_type = "linear"
    off.features.hop_length = 160
    r = model_fit_report(off.features, "mn10_as", True, 32000)
    sev = {v.get("severity") for v in r["verdicts"] if v.get("severity")}
    assert "high" in sev and "low" in sev
    assert any("feature_type" in v["text"] and v.get("severity") == "high" for v in r["verdicts"])
    assert any("Off the AudioSet-pretrained contract" in v["text"] for v in r["verdicts"])

    # Scratch initialization accepts any feature geometry (no contract penalty).
    sc = model_fit_report(off.features, "mn10_as", False, 32000)
    assert any("Scratch initialization" in v["text"] for v in sc["verdicts"])
    assert not any(v.get("severity") for v in sc["verdicts"])


def test_clink_cnn_v2_is_a_valid_clip_embedding_model():
    from sed.audio.features import model_fit_report
    from sed.config import Config

    cfg = Config()
    cfg.features.window_seconds = 0.4
    cfg.features.hop_length = 220
    clip = model_fit_report(cfg.features, "clink-cnn-v2", False, 44100, task="clip")
    assert any(
        v["level"] == "ok" and v.get("kind") == "vector_embedding"
        for v in clip["verdicts"]
    )
    assert not any(v["level"] == "warn" and "non-spatial" in v["text"] for v in clip["verdicts"])

    # ClinkCNN-v2 advertises only the clip capability.
    frame = model_fit_report(cfg.features, "clink-cnn-v2", False, 44100, task="frame")
    assert any(v["level"] == "error" and "frame-level" in v["text"] for v in frame["verdicts"])


def test_validate_feature_params_messages():
    from sed.data.inspect import validate_feature_params

    base = dict(sample_rate=32000, n_fft=1024, win_length=800, hop_length=320,
                n_mels=128, fmin=0, fmax=15000, feature_type="mel")
    assert validate_feature_params(base) is None
    assert "win_length" in validate_feature_params({**base, "n_fft": 256, "win_length": 600})
    assert "n_fft" in validate_feature_params({**base, "n_fft": 1})
    assert "power of two" in validate_feature_params({**base, "n_fft": 1000})
    assert "fmax" in validate_feature_params({**base, "fmin": 100, "fmax": 50})
    assert "Nyquist" in validate_feature_params({**base, "fmax": 20000})
    assert "n_mels" in validate_feature_params({**base, "n_fft": 256, "win_length": 200, "n_mels": 200})


def test_analyze_feature_clips_surfaces_error_not_silent():
    from sed.data.inspect import analyze_feature_clips, default_feature_preview_clip

    clip = default_feature_preview_clip()
    if not clip.exists():
        import pytest
        pytest.skip("default sample clip not present")
    bad = analyze_feature_clips([clip], feature_type="mel", n_fft=256, win_length=600)
    assert bad["clips"] == [] and bad.get("error")
    good = analyze_feature_clips([clip], feature_type="mel")
    assert good["clips"] and not good.get("error")


def test_backbone_registry_carries_reduction():
    from sed.config import backbone_registry_payload

    payload = backbone_registry_payload()
    mn10 = next(b for b in payload if b["name"] == "mn10_as")
    assert mn10["downsample"] == 32
    assert isinstance(mn10["reduction_layers"], list) and len(mn10["reduction_layers"]) == 5
    assert all(len(l) == 3 for l in mn10["reduction_layers"])
    clink_v2 = next(b for b in payload if b["name"] == "clink-cnn-v2")
    assert clink_v2["recommended"]["model.sample_rate"] == 44100
    assert clink_v2["recommended"]["features.window_seconds"] == 0.4
    assert clink_v2["recommended"]["features.fmin"] == 80
    assert clink_v2["recommended"]["features.fmax"] == 20000


def test_overlay_survival_present_and_aligned():
    from sed.data.inspect import default_feature_preview_clip, feature_analysis

    clip = default_feature_preview_clip()
    if not clip.exists():
        import pytest
        pytest.skip("default sample clip not present")
    base = feature_analysis(clip, feature_type="mel")
    assert base["survival"] is None  # no overlay → no survival
    assert base["overlay_view"] is None
    mixed = feature_analysis(clip, feature_type="mel", overlay_path=str(clip), snr_db=0.0)
    assert mixed["survival"] is not None
    assert len(mixed["survival"]) == len(mixed["freqs_hz"])
    # clip mixed with itself at 0 dB → ~0 dB margin everywhere
    assert max(abs(s) for s in mixed["survival"]) < 1.0
    # onset arrays (computed on the clean target) stay present in overlay mode
    assert len(mixed["band_energy_onset"]) == len(mixed["freqs_hz"])
    assert mixed["band_energy_onset_total"] > 0


def test_overlay_view_three_separated_specs():
    from sed.data.inspect import default_feature_preview_clip, feature_analysis

    clip = default_feature_preview_clip()
    if not clip.exists():
        import pytest
        pytest.skip("default sample clip not present")
    a = feature_analysis(clip, feature_type="mel", overlay_path=str(clip), snr_db=6.0)
    ov = a["overlay_view"]
    assert ov is not None and ov["snr_db"] == 6.0
    nfreq = len(a["freqs_hz"])
    for which in ("target", "overlay", "mixed"):
        v = ov[which]
        assert len(v["matrix"]) == nfreq
        assert len(v["band_mean"]) == len(v["band_energy"]) == nfreq
        assert v["band_energy_total"] > 0
    # overlay scaled below target at +6 dB SNR; mixed carries the most energy.
    assert ov["overlay"]["band_energy_total"] < ov["target"]["band_energy_total"]
    assert ov["mixed"]["band_energy_total"] >= ov["target"]["band_energy_total"]


def test_display_bins_align_with_freq_labels_linear():
    # Regression: display bins were array_split groups while freqs_hz labeled
    # r_idx sample rows — in linear mode (481 rows → 192 bins) the energy bins
    # drifted up to ~48 rows (~1.5 kHz) off their Hz labels.
    from sed.data.inspect import _feature_analysis_arrays

    sr = 32000
    t = np.arange(sr, dtype=np.float32) / sr
    wave = np.sin(2 * np.pi * 4000.0 * t).astype(np.float32)
    a = _feature_analysis_arrays(
        wave, sr, feature_type="linear", n_fft=1024, win_length=800,
        hop_length=320, n_mels=128, fmin=0, fmax=16000,
        log_power=True, pcen=False, pcen_params=None)
    freqs = a["freqs_hz"]
    bin_width = (freqs[-1] - freqs[0]) / max(1, len(freqs) - 1)
    peak_hz = freqs[int(np.argmax(a["band_energy"]))]
    assert abs(peak_hz - 4000.0) <= 1.5 * bin_width
    onset_peak_hz = freqs[int(np.argmax(a["band_energy_onset"]))]
    assert abs(onset_peak_hz - 4000.0) <= 1.5 * bin_width


def test_fingerprint_ranks_narrow_salient_first():
    # A narrow loud spike must outrank a wide bump that carries MORE total
    # energy — narrow concentration is what distinguishes a sound in noise.
    from sed.data.inspect import _resolve_onset_window, _spectral_fingerprint

    sr, n_fft, hop_ms = 32000, 1024, 10.0
    full_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    n_bins, n_cols = len(full_freqs), 100
    lin_power = np.full((n_bins, n_cols), 1e-8, dtype=np.float64)
    bins = np.arange(n_bins, dtype=np.float64)
    narrow = 0.6 * np.exp(-0.5 * ((bins - 100) / 3.0) ** 2)    # sharp spike ≈3.1 kHz
    wide = 0.08 * np.exp(-0.5 * ((bins - 300) / 40.0) ** 2)    # wide bump ≈9.4 kHz
    assert wide.sum() > 1.5 * narrow.sum()                     # wide carries more energy
    lin_power[:, 50:60] += (narrow + wide)[:, None]

    window = _resolve_onset_window(n_cols, lin_power, hop_ms, transient=None,
                                   window_ms=100, onset_start_ms=500.0, onset_end_ms=600.0)
    fp = _spectral_fingerprint(lin_power, full_freqs, hop_ms, window=window,
                               n_peaks=6, prominence_db=6.0,
                               n_mels=128, fmin=0.0, fmax=15000.0)
    assert fp["source"] == "manual"
    peaks = fp["peaks"]
    assert len(peaks) >= 2
    assert peaks[0]["tier"] == "narrow"
    assert 2900 < peaks[0]["center_hz"] < 3400
    wide_peak = next(p for p in peaks if 8800 < p["center_hz"] < 10000)
    assert wide_peak["tier"] == "broad"
    assert wide_peak["energy_frac"] > peaks[0]["energy_frac"]
    assert peaks[0]["salience_db"] > wide_peak["salience_db"] + 3.0


def test_manual_onset_override_and_percent_reconciliation():
    from sed.data.inspect import default_feature_preview_clip, feature_analysis

    clip = default_feature_preview_clip()
    if not clip.exists():
        import pytest
        pytest.skip("default sample clip not present")
    a = feature_analysis(clip, feature_type="linear", fmax=16000,
                         onset_start_ms=0.0, onset_end_ms=200.0)
    assert a["onset"]["source"] == "manual"
    assert a["fingerprint"]["source"] == "manual"
    assert a["onset"]["start_ms"] <= 10 and 190 <= a["onset"]["end_ms"] <= 210
    # Clicking a fingerprint pill selects the display rows inside its Hz range;
    # the readout's onset % must reproduce the pill % (same window, same total).
    freqs = a["freqs_hz"]
    beo, tot = a["band_energy_onset"], a["band_energy_onset_total"]
    peaks = a["fingerprint"]["peaks"]
    assert peaks
    for p in peaks[:2]:
        sel = [i for i, f in enumerate(freqs) if p["lo_hz"] <= f <= p["hi_hz"]]
        got = sum(beo[i] for i in sel) / tot
        assert abs(got - p["energy_frac"]) <= 0.02


def test_render_feature_preview_png(tmp_path, monkeypatch):
    import soundfile as sf

    from sed.data.inspect import render_feature_preview_png

    wav = tmp_path / "clip.wav"
    sf.write(wav, (np.random.randn(48000) * 0.1).astype(np.float32), 48000)
    for ft in ("mel", "linear"):
        png = render_feature_preview_png(wav, feature_type=ft)
        assert png[:4] == b"\x89PNG"


# --- TOML writer -----------------------------------------------------------


def test_update_config_toml_preserves_comments(tmp_path):
    from sed.config import update_config_toml_values

    path = tmp_path / "config.toml"
    path.write_text(
        '[features]\nfrontend = "efficientat"  # keep me\nn_fft = 1024\n\n[model]\nbackbone = "mn10_as"\n',
        encoding="utf-8",
    )
    update_config_toml_values(
        path, {"features": {"n_mels": 64, "feature_type": "mel"}, "model": {"sample_rate": 16000}}
    )
    import tomllib

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["features"]["n_mels"] == 64
    assert parsed["features"]["feature_type"] == "mel"
    assert parsed["features"]["n_fft"] == 1024  # untouched
    assert parsed["model"]["sample_rate"] == 16000
    assert "# keep me" in path.read_text(encoding="utf-8")  # comment preserved


# --- augment routing -------------------------------------------------------


def test_set_augment_features_routes_hop_and_win():
    from sed.config import Config
    from sed.data import augment

    cfg = Config()
    cfg.features.win_length = 400
    cfg.features.hop_length = 160
    cfg.features.n_mels = 64
    augment.set_augment_features(cfg)
    try:
        assert augment._mel_win_length(32000) == 400
        assert augment._mel_hop_length(32000) == 160
        assert augment._active_mel["n_mels"] == 64
    finally:
        augment.set_augment_features(None)  # reset module state for other tests
    assert augment._mel_win_length(32000) == 800
