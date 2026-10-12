"""Tests for the config snapshot mechanism — the safeguard that decouples
inference from edits to projects/<id>/config.toml made after a model was trained.

The contract we want:
  1. dump_config_snapshot(cfg, p)  + load_config_snapshot(p)  → round-trips
  2. load_config(project_id) reads projects/<id>/config.toml.
  3. load_config_for_inference(id) uses models/<model_id>/training_config.json
     plus generated models/<model_id>/runtime_settings.json for model-scoped
     knobs, then live config.toml for edge-local operator settings.
"""

from __future__ import annotations

import json

import pytest

from sed.config import (
    dump_config_snapshot,
    dump_runtime_settings,
    load_config,
    load_config_for_inference,
    load_config_snapshot,
    validate_pretrained_feature_contract,
    write_project_config_template,
)


def test_snapshot_round_trip(tmp_path):
    cfg = load_config()
    cfg.training.class_balance = "balanced"
    p = tmp_path / "snapshot.json"
    dump_config_snapshot(cfg, p)
    cfg2 = load_config_snapshot(p)
    assert cfg.model.sample_rate == cfg2.model.sample_rate
    assert cfg.features.n_mels == cfg2.features.n_mels
    assert cfg.inference.consecutive_frames == cfg2.inference.consecutive_frames
    assert cfg.training.focal_gamma_neg == cfg2.training.focal_gamma_neg
    assert cfg2.training.class_balance == "balanced"


@pytest.mark.parametrize("mode", ["off", "sqrt", "balanced"])
def test_class_balance_accepts_canonical_modes(mode):
    from sed.config import Config

    cfg = Config.model_validate({"training": {"class_balance": f" {mode.upper()} "}})
    assert cfg.training.class_balance == mode


def test_class_balance_defaults_to_off_and_rejects_unknown_mode():
    from sed.config import Config

    assert Config().training.class_balance == "off"
    with pytest.raises(ValueError, match="class_balance"):
        Config.model_validate({"training": {"class_balance": "aggressive"}})


@pytest.mark.parametrize(
    ("legacy_training", "expected"),
    [
        ({"sampling_strategy": "class_group_sqrt"}, "balanced"),
        ({"negative_weight": 1.5}, "sqrt"),
        ({"sampling_strategy": "class_group_sqrt", "negative_weight": 3.0}, "balanced"),
    ],
)
def test_legacy_balance_knobs_migrate_to_one_policy(legacy_training, expected):
    from sed.config import Config

    cfg = Config.model_validate({"training": legacy_training})
    assert cfg.training.class_balance == expected


@pytest.mark.parametrize("canonical", ["off", "sqrt", "balanced"])
def test_explicit_class_balance_wins_over_legacy_knobs(canonical):
    from sed.config import Config

    cfg = Config.model_validate({
        "training": {
            "class_balance": canonical,
            "sampling_strategy": "class_group_sqrt",
            "negative_weight": 9.0,
        },
    })
    assert cfg.training.class_balance == canonical


def test_legacy_balance_snapshot_is_canonicalized_and_omits_old_keys(tmp_path):
    from sed.config import Config

    cfg = Config.model_validate({
        "training": {
            "sampling_strategy": "class_group_sqrt",
            "negative_weight": 1.5,
        },
    })
    path = dump_config_snapshot(cfg, tmp_path / "snapshot.json")

    training = json.loads(path.read_text(encoding="utf-8"))["training"]
    assert training["class_balance"] == "balanced"
    assert "sampling_strategy" not in training
    assert "negative_weight" not in training
    assert load_config_snapshot(path).training.class_balance == "balanced"


@pytest.mark.parametrize(
    ("configured", "canonical"),
    [
        ("16", "16-mixed"),
        ("16-mixed", "16-mixed"),
        ("bf16", "bf16-mixed"),
        ("bf16-mixed", "bf16-mixed"),
        ("32", "32-true"),
        ("32-true", "32-true"),
    ],
)
def test_training_precision_uses_one_canonical_spelling(configured, canonical):
    from sed.config import Config

    cfg = Config.model_validate({"training": {"precision": configured}})
    assert cfg.training.precision == canonical


def test_training_precision_rejects_unsupported_modes():
    from sed.config import Config

    with pytest.raises(ValueError, match="training.precision"):
        Config.model_validate({"training": {"precision": "fp8"}})


def test_project_kind_round_trips(tmp_path, monkeypatch):
    """`--pretraining` writes [project] kind, which load_config + is_pretraining_project
    read back. A template without the flag is a venue project."""
    from sed import config as cfg_mod
    from sed.config import is_pretraining_project

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)

    venue = cfg_mod.ProjectLayout("venue")
    venue.create()
    write_project_config_template(venue, name="Venue")  # default kind
    vcfg = load_config("venue")
    assert vcfg.project.kind == "venue"
    assert is_pretraining_project(vcfg) is False

    pre = cfg_mod.ProjectLayout("pre")
    pre.create()
    write_project_config_template(pre, name="Pre", kind="pretraining")
    pcfg = load_config("pre")
    assert pcfg.project.kind == "pretraining"
    assert is_pretraining_project(pcfg) is True

    # A config that predates the field (no kind key) defaults to venue.
    legacy = cfg_mod.ProjectLayout("legacy")
    legacy.create()
    legacy.config_toml.write_text('[project]\nid = "legacy"\n', encoding="utf-8")
    assert is_pretraining_project(load_config("legacy")) is False


def test_inference_load_keeps_model_shape_frozen_but_tunes_decisions_on_edge(tmp_path, monkeypatch):
    """Model shape ([features]/[model]/hop_ms) always comes from the frozen
    snapshot, while decision/calibration keys the operator explicitly sets in
    the edge config.toml override the bundle — production tuning happens on
    the deployed device."""
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    write_project_config_template(layout, name="Project")

    frozen = load_config()
    frozen.model.pretrained = False
    frozen.features.n_mels = 64                          # what the model was trained on
    frozen.inference.hop_ms = 250
    frozen.inference.consecutive_frames = 1
    frozen.inference.negative_max = 0.37   # calibrated per-model, not the schema default
    dump_config_snapshot(frozen, layout.model_dir("m1") / "training_config.json")

    # Simulate on-site tuning of the edge config.toml AFTER training.
    layout.config_toml.write_text("""
[features]
n_mels = 128
[inference]
hop_ms = 999
consecutive_frames = 9
min_dbfs = -45
smoothing_enabled = false
""")

    cfg = load_config_for_inference(project_id)
    # Model shape stays frozen — a 128-mel input would crash the model.
    assert cfg.features.n_mels == 64
    assert cfg.inference.hop_ms == 250
    # Explicit edge decision keys win.
    assert cfg.inference.consecutive_frames == 9
    assert cfg.inference.min_dbfs == -45
    assert cfg.inference.smoothing_enabled is False
    # Keys absent from the edge file keep the bundle values (template defaults
    # must not leak over calibrated settings).
    assert cfg.inference.negative_max == 0.37


def test_capture_commands_use_project_config_by_default(tmp_path, monkeypatch):
    """Recording/meter checks are part of the next training iteration."""
    from sed import config as cfg_mod
    from sed.cli import _load_capture_config

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()

    frozen = load_config()
    frozen.model.pretrained = False
    frozen.model.sample_rate = 32000
    frozen.audio.record_sample_rate = 48000
    frozen.features.n_mels = 64
    dump_config_snapshot(frozen, layout.model_dir("m1") / "training_config.json")

    layout.config_toml.write_text("""
[model]
pretrained = false
sample_rate = 44100
[features]
n_mels = 128
[audio]
record_sample_rate = 44100
""")

    cfg = _load_capture_config(project_id)

    assert cfg.model.sample_rate == 44100
    assert cfg.audio.record_sample_rate == 44100
    assert cfg.features.n_mels == 128


def test_load_config_reads_project_config_toml(tmp_path, monkeypatch):
    """Project-local config.toml is the user-edited workflow input."""
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    layout.config_toml.write_text("""
[model]
sample_rate = 12345
""")

    cfg = load_config(project_id)

    assert cfg.model.sample_rate == 12345


def test_project_template_lists_negative_as_editable_class(tmp_path, monkeypatch):
    import tomllib

    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = cfg_mod.ProjectLayout("p")
    write_project_config_template(layout, name="Project")

    text = layout.config_toml.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    cfg = load_config("p")

    assert 'target_classes = ["_negative"]' in text
    assert "dataset_selection.json" in text
    assert "sed dataset inspect --project p --serve" in text
    assert "[dataset.sources]" not in text
    assert "[augmentation.combinations]" not in text
    assert "[augmentation]" not in text
    # Kept short deliberately: the template is a starter, not a manual.
    assert len(text.splitlines()) <= 95
    assert "====" not in text
    assert 'verbose = "none" # none | info | live' in text
    assert 'backbone = "mn10_as" # mn01 | mn10_as | mn20_as | mn40_as' in text
    assert 'frontend = "efficientat"' in text
    assert "window_seconds = 0.4" in text
    assert "legacy_torchaudio_db" not in text
    assert (
        'loss = "bce_logits" # bce_logits | cross_entropy/categorical_crossentropy | '
        "asymmetric_focal"
    ) in text
    assert "specaugment_enabled = false" in text
    assert "fmax = 15000" not in text
    assert "hop_ms = 300" not in text
    assert "smoothing_enabled = true" in text
    assert "inference_channels = [0]" in text
    # Read from the schema rather than pinned, so a default change updates the
    # template and this test together instead of failing after the fact.
    from sed.config import Config

    assert f"recording_cooldown_ms = {Config().onset.recording_cooldown_ms}" in text
    assert f"record_seconds = {Config().onset.record_seconds}" in text
    assert "cross_channel_merge_ms = 500" in text
    # The template's inline comment must list every mode the validator accepts,
    # or an operator cannot discover "best" (the mode that keeps a quiet
    # off-axis mic's `_negative` from vetoing another mic's confident fire).
    assert 'multi_mic_mode = "max" # max | best | mean | single' in text
    assert 'min_dbfs = "off"' in text
    assert '"off" or dBFS number' in text
    assert "min_dbfs = null" not in text
    assert parsed["inference"]["min_dbfs"] == "off"
    assert text.index("[dataset]") < text.index("[audio]") < text.index("[model]")
    assert text.index("[training]") < text.index("[inference]")
    assert cfg.dataset.target_classes == ["_negative"]
    assert cfg.model.classes == ["_negative"]
    assert cfg.features.frontend == "efficientat"
    assert cfg.training.loss == "bce_logits"
    assert cfg.training.lr_backbone == cfg.training.lr_head == 1e-4
    assert cfg.training.specaugment_enabled is False
    assert cfg.training.class_balance == "off"
    assert cfg.features.fmax == 15000
    assert cfg.inference.hop_ms == 300
    assert cfg.inference.inference_channels == [0]
    assert cfg.inference.smoothing_enabled is True
    assert cfg.inference.min_dbfs is None
    assert cfg.audio.record_sample_rate == 48000
    assert cfg.onset.min_flux_db == -80
    assert cfg.onset.record_seconds == Config().onset.record_seconds
    assert cfg.onset.randomize_preroll is True
    assert cfg.onset.preroll_ms_range == Config().onset.preroll_ms_range
    assert cfg.onset.recording_cooldown_ms == Config().onset.recording_cooldown_ms
    assert cfg.onset.preroll_ms == Config().onset.preroll_ms
    assert cfg.onset.cross_channel_merge_ms == 500


def test_min_dbfs_accepts_null_and_editor_safe_off(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = cfg_mod.ProjectLayout("p")
    layout.create()

    for value in ('"off"', '"none"', '"null"', "null", "-45.0"):
        layout.config_toml.write_text(f"[inference]\nmin_dbfs = {value}\n", encoding="utf-8")
        cfg = load_config("p")
        expected = -45.0 if value == "-45.0" else None
        assert cfg.inference.min_dbfs == expected


def test_renamed_config_keys_accept_legacy_toml(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = cfg_mod.ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text("""
[audio]
input_sample_rate = 44100
record_channels = 2
inference_channel = 1

[onset]
clip_seconds = 1.25
randomize_window = true
preroll_ms_range = [100, 700]
refractory_ms = 1234
""", encoding="utf-8")

    cfg = load_config("p")

    assert cfg.audio.record_sample_rate == 44100
    assert cfg.inference.inference_channels == [1]
    assert cfg.onset.record_seconds == 1.25
    assert cfg.onset.randomize_preroll is True
    assert cfg.onset.recording_cooldown_ms == 1234


def test_pretrained_contract_accepts_project_template(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = cfg_mod.ProjectLayout("p")
    write_project_config_template(layout, name="Project")

    validate_pretrained_feature_contract(load_config("p"), phase="recording")


def test_pretrained_contract_rejects_recording_frontend_drift(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.cli import _load_capture_config

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    layout.config_toml.write_text("""
[model]
backbone = "mn10_as"
pretrained = true
sample_rate = 32000

[features]
frontend = "legacy_torchaudio_db"
fmax = 15000
""", encoding="utf-8")

    # The legacy frontend is now rejected at config-load time (it was removed
    # entirely), before the recording-phase contract check even runs.
    try:
        _load_capture_config(project_id)
    except Exception as exc:  # noqa: BLE001
        assert "frontend" in str(exc)
        assert "efficientat" in str(exc)
    else:
        assert False, "expected frontend validation error"


def test_dataset_sources_parse_and_sync_classes(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    layout.config_toml.write_text("""
[dataset]
target_classes = ["_negative", "cheers_loud"]

[dataset.sources]
cheers_loud = [{ labels = ["glass_a", "glass_b"], density = "quiet" }]
_negative = [{ labels = ["cutlery"], project_ids = ["p"] }]
""", encoding="utf-8")

    cfg = load_config(project_id)

    assert cfg.model.classes == ["_negative", "cheers_loud"]
    assert cfg.dataset.sources["cheers_loud"][0].labels == ["glass_a", "glass_b"]
    assert cfg.dataset.sources["_negative"][0].project_ids == ["p"]


def test_augmentation_combinations_parse_and_sync_targets(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    layout.config_toml.write_text("""
[dataset]
target_classes = ["_negative", "cheers_quiet"]

[augmentation.combinations]
cheers_quiet = [
  { base = "cheers_quiet", overlay = "cutlery", description = "quiet with cutlery", tags = "cutlery,quiet", mask_min_mean_db = 10.0 },
]
_negative = [
  { base = "room_tone", overlay = "cutlery", notes = "room and cutlery", tags = ["negative"] },
]
""", encoding="utf-8")

    cfg = load_config(project_id)

    quiet = cfg.augmentation.combinations["cheers_quiet"][0]
    negative = cfg.augmentation.combinations["_negative"][0]
    assert quiet.base == "cheers_quiet"
    assert quiet.overlay == "cutlery"
    assert quiet.description == "quiet with cutlery"
    assert quiet.tags == ["cutlery", "quiet"]
    assert quiet.mask_min_mean_db == 10.0
    assert negative.description == "room and cutlery"
    assert negative.tags == ["negative"]


def test_legacy_augmentation_targets_normalize_to_combinations(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    layout.config_toml.write_text("""
[dataset]
target_classes = ["cheers_quiet"]

[augmentation.targets.cheers_quiet]
combinations = [
  { base = "quiet_take_a", overlay = "cutlery" },
]
""", encoding="utf-8")

    cfg = load_config(project_id)

    assert cfg.augmentation.combinations["cheers_quiet"][0].base == "quiet_take_a"


def test_legacy_dataset_aliases_normalize_to_sources(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    layout.config_toml.write_text("""
[dataset]
target_classes = ["_negative", "cling"]

[dataset.project]
cling = [{ category = "positives", labels = ["legacy_cling"] }]
_negative = [{ category = "negatives", label = "legacy_room" }]
""", encoding="utf-8")

    cfg = load_config(project_id)

    assert cfg.dataset.sources["cling"][0].labels == ["legacy_cling"]
    assert cfg.dataset.sources["_negative"][0].label == "legacy_room"


def test_target_classes_do_not_auto_inject_negative(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    layout.config_toml.write_text("""
[dataset]
target_classes = ["cling_a", "cling_b", "cling_c"]
""")

    cfg = load_config(project_id)

    assert cfg.dataset.target_classes == ["cling_a", "cling_b", "cling_c"]
    assert cfg.model.classes == ["cling_a", "cling_b", "cling_c"]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        # Nothing set: score the shared benchmark rule.
        ({}, ["fixed@0.90"]),
        # Pre-rule spellings still mean what they always meant.
        ({"auto_test": False}, []),
        ({"auto_test_threshold": 0.8}, ["fixed@0.80"]),
        # Calibration on ⇒ the bundle's own (calibrated) decision layer.
        ({"calibrate_thresholds": True}, ["deployed"]),
        # An explicit list wins over every legacy key.
        ({"auto_test_rules": ["argmax"], "auto_test": False}, ["argmax"]),
        # "no rules" is a real answer, not "fall back to the default".
        ({"auto_test_rules": []}, []),
    ],
)
def test_resolved_auto_test_rules_bridges_the_pre_rule_config_keys(overrides, expected):
    """Older config.toml files say the same thing in the old spelling; they must
    keep scoring what they always did rather than silently stopping."""
    from sed.config import TrainingCfg

    assert TrainingCfg(**overrides).resolved_auto_test_rules() == expected


def test_auto_test_rules_are_validated_and_normalized():
    """A rule id the evaluator would reject must never reach config.toml, and
    "fixed@0.9" and "fixed@0.90" are the same column band, not two."""
    from pydantic import ValidationError

    from sed.config import TrainingCfg

    cfg = TrainingCfg(auto_test_rules=["fixed@0.9", "argmax", "fixed@0.90"])
    assert cfg.auto_test_rules == ["fixed@0.90", "argmax"]
    with pytest.raises(ValidationError):
        TrainingCfg(auto_test_rules=["coin_flip"])


def test_fp_hour_calibration_is_config_not_a_session_request():
    """The FP/h sweep used to live only in the browser request, so `sed train`
    could not run it and the checkbox reset on reload."""
    from sed.config import TrainingCfg

    cfg = TrainingCfg(calibrate_after=True, calibrate_per_1000=2.0, calibrate_negsrc="library")
    assert (cfg.calibrate_after, cfg.calibrate_per_1000, cfg.calibrate_negsrc) == (
        True, 2.0, "library")
    with pytest.raises(Exception):
        TrainingCfg(calibrate_negsrc="wherever")


def test_runtime_settings_apply_over_training_snapshot(tmp_path, monkeypatch):
    """Generated runtime_settings.json carries model-scoped runtime knobs."""
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    write_project_config_template(layout, name="Project")

    base = load_config()
    base.artnet.target_ip = "192.168.1.1"
    base.features.frontend = "efficientat"
    base.features.fmax = 15000
    base.model.sample_rate = 32000
    dump_config_snapshot(base, layout.model_dir("m1") / "training_config.json")

    dump_runtime_settings({
        "artnet": {"target_ip": "10.0.0.99"},  # site-local: must be ignored here
        "model": {"sample_rate": 44100},        # model-shape: must NOT overlay
        "thresholds": {"cheers_loud": 0.77},
        "logging": {"level": "DEBUG"},          # site-local: must be ignored here
        "mining": {"enabled": False},           # site-local: must be ignored here
        "inference": {
            "hop_ms": 200,
            "default_threshold": 0.9,
            "consecutive_frames": 3,
            "smoothing_ema_alpha": 0.99,        # not in runtime allow-list
        },
    }, layout.model_dir("m1") / "runtime_settings.json")
    layout.config_toml.write_text("""
[audio]
record_channels = 2
[inference]
inference_channels = [1]
[logging]
level = "WARNING"
[mining]
enabled = true
output_dir = "edge_mined"
[deploy]
verbose = "info"
""", encoding="utf-8")
    cfg = load_config_for_inference(project_id)
    # ArtNet is site-local: it comes from the edge config.toml (the template
    # defaults to off/unset), never from the model bundle or runtime settings.
    assert cfg.artnet.enabled is False
    assert cfg.artnet.target_ip == ""
    assert cfg.logging.level == "WARNING"
    assert cfg.mining.enabled is True
    assert cfg.mining.output_dir == "edge_mined"
    assert cfg.deploy.verbose == "info"
    assert cfg.audio.record_channels == [0, 1]  # legacy int count -> ordered indexes
    assert cfg.inference.inference_channels == [1]
    # Model-shape (sample_rate, in [model]) is frozen — runtime_settings can't change it.
    assert cfg.model.sample_rate == 32000
    # Tunable knobs (thresholds) still overlay from runtime_settings.
    assert cfg.thresholds["cheers_loud"] == 0.77
    assert cfg.inference.hop_ms == 200
    assert cfg.inference.default_threshold == 0.9
    assert cfg.inference.consecutive_frames == 3
    assert cfg.inference.smoothing_ema_alpha == base.inference.smoothing_ema_alpha


def test_inference_load_requires_selected_model_snapshot(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "no_snapshot_yet"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    write_project_config_template(layout, name="Project")
    try:
        load_config_for_inference(project_id)
        assert False, "expected missing model snapshot"
    except FileNotFoundError as exc:
        assert "No model" in str(exc)
        assert "models select" in str(exc)
