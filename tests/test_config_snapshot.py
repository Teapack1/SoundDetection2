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
    p = tmp_path / "snapshot.json"
    dump_config_snapshot(cfg, p)
    cfg2 = load_config_snapshot(p)
    assert cfg.model.sample_rate == cfg2.model.sample_rate
    assert cfg.features.n_mels == cfg2.features.n_mels
    assert cfg.inference.consecutive_frames == cfg2.inference.consecutive_frames
    assert cfg.training.focal_gamma_neg == cfg2.training.focal_gamma_neg


def test_inference_load_uses_snapshot_over_live(tmp_path, monkeypatch):
    """The whole point: when training_config.json exists, edits to the live
    project config.toml must NOT affect model-shape or frozen runtime behaviour
    unless a generated runtime_settings.json explicitly overlays a tunable key."""
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    write_project_config_template(layout, name="Project")

    # Build a config and tweak something model-shape-related (n_mels), then
    # snapshot it into a model bundle subfolder.
    frozen = load_config()
    frozen.model.pretrained = False
    frozen.features.n_mels = 64                          # what the model was trained on
    frozen.inference.consecutive_frames = 1
    dump_config_snapshot(frozen, layout.model_dir("m1") / "training_config.json")

    # Simulate someone editing the live project config.toml AFTER training.
    layout.config_toml.write_text("""
[features]
n_mels = 128
[inference]
consecutive_frames = 9
""")

    cfg = load_config_for_inference(project_id)
    # Snapshot wins — the model would crash on a 128-mel input it was never trained on.
    # Live config edits alone do not override frozen inference settings.
    assert cfg.features.n_mels == 64
    assert cfg.inference.consecutive_frames == 1


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
    assert len(text.splitlines()) < 90
    assert "====" not in text
    assert 'verbose = "none" # none | info | live' in text
    assert 'backbone = "mn10_as" # mn01 | mn10_as | mn20_as | mn40_as' in text
    assert 'frontend = "efficientat"' in text
    assert "legacy_torchaudio_db" not in text
    assert "cross_entropy | asymmetric_focal | bce_logits" in text
    assert "fmax = 15000" not in text
    assert "hop_ms = 300" not in text
    assert "smoothing_enabled = true" in text
    assert "inference_channels = [0]" in text
    assert "recording_cooldown_ms = 1500" in text
    assert "cross_channel_merge_ms = 500" in text
    assert 'multi_mic_mode = "max" # max | mean | single' in text
    assert 'min_dbfs = "off"' in text
    assert '"off" or dBFS number' in text
    assert "min_dbfs = null" not in text
    assert parsed["inference"]["min_dbfs"] == "off"
    assert text.index("[dataset]") < text.index("[audio]") < text.index("[model]")
    assert text.index("[training]") < text.index("[inference]")
    assert cfg.dataset.target_classes == ["_negative"]
    assert cfg.model.classes == ["_negative"]
    assert cfg.features.frontend == "efficientat"
    assert cfg.features.fmax == 15000
    assert cfg.inference.hop_ms == 300
    assert cfg.inference.inference_channels == [0]
    assert cfg.inference.smoothing_enabled is True
    assert cfg.inference.min_dbfs is None
    assert cfg.audio.record_sample_rate == 48000
    assert cfg.onset.min_flux_db == -80
    assert cfg.onset.record_seconds == 1.0
    assert cfg.onset.randomize_preroll is True
    assert cfg.onset.preroll_ms_range == (100, 700)
    assert cfg.onset.recording_cooldown_ms == 1500
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
