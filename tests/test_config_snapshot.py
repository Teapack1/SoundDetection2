"""Tests for the config snapshot mechanism — the safeguard that decouples
inference from edits to default.toml made after a model was trained.

The contract we want:
  1. dump_config_snapshot(cfg, p)  + load_config_snapshot(p)  → round-trips
  2. load_config_for_inference(id):
       - if models/training_config.json exists → uses it (NOT live default)
       - if it doesn't → falls back to live load_config
       - either way: operational settings (artnet, thresholds, logging, and
         hardware audio fields) are RE-applied from project.toml so they can
         be changed per-deployment without retraining
"""

from __future__ import annotations

import json
from pathlib import Path

from sed.config import (
    Config,
    dump_config_snapshot,
    load_config,
    load_config_for_inference,
    load_config_snapshot,
)


def test_snapshot_round_trip(tmp_path):
    cfg = load_config()
    p = tmp_path / "snapshot.json"
    dump_config_snapshot(cfg, p)
    cfg2 = load_config_snapshot(p)
    assert cfg.audio.sample_rate == cfg2.audio.sample_rate
    assert cfg.features.n_mels == cfg2.features.n_mels
    assert cfg.inference.consecutive_frames == cfg2.inference.consecutive_frames
    assert cfg.training.focal_gamma_neg == cfg2.training.focal_gamma_neg


def test_inference_load_uses_snapshot_over_live(tmp_path, monkeypatch):
    """The whole point: when training_config.json exists, edits to the live
    default.toml must NOT affect inference behaviour."""
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()

    # Build a config and tweak something model-shape-related (n_mels), then
    # snapshot it.
    frozen = load_config()
    frozen.features.n_mels = 64                          # what the model was trained on
    frozen.inference.consecutive_frames = 1
    dump_config_snapshot(frozen, layout.models_dir / "training_config.json")

    # Simulate someone editing the live default.toml AFTER training: write
    # a fake "live" default that would, if used naively, change n_mels.
    live_default = tmp_path / "default.toml"
    live_default.write_text("""
[features]
n_mels = 128
[inference]
consecutive_frames = 9
""")
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG", live_default)

    cfg = load_config_for_inference(project_id)
    # Snapshot wins — the model would crash on a 128-mel input it was never trained on
    assert cfg.features.n_mels == 64
    assert cfg.inference.consecutive_frames == 1


def test_operational_overrides_still_apply_from_project_toml(tmp_path, monkeypatch):
    """project.toml [artnet], [audio], [thresholds] etc. are operationally
    tunable per-deployment — they must still be honoured after the snapshot
    takes precedence for everything else."""
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "p"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()

    base = load_config()
    base.artnet.target_ip = "192.168.1.1"                # value at training time
    base.audio.sample_rate = 32000                       # model/feature rate
    base.audio.input_sample_rate = 48000                 # hardware rate at training time
    dump_config_snapshot(base, layout.models_dir / "training_config.json")

    # The deployment site has a different lighting console IP and hardware rate.
    # It must not be able to change the frozen model/feature sample rate.
    layout.project_toml.write_text("""
[project]
id = "p"
name = "p"

[artnet]
target_ip = "10.0.0.99"

[audio]
sample_rate = 44100
input_sample_rate = 44100
""")
    cfg = load_config_for_inference(project_id)
    assert cfg.artnet.target_ip == "10.0.0.99"           # operational override wins
    assert cfg.audio.input_sample_rate == 44100          # hardware override wins
    assert cfg.audio.sample_rate == 32000                # model rate stays frozen


def test_inference_load_falls_back_to_live_when_no_snapshot(tmp_path, monkeypatch):
    """Before the first train/export run, models/training_config.json doesn't
    exist. Inference should still work (against the live config)."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    project_id = "no_snapshot_yet"
    layout = cfg_mod.ProjectLayout(project_id)
    layout.create()
    # No training_config.json on disk
    cfg = load_config_for_inference(project_id)
    assert isinstance(cfg, Config)                       # didn't raise
    # Got the live default value
    assert cfg.features.n_mels == load_config().features.n_mels
