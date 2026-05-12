"""Tests for calibration negative-source selection and threshold persistence."""

import numpy as np
import soundfile as sf

from sed.runtime.calibrate import _gather_negative_files, persist_calibrated_thresholds


def test_gather_negative_files_uses_library_negative_mapping(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.config import Config, ProjectLayout
    from sed.data import library as lib_mod
    from sed.data.compose import write_dataset_selection
    from sed.runtime import calibrate as cal_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(cal_mod, "LIBRARY_DIR", tmp_path / "data_library")
    layout = ProjectLayout("p")
    layout.create()
    sound_dir = tmp_path / "data_library" / "sounds" / "room"
    sound_dir.mkdir(parents=True)
    wav = sound_dir / "room.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    wav.with_suffix(".json").write_text(
        '{"schema_version":2,"uuid":"u","category":"sounds","label":"room",'
        '"project_id":"p","dataset_role":"train","density":"normal",'
        '"review_status":"accepted","sample_rate":32000,"channels":1,"duration_s":1.0}',
        encoding="utf-8",
    )
    from sed.data.library import Library
    Library().rebuild_manifest()
    write_dataset_selection(layout, {"assignments": {"_negative": [{"label": "room"}]}})
    cfg = Config()
    cfg.model.classes = ["_negative"]

    assert _gather_negative_files(layout, "venue", None, cfg) == [wav.resolve()]


def test_gather_negative_files_accepts_explicit_negative_dir(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.config import ProjectLayout

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    layout = ProjectLayout("p")
    layout.create()
    explicit = tmp_path / "custom_negatives"
    explicit.mkdir()
    wav = explicit / "custom.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)

    assert _gather_negative_files(layout, "venue", explicit) == [wav]


def test_gather_negative_files_uses_library_negatives(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.config import ProjectLayout
    from sed.runtime import calibrate as cal_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cal_mod, "LIBRARY_DIR", tmp_path / "data_library")
    layout = ProjectLayout("p")
    layout.create()
    neg_dir = tmp_path / "data_library" / "sounds" / "busy"
    neg_dir.mkdir(parents=True)
    wav = neg_dir / "busy.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)

    assert _gather_negative_files(layout, "library", None) == [wav]


def test_calibration_persists_thresholds_to_runtime_settings(tmp_path, monkeypatch):
    import json
    from sed import config as cfg_mod
    from sed.config import Config, ProjectLayout, dump_runtime_settings

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("p")
    layout.create()
    mid = "2026-05-15_1000-m1"
    # The active model subfolder (resolved by single-model fallback).
    model_dir = layout.model_dir(mid)
    model_dir.mkdir(parents=True)
    (model_dir / "training_config.json").write_text("{}", encoding="utf-8")
    dump_runtime_settings({"thresholds": {"old": 0.1}}, model_dir / "runtime_settings.json")

    out = persist_calibrated_thresholds(Config(), "p", {"cheers_loud": 0.73})

    assert out == model_dir / "runtime_settings.json"
    selected = json.loads(out.read_text(encoding="utf-8"))
    # Calibration merges into the active model's runtime_settings, preserving
    # earlier keys and adding the freshly calibrated thresholds.
    assert selected["settings"]["thresholds"]["old"] == 0.1
    assert selected["settings"]["thresholds"]["cheers_loud"] == 0.73
