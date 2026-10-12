"""Tests for calibration negative-source selection and threshold persistence."""

import numpy as np
import pytest
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


def test_calibration_records_simulation_mode(tmp_path, monkeypatch):
    import json
    from sed import config as cfg_mod
    from sed.config import Config, ProjectLayout

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("p")
    layout.create()
    model_dir = layout.model_dir("m1")
    model_dir.mkdir(parents=True)
    (model_dir / "training_config.json").write_text("{}", encoding="utf-8")

    out = persist_calibrated_thresholds(
        Config(), "p", {"cheers_loud": 0.73},
        simulation_mode="clip_batch", target_per_1000=1.0,
    )
    selected = json.loads(out.read_text(encoding="utf-8"))
    assert selected["settings"]["calibration"] == {
        "mode": "fp_budget",
        "simulation_mode": "clip_batch",
        "target_per_1000": 1.0,
    }


def test_calibration_records_achieved_rates(tmp_path, monkeypatch):
    import json
    from sed import config as cfg_mod
    from sed.config import Config, ProjectLayout

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("p")
    layout.create()
    model_dir = layout.model_dir("m1")
    model_dir.mkdir(parents=True)
    (model_dir / "training_config.json").write_text("{}", encoding="utf-8")

    out = persist_calibrated_thresholds(
        Config(), "p", {"cheers_loud": 0.73},
        simulation_mode="clip_batch", target_per_1000=1.0,
        achieved_per_1000={"cheers_loud": 0.4},
        clips_scored=1234, negative_source="venue",
    )
    calib = json.loads(out.read_text(encoding="utf-8"))["settings"]["calibration"]
    assert calib["mode"] == "fp_budget"
    assert calib["achieved_per_1000"] == {"cheers_loud": 0.4}
    assert calib["max_per_1000"] == 0.4
    assert calib["clips_scored"] == 1234
    assert calib["negative_source"] == "venue"


def test_onset_gated_and_sliding_simulations_differ():
    """The sweep must simulate the deployed decision dynamics: with smoothing
    plus multi-frame confirmation a single spike never triggers, while the
    onset-gated single-window decision fires on it."""
    from sed.runtime.smooth import TriggerArbiter

    classes = ["_negative", "cheers_loud"]
    spike = [0.1, 0.1, 0.95, 0.1, 0.1]

    def run(*, onset_gated: bool) -> int:
        arbiter = TriggerArbiter(
            classes=classes,
            thresholds={"cheers_loud": 0.9, "_negative": 1.5},
            smoothers_cfg={
                "median_window": 3,
                "ema_alpha": 0.4,
                "default_threshold": 0.9,
                "hysteresis_low": 0.5,
                "consecutive_frames": 1 if onset_gated else 3,
                "smoothing_enabled": not onset_gated,
            },
            negative_classes={"_negative"},
            vetoes={"_negative": 0.5},
            margin_to_second_best=0.08,
            global_refractory_ms=0,
        )
        triggers = 0
        now = 0.0
        for p in spike:
            ev = arbiter.step({"cheers_loud": p, "_negative": 0.05}, now=now)
            if ev is not None:
                triggers += 1
            now += 0.3
        return triggers

    assert run(onset_gated=True) == 1
    assert run(onset_gated=False) == 0


# ---- the sweep itself -------------------------------------------------------
# The tests above cover only threshold persistence, which is how six defects in
# the sweep survived. These exercise `calibrate()` end to end against a stub
# backend.


class _ScriptedBackend:
    """Returns a scripted probability row per scored window."""

    engine = "stub"
    path = "stub"

    def __init__(self, rows):
        self.rows = list(rows)
        self.i = 0

    def infer(self, batch):
        out = np.zeros((batch.shape[0], len(self.rows[0])), dtype=np.float32)
        for b in range(batch.shape[0]):
            out[b] = self.rows[min(self.i, len(self.rows) - 1)]
            self.i += 1
        return out


def _sweep_project(tmp_path, monkeypatch, *, rows, classes, seconds=8.0, n_files=1):
    """A project whose negatives replay into `rows`, one row per window."""
    from sed import config as cfg_mod
    from sed.config import Config, ProjectLayout
    from sed.runtime import calibrate as cal_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    layout = ProjectLayout("p")
    layout.create()

    neg_dir = tmp_path / "negs"
    neg_dir.mkdir()
    sr = 32000
    rng = np.random.default_rng(0)
    for i in range(n_files):
        sf.write(str(neg_dir / f"n{i}.wav"),
                 (rng.standard_normal(int(sr * seconds)) * 0.05).astype(np.float32), sr)

    cfg = Config()
    cfg.model.sample_rate = sr
    cfg.model.classes = list(classes)
    cfg.dataset.target_classes = list(classes)
    cfg.audio.record_sample_rate = sr
    cfg.audio.blocksize = 1024
    cfg.features.window_seconds = 0.4
    cfg.inference.inference_channels = [0]
    # Sliding mode gives a deterministic hop grid; consensus/smoothing off so a
    # single scripted window is a single decision (what onset gating does live).
    cfg.inference.onset_gating_enabled = False
    cfg.inference.hop_ms = 1000
    cfg.inference.consecutive_frames = 1
    cfg.inference.smoothing_enabled = False
    cfg.inference.trigger_refractory_ms = 0
    cfg.inference.negative_max = 0.5
    cfg.inference.margin_to_second_best = 0.0
    cfg.inference.min_dbfs = None

    backend = _ScriptedBackend(rows)
    monkeypatch.setattr(cal_mod, "build_feature_extractor", lambda _c: (lambda w: np.zeros((1, 8, 4), dtype=np.float32)))
    monkeypatch.setattr(
        "sed.runtime.backends.build_inference_backend",
        lambda *a, **k: backend,
    )
    monkeypatch.setattr(cal_mod, "resolve_active_model_dir", lambda *a, **k: layout.model_dir("m"))
    (layout.model_dir("m")).mkdir(parents=True, exist_ok=True)
    return cfg, neg_dir


def test_sweep_reports_the_system_rate_not_the_per_class_maximum(tmp_path, monkeypatch):
    """Two classes each firing once must be charged as TWO false alarms.

    The old sweep pinned every other class at 1.5 and stored max() of the
    per-class rates, so a system firing twice reported the rate of one.
    """
    classes = ["_negative", "a", "b"]
    # Four negative clips: one fires class a, one fires class b, two are silent.
    rows = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    cfg, neg_dir = _sweep_project(
        tmp_path, monkeypatch, rows=rows, classes=classes, n_files=4)

    from sed.runtime.calibrate import calibrate

    # A budget far above anything observed, so the loosest thresholds are kept
    # and the reported rate reflects every fire.
    result = calibrate(cfg, "p", negative_dir=neg_dir, target_per_1000=1e9)

    per_class = result["achieved_per_1000"]
    assert per_class["a"] > 0 and per_class["b"] > 0
    assert result["system_per_1000"] == pytest.approx(per_class["a"] + per_class["b"])
    assert result["system_per_1000"] > max(per_class.values()), "max() under-reports"
    # Two false fires in four clips == 500 per 1000.
    assert result["system_per_1000"] == pytest.approx(500.0)


def test_every_clip_counts_as_one_decision_regardless_of_its_length(tmp_path, monkeypatch):
    """A clip is one trigger opportunity, so the denominator is clips — not
    seconds, and not windows. A 0.2 s clip and a 4 s clip each count once."""
    classes = ["_negative", "a"]
    cfg, neg_dir = _sweep_project(
        tmp_path, monkeypatch, rows=[[1.0, 0.0]] * 3, classes=classes,
        seconds=4.0, n_files=3)
    # Replace one clip with a very short one; it must still be scored.
    import soundfile as sf_mod
    rng = np.random.default_rng(1)
    sf_mod.write(str(neg_dir / "n0.wav"),
                 (rng.standard_normal(32000 // 5) * 0.05).astype(np.float32), 32000)

    from sed.runtime.calibrate import calibrate

    result = calibrate(cfg, "p", negative_dir=neg_dir, target_per_1000=1e9)
    assert result["clips_scored"] == 3


def test_sweep_records_the_replay_substrate_and_system_rate(tmp_path, monkeypatch):
    """`simulation_mode` must distinguish a replay-based sweep from the old one,
    so a stale bundle is not mistaken for a corrected calibration."""
    classes = ["_negative", "a"]
    cfg, neg_dir = _sweep_project(
        tmp_path, monkeypatch, rows=[[1.0, 0.0]] * 8, classes=classes)

    from sed.runtime.calibrate import calibrate

    result = calibrate(cfg, "p", negative_dir=neg_dir, target_per_1000=1.0)
    assert result["simulation_mode"] == "clip_batch"
    assert "system_per_1000" in result
