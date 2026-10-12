from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

import numpy as np
from rich.console import Console

from sed.audio.features import feature_frame_count_for_cfg
from sed.config import Config, ProjectLayout
from sed.runtime.infer import (
    RealtimeInference,
    RuntimeDetection,
    _below_min_dbfs,
    _linear_to_db,
    _validate_openvino_input_shape,
    normalize_openvino_logits,
    pool_mic_probabilities,
)
from sed.runtime.smooth import TriggerArbiter, TriggerEvent
from sed.runtime.mining import DetectionMiner
from sed.runtime.run import (
    _LiveState,
    _build_live_table,
    _format_heartbeat,
    _format_trigger_line,
    _log_verbose_frame,
    _parse_target_csv,
    _resolve_log_level,
    _resolve_verbose_level,
    _startup_lines,
)


def test_parse_target_csv_accepts_comma_and_repeated_values():
    assert _parse_target_csv("cheers_loud, cheers_quiet") == [
        "cheers_loud",
        "cheers_quiet",
    ]
    assert _parse_target_csv(["cheers_loud", "cheers_medium, cheers_quiet"]) == [
        "cheers_loud",
        "cheers_medium",
        "cheers_quiet",
    ]
    assert _parse_target_csv(" , ") is None


def test_min_dbfs_gate_uses_peak_amplitude():
    assert _below_min_dbfs(-50.0, -45.0) is True
    assert _below_min_dbfs(-44.9, -45.0) is False
    assert _below_min_dbfs(-90.0, None) is False


def test_openvino_input_shape_validation_accepts_matching_dynamic_batch():
    cfg = Config()
    cfg.features.frontend = "efficientat"
    cfg.features.fmax = 15000

    # Derived, not hard-coded: the expected width follows features.window_seconds,
    # so changing the project default cannot silently invalidate this check.
    frames = feature_frame_count_for_cfg(cfg)
    _validate_openvino_input_shape([-1, 1, 128, frames], cfg)


def test_openvino_input_shape_validation_rejects_snapshot_drift():
    cfg = Config()
    cfg.features.frontend = "efficientat"
    cfg.features.fmax = 15000

    try:
        _validate_openvino_input_shape(
            [-1, 1, 64, feature_frame_count_for_cfg(cfg)], cfg)
    except RuntimeError as exc:
        assert "input shape does not match" in str(exc)
        assert "feature bands=64" in str(exc)
    else:
        assert False, "expected OpenVINO input shape mismatch"


def _cheers_arbiter(classes: list[str]) -> TriggerArbiter:
    # Smoothing off so a single window's raw probability drives the decision —
    # this isolates the pooling/gating change from the median/EMA buildup.
    return TriggerArbiter(
        classes=classes,
        thresholds={},
        smoothers_cfg={
            "median_window": 1,
            "ema_alpha": 1.0,
            "default_threshold": 0.9,
            "hysteresis_low": 0.3,
            "consecutive_frames": 1,
            "smoothing_enabled": False,
        },
        negative_classes={"_negative"},
        vetoes={"_negative": 0.5},
        margin_to_second_best=0.08,
        global_refractory_ms=0,
    )


def test_pool_mic_probabilities_best_uses_winning_mic_whole_vector():
    # mic 0 clearly heard a cheer; mics 1-2 mostly heard room tone (_negative).
    # Class order: [_negative, cheers_quiet, cheers_loud, cheers_medium].
    probs = np.array(
        [
            [0.03, 0.01, 0.01, 0.95],
            [0.85, 0.05, 0.05, 0.05],
            [0.80, 0.05, 0.05, 0.10],
        ],
        dtype=np.float32,
    )

    best = pool_mic_probabilities(probs, "best", negative_idx={0})
    # "best" returns mic 0's self-consistent distribution — low _negative.
    np.testing.assert_allclose(best, probs[0])
    assert best[0] < 0.5

    # "max" lets the worst mic's _negative dominate the pooled vector.
    pooled_max = pool_mic_probabilities(probs, "max", negative_idx={0})
    assert pooled_max[0] > 0.5
    np.testing.assert_allclose(pooled_max[3], 0.95, atol=1e-6)


def test_best_pooling_fires_where_max_pooling_is_blocked_by_negative_gate():
    classes = ["_negative", "cheers_quiet", "cheers_loud", "cheers_medium"]
    probs = np.array(
        [
            [0.03, 0.01, 0.01, 0.95],  # close mic — confident cheers_medium
            [0.85, 0.05, 0.05, 0.05],  # far mic — sounds like ambience
            [0.80, 0.05, 0.05, 0.10],  # far mic — sounds like ambience
        ],
        dtype=np.float32,
    )

    def _decide(mode: str):
        pooled = pool_mic_probabilities(probs, mode, negative_idx={0})
        prob_dict = {c: float(p) for c, p in zip(classes, pooled)}
        return _cheers_arbiter(classes).step(prob_dict)

    best_ev = _decide("best")
    assert best_ev is not None
    assert best_ev.class_name == "cheers_medium"

    # Same audio, "max" pooling: the off-axis mics' _negative crosses the gate.
    assert _decide("max") is None


def test_mining_threshold_fallback_order():
    cfg = Config()
    cfg.inference.default_threshold = 0.61
    cfg.thresholds = {"cheers_quiet": 0.66, "cheers_loud": 0.77}
    cfg.mining.per_class_min_confidence = {"cheers_loud": 0.88}

    runtime = object.__new__(RealtimeInference)
    runtime.cfg = cfg

    assert runtime._mining_threshold("cheers_loud") == 0.88
    assert runtime._mining_threshold("cheers_quiet") == 0.66
    assert runtime._mining_threshold("cheers_medium") == 0.61


def test_verbose_level_defaults_to_deploy_config_and_allows_cli_override():
    cfg = Config()
    cfg.deploy.verbose = "info"

    assert _resolve_verbose_level(cfg, None) == "info"
    assert _resolve_verbose_level(cfg, "live") == "live"
    assert _resolve_verbose_level(cfg, "none") == "none"


def test_log_level_uses_config_only_when_verbose_enabled():
    cfg = Config()
    cfg.logging.level = "DEBUG"

    assert _resolve_log_level(cfg, verbose=True) == logging.DEBUG
    assert _resolve_log_level(cfg, verbose=False) == logging.WARNING

    cfg.logging.level = "ERROR"
    assert _resolve_log_level(cfg, verbose=False) == logging.ERROR


def test_inference_channels_accept_selected_zero_based_indexes():
    cfg = Config.model_validate({
        "audio": {"record_channels": 2},
        "inference": {"inference_channels": [0, 1], "multi_mic_mode": "max"},
    })

    assert cfg.inference.inference_channels == [0, 1]


def test_inference_channels_reject_invalid_indexes():
    # record_channels and inference_channels are independent physical selections,
    # so an inference index beyond the recorded set is allowed; only genuinely
    # invalid indexes (e.g. negative) are rejected.
    try:
        Config.model_validate({
            "audio": {"record_channels": 2},
            "inference": {"inference_channels": [0, -1]},
        })
    except ValueError as exc:
        assert "inference.inference_channels" in str(exc)
    else:
        assert False, "expected invalid inference channel"


def test_single_mic_mode_requires_exactly_one_inference_channel():
    try:
        Config.model_validate({
            "audio": {"record_channels": 2},
            "inference": {"inference_channels": [0, 1], "multi_mic_mode": "single"},
        })
    except ValueError as exc:
        assert "requires exactly one inference channel" in str(exc)
    else:
        assert False, "expected single-channel validation error"


def test_detection_miner_sidecar_includes_all_probability_fields(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")
    layout = ProjectLayout("p")
    layout.create()

    cfg = Config()
    cfg.deploy.active_model = "run_123"
    cfg.model.classes = ["_negative", "cheers_loud", "cheers_quiet"]
    cfg.mining.output_dir = "mined"
    miner = DetectionMiner(cfg=cfg, project_id="p", classes={"cheers_loud"})

    det = RuntimeDetection(
        class_name="cheers_loud",
        confidence=0.91,
        timestamp=123.0,
        kind="threshold",
        probs={"_negative": 0.05, "cheers_loud": 0.91, "cheers_quiet": 0.12},
        smoothed={"_negative": 0.04, "cheers_loud": 0.87, "cheers_quiet": 0.10},
        audio=np.zeros((3200, 1), dtype="float32"),
        sample_rate=32000,
        active_channels=[0],
        best_channel=0,
        peak_dbfs=-12.5,
        rms_dbfs=-30.0,
        min_dbfs=-45.0,
        active_targets=["cheers_loud"],
    )

    wav = miner.maybe_save(det)

    assert wav is not None
    assert wav.parent == tmp_path / "data_library" / "mined" / "cheers_loud"
    meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["category"] == "mined"
    assert meta["project_id"] == "p"
    # Mined clips carry no split: a source rule decides where they compose.
    assert "dataset_role" not in meta
    assert meta["review_status"] == "accepted"
    assert meta["session_id"].startswith("inference-")
    assert meta["recording_counter"] == 1
    assert meta["probs_percent"]["cheers_loud"] == 91.0
    assert meta["smoothed_percent"]["cheers_loud"] == 87.0
    assert meta["peak_dbfs"] == -12.5
    assert meta["rms_dbfs"] == -30.0
    assert meta["min_dbfs"] == -45.0
    assert meta["active_targets"] == ["cheers_loud"]
    assert meta["active_model"] == "run_123"
    assert "review_hint" not in meta


def test_inference_help_documents_new_flags_and_hides_removed_flags():
    from typer.testing import CliRunner

    from sed.cli import app

    result = CliRunner().invoke(app, ["inference", "--help"])

    assert result.exit_code == 0
    assert "--target" in result.output
    assert "cheers_loud,cheers_quiet" in result.output
    assert "--verbose" in result.output
    assert "--device" in result.output
    assert "--min-dbfs" in result.output
    assert "--mine" in result.output
    for removed in [
        "--no-live",
        "--silence-gate",
        "--no-silence-gate",
        "--silence-dbfs",
        "--no-mine",
        "--mine-class",
        "--mine-dir",
        "--mine-min-confidence",
        "--mine-min-interval",
    ]:
        assert removed not in result.output


def test_run_is_hidden_compatibility_alias():
    from typer.testing import CliRunner

    from sed.cli import app

    help_result = CliRunner().invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "inference" in help_result.output
    assert "run " not in help_result.output

    alias_result = CliRunner().invoke(app, ["run", "--help"])
    assert alias_result.exit_code == 0
    assert "--target" in alias_result.output


def test_inference_help_shows_project_runtime_defaults(tmp_path, monkeypatch):
    import sys

    from typer.testing import CliRunner

    from sed import config as cfg_mod
    from sed.cli import app

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(
        """
[deploy]
active_model = "m1"
record_device = "UMC404HD"
verbose = "info"

[audio]
record_sample_rate = 48000
record_channels = [0, 1]

[inference]
inference_channels = [1]

[artnet]
enabled = true
target_ip = "192.168.0.50"

[mining]
enabled = false
""",
        encoding="utf-8",
    )
    snapshot = Config()
    snapshot.model.classes = ["_negative", "bell", "clap"]
    snapshot.dataset.target_classes = ["_negative", "bell", "clap"]
    snapshot.dataset.classes = ["_negative", "bell", "clap"]
    snapshot.targets = {"bell": cfg_mod.TargetCfg(enabled=True)}
    model_dir = layout.model_dir("m1")
    cfg_mod.dump_config_snapshot(snapshot, model_dir / "training_config.json")
    (model_dir / "model_fp16.xml").write_text("ir", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["sed", "inference", "--project", "p", "--verbose", "live", "--no-artnet", "--help"],
    )

    result = CliRunner().invoke(
        app,
        ["inference", "--project", "p", "--verbose", "live", "--no-artnet", "--help"],
    )

    assert result.exit_code == 0
    assert "Resolved runtime defaults (project p)" in result.output
    assert "active_model: m1" in result.output
    assert "device='UMC404HD'" in result.output
    assert "inference_channels=1" in result.output
    assert "active_targets:" in result.output
    assert "bell" in result.output
    assert "targets config" in result.output
    assert "artnet=off (disabled by --no-artnet)" in result.output
    assert "verbose=live" in result.output
    assert "effective=UMC404HD (config)" in result.output


def test_normalize_openvino_logits_accepts_legacy_singleton_batch_axis():
    logits = np.zeros((1, 2, 4), dtype=np.float32)

    normalized = normalize_openvino_logits(logits, batch_size=2, num_classes=4)

    assert normalized.shape == (2, 4)


def test_startup_lines_and_heartbeat_are_compact(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("p")
    layout.create()
    model_dir = layout.model_dir("run_1")
    model_dir.mkdir(parents=True)
    (model_dir / "training_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model_int8.xml").write_text("ir", encoding="utf-8")
    (model_dir / "model_int8.bin").write_text("bin", encoding="utf-8")

    cfg = Config()
    cfg.deploy.active_model = "run_1"
    cfg.model.classes = ["_negative", "cheers_loud", "cheers_quiet"]
    cfg.deploy.record_device = "UMC"
    cfg.audio.record_channels = [0, 1]
    cfg.audio.microphone_type = ["left", "right"]
    cfg.inference.inference_channels = [0, 1]
    cfg.inference.min_dbfs = -45.0

    lines = _startup_lines(
        cfg=cfg,
        project_id="p",
        active_targets={"cheers_loud"},
        with_artnet=False,
        artnet_enabled=False,
        live=False,
        verbose=False,
        mining_enabled=True,
    )

    joined = "\n".join(lines)
    assert "Project: p | active model: run_1" in joined
    assert "Classes: 3 | active targets: cheers_loud" in joined
    assert "mics ch0 left, ch1 right" in joined
    assert "audio_floor -45.0 dBFS" in joined
    assert "mining on" in joined
    assert "disabled by --no-artnet" in joined

    runtime = SimpleNamespace(stats={
        "windows": 123,
        "triggers": 4,
        "audio_peak_dbfs": -10.5,
        "audio_rms_dbfs": -28.25,
        "amplitude_gate": False,
        "min_dbfs": -45.0,
        "amplitude_skipped": 9,
    })
    heartbeat = _format_heartbeat(runtime)
    assert heartbeat == (
        "HEARTBEAT windows=123 detections=4 peak=-10.5dBFS "
        "rms=-28.2dBFS audio_floor=clear below_floor=9"
    )
    assert "triggers" not in heartbeat
    assert "gated" not in heartbeat
    assert "skipped" not in heartbeat


def test_onset_linear_values_display_as_db():
    assert round(_linear_to_db(1e-4), 1) == -80.0
    assert _linear_to_db(0.0) == -120.0


def test_verbose_frame_uses_clear_status_terms():
    classes = ["_negative", "cheers_loud", "cheers_quiet"]
    runtime = SimpleNamespace(
        stats={
            "last_smoothed": {
                "_negative": 0.10,
                "cheers_loud": 0.72,
                "cheers_quiet": 0.08,
            },
            "audio_peak_dbfs": -10.5,
            "audio_rms_dbfs": -28.25,
            "amplitude_gate": False,
        },
        classes=classes,
        arbiter=_cheers_arbiter(classes),
        capture=SimpleNamespace(stats={"xruns": 0}),
    )
    console = Console(record=True, color_system=None, width=160)

    _log_verbose_frame(runtime, {"cheers_loud"}, console)
    text = console.export_text()

    assert "status=listening" in text
    assert "cheers_loud=0.72/0.90" in text
    assert "negative=0.10/0.50" in text
    assert " ok " not in text
    assert "BLOCK" not in text
    assert "SKIP" not in text


def test_live_table_is_minimal_and_mic_aware():
    now = time.monotonic()
    state = _LiveState(
        last_probs={},
        last_detections=[{
            "ts": now - 2.0,
            "class": "cheers_loud",
            "score": 0.93,
            "mic": 0,
            "peak_dbfs": -12.0,
            "saved": "/tmp/project/mined/cheers_loud/example.wav",
        }],
        started_at=now - 12.0,
    )
    stats = {
        "windows": 7,
        "triggers": 2,
        "infer_ms_avg": 3.4,
        "last_smoothed": {
            "_negative": 0.02,
            "cheers_loud": 0.91,
            "cheers_quiet": 0.04,
        },
        "class_thresholds": {
            "_negative": 0.90,
            "cheers_loud": 0.90,
            "cheers_quiet": 0.90,
        },
        "negative_max": 0.50,
        "audio_peak_dbfs": -12.0,
        "audio_rms_dbfs": -30.0,
        "amplitude_gate": False,
        "amplitude_skipped": 0,
        "min_dbfs": -45.0,
        "mic_status": {
            0: {
                "label": "left",
                "peak_dbfs": -12.0,
                "rms_dbfs": -30.0,
                "onset_db": -52.0,
                "onset_gate_db": -49.0,
                "top_class": "cheers_loud",
                "top_score": 0.91,
                "detections": 2,
            },
            1: {
                "label": "right",
                "peak_dbfs": -55.0,
                "rms_dbfs": -70.0,
                "onset_db": -90.0,
                "onset_gate_db": -49.0,
                "top_class": "cheers_quiet",
                "top_score": 0.04,
                "detections": 0,
            },
        },
    }
    table = _build_live_table(state, ["_negative", "cheers_loud", "cheers_quiet"], stats, {"cheers_loud"})
    console = Console(record=True, color_system=None, width=180)

    console.print(table)
    text = console.export_text()

    assert "SED Inference" in text
    assert "ch0 left" in text
    assert "ch1 right" in text
    assert "peak -12.0 / rms -30.0 dBFS" in text
    assert "onset -52.0 / gate -49.0 dB" in text
    assert "cheers_loud 0.910" in text
    assert "detections 2" in text
    assert "mined/cheers_loud/example.wav" in text
    assert "\U0001f480" not in text
    assert "\U0001f3a4" not in text
    assert "trig" not in text.lower()
    assert "gated" not in text.lower()
    assert "skipped" not in text.lower()


def test_detection_line_uses_professional_terms():
    ev = TriggerEvent(
        class_name="cheers_loud",
        confidence=0.93,
        timestamp=time.monotonic(),
        peak_dbfs=-12.0,
        best_channel=0,
        mined_path="/tmp/project/mined/cheers_loud/example.wav",
    )

    line = _format_trigger_line(ev, {0: {"label": "left"}})

    assert "DETECTION class=cheers_loud score=0.93" in line
    assert "mic=ch0 left" in line
    assert "peak=-12.0dBFS" in line
    assert "saved=mined/cheers_loud/example.wav" in line


def test_project_new_creates_config_toml_only(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from sed import config as cfg_mod
    from sed.cli import app

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    result = CliRunner().invoke(app, ["project", "new", "venue_a"])

    assert result.exit_code == 0
    layout = ProjectLayout("venue_a")
    assert layout.config_toml.exists()
    assert not (layout.root / "project_manifest.json").exists()
    assert not (layout.root / "project.toml").exists()
