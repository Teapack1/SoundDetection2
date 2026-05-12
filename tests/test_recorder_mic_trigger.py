from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import soundfile as sf


def test_group_onset_events_merges_only_inside_window():
    from sed.audio.onset import OnsetEvent
    from sed.data.recorder import _group_onset_events

    pending = []
    _group_onset_events(
        pending,
        [
            OnsetEvent(sample_idx=100, strength=0.01, baseline=0.001, channel=2),
            OnsetEvent(sample_idx=112, strength=0.02, baseline=0.001, channel=0),
            OnsetEvent(sample_idx=140, strength=0.03, baseline=0.001, channel=1),
        ],
        merge_window=20,
        target_end_for_event=lambda ev: ev.sample_idx + 50,
    )

    assert len(pending) == 2
    assert pending[0].onset.sample_idx == 100
    assert sorted(pending[0].events_by_channel) == [0, 2]
    assert sorted(pending[1].events_by_channel) == [1]


def test_group_onset_events_blocks_new_groups_during_global_cooldown():
    from sed.audio.onset import OnsetEvent
    from sed.data.recorder import _group_onset_events

    pending = []
    last_group_onset_idx = [None]
    target_end = lambda ev: ev.sample_idx + 50

    _group_onset_events(
        pending,
        [
            OnsetEvent(sample_idx=100, strength=0.01, baseline=0.001, channel=0),
            OnsetEvent(sample_idx=130, strength=0.02, baseline=0.001, channel=1),
            OnsetEvent(sample_idx=700, strength=0.03, baseline=0.001, channel=2),
        ],
        merge_window=50,
        target_end_for_event=target_end,
        recording_cooldown=1000,
        last_group_onset_idx=last_group_onset_idx,
    )

    assert len(pending) == 1
    assert pending[0].onset.sample_idx == 100
    assert sorted(pending[0].events_by_channel) == [0, 1]
    assert last_group_onset_idx == [100]

    _group_onset_events(
        pending,
        [OnsetEvent(sample_idx=1100, strength=0.04, baseline=0.001, channel=2)],
        merge_window=50,
        target_end_for_event=target_end,
        recording_cooldown=1000,
        last_group_onset_idx=last_group_onset_idx,
    )

    assert len(pending) == 2
    assert pending[1].onset.sample_idx == 1100
    assert sorted(pending[1].events_by_channel) == [2]
    assert last_group_onset_idx == [1100]


def test_trigger_group_target_waits_for_larger_of_postroll_and_merge_window():
    from sed.audio.onset import OnsetEvent
    from sed.data.recorder import _trigger_group_target_end_idx

    event = OnsetEvent(sample_idx=3200, strength=0.01, baseline=0.001, channel=0)
    onset_input_idx = int(event.sample_idx * 1.5)

    assert _trigger_group_target_end_idx(
        event,
        sr_ratio=1.5,
        maturity_post_n=12000,
        merge_window_n=24000,
    ) == onset_input_idx + 24000
    assert _trigger_group_target_end_idx(
        event,
        sr_ratio=1.5,
        maturity_post_n=30000,
        merge_window_n=24000,
    ) == onset_input_idx + 30000


def test_recording_session_saves_triggered_channel_subset_metadata(tmp_path):
    from sed.audio.onset import OnsetEvent
    from sed.config import Config
    from sed.data.library import LibraryClip
    from sed.data.recorder import ClipTriggerInfo, RecordingSession

    cfg = Config()
    cfg.audio.record_sample_rate = 48000
    cfg.audio.record_channels = [0, 1, 2]
    session = RecordingSession(cfg, tmp_path, "glass", "p")

    one_mic = np.ones((4800, 1), dtype=np.float32) * 0.25
    wav, _ = session._save_one(
        one_mic,
        OnsetEvent(sample_idx=100, strength=0.02, baseline=0.001, channel=2),
        ClipTriggerInfo(
            recorded_channels=[2],
            triggered_channels=[2],
            triggered_flux={"2": 0.02},
            triggered_peak_dbfs={"2": -12.0},
        ),
    )
    meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))

    assert sf.info(str(wav)).channels == 1
    assert meta["channels"] == 1
    assert meta["recorded_channels"] == [2]
    assert meta["triggered_channels"] == [2]
    assert meta["triggered_flux"] == {"2": 0.02}
    assert meta["triggered_peak_dbfs"] == {"2": -12.0}
    clip = LibraryClip.from_sidecar(wav, tmp_path)
    assert clip.recorded_channels == [2]
    assert clip.triggered_channels == [2]
    assert clip.triggered_flux == {"2": 0.02}

    two_mics = np.stack([
        np.ones(4800, dtype=np.float32) * 0.10,
        np.ones(4800, dtype=np.float32) * 0.20,
    ], axis=-1)
    wav2, _ = session._save_one(
        two_mics,
        OnsetEvent(sample_idx=200, strength=0.03, baseline=0.001, channel=0),
        ClipTriggerInfo(
            recorded_channels=[0, 2],
            triggered_channels=[0, 2],
            triggered_flux={"0": 0.03, "2": 0.04},
            triggered_peak_dbfs={"0": -20.0, "2": -14.0},
        ),
    )
    meta2 = json.loads(wav2.with_suffix(".json").read_text(encoding="utf-8"))

    assert sf.info(str(wav2)).channels == 2
    assert meta2["channels"] == 2
    assert meta2["recorded_channels"] == [0, 2]
    assert meta2["triggered_channels"] == [0, 2]
    assert meta2["triggered_flux"] == {"0": 0.03, "2": 0.04}


def test_recorded_clip_line_lists_mics_flux_and_peak():
    from sed.data.recorder import _format_recorded_clip_line

    line = _format_recorded_clip_line(
        3,
        "/10",
        "glass",
        [0, 2],
        {0: -52.0, 2: -49.0},
        {0: -14.0, 2: -12.0},
        logged_at=datetime(2026, 6, 5, 13, 40, 55, 959000),
    )

    assert line.startswith("  [dim]13:40:55.959[/dim] [bold]#  3[/bold]/10")
    assert "\u2714" not in line
    assert "#  3" in line
    assert "mics 0,2 (2ch)" in line
    assert "flux -52/-49dB" in line
    assert "peak -14/-12dBFS" in line


def test_split_clip_preserves_physical_mic_metadata(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.audio.onset import OnsetEvent
    from sed.config import Config
    from sed.data import library as lib_mod
    from sed.data.library import Library
    from sed.data.recorder import ClipTriggerInfo, RecordingSession

    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    lib = Library()
    cfg = Config()
    cfg.audio.record_channels = [0, 1, 2]
    session = RecordingSession(cfg, lib.raw_dir("glass"), "glass", "p")
    wave = np.stack([
        np.ones(3200, dtype=np.float32) * 0.10,
        np.ones(3200, dtype=np.float32) * 0.20,
    ], axis=-1)
    wav, _ = session._save_one(
        wave,
        OnsetEvent(sample_idx=100, strength=0.02, baseline=0.001, channel=0),
        ClipTriggerInfo(
            recorded_channels=[0, 2],
            triggered_channels=[0, 2],
            triggered_flux={"0": 0.02, "2": 0.03},
            triggered_peak_dbfs={"0": -20.0, "2": -14.0},
        ),
    )

    split_paths = lib.split_clip(wav)
    metas = [
        json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        for path in sorted(split_paths)
    ]

    assert len(metas) == 2
    assert [m["recorded_channels"] for m in metas] == [[0], [2]]
    assert [m["triggered_channels"] for m in metas] == [[0], [2]]
    assert metas[0]["triggered_flux"] == {"0": 0.02}
    assert metas[1]["triggered_flux"] == {"2": 0.03}


def test_split_clip_preserves_multi_mic_types(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.audio.onset import OnsetEvent
    from sed.config import Config
    from sed.data import library as lib_mod
    from sed.data.library import Library
    from sed.data.recorder import ClipTriggerInfo, RecordingSession

    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    lib = Library()
    cfg = Config()
    cfg.audio.record_channels = [0, 1, 2]
    cfg.audio.microphone_type = ["generic", "rode", "shure"]
    session = RecordingSession(cfg, lib.raw_dir("glass"), "glass", "p")
    wave = np.stack([
        np.ones(3200, dtype=np.float32) * 0.10,
        np.ones(3200, dtype=np.float32) * 0.20,
        np.ones(3200, dtype=np.float32) * 0.15,
    ], axis=-1)
    wav, _ = session._save_one(
        wave,
        OnsetEvent(sample_idx=100, strength=0.02, baseline=0.001, channel=0),
        ClipTriggerInfo(
            recorded_channels=[0, 1, 2],
            triggered_channels=[0],
            triggered_flux={"0": 0.02},
            triggered_peak_dbfs={"0": -20.0},
        ),
    )

    sidecar = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["microphone_type"] == ["generic", "rode", "shure"]

    split_paths = lib.split_clip(wav)
    metas = [
        json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        for path in sorted(split_paths)
    ]

    assert len(metas) == 3
    assert metas[0]["microphone_type"] == "generic"
    assert metas[1]["microphone_type"] == "rode"
    assert metas[2]["microphone_type"] == "shure"


def test_microphone_type_coercion_and_validation():
    from sed.config import AudioCfg

    # Test default
    cfg = AudioCfg()
    assert cfg.microphone_type == ["generic"]

    # Test single string
    cfg = AudioCfg(microphone_type="rode")
    assert cfg.microphone_type == ["rode"]

    # Test comma-separated string
    cfg = AudioCfg(microphone_type="generic, rode, shure")
    assert cfg.microphone_type == ["generic", "rode", "shure"]

    # Test list
    cfg = AudioCfg(microphone_type=["generic", "rode", "shure"])
    assert cfg.microphone_type == ["generic", "rode", "shure"]

    # Test empty or invalid
    cfg = AudioCfg(microphone_type="")
    assert cfg.microphone_type == ["generic"]

    cfg = AudioCfg(microphone_type=None)
    assert cfg.microphone_type == ["generic"]


def test_model_rate_index_does_not_drift_over_long_sessions():
    from sed.data.recorder import _model_rate_index

    input_sr, model_sr, blocksize = 48000, 32000, 1024
    n_blocks = int(30 * 60 * input_sr / blocksize)        # 30-minute session
    write_idx = n_blocks * blocksize
    true_model_idx = write_idx * model_sr / input_sr

    mapped = _model_rate_index(write_idx, input_sr, model_sr)
    assert abs(mapped - true_model_idx) <= 1

    # Round-trip through the clip-slicing mapping (clip_start uses
    # int(model_idx * sr_ratio)): stays within 2 input samples of truth.
    sr_ratio = input_sr / model_sr
    assert abs(int(mapped * sr_ratio) - write_idx) <= 2

    # The bug this guards against: accumulating per-block resampled lengths
    # rounds 682.67 -> 683 every block, drifting ~29 ms per session minute.
    per_block_len = int(round(blocksize * model_sr / input_sr))
    accumulated = n_blocks * per_block_len
    drift_ms = (accumulated - true_model_idx) / model_sr * 1000.0
    assert drift_ms > 500.0
