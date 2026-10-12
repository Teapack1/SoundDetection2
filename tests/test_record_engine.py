"""RecordEngine behaviour, driven by a fake capture instead of PortAudio.

The engine is what both `sed record` and `sed record --serve` run, so these
tests are about the loop itself: does a clip land, does it carry the right
sidecar, and can the gate be retuned mid-session without dropping the stream.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import soundfile as sf

SR = 16_000
BLOCK = 512


class _FakeCapture:
    """Stands in for AudioCapture: a real RingBuffer the test writes into."""

    instances: ClassVar[list] = []

    def __init__(self, sample_rate, channels, device, blocksize, ring_seconds, on_block):
        from sed.audio.capture import RingBuffer

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.device = device
        self.blocksize = int(blocksize)
        self.ring = RingBuffer(int(ring_seconds * sample_rate), channels=self.channels)
        self._on_block = on_block
        self.stats = {"xruns": 0, "blocks": 0, "reconnects": 0}
        self.running = False
        _FakeCapture.instances.append(self)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def feed(self, block: np.ndarray) -> None:
        self.ring.write(block)
        self.stats["blocks"] += 1
        if self._on_block is not None:
            self._on_block(block, self.ring.write_idx)

    def feed_seconds(self, seconds: float, make_block) -> None:
        for i in range(int(seconds * self.sample_rate / BLOCK)):
            self.feed(make_block(i))


def _noise(channels: int, scale: float = 1e-4, seed: int | None = None):
    rng = np.random.default_rng(seed if seed is not None else 0)

    def make(_i):
        return (rng.standard_normal((BLOCK, channels)) * scale).astype(np.float32)

    return make


@pytest.fixture
def engine_env(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    from sed.data import record_engine as eng_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")
    _FakeCapture.instances = []
    monkeypatch.setattr(eng_mod, "AudioCapture", _FakeCapture)
    # The engine pre-validates the device against PortAudio before opening; the
    # fake capture has no device to validate.
    monkeypatch.setattr(eng_mod, "check_input_settings", lambda **_kw: None)
    return tmp_path


def _cfg(channels=(0, 1), **onset):
    from sed.config import Config

    base = {
        "audio": {
            "record_sample_rate": SR,
            "record_channels": list(channels),
            "blocksize": BLOCK,
            "microphone_type": ["rode", "samson"],
        },
        "model": {"sample_rate": SR},
        "onset": {
            "record_seconds": 0.5,
            "preroll_ms": 150,
            "randomize_preroll": False,
            "recording_cooldown_ms": 200,
            "min_flux_db": -80.0,
            **onset,
        },
        "deploy": {"record_device": "fake"},
    }
    return Config.model_validate(base)


def _start(cfg, **kwargs):
    from sed.data.record_engine import ClipMetadata, RecordEngine

    meta = kwargs.pop("metadata", ClipMetadata(label="clink_test"))
    engine = RecordEngine(cfg, "proj", meta, **kwargs)
    engine.start()
    return engine, _FakeCapture.instances[-1]


def _wait_for_clips(engine, capture, count, *, channels=2, timeout=6.0):
    """Keep the stream alive until the writer has flushed `count` clips."""
    make = _noise(channels, seed=7)
    deadline = time.monotonic() + timeout
    i = 0
    while engine.clips_saved < count and time.monotonic() < deadline:
        capture.feed(make(i))
        i += 1
        time.sleep(0.002)
    return engine.clips_saved


def test_manual_trigger_writes_a_clip_with_a_full_sidecar(engine_env):
    engine, capture = _start(_cfg())
    try:
        capture.feed_seconds(0.6, _noise(2, scale=0.05))
        assert engine.trigger_manual() is True
        assert _wait_for_clips(engine, capture, 1) == 1
    finally:
        engine.stop()

    wavs = sorted((engine_env / "data_library" / "sounds" / "clink_test").glob("*.wav"))
    assert len(wavs) == 1
    audio, sr = sf.read(wavs[0])
    assert sr == SR
    assert audio.shape == (int(0.5 * SR), 2)

    meta = json.loads(wavs[0].with_suffix(".json").read_text())
    assert meta["label"] == "clink_test"
    assert meta["recorded_channels"] == [0, 1]
    # A manual take keeps the normal geometry, so the crop contract downstream
    # is the same one every onset clip carries.
    assert meta["onset_offset_s"] == pytest.approx(0.15, abs=1e-3)
    assert meta["microphone_type"] == ["rode", "samson"]


def test_manual_trigger_is_refused_before_start(engine_env):
    from sed.data.record_engine import ClipMetadata, RecordEngine

    engine = RecordEngine(_cfg(), "proj", ClipMetadata(label="x"))
    assert engine.trigger_manual() is False


def test_continuous_mode_tiles_clips_without_an_onset(engine_env):
    engine, capture = _start(_cfg(), mode="continuous")
    try:
        assert _wait_for_clips(engine, capture, 3) >= 3
    finally:
        engine.stop()

    wavs = sorted((engine_env / "data_library" / "sounds" / "clink_test").glob("*.wav"))
    assert len(wavs) >= 3
    metas = [json.loads(w.with_suffix(".json").read_text()) for w in wavs]
    # No transient was detected, so the sidecar must not claim one — training
    # falls back to locating it rather than trusting a fabricated offset.
    assert all(m["onset_offset_s"] is None for m in metas)
    assert all(m["duration_s"] == pytest.approx(0.5, abs=1e-6) for m in metas)


def test_onset_detector_fires_on_a_transient(engine_env):
    engine, capture = _start(_cfg(mad_k=3.0, min_flux_db=-90.0))
    try:
        # Warm the adaptive baseline on quiet noise, then hit it with a burst.
        capture.feed_seconds(2.0, _noise(2, scale=1e-4, seed=1))
        rng = np.random.default_rng(2)
        for _ in range(4):
            capture.feed((rng.standard_normal((BLOCK, 2)) * 0.3).astype(np.float32))
        assert _wait_for_clips(engine, capture, 1) == 1
    finally:
        engine.stop()
    meta_files = list(
        (engine_env / "data_library" / "sounds" / "clink_test").glob("*.json")
    )
    meta = json.loads(meta_files[0].read_text())
    assert meta["triggered_channels"]
    assert meta["onset_strength"] is not None


def test_set_onset_retunes_live_without_reopening_the_stream(engine_env):
    engine, capture = _start(_cfg())
    try:
        capture.feed_seconds(0.3, _noise(2, scale=0.01))
        before = engine._derived.detectors
        engine.set_onset(min_flux_db=-40.0)
        # OnsetCfg resyncs dB from the linear value, so the round trip is
        # exact to float precision rather than bit-identical.
        assert engine.cfg.onset.min_flux_db == pytest.approx(-40.0)
        assert engine.cfg.onset.min_flux == pytest.approx(10.0 ** (-40.0 / 20.0))
        # Geometry-only changes keep the detectors, so their adaptive baseline
        # (and its warm-up) survives the edit.
        engine.set_onset(record_seconds=0.8)
        assert engine._derived.clip_n == int(0.8 * SR)
        assert engine._derived.detectors is not before  # min_flux rebuilt them
        assert len(_FakeCapture.instances) == 1
        assert capture.running is True
    finally:
        engine.stop()


def test_set_onset_rejects_an_impossible_combination_atomically(engine_env):
    engine, _capture = _start(_cfg())
    try:
        with pytest.raises(ValueError):
            engine.set_onset(preroll_ms=900)          # past the 0.5 s clip
        assert engine.cfg.onset.preroll_ms == 150
        assert engine.cfg.onset.record_seconds == 0.5
        with pytest.raises(ValueError):
            engine.set_onset(n_mels=64)               # not a live-tunable key
    finally:
        engine.stop()


def test_changing_the_label_rolls_the_session_and_the_directory(engine_env):
    engine, capture = _start(_cfg(), mode="continuous")
    sounds = engine_env / "data_library" / "sounds"
    other = sounds / "other_label"
    try:
        assert _wait_for_clips(engine, capture, 1) >= 1
        first_session = engine._session.session_id
        engine.set_metadata(label="other_label", density="busy")
        assert engine._session.session_id != first_session
        # A clip already in flight belongs to the old label, so wait for one
        # emitted after the change rather than for a clip count.
        make = _noise(2, seed=8)
        deadline = time.monotonic() + 6.0
        i = 0
        while time.monotonic() < deadline and not list(other.glob("*.json")):
            capture.feed(make(i))
            i += 1
            time.sleep(0.002)
    finally:
        engine.stop()

    assert list((sounds / "clink_test").glob("*.wav"))
    later = list(other.glob("*.json"))
    assert later
    assert json.loads(later[0].read_text())["density"] == "busy"


def test_set_metadata_rejects_an_empty_label_and_unknown_fields(engine_env):
    engine, _capture = _start(_cfg())
    try:
        with pytest.raises(ValueError):
            engine.set_metadata(label="   ")
        with pytest.raises(ValueError):
            engine.set_metadata(nonsense=1)
        assert engine._meta.label == "clink_test"
    finally:
        engine.stop()


def test_holdout_becomes_a_tag_not_a_split(engine_env):
    from sed.data.library import HOLDOUT_TAG
    from sed.data.record_engine import ClipMetadata

    meta = ClipMetadata(label="l", tags=["wine-glass"], holdout=True)
    assert meta.sidecar_tags() == ["wine-glass", HOLDOUT_TAG]
    assert ClipMetadata(label="l", holdout=False).sidecar_tags() == []


def test_max_clips_stops_the_session_exactly(engine_env):
    engine, capture = _start(_cfg(), mode="continuous", max_clips=2)
    try:
        deadline = time.monotonic() + 6.0
        make = _noise(2, seed=3)
        i = 0
        while engine.state == "running" and time.monotonic() < deadline:
            capture.feed(make(i))
            i += 1
            time.sleep(0.002)
        assert engine.clips_saved == 2
        assert engine._stop_reason == "target reached"
        # The state has to leave "running" on its own, or `sed record` would
        # keep waiting on a session that is already over.
        assert engine.state == "finished"
    finally:
        engine.stop()


def test_snapshot_reports_every_recorded_channel_even_in_mix_mode(engine_env):
    # "mix" runs a single detector on the downmix, so per-channel meters have to
    # come from the audio block or a dead microphone would look healthy.
    engine, capture = _start(_cfg(multi_mic_trigger="mix"))
    try:
        loud = np.zeros((BLOCK, 2), dtype=np.float32)
        loud[:, 0] = 0.5
        for _ in range(20):
            capture.feed(loud)
        snap = engine.snapshot()
        assert [c["channel"] for c in snap["channels"]] == [0, 1]
        assert [c["mic_label"] for c in snap["channels"]] == ["rode", "samson"]
        assert snap["channels"][0]["peak_dbfs"] > -10.0
        assert snap["channels"][1]["peak_dbfs"] < -60.0
        assert snap["clip_frames"] == int(0.5 * SR)
    finally:
        engine.stop()


def test_snapshot_flags_clipping(engine_env):
    engine, capture = _start(_cfg())
    try:
        for _ in range(4):
            capture.feed(np.ones((BLOCK, 2), dtype=np.float32))
        snap = engine.snapshot()
        assert all(c["clipping"] for c in snap["channels"])
    finally:
        engine.stop()


def test_calibration_suggests_a_floor_above_the_room(engine_env):
    engine, capture = _start(_cfg())
    try:
        engine.start_calibration(seconds=0.4)
        make = _noise(2, scale=0.02, seed=5)
        deadline = time.monotonic() + 6.0
        i = 0
        while time.monotonic() < deadline:
            capture.feed(make(i))
            i += 1
            time.sleep(0.002)
            cal = engine.snapshot()["calibration"]
            if cal["state"] in ("done", "failed"):
                break
        assert cal["state"] == "done", cal
        assert -120.0 < cal["suggested_min_flux_db"] < 0.0
        assert cal["suggested_min_flux_db"] > cal["background_p50_db"]
    finally:
        engine.stop()


def test_calibration_requires_a_running_session(engine_env):
    from sed.data.record_engine import ClipMetadata, RecordEngine

    engine = RecordEngine(_cfg(), "proj", ClipMetadata(label="x"))
    with pytest.raises(RuntimeError):
        engine.start_calibration()


def test_delete_clip_removes_wav_sidecar_and_manifest_row(engine_env):
    engine, capture = _start(_cfg(), mode="continuous")
    try:
        assert _wait_for_clips(engine, capture, 1) >= 1
        recent = engine.snapshot()["recent"]
        assert recent
        uuid = recent[0]["uuid"]
        path = Path(recent[0]["path"])
        assert path.exists()
        assert engine.delete_clip(uuid) is True
        assert not path.exists()
        assert not path.with_suffix(".json").exists()
        assert uuid not in {c["uuid"] for c in engine.snapshot()["recent"]}
    finally:
        engine.stop()

    from sed.data.library import Library

    assert all(c.uuid != uuid for c in Library().all_clips())


def test_set_capture_reopens_the_stream_with_the_new_topology(engine_env):
    engine, first = _start(_cfg())
    try:
        engine.set_capture(channels=[0], sample_rate=8000, device="other")
        assert len(_FakeCapture.instances) == 2
        second = _FakeCapture.instances[-1]
        assert first.running is False
        assert second.running is True
        assert second.sample_rate == 8000
        assert second.channels == 1
        assert [c["channel"] for c in engine.snapshot()["channels"]] == [0]
    finally:
        engine.stop()


def test_set_capture_rejects_duplicate_channels(engine_env):
    engine, _first = _start(_cfg())
    try:
        with pytest.raises(ValueError):
            engine.set_capture(channels=[0, 0])
    finally:
        engine.stop()


def test_level_history_is_downsampled_for_the_sparkline(engine_env):
    # A 100 px sparkline does not need a point per 50 ms tick; shipping one
    # would put 1200 numbers per channel in every status poll.
    from sed.data.record_engine import _LEVEL_HISTORY_POINTS

    engine, capture = _start(_cfg())
    try:
        _feed = _noise(2, scale=0.01)
        for i in range(40):
            capture.feed(_feed(i))
        time.sleep(1.2)
        history = engine.snapshot()["channels"][0]["peak_history"]
        assert 1 <= len(history) <= _LEVEL_HISTORY_POINTS
        assert len(history) < 20        # ~2 points per second, not ~20
    finally:
        engine.stop()


def test_continuous_mode_resyncs_instead_of_replaying_lost_audio(engine_env, caplog):
    """A starved tick loop must not emit a burst of clips whose audio has
    already been overwritten in the ring."""
    engine, capture = _start(_cfg(), mode="continuous")
    try:
        capture.feed_seconds(0.2, _noise(2))
        engine._schedule_continuous()          # seeds the tiling position
        # Pretend the loop stalled: rewind well past the ring's capacity.
        engine._continuous_next_start = capture.ring.write_idx - capture.ring.capacity * 3
        submitted_before = engine._submitted
        with caplog.at_level("WARNING"):
            engine._schedule_continuous()
        assert "fell" in caplog.text and "behind the ring buffer" in caplog.text
        # At most the audio actually still in the ring, not the whole backlog.
        emitted = engine._submitted - submitted_before
        assert emitted <= capture.ring.capacity // engine._derived.clip_n + 1
    finally:
        engine.stop()


def test_clip_records_keep_the_whole_run_not_just_the_display_buffer(engine_env):
    # `recent` holds 12 for the UI strip; discarding a pass has to be able to
    # name every clip it wrote, however long the pass was.
    engine, capture = _start(_cfg(record_seconds=0.2), mode="continuous")
    try:
        assert _wait_for_clips(engine, capture, 14, timeout=10.0) >= 14
        records = engine.clip_records()
        assert len(records) >= 14
        assert len(engine.snapshot()["recent"]) == 12
        assert all({"uuid", "path", "label", "session_id"} <= set(r) for r in records)
    finally:
        engine.stop()


def test_discard_all_deletes_every_clip_of_the_run(engine_env):
    engine, capture = _start(_cfg(), mode="continuous")
    try:
        assert _wait_for_clips(engine, capture, 3) >= 3
        out = engine_env / "data_library" / "sounds" / "clink_test"
        assert list(out.glob("*.wav"))
    finally:
        engine.stop()

    removed = engine.discard_all()
    assert removed >= 3
    assert list(out.glob("*.wav")) == []
    assert list(out.glob("*.json")) == []
    assert engine.clip_records() == []
    assert engine.snapshot()["recent"] == []

    from sed.data.library import Library

    assert Library().all_clips() == []
    # Idempotent: discarding an already-empty run is not an error.
    assert engine.discard_all() == 0


def test_discard_all_spans_a_label_change(engine_env):
    engine, capture = _start(_cfg(), mode="continuous")
    sounds = engine_env / "data_library" / "sounds"
    try:
        assert _wait_for_clips(engine, capture, 1) >= 1
        engine.set_metadata(label="second_label")
        make = _noise(2, seed=12)
        deadline = time.monotonic() + 6.0
        i = 0
        while time.monotonic() < deadline and not list(
            (sounds / "second_label").glob("*.json")
        ):
            capture.feed(make(i))
            i += 1
            time.sleep(0.002)
    finally:
        engine.stop()

    labels = {r["label"] for r in engine.clip_records()}
    assert labels == {"clink_test", "second_label"}
    engine.discard_all()
    assert list((sounds / "clink_test").glob("*.wav")) == []
    assert list((sounds / "second_label").glob("*.wav")) == []


def test_library_remove_clips_drops_files_and_rows_in_one_pass(engine_env):
    import soundfile as sf

    from sed.data.library import Library

    lib = Library()
    out = lib.raw_dir("bulk")
    kept, doomed = [], []
    for i in range(4):
        wav = out / f"clip_{i}.wav"
        sf.write(str(wav), np.zeros(1600, dtype="float32"), 16_000)
        wav.with_suffix(".json").write_text(
            json.dumps({"schema_version": 3, "uuid": f"u{i}", "label": "bulk",
                        "category": "sounds", "project_id": "p"}),
            encoding="utf-8",
        )
        lib.append_sidecar(wav)
        (doomed if i < 2 else kept).append(wav)

    removed = lib.remove_clips(doomed)
    assert len(removed) == 2
    assert all(not w.exists() and not w.with_suffix(".json").exists() for w in doomed)
    assert all(w.exists() for w in kept)
    assert {c.uuid for c in Library().all_clips()} == {"u2", "u3"}


def test_editing_any_onset_value_leaves_the_floor_exactly_where_it_was(engine_env):
    """OnsetCfg re-derives min_flux_db from the linear value on every
    revalidation, so an unrelated edit used to turn a hand-written -80 into
    -79.99999991314111 — and that is the number that reaches config.toml."""
    from sed.config import OnsetCfg
    from sed.data.record_engine import apply_onset_updates, finalize_onset

    base = OnsetCfg.model_validate({"min_flux_db": -80.0, "record_seconds": 1.0})
    assert base.min_flux_db == -80.0

    assert apply_onset_updates(base, {"record_seconds": 2.0}).min_flux_db == -80.0
    assert apply_onset_updates(base, {"mad_k": 4.0}).min_flux_db == -80.0
    assert finalize_onset(base).min_flux_db == -80.0

    # Changing the floor itself keeps the figure the operator asked for, and
    # the linear value the detector uses stays consistent with it.
    moved = apply_onset_updates(base, {"min_flux_db": -62.0})
    assert moved.min_flux_db == -62.0
    assert moved.min_flux == pytest.approx(10.0 ** (-62.0 / 20.0))


# ---------------------------------------------------------------------------
# Settings must not be able to corrupt a clip
# ---------------------------------------------------------------------------


def test_changing_the_rate_never_restamps_clips_already_captured(engine_env):
    """The rate a clip is written with must be the rate it was captured at.

    The writer used to read the live config when a clip reached the front of
    the queue, so switching the interface stamped the new rate onto old audio:
    48 kHz PCM with a 44.1 kHz header plays 8.8 % slow, and training resamples
    from the header, so it silently becomes a time-stretched example.
    """
    engine, capture = _start(_cfg(), mode="continuous")
    try:
        assert _wait_for_clips(engine, capture, 2) >= 2
        engine.set_capture(sample_rate=8000)
        new_capture = _FakeCapture.instances[-1]
        deadline = time.monotonic() + 6.0
        make = _noise(2, seed=31)
        i = 0
        while time.monotonic() < deadline and engine.clips_saved < 4:
            new_capture.feed(make(i))
            i += 1
            time.sleep(0.002)
    finally:
        engine.stop()

    import soundfile as sf

    rates = {}
    for wav in sorted((engine_env / "data_library" / "sounds" / "clink_test").rglob("*.wav")):
        meta = json.loads(wav.with_suffix(".json").read_text())
        header = sf.info(str(wav))
        # Header, sidecar and duration all have to agree with each other.
        assert meta["sample_rate"] == header.samplerate, wav.name
        assert meta["duration_s"] == pytest.approx(header.duration, abs=1e-6)
        rates.setdefault(header.samplerate, 0)
        rates[header.samplerate] += 1
    assert set(rates) <= {SR, 8000}
    assert rates.get(SR), "clips captured at the original rate must keep it"


def test_a_capture_change_starts_a_new_recording_group(engine_env):
    # session_id is the unit that must never straddle a train/val split, and
    # audio captured with a different topology is not the same group.
    engine, capture = _start(_cfg(), mode="continuous")
    try:
        assert _wait_for_clips(engine, capture, 1) >= 1
        before = engine._session.session_id
        engine.set_capture(sample_rate=8000)
        assert engine._session.session_id != before
    finally:
        engine.stop()


def test_microphone_names_follow_the_channels_actually_saved(engine_env):
    from sed.data.recorder import ClipContext, _mics_for_channels

    ctx = ClipContext(
        sample_rate=SR, device="d",
        microphone_type=("boundary", "shotgun"), record_channels=(0, 3),
    )
    # Only ch3 fired, so the clip holds one channel — and it is the shotgun.
    assert _mics_for_channels(ctx, [3]) == ("shotgun",)
    assert _mics_for_channels(ctx, [0]) == ("boundary",)
    assert _mics_for_channels(ctx, [0, 3]) == ("boundary", "shotgun")
    # An unknown channel must not silently borrow another mic's name.
    assert _mics_for_channels(ctx, [7]) == ("generic",)


def test_operator_microphone_names_reach_the_sidecar(engine_env):
    from sed.data.record_engine import ClipMetadata

    engine, capture = _start(
        _cfg(), mode="continuous",
        metadata=ClipMetadata(label="mics", microphone_type=["rode", "samson"]),
    )
    try:
        assert _wait_for_clips(engine, capture, 1) >= 1
    finally:
        engine.stop()
    meta_file = next((engine_env / "data_library" / "sounds" / "mics").glob("*.json"))
    assert json.loads(meta_file.read_text())["microphone_type"] == ["rode", "samson"]


def test_a_shorter_clip_never_leaves_the_onset_outside_it(engine_env):
    """Geometry is read once per clip.

    Taking the preroll from one retune and the clip length from another sliced
    a window that did not contain the transient, while the sidecar still said
    it did.
    """
    engine, capture = _start(_cfg(record_seconds=2.0, preroll_ms=700), mode="continuous")
    try:
        assert _wait_for_clips(engine, capture, 1, timeout=10.0) >= 1
        engine.set_onset(record_seconds=0.5, preroll_ms=100)
        deadline = time.monotonic() + 8.0
        make = _noise(2, seed=17)
        i = 0
        while time.monotonic() < deadline and engine.clips_saved < 3:
            capture.feed(make(i))
            i += 1
            time.sleep(0.002)
    finally:
        engine.stop()

    for wav in sorted((engine_env / "data_library" / "sounds" / "clink_test").glob("*.json")):
        meta = json.loads(wav.read_text())
        offset = meta["onset_offset_s"]
        if offset is not None:
            assert 0 <= offset < meta["duration_s"], meta["path"]


def test_growing_the_clip_past_the_ring_reopens_instead_of_dropping(engine_env):
    """Raising the clip length used to leave the buffer behind, after which
    every clip failed into a warning while the console still said 'running'."""
    engine, capture = _start(_cfg(record_seconds=0.4), mode="continuous")
    try:
        opened = len(_FakeCapture.instances)
        assert _wait_for_clips(engine, capture, 1) >= 1
        engine.set_onset(record_seconds=4.0)
        assert len(_FakeCapture.instances) > opened, "the stream should have reopened"
        assert engine._ring_fits()
        capture = _FakeCapture.instances[-1]
        assert capture.ring.capacity >= engine._derived.clip_n * 2
    finally:
        engine.stop()


def test_a_dropout_resets_the_detectors_and_is_reported(engine_env):
    engine, capture = _start(_cfg())
    try:
        capture.feed_seconds(0.5, _noise(2, scale=0.01))
        capture.ring.mark_discontinuity()
        capture.feed_seconds(0.2, _noise(2, scale=0.01))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and engine.snapshot()["dropouts"] == 0:
            capture.feed(_noise(2)(0))
            time.sleep(0.01)
        assert engine.snapshot()["dropouts"] == 1
    finally:
        engine.stop()


def test_a_writer_failure_does_not_wedge_the_session(engine_env, monkeypatch):
    """A dead writer used to fill the queue, block the recording loop in
    submit_clip and make stop() hang."""
    engine, capture = _start(_cfg(), mode="continuous")
    try:
        assert _wait_for_clips(engine, capture, 1) >= 1
        session = engine._session
        monkeypatch.setattr(
            session, "_save_one",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )
        make = _noise(2, seed=23)
        for i in range(400):
            capture.feed(make(i))
            time.sleep(0.001)
        assert session.write_errors > 0
        assert "disk full" in session.last_write_error
        # Still alive: the tick loop kept running rather than blocking.
        assert engine.state == "running"
    finally:
        engine.stop()          # must return, not hang


def test_max_clips_still_lands_after_a_clip_is_dropped(engine_env):
    """The accept gate is recomputed, not latched.

    A dropped clip lowers the in-flight count without raising the saved count;
    latching the gate off there meant the target could never be reached and no
    new onset was ever accepted again.
    """
    engine, capture = _start(_cfg(), mode="continuous", max_clips=3)
    try:
        engine._accept_new = False          # as a drop used to leave it
        deadline = time.monotonic() + 8.0
        make = _noise(2, seed=29)
        i = 0
        while engine.state == "running" and time.monotonic() < deadline:
            capture.feed(make(i))
            i += 1
            time.sleep(0.002)
        assert engine.clips_saved == 3
        assert engine.state == "finished"
    finally:
        engine.stop()


def test_a_label_change_does_not_block_on_the_writer(engine_env):
    """set_metadata held the lock the writer needs across a 30 s join."""
    engine, capture = _start(_cfg(), mode="continuous")
    try:
        assert _wait_for_clips(engine, capture, 2) >= 2
        started = time.monotonic()
        engine.set_metadata(label="renamed")
        assert time.monotonic() - started < 5.0
    finally:
        engine.stop()
