"""End-to-end tests for the recording console server.

Same shape as tests/test_inspect_server.py: boot the stdlib HTTP server in a
thread and drive the API with urllib. PortAudio is faked, and the capture loop
runs against a ring buffer the test writes into, so these exercise the real
handlers without any hardware.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from sed.config import Config

_HTTP_TIMEOUT = 30.0
SR = 16_000
BLOCK = 512


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _patch_project_and_library_dirs(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")


def _fake_sounddevice(monkeypatch):
    devices = [
        {"name": "default", "max_input_channels": 2,
         "default_samplerate": 48000, "hostapi": 0},
        {"name": "UMC404HD", "max_input_channels": 4,
         "default_samplerate": 48000, "hostapi": 0},
    ]

    class FakeStream:
        active = True

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def start(self):
            return None

        def stop(self):
            self.active = False

        def close(self):
            self.active = False

    def check_input_settings(device=None, channels=None, dtype=None, samplerate=None):
        info = devices[int(device)]
        if channels and channels > info["max_input_channels"]:
            raise ValueError("Invalid number of channels")

    fake = SimpleNamespace(
        default=SimpleNamespace(device=[0, None]),
        query_devices=lambda idx=None: devices if idx is None else devices[int(idx)],
        query_hostapis=lambda: [{"name": "ALSA"}],
        InputStream=FakeStream,
        PortAudioError=RuntimeError,
        check_input_settings=check_input_settings,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


class _FakeCapture:
    """AudioCapture stand-in wrapping a real RingBuffer the test feeds."""

    instances: ClassVar[list] = []

    def __init__(self, sample_rate, channels, device, blocksize, ring_seconds, on_block):
        from sed.audio.capture import RingBuffer

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.device = device
        self.ring = RingBuffer(int(ring_seconds * sample_rate), channels=self.channels)
        self._on_block = on_block
        self.stats = {"xruns": 0, "blocks": 0, "reconnects": 0}
        self.running = False
        _FakeCapture.instances.append(self)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def feed(self, block):
        self.ring.write(block)
        self.stats["blocks"] += 1
        if self._on_block is not None:
            self._on_block(block, self.ring.write_idx)


CONFIG_TOML = """\
[project]
id = "rec"
name = "rec"

[dataset]
target_classes = ["_negative", "cheers"]

[deploy]
record_device = "UMC404HD"   # keep this comment

[audio]
record_sample_rate = 16000
record_channels = [0, 1]
blocksize = 512
microphone_type = ["rode", "samson"]

[onset]
# hand-written note that must survive a save from the UI
min_flux_db = -80
record_seconds = 0.5
preroll_ms = 150
randomize_preroll = false

[model]
sample_rate = 16000
"""


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Boot the record server against a temp project and a faked sound card."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    _fake_sounddevice(monkeypatch)
    from sed.config import ProjectLayout, load_config
    from sed.data import record_engine as eng_mod

    _FakeCapture.instances = []
    monkeypatch.setattr(eng_mod, "AudioCapture", _FakeCapture)

    layout = ProjectLayout("rec")
    layout.create()
    layout.config_toml.write_text(CONFIG_TOML, encoding="utf-8")
    cfg = load_config("rec")

    from sed.data.record_server import make_handler

    shutdown = threading.Event()
    handler = make_handler(cfg, "rec", shutdown)
    httpd = ThreadingHTTPServer(("127.0.0.1", _free_port()), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    try:
        yield SimpleNamespace(base=base, layout=layout, shutdown=shutdown,
                              handler=handler, tmp_path=tmp_path)
    finally:
        handler.cleanup()
        httpd.shutdown()
        thread.join(timeout=2.0)


def _get(server, path):
    with urllib.request.urlopen(server.base + path, timeout=_HTTP_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(server, path, body=None):
    req = urllib.request.Request(
        server.base + path,
        data=json.dumps(body or {}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_expect_error(server, path, body=None):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(server, path, body)
    payload = json.loads(excinfo.value.read().decode("utf-8"))
    return excinfo.value.code, payload


def _feed(capture, blocks=40, channels=2, scale=1e-3):
    rng = np.random.default_rng(11)
    for _ in range(blocks):
        capture.feed((rng.standard_normal((BLOCK, channels)) * scale).astype("float32"))
        time.sleep(0.002)


def _record_until(capture, engine_getter, count, timeout=6.0):
    deadline = time.monotonic() + timeout
    rng = np.random.default_rng(3)
    while time.monotonic() < deadline:
        snap = engine_getter()
        if snap and snap.get("clips_saved", 0) >= count:
            return snap
        capture.feed((rng.standard_normal((BLOCK, 2)) * 1e-3).astype("float32"))
        time.sleep(0.002)
    return engine_getter()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_channels_keeps_order_and_rejects_bad_selections():
    from sed.data.record_server import parse_channels

    assert parse_channels("3,0,1") == [3, 0, 1]
    assert parse_channels([0, 2]) == [0, 2]
    for bad in ("", [], [0, 0], [-1], "a,b"):
        with pytest.raises(ValueError):
            parse_channels(bad)


def test_parse_duration_reads_the_cli_forms():
    from sed.data.record_server import parse_duration

    assert parse_duration("15m") == 900.0
    assert parse_duration("2h") == 7200.0
    assert parse_duration("90s") == 90.0
    assert parse_duration("45") == 45.0
    assert parse_duration("") is None
    with pytest.raises(ValueError):
        parse_duration("soon")
    with pytest.raises(ValueError):
        parse_duration("0")


def test_parse_tags_dedupes_and_strips_hashes():
    from sed.data.record_server import parse_tags

    assert parse_tags(" #wine-glass, quiet ,wine-glass, ") == ["wine-glass", "quiet"]
    assert parse_tags(["a", "a", ""]) == ["a"]


def test_onset_updates_only_accepts_live_tunable_keys():
    from sed.data.record_server import onset_updates

    assert onset_updates({"min_flux_db": "-70", "mad_k": 4}) == {
        "min_flux_db": -70.0, "mad_k": 4.0,
    }
    assert onset_updates({"preroll_ms_range": ["100", "600"]}) == {
        "preroll_ms_range": (100, 600),
    }
    # n_mels changes the model feature, not the trigger — it must not sneak in.
    assert onset_updates({"n_mels": 128}) == {}


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------


def test_devices_lists_inputs_and_the_configured_selection(server):
    data = _get(server, "/api/record/devices")
    assert data["ok"] is True
    assert [d["name"] for d in data["devices"]] == ["default", "UMC404HD"]
    assert data["device"] == "UMC404HD"
    assert data["channels"] == [0, 1]
    assert data["sample_rate"] == 16000
    assert data["microphone_type"] == ["rode", "samson"]
    assert data["densities"][0] == "empty"


def test_status_is_idle_before_anything_starts(server):
    data = _get(server, "/api/record/status")
    assert data["recording"] is False
    assert data["meter"]["state"] == "idle"
    assert data["onset"]["record_seconds"] == 0.5
    # The frame count is what makes clip length concrete for an operator.
    assert data["clip_frames"] == 8000


def test_page_renders_the_shared_shell(server):
    with urllib.request.urlopen(server.base + "/", timeout=_HTTP_TIMEOUT) as r:
        html = r.read().decode("utf-8")
    assert "Recording console" in html
    assert "--bg-page" in html          # shared theme tokens
    assert 'id="hover-tip"' in html     # shared tooltip element
    assert 'id="toast-stack"' in html


def test_clip_geometry_keeps_its_labels_out_of_the_bar(server):
    """The diagram used to be an SVG stretched to the panel width, which
    distorted its glyphs and printed them through the bar. The bar is now
    graphics only; every label is real text laid out beneath it."""
    import re

    with urllib.request.urlopen(server.base + "/", timeout=_HTTP_TIMEOUT) as r:
        html = r.read().decode("utf-8")

    track = re.search(
        r'<div class="geom-track">(.*?)</div>\s*<div class="geom-scale"', html, re.S
    )
    assert track, "geometry track missing"
    assert not re.sub(r"<[^>]+>", "", track.group(1)).strip()

    # Three regions and their key: guaranteed preroll, the window a randomized
    # onset lands in, and the postroll.
    for marker in ("geom-pre-fill", "geom-jitter", "geom-jitter-key",
                   "geom-scale", "geom-legend"):
        assert marker in html, marker


def test_preroll_range_has_its_own_controls(server):
    with urllib.request.urlopen(server.base + "/", timeout=_HTTP_TIMEOUT) as r:
        html = r.read().decode("utf-8")
    for marker in ('id="k-pre-lo"', 'id="k-pre-hi"', 'id="preroll-range"',
                   'id="preroll-fixed"', "clampPreroll", "prerollCeiling"):
        assert marker in html, marker
    # The sliders are bounded by the clip at runtime, not by a baked-in ceiling.
    assert 'id="k-preroll" min="0" step="10"' in html


def test_page_carries_the_run_history_and_discard_action(server):
    with urllib.request.urlopen(server.base + "/", timeout=_HTTP_TIMEOUT) as r:
        html = r.read().decode("utf-8")
    for marker in ('id="runs"', 'id="btn-cancel"', "renderRuns", "cancelRun",
                   "deleteRun", "This server run"):
        assert marker in html, marker


def test_unknown_routes_are_404(server):
    for path in ("/api/record/nope", "/nope"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(server.base + path, timeout=_HTTP_TIMEOUT)
        assert exc.value.code == 404


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def _start(server, **overrides):
    body = {"label": "cheers_quiet", "density": "busy", "mode": "continuous",
            "tags": "wine-glass", "location": "bar", "num": 0}
    body.update(overrides)
    return _post(server, "/api/record/start", body)


def test_start_status_stop_records_clips_with_provenance(server):
    data = _start(server)
    assert data["session"]["state"] == "running"
    capture = _FakeCapture.instances[-1]
    assert capture.sample_rate == 16000

    snap = _record_until(
        capture, lambda: _get(server, "/api/record/status")["session"], 1
    )
    assert snap["clips_saved"] >= 1
    assert snap["label"] == "cheers_quiet"
    assert [c["channel"] for c in snap["channels"]] == [0, 1]
    assert snap["recent"]

    summary = _post(server, "/api/record/stop", {})["summary"]
    assert summary["clips"] >= 1
    assert _get(server, "/api/record/status")["recording"] is False

    wavs = list((server.tmp_path / "data_library" / "sounds" / "cheers_quiet")
                .glob("*.wav"))
    assert wavs
    meta = json.loads(wavs[0].with_suffix(".json").read_text())
    assert meta["density"] == "busy"
    assert meta["location"] == "bar"
    assert meta["tags"] == ["wine-glass"]
    assert meta["project_id"] == "rec"


def test_start_without_a_label_is_a_400(server):
    code, payload = _post_expect_error(server, "/api/record/start", {"label": "  "})
    assert code == 400
    assert "label" in payload["error"]


def test_start_rejects_an_unknown_density(server):
    code, payload = _post_expect_error(
        server, "/api/record/start", {"label": "x", "density": "loud"}
    )
    assert code == 400
    assert "density" in payload["error"]


def test_starting_twice_is_a_409(server):
    _start(server)
    try:
        code, payload = _post_expect_error(server, "/api/record/start", {"label": "x"})
        assert code == 409
        assert "already running" in payload["error"]
    finally:
        _post(server, "/api/record/stop", {})


def test_start_rejects_channels_the_interface_does_not_have(server):
    # The engine opens max(index)+1 channels, so a bad index has to be caught
    # before the stream, not after.
    code, payload = _post_expect_error(
        server, "/api/record/start", {"label": "x", "channels": [0, 0]}
    )
    assert code == 400
    assert "repeat" in payload["error"]


def test_manual_capture_needs_a_running_session(server):
    code, payload = _post_expect_error(server, "/api/record/trigger", {})
    assert code == 409
    assert "start a session" in payload["error"]


def test_manual_capture_saves_a_clip_in_manual_mode(server):
    _start(server, mode="manual", label="manual_take")
    capture = _FakeCapture.instances[-1]
    try:
        _feed(capture, blocks=30)
        assert _post(server, "/api/record/trigger", {})["triggered"] is True
        snap = _record_until(
            capture, lambda: _get(server, "/api/record/status")["session"], 1
        )
        assert snap["clips_saved"] == 1
        assert snap["recent"][0]["kind"] == "manual"
    finally:
        _post(server, "/api/record/stop", {})


# ---------------------------------------------------------------------------
# Live retuning
# ---------------------------------------------------------------------------


def test_settings_retune_the_gate_without_reopening_the_stream(server):
    _start(server)
    try:
        before = len(_FakeCapture.instances)
        _post(server, "/api/record/settings",
              {"onset": {"min_flux_db": -55, "mad_k": 5.5}})
        snap = _get(server, "/api/record/status")["session"]
        assert snap["onset"]["min_flux_db"] == pytest.approx(-55.0)
        assert snap["onset"]["mad_k"] == 5.5
        assert len(_FakeCapture.instances) == before
    finally:
        _post(server, "/api/record/stop", {})


def test_settings_reject_an_impossible_clip_geometry(server):
    _start(server)
    try:
        code, payload = _post_expect_error(
            server, "/api/record/settings", {"onset": {"preroll_ms": 900}}
        )
        assert code == 400
        snap = _get(server, "/api/record/status")["session"]
        assert snap["onset"]["preroll_ms"] == 150
        assert "preroll" in payload["error"].lower()
    finally:
        _post(server, "/api/record/stop", {})


def test_settings_change_metadata_for_the_next_clip(server):
    _start(server)
    capture = _FakeCapture.instances[-1]
    other = server.tmp_path / "data_library" / "sounds" / "other_take"
    try:
        _record_until(capture, lambda: _get(server, "/api/record/status")["session"], 1)
        _post(server, "/api/record/settings",
              {"metadata": {"label": "other_take", "density": "empty"}})
        assert _get(server, "/api/record/status")["session"]["label"] == "other_take"
        # A clip already in flight belongs to the old label, so wait for one
        # that was emitted after the change rather than for a clip count.
        deadline = time.monotonic() + 6.0
        rng = np.random.default_rng(4)
        while time.monotonic() < deadline and not list(other.glob("*.json")):
            capture.feed((rng.standard_normal((BLOCK, 2)) * 1e-3).astype("float32"))
            time.sleep(0.002)
    finally:
        _post(server, "/api/record/stop", {})

    later = list(other.glob("*.json"))
    assert later
    assert json.loads(later[0].read_text())["density"] == "empty"
    # The previous label keeps the clips it recorded; nothing is retroactive.
    assert list((server.tmp_path / "data_library" / "sounds" / "cheers_quiet")
                .glob("*.wav"))


def test_settings_are_staged_while_idle_and_inherited_by_the_next_start(server):
    _post(server, "/api/record/settings", {"onset": {"record_seconds": 1.25}})
    idle = _get(server, "/api/record/status")
    assert idle["onset"]["record_seconds"] == 1.25
    assert idle["clip_frames"] == 20000
    _start(server)
    try:
        snap = _get(server, "/api/record/status")["session"]
        assert snap["record_seconds"] == 1.25
        assert snap["clip_frames"] == 20000
    finally:
        _post(server, "/api/record/stop", {})


def test_a_page_reload_mid_session_shows_the_live_values(server):
    """The knobs and the device list must describe the running stream, not the
    config it was started from."""
    _start(server)
    try:
        _post(server, "/api/record/settings",
              {"onset": {"min_flux_db": -47, "record_seconds": 0.75},
               "metadata": {"location": "bar-zone-2"}})
        with urllib.request.urlopen(server.base + "/", timeout=_HTTP_TIMEOUT) as r:
            html = r.read().decode("utf-8")
        assert '"min_flux_db": -47.0' in html
        assert '"record_seconds": 0.75' in html
        assert "bar-zone-2" in html
        assert _get(server, "/api/record/devices")["sample_rate"] == 16000
    finally:
        _post(server, "/api/record/stop", {})


def test_mode_switches_live(server):
    _start(server, mode="onset")
    try:
        _post(server, "/api/record/settings", {"mode": "continuous"})
        assert _get(server, "/api/record/status")["session"]["mode"] == "continuous"
    finally:
        _post(server, "/api/record/stop", {})


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_settings_writes_config_toml_and_keeps_its_comments(server):
    _post(server, "/api/record/settings",
          {"onset": {"min_flux_db": -62, "mad_k": 4.5, "record_seconds": 0.75}})
    data = _post(server, "/api/record/save-settings", {"group": "onset"})
    assert data["ok"] is True

    text = server.layout.config_toml.read_text(encoding="utf-8")
    assert "min_flux_db = -62.0" in text
    assert "mad_k = 4.5" in text
    assert "record_seconds = 0.75" in text
    # Regex surgery, not a re-dump: a venue's hand-written notes have to survive.
    assert "# hand-written note that must survive a save from the UI" in text
    assert "# keep this comment" in text

    from sed.config import load_config

    assert load_config("rec").onset.min_flux_db == pytest.approx(-62.0)


def test_save_settings_writes_the_capture_topology(server):
    _post(server, "/api/record/settings",
          {"capture": {"channels": [1, 0], "sample_rate": 44100, "device": "default"}})
    _post(server, "/api/record/save-settings", {"group": "capture"})

    from sed.config import load_config

    cfg = load_config("rec")
    assert cfg.audio.record_channels == [1, 0]
    assert cfg.audio.record_sample_rate == 44100
    assert cfg.deploy.record_device == "default"


def test_save_settings_rejects_an_unknown_group(server):
    code, payload = _post_expect_error(
        server, "/api/record/save-settings", {"group": "everything"}
    )
    assert code == 400
    assert "group" in payload["error"]


# ---------------------------------------------------------------------------
# Meter, calibration, undo, safety
# ---------------------------------------------------------------------------


def test_meter_runs_while_idle_and_is_refused_during_a_session(server):
    meter = _post(server, "/api/record/meter/start", {})["meter"]
    assert meter["state"] == "running"
    assert [c["channel"] for c in meter["channels"]] == [0, 1]
    assert _get(server, "/api/record/status")["meter"]["state"] == "running"

    # One microphone, one owner: starting a session takes the device from the
    # meter rather than letting two streams fight over it.
    _start(server)
    try:
        code, payload = _post_expect_error(server, "/api/record/meter/start", {})
        assert code == 409
        assert "stop the recording session" in payload["error"]
    finally:
        _post(server, "/api/record/stop", {})
    assert _get(server, "/api/record/status")["meter"]["state"] == "idle"


def test_meter_rejects_a_channel_the_device_does_not_have(server):
    code, payload = _post_expect_error(
        server, "/api/record/meter/start", {"channels": [0, 7], "device": "default"}
    )
    assert code == 400
    assert "input channel" in payload["error"]


def test_calibration_needs_something_listening(server):
    code, payload = _post_expect_error(server, "/api/record/calibrate", {"seconds": 1})
    assert code == 409
    assert "start the meter or a session" in payload["error"]


def test_calibration_runs_from_the_idle_meter(server):
    """The gate has to be set up BEFORE the first take, not during it."""
    _post(server, "/api/record/meter/start", {})
    try:
        data = _post(server, "/api/record/calibrate",
                     {"seconds": 0.2, "strictness": "strict"})
        assert data["calibration"]["state"] == "listening"
        assert data["calibration"]["strictness"] == "strict"
        # The idle meter reports the gate comparison too, not just levels.
        levels = _get(server, "/api/record/levels")
        assert levels["recording"] is False
        assert all("gate_db" in c for c in levels["channels"])
    finally:
        _post(server, "/api/record/meter/stop", {})


def test_calibration_rejects_an_unknown_strictness(server):
    _post(server, "/api/record/meter/start", {})
    try:
        code, payload = _post_expect_error(
            server, "/api/record/calibrate", {"seconds": 1, "strictness": "loose"}
        )
        assert code == 400
        assert "strictness" in payload["error"]
    finally:
        _post(server, "/api/record/meter/stop", {})


def test_levels_endpoint_is_small_enough_to_poll_fast(server):
    """The meters poll many times a second; the full status carries recent
    takes, the log and the run list and is far too heavy for that."""
    _start(server)
    capture = _FakeCapture.instances[-1]
    try:
        _record_until(capture, lambda: _get(server, "/api/record/status")["session"], 1)
        levels = _get(server, "/api/record/levels")
        status = _get(server, "/api/record/status")
        assert levels["recording"] is True
        assert [c["channel"] for c in levels["channels"]] == [0, 1]
        assert len(json.dumps(levels)) < len(json.dumps(status)) / 2
        # And they must agree — same builder, so a channel cannot read
        # differently depending on which poll fetched it.
        assert ([c["channel"] for c in levels["channels"]]
                == [c["channel"] for c in status["session"]["channels"]])
    finally:
        _post(server, "/api/record/stop", {})


def test_calibration_reports_a_suggested_floor(server):
    _start(server)
    capture = _FakeCapture.instances[-1]
    try:
        _post(server, "/api/record/calibrate", {"seconds": 0.4})
        deadline = time.monotonic() + 6.0
        rng = np.random.default_rng(9)
        cal = {}
        while time.monotonic() < deadline:
            capture.feed((rng.standard_normal((BLOCK, 2)) * 0.02).astype("float32"))
            time.sleep(0.003)
            cal = _get(server, "/api/record/status")["session"]["calibration"]
            if cal["state"] in ("done", "failed"):
                break
        assert cal["state"] == "done", cal
        assert -120.0 < cal["suggested_min_flux_db"] < 0.0
    finally:
        _post(server, "/api/record/stop", {})


def test_undo_removes_the_clip_from_disk_and_the_manifest(server):
    _start(server)
    capture = _FakeCapture.instances[-1]
    try:
        snap = _record_until(
            capture, lambda: _get(server, "/api/record/status")["session"], 1
        )
        take = snap["recent"][0]
        path = Path(take["path"])
        assert path.exists()
        assert _post(server, "/api/record/clip/delete",
                     {"uuid": take["uuid"]})["deleted"] == take["uuid"]
        assert not path.exists()
        assert not path.with_suffix(".json").exists()
    finally:
        _post(server, "/api/record/stop", {})

    from sed.data.library import Library

    assert all(c.uuid != take["uuid"] for c in Library().all_clips())


def test_deleting_an_unknown_clip_is_a_404(server):
    code, _payload = _post_expect_error(
        server, "/api/record/clip/delete", {"uuid": "does-not-exist"}
    )
    assert code == 404


def test_asset_routes_refuse_paths_outside_the_library(server, tmp_path):
    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"RIFF")
    for route in ("/audio", "/thumbnail"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                f"{server.base}{route}?path={urllib.parse.quote(str(outside))}",
                timeout=_HTTP_TIMEOUT,
            )
        assert exc.value.code == 403


def test_audio_route_serves_a_recorded_clip(server):
    _start(server)
    capture = _FakeCapture.instances[-1]
    try:
        snap = _record_until(
            capture, lambda: _get(server, "/api/record/status")["session"], 1
        )
        path = snap["recent"][0]["path"]
    finally:
        _post(server, "/api/record/stop", {})
    url = f"{server.base}/audio?path={urllib.parse.quote(path)}"
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as r:
        assert r.status == 200
        assert r.headers["Content-Type"] in ("audio/x-wav", "audio/wav")
        assert len(r.read()) > 0


def test_done_stops_the_session_and_signals_shutdown(server):
    _start(server)
    assert _post(server, "/api/done", {})["ok"] is True
    assert server.shutdown.is_set()
    assert _get(server, "/api/record/status")["recording"] is False


def test_preflight_returns_rows_from_the_existing_checklist(server):
    data = _get(server, "/api/record/preflight?seconds=0.1")
    assert data["ok"] is True
    assert data["worst"] in ("PASS", "WARN", "FAIL")
    names = [c["name"] for c in data["checks"]]
    assert "audio devices" in names
    assert all({"name", "status", "detail", "fix"} <= set(c) for c in data["checks"])


# ---------------------------------------------------------------------------
# Run history: cancel a pass, and delete a finished one
# ---------------------------------------------------------------------------


def test_runs_list_tracks_the_live_pass_and_then_finishes_it(server):
    assert _get(server, "/api/record/status")["runs"] == []

    _start(server, label="pass_one")
    capture = _FakeCapture.instances[-1]
    try:
        _record_until(capture, lambda: _get(server, "/api/record/status")["session"], 1)
        live = _get(server, "/api/record/status")["runs"]
        assert len(live) == 1
        assert live[0]["state"] == "recording"
        assert live[0]["label"] == "pass_one"
        assert live[0]["clips"] >= 1
        assert live[0]["mode"] == "continuous"
        assert live[0]["density"] == "busy"
        # Clip records stay server-side; the browser gets counts, not paths.
        assert "records" not in live[0]
    finally:
        _post(server, "/api/record/stop", {})

    done = _get(server, "/api/record/status")["runs"]
    assert len(done) == 1
    assert done[0]["state"] == "done"
    assert done[0]["ended_at"] is not None
    assert done[0]["clips"] >= 1


def test_a_pass_that_recorded_nothing_is_not_kept(server):
    _start(server, label="empty_pass", mode="manual")
    _post(server, "/api/record/stop", {})
    assert _get(server, "/api/record/status")["runs"] == []


def test_runs_are_listed_newest_first(server):
    for name in ("first", "second"):
        _start(server, label=name)
        capture = _FakeCapture.instances[-1]
        _record_until(capture, lambda: _get(server, "/api/record/status")["session"], 1)
        _post(server, "/api/record/stop", {})
    assert [r["label"] for r in _get(server, "/api/record/status")["runs"]] == [
        "second", "first",
    ]


def test_cancel_deletes_everything_the_pass_recorded(server):
    _start(server, label="bad_pass")
    capture = _FakeCapture.instances[-1]
    snap = _record_until(
        capture, lambda: _get(server, "/api/record/status")["session"], 2
    )
    assert snap["clips_saved"] >= 2
    out = server.tmp_path / "data_library" / "sounds" / "bad_pass"
    assert list(out.glob("*.wav"))

    data = _post(server, "/api/record/cancel", {})
    assert data["discarded"] >= 2
    assert list(out.glob("*.wav")) == []
    assert list(out.glob("*.json")) == []

    status = _get(server, "/api/record/status")
    assert status["recording"] is False
    assert status["runs"] == []          # a discarded pass leaves no history

    from sed.data.library import Library

    assert Library().all_clips() == []


def test_cancel_without_a_session_is_a_409(server):
    code, payload = _post_expect_error(server, "/api/record/cancel", {})
    assert code == 409
    assert "no recording session" in payload["error"]


def test_deleting_a_finished_run_removes_its_clips_only(server):
    _start(server, label="keep_me")
    capture = _FakeCapture.instances[-1]
    _record_until(capture, lambda: _get(server, "/api/record/status")["session"], 1)
    _post(server, "/api/record/stop", {})

    _start(server, label="drop_me")
    capture = _FakeCapture.instances[-1]
    _record_until(capture, lambda: _get(server, "/api/record/status")["session"], 1)
    _post(server, "/api/record/stop", {})

    runs = _get(server, "/api/record/status")["runs"]
    doomed = next(r for r in runs if r["label"] == "drop_me")
    data = _post(server, "/api/record/run/delete", {"id": doomed["id"]})
    assert data["clips"] >= 1

    sounds = server.tmp_path / "data_library" / "sounds"
    assert list((sounds / "drop_me").glob("*.wav")) == []
    assert list((sounds / "keep_me").glob("*.wav"))
    assert [r["label"] for r in _get(server, "/api/record/status")["runs"]] == ["keep_me"]

    from sed.data.library import Library

    assert {c.label for c in Library().all_clips()} == {"keep_me"}


def test_deleting_the_running_pass_is_refused(server):
    _start(server)
    capture = _FakeCapture.instances[-1]
    try:
        _record_until(capture, lambda: _get(server, "/api/record/status")["session"], 1)
        run_id = _get(server, "/api/record/status")["runs"][0]["id"]
        code, payload = _post_expect_error(
            server, "/api/record/run/delete", {"id": run_id}
        )
        assert code == 409
        assert "cancel it instead" in payload["error"]
    finally:
        _post(server, "/api/record/stop", {})


def test_deleting_an_unknown_run_is_a_404(server):
    code, _payload = _post_expect_error(server, "/api/record/run/delete", {"id": "run99"})
    assert code == 404


def test_a_run_spanning_two_labels_reports_and_deletes_both(server):
    _start(server, label="label_a")
    capture = _FakeCapture.instances[-1]
    other = server.tmp_path / "data_library" / "sounds" / "label_b"
    try:
        _record_until(capture, lambda: _get(server, "/api/record/status")["session"], 1)
        _post(server, "/api/record/settings", {"metadata": {"label": "label_b"}})
        deadline = time.monotonic() + 6.0
        rng = np.random.default_rng(21)
        while time.monotonic() < deadline and not list(other.glob("*.json")):
            capture.feed((rng.standard_normal((BLOCK, 2)) * 1e-3).astype("float32"))
            time.sleep(0.002)
    finally:
        _post(server, "/api/record/stop", {})

    run = _get(server, "/api/record/status")["runs"][0]
    assert run["labels"] == ["label_a", "label_b"]

    _post(server, "/api/record/run/delete", {"id": run["id"]})
    sounds = server.tmp_path / "data_library" / "sounds"
    assert list((sounds / "label_a").glob("*.wav")) == []
    assert list((sounds / "label_b").glob("*.wav")) == []


# ---------------------------------------------------------------------------
# Preroll jitter window
# ---------------------------------------------------------------------------


def test_preroll_range_is_tunable_live(server):
    _start(server)
    try:
        _post(server, "/api/record/settings",
              {"onset": {"randomize_preroll": True, "preroll_ms_range": [50, 400]}})
        onset = _get(server, "/api/record/status")["session"]["onset"]
        assert onset["randomize_preroll"] is True
        assert onset["preroll_ms_range"] == [50, 400]
    finally:
        _post(server, "/api/record/stop", {})


def test_preroll_range_must_stay_inside_the_clip(server):
    _start(server)
    try:
        # record_seconds is 0.5 s here, so 600 ms leaves no postroll at all.
        code, payload = _post_expect_error(
            server, "/api/record/settings",
            {"onset": {"randomize_preroll": True, "preroll_ms_range": [100, 600]}},
        )
        assert code == 400
        assert "preroll_ms_range" in payload["error"]
        # Rejected as a whole: neither the window nor the flag is half-applied.
        onset = _get(server, "/api/record/status")["session"]["onset"]
        assert onset["preroll_ms_range"] == list(Config().onset.preroll_ms_range)
        assert onset["randomize_preroll"] is False
    finally:
        _post(server, "/api/record/stop", {})


def test_a_longer_clip_allows_a_wider_jitter_window(server):
    # The window is bounded by the clip, not by a fixed ceiling, so lengthening
    # the clip is what makes a wider spread available.
    _post(server, "/api/record/settings",
          {"onset": {"record_seconds": 3.0,
                     "randomize_preroll": True,
                     "preroll_ms_range": [200, 2500]}})
    onset = _get(server, "/api/record/status")["onset"]
    assert onset["record_seconds"] == 3.0
    assert onset["preroll_ms_range"] == [200, 2500]


def test_preroll_range_is_saved_to_config_toml(server):
    _post(server, "/api/record/settings",
          {"onset": {"record_seconds": 2.0, "randomize_preroll": True,
                     "preroll_ms_range": [150, 1400]}})
    _post(server, "/api/record/save-settings", {"group": "onset"})
    text = server.layout.config_toml.read_text(encoding="utf-8")
    assert "preroll_ms_range = [150, 1400]" in text

    from sed.config import load_config

    assert load_config("rec").onset.preroll_ms_range == (150, 1400)


# ---------------------------------------------------------------------------
# Recording without a project
# ---------------------------------------------------------------------------


@pytest.fixture
def library_server(tmp_path, monkeypatch):
    """A console launched with no project at all — the common case."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    _fake_sounddevice(monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data import record_engine as eng_mod
    from sed.data.record_server import make_handler

    _FakeCapture.instances = []
    monkeypatch.setattr(eng_mod, "AudioCapture", _FakeCapture)

    layout = ProjectLayout("rec")           # exists, but is NOT selected
    layout.create()
    layout.config_toml.write_text(CONFIG_TOML, encoding="utf-8")

    cfg = Config.model_validate({
        "audio": {"record_sample_rate": 16000, "record_channels": [0, 1],
                  "blocksize": 512},
        "model": {"sample_rate": 16000},
        "onset": {"record_seconds": 0.5, "preroll_ms": 150,
                  "randomize_preroll": False},
        "deploy": {"record_device": "UMC404HD"},
    })
    shutdown = threading.Event()
    handler = make_handler(cfg, "", shutdown)
    httpd = ThreadingHTTPServer(("127.0.0.1", _free_port()), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(base=f"http://127.0.0.1:{httpd.server_port}",
                              layout=layout, shutdown=shutdown, handler=handler,
                              tmp_path=tmp_path)
    finally:
        handler.cleanup()
        httpd.shutdown()
        thread.join(timeout=2.0)


def test_no_project_is_the_default_and_records_to_the_shared_pool(library_server):
    from sed.config import LIBRARY_PROJECT_ID

    status = _get(library_server, "/api/record/status")
    assert status["project"] == ""
    assert status["clip_project_id"] == LIBRARY_PROJECT_ID
    assert status["can_save_settings"] is False

    _start(library_server, label="anywhere")
    capture = _FakeCapture.instances[-1]
    try:
        _record_until(
            capture, lambda: _get(library_server, "/api/record/status")["session"], 1
        )
    finally:
        _post(library_server, "/api/record/stop", {})

    meta_file = next(
        (library_server.tmp_path / "data_library" / "sounds" / "anywhere").glob("*.json")
    )
    # `_library` is the id every project's default source rule accepts
    # alongside its own, which is what makes these clips usable everywhere.
    assert json.loads(meta_file.read_text())["project_id"] == LIBRARY_PROJECT_ID


def test_saving_settings_without_a_project_explains_why(library_server):
    code, payload = _post_expect_error(
        library_server, "/api/record/save-settings", {"group": "onset"}
    )
    assert code == 400
    assert "no project selected" in payload["error"]
    # And nothing was invented on disk for the sentinel.
    assert not (library_server.tmp_path / "projects" / "_library").exists()


def test_devices_lists_the_projects_available_to_record_for(library_server):
    data = _get(library_server, "/api/record/devices")
    assert data["project"] == ""
    assert "rec" in data["projects"]


def test_selecting_a_project_stamps_it_and_loads_its_settings(library_server):
    data = _post(library_server, "/api/record/project", {"project": "rec"})
    assert data["project"] == "rec"
    assert data["clip_project_id"] == "rec"

    status = _get(library_server, "/api/record/status")
    assert status["can_save_settings"] is True
    # The project's own capture settings came with it.
    assert _get(library_server, "/api/record/devices")["device"] == "UMC404HD"

    _start(library_server, label="for_project")
    capture = _FakeCapture.instances[-1]
    try:
        _record_until(
            capture, lambda: _get(library_server, "/api/record/status")["session"], 1
        )
    finally:
        _post(library_server, "/api/record/stop", {})
    meta_file = next(
        (library_server.tmp_path / "data_library" / "sounds" / "for_project").glob("*.json")
    )
    assert json.loads(meta_file.read_text())["project_id"] == "rec"


def test_selecting_an_unknown_project_is_a_404(library_server):
    code, _payload = _post_expect_error(
        library_server, "/api/record/project", {"project": "nope"}
    )
    assert code == 404
    assert _get(library_server, "/api/record/status")["project"] == ""


def test_switching_project_mid_session_rolls_the_group(library_server):
    _post(library_server, "/api/record/project", {"project": "rec"})
    _start(library_server, label="switching")
    capture = _FakeCapture.instances[-1]
    try:
        snap = _record_until(
            capture, lambda: _get(library_server, "/api/record/status")["session"], 1
        )
        before = snap["session_id"]
        _post(library_server, "/api/record/project", {"project": ""})
        after = _get(library_server, "/api/record/status")["session"]["session_id"]
        # Clips before and after belong to different groupings, so the
        # recording-group id has to change with them.
        assert after != before
    finally:
        _post(library_server, "/api/record/stop", {})


def test_operator_microphone_names_are_session_metadata(library_server):
    _start(library_server, label="mic_names", microphone_type="rode,samson")
    capture = _FakeCapture.instances[-1]
    try:
        _record_until(
            capture, lambda: _get(library_server, "/api/record/status")["session"], 1
        )
    finally:
        _post(library_server, "/api/record/stop", {})
    meta_file = next(
        (library_server.tmp_path / "data_library" / "sounds" / "mic_names").glob("*.json")
    )
    assert json.loads(meta_file.read_text())["microphone_type"] == ["rode", "samson"]


def test_locations_already_recorded_are_offered_back(library_server):
    _start(library_server, label="loc", location="bar-zone-1")
    capture = _FakeCapture.instances[-1]
    try:
        _record_until(
            capture, lambda: _get(library_server, "/api/record/status")["session"], 1
        )
    finally:
        _post(library_server, "/api/record/stop", {})
    assert "bar-zone-1" in _get(library_server, "/api/record/devices")["locations"]


# ---------------------------------------------------------------------------
# Inspecting a take on the spot
# ---------------------------------------------------------------------------


def _recorded_take(server, label="inspect_me"):
    _start(server, label=label)
    capture = _FakeCapture.instances[-1]
    try:
        snap = _record_until(
            capture, lambda: _get(server, "/api/record/status")["session"], 1
        )
        return snap["recent"][0]
    finally:
        _post(server, "/api/record/stop", {})


def test_thumbnail_and_full_spectrogram_are_served(server):
    take = _recorded_take(server)
    for route, floor in (("/thumbnail", 100), ("/spectrogram", 1000)):
        url = f"{server.base}{route}?path={urllib.parse.quote(take['path'])}"
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as r:
            body = r.read()
        assert r.headers["Content-Type"] == "image/png"
        assert body[:4] == b"\x89PNG"
        # The enlarged one carries axes and labels, so it is materially bigger.
        assert len(body) > floor


def test_clip_check_measures_every_channel(server):
    take = _recorded_take(server, label="checked")
    data = _get(server, "/api/record/clip-check?path="
                + urllib.parse.quote(take["path"]))
    assert data["ok"] is True
    assert [c["channel"] for c in data["channels"]] == [0, 1]
    for ch in data["channels"]:
        assert {"peak_dbfs", "rms_dbfs", "dc_offset", "clipped_samples",
                "crest_db", "hum", "warnings"} <= set(ch)
    assert data["duration_s"] == pytest.approx(0.5, abs=1e-3)


def test_clip_check_refuses_paths_outside_the_library(server, tmp_path):
    outside = tmp_path / "elsewhere.wav"
    outside.write_bytes(b"RIFF")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(
            f"{server.base}/api/record/clip-check?path={urllib.parse.quote(str(outside))}",
            timeout=_HTTP_TIMEOUT,
        )
    assert exc.value.code == 403


def test_a_specific_channel_can_be_rendered(server):
    take = _recorded_take(server, label="two_channel")
    url = (f"{server.base}/spectrogram?path={urllib.parse.quote(take['path'])}"
           "&channel=1")
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as r:
        assert r.read()[:4] == b"\x89PNG"


def test_known_tags_are_offered_with_their_counts(server):
    _start(server, label="tagged", tags="wine-glass,quiet")
    capture = _FakeCapture.instances[-1]
    try:
        _record_until(capture, lambda: _get(server, "/api/record/status")["session"], 1)
    finally:
        _post(server, "/api/record/stop", {})
    tags = {t["tag"]: t["clips"] for t in _get(server, "/api/record/devices")["tags"]}
    assert tags.get("wine-glass", 0) >= 1
    assert tags.get("quiet", 0) >= 1


def test_run_rows_carry_description_and_tags(server):
    _start(server, label="documented", tags="wine-glass",
           description="thin stem, half full", location="bar")
    capture = _FakeCapture.instances[-1]
    try:
        _record_until(capture, lambda: _get(server, "/api/record/status")["session"], 1)
    finally:
        _post(server, "/api/record/stop", {})
    run = _get(server, "/api/record/status")["runs"][0]
    assert run["description"] == "thin stem, half full"
    assert run["tags"] == ["wine-glass"]
    assert run["location"] == "bar"


def test_the_meter_loop_does_not_rebuild_the_panels_it_polls(server):
    """The fast level poll must mutate the meters, not re-create them.

    Rebuilding innerHTML at the poll rate replaced every element many times a
    second: the numbers were unreadable, and the calibration Apply button was
    destroyed between a mousedown and its mouseup, so clicking it did nothing.
    """
    with urllib.request.urlopen(server.base + "/", timeout=_HTTP_TIMEOUT) as r:
        html = r.read().decode("utf-8")

    # Rows are built once and then updated in place.
    assert "function buildMeterRows(" in html
    assert "meterRows.get(c.channel)" in html
    assert "meterSignature" in html

    # The calibration panel only redraws when something actually changed...
    assert "if (key === calKey) return;" in html
    # ...and Apply is delegated, so even a redraw cannot orphan the handler.
    assert 'ev.target.closest("#btn-apply-cal")' in html
    assert "function applyCalibration()" in html

    # Levels are eased towards the measured value rather than snapping.
    assert "function meterSmooth(" in html
    assert "METER_ATTACK" in html and "METER_RELEASE" in html
