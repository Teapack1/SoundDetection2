"""End-to-end smoke test for the interactive inspect server.

We can't load an HTML page in-process, but we can boot the stdlib HTTP
server in a thread and hit the API endpoints to verify the
flag-a-clip round-trip works against real sidecar files.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from sed.config import ProjectLayout


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _seed_clip(layout: ProjectLayout, class_name: str = "clink_wine_thin") -> Path:
    out = layout.positives_dir(class_name)
    out.mkdir(parents=True, exist_ok=True)
    wav = out / "test_clip.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    side = wav.with_suffix(".json")
    side.write_text(json.dumps({
        "uuid": "abc123", "path": str(wav), "class_name": class_name,
        "project_id": "p", "session_id": "s1", "sample_rate": 32000,
        "channels": 1, "duration_s": 1.0, "recorded_at": "2026-01-01T00:00:00Z",
        "device": "", "mic_position": "",
    }))
    return wav


def test_server_flag_endpoint_round_trip(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_clip(layout)

    # Start the server in a thread on a free port
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(cfg=None, project_id="p",
                           render_html_callable=lambda: "<html>ok</html>",
                           layout=layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        # 1. GET / should serve the rendered HTML
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            assert r.status == 200
            assert b"ok" in r.read()

        # 2. POST /api/flag should flag the clip
        body = json.dumps({"path": str(wav), "flagged": True}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/flag", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            payload = json.loads(r.read())
            assert payload["ok"] is True
            assert payload["flagged"] is True

        # 3. Sidecar must now have flagged=True
        meta = json.loads(wav.with_suffix(".json").read_text())
        assert meta.get("flagged") is True

        # 4. /api/flags should list it
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/flags", timeout=2) as r:
            data = json.loads(r.read())
            assert str(wav) in data["flagged"]

        # 5. Unflagging should remove the key
        body2 = json.dumps({"path": str(wav), "flagged": False}).encode()
        req2 = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/flag", data=body2,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req2, timeout=2) as r:
            payload = json.loads(r.read())
            assert payload["ok"] is True
            assert payload["flagged"] is False
        meta = json.loads(wav.with_suffix(".json").read_text())
        assert "flagged" not in meta
    finally:
        httpd.shutdown()


def test_server_rejects_paths_outside_project(tmp_path, monkeypatch):
    """Audio + flag endpoints must NOT serve / mutate arbitrary paths on disk."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    # A WAV outside the project (e.g. attacker-controlled path)
    outside = tmp_path.parent / "evil.wav"
    sf.write(str(outside), np.zeros(100, dtype="float32"), 32000)

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(cfg=None, project_id="p",
                           render_html_callable=lambda: "<html>x</html>",
                           layout=layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        # /audio should refuse the outside path
        url = f"http://127.0.0.1:{port}/audio?path={urllib.parse.quote(str(outside))}"
        try:
            urllib.request.urlopen(url, timeout=2)
            assert False, "should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 403
    finally:
        httpd.shutdown()
        outside.unlink(missing_ok=True)
