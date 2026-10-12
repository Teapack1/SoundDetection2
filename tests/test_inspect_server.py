"""End-to-end smoke test for the interactive inspect server.

We can't load an HTML page in-process, but we can boot the stdlib HTTP
server in a thread and hit the API endpoints (audio, delete,
selection, delete) against real sidecar files.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from sed.config import ProjectLayout


def test_clink_v2_parent_selection_defaults_pitch_off_without_overriding_explicit_choices():
    from sed.config import Config
    from sed.data.inspect_server import _model_selection_augmentation_defaults

    implicit = Config()
    assert _model_selection_augmentation_defaults(
        implicit,
        architecture_id="clink-cnn-v2",
        initialization="parent",
    ) == {"pitch_shift_enabled": False, "overlay_enabled": True}
    assert _model_selection_augmentation_defaults(
        implicit,
        architecture_id="clink-cnn-v2",
        initialization="random",
    ) == {}

    explicit = Config.model_validate({
        "augmentation": {
            "pitch_shift_enabled": True,
            "overlay_enabled": False,
        },
    })
    assert _model_selection_augmentation_defaults(
        explicit,
        architecture_id="clink-cnn-v2",
        initialization="parent",
    ) == {}


def _patch_project_and_library_dirs(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# Several handlers import the scientific stack (torch, scipy, PIL) lazily, so
# whichever request happens to be first in a cold process pays several seconds.
# The assertions here are about behaviour, not latency, and a tight timeout only
# makes them fail in whatever order pytest happens to pick.
_HTTP_TIMEOUT = 30.0


def _post_json(url: str, body: dict, timeout: float = _HTTP_TIMEOUT) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post_json_status(url: str, body: dict, timeout: float = _HTTP_TIMEOUT) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get_json(url: str, timeout: float = _HTTP_TIMEOUT) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _wait_for_job(url: str, *, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = _get_json(url, timeout=timeout)
        job = last.get("job", {})
        if job.get("state") in {"finished", "error", "canceled"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")


def _review_data_via_job(port: int, timeout: float = 10.0, scopes: list[str] | None = None) -> dict:
    """Run a review-data refresh job and return its final payload."""
    body: dict = {"scopes": scopes} if scopes else {}
    started = _post_json(f"http://127.0.0.1:{port}/api/review-data/start", body, timeout=timeout)
    assert started["ok"] is True
    job_id = started["job"]["job_id"]
    final = _wait_for_job(
        f"http://127.0.0.1:{port}/api/review-data/status?job_id={job_id}", timeout=timeout,
    )
    job = final["job"]
    assert job["state"] == "finished", job
    return job["payload"]


def _delete_via_job(port: int, paths: list[str], timeout: float = 5.0) -> dict:
    """Run a delete job for `paths` and return the finished job dict."""
    started = _post_json(f"http://127.0.0.1:{port}/api/delete/start", {"paths": paths}, timeout=timeout)
    assert started["ok"] is True
    job_id = started["job"]["job_id"]
    final = _wait_for_job(
        f"http://127.0.0.1:{port}/api/delete/status?job_id={job_id}", timeout=timeout,
    )
    job = final["job"]
    assert job["state"] == "finished", job
    return job


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return (
        int.from_bytes(data[16:20], "big"),
        int.from_bytes(data[20:24], "big"),
    )


def test_feature_stats_endpoint_fits_standardized_linear_on_training_split_only(
    tmp_path, monkeypatch,
):
    import csv
    from http.server import ThreadingHTTPServer

    from sed import config as cfg_mod
    from sed.config import Config
    from sed.data.inspect_server import make_handler
    from sed.util import sha256_file

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    layout = ProjectLayout("p")
    layout.create()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    rng = np.random.default_rng(12)
    for class_name in ("_negative", "glass"):
        for group_index in range(4):
            wav = audio_dir / f"{class_name}-{group_index}.wav"
            signal = rng.normal(
                loc=0.0,
                scale=0.01 + 0.005 * group_index,
                size=17_640,
            ).astype(np.float32)
            sf.write(wav, signal, 44_100)
            rows.append({
                "path": str(wav),
                "class_name": class_name,
                "source_split_group": f"{class_name}-group-{group_index}",
                "origin": "recorded",
            })
    train_csv = layout.dataset_dir / "train.csv"
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    cfg = Config.model_validate({
        "model": {"sample_rate": 44_100},
        "features": {"window_seconds": 0.4},
        "dataset": {"val_fraction": 0.25},
        "training": {"seed": 43},
    })
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        query = urllib.parse.urlencode({
            "feature_type": "linear",
            "frontend": "efficientat",
            "sample_rate": 44_100,
            "n_fft": 1024,
            "win_length": 882,
            "hop_length": 220,
            "n_mels": 128,
            "fmin": 80,
            "fmax": 20_000,
            "window_seconds": 0.4,
            "stft_engine": "librosa",
            "clips": 2048,
        })
        result = _get_json(
            f"http://127.0.0.1:{port}/api/feature-stats-estimate?{query}",
            timeout=10.0,
        )
    finally:
        httpd.shutdown()

    assert result["ok"] is True
    assert result["clips"] == result["training_rows"] < len(rows)
    assert result["training_groups"] == result["training_rows"]
    assert result["dataset_sha256"] == sha256_file(train_csv)
    assert result["split_seed"] == 43
    assert np.isfinite(result["mean"])
    assert result["std"] > 0.0


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


def _seed_frozen_test_clip(layout: ProjectLayout, class_name: str = "cheers_loud") -> Path:
    out = layout.test_positives_dir(class_name)
    out.mkdir(parents=True, exist_ok=True)
    wav = out / "frozen_clip.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    wav.with_suffix(".json").write_text(json.dumps({
        "uuid": "frozen123",
        "session_id": "frozen-session",
        "origin": "project",
        "augmentation": "none",
        "peak_dbfs": -12.0,
    }))
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    (layout.dataset_dir / "test.csv").write_text(
        "\n".join([
            "path,class_name,session_id,sample_rate,duration_s,is_target,origin",
            f"{wav},{class_name},frozen-session,32000,1.0000,1,project",
        ]) + "\n",
        encoding="utf-8",
    )
    return wav


def _seed_test_predictions(layout: ProjectLayout, wav: Path, run_id: str = "run-1") -> Path:
    model_dir = layout.model_dir(run_id)
    # Make it a recognizable model bundle so registry.get_active_model() sees it.
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "training_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "manifest.json").write_text(json.dumps({
        "model_id": run_id,
        "created_at": f"2026-06-11T12:0{len(run_id) % 10}:00Z",
        "files": {"training_config": "training_config.json"},
    }), encoding="utf-8")
    out = model_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    report = out / "test_predictions.tsv"
    report.write_text(
        "\t".join([
            "index", "file", "true_class", "predicted_class", "correct",
            "predicted_confidence", "triggered", "decision_reason", "decision_threshold",
            "argmax_class", "argmax_confidence", "second_class", "second_confidence",
            "margin", "origin", "session_id", "path",
            "p:_negative", "p:cheers_loud", "p:cheers_quiet",
        ]) + "\n" +
        "\t".join([
            "0", wav.name, "cheers_loud", "_negative", "False",
            "0.052327", "False", "below_threshold", "0.800000",
            "cheers_loud", "0.281352", "cheers_quiet", "0.182395",
            "0.098957", "project", "frozen-session", str(wav),
            "0.052327", "0.281352", "0.182395",
        ]) + "\n",
        encoding="utf-8",
    )
    return report


def _seed_inference_bundle(layout: ProjectLayout, model_id: str, *, exported: bool = True) -> Path:
    model_dir = layout.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "training_config.json").write_text(json.dumps({
        "dataset": {"target_classes": ["_negative", "cheers_loud"]},
        "model": {"classes": ["_negative", "cheers_loud"], "sample_rate": 32000},
        "features": {"frontend": "efficientat", "feature_type": "mel", "n_mels": 128},
    }), encoding="utf-8")
    (model_dir / "manifest.json").write_text(json.dumps({
        "model_id": model_id,
        "created_at": f"2026-06-11T12:0{len(model_id) % 10}:00Z",
        "description": model_id,
        "files": {"checkpoint": "checkpoint_best_epoch-001.ckpt"},
        "test_metrics": {"f1_macro_targets": 0.91},
    }), encoding="utf-8")
    (model_dir / "checkpoint_best_epoch-001.ckpt").write_bytes(b"ckpt")
    if exported:
        (model_dir / "model_fp16.xml").write_text("<xml/>", encoding="utf-8")
        (model_dir / "model_fp16.bin").write_bytes(b"bin")
    return model_dir


def _write_inference_project_config(layout: ProjectLayout, *, active: str = "") -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.config_toml.write_text(
        "\n".join([
            "[project]",
            f'id = "{layout.id}"',
            "",
            "[dataset]",
            'target_classes = ["_negative", "cheers_loud"]',
            "",
            "[deploy]",
            f'active_model = "{active}"',
            'record_device = "UMC404HD"',
            "",
            "[audio]",
            "record_sample_rate = 48000",
            "record_channels = [0, 1, 2]",
            'microphone_type = ["rode", "samson", "joby"]',
            "",
            "[inference]",
            "inference_channels = [0, 1, 2]",
            'multi_mic_mode = "best"',
            "",
            "[targets]",
            'cheers_loud = { enabled = true, threshold = 0.6 }',
        ]) + "\n",
        encoding="utf-8",
    )


def _fake_sounddevice(monkeypatch, *, include_umc: bool = True, default_index: int = 0):
    devices = [
        {"name": "default", "max_input_channels": 8, "default_samplerate": 48000, "hostapi": 0},
    ]
    if include_umc:
        devices.append({"name": "UMC404HD", "max_input_channels": 4, "default_samplerate": 48000, "hostapi": 0})

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

    fake_sd = SimpleNamespace(
        default=SimpleNamespace(device=[default_index, None]),
        query_devices=lambda idx=None: devices if idx is None else devices[int(idx)],
        query_hostapis=lambda: [{"name": "ALSA"}],
        InputStream=FakeStream,
        PortAudioError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    return fake_sd


def _seed_mined_clip(layout: ProjectLayout, class_name: str = "cheers_loud") -> Path:
    from sed.config import LIBRARY_DIR
    out = LIBRARY_DIR / "mined" / class_name
    out.mkdir(parents=True, exist_ok=True)
    wav = out / "mined_clip.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    wav.with_suffix(".json").write_text(json.dumps({
        "schema_version": 2,
        "category": "mined",
        "label": class_name,
        "project_id": layout.id,
        "dataset_role": "ignore",
        "density": "normal",
        "review_status": "mined",
        "uuid": "mined_clip",
        "origin": "runtime_mining",
        "created_at": "2026-06-17T22:19:27Z",
        "predicted_class": class_name,
        "confidence": 0.92,
        "peak_dbfs": -11.0,
        "rms_dbfs": -30.0,
        "probs_percent": {
            "_negative": 3.0,
            "cheers_loud": 92.0,
            "cheers_quiet": 8.0,
        },
        "smoothed_percent": {
            "_negative": 2.0,
            "cheers_loud": 88.0,
            "cheers_quiet": 6.0,
        },
    }))
    return wav


def test_server_background_delete_job_reports_progress_and_deleted_paths(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.config import Config
    from sed.data.library import Library
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_clip(layout)
    sidecar = wav.with_suffix(".json")
    composed = f"path,class_name\n{wav.resolve()},clink_wine_thin\n"
    (layout.dataset_dir / "train.csv").write_text(composed, encoding="utf-8")
    monkeypatch.setattr(
        Library, "rebuild_manifest",
        lambda self: (_ for _ in ()).throw(AssertionError("full rebuild invoked")),
    )

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=Config(),
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        started = _post_json(
            f"http://127.0.0.1:{port}/api/delete/start",
            {"paths": [str(wav)]},
        )
        assert started["ok"] is True
        job_id = started["job"]["job_id"]

        final = _wait_for_job(
            f"http://127.0.0.1:{port}/api/delete/status?job_id={job_id}",
        )
        job = final["job"]
        assert job["state"] == "finished"
        assert job["total"] == 1
        assert job["completed"] == 1
        assert job["deleted_count"] == 1
        assert str(wav.resolve()) in job["deleted"]
        assert job["affected_scopes"]
        assert job["deleted_records"][0]["path"] == str(wav.resolve())
        assert set(job["phase_timings"]) >= {"unlink", "indexes", "project_metadata", "cache_cleanup"}
        assert job["compose_state"]["dirty"] is True
        assert (layout.dataset_dir / "train.csv").read_text(encoding="utf-8") == composed
        assert not wav.exists()
        assert not sidecar.exists()
    finally:
        httpd.shutdown()


def test_activity_endpoint_reports_jobs_and_resources(tmp_path, monkeypatch):
    """The redesigned top bar polls one aggregate endpoint: it must always
    return a jobs list plus a resource block with the load-meter fields."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=None,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        snap = _get_json(f"http://127.0.0.1:{port}/api/activity")
        assert snap["ok"] is True
        assert isinstance(snap["jobs"], list)
        assert snap["jobs"] == []  # nothing running on a fresh server
        res = snap["resources"]
        for key in (
            "cpu_count", "loadavg", "load_frac", "active_threads",
            "training_active", "train_workers", "level",
        ):
            assert key in res, key
        assert res["cpu_count"] >= 1
        assert 0.0 <= res["load_frac"] <= 1.0
        assert res["level"] in {"free", "busy", "heavy"}
        assert len(res["loadavg"]) == 3
        assert res["training_active"] is False
    finally:
        httpd.shutdown()


def test_unchanged_inspect_metadata_cache_is_not_rewritten(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.config import ProjectLayout
    from sed.data.inspect import _init_metadata_cache, _save_metadata_cache

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("p")
    layout.create()
    cache_path = layout.dataset_inspect_dir / "inspect_metadata_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text('{"/clip.wav":{"duration":0.4}}', encoding="utf-8")
    before = cache_path.stat().st_mtime_ns

    cache = _init_metadata_cache(layout)
    _save_metadata_cache(layout, cache)

    assert cache_path.stat().st_mtime_ns == before


def test_activity_endpoint_surfaces_a_finished_delete_job(tmp_path, monkeypatch):
    """A finished job lingers briefly in the snapshot (finished=True) so the
    bar can fire exactly one completion toast before it disappears."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_clip(layout)

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=None,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        _delete_via_job(port, [str(wav)])
        snap = _get_json(f"http://127.0.0.1:{port}/api/activity")
        delete_entries = [j for j in snap["jobs"] if j["type"] == "delete"]
        assert len(delete_entries) == 1
        assert delete_entries[0]["finished"] is True
    finally:
        httpd.shutdown()


def test_removed_legacy_routes_return_404(tmp_path, monkeypatch):
    """Deleted endpoints must stay deleted (the UI only uses the async
    /start + /status job variants and /api/inference/settings)."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()

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
        removed = [
            ("GET", "/api/review-data"),
            ("GET", "/api/previews/status"),
            ("GET", "/api/models"),
            ("GET", "/api/inference/devices"),
            ("POST", "/api/delete"),
            ("POST", "/api/compose"),
            ("POST", "/api/previews/start"),
            ("POST", "/api/previews/cancel"),
            ("POST", "/api/clip-role"),
            ("POST", "/api/model-scaffold"),
        ]
        for method, path in removed:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=b"{}" if method == "POST" else None,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(req, timeout=2)
            assert excinfo.value.code == 404, f"{method} {path} should be gone"
    finally:
        httpd.shutdown()


def test_inference_models_endpoint_marks_exported_and_checkpoint_only_models(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout, active="run-exported")
    _seed_inference_bundle(layout, "run-exported", exported=True)
    _seed_inference_bundle(layout, "run-checkpoint-only", exported=False)

    from sed.config import Config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    port = _free_port()
    handler = make_handler(
        cfg=Config(),
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        payload = _get_json(f"http://127.0.0.1:{port}/api/inference/models", timeout=10.0)
        assert payload["ok"] is True
        assert payload["active"] == "run-exported"
        assert payload["selected"] == "run-exported"
        rows = {row["model_id"]: row for row in payload["models"]}
        assert rows["run-exported"]["can_run"] is True
        assert rows["run-exported"]["has_openvino"] is True
        assert rows["run-exported"]["openvino_precision"] == "fp16"
        assert rows["run-exported"]["engines"] == ["openvino:fp16", "checkpoint"]
        assert rows["run-exported"]["default_engine"] == "openvino:fp16"
        # Runtime parity: a never-exported run is still live-runnable through
        # its PyTorch checkpoint (sed.runtime.backends chain).
        assert rows["run-checkpoint-only"]["can_run"] is True
        assert rows["run-checkpoint-only"]["engines"] == ["checkpoint"]
        assert rows["run-checkpoint-only"]["disabled_reason"] == ""
    finally:
        httpd.shutdown()


def test_inference_models_endpoint_empty_selection_without_runnable_models(tmp_path, monkeypatch):
    """No phantom auto-select: nothing runnable → selected stays empty."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout, active="")
    model_dir = _seed_inference_bundle(layout, "run-broken", exported=False)
    (model_dir / "training_config.json").unlink()
    (model_dir / "checkpoint_best_epoch-001.ckpt").unlink()

    from sed.config import Config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    port = _free_port()
    handler = make_handler(
        cfg=Config(),
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        payload = _get_json(f"http://127.0.0.1:{port}/api/inference/models", timeout=10.0)
        assert payload["ok"] is True
        assert payload["selected"] == ""
        rows = {row["model_id"]: row for row in payload["models"]}
        assert rows["run-broken"]["can_run"] is False
        assert rows["run-broken"]["disabled_reason"] == "missing training_config.json"
    finally:
        httpd.shutdown()


def test_inference_subtab_contains_cli_parity_controls():
    html_source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    for marker in [
        "train-infer-mode",
        "train-infer-trigger-onset",
        "train-infer-min-dbfs",
        "train-infer-targets",
        "train-infer-channels",
        "train-infer-save-input",
        "train-infer-mine",
        "train-infer-artnet",
    ]:
        assert marker in html_source


def test_inference_device_resolution_reports_missing_device_without_fallback(tmp_path, monkeypatch):
    """Device resolution payload (served via /api/inference/settings)."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    _fake_sounddevice(monkeypatch, include_umc=True, default_index=0)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout, active="")

    from sed.config import load_config
    from sed.data.inspect_server import _inference_device_payload

    cfg = load_config("p")
    ok_payload = _inference_device_payload(cfg)
    assert ok_payload["ok"] is True
    assert ok_payload["configured_device"] == "UMC404HD"
    assert ok_payload["resolved_index"] == 1
    assert ok_payload["warnings"] == []

    missing = _inference_device_payload(cfg, "Missing")
    assert missing["ok"] is True
    assert missing["resolved_index"] is None
    assert "No input device matching 'Missing'" in missing["resolved_error"]
    assert missing["warnings"] == ["configured device not found: Missing"]


def test_inference_settings_endpoint_reports_runtime_controls(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    _fake_sounddevice(monkeypatch, include_umc=True, default_index=0)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout, active="run-exported")
    _seed_inference_bundle(layout, "run-exported", exported=True)

    from sed.config import load_config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    cfg = load_config("p")
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        payload = _get_json(f"http://127.0.0.1:{port}/api/inference/settings")
        assert payload["ok"] is True
        assert payload["selected_model"] == "run-exported"
        assert payload["classes"] == ["_negative", "cheers_loud"]
        assert payload["active_targets"] == ["cheers_loud"]
        assert payload["inference"]["channels"] == [0, 1, 2]
        assert payload["inference"]["multi_mic_mode"] == "best"
        assert payload["device"]["resolved_index"] == 1
        assert payload["device"]["input_channels"][2]["selected"] is True
        assert payload["thresholds"]["cheers_loud"] == 0.6
    finally:
        httpd.shutdown()


def test_inference_rejects_single_mode_with_multiple_channels(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    _fake_sounddevice(monkeypatch, include_umc=True, default_index=0)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout, active="run-exported")
    _seed_inference_bundle(layout, "run-exported", exported=True)

    from sed.config import load_config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    cfg = load_config("p")
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        status, payload = _post_json_status(
            f"http://127.0.0.1:{port}/api/inference/start",
            {
                "model_id": "run-exported",
                "device": "UMC404HD",
                "channels": [0, 1],
                "multi_mic_mode": "single",
            },
        )
        assert status == 400
        assert "single microphone mode" in payload["error"]
    finally:
        cleanup = getattr(handler, "cleanup", None)
        if callable(cleanup):
            cleanup()
        httpd.shutdown()


def test_inference_save_input_setup_persists_only_input_topology(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    _fake_sounddevice(monkeypatch, include_umc=True, default_index=0)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout, active="run-exported")
    _seed_inference_bundle(layout, "run-exported", exported=True)

    from sed.config import load_config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    cfg = load_config("p")
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        payload = _post_json(
            f"http://127.0.0.1:{port}/api/inference/save-input-setup",
            {
                "device": "1",
                "channels": [1, 2],
                "multi_mic_mode": "mean",
                "onset_gating_enabled": True,
            },
        )
        assert payload["ok"] is True
        saved = load_config("p")
        assert saved.deploy.active_model == "run-exported"
        assert saved.deploy.record_device == "1"
        assert saved.inference.inference_channels == [1, 2]
        assert saved.inference.multi_mic_mode == "mean"
        assert saved.inference.onset_gating_enabled is True
    finally:
        httpd.shutdown()


def test_inference_meter_uses_selected_channels(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    _fake_sounddevice(monkeypatch, include_umc=True, default_index=0)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout, active="run-exported")
    _seed_inference_bundle(layout, "run-exported", exported=True)

    from sed.config import load_config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    cfg = load_config("p")
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        started = _post_json(
            f"http://127.0.0.1:{port}/api/inference/meter/start",
            {"device": "UMC404HD", "channels": [1, 2]},
        )
        assert started["ok"] is True
        assert started["meter"]["channels"] == [1, 2]
        stopped = _post_json(f"http://127.0.0.1:{port}/api/inference/meter/stop", {})
        assert stopped["ok"] is True
    finally:
        cleanup = getattr(handler, "cleanup", None)
        if callable(cleanup):
            cleanup()
        httpd.shutdown()


def test_inference_lifecycle_uses_selected_model_without_setting_active(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    _fake_sounddevice(monkeypatch, include_umc=True, default_index=0)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout, active="run-exported")
    _seed_inference_bundle(layout, "run-exported", exported=True)
    _seed_inference_bundle(layout, "run-candidate", exported=True)

    from sed.config import load_config
    from sed.data.inspect_server import make_handler
    from sed.runtime import infer as infer_mod
    from http.server import ThreadingHTTPServer

    class FakeRuntime:
        def __init__(
            self, cfg, project_id, on_trigger=None, on_frame=None,
            on_detection=None, active_trigger_classes=None,
            detection_classes=None, live_monitor=False,
            model_precision=None,
        ):
            self.cfg = cfg
            self.project_id = project_id
            self.on_trigger = on_trigger
            self.on_frame = on_frame
            self.on_detection = on_detection
            self.active_trigger_classes = active_trigger_classes
            self.detection_classes = detection_classes
            self.live_monitor = live_monitor
            self.classes = list(cfg.model.classes)
            self.stats = {
                "last_probs": {"_negative": 0.02, "cheers_loud": 0.92},
                "last_smoothed": {"_negative": 0.02, "cheers_loud": 0.92},
                "class_thresholds": {"_negative": 0.5, "cheers_loud": 0.6},
                "negative_max": 0.5,
                "infer_ms_avg": 1.2,
                "mic_status": {
                    0: {
                        "channel": 0,
                        "label": "rode",
                        "peak_dbfs": -12.0,
                        "rms_dbfs": -24.0,
                        "top_class": "cheers_loud",
                        "top_score": 0.92,
                        "detections": 1,
                    }
                },
            }
            self.capture = SimpleNamespace(stats={"xruns": 0, "blocks": 3, "reconnects": 0})
            self.stopped = False

        def start(self):
            if self.on_frame:
                self.on_frame({"_negative": 0.02, "cheers_loud": 0.92})
            if self.on_detection:
                best_channel = self.cfg.inference.inference_channels[0]
                self.on_detection(SimpleNamespace(
                    class_name="cheers_loud",
                    confidence=0.92,
                    timestamp=time.monotonic(),
                    kind="trigger",
                    probs={"_negative": 0.02, "cheers_loud": 0.92},
                    smoothed={"_negative": 0.02, "cheers_loud": 0.92},
                    audio=np.zeros((32000, 1), dtype="float32"),
                    sample_rate=32000,
                    active_channels=list(self.cfg.inference.inference_channels),
                    best_channel=best_channel,
                    peak_dbfs=-12.0,
                    rms_dbfs=-24.0,
                    min_dbfs=self.cfg.inference.min_dbfs,
                    active_targets=sorted(self.active_trigger_classes or []),
                ))
            if self.on_trigger:
                self.on_trigger(SimpleNamespace(
                    class_name="cheers_loud",
                    confidence=0.92,
                    best_channel=self.cfg.inference.inference_channels[0],
                    peak_dbfs=-12.0,
                    rms_dbfs=-24.0,
                ))

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(infer_mod, "RealtimeInference", FakeRuntime)

    cfg = load_config("p")
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        started = _post_json(
            f"http://127.0.0.1:{port}/api/inference/start",
            {
                "model_id": "run-candidate",
                "device": "UMC404HD",
                "channels": [1],
                "multi_mic_mode": "single",
                "onset_gating_enabled": True,
                "min_dbfs": -45.0,
                "targets": ["cheers_loud"],
            },
        )
        assert started["ok"] is True

        final = {}
        deadline = time.time() + 3.0
        while time.time() < deadline:
            final = _get_json(f"http://127.0.0.1:{port}/api/inference/status")
            if final["inference"]["state"] == "running":
                break
            time.sleep(0.05)
        inf = final["inference"]
        assert inf["state"] == "running"
        assert inf["model_id"] == "run-candidate"
        assert inf["safe_test"] is True
        assert inf["artnet"] is False
        assert inf["mining"] is False
        assert inf["stats"]["last_smoothed"]["cheers_loud"] == 0.92
        assert inf["detections"][-1]["class"] == "cheers_loud"
        assert inf["detections"][-1]["probs"]["cheers_loud"] == 0.92
        assert inf["detections"][-1]["mic"] == 1
        assert inf["settings"]["channels"] == [1]
        assert inf["settings"]["multi_mic_mode"] == "single"
        assert inf["settings"]["onset_gating_enabled"] is True
        assert inf["settings"]["min_dbfs"] == -45.0

        stopped = _post_json(f"http://127.0.0.1:{port}/api/inference/stop", {})
        assert stopped["ok"] is True

        # Starting a candidate for test does not mutate deploy.active_model.
        assert load_config("p").deploy.active_model == "run-exported"
    finally:
        cleanup = getattr(handler, "cleanup", None)
        if callable(cleanup):
            cleanup()
        httpd.shutdown()


def test_done_endpoint_replaces_soft_exclusion_manifest(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    keep = _seed_clip(layout, class_name="keep")
    drop = _seed_clip(layout, class_name="drop")

    from sed.data.compose import exclusion_manifest_path, write_excluded_clip_paths
    write_excluded_clip_paths(layout, [keep])

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    shutdown = threading.Event()
    handler = make_handler(
        cfg=None,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
        shutdown_event=shutdown,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        body = json.dumps({"exclude_paths": [str(drop)]}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/done",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            payload = json.loads(r.read())
            assert payload["ok"] is True
            assert payload["count"] == 1

        manifest = json.loads(exclusion_manifest_path(layout).read_text(encoding="utf-8"))
        assert manifest["paths"] == [str(drop.resolve())]
        assert str(keep.resolve()) not in manifest["paths"]
    finally:
        httpd.shutdown()


def test_server_rejects_paths_outside_project(tmp_path, monkeypatch):
    """Audio + delete endpoints must NOT serve / mutate arbitrary paths on disk."""
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


def test_mined_panels_show_probability_metadata(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_mined_clip(layout)

    from sed.data.inspect import _build_panels_from_mined
    panels = _build_panels_from_mined(layout, output_dir="mined")

    card = panels["cheers_loud"][0]
    assert card["path"] == str(wav)
    assert card["confidence"] == 0.92
    assert "88.0%" in card["prob_html"]
    assert "audio:" in card["meta_html"]
    assert "length 1.00 s" in card["meta_html"]
    assert "sample rate 32,000 Hz" in card["meta_html"]
    assert "peak -11.0 dBFS" in card["meta_html"]


def test_frozen_test_panels_join_composed_test_and_predictions(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_frozen_test_clip(layout)
    _seed_test_predictions(layout, wav)

    from sed.data.inspect import _build_panels_from_frozen_test
    panels, run_id, has_predictions, context = _build_panels_from_frozen_test(layout)

    assert run_id == "run-1"
    assert has_predictions is True
    # The current composed CSV remains the card/label authority; the saved
    # report is only joined as prediction metadata.
    assert context["source"] == "current_project_dataset"
    card = panels["cheers_loud"][0]
    assert card["path"] == str(wav)
    assert card["true_class"] == "cheers_loud"
    assert card["predicted_class"] == "_negative"
    assert card["correct"] is False
    assert card["triggered"] is False
    assert card["decision_reason"] == "below_threshold"
    assert abs(card["probs_percent"]["cheers_loud"] - 28.1352) < 1e-9
    assert "argmax cheers_loud" not in card["meta_html"]
    assert "28.1%" not in card["meta_html"]
    assert "margin 9.9 pp" in card["meta_html"]
    assert "audio:" in card["meta_html"]
    assert "length 1.00 s" in card["meta_html"]
    assert "sample rate 32,000 Hz" in card["meta_html"]


def test_card_metadata_includes_description_and_tags(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_clip(layout)
    meta_path = wav.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["description"] = "half-full glass near the bar"
    meta["tags"] = ["close_mic", "clean"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    from sed.data.inspect import _build_panels
    panels = _build_panels(
        items=[(wav, "clink_wine_thin", meta, 1.0)],
        classes=["clink_wine_thin"],
        max_per_class=10,
    )

    card = panels["clink_wine_thin"][0]
    assert "audio:" in card["meta_html"]
    assert "length 1.00 s" in card["meta_html"]
    assert "sample rate 32,000 Hz" in card["meta_html"]
    assert "half-full glass near the bar" in card["meta_html"]
    assert "#close_mic" in card["meta_html"]
    assert "half-full glass" in card["search_text"]
    assert "close_mic" in card["search_text"]


def test_zero_max_per_class_means_uncapped_for_interactive_panels(tmp_path, monkeypatch):
    from sed.data import inspect as inspect_mod

    monkeypatch.setattr(inspect_mod, "_get_cached_metadata", lambda _path, _cache: (16000, 1.0))
    items = [
        (tmp_path / f"clip_{i:03d}.wav", "glass", {"session_id": f"s{i:03d}"}, 1.0)
        for i in range(205)
    ]

    uncapped = inspect_mod._build_panels(
        items=items,
        classes=["glass"],
        max_per_class=0,
        is_rerender=True,
        interactive=True,
        metadata_cache={},
    )
    capped = inspect_mod._build_panels(
        items=items,
        classes=["glass"],
        max_per_class=200,
        is_rerender=True,
        interactive=True,
        metadata_cache={},
    )

    assert len(uncapped["glass"]) == 205
    assert len(capped["glass"]) == 200


def test_render_html_includes_frozen_test_tab_before_mined(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_frozen_test_clip(layout)
    _seed_test_predictions(layout, wav)

    from sed.data.inspect import _build_panels_from_frozen_test, _render_html
    frozen_panels, run_id, has_predictions, context = _build_panels_from_frozen_test(layout)
    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        aug_panels={},
        frozen_panels=frozen_panels,
        frozen_predictions_run_id=run_id,
        frozen_has_predictions=has_predictions,
        frozen_context=context,
        mined_panels={},
        interactive=True,
    )

    assert html.index("data-tab=\"frozen\"") < html.index("data-tab=\"mined\"")
    assert "Frozen test" in html
    assert "Incorrect label" in html
    assert "Predicted _negative" in html
    assert "Predicted Label" in html
    assert "id=\"frozen-search\"" in html
    assert "promote-mined-btn" not in html
    assert "mined-label" not in html
    assert "Promote selected to negatives" not in html
    assert "id=\"frozen-run-select\"" in html
    assert "run-1" in html
    assert "id=\"frozen-metric\"" in html
    assert "id=\"frozen-range-mode\"" in html
    assert "id=\"frozen-min\"" in html
    assert "id=\"frozen-max\"" in html
    assert "type=\"number\"" in html
    assert "placeholder=\"90\"" in html
    assert "max=\"100\"" in html
    assert "/audio?path=" in html
    assert "data-frozen-pred=\"_negative\"" in html
    assert "data-frozen-predicted-confidence=\"0.052327\"" in html
    assert "data-frozen-argmax-confidence=\"0.281352\"" in html
    assert "data-frozen-margin=\"0.098957\"" in html
    assert "below_threshold" in html
    assert "class=\"aug-filter " in html
    assert "id=\"mask-metric\"" in html
    assert "True _negative" not in html
    assert "No trigger" not in html


def test_render_html_includes_frozen_run_selector_without_models(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    _seed_frozen_test_clip(layout)

    from sed.data.inspect import _build_panels_from_frozen_test, _render_html
    frozen_panels, run_id, has_predictions, context = _build_panels_from_frozen_test(layout)
    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        aug_panels={},
        frozen_panels=frozen_panels,
        frozen_predictions_run_id=run_id,
        frozen_has_predictions=has_predictions,
        frozen_context=context,
        mined_panels={},
        interactive=True,
    )

    assert "id=\"frozen-run-select\"" in html
    assert "No trained models yet" in html
    assert "id=\"frozen-evaluate-btn\"" in html
    assert "id=\"frozen-analyze-btn\"" in html
    assert "Explain predictions" in html
    assert "Evaluation result" in html
    assert "up to ~1 min" not in html
    assert "id=\"frozen-insight\"" in html
    assert "class=\"explain-overlay\"" in html
    assert "disabled" in html


def test_frozen_run_endpoint_updates_predictions_without_html_rerender(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_frozen_test_clip(layout)
    _seed_test_predictions(layout, wav, run_id="run-selected")
    override = _seed_test_predictions(layout, wav, run_id="run-override")
    text = override.read_text(encoding="utf-8").replace("_negative", "cheers_loud")
    text = text.replace("\tFalse\t0.052327", "\tTrue\t0.052327")
    override.write_text(text, encoding="utf-8")

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=None,
        project_id="p",
        render_html_callable=lambda **_kwargs: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        url = f"http://127.0.0.1:{port}/api/frozen-run?run=run-override"
        with urllib.request.urlopen(url, timeout=2) as r:
            payload = json.loads(r.read())

        card = payload["cards"][str(wav)]
        assert payload["run_id"] == "run-override"
        assert card["predicted_class"] == "cheers_loud"
        assert card["correct"] is True
        assert "frozen-pred-meta" in card["prediction_html"]
        assert "prob-row" in card["prob_html"]
        assert "data-prob-class='cheers_loud'" in card["prob_html"]
        assert "class='prob-row explainable'" in card["prob_html"]
    finally:
        httpd.shutdown()


def test_frozen_run_hides_predictions_from_a_different_test_hash(tmp_path, monkeypatch):
    """A path match is not enough: predictions are valid only for the frozen
    set identity on which the run was evaluated."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_frozen_test_clip(layout)
    _seed_test_predictions(layout, wav, run_id="run-stale")
    (layout.dataset_dir / "dataset_snapshot.json").write_text(
        json.dumps({"frozen_test": {"hash": "sha256:current"}}), encoding="utf-8"
    )
    run_dataset = layout.model_dir("run-stale") / "dataset"
    run_dataset.mkdir(parents=True, exist_ok=True)
    (run_dataset / "dataset_snapshot.json").write_text(
        json.dumps({"frozen_test": {"hash": "sha256:old"}}), encoding="utf-8"
    )

    from sed.data.inspect import _build_panels_from_frozen_test, frozen_run_prediction_payload

    panels, run_id, has_predictions, context = _build_panels_from_frozen_test(
        layout, run_id="run-stale"
    )
    payload = frozen_run_prediction_payload(layout, run_id="run-stale")

    assert run_id == "run-stale"
    assert has_predictions is False
    assert context["predictions_match_current"] is False
    assert "different frozen test set" in context["prediction_warning"]
    assert panels["cheers_loud"][0]["predicted_class"] == "not_evaluated"
    assert payload["has_predictions"] is False
    assert payload["cards"][str(wav)]["has_prediction"] is False


def test_frozen_rule_state_prefers_fresh_report_and_marks_run_partition_overlap(tmp_path, monkeypatch):
    """A later Frozen-tab rerun must supersede a stale post-training artifact
    for the same rule, while the composed test CSV remains fully visible."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    train_wav = _seed_frozen_test_clip(layout)
    val_wav = train_wav.with_name("frozen_unscored.wav")
    sf.write(str(val_wav), np.zeros(32000, dtype="float32"), 32000)
    val_wav.with_suffix(".json").write_text(json.dumps({"uuid": "unscored"}))
    with open(layout.dataset_dir / "test.csv", "a", encoding="utf-8") as f:
        f.write(f"{val_wav},cheers_loud,val-session,32000,1.0000,1,project\n")
    report = _seed_test_predictions(layout, train_wav, run_id="run-1")
    (layout.dataset_dir / "dataset_snapshot.json").write_text(
        json.dumps({"frozen_test": {"hash": "sha256:current"}}), encoding="utf-8"
    )
    run_dir = layout.model_dir("run-1")
    (run_dir / "training_config.json").write_text(json.dumps({
        "training": {"auto_test_rules": ["fixed@0.90", "val_f1"]},
    }), encoding="utf-8")
    run_dataset = run_dir / "dataset"
    run_dataset.mkdir(parents=True, exist_ok=True)
    (run_dataset / "dataset_snapshot.json").write_text(
        json.dumps({"frozen_test": {"hash": "sha256:old"}}), encoding="utf-8"
    )
    (run_dataset / "train.csv").write_text(
        f"path,class_name\n{train_wav},cheers_loud\n", encoding="utf-8"
    )
    (run_dataset / "val.csv").write_text(
        f"path,class_name\n{val_wav},cheers_loud\n", encoding="utf-8"
    )
    stale_report = report.with_name("fixed_0_90_predictions.tsv")
    stale_report.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    stale_report.with_name("fixed_0_90_metrics.json").write_text(json.dumps({
        "rule": {"id": "fixed@0.90", "label": "≥ 0.90"},
        "test_set_hash": "sha256:old", "n_samples": 1,
    }), encoding="utf-8")
    report.with_name("test_metrics.json").write_text(json.dumps({
        "rule": {"id": "fixed@0.90", "label": "≥ 0.90"},
        "score_profile": "fixed_threshold_benchmark", "threshold": 0.90,
        "test_set_hash": "sha256:current", "n_samples": 1,
    }), encoding="utf-8")

    from sed.data.inspect import frozen_run_prediction_payload
    payload = frozen_run_prediction_payload(layout, "run-1", rule_id="fixed@0.90")
    context = payload["context"]

    assert context["selected_rule_id"] == "fixed@0.90"
    assert context["selected_evaluation"]["status"] == "ready"
    assert context["selected_evaluation"]["origin"] == "active review rerun"
    assert context["prediction_warning"] == ""
    assert context["frozen_sample_count"] == 2
    assert context["evaluated_sample_count"] == 1
    assert context["unevaluated_sample_count"] == 1
    assert context["training_overlap_count"] == 1
    assert context["validation_overlap_count"] == 1
    assert payload["cards"][str(train_wav)]["was_in_run_train"] is True
    assert payload["cards"][str(val_wav)]["was_in_run_val"] is True
    statuses = {row["rule_id"]: row["status"] for row in context["evaluation_options"]}
    assert statuses["fixed@0.90"] == "ready"
    assert statuses["val_f1"] == "missing"


def test_frozen_score_ranges_use_percentages_and_exclude_unscored_cards(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered inspect JavaScript")
    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p", class_summary={}, panels={}, interactive=True,
        model_classes=["_negative", "cheers"], library_items=[],
    )
    script = "\n".join(part.split("</script>", 1)[0] for part in html.split("<script>")[1:])
    start = script.index("function frozenScoreValue")
    end = script.index("function frozenVerdictHtml", start)
    range_js = script[start:end]
    harness = f"""
function assert(ok, message) {{ if (!ok) throw new Error(message); }}
{range_js}
const scored = {{ predicted_confidence: 0.93, argmax_confidence: 0.95, margin: 0.08 }};
const missing = {{ predicted_confidence: null, argmax_confidence: null, margin: null }};
assert(frozenMatchesScoreRange(scored, "predicted-confidence", "at-least", 90, null), "93% should satisfy at least 90%");
assert(!frozenMatchesScoreRange(scored, "predicted-confidence", "at-most", null, 90), "93% should not satisfy at most 90%");
assert(frozenMatchesScoreRange(scored, "margin", "between", 5, 10), "8 percentage-point margin should be in range");
assert(!frozenMatchesScoreRange(missing, "predicted-confidence", "at-least", 90, null), "unscored cards must not leak into a bounded review");
assert(frozenMatchesScoreRange(missing, "predicted-confidence", "any", null, null), "unscored cards remain visible when no score range is active");
"""
    script_path = tmp_path / "frozen-score-ranges.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_frozen_run_switch_replaces_all_prediction_state(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered inspect JavaScript")
    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    script = "\n".join(part.split("</script>", 1)[0] for part in html.split("<script>")[1:])
    start = script.index("function applyFrozenPrediction")
    end = script.index("// --- Frozen filters events", start)
    merge_js = script[start:end]
    harness = f"""
function assert(ok, message) {{ if (!ok) throw new Error(message); }}
{merge_js}
const card = {{
  correct: false, predicted_class: "old", has_prediction: true,
  prediction_html: "old html", sample_search_text: "clip.wav true_class tag",
  search_text: "clip.wav old",
}};
applyFrozenPrediction(card, {{
  correct: true, predicted_class: "new", triggered: true, has_prediction: true,
  decision_reason: "threshold", decision_threshold: 0.9,
  predicted_confidence: 0.96, argmax_class: "new", argmax_confidence: 0.96,
  second_class: "other", second_confidence: 0.03, margin: 0.93,
  was_in_run_train: true, was_in_run_val: false,
  prediction_html: "new html", prediction_search_text: "new threshold",
  prob_html: "new probabilities",
}});
assert(card.prediction_html === "new html", "old prediction prose must be replaced");
assert(card.prob_html === "new probabilities", "probability rows must be replaced");
assert(card.search_text.includes("new") && !card.search_text.split(/\\s+/).includes("old"), "search must follow the selected run");
assert(card.correct && card.has_prediction && card.predicted_confidence === 0.96, "verdict state must switch atomically");
assert(card.was_in_run_train && !card.was_in_run_val, "run-partition provenance must switch with the selected model");
"""
    script_path = tmp_path / "frozen-run-switch.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_evaluate_frozen_uses_shared_rule_contract_and_background_status(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    _seed_inference_bundle(layout, "run-1", exported=False)
    calls = []
    release_evaluation = threading.Event()

    def fake_evaluate(cfg, project_id, model_id, rule, *, progress=None):
        calls.append((project_id, model_id, rule))
        if progress is not None:
            progress({
                "phase": "scoring frozen test", "completed": 3, "total": 10,
                "current_name": "sample-003.wav", "current_class": "cheers_loud",
            })
        if rule.id == "fixed@0.92":
            release_evaluation.wait(timeout=5.0)
        return {
            "accuracy": 0.8, "f1_macro_targets": 0.7, "n_samples": 5,
            "score_profile": "fixed_threshold_benchmark",
            "prediction_mode": "runtime_thresholds",
            "thresholds_source": "override",
            "test_set_hash": "sha256:current",
        }

    monkeypatch.setattr("sed.train.eval.evaluate_run_under_rule", fake_evaluate)
    from sed.config import Config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    port = _free_port()
    handler = make_handler(
        cfg=Config(), project_id="p",
        render_html_callable=lambda **_kwargs: "<html>ok</html>", layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        fixed = _post_json(f"{base}/api/evaluate-frozen", {
            "run_id": "run-1", "rule_id": "fixed@0.92",
        })
        assert fixed["ok"] is True
        deadline = time.time() + 2.0
        frozen_jobs = []
        while time.time() < deadline:
            activity = _get_json(f"{base}/api/activity")
            frozen_jobs = [job for job in activity["jobs"] if job["type"] == "frozen_eval"]
            if frozen_jobs and frozen_jobs[0].get("progress") == pytest.approx(0.3):
                break
            time.sleep(0.02)
        assert len(frozen_jobs) == 1
        assert frozen_jobs[0]["finished"] is False
        assert frozen_jobs[0]["progress"] == pytest.approx(0.3)
        assert "3/10" in frozen_jobs[0]["detail"]
        assert "cheers_loud" in frozen_jobs[0]["detail"]
        assert "sample-003.wav" in frozen_jobs[0]["detail"]
        assert frozen_jobs[0]["tab"] == "frozen"
        release_evaluation.set()
        fixed_done = _wait_for_job(
            f"{base}/api/evaluate-frozen/status?job_id={fixed['job']['job_id']}"
        )
        assert fixed_done["job"]["state"] == "finished"
        assert fixed_done["job"]["result"]["test_set_hash"] == "sha256:current"
        assert calls[-1][2].id == "fixed@0.92"

        deployed = _post_json(f"{base}/api/evaluate-frozen", {
            "run_id": "run-1", "rule_id": "deployed",
        })
        assert deployed["ok"] is True
        _wait_for_job(
            f"{base}/api/evaluate-frozen/status?job_id={deployed['job']['job_id']}"
        )
        assert calls[-1][2].id == "deployed"

        status, body = _post_json_status(f"{base}/api/evaluate-frozen", {
            "run_id": "run-1", "rule_id": "fixed@1.20",
        })
        assert status == 400
        assert "in (0, 1)" in body["error"]
    finally:
        release_evaluation.set()
        httpd.shutdown()


def test_render_html_includes_mined_detection_analysis_controls(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    _seed_mined_clip(layout)

    from sed.data.inspect import _build_panels_from_mined, _render_html

    mined_panels = _build_panels_from_mined(layout, interactive=True)
    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        aug_panels={},
        frozen_panels={},
        mined_panels=mined_panels,
        interactive=True,
    )

    assert "id=\"mined-analyze-btn\"" in html
    assert "id=\"mined-insight\"" in html
    assert "data-prob-class='cheers_loud'" in html
    assert "class=\"explain-overlay\"" in html


def test_explanation_summary_and_map_endpoints_return_cached_artifacts(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_frozen_test_clip(layout)
    _seed_test_predictions(layout, wav, run_id="run-1")

    from sed.train.explain import explanation_artifact_path, explanation_report_dir

    report_dir = explanation_report_dir(layout.model_dir("run-1"), "frozen")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps({
        "ok": True,
        "model_id": "run-1",
        "scope": "frozen",
        "n_clips": 1,
        "n_maps": 1,
        "methods": {"cam": 1},
        "by_class": {"cheers_loud": {"count": 1}},
        "representatives": [],
    }), encoding="utf-8")
    artifact = explanation_artifact_path(layout.model_dir("run-1"), "frozen", wav, "cheers_loud")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({
        "ok": True,
        "scope": "frozen",
        "model_id": "run-1",
        "path": str(wav),
        "class_name": "cheers_loud",
        "method": "cam",
        "probability": 0.88,
        "heatmap": [[0.0, 1.0], [0.5, 0.2]],
        "hotspots": [{"label": "0.20-0.40 s, 1.0-2.0 kHz"}],
        "stats": {"concentration": 0.4},
    }), encoding="utf-8")

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=None,
        project_id="p",
        render_html_callable=lambda **_kwargs: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        summary = _get_json(f"{base}/api/explanations/summary?scope=frozen&run=run-1")
        assert summary["ok"] is True
        assert summary["summary"]["n_maps"] == 1

        params = urllib.parse.urlencode({
            "scope": "frozen",
            "run": "run-1",
            "path": str(wav),
            "class": "cheers_loud",
        })
        payload = _get_json(f"{base}/api/explanations/map?{params}")
        assert payload["ok"] is True
        assert payload["explanation"]["method"] == "cam"
        assert payload["explanation"]["heatmap"][0][1] == 1.0
    finally:
        httpd.shutdown()


def test_explanation_start_status_runs_background_job(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.config import Config
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    _seed_inference_bundle(layout, "run-a", exported=False)

    calls = []

    def fake_generate_explanations(cfg, project_id, model_id, *, scope="frozen", method="auto",
                                   classes="top", force=False, limit=None, output_dir="mined"):
        calls.append((project_id, model_id, scope, method, classes, force, limit, output_dir))
        return {
            "ok": True,
            "model_id": model_id,
            "scope": scope,
            "n_clips": 2,
            "n_maps": 2,
            "methods": {"cam": 2},
        }

    import sed.train.explain as explain_mod
    monkeypatch.setattr(explain_mod, "generate_explanations", fake_generate_explanations)

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=Config(),
        project_id="p",
        render_html_callable=lambda **_kwargs: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        started = _post_json(f"{base}/api/explanations/start", {
            "run_id": "run-a",
            "scopes": ["frozen", "mined"],
            "method": "auto",
            "classes": "top",
            "force": True,
            "limit": 3,
        })
        assert started["ok"] is True
        job_id = started["job"]["job_id"]
        final = _wait_for_job(f"{base}/api/explanations/status?job_id={job_id}", timeout=5.0)
        job = final["job"]
        assert job["state"] == "finished", job
        assert job["completed"] == 2
        assert [c[2] for c in calls] == ["frozen", "mined"]
        assert all(c[5] is True and c[6] == 3 for c in calls)
        activity = _get_json(f"{base}/api/activity")
        explanation_jobs = [j for j in activity["jobs"] if j["type"] == "explain"]
        assert len(explanation_jobs) == 1
        assert explanation_jobs[0]["title"] == "Explaining predictions"
        assert explanation_jobs[0]["finished"] is True
    finally:
        httpd.shutdown()


def test_frozen_default_prefers_new_completed_run_without_predictions(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_frozen_test_clip(layout)
    _seed_test_predictions(layout, wav, run_id="run-old")
    model_dir = _seed_inference_bundle(layout, "run-new", exported=False)
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_at"] = "2026-06-12T12:00:00Z"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    from sed.data.inspect import _build_panels_from_frozen_test, frozen_run_prediction_payload
    panels, run_id, has_predictions, context = _build_panels_from_frozen_test(layout)
    payload = frozen_run_prediction_payload(layout)

    assert run_id == "run-new"
    assert has_predictions is False
    assert context["viewed_run_id"] == "run-new"
    assert panels["cheers_loud"][0]["predicted_class"] == "not_evaluated"
    assert payload["run_id"] == "run-new"
    assert payload["has_predictions"] is False
    assert [row["run_id"] for row in payload["context"]["run_options"]][:2] == ["run-new", "run-old"]


def test_frozen_test_panels_can_override_prediction_run(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_frozen_test_clip(layout)
    _seed_test_predictions(layout, wav, run_id="run-selected")
    override = _seed_test_predictions(layout, wav, run_id="run-override")
    text = override.read_text(encoding="utf-8").replace("_negative", "cheers_loud")
    override.write_text(text, encoding="utf-8")
    layout.config_toml.write_text(
        '[deploy]\nactive_model = "run-selected"\n', encoding="utf-8"
    )

    from sed.data.inspect import _build_panels_from_frozen_test
    panels, run_id, has_predictions, context = _build_panels_from_frozen_test(
        layout,
        run_id="run-override",
    )

    assert run_id == "run-override"
    assert has_predictions is True
    assert context["selected_run_id"] == "run-selected"
    assert panels["cheers_loud"][0]["predicted_class"] == "cheers_loud"


def test_mined_bucket_auto_rule_can_be_dismissed_and_manual_mined_assignment_allowed(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_mined_clip(layout)

    from sed.config import Config
    cfg = Config()
    cfg.model.classes = ["_negative", "cheers_loud"]
    cfg.mining.output_dir = "mined"

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(cfg=cfg, project_id="p",
                           render_html_callable=lambda: "<html>ok</html>",
                           layout=layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        payload = _get_json(f"http://127.0.0.1:{port}/api/selection")
        rules = payload["selection"]["assignments"]["_negative"]
        auto_rule = next(rule for rule in rules if rule.get("auto_source") == "mined")

        assert wav.exists()
        assert auto_rule["source_bucket"] == "mined"
        assert auto_rule["split"] == "train"
        assert auto_rule["labels"] == ["cheers_loud"]
        assert auto_rule["session"] == "2026-06-17"
        assert auto_rule["count"] == 1

        manual = _post_json(
            f"http://127.0.0.1:{port}/api/selection-rule",
            {
                "action": "add",
                "class_name": "cheers_loud",
                "rule": {
                    "labels": ["cheers_loud"],
                    "project_ids": ["p"],
                    "source_bucket": "mined",
                    "split": "train",
                },
            },
        )
        assert manual["ok"] is True
        assert manual["added"] == 1

        removed = _post_json(
            f"http://127.0.0.1:{port}/api/selection-rule",
            {
                "action": "remove",
                "class_name": "_negative",
                "rule_id": auto_rule["id"],
            },
        )
        assert removed["ok"] is True
        assert removed["removed"] == 1

        refreshed = _get_json(f"http://127.0.0.1:{port}/api/selection")
        negative_rules = refreshed["selection"]["assignments"].get("_negative", [])
        assert not any(rule.get("auto_bucket_key") == auto_rule["auto_bucket_key"] for rule in negative_rules)
        assert auto_rule["auto_bucket_key"] in refreshed["selection"]["auto_mined_dismissed_bucket_keys"]
    finally:
        httpd.shutdown()


def test_selection_assign_clips_endpoint_updates_selection(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    src = tmp_path / "glass.wav"
    sf.write(str(src), np.zeros(32000, dtype="float32"), 32000)
    clip = lib_mod.Library().import_clip(src, label="glass", project_id="_library")
    wav = tmp_path / "lib" / clip.relpath

    from sed.config import Config
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(cfg=cfg, project_id="p",
                           render_html_callable=lambda: "<html>ok</html>",
                           layout=layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        assign_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/selection",
            data=json.dumps({
                "action": "assign_clips",
                "class_name": "glass",
                "clips": [{"path": str(wav), "uuid": clip.uuid, "relpath": clip.relpath}],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(assign_req, timeout=2) as r:
            assign_payload = json.loads(r.read())
        assert assign_payload["added"] == 1

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/selection", timeout=2) as r:
            selection_payload = json.loads(r.read())
        rows = selection_payload["selection"]["clip_assignments"]["glass"]
        assert rows[0]["uuid"] == clip.uuid
        assert rows[0]["relpath"] == clip.relpath
    finally:
        httpd.shutdown()


def test_inspect_server_selection_rule_api_writes_ui_manifest(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")

    layout = ProjectLayout("p")
    layout.create()

    from sed.config import Config
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        add_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/selection-rule",
            data=json.dumps({
                "action": "add",
                "class_name": "glass",
                "rule": {
                    "labels": ["ikea_wine"],
                    "project_ids": ["p"],
                    "density": "quiet",
                    "tags_all": ["wine-glass"],
                    "source_bucket": "raw",
                    "triggered_channels": [0, 2],
                    "channel_index": 1,
                    "review_status": "rejected",
                    "count": 12,
                },
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(add_req, timeout=2) as r:
            payload = json.loads(r.read())
        assert payload["ok"] is True
        assert payload["selection"]["ui_managed"] is True
        row = payload["selection"]["assignments"]["glass"][0]
        assert row["labels"] == ["ikea_wine"]
        assert row["tags_all"] == ["wine-glass"]
        assert row["triggered_channels"] == [0, 2]
        assert row["channel_index"] == 1
        assert row["count"] == 12
        assert "review_status" not in row

        remove_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/selection-rule",
            data=json.dumps({
                "action": "remove",
                "class_name": "glass",
                "rule_id": row["id"],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(remove_req, timeout=2) as r:
            removed = json.loads(r.read())
        assert removed["removed"] == 1
        assert "glass" not in removed["selection"]["assignments"]
    finally:
        httpd.shutdown()


def test_inspect_server_selection_get_is_fast_read_only_manifest(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    layout = ProjectLayout("p")
    layout.create()

    from sed.config import Config
    from sed.data.compose import write_dataset_selection

    original = {
        "ui_managed": True,
        "updated_at": "2026-01-01T00:00:00+00:00",
        "assignments": {
            "glass": [
                {
                    "id": "rule-1",
                    "labels": ["glass"],
                    "project_ids": ["p"],
                    "source_bucket": "raw",
                    "count": 12,
                },
            ],
        },
        "augmentation_recipes": [],
    }
    path = write_dataset_selection(layout, original)
    before_text = path.read_text(encoding="utf-8")

    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/selection", timeout=2) as r:
            payload = json.loads(r.read())
        assert payload["ok"] is True
        assert payload["selection"]["assignments"]["glass"][0]["id"] == "rule-1"
        assert "augmentation_settings" in payload
        assert "augmentation_generated_counts" not in payload
        assert path.read_text(encoding="utf-8") == before_text
    finally:
        httpd.shutdown()


def test_inspect_server_generated_counts_endpoint_returns_existing_shape(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    layout = ProjectLayout("p")
    layout.create()
    generated_dir = tmp_path / "data_library" / "derived" / "p" / "trainval" / "glass"
    generated_dir.mkdir(parents=True)
    wav = generated_dir / "aug.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    wav.with_suffix(".json").write_text(json.dumps({
        "recipe_id": "recipe-1",
        "base_label": "glass",
        "overlay_label": "noise",
        "augmentation_params": {"recipe_id": "recipe-1"},
    }), encoding="utf-8")

    from sed.config import Config
    from sed.data.compose import write_dataset_selection
    write_dataset_selection(layout, {"ui_managed": True, "augmentation_recipes": []})

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=Config(),
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/augmentation-generated-counts", timeout=10) as r:
            payload = json.loads(r.read())
        counts = payload["augmentation_generated_counts"]["recipe-1"]
        assert payload["ok"] is True
        assert counts["total"] == 1
        assert counts["current"] == 1
        assert counts["trainval"] == 1
        assert counts["orphan"] is True
        assert counts["base_label"] == "glass"
        assert counts["overlay_label"] == "noise"
    finally:
        httpd.shutdown()


def test_inspect_server_augmentation_rule_api_and_generate(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")

    layout = ProjectLayout("p")
    layout.create()

    from sed.config import Config
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    from sed.data import augment as augment_mod

    calls = []

    def fake_materialize(cfg_arg, project_id, recipes, seed=None, run_prefix=None, replace=False,
                         frozen_test=False, generation_id=None, progress=None, should_cancel=None):
        if progress:
            progress({"state": "running", "total": 3, "completed": 1, "generated": 1})
        calls.append({
            "project_id": project_id,
            "prefix": run_prefix,
            "replace": replace,
            "frozen_test": frozen_test,
            "recipes": recipes,
            "samples_per_source": cfg_arg.augmentation.samples_per_source,
            "pitch_shift_enabled": cfg_arg.augmentation.pitch_shift_enabled,
            "snr_db_range": cfg_arg.augmentation.snr_db_range,
        })
        return {"glass": {"overlay": 3}}

    monkeypatch.setattr(augment_mod, "materialize_library_recipes", fake_materialize)

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        add_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps({
                "action": "add",
                "recipe": {
                    "base_filters": [
                        {"labels": ["glass"], "project_ids": ["p"], "count": 4},
                    ],
                    "overlay_filters": [
                        {"labels": ["cutlery"], "tags_all": ["local"], "count": 5},
                    ],
                    "description": "glass under cutlery",
                    "tags": "cutlery,hard",
                },
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(add_req, timeout=2) as r:
            payload = json.loads(r.read())
        recipe = payload["selection"]["augmentation_recipes"][0]
        assert recipe["base"] == "glass"
        assert recipe["overlay"] == "cutlery"
        assert recipe["recipe_id"] == recipe["id"]
        assert recipe["base_filters"][0]["project_ids"] == ["p"]
        assert recipe["overlay_filters"][0]["tags_all"] == ["local"]
        assert recipe["tags"] == ["cutlery", "hard"]

        settings_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-settings",
            data=json.dumps({
                "settings": {
                    "samples_per_source": 7,
                    "pitch_shift_enabled": False,
                    "snr_db_range": [8, 12],
                    "mix_map_enabled": False,
                }
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(settings_req, timeout=2) as r:
            settings_payload = json.loads(r.read())
        assert settings_payload["selection"]["augmentation_settings"]["samples_per_source"] == 7.0
        assert settings_payload["selection"]["augmentation_settings"]["mix_map_enabled"] is False

        # Per-recipe generation knobs replace the old top-level sync payload.
        update_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps({
                "action": "update",
                "recipe": {
                    "recipe_id": recipe["id"],
                    "prefix": "ui",
                    "replace_existing": True,
                    "frozen_test": True,
                },
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(update_req, timeout=2) as r:
            updated = json.loads(r.read())
        assert updated["ok"] is True

        start_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augment/start",
            data=json.dumps({"recipe_ids": [recipe["id"]]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(start_req, timeout=2) as r:
            started = json.loads(r.read())
        job_id = started["job"]["job_id"]
        assert job_id
        status_payload = {}
        for _ in range(50):
            time.sleep(0.05)
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/augment/status?job_id={job_id}",
                timeout=2,
            ) as r:
                status_payload = json.loads(r.read())
            if status_payload["job"]["state"] in {"finished", "error"}:
                break
        assert status_payload["job"]["state"] == "finished"
        assert status_payload["job"]["counts"] == {"glass": {"overlay": 3}}

        assert calls[0]["project_id"] == "p"
        # The run cfg carries the page-level settings saved above; per-recipe
        # snapshots are applied inside materialize_library_recipes itself.
        assert calls[0]["samples_per_source"] == 7.0
        assert calls[0]["pitch_shift_enabled"] is False
        assert calls[0]["snr_db_range"] == (8.0, 12.0)
        combo = calls[0]["recipes"][0]
        assert combo["base_filters"][0]["labels"] == ["glass"]
        assert combo["overlay_filters"][0]["tags_all"] == ["local"]
        assert combo["recipe_id"] == recipe["id"]
        # Generation knobs travel on the recipe row, not the start payload.
        assert combo["prefix"] == "ui"
        assert combo["replace_existing"] is True
        assert combo["frozen_test"] is True
        # The settings snapshot was taken at Add time (cfg defaults), so the
        # later panel save must NOT have leaked into the saved recipe.
        assert combo["settings"]["samples_per_source"] == 2.0
        assert combo["settings"]["pitch_shift_enabled"] is True
    finally:
        httpd.shutdown()


def test_augmentation_recipe_normalizer_accepts_pitch_only_recipe():
    from sed.data.inspect_server import _normalise_augmentation_recipe

    recipe = _normalise_augmentation_recipe({
        "base_filters": [{"labels": ["glass"], "project_ids": ["p"], "count": 4}],
        "overlay_filters": [],
        "settings": {
            "enabled": True,
            "pitch_shift_enabled": True,
            "overlay_enabled": False,
        },
    })

    assert recipe["base"] == "glass"
    assert recipe["overlay"] == ""
    assert recipe["overlay_filter"] == {}
    assert recipe["overlay_filters"] == []
    assert recipe["overlay_count"] == 0


def test_augmentation_rule_api_recomputes_counts_and_rejects_missing_overlay(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config
    from sed.data.library import Library
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    layout = ProjectLayout("p")
    layout.create()
    for label in ("glass", "cutlery"):
        src = tmp_path / f"{label}.wav"
        sf.write(str(src), np.zeros(32000, dtype=np.float32), 32000)
        Library().import_clip(src, label=label, project_id="p")

    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        good_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps({
                "action": "add",
                "recipe": {
                    "base_filters": [{"labels": ["glass"], "project_ids": ["p"]}],
                    "overlay_filters": [{"labels": ["cutlery"], "project_ids": ["p"]}],
                    "settings": {"enabled": True, "overlay_enabled": True, "pitch_shift_enabled": False},
                },
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(good_req, timeout=2) as r:
            payload = json.loads(r.read())
        recipe = payload["selection"]["augmentation_recipes"][0]
        assert recipe["base_count"] == 1
        assert recipe["overlay_count"] == 1
        assert recipe["overlay"] == "cutlery"

        bad_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps({
                "action": "add",
                "recipe": {
                    "base_filters": [{"labels": ["glass"], "project_ids": ["p"]}],
                    "overlay_filters": [],
                    "settings": {"enabled": True, "overlay_enabled": True, "pitch_shift_enabled": False},
                },
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(bad_req, timeout=2)
        assert excinfo.value.code == 400
        assert "no overlay bucket" in excinfo.value.read().decode().lower()

        pitch_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps({
                "action": "add",
                "recipe": {
                    "base_filters": [{"labels": ["glass"], "project_ids": ["p"]}],
                    "overlay_filters": [],
                    "settings": {"enabled": True, "overlay_enabled": False, "pitch_shift_enabled": True},
                },
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(pitch_req, timeout=2) as r:
            pitch_payload = json.loads(r.read())
        pitch_recipe = pitch_payload["selection"]["augmentation_recipes"][-1]
        assert pitch_recipe["overlay"] == "pitch"
        assert pitch_recipe["overlay_count"] == 0
    finally:
        httpd.shutdown()


def test_add_recipe_keeps_overlay_bucket_when_panel_overlay_disabled(tmp_path, monkeypatch):
    """An explicitly selected overlay bucket must never be silently dropped just
    because the settings panel had the overlay stage toggled off."""
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config
    from sed.data.library import Library
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    layout = ProjectLayout("p")
    layout.create()
    for label in ("glass", "cutlery"):
        src = tmp_path / f"{label}.wav"
        sf.write(str(src), np.zeros(32000, dtype=np.float32), 32000)
        Library().import_clip(src, label=label, project_id="p")

    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps({
                "action": "add",
                "recipe": {
                    "base_filters": [{"labels": ["glass"], "project_ids": ["p"]}],
                    "overlay_filters": [{"labels": ["cutlery"], "project_ids": ["p"]}],
                    "settings": {"enabled": True, "overlay_enabled": False, "pitch_shift_enabled": True},
                },
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            payload = json.loads(r.read())
        recipe = payload["selection"]["augmentation_recipes"][0]
        assert recipe["overlay"] == "cutlery"
        assert recipe["overlay_count"] == 1
        assert recipe["overlay_filters"][0]["labels"] == ["cutlery"]
        assert recipe["settings"]["overlay_enabled"] is True
    finally:
        httpd.shutdown()


def test_preview_logmel_uses_model_frontend_and_preserves_legacy_thumbs(tmp_path, monkeypatch):
    """Preview spectrograms must show the model's features: efficientat
    frontend at the model sample rate with a fixed display range. Feature
    changes register an independent cache variant rather than deleting every
    legacy sibling thumbnail synchronously."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config
    from sed.data import inspect as inspect_mod
    from sed.audio.features import feature_frame_count

    cfg = Config.model_validate({
        "dataset": {"target_classes": ["_negative", "glass"]},
        "features": {"frontend": "efficientat"},
    })

    lib = tmp_path / "data_library"
    lib.mkdir(parents=True, exist_ok=True)
    stale_thumb = lib / "old.spec.png"
    stale_thumb.write_bytes(b"png")

    inspect_mod.set_preview_features(cfg)
    try:
        assert stale_thumb.exists()
        assert (lib / ".preview_cache" / "index.sqlite3").exists()

        # 48 kHz source resamples to the 32 kHz model rate; frame count and the
        # fixed efficientat display range prove the model frontend ran.
        wave = (0.1 * np.random.RandomState(0).randn(48000)).astype(np.float32)
        spec, vmin, vmax = inspect_mod._preview_logmel(wave, 48000)
        assert spec.shape[0] == cfg.features.n_mels
        assert spec.shape[1] == feature_frame_count(
            sample_rate=cfg.model.sample_rate,
            window_seconds=1.0,
            hop_length=cfg.features.hop_length,
            frontend="efficientat",
        )
        assert (vmin, vmax) == inspect_mod._EFFICIENTAT_PREVIEW_RANGE

        # Same contract again — registering a variant is idempotent.
        inspect_mod.set_preview_features(cfg)
    finally:
        inspect_mod.set_preview_features(None)
    spec2, v1, v2 = inspect_mod._preview_logmel(wave, 48000)
    assert v1 is None and v2 is None


def test_spectrogram_preview_pixels_keep_time_horizontal():
    from sed.data import inspect as inspect_mod

    short = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    widened = inspect_mod._spectrogram_preview_pixels(short)
    assert widened.shape == (4, 4)
    # Rows stay frequency bands, columns stay time; widening only duplicates
    # time columns to a square minimum.
    assert np.array_equal(widened[:, 0], short[:, 0])
    assert np.array_equal(widened[:, -1], short[:, -1])

    wide = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    assert inspect_mod._spectrogram_preview_pixels(wide).shape == (4, 6)


def test_recipe_normalizer_defaults_frozen_test_from_test_base():
    from sed.data.inspect_server import _normalise_augmentation_recipe

    recipe = _normalise_augmentation_recipe({
        "base_filters": [{"labels": ["glass"], "project_ids": ["p"], "split": "test"}],
        "overlay_filters": [{"labels": ["amb"], "project_ids": ["p"], "split": "train"}],
    })
    assert recipe["frozen_test"] is True

    explicit = _normalise_augmentation_recipe({
        "base_filters": [{"labels": ["glass"], "project_ids": ["p"], "split": "test"}],
        "overlay_filters": [{"labels": ["amb"], "project_ids": ["p"], "split": "train"}],
        "frozen_test": False,
    })
    assert explicit["frozen_test"] is False

    train_base = _normalise_augmentation_recipe({
        "base_filters": [{"labels": ["glass"], "project_ids": ["p"], "split": "train"}],
        "overlay_filters": [{"labels": ["amb"], "project_ids": ["p"], "split": "train"}],
    })
    assert train_base["frozen_test"] is False


def test_add_recipe_from_test_bucket_counts_clips_and_defaults_frozen(tmp_path, monkeypatch):
    """Test buckets must be augmentable: native roles keep the count non-zero
    and the recipe defaults to frozen-test output."""
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config
    from sed.data.library import Library
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    layout = ProjectLayout("p")
    layout.create()
    base_src = tmp_path / "glass.wav"
    overlay_src = tmp_path / "amb.wav"
    sf.write(str(base_src), np.zeros(32000, dtype=np.float32), 32000)
    sf.write(str(overlay_src), np.zeros(32000, dtype=np.float32), 32000)
    Library().import_clip(base_src, label="glass", project_id="p")
    Library().import_clip(overlay_src, label="amb", project_id="p")

    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps({
                "action": "add",
                "recipe": {
                    "base_filters": [{
                        "labels": ["glass"], "project_ids": ["p"],
                        "split": "test",
                    }],
                    "overlay_filters": [{
                        "labels": ["amb"], "project_ids": ["p"],
                        "split": "train",
                    }],
                    "settings": {"enabled": True, "overlay_enabled": True, "pitch_shift_enabled": False},
                },
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            payload = json.loads(r.read())
        recipe = payload["selection"]["augmentation_recipes"][0]
        assert recipe["frozen_test"] is True
        assert recipe["base_count"] == 1
        assert recipe["overlay_count"] == 1
    finally:
        httpd.shutdown()


def test_recipe_counts_use_recipe_snapshot_and_list_orphans(tmp_path, monkeypatch):
    """Generated-clip counts sign each recipe with its own settings snapshot —
    panel changes must not flip clips to stale — and generations whose recipe
    was removed are still listed (orphan) so they stay deletable."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.data.augment import recipe_signature
    from sed.data.inspect import augmentation_recipe_counts

    layout = ProjectLayout("p")
    layout.create()

    snapshot = {
        "enabled": True,
        "overlay_enabled": True,
        "pitch_shift_enabled": False,
        "samples_per_source": 2.0,
        "snr_db_range": [0.0, 20.0],
    }
    recipe = {
        "recipe_id": "keep1234",
        "id": "keep1234",
        "base_filters": [{"labels": ["glass"], "project_ids": ["p"]}],
        "overlay_filters": [{"labels": ["cutlery"], "project_ids": ["p"]}],
        "description": "",
        "tags": [],
        "settings": snapshot,
        "frozen_test": False,
    }
    signature = recipe_signature(recipe, snapshot, frozen_test=False)

    derived = tmp_path / "data_library" / "derived" / "p" / "trainval" / "glass"
    derived.mkdir(parents=True)
    (derived / "a.json").write_text(json.dumps({
        "recipe_id": "keep1234",
        "recipe_signature": "old-signature-including-volatile-counts",
        "augmentation_settings": snapshot,
        "base_label": "glass",
        "overlay_label": "cutlery",
    }), encoding="utf-8")
    (derived / "stable.json").write_text(json.dumps({
        "recipe_id": "keep1234",
        "recipe_signature": signature,
        "augmentation_settings": snapshot,
        "base_label": "glass",
        "overlay_label": "cutlery",
    }), encoding="utf-8")
    (derived / "orphan.json").write_text(json.dumps({
        "recipe_id": "gone5678",
        "recipe_signature": "stale-or-not",
        "base_label": "ikea_wine",
        "overlay_label": "ambience",
    }), encoding="utf-8")

    # Page-level settings drifted after generation; the recipe snapshot wins.
    page_settings = {**snapshot, "samples_per_source": 9.0, "snr_db_range": [5.0, 6.0]}
    counts = augmentation_recipe_counts(layout, recipes=[recipe], settings=page_settings)

    row = counts["keep1234"]
    assert row["current"] == 2
    assert row["stale"] == 0
    assert row["orphan"] is False

    changed_recipe = {**recipe, "settings": {**snapshot, "snr_db_range": [5.0, 6.0]}}
    changed_counts = augmentation_recipe_counts(layout, recipes=[changed_recipe], settings=snapshot)
    assert changed_counts["keep1234"]["current"] == 0
    assert changed_counts["keep1234"]["stale"] == 2

    orphan = counts["gone5678"]
    assert orphan["orphan"] is True
    assert orphan["base_label"] == "ikea_wine"
    assert orphan["overlay_label"] == "ambience"
    assert orphan["total"] == 1


def test_review_payload_only_shows_assigned_derived_in_augmented_tab(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config
    from sed.data.compose import write_dataset_selection
    from sed.data.inspect import review_data_payload
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    raw_src = tmp_path / "raw.wav"
    sf.write(str(raw_src), np.zeros(32000, dtype=np.float32), 32000)
    Library().import_clip(
        raw_src,
        label="riedel_wine",
        project_id="p",
        density="normal",
        review_status="accepted",
    )

    derived = tmp_path / "data_library" / "derived" / "p" / "trainval" / "riedel_wine" / "aug.wav"
    derived.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(derived), np.zeros(32000, dtype=np.float32), 32000)
    derived.with_suffix(".json").write_text(json.dumps({
        "schema_version": 2,
        "category": "derived",
        "label": "riedel_wine",
        "class_name": "riedel_wine",
        "project_id": "p",
        "dataset_role": "train",
        "review_status": "accepted",
        "sample_rate": 32000,
        "duration_s": 1.0,
        "uuid": "derived-1",
        "augmentation": "overlay",
        "base_label": "riedel_wine",
        "overlay_label": "cutlery",
        "recipe_id": "recipe-1",
        "masking_score_db": 8.0,
        "masking_score_p25_db": 5.0,
        "masking_score_min_db": 2.0,
        "mask_target_db": -12.0,
        "mask_background_db": -20.0,
        "augmentation_params": {
            "base_label": "riedel_wine",
            "overlay_label": "cutlery",
            "recipe_id": "recipe-1",
            "snr_db": 12.0,
        },
    }), encoding="utf-8")
    Library().rebuild_manifest()

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_quiet"]}})
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "cheers_quiet": [{
                "labels": ["riedel_wine"],
                "project_ids": ["p"],
                "source_bucket": "raw",
                "split": "train",
            }],
        },
    })
    payload = review_data_payload(cfg, layout)
    assert "cheers_quiet" in payload["panels"]
    assert "riedel_wine" not in payload["panels"]
    assert payload["panels"]["cheers_quiet"][0]["library_label"] == "riedel_wine"
    assert payload["aug_panels"] == {}

    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "cheers_quiet": [
                {
                    "labels": ["riedel_wine"],
                    "project_ids": ["p"],
                    "source_bucket": "raw",
                    "split": "train",
                },
                {
                    "labels": ["riedel_wine"],
                    "project_ids": ["p"],
                    "source_bucket": "derived",
                    "split": "train",
                },
            ],
        },
    })
    payload = review_data_payload(cfg, layout)
    assert "cheers_quiet" in payload["panels"]
    assert "riedel_wine" not in payload["panels"]
    assert "riedel_wine × cutlery" in payload["aug_panels"]
    card = payload["aug_panels"]["riedel_wine × cutlery"][0]
    assert card["assigned_target_class"] == "cheers_quiet"
    assert card["masking_score_p25_db"] == 5.0
    assert card["masking_score_min_db"] == 2.0

    # Paging the Augmented tab must use the same key its blocks are rendered
    # under — the augmentation group, not the target class. Matching the target
    # class here made every "Load More ... more on disk" request come back empty.
    from sed.data.inspect import clips_page
    cards, total = clips_page(
        cfg, layout, "augmented", "riedel_wine × cutlery", 0, 10, return_total=True)
    assert total >= 1 and len(cards) == total
    assert clips_page(
        cfg, layout, "augmented", "cheers_quiet", 0, 10, return_total=True)[1] == 0


def test_review_payload_keeps_split_and_multichannel_under_assigned_target(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config
    from sed.data.compose import write_dataset_selection
    from sed.data.inspect import review_data_payload
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    lib = Library()

    multi_src = tmp_path / "talk-test.wav"
    sf.write(str(multi_src), np.zeros((32000, 2), dtype=np.float32), 32000)
    multi = lib.import_clip(
        multi_src,
        label="talk-test",
        project_id="p",
        density="normal",
        review_status="accepted",
    )
    multi.clip_kind = "multichannel"
    multi.channels = 2
    multi.recorded_channels = [0, 1]
    multi.triggered_channels = [0, 1]
    multi.tags = ["talking"]
    lib.write_sidecar(multi.absolute_path(lib.root), multi)

    split_src = tmp_path / "talk-test_03.wav"
    sf.write(str(split_src), np.zeros((32000, 2), dtype=np.float32), 32000)
    split_parent = lib.import_clip(
        split_src,
        label="talk-test_03",
        project_id="p",
        density="normal",
        review_status="accepted",
    )
    split_parent.clip_kind = "multichannel"
    split_parent.channels = 2
    split_parent.recorded_channels = [0, 1]
    split_parent.triggered_channels = [0, 1]
    split_parent.tags = ["talking"]
    lib.write_sidecar(split_parent.absolute_path(lib.root), split_parent)
    lib.split_clip(split_parent.absolute_path(lib.root), force=True)
    lib.rebuild_manifest()

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_quiet"]}})
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "cheers_quiet": [
                {
                    "labels": ["talk-test"],
                    "project_ids": ["p"],
                    "source_bucket": "raw",
                    "clip_kind": "multichannel",
                    "split": "train",
                },
                {
                    "labels": ["talk-test_03"],
                    "project_ids": ["p"],
                    "source_bucket": "raw",
                    "clip_kind": "channel_split",
                    "triggered_channels": [0],
                    "split": "train",
                },
            ],
        },
    })

    payload = review_data_payload(cfg, layout)

    assert list(payload["panels"].keys()) == ["cheers_quiet"]
    cards = payload["panels"]["cheers_quiet"]
    assert len(cards) == 2
    assert {card["library_label"] for card in cards} == {"talk-test", "talk-test_03"}
    assert all(card["assigned_target_class"] == "cheers_quiet" for card in cards)
    assert not any(" rec " in key for key in payload["panels"])
    assert "talk-test" not in payload["panels"]
    assert "talk-test_03" not in payload["panels"]


def test_review_payload_hides_stale_frozen_test_rows(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")

    from sed.config import Config
    from sed.data.compose import write_dataset_selection
    from sed.data.inspect import review_data_payload

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_frozen_test_clip(layout)
    _seed_test_predictions(layout, wav)
    (layout.dataset_dir / "dataset_snapshot.json").write_text(
        json.dumps({"frozen_test": {"hash": "sha256:old"}}),
        encoding="utf-8",
    )
    write_dataset_selection(layout, {"ui_managed": True, "assignments": {}})

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_loud"]}})
    payload = review_data_payload(cfg, layout)

    assert payload["frozen_panels"] == {}
    assert payload["frozen_context"]["stale"] is True


def test_training_selection_summary_counts_dataset_and_augmented_separately():
    from pathlib import Path

    from sed.data.inspect import training_selection_summary

    summary = training_selection_summary(
        [(Path("/tmp/raw.wav"), "quiet", {})],
        [(Path("/tmp/aug.wav"), "riedel × cutlery", {"assigned_target_class": "quiet"})],
        ["quiet", "_negative"],
    )

    assert summary["quiet"] == {"dataset": 1, "augmented": 1, "count": 2}
    assert summary["_negative"] == {"dataset": 0, "augmented": 0, "count": 0}


def test_review_payload_hides_legacy_frozen_rows_when_rules_are_empty(tmp_path, monkeypatch):
    """An empty UI preset must not revive a pre-snapshot test.csv."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")

    from sed.config import Config
    from sed.data.compose import write_dataset_selection
    from sed.data.inspect import review_data_payload

    layout = ProjectLayout("p")
    layout.create()
    _seed_frozen_test_clip(layout)
    write_dataset_selection(layout, {"ui_managed": True, "assignments": {}})

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_loud"]}})
    payload = review_data_payload(cfg, layout)

    assert payload["frozen_panels"] == {}
    assert payload["frozen_context"]["stale"] is True
    assert "empty" in payload["frozen_context"]["stale_reason"]


def test_review_payload_previews_pending_frozen_test_rules(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config
    from sed.data.compose import write_dataset_selection
    from sed.data.inspect import review_data_payload
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    src = tmp_path / "heldout.wav"
    sf.write(str(src), np.zeros(32000, dtype=np.float32), 32000)
    Library().import_clip(
        src,
        label="heldout_glass",
        project_id="p",
        density="normal",
        review_status="accepted",
    )
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "cheers_loud": [{
                "labels": ["heldout_glass"],
                "project_ids": ["p"],
                "source_bucket": "raw",
                "split": "test",
            }],
        },
    })

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_loud"]}})
    payload = review_data_payload(cfg, layout)

    assert not (layout.dataset_dir / "test.csv").exists()
    assert payload["frozen_context"]["pending_compose"] is True
    assert payload["frozen_context"]["source"] == "pending_dataset_selection"
    assert len(payload["frozen_panels"]["cheers_loud"]) == 1
    assert payload["frozen_panels"]["cheers_loud"][0]["name"].endswith(".wav")


def test_review_data_endpoint_returns_fresh_panel_payload(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    from sed.config import Config

    layout = ProjectLayout("p")
    layout.create()
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        payload = _review_data_via_job(port)
        assert payload["ok"] is True
        assert set(["panels", "aug_panels", "frozen_panels", "mined_panels", "library_items"]).issubset(payload)
        assert "augmentation_generated_counts" in payload
    finally:
        httpd.shutdown()


def test_review_data_background_refresh_job_returns_final_payload(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    from sed.config import Config

    layout = ProjectLayout("p")
    layout.create()
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        started = _post_json(f"http://127.0.0.1:{port}/api/review-data/start", {})
        assert started["ok"] is True
        job_id = started["job"]["job_id"]

        final = _wait_for_job(
            f"http://127.0.0.1:{port}/api/review-data/status?job_id={job_id}",
            timeout=10,
        )
        job = final["job"]
        assert job["state"] == "finished"
        assert job["phase"] == "finished"
        assert job["phase_detail"] == "finished"
        assert job["progress"] == 1.0
        assert job["eta_s"] == 0.0
        assert job["elapsed_s"] >= 0.0
        assert job["payload"]["ok"] is True
        assert set(["panels", "aug_panels", "frozen_panels", "mined_panels", "library_items"]).issubset(job["payload"])
        assert set(["dataset", "augmented", "frozen", "mined", "library"]).issubset(job["summary_counts"])
    finally:
        httpd.shutdown()


def test_frozen_run_options_include_metric_and_loss(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")

    from sed.data.inspect import _frozen_run_options

    layout = ProjectLayout("p")
    layout.create()
    run_dir = layout.model_dir("run-a")
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "test_predictions.tsv").write_text("path\ttrue_class\n", encoding="utf-8")
    (reports / "confusion_matrix.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (reports / "training_curves.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "manifest.json").write_text(json.dumps({
        "model_id": "run-a",
        "created_at": "2026-06-11T12:00:00+00:00",
        "test_metrics": {"f1_macro_targets": 0.9974, "accuracy": 0.99},
    }), encoding="utf-8")
    (run_dir / "training_config.json").write_text(json.dumps({
        "training": {"loss": "cross_entropy"},
    }), encoding="utf-8")
    orphan_reports = layout.model_dir("orphan") / "reports"
    orphan_reports.mkdir(parents=True)
    (orphan_reports / "test_predictions.tsv").write_text("path\ttrue_class\n", encoding="utf-8")

    options = _frozen_run_options(layout, active_run_id="run-a")

    assert [row["run_id"] for row in options] == ["run-a"]
    assert options[0]["run_id"] == "run-a"
    assert "99.7% F1" in options[0]["label"]
    assert "CE" in options[0]["label"]
    assert options[0]["metric_label"] == "99.7% F1"
    assert options[0]["loss_label"] == "CE"
    assert options[0]["has_confusion"] is True
    assert options[0]["has_training_plot"] is True


def test_frozen_run_report_image_endpoint_is_limited(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")

    from sed.config import Config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    layout = ProjectLayout("p")
    layout.create()
    reports = layout.model_dir("run-a") / "reports"
    reports.mkdir(parents=True)
    png = b"\x89PNG\r\n\x1a\ncm"
    (reports / "confusion_matrix.png").write_bytes(png)
    (reports / "training_curves.png").write_bytes(b"\x89PNG\r\n\x1a\nplot")
    (layout.model_dir("run-a") / "manifest.json").write_text(json.dumps({
        "model_id": "run-a",
        "created_at": "2026-06-11T12:00:00+00:00",
        "files": {"training_config": "training_config.json"},
    }), encoding="utf-8")

    port = _free_port()
    handler = make_handler(
        cfg=Config(),
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/frozen-run/image?run=run-a&kind=confusion",
            timeout=2,
        ) as r:
            assert r.read() == png

        for url in [
            f"http://127.0.0.1:{port}/api/frozen-run/image?run=run-a&kind=../../training_config",
            f"http://127.0.0.1:{port}/api/frozen-run/image?run=../run-a&kind=confusion",
        ]:
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(url, timeout=2)
            assert excinfo.value.code == 404
    finally:
        httpd.shutdown()


def test_delete_refresh_updates_library_and_selection_counts(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config
    from sed.data.compose import write_dataset_selection
    from sed.data.library import Library
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    layout = ProjectLayout("p")
    layout.create()
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    lib = Library()
    glass_paths = []
    for idx in range(2):
        src = tmp_path / f"glass_{idx}.wav"
        sf.write(str(src), np.zeros(32000, dtype=np.float32), 32000)
        clip = lib.import_clip(src, label="glass", project_id="p")
        glass_paths.append(clip.absolute_path(lib.root).resolve())
    noise_src = tmp_path / "noise.wav"
    sf.write(str(noise_src), np.zeros(32000, dtype=np.float32), 32000)
    lib.import_clip(noise_src, label="noise", project_id="p")

    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "glass": [{
                "labels": ["glass"],
                "project_ids": ["p"],
                "source_bucket": "raw",
                "split": "train",
                "count": 2,
            }],
        },
        "augmentation_recipes": [{
            "id": "recipe-live-counts",
            "recipe_id": "recipe-live-counts",
            "base_filters": [{
                "labels": ["glass"],
                "project_ids": ["p"],
                "source_bucket": "raw",
                "split": "train",
                "count": 2,
            }],
            "overlay_filters": [{
                "labels": ["noise"],
                "project_ids": ["p"],
                "source_bucket": "raw",
                "split": "train",
                "count": 1,
            }],
            "base_count": 2,
            "overlay_count": 1,
            "settings": {"enabled": True, "overlay_enabled": True, "pitch_shift_enabled": False},
        }],
    })

    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        deleted = _delete_via_job(port, [str(glass_paths[0])])
        assert deleted["deleted_count"] == 1

        review = _review_data_via_job(port)
        live_glass = [
            item for item in review["library_items"]
            if item["label"] == "glass" and item["project_id"] == "p"
        ]
        assert [item["path"] for item in live_glass] == [str(glass_paths[1])]

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/selection", timeout=2) as r:
            selection_payload = json.loads(r.read())
        selection = selection_payload["selection"]
        assert selection["assignments"]["glass"][0]["count"] == 1
        recipe = selection["augmentation_recipes"][0]
        assert recipe["base_count"] == 1
        assert recipe["base_filters"][0]["count"] == 1
        assert recipe["overlay_count"] == 1
        assert recipe["overlay_filters"][0]["count"] == 1
    finally:
        httpd.shutdown()


def test_library_manifest_partial_jsonl_line_rebuilds_cleanly(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.data.library import Library

    src = tmp_path / "glass.wav"
    sf.write(str(src), np.zeros(32000, dtype=np.float32), 32000)
    lib = Library()
    clip = lib.import_clip(src, label="glass", project_id="p")
    wav = clip.absolute_path(lib.root)
    assert wav.exists()

    lib.manifest.write_text('{"relpath": "sounds/glass/broken.wav", "label": "glass', encoding="utf-8")
    rebuilt = lib.all_clips(rebuild_if_empty=True)

    assert len(rebuilt) == 1
    assert rebuilt[0].label == "glass"
    assert rebuilt[0].relpath == clip.relpath
    with lib.manifest.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["relpath"] == clip.relpath


def test_augmentation_rule_remove_can_delete_generated_clips(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.data import library as lib_mod
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.config import Config
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    derived = tmp_path / "data_library" / "derived" / "p" / "trainval" / "glass" / "aug.wav"
    derived.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(derived), np.zeros(32000, dtype=np.float32), 32000)
    derived.with_suffix(".json").write_text(json.dumps({
        "schema_version": 2,
        "category": "derived",
        "label": "glass",
        "class_name": "glass",
        "project_id": "p",
        "dataset_role": "train",
        "review_status": "accepted",
        "recipe_id": "recipe-remove",
    }), encoding="utf-8")
    Library().rebuild_manifest()

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(cfg=cfg, project_id="p", render_html_callable=lambda: "<html>ok</html>", layout=layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        add_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps({
                "action": "add",
                "recipe": {
                    "id": "recipe-remove",
                    "recipe_id": "recipe-remove",
                    "base_filters": [{"labels": ["glass"]}],
                    "overlay_filters": [{"labels": ["noise"]}],
                },
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(add_req, timeout=2).read()
        remove_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps({
                "action": "remove",
                "rule_id": "recipe-remove",
                "delete_generated": True,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(remove_req, timeout=2) as r:
            payload = json.loads(r.read())
        assert payload["removed"] == 1
        assert payload["deleted_generated"] == 1
        assert not derived.exists()
        assert not derived.with_suffix(".json").exists()
    finally:
        httpd.shutdown()


def test_background_augmentation_cancel_reports_canceled(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")

    from sed.config import Config
    from sed.data import augment as augment_mod

    layout = ProjectLayout("p")
    layout.create()
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    def fake_materialize(cfg_arg, project_id, recipes, **kwargs):
        progress = kwargs.get("progress")
        should_cancel = kwargs.get("should_cancel") or (lambda: False)
        if progress:
            progress({"state": "running", "total": 10, "completed": 0, "generated": 0})
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if should_cancel():
                raise augment_mod.AugmentationCancelled("stopped")
            time.sleep(0.02)
        return {}

    monkeypatch.setattr(augment_mod, "materialize_library_recipes", fake_materialize)

    from sed.data.compose import write_dataset_selection
    write_dataset_selection(layout, {
        "ui_managed": True,
        "augmentation_recipes": [{
            "id": "recipe-cancel",
            "recipe_id": "recipe-cancel",
            "base_filters": [{"labels": ["glass"]}],
            "overlay_filters": [{"labels": ["noise"]}],
        }],
    })

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(cfg=cfg, project_id="p", render_html_callable=lambda: "<html>ok</html>", layout=layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        start_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augment/start",
            data=json.dumps({}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(start_req, timeout=2) as r:
            started = json.loads(r.read())
        cancel_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augment/cancel",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(cancel_req, timeout=2).read()
        final = {}
        for _ in range(40):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/augment/status?job_id={started['job']['job_id']}",
                timeout=2,
            ) as r:
                final = json.loads(r.read())
            if final["job"]["state"] == "canceled":
                break
            time.sleep(0.05)
        assert final["job"]["state"] == "canceled"
    finally:
        httpd.shutdown()


def test_rendered_page_uses_unified_data_tip_tooltips():
    """All hover help renders through the styled [data-tip] tooltip; native
    title= attributes are banned from the template so help looks and behaves
    the same on every tab."""
    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        interactive=True,
        model_classes=["_negative"],
        is_pretraining=True,
    )
    assert ' title="' not in html
    assert 'data-tip="' in html
    assert 'id="train-stop-save"' in html
    assert "Stop &amp; save" in html


def test_rendered_page_embeds_config_default_augmentation_settings():
    """The JS form fallbacks read AUG_DEFAULTS, so UI defaults can never
    drift from AugmentationCfg (they used to be hardcoded literals)."""
    import re as _re

    from sed.config import Config
    from sed.data.inspect import _render_html
    from sed.data.source_filters import augmentation_settings_snapshot

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        interactive=True,
        model_classes=["_negative"],
    )
    match = _re.search(r"const AUG_DEFAULTS = (\{.*?\});", html)
    assert match, "AUG_DEFAULTS missing from rendered page"
    embedded = json.loads(match.group(1))
    assert embedded == augmentation_settings_snapshot(Config())
    # The mix-map toggle is a real form control wired to the snapshot key.
    assert "aug-setting-mix-map-enabled" in html
    assert embedded["mix_map_enabled"] is True


def test_rendered_clip_tabs_use_server_backed_search():
    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={"glass": []},
        interactive=True,
        model_classes=["_negative", "glass"],
    )

    assert "CLIP_SEARCH_RESULTS" in html
    assert "CLIP_SEARCH_CONTROLLERS" in html
    assert "searchClipTab(tabType, query, sequence, controller)" in html
    assert "localActivityStart(" in html
    assert '"Loading spectrograms"' in html
    assert 'limit: "1000"' in html
    assert 'limit: "1000", query' in html
    assert 'datasetSearchInput.addEventListener("input", () => scheduleClipTabSearch("dataset"))' in html
    assert 'augSearchInput.addEventListener("input", () => scheduleClipTabSearch("augmented"))' in html
    assert 'frozenSearchInput.addEventListener("input", () => scheduleClipTabSearch("frozen"))' in html
    assert 'minedSearchInput.addEventListener("input", () => scheduleClipTabSearch("mined"))' in html


def test_rendered_audio_features_support_fitted_linear_standardization():
    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        interactive=True,
        model_classes=["_negative", "glass"],
    )

    # Two controls, one per axis: compression decides what the values mean,
    # normalization only what range they arrive in.
    assert 'id="feature-compression"' in html
    assert 'id="feature-normalization"' not in html
    assert "STAGE 1 — feature generation" in html
    assert "STAGE 2 — value normalization" in html
    for compression in ("log power — ln(power + 1e-5)", "dB — 10·log10(power)",
                        "raw power — no compression", "PCEN — per-band adaptive gain"):
        assert compression in html
    assert 'id="feature-scaling"' in html
    assert "fitted z-score · per band (recommended)" in html
    assert "fitted min–max · per band" in html
    # Fixed contract constants are an option, not an invisible default.
    assert "fixed constants — " in html
    assert 'id="feature-scaling_stats"' in html
    assert 'id="feature-scaling-summary"' in html
    assert 'id="feature-standardize_stats_resolved"' in html
    assert 'id="feature-standardize_stats_dataset_sha256"' in html
    assert 'id="feature-standardize_stats_split_seed"' in html
    assert 'id="feature-standardize_stats_val_fraction"' in html
    assert 'id="feature-standardize_stats_clips"' in html
    assert 'id="feature-training-apply"' in html
    assert "featureScalingValue" in html
    assert 'params.set("scaling_method", form.scaling_method || "none");' in html
    assert 'params.set("clips", "2048")' not in html
    assert '"/api/feature-training/apply"' in html
    assert 'MODEL_PROJECT_FEATURE_SOURCE' in html
    assert 'Start training (save settings)' in html
    assert "train-only clips" in html
    assert "featureInvalidateFittedStats" in html
    assert "post-scale mean" in html  # the normalization health readout


def test_library_tab_renders_dataset_designer_controls(tmp_path):
    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        interactive=True,
        model_classes=["_negative", "glass"],
        library_items=[{
            "path": "/tmp/a.wav",
            "label": "glass",
            "project_id": "p",
            "density": "normal",
            "review_status": "accepted",
            "source_bucket": "raw",
            "tags": ["wine"],
            "search_text": "glass wine",
        }],
    )

    assert "library-target-class" in html
    assert "library-target-split" in html
    assert "library-assign-class" in html
    assert "library-delete-selected" in html
    assert "preview-cache-build" in html
    assert "preview-cache-cancel" in html
    assert "PREVIEW_FETCH_LIMIT = 4" in html
    assert "data-preview-url" in html
    assert "/api/preview-cache/start" in html
    assert "library-assign-test" not in html
    assert "Assign bucket to frozen test" not in html
    assert "library-status-filter" not in html
    assert "<th>Status</th>" not in html
    assert "library-mic-filter" in html
    assert "library-kind-filter" not in html
    assert "Kind/Mic" in html
    assert "Session/Time" in html
    assert "session/time" in html
    assert "libraryMicBucketValue" in html
    # A bucket's split comes from the rules that select it, not from its clips.
    assert "bucketSplits" in html
    assert "bucketDatasetRole" not in html
    assert "formatLocalDateTime" in html
    assert "compactUtcToMs" in html
    assert "libraryTimeLabel" in html
    assert "cssEscape(className)" in html
    assert "assigned-dot" in html
    assert "assignmentsForRule" in html
    # Source rules render as a full-width table in the Dataset Definition panel.
    assert "source-rule-row" not in html
    assert "dd-rule-table" in html
    assert "<th>bucket</th>" in html
    assert "dataset-def-panel" in html
    assert "classTintStyle" in html
    assert "CLASS_TINT_PALETTE" in html
    assert "body.ui-busy" in html
    # background-job feedback now lives in the redesigned activity bar
    assert "activity-chips" in html
    assert "initActivityBar" in html
    assert "deleteSelectedLibraryRows" in html
    assert "Rendering changes" in html
    assert "selectedOrVisibleLibraryRules(split)" in html
    assert 'parentBucket = (c.clip_kind || "") === "channel_split" ? ""' in html
    assert "rule.channel_index" in html
    assert 'groupLabel: "corpus"' in html
    assert 'group.label === "corpus" ? "class" : "bucket"' in html
    assert "libraryOverviewClipCount" in html
    assert "libraryFilteredCountForTab" in html
    assert "library-refresh-review" not in html
    assert "Rescan all" not in html
    assert "Rescan all tabs" not in html
    assert "library-scope-refresh" in html
    assert "dataset-scope-refresh" in html
    assert "augmented-scope-refresh" in html
    assert "frozen-scope-refresh" in html
    assert "mined-scope-refresh" in html
    # The dedicated Pretraining tab (and its rescan button) was removed; the
    # corpus subbucket dashboard now lives in the Dataset/Augmented tabs.
    assert "pretrain-rescan-btn" not in html
    assert "Rescan from disk" in html
    assert "compose-dataset-btn" in html
    # The compose action bar is one line: Compose plus per-view actualize chips.
    assert 'class="compose-bar"' in html
    assert "Actualize Augmented" not in html
    assert "Actualize Frozen Test" not in html
    assert 'class="section-refresh rescan-chip"' in html
    assert 'data-refresh-scopes="all"' in html
    assert 'id="library-tag-filter" multiple' in html
    assert "Refresh review tabs" not in html
    assert "refreshRequestSeq" in html
    assert "/api/review-data/start" in html
    assert "/api/review-data/status" in html
    assert "/api/delete/start" in html
    assert "/api/delete/status" in html
    delete_fn = html.split("async function deletePathsInBackground", 1)[1].split(
        "function asArray", 1,
    )[0]
    assert "applyDeletedReviewDelta(finalJob)" in delete_fn
    assert "refreshReviewData" not in delete_fn
    assert "setRefreshHeld" in html
    assert "applyReviewDataPayload(data, { render: false })" in html
    # redesigned bar: contextual delete lives inside .context-actions, and its
    # visibility is driven by the active-tab data-attribute (not .library-active)
    assert 'class="context-actions selection-only"' in html
    assert 'class="del" id="del-btn"' in html
    assert 'class="refresh" id="library-refresh-review"' not in html
    assert '[data-active-tab="dataset"]' in html
    assert ".toolbar .selection-only" in html
    assert "library-row-delete" in html
    assert "data-delete-library-row" in html
    assert "sourceRuleAudioLabel" in html
    assert 'return overlays.length ? `${base} × ${overlays.join(",")}` : base;' in html
    # The dataset tab's summary renders through the shared per-tab renderer.
    assert '["dataset", "augmented", "frozen", "mined"].forEach(renderClipTabSummary)' in html
    assert "recipe-row-description" in html
    assert "sourceRuleBriefSummary" in html
    # Destination is an inline per-row control in the Dataset Definition table,
    # not a static pill in a summary line.
    assert "lib-row-split-pill" not in html
    assert 'data-rule-field="split"' in html
    assert "className)} ←" not in html
    assert "includeLabels: false" not in html
    assert "aug-setting-seed" in html
    assert "recipe-total-samples" in html
    assert "data-recipe-total" in html
    assert "total_samples" in html
    assert "aug-setting-mask-min-mean-db" in html
    assert 'data-tip="Global recipe settings: enable augmentation' in html
    assert 'data-tip="Derived clips per matching base clip.' in html
    # Overlay and masking share one "Overlay & Masking" settings group.
    assert 'data-tip="Combine the target sound with a background overlay sound.' in html
    assert 'Overlay &amp; Masking' in html
    assert 'data-tip="Choose base sounds and optional overlay sounds' in html
    assert 'data-tip="Generate derived clips for checked recipes.' in html
    settings_panel = html.split('<div class="augment-settings-panel">', 1)[1].split(
        '<div class="recipe-actions">',
        1,
    )[0]
    assert ">?</span>" not in settings_panel
    assert "Use selected as base" in html
    assert "recipe-target-class" not in html
    assert "No composed frozen test set was found yet" in html
    assert "split <code>test</code>" in html
    # Test set panel: one toggle, one percentage, one button. No status prose.
    assert 'id="library-splittest-enabled"' in html
    assert 'id="library-splittest-fraction-pct"' in html
    assert 'id="library-testset-build"' in html
    assert 'id="library-testset-save"' not in html
    assert 'id="library-testset-state"' not in html
    assert 'id="library-testset-stale-notice"' not in html
    assert 'id="library-testset-summary"' not in html
    assert "uv run sed dataset compose --project p" in html
    launch_panel = html.split("<h3>Launch</h3>", 1)[1].split("{# ===== Live ===== #}", 1)[0]
    assert "train-launch-panel" in launch_panel
    assert "train-launch-row" in launch_panel
    assert 'id="train-description"' in launch_panel
    assert 'id="train-readiness"' in launch_panel
    assert 'id="train-launch"' in launch_panel
    assert "Start training" in launch_panel
    assert 'id="train-launch-status"' in launch_panel
    assert ".train-launch-actions .primary" in html
    assert "Recommend settings" not in launch_panel
    assert "Reset to config" not in launch_panel
    assert 'id="train-recommend"' not in html
    assert 'id="train-reset-config"' not in html
    assert "applyRecommended" not in html
    assert 'wire("recommend"' not in html
    assert 'wire("reset-config"' not in html
    assert "project current feature" not in html
    # The frozen-test rule picker + its Evaluate button, and the separate
    # deployed-IR evaluate — the old single "re-evaluate target" control is gone.
    assert 'id="train-detail-evaluate"' in html
    assert html.count('id="train-detail-evaluate"') == 1
    assert 'id="train-detail-rule"' in html
    assert 'id="train-detail-evaluate-ir"' in html
    assert 'id="train-detail-reevaluate"' not in html
    assert 'id="train-detail-eval-target"' not in html
    assert 'id="train-detail-precision"' not in html
    assert 'class="train-mode-toggle"' not in html
    assert "data-train-mode" not in html
    assert "applyTrainMode" not in html
    assert "let TRAIN_MODE =" not in html
    assert "const TRAIN_MODE =" not in html
    assert "evaluateSelectedRunRule" in html
    assert "fp16" in html
    assert 'id="train-sampling_strategy"' not in html
    assert '"sampling_strategy"' not in html.split(
        "const TRAIN_SELECT_FIELDS =", 1,
    )[1].split(";", 1)[0]
    assert 'id="train-class_balance"' in html
    assert '<option value="off">natural</option>' in html
    assert '<option value="sqrt">moderate square-root</option>' in html
    assert '<option value="balanced">full 50/50 background-target</option>' in html
    assert '"class_balance"' in html.split(
        "const TRAIN_SELECT_FIELDS =", 1,
    )[1].split(";", 1)[0]
    assert 'wire("class_balance", "change", renderTrainAdvisor);' in html
    assert 'id="train-negative_weight"' not in html
    assert '"negative_weight"' not in html.split(
        "const TRAIN_FLOAT_FIELDS =", 1,
    )[1].split(";", 1)[0]
    assert 'enableField("train-negative_weight-field"' not in html
    for field_id in (
        "train-weight_decay",
        "train-warmup_epochs",
        "train-patience",
        "train-seed",
        "train-num_workers",
        "train-fine_tune_warmup_epochs",
        "train-freeze_batchnorm",
        "train-lr_layer_decay",
        "train-focal_gamma_pos",
        "train-focal_gamma_neg",
        "train-class_balance",
        "train-mixup_alpha",
        "train-specaugment_enabled",
        "train-specaugment_freq_mask",
        "train-specaugment_time_mask",
        "train-auto_export",
        "train-auto_test",
        "train-auto_test_threshold",
        "train-export_target",
        "train-export_verify_int8",
        "train-int8_max_regression",
        "train-calibrate_after",
        "train-calibrate_mode",
    ):
        assert f'id="{field_id}"' in html
    # After training is three steps — measure, refresh normalization, package —
    # not a decision-rule picker. Choosing which metrics to read is a Runs &
    # compare question now that scoring stores the per-clip probabilities.
    assert "Score on the frozen test" in html
    assert "Refresh BatchNorm statistics" in html
    assert "Prepare for deployment" in html
    for gone in ('id="train-after-rules"', 'id="train-calibrate_thresholds"',
                 'id="train-calibrate_negsrc"', "same decision rules as Batch evaluate"):
        assert gone not in html
    assert "val-calibrate test" not in html
    assert "BN recalibrate" not in html
    assert "What runs after training" not in html
    assert "Designed in the Audio Features tab" in html
    assert "read-only — designed" not in html
    assert "<summary>? " not in html
    assert "<summary>Learning rates &amp; precision</summary>" in html
    assert "Freeze BatchNorm and layer LR decay are in" in html
    assert ">Loss &amp; regularization</span>" in html
    assert html.count(">Loss &amp; regularization</span>") == 1
    assert "<summary>Control applicability</summary>" in html
    assert "<summary>Loss choices &amp; false-positive diagnosis</summary>" in html
    assert "Class balance is sampling, not a second loss weight." in html
    assert "Regularization deliberately limits memorization and overconfidence" in html
    assert "AdamW applies it directly" in html
    assert "Read these as different mechanisms:" not in html
    for term in ("TP / FP / FN", "macro average", "independent recording group", "leakage",
                 "FFT / window / hop", "sample rate / Nyquist", "BatchNorm statistics refresh",
                 "FP/hour deployment calibration", "OpenVINO"):
        assert term in html
    assert 'id="train-loss-objective-heading"' in html
    assert 'id="train-data-sampling-heading"' in html
    assert 'id="train-optimizer-transfer-heading"' in html
    assert 'id="train-loss-status"' in html
    assert "function trainLossContext()" in html
    assert "function renderTrainLossStatus()" in html
    assert "function renderTrainPrecisionStatus()" in html
    assert 'wire("precision", "change"' in html
    assert '"focal_gamma_pos", "focal_gamma_neg"' in html
    assert '"lr_layer_decay"' in html
    assert '<option value="32">32</option>' not in html
    assert '<option value="32-true">32-true — full FP32</option>' in html
    assert 'id="train-batch_size"' in html
    assert 'id="train-accumulate_grad_batches"' in html
    assert 'id="train-effective-batch-note"' in html
    assert "function updateTrainEffectiveBatch()" in html
    assert "physical batches" in html
    assert "optimizer steps" in html
    assert "GMAC/clip" in html
    assert "Quadro M620" in html
    assert "not</b> a universal recommendation" in html
    assert "Effectively unregularized" not in html
    assert "<summary>What runs after training</summary>" not in html
    assert "full 50/50 background-target (legacy sampler)" in html
    assert "natural + legacy loss weight" in html
    assert ".train-guide > summary::before" in html

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to syntax-check rendered inspect JavaScript")
    script = "\n".join(
        part.split("</script>", 1)[0]
        for part in html.split("<script>")[1:]
    )
    script_path = tmp_path / "inspect-rendered.js"
    script_path.write_text(script, encoding="utf-8")
    subprocess.run([node, "--check", str(script_path)], check=True)
    # "Save as project defaults" button is gone — panel autosaves and each
    # Add recipe takes a snapshot. The button id should be absent; the status
    # indicator stays for the autosave hint.
    assert 'id="augment-settings-save"' not in html
    assert "augment-settings-status" in html
    assert "augment-cancel" in html
    assert "existing-augmentation-list" in html
    assert "Loading generated augmentation counts" in html
    assert "/api/augmentation-generated-counts" in html
    assert "renderSelectionRules();\n        renderAugmentationRules();\n        renderExistingAugmentations();\n        loadAugmentationGeneratedCounts();" in html
    assert "library-rule-table-body" in html
    assert "augmentation-rule-table-body" in html
    assert "Generate selected" in html
    assert "lib-rows" in html
    assert "recipe-rows" in html
    assert "aug-subsection" in html
    assert "library-compose" not in html
    # redesigned bar hides the selection/delete controls off clip tabs via the
    # active-tab data-attribute rather than the old .library-active body class
    assert 'body[data-active-tab="augmented"]' in html


def test_library_frozen_test_panel_is_one_state_aware_button(tmp_path):
    """The Test set panel is a single action whose label states what it will do.

    Off with nothing built is a no-op; off with a built holdout offers to
    return those clips; on offers build/rebuild at the typed percentage. No
    separate status prose — the button label is the only state carrier.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered inspect JavaScript")
    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    script = "\n".join(part.split("</script>", 1)[0] for part in html.split("<script>")[1:])
    start = script.index("function libTestEl(id)")
    end = script.index("async function saveLibraryTestConfig()", start)
    gating_js = script[start:end]
    harness = f"""
let composeRunning = false;
let refreshRunning = false;
const enabled = {{ checked: false }};
const fraction = {{ disabled: false, value: "10" }};
const fractionField = {{ classList: {{ disabled: false, toggle(_name, value) {{ this.disabled = value; }} }} }};
const classes = new Set();
const build = {{
  disabled: true, textContent: "", dataset: {{}},
  classList: {{ toggle(name, on) {{ if (on) classes.add(name); else classes.delete(name); }} }},
  setAttribute() {{}},
}};
const elements = {{
  "library-splittest-enabled": enabled,
  "library-splittest-fraction-pct": fraction,
  "library-splittest-fraction-field": fractionField,
  "library-testset-build": build,
}};
const document = {{ getElementById(id) {{ return elements[id] || null; }} }};
function assert(ok, message) {{ if (!ok) throw new Error(message); }}
{gating_js}

// Off, nothing built: the default. There is nothing to apply.
LIBRARY_TESTSET = {{ enabled: false, fraction: 0, built: false, builtClips: 0, loaded: true }};
updateLibraryTestSetGating();
assert(fraction.disabled, "the percentage input follows the toggle");
assert(build.disabled, "off with nothing built has no action to offer");

// Off, but a holdout exists: the one button returns those clips.
LIBRARY_TESTSET = {{ enabled: true, fraction: 0.1, built: true, builtClips: 42, loaded: true }};
updateLibraryTestSetGating();
assert(!build.disabled, "a built holdout can always be removed");
assert(build.textContent.includes("42") && build.textContent.includes("training pool"),
       "the removal label names the clips it returns: " + build.textContent);
assert(build.dataset.testsetAction === "remove", "off + built is the remove action");

// On, already built: rebuild at the typed percentage.
enabled.checked = true;
updateLibraryTestSetGating();
assert(!fraction.disabled, "enabling the toggle enables the percentage");
assert(build.textContent === "Rebuild frozen test set (10%)", build.textContent);
assert(build.dataset.testsetAction === "build");

// On, typed percentage changes: the label follows it live.
fraction.value = "25";
updateLibraryTestSetGating();
assert(build.textContent === "Rebuild frozen test set (25%)", build.textContent);

// On, nothing built yet.
LIBRARY_TESTSET.built = false;
updateLibraryTestSetGating();
assert(build.textContent === "Build frozen test set (25%)", build.textContent);
"""
    script_path = tmp_path / "library-frozen-test-gating.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_inspect_tab_counts_only_render_on_sample_tabs(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        interactive=True,
        is_pretraining=True,
    )

    def tab_button(tab: str) -> str:
        marker = f'data-tab="{tab}"'
        marker_at = html.index(marker)
        start = html.rfind("<button", 0, marker_at)
        end = html.index("</button>", marker_at) + len("</button>")
        return html[start:end]

    for tab in ("library", "dataset", "augmented", "frozen", "mined"):
        assert '<span class="count"></span>' in tab_button(tab)

    for tab in ("model", "audiofeatures", "train"):
        assert '<span class="count"' not in tab_button(tab)

    # The objective selector applies to every run (supervised vs BYOL), not just
    # pretraining, so the heading is "Training objective".
    assert ">Training objective</span>" in html
    assert ">objective type" in html
    assert "<h4>Learning signal</h4>" not in html
    assert ">learning signal" not in html

    assert (
        'const countedTabs = new Set(["library", "dataset", "augmented", '
        '"frozen", "mined", "pretraining"]);'
    ) in html


def test_train_preset_depth_control_keeps_server_transfer_policy(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered train JavaScript")
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        interactive=True,
    )
    script = "\n".join(
        part.split("</script>", 1)[0]
        for part in html.split("<script>")[1:]
    )
    start = script.index("function trainFallbackPolicies()")
    end = script.index("function updateTrainFineTuneStatus()", start)
    train_depth_js = script[start:end]
    harness = f"""
let TRAIN_CONFIG = {{ model: {{ initialization: "parent", pretrained: false }}, training: {{ fine_tune_policy: "full" }} }};
let TRAIN_FINE_TUNE_OPTIONS = {{
  has_transfer_weights: true,
  n_stages: 5,
  policies: [{{id:"head"}}, {{id:"last_stage"}}, {{id:"last_two"}}, {{id:"full"}}],
}};
let TRAIN_PARENT_CANDIDATES = [{{ok:true}}];
let TRAIN_MODEL_OVERRIDES_DIRTY = false;
const depth = {{
  _html: "",
  set innerHTML(v) {{ this._html = String(v); }},
  get innerHTML() {{ return this._html; }},
  querySelectorAll() {{ return []; }},
  querySelector(sel) {{
    if (sel.includes(":checked")) {{
      const m = this._html.match(/name="train-depth-opt" value="([^"]+)" checked/);
      return m ? {{ value: m[1] }} : null;
    }}
    return null;
  }},
}};
const elements = {{
  depth,
  fine_tune_policy: {{ value: "" }},
  parent: {{ value: "" }},
  pretrained: {{ checked: false }},
}};
const document = {{
  getElementById(id) {{
    if (id === "train-depth-n") {{
      const m = depth.innerHTML.match(/id="train-depth-n"[^>]*value="([^"]+)"/);
      return m ? {{ value: m[1], addEventListener() {{}} }} : null;
    }}
    return null;
  }},
}};
function trainEl(id) {{ return elements[id] || null; }}
function updateTrainFineTuneStatus() {{}}
function updateTrainGuideRecs() {{}}
function renderTrainRegularizationStatus() {{}}
function renderTrainAdvisor() {{}}
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
{train_depth_js}

elements.fine_tune_policy.value = "head";
renderDepthControl();
assert(elements.fine_tune_policy.value === "head", "quick baseline should keep head policy");
assert(/value="head" checked/.test(depth.innerHTML), "head radio should be checked");

elements.fine_tune_policy.value = "last_two";
renderDepthControl();
assert(elements.fine_tune_policy.value === "last_two", "recommended preset should keep last_two");
assert(/value="lastn" checked/.test(depth.innerHTML), "last-N radio should be checked for last_two");
assert(/id="train-depth-n"[^>]*value="2"/.test(depth.innerHTML), "last_two should write N=2");

elements.fine_tune_policy.value = "last_n:3";
renderDepthControl();
assert(elements.fine_tune_policy.value === "last_n:3", "parametric last_n should be preserved");
assert(/id="train-depth-n"[^>]*value="3"/.test(depth.innerHTML), "last_n:3 should write N=3");

TRAIN_MODEL_OVERRIDES_DIRTY = true;
TRAIN_CONFIG.model.initialization = "random";
elements.fine_tune_policy.value = "head";
renderDepthControl();
assert(elements.fine_tune_policy.value === "full", "manual scratch override should force full");
assert(depth.innerHTML.includes("required for scratch init"), "scratch state should render full-only explanation");
"""
    script_path = tmp_path / "train-depth.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_train_loss_controls_follow_effective_objective(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered train JavaScript")
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    script = "\n".join(part.split("</script>", 1)[0] for part in html.split("<script>")[1:])
    start = script.index("function trainLossContext()")
    # Slice up to the next section banner rather than to one function's comment,
    # so re-wording a neighbour cannot break this test.
    end = script.index("// ---- After training", start)
    objective_js = script[start:end]
    harness = f"""
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
function classList() {{
  const values = new Set();
  return {{
    toggle(name, on) {{ if (on) values.add(name); else values.delete(name); }},
    contains(name) {{ return values.has(name); }},
  }};
}}
function input(value, extra={{}}) {{
  return Object.assign({{
    value: value == null ? "" : String(value), checked: false, disabled: false,
    type: "number", tagName: "INPUT", dataset: {{}},
    closest() {{ return null; }}, removeAttribute() {{}},
  }}, extra);
}}
function field(children=[]) {{
  return {{
    classList: classList(), dataset: {{}},
    querySelectorAll() {{ return children; }}, removeAttribute() {{}},
  }};
}}
const elements = {{
  pretraining_objective: input("supervised", {{tagName:"SELECT"}}),
  loss: input("bce_logits", {{tagName:"SELECT"}}),
  focal_gamma_pos: input("1"), focal_gamma_neg: input("4"),
  mixup_alpha: input("0.1"),
  class_balance: input("sqrt", {{tagName:"SELECT"}}),
  specaugment_enabled: input("", {{type:"checkbox", checked:true}}),
  specaugment_freq_mask: input("4"), specaugment_time_mask: input("4"),
  "loss-status": {{textContent:"", className:""}},
}};
const fields = {{
  "train-ssl_ema_decay-field": field([input("0.99")]),
  "train-class_balance-field": field([elements.class_balance]),
  "train-loss-field": field([elements.loss]),
  "train-focal_gamma_pos-field": field([elements.focal_gamma_pos]),
  "train-focal_gamma_neg-field": field([elements.focal_gamma_neg]),
  "train-mixup_alpha-field": field([elements.mixup_alpha]),
  "train-specaugment_freq_mask-field": field([elements.specaugment_freq_mask]),
  "train-specaugment_time_mask-field": field([elements.specaugment_time_mask]),
  "train-focal-controls": field([elements.focal_gamma_pos, elements.focal_gamma_neg]),
}};
elements["focal-controls"] = fields["train-focal-controls"];
const document = {{ getElementById(id) {{ return fields[id] || null; }} }};
function trainEl(id) {{ return elements[id] || null; }}
function trainSetFieldReason() {{}}
function trainSetInputTip() {{}}
let regRenders = 0, advisorRenders = 0, archDropoutRenders = 0;
function renderTrainRegularizationStatus() {{ regRenders += 1; }}
function renderTrainAdvisor() {{ advisorRenders += 1; }}
function renderTrainArchitectureDropout() {{ archDropoutRenders += 1; }}
let TRAIN_CONFIG = {{
  model: {{task:"clip", classes:["_negative", "clink"]}},
  training: {{pretraining_objective:"supervised", loss:"bce_logits"}},
  after_training: {{has_negative_class:true}}, class_counts: {{_negative:10, clink:10}},
}};
{objective_js}

updateTrainConditionalFields();
assert(!elements.loss.disabled, "supervised clip should expose the loss selector");
assert(elements.focal_gamma_pos.disabled, "BCE should disable focal gamma");
assert(!elements.mixup_alpha.disabled, "supervised clip BCE should expose mixup");
assert(!elements.class_balance.disabled, "supervised sampling should expose class balance");
assert(elements["loss-status"].textContent.includes("clip BCE-with-logits"), "BCE status should name the effective objective");

elements.loss.value = "asymmetric_focal";
TRAIN_CONFIG.after_training.has_negative_class = false;
TRAIN_CONFIG.model.classes = ["clink"];
TRAIN_CONFIG.class_counts = {{clink:10}};
updateTrainConditionalFields();
assert(!elements.focal_gamma_pos.disabled, "focal gamma+ should be enabled for focal loss");
assert(elements.focal_gamma_neg.disabled, "focal gamma- should require _negative");
assert(elements.mixup_alpha.disabled, "focal loss should disable mixup");
assert(elements["loss-status"].textContent.includes("no _negative class"), "status should explain inactive gamma-");

TRAIN_CONFIG.model.task = "frame";
updateTrainConditionalFields();
assert(elements.loss.disabled, "frame task should disable clip loss");
assert(elements.focal_gamma_pos.disabled && elements.mixup_alpha.disabled, "frame task should disable focal and mixup");
assert(!elements.class_balance.disabled, "frame task still uses supervised class sampling");
assert(elements["loss-status"].textContent.includes("frame BCE-with-logits"), "frame status should name fixed BCE");

TRAIN_CONFIG.model.task = "clip";
elements.pretraining_objective.value = "byol";
updateTrainConditionalFields();
assert(elements.loss.disabled, "BYOL should disable supervised loss");
assert(elements.focal_gamma_pos.disabled && elements.mixup_alpha.disabled, "BYOL should disable focal and mixup");
assert(elements.class_balance.disabled, "BYOL should disable class balance");
assert(elements["loss-status"].textContent.includes("BYOL cosine"), "BYOL status should name its effective objective");
assert(regRenders === 4 && advisorRenders === 4, "conditional refresh should keep status and advisor live");
"""
    script_path = tmp_path / "train-loss-objective.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_train_dropout_auto_js_changes_between_frozen_and_unfrozen_encoder(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered train JavaScript")
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    script = "\n".join(part.split("</script>", 1)[0] for part in html.split("<script>")[1:])
    start = script.index("function trainDropoutValueActive")
    end = script.index("// Keep every hyperparameter visible", start)
    dropout_js = script[start:end]
    harness = f"""
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
const elements = {{
  fine_tune_policy: {{value:"head"}},
  pretraining_objective: {{value:"supervised"}},
}};
function trainEl(id) {{ return elements[id] || null; }}
function trainLossContext() {{ return {{byol: elements.pretraining_objective.value === "byol"}}; }}
let TRAIN_CONFIG = {{model:{{dropout:{{rows:[
  {{name:"head_dropout", location:"classifier_head", mechanism:"dropout", value:0.2}},
  {{name:"pool_dropout", location:"pooled_embedding", mechanism:"dropout", value:0.3}},
]}}}}}};
{dropout_js}
assert(trainAutoHeadDropoutResolved().value === 0.2, "frozen backbone needs classifier dropout");
elements.fine_tune_policy.value = "full";
assert(trainAutoHeadDropoutResolved().value === 0.0, "active pooled embedding dropout replaces classifier dropout");
elements.pretraining_objective.value = "byol";
elements.fine_tune_policy.value = "head";
assert(trainAutoHeadDropoutResolved().value === 0.0, "BYOL trains the encoder even though supervised depth is hidden");
"""
    script_path = tmp_path / "train-dropout-auto.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_model_feature_edit_js_loads_contract_and_saved_legacy_preset(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered model JavaScript")
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    script = "\n".join(part.split("</script>", 1)[0] for part in html.split("<script>")[1:])
    start = script.index("function modelLabEl")
    end = script.index("function modelLabToggleFrameSection()", start)
    model_js = script[start:end]
    harness = f"""
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
function escapeHtml(value) {{ return String(value ?? ""); }}
function makeSelect(value) {{
  return {{
    value: value || "",
    options: [],
    _html: "",
    set innerHTML(v) {{
      this._html = String(v);
      this.options = Array.from(this._html.matchAll(/<option value='([^']*)'>(.*?)<\\/option>/g))
        .map(m => ({{ value: m[1], textContent: m[2].replace(/<[^>]+>/g, "") }}));
      if (!this.value && this.options.length) this.value = this.options[0].value;
    }},
    get innerHTML() {{ return this._html; }},
  }};
}}
let written = null;
let clicked = false;
const elements = {{
  "model-family": {{ value: "efficientat-mn" }},
  "model-weight": {{ value: "efficientat:mn10_as_mels_40" }},
  "model-initialization": {{ value: "upstream" }},
  "model-feature-source": makeSelect(""),
}};
const document = {{
  getElementById(id) {{ return elements[id] || null; }},
  querySelector(sel) {{ return sel.includes("audiofeatures") ? {{ click() {{ clicked = true; }} }} : null; }},
  querySelectorAll() {{ return []; }},
}};
let MODEL_CATALOG = {{
  architectures: [{{
    id: "efficientat-mn",
    provider: "efficientat",
    source: "mn10_as",
    capabilities: ["clip"],
    weights: [{{
      id: "efficientat:mn10_as_mels_40",
      label: "mn10_as_mels_40",
      expected_feature_contract: {{
        "features.feature_type": "mel",
        "model.sample_rate": 32000,
        "features.n_fft": 1024,
        "features.win_length": 800,
        "features.hop_length": 320,
        "features.n_mels": 40,
        "features.fmin": 0,
        "features.fmax": 15000,
        "features.window_seconds": 1.0,
      }},
    }}],
    default_feature: {{feature_type:"mel", frontend:"efficientat", sample_rate:32000, n_fft:1024, win_length:800, hop_length:320, n_mels:128, fmin:0, fmax:15000, window_seconds:0.4}},
  }}],
  feature: {{ feature_type: "linear", n_mels: 99, hop_length: 123 }},
}};
function featureWriteForm(preset, source) {{ written = {{preset, source}}; }}
let FEATURE_CONTRACT = null;
const FEATURE_VIRTUAL_DEFAULTS = new Map();
function featureRememberVirtualDefault(payload, source) {{
  const saved = Object.assign({{}}, payload, {{name:source.name, label:source.label, source_kind:source.source_kind}});
  FEATURE_VIRTUAL_DEFAULTS.set(saved.name, saved); return saved;
}}
{model_js}
MODEL_PRESETS = [{{
  name: "legacy-sounddetection",
  label: "Legacy soundDetection",
  feature_type: "linear",
  sample_rate: 44100,
  n_fft: 190,
  win_length: 190,
  hop_length: 167,
  n_mels: 128,
  fmin: 0,
  fmax: 22050,
  window_seconds: 0.4,
  log_power: true,
  stft_engine: "scipy",
  db_norm: true,
  db_min: -20,
  db_max: 60,
  colormap: "jet",
  image_size: 96,
}}];

modelLabPopulateFeatureSource();
assert(elements["model-feature-source"].options[0].value === "contract:mn10_as_mels_40", "exact contract should be the first feature option");
assert(elements["model-feature-source"].options[0].textContent === "mn10_as_mels_40 contract", "contract option should be named after the selected weight");
assert(elements["model-feature-source"].value === "contract:mn10_as_mels_40", "exact contract should be the default feature source");

modelLabGotoFeatures();
assert(clicked, "audio features tab should be opened");
assert(written.source.kind === "contract", "contract source should be marked");
assert(written.source.name === "contract:mn10_as_mels_40", "contract virtual name should use selected weight");
assert(written.preset.n_mels === 40, "contract mel variant should be loaded, not stale feature state");
assert(written.preset.window_seconds === 0.4, "new clip recipes should use the task-native window");
assert(written.preset.upstream_window_seconds === 1.0, "upstream crop duration should remain visible as provenance");
assert(FEATURE_VIRTUAL_DEFAULTS.get("contract:mn10_as_mels_40").n_mels === 40, "handoff should cache the exact virtual checkpoint payload");

elements["model-initialization"].value = "random";
elements["model-weight"].value = "scratch:mn01";
written = null;
modelLabGotoFeatures();
assert(written.source.source_kind === "architecture_default", "scratch MN should be labelled as an architecture default");
assert(written.source.name === "contract:mn01", "scratch MN default should use its exact width name");
assert(FEATURE_VIRTUAL_DEFAULTS.get("contract:mn01").n_mels === 128, "scratch MN handoff should cache a loadable default payload");

elements["model-feature-source"].value = "legacy-sounddetection";
written = null;
modelLabGotoFeatures();
assert(written.source.kind === "saved", "saved preset source should be marked");
assert(written.preset.feature_type === "linear", "legacy preset feature type should load");
assert(written.preset.stft_engine === "scipy", "legacy scipy engine should load");
assert(written.preset.db_norm === true, "legacy dB normalization should load");
assert(written.preset.colormap === "jet", "legacy JET colormap should load");
"""
    script_path = tmp_path / "model-feature-edit.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_model_catalog_js_lists_all_supported_and_disables_external(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered model JavaScript")
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    script = "\n".join(part.split("</script>", 1)[0] for part in html.split("<script>")[1:])
    start = script.index("function modelLabEl")
    end = script.index("function modelLabToggleFrameSection()", start)
    model_js = script[start:end]
    harness = f"""
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
function escapeHtml(value) {{ return String(value ?? ""); }}
const elements = {{
  "model-task": {{ value: "clip" }},
  "model-family": {{ value: "", innerHTML: "" }},
  "model-weight": {{ value: "", innerHTML: "", closest() {{ return null; }} }},
  "model-initialization": {{
    value: "upstream",
    options: [
      {{value:"upstream", disabled:false, textContent:"pretrained"}},
      {{value:"parent", disabled:false, textContent:"parent"}},
      {{value:"random", disabled:false}},
    ],
  }},
  "model-source-card": {{ innerHTML: "", textContent: "" }},
  "model-constructor-note": {{ textContent: "", className: "" }},
  "model-param-chips": {{ innerHTML: "", querySelectorAll() {{ return []; }} }},
  "model-effective-params": {{ innerHTML: "", querySelectorAll() {{ return []; }} }},
  "model-effective-json": {{ textContent: "" }},
  "model-params-reset": {{ disabled: false }},
  "model-params-status": {{ textContent: "", className: "" }},
  "model-recipe-save": {{ disabled: false }},
  "model-verify": {{ disabled: false }},
}};
const document = {{
  getElementById(id) {{ return elements[id] || null; }},
  querySelectorAll() {{ return []; }},
}};
let MODEL_CATALOG = {{
  active_recipe: {{}},
  tag_definitions: {{
    task_specific_candidate: {{label:"Task-specific candidate", description:"Task-specific local benchmark required.", tone:"research"}},
    no_upstream_weights: {{label:"No upstream weights", description:"No upstream checkpoint.", tone:"neutral"}},
  }},
  architectures: [
    {{id:"efficientat-mn", label:"EfficientAT MN", provider:"efficientat", source:"mn10_as", capabilities:["clip"], export_supported:true, trusted:true, weights:[{{id:"efficientat:mn10_as", label:"mn10_as", upstream_task:"AudioSet"}}], params:[], constructor:{{fixed:true, params:[], note:"fixed"}}}},
    {{id:"transient-cnn-gap", label:"Transient CNN GAP", provider:"local", source:"sed.train.short_audio:TransientCNNGAP", capabilities:["clip"], export_supported:true, trusted:true, weights:[], params:[{{name:"time_stride", type:"choice", default:1, choices:[1,2], help:"time stride"}}], guidance:[{{name:"time_stride", default:1, recommended:1, effect:"time stride"}}], constructor:{{fixed:false, params:[], note:"declared"}}}},
    {{id:"clink-cnn-v2", label:"ClinkCNN-v2", provider:"local", source:"sed.train.clink_cnn_v2:ClinkCNNv2", capabilities:["clip"], export_supported:true, trusted:true, tags:["task_specific_candidate","no_upstream_weights"], weights:[], params:[], constructor:{{fixed:true, params:[], note:"fixed topology"}}}},
    {{id:"efficient-sed-fmn10", label:"EfficientSED fMN10", provider:"local", source:"sed.train.efficient_sed:EfficientSEDfMN10", capabilities:["clip","frame"], selection_tasks:["frame"], export_supported:false, trusted:true, weights:[{{id:"efficientsed:fmn10:strong", label:"fmn10_strong", upstream_task:"AudioSet Strong"}}], params:[], constructor:{{fixed:true, params:[]}}}},
    {{id:"untrusted-external-demo", label:"Untrusted External Demo", provider:"local", source:"sed.train.external_audio:DoesNotExist", capabilities:["clip"], export_supported:false, trusted:false, weights:[{{id:"hf:demo:untrusted", label:"demo/untrusted"}}], params:[], constructor:{{fixed:true, params:[]}}}},
  ],
}};
const MODEL_PRIMARY_VARIANTS = new Set(["efficientat:mn10_as"]);
{model_js}
modelLabRefreshFit = function() {{}};
modelLabPopulateFeatureSource = function() {{}};
modelLabRenderFamily();
const htmlOut = elements["model-family"].innerHTML;
assert(htmlOut.includes('value="transient-cnn-gap"'), "transient models should be listed");
assert(htmlOut.includes('value="clink-cnn-v2"'), "ClinkCNN-v2 should be listed");
assert(!htmlOut.includes('value="efficient-sed-fmn10"'), "frame-native fMN should be hidden from new clip recipes");
assert(/value="untrusted-external-demo"[^>]*disabled/.test(htmlOut), "unsupported external should be disabled");
assert(htmlOut.includes("unavailable"), "disabled external option should be labelled unavailable");
assert(elements["model-family"].value === "efficientat-mn", "first selectable architecture should be selected");
elements["model-family"].value = "clink-cnn-v2";
modelLabRenderVariants();
assert(elements["model-source-card"].innerHTML.includes("Task-specific candidate"), "model tags should render catalog labels");
assert(elements["model-source-card"].innerHTML.includes("Task-specific local benchmark required."), "model tag tooltip text should come from catalog metadata");
elements["model-initialization"].value = "parent";
elements["model-family"].value = "clink-cnn-v2";
modelLabRenderVariants();
assert(elements["model-initialization"].value === "parent", "no-weight architecture must preserve parent initialization");
assert(elements["model-initialization"].options[0].disabled === true, "upstream must stay disabled when no published weights exist");
assert(elements["model-source-card"].innerHTML.includes("Parent initialization"), "source card should explain encoder-only parent transfer");
elements["model-task"].value = "frame";
modelLabRenderFamily();
const frameHtml = elements["model-family"].innerHTML;
assert(frameHtml.includes('value="efficient-sed-fmn10"'), "fMN should be offered for new frame recipes");
assert(!frameHtml.includes('value="clink-cnn-v2"'), "clip-only models should be hidden from frame recipes");
"""
    script_path = tmp_path / "model-catalog-list.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_dataset_shape_reports_size_and_recording_group_redundancy(tmp_path, monkeypatch):
    """The Model tab's recommendations key off redundancy as well as size, so the
    shape summary must count distinct sessions per class — from the composed CSV
    only, never the shared library."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect_server import _dataset_shape

    layout = ProjectLayout("p")
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = ["path,class_name,session_id"]
    # cheers: 6 clips across 3 sessions. _negative: 4 clips, all one session.
    for i in range(6):
        rows.append(f"/a/{i}.wav,cheers,sess{i % 3}")
    for i in range(4):
        rows.append(f"/b/{i}.wav,_negative,neg0")
    (layout.dataset_dir / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    shape = _dataset_shape(layout)
    assert shape["rows"] == 10
    assert shape["by_class"] == {"cheers": 6, "_negative": 4}
    assert shape["n_classes"] == 2
    assert shape["negative_share"] == pytest.approx(0.4)
    # The scarcest class drives the advice, not the average.
    assert shape["min_class_groups"] == 1
    assert shape["min_class_rows"] == 4


def test_dataset_shape_is_empty_without_a_composed_pool(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect_server import _dataset_shape

    layout = ProjectLayout("p")
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    assert _dataset_shape(layout)["rows"] == 0


def test_rendered_model_and_train_tabs_use_one_constructor_and_dropout_stack(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    assert html.count("<h4>Constructor parameters</h4>") == 1
    assert 'id="model-param-chips"' in html
    assert 'id="model-effective-params"' in html
    assert 'id="model-effective-json"' in html
    assert 'id="model-params-reset"' in html
    assert "Raw constructor" not in html
    assert "Advanced constructor" not in html
    assert "model-advanced-params" not in html
    assert "+ any custom constructor kwargs" not in html

    assert 'id="train-head_dropout_auto"' in html
    assert 'id="train-edit-model-dropout"' in html
    assert "not added together" in html
    # Auto must show the value it resolved to and why, so the checkbox is not a
    # silent rewrite of the number beside it.
    assert 'id="train-head_dropout-auto-note"' in html
    assert "trainAutoHeadDropoutResolved" in html
    # Both tabs describe a mechanism's location with the same words.
    assert "function dropoutLocationLabel" in html
    assert "encoder · pooled embedding" in html
    # Architecture-fixed regularization is visible in the Model tab too, so a
    # model whose only dropout is built in does not look like it has none.
    assert "modelLabFixedRegularizationHtml" in html


def test_constructor_parameter_js_omits_disabled_optional_and_parses_integer_lists(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered model JavaScript")
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    script = "\n".join(part.split("</script>", 1)[0] for part in html.split("<script>")[1:])
    start = script.index("function modelLabReadTypedParams()")
    end = script.index("function modelLabBackbone()", start)
    parameter_js = script[start:end]
    harness = f"""
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
const architecture = {{params:[
  {{name:"strides", type:"int_list", default:[2,2,2,2], optional:true}},
  {{name:"depth", type:"int", default:4}},
]}};
const fields = [
  {{dataset:{{modelParam:"strides", modelParamType:"int_list"}}, value:"1, 2, 2, 1", disabled:true}},
  {{dataset:{{modelParam:"depth", modelParamType:"int"}}, value:"6", disabled:false}},
];
const includes = [{{dataset:{{modelParamEnabled:"strides"}}, checked:false}}];
const rows = [];
const elements = {{
  "effective-params": {{innerHTML:"", querySelectorAll() {{ return rows; }}}},
  "effective-json": {{textContent:""}},
  "params-status": {{textContent:"", className:""}},
}};
const document = {{
  querySelectorAll(sel) {{
    if (sel === "[data-model-param]") return fields;
    if (sel === "[data-model-param-enabled]") return includes;
    return [];
  }},
}};
let MODEL_CATALOG = {{train_rows:1200}};
function modelLabEl(id) {{ return elements[id] || null; }}
function modelLabCurrentArchitecture() {{ return architecture; }}
function modelLabRefreshFit() {{}}
function modelLabReloadParentSoon() {{}}
{parameter_js}
let value = modelLabReadTypedParams();
assert(JSON.stringify(value) === JSON.stringify({{depth:6}}), "disabled optional key must be omitted");
includes[0].checked = true;
fields[0].disabled = false;
value = modelLabReadTypedParams();
assert(JSON.stringify(value.strides) === JSON.stringify([1,2,2,1]), "integer list should parse");
assert(value.depth === 6, "integer field should stay typed");
modelLabResetParams();
assert(includes[0].checked === false && fields[0].disabled === true, "reset should omit optional keys");
assert(fields[1].value === "4", "reset should restore declared defaults");
"""
    script_path = tmp_path / "constructor-params.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_audio_feature_preset_js_tracks_dirty_save_and_overwrite(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered feature JavaScript")
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    script = "\n".join(part.split("</script>", 1)[0] for part in html.split("<script>")[1:])
    helpers_start = script.index("function featureIsSavedPresetName")
    helpers_end = script.index("async function featureLoadAll", helpers_start)
    save_start = script.index("async function featureSavePreset", helpers_end)
    save_end = script.index("function featureLoadSelected", save_start)
    load_start = script.index("function featureLoadSelected", save_end)
    load_end = script.index("async function featureDeletePreset", load_start)
    feature_js = script[helpers_start:helpers_end] + "\n" + script[save_start:save_end] + "\n" + script[load_start:load_end]
    harness = f"""
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
function makeSelect() {{
  return {{
    value: "",
    options: [],
    _html: "",
    set innerHTML(v) {{ this._html = String(v); this.options = []; }},
    get innerHTML() {{ return this._html; }},
    // Like the real DOM: appending an <optgroup> surfaces its child options in
    // select.options.
    appendChild(opt) {{
      const leaves = (opt && opt.children && opt.children.length) ? opt.children : [opt];
      leaves.forEach(o => {{ this.options.push(o); if (!this.value) this.value = o.value; }});
    }},
  }};
}}
const elements = {{
  "feature-preset-load": makeSelect(),
  "library-feature-preset": makeSelect(),
  "feature-preset-note": {{ className: "", textContent: "" }},
  "feature-preset-name": {{ value: "" }},
  "feature-preset-description": {{ value: "" }},
  "feature-preset-status": {{ className: "", textContent: "" }},
}};
const document = {{
  getElementById(id) {{ return elements[id] || null; }},
  createElement(tag) {{ return {{ tagName: String(tag).toUpperCase(), value: "", textContent: "", label: "", disabled: false, children: [], appendChild(c) {{ this.children.push(c); }} }}; }},
}};
function featureLabEl(id) {{ return document.getElementById("feature-" + id); }}
let loadedVirtual = null;
function featureWriteForm(preset, source) {{ loadedVirtual = {{preset, source}}; }}
function featureRefreshPreview() {{}}
let FEATURE_PRESETS_CACHE = [{{name:"existing", label:"Existing", feature_type:"mel"}}];
let FEATURE_PREVIEW_PRESET = "";
let FEATURE_LOADING_FORM = false;
let FEATURE_CURRENT_SOURCE = {{kind:"custom", name:"", label:"Custom audio feature", base_kind:"", base_name:""}};
let FEATURE_CONTRACT = null;
const FEATURE_VIRTUAL_DEFAULTS = new Map();
let MODEL_PRESETS = [];
function featureReadForm() {{ return {{ feature_type:"mel", sample_rate:32000, n_mels:64 }}; }}
let confirmCount = 0;
function confirm(msg) {{ confirmCount += 1; return true; }}
async function fetch(url, opts) {{
  const body = JSON.parse(opts.body);
  return {{
    async json() {{
      return {{ ok:true, preset:Object.assign({{label:body.name}}, body.preset, {{name:body.name}}) }};
    }}
  }};
}}
{feature_js}

(async () => {{
featureSetCurrentSource({{kind:"contract", name:"contract:mn10_as", label:"Contract: mn10_as"}}, {{n_fft:1024, n_mels:128}});
assert(FEATURE_CURRENT_SOURCE.kind === "contract", "contract source should be tracked");
assert(elements["feature-preset-load"].value === "contract:mn10_as", "virtual contract should be selected in load dropdown");
assert(elements["feature-preset-note"].textContent.includes("Loaded checkpoint contract"), "contract note should be visible");
FEATURE_CONTRACT = null; // emulate a stale API source after the Model-tab handoff
featureLoadSelected();
assert(loadedVirtual && loadedVirtual.preset.n_mels === 128, "cached virtual default should load even when the API source is stale");

featureMarkCustom();
assert(FEATURE_CURRENT_SOURCE.kind === "custom", "dirty edit should switch source to custom");
assert(FEATURE_CURRENT_SOURCE.base_kind === "contract", "custom source should remember contract base");
assert(elements["feature-preset-load"].value === "", "custom design should select the unsaved dropdown option");

elements["feature-preset-name"].value = "new-preset";
await featureSavePreset();
assert(elements["feature-preset-load"].value === "new-preset", "save-as should select the newly saved preset");
assert(FEATURE_CURRENT_SOURCE.kind === "saved", "save-as should mark saved source");
assert(FEATURE_CURRENT_SOURCE.name === "new-preset", "saved source should use new preset name");

elements["feature-preset-name"].value = "existing";
await featureSavePreset();
assert(confirmCount === 1, "overwriting an existing preset should ask for confirmation");
assert(elements["feature-preset-load"].value === "existing", "overwrite should keep overwritten preset selected");
}})().catch(err => {{ console.error(err); process.exit(1); }});
"""
    script_path = tmp_path / "feature-preset-source.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_train_metric_formatting_keeps_decimal_with_scientific_hint(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered train JavaScript")

    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        interactive=True,
    )
    script = "\n".join(
        part.split("</script>", 1)[0]
        for part in html.split("<script>")[1:]
    )
    start = script.index("function trainEl")
    end = script.index("function trainElapsed", start)
    formatting_js = script[start:end]
    harness = f"""
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
{formatting_js}
assert(trainFmtMetric(0.0001) === "0.0001", "decimal value should stay visible");
assert(trainSci(0.0001) === "1e-4", "scientific companion should be compact");
assert(trainFmtMetricHtml(0.0001, 4, true).includes("(1e-4)"), "html should include companion");
assert(trainSci(0.8912) === "", "normal score values should not get scientific noise");
assert(trainFmtMetricHtml(0.8912, 4, true) === "0.8912", "normal score html should remain plain");
"""
    script_path = tmp_path / "train-formatting.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_train_validation_section_renders_class_counts_and_zero_reason(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered train JavaScript")

    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        interactive=True,
    )
    script = "\n".join(
        part.split("</script>", 1)[0]
        for part in html.split("<script>")[1:]
    )
    render_start = script.index("function renderTrainValClassCounts()")
    render_end = script.index("function renderTrainReadiness", render_start)
    render_js = script[render_start:render_end]
    preview_start = script.index("function updateValFractionPreview()")
    preview_end = script.index("// training starts", preview_start)
    preview_js = script[preview_start:preview_end]
    harness = f"""
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
function trainEsc(s) {{ return String(s == null ? "" : s); }}
const elements = {{
  "val-class-counts": {{ innerHTML: "", textContent: "" }},
  "val-fraction-preview": {{ textContent: "" }},
  "val_fraction_pct": {{ value: "25" }},
}};
function trainEl(id) {{ return elements[id] || null; }}
let TRAIN_CONFIG = {{
  counts: {{ train: 62878, val: 7993, test: 4419 }},
  dataset_config: {{
    val_fraction: 0.25,
    training_class_counts: {{ _negative: 39731, attention: 6340, cheers: 16807 }},
    validation_class_counts: {{ _negative: 0, attention: 2249, cheers: 5744 }},
    split_diagnostics: {{ by_class: {{ _negative: {{ warnings: ["class '_negative' has no source bucket with at least two independent split groups; kept all clips in train"] }} }} }},
  }},
}};
{render_js}
{preview_js}

renderTrainValClassCounts();
assert(elements["val-class-counts"].innerHTML.includes("Validation by class"), "summary label should render");
assert(elements["val-class-counts"].innerHTML.includes("_negative"), "negative class should render even at zero");
assert(elements["val-class-counts"].innerHTML.includes("2,249"), "attention validation count should render");
assert(elements["val-class-counts"].innerHTML.includes("5,744"), "cheers validation count should render");
assert(elements["val-class-counts"].innerHTML.includes("no source bucket"), "zero-class reason should be visible");
assert(elements["val-class-counts"].innerHTML.includes("class-count zero"), "zero count should be highlighted");

updateValFractionPreview();
assert(elements["val-fraction-preview"].textContent.includes("7,993 val / 62,878 train"), "saved fraction should show the exact grouped preview");
elements["val_fraction_pct"].value = "20";
updateValFractionPreview();
assert(elements["val-fraction-preview"].textContent.includes("before grouping"), "unsaved fraction should be labelled approximate");
assert(elements["val-fraction-preview"].textContent.includes("Save to compute the exact split"), "typed fraction should explain how to refresh exact counts");
"""
    script_path = tmp_path / "train-validation-counts.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_subgroup_dropdown_lists_every_axis_best_first_and_follows_the_summary(tmp_path):
    """The rendered dropdown, exercised against the shapes real data produces.

    It must offer every axis the server kept (they suit different corpora), lead
    with the most informative one, and describe exactly the subbuckets the
    Summary panel is letting through — the two controls share one inventory.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered inspect JavaScript")

    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    script = "\n".join(
        part.split("</script>", 1)[0] for part in html.split("<script>")[1:]
    )
    start = script.index("const AXIS_LABELS = {")
    end = script.index("function subgroupFilterHtml", start)
    subgroup_js = script[start:end]
    esc_start = script.index("function escapeHtml(str)")
    esc_end = script.index("\nfunction ", esc_start + 1)
    escape_js = script[esc_start:esc_end]

    harness = f"""
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
{escape_js}
let EXCLUDED = new Set();
function summaryExcludedSet() {{ return EXCLUDED; }}
let SELECTED = null;
function activeSubgroup() {{ return SELECTED; }}
const CLIPS_DATA = {{ panel_subgroups: {{ dataset: {{ cheers: {{
  "_cheers": {{
    prefix: {{ champ_aug_01: 234, cheers_bim256: 500 }},
    folder: {{ "_cheers/_champaigne-19-PNY-299": 234, "_cheers/_cheers": 500 }},
  }},
  "cheersRandom": {{ prefix: {{ rnd_a: 3, rnd_b: 4, rnd_c: 5 }} }},
}} }} }} }};
{subgroup_js}

let html = subgroupOptionsHtml("dataset", "cheers");
// Every axis is offered, and the finest browsable one leads.
assert(html.indexOf("by name prefix (5)") < html.indexOf("by folder (2)"),
       "the most informative axis should be listed first");
assert(html.includes("all subgroups"), "an unfiltered option must exist");
// Counts merge across the checked subbuckets.
assert(html.includes("champ_aug_01 — 234"), "group label should carry its count");
assert(html.includes("rnd_a — 3"), "a second subbucket's groups should merge in");
// Folder labels drop the prefix every option shares, but the value stays whole
// so it still matches server-side.
assert(html.includes(">_champaigne-19-PNY-299 — 234<"), "shared folder prefix should be trimmed");
assert(html.includes('["folder","_cheers/_champaigne-19-PNY-299"]'.replace(/"/g, "&quot;")),
       "the option value must keep the full key");

// Unchecking a subbucket in the Summary panel removes its groups from the list.
EXCLUDED = new Set(["cheersRandom"]);
html = subgroupOptionsHtml("dataset", "cheers");
assert(!html.includes("rnd_a"), "an unchecked subbucket must not offer subgroups");
assert(html.includes("by name prefix (2)"), "axis counts should follow the checked buckets");

// A class nothing subdivides says so instead of offering a pointless option.
assert(subgroupOptionsHtml("dataset", "attention") === '<option value="">no subgroups</option>');
"""
    script_path = tmp_path / "subgroup-dropdown.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_train_runs_select_all_toggles_from_selection_state(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered train JavaScript")

    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        interactive=True,
    )
    script = "\n".join(
        part.split("</script>", 1)[0]
        for part in html.split("<script>")[1:]
    )
    start = script.index("function selectedTrainRunIds()")
    end = script.index("function clearTrainRunDetail()", start)
    selection_js = script[start:end]
    harness = f"""
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
let TRAIN_RUNS = [{{model_id:"run-a"}}, {{model_id:"run-b"}}, {{model_id:"run-c"}}];
let TRAIN_SELECTED_RUNS = new Set();
const boxes = TRAIN_RUNS.map(r => ({{ value: r.model_id, checked: false }}));
const elements = {{
  "runs-delete": {{ disabled: true, textContent: "" }},
  "runs-select-all": {{ checked: false, indeterminate: false, disabled: false }},
  "runs-status": {{ textContent: "", dataset: {{}} }},
  "runs-compare": {{ disabled: true, textContent: "" }},
  "runs-batch": {{ disabled: true, textContent: "" }},
  "batch-count": {{ textContent: "" }},
}};
const document = {{
  querySelectorAll(sel) {{
    return sel === "#train-runs-body input[data-run-select]" ? boxes : [];
  }},
}};
function trainEl(id) {{ return elements[id] || null; }}
{selection_js}

toggleAllTrainRunSelection({{target: {{checked: true}}}});
assert(selectedTrainRunIds().join(",") === "run-a,run-b,run-c", "first click should select every run");
assert(boxes.every(box => box.checked), "row boxes should all be checked after select-all");
assert(elements["runs-select-all"].checked === true, "header should be checked after select-all");

toggleAllTrainRunSelection({{target: {{checked: true}}}});
assert(selectedTrainRunIds().length === 0, "second click should clear every run even if event checked is stale");
assert(boxes.every(box => !box.checked), "row boxes should all be unchecked after clearing");
assert(elements["runs-select-all"].checked === false, "header should be unchecked after clearing");

TRAIN_SELECTED_RUNS = new Set(["run-a"]);
updateTrainRunSelectionControls();
toggleAllTrainRunSelection({{target: {{checked: false}}}});
assert(selectedTrainRunIds().length === 3, "partial selection should select all runs");
assert(elements["runs-select-all"].indeterminate === false, "header should not stay indeterminate after selecting all");
"""
    script_path = tmp_path / "train-runs-select-all.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_train_live_metrics_table_labels_accuracy_and_byol_score(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered train JavaScript")

    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={},
        interactive=True,
    )
    script = "\n".join(
        part.split("</script>", 1)[0]
        for part in html.split("<script>")[1:]
    )
    start = script.index("function trainEl")
    end = script.index("function drawTrainChart", start)
    live_js = script[start:end]
    harness = f"""
function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
const elements = {{
  "train-live-metrics": {{ innerHTML: "" }},
  "train-live-metric-rows": {{ innerHTML: "" }},
  "train-live-score-head": {{ textContent: "", title: "" }},
}};
const document = {{ getElementById(id) {{ return elements[id] || null; }} }};
let TRAIN_CONFIG = {{ metric_descriptor: {{ show_val_accuracy: true, val_accuracy_label: "val acc", val_accuracy_description: "acc" }} }};
{live_js}
renderTrainLiveMetrics([{{epoch:0, train_loss:0.1, val_loss:0.2, val_f1:0.3, val_accuracy:0.4, lr:0.0001}}]);
assert(elements["train-live-score-head"].textContent === "val acc", "supervised header should be accuracy");
assert(elements["train-live-metric-rows"].innerHTML.includes("<td>0.4</td>"), "accuracy should render beside F1");
TRAIN_CONFIG = {{ metric_descriptor: {{ show_val_accuracy: false, val_score_label: "BYOL score", val_score_description: "score" }} }};
renderTrainLiveMetrics([{{epoch:0, train_loss:0.1, val_loss:0.2, val_byol_score:-0.2, lr:0.0001}}]);
assert(elements["train-live-score-head"].textContent === "BYOL score", "BYOL header should not say accuracy");
assert(elements["train-live-metric-rows"].innerHTML.includes("<td>-0.2</td>"), "BYOL score should render in score column");
"""
    script_path = tmp_path / "train-live-metrics.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_dataset_gather_includes_selected_external_positives(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.data import library as lib_mod
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    layout = ProjectLayout("p")
    layout.create()

    src = tmp_path / "src.wav"
    sf.write(str(src), np.zeros(32000, dtype="float32"), 32000)
    lib = lib_mod.Library()
    clip = lib.add_clip(
        source_path=src,
        category="positives",
        label="cling_ikea_type1",
        sample_rate=32000,
        channels=1,
        duration_s=1.0,
        move=False,
    )

    from sed.config import load_config
    cfg = load_config()
    cfg.model.classes = ["_negative", "cheers_loud"]
    cfg.dataset.external_positives_pool = str(tmp_path / "data_library" / "positives")
    cfg.dataset.external = {
        "cheers_loud": [
            {"category": "positives", "labels": ["cling_ikea_type1"]}
        ]
    }

    from sed.data.inspect import _gather_clips
    items = _gather_clips(layout, cfg)

    assert (clip.absolute_path(), "cheers_loud") in [
        (path, label) for path, label, _meta in items
    ]


def test_augmented_search_text_includes_source_and_overlay_filenames():
    from sed.data.inspect import _augmentation_search_text

    card = {
        "name": "overlay_0001.wav",
        "source": "cheers_medium_raw.wav",
        "description": "quiet target under cutlery",
        "tags": ["hard-negative", "cutlery"],
        "augmentation_params": {
            "negative_source": "direct_spoon_hit.wav",
            "base_label": "cheers_medium",
            "overlay_label": "cutlery",
            "snr_db": 12.0,
            "semitones": -1.5,
            "transient_window_ms": 100,
        },
    }

    text = _augmentation_search_text(card)

    assert "overlay_0001.wav" in text
    assert "cheers_medium_raw.wav" in text
    assert "direct_spoon_hit.wav" in text
    assert "quiet target under cutlery" in text
    assert "hard-negative" in text
    assert "cheers_medium" in text
    assert "cutlery" in text
    assert "12.0" in text
    assert "-1.5" in text


# ---------------------------------------------------------------------------
# Phase 1 backend hardening: the silent-failure paths now surface real errors
# and the per-recipe `recipe_ids` selection is honored at /api/augment/start.
# ---------------------------------------------------------------------------


def _start_handler(cfg, project_id, layout, port):
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    handler = make_handler(
        cfg=cfg, project_id=project_id,
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    return httpd


def _seed_recipe(layout, **overrides) -> dict:
    """Persist a single augmentation recipe to dataset_selection.json."""
    from sed.data.compose import write_dataset_selection
    recipe = {
        "id": "recipe-1",
        "recipe_id": "recipe-1",
        "base_filters": [{"labels": ["glass"]}],
        "overlay_filters": [{"labels": ["noise"]}],
    }
    recipe.update(overrides)
    write_dataset_selection(layout, {
        "ui_managed": True,
        "augmentation_recipes": [recipe],
    })
    return recipe


def test_augment_start_returns_error_when_disabled(tmp_path, monkeypatch):
    """If cfg.augmentation.enabled is false, the job fails with a clear error
    instead of silently 'finishing' with zero generated clips."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")

    from sed.config import Config
    layout = ProjectLayout("p")
    layout.create()
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]
    cfg.augmentation.enabled = False  # the disabled path

    _seed_recipe(layout)

    port = _free_port()
    httpd = _start_handler(cfg, "p", layout, port)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augment/start",
            data=json.dumps({"recipe_ids": ["recipe-1"]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            started = json.loads(r.read())
        job_id = started["job"]["job_id"]
        final = {}
        for _ in range(40):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/augment/status?job_id={job_id}",
                timeout=2,
            ) as r:
                final = json.loads(r.read())
            if final["job"]["state"] in {"finished", "error", "canceled"}:
                break
            time.sleep(0.05)
        assert final["job"]["state"] == "error"
        err = (final["job"].get("error") or "").lower()
        # Expect a recognizable reason — either "disabled" or the specific
        # "augmentation.enabled is false" message we raise from augment.py.
        assert "augmentation" in err and ("false" in err or "disabled" in err)
    finally:
        httpd.shutdown()


def test_augment_start_with_no_recipes_selected_returns_400(tmp_path, monkeypatch):
    """An empty recipe_ids list (and no other selection) yields a 400 with a
    clear message, surfaced to the UI via the existing flash."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")

    from sed.config import Config
    layout = ProjectLayout("p")
    layout.create()
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    # selected=False so the fallback-by-selected path also returns 400 instead
    # of silently picking up a recipe the user didn't tick.
    _seed_recipe(layout, selected=False)

    port = _free_port()
    httpd = _start_handler(cfg, "p", layout, port)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augment/start",
            data=json.dumps({"recipe_ids": []}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read())
            assert "selected" in body["error"].lower()
        else:
            raise AssertionError("expected HTTP 400 for empty recipe_ids")
    finally:
        httpd.shutdown()


def test_augmentation_rule_update_replaces_recipe(tmp_path, monkeypatch):
    """The new `update` action edits in-place by recipe_id, preserving the
    fields the client didn't re-send (e.g. base/overlay filters)."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")

    from sed.config import Config
    layout = ProjectLayout("p")
    layout.create()
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    port = _free_port()
    httpd = _start_handler(cfg, "p", layout, port)
    try:
        add_body = {
            "action": "add",
            "recipe": {
                "base_filters": [{"labels": ["glass"], "count": 3}],
                "overlay_filters": [{"labels": ["noise"], "count": 4}],
                "total_samples": 17,
            },
        }
        with urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps(add_body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        ), timeout=2) as r:
            added = json.loads(r.read())
        rid = added["selection"]["augmentation_recipes"][0]["recipe_id"]

        update_body = {
            "action": "update",
            "recipe": {
                "recipe_id": rid,
                "prefix": "manual-pass",
                "replace_existing": True,
                "frozen_test": True,
                "selected": False,
            },
        }
        with urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps(update_body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        ), timeout=2) as r:
            updated = json.loads(r.read())
        row = updated["selection"]["augmentation_recipes"][0]
        assert row["prefix"] == "manual-pass"
        assert row["replace_existing"] is True
        assert row["frozen_test"] is True
        assert row["selected"] is False
        assert row["total_samples"] == 17
        # Base/overlay filters survived even though the update didn't resend them
        assert row["base_filters"][0]["labels"] == ["glass"]
        assert row["overlay_filters"][0]["labels"] == ["noise"]
    finally:
        httpd.shutdown()


def test_augmentation_rule_delete_generated_action_only_removes_files(tmp_path, monkeypatch):
    """`delete_generated` removes the on-disk clips for a recipe but keeps
    the recipe row itself — that's the Existing-Augmentations × button."""
    from sed import config as cfg_mod, data as data_pkg  # noqa: F401
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    from sed.config import LIBRARY_DIR  # noqa: F401
    from sed.config import Config

    import sed.data.library as lib_mod
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")

    from sed.data import augment as augment_mod
    seen: list = []
    real = augment_mod.delete_generated_for_recipes

    def spy(project_id, recipe_ids, *, frozen_test=None):
        seen.append((project_id, list(recipe_ids), frozen_test))
        return real(project_id, recipe_ids, frozen_test=frozen_test)

    monkeypatch.setattr(augment_mod, "delete_generated_for_recipes", spy)

    layout = ProjectLayout("p")
    layout.create()
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]

    # Seed a derived clip stamped with the recipe id so spy() has work to do.
    derived = tmp_path / "data_library" / "derived" / "p" / "trainval" / "glass" / "del.wav"
    derived.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(derived), np.zeros(32000, dtype=np.float32), 32000)
    derived.with_suffix(".json").write_text(json.dumps({
        "recipe_id": "recipe-del", "label": "glass", "class_name": "glass",
        "project_id": "p", "category": "derived", "schema_version": 2,
    }), encoding="utf-8")

    _seed_recipe(layout, id="recipe-del", recipe_id="recipe-del")

    port = _free_port()
    httpd = _start_handler(cfg, "p", layout, port)
    try:
        body = {"action": "delete_generated", "rule_id": "recipe-del"}
        with urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/api/augmentation-rule",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        ), timeout=2) as r:
            payload = json.loads(r.read())
        assert payload["ok"] is True
        assert payload["deleted_generated"] >= 1
        # Recipe row is still there
        rows = payload["selection"]["augmentation_recipes"]
        assert any((r.get("recipe_id") == "recipe-del") for r in rows)
        # delete helper was called with the matching id
        assert seen and seen[0][1] == ["recipe-del"]
        # The on-disk files are actually gone
        assert not derived.exists()
        assert not derived.with_suffix(".json").exists()
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Rescan scopes, uncapped panel totals, on-demand clip paging, async compose
# ---------------------------------------------------------------------------


def _setup_dataset_project(tmp_path, monkeypatch, n_glass=5):
    """Project 'p' with n_glass glass clips assigned to the dataset tab."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.config import Config
    from sed.data.compose import write_dataset_selection
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    lib = Library()
    for idx in range(n_glass):
        src = tmp_path / f"glass_{idx}.wav"
        sf.write(str(src), np.zeros(32000, dtype=np.float32), 32000)
        lib.import_clip(src, label="glass", project_id="p")
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "glass": [{
                "labels": ["glass"], "project_ids": ["p"],
                "source_bucket": "raw", "split": "train", "count": n_glass,
            }],
        },
    })
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]
    return cfg, layout


def _serve(cfg, layout, max_thumbs_per_class=0):
    from http.server import ThreadingHTTPServer
    from sed.data.inspect_server import make_handler

    handler = make_handler(
        cfg=cfg, project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout, max_thumbs_per_class=max_thumbs_per_class,
    )
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    return httpd, port


def _api_get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return json.loads(r.read())


def test_clink_feature_fit_reports_advisory_compute_budget_warning(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=0)
    httpd, port = _serve(cfg, layout)
    query = urllib.parse.urlencode({
        "backbone": "clink-cnn-v2",
        "pretrained": "0",
        "task": "clip",
        "sample_rate": "44100",
        "feature_type": "mel",
        "frontend": "efficientat",
        "n_fft": "1024",
        "win_length": "882",
        "hop_length": "220",
        "n_mels": "256",
        "fmin": "80",
        "fmax": "20000",
        "window_seconds": "0.8",
    })
    try:
        data = _api_get(port, "/api/feature-model-fit?" + query)
        fit = data["fit"]
        assert fit["gmac_per_clip"] > fit["gmac_budget_per_clip"] == 0.25
        warning = next(v for v in fit["verdicts"] if v.get("kind") == "compute_budget")
        assert warning["level"] == "warn"
        assert "Training remains allowed" in warning["text"]
    finally:
        httpd.shutdown()


def test_panel_totals_are_uncapped(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=5)
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        totals = _api_get(port, "/api/panel-totals")
        assert totals["ok"] is True
        # True count (5), independent of the display card cap of 2.
        assert totals["panel_totals"]["dataset"]["glass"] == 5

        # review-data caps the rendered cards at 2 but still reports the real total.
        review = _review_data_via_job(port)
        assert len(review["panels"]["glass"]) == 2
        assert review["panel_totals"]["dataset"]["glass"] == 5
    finally:
        httpd.shutdown()


def test_dataset_integrity_scan_flags_corrupt_audio_and_returns_card_state(tmp_path, monkeypatch):
    # The real serve entrypoint imports the renderer before opening its socket;
    # do the same here so the assertion covers scan work rather than cold torch
    # import time in the background worker.
    from sed.data.inspect import scan_dataset_audio_integrity  # noqa: F401

    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=1)
    # Keep the Library metadata/selection intact, then simulate the damaged
    # on-disk WAV that causes the training preflight warning.
    wav = next((tmp_path / "data_library").rglob("*.wav"))
    wav.write_bytes(b"this is not a WAV file")
    httpd, port = _serve(cfg, layout)
    try:
        started = _post_json(f"http://127.0.0.1:{port}/api/dataset/integrity/start", {})
        assert started["ok"] is True
        job_id = started["job"]["job_id"]
        final = _wait_for_job(
            f"http://127.0.0.1:{port}/api/dataset/integrity/status?job_id={job_id}"
        )["job"]
        assert final["state"] == "finished"
        assert final["total"] == 1
        assert final["corrupt_count"] == 1
        assert final["corrupt"][0]["path"] == str(wav.resolve())
        assert "Error" in final["corrupt"][0]["error"]
        assert final["corrupt_panels"]["glass"][0]["audio_integrity"] == "corrupt"

        cached = json.loads((layout.dataset_inspect_dir / "audio_integrity_cache.json").read_text())
        assert cached["files"][str(wav.resolve())]["error"]

        review = _review_data_via_job(port, scopes=["dataset"])
        card = review["panels"]["glass"][0]
        assert card["audio_integrity"] == "corrupt"
        assert card["audio_error"]
    finally:
        httpd.shutdown()


def test_dataset_integrity_controls_render_on_dataset_tab(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p",
        class_summary={},
        panels={"glass": [{
            "path": "/tmp/damaged.wav", "name": "damaged.wav", "meta_html": "",
            "audio_integrity": "corrupt", "audio_error": "Format not recognised",
        }]},
        interactive=True,
    )
    assert 'id="dataset-integrity-filter"' in html
    assert 'id="dataset-integrity-scan"' in html
    assert 'id="dataset-delete-corrupt"' in html
    assert "Scan audio integrity" in html
    assert "corrupt-preview" in html
    assert "corrupt-audio" in html
    assert "/api/dataset/integrity/start" in html


def test_train_config_filters_fine_tune_policy_for_scratch_models(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=2)
    # Avoid building a real backbone just to count stages (matches the other
    # train-config tests; a cold build can outlast the client timeout).
    from sed.train import backbones as bb_mod
    monkeypatch.setattr(bb_mod, "probe_spec", lambda *a, **k: {"n_stages": 5, "batchnorm_layers": 3})
    cfg.model.pretrained = False
    cfg.model.initialization = "random"
    cfg.training.fine_tune_policy = "last_two"
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        data = _api_get(port, "/api/train/config")
        assert data["ok"] is True
        assert data["training"]["accumulate_grad_batches"] == 1
        assert data["batching"] == {
            "physical_batch_size": cfg.training.batch_size,
            "accumulate_grad_batches": 1,
            "world_size": 1,
            "effective_batch_size": cfg.training.batch_size,
        }
        assert data["training"]["class_balance"] == "off"
        assert data["training"]["head_dropout_auto"] is True
        assert data["training"]["head_dropout_configured"] == pytest.approx(0.2)
        assert data["training"]["head_dropout_effective"] == pytest.approx(0.2)
        assert data["model"]["dropout"]["not_additive"] is True
        classifier = data["model"]["dropout"]["rows"][0]
        assert classifier["location"] == "classifier_head"
        assert classifier["auto"] is True
        assert classifier["value"] == pytest.approx(0.2)
        assert "sampling_strategy" not in data["training"]
        assert "negative_weight" not in data["training"]
        assert data["training"]["fine_tune_policy"] == "full"
        assert [row["id"] for row in data["fine_tune_options"]["policies"]] == ["full"]
        assert "Random/scratch" in data["fine_tune_options"]["message"]
        # Preset gating for a scratch model: transfer presets are inapplicable,
        # the scratch preset applies.
        gates = {p["id"]: p["applicable"] for p in data["presets"]}
        assert gates["scratch_full"] is True
        assert gates["venue_finetune"] is False
        assert gates["domain_pretrain"] is False
    finally:
        httpd.shutdown()


def test_train_config_metric_descriptor_tracks_loss_and_frame_task(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=2)
    cfg.training.loss = "cross_entropy"
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        data = _api_get(port, "/api/train/config")
        assert data["metric_descriptor"]["effective_loss"] == "cross_entropy"
        assert data["metric_descriptor"]["probability_mode"] == "softmax"
        assert "argmax" in data["metric_descriptor"]["val_accuracy_description"]
    finally:
        httpd.shutdown()

    cfg.model.task = "frame"
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        data = _api_get(port, "/api/train/config")
        assert data["metric_descriptor"]["effective_loss"] == "frame_bce_logits"
        assert data["metric_descriptor"]["probability_mode"] == "sigmoid"
    finally:
        httpd.shutdown()


def test_train_config_exposes_split_diagnostics(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=6)
    from sed.data.compose import compose_dataset
    from sed.train import backbones as bb_mod

    compose_dataset(cfg, "p")
    monkeypatch.setattr(bb_mod, "probe_spec", lambda *a, **k: {"n_stages": 5})
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        data = _api_get(port, "/api/train/config")
        diag = data["dataset_config"]["split_diagnostics"]
        assert diag["strategy"] == "bucket-session-grouped"
        assert "glass" in diag["by_class"]
        assert "sounds:glass" in diag["by_class"]["glass"]["buckets"]
    finally:
        httpd.shutdown()


def test_train_config_exposes_live_validation_counts_for_every_class(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=0)
    from sed.train import backbones as bb_mod

    monkeypatch.setattr(bb_mod, "probe_spec", lambda *a, **k: {"n_stages": 5})
    cfg.model.classes = ["_negative", "glass", "attention"]
    cfg.dataset.val_fraction = 0.5
    header = (
        "path,class_name,session_id,sample_rate,duration_s,is_target,origin,"
        "origin_class,source_stem,split_bucket,split_group,frozen_source"
    )
    rows = [
        "/tmp/neg-a.wav,_negative,neg-a,16000,1,0,pretraining,noise-a,,pretraining:noise-a,neg-a,",
        "/tmp/neg-b.wav,_negative,neg-b,16000,1,0,pretraining,noise-b,,pretraining:noise-b,neg-b,",
        "/tmp/glass-a1.wav,glass,glass-a,16000,1,1,pretraining,glass,,pretraining:glass,glass-a,",
        "/tmp/glass-a2.wav,glass,glass-a,16000,1,1,pretraining,glass,,pretraining:glass,glass-a,",
        "/tmp/glass-b1.wav,glass,glass-b,16000,1,1,pretraining,glass,,pretraining:glass,glass-b,",
        "/tmp/glass-b2.wav,glass,glass-b,16000,1,1,pretraining,glass,,pretraining:glass,glass-b,",
    ]
    (layout.dataset_dir / "train.csv").write_text(
        "\n".join([header, *rows]) + "\n", encoding="utf-8",
    )

    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        data = _api_get(port, "/api/train/config")
        dc = data["dataset_config"]
        assert data["counts"]["train"] == 3
        assert data["counts"]["val"] == 3
        assert dc["training_class_counts"] == {
            "_negative": 1, "glass": 2, "attention": 0,
        }
        assert dc["validation_class_counts"] == {
            "_negative": 1, "glass": 2, "attention": 0,
        }
        assert dc["training_pool_source_counts"] == {"dataset": 6, "augmented": 0}
        warnings = dc["split_diagnostics"]["by_class"]["_negative"]["warnings"]
        assert any("no source bucket" in warning for warning in warnings)
    finally:
        httpd.shutdown()


def test_train_start_rejects_invalid_ui_overrides_before_writing(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    from sed.config import Config, load_config, project_config_template

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p"), encoding="utf-8")
    cfg = Config()
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        status, payload = _post_json_status(
            f"http://127.0.0.1:{port}/api/train/start",
            {"training": {"lr_head": 0, "lr_backbone": 0.0001}},
        )
        assert status == 400
        assert payload["ok"] is False
        assert "invalid training settings" in payload["error"]
        assert load_config("p").training.lr_head == pytest.approx(1e-4)
        assert "lr_head = 0" not in layout.config_toml.read_text(encoding="utf-8")

        status, payload = _post_json_status(
            f"http://127.0.0.1:{port}/api/train/start",
            {"training": {"accumulate_grad_batches": 0}},
        )
        assert status == 400
        assert payload["ok"] is False
        assert "accumulate_grad_batches must be positive" in payload["error"]
        assert load_config("p").training.accumulate_grad_batches == 1
        assert "accumulate_grad_batches = 0" not in layout.config_toml.read_text(
            encoding="utf-8",
        )
    finally:
        httpd.shutdown()


def test_train_stage_policy_clamps_in_config_preview_and_start(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    from sed.config import Config, load_config, project_config_template
    from sed.data import inspect_server as inspect_server_mod

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    train_rows = [
        f"/tmp/clip_{i}.wav,glass,s{i},32000,1.0,1,sounds\n"
        for i in range(400)
    ]
    (layout.dataset_dir / "train.csv").write_text(
        "path,class_name,session_id,sample_rate,duration_s,is_target,origin\n"
        + "".join(train_rows),
        encoding="utf-8",
    )
    (layout.dataset_dir / "test.csv").write_text(
        "path,class_name,session_id,sample_rate,duration_s,is_target,origin\n",
        encoding="utf-8",
    )
    (layout.dataset_dir / "dataset_snapshot.json").write_text(
        json.dumps({"by_class": {"glass": 400, "_negative": 1600}}),
        encoding="utf-8",
    )

    class FakeStdout:
        def read(self, _n):
            return b""

    class FakeProc:
        stdout = FakeStdout()

        def wait(self):
            return 0

        def terminate(self):
            return None

    monkeypatch.setattr(inspect_server_mod, "_train_stage_count", lambda _cfg: 2)
    monkeypatch.setattr(inspect_server_mod.subprocess, "Popen", lambda *a, **k: FakeProc())

    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]
    cfg.training.fine_tune_policy = "last_two"
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    base = f"http://127.0.0.1:{port}"
    try:
        d = _get_json(f"{base}/api/train/config", timeout=10.0)
        assert d["fine_tune_options"]["n_stages"] == 2
        assert [p["id"] for p in d["fine_tune_options"]["policies"]] == ["head", "last_stage", "full"]
        assert d["training"]["fine_tune_policy"] == "full"
        assert d["recommended"]["values"]["fine_tune_policy"] == "full"
        venue = next(p for p in d["presets"] if p["id"] == "venue_finetune")
        auto = next(p for p in d["presets"] if p["id"] == "auto")
        assert venue["values"]["fine_tune_policy"] == "full"
        assert auto["values"]["fine_tune_policy"] == "full"

        before_preview = layout.config_toml.read_text(encoding="utf-8")
        preview = _post_json(
            f"{base}/api/train/config-preview",
            {"model": {"pretrained": False, "initialization": "random"}},
            timeout=10.0,
        )
        assert preview["fine_tune_options"]["has_transfer_weights"] is False
        assert [p["id"] for p in preview["fine_tune_options"]["policies"]] == ["full"]
        assert preview["training"]["fine_tune_policy"] == "full"
        assert layout.config_toml.read_text(encoding="utf-8") == before_preview

        payload = _post_json(
            f"{base}/api/train/start",
            {
                "model": {"pretrained": True, "initialization": "upstream"},
                "training": {"fine_tune_policy": "last_two", "epochs": 3},
            },
            timeout=10.0,
        )
        assert payload["ok"] is True
        assert load_config("p").training.fine_tune_policy == "full"
    finally:
        httpd.shutdown()


def test_train_start_persists_valid_ui_overrides_and_reloads_config(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    from sed.config import Config, load_config, project_config_template
    from sed.data import inspect_server as inspect_server_mod

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    (layout.dataset_dir / "train.csv").write_text(
        "path,class_name,session_id,sample_rate,duration_s,is_target,origin\n"
        "/tmp/clip.wav,glass,s0,32000,1.0,1,sounds\n",
        encoding="utf-8",
    )
    (layout.dataset_dir / "test.csv").write_text(
        "path,class_name,session_id,sample_rate,duration_s,is_target,origin\n",
        encoding="utf-8",
    )

    class FakeStdout:
        def read(self, _n):
            return b""

    class FakeProc:
        stdout = FakeStdout()

        def wait(self):
            return 0

        def terminate(self):
            return None

    monkeypatch.setattr(inspect_server_mod.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(inspect_server_mod, "_train_stage_count", lambda _cfg: 5)

    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]
    # A previously selected pretrained checkpoint variant: flipping to random
    # must not keep implying the mels_256 checkpoint (same architecture as mn10).
    cfg.model.backbone = "mn10_as_mels_256"
    cfg.model.weight_id = "efficientat:mn10_as_mels_256"
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        # Removed browser knobs are not accepted by config-preview and cannot
        # silently change the new canonical policy.
        stale_preview = _post_json(
            f"http://127.0.0.1:{port}/api/train/config-preview",
            {
                "training": {
                    "sampling_strategy": "class_group_sqrt",
                    "negative_weight": 9.0,
                },
            },
        )
        assert stale_preview["training"]["class_balance"] == "off"

        payload = _post_json(
            f"http://127.0.0.1:{port}/api/train/start",
            {
                "model": {"pretrained": False, "initialization": "random"},
                "training": {
                    "epochs": 7,
                    "batch_size": 12,
                    "accumulate_grad_batches": 3,
                    "lr_backbone": 0.0003,
                    "lr_head": 0.0003,
                    "loss": "categorical_crossentropy",
                    "patience": 0,
                    "fine_tune_policy": "last_two",
                    "class_balance": "sqrt",
                    # Removed Train-tab fields are ignored even if an older
                    # browser still submits them alongside the canonical mode.
                    "sampling_strategy": "class_group_sqrt",
                    "negative_weight": 9.0,
                    "auto_test_rules": ["fixed@0.87", "argmax"],
                    "calibrate_thresholds": True,
                    "calibrate_after": True,
                    "calibrate_per_1000": 2.5,
                    "calibrate_negsrc": "library",
                },
            },
        )
        assert payload["ok"] is True
        saved = load_config("p")
        assert saved.model.pretrained is False
        assert saved.model.initialization == "random"
        # Scratch runs record architecture only: no checkpoint id, and the
        # variant name reduced to its base width.
        assert saved.model.weight_id == ""
        assert saved.model.backbone == "mn10_as"
        assert saved.training.epochs == 7
        assert saved.training.batch_size == 12
        assert saved.training.accumulate_grad_batches == 3
        assert saved.training.lr_backbone == pytest.approx(0.0003)
        assert saved.training.lr_head == pytest.approx(0.0003)
        assert saved.training.loss == "categorical_crossentropy"
        assert saved.training.patience == 0
        assert saved.training.fine_tune_policy == "full"
        assert saved.training.class_balance == "sqrt"
        # Frozen-test rules persist as rule ids, so `sed train` scores exactly
        # what the Train tab asked for.
        assert saved.training.auto_test_rules == ["fixed@0.87", "argmax"]
        assert saved.training.resolved_auto_test_rules() == ["fixed@0.87", "argmax"]
        assert saved.training.calibrate_thresholds is True
        # FP/h calibration is config now, not a session-only request.
        assert saved.training.calibrate_after is True
        assert saved.training.calibrate_per_1000 == pytest.approx(2.5)
        assert saved.training.calibrate_negsrc == "library"

        refreshed = _api_get(port, "/api/train/config")
        assert refreshed["training"]["epochs"] == 7
        assert refreshed["batching"] == {
            "physical_batch_size": 12,
            "accumulate_grad_batches": 3,
            "world_size": 1,
            "effective_batch_size": 36,
        }
        assert refreshed["training"]["lr_head"] == pytest.approx(0.0003)
        assert refreshed["training"]["fine_tune_policy"] == "full"
        assert refreshed["training"]["class_balance"] == "sqrt"
        assert "sampling_strategy" not in refreshed["training"]
        assert "negative_weight" not in refreshed["training"]
        assert refreshed["training"]["auto_test_rules"] == ["fixed@0.87", "argmax"]
        assert refreshed["training"]["calibrate_thresholds"] is True
        assert refreshed["training"]["calibrate_after"] is True
        assert refreshed["training"]["calibrate_per_1000"] == pytest.approx(2.5)
        assert refreshed["training"]["calibrate_negsrc"] == "library"
    finally:
        httpd.shutdown()


def test_train_start_refits_stale_standardized_linear_stats_after_seed_change(tmp_path, monkeypatch):
    """Start is the Train-tab commit point, including seed-dependent moments."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    import csv

    import numpy as np
    import soundfile as sf
    from sed.config import (
        Config,
        load_config,
        project_config_template,
        update_config_toml_values,
    )
    from sed.data import inspect_server as inspect_server_mod
    from sed.util import sha256_file

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for class_name in ("_negative", "glass"):
        for index in range(3):
            wav = audio_dir / f"{class_name}-{index}.wav"
            sf.write(wav, np.full(12_800, 0.002 * (index + 1), dtype=np.float32), 32_000)
            rows.append({
                "path": str(wav), "class_name": class_name,
                "source_split_group": f"{class_name}-{index}", "origin": "recorded",
            })
    train_csv = layout.dataset_dir / "train.csv"
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (layout.dataset_dir / "test.csv").write_text("path,class_name\n", encoding="utf-8")

    initial = load_config("p")
    stale = initial.features.model_copy(update={
        "feature_type": "linear", "frontend": "efficientat", "n_fft": 1024,
        "win_length": 800, "hop_length": 320, "fmin": 0, "fmax": 15_000,
        "window_seconds": 0.4, "log_power": True, "stft_engine": "librosa",
        "pcen": False, "db_norm": False, "standardize": True,
        "standardize_mean": -3.0, "standardize_std": 1.5,
        "standardize_stats_resolved": True,
        "standardize_stats_dataset_sha256": sha256_file(train_csv),
        "standardize_stats_split_seed": 22,
        "standardize_stats_val_fraction": initial.dataset.val_fraction,
        "standardize_stats_clips": 6,
    })
    update_config_toml_values(layout.config_toml, {
        "model": {"pretrained": False, "initialization": "random", "sample_rate": 32_000},
        "features": stale.model_dump(),
        "training": {"seed": 22},
    })

    class FakeStdout:
        def read(self, _n):
            return b""

    class FakeProc:
        stdout = FakeStdout()

        def wait(self):
            return 0

        def terminate(self):
            return None

    monkeypatch.setattr(inspect_server_mod.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(inspect_server_mod, "_train_stage_count", lambda _cfg: 5)
    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        launched = _post_json(
            f"http://127.0.0.1:{port}/api/train/start",
            {"training": {"seed": 42, "epochs": 1}},
            timeout=30.0,
        )
        assert launched["ok"] is True
        assert launched["normalization_refit"]["split_seed"] == 42
        saved = load_config("p")
        assert saved.training.seed == 42
        assert saved.features.standardize_stats_resolved is True
        assert saved.features.standardize_stats_split_seed == 42
        assert saved.features.standardize_stats_val_fraction == pytest.approx(saved.dataset.val_fraction)
        assert saved.features.standardize_stats_dataset_sha256 == sha256_file(train_csv)
        assert saved.features.standardize_stats_clips > 0
    finally:
        httpd.shutdown()


def test_replicate_seeds_plans_a_sweep_from_the_configured_seed():
    """One run is one draw. The sweep exists so a gap between two runs has a
    spread to be measured against, and replicate 1 has to stay identical to the
    single run the user would otherwise have got."""
    from sed.data.inspect_server import MAX_REPLICATE_SEEDS, _replicate_seeds

    assert _replicate_seeds({}, 42) == [42]
    assert _replicate_seeds({"replicates": 4}, 42) == [42, 43, 44, 45]
    # An explicit list wins, and repeats collapse: two runs at the same seed are
    # re-runs, not independent samples.
    assert _replicate_seeds({"seeds": [7, 9, 7]}, 42) == [7, 9]
    with pytest.raises(ValueError):
        _replicate_seeds({"replicates": MAX_REPLICATE_SEEDS + 1}, 42)
    with pytest.raises(ValueError):
        _replicate_seeds({"seeds": ["not-a-seed"]}, 42)


def test_train_start_trains_one_run_per_replicate_seed(tmp_path, monkeypatch):
    """A replicate sweep is N trainings of one configuration, seed being the only
    thing that changes — and the project is left on the seed it started on, so a
    sweep does not silently become the new default."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    import time as _time

    from sed.config import Config, load_config, project_config_template
    from sed.data import inspect_server as inspect_server_mod

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p"), encoding="utf-8")
    (layout.dataset_dir / "train.csv").write_text(
        "path,class_name,origin\n/tmp/a.wav,glass,sounds\n", encoding="utf-8",
    )

    # Every launch records the seed config.toml holds at that moment: that is
    # what the child `sed train` would read.
    seeds_seen: list[int] = []

    class FakeStdout:
        def read(self, _n):
            return b""

    class FakeProc:
        stdout = FakeStdout()

        def wait(self):
            return 0

        def terminate(self):
            return None

    def fake_popen(*_a, **_k):
        seeds_seen.append(int(load_config("p").training.seed))
        return FakeProc()

    monkeypatch.setattr(inspect_server_mod.subprocess, "Popen", fake_popen)
    cfg = Config()
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        launched = _post_json(
            f"http://127.0.0.1:{port}/api/train/start",
            {"training": {"seed": 42, "epochs": 1}, "replicates": 3},
            timeout=30.0,
        )
        assert launched["ok"] is True

        deadline = _time.time() + 30.0
        job = {}
        while _time.time() < deadline:
            job = _api_get(port, "/api/train/status")["job"]
            if job.get("state") in ("done", "saved", "error", "cancelled"):
                break
            _time.sleep(0.05)

        assert job["state"] == "done", job.get("error")
        assert seeds_seen == [42, 43, 44]
        assert job["seed_total"] == 3
        assert [r["seed"] for r in job["replicates"]] == [42, 43, 44]
        assert all(r["state"] == "done" for r in job["replicates"])
        # Restored: the sweep borrowed the seed, it did not take it.
        assert load_config("p").training.seed == 42
    finally:
        httpd.shutdown()


def test_train_start_without_replicates_launches_exactly_once(tmp_path, monkeypatch):
    """The single-run path has to stay what it was: one child, no config rewrite
    between runs, and a status shape the UI can read unconditionally."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    import time as _time

    from sed.config import Config, load_config, project_config_template
    from sed.data import inspect_server as inspect_server_mod

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p"), encoding="utf-8")
    (layout.dataset_dir / "train.csv").write_text(
        "path,class_name,origin\n/tmp/a.wav,glass,sounds\n", encoding="utf-8",
    )
    launches: list[int] = []

    class FakeProc:
        stdout = type("S", (), {"read": lambda _s, _n: b""})()

        def wait(self):
            return 0

        def terminate(self):
            return None

    monkeypatch.setattr(
        inspect_server_mod.subprocess, "Popen",
        lambda *a, **k: (launches.append(1), FakeProc())[1],
    )
    cfg = Config()
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        assert _post_json(
            f"http://127.0.0.1:{port}/api/train/start",
            {"training": {"seed": 7, "epochs": 1}},
            timeout=30.0,
        )["ok"] is True
        deadline = _time.time() + 30.0
        job = {}
        while _time.time() < deadline:
            job = _api_get(port, "/api/train/status")["job"]
            if job.get("state") in ("done", "saved", "error", "cancelled"):
                break
            _time.sleep(0.05)
        assert job["state"] == "done", job.get("error")
        assert launches == [1]
        assert job["seed_total"] == 1
        assert job["seeds"] == [7]
        assert load_config("p").training.seed == 7
    finally:
        httpd.shutdown()


def test_train_delete_runs_endpoint_removes_bundles_and_clears_active(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    from sed.config import Config, load_config

    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout, active="run-a")
    for model_id in ("run-a", "run-b", "run-keep"):
        _seed_inference_bundle(layout, model_id, exported=True)
    nested = layout.model_dir("run-b") / "reports" / "nested" / "artifact.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("remove me", encoding="utf-8")

    cfg = Config()
    cfg.model.classes = ["_negative", "cheers_loud"]
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        started = _api_get(port, "/api/train/runs")
        assert {row["model_id"] for row in started["runs"]} == {"run-a", "run-b", "run-keep"}

        deleted = _post_json(
            f"http://127.0.0.1:{port}/api/train/delete",
            {"model_ids": ["run-a", "run-b"]},
        )
        assert deleted["ok"] is True
        assert deleted["deleted"] == ["run-a", "run-b"]
        assert deleted["deleted_count"] == 2
        assert not layout.model_dir("run-a").exists()
        assert not layout.model_dir("run-b").exists()
        assert layout.model_dir("run-keep").exists()
        assert load_config("p").deploy.active_model == ""

        remaining = _api_get(port, "/api/train/runs")
        assert [row["model_id"] for row in remaining["runs"]] == ["run-keep"]
    finally:
        httpd.shutdown()


def test_live_train_metrics_parser_keeps_epoch_values(tmp_path):
    from sed.data.inspect_server import _parse_train_metrics

    metrics_dir = tmp_path / "model" / "reports" / "training"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.csv").write_text(
        "epoch,step,train/loss_epoch,val/loss,val/accuracy,val/f1_macro_targets,val/ap_macro_targets,lr-AdamW/pg1\n"
        "0,4,0.91,0.82,0.63,0.51,0.44,0.001\n"
        "1,8,0.72,0.69,0.71,0.58,0.49,0.0008\n",
        encoding="utf-8",
    )

    rows = _parse_train_metrics(tmp_path / "model")
    assert rows[-1]["epoch"] == 1
    assert rows[-1]["train_loss"] == 0.72
    assert rows[-1]["val_loss"] == 0.69
    assert rows[-1]["val_accuracy"] == 0.71
    assert rows[-1]["val_f1"] == 0.58
    assert rows[-1]["val_ap"] == 0.49
    assert rows[-1]["lr"] == 0.0008
    assert rows[-1]["values"]["lr-AdamW/pg1"] == 0.0008


def test_live_train_metrics_parser_keeps_byol_score(tmp_path):
    from sed.data.inspect_server import _parse_train_metrics

    metrics_dir = tmp_path / "model" / "reports" / "training"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.csv").write_text(
        "epoch,step,train/loss_epoch,val/loss,val/byol_score,lr-AdamW/pg1\n"
        "0,4,0.91,0.82,-0.82,0.001\n",
        encoding="utf-8",
    )

    rows = _parse_train_metrics(tmp_path / "model")
    assert rows[-1]["val_loss"] == 0.82
    assert rows[-1]["val_byol_score"] == -0.82
    assert "val_accuracy" not in rows[-1]


def test_live_train_metrics_parser_maps_legacy_byol_accuracy_to_score(tmp_path):
    from sed.data.inspect_server import _parse_train_metrics

    metrics_dir = tmp_path / "model" / "reports" / "training"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.csv").write_text(
        "epoch,step,train/loss_epoch,val/loss,val/accuracy,lr-AdamW/pg1\n"
        "0,4,0.91,0.82,-0.82,0.001\n",
        encoding="utf-8",
    )

    rows = _parse_train_metrics(tmp_path / "model")
    assert rows[-1]["val_byol_score"] == -0.82
    assert "val_accuracy" not in rows[-1]


def test_dataset_bucket_filter_lists_and_pages_every_assigned_source_bucket(tmp_path, monkeypatch):
    """Subbucket and sub-group narrowing both reach past the display cap.

    Which subbuckets are previewed is the Summary panel's checkboxes, and which
    part of them is the per-class ``subgroup:`` dropdown; the old per-class
    ``view:`` select duplicated the first of those and is gone.  Both resolve on
    the server, because a selection can lie entirely beyond the capped cards.
    """
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.config import Config
    from sed.data.compose import write_dataset_selection
    from sed.data.inspect import _render_html, review_data_payload
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    lib = Library()
    buckets = ("bar_cheers", "concert_cheers", "stadium_cheers")
    for bucket in buckets:
        for idx in range(3):
            source = tmp_path / f"{bucket}_{idx}.wav"
            sf.write(str(source), np.zeros(32000, dtype=np.float32), 32000)
            clip = lib.import_clip(source, label=bucket, project_id="p")
            # Two takes per bucket, the shape a recorder produces: one folder,
            # several recording sessions inside it.
            sidecar = (Path(lib.root) / clip.relpath).with_suffix(".json")
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            meta["session_id"] = f"{bucket}_take{idx // 2}"
            sidecar.write_text(json.dumps(meta), encoding="utf-8")

    cfg = Config()
    cfg.model.classes = ["_negative", "cheers"]
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "cheers": [{
                "labels": list(buckets), "project_ids": ["p"],
                "source_bucket": "raw", "split": "train",
            }],
        },
    })

    # The card payload is capped, but the filter inventory represents every
    # assigned source bucket, its true count, and what is inside it.
    payload = review_data_payload(cfg, layout, max_per_class=2)
    assert len(payload["panels"]["cheers"]) == 2
    assert payload["dataset_origin_counts"]["cheers"] == {
        bucket: 3 for bucket in buckets
    }
    stadium = payload["panel_subgroups"]["dataset"]["cheers"]["stadium_cheers"]
    assert stadium["session"] == {"stadium_cheers_take0": 2, "stadium_cheers_take1": 1}
    # One folder and one date across the bucket subdivide nothing, so those axes
    # are left out rather than offered as a single pointless option.
    assert set(stadium) == {"session"}

    html = _render_html(
        project_id="p",
        class_summary={},
        panels=payload["panels"],
        interactive=True,
        dataset_origin_counts=payload["dataset_origin_counts"],
        panel_subgroups=payload["panel_subgroups"],
    )
    # One subbucket control, in the Summary panel; the redundant per-class
    # "view:" dropdown is gone and the sub-group picker took its place.
    assert "origin-filter" not in html
    assert "all subbuckets" not in html
    assert html.count('class="summary-subbucket-check"') >= len(buckets)
    for tab in ("dataset", "augmented", "frozen"):
        assert f'class="subgroup-filter" data-tab="{tab}"' in html or (
            f'data-tab="{tab}"' in html)
    assert '"panel_subgroups"' in html

    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        # One checked subbucket.
        page = _api_get(
            port,
            "/api/clips?tab=dataset&class=cheers&origins=stadium_cheers&offset=0&limit=2",
        )
        assert page["total"] == 3
        assert len(page["clips"]) == 2
        assert {clip["origin_class"] for clip in page["clips"]} == {"stadium_cheers"}

        # Several checked subbuckets are one request, not one per bucket.
        page = _api_get(
            port,
            "/api/clips?tab=dataset&class=cheers"
            "&origins=stadium_cheers%7Cbar_cheers&offset=0&limit=50",
        )
        assert page["total"] == 6
        assert {clip["origin_class"] for clip in page["clips"]} == {
            "stadium_cheers", "bar_cheers"}

        # A sub-group lists every clip in it, including past the card cap.
        page = _api_get(
            port,
            "/api/clips?tab=dataset&class=cheers&group_axis=session"
            "&group_key=stadium_cheers_take0&offset=0&limit=50",
        )
        assert page["total"] == 2
        assert {clip["origin_class"] for clip in page["clips"]} == {"stadium_cheers"}
        assert {clip["session_id"] for clip in page["clips"]} == {"stadium_cheers_take0"}

        # Unchecking every bucket shows nothing. An empty "origins" is a real
        # selection, not the absence of one — omitting the parameter is.
        assert _api_get(
            port, "/api/clips?tab=dataset&class=cheers&origins=&offset=0&limit=50",
        )["total"] == 0
        assert _api_get(
            port, "/api/clips?tab=dataset&class=cheers&offset=0&limit=50",
        )["total"] == 9

        # The legacy single-bucket spelling still resolves.
        legacy = _api_get(
            port,
            "/api/clips?tab=dataset&class=cheers&origin_class=bar_cheers&offset=0&limit=50",
        )
        assert legacy["total"] == 3

        refreshed = _review_data_via_job(port, scopes=["dataset"])
        assert refreshed["dataset_origin_counts"]["cheers"] == {
            bucket: 3 for bucket in buckets
        }
        # A scoped rescan carries only its own tab's sub-groups, so merging it
        # in the browser cannot blank another tab's dropdowns.
        assert set(refreshed["panel_subgroups"]) == {"dataset"}
    finally:
        httpd.shutdown()


def test_frozen_test_tab_narrows_to_one_recording_session(tmp_path, monkeypatch):
    """The Frozen test tab gets the same sub-group control as Dataset.

    Its clips come from the composed ``test.csv`` rather than the library scan,
    so the provenance the axes group by has to survive that round trip.
    """
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.config import Config
    from sed.data.compose import compose_dataset, write_dataset_selection
    from sed.data.inspect import clips_page, review_data_payload
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    lib = Library()
    for idx in range(6):
        source = tmp_path / f"held_{idx}.wav"
        sf.write(str(source), np.zeros(32000, dtype=np.float32), 32000)
        clip = lib.import_clip(source, label="heldout_glass", project_id="p")
        sidecar = (Path(lib.root) / clip.relpath).with_suffix(".json")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["session_id"] = f"take{idx // 3}"
        sidecar.write_text(json.dumps(meta), encoding="utf-8")

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "cheers_loud"]}})
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "cheers_loud": [{
                "labels": ["heldout_glass"], "project_ids": ["p"],
                "source_bucket": "raw", "split": "test",
            }],
        },
    })
    compose_dataset(cfg, "p")

    payload = review_data_payload(cfg, layout, max_per_class=2, scopes=["frozen"])
    axes = payload["panel_subgroups"]["frozen"]["cheers_loud"]["heldout_glass"]
    assert axes["session"] == {"take0": 3, "take1": 3}

    cards, total = clips_page(
        cfg, layout, "frozen", "cheers_loud", 0, 50,
        group_axis="session", group_key="take1", return_total=True)
    assert total == 3 and len(cards) == 3


def test_assigned_preview_cache_builds_only_selected_clips_and_keeps_variants(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=4)
    from sed.data.inspect import assigned_preview_paths
    from sed.data.library import Library

    # This Library asset is deliberately not assigned to a target class.
    extra_source = tmp_path / "unassigned.wav"
    sf.write(str(extra_source), np.zeros(32000, dtype=np.float32), 32000)
    Library().import_clip(extra_source, label="unassigned", project_id="p")
    assigned = assigned_preview_paths(cfg, layout)
    assert len(assigned) == 4
    assert all("unassigned" not in str(path) for path in assigned)

    httpd, port = _serve(cfg, layout)
    base = f"http://127.0.0.1:{port}"
    try:
        before = _get_json(f"{base}/api/preview-cache")
        assert before["ok"] is True
        assert before["variants"] == []

        started = _post_json(f"{base}/api/preview-cache/start", {}, timeout=10)
        assert started["ok"] is True
        job_id = started["job"]["job_id"]
        # A second cache build is rejected instead of stacking work.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post_json(f"{base}/api/preview-cache/start", {}, timeout=10)
        assert exc.value.code == 409

        final = _wait_for_job(f"{base}/api/preview-cache/status?job_id={job_id}", timeout=30)
        job = final["job"]
        assert job["state"] == "finished", job.get("error")
        assert job["total"] == job["completed"] == 4
        assert job["generated"] + job["skipped"] + job["migrated"] + job["failed"] == 4

        listing = _get_json(f"{base}/api/preview-cache")
        assert len(listing["variants"]) == 1
        variant = listing["variants"][0]
        assert variant["clips"] == 4
        assert variant["bytes"] > 0
        deleted = _post_json(f"{base}/api/preview-cache/delete", {"signature": variant["signature"]})
        assert deleted["ok"] is True
        assert _get_json(f"{base}/api/preview-cache")["variants"] == []
    finally:
        httpd.shutdown()


def test_clips_page_returns_cards_beyond_cap(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=5)
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        page = _api_get(port, "/api/clips?tab=dataset&class=glass&offset=2&limit=2")
        assert page["ok"] is True
        assert page["offset"] == 2
        assert len(page["clips"]) == 2
        # All five are reachable by paging past the cap.
        first = _api_get(port, "/api/clips?tab=dataset&class=glass&offset=0&limit=10")
        assert len(first["clips"]) == 5
        # Search scans the source inventory before the preview cap, so a named
        # clip that was not in the initial two cards is still discoverable.
        searched = _api_get(port, "/api/clips?tab=dataset&class=glass&offset=0&limit=1000&query=glass_4")
        assert searched["query"] == "glass_4"
        assert searched["total"] == 1
        assert len(searched["clips"]) == 1
        assert "glass" in searched["clips"][0]["name"]
        # An unknown tab is rejected.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _api_get(port, "/api/clips?tab=bogus&class=glass")
        assert exc.value.code == 400
    finally:
        httpd.shutdown()


def test_clip_search_includes_microphone_metadata_from_tile(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=2)
    wav = next((tmp_path / "data_library").rglob("*.wav"))
    sidecar = wav.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["microphone_label"] = "joby"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    httpd, port = _serve(cfg, layout)
    try:
        searched = _api_get(
            port,
            "/api/clips?tab=dataset&class=glass&offset=0&limit=1000&query=joby",
        )
        assert searched["total"] == 1
        assert searched["clips"][0]["microphone_label"] == "joby"
    finally:
        httpd.shutdown()


def test_clip_page_activity_is_visible_while_inventory_is_scanned(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=2)
    from sed.data import inspect as inspect_mod

    entered = threading.Event()
    release = threading.Event()
    original = inspect_mod.clips_page

    def slow_clips_page(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test clip request was not released"
        return original(*args, **kwargs)

    monkeypatch.setattr(inspect_mod, "clips_page", slow_clips_page)
    httpd, port = _serve(cfg, layout)
    result: list[dict] = []

    def request_page():
        result.append(_api_get(port, "/api/clips?tab=dataset&class=glass&query=joby"))

    request_thread = threading.Thread(target=request_page)
    request_thread.start()
    try:
        assert entered.wait(5), "clip request did not reach the inventory scan"
        activity = _api_get(port, "/api/activity")
        jobs = [job for job in activity["jobs"] if job["type"] == "clip_load"]
        assert jobs
        assert jobs[0]["title"] == "Searching clips"
        assert "dataset · glass" in jobs[0]["detail"]
    finally:
        release.set()
        request_thread.join(timeout=10)
        httpd.shutdown()
    assert result and result[0]["ok"] is True


def test_review_data_scoped_returns_only_requested(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=3)
    httpd, port = _serve(cfg, layout)
    try:
        scoped = _review_data_via_job(port, scopes=["frozen"])
        # A frozen-only rescan must not carry dataset/augmented panels, so the
        # frontend can merge it without wiping the other tabs.
        assert "panels" not in scoped
        assert "aug_panels" not in scoped
        assert "frozen_panels" in scoped
        assert scoped["full_refresh"] is False
        # Totals cover only the requested scope.
        assert set(scoped["panel_totals"].keys()) == {"frozen"}
    finally:
        httpd.shutdown()


def test_compose_start_status_job(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=4)
    httpd, port = _serve(cfg, layout)
    try:
        start_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/compose/start",
            data=b"{}", headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(start_req, timeout=10) as r:
            started = json.loads(r.read())
        assert started["ok"] is True
        job_id = started["job"]["job_id"]

        state, job = "", {}
        for _ in range(80):
            job = _api_get(port, f"/api/compose/status?job_id={job_id}")["job"]
            state = job["state"]
            if state in ("finished", "error"):
                break
            time.sleep(0.1)
        assert state == "finished", job.get("error")
        # compose_dataset actually wrote the CSVs.
        assert (layout.dataset_dir / "train.csv").exists()
        assert "splits" in job.get("summary", {})
    finally:
        httpd.shutdown()


def _poll_compose_job(port, job_id):
    job = {}
    for _ in range(80):
        job = _api_get(port, f"/api/compose/status?job_id={job_id}")["job"]
        if job["state"] in ("finished", "error"):
            break
        time.sleep(0.1)
    return job


def test_compose_start_build_split_test_flag_gates_the_carve(tmp_path, monkeypatch):
    """Routine compose must not create the split-derived frozen test; only
    the explicit build_split_test flag (Library tab's "Build frozen test
    set" action) does — and /api/dataset/test-config reports its state."""
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=10)
    cfg.dataset.split_test_enabled = True
    cfg.dataset.split_test_fraction = 0.2
    httpd, port = _serve(cfg, layout)
    try:
        before = _api_get(port, "/api/dataset/test-config")
        assert before["ok"] is True
        assert before["split_test_enabled"] is True
        assert before["frozen_test_present"] is False

        routine = _post_json(f"http://127.0.0.1:{port}/api/compose/start", {}, timeout=10.0)
        job = _poll_compose_job(port, routine["job"]["job_id"])
        assert job["state"] == "finished", job.get("error")
        assert job["summary"]["frozen_test"]["present"] is False
        assert not (layout.dataset_dir / "frozen_test_split.json").exists()

        built = _post_json(
            f"http://127.0.0.1:{port}/api/compose/start",
            {"build_split_test": True}, timeout=10.0,
        )
        job = _poll_compose_job(port, built["job"]["job_id"])
        assert job["state"] == "finished", job.get("error")
        assert job["summary"]["frozen_test"]["present"] is True
        assert (layout.dataset_dir / "frozen_test_split.json").exists()

        after = _api_get(port, "/api/dataset/test-config")
        assert after["frozen_test_present"] is True
    finally:
        httpd.shutdown()


def test_compose_start_build_split_test_off_clears_the_manifest(tmp_path, monkeypatch):
    """The Library tab's "Return clips to the training pool" action must fully
    reset the automatic holdout, not just leave it out of the composed
    test.csv. Before this fix, frozen_test_split.json kept its old keys
    forever once toggled off, so split_frozen_test_state() kept reporting
    built=True (and the panel kept offering to "return" clips already
    returned) even though nothing was held out anymore."""
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=10)
    cfg.dataset.split_test_enabled = True
    cfg.dataset.split_test_fraction = 0.2
    httpd, port = _serve(cfg, layout)
    try:
        built = _post_json(
            f"http://127.0.0.1:{port}/api/compose/start",
            {"build_split_test": True}, timeout=10.0,
        )
        job = _poll_compose_job(port, built["job"]["job_id"])
        assert job["state"] == "finished", job.get("error")
        assert (layout.dataset_dir / "frozen_test_split.json").exists()
        state = _api_get(port, "/api/dataset/test-config")["split_test_state"]
        assert state["built"] is True

        _post_json(
            f"http://127.0.0.1:{port}/api/dataset/split-config",
            {"split_test_enabled": False}, timeout=10.0,
        )
        returned = _post_json(
            f"http://127.0.0.1:{port}/api/compose/start",
            {"build_split_test": True, "frozen_test_confirmation": True}, timeout=10.0,
        )
        job = _poll_compose_job(port, returned["job"]["job_id"])
        assert job["state"] == "finished", job.get("error")
        assert not (layout.dataset_dir / "frozen_test_split.json").exists()

        after = _api_get(port, "/api/dataset/test-config")
        assert after["split_test_state"]["built"] is False
        assert after["split_test_state"]["clips"] == 0
    finally:
        httpd.shutdown()


def _seed_pretraining_corpus(classes: dict[str, int]) -> None:
    from sed.config import LIBRARY_DIR
    from sed.data.pretraining import PretrainIndex

    for label, n in classes.items():
        d = LIBRARY_DIR / "pretraining" / label
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            sf.write(str(d / f"{label}_{i:03d}.wav"), np.zeros(32000, dtype="float32"), 32000)
    PretrainIndex().rebuild()


def test_pretraining_clips_page_and_on_demand_thumbnails(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)

    layout = ProjectLayout("p")
    layout.create()
    _seed_pretraining_corpus({"klatsch": 3, "cheers": 2})

    from sed.config import Config, LIBRARY_DIR
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    cfg = Config.model_validate({"dataset": {"target_classes": ["cling"]}})
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        # Paginated clip cards for one pretraining class folder. Generous timeout:
        # the first /api/clips call lazily imports matplotlib (slow one-time cost).
        page = _get_json(
            f"http://127.0.0.1:{port}/api/clips?tab=pretraining&class=klatsch&offset=0&limit=5",
            timeout=30.0,
        )
        assert page["ok"] is True
        assert len(page["clips"]) == 3
        assert all("pretraining/klatsch" in c["path"] for c in page["clips"])
        first_clip = page["clips"][0]
        assert first_clip["sample_rate"] == 32000
        assert first_clip["duration_s"] == pytest.approx(1.0)
        assert first_clip["description"] == ""
        assert "audio:" in first_clip["meta_html"]
        assert "length 1.00 s" in first_clip["meta_html"]
        assert "sample rate 32,000 Hz" in first_clip["meta_html"]

        # On-demand /thumbnail requests queue renders into the shared preview
        # cache (202 while pending, 200 once the PNG exists).
        deadline = time.time() + 30.0
        for clip in page["clips"]:
            url = (
                f"http://127.0.0.1:{port}/thumbnail?path="
                + urllib.parse.quote(clip["path"])
            )
            while True:
                with urllib.request.urlopen(url, timeout=10.0) as r:
                    if r.status == 200:
                        break
                assert time.time() < deadline, "thumbnail render timed out"
                time.sleep(0.1)
        cache_files = list((LIBRARY_DIR / ".preview_cache").glob("*/*.png"))
        assert len(cache_files) == 3
        assert not list((LIBRARY_DIR / "pretraining" / "klatsch").glob("*.spec.png"))
        assert all(_png_size(thumb)[0] > 0 and _png_size(thumb)[1] > 0 for thumb in cache_files)
    finally:
        httpd.shutdown()


def test_panel_totals_includes_pretraining(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _seed_pretraining_corpus({"klatsch": 3, "cheers": 2})

    from sed.config import Config, LIBRARY_DIR  # noqa: F401
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    cfg = Config.model_validate({"dataset": {"target_classes": ["cling"]}})
    port = _free_port()
    handler = make_handler(cfg, "p", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        time.sleep(0.05)
        # The bug: /api/panel-totals omitting pretraining wiped the seeded counts
        # client-side and hid every block. It must carry pretraining counts.
        data = _get_json(f"http://127.0.0.1:{port}/api/panel-totals", timeout=30.0)
        assert data["ok"] is True
        assert data["panel_totals"].get("pretraining") == {"klatsch": 3, "cheers": 2}
    finally:
        httpd.shutdown()


def test_pretraining_rescan_picks_up_offline_folder_edits(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _seed_pretraining_corpus({"klatsch": 2})

    from sed.config import LIBRARY_DIR, Config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    # Pretraining endpoints are gated to pretraining-kind projects.
    port = _free_port()
    handler = make_handler(cfg=Config(project={"kind": "pretraining"}), project_id="p",
                           render_html_callable=lambda: "<html>ok</html>", layout=layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        time.sleep(0.05)
        # Add a new class folder + clips on disk (the offline-edit workflow).
        new_dir = LIBRARY_DIR / "pretraining" / "newclass"
        new_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            sf.write(str(new_dir / f"n_{i}.wav"), np.zeros(32000, dtype="float32"), 32000)

        resp = _post_json(f"http://127.0.0.1:{port}/api/pretraining/rescan", {}, timeout=30.0)
        assert resp["ok"] is True
        assert resp["pretraining_counts"] == {"klatsch": 2, "newclass": 4}
        assert resp["total"] == 6
        assert resp["classes"] == 2
    finally:
        httpd.shutdown()


def test_pretraining_counts_endpoint_never_walks_the_corpus(tmp_path, monkeypatch):
    """GET /api/pretraining/counts is the page-load path for every project in
    the unified UI: it must read the on-disk index only. Rebuilding there would
    reintroduce the 75k-clip first-paint hang."""
    import shutil as _shutil

    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _seed_pretraining_corpus({"klatsch": 3, "cheers": 2})

    from http.server import ThreadingHTTPServer

    from sed.config import Config, LIBRARY_DIR
    from sed.data.inspect_server import make_handler

    port = _free_port()
    handler = make_handler(Config(), "p", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        time.sleep(0.05)
        data = _get_json(f"{base}/api/pretraining/counts", timeout=30.0)
        assert data["ok"] is True
        assert data["pretraining_counts"] == {"klatsch": 3, "cheers": 2}
        assert data["total"] == 5
        assert data["index_present"] is True

        # Offline folder edits do NOT change the response until an explicit
        # rescan — the endpoint reads the index, never the disk tree.
        _shutil.rmtree(LIBRARY_DIR / "pretraining" / "klatsch")
        data = _get_json(f"{base}/api/pretraining/counts", timeout=10.0)
        assert data["pretraining_counts"] == {"klatsch": 3, "cheers": 2}

        # Without an index the endpoint reports empty and must not create one
        # (no corpus walk on a GET).
        index_path = LIBRARY_DIR / "pretraining" / "_index.jsonl"
        index_path.unlink()
        data = _get_json(f"{base}/api/pretraining/counts", timeout=10.0)
        assert data["pretraining_counts"] == {}
        assert data["index_present"] is False
        assert not index_path.exists(), "GET must never rebuild the corpus index"

        # The subbuckets GET (also a page-load path) must not rebuild either.
        sub = _get_json(f"{base}/api/pretraining/subbuckets", timeout=10.0)
        assert sub["ok"] is True
        assert sub["subbuckets"] == []
        assert not index_path.exists()
    finally:
        httpd.shutdown()


def test_feature_preset_endpoints_list_preview_and_apply(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PRESETS_DIR", tmp_path / "presets")
    monkeypatch.setattr(cfg_mod, "FEATURE_PRESETS_DIR", tmp_path / "presets" / "audio_features")

    from sed.config import ProjectLayout, load_config, project_config_template
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    layout = ProjectLayout("p")
    layout.create()
    # Feature-lab endpoints are gated to pretraining-kind projects.
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    wav = _seed_clip(layout)
    cfg = load_config("p")

    port = _free_port()
    handler = make_handler(cfg, "p", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        time.sleep(0.05)

        # List: built-in presets + backbone registry. (Generous timeouts: the
        # server thread imports matplotlib/librosa lazily on first render.)
        listing = _get_json(f"{base}/api/feature-presets", timeout=30.0)
        assert listing["ok"] is True
        names = {p["name"] for p in listing["presets"]}
        assert {"geometry-a-efficient-short", "linear-demo"} <= names
        # Retired built-ins are superseded by the virtual contract entry.
        assert not names & {"efficientat-standard", "ced-tiny-contract"}
        assert listing["contract"] is not None
        assert listing["contract"]["name"] == "contract:mn10_as"
        assert listing["contract"]["n_mels"] == 128
        assert listing["model_default"]["source_kind"] == "checkpoint_contract"
        assert any(b["name"] == "mn10_as" and b["pretrained_available"] for b in listing["backbones"])
        # A fresh project is created on the model feature (0.4 s), so the UI
        # reports it as the contract rather than a custom geometry.
        assert listing["current_source"]["kind"] == "contract"

        # A scratch-only MN width has no checkpoint, but must still return the
        # architecture default as a complete, loadable virtual payload.
        cfg.model.backbone = "mn01"
        cfg.model.pretrained = False
        cfg.model.initialization = "random"
        cfg.model.recipe_id = "efficientat-mn"
        scratch_listing = _get_json(f"{base}/api/feature-presets", timeout=30.0)
        assert scratch_listing["model_default"]["name"] == "contract:mn01"
        assert scratch_listing["model_default"]["source_kind"] == "architecture_default"
        assert scratch_listing["model_default"]["n_fft"] == 1024

        # Live preview renders a PNG for both feature types.
        for ft in ("mel", "linear"):
            url = (
                f"{base}/api/feature-preview?feature_type={ft}"
                f"&path={urllib.parse.quote(str(wav))}"
            )
            with urllib.request.urlopen(url, timeout=30) as r:
                assert r.headers.get("Content-Type") == "image/png"
                assert r.read(4) == b"\x89PNG"

        # Save an off-contract preset, then set it as the preview-only feature.
        save = _post_json(f"{base}/api/feature-preset/save", {
            "name": "test-mel",
            "preset": {"feature_type": "mel", "n_mels": 64, "sample_rate": 16000,
                       "backbone": "mn10_as", "pretrained": True},
        }, timeout=30.0)
        assert save["ok"] is True and save["preset"]["name"] == "test-mel"

        before = load_config("p")
        applied = _post_json(f"{base}/api/feature-preset/apply", {"name": "test-mel"}, timeout=30.0)
        assert applied["ok"] is True
        assert applied["preset"] == "test-mel"
        # Preview-only: no model-compatibility warning, and the project config
        # (model/training feature) is left untouched.
        assert "warning" not in applied
        reloaded = load_config("p")
        assert reloaded.features.n_mels == before.features.n_mels
        assert reloaded.features.n_mels != 64
        assert reloaded.model.sample_rate == before.model.sample_rate

        # The preview preset is recorded for dataset thumbnails and resolves to
        # the preset's geometry, independent of [features].
        from sed.data.presets import preview_feature_parts
        feats, sr = preview_feature_parts(reloaded, layout)
        assert feats.n_mels == 64 and sr == 16000

        # The listing surfaces it as the preview preset (not the training preset).
        relisted = _get_json(f"{base}/api/feature-presets", timeout=30.0)
        assert relisted["preview_preset"] == "test-mel"
        assert relisted["active_preset"] != "test-mel"

        # Delete it again.
        deleted = _post_json(f"{base}/api/feature-preset/delete", {"name": "test-mel"}, timeout=30.0)
        assert deleted["deleted"] is True
    finally:
        httpd.shutdown()


def test_model_recipe_select_applies_legacy_feature_channels_and_clears_unsaved_source(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PRESETS_DIR", tmp_path / "presets")
    monkeypatch.setattr(cfg_mod, "FEATURE_PRESETS_DIR", tmp_path / "presets" / "audio_features")

    from http.server import ThreadingHTTPServer

    from sed.config import ProjectLayout, load_config, project_config_template
    from sed.data.compose import load_dataset_selection, write_dataset_selection
    from sed.data.inspect_server import make_handler
    from sed.data.presets import DATASET_PRESET_KEY, load_feature_preset

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    cfg = load_config("p")
    write_dataset_selection(layout, {DATASET_PRESET_KEY: "stale-saved-preset"})

    port = _free_port()
    handler = make_handler(cfg, "p", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        time.sleep(0.05)
        legacy = dict(load_feature_preset("legacy-sounddetection"))
        legacy["name"] = ""
        selected = _post_json(f"{base}/api/model-recipe/select", {
            "architecture_id": "legacy-smallervggnet",
            "task": "clip",
            "initialization": "random",
            "params": {},
            "feature": legacy,
        }, timeout=30.0)

        assert selected["ok"] is True
        assert selected["feature"]["channels"] == 3
        assert selected["feature"]["source"]["kind"] == "contract"
        assert selected["feature"]["source"]["source_kind"] == "architecture_default"
        updated = load_config("p")
        assert updated.model.backbone == "legacy-smallervggnet"
        assert updated.model.channels == 3
        assert updated.model.sample_rate == 44100
        assert updated.features.feature_type == "linear"
        assert updated.features.stft_engine == "scipy"
        assert updated.features.db_norm is True
        assert updated.features.colormap == "jet"
        assert updated.features.preset == ""
        assert DATASET_PRESET_KEY not in load_dataset_selection(layout)
    finally:
        httpd.shutdown()


def test_model_recipe_select_legacy_default_feature_overrides_stale_standardize(tmp_path, monkeypatch):
    """Regression for the Model tab "Model default (unsaved)" source: picking
    Legacy SmallerVGGNet after a project already has a fitted
    standardize=True linear feature (e.g. from a previous clink-v2-linear-*
    preset) must not merge stale standardize=True with the legacy preset's
    db_norm=True — that combination is rejected by FeaturesCfg. The frontend
    sends the architecture's own default_feature dict (not the full saved
    preset), so this exercises recommended_feature_for()'s legacy branch
    exactly as the "Model default (unsaved)" dropdown entry would.
    """
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from http.server import ThreadingHTTPServer

    from sed.config import ProjectLayout, load_config, project_config_template, update_config_toml_values
    from sed.data.inspect_server import make_handler
    from sed.train.model_lab import architecture_by_id, recommended_feature_for

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    initial = load_config("p")
    stale_features = initial.features.model_copy(update={
        "feature_type": "linear", "frontend": "efficientat",
        "n_fft": 1024, "win_length": 882, "hop_length": 220,
        "fmin": 80, "fmax": 20_000, "window_seconds": 0.4,
        "log_power": True, "stft_engine": "librosa", "db_norm": False,
        "standardize": True, "standardize_mean": -3.25, "standardize_std": 1.75,
        "standardize_stats_resolved": True,
        "standardize_stats_dataset_sha256": "a" * 64,
        "standardize_stats_split_seed": 22, "standardize_stats_clips": 2_045,
    })
    update_config_toml_values(layout.config_toml, {
        "model": {"sample_rate": 44_100, "channels": 1},
        "features": stale_features.model_dump(),
    })
    cfg = load_config("p")

    # The exact dict the Model tab's catalog exposes as arch.default_feature
    # for this architecture (what "Model default (unsaved)" sends).
    default_feature = recommended_feature_for(architecture_by_id("legacy-smallervggnet", "p"))
    assert default_feature["standardize"] is False

    port = _free_port()
    handler = make_handler(cfg, "p", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        time.sleep(0.05)
        selected = _post_json(f"{base}/api/model-recipe/select", {
            "architecture_id": "legacy-smallervggnet",
            "task": "clip",
            "initialization": "random",
            "params": {},
            "feature": default_feature,
        }, timeout=30.0)

        assert selected["ok"] is True, selected
        updated = load_config("p")
        assert updated.features.feature_type == "linear"
        assert updated.features.db_norm is True
        assert updated.features.colormap == "jet"
        assert updated.features.standardize is False
        assert updated.features.standardize_stats_clips == 0
    finally:
        httpd.shutdown()


def test_model_recipe_selection_keeps_fitted_linear_stats_when_reapplying_template(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from http.server import ThreadingHTTPServer

    from sed.config import ProjectLayout, load_config, project_config_template, update_config_toml_values
    from sed.data.inspect_server import make_handler

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    initial = load_config("p")
    fitted_features = initial.features.model_copy(update={
        "feature_type": "linear",
        "frontend": "efficientat",
        "n_fft": 1024,
        "win_length": 882,
        "hop_length": 220,
        "fmin": 80,
        "fmax": 20_000,
        "window_seconds": 0.4,
        "log_power": True,
        "stft_engine": "librosa",
        "pcen": False,
        "db_norm": False,
        "colormap": "none",
        "standardize": True,
        "standardize_mean": -3.25,
        "standardize_std": 1.75,
        "standardize_stats_resolved": True,
        "standardize_stats_dataset_sha256": "a" * 64,
        "standardize_stats_split_seed": 22,
        "standardize_stats_clips": 2_048,
    })
    update_config_toml_values(layout.config_toml, {
        "model": {"sample_rate": 44_100, "channels": 1},
        "features": fitted_features.model_dump(),
    })
    cfg = load_config("p")

    port = _free_port()
    handler = make_handler(cfg, "p", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        time.sleep(0.05)
        selected = _post_json(f"{base}/api/model-recipe/select", {
            "architecture_id": "clink-cnn-v2",
            "task": "clip",
            "initialization": "random",
            "params": {},
            # The built-in experimental preset is deliberately unresolved.  A
            # verify after fitting it must not replace the fitted provenance.
            "feature": {
                "name": "clink-v2-linear-fft1024",
                "sample_rate": 44_100,
                "feature_type": "linear",
                "frontend": "efficientat",
                "n_fft": 1024,
                "win_length": 882,
                "hop_length": 220,
                "n_mels": 128,
                "fmin": 80,
                "fmax": 20_000,
                "window_seconds": 0.4,
                "log_power": True,
                "stft_engine": "librosa",
                "pcen": False,
                "db_norm": False,
                "colormap": "none",
                "standardize": True,
                "standardize_stats_resolved": False,
                "standardize_stats_dataset_sha256": "",
                "standardize_stats_split_seed": 0,
                "standardize_stats_clips": 0,
            },
        }, timeout=30.0)

        assert selected["ok"] is True
        assert selected["preserved_fitted_linear_statistics"] is True
        updated = load_config("p")
        assert updated.features.standardize_mean == -3.25
        assert updated.features.standardize_std == 1.75
        assert updated.features.standardize_stats_resolved is True
        assert updated.features.standardize_stats_dataset_sha256 == "a" * 64
        assert updated.features.standardize_stats_split_seed == 22
        assert updated.features.standardize_stats_clips == 2_048
    finally:
        httpd.shutdown()


def test_feature_training_apply_persists_full_project_contract_and_readiness(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from http.server import ThreadingHTTPServer

    from sed.config import ProjectLayout, load_config, project_config_template
    from sed.data import inspect_server as inspect_server_mod
    from sed.data.inspect_server import make_handler
    from sed.train import backbones as backbones_mod

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    cfg = load_config("p")
    monkeypatch.setattr(backbones_mod, "probe_spec", lambda *args, **kwargs: {
        "input_channels": kwargs["input_channels"], "n_stages": 5,
    })
    monkeypatch.setattr(inspect_server_mod, "_train_model_probe", lambda _cfg: {"n_stages": 5})

    port = _free_port()
    handler = make_handler(cfg, "p", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        applied = _post_json(f"{base}/api/feature-training/apply", {
            "feature": {
                "name": "", "feature_type": "linear", "frontend": "efficientat",
                "sample_rate": 44_100, "n_fft": 1024, "win_length": 882,
                "hop_length": 220, "n_mels": 128, "fmin": 80, "fmax": 20_000,
                "window_seconds": 0.4, "log_power": True, "stft_engine": "librosa",
                "pcen": False, "db_norm": False, "colormap": "none",
                "standardize": True, "standardize_mean": 0.0, "standardize_std": 1.0,
                "standardize_stats_resolved": False,
                "standardize_stats_dataset_sha256": "",
                "standardize_stats_split_seed": 0,
                "standardize_stats_val_fraction": 0.0,
                "standardize_stats_clips": 0,
            },
        }, timeout=30.0)
        assert applied["ok"] is True
        assert applied["readiness"]["ready"] is False
        assert applied["readiness"]["auto_repairable"] is True
        saved = load_config("p")
        assert saved.model.sample_rate == 44_100
        assert saved.model.channels == 1
        assert saved.features.feature_type == "linear"
        assert saved.features.standardize is True
        assert saved.features.standardize_stats_resolved is False

        train_config = _get_json(f"{base}/api/train/config", timeout=10.0)
        assert train_config["features"]["standardize_stats_val_fraction"] == 0.0
        assert train_config["feature_readiness"]["auto_repairable"] is True
    finally:
        httpd.shutdown()


def test_model_recipe_select_coerces_browser_choice_params(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from http.server import ThreadingHTTPServer

    from sed.config import ProjectLayout, load_config, project_config_template
    from sed.data.inspect_server import make_handler

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    cfg = load_config("p")

    port = _free_port()
    handler = make_handler(cfg, "p", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        time.sleep(0.05)
        selected = _post_json(f"{base}/api/model-recipe/select", {
            "architecture_id": "kws-bcresnet",
            "task": "clip",
            "weight_id": "",
            "initialization": "random",
            "params": {"time_stride": "1"},
            "feature": None,
        }, timeout=30.0)

        assert selected["ok"] is True
        assert selected["recipe"]["params"] == {
            "width_mult": 1.0,
            "time_stride": 1,
            "depth": 4,
        }
        updated = load_config("p")
        assert updated.model.backbone == "kws-bcresnet"
        assert updated.model.pretrained is False
        assert updated.model.initialization == "random"
        assert updated.model.params == selected["recipe"]["params"]
    finally:
        httpd.shutdown()


def test_model_recipe_select_random_init_is_architecture_only(tmp_path, monkeypatch):
    """Random (scratch) initialization on an EfficientAT checkpoint variant must
    persist an architecture-only recipe: the hop/mels suffix describes the
    checkpoint's training feature (not the network), and no checkpoint loads —
    so the backbone reduces to its base width and no weight id is recorded."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from http.server import ThreadingHTTPServer

    from sed.config import ProjectLayout, load_config, project_config_template
    from sed.data.inspect_server import make_handler

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    cfg = load_config("p")

    port = _free_port()
    handler = make_handler(cfg, "p", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        time.sleep(0.05)
        selected = _post_json(f"{base}/api/model-recipe/select", {
            "architecture_id": "efficientat-mn",
            "task": "clip",
            "weight_id": "efficientat:mn10_as_mels_256",
            "initialization": "random",
            "params": {},
            "feature": None,
        }, timeout=60.0)

        assert selected["ok"] is True
        assert selected["recipe"]["backbone"] == "mn10_as"
        assert selected["recipe"]["weight_id"] == ""
        assert selected["recipe"]["initialization"] == "random"
        updated = load_config("p")
        assert updated.model.backbone == "mn10_as"
        assert updated.model.weight_id == ""
        assert updated.model.pretrained is False
        assert updated.model.initialization == "random"

        # The same variant with upstream initialization keeps the exact
        # checkpoint identity (backbone + weight id) for provenance.
        selected = _post_json(f"{base}/api/model-recipe/select", {
            "architecture_id": "efficientat-mn",
            "task": "clip",
            "weight_id": "efficientat:mn10_as_mels_256",
            "initialization": "upstream",
            "params": {},
            "feature": None,
        }, timeout=60.0)
        assert selected["ok"] is True
        assert selected["recipe"]["backbone"] == "mn10_as_mels_256"
        assert selected["recipe"]["weight_id"] == "efficientat:mn10_as_mels_256"
        updated = load_config("p")
        assert updated.model.pretrained is True
        assert updated.model.initialization == "upstream"
    finally:
        httpd.shutdown()


def test_feature_audio_endpoint_wav_roundtrip_and_band_mask(tmp_path, monkeypatch):
    """/api/feature-audio renders the lab's listening config as WAV: the sample
    window, optionally the overlay mix, optionally band-limited to Hz ranges."""
    import io

    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from http.server import ThreadingHTTPServer

    from sed.config import ProjectLayout, load_config, project_config_template
    from sed.data.inspect_server import make_handler

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    # Two-tone clip (1 kHz + 6 kHz) inside the library dir (allowed root).
    out = layout.positives_dir("cheers_loud")
    out.mkdir(parents=True, exist_ok=True)
    wav = out / "two_tone.wav"
    t = np.arange(32000, dtype=np.float32) / 32000.0
    sf.write(str(wav), (0.4 * np.sin(2 * np.pi * 1000 * t)
                        + 0.4 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32), 32000)
    cfg = load_config("p")

    port = _free_port()
    handler = make_handler(cfg, "p", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    q = urllib.parse.quote(str(wav))
    try:
        time.sleep(0.05)

        # Plain render: valid WAV at the requested rate/window, peak-normalized.
        with urllib.request.urlopen(f"{base}/api/feature-audio?path={q}", timeout=30) as r:
            assert r.headers.get("Content-Type") == "audio/wav"
            body = r.read()
        data, sr = sf.read(io.BytesIO(body), dtype="float32")
        assert sr == 32000 and len(data) == 32000
        assert 0.5 < float(np.abs(data).max()) <= 1.0

        # Band mask keeps only the requested range: 6 kHz tone survives, 1 kHz dies.
        with urllib.request.urlopen(f"{base}/api/feature-audio?path={q}&bands=5500-6500",
                                    timeout=30) as r:
            filtered, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
        spec = np.abs(np.fft.rfft(filtered)) ** 2
        f = np.fft.rfftfreq(len(filtered), 1.0 / sr)
        in_band = float(spec[(f >= 5500) & (f <= 6500)].sum())
        out_band = float(spec[(f >= 900) & (f <= 1100)].sum())
        assert in_band > 100.0 * out_band

        # Overlay mix renders and decodes.
        with urllib.request.urlopen(f"{base}/api/feature-audio?path={q}&overlay={q}&snr=0",
                                    timeout=30) as r:
            mixed, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
        assert sr == 32000 and len(mixed) == 32000

        # Paths outside the allowed roots are forbidden.
        outside = tmp_path / "outside.wav"
        sf.write(str(outside), np.zeros(1000, dtype="float32"), 32000)
        try:
            urllib.request.urlopen(
                f"{base}/api/feature-audio?path={urllib.parse.quote(str(outside))}", timeout=30)
            raise AssertionError("expected 403 for a path outside the allowed roots")
        except urllib.error.HTTPError as e:
            assert e.code == 403

        # Manual onset override plumbs through the analysis endpoint.
        analysis = _get_json(
            f"{base}/api/feature-analysis?path={q}&onset_start_ms=0&onset_end_ms=200",
            timeout=30.0)
        clip0 = analysis["clips"][0]
        assert clip0["onset"]["source"] == "manual"
        assert clip0["fingerprint"]["source"] == "manual"
    finally:
        httpd.shutdown()


def test_train_config_reports_presets_and_after_training_gating(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")
    # Avoid building a real backbone just to count stages.
    from sed.train import backbones as bb_mod
    monkeypatch.setattr(bb_mod, "probe_spec", lambda *a, **k: {"n_stages": 5, "batchnorm_layers": 3})

    layout = ProjectLayout("p")
    layout.create()

    cfg = cfg_mod.Config()
    cfg.model.classes = ["cheers_loud", "_negative"]
    cfg.training.fine_tune_policy = "head"
    # Class balance the endpoint reads for recommendations/gating.
    (layout.dataset_dir / "dataset_snapshot.json").write_text(
        json.dumps({"by_class": {"cheers_loud": 400, "_negative": 1600}})
    )

    from http.server import ThreadingHTTPServer

    from sed.data.inspect_server import make_handler
    port = _free_port()
    handler = make_handler(cfg=cfg, project_id="p",
                           render_html_callable=lambda: "<html>ok</html>",
                           layout=layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        time.sleep(0.05)
        d = _get_json(f"http://127.0.0.1:{port}/api/train/config", timeout=10.0)
        assert d["ok"] is True

        presets = d["presets"]
        assert [p["id"] for p in presets][:1] == ["auto"]
        assert len(presets) >= 4
        for p in presets:
            assert p["values"] and "fine_tune_policy" in p["values"]
            assert p["requires_init"] in {"any", "transfer", "scratch"}
            assert isinstance(p["applicable"], bool)
        auto = next(p for p in presets if p["id"] == "auto")
        assert auto["values"]["fine_tune_policy"] == "last_two"
        assert "last 2 of 5 encoder stages" in auto["reasons"]["fine_tune_policy"]
        # Pretrained (upstream) model: transfer presets apply, scratch does not.
        venue = next(p for p in presets if p["id"] == "venue_finetune")
        scratch = next(p for p in presets if p["id"] == "scratch_full")
        assert venue["values"]["fine_tune_policy"] == "last_two"
        assert venue["applicable"] is True
        assert scratch["applicable"] is False

        assert d["training"]["fine_tune_policy"] == "head"
        assert d["recommended"]["values"]["fine_tune_policy"] == "last_two"
        assert "last 2 of 5 encoder stages" in d["recommended"]["reasons"]["fine_tune_policy"]
        assert d["class_counts"] == {"cheers_loud": 400, "_negative": 1600}
        assert d["metric_descriptor"]["effective_loss"] == "bce_logits"
        assert d["metric_descriptor"]["probability_mode"] == "sigmoid"
        assert "Per-label" in d["metric_descriptor"]["val_accuracy_description"]

        after = d["after_training"]
        assert after["has_negative_class"] is True     # _negative in classes
        assert after["has_frozen_test"] is False        # no composed test.csv
        assert after["has_val_split"] is False          # no composed train.csv to split yet
        assert after["venue_negatives"] == 1600
        assert after["export_supported"] is True
        assert after["export_message"] == ""
        assert after["batchnorm_layers"] == 3
        assert after["batchnorm_recalibration_supported"] is True
        # Resolved rule ids, so the form shows what the next run will score
        # even for a config.toml written before the rule system.
        assert d["training"]["auto_test_rules"] == ["fixed@0.90"]
        assert d["training"]["calibrate_thresholds"] is False
        assert d["fine_tune_options"]["n_stages"] == 5
        assert d["fine_tune_options"]["has_transfer_weights"] is True
    finally:
        httpd.shutdown()


def test_train_config_marks_export_unsupported_models(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")
    from sed.train import backbones as bb_mod
    monkeypatch.setattr(bb_mod, "probe_spec", lambda *a, **k: {"n_stages": 5, "batchnorm_layers": 0})

    layout = ProjectLayout("p")
    layout.create()
    cfg = cfg_mod.Config()
    cfg.model.backbone = "dymn10_as"
    cfg.model.recipe_id = "efficientat-dymn"
    cfg.model.weight_id = "efficientat:dymn10_as"
    cfg.model.classes = ["cheers_loud", "_negative"]
    (layout.dataset_dir / "dataset_snapshot.json").write_text(
        json.dumps({"by_class": {"cheers_loud": 400, "_negative": 1600}})
    )

    from http.server import ThreadingHTTPServer

    from sed.data.inspect_server import make_handler
    port = _free_port()
    handler = make_handler(cfg=cfg, project_id="p",
                           render_html_callable=lambda: "<html>ok</html>",
                           layout=layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        time.sleep(0.05)
        d = _get_json(f"http://127.0.0.1:{port}/api/train/config", timeout=10.0)
        assert d["ok"] is True
        after = d["after_training"]
        assert after["export_supported"] is False
        assert "does not support OpenVINO export" in after["export_message"]
        assert "PyTorch checkpoint" in after["export_message"]
        assert after["batchnorm_recalibration_supported"] is False
    finally:
        httpd.shutdown()


def test_train_config_presets_respect_parent_transfer_without_builtin_weights(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")
    from sed.train import backbones as bb_mod
    monkeypatch.setattr(bb_mod, "probe_spec", lambda *a, **k: {"n_stages": 5})

    layout = ProjectLayout("p")
    layout.create()

    cfg = cfg_mod.Config()
    cfg.model.backbone = "mn01"
    cfg.model.pretrained = False
    cfg.model.initialization = "parent"
    cfg.model.parent_project = "pretrain"
    cfg.model.parent_run = "run-a"
    cfg.model.classes = ["cheers_loud", "_negative"]
    cfg.training.fine_tune_policy = "head"
    (layout.dataset_dir / "dataset_snapshot.json").write_text(
        json.dumps({"by_class": {"cheers_loud": 400, "_negative": 1600}})
    )

    from http.server import ThreadingHTTPServer

    from sed.data.inspect_server import make_handler
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        time.sleep(0.05)
        d = _get_json(f"http://127.0.0.1:{port}/api/train/config", timeout=10.0)
        assert d["ok"] is True

        venue = next(p for p in d["presets"] if p["id"] == "venue_finetune")
        domain = next(p for p in d["presets"] if p["id"] == "domain_pretrain")
        scratch = next(p for p in d["presets"] if p["id"] == "scratch_full")
        assert venue["values"]["fine_tune_policy"] == "last_two"
        assert domain["values"]["fine_tune_policy"] == "full"
        # Parent transfer counts as real encoder weights: transfer presets
        # apply even without built-in AudioSet weights; scratch does not.
        assert venue["applicable"] is True
        assert scratch["applicable"] is False
        assert d["recommended"]["values"]["fine_tune_policy"] == "last_two"
        assert d["training"]["fine_tune_policy"] == "head"
        assert d["fine_tune_options"]["has_transfer_weights"] is True
    finally:
        httpd.shutdown()


def test_train_config_recognizes_model_lab_pretrained_recipe_weights(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")
    from sed.train import backbones as bb_mod
    monkeypatch.setattr(bb_mod, "probe_spec", lambda *a, **k: {"n_stages": 5})

    layout = ProjectLayout("p")
    layout.create()

    cfg = cfg_mod.Config.model_validate({"project": {"kind": "pretraining"}})
    cfg.model.backbone = "mn10_as_mels_256"
    cfg.model.pretrained = True
    cfg.model.initialization = "upstream"
    cfg.model.recipe_id = "efficientat-mn"
    cfg.model.weight_id = "efficientat:mn10_as_mels_256"
    cfg.model.classes = ["cheers_loud", "_negative"]
    cfg.training.fine_tune_policy = "head"
    (layout.dataset_dir / "dataset_snapshot.json").write_text(
        json.dumps({"by_class": {"cheers_loud": 400, "_negative": 1600}})
    )

    from http.server import ThreadingHTTPServer

    from sed.data.inspect_server import make_handler
    port = _free_port()
    handler = make_handler(
        cfg=cfg,
        project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        time.sleep(0.05)
        d = _get_json(f"http://127.0.0.1:{port}/api/train/config", timeout=10.0)
        assert d["ok"] is True

        venue = next(p for p in d["presets"] if p["id"] == "venue_finetune")
        auto = next(p for p in d["presets"] if p["id"] == "auto")
        assert d["fine_tune_options"]["has_transfer_weights"] is True
        assert d["training"]["fine_tune_policy"] == "head"
        assert venue["applicable"] is True
        assert venue["values"]["fine_tune_policy"] == "last_two"
        assert auto["values"]["fine_tune_policy"] == "last_two"
        assert d["recommended"]["values"]["fine_tune_policy"] == "last_two"
    finally:
        httpd.shutdown()


def test_dataset_split_config_endpoint_persists_and_reloads(tmp_path, monkeypatch):
    """POST /api/dataset/split-config writes config.toml and the running
    server picks the new values up immediately (no restart needed)."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.config import ProjectLayout, load_config, project_config_template
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(project_config_template("p", kind="pretraining"), encoding="utf-8")
    cfg = load_config("p")

    port = _free_port()
    handler = make_handler(cfg, "p", lambda: "<html>ok</html>", layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        time.sleep(0.05)

        before = _get_json(f"{base}/api/train/config", timeout=10.0)["dataset_config"]
        assert before["val_fraction"] == 0.15
        assert before["pretraining_test_enabled"] is False
        assert before["pretraining_test_fraction"] == 0.1

        saved = _post_json(f"{base}/api/dataset/split-config", {
            "val_fraction": 0.25,
            "pretraining_test_enabled": False,
            "pretraining_test_fraction": 0.05,
        }, timeout=10.0)
        assert saved["ok"] is True
        assert saved["dataset_config"]["val_fraction"] == 0.25
        assert saved["dataset_config"]["pretraining_test_enabled"] is False
        assert saved["dataset_config"]["pretraining_test_fraction"] == 0.05

        # Reflected immediately by the already-running server (in-place reload)...
        after = _get_json(f"{base}/api/train/config", timeout=10.0)["dataset_config"]
        assert after["val_fraction"] == 0.25
        assert after["pretraining_test_enabled"] is False
        assert after["pretraining_test_fraction"] == 0.05

        # ...and durably on disk, independent of the running process.
        reloaded = load_config("p")
        assert reloaded.dataset.val_fraction == 0.25
        assert reloaded.dataset.pretraining_test_enabled is False
        assert reloaded.dataset.pretraining_test_fraction == 0.05

        status, body = _post_json_status(f"{base}/api/dataset/split-config",
                                          {"val_fraction": 0.95}, timeout=10.0)
        assert status == 400
        assert body["ok"] is False
        unchanged = load_config("p")
        assert unchanged.dataset.val_fraction == 0.25   # rejected write left config untouched
    finally:
        httpd.shutdown()


# ---- Runs & compare: augmentation truth + compare endpoint ----------------


def _seed_train_run_bundle(
    layout: ProjectLayout,
    model_id: str,
    *,
    manifest_by_origin: dict | None = None,
    snapshot_by_origin: dict | None = None,
    epoch_metrics: list | None = None,
    test_f1: float = 0.9,
    model_stats: dict | None = None,
) -> Path:
    """A model bundle as `sed train` leaves it, with augmentation recorded
    either in the manifest (new runs) or only in the dataset snapshot (old)."""
    model_dir = layout.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "training_config.json").write_text(json.dumps({
        "model": {"backbone": "mn10", "pretrained": True, "sample_rate": 32000},
        "features": {"frontend": "efficientat", "feature_type": "mel", "n_fft": 1024,
                     "hop_length": 320, "n_mels": 128, "fmin": 0, "fmax": 15000,
                     "window_seconds": 1.0},
        "training": {"epochs": 50, "batch_size": 64, "lr_head": 0.0003,
                     "lr_backbone": 0.00003, "seed": 42, "loss": "bce_logits"},
        # The recipe-builder block defaults to all-enabled even for runs whose
        # dataset contains zero augmented clips — it must never be read as truth.
        "augmentation": {"enabled": True, "pitch_shift_enabled": True,
                         "overlay_enabled": True, "mask_filter_enabled": True,
                         "samples_per_source": 2},
    }), encoding="utf-8")
    dataset_block: dict = {"train": 100, "val": 20, "test": 30,
                           "by_class": {"cheers": 60, "_negative": 40}}
    if manifest_by_origin is not None:
        dataset_block["by_origin"] = manifest_by_origin
    manifest_payload = {
        "model_id": model_id,
        "created_at": "2026-07-01T12:00:00Z",
        "description": model_id,
        "dataset": dataset_block,
        "training": {"epochs_ran": 10, "epochs_max": 50, "selected_epoch": 7,
                     "selected_val_loss": 0.123, "best_val_f1_macro_targets": 0.8,
                     "best_val_ap_macro_targets": 0.82, "best_val_accuracy": 0.9},
        "test_metrics": {"f1_macro_targets": test_f1, "ap_macro_targets": 0.95,
                         "accuracy": 0.92},
        # A modern run also carries its frozen-test scores keyed by rule; the
        # default is the shared 0.90 confidence gate that `sed train` writes.
        "frozen_test": {
            "fixed@0.90": {
                "rule": {"id": "fixed@0.90", "kind": "fixed", "param": 0.9, "label": "≥ 0.90"},
                "f1_macro_targets": test_f1, "ap_macro_targets": 0.95, "accuracy": 0.92,
                "threshold": 0.9, "n_samples": 30, "test_set_hash": "hash-current",
            },
        },
        "files": {"checkpoint": "checkpoint_best_epoch-001.ckpt"},
    }
    if model_stats is not None:
        manifest_payload["model_stats"] = model_stats
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest_payload), encoding="utf-8")
    if snapshot_by_origin is not None:
        (model_dir / "dataset").mkdir(exist_ok=True)
        (model_dir / "dataset" / "dataset_snapshot.json").write_text(
            json.dumps({"by_origin": snapshot_by_origin}), encoding="utf-8")
    if epoch_metrics is not None:
        (model_dir / "reports").mkdir(exist_ok=True)
        (model_dir / "reports" / "training_report.json").write_text(
            json.dumps({"epoch_metrics": epoch_metrics}), encoding="utf-8")
    return model_dir


def _start_train_api_server(layout: ProjectLayout) -> tuple:
    from sed.config import Config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    port = _free_port()
    handler = make_handler(
        cfg=Config(),
        project_id=layout.id,
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def test_train_run_summary_uses_actual_feature_band_count_for_legacy_linear(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    model_dir = layout.model_dir("legacy")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "training_config.json").write_text(json.dumps({
        "model": {"backbone": "legacy-smallervggnet", "pretrained": False, "sample_rate": 44100},
        "features": {
            "feature_type": "linear", "n_fft": 190, "win_length": 190,
            "hop_length": 167, "fmin": 0, "fmax": 22050,
            "window_seconds": 0.4, "stft_engine": "scipy",
            "db_norm": True, "colormap": "jet", "image_size": 96,
        },
        "training": {"train_depth": "full", "loss": "bce_logits"},
    }), encoding="utf-8")

    from sed.data.inspect_server import _run_config_summary

    summary = _run_config_summary(layout, "legacy")
    assert summary["loss"] == "bce_logits"
    assert summary["feature"] == "linear 96b @ 44100Hz / 0.4s"


def test_train_run_reports_augmentation_truth_not_config_defaults(tmp_path, monkeypatch):
    """Regression: config snapshot says augmentation enabled=true, but the run's
    dataset had zero derived clips — the endpoint must report applied=False."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _seed_train_run_bundle(
        layout, "run-no-aug",
        snapshot_by_origin={
            "cheers": {"sounds": 60, "derived": 0, "total": 60},
            "_negative": {"sounds": 40, "derived": 0, "total": 40},
        },
    )
    httpd, port = _start_train_api_server(layout)
    try:
        payload = _get_json(f"http://127.0.0.1:{port}/api/train/run?model_id=run-no-aug")
        assert payload["ok"] is True
        assert payload["training_config"]["augmentation"]["enabled"] is True
        aug = payload["augmentation_applied"]
        assert aug["available"] is True
        assert aug["applied"] is False
        assert aug["derived_total"] == 0
        assert payload["run_recipe"] is None  # manifest has no model_recipe — tolerated
        assert payload["run_model_stats"] is None  # pre-backfill manifest — tolerated
    finally:
        httpd.shutdown()


def test_train_compare_returns_configs_metrics_curves_and_aug_truth(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    # New-style run: by_origin in manifest (fast path), has epoch curves.
    _seed_train_run_bundle(
        layout, "run-aug",
        manifest_by_origin={
            "cheers": {"sounds": 30, "derived": 30, "total": 60},
            "_negative": {"sounds": 40, "derived": 0, "total": 40},
        },
        epoch_metrics=[
            {"epoch": 0, "train_loss": 1.0, "val_loss": 0.8, "val_ap_macro_targets": 0.4},
            {"epoch": 1, "train_loss": 0.7, "val_loss": 0.6, "val_ap_macro_targets": 0.7},
        ],
        test_f1=0.93,
        model_stats={"total_params": 4904163, "trainable_params": 4904163,
                     "input_channels": 1, "n_bands": 128, "n_frames": 100,
                     "feature_type": "mel", "colormap": "none", "source": "train"},
    )
    # Old-style run: truth only in the copied dataset snapshot, no training report.
    _seed_train_run_bundle(
        layout, "run-old",
        snapshot_by_origin={
            "cheers": {"sounds": 60, "derived": 0, "total": 60},
            "_negative": {"sounds": 40, "derived": 0, "total": 40},
        },
        test_f1=0.88,
    )
    httpd, port = _start_train_api_server(layout)
    try:
        base = f"http://127.0.0.1:{port}"
        payload = _get_json(f"{base}/api/train/compare?model_ids=run-aug,run-old")
        assert payload["ok"] is True
        runs = {r["model_id"]: r for r in payload["runs"]}
        assert set(runs) == {"run-aug", "run-old"}

        aug_run = runs["run-aug"]
        assert aug_run["training_config"]["features"]["hop_length"] == 320
        assert aug_run["training_config"]["training"]["loss"] == "bce_logits"
        assert aug_run["run_training_stats"]["selected_val_loss"] == 0.123
        assert aug_run["frozen_test"]["fixed@0.90"]["f1"] == 0.93
        assert aug_run["metrics"]["val_f1"] == 0.8
        assert aug_run["augmentation_applied"]["applied"] is True
        assert aug_run["augmentation_applied"]["derived_total"] == 30
        assert aug_run["augmentation_applied"]["by_class"]["cheers"]["derived"] == 30
        assert len(aug_run["epoch_metrics"]) == 2
        assert aug_run["run_model_stats"]["total_params"] == 4904163
        assert aug_run["run_model_stats"]["n_bands"] == 128

        old_run = runs["run-old"]
        assert old_run["augmentation_applied"]["applied"] is False
        assert old_run["epoch_metrics"] is None

        # A missing run becomes an error entry without failing the request.
        payload = _get_json(f"{base}/api/train/compare?model_ids=run-aug,ghost")
        rows = {r["model_id"]: r for r in payload["runs"]}
        assert "error" in rows["ghost"]
        assert "error" not in rows["run-aug"]

        # Fewer than two ids is a client error.
        try:
            _get_json(f"{base}/api/train/compare?model_ids=run-aug")
            raise AssertionError("expected HTTP 400 for a single model_id")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        httpd.shutdown()


def test_train_runs_list_surfaces_manifest_augmented_counts_only(tmp_path, monkeypatch):
    """The list endpoint reports derived counts from the manifest fast path and
    leaves old runs at null — it must not fall back to snapshot parsing."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _seed_train_run_bundle(
        layout, "run-aug",
        manifest_by_origin={"cheers": {"sounds": 30, "derived": 25, "total": 55}},
    )
    _seed_train_run_bundle(
        layout, "run-old",
        snapshot_by_origin={"cheers": {"sounds": 60, "derived": 999, "total": 1059}},
    )
    httpd, port = _start_train_api_server(layout)
    try:
        payload = _get_json(f"http://127.0.0.1:{port}/api/train/runs")
        assert payload["ok"] is True
        rows = {r["model_id"]: r for r in payload["runs"]}
        assert rows["run-aug"]["augmented"] == 25
        assert rows["run-aug"]["loss"] == "bce_logits"
        assert rows["run-aug"]["selected_val_loss"] == 0.123
        assert rows["run-old"]["augmented"] is None
        assert rows["run-old"]["loss"] == "bce_logits"
        assert rows["run-old"]["selected_val_loss"] == 0.123
    finally:
        httpd.shutdown()


def test_batch_evaluate_job_scores_every_rule_across_models(tmp_path, monkeypatch):
    """One batch job scores every selected run under each requested rule and,
    separately, refits the deployed thresholds — one column per rule."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout)
    _seed_train_run_bundle(layout, "run-a")
    _seed_train_run_bundle(layout, "run-b")
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    (layout.dataset_dir / "test.csv").write_text(
        "path,class_name\nx.wav,cheers_loud\n", encoding="utf-8")

    calls = {"calibrate": [], "rule": []}

    def fake_calibrate(cfg, project_id, model_id, *, mode="val_f1", target_precision=0.90):
        calls["calibrate"].append((model_id, mode, target_precision))
        return {"cheers_loud": 0.65}

    def fake_evaluate_under_rule(cfg, project_id, model_id, rule, *, class_mapping=None):
        # The batch job forwards the pre-flight's accepted class mapping, which
        # is what lets runs on an older class space be re-scored at all.
        calls["rule"].append((model_id, rule.id))
        calls.setdefault("mapping", []).append(class_mapping)
        return {
            "f1_macro_targets": 0.8, "ap_macro_targets": 0.85, "accuracy": 0.9,
            "precision_macro_targets": 0.82, "recall_macro_targets": 0.78,
            "n_samples": 12, "thresholds_source": "fixed_threshold:0.900",
            "test_set_hash": "hash-current",
        }

    import sed.train.eval as eval_mod
    monkeypatch.setattr(eval_mod, "calibrate_run_thresholds", fake_calibrate)
    monkeypatch.setattr(eval_mod, "evaluate_run_under_rule", fake_evaluate_under_rule)

    httpd, port = _start_train_api_server(layout)
    base = f"http://127.0.0.1:{port}"
    try:
        started = _post_json(f"{base}/api/train/batch-evaluate/start", {
            "model_ids": ["run-a", "run-b"],
            "calibration": "val_precision",
            "target_precision": 0.85,
            "rules": [{"kind": "fixed", "param": 0.9}, {"kind": "argmax"}],
            "evaluate_exported": False,
        }, timeout=10.0)
        assert started["ok"] is True
        assert [r["id"] for r in started["job"]["options"]["rules"]] == ["fixed@0.90", "argmax"]
        job_id = started["job"]["job_id"]
        final = _wait_for_job(
            f"{base}/api/train/batch-evaluate/status?job_id={job_id}", timeout=30.0)
        job = final["job"]
        assert job["state"] == "finished", job
        assert job["completed"] == 2
        assert job["failed_count"] == 0, [r.get("errors") for r in job["results"]]
        assert [r["model_id"] for r in job["results"]] == ["run-a", "run-b"]
        for res in job["results"]:
            assert res["errors"] == []
            assert res["steps"]["calibrate"]["thresholds"] == {"cheers_loud": 0.65}
            assert res["steps"]["fixed@0.90"]["f1_macro_targets"] == 0.8
            assert res["steps"]["fixed@0.90"]["precision_macro_targets"] == 0.82
            assert res["steps"]["argmax"]["f1_macro_targets"] == 0.8
            assert "exported" not in res["steps"]
        # Both rules reached the eval layer for every model.
        assert calls["calibrate"] == [
            ("run-a", "val_precision", 0.85), ("run-b", "val_precision", 0.85)]
        assert calls["rule"] == [
            ("run-a", "fixed@0.90"), ("run-a", "argmax"),
            ("run-b", "fixed@0.90"), ("run-b", "argmax")]
        # Runs list exposes the comparability context for the UI.
        runs_payload = _get_json(f"{base}/api/train/runs", timeout=10.0)
        assert "current_test_hash" in runs_payload
        assert all("frozen_test" in r for r in runs_payload["runs"])
    finally:
        httpd.shutdown()


def test_batch_evaluate_check_reports_leakage_and_test_health(tmp_path, monkeypatch):
    """The pre-flight must catch, before a job runs, both that the test set
    cannot measure false positives and that a run trained on its clips."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout)
    model_dir = _seed_train_run_bundle(layout, "run-a")

    shared = str(tmp_path / "lib" / "sess_0001.wav")
    header = "path,class_name,session_id\n"
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    # A frozen test made entirely of clips the run trained on, and no negatives.
    (layout.dataset_dir / "test.csv").write_text(
        header + f"{shared},cheers_loud,sess\n", encoding="utf-8")
    (model_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (model_dir / "dataset" / "train.csv").write_text(
        header + f"{shared},cheers_loud,sess\n", encoding="utf-8")
    (model_dir / "checkpoint_best_epoch-001.ckpt").write_bytes(b"stub")

    httpd, port = _start_train_api_server(layout)
    base = f"http://127.0.0.1:{port}"
    try:
        status, body = _post_json_status(
            f"{base}/api/train/batch-evaluate/check", {"model_ids": []})
        assert status == 400

        report = _post_json(
            f"{base}/api/train/batch-evaluate/check", {"model_ids": ["run-a"]})
        assert report["ok"] is True, report
        assert report["all_clear"] is False, "leakage present"
        assert report["n_test_clips"] == 1

        run = report["runs"][0]
        assert run["run_id"] == "run-a"
        assert run["leakage"]["clip_leak_count"] == 1
        assert report["leaked_paths_union"] == [shared]
        leak = next(f for f in run["findings"] if f["code"] == "leakage")
        assert leak["remedy"]["kind"] == "drop_leaked_clips"

        # The test set's own fitness travels with the per-run verdicts.
        codes = {f["code"] for f in report["test_health"]["findings"]}
        assert "no_negatives" in codes
    finally:
        httpd.shutdown()


def test_run_lineage_walks_the_pretrain_to_finetune_chain(tmp_path, monkeypatch):
    """A fine-tune inherits whatever its parent learned, so the chain is part of
    the result — including when the parent lives in another project."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect_server import _run_lineage

    upstream = ProjectLayout("pretrain")
    upstream.create()
    venue = ProjectLayout("venue")
    venue.create()

    def _cfg(layout, run, model):
        d = layout.model_dir(run)
        d.mkdir(parents=True, exist_ok=True)
        (d / "training_config.json").write_text(json.dumps({"model": model}), encoding="utf-8")

    _cfg(upstream, "base", {"backbone": "mn10_as", "initialization": "random",
                            "classes": ["_negative", "cheers"]})
    _cfg(upstream, "domain", {"backbone": "mn10_as", "initialization": "parent",
                              "parent_run": "base", "classes": ["_negative", "cheers"]})
    _cfg(venue, "venue-a", {"backbone": "mn10_as", "initialization": "parent",
                            "parent_project": "pretrain", "parent_run": "domain",
                            "classes": ["_negative", "wine"]})

    chain = _run_lineage(venue, "venue-a")
    assert [(s["project"], s["run"]) for s in chain] == [
        ("pretrain", "domain"), ("pretrain", "base"),
    ]
    assert all(s["exists"] for s in chain)
    assert chain[0]["backbone"] == "mn10_as"

    # A scratch run has no ancestors.
    assert _run_lineage(upstream, "base") == []


def test_run_lineage_reports_a_missing_parent_instead_of_failing(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect_server import _run_lineage

    layout = ProjectLayout("p")
    layout.create()
    d = layout.model_dir("child")
    d.mkdir(parents=True, exist_ok=True)
    (d / "training_config.json").write_text(
        json.dumps({"model": {"parent_run": "deleted-run"}}), encoding="utf-8")

    chain = _run_lineage(layout, "child")
    assert len(chain) == 1
    assert chain[0]["run"] == "deleted-run"
    assert chain[0]["exists"] is False


def test_run_lineage_stops_on_a_cycle(tmp_path, monkeypatch):
    """A self-chained or circular parent link must terminate, not spin."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data.inspect_server import _run_lineage

    layout = ProjectLayout("p")
    layout.create()
    for a, b in (("x", "y"), ("y", "x")):
        d = layout.model_dir(a)
        d.mkdir(parents=True, exist_ok=True)
        (d / "training_config.json").write_text(
            json.dumps({"model": {"parent_run": b}}), encoding="utf-8")

    assert len(_run_lineage(layout, "x")) <= 3


def test_frozen_test_workbench_endpoint_pairs_named_sets_with_health(tmp_path, monkeypatch):
    """The workbench answers one question — can this benchmark support the
    comparison — so named sets and the live set's fitness arrive together."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout)
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    # No negatives: the health block must say false positives are unmeasurable.
    (layout.dataset_dir / "test.csv").write_text(
        "path,class_name,session_id\n/lib/a.wav,cheers_loud,s1\n", encoding="utf-8")

    httpd, port = _start_train_api_server(layout)
    base = f"http://127.0.0.1:{port}"
    try:
        d = _get_json(f"{base}/api/frozen-test-presets")
        assert d["ok"] is True
        assert d["state"]["current_total"] == 1
        assert d["state"]["current_unsaved"] is True, "an unnamed live set must be flagged"
        assert "no_negatives" in {f["code"] for f in d["health"]["findings"]}

        _post_json(f"{base}/api/frozen-test-preset/save", {"name": "v1"})
        d = _get_json(f"{base}/api/frozen-test-presets")
        assert d["state"]["current_unsaved"] is False
        assert d["state"]["presets"][0]["is_current"] is True
    finally:
        httpd.shutdown()


def test_advice_endpoint_and_panel_are_available_in_the_ui(tmp_path, monkeypatch):
    """The workflow lives in the UI, so recommendations must be reachable there
    and not only from the CLI."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout)

    httpd, port = _start_train_api_server(layout)
    try:
        d = _get_json(f"http://127.0.0.1:{port}/api/advice")
        assert d["ok"] is True
        assert isinstance(d["features"], list)
        assert isinstance(d["model"], list)
        # Advisory payload: findings carry a severity and never a "blocked" flag.
        for finding in d["features"] + d["model"]:
            assert finding["severity"] in ("error", "warn", "info")
            assert "blocked" not in finding
    finally:
        httpd.shutdown()

    from sed.data.inspect import _render_html
    html = _render_html(
        project_id="p", class_summary={}, panels={}, interactive=True,
        model_classes=["_negative"], library_items=[],
    )
    for marker in ("train-advice-body", "loadTrainAdvice", "/api/advice"):
        assert marker in html, marker


def test_frozen_tab_renders_review_controls_and_model_comparison():
    """Reviewing WHY a clip was wrong is the point of this tab: it needs the
    verdict on the card, failure-specific filters, and a second model to compare
    against — not just a correct/incorrect pill."""
    from sed.data.inspect import _render_html

    card = {
        "path": "/x/a.wav", "name": "a.wav", "true_class": "_negative",
        "predicted_class": "cheers", "correct": False, "triggered": True,
        "predicted_confidence": 0.93, "argmax_confidence": 0.93, "margin": 0.5,
        "has_prediction": True, "frozen_source": "", "frozen_source_label": "",
        "decision_reason": "threshold", "meta_html": "", "prob_html": "",
        "search_text": "a", "spec_b64": "", "probs_percent": {},
    }
    html = _render_html(
        project_id="p", class_summary={}, panels={}, interactive=True,
        model_classes=["_negative", "cheers"], library_items=[],
        frozen_panels={"_negative": [card]},
    )

    for marker in (
        # The verdict on the card: what it is, what the model said, how sure.
        "frozen-verdict", "fv-true", "fv-pred", "fv-conf",
        # A false positive and a miss are opposite failures and get their own filters.
        'data-frozen="false-positive"', 'data-frozen="missed"',
        # Two models on one clip, plus a filter for where they disagree.
        "frozen-compare-select", "loadFrozenCompareRun", "FROZEN_COMPARE",
        'data-frozen="compare-differs"', "fv-compare",
        # Controls that must not regress again.
        "frozen-run-select", "frozen-search", "frozen-metric",
        "frozen-min", "frozen-max",
        "frozen-hide-run-train", 'data-frozen="run-train"',
        "Evaluation result", "Explain predictions",
        "/api/evaluate-frozen/status", "frozenElapsedLabel",
        "current.current_class", "current.current_name", 'frozen_eval: "#fb7185"',
    ):
        assert marker in html, marker
    assert "up to ~1 min" not in html
    assert "analyzing frozen 0/1" not in html


def test_frozen_verdict_and_failure_filters_are_defined_once():
    """The card renders from two paths — Jinja for small classes, JS for classes
    above the pre-render threshold. Both must show the verdict, or a big class
    silently loses it."""
    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p", class_summary={}, panels={}, interactive=True,
        model_classes=["_negative", "cheers"], library_items=[],
    )
    assert "function frozenVerdictHtml" in html
    assert "function frozenIsFalsePositive" in html
    assert "function frozenIsMiss" in html


def test_train_tab_renders_the_frozen_test_workbench():
    """The P1 endpoints are unreachable without these controls."""
    from sed.data.inspect import _render_html

    html = _render_html(
        project_id="p", class_summary={}, panels={}, interactive=True,
        model_classes=["_negative", "glass"], library_items=[],
    )

    for marker in (
        # Workbench: name / list / apply / delete a benchmark.
        "train-testset-list", "train-testset-health", "train-testset-save",
        "loadFrozenTestSets", "/api/frozen-test-presets",
        # Pre-flight: verdicts, leakage, and the additive remedy.
        "train-batch-check", "checkTrainBatchCompat", "deriveCleanFrozenTest",
        "/api/train/batch-evaluate/check", "/api/frozen-test-preset/derive",
        # Runs-table columns added for replication and workflow stage.
        "seedGroupSize", "runKind", "seedSpread",
        # Benchmark provenance + the pretrain -> fine-tune chain. The table
        # reports on ONE benchmark, picked in the operating-point bar, so the
        # provenance lives there rather than in a per-row "test set" column.
        "train-op-testset", "activeTestSet", "TEST_SET_NAMES",
        "renderRunLineage", "train-detail-lineage",
    ):
        assert marker in html, marker


def test_frozen_test_preset_derive_is_additive_and_apply_is_explicit(tmp_path, monkeypatch):
    """The leakage remedy must never silently redefine the live benchmark:
    `derive` writes a new named set, `apply` is a separate deliberate step."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout)
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    (layout.dataset_dir / "test.csv").write_text(
        "path,class_name,session_id\n"
        "/lib/leaked.wav,cheers_loud,s1\n"
        "/lib/clean.wav,cheers_loud,s2\n"
        "/lib/n.wav,_negative,s3\n",
        encoding="utf-8",
    )

    httpd, port = _start_train_api_server(layout)
    base = f"http://127.0.0.1:{port}"
    try:
        saved = _post_json(f"{base}/api/frozen-test-preset/save", {"name": "v1"})
        assert saved["ok"] is True
        assert saved["preset"]["n_clips"] == 3

        derived = _post_json(f"{base}/api/frozen-test-preset/derive", {
            "name": "v1 clean", "excluded_paths": ["/lib/leaked.wav"],
        })
        assert derived["ok"] is True
        assert derived["preset"]["n_clips"] == 2

        # The live test set is untouched until apply.
        live = (layout.dataset_dir / "test.csv").read_text()
        assert "/lib/leaked.wav" in live

        applied = _post_json(f"{base}/api/frozen-test-preset/apply", {"name": "v1-clean"})
        assert applied["ok"] is True
        live = (layout.dataset_dir / "test.csv").read_text()
        assert "/lib/leaked.wav" not in live
        assert "/lib/clean.wav" in live

        # Removing everything is refused rather than writing an empty benchmark.
        status, body = _post_json_status(f"{base}/api/frozen-test-preset/derive", {
            "name": "nothing", "excluded_paths": ["/lib/clean.wav", "/lib/n.wav"],
        })
        assert status == 400
        assert "empty the test set" in body["error"]

        status, _body = _post_json_status(
            f"{base}/api/frozen-test-preset/apply", {"name": "missing"})
        assert status == 404
    finally:
        httpd.shutdown()


def test_batch_evaluate_start_validates_models_and_options(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout)
    _seed_train_run_bundle(layout, "run-a")
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    (layout.dataset_dir / "test.csv").write_text(
        "path,class_name\nx.wav,cheers_loud\n", encoding="utf-8")

    httpd, port = _start_train_api_server(layout)
    base = f"http://127.0.0.1:{port}"
    try:
        status, body = _post_json_status(
            f"{base}/api/train/batch-evaluate/start", {"model_ids": []})
        assert status == 400
        status, body = _post_json_status(
            f"{base}/api/train/batch-evaluate/start", {"model_ids": ["nope"]})
        assert status == 404
        assert "nope" in body["error"]
        status, body = _post_json_status(
            f"{base}/api/train/batch-evaluate/start",
            {"model_ids": ["run-a"], "calibration": "venue_fp"})
        assert status == 400
        # An out-of-range rule parameter is rejected.
        status, body = _post_json_status(
            f"{base}/api/train/batch-evaluate/start",
            {"model_ids": ["run-a"], "rules": [{"kind": "fixed", "param": 1.5}]})
        assert status == 400
        # An unknown rule kind is rejected.
        status, body = _post_json_status(
            f"{base}/api/train/batch-evaluate/start",
            {"model_ids": ["run-a"], "rules": [{"kind": "coin_flip"}]})
        assert status == 400
        # Nothing selected to do at all.
        status, body = _post_json_status(
            f"{base}/api/train/batch-evaluate/start",
            {"model_ids": ["run-a"], "rules": [],
             "evaluate_exported": False, "calibration": "none"})
        assert status == 400
        # No job was ever started by the rejected requests.
        r = _get_json_status(f"{base}/api/train/batch-evaluate/status")
        assert r[0] == 404
    finally:
        httpd.shutdown()


def _get_json_status(url: str, timeout: float = _HTTP_TIMEOUT) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_train_runs_list_exposes_frozen_test_rules_and_deployed(tmp_path, monkeypatch):
    """The runs list carries what the three-tier table shows: frozen-test scores
    keyed by rule (legacy blocks folded in), validation, deployed-IR scores,
    calibration provenance and model size — so the UI never re-derives it."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    model_dir = _seed_train_run_bundle(
        layout, "run-bench", model_stats={"total_params": 4934861})
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    # A second rule scored via the modern path, plus a legacy benchmark block
    # the shim must surface as fixed@0.95 without any re-evaluation.
    manifest["frozen_test"]["argmax"] = {
        "rule": {"id": "argmax", "kind": "argmax", "param": None, "label": "argmax"},
        "f1_macro_targets": 0.93, "ap_macro_targets": 0.97, "accuracy": 0.9,
        "n_samples": 197, "test_set_hash": "hash-current",
    }
    # A pre-rule benchmark block: no macro P/R and no false-fire rate, exactly
    # like the manifests written before those keys existed. Both must be
    # recovered from per_class rather than left blank.
    manifest["benchmark_metrics"] = {
        "f1_macro_targets": 0.88, "ap_macro_targets": 0.96, "accuracy": 0.86,
        "threshold": 0.95, "n_samples": 197, "test_set_hash": "hash-current",
        "per_class": {
            "cheers": {"precision": 0.8, "recall": 0.75, "f1": 0.77,
                       "tp": 30, "fp": 10, "fn": 10, "tn": 147, "support": 40},
            "_negative": {"precision": 0.9, "recall": 0.9, "f1": 0.9,
                          "tp": 90, "fp": 10, "fn": 10, "tn": 87, "support": 100},
        },
    }
    manifest["test_metrics_exported"] = {
        "f1_macro_targets": 0.89, "ap_macro_targets": 0.96, "accuracy": 0.87,
        "model_file": "model_int8.xml", "test_set_hash": "hash-current",
        "per_class": {"cheers": {"f1": 0.89}},
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (model_dir / "runtime_settings.json").write_text(json.dumps({
        "thresholds": {"cheers": 0.65},
        "calibration": {"mode": "val_precision", "target_precision": 0.9},
    }), encoding="utf-8")

    httpd, port = _start_train_api_server(layout)
    try:
        payload = _get_json(f"http://127.0.0.1:{port}/api/train/runs")
        assert payload["ok"] is True
        row = {r["model_id"]: r for r in payload["runs"]}["run-bench"]
        ft = row["frozen_test"]
        assert ft["fixed@0.90"]["f1"] == 0.9
        assert ft["argmax"]["f1"] == 0.93
        assert ft["fixed@0.95"]["f1"] == 0.88   # legacy benchmark surfaced by the shim
        assert ft["fixed@0.95"]["origin"] == "legacy_benchmark"
        assert row["val_ap"] == 0.82
        assert row["val_accuracy"] == 0.9
        assert row["calibration_mode"] == "val_precision"
        assert row["calibration_target_precision"] == 0.9
        assert row["total_params"] == 4934861
        # Deployed IR scores are keyed "<precision>:<rule id>", so FP16 and
        # INT8 accumulate instead of overwriting; the pre-section flat block is
        # bridged in under the precision its model_file names.
        dep = row["deployed_test"]["int8:deployed"]
        assert dep["f1"] == 0.89
        assert dep["ir_precision"] == "int8"
        assert dep["origin"] == "legacy_exported"
        assert row["deployed_model_file"] == "model_int8.xml"
        # Macro P/R and the false-fire rate are derived from per_class for
        # blocks written before those keys existed — no re-evaluation needed.
        assert ft["fixed@0.95"]["precision"] == pytest.approx(0.8)
        assert ft["fixed@0.95"]["recall"] == pytest.approx(0.75)
        assert ft["fixed@0.95"]["false_fire_rate"] == pytest.approx(0.1)
        assert row["has_openvino"] is False   # no IR files on disk in this seed
    finally:
        httpd.shutdown()


def test_train_run_detail_exposes_frozen_test_rule_blocks(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    model_dir = _seed_train_run_bundle(layout, "run-detail")
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["frozen_test"]["fixed@0.90"].update({
        "per_class": {"cheers": {"f1": 0.91, "ap": 0.97}},
        "thresholds_source": "fixed_threshold:0.900", "thresholds": {"cheers": 0.9},
    })
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    httpd, port = _start_train_api_server(layout)
    try:
        payload = _get_json(f"http://127.0.0.1:{port}/api/train/run?model_id=run-detail")
        assert payload["ok"] is True
        block = payload["frozen_test"]["fixed@0.90"]
        assert block["rule"]["label"] == "≥ 0.90"
        assert block["threshold"] == 0.9
        assert block["f1_macro_targets"] == 0.9
        assert block["per_class"]["cheers"]["f1"] == 0.91
        # The rule's confusion matrix is addressable via the asset key.
        assert block["confusion_key"] == "confusion_matrix:fixed_0_90"
        assert "current_test_hash" in payload
        assert payload["exported"] is False
        assert payload["available_precisions"] == []
    finally:
        httpd.shutdown()


def test_batch_evaluate_export_missing_runs_as_sole_action(tmp_path, monkeypatch):
    """`export missing IR` must be accepted alone (no frozen test needed) and
    its per-model result recorded — this is the run-detail Export shortcut."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout)
    _seed_train_run_bundle(layout, "run-a")

    exported = []

    def fake_export(cfg, project_id, model_id, **kwargs):
        exported.append(model_id)
        return {"active_precision": "fp16", "int8": False}

    import sed.data.inspect_server as srv
    monkeypatch.setattr(srv, "_export_run_ir", fake_export)

    httpd, port = _start_train_api_server(layout)
    base = f"http://127.0.0.1:{port}"
    try:
        started = _post_json(f"{base}/api/train/batch-evaluate/start", {
            "model_ids": ["run-a"],
            "export_missing": True,
            "evaluate_checkpoint": False,
            "evaluate_exported": False,
            "benchmark": False,
            "calibration": "none",
        }, timeout=10.0)
        assert started["ok"] is True
        assert started["job"]["options"]["export_missing"] is True
        job_id = started["job"]["job_id"]
        final = _wait_for_job(
            f"{base}/api/train/batch-evaluate/status?job_id={job_id}", timeout=30.0)
        job = final["job"]
        assert job["state"] == "finished", job
        assert exported == ["run-a"]
        res = job["results"][0]
        assert res["errors"] == []
        assert res["steps"]["export"] == {"active_precision": "fp16", "int8": False}
    finally:
        httpd.shutdown()


def test_export_run_ir_skips_bundles_that_already_have_an_ir(tmp_path, monkeypatch):
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    model_dir = _seed_train_run_bundle(layout, "run-a")
    (model_dir / "model_fp16.xml").write_text("<ir/>", encoding="utf-8")

    from sed.config import Config
    from sed.data.inspect_server import _export_run_ir

    result = _export_run_ir(Config(), "p", "run-a")
    assert result == {"skipped": "IR already exported"}


def test_batch_evaluate_reports_an_import_failure_on_the_request(tmp_path, monkeypatch):
    """The evaluation imports must resolve in the request thread.

    They used to run in the worker thread, ahead of its try block, so a broken
    import killed the thread silently: the job stayed "starting" (0/N) until
    the server was restarted, and every later batch was refused with "already
    running". Now it is a plain error on the start request."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout)
    _seed_train_run_bundle(layout, "run-a")

    import sed.train.eval as eval_mod
    # None in sys.modules is how CPython signals "this import is halted".
    monkeypatch.setitem(sys.modules, "sed.train.eval", None)

    httpd, port = _start_train_api_server(layout)
    base = f"http://127.0.0.1:{port}"
    body = {
        "model_ids": ["run-a"],
        "export_missing": True,
        "evaluate_checkpoint": False,
        "evaluate_exported": False,
        "benchmark": False,
        "calibration": "none",
    }
    try:
        status, payload = _post_json_status(
            f"{base}/api/train/batch-evaluate/start", body, timeout=10.0)
        assert status == 500, (status, payload)
        assert payload["ok"] is False
        assert "sed.train.eval" in payload["error"], payload["error"]
        # No half-started job may be left behind to block the next attempt.
        with pytest.raises(urllib.error.HTTPError) as no_job:
            _get_json(f"{base}/api/train/batch-evaluate/status", timeout=10.0)
        assert no_job.value.code == 404

        # …and the feature is not bricked: a later batch still starts.
        monkeypatch.setitem(sys.modules, "sed.train.eval", eval_mod)
        again = _post_json(f"{base}/api/train/batch-evaluate/start", body, timeout=10.0)
        assert again["ok"] is True
        _wait_for_job(
            f"{base}/api/train/batch-evaluate/status?job_id={again['job']['job_id']}",
            timeout=30.0)
    finally:
        httpd.shutdown()


def test_batch_evaluate_status_fails_a_job_whose_worker_thread_is_gone(tmp_path, monkeypatch):
    """Belt and braces for anything that stops the worker without reporting:
    the status/start handlers must notice the dead thread instead of polling
    0/N forever and refusing every later batch."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout)
    _seed_train_run_bundle(layout, "run-a")

    real_thread_cls = threading.Thread

    class _NeverStartsJobThread(real_thread_cls):
        """Drops the batch worker on the floor; other threads start normally."""

        def start(self):  # noqa: D102
            if getattr(getattr(self, "_target", None), "__name__", "") == "run_job":
                return
            super().start()

    monkeypatch.setattr(threading, "Thread", _NeverStartsJobThread)

    httpd, port = _start_train_api_server(layout)
    base = f"http://127.0.0.1:{port}"
    body = {
        "model_ids": ["run-a"],
        "export_missing": True,
        "evaluate_checkpoint": False,
        "evaluate_exported": False,
        "benchmark": False,
        "calibration": "none",
    }
    try:
        started = _post_json(f"{base}/api/train/batch-evaluate/start", body, timeout=10.0)
        job_id = started["job"]["job_id"]
        status = _get_json(
            f"{base}/api/train/batch-evaluate/status?job_id={job_id}", timeout=10.0)
        assert status["job"]["state"] == "error", status
        assert "stopped unexpectedly" in status["job"]["error"]

        monkeypatch.setattr(threading, "Thread", real_thread_cls)
        again = _post_json(f"{base}/api/train/batch-evaluate/start", body, timeout=10.0)
        assert again["ok"] is True
        _wait_for_job(
            f"{base}/api/train/batch-evaluate/status?job_id={again['job']['job_id']}",
            timeout=30.0)
    finally:
        httpd.shutdown()


def test_runs_tab_contains_batch_evaluation_controls():
    html_source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    for marker in [
        "train-runs-batch",
        "train-batch-calibration",
        "train-batch-target-precision",
        # The checkpoint's per-rule checkboxes are gone: one scoring pass stores
        # the per-clip probability matrix, and every threshold, metric and
        # confusion matrix in Runs & compare is re-cut from it. So the panel
        # asks which ARTIFACT to run over the clips, not which decision rule.
        "train-batch-score-checkpoint",
        "train-batch-prec-fp16",
        "train-batch-prec-int8",
        "train-batch-exported-rule",
        "train-batch-export-missing",
        "train-batch-run",
        "train-batch-cancel",
        "train-batch-plan",
        "train-runs-columns-menu",
        "train-runs-thead",
        "train-detail-export",
        "train-detail-conditions",
        "train-detail-rule-tabs",
        "train-detail-ir-tabs",
        "train-detail-ir-rule",
        "train-detail-exported-conditions",
        "/api/train/batch-evaluate/start",
        "/api/train/batch-evaluate/status",
        "/api/train/batch-evaluate/cancel",
        "What to score",
        "train-compare-warning",
        "train-runs-staleness",
    ]:
        assert marker in html_source


def test_rule_wording_has_one_source():
    """Rule labels come from the server's `rule_catalog()` and nowhere else.

    Before this, the batch checkbox said "confidence ≥", the column header said
    "≥ 0.90" and the metric guide said "≥ 0.90 (fixed)" — three names for one
    rule, so you could not tell which column a checkbox filled. The guard is
    that no client-side label table may reappear.
    """
    source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    for banned in ["const RULE_NAME", "const DETAIL_RULE_KINDS", "const RULE_HOW"]:
        assert banned not in source, (
            f"{banned} is a second source of rule wording — render from "
            f"TRAIN_RULE_CATALOG instead"
        )
    # …and the catalog is actually what the pickers and headers read.
    for reader in ["TRAIN_RULE_CATALOG", "ruleLabelFor(", "setRuleCatalog("]:
        assert reader in source

    from sed.train.test_rules import RULE_KINDS, TestRule, rule_catalog

    catalog = rule_catalog()
    assert {c["kind"] for c in catalog} == set(RULE_KINDS)
    for entry in catalog:
        # label_template is what the browser renders; it must agree with the
        # label TestRule produces server-side for the same parameter.
        rendered = entry["label_template"].replace(
            "{param}", "" if entry["param"] is None else f"{entry['param']:.2f}")
        assert rendered == TestRule(entry["kind"]).label == entry["label"]
        assert entry["how"] and isinstance(entry["benchmark"], bool)


def test_library_negatives_count_endpoint(tmp_path, monkeypatch):
    """Lazy NEG-source counts: venue from the dataset snapshot, library cached."""
    _patch_project_and_library_dirs(tmp_path, monkeypatch)
    from sed.data import inspect_server as srv_mod

    monkeypatch.setattr(srv_mod, "_LIBRARY_WAV_COUNT", None)
    layout = ProjectLayout("p")
    layout.create()
    _write_inference_project_config(layout, active="")
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    (layout.dataset_dir / "dataset_snapshot.json").write_text(
        json.dumps({"by_class": {"_negative": 3, "cheers_loud": 5}}), encoding="utf-8"
    )
    sounds = tmp_path / "data_library" / "sounds" / "room"
    sounds.mkdir(parents=True)
    (sounds / "a.wav").write_bytes(b"x")
    (sounds / "b.wav").write_bytes(b"x")

    from sed.config import Config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    port = _free_port()
    handler = make_handler(cfg=Config(), project_id="p",
                           render_html_callable=lambda: "<html>ok</html>", layout=layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        payload = _get_json(f"http://127.0.0.1:{port}/api/library/negatives-count", timeout=10.0)
        assert payload == {"ok": True, "venue": 3, "library": 2}
    finally:
        httpd.shutdown()


def test_dataset_preset_endpoints_roundtrip(tmp_path, monkeypatch):
    """Save / apply / delete dataset source-rule presets + selection meta."""
    from sed import config as cfg_mod
    from sed.data import library as lib_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()

    from sed.config import Config
    cfg = Config()
    cfg.model.classes = ["_negative", "cheers_loud"]

    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer
    port = _free_port()
    handler = make_handler(cfg=cfg, project_id="p",
                           render_html_callable=lambda: "<html>ok</html>",
                           layout=layout)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        time.sleep(0.05)
        added = _post_json(f"{base}/api/selection-rule", {
            "action": "add",
            "class_name": "cheers_loud",
            "rule": {"labels": ["cheers_loud"], "split": "train"},
        })
        assert added["ok"] is True
        # Rules exist, nothing composed -> the recompose flag is on.
        assert added["compose_state"]["stale"] is True
        assert added["preset_state"]["active"] == ""
        # The dataset definition owns the class space: assigning to a class the
        # project had not listed adds it rather than being rejected.
        assert added["target_classes"] == ["_negative", "cheers_loud"]

        saved = _post_json(f"{base}/api/dataset-preset/save", {"name": "dataset 01"})
        assert saved["ok"] is True
        # A preset is the whole dataset definition, so it carries the class list
        # alongside the source rules.
        assert saved["preset"] == {
            "name": "dataset 01", "slug": "dataset-01", "rule_count": 1, "clip_count": 0,
            "classes": ["_negative", "cheers_loud"],
        }
        assert saved["preset_state"]["active"] == "dataset-01"
        assert saved["preset_state"]["modified"] is False
        entry = next(p for p in saved["preset_state"]["presets"] if p["slug"] == "dataset-01")
        assert entry["has_classes"] is True
        assert entry["classes_match_current"] is True

        cleared = _post_json(f"{base}/api/selection-rule", {"action": "clear"})
        assert cleared["ok"] is True
        assert cleared["preset_state"]["modified"] is True

        applied = _post_json(f"{base}/api/dataset-preset/apply", {"name": "dataset-01"})
        assert applied["ok"] is True
        rules = applied["selection"]["assignments"]["cheers_loud"]
        assert rules[0]["labels"] == ["cheers_loud"]
        assert applied["preset_state"]["modified"] is False

        listing = _get_json(f"{base}/api/selection")
        assert listing["preset_state"]["active"] == "dataset-01"
        assert [p["slug"] for p in listing["preset_state"]["presets"]] == ["dataset-01"]
        assert listing["compose_state"]["stale"] is True

        status, missing = _post_json_status(
            f"{base}/api/dataset-preset/apply", {"name": "nope"})
        assert status == 404
        assert missing["ok"] is False

        deleted = _post_json(f"{base}/api/dataset-preset/delete", {"name": "dataset 01"})
        assert deleted["ok"] is True
        assert deleted["preset_state"]["presets"] == []
        assert deleted["preset_state"]["active"] == ""

        # "New blank dataset": rules clear AND the class list resets to the
        # minimal ["_negative"] space — before this fix, classes were left
        # stranded at whatever the last-loaded definition had, so the class
        # chips and assign dropdown kept showing a dataset with no rules left.
        blanked = _post_json(f"{base}/api/dataset-preset/new", {})
        assert blanked["ok"] is True
        assert blanked["selection"]["assignments"] == {}
        assert blanked["preset"]["classes"] == ["_negative"]

        from sed.config import load_config
        assert load_config("p").dataset.target_classes == ["_negative"]
    finally:
        httpd.shutdown()


def test_render_html_includes_dataset_preset_bar_and_recompose_flag():
    html_source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    for marker in [
        "dataset-preset-select",
        "dataset-preset-save",
        "dataset-preset-saveas",
        "dataset-preset-delete",
        "recompose-flag",
        "/api/dataset-preset/apply",
        "/api/dataset-preset/save",
        "/api/dataset-preset/delete",
    ]:
        assert marker in html_source


def test_library_tab_unifies_presets_classes_and_source_rules():
    """The Library tab's dataset definition is one panel: preset controls, the
    target-class list, and the source rules that connect them."""
    html_source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")

    # Preset + rules + class stats live in one full-width panel at the top.
    for marker in [
        'class="dataset-def-panel"',
        'id="dataset-def-toggle"',
        'id="dataset-def-stats"',
        'id="dataset-def-body"',
        "renderDatasetDefStats",
    ]:
        assert marker in html_source

    # A preset restores the class list too, so the UI names the classes in the
    # dropdown and pushes them into the live class controls on load.
    assert "no classes" in html_source
    assert "applyTargetClasses(presetClasses)" in html_source

    # Target classes are a first-class section next to the assign controls.
    for marker in ['class="library-section class-workbench"', "tcRenderChips", 'class="assign-bar"']:
        assert marker in html_source

    # The corpus-subbucket side panel is gone; its granularity moved into the
    # Library table itself.
    assert "corpus-subbuckets-panel" not in html_source
    assert "corpus_subbuckets_panel" not in html_source
    assert "pretraining_subbuckets" in html_source
    assert 'id="library-corpus-rescan"' in html_source

    # New Library facets: sample rate, duration band, and a tag cloud.
    for marker in [
        'id="library-rate-filter"',
        'id="library-length-filter"',
        'id="library-tag-cloud"',
        "LIBRARY_LENGTH_BANDS",
        "durationInBand",
        "renderLibraryTagCloud",
    ]:
        assert marker in html_source


def test_dataset_ui_buttons_share_one_style_system():
    """Every button in the dataset-building UI comes from one shared rule set.

    Placement decides size (section header / inline toolbar => small, main
    action row => regular), so two buttons sitting side by side always match.
    Regression guard: "Rescan from disk" and "corpus headers" used to sit
    together in the same header at different sizes, weights and colours because
    each carried its own one-off rule.
    """
    html_source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    # The button system moved to the shared UI shell when the recording console
    # started rendering it too. It is still one system; it just serves two pages.
    button_css = Path("src/sed/data/ui_theme.py").read_text(encoding="utf-8")

    # The shared base declaration names every family it covers.
    base = button_css.split("/* ==== One button system", 1)[1].split("}", 1)[0]
    for family in [
        ".section-action", "button.section-refresh", ".recipe-actions button",
        ".assign-bar button", ".tc-add-row button", ".preset-bar button",
        ".bucket-picker button", ".library-actions button", ".library-row-delete",
        ".compose-run",
        ".btn",   # the UI-neutral entry point a second front-end uses
    ]:
        assert family in base, family

    # Size follows placement, not family.
    assert ".section-head button.section-refresh" in button_css

    # The one-off rules these replaced are gone from both files.
    for dead in [
        ".assign-bar button { border: 1px solid #cbd5e1;",
        ".preset-bar button { border: 1px solid #cbd5e1;",
        ".library-row-delete { border: 1px solid #fecaca;",
        ".bucket-picker button { padding:",
        ".tc-add-row button { padding:",
        "button.section-refresh.rescan-chip { min-height: 0;",
    ]:
        assert dead not in html_source, dead
        assert dead not in button_css, dead

    # Both header rescan buttons render the same icon+label structure.
    header = html_source.split('id="library-corpus-rescan"', 1)[1].split("</button>", 1)[0]
    assert 'class="refresh-icon"' in header
    # Icon-only removes are one square, defined once for both families.
    assert ".dd-x, .lib-row-x {" in button_css


def test_compose_bar_is_one_write_action_plus_read_only_actualize_chips():
    """Compose writes; the chips only re-read. Compose already re-reads every
    view it can change, so there is no separate 'compose and refresh' button."""
    html_source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")

    # One row, one write action.
    assert 'class="compose-bar"' in html_source
    assert 'class="compose-run" id="compose-dataset-btn"' in html_source

    # Every review scope is reachable as a chip, plus an all-in-one.
    for scopes in ["library,recipes", "dataset", "augmented", "frozen", "mined", "all"]:
        assert f'data-refresh-scopes="{scopes}"' in html_source

    # Compose refreshes the views its CSVs can change, in one pass.
    compose = html_source.split("async function composeDataset(", 1)[1].split("\n}", 1)[0]
    assert 'rescanScopes("library,dataset,augmented,frozen,recipes"' in compose

    # Staleness is one signal painted in two places (top-bar pill + the button).
    flag = html_source.split("function updateRecomposeFlag() {", 1)[1].split("\n}\n", 1)[0]
    assert 'btn.classList.toggle("stale", stale)' in flag
    assert 'compose-stale-hint' in flag
    assert ".compose-run.stale" in html_source

    # The button holds an icon span, so only its label may be swapped while busy.
    assert 'btn.textContent = "Composing..."' not in html_source
    assert 'label.textContent = "Composing…"' in html_source


def test_rule_identity_ignores_split_so_frozen_test_buckets_are_marked():
    """`split` is a property of the assignment, not of the bucket.

    Keying rule identity on it meant a Library row (always built with
    split=train) never matched a frozen-test assignment, so the red frozen-test
    dot and the train/test leakage warning could not fire.
    """
    html_source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    comparable = html_source.split("function ruleComparable(rule) {", 1)[1].split("\n}", 1)[0]
    assert "split:" not in comparable
    assert "sample_rate:" in comparable   # corpus rules pin one subbucket
    # Both markers are still rendered off the matched assignments.
    assert "assigned-leakage" in html_source
    assert "assigned-test-only" in html_source


def test_source_rule_edit_preserves_filters_it_has_no_control_for():
    """Changing a rule's target class must not strip its label/tag/bucket
    filters — an earlier editor deleted every field whose control was absent,
    silently turning a narrow rule into a match-everything rule."""
    html_source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    body = html_source.split("function sourceRuleFromEditor(editor, original) {", 1)[1].split("\n}", 1)[0]
    for key in ["labels", "tags_all", "project_ids", "source_bucket", "channel_index"]:
        # every optional field is now guarded by a control-exists check
        assert f'control("{key}")' in body or 'for (const key of' in body
    assert 'if (!control(key)) continue;' in body
    assert 'if (control("project_ids"))' in body
    assert 'if (control("channel_index"))' in body


def test_dataset_definition_legend_filters_the_rule_table(tmp_path):
    """The four legend keys are the table's row filter, not decoration: a rule
    can carry several kinds at once, so the filter is a union over a set."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered inspect JavaScript")
    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    for kind in ("train", "test", "derived", "leak"):
        assert f'data-dd-filter="{kind}"' in html

    script = "\n".join(part.split("</script>", 1)[0] for part in html.split("<script>")[1:])
    start = script.index("const DD_KIND_LABELS")
    end = script.index("function renderDatasetDefLegend", start)
    filter_js = script[start:end]
    harness = f"""
function ruleComparable(rule) {{ return rule.bucket; }}
function assert(ok, message) {{ if (!ok) throw new Error(message); }}
{filter_js}

const splits = new Map([["a", new Set(["train", "test"])], ["b", new Set(["train"])]]);
const leaking = {{ bucket: "a", split: "train" }};
const plain = {{ bucket: "b", split: "train" }};
const frozen = {{ bucket: "b", split: "test" }};
const augmented = {{ bucket: "b", split: "train", source_bucket: "derived" }};

assert([...sourceRuleKinds(leaking, splits)].sort().join() === "leak,train",
       "a bucket in both pools is training AND leakage");
assert([...sourceRuleKinds(frozen, splits)].join() === "test");
assert([...sourceRuleKinds(augmented, splits)].sort().join() === "derived,train",
       "an augmented bucket is still in whichever pool it feeds");

// No selection shows everything.
assert(sourceRuleMatchesKindFilter(sourceRuleKinds(plain, splits)));
// One selection narrows to that kind.
datasetDefKindFilter = new Set(["leak"]);
assert(sourceRuleMatchesKindFilter(sourceRuleKinds(leaking, splits)));
assert(!sourceRuleMatchesKindFilter(sourceRuleKinds(plain, splits)));
// Several selections show the union, not the intersection.
datasetDefKindFilter = new Set(["test", "derived"]);
assert(sourceRuleMatchesKindFilter(sourceRuleKinds(frozen, splits)));
assert(sourceRuleMatchesKindFilter(sourceRuleKinds(augmented, splits)));
assert(!sourceRuleMatchesKindFilter(sourceRuleKinds(leaking, splits)));
"""
    script_path = tmp_path / "dataset-def-legend-filter.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, str(script_path)], check=True)


def test_dataset_definition_renders_pool_first_class_groups_and_identity_tokens(tmp_path):
    """Exercise the rendered Dataset Definition grouping in JavaScript.

    The source object deliberately starts in the opposite order.  The renderer
    must follow configured target-class order inside Training, then place all
    Frozen rows after it; colour identity stays separate from the pool names.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise rendered inspect JavaScript")
    from sed.data.inspect import _render_html

    html = _render_html(project_id="p", class_summary={}, panels={}, interactive=True)
    script = "\n".join(part.split("</script>", 1)[0] for part in html.split("<script>")[1:])
    palette_start = script.index("const CLASS_TINT_PALETTE")
    palette_end = script.index("function micChannels", palette_start)
    palette_js = script[palette_start:palette_end]
    grouping_start = script.index("const DD_KIND_LABELS")
    grouping_end = script.index("// The always-visible summary strip", grouping_start)
    grouping_js = script[grouping_start:grouping_end]
    harness = f"""
let MODEL_CLASSES = ["_negative", "attention", "cheers"];
let selectionState = {{ assignments: {{
  _negative: [
    {{ id: "n-train", labels: ["ambience"], split: "train", count: 24, bucket: "n-train" }},
  ],
  cheers: [
    {{ id: "c-train", labels: ["cheers"], split: "train", count: 12, bucket: "c-train" }},
    {{ id: "c-test", labels: ["cheers"], split: "test", count: 4, bucket: "c-test" }},
  ],
  attention: [
    {{ id: "a-train", labels: ["attention"], split: "train", count: 9, bucket: "a-train" }},
    {{ id: "a-leak", labels: ["attention"], split: "test", count: 3, bucket: "a-train" }},
  ],
}} }};
let selectionLoaded = true, selectionLoadError = "", sourceRuleQuery = "";
const table = {{ innerHTML: "", querySelectorAll() {{ return []; }} }};
const clear = {{ disabled: false }};
const document = {{ getElementById(id) {{ return {{
  "library-rule-table-body": table, "source-rules-clear": clear,
}}[id] || null; }}, querySelectorAll() {{ return []; }} }};
function escapeHtml(v) {{ return String(v || ""); }}
function ruleComparable(rule) {{ return rule.bucket || rule.id; }}
function sourceRuleHaystack(cls, rule) {{ return `${{cls}} ${{rule.bucket || ""}}`; }}
function sourceRuleAudioLabel(rule) {{ return (rule.labels || []).join(","); }}
function sourceRuleEditorHtml() {{
  return {{ target: "<select></select>", split: "<span>pool</span>" }};
}}
function ruleMicValue() {{ return ""; }}
function ruleSplitChannelValue() {{ return ""; }}
function micLabelFromValue(v) {{ return v; }}
function setSourceRulesStatus() {{}}
function tcRenderChips() {{}}
function renderDatasetDefLegend() {{}}
function renderDatasetDefStats() {{}}
function renderLibraryTable() {{}}
function updateMutatingBusy() {{}}
function removeSelectionRule() {{}}
function updateSelectionRule() {{}}
function assert(ok, message) {{ if (!ok) throw new Error(message); }}
{palette_js}
{grouping_js}
renderDatasetDefLegend = () => {{}};
renderDatasetDefStats = () => {{}};
renderSelectionRules();
const rendered = table.innerHTML;
assert(rendered.indexOf("Training pool") < rendered.indexOf("Frozen test"),
       "training pool must precede frozen test");
assert(rendered.indexOf("<strong>attention</strong>") < rendered.indexOf("<strong>cheers</strong>"),
       "configured class order must win over source object order");
assert(rendered.includes("dd-leak") && rendered.includes(">leak<"),
       "leakage remains an explicit warning treatment");
assert(rendered.includes('data-class-token="slate"'),
       "negative rows render with the neutral class token");

datasetDefClassFilter = new Set(["attention"]);
renderSelectionRules();
assert(table.innerHTML.includes("<strong>attention</strong>"));
assert(!table.innerHTML.includes("<strong>cheers</strong>"),
       "class filter narrows grouped sections");

const tokens = resolveClassVisuals({{}}, ["_negative", "attention", "cheers"]);
assert(tokens._negative === "slate", "negative stays neutral");
assert(tokens.attention !== tokens.cheers, "adjacent target classes receive unique tokens");
assert(classTintStyle("_negative").includes("--negative-class-bg"),
       "negative uses the theme-neutral background, never an event hue");
"""
    script_path = tmp_path / "dataset-definition-identity.js"
    script_path.write_text(harness, encoding="utf-8")
    subprocess.run([node, script_path], check=True)


# --- Target-class role/mining editing (negative-role generalization) ------

def _start_target_classes_server(tmp_path, monkeypatch, *, config_toml_body: str):
    from sed import config as cfg_mod
    from sed.config import Config
    from sed.data.inspect_server import make_handler
    from http.server import ThreadingHTTPServer

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(config_toml_body, encoding="utf-8")
    cfg = Config.model_validate(_read_toml_for_test(layout.config_toml))
    port = _free_port()
    handler = make_handler(
        cfg=cfg, project_id="p",
        render_html_callable=lambda: "<html>ok</html>",
        layout=layout,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, layout, port


def _read_toml_for_test(path):
    from sed.config import _read_toml
    return _read_toml(path)


_BASE_TARGET_CLASSES_TOML = """
[dataset]
target_classes = ["_negative", "glass"]

[model]
classes = ["_negative", "glass"]
"""


def test_set_target_persists_role_and_mining_fields(tmp_path, monkeypatch):
    """`set_target` merges onto (not replaces) an existing [targets] entry, and
    the written config resolves the expected role/mine_threshold afterwards —
    the persistence side of the Class roles & mining UI panel."""
    from sed.config import load_config

    httpd, layout, port = _start_target_classes_server(
        tmp_path, monkeypatch, config_toml_body=_BASE_TARGET_CLASSES_TOML,
    )
    try:
        d = _post_json(f"http://127.0.0.1:{port}/api/target-classes", {
            "action": "set_target", "name": "glass",
            "fields": {"role": "negative", "mine": True},
        })
        assert d["ok"] is True, d
        assert d["role"] == "negative"

        # A second, independent field update must not clobber the first.
        d2 = _post_json(f"http://127.0.0.1:{port}/api/target-classes", {
            "action": "set_target", "name": "glass",
            "fields": {"mine_threshold": 0.3},
        })
        assert d2["ok"] is True, d2

        cfg = load_config("p")
        assert cfg.class_roles(["_negative", "glass"])["glass"] == "negative"
        target = cfg.targets["glass"]
        assert target.mine is True
        assert target.mine_threshold == pytest.approx(0.3)
        assert cfg.mine_threshold("glass") == pytest.approx(0.3)
        # Setting role="negative" moved glass onto the shared veto ceiling.
        assert cfg.class_threshold("glass") == pytest.approx(cfg.inference.negative_max)
    finally:
        httpd.shutdown()


def test_set_target_rejects_unknown_class_and_unknown_field(tmp_path, monkeypatch):
    httpd, layout, port = _start_target_classes_server(
        tmp_path, monkeypatch, config_toml_body=_BASE_TARGET_CLASSES_TOML,
    )
    try:
        status, body = _post_json_status(f"http://127.0.0.1:{port}/api/target-classes", {
            "action": "set_target", "name": "not_a_class", "fields": {"role": "negative"},
        })
        assert status == 400
        assert body["ok"] is False

        status, body = _post_json_status(f"http://127.0.0.1:{port}/api/target-classes", {
            "action": "set_target", "name": "glass", "fields": {"bogus_field": 1},
        })
        assert status == 400
        assert body["ok"] is False

        status, body = _post_json_status(f"http://127.0.0.1:{port}/api/target-classes", {
            "action": "set_target", "name": "glass", "fields": {"role": "not_a_role"},
        })
        assert status == 400
        assert body["ok"] is False
    finally:
        httpd.shutdown()


def test_rename_and_remove_keep_targets_table_in_sync(tmp_path, monkeypatch):
    """Renaming/removing a class must carry/drop its [targets] entry — before
    this fix, `_handle_target_classes_post` only rewrote `dataset.target_classes`,
    leaving role/enabled/threshold/veto_max/mine silently pointed at a class
    name that no longer existed."""
    from sed.config import load_config

    toml_body = _BASE_TARGET_CLASSES_TOML + """
[targets]
glass = { role = "suppressor", mine = true, mine_threshold = 0.42 }
"""
    httpd, layout, port = _start_target_classes_server(
        tmp_path, monkeypatch, config_toml_body=toml_body,
    )
    try:
        d = _post_json(f"http://127.0.0.1:{port}/api/target-classes", {
            "action": "rename", "name": "glass", "new_name": "cutlery",
        })
        assert d["ok"] is True, d
        assert d["class_visuals"]["_negative"] == "slate"
        assert d["class_visuals"]["cutlery"] == "blue"
        assert "glass" not in d["class_visuals"]
        cfg = load_config("p")
        assert "glass" not in cfg.targets
        assert "cutlery" in cfg.targets
        assert cfg.targets["cutlery"].role == "suppressor"
        assert cfg.targets["cutlery"].mine_threshold == pytest.approx(0.42)

        d2 = _post_json(f"http://127.0.0.1:{port}/api/target-classes", {
            "action": "remove", "name": "cutlery",
        })
        assert d2["ok"] is True, d2
        assert d2["class_visuals"] == {"_negative": "slate"}
        cfg2 = load_config("p")
        assert "cutlery" not in cfg2.targets
        assert "cutlery" not in cfg2.model.classes
    finally:
        httpd.shutdown()
