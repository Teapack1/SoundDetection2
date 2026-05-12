"""End-to-end smoke test for the interactive inspect server.

We can't load an HTML page in-process, but we can boot the stdlib HTTP
server in a thread and hit the API endpoints (audio, delete,
promote-mined) against real sidecar files.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sed.config import ProjectLayout


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


def _post_json(url: str, body: dict, timeout: float = 2.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get_json(url: str, timeout: float = 2.0) -> dict:
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


def test_server_delete_endpoint_removes_wav_and_sidecar(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_clip(layout)
    sidecar = wav.with_suffix(".json")
    assert wav.exists() and sidecar.exists()

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
        body = json.dumps({"paths": [str(wav)]}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/delete", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            payload = json.loads(r.read())
            assert payload["ok"] is True
            assert payload["count"] == 1
        assert not wav.exists()
        assert not sidecar.exists()
    finally:
        httpd.shutdown()


def test_server_background_delete_job_reports_progress_and_deleted_paths(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "lib")

    layout = ProjectLayout("p")
    layout.create()
    wav = _seed_clip(layout)
    sidecar = wav.with_suffix(".json")

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
        assert not wav.exists()
        assert not sidecar.exists()
    finally:
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
    assert context["source"] == "test_predictions.tsv"
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
    assert "id=\"frozen-run-select\"" in html
    assert "run-1" in html
    assert "id=\"frozen-metric\"" in html
    assert "id=\"frozen-min\"" in html
    assert "id=\"frozen-max\"" in html
    assert "type=\"number\"" in html
    assert "placeholder=\"0.00\"" in html
    assert "placeholder=\"1.00\"" in html
    assert "data-frozen-pred=\"_negative\"" in html
    assert "data-frozen-predicted-confidence=\"0.052327\"" in html
    assert "data-frozen-argmax-confidence=\"0.281352\"" in html
    assert "data-frozen-margin=\"0.098957\"" in html
    assert "below_threshold" in html
    assert "class=\"aug-filter " in html
    assert "id=\"mask-metric\"" in html
    assert "True _negative" not in html
    assert "No trigger" not in html


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
    finally:
        httpd.shutdown()


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


def test_promote_mined_endpoint_moves_wav_and_preserves_metadata(tmp_path, monkeypatch):
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
        body = json.dumps({
            "paths": [str(wav)],
            "label": "mined false positive",
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/promote-mined",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            payload = json.loads(r.read())
            assert payload["ok"] is True
            assert payload["count"] == 1

        assert not wav.exists()
        dest = Path(payload["moved"][0]["dest"])
        assert dest.exists()
        assert dest.parent == tmp_path / "lib" / "sounds" / "mined_false_positive"
        meta = json.loads(dest.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["category"] == "sounds"
        assert meta["project_id"] == "p"
        assert meta["review_status"] == "accepted"
        assert meta["probs_percent"]["cheers_loud"] == 92.0
        assert meta["promoted_to"] == "_negative"
        assert meta["promoted_negative_label"] == "mined_false_positive"
        assert meta["promoted_from"] == str(wav)
        assert "flagged" not in meta
    finally:
        httpd.shutdown()


def test_selection_and_clip_role_endpoints_update_library_metadata(tmp_path, monkeypatch):
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
        role_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/clip-role",
            data=json.dumps({"paths": [str(wav)], "dataset_role": "test"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(role_req, timeout=2) as r:
            role_payload = json.loads(r.read())
        assert role_payload["count"] == 1
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["dataset_role"] == "test"
        assert meta["project_id"] == "p"

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
                }
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(settings_req, timeout=2) as r:
            settings_payload = json.loads(r.read())
        assert settings_payload["selection"]["augmentation_settings"]["samples_per_source"] == 7.0

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
        Library().import_clip(src, label=label, project_id="p", dataset_role="train")

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
                    "base_filters": [{"labels": ["glass"], "project_ids": ["p"], "count": 999}],
                    "overlay_filters": [{"labels": ["cutlery"], "project_ids": ["p"], "count": 999}],
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
        Library().import_clip(src, label=label, project_id="p", dataset_role="train")

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


def test_preview_logmel_uses_model_frontend_and_clears_stale_thumbs(tmp_path, monkeypatch):
    """Preview spectrograms must show the model's features: efficientat
    frontend at the model sample rate with a fixed display range. Cached
    thumbnails from another feature contract are dropped once."""
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
        assert not stale_thumb.exists()
        assert (lib / ".spec_thumbs_signature").exists()

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

        # Same contract again — the signature short-circuits, nothing breaks.
        inspect_mod.set_preview_features(cfg)
    finally:
        inspect_mod.set_preview_features(None)
    spec2, v1, v2 = inspect_mod._preview_logmel(wave, 48000)
    assert v1 is None and v2 is None


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
    Library().import_clip(base_src, label="glass", project_id="p", dataset_role="test")
    Library().import_clip(overlay_src, label="amb", project_id="p", dataset_role="train")

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
                        "split": "test", "dataset_role": "test",
                    }],
                    "overlay_filters": [{
                        "labels": ["amb"], "project_ids": ["p"],
                        "split": "train", "dataset_role": "train",
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
        dataset_role="train",
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
        dataset_role="train",
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
        dataset_role="train",
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
        dataset_role="test",
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
                "dataset_role": "test",
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
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/review-data", timeout=10) as r:
            payload = json.loads(r.read())
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
        "created_at": "2026-06-11T12:00:00+00:00",
        "test_metrics": {"f1_macro_targets": 0.9974, "accuracy": 0.99},
    }), encoding="utf-8")
    (run_dir / "training_config.json").write_text(json.dumps({
        "training": {"loss": "cross_entropy"},
    }), encoding="utf-8")

    options = _frozen_run_options(layout, active_run_id="run-a")

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
        clip = lib.import_clip(src, label="glass", project_id="p", dataset_role="train")
        glass_paths.append(clip.absolute_path(lib.root).resolve())
    noise_src = tmp_path / "noise.wav"
    sf.write(str(noise_src), np.zeros(32000, dtype=np.float32), 32000)
    lib.import_clip(noise_src, label="noise", project_id="p", dataset_role="train")

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
        delete_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/delete",
            data=json.dumps({"paths": [str(glass_paths[0])]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(delete_req, timeout=2) as r:
            deleted = json.loads(r.read())
        assert deleted["ok"] is True
        assert deleted["count"] == 1

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/review-data", timeout=10) as r:
            review = json.loads(r.read())
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
            "dataset_role": "train",
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
    assert "bucketDatasetRole" in html
    assert "formatLocalDateTime" in html
    assert "compactUtcToMs" in html
    assert "libraryTimeLabel" in html
    assert "cssEscape(className)" in html
    assert "assigned-dot" in html
    assert "assignedClassesForRule" in html
    assert "source-rule-row" in html
    assert "classTintStyle" in html
    assert "CLASS_TINT_PALETTE" in html
    assert "body.ui-busy" in html
    assert "background-job-status" in html
    assert "backgroundJobHint" in html
    assert "deleteSelectedLibraryRows" in html
    assert "Rendering changes" in html
    assert "selectedOrVisibleLibraryRules(split)" in html
    assert 'parentBucket = (c.clip_kind || "") === "channel_split" ? ""' in html
    assert "rule.channel_index" in html
    assert "library-refresh-review" in html
    assert 'class="refresh" id="library-refresh-review"' in html
    assert "Rescan all" in html
    assert "Rescan all tabs" in html
    assert "compose-dataset-btn" in html
    assert "Compose dataset" in html
    assert "Refresh review tabs" not in html
    assert "refreshRequestSeq" in html
    assert "/api/review-data/start" in html
    assert "/api/review-data/status" in html
    assert "/api/delete/start" in html
    assert "/api/delete/status" in html
    assert "setRefreshHeld" in html
    assert "applyReviewDataPayload(data, { render: false })" in html
    assert 'id="review-refresh-status"' in html
    assert 'class="del selection-only" id="del-btn"' in html
    assert '<button type="button" class="refresh" id="library-refresh-review" disabled>' not in html
    assert ".library-active .toolbar .selection-only" in html
    assert "library-row-delete" in html
    assert "data-delete-library-row" in html
    assert "sourceRuleAudioLabel" in html
    assert "sourceRuleDetailsHtml" in html
    assert "sourceRuleBriefSummary" in html
    assert "lib-row-split-pill" in html
    assert "className)} ·" in html
    assert "className)} ←" not in html
    assert "includeLabels: false" not in html
    assert "aug-setting-seed" in html
    assert "aug-setting-mask-min-mean-db" in html
    assert "Use selected as base" in html
    assert "recipe-target-class" not in html
    assert "No composed frozen test set was found yet" in html
    assert "split <code>test</code>" in html
    assert "uv run sed dataset compose --project p" in html

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
    assert ".library-active .toolbar" in html


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
        lib.import_clip(src, label="glass", project_id="p", dataset_role="train")
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


def test_panel_totals_are_uncapped(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=5)
    httpd, port = _serve(cfg, layout, max_thumbs_per_class=2)
    try:
        totals = _api_get(port, "/api/panel-totals")
        assert totals["ok"] is True
        # True count (5), independent of the display card cap of 2.
        assert totals["panel_totals"]["dataset"]["glass"] == 5

        # review-data caps the rendered cards at 2 but still reports the real total.
        review = _api_get(port, "/api/review-data")
        assert len(review["panels"]["glass"]) == 2
        assert review["panel_totals"]["dataset"]["glass"] == 5
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
        # An unknown tab is rejected.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _api_get(port, "/api/clips?tab=bogus&class=glass")
        assert exc.value.code == 400
    finally:
        httpd.shutdown()


def test_review_data_scoped_returns_only_requested(tmp_path, monkeypatch):
    cfg, layout = _setup_dataset_project(tmp_path, monkeypatch, n_glass=3)
    httpd, port = _serve(cfg, layout)
    try:
        scoped = _api_get(port, "/api/review-data?scopes=frozen")
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
