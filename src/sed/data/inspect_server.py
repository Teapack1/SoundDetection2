"""Local HTTP server for the interactive inspect UI.

`clinkd inspect --serve` opens this on http://127.0.0.1:PORT. The HTML
report is rendered live (with current sidecar flag state baked in) and
each clip carries a checkbox. Ticking the box POSTs to /api/flag, which
flips `meta["flagged"]` in the per-clip .json sidecar. The dataset
composer already skips flagged clips, so the next `clinkd dataset compose`
run automatically respects the curation done here.

No external deps — uses stdlib http.server. Single-process, localhost-only;
no auth needed because the server only binds to 127.0.0.1.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from sed.config import Config, ProjectLayout

log = logging.getLogger(__name__)


def _safe_resolve_audio_path(layout: ProjectLayout, raw_path: str) -> Path | None:
    """Resolve a path requested via /audio?path=... and only allow files
    inside the project's recordings dir, the library dir, or the report's
    augmented preview dir. This keeps the server from being abused as a
    file-read primitive."""
    from sed.config import LIBRARY_DIR
    try:
        candidate = Path(raw_path).resolve()
    except Exception:  # noqa: BLE001
        return None
    allowed_roots = [
        layout.recordings_dir.resolve(),
        (layout.reports_dir / "augmented").resolve(),
        LIBRARY_DIR.resolve(),
    ]
    for root in allowed_roots:
        try:
            candidate.relative_to(root)
            if candidate.is_file() and candidate.suffix.lower() == ".wav":
                return candidate
        except ValueError:
            continue
    return None


def _flag_clip(path: Path, flagged: bool) -> bool:
    """Set `meta['flagged'] = flagged` in the sidecar JSON. Returns whether
    the sidecar existed and was updated. Library clips don't have per-WAV
    sidecars (they live in data_library/manifest.jsonl), so this is a no-op
    for those — those rows shouldn't show up in the interactive UI."""
    side = path.with_suffix(".json")
    if not side.exists():
        return False
    try:
        with open(side, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:  # noqa: BLE001
        return False
    if flagged:
        meta["flagged"] = True
    else:
        meta.pop("flagged", None)
    with open(side, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return True


def make_handler(
    cfg: Config,
    project_id: str,
    render_html_callable,
    layout: ProjectLayout,
    shutdown_event: threading.Event,
):
    """Factory that closes over the project/config so we can build a stdlib
    BaseHTTPRequestHandler subclass without globals."""

    class Handler(BaseHTTPRequestHandler):

        # Silence the default access logging to stderr per request — too
        # noisy when 100 audio clips load on one page.
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            if self.path.startswith("/api/"):
                log.info("%s %s", self.command, self.path)

        # ---- GET --------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send_html(render_html_callable())
            elif url.path == "/audio":
                qs = parse_qs(url.query)
                paths = qs.get("path", [])
                if not paths:
                    self._send(400, b"missing ?path"); return
                resolved = _safe_resolve_audio_path(layout, paths[0])
                if resolved is None:
                    self._send(403, b"forbidden"); return
                self._send_file(resolved)
            elif url.path == "/api/flags":
                # Diagnostic: list currently flagged clip paths
                flagged = []
                for wav in layout.recordings_dir.rglob("*.wav"):
                    side = wav.with_suffix(".json")
                    if side.exists():
                        try:
                            if json.loads(side.read_text()).get("flagged"):
                                flagged.append(str(wav))
                        except Exception:  # noqa: BLE001
                            pass
                self._send_json({"flagged": flagged})
            else:
                self._send(404, b"not found")

        # ---- POST -------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/api/done":
                self._send_json({"ok": True})
                threading.Thread(target=shutdown_event.set, daemon=True).start()
                return
            if url.path == "/api/delete":
                self._handle_delete(); return
            if url.path != "/api/flag":
                self._send(404, b"not found"); return
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON"); return
            raw_path = body.get("path")
            flagged = bool(body.get("flagged"))
            if not isinstance(raw_path, str):
                self._send(400, b"missing path"); return
            resolved = _safe_resolve_audio_path(layout, raw_path)
            if resolved is None:
                self._send(403, b"forbidden"); return
            ok = _flag_clip(resolved, flagged)
            if not ok:
                self._send_json({"ok": False, "reason": "no_sidecar"}, status=200)
                return
            self._send_json({"ok": True, "path": str(resolved), "flagged": flagged})

        def _handle_delete(self) -> None:
            """Permanently delete WAV + JSON sidecar for requested paths."""
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                self._send(400, b"invalid JSON"); return
            raw_paths = body.get("paths", [])
            if not isinstance(raw_paths, list):
                self._send(400, b"paths must be an array"); return
            deleted = []
            for rp in raw_paths:
                resolved = _safe_resolve_audio_path(layout, rp)
                if resolved is None:
                    continue
                sidecar = resolved.with_suffix(".json")
                try:
                    resolved.unlink(missing_ok=True)
                    sidecar.unlink(missing_ok=True)
                    deleted.append(str(resolved))
                    log.info("deleted %s", resolved.name)
                except Exception as e:  # noqa: BLE001
                    log.warning("delete failed %s: %s", resolved, e)
            self._send_json({"ok": True, "deleted": deleted, "count": len(deleted)})

        # ---- helpers ----------------------------------------------------

        def _send(self, status: int, body: bytes, ctype: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def _send_json(self, payload, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _send_html(self, html: str) -> None:
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

        def _send_file(self, path: Path) -> None:
            ctype, _ = mimetypes.guess_type(str(path))
            ctype = ctype or "application/octet-stream"
            try:
                size = path.stat().st_size
            except OSError:
                self._send(404, b"not found"); return
            # Honour Range requests so the browser can seek long audio
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            status = 200
            if rng and rng.startswith("bytes="):
                try:
                    spec = rng[6:]
                    sp, ep = spec.split("-")
                    if sp: start = int(sp)
                    if ep: end = int(ep)
                    end = min(end, size - 1)
                    status = 206
                except Exception:  # noqa: BLE001
                    status = 200; start, end = 0, size - 1
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = f.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(
    cfg: Config,
    project_id: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    max_thumbs_per_class: int = 200,
    augmentation_preview: bool = True,
) -> None:
    """Start the interactive inspect server. Blocks until Ctrl+C."""
    from sed.data.inspect import inspect_project, _render_html as _render
    # Re-render the HTML on every GET / so adding/quarantining clips and
    # then refreshing the browser shows fresh state. Each render is cheap
    # because the heavy work (UMAP, augmentation preview) is cached on disk.
    layout = ProjectLayout(project_id)

    # First render: also generates the on-disk artefacts (UMAP, augmented
    # WAVs) that subsequent renders re-use.
    log.info("rendering initial report ...")
    out_path = inspect_project(
        cfg, project_id,
        max_thumbs_per_class=max_thumbs_per_class,
        augmentation_preview=augmentation_preview,
        interactive=True,
    )
    log.info("static report also written to %s", out_path)

    def render_html_now() -> str:
        # Re-render only the HTML (no UMAP/aug regen) so the live flag state
        # reflects sidecars on every page refresh.
        from sed.data.inspect import _quick_rerender
        return _quick_rerender(cfg, project_id,
                               max_thumbs_per_class=max_thumbs_per_class,
                               augmentation_preview=augmentation_preview)

    shutdown_event = threading.Event()
    Handler = make_handler(cfg, project_id, render_html_now, layout, shutdown_event)

    # Bind, possibly auto-incrementing the port if busy
    for try_port in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, try_port), Handler)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"could not bind any port in [{port},{port+19}]")

    url = f"http://{host}:{httpd.server_port}/"
    log.info("inspect server listening on %s", url)
    log.info("Ctrl+C, or press 'Done' in the browser, to stop.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    # Serve on a worker thread so the main thread can also watch for the
    # in-browser "Done" button (which sets shutdown_event).
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    try:
        while not shutdown_event.is_set():
            if shutdown_event.wait(timeout=0.5):
                break
        log.info("Done received from browser — shutting down.")
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        httpd.shutdown()
        server_thread.join(timeout=2.0)
