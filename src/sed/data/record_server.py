"""Local HTTP server for the recording console.

``sed record --serve`` opens this on http://127.0.0.1:8790. It is the operator
surface for a capture session: pick the interface and the channels, watch every
microphone live, start onset-triggered recording, and retune the gate while the
stream keeps running.

It is deliberately a separate server from ``sed dataset inspect --serve`` rather
than another tab in it. Inspect scans the whole library and renders thumbnails
before it answers the first request; on site you want a page in a second, and
you want it on a machine that may be doing nothing else. The two also cannot
share one PortAudio input, so keeping them apart keeps the interlock simple.

No external deps — stdlib http.server, same as the inspect server.
Single-process, localhost-only; no auth needed because it only binds 127.0.0.1.
"""

from __future__ import annotations

import itertools
import json
import logging
import mimetypes
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from sed.config import LIBRARY_PROJECT_ID, Config, ProjectLayout
from sed.data.recorder import _linear_to_db

log = logging.getLogger(__name__)

DEFAULT_PORT = 8790

_DENSITIES = ("empty", "quiet", "normal", "busy", "peak")
_AUDIO_SUFFIXES = frozenset({".wav"})
# Peaking this close to full scale means the converter is clipping.
CLIPPING_PEAK_DBFS = -0.5


# ---------------------------------------------------------------------------
# Request-value coercion
# ---------------------------------------------------------------------------


def parse_channels(value) -> list[int]:
    """Ordered, zero-based physical channel indexes.

    Accepts a list or a ``"0,2"`` string, because the same field arrives from a
    JSON body and from a CLI prefill.
    """
    if value is None:
        raise ValueError("channels are required")
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        parts = list(value)
    try:
        idx = [int(p) for p in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"channels must be integer indexes, got {value!r}") from exc
    if not idx:
        raise ValueError("select at least one channel")
    if any(c < 0 for c in idx):
        raise ValueError("channel indexes must be >= 0")
    if len(set(idx)) != len(idx):
        raise ValueError("channel indexes must not repeat")
    return idx


def parse_tags(value) -> list[str]:
    """Comma-separated tags, deduplicated, '#' stripped."""
    raw = value if isinstance(value, list) else str(value or "").split(",")
    tags: list[str] = []
    seen: set[str] = set()
    for item in raw:
        tag = str(item).strip().lstrip("#").strip()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def parse_duration(value) -> float | None:
    """Wall-clock cap: ``15m``, ``2h``, ``90s``, a bare number of seconds, or
    empty for no limit."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    try:
        if text.endswith("h"):
            seconds = float(text[:-1]) * 3600
        elif text.endswith("m"):
            seconds = float(text[:-1]) * 60
        elif text.endswith("s"):
            seconds = float(text[:-1])
        else:
            seconds = float(text)
    except ValueError as exc:
        raise ValueError(f"could not read a duration from {value!r}") from exc
    if seconds <= 0:
        raise ValueError("duration must be positive")
    return seconds


def onset_updates(body: dict) -> dict:
    """The live-tunable onset fields present in a request body, typed."""
    from sed.data.record_engine import LIVE_ONSET_KEYS

    casts = {
        "min_flux_db": float,
        "mad_k": float,
        "median_window_s": float,
        "recording_cooldown_ms": int,
        "cross_channel_merge_ms": int,
        "record_seconds": float,
        "preroll_ms": int,
        "randomize_preroll": bool,
        "multi_mic_trigger": str,
        "hop_ms": int,
        "trigger_n_mels": int,
    }
    out: dict = {}
    for key in LIVE_ONSET_KEYS:
        if key not in body:
            continue
        if key == "preroll_ms_range":
            lo, hi = body[key]
            out[key] = (int(lo), int(hi))
            continue
        try:
            out[key] = casts[key](body[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: {exc}") from exc
    return out


def safe_library_path(raw_path: str) -> Path | None:
    """Resolve a client-supplied clip path, refusing anything outside the
    library. Keeps the server from becoming a file-read primitive."""
    from sed.config import LIBRARY_DIR

    try:
        candidate = Path(raw_path).resolve()
        root = LIBRARY_DIR.resolve()
    except Exception:  # noqa: BLE001
        return None
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if candidate.is_file() and candidate.suffix.lower() in _AUDIO_SUFFIXES:
        return candidate
    return None


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def devices_payload(cfg: Config, project: str = "") -> dict:
    """Input devices, the configured selection, and the library's labels."""
    from sed.data.library import RAW_CATEGORY, Library

    devices: list[dict] = []
    error = ""
    try:
        from sed.audio.capture import list_input_devices

        devices = [
            {
                "index": d.index,
                "name": d.name,
                "max_input_channels": d.max_input_channels,
                "default_samplerate": d.default_samplerate,
                "hostapi": d.hostapi_name,
            }
            for d in list_input_devices()
        ]
    except Exception as exc:  # noqa: BLE001 — no PortAudio is a message, not a crash
        error = f"{type(exc).__name__}: {exc}"

    labels: list[dict] = []
    locations: list[str] = []
    tags: list[dict] = []
    try:
        library = Library()
        counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}
        seen_locations: set[str] = set()
        for clip in library.all_clips():
            if clip.category != RAW_CATEGORY:
                continue
            counts[clip.label] = counts.get(clip.label, 0) + 1
            for tag in getattr(clip, "tags", None) or []:
                tag = str(tag).strip()
                if tag:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            # Locations already recorded — narrowed to the selected project, so
            # a venue's own places come back rather than every site ever used.
            where = str(getattr(clip, "location", "") or "").strip()
            if where and (not project or str(clip.project_id) == project):
                seen_locations.add(where)
        labels = [{"label": k, "clips": v} for k, v in sorted(counts.items())]
        locations = sorted(seen_locations)
        # Most-used first: on site the tag you want is usually one you have
        # already been using this session.
        tags = [
            {"tag": k, "clips": v}
            for k, v in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read library labels: %s", exc)

    try:
        from sed.config import list_projects

        projects = list_projects()
    except Exception:  # noqa: BLE001
        projects = []

    return {
        "ok": True,
        "devices": devices,
        "device_error": error,
        "device": cfg.deploy.record_device,
        "sample_rate": cfg.audio.record_sample_rate,
        "blocksize": cfg.audio.blocksize,
        "channels": list(cfg.audio.record_channels),
        "microphone_type": list(cfg.audio.microphone_type),
        "labels": labels,
        "locations": locations,
        "tags": tags,
        "projects": projects,
        "project": project,
        "clip_project_id": project or LIBRARY_PROJECT_ID,
        "densities": list(_DENSITIES),
    }


def preflight_payload(cfg: Config, seconds: float = 3.0) -> dict:
    """The existing pre-record checklist, as rows the page can render."""
    from sed.doctor import doctor_record

    checks = doctor_record(cfg, seconds=seconds)
    return {
        "ok": True,
        "checks": [
            {"name": c.name, "status": c.status, "detail": c.detail, "fix": c.fix}
            for c in checks
        ],
        "worst": (
            "FAIL" if any(c.status == "FAIL" for c in checks)
            else "WARN" if any(c.status == "WARN" for c in checks)
            else "PASS"
        ),
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def make_handler(
    cfg: Config,
    project_id: str = "",
    shutdown_event: threading.Event | None = None,
    prefill: dict | None = None,
):
    """Build the request handler, closing over the session state.

    All mutable state lives here rather than in module globals, so a test can
    boot several servers in one process.
    """
    from sed.data.record_engine import (
        ClipMetadata,
        RecordEngine,
        onset_snapshot,
        warm_clip_writer,
    )

    # Writing a clip and rendering its preview pull in librosa, scipy and
    # matplotlib — about six seconds on a cold process, which would otherwise be
    # paid by the writer thread while the operator waits for their first take to
    # appear. The operator is in Setup for longer than that, so spend it here.
    threading.Thread(target=warm_clip_writer, daemon=True, name="record-warmup").start()

    # Runs recorded since this server started. One entry per Start..Stop, kept
    # in memory only: it is a working list for the session, not a second index
    # of the library. Each carries the clips it wrote so a whole pass can be
    # thrown away without going hunting for its session id in Inspect.
    state = {
        "cfg": cfg,
        "engine": None,
        "meter": None,
        "runs": [],
        "run": None,
        # Recording is mostly not project work — it fills the shared library —
        # so no project is the default. "" means none; clips then carry
        # LIBRARY_PROJECT_ID, which every project's default source rule accepts.
        "project": project_id or "",
    }
    run_counter = itertools.count(1)

    def active_project() -> str:
        return str(state["project"] or "")

    def clip_project_id() -> str:
        """What gets stamped on a clip: the chosen project, or the shared pool."""
        return active_project() or LIBRARY_PROJECT_ID

    def active_layout() -> ProjectLayout | None:
        """The selected project's layout, or None when recording library-wide.

        Deliberately None rather than a ProjectLayout("_library"): that path
        does not exist, and update_config_toml_values creates missing parents,
        so a stray write would invent a project directory named after the
        sentinel.
        """
        chosen = active_project()
        return ProjectLayout(chosen) if chosen else None

    def select_project(chosen: str) -> dict:
        """Point recording at a project, or at no project at all.

        Loads that project's capture and onset settings so the venue's own mic
        setup comes with it, and stamps its id on subsequent clips. Clips keep
        their original stamp, so this rolls the recording session rather than
        retroactively relabelling anything.
        """
        from sed.config import list_projects, load_config

        chosen = str(chosen or "").strip()
        if chosen and chosen not in list_projects():
            raise FileNotFoundError(f"no project named {chosen!r}")
        state["project"] = chosen
        state["cfg"] = load_config(chosen) if chosen else Config()
        engine = state["engine"]
        if engine is not None:
            engine.set_project(clip_project_id())
        return {"project": chosen, "clip_project_id": clip_project_id()}

    def active_cfg() -> Config:
        """The config that is actually in force right now.

        While a session runs, the engine owns a copy that live retuning has
        edited. Reading ``state["cfg"]`` instead would make a page reload — or
        the device list, or a pre-flight check — describe settings the running
        stream stopped using several adjustments ago.
        """
        engine = state["engine"]
        return engine.cfg if engine is not None else state["cfg"]
    # One microphone, one owner. The idle meter and a recording session both
    # want the same PortAudio input, so every transition takes this lock and
    # the loser gets a 409 rather than a confusing PortAudio error.
    device_lock = threading.RLock()
    form = dict(prefill or {})

    def stop_meter() -> None:
        meter = state["meter"]
        state["meter"] = None
        if not meter:
            return
        stream = meter.get("stream")
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("meter stream close failed: %s", exc)

    def start_meter(body: dict) -> dict:
        import numpy as np
        import sounddevice as sd

        from sed.audio.capture import resolve_device

        active = active_cfg()
        device = str(body.get("device") or active.deploy.record_device or "default")
        channels = parse_channels(
            body.get("channels") or list(active.audio.record_channels)
        )
        sample_rate = int(body.get("sample_rate") or active.audio.record_sample_rate)
        blocksize = int(body.get("blocksize") or active.audio.blocksize)
        n_open = max(channels) + 1
        idx = resolve_device(device)
        info = sd.query_devices(idx)
        if n_open > int(info["max_input_channels"]):
            raise ValueError(
                f"channels {channels} need {n_open} input channel(s), but "
                f"{info['name']} has {info['max_input_channels']}"
            )
        # The idle meter runs the real onset detector too. Setting the gate is
        # something to do BEFORE the first take, not discover during it, and a
        # level meter alone cannot tell you whether a footstep would trigger.
        from sed.audio.onset import StreamingSuperflux
        from sed.audio.resample import StreamingResampler
        from sed.data.record_engine import FluxCalibrator

        o = active.onset
        model_sr = int(active.model.sample_rate)
        detectors = {
            c: StreamingSuperflux(
                sample_rate=model_sr, hop_ms=o.hop_ms, n_mels=o.trigger_n_mels,
                median_window_s=o.median_window_s, mad_k=o.mad_k,
                refractory_ms=min(150, int(o.recording_cooldown_ms)),
                min_flux=o.min_flux,
            )
            for c in channels
        }
        resamplers = {c: StreamingResampler(sample_rate, model_sr) for c in channels}
        calibrator = FluxCalibrator()
        min_flux = float(o.min_flux)

        job = {
            "detectors": detectors,
            "resamplers": resamplers,
            "calibrator": calibrator,
            "state": "running",
            "device": device,
            "device_index": idx,
            "device_name": str(info.get("name") or ""),
            "sample_rate": sample_rate,
            "channels": channels,
            # Seeded with the SAME shape the callback produces, so a client
            # polling before the first block never sees a different payload.
            "levels": [
                {"channel": c, "peak_dbfs": -120.0, "rms_dbfs": -120.0,
                 "flux_db": None, "gate_db": None, "gate_source": "",
                 "will_trigger": False, "detector": True}
                for c in channels
            ],
            "xruns": 0,
            "stream": None,
        }

        def callback(indata, frames, time_info, status):
            _ = (frames, time_info)
            block = indata.astype("float32", copy=False)
            levels = []
            for c in channels:
                x = block[:, c]
                peak = float(np.max(np.abs(x))) if x.size else 0.0
                rms = float(np.sqrt(np.mean(x**2) + 1e-12)) if x.size else 0.0
                chunk = resamplers[c].process(x)
                if chunk.size:
                    detectors[c].process_block(chunk, resamplers[c].output_index)
                _d, flux, thresh = detectors[c].get_status()
                flux, thresh = float(flux), float(thresh)
                gate = max(min_flux, thresh)
                calibrator.observe(flux)
                levels.append({
                    "channel": c,
                    "peak_dbfs": float(20.0 * np.log10(peak + 1e-12)),
                    "rms_dbfs": float(20.0 * np.log10(rms + 1e-12)),
                    "flux_db": round(_linear_to_db(flux), 1),
                    "gate_db": round(_linear_to_db(gate), 1),
                    "gate_source": "floor" if min_flux >= thresh else "adaptive",
                    "will_trigger": bool(flux >= gate),
                    "detector": True,
                })
            calibrator.advance()
            if state["meter"] is job:
                job["levels"] = levels
                if status:
                    job["xruns"] += 1

        stream = sd.InputStream(
            device=idx, channels=n_open, samplerate=sample_rate,
            blocksize=blocksize, dtype="float32", callback=callback,
        )
        job["stream"] = stream
        state["meter"] = job
        try:
            stream.start()
        except Exception:
            state["meter"] = None
            raise
        return meter_snapshot()

    def meter_snapshot() -> dict:
        meter = state["meter"]
        if not meter:
            return {"state": "idle", "channels": []}
        return {
            "state": "running",
            "device_name": meter["device_name"],
            "sample_rate": meter["sample_rate"],
            "xruns": meter["xruns"],
            "calibration": meter["calibrator"].snapshot(),
            "channels": [
                {**lvl, "clipping": lvl["peak_dbfs"] >= -0.5} for lvl in meter["levels"]
            ],
        }

    def levels_payload() -> dict:
        """Meters only, for the fast poll.

        Deliberately tiny: the full status carries recent takes, the session
        log and the run list, and sending all of that many times a second is
        what made the meters feel like they lagged the room.
        """
        engine = state["engine"]
        if engine is not None:
            return {"ok": True, "recording": True, **engine.levels()}
        meter = state["meter"]
        if not meter:
            return {"ok": True, "recording": False, "channels": []}
        return {
            "ok": True,
            "recording": False,
            "calibration": meter["calibrator"].snapshot(),
            "channels": [
                {**lvl, "clipping": lvl["peak_dbfs"] >= CLIPPING_PEAK_DBFS}
                for lvl in meter["levels"]
            ],
        }

    def status_payload() -> dict:
        engine = state["engine"]
        if engine is not None:
            return {"ok": True, "recording": True, "session": engine.snapshot(),
                    "runs": runs_payload(), "form": form,
                    "project": active_project(),
                    "clip_project_id": clip_project_id()}
        active = active_cfg()
        return {
            "ok": True,
            "recording": False,
            "runs": runs_payload(),
            "project": active_project(),
            "clip_project_id": clip_project_id(),
            "can_save_settings": bool(active_project()),
            "meter": meter_snapshot(),
            "onset": onset_snapshot(active.onset),
            "form": form,
            "sample_rate": active.audio.record_sample_rate,
            "clip_frames": round(active.onset.record_seconds
                                 * active.audio.record_sample_rate),
        }

    def start_session(body: dict) -> dict:
        from sed.data.library import RAW_CATEGORY

        if state["engine"] is not None:
            raise RuntimeError("a recording session is already running")
        stop_meter()

        active = state["cfg"].model_copy(deep=True)
        if body.get("device"):
            active.deploy.record_device = str(body["device"])
        if body.get("sample_rate"):
            active.audio.record_sample_rate = int(body["sample_rate"])
        if body.get("blocksize"):
            active.audio.blocksize = int(body["blocksize"])
        if body.get("channels") is not None:
            active.audio.record_channels = parse_channels(body["channels"])
        active.audio = type(active.audio).model_validate(active.audio.model_dump())

        updates = onset_updates(body.get("onset") or {})
        if updates:
            from sed.data.record_engine import apply_onset_updates

            active.onset = apply_onset_updates(active.onset, updates)

        label = str(body.get("label") or "").strip()
        if not label:
            raise ValueError("a label is required — clips are saved under it")
        density = str(body.get("density") or "normal")
        if density not in _DENSITIES:
            raise ValueError(f"density must be one of: {', '.join(_DENSITIES)}")
        mode = str(body.get("mode") or "onset")
        num = int(body.get("num") or 0)

        metadata = ClipMetadata(
            label=label,
            density=density,
            description=str(body.get("description") or ""),
            location=str(body.get("location") or ""),
            mic_position=str(body.get("mic_position") or ""),
            tags=parse_tags(body.get("tags")),
            holdout=bool(body.get("holdout")),
            split_channels=bool(body.get("split_channels")),
            microphone_type=parse_tags(body.get("microphone_type")),
        )
        engine = RecordEngine(
            active,
            clip_project_id(),
            metadata,
            max_clips=num if num > 0 else None,
            duration_s=parse_duration(body.get("duration")),
            mode=mode,
            category=RAW_CATEGORY,
        )
        engine.start()
        state["engine"] = engine
        run = {
            "id": f"run{next(run_counter)}",
            "state": "recording",
            "label": label,
            "labels": [label],
            "density": density,
            "mode": mode,
            "started_at": time.time(),
            "ended_at": None,
            "clips": 0,
            "target": num if num > 0 else None,
            "out_dir": "",
            "reason": "",
            "description": metadata.description,
            "location": metadata.location,
            "tags": list(metadata.tags),
        }
        state["runs"].insert(0, run)
        state["run"] = run
        form.update({
            "label": label, "density": density, "mode": mode,
            "tags": metadata.tags, "location": metadata.location,
            "mic_position": metadata.mic_position,
            "description": metadata.description,
            "holdout": metadata.holdout, "split_channels": metadata.split_channels,
            "num": num, "duration": str(body.get("duration") or ""),
            "channels": list(active.audio.record_channels),
            "device": active.deploy.record_device,
            "sample_rate": active.audio.record_sample_rate,
        })
        return engine.snapshot()

    def _refresh_run(run: dict, engine) -> None:
        """Mirror the engine's live counters onto its run entry."""
        if run is None or engine is None:
            return
        records = engine.clip_records()
        run["records"] = records
        meta = engine.metadata()
        run["description"] = meta.description
        run["location"] = meta.location
        run["tags"] = list(meta.tags)
        run["clips"] = len(records)
        labels = list(dict.fromkeys(c["label"] for c in records)) or [run["label"]]
        run["labels"] = labels
        run["out_dir"] = engine.snapshot().get("out_dir", "")

    def stop_session() -> dict:
        engine = state["engine"]
        run = state["run"]
        if engine is None:
            return {"clips": 0}
        summary = engine.stop()
        if run is not None:
            _refresh_run(run, engine)
            run["state"] = "done"
            run["ended_at"] = time.time()
            run["reason"] = summary.get("reason", "")
            run["out_dir"] = summary.get("out_dir", run.get("out_dir", ""))
            if not run["clips"]:
                # An empty pass is noise in the list, not history.
                state["runs"].remove(run)
        state["engine"] = None
        state["run"] = None
        return summary

    def cancel_session() -> dict:
        """Stop the current pass and delete everything it recorded."""
        engine = state["engine"]
        run = state["run"]
        if engine is None:
            raise RuntimeError("no recording session to cancel")
        engine.stop()
        discarded = engine.discard_all()
        if run is not None and run in state["runs"]:
            state["runs"].remove(run)
        state["engine"] = None
        state["run"] = None
        return {"discarded": discarded}

    def delete_run(run_id: str) -> dict:
        """Delete a finished run's clips, and drop it from the list."""
        run = next((r for r in state["runs"] if r["id"] == run_id), None)
        if run is None:
            raise FileNotFoundError(f"no run {run_id!r} in this server session")
        if run is state["run"]:
            raise RuntimeError("this run is still recording — cancel it instead")
        from sed.data.library import Library

        paths = [c["path"] for c in run.get("records", [])]
        removed = len(Library().remove_clips(paths)) if paths else 0
        state["runs"].remove(run)
        return {"deleted": run_id, "clips": removed}

    def runs_payload() -> list[dict]:
        """Run list for the UI. Clip records stay server-side — the browser
        only needs the counts and the identity to delete by."""
        engine = state["engine"]
        if state["run"] is not None and engine is not None:
            _refresh_run(state["run"], engine)
        return [
            {k: v for k, v in run.items() if k != "records"}
            for run in state["runs"]
        ]

    def save_settings(body: dict) -> dict:
        """Persist one group of settings to the project's config.toml.

        Written through :func:`update_config_toml_values`, which edits keys in
        place, so the hand-written comments in a venue's config survive an
        on-site tweak. The config is then re-read rather than patched in
        memory, so what the server holds is what is on disk.
        """
        from sed.config import load_config, update_config_toml_values

        layout = active_layout()
        if layout is None:
            raise ValueError(
                "no project selected — pick one to save these settings, or keep "
                "them for this session only"
            )
        group = str(body.get("group") or "")
        active = active_cfg()
        if group == "capture":
            updates = {
                "audio": {
                    "record_sample_rate": int(active.audio.record_sample_rate),
                    "record_channels": [int(c) for c in active.audio.record_channels],
                    "blocksize": int(active.audio.blocksize),
                },
                "deploy": {"record_device": str(active.deploy.record_device)},
            }
        elif group == "onset":
            onset = active.onset
            updates = {
                "onset": {
                    "min_flux_db": float(onset.min_flux_db),
                    "mad_k": float(onset.mad_k),
                    "recording_cooldown_ms": int(onset.recording_cooldown_ms),
                    "cross_channel_merge_ms": int(onset.cross_channel_merge_ms),
                    "record_seconds": float(onset.record_seconds),
                    "preroll_ms": int(onset.preroll_ms),
                    "randomize_preroll": bool(onset.randomize_preroll),
                    "preroll_ms_range": list(onset.preroll_ms_range),
                    "multi_mic_trigger": str(onset.multi_mic_trigger),
                },
            }
        elif group == "microphones":
            updates = {"audio": {"microphone_type": list(active.audio.microphone_type)}}
        else:
            raise ValueError("group must be capture, onset or microphones")
        update_config_toml_values(layout.config_toml, updates)
        state["cfg"] = load_config(layout.id)
        return {"saved": updates, "path": str(layout.config_toml)}

    class Handler(BaseHTTPRequestHandler):
        server_version = "sed-record/1.0"

        def log_message(self, fmt, *args):  # noqa: A003 - stdlib hook name
            if self.path.startswith("/api/"):
                log.debug("%s - %s", self.address_string(), fmt % args)

        # ---- dispatch --------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._handle_page()
            elif url.path == "/api/record/devices":
                self._guard(lambda: devices_payload(active_cfg(), active_project()))
            elif url.path == "/api/record/status":
                self._guard(status_payload)
            elif url.path == "/api/record/levels":
                self._guard(levels_payload)
            elif url.path == "/api/record/preflight":
                qs = parse_qs(url.query)
                seconds = float((qs.get("seconds", ["3"])[0]) or 3)
                self._guard(lambda: preflight_payload(active_cfg(), seconds))
            elif url.path == "/audio":
                self._handle_asset(parse_qs(url.query))
            elif url.path == "/thumbnail":
                self._handle_thumbnail(parse_qs(url.query))
            elif url.path == "/spectrogram":
                self._handle_thumbnail(parse_qs(url.query), large=True)
            elif url.path == "/api/record/clip-check":
                self._handle_clip_check(parse_qs(url.query))
            else:
                self._send(404, b"not found")

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            routes = {
                "/api/record/start": self._handle_start,
                "/api/record/stop": self._handle_stop,
                "/api/record/cancel": self._handle_cancel,
                "/api/record/project": self._handle_select_project,
                "/api/record/run/delete": self._handle_run_delete,
                "/api/record/settings": self._handle_settings,
                "/api/record/save-settings": self._handle_save_settings,
                "/api/record/trigger": self._handle_trigger,
                "/api/record/calibrate": self._handle_calibrate,
                "/api/record/clip/delete": self._handle_clip_delete,
                "/api/record/meter/start": self._handle_meter_start,
                "/api/record/meter/stop": self._handle_meter_stop,
                "/api/done": self._handle_done,
            }
            handler = routes.get(url.path)
            if handler is None:
                self._send(404, b"not found")
                return
            handler()

        # ---- handlers --------------------------------------------------

        def _handle_page(self) -> None:
            from sed.data.record_ui import render_record_html

            self._send_html(
                render_record_html(active_cfg(), active_project(), form)
            )

        def _handle_asset(self, qs: dict) -> None:
            path = safe_library_path(qs.get("path", [""])[0])
            if path is None:
                self._send(403, b"forbidden")
                return
            self._send_file(path)

        def _handle_thumbnail(self, qs: dict, large: bool = False) -> None:
            """Linear, full-range spectrogram of a take.

            Not the mel preview the dataset UI uses: judging a recording means
            looking for hum and cable hash, and mel spends its resolution
            everywhere except where those live.
            """
            path = safe_library_path(qs.get("path", [""])[0])
            if path is None:
                self._send(403, b"forbidden")
                return
            channel = int((qs.get("channel", ["0"])[0]) or 0)
            try:
                import soundfile as sf

                from sed.data.clip_check import render_spectrogram

                wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
                channel = min(max(channel, 0), wave.shape[1] - 1)
                png = render_spectrogram(
                    wave[:, channel], sr,
                    width=1000 if large else 260,
                    height=420 if large else 90,
                    with_axes=large,
                )
                self._send(200, png, "image/png")
            except Exception as exc:  # noqa: BLE001
                log.debug("spectrogram failed for %s: %s", path, exc)
                self._send(404, b"no preview")

        def _handle_clip_check(self, qs: dict) -> None:
            """Measure a take: levels, DC, clipping, mains hum."""
            path = safe_library_path(qs.get("path", [""])[0])
            if path is None:
                self._send_json({"ok": False, "error": "forbidden"}, status=403)
                return

            def run() -> dict:
                import soundfile as sf

                from sed.data.clip_check import analyze_clip

                wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
                channels = [
                    {"channel": c, **analyze_clip(wave[:, c], sr)}
                    for c in range(wave.shape[1])
                ]
                return {
                    "name": path.name,
                    "sample_rate": sr,
                    "duration_s": round(wave.shape[0] / sr, 3),
                    "channels": channels,
                    "warnings": [
                        f"ch{c['channel']}: {w}" for c in channels for w in c["warnings"]
                    ],
                }

            self._guard(run)

        def _handle_start(self) -> None:
            body = self._read_json_body()
            with device_lock:
                self._guard(lambda: {"session": start_session(body)})

        def _handle_stop(self) -> None:
            with device_lock:
                self._guard(lambda: {"summary": stop_session()})

        def _handle_select_project(self) -> None:
            body = self._read_json_body()
            with device_lock:
                self._guard(lambda: select_project(body.get("project", "")))

        def _handle_cancel(self) -> None:
            with device_lock:
                self._guard(cancel_session)

        def _handle_run_delete(self) -> None:
            body = self._read_json_body()

            def run() -> dict:
                run_id = str(body.get("id") or "").strip()
                if not run_id:
                    raise ValueError("id is required")
                with device_lock:
                    return delete_run(run_id)

            self._guard(run)

        def _handle_settings(self) -> None:
            """Apply a live retune. Onset changes never touch the stream;
            capture changes reopen it."""
            body = self._read_json_body()

            def apply() -> dict:
                engine = state["engine"]
                onset = onset_updates(body.get("onset") or {})
                metadata = {
                    k: v for k, v in (body.get("metadata") or {}).items()
                }
                for key in ("tags", "microphone_type"):
                    if key in metadata:
                        metadata[key] = parse_tags(metadata[key])
                capture = body.get("capture") or {}
                if "channels" in capture:
                    capture["channels"] = parse_channels(capture["channels"])
                mode = body.get("mode")

                if engine is None:
                    # Idle: stage the change on the server's config so the next
                    # Start inherits it, and remember the form for a reload.
                    if onset:
                        from sed.data.record_engine import apply_onset_updates

                        state["cfg"].onset = apply_onset_updates(
                            state["cfg"].onset, onset
                        )
                    if capture:
                        if "device" in capture:
                            state["cfg"].deploy.record_device = str(capture["device"])
                        if "sample_rate" in capture:
                            state["cfg"].audio.record_sample_rate = int(
                                capture["sample_rate"]
                            )
                        if "blocksize" in capture:
                            state["cfg"].audio.blocksize = int(capture["blocksize"])
                        if "channels" in capture:
                            state["cfg"].audio.record_channels = capture["channels"]
                        state["cfg"].audio = type(state["cfg"].audio).model_validate(
                            state["cfg"].audio.model_dump()
                        )
                    form.update(metadata)
                    if mode:
                        form["mode"] = mode
                    return {"staged": True}

                if mode:
                    engine.set_mode(str(mode))
                    form["mode"] = str(mode)
                if onset:
                    engine.set_onset(**onset)
                if metadata:
                    engine.set_metadata(**metadata)
                    form.update(metadata)
                if capture:
                    with device_lock:
                        engine.set_capture(**capture)
                return {"session": engine.snapshot()}

            self._guard(apply)

        def _handle_save_settings(self) -> None:
            body = self._read_json_body()
            self._guard(lambda: save_settings(body))

        def _handle_trigger(self) -> None:
            def fire() -> dict:
                engine = state["engine"]
                if engine is None:
                    raise RuntimeError("start a session before capturing")
                return {"triggered": engine.trigger_manual()}

            self._guard(fire)

        def _handle_calibrate(self) -> None:
            body = self._read_json_body()

            def run() -> dict:
                seconds = float(body.get("seconds") or 10.0)
                strictness = str(body.get("strictness") or "normal")
                engine = state["engine"]
                if engine is not None:
                    return {"calibration": engine.start_calibration(seconds, strictness)}
                meter = state["meter"]
                if meter:
                    meter["calibrator"].start(seconds, strictness)
                    return {"calibration": meter["calibrator"].snapshot()}
                raise RuntimeError(
                    "start the meter or a session first — calibration listens to "
                    "the live input"
                )

            self._guard(run)

        def _handle_clip_delete(self) -> None:
            body = self._read_json_body()

            def run() -> dict:
                from sed.data.library import Library

                clip_uuid = str(body.get("uuid") or "").strip()
                if not clip_uuid:
                    raise ValueError("uuid is required")
                engine = state["engine"]
                removed = (
                    engine.delete_clip(clip_uuid) if engine is not None
                    else Library().remove_clip(clip_uuid)
                )
                if not removed:
                    raise FileNotFoundError(f"no library clip with uuid {clip_uuid}")
                return {"deleted": clip_uuid}

            self._guard(run)

        def _handle_meter_start(self) -> None:
            body = self._read_json_body()
            with device_lock:
                if state["engine"] is not None:
                    self._send_json(
                        {"ok": False,
                         "error": "stop the recording session before metering"},
                        status=409,
                    )
                    return
                self._guard(lambda: {"meter": start_meter(body)})

        def _handle_meter_stop(self) -> None:
            with device_lock:
                self._guard(lambda: (stop_meter(), {"meter": meter_snapshot()})[1])

        def _handle_done(self) -> None:
            with device_lock:
                stop_session()
                stop_meter()
            # Signal before answering. The other order lets the client see "ok"
            # while the server thread has not yet set the flag, so a caller that
            # checks whether shutdown began can win the race and find it has
            # not. `serve()` stops accepting new connections but lets this
            # handler finish writing, so the reply still lands.
            if shutdown_event is not None:
                shutdown_event.set()
            self._send_json({"ok": True})

        # ---- plumbing --------------------------------------------------

        def _guard(self, build) -> None:
            """Run a payload builder, mapping failures onto HTTP statuses.

            Recording is full of operator-fixable mistakes — a channel index
            past the end of the interface, a preroll longer than the clip — and
            each has to come back as its own sentence, not a 500.
            """
            try:
                payload = build()
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            except FileNotFoundError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=404)
                return
            except RuntimeError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=409)
                return
            except Exception as exc:  # noqa: BLE001
                log.exception("record API failure on %s", self.path)
                self._send_json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500
                )
                return
            if "ok" not in payload:
                payload = {"ok": True, **payload}
            self._send_json(payload)

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return {}

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
                self._send(404, b"not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    while chunk := f.read(64 * 1024):
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def cleanup() -> None:
        with device_lock:
            stop_session()
            stop_meter()

    Handler.cleanup = staticmethod(cleanup)
    return Handler


def serve(
    cfg: Config,
    project_id: str,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    prefill: dict | None = None,
) -> None:
    """Start the recording console. Blocks until Ctrl+C or Done in the browser."""
    shutdown_event = threading.Event()
    handler = make_handler(cfg, project_id, shutdown_event, prefill=prefill)

    for try_port in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, try_port), handler)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"could not bind any port in [{port},{port + 19}]")

    def _log_thread_exception(args) -> None:
        if args.exc_type is SystemExit:
            return
        log.error(
            "unhandled exception in background thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _log_thread_exception

    url = f"http://{host}:{httpd.server_port}/"
    log.info("recording console listening on %s", url)
    log.info("Ctrl+C, or press 'Done' in the browser, to stop.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

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
        cleanup = getattr(handler, "cleanup", None)
        if callable(cleanup):
            cleanup()
        httpd.shutdown()
        server_thread.join(timeout=2.0)
        time.sleep(0.05)
