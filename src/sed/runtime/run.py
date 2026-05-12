"""Main 24/7 runtime entry.

Wires together:
    AudioCapture → RealtimeInference → TriggerArbiter → (terminal log + ZMQ pub + ArtNet)

Design:
  - Watchdog: pings systemd if Type=notify, so the unit auto-restarts on hang.
  - Clean shutdown on SIGTERM / SIGINT.
  - Optional terminal "live" view (probabilities + recent triggers) via Rich.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table

from sed.config import (
    Config,
    ProjectLayout,
    load_config_for_inference,
    resolve_active_model_dir,
)
from sed.lighting.artnet import ArtnetController, trigger_to_target
from sed.runtime.infer import RealtimeInference, RuntimeDetection
from sed.runtime.manifest import resolve_model_path
from sed.runtime.mining import DetectionMiner
from sed.runtime.smooth import TriggerEvent
from sed.runtime.trigger import EventPublisher
from sed.train.objective import probability_mode

log = logging.getLogger(__name__)

# Third-party libraries that emit very noisy DEBUG logs (e.g. numba dumps its
# entire JIT bytecode compilation on first call). We never want their DEBUG
# output even when --verbose raises the root level, so pin them to WARNING.
_NOISY_LOGGERS = ("numba", "matplotlib", "PIL", "asyncio", "h5py")


def quiet_noisy_loggers() -> None:
    """Keep chatty third-party loggers at WARNING regardless of root level."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def sd_notify(state: str) -> bool:
    """Send a message to the systemd notify socket (e.g. "READY=1",
    "WATCHDOG=1"). Pure-Python implementation of the sd_notify protocol so the
    runtime needs no C-extension dependency (systemd-python) on the edge.

    Returns True if the message was sent. False when not run under systemd
    Type=notify (no NOTIFY_SOCKET) or on any socket error — callers treat that
    as "watchdog/readiness disabled" and keep running normally.
    """
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    # Abstract-namespace sockets start with '@' (or NUL); map to NUL prefix.
    if addr[0] == "@":
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(addr)
            sock.sendall(state.encode("utf-8"))
        return True
    except OSError:
        return False


@dataclass
class _LiveState:
    last_probs: dict[str, float]
    last_detections: list[dict[str, object]]
    started_at: float
    freeze_until: float = 0.0
    freeze_probs: dict[str, float] | None = None
    freeze_class: str | None = None


def _build_live_table(
    state: _LiveState,
    classes: list[str],
    stats: dict,
    active_targets: set[str],
) -> Table:
    from rich.panel import Panel
    from rich.table import Table

    now = time.monotonic()
    uptime = now - state.started_at
    frozen = state.freeze_probs is not None and now < state.freeze_until
    probs = state.freeze_probs if frozen else stats.get("last_smoothed", {}) or state.last_probs
    windows = stats.get("windows", stats.get("frames", 0))
    detections = stats.get("triggers", 0)
    mined = stats.get("mined_saved", 0)
    infer_ms = stats.get("infer_ms_avg", 0.0)

    # Root grid to organize the dashboard layout
    root = Table.grid(expand=True)
    root.add_column("main")

    # Header Panel (Status and Stats)
    if frozen:
        status_str = f"[bold white on red] DETECTION: {state.freeze_class} [/bold white on red]"
    elif stats.get("amplitude_gate"):
        status_str = "[bold black on yellow] SILENT [/bold black on yellow]"
    else:
        status_str = "[bold black on green] LISTENING [/bold black on green]"

    header_text = (
        f"{status_str}   "
        f"[dim]Uptime:[/dim] [bold]{uptime:5.0f}s[/bold]   "
        f"[dim]Infer Avg:[/dim] [bold]{infer_ms:4.1f} ms[/bold]   "
        f"[dim]Windows:[/dim] [bold]{windows}[/bold]   "
        f"[dim]Detections:[/dim] [bold green]{detections}[/bold green]   "
        f"[dim]Mined:[/dim] [bold]{mined}[/bold]"
    )
    header_panel = Panel(
        header_text,
        border_style="cyan",
        title="[bold]SED Inference Live Monitor[/bold]",
        title_align="left",
    )
    root.add_row(header_panel)
    root.add_row("")

    # Sound Event Classes monitoring table
    classes_table = Table(box=None, expand=True, show_header=True, header_style="bold cyan")
    classes_table.add_column("Class", justify="left")
    classes_table.add_column("Level", justify="center", width=17)
    classes_table.add_column("Value", justify="right")
    classes_table.add_column("Limit/State", justify="left")

    neg_name = "_negative" if "_negative" in classes else None
    negative_max = stats.get("negative_max")
    thresholds = stats.get("class_thresholds", {})

    for cls in classes:
        prob = probs.get(cls, 0.0)
        is_active = cls in active_targets

        if frozen and cls == state.freeze_class:
            cls_styled = f"[bold red]★ {cls}[/bold red]"
        elif is_active:
            cls_styled = f"[bold green]{cls}[/bold green]"
        else:
            cls_styled = f"[dim]{cls}[/dim]"

        # Generate custom bar
        bar_width = 15
        filled = max(0, min(bar_width, int(prob * bar_width)))
        empty = bar_width - filled

        if cls == neg_name:
            bar_color = "yellow" if (negative_max is not None and prob > float(negative_max)) else "dim yellow"
        elif is_active:
            threshold = thresholds.get(cls, 0.9)
            bar_color = "green" if prob >= threshold else "cyan"
        else:
            bar_color = "dim white"

        bar_str = f"[{bar_color}]" + "█" * filled + "░" * empty + f"[/{bar_color}]"

        if cls == neg_name:
            limit = f"limit {float(negative_max):.2f}" if negative_max is not None else "observation"
            state_text = "blocking" if negative_max is not None and prob > float(negative_max) else "clear"
        else:
            threshold = thresholds.get(cls)
            limit = f"threshold {float(threshold):.2f}" if threshold is not None else "threshold n/a"
            if not is_active:
                state_text = "monitor"
            elif threshold is not None and prob >= float(threshold):
                state_text = "ready"
            else:
                state_text = "listening"

        if frozen and cls == state.freeze_class:
            state_text = f"[bold]{state_text}[/bold]"

        classes_table.add_row(
            cls_styled,
            bar_str,
            f"{prob:0.3f}",
            f"{limit} | {state_text}"
        )

    classes_panel = Panel(
        classes_table,
        border_style="green",
        title="[bold green]Sound Event Classes[/bold green]",
        title_align="left",
    )

    root.add_row(classes_panel)
    root.add_row("")

    # Microphone monitoring table
    mics_table = Table(box=None, expand=True, show_header=True, header_style="bold cyan")
    mics_table.add_column("Microphone", justify="left", width=15)
    mics_table.add_column("Level", justify="center", width=14)
    mics_table.add_column("dBFS (Peak/RMS)", justify="left", width=30)
    mics_table.add_column("Onset / Gate Status", justify="left", width=32)
    mics_table.add_column("Top Class", justify="left", width=25)
    mics_table.add_column("Detections", justify="left")

    mic_status = stats.get("mic_status", {})
    if mic_status:
        for ch in sorted(mic_status.keys(), key=lambda v: int(v)):
            status = mic_status[ch]
            mic_name = _format_mic_name(int(ch), mic_status)
            mic_name_styled = f"[bold cyan]{mic_name}[/bold cyan]" if status.get("detections", 0) > 0 else mic_name

            peak_dbfs = status.get("peak_dbfs", -120.0)
            rms_dbfs = status.get("rms_dbfs", -120.0)

            # Draw mic peak bar
            filled_mic = max(0, min(12, int((peak_dbfs + 72.0) / 72.0 * 12)))
            empty_mic = 12 - filled_mic
            mic_bar_color = "red" if peak_dbfs >= -3.0 else ("green" if peak_dbfs >= -18.0 else "dim green")
            mic_bar = f"[{mic_bar_color}]" + "█" * filled_mic + "░" * empty_mic + f"[/{mic_bar_color}]"

            peak_rms_text = f"peak {peak_dbfs:.1f} / rms {rms_dbfs:.1f} dBFS"
            onset_pair = _format_onset_pair(status)
            top_class = _format_top_class(status)
            dets = status.get("detections", 0)

            mics_table.add_row(
                mic_name_styled,
                mic_bar,
                peak_rms_text,
                f"onset {onset_pair}",
                f"top {top_class}",
                f"detections {dets}"
            )
    else:
        mics_table.add_row("[dim]No microphones active[/dim]", "", "", "", "", "")

    mics_panel = Panel(
        mics_table,
        border_style="cyan",
        title="[bold cyan]Microphone Inputs[/bold cyan]",
        title_align="left",
    )

    root.add_row(mics_panel)
    root.add_row("")

    # Bottom Area: Recent Detections Panel
    detections_table = Table(box=None, expand=True)
    detections_table.add_column("Time", width=12, style="dim")
    detections_table.add_column("Class & Score", width=25)
    detections_table.add_column("Microphone & Level", width=35)
    detections_table.add_column("Saved File Destination", justify="left")

    if state.last_detections:
        for det in state.last_detections[-6:]:
            ago = time.monotonic() - float(det.get("ts", time.monotonic()))
            saved = _mined_relpath(det.get("saved"))
            mic_text = _format_mic_name(det.get("mic"), mic_status)
            peak_text = _format_optional_dbfs(det.get("peak_dbfs"))

            time_str = f"{ago:5.1f}s ago"
            class_str = f"[bold green]{det.get('class', '?')}[/bold green] {float(det.get('score', 0.0)):.3f}"
            mic_info = f"{mic_text} | peak {peak_text}"

            saved_str = f"saved {saved}" if saved else ""

            detections_table.add_row(
                time_str,
                class_str,
                mic_info,
                saved_str
            )
    else:
        detections_table.add_row("[dim]No recent detections[/dim]", "", "", "")

    recent_panel = Panel(
        detections_table,
        border_style="yellow",
        title="[bold yellow]Recent Detections[/bold yellow]",
        title_align="left",
    )
    root.add_row(recent_panel)
    return root



def run(
    project_id: str,
    with_artnet: bool | None = None,
    live: bool | None = None,
    targets_override: list[str] | None = None,
    verbose_level: str | None = None,
    mine_detections: bool | None = None,
    min_dbfs: float | None = None,
    device: str | None = None,
) -> None:
    cfg = load_config_for_inference(project_id)
    if min_dbfs is not None:
        cfg.inference.min_dbfs = float(min_dbfs)
    if device is not None:
        cfg.deploy.record_device = device
    resolved_verbose = _resolve_verbose_level(cfg, verbose_level)
    verbose = resolved_verbose in ("info", "live")
    live = (resolved_verbose == "live") if live is None else bool(live)
    with_artnet = True if with_artnet is None else bool(with_artnet)
    # deploy.verbose controls operator chatter; [logging] level only refines
    # Python logger verbosity once chatter is enabled.
    logging.basicConfig(
        level=_resolve_log_level(cfg, verbose),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    quiet_noisy_loggers()

    active_requested = _active_requested_classes(cfg, targets_override)
    active_trigger_classes = {c for c in active_requested if c != "_negative"}

    # Optional: override which target classes can fire ArtNet at runtime.
    # The same active set is also passed to the arbiter, so non-selected target
    # classes are visible in verbose probabilities but cannot emit triggers.
    if targets_override is not None:
        for cls, t in cfg.targets.items():
            t.enabled = cls in targets_override
    if verbose:
        log.info(
            "active detection classes: %s",
            sorted(active_trigger_classes) if active_trigger_classes else "none",
        )
        if "_negative" in active_requested:
            log.info("_negative selected for observation only; it does not send ArtNet")

    publisher = EventPublisher(cfg.zmq.publish_endpoint)
    artnet = ArtnetController(cfg) if (with_artnet and cfg.artnet.enabled) else None

    console = Console()
    state = _LiveState(last_probs={}, last_detections=[], started_at=time.monotonic())
    mining_enabled = cfg.mining.enabled if mine_detections is None else bool(mine_detections)
    # Mining class selection comes from [targets] (classes with mine = true,
    # which may include _negative). If none are flagged, fall back to the active
    # trigger classes so `--mine` alone still captures the obvious detections.
    flagged_mining = {c for c in cfg.mining_classes() if c in cfg.model.classes}
    mining_classes = (
        (flagged_mining or {c for c in active_requested if c in cfg.model.classes})
        if mining_enabled else set()
    )
    miner = (
        DetectionMiner(
            cfg=cfg,
            project_id=project_id,
            classes=mining_classes,
        )
        if mining_enabled and mining_classes else None
    )
    if miner is not None and verbose:
        log.info("runtime mining enabled: classes=%s dir=%s", sorted(mining_classes), miner.root)

    if resolved_verbose != "none":
        for line in _startup_lines(
            cfg=cfg,
            project_id=project_id,
            active_targets=active_trigger_classes,
            with_artnet=with_artnet,
            artnet_enabled=artnet is not None,
            live=live,
            verbose=verbose,
            mining_enabled=miner is not None,
        ):
            console.print(line)

    def on_trigger(ev: TriggerEvent) -> None:
        console.print(
            _format_trigger_line(ev, runtime.stats.get("mic_status", {}), verbose_level=resolved_verbose),
            highlight=False,
            soft_wrap=True,
        )
        publisher.publish(ev)
        state.last_detections.append({
            "ts": ev.timestamp,
            "class": ev.class_name,
            "score": ev.confidence,
            "mic": ev.best_channel,
            "peak_dbfs": ev.peak_dbfs,
            "saved": ev.mined_path,
        })
        state.last_detections = state.last_detections[-20:]
        state.freeze_until = time.monotonic() + 1.0
        state.freeze_probs = dict(runtime.stats.get("last_smoothed", {}) or state.last_probs)
        state.freeze_class = ev.class_name
        if artnet is not None:
            target = trigger_to_target(cfg, ev.class_name)
            if target is not None:
                ch, value = target
                artnet.fire(ch, peak=value)

    def on_frame(probs: dict[str, float]) -> None:
        state.last_probs = probs

    def on_detection(det: RuntimeDetection):
        if miner is not None:
            return miner.maybe_save(det)
        return None

    runtime = RealtimeInference(
        cfg=cfg,
        project_id=project_id,
        on_trigger=on_trigger,
        on_frame=on_frame,
        on_detection=on_detection if miner is not None else None,
        active_trigger_classes=active_trigger_classes,
        detection_classes=mining_classes,
        live_monitor=bool(live and verbose),
    )

    # Signal handlers
    stop = False

    def _sig(_s, _f):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    runtime.start()
    if resolved_verbose != "none":
        console.print("READY inference running; detection lines will appear below")
    else:
        console.print("[dim]Listening... (Ctrl+C to stop)[/dim]")

    # Tell systemd we are up (Type=notify). watchdog stays True only when run
    # under systemd; otherwise sd_notify is a harmless no-op.
    watchdog = sd_notify("READY=1")

    if verbose:
        console.print(
            "[dim]Ctrl+C to stop. Live monitor shows class scores, "
            "microphone levels, onset levels, and recent detections.[/dim]"
        )
    last_verbose = 0.0
    last_onset_windows = 0
    mining_on = miner is not None
    try:
        if verbose and live and sys.stdout.isatty():
            with Live(_build_live_table(
                state, runtime.classes, runtime.stats, active_trigger_classes,
            ),
                      console=console, refresh_per_second=4) as live_:
                while not stop:
                    live_.update(_build_live_table(
                        state, runtime.classes, runtime.stats, active_trigger_classes,
                    ))
                    if watchdog:
                        sd_notify("WATCHDOG=1")
                    time.sleep(0.25)
        else:
            while not stop:
                time.sleep(0.2)
                now = time.monotonic()
                if verbose:
                    if runtime.onset_gating:
                        # Onset-gated: log one line each time the onset threshold is
                        # passed, plus a slow heartbeat so silence still shows life.
                        onset_windows = runtime.stats.get("onset_windows", 0)
                        if onset_windows > last_onset_windows:
                            last_onset_windows = onset_windows
                            last_verbose = now
                            _log_verbose_frame(runtime, active_trigger_classes, console, mining_on)
                        elif now - last_verbose >= 60.0:
                            last_verbose = now
                            _log_verbose_frame(runtime, active_trigger_classes, console, mining_on)
                    elif now - last_verbose >= 1.0:
                        # Sliding mode: steady ~1 Hz frame line.
                        last_verbose = now
                        _log_verbose_frame(runtime, active_trigger_classes, console, mining_on)
                if watchdog:
                    sd_notify("WATCHDOG=1")
    finally:
        runtime.stop()
        if artnet is not None:
            artnet.stop()
        publisher.close()
        if verbose:
            log.info("shutdown clean")


def _active_requested_classes(cfg: Config, targets_override: list[str] | None) -> set[str]:
    known = set(cfg.model.classes)
    if targets_override is not None:
        unknown = sorted(set(targets_override) - known)
        if unknown:
            log.warning("ignoring unknown --target classes: %s", unknown)
        return {c for c in targets_override if c in known}

    configured = {
        cls for cls, target in cfg.targets.items()
        if cls in known and cls != "_negative" and target.enabled
    }
    if configured:
        return configured
    return {c for c in cfg.model.classes if c != "_negative"}


def _decision_status(
    stats: dict,
    smoothed: dict[str, float],
    neg_name: str | None,
    negative_max: float | None,
) -> str:
    if stats.get("amplitude_gate"):
        return "below_audio_floor"
    if (
        neg_name is not None
        and negative_max is not None
        and smoothed.get(neg_name, 0.0) > float(negative_max)
    ):
        return "blocked_negative"
    return "listening"


def _audio_floor_state(stats: dict) -> str:
    min_dbfs = stats.get("min_dbfs")
    if min_dbfs is None:
        return "off"
    return "below_audio_floor" if stats.get("amplitude_gate") else "clear"


def _format_audio_floor(min_dbfs: object) -> str:
    if min_dbfs is None:
        return "off"
    return f"{float(min_dbfs):.1f} dBFS"


def _format_optional_dbfs(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}dBFS"


def _format_onset_pair(status: dict) -> str:
    onset = status.get("onset_db")
    gate = status.get("onset_gate_db")
    if onset is None or gate is None:
        return "off"
    return f"{float(onset):.1f} / gate {float(gate):.1f} dB"


def _format_top_class(status: dict) -> str:
    cls = status.get("top_class")
    if not cls:
        return "n/a"
    return f"{cls} {float(status.get('top_score', 0.0)):.3f}"


def _format_mic_name(channel: object, mic_status: dict | None = None) -> str:
    if channel is None:
        return "unknown"
    ch = int(channel)
    label = ""
    if mic_status:
        status = mic_status.get(ch)
        if status is None:
            status = mic_status.get(str(ch))
        if isinstance(status, dict):
            label = str(status.get("label", "") or "").strip()
    if label and label.lower() not in {"generic", "unknown"}:
        return f"ch{ch} {label}"
    return f"ch{ch}"


def _format_startup_mics(channels: list[int], labels: list[str]) -> str:
    parts = []
    for i, ch in enumerate(channels):
        label = ""
        if len(labels) == len(channels):
            label = labels[i]
        elif 0 <= ch < len(labels):
            label = labels[ch]
        elif len(labels) == 1:
            label = labels[0]
        label = str(label or "").strip()
        if label and label.lower() not in {"generic", "unknown"}:
            parts.append(f"ch{ch} {label}")
        else:
            parts.append(f"ch{ch}")
    return ", ".join(parts) if parts else "none"


def _mined_relpath(value: object) -> str:
    if not value:
        return ""
    path = str(value)
    idx = path.find("/mined/")
    if idx != -1:
        return path[idx + 1:]
    return path.rsplit("/", 1)[-1]


def _log_verbose_frame(
    runtime: RealtimeInference,
    active_targets: set[str],
    console: Console,
    mining: bool = False,
) -> None:
    """Print one compact line for the latest inference hop.

    Shows smoothed probabilities only (what drives decisions). The status token
    uses the same names as the live dashboard, and ``input_drop`` appears only
    when the sound card dropped buffers since the last line. With onset gating
    this is called once per onset (each time the onset threshold is passed).
    """
    stats = runtime.stats
    smoothed = stats.get("last_smoothed", {})
    classes = runtime.classes or sorted(smoothed)

    arbiter = getattr(runtime, "arbiter", None)
    thresholds = {c: s.threshold for c, s in arbiter.smoothers.items()} if arbiter else {}
    negative_max = getattr(arbiter, "negative_max", None)
    neg_name = (
        arbiter.classes[arbiter.neg_idx]
        if arbiter is not None and arbiter.neg_idx is not None else None
    )

    def _rank(c: str) -> tuple[int, str]:
        return (0 if c in active_targets else (2 if c == neg_name else 1), c)

    ordered = sorted(classes, key=_rank)

    peak = stats.get("audio_peak_dbfs", -120.0)
    rms = stats.get("audio_rms_dbfs", -120.0)
    status = _decision_status(stats, smoothed, neg_name, negative_max)

    cells = []
    for c in ordered:
        s = smoothed.get(c, 0.0)
        if c == neg_name:
            limit = f"/{float(negative_max):.2f}" if negative_max is not None else ""
            cells.append(f"negative={s:.2f}{limit}")
        elif c in active_targets:
            threshold = thresholds.get(c)
            limit = f"/{float(threshold):.2f}" if threshold is not None else ""
            cells.append(f"{c}={s:.2f}{limit}")
        else:
            cells.append(f"{c}={s:.2f}")

    # Audio drops since the last frame (PortAudio input overflow).
    xruns = runtime.capture.stats.get("xruns", 0) if getattr(runtime, "capture", None) else 0
    delta = xruns - getattr(runtime, "_last_xruns_shown", 0)
    runtime._last_xruns_shown = xruns
    drop = f" input_drop={delta}" if delta > 0 else ""

    mined = f" mined={runtime.stats.get('mined_saved', 0)}" if mining else ""

    wall = datetime.now().strftime("%H:%M:%S")
    console.print(
        (
            f"{wall} status={status} peak={peak:.1f}dBFS "
            f"rms={rms:.1f}dBFS "
        ) + " ".join(cells) + mined + drop,
        highlight=False, soft_wrap=True,
    )


def _startup_lines(
    cfg: Config,
    project_id: str,
    active_targets: set[str],
    with_artnet: bool,
    artnet_enabled: bool,
    live: bool,
    verbose: bool,
    mining_enabled: bool,
) -> list[str]:
    layout = ProjectLayout(project_id)
    model_dir = resolve_active_model_dir(layout, cfg.deploy.active_model)
    model_path = resolve_model_path(model_dir)
    active_model = cfg.deploy.active_model or model_dir.name
    audio_floor = _format_audio_floor(cfg.inference.min_dbfs)
    mic_summary = _format_startup_mics(
        cfg.inference.inference_channels,
        cfg.audio.microphone_type,
    )
    return [
        (
            f"Project: {project_id} | active model: {active_model}"
        ),
        (
            f"Model file: {model_path}"
        ),
        (
            f"Classes: {len(cfg.model.classes)} | active targets: "
            f"{','.join(sorted(active_targets)) or 'none'}"
        ),
        (
            f"Audio: device {cfg.deploy.record_device!r} | mics {mic_summary} | "
            f"capture {cfg.audio.record_sample_rate} Hz -> features {cfg.model.sample_rate} Hz"
        ),
        (
            f"Decision: window {cfg.features.window_seconds:.3g}s | "
            f"hop {cfg.inference.hop_ms} ms | probabilities {probability_mode(cfg)} | "
            f"audio_floor {audio_floor} | onset trigger "
            f"{'on' if cfg.inference.onset_gating_enabled else 'off'}"
        ),
        (
            f"Outputs: ArtNet {'on' if artnet_enabled else 'off'}"
            f"{' (disabled by --no-artnet)' if not with_artnet else ''} "
            f"| mining {'on' if mining_enabled else 'off'} "
            f"| terminal verbose {'on' if verbose else 'off'} "
            f"| live monitor {'on' if live and verbose else 'off'}"
        ),
    ]


def _format_heartbeat(runtime: RealtimeInference) -> str:
    stats = runtime.stats
    windows = stats.get("windows", stats.get("frames", 0))
    return (
        "HEARTBEAT "
        f"windows={windows} "
        f"detections={stats.get('triggers', 0)} "
        f"peak={stats.get('audio_peak_dbfs', -120.0):.1f}dBFS "
        f"rms={stats.get('audio_rms_dbfs', -120.0):.1f}dBFS "
        f"audio_floor={_audio_floor_state(stats)} "
        f"below_floor={stats.get('amplitude_skipped', 0)}"
    )


def _format_trigger_line(
    ev: TriggerEvent,
    mic_status: dict | None = None,
    verbose_level: str = "info",
) -> str:
    wall = datetime.now().strftime("%H:%M:%S")
    mic_text = _format_mic_name(ev.best_channel, mic_status or {})

    if verbose_level == "none":
        return f"[dim][{wall}][/dim] [cyan]{mic_text}[/cyan]: [bold green]{ev.class_name}[/bold green]"

    peak_text = _format_optional_dbfs(ev.peak_dbfs)
    if ev.mined_path:
        saved = _mined_relpath(ev.mined_path) or ev.mined_path.rsplit("/", 1)[-1]
    else:
        saved = "not_mined"
    return (
        f"{wall}  DETECTION class={ev.class_name} score={ev.confidence:.2f} "
        f"mic={mic_text} peak={peak_text} saved={saved}"
    )


def _parse_target_csv(value: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    targets = [
        part.strip()
        for item in values
        for part in str(item).split(",")
        if part.strip()
    ]
    return targets or None


def _resolve_verbose_level(cfg: Config, override: str | None = None) -> str:
    level = (override if override is not None else cfg.deploy.verbose) or "none"
    level = str(level).lower()
    if level not in {"none", "info", "live"}:
        raise ValueError("verbose level must be one of: none, info, live")
    return level


def _resolve_log_level(cfg: Config, verbose: bool) -> int:
    configured = str(getattr(cfg.logging, "level", "INFO") or "INFO").upper()
    level = logging._nameToLevel.get(configured, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO
    # deploy.verbose controls whether runtime is chatty at all. logging.level
    # can make verbose runs more/less detailed, but cannot make "none" chatty.
    return level if verbose else max(logging.WARNING, level)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--no-artnet", action="store_true")
    p.add_argument("--target", action="append", default=None)
    p.add_argument("--verbose", choices=("none", "info", "live"),
                   nargs="?", const="live", default=None)
    p.add_argument("--mine", "--mine-detections", dest="mine_detections", action="store_true")
    p.add_argument("--min-dbfs", type=float, default=None)
    args = p.parse_args()
    targets = _parse_target_csv(args.target)
    run(args.project, with_artnet=False if args.no_artnet else None,
        live=args.verbose == "live" if args.verbose is not None else None,
        targets_override=targets, verbose_level=args.verbose,
        mine_detections=True if args.mine_detections else None,
        min_dbfs=args.min_dbfs)
