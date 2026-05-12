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
import signal
import sys
import time
from dataclasses import dataclass

from rich.console import Console
from rich.live import Live
from rich.table import Table

from sed.config import Config, load_config_for_inference
from sed.lighting.artnet import ArtnetController, trigger_to_target
from sed.runtime.infer import RealtimeInference
from sed.runtime.smooth import TriggerEvent
from sed.runtime.trigger import EventPublisher

log = logging.getLogger(__name__)


@dataclass
class _LiveState:
    last_probs: dict[str, float]
    last_triggers: list[tuple[float, str, float]]      # (ts, cls, conf)
    started_at: float


def _build_live_table(state: _LiveState, classes: list[str], stats: dict) -> Table:
    uptime = time.monotonic() - state.started_at
    t = Table(title=f"SED — uptime {uptime:5.0f}s · "
                    f"infer {stats.get('infer_ms_avg', 0.0):4.1f} ms · "
                    f"frames {stats.get('frames', 0)} · "
                    f"triggers {stats.get('triggers', 0)}",
              show_lines=False, header_style="bold")
    t.add_column("class")
    t.add_column("smoothed", justify="right")
    t.add_column("bar")
    for cls in classes:
        prob = state.last_probs.get(cls, 0.0)
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        style = "bold red" if cls == "_negative" else "bold green"
        t.add_row(cls, f"{prob:0.3f}", f"[{style}]{bar}[/{style}]")

    # Per-mic section — surfaces dead microphones immediately
    per_mic = stats.get("per_mic_max_prob", {})
    if per_mic:
        t.add_section()
        for ch in sorted(per_mic.keys()):
            prob = per_mic[ch]
            trigs = stats.get("per_mic_triggers", {}).get(ch, 0)
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            # Dead-mic indicator: if ema-max stays below 0.05 it's probably muted
            tag = "🎤"
            if prob < 0.02:
                tag = "💀"
            t.add_row(f"{tag} mic {ch}", f"{prob:0.3f}",
                      f"[cyan]{bar}[/cyan] · {trigs} trig")

    if state.last_triggers:
        t.add_section()
        for ts, cls, conf in state.last_triggers[-6:]:
            ago = time.monotonic() - ts
            t.add_row(f"⏱ {ago:5.1f}s ago", f"{conf:0.3f}", f"→ {cls}")
    return t


def run(project_id: str, with_artnet: bool = True, live: bool = True,
        targets_override: list[str] | None = None) -> None:
    cfg = load_config_for_inference(project_id)
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Optional: override which target classes can fire ArtNet at runtime
    if targets_override is not None:
        for cls, t in cfg.targets.items():
            t.enabled = cls in targets_override

    publisher = EventPublisher(cfg.zmq.publish_endpoint)
    artnet = ArtnetController(cfg) if (with_artnet and cfg.artnet.enabled) else None

    console = Console()
    state = _LiveState(last_probs={}, last_triggers=[], started_at=time.monotonic())

    def on_trigger(ev: TriggerEvent) -> None:
        log.info("TRIGGER cls=%-20s conf=%.3f", ev.class_name, ev.confidence)
        publisher.publish(ev)
        state.last_triggers.append((ev.timestamp, ev.class_name, ev.confidence))
        state.last_triggers = state.last_triggers[-20:]
        if artnet is not None:
            target = trigger_to_target(cfg, ev.class_name)
            if target is not None:
                ch, value = target
                artnet.fire(ch, peak=value)

    def on_frame(probs: dict[str, float]) -> None:
        state.last_probs = probs

    runtime = RealtimeInference(
        cfg=cfg, project_id=project_id, on_trigger=on_trigger, on_frame=on_frame,
    )

    # Signal handlers
    stop = False

    def _sig(_s, _f):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    runtime.start()

    # Optional: notify systemd watchdog
    try:
        import systemd.daemon as sd                                # type: ignore
        sd.notify("READY=1")
        watchdog = True
    except ImportError:
        watchdog = False

    log.info("running — Ctrl+C to stop")

    try:
        if live and sys.stdout.isatty():
            with Live(_build_live_table(state, runtime.classes, runtime.stats),
                      console=console, refresh_per_second=4) as live_:
                while not stop:
                    live_.update(_build_live_table(state, runtime.classes, runtime.stats))
                    if watchdog:
                        try:
                            sd.notify("WATCHDOG=1")
                        except Exception:                          # noqa: BLE001
                            pass
                    time.sleep(0.25)
        else:
            while not stop:
                time.sleep(2.0)
                log.info(
                    "frames=%d triggers=%d infer=%.1fms per_mic=%s",
                    runtime.stats.get("frames", 0),
                    runtime.stats.get("triggers", 0),
                    runtime.stats.get("infer_ms_avg", 0.0),
                    {k: f"{v:.2f}" for k, v in runtime.stats.get("per_mic_max_prob", {}).items()},
                )
                if watchdog:
                    try:
                        sd.notify("WATCHDOG=1")
                    except Exception:                              # noqa: BLE001
                        pass
    finally:
        runtime.stop()
        if artnet is not None:
            artnet.stop()
        publisher.close()
        log.info("shutdown clean")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--no-artnet", action="store_true")
    p.add_argument("--no-live", action="store_true")
    p.add_argument("--targets", nargs="*", default=None)
    args = p.parse_args()
    run(args.project, with_artnet=not args.no_artnet, live=not args.no_live,
        targets_override=args.targets)
