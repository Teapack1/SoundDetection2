"""ArtNet output for trigger-driven lighting.

Uses `stupidArtnet` (MIT, threaded 30 Hz keep-alive). On each TriggerEvent we
schedule a per-class effect on a target channel: fade-in, hold, fade-out.
Effects are computed in real time on a background thread that ticks at the
configured FPS and writes the channel buffer.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from stupidArtnet import StupidArtnet

from sed.config import Config

log = logging.getLogger(__name__)


@dataclass
class Cue:
    """One running effect on the universe."""
    channel: int                                 # 1-indexed DMX channel
    start_ts: float
    fade_in_ms: int = 80
    hold_ms: int = 600
    fade_out_ms: int = 1000
    peak: int = 255                              # 0-255

    def value(self, now: float) -> int:
        elapsed_ms = (now - self.start_ts) * 1000
        if elapsed_ms < self.fade_in_ms:
            return int(self.peak * elapsed_ms / max(1, self.fade_in_ms))
        elapsed_ms -= self.fade_in_ms
        if elapsed_ms < self.hold_ms:
            return self.peak
        elapsed_ms -= self.hold_ms
        if elapsed_ms < self.fade_out_ms:
            return int(self.peak * (1.0 - elapsed_ms / max(1, self.fade_out_ms)))
        return 0

    def is_done(self, now: float) -> bool:
        total = self.fade_in_ms + self.hold_ms + self.fade_out_ms
        return (now - self.start_ts) * 1000 >= total


class ArtnetController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.universe = StupidArtnet(
            target_ip=cfg.artnet.target_ip,
            universe=cfg.artnet.universe,
            packet_size=512,
            fps=cfg.artnet.fps,
        )
        self.universe.start()
        self._cues: list[Cue] = []
        self._buf = bytearray(512)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._tick = threading.Thread(target=self._loop, daemon=True)
        self._tick.start()
        log.info(
            "ArtNet → %s universe %d @ %d fps",
            cfg.artnet.target_ip, cfg.artnet.universe, cfg.artnet.fps,
        )

    def fire(self, channel: int, peak: int = 255) -> None:
        with self._lock:
            self._cues.append(Cue(channel=channel, start_ts=time.monotonic(), peak=peak))

    def stop(self) -> None:
        self._stop.set()
        self._tick.join(timeout=1.0)
        try:
            self.universe.stop()
            self.universe.close()
        except Exception:  # noqa: BLE001
            pass

    def _loop(self) -> None:
        period = 1.0 / max(1, self.cfg.artnet.fps)
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                self._cues = [c for c in self._cues if not c.is_done(now)]
                values: dict[int, int] = {}
                for c in self._cues:
                    v = c.value(now)
                    values[c.channel] = max(values.get(c.channel, 0), v)
                for i in range(512):
                    self._buf[i] = values.get(i + 1, 0)
                self.universe.set(bytes(self._buf))
            time.sleep(period)


def trigger_to_target(cfg: Config, class_name: str) -> tuple[int, int] | None:
    """Look up the DMX (channel, value) for a triggered class.

    Returns None if the class isn't in [targets] or is disabled. The value
    is what the channel will fade up to at the cue peak — defaults to 255
    (full) but can be set per-class to use one channel for multiple classes
    (see TargetCfg docstring)."""
    target = cfg.targets.get(class_name)
    if target is None or not target.enabled:
        return None
    return int(target.channel), int(target.value)


# Back-compat alias — callers that only need the channel.
def trigger_to_channel(cfg: Config, class_name: str) -> int | None:
    ct = trigger_to_target(cfg, class_name)
    return ct[0] if ct else None
