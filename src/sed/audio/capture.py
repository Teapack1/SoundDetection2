"""Multi-microphone audio capture with a thread-safe ring buffer.

Built on PortAudio via `sounddevice`. Supports 1 to 4 channels (extends
trivially to more). Each channel is captured into its own ring buffer slot;
downstream code can pick any channel as the inference source, or mix them.

Design goals:
  - lock-free for the audio callback (uses a single atomic write index)
  - never drops samples in normal operation
  - graceful USB-disconnect recovery
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


def _sd():
    """Lazy import of sounddevice so the rest of this module is usable on
    machines without PortAudio (CI, test sandboxes, doc generation)."""
    import sounddevice as _sounddevice
    return _sounddevice


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------


@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float
    hostapi_name: str

    def __str__(self) -> str:
        return (
            f"[{self.index:2d}] {self.name}  "
            f"(in={self.max_input_channels}ch, "
            f"sr={self.default_samplerate:.0f} Hz, api={self.hostapi_name})"
        )


def list_input_devices() -> list[AudioDevice]:
    sd = _sd()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    out: list[AudioDevice] = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        out.append(
            AudioDevice(
                index=i,
                name=d["name"],
                max_input_channels=d["max_input_channels"],
                default_samplerate=d["default_samplerate"],
                hostapi_name=hostapis[d["hostapi"]]["name"],
            )
        )
    return out


def check_input_settings(
    *, device: str | int, channels: int, sample_rate: int, dtype: str = "float32"
) -> None:
    """Raise if this interface cannot deliver these settings.

    PortAudio only reports an unsupported rate or channel count when the stream
    is opened, and the supervisor treats that as a transient error and retries
    forever behind a log line — so an interface that simply cannot do 48 kHz
    presents as a session that is "recording" with meters at silence. Asking
    first turns that into one clear sentence before anything starts.
    """
    sd = _sd()
    index = resolve_device(device)
    try:
        sd.check_input_settings(
            device=index, channels=int(channels), dtype=dtype,
            samplerate=float(sample_rate),
        )
    except Exception as exc:  # noqa: BLE001 — reworded for an operator
        name = str(sd.query_devices(index).get("name") or index)
        raise ValueError(
            f"{name} cannot capture {channels} channel(s) at {sample_rate} Hz: {exc}"
        ) from exc


def resolve_device(name_or_index: str | int) -> int:
    """Accepts either a device index, a substring of the name, or 'default'."""
    sd = _sd()
    if name_or_index == "default":
        return sd.default.device[0]
    try:
        return int(name_or_index)
    except (TypeError, ValueError):
        pass
    needle = str(name_or_index).lower()
    devs = list_input_devices()
    exact = [d for d in devs if d.name.lower() == needle]
    if len(exact) == 1:
        return exact[0].index

    matches = [d for d in devs if needle in d.name.lower()]
    if len(matches) == 1:
        return matches[0].index
    if len(matches) > 1:
        choices = ", ".join(f"[{d.index}] {d.name}" for d in matches)
        raise ValueError(f"Ambiguous input device {name_or_index!r}: {choices}")
    raise ValueError(f"No input device matching {name_or_index!r}")


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------


class RingBuffer:
    """Float32 ring buffer of shape (capacity, channels)."""

    def __init__(self, capacity_samples: int, channels: int = 1):
        self.capacity = int(capacity_samples)
        self.channels = int(channels)
        self.buf = np.zeros((self.capacity, self.channels), dtype=np.float32)
        # write_idx is the absolute number of samples written since start
        # (modulo with `capacity` to get the ring position)
        self._write_idx = 0
        # Absolute index of the most recent stream restart, or None. Audio
        # either side of it is contiguous in the buffer but not in time.
        self.discontinuity_idx: int | None = None

    @property
    def write_idx(self) -> int:
        return self._write_idx

    def write(self, frames: np.ndarray) -> None:
        """frames: (n, channels) float32."""
        n = frames.shape[0]
        if n > self.capacity:
            raise ValueError(
                f"block of {n} samples exceeds ring capacity {self.capacity}"
            )
        if n == 0:
            return
        start = self._write_idx % self.capacity
        end = start + n
        if end <= self.capacity:
            self.buf[start:end] = frames
        else:
            split = self.capacity - start
            self.buf[start:] = frames[:split]
            self.buf[: end - self.capacity] = frames[split:]
        # Single 64-bit int store on CPython is atomic in practice; the
        # GIL makes this safe across our threads.
        self._write_idx += n

    def mark_discontinuity(self) -> None:
        """Record that the stream restarted here.

        The buffer is not reset across a reconnect — the sample clock simply
        stops while wall-clock time keeps going — so without this marker a
        window straddling the outage looks like ordinary contiguous audio.
        """
        self.discontinuity_idx = self._write_idx

    def read_last(self, n: int, channel: int = 0) -> np.ndarray:
        """Return the last `n` samples from one channel."""
        n = int(n)
        if n > self.capacity:
            raise ValueError(f"Requested {n} > capacity {self.capacity}")
        end = self._write_idx
        if n > end:
            # Before the stream has produced n samples the window reaches back
            # past the start of the recording; the modulo would wrap it onto
            # the unwritten tail of the buffer and return zeros as if they were
            # audio.
            raise ValueError(f"only {end} sample(s) captured so far, need {n}")
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        start = end - n
        s = start % self.capacity
        e = end % self.capacity
        if s < e:
            return self.buf[s:e, channel].copy()
        out = np.empty(n, dtype=np.float32)
        first = self.capacity - s
        out[:first] = self.buf[s:, channel]
        out[first:] = self.buf[:e, channel]
        return out

    def read_window(self, start_idx: int, n: int, channel: int = 0) -> np.ndarray:
        """Read n samples starting at absolute sample index `start_idx`.

        Lock-free against the audio callback, and validated on both sides of
        the copy. Checking only before it would prove the window was intact at
        that instant and nothing more: the writer keeps running during the
        copy, so a window close to the buffer's age could be overwritten
        underneath it and the caller would get a clip spliced from two
        different laps of the ring, silently. The second check turns that into
        an error, and callers size the ring so it stays rare.
        """
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        end = self._write_idx
        if start_idx + n > end:
            raise ValueError(
                f"Window [{start_idx}, {start_idx + n}) not yet written (have {end})"
            )
        if start_idx < 0 or end - start_idx > self.capacity:
            raise ValueError("Window already overwritten in ring buffer")
        gap = self.discontinuity_idx
        if gap is not None and start_idx < gap < start_idx + n:
            # The stream restarted inside this window: the samples on either
            # side of `gap` are adjacent in the buffer but seconds apart in
            # time. Splicing them would write a clip with a step in it.
            raise ValueError("Window spans an audio dropout")
        s = start_idx % self.capacity
        e = (start_idx + n) % self.capacity
        if s < e:
            out = self.buf[s:e, channel].copy()
        else:
            out = np.empty(n, dtype=np.float32)
            first = self.capacity - s
            out[:first] = self.buf[s:, channel]
            out[first:] = self.buf[:e, channel]
        if self._write_idx - start_idx > self.capacity:
            raise ValueError("Window was overwritten while being read")
        return out


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class AudioCapture:
    """Continuous capture into a ring buffer.

    Optionally runs an extra callback on each block (for onset detection).
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        channels: int = 1,
        device: str | int = "default",
        blocksize: int = 1024,
        ring_seconds: float = 4.0,
        on_block: Callable[[np.ndarray, int], None] | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = resolve_device(device)
        self.blocksize = blocksize
        self.ring = RingBuffer(int(ring_seconds * sample_rate), channels=channels)
        self._on_block = on_block

        self._stream = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._first_error = ""
        self._supervisor: threading.Thread | None = None

        self.stats = {"xruns": 0, "blocks": 0, "reconnects": 0}
        # Stream health, for callers that have no terminal to watch. Without
        # this a stream that never opens is indistinguishable from a silent room.
        self.state = "idle"          # idle | starting | running | reconnecting
        self.last_error = ""
        # Rate-limit the "sustained overflow" warning (audio thread).
        self._last_xrun_warn = 0.0
        self._xruns_at_last_warn = 0

    # ---- public API ---------------------------------------------------

    def start(self, wait_s: float = 3.0) -> None:
        """Open the stream, waiting briefly for the first attempt to resolve.

        Returning before the first open would report success for a device that
        cannot deliver these settings at all — the supervisor would retry it
        forever behind a log line while the caller believed it was recording.
        """
        self._stop.clear()
        self._started.clear()
        self._first_error = ""
        self._supervisor = threading.Thread(target=self._supervise, daemon=True)
        self._supervisor.start()
        if wait_s > 0 and self._started.wait(timeout=wait_s) and self._first_error:
            self.stop()
            raise RuntimeError(self._first_error)

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
        if self._supervisor is not None:
            self._supervisor.join(timeout=2.0)

    # ---- internals ----------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        _ = time_info
        if status:
            self.stats["xruns"] += 1
            log.debug("PortAudio status: %s", status)
            # Occasional overflow is harmless (a few ms of dropped input). Only
            # warn — rate-limited — when drops are SUSTAINED (>=5 in a 10 s
            # window), which signals real audio loss (raise blocksize / reduce
            # CPU load). Per-event noise stays at DEBUG above.
            now = time.monotonic()
            if (now - self._last_xrun_warn >= 10.0
                    and self.stats["xruns"] - self._xruns_at_last_warn >= 5):
                log.warning(
                    "sustained audio input overflow: %d dropped-buffer events "
                    "(raise [audio] blocksize or reduce CPU load)",
                    self.stats["xruns"],
                )
                self._last_xrun_warn = now
                self._xruns_at_last_warn = self.stats["xruns"]
        block = indata.copy()
        self.ring.write(block)
        self.stats["blocks"] += 1
        if self._on_block is not None:
            try:
                self._on_block(block, self.ring.write_idx)
            except Exception:  # noqa: BLE001
                log.exception("on_block callback raised")

    def _open_stream(self):
        sd = _sd()
        log.debug(
            "Opening input stream: device=%s ch=%d sr=%d block=%d",
            self.device, self.channels, self.sample_rate, self.blocksize,
        )
        return sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._audio_callback,
        )

    def _supervise(self) -> None:
        """Restart the stream on errors (e.g. USB disconnects)."""
        sd = _sd()
        backoff = 0.5
        opened_once = False
        while not self._stop.is_set():
            try:
                self.state = "starting"
                self._stream = self._open_stream()
                self._stream.start()
                if opened_once:
                    # Coming back from a dropout. The buffer keeps its sample
                    # clock across the gap, so mark where the join is: audio
                    # either side of it is adjacent in the ring but seconds
                    # apart in time, and a clip spanning it would have a step
                    # in the middle.
                    self.ring.mark_discontinuity()
                    self.stats["reconnects"] += 1
                opened_once = True
                self.state = "running"
                self.last_error = ""
                self._started.set()
                backoff = 0.5
                while not self._stop.is_set():
                    if not self._stream.active:
                        raise sd.PortAudioError("stream became inactive")
                    time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001
                if self._stop.is_set():
                    break
                log.warning("Audio stream error: %s — reconnecting in %.1fs", exc, backoff)
                self.state = "reconnecting"
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._first_error = self._first_error or self.last_error
                self._started.set()      # release start(); the error is readable
                try:
                    if self._stream is not None:
                        self._stream.close()
                except Exception:  # noqa: BLE001
                    pass
                self._stream = None
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
        self.state = "idle"
