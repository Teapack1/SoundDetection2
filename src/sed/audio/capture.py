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
        self._lock = threading.Lock()

    @property
    def write_idx(self) -> int:
        return self._write_idx

    def write(self, frames: np.ndarray) -> None:
        """frames: (n, channels) float32."""
        n = frames.shape[0]
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

    def read_last(self, n: int, channel: int = 0) -> np.ndarray:
        """Return the last `n` samples from one channel."""
        n = int(n)
        if n > self.capacity:
            raise ValueError(f"Requested {n} > capacity {self.capacity}")
        end = self._write_idx
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
        """Read n samples starting at absolute sample index `start_idx`."""
        end = self._write_idx
        if start_idx + n > end:
            raise ValueError(
                f"Window [{start_idx}, {start_idx + n}) not yet written (have {end})"
            )
        if end - start_idx > self.capacity:
            raise ValueError("Window already overwritten in ring buffer")
        s = start_idx % self.capacity
        e = (start_idx + n) % self.capacity
        if s < e:
            return self.buf[s:e, channel].copy()
        out = np.empty(n, dtype=np.float32)
        first = self.capacity - s
        out[:first] = self.buf[s:, channel]
        out[first:] = self.buf[:e, channel]
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
        gain_db: float = 0.0,
        on_block: Callable[[np.ndarray, int], None] | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = resolve_device(device)
        self.blocksize = blocksize
        self.gain = float(10 ** (gain_db / 20.0))
        self.ring = RingBuffer(int(ring_seconds * sample_rate), channels=channels)
        self._on_block = on_block

        self._stream = None
        self._stop = threading.Event()
        self._supervisor: threading.Thread | None = None

        self.stats = {"xruns": 0, "blocks": 0, "reconnects": 0}

    # ---- public API ---------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._supervisor = threading.Thread(target=self._supervise, daemon=True)
        self._supervisor.start()

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
        if status:
            self.stats["xruns"] += 1
            log.debug("PortAudio status: %s", status)
        block = indata.copy()
        if self.gain != 1.0:
            block *= self.gain
            np.clip(block, -1.0, 1.0, out=block)
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
        while not self._stop.is_set():
            try:
                self._stream = self._open_stream()
                self._stream.start()
                backoff = 0.5
                while not self._stop.is_set():
                    if not self._stream.active:
                        raise sd.PortAudioError("stream became inactive")
                    time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001
                log.warning("Audio stream error: %s — reconnecting in %.1fs", exc, backoff)
                self.stats["reconnects"] += 1
                try:
                    if self._stream is not None:
                        self._stream.close()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
