"""Recording session orchestration.

Two modes:
  1. continuous — write the entire raw stream to disk (for ambience / RIRs)
  2. auto       — onset-triggered. Save only the frames around detected events.

For auto mode, each detected onset writes a 1.0 s clip (300 ms preroll +
700 ms postroll) plus a JSON sidecar with full provenance.

Multi-channel: when `record_channels > 1` the saved WAV is multichannel; the
inference channel is whichever the project config chooses. The library /
inspector tools use the first channel as the canonical "preview" channel
unless told otherwise.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid as _uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from sed.audio.capture import AudioCapture
from sed.audio.onset import OnsetEvent, StreamingSuperflux
from sed.config import Config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sidecar JSON metadata for every saved clip
# ---------------------------------------------------------------------------


@dataclass
class ClipMeta:
    uuid: str
    path: str                                  # absolute
    class_name: str
    project_id: str
    session_id: str
    sample_rate: int
    channels: int
    duration_s: float
    recorded_at: str
    device: str
    mic_position: str = ""
    onset_strength: float | None = None
    onset_baseline: float | None = None
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    flagged: bool = False                      # set by inspector to mark bad samples


# ---------------------------------------------------------------------------
# Recording session
# ---------------------------------------------------------------------------


class RecordingSession:
    def __init__(
        self,
        cfg: Config,
        out_dir: Path,
        class_name: str,
        project_id: str,
        mic_position: str = "",
        on_save: "callable | None" = None,
    ):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.class_name = class_name
        self.project_id = project_id
        self.mic_position = mic_position
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_id += "-" + _uuid.uuid4().hex[:6]
        # Optional callback fired synchronously AFTER each clip is written to
        # disk. Receives (wav_path, ClipMeta). Used to register clips into a
        # library manifest as they're recorded, so Ctrl+C doesn't lose work.
        self.on_save = on_save

        # Save audio + meta on a writer thread to keep the audio thread free
        self._save_q: queue.Queue = queue.Queue(maxsize=64)
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._stop = threading.Event()
        self.count = 0

    def start(self) -> None:
        self._writer.start()

    def stop(self) -> None:
        # Drain the queue before shutting the writer down so clips already
        # submitted but not yet written get saved (essential for Ctrl+C UX).
        self._save_q.put(None)
        self._writer.join(timeout=30.0)
        self._stop.set()

    def submit_clip(
        self,
        wave: np.ndarray,
        onset: OnsetEvent | None = None,
    ) -> None:
        """wave: (n, channels) float32."""
        self._save_q.put((wave.copy(), onset))

    # -- internals ----------------------------------------------------------

    def _writer_loop(self) -> None:
        while True:
            try:
                item = self._save_q.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if item is None:
                break
            wave, onset = item
            wav_path, meta = self._save_one(wave, onset)
            if self.on_save is not None:
                try:
                    self.on_save(wav_path, meta)
                except Exception as e:  # noqa: BLE001
                    log.warning("on_save callback failed: %s", e)

    def _save_one(self, wave: np.ndarray, onset: OnsetEvent | None) -> tuple[Path, ClipMeta]:
        clip_uuid = _uuid.uuid4().hex
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
        wav_path = self.out_dir / f"{ts}_{clip_uuid[:8]}.wav"
        # Soundfile expects (n, ch) and writes 16-bit PCM by default — good for size.
        sf.write(str(wav_path), wave, self.cfg.audio.input_sample_rate, subtype="PCM_16")

        meta = ClipMeta(
            uuid=clip_uuid,
            path=str(wav_path),
            class_name=self.class_name,
            project_id=self.project_id,
            session_id=self.session_id,
            sample_rate=self.cfg.audio.input_sample_rate,
            channels=wave.shape[1] if wave.ndim > 1 else 1,
            duration_s=wave.shape[0] / self.cfg.audio.input_sample_rate,
            recorded_at=ts,
            device=self.cfg.audio.record_device,
            mic_position=self.mic_position,
            onset_strength=getattr(onset, "strength", None),
            onset_baseline=getattr(onset, "baseline", None),
        )
        with open(wav_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

        self.count += 1
        log.debug("saved %s (#%d)", wav_path.name, self.count)
        return wav_path, meta


# ---------------------------------------------------------------------------
# High-level entry points used by the CLI
# ---------------------------------------------------------------------------


def record_continuous(
    cfg: Config,
    out_dir: Path,
    class_name: str,
    project_id: str,
    duration_s: float,
    mic_position: str = "",
    chunk_s: float = 30.0,
    on_save: "callable | None" = None,
) -> int:
    """Save the entire stream to disk in `chunk_s`-second WAV files.

    Stops on duration_s OR Ctrl+C. On Ctrl+C, the in-flight partial chunk
    is also saved (so a 28-second leftover after 12 minutes is not lost).
    """
    session = RecordingSession(
        cfg, out_dir, class_name, project_id, mic_position, on_save=on_save,
    )
    session.start()
    input_sr = cfg.audio.input_sample_rate
    capture = AudioCapture(
        sample_rate=input_sr,
        channels=cfg.audio.record_channels,
        device=cfg.audio.record_device,
        blocksize=cfg.audio.blocksize,
        ring_seconds=max(chunk_s + 1.0, 4.0),
        gain_db=cfg.audio.input_gain_db,
    )
    capture.start()
    chunk_samples = int(chunk_s * input_sr)
    log.info(
        "Recording continuous ≤ %.1f min into %s. Ctrl+C stops early (partial chunk saved).",
        duration_s / 60, out_dir,
    )
    start = time.monotonic()
    last_idx = 0

    def _save_partial() -> None:
        """If at least 1 second has been buffered since last_idx, save it."""
        remaining = capture.ring.write_idx - last_idx
        if remaining < input_sr:
            return
        try:
            if cfg.audio.record_channels > 1:
                chans = [
                    capture.ring.read_window(last_idx, remaining, channel=c)
                    for c in range(cfg.audio.record_channels)
                ]
                wave = np.stack(chans, axis=-1)
            else:
                wave = capture.ring.read_window(last_idx, remaining, channel=0)[:, None]
            session.submit_clip(wave)
            log.info("saved %.1f s partial chunk on stop", remaining / input_sr)
        except ValueError:
            pass

    try:
        while time.monotonic() - start < duration_s:
            # Wait until enough new samples have been captured
            while capture.ring.write_idx - last_idx < chunk_samples:
                time.sleep(0.05)
                if time.monotonic() - start > duration_s:
                    break
            try:
                if cfg.audio.record_channels > 1:
                    chans = [
                        capture.ring.read_window(last_idx, chunk_samples, channel=c)
                        for c in range(cfg.audio.record_channels)
                    ]
                    wave = np.stack(chans, axis=-1)
                else:
                    wave = capture.ring.read_window(last_idx, chunk_samples, channel=0)[:, None]
                session.submit_clip(wave)
                last_idx += chunk_samples
            except ValueError:
                # ring overran — bump last_idx and continue
                last_idx = capture.ring.write_idx - chunk_samples
        _save_partial()
    except KeyboardInterrupt:
        log.info("Ctrl+C — stopping recording, flushing in-flight chunks ...")
        _save_partial()
    finally:
        capture.stop()
        session.stop()
    return session.count


def record_auto(
    cfg: Config,
    out_dir: Path,
    class_name: str,
    project_id: str,
    duration_s: float,
    mic_position: str = "",
    max_clips: int | None = None,
    on_save: "callable | None" = None,
) -> int:
    """Onset-triggered recording. Saves a clip per detected transient.

    Stops when EITHER duration_s elapses, max_clips is reached, OR the
    user sends KeyboardInterrupt — in all three cases any clips already
    saved stay on disk and the in-flight write queue is flushed before
    return. The caller's `on_save` callback (if any) is invoked
    synchronously after each clip lands, so callers can register clips
    into a library manifest as they're recorded (rather than at the end,
    which would lose work on Ctrl+C).
    """
    session = RecordingSession(
        cfg, out_dir, class_name, project_id, mic_position, on_save=on_save,
    )
    session.start()
    input_sr = cfg.audio.input_sample_rate
    model_sr = cfg.audio.sample_rate        # 32 kHz — onset runs at model rate

    onset = StreamingSuperflux(
        sample_rate=model_sr,
        hop_ms=cfg.onset.hop_ms,
        median_window_s=cfg.onset.median_window_s,
        mad_k=cfg.onset.mad_k,
        refractory_ms=cfg.onset.refractory_ms,
        min_flux=cfg.onset.min_flux,
    )

    # Pending events: detected but waiting until the postroll has been captured.
    pending: list[tuple[OnsetEvent, int]] = []  # (event, target_end_idx)
    pending_lock = threading.Lock()

    pre_n = int(cfg.onset.preroll_ms / 1000 * input_sr)
    post_n = int(cfg.onset.postroll_ms / 1000 * input_sr)
    clip_n = int(cfg.onset.clip_seconds * input_sr)

    # Resampling state: track a parallel write index in model_sr space.
    from sed.audio.resample import resample_audio
    sr_ratio = input_sr / model_sr            # 48000/32000 = 1.5
    _onset_write_idx = [0]                    # mutable from closure

    def on_block(block: np.ndarray, write_idx: int) -> None:
        # Onset detection runs on channel 0, resampled to model rate
        mono = block[:, 0] if block.ndim > 1 else block
        mono_rs = resample_audio(mono, orig_sr=input_sr, target_sr=model_sr)
        _onset_write_idx[0] += len(mono_rs)
        events = onset.process_block(mono_rs, _onset_write_idx[0])
        if events:
            with pending_lock:
                for ev in events:
                    # Map sample index back to input_sr (ring buffer) space
                    ev_idx_input = int(ev.sample_idx * sr_ratio)
                    target_end = ev_idx_input + post_n
                    pending.append((ev, target_end))

    capture = AudioCapture(
        sample_rate=input_sr,
        channels=cfg.audio.record_channels,
        device=cfg.audio.record_device,
        blocksize=cfg.audio.blocksize,
        ring_seconds=4.0,
        gain_db=cfg.audio.input_gain_db,
        on_block=on_block,
    )
    capture.start()

    from rich.console import Console
    from rich.live import Live
    from rich.text import Text

    _console = Console(highlight=False)
    limit_str = f"/{max_clips}" if max_clips else ""

    # One-time header: class, duration, and the two sensitivity knobs
    _console.print(
        f"[bold]● Recording[/bold]  [dim]{class_name}[/dim]  "
        f"[dim]≤ {duration_s/60:.0f} min · Ctrl+C to stop[/dim]\n"
        f"  [dim]sensitivity:  "
        f"mad_k [/dim][bold]{cfg.onset.mad_k}[/bold]  "
        f"[dim]·  min_flux [/dim][bold]{cfg.onset.min_flux:.3f}[/bold]"
    )

    def _level_bar(dbfs: float, width: int = 20) -> str:
        """Map dBFS (typically -80…0) to a filled bar string."""
        frac = max(0.0, min(1.0, (dbfs + 60) / 60))  # -60 dBFS = empty, 0 dBFS = full
        filled = int(frac * width)
        color = "green" if dbfs < -12 else ("yellow" if dbfs < -3 else "red")
        return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"

    def _status_line() -> Text:
        dbfs, flux, thresh = onset.get_status()
        bar = _level_bar(dbfs)
        saved = session.count
        label = f"[dim]clips:[/dim] [bold]{saved}{limit_str}[/bold]"
        level = f"{dbfs:+.0f} dBFS"
        # Show flux vs min_flux so user can see how close they are to triggering
        flux_color = "green" if flux >= cfg.onset.min_flux else "dim"
        flux_str = f"[{flux_color}]flux {flux:.3f}[/{flux_color}][dim]/{cfg.onset.min_flux:.3f}[/dim]"
        t = Text.from_markup(f"  {bar}  {level:>9}   {flux_str}   {label}")
        return t

    prev_count = 0
    start = time.monotonic()
    try:
        with Live(_status_line(), console=_console, refresh_per_second=8,
                  transient=False) as live:
            while time.monotonic() - start < duration_s:
                time.sleep(0.05)

                # Honor max_clips
                if max_clips is not None:
                    with pending_lock:
                        in_flight = len(pending) + session._save_q.qsize()
                    if session.count + in_flight >= max_clips:
                        break

                now_idx = capture.ring.write_idx
                ready: list[tuple[OnsetEvent, int]] = []
                with pending_lock:
                    still_pending = []
                    for ev, target in pending:
                        if target <= now_idx:
                            ready.append((ev, target))
                        else:
                            still_pending.append((ev, target))
                    pending[:] = still_pending

                for ev, _target in ready:
                    if max_clips is not None and session.count >= max_clips:
                        break
                    clip_start = int(ev.sample_idx * sr_ratio) - pre_n
                    try:
                        if cfg.audio.record_channels > 1:
                            chans = [
                                capture.ring.read_window(clip_start, clip_n, channel=c)
                                for c in range(cfg.audio.record_channels)
                            ]
                            wave = np.stack(chans, axis=-1)
                        else:
                            wave = capture.ring.read_window(clip_start, clip_n, channel=0)[:, None]
                        session.submit_clip(wave, onset=ev)
                    except ValueError as e:
                        log.warning("ring overrun, clip dropped: %s", e)

                # Print a line for each newly saved clip (count bumped by writer thread)
                if session.count > prev_count:
                    for _ in range(session.count - prev_count):
                        n = prev_count + 1
                        prev_count += 1
                        live.console.print(
                            f"  [green]✔[/green] [bold]#{n:>3}[/bold]{limit_str}  "
                            f"[dim]{class_name}[/dim]"
                        )

                # Decay peak so the bar responds to silence
                onset._peak_rms *= 0.85
                live.update(_status_line())

    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        session.stop()

    _console.print(
        f"\n[bold]Done.[/bold]  {session.count} clip(s) saved → [dim]{out_dir}[/dim]"
    )
    return session.count
