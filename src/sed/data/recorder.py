"""Recording session orchestration.

Recording is onset-triggered: each detected onset writes one clip plus a JSON
sidecar with full provenance. The CLI records directly into
``data_library/sounds/<label>/``.

Multi-channel: `record_channels` is an ordered list of physical input-channel
indexes (e.g. [0, 3, 1]). With `multi_mic_trigger="any"` the recorder saves only
the physical channels whose own onset detector fired for that merged trigger,
still ordered by `record_channels`. Other trigger modes keep the legacy full-set
save behaviour.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid as _uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from sed.audio.capture import AudioCapture
from sed.audio.onset import OnsetEvent
from sed.config import Config
from sed.data.library import RAW_CATEGORY, SCHEMA_VERSION, Library, safe_label

log = logging.getLogger(__name__)

# How long submit_clip waits for room in the write queue before giving up on a
# clip. Generous enough to ride out a slow disk, short enough that a dead
# writer cannot wedge the recording loop.
SUBMIT_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# Sidecar JSON metadata for every saved clip
# ---------------------------------------------------------------------------


@dataclass
class ClipMeta:
    schema_version: int
    uuid: str
    path: str                                  # absolute
    class_name: str
    label: str
    category: str
    project_id: str
    recording_counter: int
    session_id: str
    sample_rate: int
    channels: int
    duration_s: float
    recorded_at: str
    device: str
    density: str = "normal"
    review_status: str = "accepted"
    # Multichannel split provenance (see LibraryClip for the field contract).
    clip_kind: str = "mono"                    # mono | multichannel | channel_split
    channel_index: int | None = None
    channel_count: int = 1
    parent_uuid: str = ""
    group_id: str = ""
    mic_position: str = ""
    location: str = ""                         # venue/site note (e.g. "rnd-office", "bar-zone-1")
    microphone_type: str | list[str] = "generic"           # from cfg.audio.microphone_type
    onset_strength: float | None = None
    onset_baseline: float | None = None
    # Seconds from the start of the saved clip to the detected transient. The
    # recorder knows this exactly (it is the preroll it just drew), and every
    # consumer that needs to fit a shorter model window around the event would
    # otherwise have to re-detect it. See Config.window_lead_in_s.
    onset_offset_s: float | None = None
    peak_dbfs: float | None = None             # peak amplitude of the saved clip (ch0)
    rms_dbfs: float | None = None              # RMS amplitude
    transient_mel_db: float | None = None      # dominant mel intensity near transient
    recorded_channels: list[int] = field(default_factory=list)
    triggered_channels: list[int] = field(default_factory=list)
    triggered_flux: dict[str, float] = field(default_factory=dict)
    triggered_peak_dbfs: dict[str, float] = field(default_factory=dict)
    description: str = ""                      # free-form description (e.g. "half-full", "empty")
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ClipContext:
    """The capture settings a clip was actually recorded with.

    Frozen at the moment the audio is sliced, and carried with it through the
    write queue. Reading these from the live config at write time instead —
    which is what used to happen — meant that changing the interface while
    clips were still queued stamped the NEW sample rate onto audio captured at
    the old one: a 48 kHz clip written with a 44.1 kHz header plays 8.8 % slow,
    and training resamples from the header, so the error is invisible on disk
    and silently becomes a time-stretched example.
    """

    sample_rate: int
    device: str
    microphone_type: tuple[str, ...]
    record_channels: tuple[int, ...]


@dataclass
class ClipTriggerInfo:
    recorded_channels: list[int] = field(default_factory=list)
    triggered_channels: list[int] = field(default_factory=list)
    triggered_flux: dict[str, float] = field(default_factory=dict)
    triggered_peak_dbfs: dict[str, float] = field(default_factory=dict)


@dataclass
class PendingTriggerGroup:
    onset: OnsetEvent
    target_end_idx: int
    events_by_channel: dict[int, OnsetEvent] = field(default_factory=dict)

    def add_event(self, event: OnsetEvent) -> None:
        if event.channel is None:
            return
        existing = self.events_by_channel.get(event.channel)
        if existing is None or event.strength > existing.strength:
            self.events_by_channel[event.channel] = event


def _mics_for_channels(context: ClipContext, recorded: list[int]) -> tuple[str, ...]:
    """The microphone names for the channels this clip actually holds.

    ``microphone_type`` is positional over ``record_channels``, but with
    ``multi_mic_trigger="any"`` a clip holds only the channels that fired.
    Stamping the whole list left ``Library.split_clip`` — which indexes it by
    WAV column — attributing the wrong microphone to every split of a clip that
    came from a secondary mic.
    """
    order = {ch: i for i, ch in enumerate(context.record_channels)}
    mics = context.microphone_type
    out = []
    for ch in recorded:
        i = order.get(int(ch))
        out.append(str(mics[i]) if i is not None and i < len(mics) else "generic")
    return tuple(out) if out else tuple(mics)


def _order_channels(channels: list[int], record_indices: list[int]) -> list[int]:
    order = {int(ch): i for i, ch in enumerate(record_indices)}
    return sorted(
        {int(ch) for ch in channels},
        key=lambda ch: (order.get(ch, len(order)), ch),
    )


def _group_onset_events(
    pending: list[PendingTriggerGroup],
    events: list[OnsetEvent],
    *,
    merge_window: int,
    target_end_for_event: Callable[[OnsetEvent], int],
    recording_cooldown: int = 0,
    last_group_onset_idx: list[int | None] | None = None,
) -> None:
    for ev in sorted(events, key=lambda e: e.sample_idx):
        group = next(
            (
                g for g in reversed(pending)
                if 0 <= ev.sample_idx - g.onset.sample_idx <= merge_window
            ),
            None,
        )
        if group is None:
            last_onset = last_group_onset_idx[0] if last_group_onset_idx else None
            if (
                last_onset is not None
                and ev.sample_idx - last_onset < recording_cooldown
            ):
                continue
            group = PendingTriggerGroup(
                onset=ev,
                target_end_idx=target_end_for_event(ev),
            )
            pending.append(group)
            if last_group_onset_idx is not None:
                last_group_onset_idx[0] = ev.sample_idx
        group.add_event(ev)


def _linear_to_db(x: float) -> float:
    """Convert a linear flux value to dB. Floors at -120 to avoid log(0)."""
    return 20.0 * float(np.log10(max(float(x), 1e-12)))


def _format_db_list(values_by_channel: dict[int, float], channels: list[int]) -> str:
    return "/".join(f"{values_by_channel.get(ch, float('nan')):+.0f}" for ch in channels)


def _format_log_time(now: datetime | None = None) -> str:
    """Local wall-clock time for compact recording log lines."""
    current = now if now is not None else datetime.now().astimezone()
    return current.strftime("%H:%M:%S.") + f"{current.microsecond // 1000:03d}"


def _format_saved_count_line(
    count: int,
    limit_str: str,
    logged_at: datetime | None = None,
) -> str:
    return f"  [dim]{_format_log_time(logged_at)}[/dim] [bold]#{count:>3}[/bold]{limit_str}"


def _format_recorded_clip_line(
    count: int,
    limit_str: str,
    class_name: str,
    channels: list[int],
    flux_db_by_channel: dict[int, float],
    peak_db_by_channel: dict[int, float],
    logged_at: datetime | None = None,
) -> str:
    mic_word = "mic" if len(channels) == 1 else "mics"
    mic_text = ",".join(str(c) for c in channels) if channels else "?"
    flux_text = _format_db_list(flux_db_by_channel, channels)
    peak_text = _format_db_list(peak_db_by_channel, channels)
    return (
        f"{_format_saved_count_line(count, limit_str, logged_at)}  "
        f"[dim]{class_name}[/dim]  "
        f"{mic_word} {mic_text} ({len(channels)}ch)  "
        f"flux {flux_text}dB  "
        f"[dim]peak {peak_text}dBFS[/dim]"
    )


def _trigger_group_target_end_idx(
    event: OnsetEvent,
    *,
    sr_ratio: float,
    maturity_post_n: int,
    merge_window_n: int,
) -> int:
    return int(event.sample_idx * sr_ratio) + max(maturity_post_n, merge_window_n)


# Shared with the runtime (onset->scoring window) so the input->model rate
# mapping stays identical on both paths. Re-exported: the engine and the tests
# both reach for it here, where the recording geometry is documented.
from sed.audio.resample import model_rate_index as _model_rate_index  # noqa: E402,F401


# ---------------------------------------------------------------------------
class RecordingSession:
    def __init__(
        self,
        cfg: Config,
        out_dir: Path,
        class_name: str,
        project_id: str,
        mic_position: str = "",
        location: str = "",
        on_save: "callable | None" = None,
        description: str = "",
        label: str | None = None,
        category: str = RAW_CATEGORY,
        density: str = "normal",
        review_status: str = "accepted",
        tags: list[str] | None = None,
    ):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.class_name = class_name
        self.label = label or class_name
        self.category = category
        self.project_id = project_id
        self.density = density
        self.review_status = review_status
        self.mic_position = mic_position
        self.location = location
        self.description = description
        self.tags = list(tags or [])
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
        # Surfaced so a stalled or failing writer is visible to the operator
        # rather than only to the log.
        self.write_errors = 0
        self.last_write_error = ""

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
        trigger_info: ClipTriggerInfo | None = None,
        onset_offset_s: float | None = None,
        context: ClipContext | None = None,
    ) -> None:
        """Queue a clip for writing. wave: (n, channels) float32.

        ``context`` pins the capture settings this audio was recorded with;
        without it the writer falls back to the live config, which is only
        correct while nothing changes mid-session.

        Never blocks indefinitely. An unbounded put on a full queue whose
        consumer has died stalls the recording loop and makes stop() hang; one
        lost clip and a loud error is the better failure.
        """
        try:
            self._save_q.put(
                (wave.copy(), onset, trigger_info, onset_offset_s, context),
                timeout=SUBMIT_TIMEOUT_S,
            )
        except queue.Full:
            self.write_errors += 1
            self.last_write_error = "writer is not keeping up; a clip was dropped"
            log.error(
                "clip dropped: the write queue has been full for %.0fs "
                "(slow disk, or the writer thread died)", SUBMIT_TIMEOUT_S,
            )

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
            wave, onset, trigger_info, onset_offset_s, context = item
            try:
                wav_path, meta = self._save_one(
                    wave, onset, trigger_info, onset_offset_s=onset_offset_s,
                    context=context,
                )
            except Exception as e:  # noqa: BLE001
                # Never let one bad write kill the writer. A dead writer is not
                # quiet: the queue fills, and then every submit_clip blocks the
                # recording loop forever and stop() never returns.
                self.write_errors += 1
                self.last_write_error = f"{type(e).__name__}: {e}"
                log.exception("failed to write a clip")
                continue
            if self.on_save is not None:
                try:
                    self.on_save(wav_path, meta)
                except Exception as e:  # noqa: BLE001
                    log.warning("on_save callback failed: %s", e)

    def _save_one(
        self,
        wave: np.ndarray,
        onset: OnsetEvent | None,
        trigger_info: ClipTriggerInfo | None = None,
        onset_offset_s: float | None = None,
        context: ClipContext | None = None,
    ) -> tuple[Path, ClipMeta]:
        context = context or self.live_context()
        clip_uuid = _uuid.uuid4().hex
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
        counter = self.count + 1
        safe_class = safe_label(self.label)
        wav_path = self.out_dir / f"{ts}_{safe_class}_{counter:04d}_{clip_uuid[:8]}.wav"
        save_sr = int(context.sample_rate)
        channels_n = wave.shape[1] if wave.ndim > 1 else 1
        recorded_channels = (
            list(trigger_info.recorded_channels)
            if trigger_info and trigger_info.recorded_channels
            else self._default_recorded_channels(channels_n, context)
        )
        triggered_channels = (
            list(trigger_info.triggered_channels)
            if trigger_info and trigger_info.triggered_channels
            else list(recorded_channels)
        )
        triggered_flux = (
            {str(k): float(v) for k, v in trigger_info.triggered_flux.items()}
            if trigger_info
            else {}
        )
        triggered_peak_dbfs = (
            {str(k): float(v) for k, v in trigger_info.triggered_peak_dbfs.items()}
            if trigger_info
            else {}
        )

        # Preserve the raw capture rate in the library. Training/export/runtime
        # resample at the model boundary so old/new clips can have mixed rates.
        sf.write(str(wav_path), wave, save_sr, subtype="PCM_16")

        # Peak dBFS of the saved clip (channel 0 if multi-channel). This is
        # the loudness of the actual saved audio, separate from the onset's
        # internal flux strength.
        ch0 = wave[:, 0] if wave.ndim > 1 else wave
        peak = float(np.max(np.abs(ch0))) if ch0.size else 0.0
        peak_dbfs = 20.0 * float(np.log10(peak + 1e-10)) if peak > 0 else -120.0

        rms = float(np.sqrt(np.mean(ch0**2))) if ch0.size else 0.0
        rms_dbfs = 20.0 * float(np.log10(rms + 1e-10)) if rms > 0 else -120.0
        try:
            from sed.data.augment import transient_mel_db
            tmel_db = transient_mel_db(ch0, save_sr)
        except Exception:  # noqa: BLE001
            tmel_db = None

        meta = ClipMeta(
            schema_version=SCHEMA_VERSION,
            uuid=clip_uuid,
            path=str(wav_path),
            class_name=self.class_name,
            label=self.label,
            category=self.category,
            project_id=self.project_id,
            density=self.density,
            review_status=self.review_status,
            recording_counter=counter,
            session_id=self.session_id,
            sample_rate=save_sr,
            channels=channels_n,
            clip_kind="multichannel" if channels_n > 1 else "mono",
            channel_count=channels_n,
            duration_s=wave.shape[0] / save_sr,
            recorded_at=ts,
            device=context.device,
            mic_position=self.mic_position,
            location=self.location,
            microphone_type=list(_mics_for_channels(context, recorded_channels)),
            onset_strength=getattr(onset, "strength", None),
            onset_baseline=getattr(onset, "baseline", None),
            onset_offset_s=onset_offset_s,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            transient_mel_db=tmel_db,
            recorded_channels=recorded_channels,
            triggered_channels=triggered_channels,
            triggered_flux=triggered_flux,
            triggered_peak_dbfs=triggered_peak_dbfs,
            description=self.description,
            tags=self.tags,
        )
        with open(wav_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

        self.count += 1
        log.debug("saved %s (#%d)", wav_path.name, self.count)
        return wav_path, meta

    def live_context(self) -> ClipContext:
        """The capture settings in force right now."""
        return ClipContext(
            sample_rate=int(self.cfg.audio.record_sample_rate),
            device=str(self.cfg.deploy.record_device),
            microphone_type=tuple(str(m) for m in self.cfg.audio.microphone_type),
            record_channels=tuple(int(c) for c in self.cfg.audio.record_channels),
        )

    def _default_recorded_channels(self, channels_n: int, context: ClipContext) -> list[int]:
        configured = list(context.record_channels)
        if len(configured) >= channels_n:
            return [int(c) for c in configured[:channels_n]]
        return list(range(channels_n))


# ---------------------------------------------------------------------------
# High-level entry points used by the CLI
# ---------------------------------------------------------------------------


# A selected channel sitting this far down on BOTH peak and RMS over the
# startup probe window is almost certainly unconnected (preamp / digital noise
# floor) rather than a live-but-quiet source. Measured Scarlett preamp
# self-noise with nothing plugged in is ~-86 dBFS peak / ~-98 dBFS RMS; a real
# connected source clears these thresholds with margin.
_SILENT_PEAK_DBFS = -78.0
_SILENT_RMS_DBFS = -85.0


def _probe_channel_levels(
    capture: AudioCapture,
    channels: list[int],
    *,
    settle_s: float = 0.6,
    window_s: float = 0.4,
) -> list[tuple[int, float, float]]:
    """Read a brief window from the already-running stream.

    Returns ``(channel, peak_dbfs, rms_dbfs)`` for each requested channel, or
    an empty list if the stream hasn't delivered enough audio yet — the probe
    must never block the recording session.
    """
    need = int(window_s * capture.sample_rate)
    deadline = time.monotonic() + settle_s + window_s + 1.5
    while capture.ring.write_idx < need and time.monotonic() < deadline:
        time.sleep(0.05)
    if capture.ring.write_idx < need:
        return []
    out: list[tuple[int, float, float]] = []
    for c in channels:
        try:
            x = capture.ring.read_last(need, channel=c)
        except (ValueError, IndexError):
            continue
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
        peak_db = 20.0 * float(np.log10(peak + 1e-10)) if peak > 0 else -120.0
        rms_db = 20.0 * float(np.log10(rms + 1e-10)) if rms > 0 else -120.0
        out.append((c, peak_db, rms_db))
    return out


def record_auto(
    cfg: Config,
    out_dir: Path,
    class_name: str,
    project_id: str,
    duration_s: float,
    mic_position: str = "",
    location: str = "",
    description: str = "",
    max_clips: int | None = None,
    on_save: "callable | None" = None,
    label: str | None = None,
    category: str = RAW_CATEGORY,
    density: str = "normal",
    review_status: str = "accepted",
    tags: list[str] | None = None,
) -> int:
    """Onset-triggered recording with a live terminal display.

    The capture loop itself lives in :class:`sed.data.record_engine.RecordEngine`,
    which the recording web console drives too; this function is the Rich
    front-end for it. Everything the loop does — cross-mic merging, the global
    cooldown, randomized preroll, the exact ``max_clips`` gate — is therefore
    identical in the terminal and in the browser.

    Stops when EITHER duration_s elapses, max_clips is reached, OR the user
    sends KeyboardInterrupt. In all three cases clips already saved stay on
    disk and the in-flight write queue is flushed before return.
    """
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text

    from sed.data.record_engine import ClipMetadata, RecordEngine

    # `out_dir` is where this label's clips belong; the engine derives the same
    # path from the label, so a caller passing a different one is telling us
    # about a layout the library does not know about.
    engine = RecordEngine(
        cfg,
        project_id,
        ClipMetadata(
            label=label or class_name,
            density=density,
            description=description,
            location=location,
            mic_position=mic_position,
            tags=list(tags or []),
        ),
        on_save=on_save,
        max_clips=max_clips,
        duration_s=duration_s,
        category=category,
        review_status=review_status,
        library=_LibraryAt(out_dir),
    )

    console = Console(highlight=False)
    limit_str = f"/{max_clips}" if max_clips else ""
    input_sr = int(cfg.audio.record_sample_rate)
    clip_frames = round(cfg.onset.record_seconds * input_sr)
    location_str = f"  ·  loc: [cyan]{location}[/cyan]" if location else ""
    console.print(
        f"[bold]● Recording[/bold]  [cyan]{class_name}[/cyan]{location_str}  "
        f"[dim]clip duration: {cfg.onset.record_seconds:.2f}s ({clip_frames:,} frames @ {input_sr} Hz) · "
        f"≤ {duration_s/60:.0f} min · Ctrl+C to stop[/dim]\n"
        f"  [dim]onset gates:  floor [/dim][bold]{cfg.onset.min_flux_db:+.0f} dB[/bold]"
        f"[dim]  ·  mad_k [/dim][bold]{cfg.onset.mad_k}[/bold]"
        f"[dim]   (effective gate = max(floor, median+mad_k·MAD))[/dim]"
    )

    def _level_bar(dbfs: float, flashing: bool, width: int = 20) -> str:
        """Map dBFS (typically -80…0) to a filled bar string."""
        if flashing:
            return f"[bright_cyan]{'█' * width}[/bright_cyan]"
        frac = max(0.0, min(1.0, (dbfs + 60) / 60))  # -60 dBFS = empty, 0 dBFS = full
        filled = int(frac * width)
        color = "green" if dbfs < -12 else ("yellow" if dbfs < -3 else "red")
        return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"

    def _status_line(snap: dict) -> Text:
        """Drive the display from the hottest channel.

        Level comes from the loudest mic; flux and gate come from the mic with
        the highest flux, so the comparison shown is self-consistent.
        """
        chans = snap["channels"] or [{}]
        dbfs = max((c.get("peak_dbfs") or -120.0) for c in chans)
        gated = [c for c in chans if c.get("flux_db") is not None]
        hot = max(gated, key=lambda c: c["flux_db"]) if gated else {}
        flashing = any(c.get("flash") for c in chans)
        level = f"{dbfs:+.0f} dBFS"
        flux_db = hot.get("flux_db")
        gate_db = hot.get("gate_db")
        if flux_db is None or gate_db is None:
            flux_str = "[dim]flux —  (warming up)[/dim]"
        else:
            will = bool(hot.get("will_trigger"))
            mark = "[bright_green]✓[/bright_green]" if will else "[red]✗[/red]"
            fc = "bright_green" if will else "dim"
            flux_str = (
                f"[{fc}]flux {flux_db:+5.1f} dB[/{fc}]  "
                f"[dim]vs gate[/dim] {mark}[dim] {gate_db:+5.1f} dB[/dim]  "
                f"[dim]({hot.get('gate_source', '')})[/dim]"
            )
        count = f"[dim]clips:[/dim] [bold]{snap['clips_saved']}{limit_str}[/bold]"
        return Text.from_markup(
            f"  {_level_bar(dbfs, flashing)}  {level:>9}   {flux_str}   {count}"
        )

    engine.start()
    printed = 0
    warned_dead = False
    try:
        with Live(_status_line(engine.snapshot()), console=console,
                  refresh_per_second=8, transient=False) as live:
            while engine.state == "running":
                time.sleep(0.05)
                snap = engine.snapshot()

                if not warned_dead and snap["elapsed_s"] > 2.0:
                    warned_dead = True
                    dead = [c for c in snap["channels"] if c["silent"]]
                    if dead:
                        chans = ", ".join(
                            f"ch{c['channel']} ({c['peak_dbfs']:+.0f} dBFS peak)"
                            for c in dead
                        )
                        live.console.print(
                            f"  [yellow]⚠ selected channel(s) look silent:[/yellow] {chans}\n"
                            f"    [dim]Channels are 0-based indexes — interface input N is "
                            f"index N-1 (input 1 → ch0, input 4 → ch3). '-ch 4' selects index "
                            f"4 (the 5th input), not four channels. Check the index or the "
                            f"connection.[/dim]"
                        )

                printed = _print_new_clips(live.console, snap, printed, limit_str)
                live.update(_status_line(snap))
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        _print_new_clips(console, engine.snapshot(), printed, limit_str)

    saved = engine.clips_saved
    console.print(f"\n[bold]Done.[/bold]  {saved} clip(s) saved → [dim]{out_dir}[/dim]")
    return saved


def _print_new_clips(console, snap: dict, printed: int, limit_str: str) -> int:
    """Emit one summary line per clip saved since the last call.

    ``log`` is newest-first and capped, so replay it oldest-first and skip the
    ones already on screen.
    """
    entries = [e for e in reversed(snap["log"]) if e["index"] > printed]
    for entry in entries:
        channels = [int(c) for c in entry["channels"]]
        console.print(_format_recorded_clip_line(
            entry["index"],
            limit_str,
            entry["label"],
            channels,
            {c: float(entry["flux_db"].get(str(c), float("nan"))) for c in channels},
            {c: float(entry.get("peak_dbfs_by_channel", {}).get(str(c), float("nan")))
             for c in channels},
        ))
        printed = entry["index"]
    return max(printed, snap["clips_saved"])


class _LibraryAt(Library):
    """A library pinned to one output directory.

    ``sed record`` resolves the destination before it starts (and tests point it
    at a temp dir), while the engine asks the library for a label's directory so
    a mid-session label change lands in the right place. This keeps both true.
    """

    def __init__(self, out_dir: Path) -> None:
        super().__init__()
        self._pinned = Path(out_dir)
        self._pinned_for: str | None = None

    def raw_dir(self, label: str) -> Path:
        if self._pinned_for is None:
            self._pinned_for = label
        if label == self._pinned_for:
            self._pinned.mkdir(parents=True, exist_ok=True)
            return self._pinned
        return super().raw_dir(label)
