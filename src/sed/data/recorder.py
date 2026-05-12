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
import random
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
from sed.audio.onset import OnsetEvent, StreamingSuperflux
from sed.config import Config
from sed.data.library import RAW_CATEGORY, SCHEMA_VERSION, safe_label

log = logging.getLogger(__name__)


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
# mapping stays identical on both paths.
from sed.audio.resample import model_rate_index as _model_rate_index  # noqa: E402


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
    ) -> None:
        """wave: (n, channels) float32."""
        self._save_q.put((wave.copy(), onset, trigger_info, onset_offset_s))

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
            wave, onset, trigger_info, onset_offset_s = item
            wav_path, meta = self._save_one(
                wave, onset, trigger_info, onset_offset_s=onset_offset_s
            )
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
    ) -> tuple[Path, ClipMeta]:
        clip_uuid = _uuid.uuid4().hex
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
        counter = self.count + 1
        safe_class = safe_label(self.label)
        wav_path = self.out_dir / f"{ts}_{safe_class}_{counter:04d}_{clip_uuid[:8]}.wav"
        save_sr = int(self.cfg.audio.record_sample_rate)
        channels_n = wave.shape[1] if wave.ndim > 1 else 1
        recorded_channels = (
            list(trigger_info.recorded_channels)
            if trigger_info and trigger_info.recorded_channels
            else self._default_recorded_channels(channels_n)
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
            device=self.cfg.deploy.record_device,
            mic_position=self.mic_position,
            location=self.location,
            microphone_type=self.cfg.audio.microphone_type,
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

    def _default_recorded_channels(self, channels_n: int) -> list[int]:
        configured = list(getattr(self.cfg.audio, "record_channels", []) or [])
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
        cfg, out_dir, class_name, project_id, mic_position=mic_position,
        location=location, on_save=on_save,
        description=description, label=label, category=category,
        density=density, review_status=review_status,
        tags=tags,
    )
    session.start()
    input_sr = cfg.audio.record_sample_rate
    model_sr = cfg.model.sample_rate        # 32 kHz — onset runs at model rate

    # Ordered physical channels to record. We open enough physical channels on
    # the device to cover the highest requested index, then store/monitor the
    # selected channels in this exact order (recorded ch i = device channel
    # record_indices[i]).
    record_indices = list(cfg.audio.record_channels) or [0]
    n_physical = max(record_indices) + 1
    n_recorded = len(record_indices)

    # Multichannel triggering: run one detector per recorded channel and OR their
    # detections so ANY recorded channel can start a recording. "mix" feeds a
    # single detector the channel downmix; "ch0" is the legacy single-channel
    # behaviour. With one recorded channel every mode collapses to one detector.
    trigger_mode = cfg.onset.multi_mic_trigger
    if trigger_mode in {"any", "any_all"}:
        n_trig = max(1, n_recorded)
        detector_channels: list[int | None] = list(record_indices)
    elif trigger_mode == "ch0":
        n_trig = 1
        detector_channels = [record_indices[0]]
    else:                                     # "mix"
        n_trig = 1
        detector_channels = [None]
    # Keep the detector's own refractory small so suppressed cross-mic events do
    # not extend the operator-facing recording cooldown. The public cooldown is
    # enforced globally below when new trigger groups are accepted.
    detector_refractory_ms = min(max(0, int(cfg.onset.recording_cooldown_ms)), 150)
    detectors = [
        StreamingSuperflux(
            sample_rate=model_sr,
            hop_ms=cfg.onset.hop_ms,
            n_mels=cfg.onset.trigger_n_mels,
            median_window_s=cfg.onset.median_window_s,
            mad_k=cfg.onset.mad_k,
            refractory_ms=detector_refractory_ms,
            min_flux=cfg.onset.min_flux,
        )
        for _ in range(n_trig)
    ]
    merge_window = int(cfg.onset.cross_channel_merge_ms / 1000 * model_sr)
    merge_window_input_n = int(cfg.onset.cross_channel_merge_ms / 1000 * input_sr)
    recording_cooldown = int(cfg.onset.recording_cooldown_ms / 1000 * model_sr)

    def aggregate_status() -> tuple[float, float, float]:
        """Drive the live display from the hottest channel.

        peak level = loudest channel; flux/threshold are taken from the same
        channel (the one with the highest flux) so the gate comparison stays
        self-consistent.
        """
        stats = [d.get_status() for d in detectors]
        dbfs = max(s[0] for s in stats)
        best = max(stats, key=lambda s: s[1])
        return dbfs, best[1], best[2]

    # Pending trigger groups: detected but waiting until the postroll has been
    # captured and the cross-mic merge window has closed. In "any" mode a group
    # accumulates every mic-specific onset within cross_channel_merge_ms of the
    # first hit; recording_cooldown_ms then blocks any brand-new group across all
    # mics.
    pending: list[PendingTriggerGroup] = []
    pending_lock = threading.Lock()
    last_group_onset_idx: list[int | None] = [None]

    fixed_pre_n = int(cfg.onset.preroll_ms / 1000 * input_sr)
    fixed_post_n = int(cfg.onset.postroll_ms / 1000 * input_sr)
    clip_n = int(cfg.onset.record_seconds * input_sr)

    # When randomize_preroll is on, each clip gets its own preroll drawn from
    # cfg.onset.preroll_ms_range (postroll auto-fills the rest of record_seconds).
    # The maturity check needs the WORST-case (largest) postroll so a clip is
    # always fully present in the ring before we slice — that corresponds to
    # the SMALLEST preroll.
    if cfg.onset.randomize_preroll:
        clip_ms_total = int(cfg.onset.record_seconds * 1000)
        rand_lo_ms, _rand_hi_ms = cfg.onset.preroll_ms_range
        maturity_post_n = int((clip_ms_total - rand_lo_ms) / 1000 * input_sr)
    else:
        maturity_post_n = fixed_post_n

    def pick_preroll_n() -> int:
        if cfg.onset.randomize_preroll:
            lo_ms, hi_ms = cfg.onset.preroll_ms_range
            pre_ms = random.randint(lo_ms, hi_ms)
            return int(pre_ms / 1000 * input_sr)
        return fixed_pre_n

    from sed.audio.resample import resample_audio
    sr_ratio = input_sr / model_sr            # 48000/32000 = 1.5
    # When False, on_block stops queueing new events but already-pending ones
    # are still allowed to mature into saves. Lets us hit max_clips precisely.
    accept_new = [True]
    def channel_signals(block: np.ndarray) -> list[np.ndarray]:
        """One 1-d signal per detector, in input-rate space.

        Indexes into the physical device channels via ``record_indices`` so the
        detectors monitor exactly the channels being recorded, in record order.
        """
        if block.ndim == 1:
            return [block]
        if trigger_mode == "ch0":
            return [block[:, record_indices[0]]]
        if trigger_mode == "mix":
            return [block[:, record_indices].mean(axis=-1).astype(np.float32)]
        # "any": one signal per recorded channel, in record order.
        return [block[:, c] for c in record_indices]

    def on_block(block: np.ndarray, write_idx: int) -> None:
        # Onset detection runs at model rate; the model-rate index is derived
        # from the ring's input-rate write index so the mapping cannot drift.
        signals = channel_signals(block)
        resampled = [
            resample_audio(sig, orig_sr=input_sr, target_sr=model_sr) for sig in signals
        ]
        if not resampled:
            return
        end_idx = _model_rate_index(write_idx, input_sr, model_sr)
        raw_events: list[OnsetEvent] = []
        for det, sig_rs, channel in zip(detectors, resampled, detector_channels):
            for ev in det.process_block(sig_rs, end_idx):
                raw_events.append(OnsetEvent(
                    sample_idx=ev.sample_idx,
                    strength=ev.strength,
                    baseline=ev.baseline,
                    channel=channel,
                ))
        if not raw_events or not accept_new[0]:
            return
        with pending_lock:
            _group_onset_events(
                pending,
                raw_events,
                merge_window=merge_window,
                target_end_for_event=lambda ev: _trigger_group_target_end_idx(
                    ev,
                    sr_ratio=sr_ratio,
                    maturity_post_n=maturity_post_n,
                    merge_window_n=merge_window_input_n,
                ),
                recording_cooldown=recording_cooldown,
                last_group_onset_idx=last_group_onset_idx,
            )

    capture = AudioCapture(
        sample_rate=input_sr,
        channels=n_physical,
        device=cfg.deploy.record_device,
        blocksize=cfg.audio.blocksize,
        ring_seconds=4.0,
        on_block=on_block,
    )
    capture.start()

    from rich.console import Console
    from rich.live import Live
    from rich.text import Text

    _console = Console(highlight=False)
    limit_str = f"/{max_clips}" if max_clips else ""

    # One-time header. Express duration/frame count and onset knobs in dB.
    clip_frames = int(round(cfg.onset.record_seconds * input_sr))
    location_str = f"  ·  loc: [cyan]{location}[/cyan]" if location else ""
    _console.print(
        f"[bold]● Recording[/bold]  [cyan]{class_name}[/cyan]{location_str}  "
        f"[dim]clip duration: {cfg.onset.record_seconds:.2f}s ({clip_frames:,} frames @ {input_sr} Hz) · "
        f"≤ {duration_s/60:.0f} min · Ctrl+C to stop[/dim]\n"
        f"  [dim]onset gates:  floor [/dim][bold]{cfg.onset.min_flux_db:+.0f} dB[/bold]"
        f"[dim]  ·  mad_k [/dim][bold]{cfg.onset.mad_k}[/bold]"
        f"[dim]   (effective gate = max(floor, median+mad_k·MAD))[/dim]"
    )

    # Surface the most common multichannel footgun before the session starts:
    # a selected channel that has nothing connected — or the wrong INDEX, since
    # channels are 0-based, so interface input N is index N-1. Such a channel
    # never rises above the noise floor, so onset-triggered recording would
    # silently capture nothing and the user is left wondering why.
    levels = _probe_channel_levels(capture, record_indices)
    dead = [
        (c, pk) for (c, pk, rms) in levels
        if pk < _SILENT_PEAK_DBFS and rms < _SILENT_RMS_DBFS
    ]
    if dead:
        chans = ", ".join(f"ch{c} ({pk:+.0f} dBFS peak)" for c, pk in dead)
        _console.print(
            f"  [yellow]⚠ selected channel(s) look silent:[/yellow] {chans}\n"
            f"    [dim]Channels are 0-based indexes — interface input N is "
            f"index N-1 (input 1 → ch0, input 4 → ch3). '-ch 4' selects index "
            f"4 (the 5th input), not four channels. Check the index or the "
            f"connection.[/dim]"
        )

    # Flash state: when an onset fires, flash_until is set to time + 0.5s.
    # While active, the level bar renders fully filled in bright_cyan.
    flash_until = [0.0]   # mutable from closure; monotonic time

    def _level_bar(dbfs: float, width: int = 20) -> str:
        """Map dBFS (typically -80…0) to a filled bar string."""
        if time.monotonic() < flash_until[0]:
            return f"[bright_cyan]{'█' * width}[/bright_cyan]"
        frac = max(0.0, min(1.0, (dbfs + 60) / 60))  # -60 dBFS = empty, 0 dBFS = full
        filled = int(frac * width)
        color = "green" if dbfs < -12 else ("yellow" if dbfs < -3 else "red")
        return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"

    def _status_line() -> Text:
        dbfs, flux, thresh = aggregate_status()
        bar = _level_bar(dbfs)
        saved = session.count
        level = f"{dbfs:+.0f} dBFS"
        # Single effective gate = max(absolute floor, adaptive threshold).
        # That's the actual decision threshold; flux must exceed it to trigger.
        floor_lin = cfg.onset.min_flux
        gate_lin  = max(floor_lin, thresh)
        which = "floor" if floor_lin >= thresh else "adaptive"
        flux_db = _linear_to_db(flux)
        gate_db = _linear_to_db(gate_lin)
        will_trigger = flux >= gate_lin
        mark   = "[bright_green]✓[/bright_green]" if will_trigger else "[red]✗[/red]"
        fc     = "bright_green" if will_trigger else "dim"
        flux_str = (
            f"[{fc}]flux {flux_db:+5.1f} dB[/{fc}]  "
            f"[dim]vs gate[/dim] {mark}[dim] {gate_db:+5.1f} dB[/dim]  "
            f"[dim]({which})[/dim]"
        )
        label = f"[dim]clips:[/dim] [bold]{saved}{limit_str}[/bold]"
        t = Text.from_markup(f"  {bar}  {level:>9}   {flux_str}   {label}")
        return t

    def _peak_dbfs_1d(wave: np.ndarray) -> float:
        peak = float(np.max(np.abs(wave))) if wave.size else 0.0
        return 20.0 * float(np.log10(peak + 1e-10)) if peak > 0 else -120.0

    def _peak_dbfs_by_channel(wave: np.ndarray, channels: list[int]) -> dict[int, float]:
        if wave.ndim == 1:
            return {channels[0]: _peak_dbfs_1d(wave)} if channels else {}
        return {
            ch: _peak_dbfs_1d(wave[:, i])
            for i, ch in enumerate(channels)
            if i < wave.shape[1]
        }

    prev_count = 0
    last_saved_lines: list[str] = []      # FIFO of terminal summaries
    start = time.monotonic()
    try:
        with Live(_status_line(), console=_console, refresh_per_second=8,
                  transient=False) as live:
            while time.monotonic() - start < duration_s:
                time.sleep(0.05)

                # ---- mature ready events into saved clips ----
                now_idx = capture.ring.write_idx
                ready: list[PendingTriggerGroup] = []
                with pending_lock:
                    still_pending = []
                    for group in pending:
                        if group.target_end_idx <= now_idx:
                            ready.append(group)
                        else:
                            still_pending.append(group)
                    pending[:] = still_pending

                for group in ready:
                    if max_clips is not None and session.count >= max_clips:
                        break
                    ev = group.onset
                    if trigger_mode == "any":
                        saved_channels = _order_channels(
                            list(group.events_by_channel.keys()),
                            record_indices,
                        )
                        if not saved_channels:
                            saved_channels = [record_indices[0]]
                        event_by_channel = group.events_by_channel
                    elif trigger_mode == "any_all":
                        saved_channels = list(record_indices)
                        event_by_channel = group.events_by_channel
                    else:
                        saved_channels = list(record_indices)
                        triggered = (
                            [record_indices[0]]
                            if trigger_mode == "ch0"
                            else list(record_indices)
                        )
                        event_by_channel = {ch: ev for ch in triggered}
                    pre_n = pick_preroll_n()
                    clip_start = int(ev.sample_idx * sr_ratio) - pre_n
                    try:
                        chans = [
                            capture.ring.read_window(clip_start, clip_n, channel=c)
                            for c in saved_channels
                        ]
                        wave = np.stack(chans, axis=-1)
                        triggered_channels = _order_channels(
                            list(event_by_channel.keys()),
                            saved_channels,
                        )
                        peak_db_by_channel = _peak_dbfs_by_channel(wave, saved_channels)
                        flux_by_channel = {
                            ch: float(event_by_channel.get(ch, ev).strength)
                            for ch in triggered_channels
                        }
                        flux_db_by_channel = {
                            ch: _linear_to_db(value)
                            for ch, value in flux_by_channel.items()
                        }
                        trigger_info = ClipTriggerInfo(
                            recorded_channels=list(saved_channels),
                            triggered_channels=list(triggered_channels),
                            triggered_flux={str(k): v for k, v in flux_by_channel.items()},
                            triggered_peak_dbfs={
                                str(k): peak_db_by_channel[k]
                                for k in triggered_channels
                                if k in peak_db_by_channel
                            },
                        )
                        last_saved_lines.append(_format_recorded_clip_line(
                            session.count + len(last_saved_lines) + 1,
                            limit_str,
                            class_name,
                            list(saved_channels),
                            flux_db_by_channel,
                            peak_db_by_channel,
                        ))
                        flash_until[0] = time.monotonic() + 0.5
                        session.submit_clip(
                            wave, onset=ev, trigger_info=trigger_info,
                            onset_offset_s=pre_n / float(input_sr),
                        )
                    except ValueError as e:
                        log.warning("ring overrun, clip dropped: %s", e)

                # ---- gate logic: stop accepting NEW onsets once we have
                #      enough in-flight, but keep maturing what's pending.
                if max_clips is not None:
                    with pending_lock:
                        in_flight = len(pending) + session._save_q.qsize()
                    if session.count + in_flight >= max_clips:
                        accept_new[0] = False
                    if session.count >= max_clips and in_flight == 0:
                        break

                # ---- print one line per newly-saved clip ----
                if session.count > prev_count:
                    while prev_count < session.count:
                        prev_count += 1
                        line = last_saved_lines.pop(0) if last_saved_lines else ""
                        live.console.print(line or _format_saved_count_line(prev_count, limit_str))

                # Decay peak so the bar responds to silence
                for det in detectors:
                    det._peak_rms *= 0.85
                live.update(_status_line())

    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        session.stop()
        # Flush any straggler "saved" prints between the last loop iteration
        # and writer-thread completion.
        while prev_count < session.count:
            prev_count += 1
            line = last_saved_lines.pop(0) if last_saved_lines else ""
            _console.print(line or _format_saved_count_line(prev_count, limit_str))

    _console.print(
        f"\n[bold]Done.[/bold]  {session.count} clip(s) saved → [dim]{out_dir}[/dim]"
    )
    return session.count
