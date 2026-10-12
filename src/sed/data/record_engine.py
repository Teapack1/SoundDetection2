"""Headless onset-triggered recording.

This is the recording loop itself, with no terminal and no HTTP in it. Two
front-ends drive it: ``sed record`` renders it with Rich
(:func:`sed.data.recorder.record_auto`), and ``sed record --serve`` polls
:meth:`RecordEngine.snapshot` from a browser
(:mod:`sed.data.record_server`).

There is one implementation because the loop is not simple: cross-microphone
onset merging, a global recording cooldown, per-clip randomized preroll,
ring-buffer maturity before slicing, and an exact ``max_clips`` gate that still
lets already-detected events finish. A second copy of that in the web server
would drift from what the CLI records, and the library would end up holding two
subtly different kinds of clip.

What the engine adds over the original loop is the ability to change your mind
without stopping the audio stream. On site the gate is always slightly wrong at
first, so :meth:`set_onset` recomputes the derived sample counts in place, and
:meth:`set_metadata` re-aims where the next clip lands. Only the capture
topology — device, rate, channels — needs the stream reopened.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar

import numpy as np

from sed.audio.capture import AudioCapture, check_input_settings
from sed.audio.onset import OnsetEvent, StreamingSuperflux
from sed.audio.resample import StreamingResampler, model_rate_index
from sed.config import LIBRARY_PROJECT_ID, Config, OnsetCfg
from sed.data.library import HOLDOUT_TAG, RAW_CATEGORY, Library
from sed.data.recorder import (
    _SILENT_PEAK_DBFS,
    _SILENT_RMS_DBFS,
    ClipContext,
    ClipTriggerInfo,
    PendingTriggerGroup,
    RecordingSession,
    _group_onset_events,
    _linear_to_db,
    _order_channels,
    _probe_channel_levels,
    _trigger_group_target_end_idx,
)

log = logging.getLogger(__name__)

# A channel peaking this close to full scale is being driven into the converter
# ceiling; the transient we care about is already destroyed by the time the
# clip is written, so the operator has to see it while it is still fixable.
CLIPPING_PEAK_DBFS = -0.5

# The engine ticks fast enough that a browser polling levels at ~20 Hz always
# has a fresh number to draw.
_TICK_SECONDS = 0.02
# Peak release, per 20 ms tick — about 25 dB/s, in the range a hardware peak
# meter uses. The attack is instantaneous (the level is a running max), so a
# transient still registers immediately; only the fall is eased, which is what
# makes the bar readable instead of flickering.
PEAK_RELEASE_DB_PER_TICK = 0.5
# The UI sparkline is about 100 px wide, so sampling it every tick would ship
# 1200 points per channel on every poll to draw the same line. One point every
# half second over a minute is 120, which is already more than it can show.
_LEVEL_HISTORY_SECONDS = 60.0
_LEVEL_HISTORY_INTERVAL_S = 0.5
_LEVEL_HISTORY_POINTS = int(_LEVEL_HISTORY_SECONDS / _LEVEL_HISTORY_INTERVAL_S)

MODES = ("onset", "continuous", "manual")

# Ring buffer length, as a multiple of the clip length. A clip is read some time
# after it matured, so the buffer must hold the clip plus that lag with margin;
# at less than ~2x the read races the writer for the samples it is copying.
RING_CLIP_MULTIPLE = 4.0
RING_MIN_SECONDS = 8.0

# Keys the operator may retune mid-session. Everything else in [onset] is
# either derived (postroll_ms) or fixed by the detector's construction.
LIVE_ONSET_KEYS = (
    "min_flux_db",
    "mad_k",
    "median_window_s",
    "recording_cooldown_ms",
    "cross_channel_merge_ms",
    "record_seconds",
    "preroll_ms",
    "randomize_preroll",
    "preroll_ms_range",
    "multi_mic_trigger",
    "hop_ms",
    "trigger_n_mels",
)

# Changing one of these reopens the PortAudio stream.
CAPTURE_KEYS = ("device", "sample_rate", "channels", "blocksize")


@dataclass
class ClipMetadata:
    """Provenance stamped onto the next clip the engine writes.

    Every field is live: an operator who notices the room filled up can switch
    density from ``quiet`` to ``busy`` without losing the session.
    """

    label: str
    density: str = "normal"
    description: str = ""
    location: str = ""
    mic_position: str = ""
    tags: list[str] = field(default_factory=list)
    holdout: bool = False
    split_channels: bool = False
    # Per recorded channel, in record_channels order. Empty means "use whatever
    # the config says" — which for a project-less session is only ever
    # "generic", so the console lets an operator name the mics they brought.
    microphone_type: list[str] = field(default_factory=list)

    def sidecar_tags(self) -> list[str]:
        """Tags as saved. ``holdout`` is a tag, not clip-level membership — a
        frozen-test source rule decides what actually lands in test.csv."""
        tags = [t for t in self.tags if t]
        if self.holdout and HOLDOUT_TAG not in tags:
            tags = [*tags, HOLDOUT_TAG]
        return tags


@dataclass(frozen=True)
class _Derived:
    """Values the audio callback and tick loop read on every block.

    Frozen and swapped as a whole, so a retune can never be observed halfway —
    the callback either sees the old geometry or the new one, never a clip
    length from one and a preroll from the other.
    """

    onset: OnsetCfg
    input_sr: int
    model_sr: int
    record_indices: tuple[int, ...]
    trigger_mode: str
    detectors: tuple[StreamingSuperflux, ...]
    detector_channels: tuple[int | None, ...]
    # One per detector. Stateful, so they travel with the detectors and are
    # rebuilt only when the rates actually change — see set_onset.
    resamplers: tuple[StreamingResampler, ...]
    sr_ratio: float
    clip_n: int
    fixed_pre_n: int
    maturity_post_n: int
    merge_window: int
    merge_window_input_n: int
    recording_cooldown: int

    def preroll_n(self) -> int:
        if self.onset.randomize_preroll:
            lo_ms, hi_ms = self.onset.preroll_ms_range
            return round(random.randint(lo_ms, hi_ms) / 1000 * self.input_sr)
        return self.fixed_pre_n


class FluxCalibrator:
    """Measure a room's background spectral flux and propose an onset floor.

    Owned by whatever is listening — a recording session, or the idle meter
    before one starts — so the gate can be set up before the first take rather
    than discovered during it.

    The floor's job is to sit above everything the room does on its own. That
    includes the operator: walking, setting a glass down, brushing a stand are
    all real transients, and a floor fitted to the quiet median will fire on
    every one of them. So the suggestion comes from a high percentile of what
    was actually heard, plus margin — calibrate while moving around normally
    and the result is a gate that ignores you.
    """

    # Percentile of observed background flux the floor is placed above.
    STRICTNESS: ClassVar[dict[str, tuple[float, float]]] = {
        "relaxed": (90.0, 3.0),    # catches quiet events; fires on movement
        "normal": (99.0, 6.0),     # ignores incidental noise
        "strict": (99.9, 10.0),    # deliberate sounds only
    }

    def __init__(self) -> None:
        self._state = "idle"
        self._samples: list[float] = []
        self._until = 0.0
        self._seconds = 0.0
        self._strictness = "normal"
        self._result: dict = {}
        self._error = ""

    def start(self, seconds: float = 10.0, strictness: str = "normal") -> None:
        if strictness not in self.STRICTNESS:
            raise ValueError(
                f"strictness must be one of {'|'.join(self.STRICTNESS)}"
            )
        self._state = "listening"
        self._samples = []
        self._seconds = float(seconds)
        self._until = time.monotonic() + float(seconds)
        self._strictness = strictness
        self._result = {}
        self._error = ""

    @property
    def listening(self) -> bool:
        return self._state == "listening"

    def observe(self, flux: float) -> None:
        if self._state == "listening" and flux > 0.0:
            self._samples.append(float(flux))

    def advance(self) -> None:
        """Finish once the listening window has elapsed."""
        if self._state != "listening" or time.monotonic() < self._until:
            return
        if len(self._samples) < 8:
            self._state = "failed"
            self._error = "not enough audio to measure — is the stream live?"
            return
        arr = np.asarray(self._samples, dtype=np.float64)
        percentile, margin_db = self.STRICTNESS[self._strictness]
        self._result = {
            "suggested_min_flux_db": round(
                _linear_to_db(float(np.percentile(arr, percentile))) + margin_db, 1
            ),
            "background_p50_db": round(_linear_to_db(float(np.median(arr))), 1),
            "background_p95_db": round(
                _linear_to_db(float(np.percentile(arr, 95))), 1),
            "background_max_db": round(_linear_to_db(float(arr.max())), 1),
            "strictness": self._strictness,
            "margin_db": margin_db,
        }
        self._state = "done"

    def snapshot(self) -> dict:
        out = {"state": self._state, "sample_count": len(self._samples),
               "seconds": self._seconds, "strictness": self._strictness}
        if self._state == "listening":
            out["remaining_s"] = round(max(0.0, self._until - time.monotonic()), 1)
        if self._error:
            out["error"] = self._error
        out.update(self._result)
        return out


class _ManualGroup(PendingTriggerGroup):
    """A trigger group the operator asked for.

    It rides the same maturity path as a detected onset — the clip geometry is
    identical — but the take has to be labelled honestly, because "did the
    detector find this?" is exactly what an operator is judging when they look
    back over the session.
    """


@dataclass
class _ChannelLevel:
    peak_dbfs: float = -120.0
    rms_dbfs: float = -120.0
    clips: int = 0
    flash_until: float = 0.0
    history: deque = field(
        default_factory=lambda: deque(maxlen=_LEVEL_HISTORY_POINTS)
    )


def apply_onset_updates(onset: OnsetCfg, updates: dict) -> OnsetCfg:
    """Return ``onset`` with ``updates`` applied, validated as a whole.

    Two details make this worth having in one place rather than at each call
    site. ``OnsetCfg`` lets the linear ``min_flux`` win when both it and
    ``min_flux_db`` are set, so a dB edit has to carry the matching linear
    value or it is silently discarded — and the validator then re-derives dB
    from that linear value, which turns the operator's -62 into
    -61.99999998906512.

    That rewrite happens on *every* revalidation, not just when the floor is
    the thing being changed: dragging the clip length was enough to turn a
    hand-written -80 into -79.99999991314111 and write it back to config.toml.
    So the gate is always restored afterwards — to the requested figure when
    the floor is what moved, and to the untouched one when it is not.
    """
    unknown = set(updates) - set(LIVE_ONSET_KEYS)
    if unknown:
        raise ValueError(
            f"not live-tunable: {', '.join(sorted(unknown))} "
            f"(tunable: {', '.join(LIVE_ONSET_KEYS)})"
        )
    data = onset.model_dump()
    data.pop("postroll_ms", None)          # derived from record_seconds
    data.update(updates)
    if "min_flux_db" in updates:
        gate_db = float(updates["min_flux_db"])
        gate_lin = float(10.0 ** (gate_db / 20.0))
        data["min_flux"] = gate_lin
    else:
        gate_db, gate_lin = float(onset.min_flux_db), float(onset.min_flux)
    return OnsetCfg.model_validate(data).model_copy(
        update={"min_flux_db": gate_db, "min_flux": gate_lin}
    )


def finalize_onset(onset: OnsetCfg) -> OnsetCfg:
    """Re-derive ``postroll_ms`` and validate the timing as a whole.

    ``OnsetCfg``'s own validators do this on construction; re-running them
    through ``model_validate`` is what makes a live retune reject an impossible
    combination (preroll past the end of the clip) instead of half-applying it.

    Timing only: the gate is carried through untouched. Revalidating a section
    that already has both ``min_flux`` and ``min_flux_db`` set makes the linear
    value win and the dB be recomputed from it, so an operator's ``-ot -77``
    would come back as -76.99999993850861 — and that is the number that ends up
    in their config.toml and in the session header.
    """
    data = onset.model_dump()
    data.pop("postroll_ms", None)
    return OnsetCfg.model_validate(data).model_copy(
        update={"min_flux_db": float(onset.min_flux_db),
                "min_flux": float(onset.min_flux)}
    )


class RecordEngine:
    """One recording session: capture, detect, slice, write.

    Not reusable across sessions — ``stop()`` is terminal. The web server
    constructs a fresh engine per Start.
    """

    def __init__(
        self,
        cfg: Config,
        project_id: str,
        metadata: ClipMetadata,
        *,
        on_save=None,
        max_clips: int | None = None,
        duration_s: float | None = None,
        mode: str = "onset",
        library: Library | None = None,
        category: str = RAW_CATEGORY,
        review_status: str = "accepted",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {'|'.join(MODES)} (got {mode!r})")
        self.cfg = cfg.model_copy(deep=True)
        self.project_id = project_id
        self.category = category
        self.review_status = review_status
        self.library = library if library is not None else Library()
        self.max_clips = max_clips
        self.duration_s = duration_s
        self.on_save = on_save

        self._meta = metadata
        self._mode = mode
        self._state = "idle"
        self._error = ""
        self._started_at: float | None = None
        self._stop_reason = ""

        self._lock = threading.Lock()          # guards metadata + published state
        # Separate, and never held while the metadata lock is: swapping a
        # session joins the writer, and the writer needs the metadata lock.
        self._session_lock = threading.RLock()
        self._pending_lock = threading.Lock()  # guards pending + last-group index
        self._stop_event = threading.Event()

        self._capture: AudioCapture | None = None
        self._session: RecordingSession | None = None
        self._tick: threading.Thread | None = None

        self._pending: list[PendingTriggerGroup] = []
        self._last_group_onset_idx: list[int | None] = [None]
        self._accept_new = True
        self._levels: dict[int, _ChannelLevel] = {}
        self._recent: deque = deque(maxlen=12)
        self._log: deque = deque(maxlen=200)
        # Every clip this run wrote, uncapped. `_recent` and `_log` are display
        # buffers and forget the beginning of a long pass; discarding a run has
        # to be able to name all of it.
        self._clips: list[dict] = []
        # Per-clip detail the sidecar does not carry (which mics fired, and at
        # what flux). Written by the tick thread, drained by the writer thread;
        # both are FIFO over the same clips, so the pairing holds.
        self._clip_notes: deque = deque()
        # Counted here, not on the RecordingSession: a label change starts a
        # fresh session (and a fresh per-session file counter), but the
        # operator's clip total and the max_clips gate span the whole run.
        # `_submitted` leads `_written`; the difference is the write queue.
        self._submitted = 0
        self._written = 0
        self._dropped_clips = 0
        self._dropouts = 0
        self._last_dropout_idx: int | None = None
        self._dead_channels: list[int] = []
        self._continuous_next_start: int | None = None
        self._ticks = 0

        self._calibrator = FluxCalibrator()

        self._derived = self._build_derived(self.cfg.onset)
        self._reset_levels()

    # ---- construction helpers ------------------------------------------

    def _build_derived(self, onset: OnsetCfg) -> _Derived:
        onset = finalize_onset(onset)
        input_sr = int(self.cfg.audio.record_sample_rate)
        model_sr = int(self.cfg.model.sample_rate)
        record_indices = tuple(int(c) for c in (self.cfg.audio.record_channels or [0]))

        trigger_mode = onset.multi_mic_trigger
        if trigger_mode in {"any", "any_all"}:
            detector_channels: tuple[int | None, ...] = record_indices
        elif trigger_mode == "ch0":
            detector_channels = (record_indices[0],)
        else:  # "mix"
            detector_channels = (None,)

        # The detector's own refractory stays small so a suppressed cross-mic
        # hit cannot stretch the operator-facing cooldown, which is enforced
        # globally when a new trigger group is accepted.
        detector_refractory_ms = min(max(0, int(onset.recording_cooldown_ms)), 150)
        detectors = tuple(
            StreamingSuperflux(
                sample_rate=model_sr,
                hop_ms=onset.hop_ms,
                n_mels=onset.trigger_n_mels,
                median_window_s=onset.median_window_s,
                mad_k=onset.mad_k,
                refractory_ms=detector_refractory_ms,
                min_flux=onset.min_flux,
            )
            for _ in detector_channels
        )

        # round(), not int(): truncating a float second count made record_seconds
        # 2.9 read as 2899 ms, and left maturity a sample or two short of what
        # the slice needs, so an occasional clip matured before its own audio
        # existed and was dropped.
        clip_ms_total = round(onset.record_seconds * 1000)
        if onset.randomize_preroll:
            # The maturity check needs the WORST case: the smallest preroll
            # leaves the largest postroll still to be captured.
            rand_lo_ms, _hi = onset.preroll_ms_range
            maturity_post_n = round((clip_ms_total - rand_lo_ms) / 1000 * input_sr)
        else:
            maturity_post_n = round(onset.postroll_ms / 1000 * input_sr)
        # One block of slack so a boundary-case tick cannot ask for a sample
        # that lands after the ring's last write.
        maturity_post_n += int(self.cfg.audio.blocksize)

        resamplers = tuple(
            StreamingResampler(input_sr, model_sr) for _ in detector_channels
        )

        return _Derived(
            onset=onset,
            input_sr=input_sr,
            model_sr=model_sr,
            record_indices=record_indices,
            trigger_mode=trigger_mode,
            detectors=detectors,
            detector_channels=detector_channels,
            resamplers=resamplers,
            sr_ratio=input_sr / model_sr,
            clip_n=round(onset.record_seconds * input_sr),
            fixed_pre_n=round(onset.preroll_ms / 1000 * input_sr),
            maturity_post_n=maturity_post_n,
            merge_window=round(onset.cross_channel_merge_ms / 1000 * model_sr),
            merge_window_input_n=round(onset.cross_channel_merge_ms / 1000 * input_sr),
            recording_cooldown=round(onset.recording_cooldown_ms / 1000 * model_sr),
        )

    def _mic_names(self) -> tuple[str, ...]:
        """Microphone names per recorded channel, operator override first."""
        chosen = [str(m).strip() for m in self._meta.microphone_type if str(m).strip()]
        if chosen:
            return tuple(chosen)
        return tuple(str(m) for m in self.cfg.audio.microphone_type)

    def _reset_levels(self) -> None:
        keep = self._levels
        self._levels = {
            ch: keep.get(ch, _ChannelLevel()) for ch in self._derived.record_indices
        }

    def _new_session(self, meta: ClipMetadata) -> RecordingSession:
        return RecordingSession(
            self.cfg,
            self.library.raw_dir(meta.label),
            class_name=meta.label,
            label=meta.label,
            category=self.category,
            project_id=self.project_id,
            density=meta.density,
            review_status=self.review_status,
            mic_position=meta.mic_position,
            location=meta.location,
            description=meta.description,
            tags=meta.sidecar_tags(),
            on_save=self._on_save,
        )

    # ---- lifecycle ------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def clips_saved(self) -> int:
        """Clips fully written AND registered in the library.

        Deliberately lags submission: a front-end that shows this number can
        also show the clip it refers to, because both are published by the same
        writer-thread callback.
        """
        return self._written

    def start(self) -> None:
        if self._state != "idle":
            raise RuntimeError(f"engine already {self._state}")
        self._session = self._new_session(self._meta)
        self._session.start()
        try:
            self._open_capture()
        except Exception:
            self._session.stop()
            self._session = None
            self._state = "idle"
            raise
        self._started_at = time.monotonic()
        self._state = "running"
        self._tick = threading.Thread(target=self._tick_loop, daemon=True,
                                      name="record-engine")
        self._tick.start()
        threading.Thread(target=self._probe_dead_channels, daemon=True).start()

    def _ring_seconds(self, d: _Derived) -> float:
        """Ring length for this geometry.

        A clip is sliced some time after its onset matured, so the buffer has to
        hold the clip plus the lag, with room to spare. RING_CLIP_MULTIPLE is
        that margin; below roughly 2x the clip length a read starts racing the
        writer for the samples it is copying.
        """
        return max(RING_MIN_SECONDS, float(d.onset.record_seconds) * RING_CLIP_MULTIPLE)

    def _ring_fits(self) -> bool:
        capture = self._capture
        if capture is None:
            return True
        d = self._derived
        return capture.ring.capacity >= int(self._ring_seconds(d) * d.input_sr)

    def _ensure_ring_fits(self) -> None:
        """Reopen the stream if a retune outgrew the buffer.

        The ring is sized when the stream opens, but the clip length is tunable
        while it runs. Growing the clip past the buffer used to make every
        subsequent read fail into a warning while the console still showed a
        healthy meter and 'running' — a session that records nothing and says
        nothing. Reopening costs about a second of audio; recording nothing
        costs the session.
        """
        if self._state != "running" or self._ring_fits():
            return
        log.info("clip length outgrew the ring buffer — reopening the input stream")
        self._reopen_capture()

    def _reopen_capture(self) -> None:
        """Close and reopen the input stream, discarding stale in-flight state.

        The new ring restarts its sample clock at zero, so anything still
        holding an index from the old stream is meaningless and must go — a
        stale group would otherwise never mature and would hold the max_clips
        gate shut for the rest of the session.
        """
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.stop()
        with self._pending_lock:
            self._pending.clear()
            self._last_group_onset_idx[0] = None
            self._continuous_next_start = None
            for rs in self._derived.resamplers:
                rs.reset()
            for det in self._derived.detectors:
                det.reset()
        self._open_capture()

    def _open_capture(self) -> None:
        if not wait_for_warm_clip_writer():
            log.warning("clip-writer warm-up did not finish; opening the stream anyway")
        d = self._derived
        check_input_settings(
            device=self.cfg.deploy.record_device,
            channels=max(d.record_indices) + 1,
            sample_rate=d.input_sr,
        )
        self._capture = AudioCapture(
            sample_rate=d.input_sr,
            channels=max(d.record_indices) + 1,
            device=self.cfg.deploy.record_device,
            blocksize=int(self.cfg.audio.blocksize),
            ring_seconds=self._ring_seconds(d),
            on_block=self._on_block,
        )
        self._capture.start()

    def stop(self) -> dict:
        """Stop capture, flush queued writes, and return a session summary.

        Safe to call on a session that already finished on its own — that is
        the normal path when ``max_clips`` or the time cap is reached.
        """
        if self._state in ("idle", "stopped"):
            return self.summary()
        self._state = "stopping"
        self._stop_event.set()
        if self._tick is not None:
            self._tick.join(timeout=5.0)
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        if self._session is not None:
            self._session.stop()
        self._state = "stopped"
        return self.summary()

    def summary(self) -> dict:
        session = self._session
        records = self.clip_records()
        return {
            "clips": self.clips_saved,
            "label": self._meta.label,
            # A label change rolls a new session, so one pass can span several.
            "labels": list(dict.fromkeys(c["label"] for c in records)),
            "session_ids": list(dict.fromkeys(c["session_id"] for c in records)),
            "session_id": getattr(session, "session_id", ""),
            "out_dir": str(getattr(session, "out_dir", "")),
            "elapsed_s": self.elapsed_s(),
            "reason": self._stop_reason,
        }

    def elapsed_s(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    # ---- audio thread ---------------------------------------------------

    def _channel_signals(self, block: np.ndarray, d: _Derived) -> list[np.ndarray]:
        """One 1-d signal per detector, in input-rate space."""
        if block.ndim == 1:
            return [block]
        if d.trigger_mode == "ch0":
            return [block[:, d.record_indices[0]]]
        if d.trigger_mode == "mix":
            return [block[:, list(d.record_indices)].mean(axis=-1).astype(np.float32)]
        return [block[:, c] for c in d.record_indices]

    def _update_levels(self, block: np.ndarray, d: _Derived) -> None:
        """Per-channel peak/RMS for every recorded channel, whatever the
        trigger mode.

        The detectors only cover the channels that can trigger — with ``mix``
        or ``ch0`` that is one detector for the whole card — so meters have to
        come from the block directly or a console would show a dead microphone
        as healthy.
        """
        for ch in d.record_indices:
            if block.ndim > 1:
                if ch >= block.shape[1]:
                    continue
                x = block[:, ch]
            else:
                x = block
            level = self._levels.get(ch)
            if level is None:
                continue
            peak = float(np.max(np.abs(x))) if x.size else 0.0
            rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
            peak_db = 20.0 * math.log10(peak + 1e-10)
            rms_db = 20.0 * math.log10(rms + 1e-10)
            # Peak holds and is decayed by the tick loop, so a short transient
            # is still visible on a browser polling at 5 Hz.
            level.peak_dbfs = max(level.peak_dbfs, peak_db)
            level.rms_dbfs = rms_db

    def _on_block(self, block: np.ndarray, write_idx: int) -> None:
        d = self._derived                       # one atomic read per block
        self._update_levels(block, d)

        signals = self._channel_signals(block, d)
        if not signals:
            return
        raw_events: list[OnsetEvent] = []
        for det, rs, sig, channel in zip(
            d.detectors, d.resamplers, signals, d.detector_channels, strict=False
        ):
            # Streaming, not per-block: resampling each block in isolation puts
            # a filter seam at every boundary, which a spectral-flux detector
            # reads as a transient. The resampler's own output counter is the
            # detector's timebase, so the two cannot drift.
            sig_rs = rs.process(sig)
            if sig_rs.size == 0:
                continue
            for ev in det.process_block(sig_rs, rs.output_index):
                raw_events.append(
                    OnsetEvent(sample_idx=ev.sample_idx, strength=ev.strength,
                               baseline=ev.baseline, channel=channel)
                )
        if not raw_events:
            return
        if self._mode != "onset" or not self._accept_new:
            return
        with self._pending_lock:
            _group_onset_events(
                self._pending,
                raw_events,
                merge_window=d.merge_window,
                target_end_for_event=lambda ev: _trigger_group_target_end_idx(
                    ev,
                    sr_ratio=d.sr_ratio,
                    maturity_post_n=d.maturity_post_n,
                    merge_window_n=d.merge_window_input_n,
                ),
                recording_cooldown=d.recording_cooldown,
                last_group_onset_idx=self._last_group_onset_idx,
            )

    # ---- tick loop ------------------------------------------------------

    def _tick_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(_TICK_SECONDS)
                if self._capture is None:
                    continue
                self._handle_dropout()
                self._mature_pending()
                self._schedule_continuous()
                self._decay_levels()
                self._advance_calibration()
                if self._should_finish():
                    # The session met its own target or time cap. Say so in the
                    # state, or every front-end waiting on "running" would hang:
                    # the loop is over but nothing told them.
                    self._state = "finished"
                    break
        except Exception as exc:  # noqa: BLE001 — surface it, never die silently
            log.exception("recording tick loop failed")
            self._error = f"{type(exc).__name__}: {exc}"
            self._state = "error"

    def _handle_dropout(self) -> None:
        """Recover cleanly from an audio dropout.

        The ring keeps its sample clock across a reconnect — the clock simply
        stops while wall-clock time carries on — so the samples either side of
        the join are adjacent in the buffer but seconds apart in reality.
        `read_window` already refuses a clip that spans the join; the detectors
        need telling too, because to a spectral-flux detector that step is the
        largest transient it has ever seen and would fire an onset centred on
        the splice. The resamplers hold filter state from before the gap, which
        is equally meaningless afterwards.
        """
        capture = self._capture
        if capture is None:
            return
        gap = capture.ring.discontinuity_idx
        if gap is None or gap == self._last_dropout_idx:
            return
        self._last_dropout_idx = gap
        d = self._derived
        with self._pending_lock:
            stale = len(self._pending)
            self._pending.clear()
            self._last_group_onset_idx[0] = None
            self._continuous_next_start = None
            for rs in d.resamplers:
                rs.reset()
            for det in d.detectors:
                det.reset()
        self._dropouts += 1
        log.warning(
            "audio device dropped out and reconnected; detector state reset"
            "%s", f" ({stale} pending onset(s) discarded)" if stale else "",
        )

    def _should_finish(self) -> bool:
        if self.duration_s is not None and self.elapsed_s() >= self.duration_s:
            self._stop_reason = "duration reached"
            return True
        if self.max_clips is None:
            return False
        with self._pending_lock:
            in_flight = len(self._pending) + (self._submitted - self._written)
        # Recomputed every tick, in BOTH directions. Latching it off was a trap:
        # a clip that is dropped rather than written lowers the in-flight count
        # without raising the saved count, so the target could never be reached
        # and no new onset was ever accepted again — a session that reports
        # "running" forever and records nothing.
        self._accept_new = self.clips_saved + in_flight < self.max_clips
        if self.clips_saved >= self.max_clips and in_flight == 0:
            self._stop_reason = "target reached"
            return True
        return False

    def _decay_levels(self) -> None:
        # A VU meter has to fall fast enough to look live. The old 30 dB/s
        # release made every transient smear across half a second, which read
        # as a laggy meter rather than a held peak.
        for det in self._derived.detectors:
            det._peak_rms *= 0.7
        self._ticks += 1
        sample = self._ticks % max(1, round(_LEVEL_HISTORY_INTERVAL_S / _TICK_SECONDS)) == 0
        for level in self._levels.values():
            level.peak_dbfs = max(-120.0, level.peak_dbfs - PEAK_RELEASE_DB_PER_TICK)
            if sample:
                level.history.append(round(level.peak_dbfs, 1))

    def _mature_pending(self) -> None:
        d = self._derived
        now_idx = self._capture.ring.write_idx
        with self._pending_lock:
            ready = [g for g in self._pending if g.target_end_idx <= now_idx]
            self._pending = [g for g in self._pending if g.target_end_idx > now_idx]

        for group in ready:
            if self.max_clips is not None and self._submitted >= self.max_clips:
                break
            ev = group.onset
            if d.trigger_mode == "any":
                saved_channels = _order_channels(
                    list(group.events_by_channel.keys()), list(d.record_indices)
                ) or [d.record_indices[0]]
                event_by_channel = group.events_by_channel
            elif d.trigger_mode == "any_all":
                saved_channels = list(d.record_indices)
                event_by_channel = group.events_by_channel
            else:
                saved_channels = list(d.record_indices)
                triggered = (
                    [d.record_indices[0]] if d.trigger_mode == "ch0"
                    else list(d.record_indices)
                )
                event_by_channel = {ch: ev for ch in triggered}
            pre_n = d.preroll_n()
            self._emit_clip(
                d,
                start_idx=round(ev.sample_idx * d.sr_ratio) - pre_n,
                channels=saved_channels,
                onset=ev,
                event_by_channel=event_by_channel,
                onset_offset_s=pre_n / float(d.input_sr),
                kind="manual" if isinstance(group, _ManualGroup) else "onset",
            )

    def _schedule_continuous(self) -> None:
        """Tile the timeline with back-to-back clips, gate off.

        Room tone, crowd beds and calibration material have no transient to
        trigger on, and waiting for the onset detector to fire on them is
        exactly the wrong behaviour: what you want is every second of it.
        """
        if self._mode != "continuous":
            self._continuous_next_start = None
            return
        d = self._derived
        now_idx = self._capture.ring.write_idx
        if self._continuous_next_start is None:
            self._continuous_next_start = now_idx
            return
        # If the tick loop was starved (a slow disk, a busy machine), the next
        # position can fall out of the ring entirely. Replaying it would emit a
        # burst of clips that all fail to read; skip to the oldest audio still
        # present and say so once.
        oldest = now_idx - self._capture.ring.capacity + d.clip_n
        if self._continuous_next_start < oldest:
            dropped = (oldest - self._continuous_next_start) / d.input_sr
            log.warning(
                "continuous recording fell %.1fs behind the ring buffer; "
                "resuming from the oldest audio still held", dropped,
            )
            self._continuous_next_start = oldest
        while now_idx - self._continuous_next_start >= d.clip_n:
            if self.max_clips is not None and self._submitted >= self.max_clips:
                return
            start = self._continuous_next_start
            self._continuous_next_start = start + d.clip_n
            self._emit_clip(
                d,
                start_idx=start,
                channels=list(d.record_indices),
                onset=None,
                event_by_channel={},
                # No transient was detected, so claiming one would put a false
                # crop centre in the sidecar. Training falls back to locating
                # it, or to a centre crop.
                onset_offset_s=None,
                kind="continuous",
            )

    def trigger_manual(self) -> bool:
        """Capture around this instant, ignoring the gate and the cooldown.

        The clip keeps the normal preroll/postroll geometry, so the sound the
        operator just reacted to sits inside the preroll and the sidecar's
        ``onset_offset_s`` still means what it means everywhere else.
        """
        if self._state != "running" or self._capture is None:
            return False
        d = self._derived
        now_model_idx = model_rate_index(
            self._capture.ring.write_idx, d.input_sr, d.model_sr
        )
        ev = OnsetEvent(sample_idx=now_model_idx, strength=0.0, baseline=0.0,
                        channel=d.record_indices[0])
        group = _ManualGroup(
            onset=ev,
            target_end_idx=_trigger_group_target_end_idx(
                ev,
                sr_ratio=d.sr_ratio,
                maturity_post_n=d.maturity_post_n,
                merge_window_n=d.merge_window_input_n,
            ),
        )
        for ch in d.record_indices:
            group.add_event(replace(ev, channel=ch))
        with self._pending_lock:
            self._pending.append(group)
        return True

    def _emit_clip(
        self,
        d: _Derived,
        *,
        start_idx: int,
        channels: list[int],
        onset: OnsetEvent | None,
        event_by_channel: dict[int, OnsetEvent],
        onset_offset_s: float | None,
        kind: str,
    ) -> None:
        # `d` is the caller's geometry, deliberately not re-read: taking the
        # preroll from one retune and the clip length from another can slice a
        # window that does not contain the transient at all, while the sidecar
        # still claims it does.
        capture = self._capture
        if capture is None:
            return
        try:
            wave = np.stack(
                [capture.ring.read_window(start_idx, d.clip_n, channel=c)
                 for c in channels],
                axis=-1,
            )
        except (ValueError, IndexError) as exc:
            # IndexError as well as ValueError: a channel can disappear from
            # under a pending clip when the capture topology changes, and an
            # uncaught one used to kill the tick loop for the whole session.
            log.warning("clip dropped: %s", exc)
            self._dropped_clips += 1
            return

        peak_db_by_channel = {
            ch: _peak_dbfs(wave[:, i]) for i, ch in enumerate(channels)
            if i < wave.shape[1]
        }
        triggered_channels = _order_channels(list(event_by_channel.keys()), channels)
        flux_by_channel = {
            ch: float(event_by_channel[ch].strength) for ch in triggered_channels
        }
        trigger_info = ClipTriggerInfo(
            recorded_channels=list(channels),
            triggered_channels=list(triggered_channels),
            triggered_flux={str(k): v for k, v in flux_by_channel.items()},
            triggered_peak_dbfs={
                str(k): peak_db_by_channel[k]
                for k in triggered_channels if k in peak_db_by_channel
            },
        )
        now = time.monotonic()
        for ch in channels:
            level = self._levels.get(ch)
            if level is not None:
                level.clips += 1
                level.flash_until = now + 0.5
        session = self._session
        if session is None:
            return
        self._clip_notes.append({
            "kind": kind,
            "channels": list(channels),
            "flux_db": {str(ch): round(_linear_to_db(v), 1)
                        for ch, v in flux_by_channel.items()},
            "peak_dbfs": {str(ch): round(peak_db_by_channel[ch], 1)
                          for ch in channels if ch in peak_db_by_channel},
        })
        self._submitted += 1
        session.submit_clip(
            wave, onset=onset, trigger_info=trigger_info,
            onset_offset_s=onset_offset_s,
            context=ClipContext(
                sample_rate=d.input_sr,
                device=str(self.cfg.deploy.record_device),
                microphone_type=self._mic_names(),
                record_channels=tuple(d.record_indices),
            ),
        )

    def _on_save(self, wav_path: Path, meta) -> None:
        """Runs on the writer thread, right after the WAV and sidecar land."""
        try:
            note = self._clip_notes.popleft()
        except IndexError:  # pragma: no cover — writer outpaced the emitter
            note = {}
        self.library.append_sidecar(wav_path)
        if self._meta.split_channels and meta.channels > 1:
            try:
                self.library.split_clip(wav_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("channel split failed for %s: %s", wav_path.name, exc)
        entry = {
            "uuid": meta.uuid,
            "name": wav_path.name,
            "path": str(wav_path),
            "label": meta.label,
            "density": meta.density,
            "session_id": meta.session_id,
            "recorded_at": meta.recorded_at,
            "channels": list(meta.recorded_channels),
            "peak_dbfs": meta.peak_dbfs,
            "kind": note.get("kind", "onset"),
            "flux_db": note.get("flux_db", {}),
            "peak_dbfs_by_channel": note.get("peak_dbfs", {}),
            "index": meta.recording_counter,
        }
        with self._lock:
            self._recent.appendleft(entry)
            self._log.appendleft(entry)
            self._clips.append(entry)
            self._written += 1
        if self.on_save is not None:
            try:
                self.on_save(wav_path, meta)
            except Exception as exc:  # noqa: BLE001
                log.warning("on_save callback failed: %s", exc)

    # ---- live retuning --------------------------------------------------

    def set_project(self, project_id: str) -> None:
        """Change which project subsequent clips are stamped with.

        Clips already written keep the stamp they were saved with, so this
        rolls the recording session rather than rewriting history — the new
        clips belong to a different grouping than the old ones.
        """
        project_id = str(project_id or LIBRARY_PROJECT_ID)
        if project_id == self.project_id:
            return
        self.project_id = project_id
        if self._session is not None:
            self._roll_session(self._meta)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {'|'.join(MODES)} (got {mode!r})")
        with self._lock:
            self._mode = mode
            self._continuous_next_start = None

    def set_metadata(self, **fields) -> ClipMetadata:
        """Re-aim the next clip. Changing the label rolls a new session.

        A recording group is the unit that must never straddle a train/val
        split, and it is scoped by ``session_id`` — so a new label gets a new
        session rather than inheriting the previous one's identity.
        """
        unknown = set(fields) - set(ClipMetadata.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown metadata field(s): {', '.join(sorted(unknown))}")
        with self._lock:
            previous = self._meta
            new = replace(previous, **fields)
            if not new.label.strip():
                raise ValueError("label must not be empty")
            self._meta = new
            session = self._session
            if session is None:
                return new
            rolls = new.label != previous.label
            if not rolls:
                session.density = new.density
                session.description = new.description
                session.location = new.location
                session.mic_position = new.mic_position
                session.tags = new.sidecar_tags()
                return new
        # Outside the lock: session.stop() joins the writer thread, and the
        # writer takes this same lock in _on_save. Holding it across the join
        # deadlocked the two until the 30 s timeout expired, freezing the whole
        # console on every label change.
        self._roll_session(new)
        return new

    def _roll_session(self, meta: ClipMetadata) -> None:
        """Retire the current recording session and open a fresh one.

        A new session id is a new recording group — the unit that must never
        straddle a train/val split — so anything that changes what the clips
        are (the label) or how they were captured (the topology) rolls one.
        """
        with self._session_lock:
            old_session = self._session
            if old_session is not None:
                old_session.stop()
            session = self._new_session(meta)
            session.start()
            self._session = session

    def set_onset(self, **fields) -> OnsetCfg:
        """Retune the gate and the clip geometry without touching the stream.

        Rejected as a whole if the combination is invalid, so a bad preroll can
        never be applied on top of a good record length.
        """
        candidate = apply_onset_updates(self.cfg.onset, fields)

        keeps_detectors = all(
            getattr(candidate, k) == getattr(self.cfg.onset, k)
            for k in ("hop_ms", "trigger_n_mels", "median_window_s", "mad_k",
                      "min_flux", "multi_mic_trigger", "recording_cooldown_ms")
        )
        previous = self._derived
        # Built OUTSIDE the lock: this constructs a detector per channel, each
        # with a librosa mel filterbank, and the audio callback takes the same
        # lock on every block. Holding it across that reliably cost input
        # overflow on a retune.
        new_derived = self._build_derived(candidate)
        if keeps_detectors:
            # Rebuilding a detector throws away its adaptive baseline and costs
            # a fresh warm-up, so when only geometry changed: keep them.
            new_derived = replace(
                new_derived,
                detectors=previous.detectors,
                detector_channels=previous.detector_channels,
            )
        # The resamplers are stream state, and set_onset never changes a rate —
        # dropping them here would put a seam in the detector input at every
        # retune, which is the artifact this whole path exists to avoid.
        new_derived = replace(new_derived, resamplers=previous.resamplers)

        with self._pending_lock:
            self.cfg.onset = candidate
            self._derived = new_derived
            self._retime_pending(previous, new_derived)
            self._reset_levels()
        self._ensure_ring_fits()
        return candidate

    def _retime_pending(self, old: _Derived, new: _Derived) -> None:
        """Re-derive maturity for groups detected under the old geometry.

        A group's target_end_idx was computed from the clip length in force
        when it was detected, and the nominal margin is exactly zero
        (maturity_post_n == clip_n - preroll). Lengthening the clip therefore
        matures a group before its own audio has been captured, and it is
        dropped with nothing but a warning. Re-derive instead, and only give up
        on groups whose audio the ring can no longer hold.
        """
        if not self._pending or old is new:
            return
        kept = []
        for group in self._pending:
            group.target_end_idx = _trigger_group_target_end_idx(
                group.onset,
                sr_ratio=new.sr_ratio,
                maturity_post_n=new.maturity_post_n,
                merge_window_n=new.merge_window_input_n,
            )
            kept.append(group)
        dropped = len(self._pending) - len(kept)
        if dropped:
            log.warning("%d pending onset(s) dropped by the retune", dropped)
        self._pending = kept

    def set_capture(self, **fields) -> None:
        """Change device / rate / channels / blocksize, reopening the stream."""
        unknown = set(fields) - set(CAPTURE_KEYS)
        if unknown:
            raise ValueError(f"unknown capture field(s): {', '.join(sorted(unknown))}")
        device = fields.get("device")
        if device is not None:
            self.cfg.deploy.record_device = str(device)
        if "sample_rate" in fields:
            self.cfg.audio.record_sample_rate = int(fields["sample_rate"])
        if "channels" in fields:
            self.cfg.audio.record_channels = [int(c) for c in fields["channels"]]
        if "blocksize" in fields:
            self.cfg.audio.blocksize = int(fields["blocksize"])
        # Re-validate the section as a whole (duplicate/negative channel guards).
        self.cfg.audio = type(self.cfg.audio).model_validate(self.cfg.audio.model_dump())

        derived = self._build_derived(self.cfg.onset)
        with self._pending_lock:
            self._derived = derived
            self._reset_levels()
        if self._state != "running":
            return
        # The topology changed, so the clips before and after were captured
        # differently: that is a new recording group, not a continuation.
        self._roll_session(self._meta)
        # _reopen_capture stops the stream FIRST and only then clears pending.
        # The other order let a block still in flight insert a group indexed
        # against the old stream, which the new ring — whose clock restarts at
        # zero — could never mature, holding the max_clips gate shut forever.
        self._reopen_capture()

    # ---- calibration ----------------------------------------------------

    def start_calibration(self, seconds: float = 10.0,
                          strictness: str = "normal") -> dict:
        """Listen to the room and propose an onset floor."""
        if self._state != "running":
            raise RuntimeError("start the session before calibrating")
        self._calibrator.start(seconds, strictness)
        return self._calibrator.snapshot()

    def _advance_calibration(self) -> None:
        if self._calibrator.listening:
            for det in self._derived.detectors:
                _dbfs, flux, _thresh = det.get_status()
                self._calibrator.observe(flux)
        self._calibrator.advance()

    # ---- health ---------------------------------------------------------

    def _probe_dead_channels(self) -> None:
        """The most common multichannel footgun, caught before the session.

        Channel indexes are 0-based, so interface input N is index N-1. A
        mis-indexed channel never rises above the preamp noise floor, and
        onset-triggered recording on it silently captures nothing.
        """
        capture = self._capture
        if capture is None:
            return
        try:
            levels = _probe_channel_levels(capture, list(self._derived.record_indices))
        except Exception as exc:  # noqa: BLE001
            log.debug("channel probe failed: %s", exc)
            return
        self._dead_channels = [
            c for (c, pk, rms) in levels
            if pk < _SILENT_PEAK_DBFS and rms < _SILENT_RMS_DBFS
        ]

    def metadata(self) -> ClipMetadata:
        """The provenance the next clip will carry."""
        with self._lock:
            return self._meta

    def clip_records(self) -> list[dict]:
        """Every clip this run has written, oldest first."""
        with self._lock:
            return list(self._clips)

    def delete_clip(self, clip_uuid: str) -> bool:
        """Undo a take: WAV, sidecar, preview and manifest row."""
        removed = self.library.remove_clip(clip_uuid)
        if removed:
            self._forget(lambda c: c["uuid"] == clip_uuid)
        return removed

    def discard_all(self) -> int:
        """Delete every clip this run wrote. Returns how many were removed.

        For a pass that turned out to be unusable — the wrong glass, the wrong
        mic, a room that filled up halfway through — throwing it away here beats
        hunting for it in the library afterwards, where the only thing marking
        it out is a session id.
        """
        records = self.clip_records()
        if not records:
            return 0
        self.library.remove_clips([c["path"] for c in records])
        uuids = {c["uuid"] for c in records}
        self._forget(lambda c: c["uuid"] in uuids)
        return len(records)

    def _forget(self, matches) -> None:
        with self._lock:
            self._recent = deque(
                (c for c in self._recent if not matches(c)), maxlen=12
            )
            self._log = deque((c for c in self._log if not matches(c)), maxlen=200)
            self._clips = [c for c in self._clips if not matches(c)]

    # ---- snapshot -------------------------------------------------------

    def levels(self) -> dict:
        """Just the meters. Small enough to poll many times a second.

        The full snapshot carries recent takes, the session log and clip
        records — kilobytes per poll, which is what made the meters feel like
        they were lagging the room.
        """
        d = self._derived
        now = time.monotonic()
        status = self._detector_status(d)
        return {
            "state": self._state,
            "clips_saved": self._written,
            "channels": [
                self._channel_levels(d, i, ch, status, now)
                for i, ch in enumerate(d.record_indices)
            ],
        }

    def _detector_status(self, d: _Derived) -> dict:
        per_channel = {
            ch: det.get_status()
            for det, ch in zip(d.detectors, d.detector_channels, strict=True)
            if ch is not None
        }
        shared = d.detectors[0].get_status() if (
            d.trigger_mode in ("mix", "ch0") and d.detectors
        ) else None
        return {"per_channel": per_channel, "shared": shared}

    def _channel_levels(self, d, i, ch, status, now) -> dict:
        level = self._levels.get(ch) or _ChannelLevel()
        mic_labels = list(self._mic_names())
        detector = status["per_channel"].get(ch, status["shared"])
        flux_db = gate_db = None
        gate_source = ""
        will_trigger = False
        if detector is not None:
            _peak, flux, thresh = detector
            flux, thresh = float(flux), float(thresh)
            gate_lin = max(d.onset.min_flux, thresh)
            gate_source = "floor" if d.onset.min_flux >= thresh else "adaptive"
            flux_db = round(_linear_to_db(flux), 1)
            gate_db = round(_linear_to_db(gate_lin), 1)
            will_trigger = flux >= gate_lin
        return {
            "channel": ch,
            "mic_label": mic_labels[i] if i < len(mic_labels) else "",
            "peak_dbfs": round(level.peak_dbfs, 1),
            "rms_dbfs": round(level.rms_dbfs, 1),
            "peak_history": list(level.history),
            "flux_db": flux_db,
            "gate_db": gate_db,
            "gate_source": gate_source,
            "will_trigger": bool(will_trigger),
            "flash": bool(level.flash_until > now),
            "clips": int(level.clips),
            "detector": detector is not None,
            "silent": bool(ch in self._dead_channels),
            "clipping": bool(level.peak_dbfs >= CLIPPING_PEAK_DBFS),
        }

    def snapshot(self) -> dict:
        """Everything a front-end needs, in one read.

        The meters come from the same builder :meth:`levels` uses, so the fast
        poll and the full poll can never disagree about what a channel is doing.
        """
        d = self._derived
        now = time.monotonic()
        status = self._detector_status(d)
        channels = [
            self._channel_levels(d, i, ch, status, now)
            for i, ch in enumerate(d.record_indices)
        ]

        capture = self._capture
        session = self._session
        with self._lock:
            recent = list(self._recent)
            log_lines = list(self._log)[:40]
        return {
            "state": self._state,
            "mode": self._mode,
            "error": self._error,
            "stop_reason": self._stop_reason,
            "elapsed_s": round(self.elapsed_s(), 1),
            "clips_saved": self.clips_saved,
            "target": self.max_clips,
            "duration_s": self.duration_s,
            "label": self._meta.label,
            "density": self._meta.density,
            "session_id": getattr(session, "session_id", ""),
            "out_dir": str(getattr(session, "out_dir", "")),
            "sample_rate": d.input_sr,
            "device": self.cfg.deploy.record_device,
            "record_seconds": d.onset.record_seconds,
            "clip_frames": round(d.onset.record_seconds * d.input_sr),
            "onset": onset_snapshot(d.onset),
            "channels": channels,
            "recent": recent,
            "log": log_lines,
            "calibration": self._calibrator.snapshot(),
            "stats": dict(capture.stats) if capture is not None else {},
            "capture_state": getattr(capture, "state", "idle"),
            "capture_error": getattr(capture, "last_error", ""),
            "write_error": getattr(session, "last_write_error", ""),
            "dropped_clips": self._dropped_clips,
            "dropouts": self._dropouts,
        }


# A warm-up must never overlap a live stream: see wait_for_warm_clip_writer.
_warm_started = threading.Event()
_warm_done = threading.Event()
_warm_lock = threading.Lock()


def wait_for_warm_clip_writer(timeout: float = 30.0) -> bool:
    """Block until any in-flight :func:`warm_clip_writer` has finished.

    The warm-up imports torch (by way of ``sed.data.augment``), and scipy's
    resampler — which the audio callback calls on every block — probes
    ``sys.modules["torch"]`` each time. A warm-up running while audio flows
    therefore hands the callback a partially initialised module and the block is
    dropped. Opening the stream waits instead; by the time an operator has
    chosen a device the wait is already zero.

    Returns True when nothing is pending or the warm-up completed in time.
    """
    if not _warm_started.is_set():
        return True
    return _warm_done.wait(timeout)


def warm_clip_writer() -> None:
    """Pay the scientific stack's import cost before the first clip needs it.

    Writing a clip calls ``transient_mel_db``, whose module pulls in librosa and
    scipy — about six seconds on a cold process, spent on the writer thread
    while the operator watches nothing appear. The audio is sliced from the ring
    at the right instants either way, but the first take's ``recorded_at`` and
    its arrival in the UI were both late by that much. Rendering a preview pulls
    in matplotlib for the same reason.

    Idempotent and safe to call more than once. The recording server calls it
    once at boot, while the operator is still choosing a device and no stream
    exists; ``sed record`` does not call it at all, because the earliest it
    could is the moment it opens the stream, and a warm-up next to a live
    stream costs more than it saves (see wait_for_warm_clip_writer).
    """
    import numpy as _np

    _warm_started.set()
    with _warm_lock:
        if _warm_done.is_set():
            return
        try:
            from sed.data.augment import transient_mel_db

            transient_mel_db(_np.zeros(16_000, dtype=_np.float32), 16_000)
        except Exception as exc:  # noqa: BLE001 — a cold first clip is the only cost
            log.debug("clip-writer warm-up skipped: %s", exc)
        try:
            import matplotlib  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            log.debug("preview warm-up skipped: %s", exc)
        _warm_done.set()


def onset_snapshot(onset: OnsetCfg) -> dict:
    """The live-tunable onset values, as the UI's form reads them."""
    return {
        key: (list(getattr(onset, key)) if key == "preroll_ms_range"
              else getattr(onset, key))
        for key in LIVE_ONSET_KEYS
    }


def _peak_dbfs(wave: np.ndarray) -> float:
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    return 20.0 * math.log10(peak + 1e-10) if peak > 0 else -120.0
