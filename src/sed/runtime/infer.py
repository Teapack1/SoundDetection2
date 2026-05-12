"""Real-time inference loop.

Architecture:

    AudioCapture (PortAudio thread)
        └─► RingBuffer
                └─► Inference thread (every hop_ms):
                      read 1 s window → log-mel → inference backend
                                                   (OpenVINO IR → ONNX →
                                                    PyTorch checkpoint)
                                                   → objective-aware probs
                                                   → smoother + arbiter
                                                   → TriggerEvent → callback

The inference thread is a normal Python thread; OpenVINO/ONNX Runtime release
the GIL during inference, so audio capture is not blocked. Engine selection
lives in sed.runtime.backends — this loop is engine-agnostic.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from sed.audio.capture import AudioCapture
from sed.audio.features import (
    build_feature_extractor,
    window_sample_count,
)
from sed.audio.resample import model_rate_index, resample_audio
from sed.config import Config, ProjectLayout, resolve_active_model_dir
from sed.runtime.backends import (  # noqa: F401 — re-exported for compat
    _validate_openvino_input_shape,
    build_inference_backend,
    compile_openvino_model,
    load_openvino_model,
    normalize_openvino_logits,
    openvino_input_shape,
)
from sed.runtime.manifest import resolve_model_path, resolve_runtime_artifact  # re-exported
from sed.runtime.scoring import build_arbiter as _build_arbiter
from sed.runtime.scoring import pool_mic_probabilities as _pool_mic_probabilities
from sed.runtime.smooth import TriggerEvent
from sed.train.objective import logits_to_probabilities

log = logging.getLogger(__name__)


@dataclass
class RuntimeDetection:
    class_name: str
    confidence: float
    timestamp: float
    kind: str
    probs: dict[str, float]
    smoothed: dict[str, float]
    audio: np.ndarray
    sample_rate: int
    active_channels: list[int]
    best_channel: int | None
    peak_dbfs: float
    rms_dbfs: float
    min_dbfs: float | None
    active_targets: list[str]


__all__ = [
    "RealtimeInference",
    "RuntimeDetection",
    "build_inference_backend",
    "compile_openvino_model",
    "load_openvino_model",
    "_linear_to_db",
    "normalize_openvino_logits",
    "openvino_input_shape",
    "resolve_model_path",
    "resolve_runtime_artifact",
]


class RealtimeInference:
    def __init__(
        self,
        cfg: Config,
        project_id: str,
        on_trigger: Callable[[TriggerEvent], None] | None = None,
        on_frame: Callable[[dict[str, float]], None] | None = None,
        on_detection: Callable[[RuntimeDetection], Path | None] | None = None,
        active_trigger_classes: set[str] | None = None,
        detection_classes: set[str] | None = None,
        live_monitor: bool = False,
        model_precision: str | None = None,
    ):
        self.cfg = cfg
        self.layout = ProjectLayout(project_id)
        # Live monitor wants probabilities refreshed every hop; the edge service
        # (no monitor) keeps onset-only scoring so silence costs no inference.
        self.live_monitor = bool(live_monitor)
        self.classes: list[str] = list(cfg.model.classes)
        self.neg_idx: int | None = (
            self.classes.index("_negative") if "_negative" in self.classes else None
        )
        known = set(self.classes)
        self.active_trigger_classes = (
            {c for c in active_trigger_classes if c in known}
            if active_trigger_classes is not None else None
        )
        self.detection_classes = (
            {c for c in detection_classes if c in known}
            if detection_classes is not None else set()
        )

        # Inference backend, shaped to one window per inference mic.
        # `model_precision` selects the artifact/engine: auto|int8|fp16 (OpenVINO)
        # |onnx|checkpoint — see sed.runtime.backends.build_inference_backend.
        model_dir = resolve_active_model_dir(self.layout, cfg.deploy.active_model)
        batch_size = max(1, len(cfg.inference.inference_channels))
        self.backend = build_inference_backend(
            cfg, model_dir, prefer=model_precision, batch_size=batch_size
        )
        log.info("loaded model: %s (%s engine)", self.backend.path, self.backend.engine)

        self.record_sample_rate = cfg.audio.record_sample_rate
        self.feature_sample_rate = int(cfg.model.sample_rate)
        self.window_seconds = float(cfg.features.window_seconds)

        # Feature extractor (shared numpy/torch code), matched to the exported
        # model: mel+efficientat uses MelExtractor, linear/PCEN use FeatureExtractor.
        self.mel = build_feature_extractor(cfg)

        # Onset-gated mode scores ONE onset-centered window per transient, so the
        # temporal-consensus machinery (median/EMA/consecutive frames) does not
        # apply — each window is gated on the spot, exactly like the frozen test.
        self.onset_gating = bool(cfg.inference.onset_gating_enabled)

        # Smoother + arbiter — ONE arbiter sees the pooled per-class probability
        # across mics. So "any mic detects → trigger" is enforced naturally.
        # Per-class thresholds come from [targets].<cls>.threshold, falling back to
        # legacy [thresholds] then inference.default_threshold (Config helper).
        # Built through the shared factory so offline replay and FP/h
        # calibration decide exactly as production does — including the
        # onset-gating override that switches temporal consensus off.
        self.arbiter = _build_arbiter(
            cfg, self.classes, self.neg_idx,
            trigger_classes=self.active_trigger_classes,
        )

        # Audio
        self.window_input_n = window_sample_count(self.record_sample_rate, self.window_seconds)
        self.window_feature_n = window_sample_count(self.feature_sample_rate, self.window_seconds)
        self.hop_input_n = int(round(cfg.inference.hop_ms / 1000.0 * self.record_sample_rate))
        if self.window_input_n <= 0 or self.window_feature_n <= 0 or self.hop_input_n <= 0:
            raise ValueError("runtime window_seconds and inference.hop_ms must produce samples")

        # Determine which physical mics feed inference. inference_channels is an
        # ordered list of physical channel indexes (independent of the recording
        # selection); we open enough physical channels to cover the highest one.
        self.mic_mode = cfg.inference.multi_mic_mode  # "max" | "best" | "single" | "mean"
        self.active_channels = list(cfg.inference.inference_channels)
        if self.mic_mode == "single" and len(self.active_channels) != 1:
            raise ValueError(
                "inference.multi_mic_mode = 'single' requires exactly one inference channel"
            )
        n_physical = max(self.active_channels) + 1
        log.info(
            "microphones: pooling=%s | inference_channels=%s | opened_channels=%d",
            self.mic_mode, self.active_channels, n_physical,
        )

        # Rolling inference-time accounting (was a _loop local; now shared by the
        # sliding and onset-gated paths).
        self._infer_ms_total = 0.0
        self._infer_count = 0

        # Onset-gated inference: one StreamingSuperflux per active mic, fed from
        # the audio callback. Detector params come from [onset] — the same
        # settings that recorded/trained the data, so live onsets match.
        self._pending_onsets: collections.deque[int] = collections.deque()
        self._onset_lock = threading.Lock()
        on_block_cb: Callable[[np.ndarray, int], None] | None = None
        if self.onset_gating:
            from sed.audio.onset import OnsetMerger, StreamingSuperflux

            o = cfg.onset
            # Where the onset sits inside the scored window. Not the recording
            # preroll: `record_seconds` is generous on purpose, so at a window
            # shorter than the recording the preroll would eat the event's decay
            # (see Config.window_lead_in_s). Training crops to the same geometry.
            self._onset_preroll_n = int(
                round(cfg.window_lead_in_s() * self.record_sample_rate)
            )
            # Collapse the same physical transient (seen on several mics or in
            # successive detector frames) into a single scoring window.
            self._onset_cooldown_n = int(
                round(max(o.cross_channel_merge_ms, 1) / 1000.0 * self.record_sample_rate)
            )
            self._onset_merger = OnsetMerger(self._onset_cooldown_n)
            self._onset_detectors = [
                StreamingSuperflux(
                    sample_rate=self.feature_sample_rate,
                    hop_ms=o.hop_ms,
                    n_mels=o.trigger_n_mels,
                    median_window_s=o.median_window_s,
                    mad_k=o.mad_k,
                    refractory_ms=min(150, int(o.recording_cooldown_ms)),
                    min_flux=o.min_flux,
                )
                for _ in self.active_channels
            ]
            on_block_cb = self._on_audio_block
            log.info(
                "onset trigger enabled | preroll=%d ms | detectors=%d | mad_k=%.1f | onset_floor=%.1f dB",
                o.preroll_ms, len(self._onset_detectors), o.mad_k, o.min_flux_db,
            )

        self.capture = AudioCapture(
            sample_rate=self.record_sample_rate,
            channels=n_physical,
            device=cfg.deploy.record_device,
            blocksize=cfg.audio.blocksize,
            ring_seconds=4.0,
            on_block=on_block_cb,
        )

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Per-class cooldown for "threshold"-kind mining (monotonic seconds).
        self._mining_last_ts: dict[str, float] = {}
        self._on_trigger = on_trigger
        self._on_frame = on_frame
        self._on_detection = on_detection

        mic_status = {
            ch: self._initial_mic_status(ch, i)
            for i, ch in enumerate(self.active_channels)
        }
        # Stats — keep the legacy flat counters, and publish richer per-mic
        # state under mic_status for the operator display.
        self.stats = {
            "frames": 0,
            "windows": 0,
            "onset_windows": 0,  # onset-gated frames scored (== onsets passed)
            "triggers": 0,
            "amplitude_skipped": 0,
            "audio_rms_dbfs": -120.0,
            "audio_peak_dbfs": -120.0,
            "amplitude_gate": False,
            "min_dbfs": cfg.inference.min_dbfs,
            "last_probs": {},
            "last_smoothed": {},
            "infer_ms_avg": 0.0,
            "per_mic_max_prob": {ch: 0.0 for ch in self.active_channels},
            "per_mic_triggers": {ch: 0 for ch in self.active_channels},
            "mic_status": mic_status,
            "class_thresholds": {
                cls: smoother.threshold
                for cls, smoother in self.arbiter.smoothers.items()
            },
            "negative_max": cfg.inference.negative_max,
            "mined_saved": 0,
            "last_trigger": None,    # {"class", "conf", "ts", "mined"} of the most recent fire
            "last_detection": None,  # operator-facing alias with mic/level detail
            "last_mined": None,      # {"class", "ts", "path"} of the most recent saved clip
        }

    def _initial_mic_status(self, channel: int, order_index: int) -> dict:
        return {
            "channel": channel,
            "label": self._mic_label(channel, order_index),
            "peak_dbfs": -120.0,
            "rms_dbfs": -120.0,
            "onset_db": None,
            "onset_gate_db": None,
            "top_class": None,
            "top_score": 0.0,
            "detections": 0,
            "detections_by_class": {},
            "last_detection": None,
        }

    def _mic_label(self, channel: int, order_index: int) -> str:
        labels = list(getattr(self.cfg.audio, "microphone_type", []) or [])
        if len(labels) == len(self.active_channels):
            return labels[order_index]
        if 0 <= channel < len(labels):
            return labels[channel]
        if len(labels) == 1:
            return labels[0]
        return ""

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self.capture.start()
        # wait for the ring to fill
        log.info(
            "waiting for first %d input samples in ring (%d Hz -> %d Hz features) ...",
            self.window_input_n,
            self.record_sample_rate,
            self.feature_sample_rate,
        )
        while self.capture.ring.write_idx < self.window_input_n and not self._stop.is_set():
            time.sleep(0.05)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("inference loop started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.capture.stop()

    # ---- loop -------------------------------------------------------------

    def _loop(self) -> None:
        """Unified inference loop.

        Two cadences share one thread (and one OpenVINO infer request):

          * **Display frames** — every ``hop_ms`` we score the latest sliding
            window and refresh the live monitor's mic levels + class probabilities.
            This keeps the bars responsive (~hop rate) instead of frozen between
            transients. When onset gating is OFF this is also the decision path
            (trigger each hop, the legacy sliding behaviour).

          * **Onset frames** — when onset gating is ON, each detected transient is
            scored on its own onset-centered window (matching the recorded/frozen
            distribution) and only those frames run the arbiter / trigger / mining.

        So with gating on: bars update every hop, but a trigger fires only on a
        real onset. With gating ON and no live monitor (e.g. the edge service),
        display frames are skipped so silence costs no inference.
        """
        # Hop-paced display frames only matter for the live monitor. When onset
        # gating is OFF they are the decision path, so they always run.
        do_display = (not self.onset_gating) or self.live_monitor
        next_idx = self.capture.ring.write_idx       # absolute end-sample of next display frame
        while not self._stop.is_set():
            did_work = False
            now_idx = self.capture.ring.write_idx

            if self.onset_gating:
                for onset_input_idx in self._pop_due_onsets(now_idx):
                    window_start = onset_input_idx - self._onset_preroll_n
                    try:
                        channel_waves = self._read_channel_window(window_start)
                    except ValueError as e:
                        # Window not yet written, or already overwritten under load.
                        log.debug("onset window read failed: %s", e)
                        continue
                    self.stats["onset_windows"] += 1
                    self._process_window(channel_waves, trigger=True)
                    did_work = True

            if do_display and now_idx - next_idx >= self.hop_input_n:
                # Catch up if we're behind (>1 hop).
                while now_idx - next_idx >= self.hop_input_n:
                    next_idx += self.hop_input_n
                window_start = next_idx - self.window_input_n
                try:
                    channel_waves = self._read_channel_window(window_start)
                except ValueError as e:
                    log.warning("ring read failed: %s", e)
                else:
                    self._process_window(channel_waves, trigger=not self.onset_gating)
                did_work = True

            if not did_work:
                time.sleep(0.005)

    def _pop_due_onsets(self, now_idx: int) -> list[int]:
        """Pop queued onsets whose full window (preroll + postroll) is captured."""
        due: list[int] = []
        with self._onset_lock:
            while self._pending_onsets and (
                self._pending_onsets[0] - self._onset_preroll_n + self.window_input_n
                <= now_idx
            ):
                due.append(self._pending_onsets.popleft())
        return due

    def _read_channel_window(self, window_start: int) -> list[np.ndarray]:
        """Read `window_input_n` samples from each active mic, starting at the
        absolute ring index `window_start`."""
        return [
            self.capture.ring.read_window(window_start, self.window_input_n, channel=ch)
            for ch in self.active_channels
        ]

    def _process_window(self, channel_waves: list[np.ndarray], *, trigger: bool) -> None:
        """Audio stats + amplitude gate + scoring for one window.

        ``trigger`` controls whether the scored window runs the full decision
        path (arbiter → trigger → mining). Display-only frames (``trigger=False``)
        refresh the live monitor's levels/probabilities without advancing the
        arbiter, so they never fire on non-onset windows.
        """
        audio_window = np.stack(channel_waves, axis=1)
        audio_mono = audio_window.mean(axis=1)
        rms = float(np.sqrt(np.mean(np.square(audio_mono))) + 1e-12)
        peak = float(np.max(np.abs(audio_window)) + 1e-12)
        rms_dbfs = 20.0 * np.log10(rms)
        peak_dbfs = 20.0 * np.log10(peak)
        self.stats["windows"] += 1
        self.stats["audio_rms_dbfs"] = rms_dbfs
        self.stats["audio_peak_dbfs"] = peak_dbfs
        self._update_mic_audio_status(channel_waves)
        min_dbfs = self.cfg.inference.min_dbfs
        amplitude_gate = _below_min_dbfs(peak_dbfs, min_dbfs)
        self.stats["amplitude_gate"] = amplitude_gate
        if amplitude_gate:
            self.stats["amplitude_skipped"] += 1
            zero_probs = {c: 0.0 for c in self.classes}
            self.stats["last_probs"] = zero_probs
            self.stats["last_smoothed"] = zero_probs
            self._clear_mic_score_status()
            if self._on_frame is not None:
                self._on_frame(zero_probs)
            return

        prob_dict, probs_per_mic = self._score_window(channel_waves)
        if trigger:
            self._decide_and_fire(
                prob_dict, probs_per_mic, audio_window, peak_dbfs, rms_dbfs, min_dbfs
            )
        else:
            # Display-only refresh: show the latest probabilities (no smoothing in
            # onset-gated mode) but leave the arbiter untouched so quiet sliding
            # windows can never trigger.
            self.stats["last_probs"] = prob_dict
            self.stats["last_smoothed"] = dict(prob_dict)
            if self._on_frame is not None:
                self._on_frame(prob_dict)

    def _score_window(
        self, channel_waves: list[np.ndarray]
    ) -> tuple[dict[str, float], np.ndarray]:
        """Mel features → inference backend → per-mic pooling for one window.

        Returns the pooled per-class probability dict and the raw per-mic matrix.
        """
        t0 = time.perf_counter()
        # Build the mel batch
        specs = []
        for w in channel_waves:
            w = resample_audio(
                w,
                orig_sr=self.record_sample_rate,
                target_sr=self.feature_sample_rate,
            )
            if len(w) != self.window_feature_n:
                if len(w) > self.window_feature_n:
                    w = w[:self.window_feature_n]
                else:
                    w = np.pad(w, (0, self.window_feature_n - len(w)))
            spec = self.mel(w)
            if hasattr(spec, "numpy"):
                spec = spec.numpy()
            # Drop the leading batch axis the extractor adds: a spectrogram is
            # (1, n_mels, T) → (n_mels, T); a colormap image is (1, C, H, W) →
            # (C, H, W) (already channel-first).
            spec = spec[0]
            specs.append(spec)
        stacked = np.stack(specs, axis=0).astype(np.float32)
        # (B, n_mels, T) spectrogram → add the single channel; (B, C, H, W) image
        # already carries its channels.
        batch = stacked[:, None, :, :] if stacked.ndim == 3 else stacked

        # Single batched inference call — much cheaper than N sequential calls.
        # The backend returns normalized (B, C) logits regardless of engine.
        logits = self.backend.infer(batch)
        probs_per_mic = logits_to_probabilities(logits, self.cfg)       # (B, num_classes)
        self._accumulate_infer_ms((time.perf_counter() - t0) * 1000)

        # Which classes are targets (everything except the optional _negative).
        target_mask = np.array(
            [self.neg_idx is None or i != self.neg_idx for i in range(len(self.classes))]
        )
        # Pool per-class across mics into one decision vector.
        pooled = pool_mic_probabilities(probs_per_mic, self.mic_mode, self.neg_idx)
        # Per-mic stats: which mic had the highest TARGET-class probability?
        for i, ch in enumerate(self.active_channels):
            m = float(probs_per_mic[i, target_mask].max()) if target_mask.any() else 0.0
            # exponential moving max for the live display
            self.stats["per_mic_max_prob"][ch] = (
                0.85 * self.stats["per_mic_max_prob"].get(ch, 0.0) + 0.15 * m
            )
            self._update_mic_score_status(ch, probs_per_mic[i], target_mask)
        prob_dict = {c: float(p) for c, p in zip(self.classes, pooled)}
        return prob_dict, probs_per_mic

    def _update_mic_audio_status(self, channel_waves: list[np.ndarray]) -> None:
        mic_status = self.stats.get("mic_status", {})
        for ch, wave in zip(self.active_channels, channel_waves):
            if wave.size:
                rms = float(np.sqrt(np.mean(np.square(wave))) + 1e-12)
                peak = float(np.max(np.abs(wave)) + 1e-12)
                rms_dbfs = 20.0 * np.log10(rms)
                peak_dbfs = 20.0 * np.log10(peak)
            else:
                rms_dbfs = peak_dbfs = -120.0
            status = mic_status.setdefault(ch, self._initial_mic_status(ch, 0))
            status["rms_dbfs"] = float(rms_dbfs)
            status["peak_dbfs"] = float(peak_dbfs)

    def _update_mic_score_status(
        self,
        channel: int,
        probs: np.ndarray,
        target_mask: np.ndarray,
    ) -> None:
        mic_status = self.stats.get("mic_status", {})
        status = mic_status.setdefault(channel, self._initial_mic_status(channel, 0))
        target_indices = np.flatnonzero(target_mask)
        if target_indices.size:
            best_idx = int(target_indices[int(np.argmax(probs[target_indices]))])
        else:
            best_idx = int(np.argmax(probs))
        status["top_class"] = self.classes[best_idx]
        status["top_score"] = float(probs[best_idx])

    def _clear_mic_score_status(self) -> None:
        for status in self.stats.get("mic_status", {}).values():
            status["top_class"] = None
            status["top_score"] = 0.0

    def _accumulate_infer_ms(self, ms: float) -> None:
        self._infer_ms_total += ms
        self._infer_count += 1
        if self._infer_count >= 20:
            self.stats["infer_ms_avg"] = self._infer_ms_total / self._infer_count
            self._infer_ms_total = 0.0
            self._infer_count = 0

    def _decide_and_fire(
        self,
        prob_dict: dict[str, float],
        probs_per_mic: np.ndarray,
        audio_window: np.ndarray,
        peak_dbfs: float,
        rms_dbfs: float,
        min_dbfs: float | None,
    ) -> None:
        ev = self.arbiter.step(prob_dict)
        self.stats["frames"] += 1
        smoothed = dict(self.arbiter.last_smoothed)
        self.stats["last_probs"] = prob_dict
        self.stats["last_smoothed"] = smoothed
        if self._on_frame is not None:
            self._on_frame(prob_dict)
        emitted_detection_classes: set[str] = set()
        if ev is not None:
            self.stats["triggers"] += 1
            # Which mic detected most strongly?
            ci = self.classes.index(ev.class_name)
            best_mic_i = int(np.argmax(probs_per_mic[:, ci]))
            best_mic = self.active_channels[best_mic_i]
            self.stats["per_mic_triggers"][best_mic] = (
                self.stats["per_mic_triggers"].get(best_mic, 0) + 1
            )
            ev.peak_dbfs = peak_dbfs
            ev.rms_dbfs = rms_dbfs
            ev.min_dbfs = min_dbfs
            ev.best_channel = best_mic
            if self._on_detection is not None:
                mined_path = self._on_detection(
                    self._runtime_detection(
                        ev.class_name,
                        ev.confidence,
                        ev.timestamp,
                        "trigger",
                        prob_dict,
                        smoothed,
                        audio_window,
                        best_mic,
                        peak_dbfs,
                        rms_dbfs,
                        min_dbfs,
                    )
                )
                ev.mined_path = str(mined_path) if mined_path is not None else None
                emitted_detection_classes.add(ev.class_name)
                # Start the cooldown so threshold-mining doesn't immediately
                # pile clips on top of a just-saved trigger clip.
                self._mining_last_ts[ev.class_name] = time.monotonic()
                if mined_path is not None:
                    self.stats["mined_saved"] += 1
                    self.stats["last_mined"] = {
                        "class": ev.class_name, "ts": ev.timestamp,
                        "path": str(mined_path),
                    }
            self.stats["last_trigger"] = {
                "class": ev.class_name, "conf": ev.confidence,
                "ts": ev.timestamp, "mined": ev.mined_path,
            }
            self._record_mic_detection(best_mic, ev.class_name, ev.confidence, ev.timestamp, peak_dbfs)
            self.stats["last_detection"] = {
                "class": ev.class_name,
                "conf": ev.confidence,
                "ts": ev.timestamp,
                "mic": best_mic,
                "peak_dbfs": peak_dbfs,
                "mined": ev.mined_path,
            }
            if self._on_trigger is not None:
                self._on_trigger(ev)

        if self._on_detection is not None and self.detection_classes:
            cooldown = self.cfg.mining.min_interval_ms / 1000.0
            now_ts = time.monotonic()
            for cls in sorted(self.detection_classes):
                if cls in emitted_detection_classes:
                    continue
                score = float(smoothed.get(cls, 0.0))
                threshold = self._mining_threshold(cls)
                if score < threshold:
                    continue
                # Anti-flood: at most one mined clip per class per cooldown.
                if now_ts - self._mining_last_ts.get(cls, -1e9) < cooldown:
                    continue
                self._mining_last_ts[cls] = now_ts
                ci = self.classes.index(cls)
                best_mic_i = int(np.argmax(probs_per_mic[:, ci]))
                mined_path = self._on_detection(
                    self._runtime_detection(
                        cls,
                        score,
                        now_ts,
                        "threshold",
                        prob_dict,
                        smoothed,
                        audio_window,
                        self.active_channels[best_mic_i],
                        peak_dbfs,
                        rms_dbfs,
                        min_dbfs,
                    )
                )
                if mined_path is not None:
                    self.stats["mined_saved"] += 1
                    self.stats["last_mined"] = {
                        "class": cls, "ts": now_ts, "path": str(mined_path),
                    }

    def _record_mic_detection(
        self,
        channel: int,
        class_name: str,
        confidence: float,
        timestamp: float,
        peak_dbfs: float,
    ) -> None:
        mic_status = self.stats.get("mic_status", {})
        status = mic_status.setdefault(channel, self._initial_mic_status(channel, 0))
        status["detections"] = int(status.get("detections", 0)) + 1
        by_class = status.setdefault("detections_by_class", {})
        by_class[class_name] = int(by_class.get(class_name, 0)) + 1
        status["last_detection"] = {
            "class": class_name,
            "conf": confidence,
            "ts": timestamp,
            "peak_dbfs": peak_dbfs,
        }

    def _on_audio_block(self, block: np.ndarray, write_idx: int) -> None:
        """Audio-thread callback (onset gating): detect onsets and queue them.

        Deliberately light — resample + spectral flux only; all model work runs
        in the scoring thread. Onsets from any mic for the same physical
        transient are collapsed via `_onset_cooldown_n`.
        """
        try:
            end_idx_model = model_rate_index(
                write_idx, self.record_sample_rate, self.feature_sample_rate
            )
            new_onsets: list[int] = []
            for det, ch in zip(self._onset_detectors, self.active_channels):
                sig = block[:, ch] if block.ndim > 1 else block
                sig_rs = resample_audio(
                    sig, orig_sr=self.record_sample_rate, target_sr=self.feature_sample_rate
                )
                events = det.process_block(sig_rs, end_idx_model)
                self._update_mic_onset_status(det, ch, events)
                for ev in events:
                    new_onsets.append(
                        model_rate_index(
                            ev.sample_idx, self.feature_sample_rate, self.record_sample_rate
                        )
                    )
            if not new_onsets:
                return
            with self._onset_lock:
                self._pending_onsets.extend(self._onset_merger.push(new_onsets))
        except Exception:  # noqa: BLE001 — never let the audio callback die
            log.exception("onset block callback failed")

    def _update_mic_onset_status(self, detector, channel: int, events: list) -> None:
        mic_status = self.stats.get("mic_status", {})
        status = mic_status.setdefault(channel, self._initial_mic_status(channel, 0))
        _dbfs, flux, threshold = detector.get_status()
        gate = max(float(self.cfg.onset.min_flux), float(threshold))
        status["onset_db"] = _linear_to_db(flux)
        status["onset_gate_db"] = _linear_to_db(gate)
        if events:
            status["last_onset_db"] = max(_linear_to_db(ev.strength) for ev in events)

    def _mining_threshold(self, class_name: str) -> float:
        # Legacy [mining].per_class_min_confidence still wins if set; otherwise
        # mining shares the class's decision threshold from [targets]/[thresholds].
        per_class = self.cfg.mining.per_class_min_confidence
        if class_name in per_class:
            return float(per_class[class_name])
        return float(self.cfg.class_threshold(class_name))

    def _runtime_detection(
        self,
        class_name: str,
        confidence: float,
        timestamp: float,
        kind: str,
        probs: dict[str, float],
        smoothed: dict[str, float],
        audio: np.ndarray,
        best_channel: int | None,
        peak_dbfs: float,
        rms_dbfs: float,
        min_dbfs: float | None,
    ) -> RuntimeDetection:
        return RuntimeDetection(
            class_name=class_name,
            confidence=confidence,
            timestamp=timestamp,
            kind=kind,
            probs=probs,
            smoothed=smoothed,
            audio=audio,
            sample_rate=self.record_sample_rate,
            active_channels=list(self.active_channels),
            best_channel=best_channel,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            min_dbfs=min_dbfs,
            active_targets=sorted(self.detection_classes or self.active_trigger_classes or []),
        )


def _below_min_dbfs(peak_dbfs: float, min_dbfs: float | None) -> bool:
    """Return whether a window should be skipped by the peak-amplitude gate."""
    return min_dbfs is not None and peak_dbfs < float(min_dbfs)


def _linear_to_db(value: float | None) -> float:
    """Convert a positive linear onset value to dB for operator displays."""
    if value is None or float(value) <= 0.0:
        return -120.0
    return 20.0 * float(np.log10(float(value) + 1e-12))


# Multi-mic pooling lives in sed.runtime.scoring with the rest of the decision
# core; re-exported here because this module is the runtime's public surface
# (tests and tooling import it from sed.runtime.infer).
pool_mic_probabilities = _pool_mic_probabilities
