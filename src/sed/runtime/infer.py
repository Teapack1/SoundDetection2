"""Real-time inference loop.

Architecture:

    AudioCapture (PortAudio thread)
        └─► RingBuffer
                └─► Inference thread (every hop_ms):
                      read 1 s window → log-mel → OpenVINO compiled model
                                                   → sigmoid probs
                                                   → smoother + arbiter
                                                   → TriggerEvent → callback

The inference thread is a normal Python thread; OpenVINO releases the GIL
during `infer_request.infer()`, so audio capture is not blocked.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import openvino as ov

from sed.audio.capture import AudioCapture
from sed.audio.features import MelExtractor
from sed.audio.resample import resample_audio
from sed.config import Config, ProjectLayout
from sed.runtime.manifest import load_manifest, resolve_model_path  # re-exported
from sed.runtime.smooth import TriggerArbiter, TriggerEvent

log = logging.getLogger(__name__)


__all__ = ["RealtimeInference", "load_manifest", "resolve_model_path"]


class RealtimeInference:
    def __init__(
        self,
        cfg: Config,
        project_id: str,
        on_trigger: Callable[[TriggerEvent], None] | None = None,
        on_frame: Callable[[dict[str, float]], None] | None = None,
    ):
        self.cfg = cfg
        self.layout = ProjectLayout(project_id)
        self.manifest = load_manifest(self.layout)
        self.classes: list[str] = self.manifest["classes"]
        self.neg_idx: int = self.manifest["negative_index"]

        # OpenVINO compiled model (dynamic batch)
        model_path = resolve_model_path(self.layout, self.manifest)
        log.info("loading model: %s", model_path)
        self.core = ov.Core()
        self.model = self.core.read_model(str(model_path))
        # Allow batch>1 for multi-mic. The model was exported with dynamic batch.
        self.compiled = self.core.compile_model(
            self.model,
            "CPU",
            config={
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_NUM_THREADS": str(cfg.inference.inference_threads),
            },
        )
        self.infer_req = self.compiled.create_infer_request()

        self.input_sample_rate = cfg.audio.input_sample_rate
        self.feature_sample_rate = int(self.manifest["sample_rate"])
        self.window_seconds = float(self.manifest["window_seconds"])

        # Mel extractor (pure-numpy/torch shared code), matched to the exported model.
        self.mel = MelExtractor(
            sample_rate=self.feature_sample_rate,
            n_fft=self.manifest["n_fft"],
            win_length=self.manifest["win_length"],
            hop_length=self.manifest["hop_length"],
            n_mels=self.manifest["n_mels"],
            fmin=self.manifest["fmin"],
            fmax=self.manifest["fmax"],
        )

        # Smoother + arbiter — ONE arbiter sees the per-class max (or mean)
        # across mics. So "any mic detects → trigger" is enforced naturally.
        thresholds = dict(cfg.thresholds)
        smoothers_cfg = {
            "median_window": cfg.inference.smoothing_median_window,
            "ema_alpha": cfg.inference.smoothing_ema_alpha,
            "default_threshold": cfg.inference.default_threshold,
            "hysteresis_low": cfg.inference.hysteresis_low,
            "consecutive_frames": cfg.inference.consecutive_frames,
        }
        self.arbiter = TriggerArbiter(
            classes=self.classes,
            thresholds=thresholds,
            smoothers_cfg=smoothers_cfg,
            negative_index=self.neg_idx,
            negative_max=cfg.inference.negative_max,
            margin_to_second_best=cfg.inference.margin_to_second_best,
            global_refractory_ms=cfg.inference.trigger_refractory_ms,
        )

        # Audio
        self.window_input_n = int(round(self.window_seconds * self.input_sample_rate))
        self.window_feature_n = int(round(self.window_seconds * self.feature_sample_rate))
        self.hop_input_n = int(round(cfg.inference.hop_ms / 1000.0 * self.input_sample_rate))

        # Determine how many mics actually feed inference
        self.mic_mode = cfg.inference.multi_mic_mode  # "max" | "single" | "mean"
        if self.mic_mode == "single":
            self.active_channels = [cfg.audio.inference_channel]
        else:
            self.active_channels = list(range(max(1, cfg.audio.record_channels)))
        log.info(
            "multi-mic mode=%s · active channels=%s",
            self.mic_mode, self.active_channels,
        )

        self.capture = AudioCapture(
            sample_rate=self.input_sample_rate,
            channels=max(1, cfg.audio.record_channels),
            device=cfg.audio.record_device,
            blocksize=cfg.audio.blocksize,
            ring_seconds=4.0,
            gain_db=cfg.audio.input_gain_db,
        )

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_trigger = on_trigger
        self._on_frame = on_frame

        # Stats — also report per-mic activity so the operator can see if a
        # mic is dead.
        self.stats = {
            "frames": 0,
            "triggers": 0,
            "infer_ms_avg": 0.0,
            "per_mic_max_prob": {ch: 0.0 for ch in self.active_channels},
            "per_mic_triggers": {ch: 0 for ch in self.active_channels},
        }

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self.capture.start()
        # wait for the ring to fill
        log.info(
            "waiting for first %d input samples in ring (%d Hz -> %d Hz features) ...",
            self.window_input_n,
            self.input_sample_rate,
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
        next_idx = self.capture.ring.write_idx           # absolute end-sample of next window
        infer_ms_total = 0.0
        infer_count = 0
        while not self._stop.is_set():
            now_idx = self.capture.ring.write_idx
            if now_idx - next_idx < self.hop_input_n:
                time.sleep(0.01)
                continue

            # Catch up if we're behind (>1 hop)
            while now_idx - next_idx >= self.hop_input_n:
                next_idx += self.hop_input_n

            window_end = next_idx
            window_start = window_end - self.window_input_n

            # Read N channels of audio
            try:
                channel_waves: list[np.ndarray] = []
                for ch in self.active_channels:
                    w = self.capture.ring.read_window(
                        window_start, self.window_input_n, channel=ch
                    )
                    channel_waves.append(w)
            except ValueError as e:
                log.warning("ring read failed: %s", e)
                continue

            t0 = time.perf_counter()
            # Build the mel batch
            specs = []
            for w in channel_waves:
                w = resample_audio(
                    w,
                    orig_sr=self.input_sample_rate,
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
                # spec is (1, n_mels, T) → (n_mels, T)
                if spec.ndim == 3:
                    spec = spec[0]
                specs.append(spec)
            batch = np.stack(specs, axis=0)[:, None, :, :].astype(np.float32)  # (B, 1, n_mels, T)

            # Single batched OpenVINO call — much cheaper than N sequential calls
            logits = self.infer_req.infer({0: batch})
            logits = list(logits.values())[0]                               # (B, num_classes)
            probs_per_mic = 1.0 / (1.0 + np.exp(-logits))                   # (B, num_classes)
            t1 = time.perf_counter()
            infer_ms_total += (t1 - t0) * 1000
            infer_count += 1

            # Pool per-class across mics
            if self.mic_mode == "max" or self.mic_mode == "single":
                pooled = probs_per_mic.max(axis=0)                          # (num_classes,)
            elif self.mic_mode == "mean":
                pooled = probs_per_mic.mean(axis=0)
            else:
                pooled = probs_per_mic.max(axis=0)

            # Per-mic stats: which mic had the highest TARGET-class probability?
            target_mask = np.array([i != self.neg_idx for i in range(len(self.classes))])
            for i, ch in enumerate(self.active_channels):
                m = float(probs_per_mic[i, target_mask].max())
                # exponential moving max for the live display
                self.stats["per_mic_max_prob"][ch] = (
                    0.85 * self.stats["per_mic_max_prob"].get(ch, 0.0) + 0.15 * m
                )

            prob_dict = {c: float(p) for c, p in zip(self.classes, pooled)}
            if self._on_frame is not None:
                self._on_frame(prob_dict)

            ev = self.arbiter.step(prob_dict)
            self.stats["frames"] += 1
            if ev is not None:
                self.stats["triggers"] += 1
                # Which mic detected most strongly?
                ci = self.classes.index(ev.class_name)
                best_mic_i = int(np.argmax(probs_per_mic[:, ci]))
                best_mic = self.active_channels[best_mic_i]
                self.stats["per_mic_triggers"][best_mic] = (
                    self.stats["per_mic_triggers"].get(best_mic, 0) + 1
                )
                if self._on_trigger is not None:
                    self._on_trigger(ev)

            if infer_count >= 20:
                self.stats["infer_ms_avg"] = infer_ms_total / infer_count
                infer_ms_total = 0.0
                infer_count = 0
