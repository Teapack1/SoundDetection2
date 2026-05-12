"""The production decision core, in one place.

Turning audio into a trigger is one rule, but it existed three times: the live
runtime (`infer.RealtimeInference`), the FP/h sweep (`runtime.calibrate`) and
the clip evaluator (`train.eval._runtime_decisions`). Each rendered the rule
slightly differently, and the differences were not deliberate — the calibrator's
synthetic clock never engaged the refractory that production applies, and the
evaluator drops the temporal state entirely.

This module owns the two halves that must not diverge:

  * `WindowScorer` — one multichannel window to a pooled per-class probability
    (resample, features, backend, mic pooling). Owns no clock and no device, so
    it works identically on a live ring buffer and on a file.
  * `build_arbiter` — the single place the onset-gating override is applied.

Anything that wants to reproduce production behaviour builds these rather than
re-implementing the chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sed.audio.features import window_sample_count
from sed.audio.resample import resample_audio
from sed.config import Config
from sed.runtime.smooth import TriggerArbiter
from sed.train.objective import logits_to_probabilities

log = logging.getLogger(__name__)


def pool_mic_probabilities(
    probs_per_mic: np.ndarray,
    mic_mode: str,
    neg_idx: int | None,
) -> np.ndarray:
    """Collapse per-mic class probabilities ``(n_mics, n_classes)`` to ``(n_classes,)``.

    ``best``  — trust the single mic that best heard a target and return its
                *whole* softmax vector (including its own ``_negative``). The
                negative gate then stays self-consistent: it reflects the mic
                that actually heard the event, not the quietest off-axis mic.
    ``mean``  — average across mics.
    ``max`` / ``single`` — per-class maximum across mics. Cheap, but with an
                explicit ``_negative`` class this lets the worst mic's negative
                veto a fire another mic is confident about, hurting recall.
    """
    if probs_per_mic.ndim != 2:
        raise ValueError("probs_per_mic must be 2-D (n_mics, n_classes)")
    if mic_mode == "best":
        n_classes = probs_per_mic.shape[1]
        target_cols = [c for c in range(n_classes) if neg_idx is None or c != neg_idx]
        scores = (
            probs_per_mic[:, target_cols].max(axis=1)
            if target_cols
            else probs_per_mic.max(axis=1)
        )
        return probs_per_mic[int(np.argmax(scores))]
    if mic_mode == "mean":
        return probs_per_mic.mean(axis=0)
    return probs_per_mic.max(axis=0)  # "max" / "single"


def build_arbiter(
    cfg: Config,
    classes: list[str],
    neg_idx: int | None,
    *,
    thresholds: dict[str, float] | None = None,
    trigger_classes: list[str] | None = None,
) -> TriggerArbiter:
    """The production arbiter, including the onset-gating override.

    With onset gating on, each transient is scored on its own onset-centred
    window, so the temporal-consensus machinery (median / EMA / N-of-N) does not
    apply and is switched off. Reproducing production without this override
    measures a system that is not deployed.
    """
    onset_gating = bool(cfg.inference.onset_gating_enabled)
    return TriggerArbiter(
        classes=classes,
        thresholds=thresholds if thresholds is not None else cfg.effective_thresholds(classes),
        smoothers_cfg={
            "median_window": cfg.inference.smoothing_median_window,
            "ema_alpha": cfg.inference.smoothing_ema_alpha,
            "default_threshold": cfg.inference.default_threshold,
            "hysteresis_low": cfg.inference.hysteresis_low,
            "consecutive_frames": 1 if onset_gating else cfg.inference.consecutive_frames,
            "smoothing_enabled": False if onset_gating else cfg.inference.smoothing_enabled,
        },
        negative_index=neg_idx,
        negative_max=cfg.inference.negative_max,
        margin_to_second_best=cfg.inference.margin_to_second_best,
        global_refractory_ms=cfg.inference.trigger_refractory_ms,
        trigger_classes=trigger_classes,
    )


@dataclass(frozen=True)
class ScoredWindow:
    """One window's scores: pooled decision vector plus the per-mic detail."""

    probs: dict[str, float]
    probs_per_mic: np.ndarray      # (n_mics, n_classes)
    peak_dbfs: float
    rms_dbfs: float


def _dbfs(value: float) -> float:
    return 20.0 * float(np.log10(max(float(value), 1e-12)))


class WindowScorer:
    """Score one multichannel window exactly as the runtime does.

    Deliberately clock-free and device-free: the live runtime feeds it slices of
    a ring buffer, the replay engine feeds it slices of a file, and both get the
    same numbers.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        backend,
        classes: list[str],
        feature_extractor=None,
        active_channels: list[int] | None = None,
        record_sample_rate: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.backend = backend
        self.classes = list(classes)
        self.neg_idx = self.classes.index("_negative") if "_negative" in self.classes else None
        self.mic_mode = cfg.inference.multi_mic_mode
        self.active_channels = list(
            active_channels if active_channels is not None else cfg.inference.inference_channels
        )
        self.record_sample_rate = int(record_sample_rate or cfg.audio.record_sample_rate)
        self.feature_sample_rate = int(cfg.model.sample_rate)
        self.window_seconds = float(cfg.features.window_seconds)
        self.window_input_n = window_sample_count(self.record_sample_rate, self.window_seconds)
        self.window_feature_n = window_sample_count(self.feature_sample_rate, self.window_seconds)

        if feature_extractor is None:
            from sed.audio.features import build_feature_extractor

            feature_extractor = build_feature_extractor(cfg)
        self.mel = feature_extractor

    @classmethod
    def from_model_dir(
        cls,
        cfg: Config,
        model_dir: Path,
        *,
        classes: list[str] | None = None,
        precision: str | None = None,
        active_channels: list[int] | None = None,
        record_sample_rate: int | None = None,
    ) -> "WindowScorer":
        """Build a scorer on the engine the runtime would resolve (IR → ONNX → ckpt)."""
        from sed.runtime.backends import build_inference_backend

        channels = list(
            active_channels if active_channels is not None else cfg.inference.inference_channels
        )
        backend = build_inference_backend(
            cfg, model_dir, prefer=precision, batch_size=max(1, len(channels))
        )
        return cls(
            cfg,
            backend=backend,
            classes=list(classes if classes is not None else cfg.model.classes),
            active_channels=channels,
            record_sample_rate=record_sample_rate,
        )

    def features_for(self, channel_waves: list[np.ndarray]) -> np.ndarray:
        """Stack per-mic windows into one model input batch."""
        specs = []
        for wave in channel_waves:
            w = resample_audio(
                wave, orig_sr=self.record_sample_rate, target_sr=self.feature_sample_rate
            )
            if len(w) != self.window_feature_n:
                w = (w[:self.window_feature_n] if len(w) > self.window_feature_n
                     else np.pad(w, (0, self.window_feature_n - len(w))))
            spec = self.mel(w)
            if hasattr(spec, "numpy"):
                spec = spec.numpy()
            # Drop the extractor's batch axis: (1, bands, T) → (bands, T), or
            # (1, C, H, W) → (C, H, W) for colormap images.
            specs.append(spec[0])
        stacked = np.stack(specs, axis=0).astype(np.float32)
        return stacked[:, None, :, :] if stacked.ndim == 3 else stacked

    def score(self, channel_waves: list[np.ndarray]) -> ScoredWindow:
        batch = self.features_for(channel_waves)
        logits = self.backend.infer(batch)
        probs_per_mic = np.asarray(logits_to_probabilities(logits, self.cfg))
        pooled = pool_mic_probabilities(probs_per_mic, self.mic_mode, self.neg_idx)

        stacked = np.stack([np.asarray(w) for w in channel_waves], axis=-1) if channel_waves else None
        peak = float(np.max(np.abs(stacked))) if stacked is not None and stacked.size else 0.0
        rms = (float(np.sqrt(np.mean(np.square(stacked.mean(axis=-1)))))
               if stacked is not None and stacked.size else 0.0)
        return ScoredWindow(
            probs={c: float(p) for c, p in zip(self.classes, pooled, strict=False)},
            probs_per_mic=probs_per_mic,
            peak_dbfs=_dbfs(peak),
            rms_dbfs=_dbfs(rms),
        )
