"""Threshold calibration against negative/background recordings.

The procedure:
  1. Get project or library negative/background material.
  2. Run the trained model over it offline (the full inference pipeline,
     including smoothing + arbiter).
  3. For each target class, find the lowest threshold that yields
     ≤ target FP/h.
  4. Write the calibrated thresholds to generated runtime_settings.json.

This is the single biggest contributor to a low false-positive rate in
production. The default starting threshold is 0.90, then calibration can
adjust per class from recorded negative/background material.

Record room tone, crowd beds, and distractors into ``data_library/sounds/``
and assign the labels to ``_negative``. Calibration sweeps target thresholds against
those non-target clips to control false positives.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from sed.audio.features import MelExtractor
from sed.audio.resample import resample_audio
from sed.config import (
    LIBRARY_DIR,
    Config,
    ProjectLayout,
    dump_runtime_settings,
    load_runtime_settings,
    resolve_active_model_dir,
    runtime_settings_from_config,
)
from sed.runtime.manifest import resolve_model_path
from sed.runtime.smooth import TriggerArbiter
from sed.train.objective import logits_to_probabilities

log = logging.getLogger(__name__)


NegativeSource = Literal["venue", "library"]


def _iter_windows(wave: np.ndarray, sr: int, win_s: float, hop_ms: int):
    win_n = int(sr * win_s)
    hop_n = int(sr * hop_ms / 1000.0)
    for start in range(0, len(wave) - win_n + 1, hop_n):
        yield wave[start:start + win_n]


def _gather_negative_files(
    layout: ProjectLayout,
    source: NegativeSource,
    negative_dir: Path | None,
    cfg: Config | None = None,
) -> list[Path]:
    """Return calibration negative/background files for the selected source."""
    if negative_dir is not None:
        return sorted(negative_dir.rglob("*.wav"))

    if source == "venue" and cfg is not None:
        from sed.data.compose import _library_clips_for_class, load_dataset_selection

        selection = load_dataset_selection(layout)
        return [
            Path(clip.path)
            for clip in _library_clips_for_class(
                cfg, layout.id, "_negative", dataset_role="train", selection=selection
            )
        ]

    if source == "library":
        return sorted((LIBRARY_DIR / "sounds").rglob("*.wav"))

    if source == "venue":
        return []

    raise ValueError(f"unknown negative source: {source}")


def calibrate(
    cfg: Config,
    project_id: str,
    negative_dir: Path | None = None,
    target_fp_per_hour: float = 1.0,
    negative_source: NegativeSource = "venue",
) -> dict[str, float]:
    layout = ProjectLayout(project_id)
    classes = list(cfg.model.classes)
    neg_idx = classes.index("_negative") if "_negative" in classes else None

    negative_files = _gather_negative_files(layout, negative_source, negative_dir, cfg)
    if not negative_files:
        raise RuntimeError(
            f"No negative clips for source='{negative_source}'. "
            "Record/import sounds and assign one or more labels to _negative."
        )
    log.info(
        "calibration negatives: source=%s files=%d",
        negative_source, len(negative_files),
    )

    # Compile model (openvino import deferred — keeps pure helpers testable
    # in environments without the runtime, e.g. CI sandboxes)
    import openvino as ov
    from sed.runtime.infer import compile_openvino_model, normalize_openvino_logits
    model_dir = resolve_active_model_dir(layout, cfg.deploy.active_model)
    core = ov.Core()
    _model, cm = compile_openvino_model(
        core,
        resolve_model_path(model_dir),
        cfg,
        batch_size=1,
        config={"PERFORMANCE_HINT": "LATENCY"},
    )
    req = cm.create_infer_request()

    mel = MelExtractor(
        sample_rate=cfg.model.sample_rate,
        n_fft=cfg.features.n_fft,
        win_length=cfg.features.win_length,
        hop_length=cfg.features.hop_length,
        n_mels=cfg.features.n_mels,
        fmin=cfg.features.fmin,
        fmax=cfg.features.fmax,
        frontend=cfg.features.frontend,
    )

    sr = cfg.model.sample_rate
    # Score every window once, store probabilities per class
    log.info("scoring negatives ...")
    all_probs: list[np.ndarray] = []
    total_seconds = 0.0

    for f in negative_files:
        try:
            wave, file_sr = sf.read(str(f), dtype="float32", always_2d=True)
            wave = wave[:, 0]
            if file_sr != sr:
                wave = resample_audio(wave, orig_sr=file_sr, target_sr=sr)
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", f, e)
            continue
        for w in _iter_windows(wave, sr, cfg.features.window_seconds,
                               cfg.inference.hop_ms):
            spec = mel(w)
            if hasattr(spec, "numpy"):
                spec = spec.numpy()
            x = spec.reshape(1, 1, cfg.features.n_mels, -1).astype(np.float32)
            logits = normalize_openvino_logits(
                list(req.infer({0: x}).values())[0],
                batch_size=x.shape[0],
                num_classes=len(classes),
            )
            # Use the SAME logits→prob mapping as live inference (softmax for
            # single-label cross_entropy, sigmoid for multilabel). Hardcoding
            # sigmoid here calibrated thresholds in the wrong probability space.
            probs = logits_to_probabilities(logits, cfg)          # (1, num_classes)
            all_probs.append(np.asarray(probs)[0])
        total_seconds += len(wave) / sr

    if not all_probs:
        raise RuntimeError("No usable negative samples after preprocessing.")

    probs_arr = np.stack(all_probs, axis=0)             # (T, num_classes)
    log.info(
        "scored %d windows / %.1f minutes of negatives (source=%s)",
        len(probs_arr), total_seconds / 60, negative_source,
    )

    # Sweep threshold per target class
    chosen: dict[str, float] = {}
    grid = np.linspace(0.30, 0.95, 14)

    for cls in classes:
        if cls == "_negative":
            continue
        best_thr = 0.95                                  # safest default
        for thr in grid:
            arbiter = TriggerArbiter(
                classes=classes,
                thresholds={c: thr if c == cls else 1.5 for c in classes},
                smoothers_cfg={
                    "median_window": cfg.inference.smoothing_median_window,
                    "ema_alpha": cfg.inference.smoothing_ema_alpha,
                    "default_threshold": thr,
                    "hysteresis_low": cfg.inference.hysteresis_low,
                    "consecutive_frames": cfg.inference.consecutive_frames,
                    "smoothing_enabled": cfg.inference.smoothing_enabled,
                },
                negative_index=neg_idx,
                negative_max=cfg.inference.negative_max,
                margin_to_second_best=cfg.inference.margin_to_second_best,
                global_refractory_ms=cfg.inference.trigger_refractory_ms,
            )
            triggers = 0
            now = 0.0
            dt = cfg.inference.hop_ms / 1000.0
            for row in probs_arr:
                ev = arbiter.step({c: float(p) for c, p in zip(classes, row)}, now=now)
                if ev is not None:
                    triggers += 1
                now += dt
            fp_per_hour = triggers / max(total_seconds, 1.0) * 3600.0
            log.info("  %s thr=%.2f → %d triggers (%.2f/h)", cls, thr, triggers, fp_per_hour)
            if fp_per_hour <= target_fp_per_hour:
                best_thr = float(thr)
                break
        chosen[cls] = best_thr
        log.info("→ %s threshold = %.2f", cls, best_thr)

    persist_calibrated_thresholds(cfg, project_id, chosen)
    return chosen


def persist_calibrated_thresholds(
    cfg: Config,
    project_id: str,
    thresholds: dict[str, float],
) -> Path:
    """Persist calibrated thresholds into the active model's runtime settings."""
    layout = ProjectLayout(project_id)
    model_dir = resolve_active_model_dir(layout, cfg.deploy.active_model)
    runtime_path = model_dir / "runtime_settings.json"
    settings = runtime_settings_from_config(cfg)
    existing = load_runtime_settings(runtime_path)
    if existing:
        settings.update(existing)
    settings.setdefault("thresholds", {})
    settings["thresholds"].update({k: float(v) for k, v in thresholds.items()})
    out = dump_runtime_settings(settings, runtime_path)
    log.info("thresholds written to %s", out)
    return out
