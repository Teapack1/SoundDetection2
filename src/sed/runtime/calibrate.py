"""Threshold calibration against batches of recorded negative clips.

The procedure:
  1. Get the project's or library's negative/background clips.
  2. Score each one through the deployed engine.
  3. Relax all target thresholds together until the system reaches its
     false-alarm budget, expressed as **false fires per 1000 clips**.
  4. Write the calibrated thresholds into runtime_settings.json.

Why per 1000 clips and not per hour
-----------------------------------
The material is recorded the way the venue is worked: onset-triggered batches
captured while the room is busy, not continuous audio. Each clip is therefore
*already* a gate survivor — the window production would have scored — so the
observable denominator is "clips", not "hours". "1 false fire in 1000 samples"
is a number this corpus can actually support; FP/hour is not, because nothing
here records how many hours those clips were drawn from.

That also means the onset detector must NOT be re-run here. It needs about a
second of audio to establish its adaptive baseline, so re-detecting on a 1 s
clip finds nothing and the clip would be silently dropped from the sweep.

Each clip is one independent decision opportunity, so the cross-clip refractory
does not apply: two clips are two separate moments in the venue, not two
windows a refractory would collapse.

Record room tone, crowd beds and distractors into ``data_library/sounds/``,
assign the labels to ``_negative``, and this sweeps against them. The same
batches can be assigned to the frozen test so one corpus serves both.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

from sed.audio.features import build_feature_extractor
from sed.config import (
    LIBRARY_DIR,
    Config,
    ProjectLayout,
    dump_runtime_settings,
    load_runtime_settings,
    resolve_active_model_dir,
    runtime_settings_from_config,
)

log = logging.getLogger(__name__)


NegativeSource = Literal["venue", "library"]


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
                cfg, layout.id, "_negative", split="train", selection=selection
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
    target_per_1000: float = 1.0,
    negative_source: NegativeSource = "venue",
    model_id: str | None = None,
) -> dict:
    """Sweep thresholds against negative clips to hold ≤ N false fires per 1000.

    Returns a result dict — ``thresholds`` (the chosen per-class values, the
    historical return), plus the achieved rate per class and how many clips were
    scored, so the UI can show what the calibration actually bought.

    The budget is per 1000 clips because that is the denominator the recorded
    material supports; see the module docstring.
    """
    layout = ProjectLayout(project_id)
    classes = list(cfg.model.classes)
    # `model_id` exists for the post-training call: at that point the freshly
    # trained run is not yet the active model, so resolving "active" would
    # calibrate the PREVIOUS bundle. Defaulting to None keeps every existing
    # caller on the active model.

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

    # Resolve the runnable artifact (heavy imports deferred — keeps pure
    # helpers testable in environments without the runtime, e.g. CI sandboxes).
    # Any engine works: OpenVINO IR, ONNX, or the raw checkpoint, so FP/h
    # calibration is available even for never-exported runs.
    from sed.runtime.backends import build_inference_backend
    model_dir = (layout.model_dir(model_id) if model_id
                 else resolve_active_model_dir(layout, cfg.deploy.active_model))
    backend = build_inference_backend(cfg, model_dir, prefer=None, batch_size=1)
    log.info("calibration model: %s (%s engine)", backend.path, backend.engine)

    mel = build_feature_extractor(cfg)

    # One scorer, shared with the live runtime (sed.runtime.scoring), so the
    # windowing and mic pooling being calibrated are the deployed ones.
    from sed.runtime.scoring import WindowScorer

    scorer = WindowScorer(
        cfg, backend=backend, classes=classes, feature_extractor=mel,
        record_sample_rate=cfg.model.sample_rate,
    )

    log.info("scoring %d negative clip(s) ...", len(negative_files))
    probs = _score_clip_batch(scorer, negative_files)
    if probs.size == 0:
        raise RuntimeError("No usable negative samples after preprocessing.")
    n_clips = int(probs.shape[0])
    log.info("scored %d/%d clips (source=%s)", n_clips, len(negative_files), negative_source)

    roles = cfg.class_roles(classes)
    negative_classes = {cls for cls, role in roles.items() if role == "negative"}
    targets = [c for c in classes if c not in negative_classes]
    grid = np.linspace(0.30, 0.95, 14)

    def _system_rate(thresholds: dict[str, float]) -> tuple[float, dict[str, int]]:
        """Fires per 1000 clips with ALL classes armed at once.

        The old sweep pinned every other class at 1.5 and stored max() of the
        per-class rates. With several targets that under-reports the system
        rate, which is nearer the sum: a clip where a *different* target scored
        highest was dropped from the swept class's count instead of being
        charged to the system.
        """
        hits = _count_fires(
            probs, classes,
            thresholds={**thresholds, "_negative": cfg.inference.negative_max},
            negative_max=cfg.inference.negative_max,
            margin=cfg.inference.margin_to_second_best,
            negative_classes=negative_classes,
        )
        total = sum(hits.values())
        return total / n_clips * 1000.0, hits

    # Start strict and relax every class together until the SYSTEM budget is
    # spent, so the rate reported is the one being bought.
    chosen: dict[str, float] = {c: float(grid[-1]) for c in targets}
    system_rate, per_class_hits = _system_rate(chosen)
    for thr in reversed(grid[:-1]):
        candidate = {c: float(thr) for c in targets}
        rate, hits = _system_rate(candidate)
        log.info("  all classes thr=%.2f → %.2f fires/1000 clips (system)", thr, rate)
        if rate > target_per_1000:
            break
        chosen, system_rate, per_class_hits = candidate, rate, hits

    achieved = {c: per_class_hits.get(c, 0) / n_clips * 1000.0 for c in targets}
    for cls in targets:
        log.info("→ %s threshold = %.2f (%.2f/1000)", cls, chosen[cls], achieved[cls])
    log.info(
        "system false-alarm rate: %.2f per 1000 clips (budget %.2f)",
        system_rate, target_per_1000,
    )

    persist_calibrated_thresholds(
        cfg, project_id, chosen,
        simulation_mode="clip_batch",
        target_per_1000=target_per_1000,
        achieved_per_1000=achieved,
        clips_scored=n_clips,
        negative_source=negative_source,
        system_per_1000=system_rate,
        model_id=model_id,
    )
    return {
        "thresholds": chosen,
        "achieved_per_1000": achieved,
        "system_per_1000": float(system_rate),
        "target_per_1000": float(target_per_1000),
        "clips_scored": n_clips,
        "negative_files": len(negative_files),
        "negative_source": negative_source,
        "simulation_mode": "clip_batch",
    }


def _score_clip_batch(scorer, files: list[Path]) -> np.ndarray:
    """Per-class probabilities for each clip, one row per clip.

    Clips are scored directly rather than re-detected: they were captured by the
    onset trigger, so each one already IS the window production would score.
    Re-running the detector would find nothing (it needs ~1 s to establish its
    baseline) and drop the whole corpus.
    """
    import soundfile as sf

    from sed.audio.resample import resample_audio

    rows: list[np.ndarray] = []
    target_sr = scorer.record_sample_rate
    for path in files:
        try:
            audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as exc:  # noqa: BLE001 — one bad clip must not end the sweep
            log.warning("skip %s: %s", path, exc)
            continue
        if audio.size == 0:
            continue
        channels = [c for c in scorer.active_channels if c < audio.shape[1]] or [0]
        waves = []
        for ch in channels:
            w = audio[:, ch]
            if file_sr != target_sr:
                w = resample_audio(w, orig_sr=file_sr, target_sr=target_sr)
            waves.append(w)
        try:
            scored = scorer.score(waves)
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", path, exc)
            continue
        rows.append(np.array([scored.probs[c] for c in scorer.classes], dtype=np.float64))
    return np.stack(rows) if rows else np.zeros((0, len(scorer.classes)))


def _count_fires(
    probs: np.ndarray,
    classes: list[str],
    *,
    thresholds: dict[str, float],
    negative_max: float,
    margin: float,
    negative_classes: set[str] | None = None,
) -> dict[str, int]:
    """Per-class fire counts over a clip batch, using the runtime's gate order.

    Each clip is one independent decision opportunity, so no refractory applies:
    two clips are two separate moments in the venue, not two windows a
    refractory would collapse.
    """
    negative_classes = {"_negative"} if negative_classes is None else negative_classes
    negative_idx = {i for i, c in enumerate(classes) if c in negative_classes}
    target_idx = [i for i in range(len(classes)) if i not in negative_idx]
    hits = {classes[i]: 0 for i in target_idx}
    if not target_idx or probs.size == 0:
        return hits

    target_probs = probs[:, target_idx]
    order = np.argsort(-target_probs, axis=1)
    winner_col = order[:, 0]
    winner_val = np.take_along_axis(target_probs, winner_col[:, None], axis=1)[:, 0]
    runner_val = (
        np.take_along_axis(target_probs, order[:, 1][:, None], axis=1)[:, 0]
        if target_probs.shape[1] > 1 else np.zeros(len(probs))
    )

    ok = np.ones(len(probs), dtype=bool)
    if negative_idx:                               # negative veto
        ok &= (probs[:, sorted(negative_idx)] <= negative_max).all(axis=1)
    ok &= (winner_val - runner_val) >= margin     # margin gate
    thr = np.array([thresholds.get(classes[target_idx[c]], 1.0) for c in winner_col])
    ok &= winner_val >= thr                       # per-class threshold

    for row in np.nonzero(ok)[0]:
        hits[classes[target_idx[winner_col[row]]]] += 1
    return hits


def persist_calibrated_thresholds(
    cfg: Config,
    project_id: str,
    thresholds: dict[str, float],
    *,
    simulation_mode: str = "",
    target_per_1000: float | None = None,
    achieved_per_1000: dict[str, float] | None = None,
    clips_scored: int | None = None,
    negative_source: str | None = None,
    system_per_1000: float | None = None,
    model_id: str | None = None,
) -> Path:
    """Persist calibrated thresholds into a model's runtime settings.

    The optional achieved-rate arguments are stored as provenance in the
    ``calibration`` block so the Runs UI can show what the sweep actually bought
    (peak per-class rate, clips scored) without re-running it.
    """
    layout = ProjectLayout(project_id)
    model_dir = (layout.model_dir(model_id) if model_id
                 else resolve_active_model_dir(layout, cfg.deploy.active_model))
    runtime_path = model_dir / "runtime_settings.json"
    settings = runtime_settings_from_config(cfg)
    existing = load_runtime_settings(runtime_path)
    if existing:
        settings.update(existing)
    settings.setdefault("thresholds", {})
    settings["thresholds"].update({k: float(v) for k, v in thresholds.items()})
    if simulation_mode or achieved_per_1000 is not None:
        # Provenance only — apply_runtime_settings ignores this block.
        block: dict = {
            "mode": "fp_budget",
            "simulation_mode": simulation_mode,
            "target_per_1000": target_per_1000,
        }
        if achieved_per_1000 is not None:
            block["achieved_per_1000"] = {
                k: float(v) for k, v in achieved_per_1000.items() if v is not None
            }
            rates = [float(v) for v in achieved_per_1000.values() if v is not None]
            # Peak per-class rate, kept alongside the system rate: one says which
            # class is loosest, the other says what the venue actually gets.
            block["max_per_1000"] = max(rates) if rates else None
        if system_per_1000 is not None:
            block["system_per_1000"] = float(system_per_1000)
        if clips_scored is not None:
            block["clips_scored"] = int(clips_scored)
        if negative_source:
            block["negative_source"] = negative_source
        settings["calibration"] = block
    out = dump_runtime_settings(settings, runtime_path)
    log.info("thresholds written to %s", out)
    return out
