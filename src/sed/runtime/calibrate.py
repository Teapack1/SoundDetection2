"""Threshold calibration against ambience.

The procedure:
  1. Get ambience material — venue-recorded, library-pooled, or *simulated*
     by mixing empty-venue room tone with library "busy" ambience.
  2. Run the trained model over it offline (the full inference pipeline,
     including smoothing + arbiter).
  3. For each target class, find the lowest threshold that yields
     ≤ target FP/h.
  4. Write the calibrated thresholds back to project.toml.

This is the single biggest contributor to a low false-positive rate in
production. Calibrated thresholds typically end up at 0.7–0.85.

## Ambience sources

In practice, system setup almost always happens BEFORE the venue opens —
HVAC running, dim lights, no crowd. Real busy-hour ambience can't be
recorded on-site at install time. The three modes solve this:

  - ``venue`` (default if recordings exist): use whatever ambience was
    recorded at the venue. Fine if the team came back during peak hours
    or if the install includes a sound system playing crowd noise.
  - ``library``: use the shared library ambience pool only — assumes the
    library has crowd/restaurant recordings representative of the venue.
  - ``simulated_busy``: combine venue-recorded *empty* ambience (room
    tone, HVAC, reverb signature) with library *busy* ambience, mixed at
    randomised SNRs that bracket realistic peak conditions. This lets you
    calibrate against a plausible peak-hour soundscape even when only
    empty-venue recordings exist.

After deployment, recapture an hour of real busy-hour ambience and
re-run ``clinkd calibrate --ambience-source venue`` to refine.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from sed.audio.features import MelExtractor
from sed.audio.resample import resample_audio
from sed.config import LIBRARY_DIR, Config, ProjectLayout
from sed.data.augment import _load_wav_resampled, mix_at_snr
from sed.runtime.manifest import load_manifest, resolve_model_path
from sed.runtime.smooth import TriggerArbiter

log = logging.getLogger(__name__)


AmbienceSource = Literal["venue", "library", "simulated_busy"]


def _iter_windows(wave: np.ndarray, sr: int, win_s: float, hop_ms: int):
    win_n = int(sr * win_s)
    hop_n = int(sr * hop_ms / 1000.0)
    for start in range(0, len(wave) - win_n + 1, hop_n):
        yield wave[start:start + win_n]


def _gather_ambience_files(
    layout: ProjectLayout,
    source: AmbienceSource,
    ambience_dir: Path | None,
) -> tuple[list[Path], list[Path]]:
    """Return (primary_files, busy_overlay_files) depending on source.

    For ``simulated_busy`` we need both: primary = empty-venue room tone,
    overlay = library busy ambience.
    """
    if source == "venue":
        primary = (
            sorted(ambience_dir.rglob("*.wav")) if ambience_dir is not None
            else sorted(layout.ambience_dir.rglob("*.wav"))
        )
        return primary, []

    library_amb = sorted((LIBRARY_DIR / "ambience").rglob("*.wav"))

    if source == "library":
        return library_amb, []

    if source == "simulated_busy":
        venue_files = (
            sorted(ambience_dir.rglob("*.wav")) if ambience_dir is not None
            else sorted(layout.ambience_dir.rglob("*.wav"))
        )
        # Fall back to library-only if the venue has no recordings — strictly
        # less realistic but better than refusing to calibrate.
        if not venue_files:
            log.warning(
                "simulated_busy: no venue ambience found, using library-only"
            )
            return library_amb, []
        return venue_files, library_amb

    raise ValueError(f"unknown ambience source: {source}")


def _generate_simulated_window(
    primary_wave: np.ndarray,
    overlay_pool: list[np.ndarray],
    win_n: int,
    rng: random.Random,
    snr_range_db: tuple[float, float] = (-3.0, 12.0),
) -> np.ndarray:
    """Build a single 'simulated busy' window: empty-venue tone + busy overlay.

    SNR is the busy-overlay-relative-to-empty ratio: higher = louder crowd.
    The default range (-3 to +12 dB) brackets realistic peak conditions:
    -3 dB = quiet evening, +12 dB = packed Friday night.
    """
    if len(primary_wave) < win_n:
        primary = np.tile(primary_wave, win_n // max(1, len(primary_wave)) + 1)[:win_n]
    else:
        start = rng.randint(0, len(primary_wave) - win_n)
        primary = primary_wave[start:start + win_n].copy()

    if not overlay_pool:
        return primary

    overlay_full = rng.choice(overlay_pool)
    if len(overlay_full) < win_n:
        overlay = np.tile(overlay_full, win_n // max(1, len(overlay_full)) + 1)[:win_n]
    else:
        ostart = rng.randint(0, len(overlay_full) - win_n)
        overlay = overlay_full[ostart:ostart + win_n].copy()

    # mix_at_snr(signal, noise, snr_db): the busy overlay is the signal we
    # want at the given SNR over the empty-venue tone (which is the noise
    # floor). Higher SNR → louder crowd over the room tone.
    snr_db = rng.uniform(*snr_range_db)
    return mix_at_snr(signal=overlay, noise=primary, snr_db=snr_db)


def calibrate(
    cfg: Config,
    project_id: str,
    ambience_dir: Path | None = None,
    target_fp_per_hour: float = 1.0,
    ambience_source: AmbienceSource = "venue",
    simulated_minutes: float = 30.0,
    simulated_snr_db_range: tuple[float, float] = (-3.0, 12.0),
    seed: int = 0,
) -> dict[str, float]:
    layout = ProjectLayout(project_id)
    manifest = load_manifest(layout)
    classes = manifest["classes"]
    neg_idx = manifest["negative_index"]

    primary_files, overlay_files = _gather_ambience_files(
        layout, ambience_source, ambience_dir
    )
    if not primary_files:
        raise RuntimeError(
            f"No ambience clips for source='{ambience_source}'. "
            "Record some with `clinkd record ambience` (venue) or "
            "`clinkd library record-ambience` (library)."
        )
    log.info(
        "calibration ambience: source=%s primary=%d overlay=%d",
        ambience_source, len(primary_files), len(overlay_files),
    )

    # Compile model (openvino import deferred — keeps pure helpers testable
    # in environments without the runtime, e.g. CI sandboxes)
    import openvino as ov
    core = ov.Core()
    cm = core.compile_model(
        core.read_model(str(resolve_model_path(layout, manifest))),
        "CPU",
        config={"PERFORMANCE_HINT": "LATENCY"},
    )
    req = cm.create_infer_request()

    mel = MelExtractor(
        sample_rate=manifest["sample_rate"],
        n_fft=manifest["n_fft"],
        win_length=manifest["win_length"],
        hop_length=manifest["hop_length"],
        n_mels=manifest["n_mels"],
        fmin=manifest["fmin"],
        fmax=manifest["fmax"],
    )

    sr = manifest["sample_rate"]
    win_n = int(sr * manifest["window_seconds"])
    hop_n = int(sr * cfg.inference.hop_ms / 1000.0)

    # Score every window once, store probabilities per class
    log.info("scoring ambience ...")
    all_probs: list[np.ndarray] = []
    total_seconds = 0.0

    if ambience_source == "simulated_busy" and overlay_files:
        # Build a synthetic stream of simulated_minutes by drawing windows
        # at hop_n stride. Pre-load all material once (small RAM footprint
        # per file, total typically < 200 MB).
        rng = random.Random(seed)
        primaries = []
        for f in primary_files:
            try:
                primaries.append(_load_wav_resampled(f, sr))
            except Exception as e:  # noqa: BLE001
                log.warning("skip primary %s: %s", f, e)
        overlays = []
        for f in overlay_files:
            try:
                overlays.append(_load_wav_resampled(f, sr))
            except Exception as e:  # noqa: BLE001
                log.warning("skip overlay %s: %s", f, e)
        if not primaries:
            raise RuntimeError("no usable primary (empty-venue) clips")
        if not overlays:
            log.warning("no usable overlay (library) clips, falling through")

        n_windows = int(simulated_minutes * 60 * sr / hop_n)
        log.info(
            "synthesising %d simulated busy windows (%.1f min @ SNR %s dB)",
            n_windows, simulated_minutes, list(simulated_snr_db_range),
        )
        for _ in range(n_windows):
            primary = rng.choice(primaries)
            w = _generate_simulated_window(
                primary, overlays, win_n, rng, simulated_snr_db_range,
            )
            spec = mel(w)
            if hasattr(spec, "numpy"):
                spec = spec.numpy()
            x = spec.reshape(1, 1, manifest["n_mels"], -1).astype(np.float32)
            logits = list(req.infer({0: x}).values())[0][0]
            all_probs.append(1.0 / (1.0 + np.exp(-logits)))
        total_seconds = simulated_minutes * 60
    else:
        for f in primary_files:
            try:
                wave, file_sr = sf.read(str(f), dtype="float32", always_2d=True)
                wave = wave[:, 0]
                if file_sr != sr:
                    wave = resample_audio(wave, orig_sr=file_sr, target_sr=sr)
            except Exception as e:  # noqa: BLE001
                log.warning("skip %s: %s", f, e)
                continue
            for w in _iter_windows(wave, sr, manifest["window_seconds"],
                                   cfg.inference.hop_ms):
                spec = mel(w)
                if hasattr(spec, "numpy"):
                    spec = spec.numpy()
                x = spec.reshape(1, 1, manifest["n_mels"], -1).astype(np.float32)
                logits = list(req.infer({0: x}).values())[0][0]
                all_probs.append(1.0 / (1.0 + np.exp(-logits)))
            total_seconds += len(wave) / sr

    if not all_probs:
        raise RuntimeError("No usable ambience samples after preprocessing.")

    probs_arr = np.stack(all_probs, axis=0)             # (T, num_classes)
    log.info(
        "scored %d windows / %.1f minutes of ambience (source=%s)",
        len(probs_arr), total_seconds / 60, ambience_source,
    )

    # Sweep threshold per target class
    chosen: dict[str, float] = {}
    grid = np.linspace(0.30, 0.95, 14)

    for ci, cls in enumerate(classes):
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

    # Persist back into project.toml
    _write_thresholds(layout.project_toml, chosen)
    return chosen


def _write_thresholds(toml_path: Path, thresholds: dict[str, float]) -> None:
    """Append/replace [thresholds] block in project.toml. Minimal-diff approach."""
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    if not toml_path.exists():
        toml_path.write_text("")

    text = toml_path.read_text(encoding="utf-8")
    new_block = "[thresholds]\n" + "\n".join(
        f"{k} = {v:.3f}" for k, v in thresholds.items()
    ) + "\n"

    if "[thresholds]" not in text:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n" + new_block
    else:
        # Replace existing section up to next [section] header
        lines = text.splitlines()
        out_lines: list[str] = []
        in_section = False
        replaced = False
        for line in lines:
            if line.strip() == "[thresholds]":
                in_section = True
                if not replaced:
                    out_lines.append(new_block.rstrip())
                    replaced = True
                continue
            if in_section and line.startswith("[") and line.endswith("]"):
                in_section = False
            if not in_section:
                out_lines.append(line)
        text = "\n".join(out_lines) + "\n"

    toml_path.write_text(text, encoding="utf-8")
    log.info("thresholds written to %s", toml_path)
