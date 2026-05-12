"""Dataset inspection: visualize and quality-check recorded clips.

Generates an HTML report with:
  - Class distribution (counts, durations)
  - Per-clip thumbnails: log-mel spectrogram + waveform + audio playback
  - Outlier flagging: clips farthest from their class centroid in feature space
  - 2-D UMAP scatter of all clips colored by class — visually find mislabeled
    or anomalous samples

The report is a single self-contained HTML file (with inline base64 PNGs for
spectrograms and a copy of audio files served from a relative path).

Outliers are written to `outliers.csv` for easy quarantine: copy/paste UUIDs
into a script that adds `"flagged": true` to each clip's sidecar JSON.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from sed.audio.resample import resample_audio
from sed.audio.features import waveform_to_logmel
from sed.config import Config, ProjectLayout

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-clip features for outlier detection
# ---------------------------------------------------------------------------


def _summary_features(wave: np.ndarray, sr: int) -> np.ndarray:
    """Compact embedding for outlier detection (no NN required).

    Concatenates: mean+std log-mel per band (256 dims for 128 mels) + RMS,
    spectral centroid, zero-crossing-rate. Cheap, deterministic, robust.
    """
    import librosa
    if wave.ndim > 1:
        wave = wave.mean(axis=-1)
    wave = wave.astype(np.float32)
    if len(wave) < sr // 2:
        wave = np.pad(wave, (0, sr // 2 - len(wave)))
    mel = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=64, n_fft=1024, hop_length=256)
    log_mel = np.log1p(mel)
    feat = np.concatenate(
        [
            log_mel.mean(axis=1),
            log_mel.std(axis=1),
            np.array([
                _safe(librosa.feature.rms(y=wave).mean()),
                _safe(librosa.feature.spectral_centroid(y=wave, sr=sr).mean()),
                _safe(librosa.feature.zero_crossing_rate(wave).mean()),
            ]),
        ]
    )
    return feat.astype(np.float32)


def _safe(x) -> float:
    v = float(np.asarray(x).item() if hasattr(x, "item") else x)
    return 0.0 if not np.isfinite(v) else v


# ---------------------------------------------------------------------------
# Plotting helpers — return base64 PNG strings
# ---------------------------------------------------------------------------


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _spectrogram_thumb(wave: np.ndarray, sr: int, height: int = 96) -> str:
    spec = waveform_to_logmel(wave, sample_rate=sr)[0]   # (n_mels, T)
    fig, ax = plt.subplots(figsize=(3.0, 1.0))
    ax.imshow(spec, aspect="auto", origin="lower", cmap="magma",
              interpolation="nearest")
    ax.axis("off")
    return _png_b64(fig)


def _waveform_thumb(wave: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(3.0, 0.6))
    ax.plot(wave, linewidth=0.5, color="#222")
    ax.axis("off")
    ax.set_ylim(-1, 1)
    return _png_b64(fig)


def _umap_scatter(
    feats: np.ndarray,
    labels: list[str],
    classes: list[str],
) -> str:
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        emb = reducer.fit_transform(feats)
    except Exception as e:  # noqa: BLE001
        log.warning("UMAP failed (%s) — falling back to PCA", e)
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=42).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(classes):
        mask = np.array(labels) == cls
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, label=cls,
                   color=cmap(i % 10), alpha=0.7)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Per-clip feature scatter (UMAP)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    return _png_b64(fig)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def _detect_outliers(
    feats: np.ndarray, labels: list[str], top_k_per_class: int = 5
) -> list[int]:
    """Return indices of clips farthest from their class centroid (per class)."""
    out: list[int] = []
    for cls in set(labels):
        mask = np.array([lab == cls for lab in labels])
        if mask.sum() < 4:
            continue
        sub = feats[mask]
        centroid = sub.mean(axis=0)
        dists = np.linalg.norm(sub - centroid, axis=1)
        idxs_sub = np.argsort(-dists)[:top_k_per_class]
        global_idxs = np.flatnonzero(mask)[idxs_sub]
        out.extend(global_idxs.tolist())
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _gather_clips(layout: ProjectLayout) -> list[tuple[Path, str, dict]]:
    """Return (wav_path, class_name, sidecar_meta_or_empty) tuples."""
    items: list[tuple[Path, str, dict]] = []
    pos_root = layout.recordings_dir / "positives"
    if pos_root.exists():
        for class_dir in sorted(pos_root.iterdir()):
            if not class_dir.is_dir():
                continue
            for wav in sorted(class_dir.rglob("*.wav")):
                meta = _load_sidecar(wav)
                items.append((wav, class_dir.name, meta))
    hn_root = layout.recordings_dir / "hardneg"
    if hn_root.exists():
        for class_dir in sorted(hn_root.iterdir()):
            if not class_dir.is_dir():
                continue
            for wav in sorted(class_dir.rglob("*.wav")):
                meta = _load_sidecar(wav)
                items.append((wav, f"hardneg/{class_dir.name}", meta))
    if layout.ambience_dir.exists():
        for wav in sorted(layout.ambience_dir.rglob("*.wav")):
            meta = _load_sidecar(wav)
            items.append((wav, "ambience", meta))
    return items


def _load_sidecar(wav: Path) -> dict:
    side = wav.with_suffix(".json")
    if not side.exists():
        return {}
    try:
        with open(side, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def inspect_project(
    cfg: Config, project_id: str, max_thumbs_per_class: int = 30,
    augmentation_preview: bool = True, aug_per_class: int = 12,
    interactive: bool = False,
) -> Path:
    """Generate the HTML inspection report. Returns its path."""
    layout = ProjectLayout(project_id)
    layout.reports_dir.mkdir(parents=True, exist_ok=True)
    items = _gather_clips(layout)
    if not items:
        raise RuntimeError(f"No clips found under {layout.recordings_dir}")

    log.info("Inspecting %d clips ...", len(items))

    feats_list: list[np.ndarray] = []
    labels: list[str] = []
    durations: list[float] = []
    paths: list[Path] = []
    metas: list[dict] = []
    for wav, label, meta in items:
        try:
            data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=-1)
            feats_list.append(_summary_features(data, sr))
            labels.append(label)
            durations.append(len(data) / sr)
            paths.append(wav)
            metas.append(meta)
        except Exception as e:  # noqa: BLE001
            log.warning("skipping %s: %s", wav, e)

    feats = np.stack(feats_list, axis=0)
    classes = sorted(set(labels))

    # Outliers
    outlier_idxs = set(_detect_outliers(feats, labels, top_k_per_class=5))
    outliers_csv = layout.reports_dir / "outliers.csv"
    with open(outliers_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "path", "uuid", "session_id"])
        for i in outlier_idxs:
            w.writerow([
                labels[i],
                str(paths[i]),
                metas[i].get("uuid", ""),
                metas[i].get("session_id", ""),
            ])
    log.info("Flagged %d candidate outliers → %s", len(outlier_idxs), outliers_csv)

    # UMAP
    umap_b64 = _umap_scatter(feats, labels, classes)
    # Cache the UMAP PNG on disk so the interactive server can reuse it on
    # each page refresh without recomputing the embedding (the slow step).
    try:
        import base64 as _b64
        (layout.reports_dir / "umap.png").write_bytes(_b64.b64decode(umap_b64))
    except Exception:  # noqa: BLE001
        pass

    # Class summary
    class_summary = {
        cls: {
            "count": int(sum(1 for l in labels if l == cls)),
            "total_seconds": float(sum(d for l, d in zip(labels, durations) if l == cls)),
        }
        for cls in classes
    }

    # Per-class thumbnail panels
    panels = _build_panels(
        items=list(zip(paths, labels, metas, durations)),
        classes=classes,
        outlier_idxs=outlier_idxs,
        max_per_class=max_thumbs_per_class,
    )

    # Augmentation preview — shows what the model will actually train on
    aug_panels: dict[str, list[dict]] = {}
    aug_summary: dict = {}
    if augmentation_preview:
        try:
            aug_panels, aug_summary = _build_augmentation_preview(
                cfg, project_id, classes, paths, labels, n_per_class=aug_per_class,
                out_dir=layout.reports_dir / "augmented",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("augmentation preview skipped: %s", e)

    html = _render_html(
        project_id=project_id,
        class_summary=class_summary,
        umap_b64=umap_b64,
        panels=panels,
        outliers_count=len(outlier_idxs),
        aug_panels=aug_panels,
        aug_summary=aug_summary,
        interactive=interactive,
    )
    out = layout.reports_dir / "inspection.html"
    out.write_text(html, encoding="utf-8")
    log.info("Wrote %s", out)
    return out


def _quick_rerender(
    cfg: Config, project_id: str, max_thumbs_per_class: int = 200,
    augmentation_preview: bool = True,
) -> str:
    """Re-render the HTML using cached UMAP and current sidecar flags.

    The server calls this on every GET / so freshly-flagged clips show as
    flagged after a page refresh, without re-computing the (slow) UMAP
    embedding. We DO re-scan the recordings dir so newly recorded clips
    are picked up.
    """
    import base64
    layout = ProjectLayout(project_id)
    items = _gather_clips(layout)
    if not items:
        return _render_html(
            project_id=project_id, class_summary={}, umap_b64="",
            panels={}, outliers_count=0, interactive=True,
        )

    paths = [w for w, _l, _m in items]
    labels = [l for _w, l, _m in items]
    metas = [m for _w, _l, m in items]
    durations: list[float] = []
    for wav in paths:
        try:
            data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=-1)
            durations.append(len(data) / sr)
        except Exception:  # noqa: BLE001
            durations.append(0.0)
    classes = sorted({l for l in labels})
    class_summary = {
        cls: {
            "count": int(sum(1 for l in labels if l == cls)),
            "total_seconds": float(sum(d for l, d in zip(labels, durations) if l == cls)),
        }
        for cls in classes
    }
    panels = _build_panels(
        items=list(zip(paths, labels, metas, durations)),
        classes=classes,
        outlier_idxs=set(),
        max_per_class=max_thumbs_per_class,
    )

    umap_png = layout.reports_dir / "umap.png"
    umap_b64 = ""
    if umap_png.exists():
        umap_b64 = base64.b64encode(umap_png.read_bytes()).decode("ascii")

    return _render_html(
        project_id=project_id,
        class_summary=class_summary,
        umap_b64=umap_b64,
        panels=panels,
        outliers_count=0,
        interactive=True,
    )


# ---------------------------------------------------------------------------
# Augmentation preview
# ---------------------------------------------------------------------------


def _build_augmentation_preview(
    cfg: Config,
    project_id: str,
    classes: list[str],
    paths: list[Path],
    labels: list[str],
    n_per_class: int,
    out_dir: Path,
) -> tuple[dict[str, list[dict]], dict]:
    """For each TARGET class, generate `n_per_class` augmented samples,
    save the WAV + spectrogram thumbnail, compute a masking score, and
    return a panel dict for the HTML report."""
    from sed.data.augment import (
        AudioPool,
        WaveformAugmenter,
        masking_score_db,
        resolve_pool_path,
        _ensure_length,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    sr = cfg.audio.sample_rate
    target_n = int(cfg.features.window_seconds * sr)

    amb_pool = AudioPool(resolve_pool_path(cfg.augmentation.ambience_pool), sr)
    ext_pool = AudioPool(resolve_pool_path(cfg.augmentation.external_negative_pool), sr)

    aug = WaveformAugmenter(
        sample_rate=sr,
        ambience_pool=amb_pool,
        external_neg_pool=ext_pool,
        domain_mix_prob=cfg.augmentation.domain_mix_prob,
        snr_db_range=tuple(cfg.augmentation.snr_db_range),
        per_class_min_snr_db=dict(cfg.augmentation.per_class_min_snr_db),
        external_neg_mix_prob=cfg.augmentation.external_negative_mix_prob,
        masking_check=cfg.augmentation.masking_check,
        masking_min_db=cfg.augmentation.masking_min_db,
        time_shift_ms=cfg.augmentation.time_shift_ms,
        pitch_shift_prob=cfg.augmentation.pitch_shift_prob,
        pitch_semitones=tuple(cfg.augmentation.pitch_semitones),
        time_stretch_prob=cfg.augmentation.time_stretch_prob,
        time_stretch=tuple(cfg.augmentation.time_stretch),
        gaussian_noise_amp=tuple(cfg.augmentation.gaussian_noise_amp),
        seed=42,
    )

    panels: dict[str, list[dict]] = {}
    masked_total = 0
    generated_total = 0

    # Group source paths by class
    by_class: dict[str, list[Path]] = {}
    for p, lab in zip(paths, labels):
        by_class.setdefault(lab, []).append(p)

    for cls in classes:
        if cls.startswith("hardneg/") or cls in ("ambience",):
            # show augmentation only for targets — non-targets aren't augmented
            # in the same way
            continue
        if cls == "_negative":
            continue
        sources = by_class.get(cls, [])
        if not sources:
            continue
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        panel: list[dict] = []
        for i in range(n_per_class):
            src = sources[i % len(sources)]
            try:
                data, file_sr = sf.read(str(src), dtype="float32", always_2d=True)
                clean = data[:, 0]
                if file_sr != sr:
                    clean = resample_audio(clean, orig_sr=file_sr, target_sr=sr)
                clean = _ensure_length(clean, target_n)
            except Exception as e:  # noqa: BLE001
                log.warning("aug preview load %s failed: %s", src, e)
                continue

            mixed = aug(clean.copy(), is_target=True, class_name=cls)
            generated_total += 1

            mask_db = masking_score_db(clean, mixed, sr)
            is_masked = mask_db < cfg.augmentation.masking_min_db
            if is_masked:
                masked_total += 1

            # Save WAV (relative path used in HTML so audio plays via file://)
            wav_path = cls_dir / f"aug_{i:02d}_mask{int(mask_db):+03d}dB.wav"
            sf.write(str(wav_path), mixed, sr, subtype="PCM_16")

            spec_b64 = _spectrogram_thumb(mixed, sr)
            wav_b64 = _waveform_thumb(mixed)
            clean_spec_b64 = _spectrogram_thumb(clean, sr)

            panel.append({
                "name": wav_path.name,
                "path": str(wav_path),
                "source": src.name,
                "mask_db": mask_db,
                "warn": is_masked,
                "spec_b64": spec_b64,
                "wav_b64": wav_b64,
                "clean_spec_b64": clean_spec_b64,
            })
        panels[cls] = panel

    summary = {
        "generated": generated_total,
        "masked_skipped": masked_total,
        "masked_pct": (100.0 * masked_total / generated_total) if generated_total else 0.0,
        "snr_range": list(cfg.augmentation.snr_db_range),
        "per_class_min_snr_db": dict(cfg.augmentation.per_class_min_snr_db),
        "masking_min_db": cfg.augmentation.masking_min_db,
    }
    return panels, summary


def _build_panels(
    items: list[tuple[Path, str, dict, float]],
    classes: list[str],
    outlier_idxs: set[int],
    max_per_class: int,
) -> dict[str, list[dict]]:
    panels: dict[str, list[dict]] = {c: [] for c in classes}
    for global_idx, (path, label, meta, duration) in enumerate(items):
        if len(panels[label]) >= max_per_class:
            continue
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=-1)
        except Exception:  # noqa: BLE001
            continue
        spec_b64 = _spectrogram_thumb(data, sr)
        wav_b64 = _waveform_thumb(data)
        # Read current quarantine state from the sidecar (if any). This makes
        # the report show pre-existing flags AND lets the interactive server
        # mode round-trip without losing state between page loads.
        flagged = False
        side = Path(path).with_suffix(".json")
        if side.exists():
            try:
                with open(side, "r", encoding="utf-8") as _f:
                    flagged = bool(json.load(_f).get("flagged", False))
            except Exception:  # noqa: BLE001
                flagged = False
        panels[label].append({
            "path": str(path),
            "name": path.name,
            "uuid": meta.get("uuid", ""),
            "session_id": meta.get("session_id", ""),
            "duration_s": float(duration),
            "outlier": global_idx in outlier_idxs,
            "flagged": flagged,
            "spec_b64": spec_b64,
            "wav_b64": wav_b64,
        })
    return panels


def _render_html(
    project_id: str,
    class_summary: dict,
    umap_b64: str,
    panels: dict[str, list[dict]],
    outliers_count: int,
    aug_panels: dict[str, list[dict]] | None = None,
    aug_summary: dict | None = None,
    interactive: bool = False,
) -> str:
    from jinja2 import Template
    tpl = Template(_HTML_TEMPLATE)
    return tpl.render(
        project_id=project_id,
        class_summary=class_summary,
        umap_b64=umap_b64,
        panels=panels,
        outliers_count=outliers_count,
        aug_panels=aug_panels or {},
        aug_summary=aug_summary or {},
        interactive=interactive,
    )


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SED Inspection — {{ project_id }}</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #f6f6f6; color: #222; max-width: 1200px; margin: 1em auto;
       padding: 0 1em; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }
h2 { margin-top: 1.6em; }
h3.section { background: #333; color: #fff; padding: 0.4em 0.8em; border-radius: 4px; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
           gap: 0.5em; }
.card { background: #fff; padding: 0.6em 1em; border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08); }
.clips { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
         gap: 0.5em; }
.clip { background: #fff; padding: 0.5em; border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06); position: relative;
        transition: opacity 0.15s, background 0.15s, box-shadow 0.15s; }
.clip.outlier { border-left: 4px solid #d33; }
.clip.warn { border-left: 4px solid #d4a017; }
.clip.flagged { background: #fbeaea; opacity: 0.55; }
.clip.flagged .clip-body { filter: grayscale(70%); }
/* Whole-card click toggles flag (interactive mode). Excludes audio + checkbox. */
.interactive .clip .clip-body { cursor: pointer; }
.interactive .clip:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.18); }
.interactive .clip:hover:not(.flagged) { background: #fff7e6; }
.clip h3 { font-size: 0.85em; margin: 0 0 0.2em; word-break: break-all; }
.clip .meta { font-size: 0.75em; color: #666; }
.clip img { display: block; width: 100%; }
.clip .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
.clip .pair-label { font-size: 0.65em; text-align: center; color: #999; }
audio { width: 100%; height: 32px; margin-top: 0.3em; }
.umap { background: #fff; padding: 1em; text-align: center;
        border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); }
.notice { background: #fff8c6; padding: 0.6em 1em; border-left: 4px solid #d4a017;
          margin: 1em 0; border-radius: 0 4px 4px 0; }
.notice.danger { background: #ffe0e0; border-left-color: #d33; }
.mask-good { color: #060; font-weight: 600; }
.mask-warn { color: #c70; font-weight: 600; }
.mask-bad  { color: #c00; font-weight: 600; }

/* Interactive mode: checkbox, sticky toolbar */
.flag-box { position: absolute; top: 6px; right: 8px;
            width: 22px; height: 22px; cursor: pointer; z-index: 2;
            accent-color: #d33; }
.flag-box:hover { transform: scale(1.15); }
.toolbar { position: sticky; top: 0; z-index: 100; background: #222; color: #fff;
           padding: 0.6em 1em; margin: -1em -1em 1em -1em;
           display: flex; gap: 1em; align-items: center; justify-content: space-between;
           box-shadow: 0 2px 6px rgba(0,0,0,0.2); }
.toolbar .count { font-size: 1.1em; font-weight: 600; }
.toolbar .count .num { color: #ff7a7a; }
.toolbar .hint { font-size: 0.85em; opacity: 0.75; }
.toolbar .status { font-size: 0.8em; opacity: 0.6; min-width: 90px; text-align: right; }
.toolbar .status.ok { color: #8eda8e; opacity: 1; }
.toolbar .status.err { color: #ff7a7a; opacity: 1; }
.toolbar button.done {
    background: #2a9d4a; color: #fff; border: none; padding: 0.45em 1em;
    border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 0.95em;
}
.toolbar button.done:hover { background: #36b85a; }
.toolbar button.done:disabled { background: #555; cursor: default; }
.toolbar button.del {
    background: #c33; color: #fff; border: none; padding: 0.45em 1em;
    border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 0.95em;
}
.toolbar button.del:hover { background: #e44; }
.toolbar button.del:disabled { background: #555; cursor: default; }
.clip.deleted { display: none; }
.kbd { background: #444; padding: 1px 6px; border-radius: 3px; font-family: monospace;
       font-size: 0.85em; }
.done-overlay {
    position: fixed; inset: 0; background: rgba(20,20,20,0.92); color: #fff;
    display: none; align-items: center; justify-content: center; flex-direction: column;
    z-index: 1000; font-size: 1.4em; text-align: center; padding: 2em;
}
.done-overlay.show { display: flex; }
.done-overlay .big { font-size: 2em; margin-bottom: 0.4em; }
</style>
</head>
<body class="{% if interactive %}interactive{% endif %}">

{% if interactive %}
<div class="toolbar">
  <div class="count">Flagged: <span class="num" id="flag-count">0</span></div>
  <div class="hint">
    Click any clip to flag it — flagged clips are skipped by
    <span class="kbd">dataset compose</span>. Saves are live.
  </div>
  <div class="status" id="status"></div>
  <button class="del" id="del-btn" title="Permanently delete flagged clips from disk">Delete flagged</button>
  <button class="done" id="done-btn" title="Confirm and stop the inspect server">Done</button>
</div>
<div class="done-overlay" id="done-overlay">
  <div class="big">✅ All changes saved</div>
  <div>You can safely close this tab.</div>
  <div style="opacity:0.6; margin-top:1em; font-size:0.7em;">
    The inspect server has been stopped.
  </div>
</div>
{% endif %}

<h1>SED Inspection — {{ project_id }}</h1>

<div class="notice">
  <b>{{ outliers_count }}</b> candidate outliers flagged (highlighted in red below).
  {% if interactive %}
  Tick the box on any clip to mark it as quarantined — changes are saved live to disk
  and that clip will be skipped at the next <code>clinkd dataset compose</code>.
  {% else %}
  To quarantine clips, either re-run with <code>--serve</code> for the interactive
  UI, or pass the clip paths/UUIDs to <code>clinkd quarantine</code>.
  {% endif %}
</div>

<h2>Class summary</h2>
<div class="summary">
{% for cls, info in class_summary.items() %}
<div class="card">
  <b>{{ cls }}</b><br>
  {{ info.count }} clips · {{ "%.1f"|format(info.total_seconds) }}s total
</div>
{% endfor %}
</div>

<h2>Feature scatter</h2>
<div class="umap">
  <img src="data:image/png;base64,{{ umap_b64 }}" alt="UMAP">
  <p>Same-class clips should cluster together. Stray points are likely
  mislabeled or noisy samples worth investigating.</p>
</div>

<h3 class="section">Raw clips per class</h3>
{% for cls, clips in panels.items() %}
<h2>{{ cls }} ({{ clips|length }} shown)</h2>
<div class="clips">
{% for c in clips %}
<div class="clip{% if c.outlier %} outlier{% endif %}{% if c.flagged %} flagged{% endif %}"
     data-path="{{ c.path|e }}">
  {% if interactive %}
  <input class="flag-box" type="checkbox"
         title="Mark as quarantined (excluded from training)"
         {% if c.flagged %}checked{% endif %}>
  {% endif %}
  <div class="clip-body">
    <h3>{{ c.name }}{% if c.outlier %} ⚠️ outlier{% endif %}</h3>
    <div class="meta">{{ "%.2f"|format(c.duration_s) }}s · session {{ c.session_id[:14] }}</div>
    <img src="data:image/png;base64,{{ c.spec_b64 }}" alt="mel">
    <img src="data:image/png;base64,{{ c.wav_b64 }}" alt="wav">
    {% if interactive %}
    <audio controls preload="none"
           src="/audio?path={{ c.path|urlencode }}"></audio>
    {% else %}
    <audio controls preload="none" src="file://{{ c.path }}"></audio>
    {% endif %}
  </div>
</div>
{% endfor %}
</div>
{% endfor %}

{% if aug_panels %}
<h3 class="section">Augmentation preview — what the model trains on</h3>
<div class="notice {% if aug_summary.masked_pct > 30 %}danger{% endif %}">
  Generated <b>{{ aug_summary.generated }}</b> augmented samples.
  <b>{{ aug_summary.masked_skipped }}</b> ({{ "%.1f"|format(aug_summary.masked_pct) }}%)
  were flagged as <b>over-masked</b> (target buried under noise) and would be
  replaced with the clean target during training.
  {% if aug_summary.masked_pct > 30 %}
  <br>⚠️ More than 30% masking — consider raising the lower SNR bound in
  <code>[augmentation] snr_db_range</code> or adding a <code>per_class_min_snr_db</code>
  entry for the affected class.
  {% endif %}
  <br><span style="font-size:0.85em">SNR range {{ aug_summary.snr_range }} dB · masking floor {{ aug_summary.masking_min_db }} dB</span>
</div>
{% for cls, clips in aug_panels.items() %}
<h2>aug · {{ cls }}</h2>
<div class="clips">
{% for c in clips %}
<div class="clip {% if c.warn %}warn{% endif %}">
  <h3>{{ c.name }}</h3>
  <div class="meta">
    source: {{ c.source }}
    · masking Δ:
    <span class="mask-{% if c.warn %}bad{% elif c.mask_db < 6 %}warn{% else %}good{% endif %}">
      {{ "%+.1f"|format(c.mask_db) }} dB
    </span>
    {% if c.warn %} — TOO MASKED, training will skip this mix{% endif %}
  </div>
  <div class="pair">
    <div><div class="pair-label">clean (source)</div>
      <img src="data:image/png;base64,{{ c.clean_spec_b64 }}" alt="clean"></div>
    <div><div class="pair-label">augmented (mixed)</div>
      <img src="data:image/png;base64,{{ c.spec_b64 }}" alt="mixed"></div>
  </div>
  <img src="data:image/png;base64,{{ c.wav_b64 }}" alt="wav">
  {% if interactive %}
  <audio controls preload="none" src="/audio?path={{ c.path|urlencode }}"></audio>
  {% else %}
  <audio controls preload="none" src="file://{{ c.path }}"></audio>
  {% endif %}
</div>
{% endfor %}
</div>
{% endfor %}
{% endif %}

{% if interactive %}
<script>
// --- Live quarantine: each checkbox POSTs to /api/flag and updates UI -----
const statusEl = document.getElementById("status");
const countEl  = document.getElementById("flag-count");

function refreshCount() {
    const n = document.querySelectorAll(".clip.flagged").length;
    countEl.textContent = n;
}
function flash(msg, cls) {
    statusEl.textContent = msg;
    statusEl.className = "status " + (cls || "");
    if (cls === "ok") setTimeout(() => { statusEl.textContent = ""; statusEl.className = "status"; }, 1500);
}

async function setFlag(path, flagged) {
    flash("saving…");
    try {
        const r = await fetch("/api/flag", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path, flagged }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        flash("saved", "ok");
    } catch (e) {
        flash("save failed", "err");
        throw e;
    }
}

async function toggleCard(card) {
    const path = card.dataset.path;
    const willFlag = !card.classList.contains("flagged");
    const box = card.querySelector(".flag-box");
    try {
        await setFlag(path, willFlag);
        card.classList.toggle("flagged", willFlag);
        if (box) box.checked = willFlag;
        refreshCount();
    } catch { /* status already shown */ }
}

// Checkbox itself
document.querySelectorAll(".flag-box").forEach(box => {
    box.addEventListener("click", ev => ev.stopPropagation());
    box.addEventListener("change", async ev => {
        const card = ev.target.closest(".clip");
        const path = card.dataset.path;
        const flagged = ev.target.checked;
        try {
            await setFlag(path, flagged);
            card.classList.toggle("flagged", flagged);
            refreshCount();
        } catch {
            ev.target.checked = !flagged;
        }
    });
});

// Click anywhere on the card body (except audio controls) to toggle
document.querySelectorAll(".clip").forEach(card => {
    const body = card.querySelector(".clip-body");
    if (!body) return;
    body.addEventListener("click", ev => {
        // Don't toggle when the user clicks inside the audio player
        if (ev.target.closest("audio")) return;
        toggleCard(card);
    });
});

// Delete flagged button — permanently removes WAV + JSON from disk
const delBtn = document.getElementById("del-btn");
if (delBtn) {
    delBtn.addEventListener("click", async () => {
        const flagged = document.querySelectorAll(".clip.flagged:not(.deleted)");
        if (flagged.length === 0) {
            alert("No clips are flagged. Click clips to flag them first.");
            return;
        }
        if (!confirm(`Permanently delete ${flagged.length} clip(s) from disk?\n\nThis cannot be undone.`)) return;
        delBtn.disabled = true;
        flash("deleting…");
        const paths = Array.from(flagged).map(c => c.dataset.path);
        try {
            const r = await fetch("/api/delete", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ paths }),
            });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const data = await r.json();
            const deletedSet = new Set(data.deleted || []);
            flagged.forEach(card => {
                if (deletedSet.has(card.dataset.path)) card.classList.add("deleted");
            });
            flash(`deleted ${data.count} clip(s)`, "ok");
            refreshCount();
        } catch (e) {
            flash("delete failed", "err");
        } finally {
            delBtn.disabled = false;
        }
    });
}

// Done button
const doneBtn = document.getElementById("done-btn");
const doneOverlay = document.getElementById("done-overlay");
if (doneBtn) {
    doneBtn.addEventListener("click", async () => {
        doneBtn.disabled = true;
        flash("finishing…");
        try {
            await fetch("/api/done", { method: "POST" });
        } catch { /* server may close before responding */ }
        flash("done", "ok");
        doneOverlay.classList.add("show");
    });
}

refreshCount();
</script>
{% endif %}

</body>
</html>
"""


def quarantine(cfg: Config, project_id: str, paths_or_uuids: Iterable[str]) -> int:
    """Mark sidecar metadata `flagged=True` for given clips so they're excluded."""
    layout = ProjectLayout(project_id)
    targets = set(paths_or_uuids)
    n = 0
    for wav in layout.recordings_dir.rglob("*.wav"):
        side = wav.with_suffix(".json")
        if not side.exists():
            continue
        try:
            with open(side, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if str(wav) in targets or meta.get("uuid") in targets or wav.name in targets:
            meta["flagged"] = True
            with open(side, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            n += 1
    return n
