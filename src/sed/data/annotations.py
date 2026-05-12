"""Strong interval annotations and onset-guided proposals for frame SED."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from sed.data.library import normalize_event_intervals


def propose_event_intervals(wav_path: Path, cfg: Any, label: str) -> list[dict[str, Any]]:
    """Propose one short event around the strongest transient.

    This deliberately reuses the transient locator used to audit augmentation
    masking.  The result is a *proposal* only: callers must show it to an
    annotator and keep ``confirmed=false`` until they explicitly save it.
    """
    from sed.data.augment import _locate_transient_frame, set_augment_features

    data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    wave = np.asarray(data.mean(axis=1), dtype=np.float32)
    duration_s = len(wave) / sample_rate if sample_rate else 0.0
    set_augment_features(cfg)
    frame = _locate_transient_frame(wave, int(sample_rate))
    hop_s = float(cfg.features.hop_length) / float(cfg.model.sample_rate)
    onset_s = max(0.0, frame * hop_s)
    # A glass impact often has a very short attack but rings out. This is a
    # conservative editable default, not an inferred offset claim.
    offset_s = min(duration_s, onset_s + min(0.18, max(0.06, duration_s * 0.35)))
    return normalize_event_intervals([{
        "label": label, "onset_s": onset_s, "offset_s": offset_s,
        "confirmed": False, "proposed_by": "onset_transient_locator",
    }], duration_s=duration_s)


def _dataset_rows(dataset_dir: Path | None) -> list[dict[str, Any]]:
    if dataset_dir is None:
        return []
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        csv_path = Path(dataset_dir) / f"{split}.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("path"):
                        rows.append({**row, "split": split})
        except Exception:  # noqa: BLE001
            continue
    return rows


def _confirmed_intervals(wav_path: Path) -> list[dict[str, Any]]:
    sidecar = wav_path.with_suffix(".json")
    if not sidecar.exists():
        return []
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    return [
        row for row in (raw.get("event_intervals") or [])
        if isinstance(row, dict) and bool(row.get("confirmed"))
    ]


def task_dataset_readiness(cfg: Any, task: str, dataset_dir: Path | None = None) -> dict[str, Any]:
    """Judge whether the composed dataset + feature window suit the task.

    Clip classification uses onset-centred windows and needs no strong labels.
    Frame-level SED trains only on *confirmed* onset/offset intervals and wants a
    long-enough window for temporal resolution — so it reports interval coverage
    and warns on short windows. Returns ``{task, stats, verdicts}`` for the UI.
    """
    task = str(task or "clip").strip().lower()
    window_seconds = float(getattr(cfg.features, "window_seconds", 1.0))

    rows = _dataset_rows(Path(dataset_dir) if dataset_dir is not None else None)
    split_counts = {split: 0 for split in ("train", "val", "test")}
    # Composed CSVs store absolute paths, so a string normpath is a stable key;
    # Path.resolve() per row costs minutes on a 75k-row dataset.
    paths: dict[str, Path] = {}
    for row in rows:
        split_counts[str(row.get("split") or "train")] = split_counts.get(str(row.get("split") or "train"), 0) + 1
        raw = str(row.get("path") or "")
        if raw:
            paths[os.path.normpath(raw)] = Path(os.path.normpath(raw))

    confirmed_clips = 0
    confirmed_positive_rows = 0
    frame_usable_train_rows = 0
    confirmed_labels: set[str] = set()
    confirmed_by_path: dict[str, list[dict[str, Any]]] = {}
    for key, wav in paths.items():
        confirmed = _confirmed_intervals(wav)
        confirmed_by_path[key] = confirmed
        if confirmed:
            confirmed_clips += 1
            for iv in confirmed:
                label = str(iv.get("label") or "")
                if label:
                    confirmed_labels.add(label)

    for row in rows:
        label = str(row.get("class_name") or "")
        split = str(row.get("split") or "")
        key = os.path.normpath(str(row.get("path") or "")) if row.get("path") else ""
        confirmed = confirmed_by_path.get(key, [])
        has_matching_interval = any(str(iv.get("label") or "") == label for iv in confirmed)
        if label != "_negative" and has_matching_interval:
            confirmed_positive_rows += 1
        if split == "train" and (label == "_negative" or has_matching_interval):
            frame_usable_train_rows += 1

    stats = {
        "total_clips": len(paths),
        "dataset_rows": len(rows),
        "dataset_unique_wavs": len(paths),
        "split_counts": split_counts,
        "clips_with_confirmed_intervals": confirmed_clips,
        "positive_rows_with_confirmed_intervals": confirmed_positive_rows,
        "frame_usable_train_rows": frame_usable_train_rows,
        "confirmed_labels": sorted(confirmed_labels),
        "window_seconds": window_seconds,
    }
    verdicts: list[dict[str, Any]] = []

    if not rows:
        verdicts.append({"level": "error", "text":
                         "No composed train/val/test rows yet — compose the dataset before training."})
        return {"task": task, "stats": stats, "verdicts": verdicts}

    if task == "frame":
        if window_seconds < 0.5:
            verdicts.append({"level": "warn", "text":
                f"Window is {window_seconds:.2f}s — short for frame-level SED. A longer "
                "window (≥1 s) gives the encoder more temporal cells to localize events."})
        if confirmed_positive_rows == 0:
            verdicts.append({"level": "error", "text":
                "Frame-level SED trains positive rows only when their WAV sidecar has a "
                "matching confirmed onset/offset interval. None of the composed positive "
                "rows are confirmed yet."})
        else:
            frac = confirmed_positive_rows / max(1, len(rows))
            level = "ok" if (confirmed_positive_rows >= 30 and frac >= 0.1) else "warn"
            verdicts.append({"level": level, "text":
                f"{confirmed_positive_rows}/{len(rows)} composed rows have matching "
                f"confirmed intervals across {len(confirmed_labels)} label(s). Frame "
                "SED skips weak positive rows and keeps negative rows as all-zero targets."})
    else:  # clip
        verdicts.append({"level": "ok", "text":
            f"{len(rows)} composed dataset rows ({len(paths)} unique WAVs) are available. "
            "Training loads those WAVs and computes features on the fly; preview images "
            "do not cap the dataset."})

    return {"task": task, "stats": stats, "verdicts": verdicts}


__all__ = ["propose_event_intervals", "task_dataset_readiness"]
