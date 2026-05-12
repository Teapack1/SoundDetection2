"""Pre-flight checklists. Each `doctor_*` function returns a list of
`Check` results; the CLI prints them as a colored table.

Intent: catch problems BEFORE the expensive step. A failed pre-record check
saves an hour of bad recordings; a failed pre-train check saves an hour of
GPU time; a failed pre-deploy check saves a service that crashes at 11 PM.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sed.config import Config, ProjectLayout, resolve_active_model_dir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    status: str               # "PASS" | "WARN" | "FAIL"
    detail: str = ""
    fix: str = ""             # actionable hint when status != PASS

    @property
    def emoji(self) -> str:
        return {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(self.status, "?")


def _ok(name: str, detail: str = "") -> Check:
    return Check(name, "PASS", detail)


def _warn(name: str, detail: str, fix: str = "") -> Check:
    return Check(name, "WARN", detail, fix)


def _fail(name: str, detail: str, fix: str = "") -> Check:
    return Check(name, "FAIL", detail, fix)


# ---------------------------------------------------------------------------
# 1. Pre-record
# ---------------------------------------------------------------------------


def doctor_record(cfg: Config, seconds: float = 5.0) -> list[Check]:
    """Verify the audio interface BEFORE you spend an hour recording.

    Tests:
      - the configured device exists and reports the expected channel count
      - sample-rate is supported
      - all channels actually carry signal (catches dead/unplugged mic)
      - per-channel RMS / peak in healthy ranges
      - no clipping
      - latency / xrun count over a short capture window
    """
    out: list[Check] = []

    # --- (a) device discovery ---
    try:
        from sed.audio.capture import list_input_devices, resolve_device
        devs = list_input_devices()
        if not devs:
            out.append(_fail("audio devices",
                             "no input devices found",
                             "plug the audio interface in; "
                             "check `arecord -l` (Linux) or your OS sound settings"))
            return out
        out.append(_ok("audio devices", f"{len(devs)} input device(s) available"))

        try:
            idx = resolve_device(cfg.deploy.record_device)
        except ValueError as e:
            out.append(_fail("device resolve",
                             str(e),
                             "check `[deploy] record_device` in projects/<id>/config.toml — "
                             f"available devices: {', '.join(d.name for d in devs[:5])}"))
            return out

        chosen = next((d for d in devs if d.index == idx), None)
        if chosen is None:
            out.append(_fail("device resolve", f"index {idx} not in input list"))
            return out
        out.append(_ok("device selected", f"[{chosen.index}] {chosen.name} "
                                          f"({chosen.max_input_channels}ch max)"))
        if cfg.deploy.record_device == "default":
            out.append(_warn(
                "device pinned",
                "`record_device` is `default`; OS default inputs can change",
                "set `[deploy] record_device` to a stable device name substring "
                "from `sed audio devices`",
            ))

        # Channel selection sanity. record_channels is an ordered list of
        # physical channel indexes; we must open at least max(index)+1 channels.
        req = list(cfg.audio.record_channels)
        n_open = max(req) + 1
        if n_open > chosen.max_input_channels:
            out.append(_fail(
                "channel selection",
                f"record_channels={req} needs {n_open} channel(s) but device has "
                f"{chosen.max_input_channels}",
                f"use channel indexes < {chosen.max_input_channels} in [audio] record_channels",
            ))
            return out
        else:
            out.append(_ok("channel selection",
                           f"record_channels={req} (opens {n_open}ch), "
                           f"available={chosen.max_input_channels}"))
        out.append(_ok(
            "sample rates",
            f"record={cfg.audio.record_sample_rate} Hz -> features={cfg.model.sample_rate} Hz",
        ))
    except ImportError as e:
        out.append(_fail("portaudio",
                         f"sounddevice import failed: {e}",
                         "install libasound2 + portaudio: "
                         "`apt install libportaudio2 libasound2`"))
        return out

    # --- (b) capture a few seconds and analyse ---
    import sounddevice as sd

    try:
        rec = sd.rec(
            int(seconds * cfg.audio.record_sample_rate),
            samplerate=cfg.audio.record_sample_rate,
            channels=n_open,
            dtype="float32",
            device=idx,
        )
        sd.wait()
    except Exception as e:  # noqa: BLE001
        out.append(_fail("capture",
                         f"sd.rec failed: {e}",
                         "check sample rate support: try 44100 or 48000 instead"))
        return out

    out.append(_ok("capture", f"{seconds:.0f}s recorded successfully"))

    # --- (c) per-channel signal analysis (only the recorded channels) ---
    for c in req:
        x = rec[:, c]
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        peak = float(np.max(np.abs(x)))
        rms_db = 20 * np.log10(rms + 1e-12)
        peak_db = 20 * np.log10(peak + 1e-12)

        # Heuristics tuned for room-mic recording
        if peak < 1e-3:
            out.append(_fail(
                f"mic ch{c}",
                f"essentially silent (peak {peak_db:.0f} dBFS)",
                "mic unplugged / muted / phantom power off / wrong input channel",
            ))
        elif peak > 0.99:
            out.append(_fail(
                f"mic ch{c}",
                f"clipping (peak {peak_db:.1f} dBFS)",
                "lower preamp gain on the interface; aim for peaks at -10 to -3 dBFS",
            ))
        elif peak > 0.7:
            out.append(_warn(
                f"mic ch{c}",
                f"peaks high (peak {peak_db:.1f} dBFS, RMS {rms_db:.1f} dBFS)",
                "consider lowering preamp gain by 3-6 dB to leave headroom",
            ))
        elif peak < 0.02:
            out.append(_warn(
                f"mic ch{c}",
                f"very quiet (peak {peak_db:.1f} dBFS, RMS {rms_db:.1f} dBFS)",
                "raise preamp gain by 6-12 dB; aim for typical-source peaks at -20 to -10 dBFS",
            ))
        else:
            out.append(_ok(
                f"mic ch{c}",
                f"peak {peak_db:.1f} dBFS · RMS {rms_db:.1f} dBFS — healthy",
            ))

    # --- (d) DC offset ---
    for c in req:
        dc = float(rec[:, c].mean())
        if abs(dc) > 0.005:
            out.append(_warn(
                f"DC offset ch{c}",
                f"DC = {dc:+.4f} (≈{20*np.log10(abs(dc)+1e-12):.0f} dBFS)",
                "interface or DAC issue; usually harmless, but if persistent, "
                "enable a high-pass filter on your interface (typ. 80 Hz)",
            ))

    # --- (e) inter-channel correlation: catches "all channels are actually 1 mic" ---
    if len(req) >= 2:
        base = req[0]
        for c2 in req[1:]:
            corr = float(np.corrcoef(rec[:, base], rec[:, c2])[0, 1])
            if abs(corr) > 0.98:
                out.append(_warn(
                    f"corr ch{base}↔ch{c2}",
                    f"channels appear identical (r={corr:.3f})",
                    "either the same mic is wired to multiple inputs, "
                    "or the device is summing channels — check interface routing",
                ))
            else:
                out.append(_ok(
                    f"corr ch{base}↔ch{c2}",
                    f"r={corr:+.3f} — independent signals",
                ))

    return out


# ---------------------------------------------------------------------------
# 2. Pre-train
# ---------------------------------------------------------------------------


def doctor_train(cfg: Config, project_id: str) -> list[Check]:
    """Sanity-check the dataset and training config BEFORE training."""
    out: list[Check] = []
    layout = ProjectLayout(project_id)

    if not layout.dataset_dir.exists():
        out.append(_fail("dataset", f"no dataset/ at {layout.dataset_dir}",
                         "run `sed dataset compose --project " + project_id + "`"))
        return out

    # --- (a) split files exist and parse ---
    # train.csv is the whole train+val pool; val is carved out live at train
    # time (seeded by cfg.training.seed), so preview that same split here
    # rather than requiring a persisted val.csv.
    import csv as csvlib
    from sed.data.splits import split_pool_rows
    pool_rows: dict[str, list[dict]] = {}
    for name in ("train", "test"):
        p = layout.dataset_dir / f"{name}.csv"
        if not p.exists():
            out.append(_fail(f"{name}.csv", f"missing {p}",
                             "run `sed dataset compose`"))
            return out
        with open(p, "r", encoding="utf-8") as f:
            pool_rows[name] = list(csvlib.DictReader(f))
    val_fraction = float(getattr(cfg.dataset, "val_fraction", 0.15) or 0.15)
    train_rows, val_rows, _modes = split_pool_rows(
        pool_rows["train"], val_frac=val_fraction, seed=cfg.training.seed,
    )
    splits: dict[str, list[dict]] = {"train": train_rows, "val": val_rows, "test": pool_rows["test"]}
    test_note = " (frozen)" if len(splits["test"]) else " (empty — no library test clips)"
    out.append(_ok("split files", f"train+val pool={len(pool_rows['train'])} "
                                  f"(preview: train={len(train_rows)} val={len(val_rows)}) "
                                  f"test={len(splits['test'])}{test_note}"))

    # --- (b) classes.json + agreement with config ---
    classes_path = layout.dataset_dir / "classes.json"
    if not classes_path.exists():
        out.append(_fail("classes.json", "missing", "run `sed dataset compose`"))
        return out
    with open(classes_path, "r", encoding="utf-8") as f:
        ds_classes = json.load(f)["classes"]

    if ds_classes != cfg.model.classes:
        out.append(_warn(
            "class agreement",
            f"dataset has {ds_classes} but projects/<id>/config.toml says {cfg.model.classes}",
            "re-run `sed dataset compose` after editing projects/<id>/config.toml",
        ))
    else:
        out.append(_ok("class agreement", f"{len(ds_classes)} classes"))

    # --- (c) per-class counts (warn if any class has < 50 in train) ---
    train_counts: dict[str, int] = {c: 0 for c in ds_classes}
    for r in splits["train"]:
        train_counts[r["class_name"]] = train_counts.get(r["class_name"], 0) + 1
    for cls, n in train_counts.items():
        if n == 0:
            # Declared in [dataset] target_classes but no recordings yet → training will
            # fail because the classifier head expects N classes. Either record
            # some clips for this class, or drop it from projects/<id>/config.toml until you do.
            out.append(_fail(
                f"count {cls}",
                "no training samples (class is declared but empty)",
                f"record with `sed record --class {cls}` "
                f"OR remove '{cls}' from [dataset] target_classes in projects/<id>/config.toml "
                "until you're ready",
            ))
        elif n < 50:
            out.append(_warn(f"count {cls}", f"{n} train samples",
                             "aim for ≥200; recall will likely suffer"))
        else:
            out.append(_ok(f"count {cls}", f"{n} train samples"))

    # --- (d) val must contain every class for meaningful metrics. The test
    # split is the frozen one and is covered by its own
    # row below — no need to re-check coverage here. ---
    val_present = {r["class_name"] for r in splits["val"]}
    val_missing = set(ds_classes) - val_present
    if val_missing:
        out.append(_warn(
            "val coverage",
            f"missing classes: {sorted(val_missing)}",
            f"record more clips of {sorted(val_missing)} so the splitter can "
            "reserve some for val (need ≥ 2-3 clips per class).",
        ))
    else:
        out.append(_ok("val coverage", "all classes present"))

    # --- (e) negative ratio sanity ---
    has_negative_class = "_negative" in ds_classes
    neg = train_counts.get("_negative", 0)
    pos = sum(n for c, n in train_counts.items() if c != "_negative")
    if not has_negative_class:
        out.append(_ok("pos:neg ratio", "_negative class not configured"))
    elif pos == 0:
        out.append(_warn(
            "pos:neg ratio",
            "no positives in train",
            "early-state OK while you bootstrap; record target clips with "
            "`sed record --class <label>` before training",
        ))
    else:
        ratio = neg / max(pos, 1)
        if ratio < 1.0:
            out.append(_warn(
                "pos:neg ratio",
                f"only {ratio:.1f}x more negatives than positives",
                "aim for 3-5x; record more negative/background clips "
                "to cut FP rate in production",
            ))
        else:
            out.append(_ok("pos:neg ratio",
                           f"{ratio:.1f}x more negatives than positives"))

    # --- (e1) frozen project test set ---
    if splits["test"]:
        per_class: dict[str, int] = {c: 0 for c in ds_classes}
        for row in splits["test"]:
            cls = row.get("class_name", "")
            per_class[cls] = per_class.get(cls, 0) + 1
        total_test = sum(per_class.values())
        missing = [c for c, n in per_class.items() if n == 0]
        summary = " ".join(f"{c}={n}" for c, n in per_class.items())
        if missing:
            out.append(_warn(
                "frozen test set",
                f"{total_test} clips — missing classes: {missing}",
                f"record clips for {missing} with `sed record --project {project_id} "
                "--test --class <name>` so every class has held-out test material.",
            ))
        elif total_test < 50:
            out.append(_warn(
                "frozen test set",
                f"only {total_test} clips ({summary}) — metrics will be noisy",
                "record/import more with `sed record --test`; aim for ≥30 per class.",
            ))
        else:
            out.append(_ok("frozen test set", f"{total_test} clips ({summary})"))
    else:
        out.append(_warn(
            "frozen test set",
            "no library test clips — post-training evaluation will be skipped",
            "without a frozen test set you cannot compare model iterations "
            f"fairly. Record one with `sed record --project {project_id} "
            "--test --class <label>`.",
        ))

    # --- (e2) offline augmentation must have produced WAVs if enabled ---
    aug_cfg = cfg.augmentation
    any_enabled = aug_cfg.enabled and aug_cfg.overlay_enabled
    from sed.config import LIBRARY_DIR
    aug_dir = LIBRARY_DIR / "derived" / project_id / "trainval"
    if any_enabled:
        if not aug_dir.exists() or not any(aug_dir.rglob("*.wav")):
            out.append(_warn(
                "augmented samples",
                f"none under {aug_dir}",
                "run `sed dataset augment --project <id>` before training "
                "so the augmented WAVs actually land in train.csv (otherwise "
                "training sees only the raw recordings).",
            ))
        else:
            n_aug = sum(1 for _ in aug_dir.rglob("*.wav"))
            out.append(_ok("augmented samples", f"{n_aug} files under {aug_dir.name}/"))

    # --- (f) negative/background pool: single consolidated row.
    if has_negative_class:
        neg_summary = _summarise_negatives(layout, cfg)
        if neg_summary["n_clips"] == 0:
            out.append(_warn(
                "negatives",
                "no negative/background clips found in library mappings",
                "record/import sounds and assign labels to _negative.",
            ))
        else:
            labels = ", ".join(sorted(neg_summary["labels"])[:5]) or "unlabeled"
            extra = "" if len(neg_summary["labels"]) <= 5 else f" +{len(neg_summary['labels']) - 5}"
            density = " ".join(
                f"{d}={neg_summary['by_density'].get(d, 0.0):.1f}m"
                for d in ("empty", "quiet", "normal", "busy", "peak", "unknown")
                if neg_summary['by_density'].get(d, 0.0) > 0
            )
            detail = (
                f"{neg_summary['n_clips']} clips · {len(neg_summary['labels'])} labels "
                f"({labels}{extra}) · {neg_summary['total_min']:.1f} min"
                + (f" · {density}" if density else "")
            )
            if neg_summary["n_clips"] < 100 or len(neg_summary["labels"]) < 2:
                out.append(_warn(
                    "negatives",
                    detail,
                    "add more diverse negative labels/clips; aim for at least "
                    "100 clips across 2+ distractor/background labels.",
                ))
            else:
                out.append(_ok("negatives", detail))

    # Splits are random clip-level (per class). There is no session-aware
    # splitting any more, so no leakage check is needed.

    return out


def _summarise_negatives(layout: ProjectLayout, cfg: Config | None = None) -> dict:
    """Roll up every negative/background clip into duration + density.

    The reader looks at accepted library clips mapped to ``_negative``.
    """
    by_density: dict[str, float] = {}
    labels: set[str] = set()
    n_clips = 0
    total_seconds = 0.0

    if cfg is not None:
        try:
            from sed.data.compose import _library_clips_for_class, load_dataset_selection
            selection = load_dataset_selection(layout)
            for clip in _library_clips_for_class(
                cfg, layout.id, "_negative", split="train", selection=selection
            ):
                meta = {}
                try:
                    meta = json.loads(Path(clip.path).with_suffix(".json").read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    pass
                density = meta.get("density") or "unknown"
                label = meta.get("label") or Path(clip.path).parent.name
                labels.add(label)
                n_clips += 1
                by_density[density] = by_density.get(density, 0.0) + clip.duration_s
                total_seconds += clip.duration_s
            if n_clips == 0:
                from sed.data.library import RAW_CATEGORY, Library
                for c in Library().query(category=RAW_CATEGORY, review_status="accepted"):
                    density = c.density or "unknown"
                    labels.add(c.label)
                    n_clips += 1
                    by_density[density] = by_density.get(density, 0.0) + c.duration_s
                    total_seconds += c.duration_s
        except Exception:  # noqa: BLE001
            pass
    else:
        try:
            from sed.data.library import RAW_CATEGORY, Library
            for c in Library().query(category=RAW_CATEGORY, review_status="accepted"):
                density = c.density or "unknown"
                labels.add(c.label)
                n_clips += 1
                by_density[density] = by_density.get(density, 0.0) + c.duration_s
                total_seconds += c.duration_s
        except Exception:  # noqa: BLE001
            pass

    return {
        "total_min": total_seconds / 60.0,
        "by_density": {d: s / 60.0 for d, s in by_density.items()},
        "labels": labels,
        "n_clips": n_clips,
    }


# ---------------------------------------------------------------------------
# Recommended hyperparameters (advisory, never auto-applied)
# ---------------------------------------------------------------------------


@dataclass
class Recommendation:
    signal: str        # what the recommendation is based on
    suggestion: str    # what to change
    current: str = ""  # current value(s) for context


def recommend_training_hyperparams(cfg: Config, project_id: str) -> list[Recommendation]:
    """Heuristic hyperparameter advice derived from dataset stats.

    Returns rows for `sed check train` to render. The advice is intentionally
    coarse — each rule is an explicit cutoff visible in this file. Nothing here
    modifies projects/<id>/config.toml; the user decides what to apply.
    """
    out: list[Recommendation] = []
    layout = ProjectLayout(project_id)
    if not layout.dataset_dir.exists():
        return out

    import csv as _csv
    from sed.data.splits import split_pool_rows
    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    totals: dict[str, int] = {}
    pool_rows: dict[str, list[dict]] = {}
    for split in ("train", "test"):
        p = layout.dataset_dir / f"{split}.csv"
        if not p.exists():
            return out
        with open(p, "r", encoding="utf-8") as f:
            pool_rows[split] = list(_csv.DictReader(f))
    # train.csv is the whole train+val pool; preview the same live split
    # train time uses (seeded by cfg.training.seed) rather than requiring a
    # persisted val.csv.
    val_fraction = float(getattr(cfg.dataset, "val_fraction", 0.15) or 0.15)
    train_rows, val_rows, _modes = split_pool_rows(
        pool_rows["train"], val_frac=val_fraction, seed=cfg.training.seed,
    )
    rows_by_split = {"train": train_rows, "val": val_rows, "test": pool_rows["test"]}
    for split, rows in rows_by_split.items():
        for r in rows:
            cls = r["class_name"]
            counts[split][cls] = counts[split].get(cls, 0) + 1
        totals[split] = len(rows)

    train_total = totals["train"]
    if train_total < 500:
        out.append(Recommendation(
            f"train_total={train_total} (<500)",
            "patience: 5, epochs: 30 — small dataset, train short",
            f"patience={cfg.training.patience}, epochs={cfg.training.epochs}",
        ))
    elif train_total < 3000:
        out.append(Recommendation(
            f"train_total={train_total} (500-3000)",
            "patience: 8, epochs: 60 — moderate dataset",
            f"patience={cfg.training.patience}, epochs={cfg.training.epochs}",
        ))
    else:
        out.append(Recommendation(
            f"train_total={train_total} (≥3000)",
            "patience: 12, epochs: 80 — larger dataset can train longer",
            f"patience={cfg.training.patience}, epochs={cfg.training.epochs}",
        ))

    present_train = {name: n for name, n in counts["train"].items() if n > 0}
    if len(present_train) >= 2:
        target_counts = [n for name, n in present_train.items() if name != "_negative"]
        negative_count = present_train.get("_negative", 0)
        ratios: list[float] = []
        if negative_count and target_counts:
            target_total = sum(target_counts)
            ratios.append(max(negative_count, target_total) / min(negative_count, target_total))
        if len(target_counts) >= 2:
            ratios.append(max(target_counts) / min(target_counts))
        if not negative_count:
            ratios.append(max(present_train.values()) / min(present_train.values()))
        count_ratio = max(ratios, default=1.0)
        suggested_balance = "off" if count_ratio <= 1.25 else "sqrt"
        if cfg.training.class_balance != suggested_balance:
            rationale = (
                "natural row frequency is appropriate because background/target and "
                "target-class counts are within 1.25x"
                if suggested_balance == "off"
                else "temper class-frequency skew once in the group-aware sampler"
            )
            out.append(Recommendation(
                f"class count ratio max:min = {count_ratio:.2f}:1",
                f'training.class_balance = "{suggested_balance}" — {rationale}',
                f"class_balance={cfg.training.class_balance}",
            ))

    nonneg_train = {c: n for c, n in counts["train"].items() if c != "_negative" and n > 0}
    if len(nonneg_train) >= 2:
        nmin, nmax = min(nonneg_train.values()), max(nonneg_train.values())
        imbalance = nmin / nmax if nmax else 1.0
        if imbalance < 0.3:
            weakest = min(nonneg_train, key=nonneg_train.get)
            out.append(Recommendation(
                f"class imbalance: min/max = {imbalance:.2f} ('{weakest}' = {nmin})",
                f"record more of '{weakest}' so train/val/test represent the real venue",
                "",
            ))

    if totals["val"] < 100:
        out.append(Recommendation(
            f"val rows = {totals['val']} (<100)",
            "val metric will be noisy — record more, then `sed dataset compose` again",
            "",
        ))
    # Only recommend more test data when SOME test rows exist; absence is
    # already covered by the "frozen test set" WARN row above, no need to
    # repeat the same advice in the Recommendations section.
    if 0 < totals["test"] < 100:
        out.append(Recommendation(
            f"test rows = {totals['test']} (<100)",
            "test signal is weak — record more frozen test material with "
            "`sed record --project <id> --test --class <label>`",
            "",
        ))

    return out


# ---------------------------------------------------------------------------
# 3. Pre-deploy
# ---------------------------------------------------------------------------


def doctor_deploy(cfg: Config, project_id: str) -> list[Check]:
    """Final checks before leaving the system running 24/7."""
    out: list[Check] = []
    layout = ProjectLayout(project_id)

    # --- (a) deployment slot: training_config.json + classes.json + an IR ---
    try:
        model_dir = resolve_active_model_dir(layout, cfg.deploy.active_model)
    except FileNotFoundError as e:
        out.append(_fail(
            "deployment slot", str(e),
            "train a model (`sed train`) or copy a bundle into models/, then "
            "`sed models select --project " + project_id + " <model_id>`.",
        ))
        return out
    out.append(_ok("deployment slot",
                   f"{model_dir.name} classes={list(cfg.model.classes)}"))
    deployed_classes = list(cfg.model.classes)

    int8 = model_dir / "model_int8.xml"
    fp16 = model_dir / "model_fp16.xml"
    if not int8.exists() and not fp16.exists():
        out.append(_fail("model files", "no .xml found",
                         f"re-train or re-copy the model bundle for {project_id}"))
        return out
    out.append(_ok("model files",
                   f"INT8={'✓' if int8.exists() else '✗'} "
                   f"FP16={'✓' if fp16.exists() else '✗'}"))

    # --- (b) thresholds calibrated? ---
    calibrated = [c for c in deployed_classes if c != "_negative" and c in cfg.thresholds]
    needed = [c for c in deployed_classes if c != "_negative"]
    if not calibrated:
        out.append(_warn(
            "thresholds",
            "no per-class thresholds set; using defaults",
            "run `sed calibrate --project " + project_id + "` against negatives",
        ))
    elif set(calibrated) < set(needed):
        out.append(_warn(
            "thresholds",
            f"some classes uncalibrated: {set(needed) - set(calibrated)}",
            "run `sed calibrate` again",
        ))
    else:
        out.append(_ok("thresholds",
                       " ".join(f"{c}={cfg.thresholds[c]:.2f}" for c in calibrated)))

    # --- (c) load model & measure latency ---
    try:
        import openvino as ov
        from sed.runtime.infer import compile_openvino_model
        from sed.runtime.manifest import resolve_model_path
        core = ov.Core()
        b = max(1, len(cfg.inference.inference_channels))
        _model, cm = compile_openvino_model(
            core,
            resolve_model_path(model_dir),
            cfg,
            batch_size=b,
            config={"PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_NUM_THREADS": str(cfg.inference.inference_threads)},
        )
        req = cm.create_infer_request()
        from sed.audio.features import (
            feature_band_count,
            feature_channels,
            feature_frame_count_for_cfg,
        )

        n_mels = feature_band_count(cfg.features, cfg.model.sample_rate)
        n_frames = feature_frame_count_for_cfg(cfg)
        n_ch = feature_channels(cfg.features)
        x = np.random.randn(b, n_ch, n_mels, n_frames).astype(np.float32)
        # warm
        for _ in range(10):
            req.infer({0: x})
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            req.infer({0: x})
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        p50, p99 = times[25], times[49]
        budget_ms = cfg.inference.hop_ms
        if p99 > budget_ms:
            out.append(_warn(
                "inference latency",
                f"p50={p50:.1f}ms p99={p99:.1f}ms (batch={b}) — exceeds "
                f"hop {budget_ms}ms",
                f"reduce inference_threads, switch to mn01_as backbone, "
                f"or increase [inference] hop_ms above {p99:.0f}",
            ))
        else:
            out.append(_ok(
                "inference latency",
                f"p50={p50:.1f}ms p99={p99:.1f}ms (batch={b}, budget {budget_ms}ms)",
            ))
    except Exception as e:  # noqa: BLE001
        out.append(_fail("model load", str(e),
                         "check OpenVINO install and that the model exported cleanly"))

    # --- (d) ArtNet target reachable? ---
    if cfg.artnet.enabled:
        target = cfg.artnet.target_ip.strip()
        if not target:
            out.append(_fail("artnet target", "target_ip is not set",
                             "set [artnet] target_ip to the receiver's unicast IP"))
        elif target.endswith(".255") or target == "255.255.255.255":
            out.append(_fail("artnet target",
                             f"{target} is a broadcast address",
                             "broadcasting is disabled by design; set the "
                             "receiver's specific unicast IP"))
        else:
            try:
                socket.inet_aton(target)
                # quick UDP probe to ArtNet port (we don't expect a reply)
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(0.2)
                try:
                    s.sendto(b"\x41\x72\x74\x2d\x4e\x65\x74\x00", (target, 6454))
                    out.append(_ok("artnet target", f"{target}:6454 reachable"))
                except OSError as e:
                    out.append(_warn("artnet target",
                                     f"{target}:6454 unreachable: {e}",
                                     "check the network cable and switch"))
                finally:
                    s.close()
            except OSError:
                out.append(_fail("artnet target",
                                 f"invalid IP: {target}",
                                 "set [artnet] target_ip to a valid IPv4 address"))
    else:
        out.append(_ok("artnet", "disabled in config"))

    # --- (e) systemd unit ready? ---
    unit = Path("/etc/systemd/system/sed.service")
    if unit.exists():
        out.append(_ok("systemd unit", f"{unit} present"))
    else:
        out.append(_warn("systemd unit", "/etc/systemd/system/sed.service missing",
                         "run `sudo PROJECT_ID=" + project_id + " ./deploy/install.sh`"))

    # --- (f) writable output paths ---
    # Only models/ matters at deploy time (the runtime writes nothing else).
    # Inspect/training write into dataset/, which the edge doesn't touch.
    p = layout.models_dir
    try:
        test_file = p / ".doctor_write_test"
        test_file.touch()
        test_file.unlink()
        out.append(_ok(f"writable {p.name}", str(p)))
    except OSError as e:
        out.append(_fail(f"writable {p.name}", str(e),
                         f"chown -R sed:sed {p}"))

    return out


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------


def render(
    checks: list[Check],
    title: str,
    recommendations: "list[Recommendation] | None" = None,
) -> None:
    """Print the check table, then optional fix bullets and recommendations.

    The output follows one consistent visual language:
      • table with status/detail
      • bullets for non-PASS rows under "Suggested fixes"
      • bullets for advisory rows under "Recommendations" (advisory; informational)
      • one-line summary footer (PASS/WARN/FAIL counts)
    """
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title=title, header_style="bold")
    table.add_column("", width=2)
    table.add_column("check", style="bold")
    table.add_column("status")
    table.add_column("detail")
    for c in checks:
        style = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[c.status]
        table.add_row(c.emoji, c.name, f"[{style}]{c.status}[/{style}]", c.detail)
    console.print(table)
    # Show fix advice for any non-PASS
    bad = [c for c in checks if c.status != "PASS"]
    if bad:
        console.print("\n[bold]Suggested fixes:[/bold]")
        for c in bad:
            if c.fix:
                console.print(f"  [{('red' if c.status == 'FAIL' else 'yellow')}]"
                              f"• {c.name}[/]: {c.fix}")
    if recommendations:
        console.print("\n[bold]Recommendations:[/bold]"
                      "  [dim](heuristics from dataset stats — review and apply manually)[/dim]")
        for r in recommendations:
            console.print(f"  [cyan]•[/cyan] [bold]{r.signal}[/bold] → {r.suggestion}"
                          + (f"  [dim](current: {r.current})[/dim]" if r.current else ""))
    fail_count = sum(1 for c in checks if c.status == "FAIL")
    warn_count = sum(1 for c in checks if c.status == "WARN")
    if fail_count:
        console.print(f"\n[red bold]{fail_count} FAIL[/red bold] · "
                      f"[yellow]{warn_count} WARN[/yellow] — "
                      f"address FAILs before proceeding.")
    elif warn_count:
        console.print(f"\n[yellow bold]{warn_count} WARN[/yellow bold] — "
                      f"safe to proceed but check the warnings.")
    else:
        console.print(f"\n[green bold]all {len(checks)} checks passed[/green bold]")


def all_failed(checks: list[Check]) -> bool:
    return any(c.status == "FAIL" for c in checks)
