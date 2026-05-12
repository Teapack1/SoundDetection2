"""Test-set evaluation for a trained model.

Used in two places:
  1. Inside `sed train` when `[training] auto_test` is on — runs against
     the newly trained model's frozen dataset snapshot and writes
     test_metrics.json + confusion_matrix.{png,json} into the model's reports/ dir.
  2. `sed evaluate run --project <id> [model_id]` — re-scores an existing
     model against the current project frozen test set. Omitting model_id uses
     the active project model.

Confusion matrix derivation
---------------------------
Models trained with cross entropy are evaluated with softmax probabilities;
legacy BCE/focal runs keep sigmoid probabilities. If `_negative` is configured,
the matrix uses the same unsmoothed runtime gates as production. Target-only
runs are evaluated as closed-set argmax classification without no-trigger
threshold wording.
"""

from __future__ import annotations

import csv
import json
import logging
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sed.config import Config, ProjectLayout, load_config_snapshot
from sed.train.datamodule import SedDataset
from sed.train.lit_module import SedLitModule, load_classes
from sed.train.objective import (
    logits_to_probabilities,
    probability_mode,
    targets_to_one_hot_np,
)

# The decision replay and the metric math live in a torch-free module so the web
# UI can recompute them on demand from stored scores. Importing them here (under
# the historical private names) keeps this module's call sites and its tests
# unchanged, and guarantees a number recomputed in the browser is produced by
# the same code that wrote the manifest.
from sed.train.metrics import (  # noqa: F401
    average_precision as _average_precision,
    closed_set_decisions as _closed_set_decisions,
    confusion_matrix as _confusion_matrix,
    cross_target_confusion,
    decision_metrics as _decision_metrics,
    fit_thresholds_for_precision,
    fit_thresholds_from_val,
    operating_point_block as _operating_point_block,
    runtime_decisions as _runtime_decisions,
    runtime_pred_indices as _runtime_pred_indices,
)

# Val-fit calibration modes (calibrate_run_thresholds). "val_f1" maximizes each
# class's F1; "val_precision" picks the lowest threshold whose val precision
# reaches a shared target, giving every model the same operating point.
CALIBRATION_MODES: tuple[str, ...] = ("val_f1", "val_precision")

# The decision-rule vocabulary and the stored-score readers live in a
# torch-free module so the web UI can read them without paying for the training
# runtime; re-exported here so `from sed.train.eval import TestRule` keeps working.
from sed.train.test_rules import (  # noqa: F401
    DEPLOYED_TEST_SECTION,
    EXPORTED_RULE_KINDS,
    FROZEN_TEST_SECTION,
    RULE_KIND_INFO,
    RULE_KINDS,
    RULE_PARAM_DEFAULTS,
    TestRule,
    false_fire_rate,
    false_per_1000_negatives,
    macro_targets_from_per_class,
    parse_rule_id,
    read_deployed_test_evals,
    read_frozen_test_evals,
    rule_catalog,
)

log = logging.getLogger(__name__)

EvaluationProgress = Callable[[dict[str, object]], None]


def _report_evaluation_progress(
    callback: EvaluationProgress | None,
    **update: object,
) -> None:
    """Best-effort progress reporting; telemetry must not break evaluation."""
    if callback is None:
        return
    try:
        callback(update)
    except Exception:  # noqa: BLE001
        log.debug("evaluation progress callback failed", exc_info=True)


def _collect_predictions(
    cfg: Config,
    project_id: str,
    ckpt_path: Path,
    dataset_dir: Path,
    classes: list[str],
    csv_name: str = "test.csv",
    *,
    progress: EvaluationProgress | None = None,
    progress_phase: str = "scoring frozen test",
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    """Run one dataset CSV through the checkpoint once.

    Returns:
        probs: (N, C) objective-aware probabilities
        targets: (N, C) one-hot ground truth
        classes: class names in order
        rows: csv rows in dataloader order
    """
    _report_evaluation_progress(
        progress, phase="loading model checkpoint", completed=0, total=0,
        current_name="", current_class="",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"You are using `torch.load` with `weights_only=False`.*",
        )
        lit = SedLitModule.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, classes=classes, map_location="cpu"
        )
    lit.eval()

    ds = SedDataset(dataset_dir / csv_name, classes, cfg, layout=ProjectLayout(project_id))
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    probs_chunks: list[np.ndarray] = []
    targets_chunks: list[np.ndarray] = []
    total = len(ds)
    if progress is not None:
        first = ds.rows[0] if ds.rows else {}
        _report_evaluation_progress(
            progress,
            phase=progress_phase,
            completed=0,
            total=total,
            current_name=Path(str(first.get("path") or "")).name,
            current_path=str(first.get("path") or ""),
            current_class=str(first.get("class_name") or ""),
        )
    completed = 0
    with torch.no_grad():
        for x, y in loader:
            if progress is not None:
                current = ds.rows[completed] if completed < total else {}
                _report_evaluation_progress(
                    progress,
                    phase=progress_phase,
                    completed=completed,
                    total=total,
                    current_name=Path(str(current.get("path") or "")).name,
                    current_path=str(current.get("path") or ""),
                    current_class=str(current.get("class_name") or ""),
                )
            logits = lit(x)
            probs_chunks.append(logits_to_probabilities(logits, cfg).cpu().numpy())
            targets_chunks.append(y.cpu().numpy())
            completed = min(total, completed + int(y.shape[0]))
            if progress is not None:
                last = ds.rows[completed - 1] if completed else {}
                _report_evaluation_progress(
                    progress,
                    phase=progress_phase,
                    completed=completed,
                    total=total,
                    current_name=Path(str(last.get("path") or "")).name,
                    current_path=str(last.get("path") or ""),
                    current_class=str(last.get("class_name") or ""),
                )
    if not probs_chunks:
        return np.zeros((0, len(classes))), np.zeros((0, len(classes))), classes, []
    return (
        np.concatenate(probs_chunks, axis=0),
        targets_to_one_hot_np(np.concatenate(targets_chunks, axis=0), len(classes)),
        classes,
        list(ds.rows),
    )


def _collect_predictions_openvino(
    cfg: Config,
    project_id: str,
    model_path: Path,
    dataset_dir: Path,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    """Run the test loader through an exported OpenVINO IR once.

    Same contract as ``_collect_predictions``, but scores the deployed
    artifact (typically the INT8 IR) instead of the torch checkpoint.
    """
    import openvino as ov
    from sed.runtime.infer import compile_openvino_model, normalize_openvino_logits

    core = ov.Core()
    _model, compiled = compile_openvino_model(
        core,
        model_path,
        cfg,
        batch_size=1,
    )
    req = compiled.create_infer_request()

    ds = SedDataset(dataset_dir / "test.csv", classes, cfg, layout=ProjectLayout(project_id))
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    probs_chunks: list[np.ndarray] = []
    targets_chunks: list[np.ndarray] = []
    for x, y in loader:
        batch = x.numpy().astype(np.float32)
        logits = normalize_openvino_logits(
            list(req.infer({0: batch}).values())[0],
            batch_size=batch.shape[0],
            num_classes=len(classes),
        )
        probs_chunks.append(np.asarray(logits_to_probabilities(logits, cfg)))
        targets_chunks.append(y.cpu().numpy())
    if not probs_chunks:
        return np.zeros((0, len(classes))), np.zeros((0, len(classes))), classes, []
    return (
        np.concatenate(probs_chunks, axis=0),
        targets_to_one_hot_np(np.concatenate(targets_chunks, axis=0), len(classes)),
        classes,
        list(ds.rows),
    )




def _configured_thresholds(
    cfg: Config,
    classes: list[str],
    override_threshold: float | None = None,
) -> dict[str, float]:
    """Evaluation thresholds per class.

    Positive classes use [targets].<class>.threshold, falling back to legacy
    [thresholds].<class> then [inference].default_threshold (Config.class_threshold).
    `_negative` uses [inference].negative_max because that is the runtime
    suppression gate. An explicit `override_threshold` wins for positive classes.
    """
    thresholds: dict[str, float] = {}
    for cls in classes:
        if cls == "_negative":
            thresholds[cls] = float(cfg.inference.negative_max)
        elif override_threshold is not None:
            thresholds[cls] = float(override_threshold)
        else:
            thresholds[cls] = float(cfg.class_threshold(cls))
    return thresholds


def fit_run_thresholds(
    cfg: Config,
    project_id: str,
    model_id: str,
    *,
    mode: str = "val_f1",
    target_precision: float = 0.90,
    progress: EvaluationProgress | None = None,
) -> tuple[dict[str, float], int] | None:
    """Fit per-class thresholds on the run's frozen val split — without saving.

    The pure fit, split out of `calibrate_run_thresholds` so that *scoring* the
    frozen test under a val-fitted rule can never change what the bundle
    deploys. Returns (thresholds, n_val_clips), or None when the fit does not
    apply (no `_negative` class / no val split).
    """
    from sed.train.registry import get_checkpoint

    if mode not in CALIBRATION_MODES:
        raise ValueError(
            f"unknown calibration mode: {mode!r} (expected one of {CALIBRATION_MODES})"
        )
    layout = ProjectLayout(project_id)
    model_dir = layout.model_dir(model_id)
    run_cfg = _model_config(model_dir, cfg)
    classes = _load_model_classes(model_dir)
    if "_negative" not in classes:
        log.info("threshold fit skipped: closed-set model (no _negative)")
        return None
    val_csv = model_dir / "dataset" / "val.csv"
    if not val_csv.exists() or _csv_row_count(val_csv) == 0:
        log.warning("threshold fit skipped: no val split in %s", model_dir / "dataset")
        return None

    ckpt_path = get_checkpoint(project_id, model_id)
    probs, targets, classes, rows = _collect_predictions(
        run_cfg, project_id, ckpt_path, model_dir / "dataset", classes,
        csv_name="val.csv",
        progress=progress,
        progress_phase="fitting decision thresholds on validation",
    )
    # The val split has now been scored anyway; storing it costs nothing extra
    # and is what lets Runs & compare show the val→test gap at whatever
    # confidence the user picks, and re-fit thresholds without another pass.
    _store_scores(
        model_dir,
        engine="checkpoint",
        probs=probs,
        targets=targets,
        classes=classes,
        rows=rows,
        metrics={"test_set_hash": _val_split_hash(model_dir)},
        run_cfg=run_cfg,
        split="val",
    )
    _report_evaluation_progress(
        progress,
        phase="calculating validation thresholds",
        completed=int(probs.shape[0]),
        total=int(probs.shape[0]),
        current_name="",
        current_class="",
    )
    base = _configured_thresholds(run_cfg, classes)
    if mode == "val_precision":
        fitted = fit_thresholds_for_precision(
            probs, targets, classes,
            target_precision=target_precision,
            negative_max=run_cfg.inference.negative_max,
            margin_to_second_best=run_cfg.inference.margin_to_second_best,
            base_thresholds=base,
        )
    else:
        fitted = fit_thresholds_from_val(
            probs, targets, classes,
            negative_max=run_cfg.inference.negative_max,
            margin_to_second_best=run_cfg.inference.margin_to_second_best,
            base_thresholds=base,
        )
    return fitted, int(probs.shape[0])


def calibrate_run_thresholds(
    cfg: Config,
    project_id: str,
    model_id: str,
    *,
    mode: str = "val_f1",
    target_precision: float = 0.90,
) -> dict[str, float] | None:
    """Fit per-class thresholds on the run's frozen val split and persist them.

    Writes into the bundle's runtime_settings.json — the same overlay the
    runtime (and `evaluate_run`) applies — so the deployed decision layer and
    the test report both use thresholds matched to this model's probability
    scale instead of one-size config defaults. ``mode`` selects the target:
    "val_f1" maximizes per-class F1 (the default automatic setup);
    "val_precision" fits every class to the shared ``target_precision`` floor,
    which makes different runs directly comparable at one operating point.
    Returns the fitted thresholds, or None when calibration does not apply
    (no `_negative` class / no val split).

    This MUTATES the bundle. To score a val-fitted rule without changing what
    ships, use `fit_run_thresholds` (or the `val_f1` / `val_precision` rules).
    """
    from datetime import datetime, timezone

    from sed.config import dump_runtime_settings, load_runtime_settings

    result = fit_run_thresholds(
        cfg, project_id, model_id, mode=mode, target_precision=target_precision,
    )
    if result is None:
        return None
    fitted, n_val = result
    model_dir = ProjectLayout(project_id).model_dir(model_id)

    runtime_path = model_dir / "runtime_settings.json"
    settings = load_runtime_settings(runtime_path)
    settings.setdefault("thresholds", {})
    settings["thresholds"].update({k: float(v) for k, v in fitted.items()})
    # Provenance only — apply_runtime_settings ignores this block.
    calibration: dict[str, object] = {
        "mode": mode,
        "n_val": n_val,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }
    if mode == "val_precision":
        calibration["target_precision"] = float(target_precision)
    settings["calibration"] = calibration
    dump_runtime_settings(settings, runtime_path)
    log.info(
        "val-calibrated thresholds (%s, %d clips) → %s: %s",
        mode, n_val, runtime_path,
        {k: round(v, 3) for k, v in fitted.items()},
    )
    return fitted


def _prediction_records(
    probs: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    rows: list[dict[str, str]] | None = None,
    decisions: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Per-clip prediction rows with all class confidences.

    This is intentionally tabular rather than visual: frozen test sets can
    easily contain hundreds of clips, and operators need something sortable,
    greppable, and spreadsheet-friendly.
    """
    records: list[dict[str, object]] = []
    if probs.shape[0] == 0:
        return records

    pred_idx = np.argmax(probs, axis=1)
    true_idx = np.argmax(targets, axis=1)
    order = np.argsort(-probs, axis=1)
    rows = rows or [{} for _ in range(probs.shape[0])]

    for i in range(probs.shape[0]):
        top_i = int(pred_idx[i])
        true_i = int(true_idx[i])
        second_i = int(order[i, 1]) if probs.shape[1] > 1 else top_i
        row = rows[i] if i < len(rows) else {}
        decision = decisions[i] if decisions and i < len(decisions) else {}
        predicted_class = str(decision.get("predicted_class", classes[top_i]))
        predicted_confidence = float(decision.get("predicted_confidence", probs[i, top_i]))
        rec: dict[str, object] = {
            "index": i,
            "path": row.get("path", ""),
            "file": Path(row.get("path", "")).name,
            "true_class": classes[true_i],
            "predicted_class": predicted_class,
            "correct": classes[true_i] == predicted_class,
            "predicted_confidence": predicted_confidence,
            "triggered": bool(decision.get("triggered", predicted_class != "_negative")),
            "decision_reason": decision.get("decision_reason", "argmax"),
            "decision_threshold": decision.get("decision_threshold", ""),
            "argmax_class": decision.get("argmax_class", classes[top_i]),
            "argmax_confidence": float(decision.get("argmax_confidence", probs[i, top_i])),
            "second_class": classes[second_i],
            "second_confidence": float(probs[i, second_i]),
            "margin": float(probs[i, top_i] - probs[i, second_i]),
            "origin": row.get("origin", ""),
            "session_id": row.get("session_id", ""),
        }
        for cls_i, cls in enumerate(classes):
            rec[f"p:{cls}"] = float(probs[i, cls_i])
        records.append(rec)
    return records


def _write_prediction_table(
    records: list[dict[str, object]],
    classes: list[str],
    out_path: Path,
) -> Path:
    """Write all per-clip confidences as TSV."""
    base_fields = [
        "index", "file", "true_class", "predicted_class", "correct",
        "predicted_confidence", "triggered", "decision_reason", "decision_threshold",
        "argmax_class", "argmax_confidence", "second_class", "second_confidence", "margin",
        "origin", "session_id", "path",
    ]
    fieldnames = base_fields + [f"p:{cls}" for cls in classes]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            for key, value in list(row.items()):
                if isinstance(value, float):
                    row[key] = f"{value:.6f}"
            writer.writerow(row)
    return out_path


def _plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    out_path: Path,
    *,
    prediction_mode: str,
    accuracy: float | None = None,
    f1_macro_targets: float | None = None,
    ap_macro_targets: float | None = None,
) -> Path:
    """Render the matrix as a heatmap with annotated counts."""
    import matplotlib.pyplot as plt
    import textwrap

    n = len(classes)
    fig_w = max(6.5, n * 1.7)
    fig_h = max(5.2, n * 1.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = ["\n".join(textwrap.wrap(c, width=14, break_long_words=False)) for c in classes]
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    title = (
        "Frozen test confusion (runtime gates)"
        if prediction_mode == "runtime_thresholds"
        else "Frozen test confusion (closed set)"
    )
    # Lead the numeric summary with the metrics we actually rank by — macro-F1
    # over targets (this matrix's own operating point) and macro-AP (the
    # threshold-free trust metric) — and keep accuracy last as context, since
    # the grid already makes the per-class breakdown obvious.
    summary = []
    if f1_macro_targets is not None:
        summary.append(f"F1(targets) {f1_macro_targets:.3f}")
    if ap_macro_targets is not None:
        summary.append(f"AP {ap_macro_targets:.3f}")
    if accuracy is not None:
        summary.append(f"acc {accuracy:.3f}")
    if summary:
        title += " · " + " · ".join(summary)
    ax.set_title(title, fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    vmax = cm.max() if cm.size else 1
    row_totals = cm.sum(axis=1)
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > vmax / 2 else "black"
            pct = (cm[i, j] / row_totals[i] * 100.0) if row_totals[i] else 0.0
            label = f"{cm[i, j]}\n{pct:.0f}%"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
    ax.set_ylim(n - 0.5, -0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _load_model_classes(model_dir: Path) -> list[str]:
    if (model_dir / "classes.json").exists():
        return load_classes(model_dir)
    return load_classes(model_dir / "dataset")


def _snapshot_frozen_test(dataset_dir: Path) -> dict:
    snapshot_path = dataset_dir / "dataset_snapshot.json"
    if not snapshot_path.exists():
        return {}
    with open(snapshot_path, encoding="utf-8") as f:
        snap = json.load(f)
    frozen = snap.get("frozen_test", {})
    return frozen if isinstance(frozen, dict) else {}


def _materialize_mapped_dataset(
    dataset_dir: Path, mapping: dict[str, str], out_dir: Path
) -> tuple[Path, dict[str, str]]:
    """Write a copy of ``test.csv`` with its class labels folded by ``mapping``.

    Returns ``(dataset_dir_to_use, mapping_actually_applied)``.

    A remapped CSV rather than a remapped loader: every scoring path already
    reads ``<dataset_dir>/test.csv``, so this keeps one code path and the
    remapped file stays on disk as evidence of exactly what was scored.

    Only merges are honoured — a mapping entry whose target is not already a
    label in the test set is still applied, but the caller is responsible for
    checking the run can score the result (`_assert_model_covers_test_classes`
    runs immediately after).
    """
    src = Path(dataset_dir) / "test.csv"
    if not src.exists() or not mapping:
        return Path(dataset_dir), {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    applied: dict[str, str] = {}
    with open(src, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            name = str(row.get("class_name") or "")
            target = mapping.get(name)
            if target and target != name:
                applied[name] = target
                row = {**row, "class_name": target}
            rows.append(row)

    dest = out_dir / "test.csv"
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_dir, applied


def _csv_classes(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {str(row.get("class_name", "")) for row in csv.DictReader(f) if row.get("class_name")}


def _model_config(model_dir: Path, fallback: Config) -> Config:
    path = model_dir / "training_config.json"
    if path.exists():
        return load_config_snapshot(path)
    return fallback


def _overlay_runtime_settings(
    model_dir: Path,
    run_cfg: Config,
    *,
    enabled: bool = True,
) -> tuple[Config, str]:
    """Apply the bundle's runtime_settings.json — the deployed decision layer.

    The runtime overlays calibrated thresholds/targets from the bundle onto the
    frozen training config; evaluation must replay the same overlay or its F1
    describes a decision layer production never runs. Returns the overlaid
    config plus where the thresholds came from ("val_calibrated" /
    "runtime_settings" / "config_defaults") for the report.
    """
    from sed.config import apply_runtime_settings, load_runtime_settings

    if not enabled:
        return run_cfg, "config_snapshot"
    settings = load_runtime_settings(model_dir / "runtime_settings.json")
    source = "config_defaults"
    if isinstance(settings.get("thresholds"), dict) and settings["thresholds"]:
        calibration = settings.get("calibration") or {}
        mode = str(calibration.get("mode") or "")
        if mode in CALIBRATION_MODES:
            source = "val_calibrated"
        elif mode == "fp_budget":
            # An FP/hour sweep against background audio — a real calibration, so
            # it must not read as "somebody hand-edited the file".
            source = "fp_calibrated"
        else:
            source = "runtime_settings"
    return apply_runtime_settings(run_cfg, settings), source


def _evaluation_artifacts(report_name: str) -> dict[str, str]:
    """Report filenames + manifest file keys for one evaluation profile."""
    name = (report_name or "test").strip() or "test"
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)
    if safe == "test":
        return {
            "metrics_file": "test_metrics.json",
            "predictions_file": "test_predictions.tsv",
            "confusion_png": "confusion_matrix.png",
            "confusion_json": "confusion_matrix.json",
            "metrics_key": "test_metrics",
            "predictions_key": "test_predictions",
            "confusion_key": "confusion_matrix",
            "confusion_json_key": "confusion_matrix_json",
        }
    return {
        "metrics_file": f"{safe}_metrics.json",
        "predictions_file": f"{safe}_predictions.tsv",
        "confusion_png": f"{safe}_confusion_matrix.png",
        "confusion_json": f"{safe}_confusion_matrix.json",
        "metrics_key": f"{safe}_metrics",
        "predictions_key": f"{safe}_predictions",
        "confusion_key": f"{safe}_confusion_matrix",
        "confusion_json_key": f"{safe}_confusion_matrix_json",
    }


def _thresholds_source_label(
    runtime_source: str,
    threshold: float | None,
    *,
    use_runtime_settings: bool,
) -> str:
    if threshold is None:
        return runtime_source if use_runtime_settings else "config_snapshot"
    fixed = f"fixed_threshold:{float(threshold):.3f}"
    if not use_runtime_settings:
        return fixed
    return f"{fixed}+{runtime_source}"


def _metrics_manifest_block(metrics: dict) -> dict:
    """Compact metrics payload shared by every stored evaluation block.

    One shape for every scoring path — checkpoint under a rule, exported IR,
    train-time auto-test — so the runs table reads one set of keys and a metric
    can never be present in one block and missing from another.
    """
    # NOTE: `precision_macro_targets` is the metric; `ir_precision` is the
    # exported variant (fp16/int8). They are deliberately not both "precision".
    block = {
        "rule": metrics.get("rule"),
        "threshold": metrics["threshold"],
        "thresholds": metrics["thresholds"],
        "thresholds_source": metrics["thresholds_source"],
        "runtime_settings_applied": metrics.get("runtime_settings_applied"),
        "score_profile": metrics.get("score_profile"),
        "prediction_mode": metrics["prediction_mode"],
        "probability_mode": metrics["probability_mode"],
        "negative_max": metrics["negative_max"],
        "margin_to_second_best": metrics["margin_to_second_best"],
        "f1_macro_targets": metrics["f1_macro_targets"],
        "ap_macro_targets": metrics["ap_macro_targets"],
        "precision_macro_targets": metrics.get("precision_macro_targets"),
        "recall_macro_targets": metrics.get("recall_macro_targets"),
        "false_fire_rate": metrics.get("false_fire_rate"),
        "accuracy": metrics["accuracy"],
        "n_samples": metrics["n_samples"],
        "per_class": metrics["per_class"],
        "test_source": metrics["test_source"],
        "test_set_hash": metrics["test_set_hash"],
    }
    # Only the exported-IR path sets these; keeping them out of the checkpoint
    # blocks keeps "which engine produced this" unambiguous.
    for key in ("model_file", "ir_precision"):
        if metrics.get(key) is not None:
            block[key] = metrics[key]
    return block


def _store_scores(
    model_dir: Path,
    *,
    engine: str,
    probs: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    rows: list[dict[str, str]],
    metrics: dict,
    run_cfg: Config,
    class_mapping: dict[str, str] | None = None,
    ir_precision: str | None = None,
    split: str = "test",
) -> str | None:
    """Persist the probability matrix behind one scoring pass.

    Best-effort: a failure here must never lose an evaluation that already
    succeeded, so it logs and returns None rather than raising.
    """
    from sed.train import score_store

    test_hash = str(metrics.get("test_set_hash") or "")
    if not test_hash:
        # Without a hash the scores cannot be matched to a test set later, which
        # is the whole point of storing them.
        log.debug("not storing scores for %s — no test_set_hash", model_dir.name)
        return None
    try:
        return score_store.write_scores(
            model_dir,
            engine=engine,
            test_set_hash=test_hash,
            classes=list(classes),
            probs=probs,
            true_idx=np.argmax(targets, axis=1),
            paths=[str(r.get("path") or "") for r in rows],
            probability_mode=probability_mode(run_cfg),
            ir_precision=ir_precision,
            class_mapping=class_mapping or {},
            split=split,
        )
    except Exception as exc:  # noqa: BLE001 — an archive must not fail a scoring run
        log.warning("could not store scores for %s: %s", model_dir.name, exc)
        return None


def _val_split_hash(model_dir: Path) -> str:
    """Identity of a run's frozen validation split.

    The val split is per-run, not shared, so it is keyed by the content of the
    run's own val.csv. Prefixed `val:` so a val score store can never be
    mistaken for a frozen-test one when both live in the same directory.
    """
    from sed.util import sha256_file

    val_csv = model_dir / "dataset" / "val.csv"
    if not val_csv.is_file():
        return ""
    try:
        return f"val:{sha256_file(val_csv)}"
    except Exception:  # noqa: BLE001
        return ""


def _register_scores(manifest: dict, model_dir: Path, key: str | None) -> None:
    """Record a stored score pass in the manifest, if one was written."""
    if not key:
        return
    from sed.train import score_store

    sidecar = score_store.read_sidecar(model_dir, key)
    if sidecar is not None:
        score_store.register(manifest, sidecar)


def evaluate_run(
    cfg: Config,
    project_id: str,
    model_id: str,
    threshold: float | None = None,
    *,
    use_model_dataset: bool = False,
    use_runtime_settings: bool = True,
    report_name: str = "test",
    manifest_key: str | None = None,
    manifest_section: tuple[str, str] | None = None,
    thresholds_override: dict[str, float] | None = None,
    decision: str = "auto",
    rule: "TestRule | None" = None,
    class_mapping: dict[str, str] | None = None,
    progress: EvaluationProgress | None = None,
) -> dict:
    """Run a test CSV through a model's checkpoint.

    Writes test_metrics.json, test_predictions.tsv, confusion_matrix.json and
    confusion_matrix.png into the model's reports/ dir. Returns the metrics dict.

    The decision step is what distinguishes one frozen-test *rule* from another
    (see `TestRule`): `threshold` applies one shared value to every target class,
    `thresholds_override` supplies explicit per-class values (e.g. fitted on val),
    and `decision="argmax"` drops thresholds entirely. `manifest_section` nests the
    result under `manifest[section][key]` instead of a flat top-level key.
    """
    from sed.train.registry import get_checkpoint, load_manifest, save_manifest

    layout = ProjectLayout(project_id)
    model_dir = layout.model_dir(model_id)
    ckpt_path = get_checkpoint(project_id, model_id)
    run_cfg = _model_config(model_dir, cfg)
    run_cfg, runtime_thresholds_source = _overlay_runtime_settings(
        model_dir, run_cfg, enabled=use_runtime_settings
    )
    classes = _load_model_classes(model_dir)
    dataset_dir = model_dir / "dataset" if use_model_dataset else layout.dataset_dir

    # A class mapping folds the test set's labels onto the space this run was
    # trained on, so a taxonomy refinement (loud|medium|quiet -> cheers) does not
    # orphan the entire run history. It only ever MERGES labels, so it cannot
    # invent a distinction the run was never trained to make.
    mapping_applied: dict[str, str] = {}
    if class_mapping:
        dataset_dir, mapping_applied = _materialize_mapped_dataset(
            dataset_dir, class_mapping, model_dir / "reports" / "mapped_test"
        )

    test_csv = dataset_dir / "test.csv"
    if test_csv.exists():
        _assert_model_covers_test_classes(test_csv, classes, model_id)
    if not test_csv.exists() or _csv_row_count(test_csv) == 0:
        raise RuntimeError(
            "test.csv is empty — record frozen test material with "
            "`sed record --project <id> --test --class <label>` and re-run "
            "`sed dataset compose`."
        )
    unknown_classes = sorted(_csv_classes(test_csv) - set(classes))
    if unknown_classes:
        raise RuntimeError(
            f"test.csv contains classes this model was not trained with: {unknown_classes}. "
            f"Model {model_id} classes are: {classes}."
        )

    probs, targets, classes, rows = _collect_predictions(
        run_cfg,
        project_id,
        ckpt_path,
        dataset_dir,
        classes,
        progress=progress,
        progress_phase="scoring frozen test",
    )
    _report_evaluation_progress(
        progress,
        phase="calculating evaluation metrics",
        completed=len(rows),
        total=len(rows),
        current_name="",
        current_class="",
    )
    thresholds = _configured_thresholds(run_cfg, classes, override_threshold=threshold)
    if thresholds_override:
        # Explicit per-class values (a val-fitted rule) win over both the shared
        # `threshold` and the config; `_negative` stays the runtime gate.
        thresholds.update({
            cls: float(v) for cls, v in thresholds_override.items()
            if cls in thresholds and cls != "_negative"
        })
    thresholds_source = _thresholds_source_label(
        runtime_thresholds_source,
        threshold,
        use_runtime_settings=use_runtime_settings,
    )
    if thresholds_override:
        thresholds_source = "fitted_thresholds"
    if decision == "argmax" or "_negative" not in classes:
        # Forced argmax scores a model with a `_negative` class as a closed set:
        # no threshold, no gates — pure ranking quality, immune to calibration.
        pred_idx, decisions = _closed_set_decisions(probs, classes)
        prediction_mode = "closed_set_argmax"
        if decision == "argmax":
            thresholds_source = "argmax"
    else:
        pred_idx, decisions = _runtime_decisions(
            probs,
            classes,
            thresholds,
            negative_max=run_cfg.inference.negative_max,
            margin_to_second_best=run_cfg.inference.margin_to_second_best,
        )
        prediction_mode = "runtime_thresholds"
    cm = _confusion_matrix(probs, targets, classes, pred_idx=pred_idx)
    metrics = _decision_metrics(
        probs,
        targets,
        classes,
        cm,
        thresholds=thresholds,
        threshold=threshold,
        # The sweep must replay the gate the point metrics just used; argmax
        # scoring applies no margin at all.
        margin_to_second_best=(
            0.0 if prediction_mode == "closed_set_argmax"
            else float(run_cfg.inference.margin_to_second_best)
        ),
        budget_per_1000=float(run_cfg.training.calibrate_per_1000),
    )
    metrics["prediction_mode"] = prediction_mode
    metrics["probability_mode"] = probability_mode(run_cfg)
    metrics["thresholds_source"] = thresholds_source
    metrics["runtime_settings_applied"] = bool(use_runtime_settings)
    metrics["evaluation_name"] = report_name
    metrics["rule"] = rule.as_dict() if rule is not None else None
    metrics["score_profile"] = (
        "fixed_threshold_benchmark"
        if threshold is not None and not use_runtime_settings
        else ("deployment" if use_runtime_settings else "config_snapshot")
    )
    metrics["negative_max"] = float(run_cfg.inference.negative_max)
    metrics["margin_to_second_best"] = float(run_cfg.inference.margin_to_second_best)
    # The frozen test scores each clip as one independent window, so temporal
    # smoothing (median/EMA/consecutive) never applies here — this metric set is
    # always the per-window ("smoothing off") decision. We record the live
    # setting so the report says whether production adds smoothing on top.
    metrics["smoothing_enabled"] = bool(run_cfg.inference.smoothing_enabled)
    metrics["test_source"] = "model" if use_model_dataset else "current"
    # A mapped score is NOT the same statement as an unmapped one — the run was
    # measured at a coarser granularity than the test set carries. Recorded so
    # the runs table can say so rather than letting it read as a like-for-like
    # comparison with a run scored on the full taxonomy.
    metrics["class_mapping"] = dict(mapping_applied)
    frozen_snapshot = _snapshot_frozen_test(dataset_dir)
    metrics["test_set_hash"] = frozen_snapshot.get("hash", "")
    prediction_records = _prediction_records(probs, targets, classes, rows, decisions)

    artifacts = _evaluation_artifacts(report_name)
    metrics_key = manifest_key or artifacts["metrics_key"]
    reports = model_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / artifacts["metrics_file"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)
    with open(reports / artifacts["confusion_json"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": classes,
                "matrix": cm.tolist(),
                "prediction_mode": prediction_mode,
                "probability_mode": metrics["probability_mode"],
                "thresholds_source": thresholds_source,
                "runtime_settings_applied": bool(use_runtime_settings),
                "score_profile": metrics["score_profile"],
                "thresholds": thresholds,
                "negative_max": run_cfg.inference.negative_max,
                "margin_to_second_best": run_cfg.inference.margin_to_second_best,
                "accuracy": metrics["accuracy"],
                "test_source": metrics["test_source"],
                "test_set_hash": metrics["test_set_hash"],
            },
            f,
            indent=2,
        )
    _plot_confusion_matrix(
        cm,
        classes,
        reports / artifacts["confusion_png"],
        prediction_mode=prediction_mode,
        accuracy=metrics["accuracy"],
        f1_macro_targets=metrics.get("f1_macro_targets"),
        ap_macro_targets=metrics.get("ap_macro_targets"),
    )
    _write_prediction_table(prediction_records, classes, reports / artifacts["predictions_file"])

    # The probability matrix, stored once for this (engine, test set). Every rule
    # of this run produces the SAME matrix — only the decision differs — so this
    # is what makes re-thresholding in Runs & compare a lookup rather than
    # another forward pass. Written unconditionally: it costs a few hundred kB
    # and it is the only artifact that keeps a score re-derivable.
    scores_key = _store_scores(
        model_dir,
        engine="checkpoint",
        probs=probs,
        targets=targets,
        classes=classes,
        rows=rows,
        metrics=metrics,
        run_cfg=run_cfg,
        class_mapping=mapping_applied,
    )

    # Update model manifest with fresh metrics + the current test_set_hash.
    try:
        manifest = load_manifest(project_id, model_id)
    except FileNotFoundError:
        manifest = {}
    manifest.setdefault("dataset", {})
    manifest.setdefault("files", {})
    _register_scores(manifest, model_dir, scores_key)
    if metrics_key == "test_metrics":
        manifest["dataset"]["test_set_hash"] = metrics["test_set_hash"]
        manifest["dataset"]["frozen_test_source"] = frozen_snapshot.get("source", "")
        manifest["dataset"]["frozen_test_source_counts"] = frozen_snapshot.get("source_counts", {})
        manifest["dataset"]["frozen_test_manifest"] = frozen_snapshot.get("manifest", "")
    if manifest_section is not None:
        section, key = manifest_section
        manifest.setdefault(section, {})[key] = _metrics_manifest_block(metrics)
    else:
        manifest[metrics_key] = _metrics_manifest_block(metrics)
    manifest["files"][artifacts["metrics_key"]] = f"reports/{artifacts['metrics_file']}"
    manifest["files"][artifacts["confusion_key"]] = f"reports/{artifacts['confusion_png']}"
    manifest["files"][artifacts["confusion_json_key"]] = f"reports/{artifacts['confusion_json']}"
    manifest["files"][artifacts["predictions_key"]] = f"reports/{artifacts['predictions_file']}"
    save_manifest(project_id, model_id, manifest)

    log.info(
        "evaluated model %s (%s) on %d samples — F1(targets)=%.3f AP(targets)=%.3f",
        model_id, report_name, metrics["n_samples"],
        metrics["f1_macro_targets"], metrics["ap_macro_targets"],
    )
    _report_evaluation_progress(
        progress,
        phase="evaluation saved",
        completed=len(rows),
        total=len(rows),
        current_name="",
        current_class="",
    )
    return metrics


def evaluate_run_under_rule(
    cfg: Config,
    project_id: str,
    model_id: str,
    rule: TestRule,
    *,
    class_mapping: dict[str, str] | None = None,
    progress: EvaluationProgress | None = None,
) -> dict:
    """Score a run on the CURRENT frozen test under one decision rule.

    Every rule reads the same clips and the same checkpoint; only the decision
    step differs. Results land in `manifest["frozen_test"][rule.id]` and in
    `reports/<rule.slug>_*`, so rules accumulate side by side instead of
    overwriting each other.
    """
    kwargs: dict = {
        "use_model_dataset": False,
        "report_name": rule.slug,
        "manifest_section": (FROZEN_TEST_SECTION, rule.id),
        "rule": rule,
        # Folds a refined test taxonomy onto the space this run was trained on,
        # so a class-space change does not orphan the run's history.
        "class_mapping": class_mapping,
    }
    if rule.kind == "fixed":
        kwargs.update(threshold=float(rule.param), use_runtime_settings=False)
    elif rule.kind == "argmax":
        kwargs.update(decision="argmax", use_runtime_settings=False)
    elif rule.kind == "deployed":
        kwargs.update(use_runtime_settings=True)
    else:  # val_f1 / val_precision — fit on val, score the frozen test, save nothing
        fit_kwargs = {
            "mode": rule.kind,
            "target_precision": float(rule.param or 0.90),
        }
        if progress is not None:
            fit_kwargs["progress"] = progress
        fitted = fit_run_thresholds(cfg, project_id, model_id, **fit_kwargs)
        if fitted is None:
            raise RuntimeError(
                f"rule {rule.id} needs a val split and a _negative class — "
                f"model {model_id} has neither"
            )
        kwargs.update(thresholds_override=fitted[0], use_runtime_settings=False)
    if progress is not None:
        kwargs["progress"] = progress
    return evaluate_run(cfg, project_id, model_id, **kwargs)


def _resolved_ir_precision(model_path: Path, requested: str) -> str:
    """Which IR variant `resolve_model_path` actually landed on.

    "auto" follows the deployment pointer, so the request alone does not say
    what was scored — and the storage key has to, or an auto run would overwrite
    whichever explicit variant it happens to match.
    """
    if requested in ("fp16", "int8"):
        return requested
    stem = model_path.stem.lower()
    for known in ("int8", "fp16"):
        if known in stem:
            return known
    return "ir"


def evaluate_exported_run(
    cfg: Config,
    project_id: str,
    model_id: str,
    *,
    use_model_dataset: bool = True,
    precision: str = "auto",
    rule: "TestRule | None" = None,
    persist: bool = True,
) -> dict:
    """Score the exported (deployed) OpenVINO model on the frozen test set.

    The torch-checkpoint metrics describe what was trained; production runs the
    exported IR — typically INT8 — whose quantisation can shift class boundaries
    exactly where false-positive behaviour matters.

    ``precision`` picks which IR to score ("fp16" / "int8", or "auto" for
    whichever variant is deployed). ``rule`` picks how its probabilities become
    a pass/fail: the default ``deployed`` scores the run's own decision layer
    (what production delivers), while a benchmark rule holds the decision fixed
    so the gap to the checkpoint's score under the SAME rule is quantisation
    cost alone.

    Results land in ``manifest["deployed_test"][<precision>][<rule.id>]`` — so
    FP16 and INT8, and one rule and the next, accumulate side by side instead of
    overwriting each other — and in ``reports/deployed_<precision>_<slug>_*``.
    The newest block is mirrored into the flat ``test_metrics_exported`` key for
    readers that predate the section. ``persist=False`` skips all writes.
    """
    from sed.runtime.manifest import resolve_model_path
    from sed.train.registry import load_manifest, save_manifest

    rule = rule or TestRule("deployed")
    if rule.kind not in EXPORTED_RULE_KINDS:
        raise ValueError(
            f"rule {rule.id} cannot score an exported IR: {rule.kind} fits thresholds on "
            f"checkpoint probabilities. Use one of {', '.join(EXPORTED_RULE_KINDS)}."
        )

    layout = ProjectLayout(project_id)
    model_dir = layout.model_dir(model_id)
    run_cfg = _model_config(model_dir, cfg)
    run_cfg, thresholds_source = _overlay_runtime_settings(
        model_dir, run_cfg, enabled=rule.kind == "deployed"
    )
    classes = _load_model_classes(model_dir)
    dataset_dir = model_dir / "dataset" if use_model_dataset else layout.dataset_dir

    test_csv = dataset_dir / "test.csv"
    if not test_csv.exists() or _csv_row_count(test_csv) == 0:
        raise RuntimeError("test.csv is empty — nothing to score the exported model on.")
    _assert_model_covers_test_classes(test_csv, classes, model_id)

    model_path = resolve_model_path(model_dir, prefer=None if precision == "auto" else precision)
    probs, targets, classes, rows = _collect_predictions_openvino(
        run_cfg,
        project_id,
        model_path,
        dataset_dir,
        classes,
    )
    override = float(rule.param) if rule.kind == "fixed" else None
    thresholds = _configured_thresholds(run_cfg, classes, override_threshold=override)
    if rule.kind == "argmax" or "_negative" not in classes:
        pred_idx, _decisions = _closed_set_decisions(probs, classes)
        prediction_mode = "closed_set_argmax"
    else:
        pred_idx, _decisions = _runtime_decisions(
            probs,
            classes,
            thresholds,
            negative_max=run_cfg.inference.negative_max,
            margin_to_second_best=run_cfg.inference.margin_to_second_best,
        )
        prediction_mode = "runtime_thresholds"
    cm = _confusion_matrix(probs, targets, classes, pred_idx=pred_idx)
    metrics = _decision_metrics(
        probs, targets, classes, cm, thresholds=thresholds, threshold=override,
        margin_to_second_best=(
            0.0 if prediction_mode == "closed_set_argmax"
            else float(run_cfg.inference.margin_to_second_best)
        ),
        budget_per_1000=float(run_cfg.training.calibrate_per_1000),
    )
    metrics["rule"] = rule.as_dict()
    metrics["prediction_mode"] = prediction_mode
    metrics["probability_mode"] = probability_mode(run_cfg)
    metrics["thresholds_source"] = "argmax" if rule.kind == "argmax" else thresholds_source
    metrics["runtime_settings_applied"] = rule.kind == "deployed"
    metrics["score_profile"] = "deployment" if rule.kind == "deployed" else "fixed_threshold_benchmark"
    metrics["negative_max"] = float(run_cfg.inference.negative_max)
    metrics["margin_to_second_best"] = float(run_cfg.inference.margin_to_second_best)
    metrics["model_file"] = Path(model_path).name
    # "auto" resolves to a concrete variant; store what was actually scored so
    # the FP16 and INT8 blocks can never collide.
    resolved_precision = _resolved_ir_precision(Path(model_path), precision)
    metrics["ir_precision"] = resolved_precision
    metrics["test_source"] = "model" if use_model_dataset else "current"
    frozen_snapshot = _snapshot_frozen_test(dataset_dir)
    metrics["test_set_hash"] = frozen_snapshot.get("hash", "")

    if persist:
        reports = model_dir / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        report_stem = f"deployed_{resolved_precision}_{rule.slug}"
        with open(reports / f"{report_stem}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=float)

        # An IR's probabilities are its own — quantization moves class boundaries
        # — so they get their own engine key and never overwrite the checkpoint's.
        scores_key = _store_scores(
            model_dir,
            engine=f"ov_{resolved_precision}",
            probs=probs,
            targets=targets,
            classes=classes,
            rows=rows,
            metrics=metrics,
            run_cfg=run_cfg,
            ir_precision=resolved_precision,
        )

        try:
            manifest = load_manifest(project_id, model_id)
        except FileNotFoundError:
            manifest = {}
        _register_scores(manifest, model_dir, scores_key)
        manifest.setdefault("dataset", {})
        # Only a score against the CURRENT frozen test may claim the project's
        # hash; scoring the run's own snapshot must not restamp it.
        if not use_model_dataset:
            manifest["dataset"]["test_set_hash"] = metrics["test_set_hash"]
            manifest["dataset"]["frozen_test_source"] = frozen_snapshot.get("source", "")
            manifest["dataset"]["frozen_test_source_counts"] = frozen_snapshot.get("source_counts", {})
            manifest["dataset"]["frozen_test_manifest"] = frozen_snapshot.get("manifest", "")
        manifest.setdefault("files", {})
        block = _metrics_manifest_block(metrics)
        section = manifest.setdefault(DEPLOYED_TEST_SECTION, {})
        section.setdefault(resolved_precision, {})[rule.id] = block
        # Flat mirror for readers that predate the section.
        manifest["test_metrics_exported"] = block
        manifest["files"][f"{report_stem}_metrics"] = f"reports/{report_stem}_metrics.json"
        manifest["files"]["test_metrics_exported"] = f"reports/{report_stem}_metrics.json"
        save_manifest(project_id, model_id, manifest)

    log.info(
        "evaluated exported model %s (%s) on %d samples — F1(targets)=%.3f AP(targets)=%.3f",
        model_id, metrics["model_file"], metrics["n_samples"],
        metrics["f1_macro_targets"], metrics["ap_macro_targets"],
    )
    return metrics


def _recall_at_budget(metrics: dict, budget_per_1000: float) -> float | None:
    """Recall reachable inside a false-alarm budget, from a scored block."""
    from sed.train.operating_point import recall_at_budget

    point = recall_at_budget({"points": metrics.get("operating_curve") or []}, budget_per_1000)
    return None if point is None else point.get("recall_macro")


def decide_ir_precision(
    cfg: Config,
    project_id: str,
    model_id: str,
    *,
    tolerance: float | None = None,
    budget_per_1000: float | None = None,
    rule: "TestRule | None" = None,
) -> dict:
    """Score FP16 and INT8 on the frozen test and pick which one deploys.

    Quantization is lossy in a way that "the export succeeded" cannot detect: it
    moves class boundaries, and it moves them exactly where a confident model
    separates a clink from the room. So INT8 is deployed only when it holds its
    recall at the false-alarm budget to within `tolerance` of FP16.

    Recall-at-budget rather than F1: F1 shifts with a test set's
    negative:positive balance, so an F1 tolerance means different things on
    different benchmarks, while a recall-at-budget tolerance means the same
    thing everywhere — "how many real events did quantization cost us at the
    false-alarm rate we will actually run at".

    Returns the decision and both scores; the caller records it in the manifest.
    """
    from sed.export.openvino import set_active_precision

    layout = ProjectLayout(project_id)
    model_dir = layout.model_dir(model_id)
    tolerance = float(cfg.training.int8_max_regression if tolerance is None else tolerance)
    budget = float(cfg.training.calibrate_per_1000 if budget_per_1000 is None else budget_per_1000)
    rule = rule or TestRule("fixed", cfg.training.auto_test_threshold)

    scores: dict[str, float | None] = {}
    for precision in ("fp16", "int8"):
        if not (model_dir / f"model_{precision}.xml").exists():
            scores[precision] = None
            continue
        try:
            metrics = evaluate_exported_run(
                cfg, project_id, model_id, precision=precision, rule=rule,
            )
            scores[precision] = _recall_at_budget(metrics, budget)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not score the %s IR of %s: %s", precision, model_id, exc)
            scores[precision] = None

    fp16, int8 = scores.get("fp16"), scores.get("int8")
    if int8 is None:
        choice, reason = "fp16", "int8 IR missing or unscorable"
    elif fp16 is None:
        choice, reason = "int8", "fp16 IR unscorable — nothing to compare against"
    elif tolerance < 0:
        choice, reason = "int8", "tolerance is negative: int8 always kept"
    else:
        regression = fp16 - int8
        if regression > tolerance:
            choice = "fp16"
            reason = (f"int8 lost {regression:.3f} recall at ≤{budget:g}/1000, "
                      f"over the {tolerance:.3f} tolerance")
        else:
            choice = "int8"
            reason = (f"int8 within tolerance ({regression:+.3f} recall "
                      f"at ≤{budget:g}/1000, allowed {tolerance:.3f})")

    decision = {
        "precision": choice,
        "reason": reason,
        "metric": f"recall_at_{budget:g}_per_1000",
        "tolerance": tolerance,
        "fp16": fp16,
        "int8": int8,
        "rule": rule.id,
        "decided_at": datetime.now(UTC).isoformat(),
    }
    try:
        set_active_precision(model_dir, choice)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not point active_openvino.xml at %s: %s", choice, exc)
        decision["error"] = str(exc)
    log.info("deploy precision for %s → %s (%s)", model_id, choice, reason)
    return decision


def _csv_row_count(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _assert_model_covers_test_classes(test_csv: Path, classes: list[str], model_id: str) -> None:
    """Fail early and legibly when a run cannot be scored on this frozen test.

    Scoring against the CURRENT frozen test is what makes runs comparable, but a
    run trained on a different label space has no output for the classes it has
    never seen. Left to the dataloader that surfaces as a bare
    ``KeyError: '<class>'`` with no indication of which model or why.
    """
    known = set(classes)
    with open(test_csv, encoding="utf-8") as f:
        missing = sorted({
            row["class_name"] for row in csv.DictReader(f)
            if row.get("class_name") and row["class_name"] not in known
        })
    if missing:
        raise RuntimeError(
            f"model {model_id} was trained on classes {sorted(known)} and cannot be "
            f"scored on this frozen test, which also contains {missing}. Its stored "
            f"scores stay valid for the test set it was trained against; re-train on "
            f"the current classes to compare it with the rest."
        )


# ---------------------------------------------------------------------------
# Augmentation audit — which augmented clips does a trained model recognize?
# ---------------------------------------------------------------------------


# Band-masking score (dB): how far the base transient sits above the overlay in
# its own mel bands. Can go negative when the overlay buries the event.
_AUDIT_MASK_BINS: list[tuple[float, float, str]] = [
    (float("-inf"), 0.0, "<0"),
    (0.0, 5.0, "0–5"),
    (5.0, 10.0, "5–10"),
    (10.0, 15.0, "10–15"),
    (15.0, 20.0, "15–20"),
    (20.0, float("inf"), "≥20"),
]

# Effective broadband SNR (dB): recognition falls off hard below −10 dB, so the
# bins expose the negative floor rather than lumping it into "<0".
_AUDIT_SNR_BINS: list[tuple[float, float, str]] = [
    (float("-inf"), -10.0, "<-10"),
    (-10.0, 0.0, "-10–0"),
    (0.0, 5.0, "0–5"),
    (5.0, 10.0, "5–10"),
    (10.0, 15.0, "10–15"),
    (15.0, float("inf"), "≥15"),
]


def _gather_derived_rows(project_id: str) -> list[dict]:
    """One row per derived WAV in the project's library, with audit metadata."""
    from sed.config import LIBRARY_DIR

    rows: list[dict] = []
    root = LIBRARY_DIR / "derived" / project_id
    for split_name in ("trainval", "test"):
        split_root = root / split_name
        if not split_root.exists():
            continue
        for sidecar in sorted(split_root.rglob("*.json")):
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            wav = sidecar.with_suffix(".wav")
            if not wav.exists():
                continue
            params = meta.get("augmentation_params") if isinstance(meta.get("augmentation_params"), dict) else {}
            base_label = str(meta.get("base_label") or params.get("base_label") or "")
            overlay_label = str(meta.get("overlay_label") or params.get("overlay_label") or "")
            # Readable recipe name for the audit's by-recipe table: the sidecar
            # description ("riedel_wine x ambience_rnd") when present, else the
            # base/overlay combo. Falls back to recipe_id (below) so distinct
            # recipes never collapse under a shared base label.
            recipe_label = str(meta.get("description") or "").strip()
            if not recipe_label and base_label and overlay_label:
                recipe_label = f"{base_label} x {overlay_label}"
            rows.append({
                "path": str(wav),
                "label": str(meta.get("base_label") or meta.get("label") or ""),
                "split": split_name,
                "recipe_id": str(meta.get("recipe_id") or params.get("recipe_id") or ""),
                "recipe_label": recipe_label,
                "masking_score_db": meta.get("masking_score_db"),
                "effective_snr_db": meta.get("effective_snr_db", params.get("effective_snr_db")),
            })
    return rows


def _training_paths_for_model(model_dir: Path) -> set[str]:
    """Resolved paths of the clips in the model bundle's train.csv, if kept."""
    train_csv = model_dir / "dataset" / "train.csv"
    if not train_csv.exists():
        return set()
    paths: set[str] = set()
    try:
        with open(train_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = str(row.get("path") or "")
                if raw:
                    paths.add(str(Path(raw).resolve()))
    except Exception:  # noqa: BLE001
        return set()
    return paths


def _bin_label(value: object, bins: list[tuple[float, float, str]]) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    for lo, hi, name in bins:
        if lo <= v < hi:
            return name
    return "n/a"


def _derived_class_assignments(
    cfg: Config,
    project_id: str,
    classes: list[str],
) -> dict[str, str]:
    """Map derived clip paths to the model class they compose into.

    Library labels (e.g. riedel_veloce) usually differ from model classes
    (e.g. cheers_quiet); the dataset selection rules are the source of truth
    for that mapping, so the audit resolves it exactly like compose does.
    """
    from sed.data.compose import _library_clips_for_class, load_dataset_selection
    from sed.data.library import DERIVED_CATEGORY

    layout = ProjectLayout(project_id)
    try:
        selection = load_dataset_selection(layout)
    except Exception:  # noqa: BLE001
        return {}
    mapping: dict[str, str] = {}
    for class_name in classes:
        for role in ("train", "test"):
            try:
                clips = _library_clips_for_class(
                    cfg, project_id, class_name,
                    split=role, selection=selection,
                )
            except Exception:  # noqa: BLE001
                continue
            for clip in clips:
                if clip.origin == DERIVED_CATEGORY:
                    mapping[str(Path(clip.path).resolve())] = class_name
    return mapping


def audit_augmented(
    cfg: Config,
    project_id: str,
    model_id: str | None = None,
    *,
    exported: bool = False,
) -> dict:
    """Score every derived (augmented) clip with a trained model.

    Each clip is judged with the same unsmoothed runtime gates as the frozen
    test: a clip counts as recognized when the decision equals the model class
    its bucket is assigned to (per the dataset selection rules; falling back to
    the raw label when that already is a model class). The report tells which
    recipes / mask-score ranges produce material the model can actually use —
    clips the model also trained on are flagged (`seen_in_training`) because
    their numbers are optimistic.

    Writes ``projects/<id>/dataset/inspect/augment_audit.json`` and returns it.
    """
    from sed.train.registry import get_active_model, get_checkpoint

    layout = ProjectLayout(project_id)
    model_id = model_id or get_active_model(project_id)
    if not model_id:
        raise RuntimeError(
            "no trained model found — train one first or pass a model id"
        )
    model_dir = layout.model_dir(model_id)
    run_cfg = _model_config(model_dir, cfg)
    classes = _load_model_classes(model_dir)

    all_rows = _gather_derived_rows(project_id)
    path_to_class = _derived_class_assignments(cfg, project_id, classes)
    rows: list[dict] = []
    skipped: set[str] = set()
    for row in all_rows:
        expected = path_to_class.get(str(Path(row["path"]).resolve()))
        if expected is None and row["label"] in classes:
            expected = row["label"]
        if expected is None:
            skipped.add(row["label"])
            continue
        row["expected_class"] = expected
        rows.append(row)
    skipped_labels = sorted(skipped)
    if not rows:
        raise RuntimeError(
            f"no augmented clips assigned to model classes found for project '{project_id}' — "
            "assign the derived buckets to classes in the Inspect Library tab first"
            + (f" (unassigned labels: {skipped_labels})" if skipped_labels else "")
        )

    audit_dir = layout.dataset_inspect_dir / "augment_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = audit_dir / "test.csv"
    with open(audit_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_name"])
        for row in rows:
            w.writerow([row["path"], row["expected_class"]])

    if exported:
        from sed.runtime.manifest import resolve_model_path
        model_path = resolve_model_path(model_dir)
        probs, _targets, classes, ds_rows = _collect_predictions_openvino(
            run_cfg, project_id, model_path, audit_dir, classes,
        )
        scored_with = Path(model_path).name
    else:
        ckpt_path = get_checkpoint(project_id, model_id)
        probs, _targets, classes, ds_rows = _collect_predictions(
            run_cfg, project_id, ckpt_path, audit_dir, classes,
        )
        scored_with = Path(ckpt_path).name

    thresholds = _configured_thresholds(run_cfg, classes)
    if "_negative" in classes:
        _, decisions = _runtime_decisions(
            probs,
            classes,
            thresholds,
            negative_max=run_cfg.inference.negative_max,
            margin_to_second_best=run_cfg.inference.margin_to_second_best,
        )
    else:
        _, decisions = _closed_set_decisions(probs, classes)

    seen_paths = _training_paths_for_model(model_dir)
    per_clip: dict[str, dict] = {}
    for row, csv_row, decision in zip(rows, ds_rows, decisions):
        assert row["path"] == csv_row["path"]
        predicted = str(decision.get("predicted_class") or "")
        per_clip[row["path"]] = {
            "label": row["label"],
            "expected_class": row["expected_class"],
            "predicted_class": predicted,
            "predicted_confidence": float(decision.get("predicted_confidence") or 0.0),
            "correct": predicted == row["expected_class"],
            "decision_reason": str(decision.get("decision_reason") or ""),
            "split": row["split"],
            "recipe_id": row["recipe_id"],
            "recipe_label": row.get("recipe_label") or "",
            "masking_score_db": row["masking_score_db"],
            "effective_snr_db": row["effective_snr_db"],
            "seen_in_training": str(Path(row["path"]).resolve()) in seen_paths,
        }

    def _rate_table(key_fn) -> dict[str, dict]:
        table: dict[str, dict] = {}
        for clip in per_clip.values():
            key = key_fn(clip)
            entry = table.setdefault(key, {"total": 0, "recognized": 0, "unseen_total": 0, "unseen_recognized": 0})
            entry["total"] += 1
            entry["recognized"] += int(clip["correct"])
            if not clip["seen_in_training"]:
                entry["unseen_total"] += 1
                entry["unseen_recognized"] += int(clip["correct"])
        for entry in table.values():
            entry["rate"] = entry["recognized"] / entry["total"] if entry["total"] else 0.0
            entry["unseen_rate"] = (
                entry["unseen_recognized"] / entry["unseen_total"] if entry["unseen_total"] else None
            )
        return table

    from datetime import UTC, datetime

    report = {
        "model_id": model_id,
        "scored_with": scored_with,
        "exported": bool(exported),
        "generated_at": datetime.now(UTC).isoformat(),
        "n_clips": len(per_clip),
        "skipped_labels": skipped_labels,
        "summary": {
            "overall": _rate_table(lambda c: "all"),
            "by_recipe": _rate_table(lambda c: c["recipe_label"] or c["recipe_id"] or "(no recipe)"),
            "by_mask_bin": _rate_table(lambda c: _bin_label(c["masking_score_db"], _AUDIT_MASK_BINS)),
            "by_effective_snr_bin": _rate_table(lambda c: _bin_label(c["effective_snr_db"], _AUDIT_SNR_BINS)),
            "by_split": _rate_table(lambda c: c["split"]),
        },
        "per_clip": per_clip,
    }
    out_path = layout.dataset_inspect_dir / "augment_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    log.info(
        "audited %d augmented clip(s) with model %s — overall recognition %.1f%%",
        len(per_clip), model_id,
        100.0 * report["summary"]["overall"]["all"]["rate"],
    )
    return report
