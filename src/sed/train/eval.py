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

log = logging.getLogger(__name__)


def _collect_predictions(
    cfg: Config,
    project_id: str,
    ckpt_path: Path,
    dataset_dir: Path,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    """Run the test loader through the checkpoint once.

    Returns:
        probs: (N, C) objective-aware probabilities
        targets: (N, C) one-hot ground truth
        classes: class names in order
        rows: test.csv rows in dataloader order
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"You are using `torch.load` with `weights_only=False`.*",
        )
        lit = SedLitModule.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, classes=classes, map_location="cpu"
        )
    lit.eval()

    ds = SedDataset(dataset_dir / "test.csv", classes, cfg, layout=ProjectLayout(project_id))
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    probs_chunks: list[np.ndarray] = []
    targets_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            logits = lit(x)
            probs_chunks.append(logits_to_probabilities(logits, cfg).cpu().numpy())
            targets_chunks.append(y.cpu().numpy())
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


def _per_class_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    threshold: float | None = 0.5,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Per-class AP / F1 / precision / recall plus macros over target classes."""
    if thresholds is None:
        fallback = 0.5 if threshold is None else float(threshold)
        thresholds = {cls: fallback for cls in classes}

    preds_bin = np.zeros_like(probs, dtype=np.int32)
    for i, cls in enumerate(classes):
        preds_bin[:, i] = (probs[:, i] >= thresholds.get(cls, 0.5)).astype(np.int32)
    target_bin = (targets >= 0.5).astype(np.int32)

    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(classes):
        tp = int(np.sum((preds_bin[:, i] == 1) & (target_bin[:, i] == 1)))
        fp = int(np.sum((preds_bin[:, i] == 1) & (target_bin[:, i] == 0)))
        fn = int(np.sum((preds_bin[:, i] == 0) & (target_bin[:, i] == 1)))
        tn = int(np.sum((preds_bin[:, i] == 0) & (target_bin[:, i] == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        ap = _average_precision(probs[:, i], target_bin[:, i])
        per_class[name] = {
            "ap": ap,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    target_names = [c for c in classes if c != "_negative"]
    f1_macro_targets = (
        float(np.mean([per_class[c]["f1"] for c in target_names]))
        if target_names else 0.0
    )
    ap_macro_targets = (
        float(np.mean([per_class[c]["ap"] for c in target_names]))
        if target_names else 0.0
    )

    return {
        "threshold": threshold,
        "thresholds": dict(thresholds),
        "per_class": per_class,
        "f1_macro_targets": f1_macro_targets,
        "ap_macro_targets": ap_macro_targets,
        "n_samples": int(probs.shape[0]),
    }


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute area under the precision-recall curve.

    Mirrors torchmetrics' MultilabelAveragePrecision per-class; we recompute
    it here so eval is self-contained and works on a numpy stack of probs.
    """
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp_cum = np.cumsum(labels_sorted)
    fp_cum = np.cumsum(1 - labels_sorted)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / labels.sum()
    # AP = sum_k (R_k - R_{k-1}) * P_k
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def _confusion_matrix(
    probs: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    pred_idx: np.ndarray | None = None,
) -> np.ndarray:
    """NxN matrix. Rows = true class, cols = predicted class."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    if probs.shape[0] == 0:
        return cm
    if pred_idx is None:
        pred_idx = np.argmax(probs, axis=1)
    true_idx = np.argmax(targets, axis=1)
    for t, p in zip(true_idx, pred_idx, strict=True):
        cm[t, p] += 1
    return cm


def _decision_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    cm: np.ndarray,
    thresholds: dict[str, float],
    threshold: float | None = None,
) -> dict:
    """Precision/recall/F1 from threshold-aware single-label decisions."""
    target_bin = (targets >= 0.5).astype(np.int32)
    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        tn = int(np.sum(cm) - tp - fp - fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        ap = _average_precision(probs[:, i], target_bin[:, i])
        per_class[name] = {
            "ap": ap,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": int(np.sum(cm[i, :])),
        }

    target_names = [c for c in classes if c != "_negative"]
    f1_macro_targets = (
        float(np.mean([per_class[c]["f1"] for c in target_names]))
        if target_names else 0.0
    )
    ap_macro_targets = (
        float(np.mean([per_class[c]["ap"] for c in target_names]))
        if target_names else 0.0
    )
    accuracy = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) else 0.0
    return {
        "threshold": threshold,
        "thresholds": dict(thresholds),
        "per_class": per_class,
        "f1_macro_targets": f1_macro_targets,
        "ap_macro_targets": ap_macro_targets,
        "accuracy": accuracy,
        "n_samples": int(probs.shape[0]),
    }


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


def _closed_set_decisions(
    probs: np.ndarray,
    classes: list[str],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Argmax decisions for target-only models without an explicit no-trigger class."""
    if probs.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), []
    pred_idx = np.argmax(probs, axis=1).astype(np.int64)
    order = np.argsort(-probs, axis=1)
    decisions: list[dict[str, object]] = []
    for i, idx in enumerate(pred_idx):
        second_idx = int(order[i, 1]) if probs.shape[1] > 1 else int(idx)
        decisions.append({
            "predicted_class": classes[int(idx)],
            "predicted_confidence": float(probs[i, idx]),
            "triggered": True,
            "decision_reason": "closed_set_argmax",
            "decision_threshold": "",
            "argmax_class": classes[int(idx)],
            "argmax_confidence": float(probs[i, idx]),
            "runner_up_class": classes[second_idx],
            "runner_up_confidence": float(probs[i, second_idx]),
        })
    return pred_idx, decisions


def _runtime_decisions(
    probs: np.ndarray,
    classes: list[str],
    thresholds: dict[str, float],
    negative_max: float,
    margin_to_second_best: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Single-window version of the runtime trigger decision.

    The live system additionally smooths probabilities over time. Frozen test
    clips are independent one-second windows, so we apply the same gates without
    median/EMA/consecutive-frame state.
    """
    if probs.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), []

    has_negative = "_negative" in classes
    neg_idx = classes.index("_negative") if has_negative else None
    positive_indices = [i for i, cls in enumerate(classes) if cls != "_negative"]
    pred_idx = np.zeros((probs.shape[0],), dtype=np.int64)
    decisions: list[dict[str, object]] = []

    for row in probs:
        argmax_idx = int(np.argmax(row))
        if not positive_indices:
            fallback_idx = neg_idx if neg_idx is not None else argmax_idx
            decisions.append({
                "predicted_class": classes[fallback_idx],
                "predicted_confidence": float(row[fallback_idx]),
                "triggered": False,
                "decision_reason": "no_positive_classes",
                "decision_threshold": float("nan"),
                "argmax_class": classes[argmax_idx],
                "argmax_confidence": float(row[argmax_idx]),
            })
            pred_idx[len(decisions) - 1] = fallback_idx
            continue

        winner_idx = max(positive_indices, key=lambda idx: row[idx])
        winner_score = float(row[winner_idx])
        winner_class = classes[winner_idx]
        positive_scores = sorted(
            [(float(row[idx]), idx) for idx in positive_indices],
            reverse=True,
        )
        runner_score, runner_idx = (
            positive_scores[1] if len(positive_scores) > 1 else (0.0, winner_idx)
        )
        threshold = thresholds.get(winner_class, 0.5)
        neg_score = float(row[neg_idx]) if neg_idx is not None else 0.0

        reason = "trigger"
        triggered = True
        predicted_idx = winner_idx
        predicted_confidence = winner_score
        if neg_idx is not None and neg_score > negative_max:
            reason = "negative_gate"
            triggered = False
            predicted_idx = neg_idx
            predicted_confidence = neg_score
        elif len(positive_scores) > 1 and winner_score - runner_score < margin_to_second_best:
            reason = "margin_gate"
            triggered = False
            predicted_idx = neg_idx if neg_idx is not None else winner_idx
            predicted_confidence = neg_score if neg_idx is not None else winner_score
        elif winner_score < threshold:
            reason = "below_threshold"
            triggered = False
            predicted_idx = neg_idx if neg_idx is not None else winner_idx
            predicted_confidence = neg_score if neg_idx is not None else winner_score

        pred_idx[len(decisions)] = predicted_idx
        decisions.append({
            "predicted_class": classes[predicted_idx],
            "predicted_confidence": float(predicted_confidence),
            "triggered": triggered,
            "decision_reason": reason,
            "decision_threshold": float(threshold),
            "winner_class": winner_class,
            "winner_confidence": winner_score,
            "runner_up_class": classes[runner_idx],
            "runner_up_confidence": float(runner_score),
            "argmax_class": classes[argmax_idx],
            "argmax_confidence": float(row[argmax_idx]),
        })

    return pred_idx, decisions


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
    if accuracy is not None:
        title += f" · acc {accuracy:.3f}"
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


def _snapshot_hash(dataset_dir: Path) -> str:
    snapshot_path = dataset_dir / "dataset_snapshot.json"
    if not snapshot_path.exists():
        return ""
    with open(snapshot_path, encoding="utf-8") as f:
        snap = json.load(f)
    return snap.get("frozen_test", {}).get("hash", "")


def _csv_classes(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {str(row.get("class_name", "")) for row in csv.DictReader(f) if row.get("class_name")}


def _model_config(model_dir: Path, fallback: Config) -> Config:
    path = model_dir / "training_config.json"
    if path.exists():
        return load_config_snapshot(path)
    return fallback


def evaluate_run(
    cfg: Config,
    project_id: str,
    model_id: str,
    threshold: float | None = None,
    *,
    use_model_dataset: bool = False,
) -> dict:
    """Run a test CSV through a model's checkpoint.

    Writes test_metrics.json, test_predictions.tsv, confusion_matrix.json and
    confusion_matrix.png into the model's reports/ dir. Returns the metrics dict.
    """
    from sed.train.registry import get_checkpoint, load_manifest, save_manifest

    layout = ProjectLayout(project_id)
    model_dir = layout.model_dir(model_id)
    ckpt_path = get_checkpoint(project_id, model_id)
    run_cfg = _model_config(model_dir, cfg)
    classes = _load_model_classes(model_dir)
    dataset_dir = model_dir / "dataset" if use_model_dataset else layout.dataset_dir

    test_csv = dataset_dir / "test.csv"
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
    )
    thresholds = _configured_thresholds(run_cfg, classes, override_threshold=threshold)
    if "_negative" in classes:
        pred_idx, decisions = _runtime_decisions(
            probs,
            classes,
            thresholds,
            negative_max=run_cfg.inference.negative_max,
            margin_to_second_best=run_cfg.inference.margin_to_second_best,
        )
        prediction_mode = "runtime_thresholds"
    else:
        pred_idx, decisions = _closed_set_decisions(probs, classes)
        prediction_mode = "closed_set_argmax"
    cm = _confusion_matrix(probs, targets, classes, pred_idx=pred_idx)
    metrics = _decision_metrics(
        probs,
        targets,
        classes,
        cm,
        thresholds=thresholds,
        threshold=threshold,
    )
    metrics["prediction_mode"] = prediction_mode
    metrics["probability_mode"] = probability_mode(run_cfg)
    metrics["negative_max"] = float(run_cfg.inference.negative_max)
    metrics["margin_to_second_best"] = float(run_cfg.inference.margin_to_second_best)
    # The frozen test scores each clip as one independent window, so temporal
    # smoothing (median/EMA/consecutive) never applies here — this metric set is
    # always the per-window ("smoothing off") decision. We record the live
    # setting so the report says whether production adds smoothing on top.
    metrics["smoothing_enabled"] = bool(run_cfg.inference.smoothing_enabled)
    metrics["test_source"] = "model" if use_model_dataset else "current"
    metrics["test_set_hash"] = _snapshot_hash(dataset_dir)
    prediction_records = _prediction_records(probs, targets, classes, rows, decisions)

    reports = model_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)
    with open(reports / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": classes,
                "matrix": cm.tolist(),
                "prediction_mode": prediction_mode,
                "probability_mode": metrics["probability_mode"],
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
        reports / "confusion_matrix.png",
        prediction_mode=prediction_mode,
        accuracy=metrics["accuracy"],
    )
    _write_prediction_table(prediction_records, classes, reports / "test_predictions.tsv")

    # Update model manifest with fresh metrics + the current test_set_hash.
    try:
        manifest = load_manifest(project_id, model_id)
    except FileNotFoundError:
        manifest = {}
    manifest.setdefault("dataset", {})
    manifest.setdefault("files", {})
    manifest["dataset"]["test_set_hash"] = metrics["test_set_hash"]
    manifest["test_metrics"] = {
        "threshold": metrics["threshold"],
        "thresholds": metrics["thresholds"],
        "prediction_mode": metrics["prediction_mode"],
        "probability_mode": metrics["probability_mode"],
        "negative_max": metrics["negative_max"],
        "margin_to_second_best": metrics["margin_to_second_best"],
        "f1_macro_targets": metrics["f1_macro_targets"],
        "ap_macro_targets": metrics["ap_macro_targets"],
        "accuracy": metrics["accuracy"],
        "n_samples": metrics["n_samples"],
        "per_class": metrics["per_class"],
        "test_source": metrics["test_source"],
        "test_set_hash": metrics["test_set_hash"],
    }
    manifest["files"]["test_metrics"] = "reports/test_metrics.json"
    manifest["files"]["confusion_matrix"] = "reports/confusion_matrix.png"
    manifest["files"]["confusion_matrix_json"] = "reports/confusion_matrix.json"
    manifest["files"]["test_predictions"] = "reports/test_predictions.tsv"
    save_manifest(project_id, model_id, manifest)

    log.info(
        "evaluated model %s on %d samples — F1(targets)=%.3f AP(targets)=%.3f",
        model_id, metrics["n_samples"],
        metrics["f1_macro_targets"], metrics["ap_macro_targets"],
    )
    return metrics


def evaluate_exported_run(
    cfg: Config,
    project_id: str,
    model_id: str,
    *,
    use_model_dataset: bool = True,
) -> dict:
    """Score the exported (deployed) OpenVINO model on the frozen test set.

    The torch-checkpoint metrics describe what was trained; production runs
    the exported IR — typically INT8 — whose quantisation can shift class
    boundaries exactly where false-positive behaviour matters. This writes
    reports/test_metrics_exported.json and a ``test_metrics_exported`` block
    into the model manifest so both views sit side by side.
    """
    from sed.runtime.manifest import resolve_model_path
    from sed.train.registry import load_manifest, save_manifest

    layout = ProjectLayout(project_id)
    model_dir = layout.model_dir(model_id)
    run_cfg = _model_config(model_dir, cfg)
    classes = _load_model_classes(model_dir)
    dataset_dir = model_dir / "dataset" if use_model_dataset else layout.dataset_dir

    test_csv = dataset_dir / "test.csv"
    if not test_csv.exists() or _csv_row_count(test_csv) == 0:
        raise RuntimeError("test.csv is empty — nothing to score the exported model on.")

    model_path = resolve_model_path(model_dir)
    probs, targets, classes, _rows = _collect_predictions_openvino(
        run_cfg,
        project_id,
        model_path,
        dataset_dir,
        classes,
    )
    thresholds = _configured_thresholds(run_cfg, classes)
    if "_negative" in classes:
        pred_idx, _decisions = _runtime_decisions(
            probs,
            classes,
            thresholds,
            negative_max=run_cfg.inference.negative_max,
            margin_to_second_best=run_cfg.inference.margin_to_second_best,
        )
        prediction_mode = "runtime_thresholds"
    else:
        pred_idx, _decisions = _closed_set_decisions(probs, classes)
        prediction_mode = "closed_set_argmax"
    cm = _confusion_matrix(probs, targets, classes, pred_idx=pred_idx)
    metrics = _decision_metrics(probs, targets, classes, cm, thresholds=thresholds)
    metrics["prediction_mode"] = prediction_mode
    metrics["probability_mode"] = probability_mode(run_cfg)
    metrics["negative_max"] = float(run_cfg.inference.negative_max)
    metrics["margin_to_second_best"] = float(run_cfg.inference.margin_to_second_best)
    metrics["model_file"] = Path(model_path).name
    metrics["test_set_hash"] = _snapshot_hash(dataset_dir)

    reports = model_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / "test_metrics_exported.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    try:
        manifest = load_manifest(project_id, model_id)
    except FileNotFoundError:
        manifest = {}
    manifest.setdefault("files", {})
    manifest["test_metrics_exported"] = {
        "model_file": metrics["model_file"],
        "thresholds": metrics["thresholds"],
        "prediction_mode": metrics["prediction_mode"],
        "f1_macro_targets": metrics["f1_macro_targets"],
        "ap_macro_targets": metrics["ap_macro_targets"],
        "accuracy": metrics["accuracy"],
        "n_samples": metrics["n_samples"],
        "per_class": metrics["per_class"],
        "test_set_hash": metrics["test_set_hash"],
    }
    manifest["files"]["test_metrics_exported"] = "reports/test_metrics_exported.json"
    save_manifest(project_id, model_id, manifest)

    log.info(
        "evaluated exported model %s (%s) on %d samples — F1(targets)=%.3f AP(targets)=%.3f",
        model_id, metrics["model_file"], metrics["n_samples"],
        metrics["f1_macro_targets"], metrics["ap_macro_targets"],
    )
    return metrics


def _csv_row_count(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


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
                    dataset_role=role, selection=selection,
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
