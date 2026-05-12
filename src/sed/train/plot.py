"""Render the per-run training-curves figure from Lightning's CSV log.

Lightning's CSVLogger writes one row per "log event" — that means metrics
emitted per-step live on different rows than metrics emitted per-epoch (val/*,
train/loss_epoch). We collapse them onto a single x-axis keyed by epoch so the
plot is readable, and we annotate the validation-loss checkpoint selection.
"""

from __future__ import annotations

import csv as csvlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def _read_epoch_series(metrics_csv: Path, columns: list[str]) -> dict[str, dict[int, float]]:
    """Return {column → {epoch_int → last_value_seen}} from a CSV.

    Lightning emits multiple rows per epoch (one for each log call). We keep
    the last value seen per epoch — that's the end-of-epoch summary.
    """
    out: dict[str, dict[int, float]] = {c: {} for c in columns}
    with open(metrics_csv, encoding="utf-8") as f:
        for row in csvlib.DictReader(f):
            epoch_raw = row.get("epoch")
            if not epoch_raw:
                continue
            try:
                epoch = int(float(epoch_raw))
            except ValueError:
                continue
            for col in columns:
                v = row.get(col)
                if v in (None, "", "nan"):
                    continue
                try:
                    out[col][epoch] = float(v)
                except ValueError:
                    continue
    return out


def _series_xy(epoch_map: dict[int, float]) -> tuple[list[int], list[float]]:
    xs = sorted(epoch_map.keys())
    return xs, [epoch_map[x] for x in xs]


def _read_lr_series(metrics_csv: Path, columns: list[str]) -> dict[str, dict[int, float]]:
    """Read Lightning LR rows, which often have `step` but no `epoch`."""
    out: dict[str, dict[int, float]] = {c: {} for c in columns}
    lr_epoch = 0
    with open(metrics_csv, encoding="utf-8") as f:
        for row in csvlib.DictReader(f):
            values: dict[str, float] = {}
            for col in columns:
                v = row.get(col)
                if v in (None, "", "nan"):
                    continue
                try:
                    values[col] = float(v)
                except ValueError:
                    continue
            if not values:
                continue
            epoch_raw = row.get("epoch")
            if epoch_raw:
                try:
                    epoch = int(float(epoch_raw))
                except ValueError:
                    epoch = lr_epoch
            else:
                epoch = lr_epoch
                lr_epoch += 1
            for col, value in values.items():
                out[col][epoch] = value
    return out


def _target_ap_columns(header: list[str]) -> list[str]:
    return [
        h for h in header
        if h.startswith("val/ap/") and h.split("/")[-1] != "_negative"
    ]


def summarize_training_metrics(metrics_csv: Path) -> dict:
    with open(metrics_csv, encoding="utf-8") as f:
        header = next(csvlib.reader(f), [])
    wanted = [
        "train/loss_epoch",
        "val/loss",
        "val/accuracy",
        "val/f1_macro_targets",
        "val/ap_macro_targets",
    ]
    series = _read_epoch_series(metrics_csv, [c for c in wanted if c in header])

    epochs = sorted({epoch for values in series.values() for epoch in values})
    epoch_rows = []
    for epoch in epochs:
        epoch_rows.append({
            "epoch": epoch,
            "train_loss": series.get("train/loss_epoch", {}).get(epoch),
            "val_loss": series.get("val/loss", {}).get(epoch),
            "val_accuracy": series.get("val/accuracy", {}).get(epoch),
            "val_f1_macro_targets": series.get("val/f1_macro_targets", {}).get(epoch),
            "val_ap_macro_targets": series.get("val/ap_macro_targets", {}).get(epoch),
        })

    loss_map = series.get("val/loss", {})
    acc_map = series.get("val/accuracy", {})
    best_epoch = None
    for epoch, val_loss in sorted(loss_map.items()):
        if best_epoch is None:
            best_epoch = epoch
            continue
        best_loss = loss_map[best_epoch]
        tied_with_better_accuracy = (
            abs(val_loss - best_loss) <= 0.001
            and acc_map.get(epoch, float("-inf")) > acc_map.get(best_epoch, float("-inf"))
        )
        if val_loss < best_loss - 0.001 or tied_with_better_accuracy:
            best_epoch = epoch
    best = None
    if best_epoch is not None:
        best = {
            "epoch": best_epoch,
            "monitor": "val/loss",
            "mode": "min",
            "selection_tiebreak": "val/accuracy",
            "score": loss_map[best_epoch],
            "val_loss": series.get("val/loss", {}).get(best_epoch),
            "val_accuracy": series.get("val/accuracy", {}).get(best_epoch),
            "train_loss": series.get("train/loss_epoch", {}).get(best_epoch),
            "val_f1_macro_targets": series.get("val/f1_macro_targets", {}).get(best_epoch),
            "val_ap_macro_targets": series.get("val/ap_macro_targets", {}).get(best_epoch),
        }

    return {
        "best": best,
        "epochs": epoch_rows,
    }


def write_training_report(
    metrics_csv: Path,
    out_json: Path,
    *,
    epochs_ran: int,
    epochs_max: int,
    patience: int,
) -> dict:
    summary = summarize_training_metrics(metrics_csv)
    report = {
        "selection": summary["best"],
        "early_stopping": {
            "epochs_ran": epochs_ran,
            "epochs_max": epochs_max,
            "stopped_early": epochs_ran < epochs_max,
            "patience": patience,
        },
        "epoch_metrics": summary["epochs"],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("training report → %s", out_json)
    return report


def plot_training_curves(metrics_csv: Path, out_png: Path, summary: dict | None = None) -> Path:
    """Render a 2x2 panel: loss · target-F1 · target AP · learning rate.

    The best epoch (by val/loss, accuracy tie-break) is marked on every panel.
    """
    # Sniff what columns the CSV actually has so we can fan out per-class lines
    # without depending on the class list being passed in.
    with open(metrics_csv, encoding="utf-8") as f:
        header = next(csvlib.reader(f), [])
    ap_cols = _target_ap_columns(header)
    lr_cols = [h for h in header if h.startswith("lr-")]

    wanted = [
        "train/loss_epoch", "val/loss",
        "val/accuracy", "val/f1_macro_targets", "val/ap_macro_targets",
        *ap_cols, *lr_cols,
    ]
    series = _read_epoch_series(metrics_csv, wanted)
    series.update(_read_lr_series(metrics_csv, lr_cols))

    f1_map = series.get("val/f1_macro_targets", {})
    best = (summary or {}).get("best")
    best_epoch = best.get("epoch") if best else None
    if best_epoch is None and f1_map:
        best_epoch = max(f1_map, key=lambda e: f1_map[e])
    best_f1 = f1_map.get(best_epoch) if best_epoch is not None else None
    best_val_loss = series.get("val/loss", {}).get(best_epoch) if best_epoch is not None else None
    best_val_acc = (
        series.get("val/accuracy", {}).get(best_epoch)
        if best_epoch is not None else None
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    title = "Training curves"
    if best_epoch is not None:
        title += f"  ·  best epoch {best_epoch}"
        if best_val_loss is not None:
            title += f" · val loss = {best_val_loss:.4f}"
        if best_val_acc is not None:
            title += f" · val acc = {best_val_acc:.3f}"
    fig.suptitle(title, fontsize=13, y=0.99)

    def _mark_best(ax):
        if best_epoch is None:
            return
        ax.axvline(best_epoch, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Loss
    ax = axes[0][0]
    for tag, label, color in [("train/loss_epoch", "train", "C0"),
                              ("val/loss", "val", "C1")]:
        xs, ys = _series_xy(series.get(tag, {}))
        if ys:
            ax.plot(xs, ys, label=label, color=color)
            if tag == "val/loss" and best_epoch is not None and best_val_loss is not None:
                ax.scatter([best_epoch], [best_val_loss], color="black", zorder=3, s=35)
                ax.annotate(
                    f"{best_val_loss:.4f}",
                    (best_epoch, best_val_loss),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=9,
                )
    _mark_best(ax)
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    # Target-class macro F1
    ax = axes[0][1]
    xs, ys = _series_xy(f1_map)
    if ys:
        ax.plot(xs, ys, color="C2", label="val F1 (target classes, macro)")
        if best_epoch is not None:
            ax.scatter([best_epoch], [best_f1], color="black", zorder=3, s=40)
            ax.annotate(f"{best_f1:.3f}", (best_epoch, best_f1),
                        textcoords="offset points", xytext=(6, -10), fontsize=9)
    _mark_best(ax)
    ax.set_title("F1 (target classes)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()

    # Target-class AP
    ax = axes[1][0]
    for col in ap_cols:
        xs, ys = _series_xy(series.get(col, {}))
        if ys:
            ax.plot(xs, ys, label=col.split("/")[-1], alpha=0.85)
    _mark_best(ax)
    ax.set_title("Target-class AP (val)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    # Learning rate
    ax = axes[1][1]
    plotted_any_lr = False
    for col in lr_cols:
        xs, ys = _series_xy(series.get(col, {}))
        if ys:
            ax.plot(xs, ys, label=col)
            plotted_any_lr = True
    _mark_best(ax)
    ax.set_title("Learning rate")
    ax.set_xlabel("epoch")
    if plotted_any_lr:
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    log.info("training curves → %s", out_png)
    return out_png
