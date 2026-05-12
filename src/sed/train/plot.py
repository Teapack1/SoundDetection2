"""Render the per-run training-curves figure from Lightning's CSV log.

Lightning's CSVLogger writes one row per "log event" — that means metrics
emitted per-step live on different rows than metrics emitted per-epoch (val/*,
train/loss_epoch). We collapse them onto a single x-axis keyed by epoch so the
plot is readable, and we annotate the checkpoint-selected epoch.
"""

from __future__ import annotations

import csv as csvlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sed.train.lit_module import (
    FIXED_SELECTION_RECALL_FLOOR,
    FIXED_SELECTION_THRESHOLD,
    VAL_FIXED_FP_KEY,
    VAL_FIXED_NEGATIVE_COUNT_KEY,
    VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY,
    VAL_FIXED_RECALL_KEY,
)

log = logging.getLogger(__name__)


SUPERVISED_SELECTION_POLICY_NAME = "fixed_threshold_false_positive"


def supervised_selection_policy() -> dict:
    """Serializable contract used by checkpointing, reports, and manifests."""
    return {
        "name": SUPERVISED_SELECTION_POLICY_NAME,
        "threshold": FIXED_SELECTION_THRESHOLD,
        "target_recall_floor": FIXED_SELECTION_RECALL_FLOOR,
        "eligibility": (
            f"{VAL_FIXED_RECALL_KEY}>={FIXED_SELECTION_RECALL_FLOOR}"
        ),
        "decision_contract": "runtime_winner_negative_veto_margin_abstention",
        "eligible_order": [
            f"min:{VAL_FIXED_FP_KEY}",
            "max:val/ap_macro_targets",
            "min:val/loss",
        ],
        "fallback_order": [
            f"max:{VAL_FIXED_RECALL_KEY}",
            f"min:{VAL_FIXED_FP_KEY}",
            "max:val/ap_macro_targets",
            "min:val/loss",
        ],
        "no_negative_clips": (
            f"{VAL_FIXED_FP_KEY}=0.0 and {VAL_FIXED_NEGATIVE_COUNT_KEY}=0"
        ),
    }


def supervised_selection_rank(metrics: dict[str, float]) -> tuple | None:
    """Return a lexicographic rank (larger wins) for one supervised epoch."""
    try:
        recall = float(metrics[VAL_FIXED_RECALL_KEY])
        false_triggers = float(metrics[VAL_FIXED_FP_KEY])
        target_ap = float(metrics["val/ap_macro_targets"])
        val_loss = float(metrics["val/loss"])
    except (KeyError, TypeError, ValueError):
        return None
    values = (recall, false_triggers, target_ap, val_loss)
    if not all(float("-inf") < value < float("inf") for value in values):
        return None
    # Lightning/CSV may have serialized a float32 ratio (for example 19/20) a
    # few ulps below its mathematical value. Treat that representation as the
    # exact inclusive floor promised by the policy.
    eligible = recall >= FIXED_SELECTION_RECALL_FLOOR - 1e-7
    if eligible:
        # Recall only gates eligibility.  Once the floor is met, operational
        # false triggers dominate all further quality comparisons.
        return (1, -false_triggers, target_ap, -val_loss)
    return (0, recall, -false_triggers, target_ap, -val_loss)


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


def _lr_curves_to_plot(series: dict, lr_cols: list[str]) -> list[tuple[str, list, list]]:
    """Collapse many per-param-group LR columns into ≤3 representative curves.

    Lightning's ``LearningRateMonitor`` logs one ``lr-AdamW/pgK`` column per
    optimizer param group (backbone stage × decay/no-decay split), and layer-wise
    decay makes them differ only by a constant scale of one shared warmup+cosine
    schedule — so plotting every column is a dozen near-identical lines with an
    unreadable ``pg0…pgN`` legend. Keep the distinct trajectories only, then show
    at most the lowest / median / highest of them (by peak LR), labelled by value.
    """
    uniq: dict[tuple, tuple[list, list]] = {}
    for col in lr_cols:
        xs, ys = _series_xy(series.get(col, {}))
        if not ys:
            continue
        uniq.setdefault(tuple(round(float(v), 12) for v in ys), (xs, ys))
    curves = sorted(uniq.values(), key=lambda xy: max(xy[1]))
    if not curves:
        return []
    if len(curves) == 1:
        xs, ys = curves[0]
        return [("learning rate", xs, ys)]
    picks = [curves[0], curves[-1]] if len(curves) == 2 else \
        [curves[0], curves[len(curves) // 2], curves[-1]]
    return [(f"peak {max(ys):.1e}", xs, ys) for xs, ys in picks]


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
        "val/byol_score",
        "val/f1_macro_targets",
        "val/ap_macro_targets",
        VAL_FIXED_RECALL_KEY,
        VAL_FIXED_FP_KEY,
        VAL_FIXED_NEGATIVE_COUNT_KEY,
        VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY,
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
            "val_byol_score": series.get("val/byol_score", {}).get(epoch),
            "val_f1_macro_targets": series.get("val/f1_macro_targets", {}).get(epoch),
            "val_ap_macro_targets": series.get("val/ap_macro_targets", {}).get(epoch),
            "val_recall_macro_targets_at_0_9": series.get(VAL_FIXED_RECALL_KEY, {}).get(epoch),
            "val_positive_triggers_per_1000_negatives_at_0_9": series.get(
                VAL_FIXED_FP_KEY, {}
            ).get(epoch),
            "val_negative_clip_count": series.get(
                VAL_FIXED_NEGATIVE_COUNT_KEY, {}
            ).get(epoch),
            "val_positive_trigger_count_on_negatives_at_0_9": series.get(
                VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY, {}
            ).get(epoch),
        })

    loss_map = series.get("val/loss", {})
    acc_map = series.get("val/accuracy", {})
    ap_map = series.get("val/ap_macro_targets", {})
    fixed_recall_map = series.get(VAL_FIXED_RECALL_KEY, {})
    fixed_fp_map = series.get(VAL_FIXED_FP_KEY, {})
    best_epoch = None
    selection_policy = None
    supervised_epochs = sorted(
        set(loss_map) & set(ap_map) & set(fixed_recall_map) & set(fixed_fp_map)
    )
    if supervised_epochs:
        selection_policy = supervised_selection_policy()
        monitor, mode = VAL_FIXED_RECALL_KEY, "lexicographic"
        tiebreak = f"{VAL_FIXED_FP_KEY}, val/ap_macro_targets, val/loss"
        best_rank = None
        for epoch in supervised_epochs:
            candidate = {
                VAL_FIXED_RECALL_KEY: fixed_recall_map[epoch],
                VAL_FIXED_FP_KEY: fixed_fp_map[epoch],
                "val/ap_macro_targets": ap_map[epoch],
                "val/loss": loss_map[epoch],
            }
            candidate_rank = supervised_selection_rank(candidate)
            if candidate_rank is None:
                continue
            if best_epoch is None or best_rank is None:
                best_epoch = epoch
                best_rank = candidate_rank
                continue
            if candidate_rank > best_rank:
                best_epoch = epoch
                best_rank = candidate_rank
        score_map = fixed_recall_map
    elif ap_map:
        # Backward-compatible summary for metrics.csv files created before the
        # fixed-threshold selection metrics were introduced.
        # Selection follows ranking quality, not calibration: BCE overconfidence
        # inflates val/loss epochs before AP actually degrades. Ties break
        # toward the lower (better-calibrated) loss.
        monitor, mode, tiebreak = "val/ap_macro_targets", "max", "val/loss"
        for epoch, val_ap in sorted(ap_map.items()):
            if best_epoch is None:
                best_epoch = epoch
                continue
            best_ap = ap_map[best_epoch]
            tied_with_better_loss = (
                abs(val_ap - best_ap) <= 0.001
                and loss_map.get(epoch, float("inf")) < loss_map.get(best_epoch, float("inf"))
            )
            if val_ap > best_ap + 0.001 or tied_with_better_loss:
                best_epoch = epoch
        score_map = ap_map
    else:
        # No AP series (e.g. BYOL): fall back to loss with BYOL-score/accuracy tie-break.
        byol_map = series.get("val/byol_score", {})
        tiebreak_map = byol_map or acc_map
        tiebreak = "val/byol_score" if byol_map else "val/accuracy"
        monitor, mode = "val/loss", "min"
        for epoch, val_loss in sorted(loss_map.items()):
            if best_epoch is None:
                best_epoch = epoch
                continue
            best_loss = loss_map[best_epoch]
            tied_with_better_tiebreak = (
                abs(val_loss - best_loss) <= 0.001
                and tiebreak_map.get(epoch, float("-inf")) > tiebreak_map.get(best_epoch, float("-inf"))
            )
            if val_loss < best_loss - 0.001 or tied_with_better_tiebreak:
                best_epoch = epoch
        score_map = loss_map
    best = None
    if best_epoch is not None:
        best = {
            "epoch": best_epoch,
            "monitor": monitor,
            "mode": mode,
            "selection_tiebreak": tiebreak,
            "selection_policy": selection_policy,
            "score": score_map[best_epoch],
            "val_loss": series.get("val/loss", {}).get(best_epoch),
            "val_accuracy": series.get("val/accuracy", {}).get(best_epoch),
            "val_byol_score": series.get("val/byol_score", {}).get(best_epoch),
            "train_loss": series.get("train/loss_epoch", {}).get(best_epoch),
            "val_f1_macro_targets": series.get("val/f1_macro_targets", {}).get(best_epoch),
            "val_ap_macro_targets": series.get("val/ap_macro_targets", {}).get(best_epoch),
            "val_recall_macro_targets_at_0_9": series.get(
                VAL_FIXED_RECALL_KEY, {}
            ).get(best_epoch),
            "val_positive_triggers_per_1000_negatives_at_0_9": series.get(
                VAL_FIXED_FP_KEY, {}
            ).get(best_epoch),
            "val_negative_clip_count": series.get(
                VAL_FIXED_NEGATIVE_COUNT_KEY, {}
            ).get(best_epoch),
            "val_positive_trigger_count_on_negatives_at_0_9": series.get(
                VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY, {}
            ).get(best_epoch),
        }

    return {
        "best": best,
        "epochs": epoch_rows,
        "selection_policy": selection_policy,
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
        "selection_policy": summary.get("selection_policy"),
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

    The checkpoint-selected epoch is marked on every panel.
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
        VAL_FIXED_RECALL_KEY, VAL_FIXED_FP_KEY,
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
    best_fixed_recall = (
        series.get(VAL_FIXED_RECALL_KEY, {}).get(best_epoch)
        if best_epoch is not None else None
    )
    best_fixed_fp = (
        series.get(VAL_FIXED_FP_KEY, {}).get(best_epoch)
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
        if best_fixed_recall is not None:
            title += f" · recall@0.9 = {best_fixed_recall:.3f}"
        if best_fixed_fp is not None:
            title += f" · FP/1k = {best_fixed_fp:.1f}"
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
    fixed_recall_x, fixed_recall_y = _series_xy(series.get(VAL_FIXED_RECALL_KEY, {}))
    if fixed_recall_y:
        ax.plot(
            fixed_recall_x,
            fixed_recall_y,
            color="C3",
            linestyle="--",
            label="val recall @ 0.9 (targets, macro)",
        )
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

    # Learning rate — collapsed to ≤3 representative curves (see _lr_curves_to_plot);
    # the raw per-param-group columns would be a dozen overlapping pgK lines.
    ax = axes[1][1]
    lr_curves = _lr_curves_to_plot(series, lr_cols)
    for label, xs, ys in lr_curves:
        ax.plot(xs, ys, label=label)
    _mark_best(ax)
    ax.set_title("Learning rate")
    ax.set_xlabel("epoch")
    if lr_curves:
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    log.info("training curves → %s", out_png)
    return out_png
