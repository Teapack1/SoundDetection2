"""Plot training curves from Lightning's CSV logger output."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


def plot_training_curves(metrics_csv: Path, out_png: Path) -> Path:
    """Render a 2x2 panel: loss, F1, AP, learning rate."""
    import csv as csvlib

    rows: list[dict] = []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        for r in csvlib.DictReader(f):
            rows.append(r)
    if not rows:
        log.warning("metrics.csv is empty")
        return out_png

    def col(name: str) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for r in rows:
            v = r.get(name)
            if v in (None, "", "nan"):
                continue
            try:
                ys.append(float(v))
                xs.append(float(r.get("epoch") or r.get("step") or len(ys)))
            except ValueError:
                pass
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    ax = axes[0][0]
    for tag, label in [("train/loss_epoch", "train"), ("val/loss", "val")]:
        xs, ys = col(tag)
        if ys:
            ax.plot(xs, ys, label=label)
    ax.set_title("Loss"); ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)

    # F1 macro targets (the metric we care about)
    ax = axes[0][1]
    xs, ys = col("val/f1_macro_targets")
    if ys:
        ax.plot(xs, ys, label="val F1 (target classes, macro)", color="C2")
    ax.set_title("F1 (target classes)"); ax.set_xlabel("epoch")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()

    # AP per class
    ax = axes[1][0]
    keys = [r for r in rows[0].keys() if r.startswith("val/ap/")]
    for k in keys:
        xs, ys = col(k)
        if ys:
            ax.plot(xs, ys, label=k.split("/")[-1], alpha=0.85)
    ax.set_title("Per-class AP (val)"); ax.set_xlabel("epoch")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    # Learning rate
    ax = axes[1][1]
    lr_keys = [k for k in rows[0].keys() if k.startswith("lr-")]
    for k in lr_keys:
        xs, ys = col(k)
        if ys:
            ax.plot(xs, ys, label=k)
    ax.set_title("Learning rate"); ax.set_xlabel("epoch")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    log.info("training curves → %s", out_png)
    return out_png
