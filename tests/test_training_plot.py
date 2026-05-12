from __future__ import annotations

import csv
import json

from sed.train.lit_module import (
    FIXED_SELECTION_RECALL_FLOOR,
    FIXED_SELECTION_THRESHOLD,
    VAL_FIXED_FP_KEY,
    VAL_FIXED_NEGATIVE_COUNT_KEY,
    VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY,
    VAL_FIXED_RECALL_KEY,
)
from sed.train.plot import (
    _read_lr_series,
    _target_ap_columns,
    summarize_training_metrics,
    write_training_report,
)


def test_read_lr_series_handles_lightning_rows_without_epoch(tmp_path):
    path = tmp_path / "metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "lr-AdamW/pg1"])
        writer.writeheader()
        writer.writerow({"epoch": "", "step": "0", "lr-AdamW/pg1": "0.001"})
        writer.writerow({"epoch": "0", "step": "9", "lr-AdamW/pg1": ""})
        writer.writerow({"epoch": "", "step": "10", "lr-AdamW/pg1": "0.0005"})

    series = _read_lr_series(path, ["lr-AdamW/pg1"])

    assert series["lr-AdamW/pg1"] == {0: 0.001, 1: 0.0005}


def test_target_ap_columns_exclude_negative_class():
    header = [
        "val/ap/_negative",
        "val/ap/cheers_loud",
        "val/ap/cheers_quiet",
        "val/ap_macro_targets",
    ]

    assert _target_ap_columns(header) == [
        "val/ap/cheers_loud",
        "val/ap/cheers_quiet",
    ]


def _write_selection_metrics(path, rows):
    fieldnames = [
        "epoch",
        "train/loss_epoch",
        "val/loss",
        "val/ap_macro_targets",
        VAL_FIXED_RECALL_KEY,
        VAL_FIXED_FP_KEY,
        VAL_FIXED_NEGATIVE_COUNT_KEY,
        VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY,
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_training_summary_uses_fixed_threshold_false_positive_selection(tmp_path):
    path = tmp_path / "metrics.csv"
    rows = [
        # Ineligible despite the strongest AP/loss and no false triggers.
        {"epoch": 0, "val/loss": 0.1, "val/ap_macro_targets": 0.99,
         VAL_FIXED_RECALL_KEY: 0.94, VAL_FIXED_FP_KEY: 0.0,
         VAL_FIXED_NEGATIVE_COUNT_KEY: 100, VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY: 0},
        # Eligible baseline.
        {"epoch": 1, "val/loss": 0.3, "val/ap_macro_targets": 0.80,
         VAL_FIXED_RECALL_KEY: 0.95, VAL_FIXED_FP_KEY: 20.0,
         VAL_FIXED_NEGATIVE_COUNT_KEY: 100, VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY: 2},
        # More recall cannot compensate for more false triggers after eligibility.
        {"epoch": 2, "val/loss": 0.2, "val/ap_macro_targets": 0.95,
         VAL_FIXED_RECALL_KEY: 0.99, VAL_FIXED_FP_KEY: 30.0,
         VAL_FIXED_NEGATIVE_COUNT_KEY: 100, VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY: 3},
        # Fewer false triggers wins, then AP and finally loss break exact ties.
        {"epoch": 3, "val/loss": 0.4, "val/ap_macro_targets": 0.70,
         VAL_FIXED_RECALL_KEY: 0.95, VAL_FIXED_FP_KEY: 10.0,
         VAL_FIXED_NEGATIVE_COUNT_KEY: 100, VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY: 1},
        {"epoch": 4, "val/loss": 0.4, "val/ap_macro_targets": 0.85,
         VAL_FIXED_RECALL_KEY: 0.96, VAL_FIXED_FP_KEY: 10.0,
         VAL_FIXED_NEGATIVE_COUNT_KEY: 100, VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY: 1},
        {"epoch": 5, "val/loss": 0.2, "val/ap_macro_targets": 0.85,
         VAL_FIXED_RECALL_KEY: 0.97, VAL_FIXED_FP_KEY: 10.0,
         VAL_FIXED_NEGATIVE_COUNT_KEY: 100, VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY: 1},
    ]
    _write_selection_metrics(path, rows)

    summary = summarize_training_metrics(path)

    assert summary["best"]["epoch"] == 5
    assert summary["best"]["mode"] == "lexicographic"
    assert summary["best"]["val_recall_macro_targets_at_0_9"] == 0.97
    assert summary["best"]["val_positive_triggers_per_1000_negatives_at_0_9"] == 10.0
    assert summary["selection_policy"]["threshold"] == FIXED_SELECTION_THRESHOLD
    assert (
        summary["selection_policy"]["target_recall_floor"]
        == FIXED_SELECTION_RECALL_FLOOR
    )


def test_training_summary_fallback_maximizes_recall_before_fp(tmp_path):
    path = tmp_path / "metrics.csv"
    rows = [
        {"epoch": 0, "val/loss": 0.1, "val/ap_macro_targets": 0.99,
         VAL_FIXED_RECALL_KEY: 0.90, VAL_FIXED_FP_KEY: 0.0,
         VAL_FIXED_NEGATIVE_COUNT_KEY: 0, VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY: 0},
        {"epoch": 1, "val/loss": 0.8, "val/ap_macro_targets": 0.50,
         VAL_FIXED_RECALL_KEY: 0.94, VAL_FIXED_FP_KEY: 100.0,
         VAL_FIXED_NEGATIVE_COUNT_KEY: 10, VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY: 1},
    ]
    _write_selection_metrics(path, rows)

    summary = summarize_training_metrics(path)

    assert summary["best"]["epoch"] == 1


def test_training_report_persists_exact_fixed_selection_policy(tmp_path):
    metrics = tmp_path / "metrics.csv"
    report_path = tmp_path / "training_report.json"
    _write_selection_metrics(metrics, [{
        "epoch": 0,
        "val/loss": 0.2,
        "val/ap_macro_targets": 0.9,
        VAL_FIXED_RECALL_KEY: 0.96,
        VAL_FIXED_FP_KEY: 0.0,
        VAL_FIXED_NEGATIVE_COUNT_KEY: 0,
        VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY: 0,
    }])

    write_training_report(
        metrics,
        report_path,
        epochs_ran=1,
        epochs_max=40,
        patience=8,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["selection_policy"]["threshold"] == 0.9
    assert report["selection_policy"]["target_recall_floor"] == 0.95
    assert report["selection"]["val_negative_clip_count"] == 0.0
    assert report["selection"]["val_positive_triggers_per_1000_negatives_at_0_9"] == 0.0


def test_training_summary_keeps_legacy_ap_selection_for_old_csv(tmp_path):
    path = tmp_path / "legacy_metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "val/loss", "val/ap_macro_targets"],
        )
        writer.writeheader()
        writer.writerows([
            {"epoch": 0, "val/loss": 0.2, "val/ap_macro_targets": 0.8},
            {"epoch": 1, "val/loss": 0.4, "val/ap_macro_targets": 0.9},
        ])

    summary = summarize_training_metrics(path)

    assert summary["best"]["epoch"] == 1
    assert summary["best"]["monitor"] == "val/ap_macro_targets"
    assert summary["best"]["mode"] == "max"
    assert summary["selection_policy"] is None
