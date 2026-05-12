from __future__ import annotations

import csv

from sed.train.plot import _read_lr_series, _target_ap_columns


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
