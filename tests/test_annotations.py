"""Strong interval sidecar support used by the frame-SED workbench."""

from __future__ import annotations

import csv
import json


def test_normalize_event_intervals_rejects_invalid_and_marks_confirmation():
    from sed.data.library import normalize_event_intervals

    intervals = normalize_event_intervals([
        {"label": "glass", "onset_s": -1, "offset_s": 0.12, "confirmed": True},
        {"label": "bad", "onset_s": 0.5, "offset_s": 0.5},
        {"label": "late", "onset_s": 0.8, "offset_s": 2.0},
    ], duration_s=1.0)
    assert intervals == [
        {"label": "glass", "onset_s": 0.0, "offset_s": 0.12, "confirmed": True, "proposed_by": ""},
        {"label": "late", "onset_s": 0.8, "offset_s": 1.0, "confirmed": False, "proposed_by": ""},
    ]


def test_task_readiness_counts_composed_dataset_rows_not_library_or_previews(tmp_path):
    from sed.config import Config
    from sed.data.annotations import task_dataset_readiness

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    glass = tmp_path / "glass.wav"
    negative = tmp_path / "room.wav"
    glass.write_bytes(b"")
    negative.write_bytes(b"")
    glass.with_suffix(".json").write_text(json.dumps({
        "event_intervals": [{
            "label": "glass", "onset_s": 0.02, "offset_s": 0.12,
            "confirmed": True,
        }],
    }), encoding="utf-8")
    rows = [
        {"path": str(glass), "class_name": "glass"},
        {"path": str(negative), "class_name": "_negative"},
    ]
    with open(dataset_dir / "train.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "class_name"])
        writer.writeheader()
        writer.writerows(rows)
    with open(dataset_dir / "val.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "class_name"])
        writer.writeheader()
        writer.writerow({"path": str(glass), "class_name": "glass"})

    cfg = Config()
    clip = task_dataset_readiness(cfg, "clip", dataset_dir)
    assert clip["stats"]["dataset_rows"] == 3
    assert clip["stats"]["dataset_unique_wavs"] == 2
    assert clip["stats"]["split_counts"] == {"train": 2, "val": 1, "test": 0}
    assert "preview images do not cap the dataset" in clip["verdicts"][0]["text"]

    frame = task_dataset_readiness(cfg, "frame", dataset_dir)
    assert frame["stats"]["positive_rows_with_confirmed_intervals"] == 2
    assert frame["stats"]["frame_usable_train_rows"] == 2
