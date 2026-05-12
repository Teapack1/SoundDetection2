"""Detect train/test leakage between a frozen test set and an existing run.

Rebuilding a frozen test after models have been trained is the point of the
research workflow — you gather new test material, define the classes you care
about, and re-score the whole run history against it. The hazard is that the
new test may contain clips a given run was *trained* on, which inflates that
run's score and nothing about the number says so.

Nothing extra has to be recorded at training time to catch this: every run
directory already keeps a complete copy of the data it saw ::

    projects/<proj>/models/<run>/dataset/{train,val,test}.csv

so leakage is an intersection over files already on disk.

Two levels, and the second is the one that bites:

  * **clip leakage** — the identical file appears in the run's train or val split;
  * **group leakage** — a *different* file from the same recording group is in
    the run's training data. Clips cut from one recording seconds apart are
    near-duplicates, so this inflates scores just as effectively while looking
    like a clean split. The unit is `splits.split_group_key`, the same one
    composition uses to hold recordings out of validation.

Torch-free: the inspect server calls this from the batch-evaluate pre-flight.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from sed.data.splits import split_group_key

# Splits a run may have trained on. `test.csv` is deliberately excluded: a run's
# own frozen test is not training data.
TRAINING_SPLITS = ("train.csv", "val.csv")


@lru_cache(maxsize=None)
def _realdir(directory: str) -> str:
    return os.path.realpath(directory)


def normalize_clip_path(path: str | Path) -> str:
    """Canonical key for comparing clip paths across runs.

    Equivalent to ``Path(path).expanduser().resolve()`` but resolves each unique
    *directory* once. Per-file resolve() walks every path component and has
    already caused multi-second hangs on this corpus at 75k clips; a frozen test
    checked against 40+ runs multiplies that.
    """
    p = os.path.expanduser(os.fspath(path))
    directory, name = os.path.split(os.path.normpath(os.path.join(os.getcwd(), p)))
    return os.path.join(_realdir(directory), name)


def read_clip_rows(csv_path: Path) -> list[dict]:
    """Rows of a composed dataset CSV, or [] when it is absent/unreadable."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            return [row for row in csv.DictReader(fh) if row.get("path")]
    except (OSError, csv.Error):
        return []


@dataclass(frozen=True)
class TrainingIndex:
    """What one run was trained on, keyed for membership tests."""

    run_id: str
    paths: frozenset[str] = field(default_factory=frozenset)
    groups: frozenset[str] = field(default_factory=frozenset)
    n_clips: int = 0
    # False when the run kept no dataset copy — absence of evidence, not
    # evidence of absence, and reported that way rather than as "no leakage".
    available: bool = True


def training_index(model_dir: Path, run_id: str = "") -> TrainingIndex:
    """Index a run's train+val splits by clip path and recording group."""
    dataset_dir = Path(model_dir) / "dataset"
    paths: set[str] = set()
    groups: set[str] = set()
    n = 0
    found = False
    for name in TRAINING_SPLITS:
        rows = read_clip_rows(dataset_dir / name)
        if rows:
            found = True
        for row in rows:
            paths.add(normalize_clip_path(row["path"]))
            groups.add(split_group_key(row))
            n += 1
    return TrainingIndex(
        run_id=run_id or Path(model_dir).name,
        paths=frozenset(paths),
        groups=frozenset(groups),
        n_clips=n,
        available=found,
    )


def find_leakage(test_rows: list[dict], index: TrainingIndex, *, limit: int = 200) -> dict:
    """Which clips of a candidate test set the run has already seen.

    ``clip_leaks`` and ``group_leaks`` are disjoint: a clip counted as an exact
    match is not counted again as a group match, so the two totals sum to the
    number of compromised clips.
    """
    clip_leaks: list[dict] = []
    group_leaks: list[dict] = []
    for row in test_rows:
        path = row.get("path")
        if not path:
            continue
        entry = {"path": path, "class_name": row.get("class_name", "")}
        if normalize_clip_path(path) in index.paths:
            clip_leaks.append(entry)
            continue
        group = split_group_key(row)
        if group in index.groups:
            group_leaks.append({**entry, "group": group})

    leaked = [e["path"] for e in clip_leaks] + [e["path"] for e in group_leaks]
    return {
        "run_id": index.run_id,
        "available": index.available,
        "n_test_clips": len(test_rows),
        "n_train_clips": index.n_clips,
        "clip_leak_count": len(clip_leaks),
        "group_leak_count": len(group_leaks),
        "leak_count": len(leaked),
        "leak_fraction": (len(leaked) / len(test_rows)) if test_rows else 0.0,
        # Truncated for transport; the counts above are always exact.
        "clip_leaks": clip_leaks[:limit],
        "group_leaks": group_leaks[:limit],
        "truncated": len(clip_leaks) > limit or len(group_leaks) > limit,
        "leaked_paths": leaked,
    }


def find_leakage_for_run(test_rows: list[dict], model_dir: Path, run_id: str = "") -> dict:
    return find_leakage(test_rows, training_index(model_dir, run_id))


