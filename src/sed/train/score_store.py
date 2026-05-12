"""Per-clip probabilities, stored once and re-scored forever.

Scoring a frozen test used to mean one forward pass *per decision rule*, with
each rule writing its own near-identical prediction table and freezing its own
metrics into the manifest. Changing a confidence from 0.90 to 0.95 meant running
the model again.

Nothing about the model changes between rules — only the decision step does. So
a run is scored **once** per (test set × engine), the probability matrix is
saved here, and every metric, threshold, confusion matrix and benchmark is
derived from it on demand. Re-inference is only needed when the test set itself
changes, which is exactly the case a hash comparison can detect.

A run may hold scores for several test sets at once, which is what lets an old
test set stay readable after the project moves to a new one.

Layout, under ``<run>/reports/scores/``::

    <engine>__<hash8>.npz    probs float32 (N, C) · true_idx int16 (N,)
    <engine>__<hash8>.json   classes, paths, test_set_hash, n, engine, …

float32 is what the model produced, so the npz is lossless — unlike the
human-readable prediction TSV, which rounds to six decimals.

Deliberately free of torch: the inspect server reads these on a request path.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SCORES_DIRNAME = "scores"
MANIFEST_SECTION = "frozen_scores"

# Which artifact produced the probabilities. The checkpoint and a quantized IR
# disagree by design, so their scores must never share a key.
ENGINES: tuple[str, ...] = ("checkpoint", "onnx", "ov_fp16", "ov_int8")
# The validation split gets the same treatment, keyed by its own CSV hash. It is
# what makes val-fitted thresholds and the val→test gap available without
# re-running anything.
VAL_SPLIT = "val"


def hash8(test_set_hash: str) -> str:
    """Short, filesystem-safe form of a test-set hash.

    Accepts the stored ``sha256:abcd…`` spelling as well as a bare digest.
    """
    raw = str(test_set_hash or "").strip()
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    safe = "".join(ch for ch in raw if ch.isalnum())
    return (safe[:8] or "unknown").lower()


def score_key(engine: str, test_set_hash: str) -> str:
    """Storage key for one scored (engine, test set) pair."""
    return f"{_safe_engine(engine)}__{hash8(test_set_hash)}"


def _safe_engine(engine: str) -> str:
    name = str(engine or "checkpoint").strip().lower()
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name) or "checkpoint"


def scores_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "reports" / SCORES_DIRNAME


def _paths_for(run_dir: Path, key: str) -> tuple[Path, Path]:
    base = scores_dir(run_dir)
    return base / f"{key}.npz", base / f"{key}.json"


def write_scores(
    run_dir: Path,
    *,
    engine: str,
    test_set_hash: str,
    classes: list[str],
    probs: np.ndarray,
    true_idx: np.ndarray,
    paths: list[str],
    probability_mode: str = "",
    ir_precision: str | None = None,
    checkpoint_sha256: str = "",
    class_mapping: dict[str, str] | None = None,
    split: str = "test",
) -> str:
    """Persist one scored pass and return its storage key.

    ``true_idx`` is the ground-truth class index per clip, not a one-hot matrix:
    the frozen test is single-label, and an index array is a tenth the size.
    """
    key = score_key(engine, test_set_hash)
    npz_path, json_path = _paths_for(run_dir, key)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    probs = np.asarray(probs, dtype=np.float32)
    true_idx = np.asarray(true_idx, dtype=np.int16)
    if probs.ndim != 2 or probs.shape[1] != len(classes):
        raise ValueError(
            f"probs must be (N, {len(classes)}) for classes {classes}, got {probs.shape}"
        )
    if probs.shape[0] != true_idx.shape[0]:
        raise ValueError(f"probs has {probs.shape[0]} rows but true_idx has {true_idx.shape[0]}")

    np.savez_compressed(npz_path, probs=probs, true_idx=true_idx)
    sidecar = {
        "key": key,
        "split": split,
        "engine": _safe_engine(engine),
        "ir_precision": ir_precision,
        "test_set_hash": str(test_set_hash or ""),
        "classes": list(classes),
        "n": int(probs.shape[0]),
        "probability_mode": str(probability_mode or ""),
        "checkpoint_sha256": str(checkpoint_sha256 or ""),
        "class_mapping": dict(class_mapping or {}),
        "paths": [str(p) for p in paths],
        "scored_at": datetime.now(UTC).isoformat(),
        "source": "scored",
    }
    json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    log.info("scores → %s (%d clips, %d classes)", npz_path.name, probs.shape[0], len(classes))
    return key


def read_sidecar(run_dir: Path, key: str) -> dict | None:
    """The metadata of one stored pass, without loading the probabilities."""
    _npz, json_path = _paths_for(run_dir, key)
    if not json_path.is_file():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("score sidecar unreadable %s: %s", json_path, exc)
        return None
    return data if isinstance(data, dict) else None


def read_scores(run_dir: Path, key: str) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Load ``(probs, true_idx, sidecar)`` for one stored pass, or None."""
    npz_path, _json = _paths_for(run_dir, key)
    sidecar = read_sidecar(run_dir, key)
    if sidecar is None or not npz_path.is_file():
        return None
    try:
        with np.load(npz_path) as data:
            probs = np.asarray(data["probs"], dtype=np.float32)
            true_idx = np.asarray(data["true_idx"], dtype=np.int64)
    except Exception as exc:  # noqa: BLE001
        log.warning("score matrix unreadable %s: %s", npz_path, exc)
        return None
    return probs, true_idx, sidecar


def one_hot(true_idx: np.ndarray, n_classes: int) -> np.ndarray:
    """Ground-truth indices back to the one-hot matrix the metric core expects."""
    true_idx = np.asarray(true_idx, dtype=np.int64)
    out = np.zeros((true_idx.shape[0], int(n_classes)), dtype=np.float32)
    if true_idx.size:
        valid = (true_idx >= 0) & (true_idx < int(n_classes))
        out[np.nonzero(valid)[0], true_idx[valid]] = 1.0
    return out


def available(run_dir: Path) -> dict[str, dict]:
    """Every stored pass for a run, keyed by storage key."""
    base = scores_dir(run_dir)
    if not base.is_dir():
        return {}
    out: dict[str, dict] = {}
    for json_path in sorted(base.glob("*.json")):
        key = json_path.stem
        sidecar = read_sidecar(run_dir, key)
        if sidecar is not None and (base / f"{key}.npz").is_file():
            out[key] = sidecar
    return out


def manifest_entry(sidecar: dict) -> dict:
    """The compact form recorded in the run manifest (no per-clip paths)."""
    return {
        k: sidecar.get(k)
        for k in (
            "key", "split", "engine", "ir_precision", "test_set_hash",
            "classes", "n", "probability_mode", "scored_at", "source",
        )
    }


def register(manifest: dict, sidecar: dict) -> dict:
    """Record a stored pass in the manifest's ``frozen_scores`` section."""
    section = manifest.setdefault(MANIFEST_SECTION, {})
    if isinstance(section, dict):
        section[str(sidecar.get("key"))] = manifest_entry(sidecar)
    return manifest


# --- Backfill from existing prediction tables -----------------------------
# Every run scored before the score store existed already wrote every class
# probability into `reports/<rule>_predictions.tsv`. Lifting those columns
# recovers the full run history with no GPU time — the one caveat being that the
# TSV writer rounds to six decimals, which can shift an AP tie-break but cannot
# move a decision boundary.

PREDICTION_SUFFIX = "_predictions.tsv"
FLAT_PREDICTION_NAME = "test_predictions.tsv"


def prediction_reports(run_dir: Path) -> list[Path]:
    """Prediction tables in a run, newest-looking first (rule slugs before the flat one)."""
    reports = Path(run_dir) / "reports"
    if not reports.is_dir():
        return []
    rule_tables = sorted(
        p for p in reports.glob(f"*{PREDICTION_SUFFIX}") if p.name != FLAT_PREDICTION_NAME
    )
    flat = reports / FLAT_PREDICTION_NAME
    return rule_tables + ([flat] if flat.is_file() else [])


def _metrics_beside(report: Path) -> dict:
    """The `_metrics.json` written alongside a prediction table."""
    name = (
        "test_metrics.json"
        if report.name == FLAT_PREDICTION_NAME
        else report.name.replace(PREDICTION_SUFFIX, "_metrics.json")
    )
    candidate = report.with_name(name)
    if not candidate.is_file():
        return {}
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def scores_from_prediction_table(report: Path) -> dict | None:
    """Read a prediction TSV back into a probability matrix.

    Returns ``None`` when the table predates per-class probability columns.
    """
    try:
        with open(report, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
    except Exception as exc:  # noqa: BLE001
        log.warning("prediction table unreadable %s: %s", report, exc)
        return None
    if not rows:
        return None
    prob_cols = [c for c in (rows[0].keys() or []) if c and c.startswith("p:")]
    if not prob_cols:
        return None

    classes = [c[2:] for c in prob_cols]
    index = {name: i for i, name in enumerate(classes)}
    probs = np.zeros((len(rows), len(classes)), dtype=np.float32)
    true_idx = np.full((len(rows),), -1, dtype=np.int16)
    paths: list[str] = []
    for r, row in enumerate(rows):
        for c, col in enumerate(prob_cols):
            try:
                probs[r, c] = float(row.get(col) or 0.0)
            except (TypeError, ValueError):
                probs[r, c] = 0.0
        true_idx[r] = index.get(str(row.get("true_class") or ""), -1)
        paths.append(str(row.get("path") or ""))

    metrics = _metrics_beside(report)
    return {
        "classes": classes,
        "probs": probs,
        "true_idx": true_idx,
        "paths": paths,
        "test_set_hash": str(metrics.get("test_set_hash") or ""),
        "probability_mode": str(metrics.get("probability_mode") or ""),
        "class_mapping": metrics.get("class_mapping") or {},
    }


def backfill_run(run_dir: Path, *, overwrite: bool = False) -> dict:
    """Recover a score store for one run from its stored prediction tables.

    Uses the first table that carries probability columns *and* a test-set hash;
    every rule of a run shares the same probabilities, so one is enough.
    """
    run_dir = Path(run_dir)
    result: dict = {"run": run_dir.name, "written": None, "reason": ""}
    reports = prediction_reports(run_dir)
    if not reports:
        result["reason"] = "no prediction table"
        return result

    for report in reports:
        parsed = scores_from_prediction_table(report)
        if parsed is None:
            continue
        test_hash = parsed["test_set_hash"]
        if not test_hash:
            result["reason"] = f"{report.name} has no test_set_hash — comparability unknown"
            continue
        key = score_key("checkpoint", test_hash)
        if not overwrite and read_sidecar(run_dir, key) is not None:
            result["written"] = key
            result["reason"] = "already present"
            return result
        write_scores(
            run_dir,
            engine="checkpoint",
            test_set_hash=test_hash,
            classes=parsed["classes"],
            probs=parsed["probs"],
            true_idx=parsed["true_idx"],
            paths=parsed["paths"],
            probability_mode=parsed["probability_mode"],
            class_mapping=parsed["class_mapping"],
        )
        # Mark the provenance: these probabilities came back through a
        # six-decimal text round-trip, so an AP recomputed from them can differ
        # in the last places from the one stored at scoring time.
        _npz, json_path = _paths_for(run_dir, key)
        sidecar = read_sidecar(run_dir, key) or {}
        sidecar["source"] = "backfilled"
        sidecar["backfilled_from"] = report.name
        json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        result["written"] = key
        result["reason"] = f"from {report.name}"
        return result

    if not result["reason"]:
        result["reason"] = "prediction tables carry no per-class probability columns"
    return result
