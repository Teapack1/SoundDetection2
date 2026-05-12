"""Named, versioned frozen-test sets.

The research workflow needs to build a benchmark *after* models exist: gather
new test material, pick the classes to measure, then re-score the whole run
history against it. That requires a test set to be a first-class named object
rather than whatever `test.csv` happens to hold right now — otherwise every
rebuild silently redefines what past scores meant.

A preset is stored as a CSV in the same shape `compose` writes (so it can be
swapped into `test.csv` verbatim) plus a row in a small JSON index. Keeping the
clips in CSV rather than inlined in the index matters at this corpus size: the
pretraining frozen test is 42k rows.

The leakage remedy lives here too, as `derive_preset_excluding`: dropping the
clips a run trained on produces a **new** preset rather than editing the live
test set, so a rebuild is always additive and the old benchmark stays
reproducible for the scores that were computed against it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from sed.config import ProjectLayout

log = logging.getLogger(__name__)

FROZEN_TEST_PRESETS_FILENAME = "frozen_test_presets.json"
FROZEN_TEST_DIRNAME = "frozen_tests"

# Larger than any real frozen test; guards against a runaway CSV read.
_MAX_ROWS = 5_000_000


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _slugify(name: str) -> str:
    from sed.data.presets import slugify_preset_name

    return slugify_preset_name(name)


def presets_path(layout: ProjectLayout) -> Path:
    return layout.dataset_dir / FROZEN_TEST_PRESETS_FILENAME


def presets_dir(layout: ProjectLayout) -> Path:
    return layout.dataset_dir / FROZEN_TEST_DIRNAME


def frozen_test_hash(rows: list[dict]) -> str:
    """Hash a test set from CSV rows.

    Byte-identical to `compose._test_set_hash` for the same clips — the stored
    `test_set_hash` on every manifest block is what tells two runs they were
    scored on the same ground truth, so this must not drift from it.
    """
    pairs = sorted((str(r.get("class_name", "")), str(r.get("path", ""))) for r in rows)
    h = hashlib.sha256()
    for cls, path in pairs:
        h.update(cls.encode("utf-8"))
        h.update(b"\0")
        h.update(path.encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


def read_test_csv(path: Path) -> tuple[list[str], list[dict]]:
    """(fieldnames, rows) of a composed test CSV; ([], []) when absent."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = []
            for i, row in enumerate(reader):
                if i >= _MAX_ROWS:
                    log.warning("test csv %s exceeds %d rows — truncated", path, _MAX_ROWS)
                    break
                if row.get("path"):
                    rows.append(row)
            return list(reader.fieldnames or []), rows
    except (OSError, csv.Error):
        return [], []


def write_test_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _by_class(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        name = str(row.get("class_name") or "")
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _empty_index() -> dict:
    return {"schema_version": 1, "presets": []}


def load_presets(layout: ProjectLayout) -> dict:
    path = presets_path(layout)
    if not path.exists():
        return _empty_index()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to read frozen-test presets %s: %s", path, exc)
        return _empty_index()
    if not isinstance(payload, dict) or not isinstance(payload.get("presets"), list):
        return _empty_index()
    payload.setdefault("schema_version", 1)
    return payload


def write_presets(layout: ProjectLayout, payload: dict) -> Path:
    out = presets_path(layout)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def find_preset(payload: dict, name_or_slug: str) -> dict | None:
    wanted = str(name_or_slug or "").strip()
    slug = _slugify(wanted)
    for record in payload.get("presets", []):
        if isinstance(record, dict) and str(record.get("slug") or "") in (wanted, slug):
            return record
    return None


def save_frozen_test_preset(
    layout: ProjectLayout,
    name: str,
    *,
    rows: list[dict] | None = None,
    fieldnames: list[str] | None = None,
    notes: str = "",
    derived_from: str = "",
    excluded_paths: list[str] | None = None,
) -> dict:
    """Snapshot a clip list as a named test set.

    With no ``rows`` the project's current ``test.csv`` is captured, which is
    how an existing benchmark gets a name before anything else touches it.
    """
    display = str(name or "").strip()
    if not display:
        raise ValueError("preset name is required")
    slug = _slugify(display)

    if rows is None:
        fieldnames, rows = read_test_csv(layout.dataset_dir / "test.csv")
    if not rows:
        raise ValueError("no test clips to save — compose a frozen test first")
    fieldnames = list(fieldnames or list(rows[0].keys()))

    csv_path = presets_dir(layout) / f"{slug}.csv"
    write_test_csv(csv_path, fieldnames, rows)

    payload = load_presets(layout)
    existing = find_preset(payload, slug)
    record = {
        "name": display,
        "slug": slug,
        "file": csv_path.name,
        "n_clips": len(rows),
        "by_class": _by_class(rows),
        "classes": sorted(_by_class(rows)),
        "hash": frozen_test_hash(rows),
        "notes": str(notes or ""),
        "derived_from": str(derived_from or ""),
        "excluded_count": len(excluded_paths or []),
        "created_at": str((existing or {}).get("created_at") or "") or _now_iso(),
        "updated_at": _now_iso(),
    }
    if existing is not None:
        payload["presets"] = [record if r is existing else r for r in payload.get("presets", [])]
    else:
        payload.setdefault("presets", []).append(record)
    write_presets(layout, payload)
    return record


def preset_rows(layout: ProjectLayout, name_or_slug: str) -> tuple[list[str], list[dict]]:
    record = find_preset(load_presets(layout), name_or_slug)
    if record is None:
        raise KeyError(f"frozen-test preset '{name_or_slug}' not found")
    return read_test_csv(presets_dir(layout) / str(record.get("file") or ""))


def apply_frozen_test_preset(layout: ProjectLayout, name_or_slug: str) -> dict:
    """Make a preset the project's live frozen test.

    Rewrites ``test.csv`` and the ``frozen_test`` block of
    ``dataset_snapshot.json`` so the runs table's "scored on an older test set"
    staleness logic keeps working — a changed test set MUST invalidate the
    comparability of scores computed against the previous one, and the hash is
    how that is detected.
    """
    payload = load_presets(layout)
    record = find_preset(payload, name_or_slug)
    if record is None:
        raise KeyError(f"frozen-test preset '{name_or_slug}' not found")

    fieldnames, rows = read_test_csv(presets_dir(layout) / str(record.get("file") or ""))
    if not rows:
        raise ValueError(f"preset '{name_or_slug}' has no clips on disk")

    write_test_csv(layout.dataset_dir / "test.csv", fieldnames, rows)

    snapshot_path = layout.dataset_dir / "dataset_snapshot.json"
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a missing snapshot is not fatal here
        snapshot = {}
    frozen = dict(snapshot.get("frozen_test") or {})
    frozen.update({
        "present": True,
        "total": len(rows),
        "by_class": _by_class(rows),
        "hash": record.get("hash") or frozen_test_hash(rows),
        "source": "preset",
        "preset": record.get("slug", ""),
    })
    snapshot["frozen_test"] = frozen
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return record


def derive_preset_excluding(
    layout: ProjectLayout,
    new_name: str,
    excluded_paths: list[str],
    *,
    source: str = "",
    notes: str = "",
) -> dict:
    """A copy of a test set with the given clips removed — the leakage remedy.

    Additive by design: the source test set is left exactly as it was, so scores
    already computed against it stay reproducible. Nothing becomes the live test
    until `apply_frozen_test_preset` is called.
    """
    from sed.train.leakage import normalize_clip_path

    if source:
        fieldnames, rows = preset_rows(layout, source)
    else:
        fieldnames, rows = read_test_csv(layout.dataset_dir / "test.csv")
    if not rows:
        raise ValueError("no test clips to derive from")

    drop = {normalize_clip_path(p) for p in (excluded_paths or [])}
    kept = [r for r in rows if normalize_clip_path(r.get("path", "")) not in drop]
    removed = len(rows) - len(kept)
    if not kept:
        raise ValueError(
            f"removing {removed} clip(s) would empty the test set — "
            "the runs share too much training data with it to benchmark this way"
        )
    return save_frozen_test_preset(
        layout, new_name, rows=kept, fieldnames=fieldnames,
        notes=notes or f"derived by removing {removed} leaked clip(s)",
        derived_from=source or "current test.csv",
        excluded_paths=list(excluded_paths or []),
    )


def delete_frozen_test_preset(layout: ProjectLayout, name_or_slug: str) -> bool:
    payload = load_presets(layout)
    record = find_preset(payload, name_or_slug)
    if record is None:
        return False
    payload["presets"] = [r for r in payload.get("presets", []) if r is not record]
    write_presets(layout, payload)
    (presets_dir(layout) / str(record.get("file") or "")).unlink(missing_ok=True)
    return True


def frozen_test_preset_state(layout: ProjectLayout) -> dict:
    """Preset list for the UI, with which one matches the live test set."""
    payload = load_presets(layout)
    _fields, current = read_test_csv(layout.dataset_dir / "test.csv")
    current_hash = frozen_test_hash(current) if current else ""
    presets = [
        {
            "name": str(r.get("name") or ""),
            "slug": str(r.get("slug") or ""),
            "n_clips": int(r.get("n_clips") or 0),
            "classes": list(r.get("classes") or []),
            "by_class": dict(r.get("by_class") or {}),
            "hash": str(r.get("hash") or ""),
            "notes": str(r.get("notes") or ""),
            "derived_from": str(r.get("derived_from") or ""),
            "excluded_count": int(r.get("excluded_count") or 0),
            "updated_at": str(r.get("updated_at") or ""),
            "is_current": str(r.get("hash") or "") == current_hash and bool(current_hash),
        }
        for r in payload.get("presets", [])
        if isinstance(r, dict)
    ]
    presets.sort(key=lambda p: p["name"].lower())
    return {
        "presets": presets,
        "current_hash": current_hash,
        "current_total": len(current),
        # True when the live test set has no named counterpart — i.e. it cannot
        # be restored after the next rebuild.
        "current_unsaved": bool(current) and not any(p["is_current"] for p in presets),
    }
