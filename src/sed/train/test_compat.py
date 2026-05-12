"""Can these runs be scored on this frozen test — and would the answer be fair?

Batch evaluate scores a set of runs on the *current* frozen test so their
numbers are comparable. Two things can make that impossible or misleading, and
both currently surface only as a failed job or not at all:

  * **class-space mismatch** — a run has no output for a class it never trained
    on. This is why the pretraining project's runs (`_negative/cheers/attention`)
    cannot be scored against its present `_negative/loud/medium/quiet` dataset:
    41 runs of history orphaned by one taxonomy change. A **class mapping**
    folds the newer labels onto the older space (`loud|medium|quiet -> cheers`)
    so a refinement of a taxonomy does not throw the history away.
  * **leakage** — the test contains clips the run trained on (see
    `sed.train.leakage`). The score comes out high and looks legitimate.

This module answers both *before* a job runs and says what would fix each, so
the UI can offer a remedy instead of a traceback. Torch-free: it is called from
a request handler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sed.train.leakage import find_leakage, read_clip_rows, training_index

# Worst first — a run's verdict is its most severe finding.
SEVERITY_ORDER = ("error", "warn", "info")


@dataclass(frozen=True)
class ClassMapping:
    """Fold test-set class names onto the space a run was trained on.

    Only ever *merges* labels (a finer taxonomy onto a coarser one). It cannot
    invent a distinction a run was never trained to make, so a mapping never
    flatters a model — it lets an older, coarser run be scored on a newer test
    at the granularity it actually supports.
    """

    mapping: dict[str, str]

    def apply(self, class_name: str) -> str:
        return self.mapping.get(class_name, class_name)

    def map_rows(self, rows: list[dict]) -> list[dict]:
        out = []
        for row in rows:
            name = row.get("class_name", "")
            mapped = self.apply(name)
            out.append(row if mapped == name else {**row, "class_name": mapped})
        return out

    @property
    def is_empty(self) -> bool:
        return not any(k != v for k, v in self.mapping.items())


def parse_class_mapping(raw: dict | None) -> ClassMapping:
    """Build a mapping from a ``{test_class: run_class}`` dict."""
    return ClassMapping({str(k): str(v) for k, v in (raw or {}).items() if k and v})


def suggest_class_mapping(test_classes: set[str], run_classes: list[str]) -> dict[str, str]:
    """Best-effort mapping proposal for classes a run cannot score.

    Only suggests where the intent is unambiguous from the names — a
    ``<target>_<qualifier>`` split of a class the run does have (``cheers_quiet``
    -> ``cheers``), or a prefix match. Anything else is left for a human: a wrong
    merge silently changes what is being measured.
    """
    known = set(run_classes)
    suggestion: dict[str, str] = {}
    for name in sorted(test_classes - known):
        if name == "_negative":
            continue
        for run_class in sorted(known, key=len, reverse=True):
            if run_class == "_negative":
                continue
            if name.startswith(f"{run_class}_") or name.split("_")[0] == run_class:
                suggestion[name] = run_class
                break
    return suggestion


def check_run(
    test_rows: list[dict],
    model_dir: Path,
    run_id: str,
    run_classes: list[str],
    *,
    mapping: ClassMapping | None = None,
) -> dict:
    """Pre-flight one run against a candidate frozen test.

    Returns findings plus a machine-readable ``remedy`` per finding, so the UI
    can offer the fix rather than describing the problem.
    """
    mapping = mapping or ClassMapping({})
    rows = mapping.map_rows(test_rows) if not mapping.is_empty else test_rows
    test_classes = {r.get("class_name", "") for r in rows if r.get("class_name")}
    known = set(run_classes)

    findings: list[dict] = []

    unscorable = sorted(test_classes - known)
    if unscorable:
        findings.append({
            "severity": "error", "code": "class_space",
            "message": (
                f"The test set contains {unscorable}, which this run was never trained on "
                f"(its classes are {sorted(known)}). It has no output for them and cannot "
                "be scored."
            ),
            "classes": unscorable,
            "remedy": {
                "kind": "class_mapping",
                "label": "Map these test classes onto the run's classes",
                "suggestion": suggest_class_mapping(test_classes, run_classes),
            },
        })

    uncovered = sorted(c for c in known - test_classes if c != "_negative")
    if uncovered:
        findings.append({
            "severity": "warn", "code": "class_coverage",
            "message": (
                f"The run has target class(es) {uncovered} with no clips in this test set. "
                "They will contribute no recall, and macro averages will describe only the "
                "classes present."
            ),
            "classes": uncovered,
            "remedy": {"kind": "add_test_clips", "label": "Add test clips for these classes"},
        })

    index = training_index(model_dir, run_id)
    leak = find_leakage(test_rows, index)
    if not index.available:
        findings.append({
            "severity": "warn", "code": "no_dataset_copy",
            "message": (
                "This run kept no dataset copy, so leakage cannot be checked. Its score on "
                "a rebuilt test set is unverified, not verified-clean."
            ),
            "remedy": None,
        })
    elif leak["leak_count"]:
        parts = []
        if leak["clip_leak_count"]:
            parts.append(f"{leak['clip_leak_count']} identical clip(s)")
        if leak["group_leak_count"]:
            parts.append(
                f"{leak['group_leak_count']} clip(s) from a recording this run trained on"
            )
        findings.append({
            "severity": "error", "code": "leakage",
            "message": (
                f"{leak['leak_count']} of {leak['n_test_clips']} test clips "
                f"({leak['leak_fraction']:.0%}) were in this run's training data — "
                + " and ".join(parts)
                + ". Its score on this test set would be inflated."
            ),
            "remedy": {
                "kind": "drop_leaked_clips",
                "label": f"Remove {leak['leak_count']} leaked clip(s) from the frozen test",
                "paths": leak["leaked_paths"],
            },
        })

    ckpt = any(Path(model_dir).glob("checkpoint_*.ckpt"))
    if not ckpt:
        findings.append({
            "severity": "error", "code": "no_checkpoint",
            "message": "No checkpoint in the run directory — nothing to score.",
            "remedy": None,
        })

    findings.sort(key=lambda f: SEVERITY_ORDER.index(f["severity"]))
    severities = {f["severity"] for f in findings}
    if "error" in severities:
        verdict = "fixable" if all(
            f.get("remedy") for f in findings if f["severity"] == "error"
        ) else "incompatible"
    else:
        verdict = "ok"

    return {
        "run_id": run_id,
        "verdict": verdict,
        "classes": sorted(known),
        "findings": findings,
        "leakage": {
            k: leak[k] for k in
            ("clip_leak_count", "group_leak_count", "leak_count", "leak_fraction", "available")
        },
    }


def check_runs(
    test_csv: Path,
    runs: list[dict],
    *,
    mapping: ClassMapping | None = None,
) -> dict:
    """Pre-flight a batch. ``runs`` need ``id``, ``model_dir`` and ``classes``.

    The union of leaked clips across the batch is reported separately: a shared
    benchmark has to be clean for every model it ranks, so the remedy drops the
    union rather than per-run subsets.
    """
    test_rows = read_clip_rows(Path(test_csv))
    results = [
        check_run(
            test_rows, Path(r["model_dir"]), str(r["id"]), list(r.get("classes") or []),
            mapping=mapping,
        )
        for r in runs
    ]
    leaked_union: set[str] = set()
    for run, result in zip(runs, results, strict=True):
        for finding in result["findings"]:
            remedy = finding.get("remedy") or {}
            if remedy.get("kind") == "drop_leaked_clips":
                leaked_union.update(remedy.get("paths") or [])

    counts = {"ok": 0, "fixable": 0, "incompatible": 0}
    for result in results:
        counts[result["verdict"]] += 1
    return {
        "n_test_clips": len(test_rows),
        "runs": results,
        "counts": counts,
        "leaked_paths_union": sorted(leaked_union),
        # Deliberately NOT "ok": this dict is spread into a JSON envelope whose
        # own "ok" means "the request succeeded". A clean check and a successful
        # request are different questions, and one silently overwrote the other.
        "all_clear": counts["fixable"] == 0 and counts["incompatible"] == 0,
    }
