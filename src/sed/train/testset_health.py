"""Is a frozen test set capable of answering the question asked of it?

Two failures made model selection on this project uninterpretable, and neither
was visible from any score:

  * a test set with **no `_negative` clips** cannot measure false positives at
    all, so every false-alarm metric silently reads `None`;
  * a test set with **~24 clips per target class** cannot separate models whose
    recall differs by a few points, so single-seed rankings over such a set are
    noise dressed as a result.

Both are properties of the test set alone — no model, no scoring run — so they
can be checked before spending GPU time, and re-checked whenever a test set is
rebuilt. Deliberately torch-free: the inspect server reads this on page load.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

# 95% two-sided normal quantile, for Wilson intervals.
_Z = 1.959963984540054

# Below this many clips a per-class recall estimate is too wide to rank models.
LOW_SUPPORT_CLIPS = 100
# The operating recall the reported resolution is quoted at. This product cares
# about the high-recall region, and interval width depends on where you sit.
RESOLUTION_AT_RECALL = 0.9


def wilson_halfwidth(p: float, n: int, z: float = _Z) -> float | None:
    """Half-width of the Wilson score interval for a proportion.

    Wilson rather than the normal approximation because the counts here are
    small and the proportions near 1.0, exactly where the normal approximation
    understates the interval (and can stray above 1).
    """
    if n <= 0:
        return None
    denom = 1.0 + z * z / n
    return (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom


def read_class_counts(test_csv: Path) -> dict[str, int]:
    """Clip count per class in a composed test.csv."""
    counts: dict[str, int] = {}
    with open(test_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = str(row.get("class_name") or "").strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


def frozen_test_health(counts: dict[str, int]) -> dict:
    """Grade a frozen test set from its per-class clip counts.

    Returns per-class support with the recall resolution each class can
    actually achieve, the negative prior every prior-bound metric depends on,
    and plain-language findings ordered worst first.
    """
    counts = {str(k): int(v) for k, v in (counts or {}).items()}
    total = sum(counts.values())
    n_negative = counts.get("_negative", 0)
    targets = {k: v for k, v in counts.items() if k != "_negative"}

    per_class = {
        name: {
            "support": n,
            # ± percentage points on this class's recall, at the high-recall
            # operating point the product actually runs at.
            "recall_ci_halfwidth": wilson_halfwidth(RESOLUTION_AT_RECALL, n),
        }
        for name, n in sorted(counts.items())
    }

    findings: list[dict] = []
    if total == 0:
        findings.append({
            "severity": "error", "code": "empty",
            "message": "The test set is empty — nothing can be scored.",
        })
    if n_negative == 0:
        findings.append({
            "severity": "error", "code": "no_negatives",
            "message": (
                "No `_negative` clips: false positives cannot be measured at all. "
                "Every false-alarm metric will read as missing, and reported "
                "precision only describes confusion between target classes."
            ),
        })
    if not targets:
        findings.append({
            "severity": "error", "code": "no_targets",
            "message": "No target classes — recall cannot be measured.",
        })

    thin = {n: c for n, c in targets.items() if c < LOW_SUPPORT_CLIPS}
    if thin:
        worst = min(thin.values())
        halfwidth = wilson_halfwidth(RESOLUTION_AT_RECALL, worst)
        findings.append({
            "severity": "warn", "code": "low_support",
            "message": (
                f"{len(thin)} target class(es) have under {LOW_SUPPORT_CLIPS} clips "
                f"(smallest {worst}). At {RESOLUTION_AT_RECALL:.0%} recall that class "
                f"resolves to ±{halfwidth * 100:.0f} points, so model differences "
                f"smaller than about {2 * halfwidth * 100:.0f} points are not "
                "distinguishable on this test set."
            ),
        })

    prior = (n_negative / total) if total else None
    if prior is not None and total > 0:
        findings.append({
            "severity": "info", "code": "prior",
            "message": (
                f"Negative prior is {prior:.1%}. Precision, F1, accuracy and AP are "
                "only comparable against runs scored at this same prior; recall and "
                "false-alarms-per-1000-negatives are not affected by it."
            ),
        })

    order = {"error": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda f: order.get(f["severity"], 3))
    return {
        "n_samples": total,
        "n_negatives": n_negative,
        "negative_prior": prior,
        "n_target_classes": len(targets),
        "n_target_clips": sum(targets.values()),
        "per_class": per_class,
        "findings": findings,
        "can_measure_false_positives": n_negative > 0,
        "ok": not any(f["severity"] == "error" for f in findings),
    }


def health_for_test_csv(test_csv: Path) -> dict:
    """`test_set_health` for a composed test.csv on disk."""
    if not Path(test_csv).exists():
        return {
            "n_samples": 0, "n_negatives": 0, "negative_prior": None,
            "n_target_classes": 0, "n_target_clips": 0, "per_class": {},
            "can_measure_false_positives": False, "ok": False,
            "findings": [{
                "severity": "error", "code": "missing",
                "message": f"No test set at {test_csv} — compose one first.",
            }],
        }
    return frozen_test_health(read_class_counts(Path(test_csv)))
