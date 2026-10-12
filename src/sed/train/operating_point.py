"""Choosing and comparing operating points on a clip corpus.

A single F1 describes one threshold. The deployment question is a *curve*: how
much recall is available at a false-alarm budget the venue will accept. This
module sweeps thresholds over already-computed probabilities and summarises the
curve, so two models can be ranked by the region they will actually be operated
in rather than by a threshold one of them happened to suit.

Everything is expressed per 1000 clips because that is the denominator the
recorded material supports: negatives are captured as onset-triggered batches
while the room is busy, so "clips" is observable and "hours" is not.

Recall is macro-averaged over target classes. The classes are distinct events
that drive distinct responses, so a `loud` clip heard as `medium` is a miss for
`loud` — not a detection.

Deliberately free of torch — this runs over a probability matrix that the
evaluator has already produced.
"""

from __future__ import annotations

import math

# The band a venue product lives in: from a very strict 0.1 false fires per 1000
# clips up to a permissive 20. Integrated in log10 because linear spacing would
# give 15→20 many times the weight of 0.1→1, which is backwards for a budget.
BAND_LO_PER_1000 = 0.1
BAND_HI_PER_1000 = 20.0

# Coarse anchors, kept so a curve is still legible when plotted, plus the
# high-confidence tail. A strict false-alarm budget is reached at 0.97–0.999 on
# a confident model, and a grid that stops at 0.95 simply cannot express it —
# which is why band-pAUC used to read 0.0 for models that had a perfectly good
# operating point just past the end of the sweep.
_ANCHORS: tuple[float, ...] = tuple(
    [round(0.05 * i, 2) for i in range(1, 20)]
    + [0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.998, 0.999]
)
# Upper bound on sweep points. The interesting resolution is around the negative
# score distribution, so the rest of the budget goes to quantiles of that.
MAX_GRID_POINTS = 256


def _default_grid(win_val, is_negative) -> list[float]:
    """Thresholds worth evaluating, derived from the scores themselves.

    The anchors keep the curve comparable and plottable; the quantiles of the
    *negative* winner scores put resolution exactly where the false-alarm rate
    changes, so a budget of 0.5 per 1000 lands on a real operating point instead
    of being rounded away by a fixed 0.05 step.
    """
    import numpy as np

    grid = list(_ANCHORS)
    neg_scores = np.asarray(win_val)[np.asarray(is_negative, dtype=bool)]
    if neg_scores.size:
        n_q = max(0, MAX_GRID_POINTS - len(grid))
        if n_q > 0:
            qs = np.quantile(neg_scores, np.linspace(0.0, 1.0, min(n_q, neg_scores.size)))
            grid.extend(float(q) for q in qs)
    # Round before de-duplicating: float noise would otherwise leave hundreds of
    # thresholds that separate nothing.
    out = sorted({round(float(t), 6) for t in grid if 0.0 <= float(t) <= 1.0})
    return out[:MAX_GRID_POINTS]


def sweep(
    probs,
    targets,
    classes: list[str],
    *,
    negative_max: float,
    margin: float,
    grid: list[float] | None = None,
    roles: dict[str, str] | None = None,
) -> dict:
    """Recall and false-alarm rate across a shared threshold sweep.

    ``probs`` / ``targets`` are the (N, C) matrices the evaluator already built.
    One shared threshold moves across all target classes, so the curve describes
    the system rather than a per-class operating point.

    ``negative_max`` and ``margin`` are the other two runtime gates, applied
    exactly as the live decision applies them so that a point on this curve and
    the point metrics at the same threshold agree.

    ``roles`` (``Config.class_roles`` output; defaults to "``_negative`` is the
    only negative-role class, everything else is a target") decides two
    separate sets: every ``"negative"``-role class is excluded from window
    competition and must clear ``negative_max`` (mirroring the runtime veto
    gate), while a clip whose *true* label is any non-``"target"`` class
    (negative or suppressor) counts as a false-alarm opportunity — a
    suppressor-labelled clip is never a real detection, so it must not inflate
    "correct", matching the deployment `false_fire_rate` definition.
    """
    import numpy as np

    probs = np.asarray(probs)
    targets = np.asarray(targets)
    if probs.size == 0:
        return {"points": [], "n_negatives": 0, "n_targets": 0, "classes": []}

    if roles is None:
        roles = {c: ("negative" if c == "_negative" else "target") for c in classes}
    negative_idx = [i for i, c in enumerate(classes) if roles.get(c) == "negative"]
    non_target_idx = {i for i, c in enumerate(classes) if roles.get(c) != "target"}
    target_idx = [i for i in range(len(classes)) if i not in negative_idx]
    if not target_idx:
        return {"points": [], "n_negatives": 0, "n_targets": 0, "classes": []}

    true_idx = np.argmax(targets, axis=1)
    is_negative = (
        np.isin(true_idx, list(non_target_idx)) if non_target_idx else np.zeros(len(probs), bool)
    )
    n_neg = int(is_negative.sum())
    n_pos = int((~is_negative).sum())

    tgt = probs[:, target_idx]
    order = np.argsort(-tgt, axis=1)
    win_col = order[:, 0]
    win_val = np.take_along_axis(tgt, win_col[:, None], axis=1)[:, 0]
    # Gates that do not depend on the swept threshold, applied once.
    base_ok = np.ones(len(probs), bool)
    if negative_idx:
        base_ok &= (probs[:, negative_idx] <= negative_max).all(axis=1)
    # With a single target class there is no runner-up, and the live decision
    # skips the margin gate entirely — comparing the winner against a phantom
    # 0.0 would silently re-impose it as a second confidence threshold.
    if tgt.shape[1] > 1 and margin > 0:
        run_val = np.take_along_axis(tgt, order[:, 1][:, None], axis=1)[:, 0]
        base_ok &= (win_val - run_val) >= margin

    winner_class_idx = np.asarray(target_idx)[win_col]
    target_names = [classes[i] for i in target_idx]
    class_support = {
        classes[i]: int((true_idx == i).sum()) for i in target_idx
    }

    points = []
    for thr in (grid if grid is not None else _default_grid(win_val, is_negative)):
        fires = base_ok & (win_val >= thr)
        correct = fires & ~is_negative & (winner_class_idx == true_idx)
        false_fires = int((fires & is_negative).sum())
        per_class = {}
        for i in target_idx:
            support = class_support[classes[i]]
            per_class[classes[i]] = (
                float((correct & (true_idx == i)).sum() / support) if support else None
            )
        macro = [v for v in per_class.values() if v is not None]
        points.append({
            "threshold": float(thr),
            # Pooled over every target clip — the detection rate.
            "recall": float(correct.sum() / n_pos) if n_pos else None,
            # Averaged over classes, so a thin class counts as much as a fat one.
            # This is the headline: it is what the runs table ranks on.
            "recall_macro": (sum(macro) / len(macro)) if macro else None,
            "recall_per_class": per_class,
            "false_per_1000": (false_fires / n_neg * 1000.0) if n_neg else None,
            "n_false": false_fires,
        })
    return {
        "points": points,
        "n_negatives": n_neg,
        "n_targets": n_pos,
        "classes": target_names,
        "class_support": class_support,
    }


def _usable(curve: dict, key: str = "recall_macro") -> list[dict]:
    """Curve points that carry both a recall and a false-alarm rate."""
    return [
        p for p in (curve.get("points") or [])
        if p.get("false_per_1000") is not None and p.get(key) is not None
    ]


def recall_at_budget(curve: dict, budget_per_1000: float) -> dict | None:
    """The most recall available without exceeding a false-alarm budget.

    This is the headline the product is judged on: everything else on the curve
    is context for choosing it. Ties on recall are broken toward the *higher*
    threshold — the same recall with more headroom is strictly better.
    """
    usable = [p for p in _usable(curve) if p["false_per_1000"] <= budget_per_1000]
    if not usable:
        return None
    return max(usable, key=lambda p: (p["recall_macro"], p["threshold"]))


def operating_window(
    curve: dict,
    budget_per_1000: float,
    recall_floor: float,
) -> dict | None:
    """The span of thresholds that satisfy the budget AND the recall floor.

    A model that only meets its target at one exact threshold is not deployable
    in a room whose noise floor moves. A wide window means the operating point
    can drift without the product breaking; a negative width means no threshold
    satisfies both constraints at once, which no single-point metric reveals.
    """
    usable = _usable(curve)
    if not usable:
        return None
    under_budget = [p["threshold"] for p in usable if p["false_per_1000"] <= budget_per_1000]
    above_floor = [p["threshold"] for p in usable if p["recall_macro"] >= recall_floor]
    if not under_budget or not above_floor:
        return {"lo": None, "hi": None, "width": None, "feasible": False}
    # False alarms fall as the threshold rises, recall falls with it: the budget
    # sets the floor of the window and the recall floor sets its ceiling.
    lo, hi = min(under_budget), max(above_floor)
    return {"lo": float(lo), "hi": float(hi), "width": float(hi - lo), "feasible": hi >= lo}


def band_pauc(
    curve: dict,
    *,
    lo: float = BAND_LO_PER_1000,
    hi: float = BAND_HI_PER_1000,
    n_grid: int = 64,
) -> float | None:
    """Normalised partial AUC of the recall-vs-false-alarm curve over the band.

    Threshold-free like AP, but restricted to the false-alarm range the product
    is deployed in — which is exactly why AP saturated on this corpus while
    telling nobody anything: it integrates over a region nothing will operate in.

    The recall envelope is the best recall reachable at or below each rate, so a
    non-monotone sweep cannot inflate the score.
    """
    points = _usable(curve)
    if not points or hi <= lo:
        return None

    grid = [
        10 ** (math.log10(lo) + (math.log10(hi) - math.log10(lo)) * i / (n_grid - 1))
        for i in range(n_grid)
    ]
    envelope = []
    for rate in grid:
        reachable = [p["recall_macro"] for p in points if p["false_per_1000"] <= rate]
        envelope.append(max(reachable) if reachable else 0.0)
    total = 0.0
    for i in range(1, n_grid):
        dx = math.log10(grid[i]) - math.log10(grid[i - 1])
        total += 0.5 * (envelope[i] + envelope[i - 1]) * dx
    return total / (math.log10(hi) - math.log10(lo))
