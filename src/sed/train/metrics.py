"""Decision replay and metric computation over a probability matrix.

Everything here operates on an ``(N, C)`` array of class probabilities that
somebody else has already produced — a checkpoint forward pass, an OpenVINO IR,
or a score file read back off disk. Nothing in this module knows how the
probabilities were made.

That separation is the point. `sed.train.eval` computes metrics once at training
time; the inspect server recomputes them on demand whenever the user moves a
confidence slider in Runs & compare. Both call the functions below, so a number
shown in the browser can never drift from the number stored in the manifest.

Deliberately free of torch, lightning and the dataset stack — importing the
training runtime costs seconds, and the web UI reaches this on a request path.
"""

from __future__ import annotations

import numpy as np

from sed.train.test_rules import false_fire_rate, false_per_1000_negatives

NEGATIVE_CLASS = "_negative"


# --- Threshold-free ranking ----------------------------------------------

def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the precision-recall curve, or None when the class has no positives.

    Mirrors torchmetrics' MultilabelAveragePrecision per-class; we recompute
    it here so eval is self-contained and works on a numpy stack of probs.

    Tied scores are collapsed into a single operating point. Without that, the
    result depends on how ``argsort`` happens to order equal scores — which is
    exactly the saturated regime a confident model produces, where float32
    probabilities pile up on 0.0 and 1.0. (An earlier version scored the same
    16-sample input anywhere from 0.36 to 0.86 depending on tie order; the
    correct value is 0.458.) ``test_eval.py`` pins this against sklearn.

    A class with no positive samples has no defined AP, so it returns None
    rather than 0.0 — a zero would silently drag the macro average down for
    every class a frozen test happens not to cover.
    """
    labels = np.asarray(labels, dtype=np.float64)
    n_pos = float(labels.sum())
    if n_pos == 0:
        return None
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    tp_cum = np.cumsum(labels_sorted)
    fp_cum = np.cumsum(1.0 - labels_sorted)
    # One operating point per distinct score: keep the last index of each run
    # of equal scores. A threshold cannot separate samples that tie.
    last = np.r_[np.nonzero(np.diff(scores_sorted))[0], len(scores_sorted) - 1]
    tp_cum, fp_cum = tp_cum[last], fp_cum[last]
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    # AP = sum_k (R_k - R_{k-1}) * P_k
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


# --- Decision replay ------------------------------------------------------

def closed_set_decisions(
    probs: np.ndarray,
    classes: list[str],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Argmax decisions for target-only models without an explicit no-trigger class."""
    if probs.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), []
    pred_idx = np.argmax(probs, axis=1).astype(np.int64)
    order = np.argsort(-probs, axis=1)
    decisions: list[dict[str, object]] = []
    for i, idx in enumerate(pred_idx):
        second_idx = int(order[i, 1]) if probs.shape[1] > 1 else int(idx)
        decisions.append({
            "predicted_class": classes[int(idx)],
            "predicted_confidence": float(probs[i, idx]),
            "triggered": True,
            "decision_reason": "closed_set_argmax",
            "decision_threshold": "",
            "argmax_class": classes[int(idx)],
            "argmax_confidence": float(probs[i, idx]),
            "runner_up_class": classes[second_idx],
            "runner_up_confidence": float(probs[i, second_idx]),
        })
    return pred_idx, decisions


def firing_classes(cfg, classes: list[str]) -> set[str] | None:
    """Classes allowed to fire a trigger, or ``None`` when every one of them may.

    Offline scoring has to reproduce the deployed decision layer, and that layer
    lets suppressor classes compete without firing (see
    ``sed.runtime.smooth.TriggerArbiter``). Returning ``None`` for the common
    all-targets case keeps the vectorized sweeps on their fast path.
    """
    roles = cfg.class_roles(list(classes))
    firing = {cls for cls, role in roles.items() if role == "target"}
    positives = {cls for cls, role in roles.items() if role != "negative"}
    return None if firing == positives else firing


def negative_role_classes(cfg, classes: list[str]) -> set[str]:
    """Every class with role "negative" — `_negative` by default, plus any
    class explicitly given that role (see `Config.class_roles`)."""
    roles = cfg.class_roles(list(classes))
    return {cls for cls, role in roles.items() if role == "negative"}


def runtime_decisions(
    probs: np.ndarray,
    classes: list[str],
    thresholds: dict[str, float],
    negative_max: float,
    margin_to_second_best: float,
    trigger_classes: set[str] | None = None,
    negative_classes: set[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Single-window version of the runtime trigger decision.

    The live system additionally smooths probabilities over time. Frozen test
    clips are independent one-second windows, so we apply the same gates without
    median/EMA/consecutive-frame state.

    ``trigger_classes`` restricts which classes may actually fire. Every class
    outside ``negative_classes`` still competes for the window, so a suppressor
    (``hardnegs`` and friends) that wins blocks the trigger instead of handing it
    to the runner-up — matching ``TriggerArbiter``. ``None`` means every
    non-negative class can fire. ``negative_classes`` defaults to ``{"_negative"}``
    — pass `sed.train.metrics.negative_role_classes(cfg, classes)` to also cover
    a project that split its background pool into named buckets.
    """
    if probs.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), []

    negative_classes = {NEGATIVE_CLASS} if negative_classes is None else negative_classes
    negative_indices = [i for i, cls in enumerate(classes) if cls in negative_classes]
    has_negative = bool(negative_indices)
    positive_indices = [i for i, cls in enumerate(classes) if cls not in negative_classes]
    firing = (
        {i for i in positive_indices if classes[i] in trigger_classes}
        if trigger_classes is not None else set(positive_indices)
    )
    pred_idx = np.zeros((probs.shape[0],), dtype=np.int64)
    decisions: list[dict[str, object]] = []

    for row in probs:
        argmax_idx = int(np.argmax(row))
        # The negative-role class with the highest score represents the veto —
        # with one negative class (the common case) this is just its own score.
        neg_winner_idx = max(negative_indices, key=lambda idx: row[idx]) if has_negative else None
        if not positive_indices:
            fallback_idx = neg_winner_idx if neg_winner_idx is not None else argmax_idx
            decisions.append({
                "predicted_class": classes[fallback_idx],
                "predicted_confidence": float(row[fallback_idx]),
                "triggered": False,
                "decision_reason": "no_positive_classes",
                "decision_threshold": float("nan"),
                "argmax_class": classes[argmax_idx],
                "argmax_confidence": float(row[argmax_idx]),
            })
            pred_idx[len(decisions) - 1] = fallback_idx
            continue

        winner_idx = max(positive_indices, key=lambda idx: row[idx])
        winner_score = float(row[winner_idx])
        winner_class = classes[winner_idx]
        positive_scores = sorted(
            [(float(row[idx]), idx) for idx in positive_indices],
            reverse=True,
        )
        runner_score, runner_idx = (
            positive_scores[1] if len(positive_scores) > 1 else (0.0, winner_idx)
        )
        threshold = thresholds.get(winner_class, 0.5)
        neg_score = float(row[neg_winner_idx]) if neg_winner_idx is not None else 0.0

        reason = "trigger"
        triggered = True
        predicted_idx = winner_idx
        predicted_confidence = winner_score
        if neg_winner_idx is not None and neg_score > negative_max:
            reason = "negative_gate"
            triggered = False
            predicted_idx = neg_winner_idx
            predicted_confidence = neg_score
        elif len(positive_scores) > 1 and winner_score - runner_score < margin_to_second_best:
            reason = "margin_gate"
            triggered = False
            predicted_idx = neg_winner_idx if neg_winner_idx is not None else winner_idx
            predicted_confidence = neg_score if neg_winner_idx is not None else winner_score
        elif winner_idx not in firing:
            # A class that never fires won the window: abstain. Reporting the
            # winner keeps the confusion matrix honest about what the model said.
            reason = "suppressor_gate"
            triggered = False
            predicted_idx = neg_winner_idx if neg_winner_idx is not None else winner_idx
            predicted_confidence = neg_score if neg_winner_idx is not None else winner_score
        elif winner_score < threshold:
            reason = "below_threshold"
            triggered = False
            predicted_idx = neg_winner_idx if neg_winner_idx is not None else winner_idx
            predicted_confidence = neg_score if neg_winner_idx is not None else winner_score

        pred_idx[len(decisions)] = predicted_idx
        decisions.append({
            "predicted_class": classes[predicted_idx],
            "predicted_confidence": float(predicted_confidence),
            "triggered": triggered,
            "decision_reason": reason,
            "decision_threshold": float(threshold),
            "winner_class": winner_class,
            "winner_confidence": winner_score,
            "runner_up_class": classes[runner_idx],
            "runner_up_confidence": float(runner_score),
            "argmax_class": classes[argmax_idx],
            "argmax_confidence": float(row[argmax_idx]),
        })

    return pred_idx, decisions


def runtime_pred_indices(
    probs: np.ndarray,
    classes: list[str],
    thresholds: dict[str, float],
    negative_max: float,
    margin_to_second_best: float,
    trigger_classes: set[str] | None = None,
    negative_classes: set[str] | None = None,
) -> np.ndarray:
    """Vectorized `runtime_decisions` returning only the predicted indices.

    Used for threshold sweeps and for on-demand re-scoring, where the per-row
    decision dicts of the loop version would dominate the cost. Must stay
    decision-identical to `runtime_decisions` (covered by a parity test).
    """
    n = probs.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    negative_classes = {NEGATIVE_CLASS} if negative_classes is None else negative_classes
    negative_indices = [i for i, cls in enumerate(classes) if cls in negative_classes]
    has_negative = bool(negative_indices)
    positive_indices = [i for i, cls in enumerate(classes) if cls not in negative_classes]
    neg = np.asarray(negative_indices) if has_negative else None
    if not positive_indices:
        if has_negative:
            neg_probs = probs[:, neg]
            neg_win_col = np.argmax(neg_probs, axis=1)
            return neg[neg_win_col].astype(np.int64)
        return np.argmax(probs, axis=1).astype(np.int64)

    pos = np.asarray(positive_indices)
    pos_probs = probs[:, pos]                      # (N, P)
    win_col = np.argmax(pos_probs, axis=1)         # first max, like the loop's max()
    rows_range = np.arange(n)
    winner_idx = pos[win_col]
    winner_score = pos_probs[rows_range, win_col].astype(np.float64)
    if len(positive_indices) > 1:
        rest = pos_probs.astype(np.float64).copy()
        rest[rows_range, win_col] = -np.inf
        runner_score = rest.max(axis=1)
    else:
        runner_score = None

    thr_by_col = np.asarray(
        [float(thresholds.get(classes[i], 0.5)) for i in positive_indices], dtype=np.float64
    )
    threshold = thr_by_col[win_col]

    blocked = np.zeros((n,), dtype=bool)
    if has_negative:
        # The negative-role class with the highest score per row represents
        # the veto — with one negative class this reduces to its own score.
        neg_probs = probs[:, neg].astype(np.float64)
        neg_win_col = np.argmax(neg_probs, axis=1)
        neg_winner_idx = neg[neg_win_col]
        neg_score = neg_probs[rows_range, neg_win_col]
        blocked |= neg_score > float(negative_max)
    if runner_score is not None:
        blocked |= (winner_score - runner_score) < float(margin_to_second_best)
    if trigger_classes is not None:
        can_fire = np.asarray(
            [classes[i] in trigger_classes for i in positive_indices], dtype=bool
        )
        blocked |= ~can_fire[win_col]
    blocked |= winner_score < threshold

    pred_idx = winner_idx.astype(np.int64)
    if has_negative:
        pred_idx[blocked] = neg_winner_idx[blocked]
    # Without a negative-role class the loop version keeps the winner index for
    # blocked rows, so there is nothing to overwrite here.
    return pred_idx


# --- Metrics --------------------------------------------------------------

def confusion_matrix(
    probs: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    pred_idx: np.ndarray | None = None,
) -> np.ndarray:
    """NxN matrix. Rows = true class, cols = predicted class."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    if probs.shape[0] == 0:
        return cm
    if pred_idx is None:
        pred_idx = np.argmax(probs, axis=1)
    true_idx = np.argmax(targets, axis=1)
    for t, p in zip(true_idx, pred_idx, strict=True):
        cm[t, p] += 1
    return cm


def decision_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    cm: np.ndarray,
    thresholds: dict[str, float],
    threshold: float | None = None,
    *,
    margin_to_second_best: float = 0.0,
    budget_per_1000: float | None = None,
    roles: dict[str, str] | None = None,
) -> dict:
    """Precision/recall/F1 from threshold-aware single-label decisions.

    ``roles`` (``Config.class_roles`` output; defaults to "`_negative` is the
    only negative-role class, everything else is a target") decides which
    classes are "targets" for every macro average and false-alarm figure below
    — a project that split its background pool into named buckets (`hardneg`,
    `cutlery`, …) gets full per-class AP/F1/precision/recall for each bucket
    (stored in `per_class` regardless of role, for diagnostics) while none of
    them count toward `*_macro_targets` or inflate recall.
    """
    if roles is None:
        roles = {c: ("negative" if c == NEGATIVE_CLASS else "target") for c in classes}
    target_bin = (targets >= 0.5).astype(np.int32)
    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        tn = int(np.sum(cm) - tp - fp - fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        ap = average_precision(probs[:, i], target_bin[:, i])
        per_class[name] = {
            "ap": ap,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": int(np.sum(cm[i, :])),
            "role": roles.get(name, "target"),
        }

    target_names = [c for c in classes if roles.get(c) == "target"]
    non_target_names = [c for c in classes if roles.get(c) != "target"]

    def _macro(key: str) -> float:
        # Undefined per-class values (AP of a class the test set does not cover)
        # are skipped rather than counted as 0.0 — see `average_precision`.
        values = [
            float(per_class[c][key])
            for c in target_names
            if per_class[c].get(key) is not None
        ]
        return float(np.mean(values)) if values else 0.0

    accuracy = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) else 0.0

    # The share of the test set that is background. Precision, F1, accuracy and
    # AP all move with this number, so a score is only comparable against
    # another scored at the same prior — store it next to them rather than
    # leaving readers to guess. Production sits near 0.999; a frozen test rarely
    # does, which is why `false_per_1000_negatives` (below) is the transferable
    # false-alarm figure and macro precision is not.
    n_samples = int(probs.shape[0])
    n_negatives = sum(int(per_class[c]["support"]) for c in non_target_names)
    negative_prior = (n_negatives / n_samples) if n_samples else None

    return {
        "threshold": threshold,
        "thresholds": dict(thresholds),
        "per_class": per_class,
        "f1_macro_targets": _macro("f1"),
        "ap_macro_targets": _macro("ap"),
        # Macro precision/recall over targets: the runs table lets a rule column
        # switch between these without re-deriving them from per_class client-side.
        "precision_macro_targets": _macro("precision"),
        "recall_macro_targets": _macro("recall"),
        # The deployment false-alarm number: the share of background clips that
        # fired a target. Persisted rather than derived so the CLI and the
        # compare view see the same figure the runs table shows.
        "false_fire_rate": false_fire_rate(per_class),
        # Same quantity as a rate with an observable denominator. Unlike
        # precision/F1 this survives a change of test prior, so it is the
        # false-alarm figure to compare across differently-balanced test sets.
        "false_per_1000_negatives": false_per_1000_negatives(per_class),
        # Target clips that fired the WRONG target. The classes are distinct
        # events, so this is a real error — but it is not a background false
        # alarm, and lumping the two together hides which one a model has.
        "cross_target_confusion": cross_target_confusion(cm, classes, roles=roles),
        "accuracy": accuracy,
        "n_samples": n_samples,
        # Denominator + prior for every rate above, so a reader can compute a
        # confidence interval and know which prior the prior-bound metrics used.
        "n_negatives": n_negatives,
        "negative_prior": negative_prior,
        # The threshold sweep this model supports, so a reader can pick an
        # operating point instead of accepting the one this block was scored at.
        **operating_point_block(
            probs, targets, classes, thresholds,
            margin_to_second_best=margin_to_second_best,
            budget_per_1000=budget_per_1000,
            roles=roles,
        ),
    }


def argmax_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    *,
    roles: dict[str, str] | None = None,
) -> dict | None:
    """The same matrix read with no decision layer at all: highest score wins.

    Every other metric here depends on at least one chosen number — a
    confidence, a negative gate, a margin. That makes them measurements of a
    *model plus a decision layer*, and two models can trade places purely by
    being calibrated differently. This one cannot: it asks only whether the
    right class outranks the others, so it isolates what the network learned
    from how its outputs happen to be scaled.

    Read it against the gated numbers. A high argmax recall next to a low
    gated recall means the model knows the answer but is not confident enough
    to pass the threshold — a calibration problem, fixable without retraining.
    Both low means the model does not separate the classes at all.

    Every non-target-role class (`_negative` by default, plus any class a
    project gave that role) participates as an ordinary class, which is what
    makes this honest: without it, "the best target wins" would fire on every
    background clip and recall would be free.
    """
    if probs.shape[0] == 0 or not classes:
        return None
    if roles is None:
        roles = {c: ("negative" if c == NEGATIVE_CLASS else "target") for c in classes}
    pred_idx = np.argmax(probs, axis=1)
    cm = confusion_matrix(probs, targets, classes, pred_idx=pred_idx)
    target_names = [c for c in classes if roles.get(c) == "target"]
    non_target_idx = [i for i, c in enumerate(classes) if roles.get(c) != "target"]

    recalls: list[float] = []
    f1s: list[float] = []
    for i, name in enumerate(classes):
        if roles.get(name) != "target":
            continue
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)

    false_per_1000 = None
    if non_target_idx:
        n_neg = int(sum(int(np.sum(cm[i, :])) for i in non_target_idx))
        if n_neg > 0:
            correct = int(sum(int(cm[i, i]) for i in non_target_idx))
            fired = n_neg - correct
            false_per_1000 = fired / n_neg * 1000.0

    total = int(np.sum(cm))
    return {
        "recall_macro": float(np.mean(recalls)) if recalls else None,
        "f1_macro": float(np.mean(f1s)) if f1s else None,
        "accuracy": float(np.trace(cm) / total) if total else None,
        "false_per_1000_negatives": false_per_1000,
        "cross_target_confusion": cross_target_confusion(cm, classes, roles=roles),
        "confusion": cm.tolist(),
        "n_target_classes": len(target_names),
    }


def cross_target_confusion(
    cm: np.ndarray, classes: list[str], *, roles: dict[str, str] | None = None,
) -> float | None:
    """Share of target clips that fired a *different* target class.

    `loud` heard as `medium` is a real error — the classes drive different
    events — but it is not the same failure as firing on background, and it has
    a different denominator in production (targets are rare, background is not).
    Reported separately so a model that confuses two glass types is
    distinguishable from one that trips on the room.

    A suppressor is excluded here just like a negative-role class: it is not a
    real detection either, so a target clip confused with a suppressor is
    "fired on background," not "fired the wrong target" — and vice versa, a
    suppressor clip predicted as a target must not count as a target's own
    error.
    """
    if roles is None:
        roles = {c: ("negative" if c == NEGATIVE_CLASS else "target") for c in classes}
    target_idx = [i for i, c in enumerate(classes) if roles.get(c) == "target"]
    if not target_idx or cm.size == 0:
        return None
    support = int(sum(int(cm[i, :].sum()) for i in target_idx))
    if support <= 0:
        return None
    wrong = int(sum(int(cm[i, j]) for i in target_idx for j in target_idx if i != j))
    return wrong / support


def operating_point_block(
    probs,
    targets,
    classes,
    thresholds,
    *,
    margin_to_second_best: float = 0.0,
    budget_per_1000: float | None = None,
    roles: dict[str, str] | None = None,
) -> dict:
    """Recall-vs-false-alarm curve and its summary, alongside the point metrics.

    A single F1 describes one threshold; two models are only comparable at the
    budget they will actually run at. Cheap — a sweep over probabilities that
    have already been computed.

    The sweep replays the SAME gates the point metrics used, margin included:
    a curve drawn without the margin gate describes a decision layer nothing
    runs.
    """
    from sed.train.operating_point import band_pauc, recall_at_budget, sweep

    try:
        curve = sweep(
            probs, targets, classes,
            negative_max=float(thresholds.get(NEGATIVE_CLASS, 0.5)),
            margin=float(margin_to_second_best),
            roles=roles,
        )
    except Exception:  # noqa: BLE001 — a summary must never fail a scoring run
        return {}
    if not curve.get("points"):
        return {}
    out = {
        "operating_curve": curve["points"],
        "band_pauc": band_pauc(curve),
    }
    for budget in _summary_budgets(budget_per_1000):
        at = recall_at_budget(curve, budget)
        key = _budget_key(budget)
        out[f"recall_at_{key}_per_1000"] = (at or {}).get("recall_macro")
        out[f"threshold_at_{key}_per_1000"] = (at or {}).get("threshold")
    return out


def _summary_budgets(budget_per_1000: float | None) -> list[float]:
    """Budgets to summarise in a stored block: always 1/1000, plus a chosen one."""
    budgets = [1.0]
    if budget_per_1000 is not None and abs(float(budget_per_1000) - 1.0) > 1e-9:
        budgets.append(float(budget_per_1000))
    return budgets


def _budget_key(budget: float) -> str:
    """`1.0` -> `1`, `0.5` -> `0_5` — a stable suffix for a metric key."""
    text = f"{float(budget):g}".replace(".", "_").replace("-", "neg")
    return text


# --- Threshold fitting on a validation pass -------------------------------
# Lives here rather than in `eval` so that re-fitting an operating point from a
# stored val score matrix does not pull the training runtime into the web
# server. `eval` re-exports both fitters, and calls them — these are the only
# implementations. (They used to be duplicated in `eval`, where the local copies
# silently shadowed these on import; the two searched different grids and so
# returned different thresholds for the same scores, meaning a run calibrated by
# `evaluate_run` and the same run re-calibrated from its stored scores did not
# agree. Keep exactly one implementation here.)


def _threshold_sweep_grid(scores: np.ndarray, base: float) -> list[float]:
    """Stable calibration grid with exact observed boundaries included.

    A regular 0.025 lattice alone lands next to — not on — the value that
    actually separates this class's positives from its negatives, so every
    observed score and the incumbent threshold join the sweep.
    """
    regular = np.arange(0.05, 0.975 + 1e-9, 0.025, dtype=np.float64)
    observed = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = [*regular.tolist(), *observed.tolist(), float(base)]
    return sorted({round(float(value), 6) for value in values if 0.0 <= value <= 1.0})


def _class_calibration_rows(
    probs: np.ndarray,
    classes: list[str],
    class_index: int,
    *,
    negative_max: float,
    margin_to_second_best: float,
    negative_classes: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return this class's score and the threshold-independent gate mask.

    The runtime fires class *i* when it wins the positive argmax, clears the
    negative veto and the runner-up margin, and beats its own threshold. Only
    the last term depends on a threshold, so the other gates are evaluated once
    per class and the sweep varies a single comparison.
    """
    negative_classes = {NEGATIVE_CLASS} if negative_classes is None else negative_classes
    negative_indexes = [i for i, cls in enumerate(classes) if cls in negative_classes]
    target_indexes = [i for i in range(len(classes)) if i not in negative_indexes]
    scores = np.asarray(probs[:, class_index], dtype=np.float64)
    eligible = np.ones(scores.shape[0], dtype=bool)
    if negative_indexes:
        eligible &= (np.asarray(probs[:, negative_indexes]) <= float(negative_max)).all(axis=1)
    if len(target_indexes) > 1:
        target_probs = np.asarray(probs[:, target_indexes], dtype=np.float64)
        order = np.argsort(-target_probs, axis=1)
        winner = np.asarray(target_indexes)[order[:, 0]]
        eligible &= winner == class_index
        if margin_to_second_best > 0:
            first = np.take_along_axis(target_probs, order[:, :1], axis=1)[:, 0]
            second = np.take_along_axis(target_probs, order[:, 1:2], axis=1)[:, 0]
            eligible &= (first - second) >= float(margin_to_second_best)
    return scores, eligible


def fit_thresholds_from_val(
    probs: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    *,
    negative_max: float,
    margin_to_second_best: float,
    base_thresholds: dict[str, float],
    negative_classes: set[str] | None = None,
) -> dict[str, float]:
    """Fit per-target thresholds for F1; ties choose the safer high gate.

    Each positive class is swept independently: under the runtime rule a class
    fires only when it already won the positive argmax, so another class's
    threshold cannot change this one's decisions. Every negative-role class
    (`_negative` by default) keeps `negative_max`; it is a gate, not a class
    threshold.
    """
    negative_classes = {NEGATIVE_CLASS} if negative_classes is None else negative_classes
    if np.asarray(probs).size == 0:
        return {
            c: float(base_thresholds.get(c, 0.5))
            for c in classes if c not in negative_classes
        }
    fitted: dict[str, float] = {}
    for index, class_name in enumerate(classes):
        if class_name in negative_classes:
            continue
        base = float(base_thresholds.get(class_name, 0.5))
        scores, eligible = _class_calibration_rows(
            probs, classes, index,
            negative_max=negative_max,
            margin_to_second_best=margin_to_second_best,
            negative_classes=negative_classes,
        )
        truth = np.asarray(targets[:, index]) >= 0.5
        best = (-1.0, base)
        for threshold in _threshold_sweep_grid(scores, base):
            predicted = eligible & (scores >= threshold)
            tp = int(np.count_nonzero(predicted & truth))
            fp = int(np.count_nonzero(predicted & ~truth))
            fn = int(np.count_nonzero(~predicted & truth))
            denom = 2 * tp + fp + fn
            f1 = (2.0 * tp / denom) if denom else 0.0
            best = max(best, (f1, threshold))
        fitted[class_name] = float(best[1])
    return fitted


def fit_thresholds_for_precision(
    probs: np.ndarray,
    targets: np.ndarray,
    classes: list[str],
    *,
    target_precision: float,
    negative_max: float,
    margin_to_second_best: float,
    base_thresholds: dict[str, float],
    negative_classes: set[str] | None = None,
) -> dict[str, float]:
    """Fit the lowest gate meeting precision; otherwise use safest best precision.

    A class that never reaches ``target_precision`` falls back to its most
    precise threshold (ties → highest), so a weak class degrades toward fewer
    triggers rather than more false positives.
    """
    negative_classes = {NEGATIVE_CLASS} if negative_classes is None else negative_classes
    if np.asarray(probs).size == 0:
        return {
            c: float(base_thresholds.get(c, 0.5))
            for c in classes if c not in negative_classes
        }
    fitted: dict[str, float] = {}
    for index, class_name in enumerate(classes):
        if class_name in negative_classes:
            continue
        base = float(base_thresholds.get(class_name, 0.5))
        scores, eligible = _class_calibration_rows(
            probs, classes, index,
            negative_max=negative_max,
            margin_to_second_best=margin_to_second_best,
            negative_classes=negative_classes,
        )
        truth = np.asarray(targets[:, index]) >= 0.5
        candidates: list[tuple[float, float, int]] = []
        for threshold in _threshold_sweep_grid(scores, base):
            predicted = eligible & (scores >= threshold)
            fired = int(np.count_nonzero(predicted))
            if not fired:
                continue
            tp = int(np.count_nonzero(predicted & truth))
            candidates.append((tp / fired, threshold, tp))
        meeting = [row for row in candidates if row[0] >= float(target_precision)]
        if meeting:
            fitted[class_name] = float(min(meeting, key=lambda row: row[1])[1])
        elif candidates:
            fitted[class_name] = float(max(candidates, key=lambda row: (row[0], row[1]))[1])
        else:
            fitted[class_name] = base
    return fitted
