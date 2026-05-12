"""Tests for operating-point selection on a clip corpus.

The deployment question is a curve, not a single F1: how much recall is
available at a false-alarm budget the venue accepts. Rates are per 1000 clips
because that is the denominator onset-triggered batches actually support.
"""

from __future__ import annotations

import numpy as np
import pytest

from sed.train.metrics import confusion_matrix, decision_metrics, runtime_pred_indices
from sed.train.operating_point import (
    band_pauc,
    operating_window,
    recall_at_budget,
    sweep,
)

CLASSES = ["_negative", "cheers"]
# The multi-target space this product actually runs: three glass types that are
# distinct events, so firing the wrong one is an error rather than a near-miss.
TARGETS = ["_negative", "loud", "medium", "quiet"]


def _corpus(pos_scores: list[float], neg_scores: list[float]):
    """(probs, targets) for a two-class corpus with the given target scores."""
    rows, tgts = [], []
    for s in pos_scores:
        rows.append([1.0 - s, s])
        tgts.append([0, 1])
    for s in neg_scores:
        rows.append([1.0 - s, s])
        tgts.append([1, 0])
    return np.array(rows), np.array(tgts)


def _sweep(probs, targets, **kw):
    return sweep(probs, targets, CLASSES, negative_max=1.0, margin=0.0, **kw)


def test_sweep_reports_recall_and_a_per_1000_rate_at_each_threshold():
    probs, targets = _corpus([0.9] * 10, [0.1] * 990 + [0.95] * 10)
    curve = _sweep(probs, targets)

    assert curve["n_targets"] == 10
    assert curve["n_negatives"] == 1000
    loose = next(p for p in curve["points"] if p["threshold"] == pytest.approx(0.5))
    assert loose["recall"] == pytest.approx(1.0)
    # 10 of 1000 negatives score 0.95, so they fire at any threshold below it.
    assert loose["false_per_1000"] == pytest.approx(10.0)


def test_raising_the_threshold_never_increases_false_alarms():
    rng = np.random.default_rng(0)
    probs, targets = _corpus(list(rng.random(50)), list(rng.random(500)))
    rates = [p["false_per_1000"] for p in _sweep(probs, targets)["points"]]
    assert rates == sorted(rates, reverse=True)


def test_recall_at_budget_picks_the_most_recall_within_the_budget():
    # Positives at 0.6; negatives mostly silent but 5/1000 sit at 0.7.
    probs, targets = _corpus([0.6] * 100, [0.05] * 995 + [0.7] * 5)
    curve = _sweep(probs, targets)

    strict = recall_at_budget(curve, 1.0)
    assert strict is not None
    # Only thresholds above 0.7 keep the rate under 1/1000, and those miss the
    # 0.6 positives entirely.
    assert strict["false_per_1000"] <= 1.0
    assert strict["recall"] == pytest.approx(0.0)

    relaxed = recall_at_budget(curve, 10.0)
    assert relaxed["recall"] == pytest.approx(1.0)
    assert relaxed["false_per_1000"] == pytest.approx(5.0)


def test_a_strict_threshold_meets_any_budget_by_firing_nothing():
    """Not a failure mode: silence is always within budget. The point returned
    is the honest one — zero false alarms bought with zero recall."""
    probs, targets = _corpus([0.9] * 10, [0.9] * 100)   # negatives as hot as positives
    point = recall_at_budget(_sweep(probs, targets), 0.001)
    assert point is not None
    assert point["recall"] == pytest.approx(0.0)
    assert point["false_per_1000"] == pytest.approx(0.0)


def test_recall_at_budget_is_none_when_even_the_strictest_sweep_overshoots():
    # Negatives above the top of the sweep grid: every point fires all of them.
    probs, targets = _corpus([0.99] * 10, [0.999] * 100)
    assert recall_at_budget(_sweep(probs, targets), 1.0) is None


def test_band_pauc_separates_models_that_share_a_peak_recall():
    """Two models both reaching full recall must not tie when one only gets
    there by firing far more often — the reason AP was uninformative here."""
    clean_probs, clean_targets = _corpus([0.9] * 50, [0.05] * 1000)
    # The noisy model's false alarms score ABOVE its positives, so catching the
    # positives necessarily admits them — that is what "not separable" means.
    noisy_probs, noisy_targets = _corpus([0.9] * 50, [0.05] * 900 + [0.95] * 100)

    clean = band_pauc(_sweep(clean_probs, clean_targets))
    noisy = band_pauc(_sweep(noisy_probs, noisy_targets))

    assert clean > noisy
    assert 0.0 <= noisy < clean <= 1.0


def test_band_pauc_is_one_for_perfect_separation_and_zero_for_none():
    perfect_p, perfect_t = _corpus([0.99] * 20, [0.01] * 1000)
    assert band_pauc(_sweep(perfect_p, perfect_t)) == pytest.approx(1.0)

    useless_p, useless_t = _corpus([0.01] * 20, [0.01] * 1000)
    assert band_pauc(_sweep(useless_p, useless_t)) == pytest.approx(0.0)


def test_the_negative_veto_is_applied_before_the_sweep():
    """A clip whose `_negative` score exceeds the gate cannot fire at any target
    threshold — the runtime rejects it first."""
    probs, targets = _corpus([0.9] * 10, [0.9] * 100)
    vetoed = sweep(probs, targets, CLASSES, negative_max=0.05, margin=0.0)
    assert all(p["n_false"] == 0 for p in vetoed["points"])


def test_empty_and_target_only_corpora_do_not_raise():
    assert sweep(np.zeros((0, 2)), np.zeros((0, 2)), CLASSES,
                 negative_max=0.5, margin=0.0)["points"] == []
    only_targets = sweep(np.array([[0.1, 0.9]]), np.array([[0, 1]]), ["_negative", "c"],
                         negative_max=1.0, margin=0.0)
    assert only_targets["n_negatives"] == 0
    assert only_targets["points"][0]["false_per_1000"] is None
    assert band_pauc(only_targets) is None


def _confident_corpus(n_neg=2000, n_pos=210, seed=0, spread=0.02):
    """A confident multi-target model: negatives near 0, targets near 1.

    This is the regime a trained detector actually produces, and the one the old
    fixed 0.05…0.95 grid could not describe — every threshold worth deploying
    lives above 0.95.
    """
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n_neg):
        p = rng.random(len(TARGETS)) * spread
        p[0] = 1.0 - rng.random() * spread
        rows.append(p)
        labels.append(0)
    for i in range(n_pos):
        cls = 1 + i % 3
        p = rng.random(len(TARGETS)) * spread
        p[cls] = 1.0 - rng.random() * spread
        rows.append(p)
        labels.append(cls)
    probs = np.asarray(rows, dtype=np.float32)
    true_idx = np.asarray(labels)
    targets = np.zeros_like(probs)
    targets[np.arange(len(true_idx)), true_idx] = 1.0
    return probs, targets


def test_sweep_reaches_the_low_false_alarm_region():
    """A grid stopping at 0.95 makes a strict budget unreachable on a confident
    model, which read as band-pAUC 0.0 and recall-at-budget None — a model with
    a perfectly good operating point just past the end of the sweep looked
    worthless. Real runs showed exactly this."""
    probs, targets = _confident_corpus()
    curve = sweep(probs, targets, TARGETS, negative_max=0.5, margin=0.08)

    assert max(p["threshold"] for p in curve["points"]) > 0.95
    assert min(p["false_per_1000"] for p in curve["points"]) <= 1.0
    assert band_pauc(curve) > 0.0

    at_budget = recall_at_budget(curve, 1.0)
    assert at_budget is not None
    assert at_budget["false_per_1000"] <= 1.0
    assert at_budget["recall_macro"] > 0.5


def test_recall_is_macro_over_target_classes_not_pooled():
    """The three glass types drive different events, so `loud` heard as `medium`
    is a miss — and a thin class must weigh as much as a fat one."""
    probs = np.array([
        [0.0, 0.99, 0.0, 0.0],   # loud, correct
        [0.0, 0.99, 0.0, 0.0],   # loud, correct
        [0.0, 0.99, 0.0, 0.0],   # medium heard as loud → a miss, not a hit
        [0.99, 0.0, 0.0, 0.0],   # negative
    ], dtype=np.float32)
    targets = np.zeros_like(probs)
    for row, cls in enumerate([1, 1, 2, 0]):
        targets[row, cls] = 1.0

    curve = sweep(probs, targets, TARGETS, negative_max=0.5, margin=0.0, grid=[0.9])
    point = curve["points"][0]
    assert point["recall_per_class"]["loud"] == pytest.approx(1.0)
    assert point["recall_per_class"]["medium"] == pytest.approx(0.0)
    assert point["recall_per_class"]["quiet"] is None       # no clips, no opinion
    assert point["recall_macro"] == pytest.approx(0.5)      # macro over the two with support
    assert point["recall"] == pytest.approx(2 / 3)          # pooled, for contrast


def test_curve_and_point_metrics_agree_at_the_same_threshold():
    """The sweep used to run with margin=0 while the point metrics used 0.08, so
    a curve and the F1 printed beside it described different decision layers."""
    probs, targets = _confident_corpus(n_neg=500, n_pos=90, seed=5, spread=0.35)
    margin, negative_max, thr = 0.08, 0.5, 0.90
    thresholds = {"_negative": negative_max, **{c: thr for c in TARGETS[1:]}}

    pred = runtime_pred_indices(probs, TARGETS, thresholds, negative_max, margin)
    cm = confusion_matrix(probs, targets, TARGETS, pred_idx=pred)
    point = decision_metrics(
        probs, targets, TARGETS, cm, thresholds=thresholds, threshold=thr,
        margin_to_second_best=margin,
    )
    on_curve = sweep(probs, targets, TARGETS, negative_max=negative_max,
                     margin=margin, grid=[thr])["points"][0]

    assert on_curve["recall_macro"] == pytest.approx(point["recall_macro_targets"])
    assert on_curve["false_per_1000"] == pytest.approx(point["false_per_1000_negatives"])


def test_margin_gate_is_skipped_with_a_single_target_class():
    """With no runner-up the live decision applies no margin, so comparing the
    winner against a phantom 0.0 would silently re-impose it as a second
    confidence threshold."""
    probs = np.array([[0.1, 0.5], [0.9, 0.05]])
    targets = np.array([[0, 1], [1, 0]])
    curve = sweep(probs, targets, CLASSES, negative_max=0.5, margin=0.8, grid=[0.4])
    assert curve["points"][0]["recall_macro"] == pytest.approx(1.0)


def test_operating_window_reports_whether_any_threshold_satisfies_both():
    """A model that only works at one exact threshold is not deployable in a
    room whose noise floor moves — and no single-point metric says so."""
    probs, targets = _confident_corpus()
    separable = operating_window(
        sweep(probs, targets, TARGETS, negative_max=0.5, margin=0.08),
        budget_per_1000=1.0, recall_floor=0.9,
    )
    assert separable["feasible"] and separable["width"] > 0

    # Positives at 0.6, but 5 negatives per 1000 score 0.7. Staying under one
    # false fire per 1000 forces a threshold above 0.7, which catches nothing —
    # so no threshold delivers 90% recall inside the budget, and the window says
    # so instead of reporting whichever end happened to be measured.
    overlapped, overlapped_t = _corpus([0.6] * 100, [0.05] * 995 + [0.7] * 5)
    infeasible = operating_window(
        _sweep(overlapped, overlapped_t), budget_per_1000=1.0, recall_floor=0.9,
    )
    assert infeasible["feasible"] is False


def test_operating_point_stays_torch_free():
    import subprocess
    import sys

    probe = "import sed.train.operating_point, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
