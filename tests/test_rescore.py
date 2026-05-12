"""Tests for re-deriving frozen-test metrics from stored scores.

The Runs & compare tab moves a confidence slider and expects every number, the
confusion matrix included, to follow — without loading a model. That only holds
if the re-cut agrees with what the evaluator would have written, and if the
things it cannot re-derive (a different test set, a missing val pass) fail
visibly rather than silently substituting something else.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from sed.train import rescore, score_store

CLASSES = ["_negative", "loud", "medium", "quiet"]


def _scored_run(tmp_path, *, name="2026-07-30_1200", n_neg=400, n_pos=90, seed=1, spread=0.05):
    """A run directory with a stored score matrix from a confident model."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n_neg):
        p = rng.random(len(CLASSES)) * spread
        p[0] = 1.0 - rng.random() * spread
        rows.append(p)
        labels.append(0)
    for i in range(n_pos):
        cls = 1 + i % 3
        p = rng.random(len(CLASSES)) * spread
        p[cls] = 1.0 - rng.random() * spread
        rows.append(p)
        labels.append(cls)
    probs = np.asarray(rows, dtype=np.float32)
    true_idx = np.asarray(labels, dtype=np.int16)

    run_dir = tmp_path / name
    (run_dir / "reports").mkdir(parents=True)
    key = score_store.write_scores(
        run_dir, engine="checkpoint", test_set_hash="sha256:abcdef0123456789",
        classes=CLASSES, probs=probs, true_idx=true_idx, paths=[""] * len(true_idx),
    )
    return run_dir, key, probs, true_idx


def test_rescore_matches_the_evaluator_at_the_same_operating_point(tmp_path):
    """The whole premise of the tab: a re-cut is the same measurement, not a
    second opinion."""
    from sed.train.metrics import confusion_matrix, decision_metrics, runtime_pred_indices

    run_dir, key, probs, true_idx = _scored_run(tmp_path)
    targets = score_store.one_hot(true_idx, len(CLASSES))
    thresholds = {"_negative": 0.5, "loud": 0.9, "medium": 0.9, "quiet": 0.9}

    pred = runtime_pred_indices(probs, CLASSES, thresholds, 0.5, 0.08)
    cm = confusion_matrix(probs, targets, CLASSES, pred_idx=pred)
    direct = decision_metrics(
        probs, targets, CLASSES, cm, thresholds=thresholds, threshold=0.9,
        margin_to_second_best=0.08,
    )
    got = rescore.rescore(run_dir, key, rescore.Decision(confidence=0.90))

    for metric in ("f1_macro_targets", "recall_macro_targets", "precision_macro_targets",
                   "false_per_1000_negatives", "accuracy", "ap_macro_targets"):
        assert got[metric] == pytest.approx(direct[metric]), metric
    assert got["confusion"] == cm.tolist()


def test_raising_the_confidence_trades_recall_for_quiet(tmp_path):
    run_dir, key, _probs, _true = _scored_run(tmp_path, spread=0.4, seed=4)
    loose = rescore.rescore(run_dir, key, rescore.Decision(confidence=0.50))
    tight = rescore.rescore(run_dir, key, rescore.Decision(confidence=0.99))

    assert tight["recall_macro_targets"] <= loose["recall_macro_targets"]
    assert tight["false_per_1000_negatives"] <= loose["false_per_1000_negatives"]
    # Threshold-free numbers describe the model, not the operating point, so
    # they must NOT move when only the confidence does.
    assert tight["ap_macro_targets"] == pytest.approx(loose["ap_macro_targets"])
    assert tight["band_pauc"] == pytest.approx(loose["band_pauc"])


def test_headline_is_reported_with_the_threshold_that_reaches_it(tmp_path):
    run_dir, key, _p, _t = _scored_run(tmp_path)
    got = rescore.rescore(run_dir, key, rescore.Decision(budget_per_1000=1.0))

    assert got["recall_at_budget"] is not None
    assert got["threshold_at_budget"] is not None
    assert got["false_at_budget"] <= 1.0
    # Per-class, because a macro number hides the class that is actually hard.
    assert set(got["recall_at_budget_per_class"]) == {"loud", "medium", "quiet"}


def test_a_missing_score_store_is_reported_not_guessed(tmp_path):
    run_dir, _key, _p, _t = _scored_run(tmp_path)
    # A test set this run was never scored on has no scores to re-cut.
    other = score_store.score_key("checkpoint", "sha256:9999999999999999")
    assert rescore.rescore(run_dir, other, rescore.Decision()) is None


def test_deployed_mode_falls_back_visibly_when_there_is_no_saved_layer(tmp_path):
    """Silently substituting the shared confidence would make two rows in one
    column mean different things."""
    run_dir, key, _p, _t = _scored_run(tmp_path)
    got = rescore.rescore(run_dir, key, rescore.Decision(mode="deployed"))
    assert got["thresholds_source"] == "shared_fallback_no_deployed"

    (run_dir / "runtime_settings.json").write_text(
        json.dumps({"settings": {"thresholds": {"loud": 0.7, "medium": 0.8, "quiet": 0.95}}}),
        encoding="utf-8")
    got = rescore.rescore(run_dir, key, rescore.Decision(mode="deployed"))
    assert got["thresholds_source"] == "deployed"
    assert got["thresholds"]["loud"] == pytest.approx(0.7)
    assert got["thresholds"]["quiet"] == pytest.approx(0.95)


def test_val_fitted_modes_need_a_stored_val_pass(tmp_path):
    run_dir, key, probs, true_idx = _scored_run(tmp_path)
    got = rescore.rescore(run_dir, key, rescore.Decision(mode="val_f1"))
    assert got["thresholds_source"] == "shared_fallback_no_val_scores"

    # With a val pass stored, the same request fits real thresholds — no
    # checkpoint, no forward pass.
    score_store.write_scores(
        run_dir, engine="checkpoint", test_set_hash="val:1111222233334444",
        classes=CLASSES, probs=probs, true_idx=true_idx,
        paths=[""] * len(true_idx), split="val",
    )
    got = rescore.rescore(run_dir, key, rescore.Decision(mode="val_f1"))
    assert got["thresholds_source"] == "val_f1"
    assert got["thresholds"]["_negative"] == pytest.approx(0.5)


def test_argmax_drops_every_gate(tmp_path):
    run_dir, key, _p, _t = _scored_run(tmp_path)
    got = rescore.rescore(run_dir, key, rescore.Decision(mode="argmax", confidence=0.99))
    assert got["prediction_mode"] == "closed_set_argmax"
    assert got["margin_to_second_best"] == 0.0
    # Nothing abstains under argmax, so recall cannot be held back by a gate.
    assert got["recall_macro_targets"] > 0.9


def test_argmax_block_is_reported_whatever_the_bar_is_set_to(tmp_path):
    """The threshold-free reading travels with every re-cut, not behind a mode.

    Its whole value is being readable NEXT TO the gated numbers — the gap
    between them is the recall the decision layer is costing — so it has to be
    in the same block, and it must not move when the bar does.
    """
    run_dir, key, _p, _t = _scored_run(tmp_path)
    strict = rescore.rescore(run_dir, key, rescore.Decision(confidence=0.99, margin=0.5))
    loose = rescore.rescore(run_dir, key, rescore.Decision(confidence=0.10, margin=0.0))

    assert strict["argmax"] == loose["argmax"], "argmax moved with the operating point"
    assert strict["argmax"]["recall_macro"] > 0.9
    # The gates cost recall; argmax is the ceiling they are costing it from.
    assert strict["recall_macro_targets"] < strict["argmax"]["recall_macro"]
    assert loose["recall_macro_targets"] == pytest.approx(loose["argmax"]["recall_macro"], abs=0.05)


def test_argmax_counts_background_as_a_competing_class(tmp_path):
    """Without `_negative` in the argmax, "best target wins" would fire on every
    background clip and recall would be free. The false-alarm figure is what
    catches that, so it has to be reported."""
    run_dir, key, _p, _t = _scored_run(tmp_path)
    block = rescore.rescore(run_dir, key, rescore.Decision())["argmax"]
    assert block["false_per_1000_negatives"] is not None
    assert block["false_per_1000_negatives"] < 50.0
    assert block["accuracy"] is not None


def test_separation_and_abstention_describe_confidence(tmp_path):
    run_dir, key, _p, _t = _scored_run(tmp_path)
    got = rescore.rescore(run_dir, key, rescore.Decision(margin=0.08))
    sep = got["separation"]
    # A confident model leaves its background far from the trigger and its
    # targets far from their runner-up.
    assert sep["negative_headroom"] < 0.1
    assert sep["target_margin"] > 0.8
    assert got["margin_abstention"] is not None


def test_decision_payload_clamps_rather_than_rejects():
    """A control cannot send something meaningful out of range, and failing a
    whole table over one bad field is worse than using its default."""
    d = rescore.Decision.from_payload({
        "confidence": 5, "negative_max": -2, "margin": "nonsense",
        "mode": "not-a-mode", "budget_per_1000": 1e9,
    })
    assert d.confidence == 1.0
    assert d.negative_max == 0.0
    assert d.margin == 0.08          # unparseable → default
    assert d.mode == "shared"        # unknown → the benchmark mode
    assert d.budget_per_1000 == 1000.0
    assert rescore.Decision.from_payload(None).confidence == 0.90


def test_rescore_stays_torch_free():
    probe = "import sed.train.rescore, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", "the re-scoring path pulled in torch"
