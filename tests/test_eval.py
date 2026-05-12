"""Tests for the test-set evaluation helpers."""

from __future__ import annotations

import csv
import subprocess
import sys

import numpy as np
import pytest

from sed.config import Config, InferenceCfg
from sed.train.eval import (
    _average_precision,
    _closed_set_decisions,
    _configured_thresholds,
    _confusion_matrix,
    _decision_metrics,
    _prediction_records,
    _runtime_decisions,
    _write_prediction_table,
)
from sed.train.objective import logits_to_probabilities, probability_mode


def test_average_precision_perfect_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert _average_precision(scores, labels) == 1.0


def test_average_precision_no_positives_is_undefined_not_zero():
    """A class the test set does not cover has no AP — returning 0.0 would
    silently drag `ap_macro_targets` down for every such class."""
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 0, 0])
    assert _average_precision(scores, labels) is None


def test_average_precision_collapses_tied_scores():
    """Tied scores must form ONE operating point.

    Regression for a tie bug that mattered most in the saturated regime a
    confident model produces: the same 16 samples scored 0.859 or 0.362 purely
    from how argsort ordered the ties. The correct value is 11/24.
    """
    scores = np.array([1.0] * 8 + [0.0] * 8)
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    expected = 11.0 / 24.0  # (4/8)*(4/6) + (6/16)*(2/6)

    assert _average_precision(scores, labels) == pytest.approx(expected)
    # Permuting within the tie groups must not change the answer.
    perm = np.array([4, 5, 6, 7, 0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 8, 9])
    assert _average_precision(scores[perm], labels[perm]) == pytest.approx(expected)
    # All-tied scores carry no ranking information: AP is the positive rate.
    assert _average_precision(np.full(6, 0.5), np.array([1, 0, 1, 0, 0, 0])) == pytest.approx(1 / 3)


def test_average_precision_matches_sklearn_on_tie_heavy_inputs():
    """sklearn is the oracle for the standard AP definition; we keep our own
    numpy implementation so eval stays self-contained on a stack of probs."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(300):
        n = int(rng.integers(2, 40))
        # Round hard so ties are frequent — the case that used to break.
        scores = np.round(rng.random(n), int(rng.integers(1, 3)))
        labels = rng.integers(0, 2, n)
        if labels.sum() == 0:
            continue
        expected = sklearn_metrics.average_precision_score(labels, scores)
        assert _average_precision(scores, labels) == pytest.approx(expected, abs=1e-12)
        checked += 1
    assert checked > 200, "sampling degenerated — not enough cases exercised"


def test_decision_metrics_skips_undefined_ap_in_macro():
    """A target class absent from the test set must not be averaged in as 0.0."""
    classes = ["_negative", "cheers", "clap"]
    # Two negatives and two `cheers`; `clap` has no support at all.
    probs = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.2, 0.8, 0.0]])
    targets = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
    cm = _confusion_matrix(probs, targets, classes)
    metrics = _decision_metrics(probs, targets, classes, cm, {c: 0.5 for c in classes})

    assert metrics["per_class"]["clap"]["ap"] is None
    assert metrics["per_class"]["cheers"]["ap"] == pytest.approx(1.0)
    # Macro over targets == cheers alone, not (1.0 + 0.0) / 2.
    assert metrics["ap_macro_targets"] == pytest.approx(1.0)


def test_confusion_matrix_argmax():
    # Two clips of class 0 (correct), one clip of class 1 misclassified as 0.
    classes = ["_negative", "cheers"]
    probs = np.array([
        [0.9, 0.1],          # true 0, predicted 0
        [0.6, 0.4],          # true 0, predicted 0
        [0.7, 0.3],          # true 1, predicted 0  (mistake)
    ])
    targets = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
    ])
    cm = _confusion_matrix(probs, targets, classes)
    assert cm.shape == (2, 2)
    # rows = true, cols = pred
    assert cm[0, 0] == 2
    assert cm[0, 1] == 0
    assert cm[1, 0] == 1
    assert cm[1, 1] == 0


def test_prediction_records_include_all_class_confidences(tmp_path):
    classes = ["_negative", "cheers"]
    probs = np.array([
        [0.2, 0.8],
        [0.7, 0.3],
    ])
    targets = np.array([
        [0, 1],
        [0, 1],
    ])
    rows = [
        {"path": "/tmp/a.wav", "origin": "project", "session_id": "s1"},
        {"path": "/tmp/b.wav", "origin": "project", "session_id": "s2"},
    ]

    pred_idx, decisions = _runtime_decisions(
        probs,
        classes,
        thresholds={"_negative": 0.5, "cheers": 0.5},
        negative_max=0.5,
        margin_to_second_best=0.0,
    )
    records = _prediction_records(probs, targets, classes, rows, decisions)

    assert pred_idx.tolist() == [1, 0]
    assert records[0]["file"] == "a.wav"
    assert records[0]["true_class"] == "cheers"
    assert records[0]["predicted_class"] == "cheers"
    assert records[0]["correct"] is True
    assert records[0]["p:_negative"] == 0.2
    assert records[0]["p:cheers"] == 0.8
    assert records[1]["correct"] is False
    assert records[1]["second_class"] == "cheers"
    assert records[1]["decision_reason"] == "negative_gate"

    out = _write_prediction_table(records, classes, tmp_path / "predictions.tsv")
    text = out.read_text(encoding="utf-8")
    assert "true_class\tpredicted_class\tcorrect" in text
    assert "p:_negative\tp:cheers" in text
    assert "0.800000" in text


def test_runtime_decisions_use_project_thresholds_for_low_confidence_negatives():
    classes = ["_negative", "cheers_loud", "cheers_quiet"]
    probs = np.array([
        [0.05, 0.20, 0.10],  # argmax target, but below configured 0.8 threshold
        [0.05, 0.85, 0.10],
    ])

    pred_idx, decisions = _runtime_decisions(
        probs,
        classes,
        thresholds={"_negative": 0.5, "cheers_loud": 0.8, "cheers_quiet": 0.8},
        negative_max=0.5,
        margin_to_second_best=0.0,
    )

    assert pred_idx.tolist() == [0, 1]
    assert decisions[0]["predicted_class"] == "_negative"
    assert decisions[0]["argmax_class"] == "cheers_loud"
    assert decisions[0]["decision_reason"] == "below_threshold"
    assert decisions[1]["predicted_class"] == "cheers_loud"
    assert decisions[1]["decision_reason"] == "trigger"


def test_configured_thresholds_use_config_values_and_negative_gate():
    cfg = Config(
        thresholds={"cheers_loud": 0.8},
        inference=InferenceCfg(default_threshold=0.6, negative_max=0.4),
    )

    thresholds = _configured_thresholds(cfg, ["_negative", "cheers_loud", "cheers_quiet"])

    assert thresholds == {
        "_negative": 0.4,
        "cheers_loud": 0.8,
        "cheers_quiet": 0.6,
    }


def test_decision_metrics_treat_no_trigger_as_negative():
    classes = ["_negative", "cheers"]
    probs = np.array([
        [0.05, 0.2],
        [0.05, 0.9],
    ])
    targets = np.array([
        [1, 0],
        [0, 1],
    ])
    pred_idx = np.array([0, 1])
    cm = _confusion_matrix(probs, targets, classes, pred_idx=pred_idx)

    metrics = _decision_metrics(
        probs,
        targets,
        classes,
        cm,
        thresholds={"_negative": 0.5, "cheers": 0.8},
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["per_class"]["_negative"]["f1"] == 1.0
    assert metrics["per_class"]["cheers"]["f1"] == 1.0


def test_cross_entropy_probabilities_use_softmax():
    cfg = Config()
    cfg.training.loss = "cross_entropy"
    logits = np.array([[1.0, 2.0, 3.0]])

    probs = logits_to_probabilities(logits, cfg)

    assert probability_mode(cfg) == "softmax"
    assert np.allclose(probs.sum(axis=1), np.array([1.0]))
    assert probs[0, 2] > probs[0, 1] > probs[0, 0]


def test_closed_set_decisions_do_not_apply_runtime_negative_gate():
    classes = ["cheers_loud", "cheers_quiet"]
    probs = np.array([[0.49, 0.51], [0.9, 0.1]])

    pred_idx, decisions = _closed_set_decisions(probs, classes)

    assert pred_idx.tolist() == [1, 0]
    assert [d["decision_reason"] for d in decisions] == [
        "closed_set_argmax",
        "closed_set_argmax",
    ]


def test_runtime_pred_indices_matches_loop_decisions():
    """The vectorized sweep helper must stay decision-identical to
    `_runtime_decisions` across class layouts and gate settings."""
    from sed.train.eval import _runtime_pred_indices

    rng = np.random.default_rng(7)
    layouts = [
        ["_negative", "cheers", "attention"],
        ["_negative", "cheers"],
        ["cheers", "attention"],
    ]
    for classes in layouts:
        for _ in range(10):
            probs = rng.random((200, len(classes)))
            thresholds = {c: float(rng.random()) for c in classes if c != "_negative"}
            negative_max = float(rng.random())
            margin = float(rng.random() * 0.3)
            loop_idx, _ = _runtime_decisions(
                probs, classes, thresholds,
                negative_max=negative_max, margin_to_second_best=margin,
            )
            vec_idx = _runtime_pred_indices(
                probs, classes, thresholds,
                negative_max=negative_max, margin_to_second_best=margin,
            )
            assert np.array_equal(loop_idx, vec_idx)


def test_threshold_fitters_have_exactly_one_implementation():
    """`eval` must re-export the torch-free fitters, never shadow them.

    Local copies in `eval` used to override the `metrics` import: the two
    searched different grids, so a run calibrated by `evaluate_run` and the same
    run re-calibrated from its stored scores (`sed.train.rescore`, the Runs UI)
    disagreed on every input.
    """
    from sed.train import eval as eval_mod
    from sed.train import metrics, rescore

    assert eval_mod.fit_thresholds_from_val is metrics.fit_thresholds_from_val
    assert eval_mod.fit_thresholds_for_precision is metrics.fit_thresholds_for_precision
    assert rescore.fit_thresholds_from_val is metrics.fit_thresholds_from_val
    assert rescore.fit_thresholds_for_precision is metrics.fit_thresholds_for_precision


def test_per_class_calibration_mask_replays_the_runtime_rule_exactly():
    """The fitters sweep each class independently rather than by coordinate
    ascent. That is only sound because a class fires solely when it already won
    the positive argmax — another class's threshold cannot change its decisions.
    Pin the equivalence against the full runtime replay.
    """
    from sed.train.metrics import _class_calibration_rows, runtime_pred_indices

    rng = np.random.default_rng(11)
    classes = ["_negative", "a", "b", "c"]
    negative_max, margin = 0.5, 0.08
    for _ in range(50):
        probs = rng.dirichlet(np.ones(4) * rng.uniform(0.3, 3.0), size=60)
        thresholds = {c: float(rng.uniform(0.1, 0.95)) for c in classes}
        predicted = runtime_pred_indices(
            probs, classes, thresholds,
            negative_max=negative_max, margin_to_second_best=margin,
        )
        for index, class_name in enumerate(classes):
            if class_name == "_negative":
                continue
            scores, eligible = _class_calibration_rows(
                probs, classes, index,
                negative_max=negative_max, margin_to_second_best=margin,
            )
            fires = eligible & (scores >= thresholds[class_name])
            assert np.array_equal(fires, predicted == index)


def test_fit_thresholds_from_val_recovers_separation_boundary():
    """Positives at 0.9, hard negatives at 0.6: every grid point in (0.6, 0.9]
    gives perfect F1 and the fitter must take the highest (most conservative)."""
    from sed.train.eval import fit_thresholds_from_val

    classes = ["_negative", "cheers"]
    pos = np.tile([0.05, 0.90], (40, 1))    # true cheers, confidently scored
    hard = np.tile([0.10, 0.60], (40, 1))   # true negatives that score 0.6 on cheers
    easy = np.tile([0.90, 0.05], (40, 1))   # true negatives caught by the negative gate
    probs = np.vstack([pos, hard, easy])
    targets = np.vstack([
        np.tile([0.0, 1.0], (40, 1)),
        np.tile([1.0, 0.0], (80, 1)),
    ])

    fitted = fit_thresholds_from_val(
        probs, targets, classes,
        negative_max=0.5, margin_to_second_best=0.08,
        base_thresholds={"_negative": 0.5, "cheers": 0.5},
    )

    assert set(fitted) == {"cheers"}          # _negative is a gate, not a threshold
    assert fitted["cheers"] == 0.9            # highest grid point with perfect F1


def test_fit_thresholds_from_val_empty_probs_returns_base():
    from sed.train.eval import fit_thresholds_from_val

    fitted = fit_thresholds_from_val(
        np.zeros((0, 2)), np.zeros((0, 2)), ["_negative", "cheers"],
        negative_max=0.5, margin_to_second_best=0.08,
        base_thresholds={"cheers": 0.7},
    )
    assert fitted == {"cheers": 0.7}


def test_fit_thresholds_for_precision_takes_lowest_threshold_meeting_floor():
    """Positives at 0.9, hard negatives at 0.6: precision is 0.5 up to the 0.6
    grid point and 1.0 above it. The precision fitter must take the LOWEST
    threshold reaching the floor (max recall at the shared operating point) —
    unlike the F1 fitter, which resolves the same sweep to the highest (0.9)."""
    from sed.train.eval import fit_thresholds_for_precision

    classes = ["_negative", "cheers"]
    pos = np.tile([0.05, 0.90], (40, 1))
    hard = np.tile([0.10, 0.60], (40, 1))   # false triggers below/at 0.6
    easy = np.tile([0.90, 0.05], (40, 1))   # caught by the negative gate
    probs = np.vstack([pos, hard, easy])
    targets = np.vstack([
        np.tile([0.0, 1.0], (40, 1)),
        np.tile([1.0, 0.0], (80, 1)),
    ])

    fitted = fit_thresholds_for_precision(
        probs, targets, classes,
        target_precision=0.9,
        negative_max=0.5, margin_to_second_best=0.08,
        base_thresholds={"_negative": 0.5, "cheers": 0.5},
    )

    assert set(fitted) == {"cheers"}
    assert fitted["cheers"] == 0.625   # first grid point past the 0.6 false triggers


def test_fit_thresholds_for_precision_falls_back_to_most_precise():
    """A class that never reaches the target keeps its most precise threshold
    (ties resolve to the highest grid point with any triggers)."""
    from sed.train.eval import fit_thresholds_for_precision

    classes = ["_negative", "cheers"]
    pos = np.tile([0.05, 0.90], (40, 1))
    neg = np.tile([0.05, 0.90], (40, 1))    # indistinguishable → precision 0.5 everywhere
    probs = np.vstack([pos, neg])
    targets = np.vstack([
        np.tile([0.0, 1.0], (40, 1)),
        np.tile([1.0, 0.0], (40, 1)),
    ])

    fitted = fit_thresholds_for_precision(
        probs, targets, classes,
        target_precision=0.99,
        negative_max=0.95, margin_to_second_best=0.08,
        base_thresholds={"_negative": 0.5, "cheers": 0.5},
    )

    assert fitted["cheers"] == 0.9     # highest grid point that still triggers


def test_fit_thresholds_for_precision_empty_probs_returns_base():
    from sed.train.eval import fit_thresholds_for_precision

    fitted = fit_thresholds_for_precision(
        np.zeros((0, 2)), np.zeros((0, 2)), ["_negative", "cheers"],
        target_precision=0.9,
        negative_max=0.5, margin_to_second_best=0.08,
        base_thresholds={"cheers": 0.7},
    )
    assert fitted == {"cheers": 0.7}


def test_calibrate_run_thresholds_rejects_unknown_mode(tmp_path):
    from sed.train.eval import calibrate_run_thresholds

    with pytest.raises(ValueError, match="unknown calibration mode"):
        calibrate_run_thresholds(Config(), "p", "m", mode="venue_fp")


def test_overlay_runtime_settings_recognizes_val_precision_mode(tmp_path):
    from sed.config import dump_runtime_settings
    from sed.train.eval import _overlay_runtime_settings

    dump_runtime_settings(
        {
            "thresholds": {"cheers": 0.625},
            "calibration": {"mode": "val_precision", "target_precision": 0.9},
        },
        tmp_path / "runtime_settings.json",
    )
    _overlaid, source = _overlay_runtime_settings(tmp_path, Config())
    assert source == "val_calibrated"


def test_overlay_runtime_settings_sources_and_threshold_effect(tmp_path):
    from sed.config import dump_runtime_settings
    from sed.train.eval import _configured_thresholds, _overlay_runtime_settings

    cfg = Config()
    classes = ["_negative", "cheers"]

    # No runtime_settings.json → config defaults untouched.
    overlaid, source = _overlay_runtime_settings(tmp_path, cfg)
    assert source == "config_defaults"
    assert _configured_thresholds(overlaid, classes)["cheers"] == cfg.class_threshold("cheers")

    # Calibrated thresholds present → overlaid config resolves them.
    dump_runtime_settings(
        {"thresholds": {"cheers": 0.325}, "calibration": {"mode": "val_f1"}},
        tmp_path / "runtime_settings.json",
    )
    overlaid, source = _overlay_runtime_settings(tmp_path, cfg)
    assert source == "val_calibrated"
    assert _configured_thresholds(overlaid, classes)["cheers"] == 0.325

    # Thresholds without calibration provenance → plain runtime_settings.
    dump_runtime_settings(
        {"thresholds": {"cheers": 0.4}},
        tmp_path / "runtime_settings.json",
    )
    _overlaid, source = _overlay_runtime_settings(tmp_path, cfg)
    assert source == "runtime_settings"


def test_benchmark_profile_ignores_runtime_settings(tmp_path):
    from sed.config import dump_runtime_settings
    from sed.train.eval import (
        _configured_thresholds,
        _evaluation_artifacts,
        _overlay_runtime_settings,
        _thresholds_source_label,
    )

    cfg = Config()
    classes = ["_negative", "cheers"]
    dump_runtime_settings(
        {"thresholds": {"cheers": 0.325}, "calibration": {"mode": "val_f1"}},
        tmp_path / "runtime_settings.json",
    )

    overlaid, source = _overlay_runtime_settings(tmp_path, cfg, enabled=False)

    assert source == "config_snapshot"
    assert _configured_thresholds(overlaid, classes)["cheers"] == cfg.class_threshold("cheers")
    assert _thresholds_source_label(
        "val_calibrated", 0.9, use_runtime_settings=False
    ) == "fixed_threshold:0.900"
    assert _thresholds_source_label(
        "val_calibrated", 0.9, use_runtime_settings=True
    ) == "fixed_threshold:0.900+val_calibrated"
    assert _evaluation_artifacts("benchmark")["metrics_file"] == "benchmark_metrics.json"
    assert _evaluation_artifacts("test")["metrics_file"] == "test_metrics.json"


# ---- Frozen-test decision rules -----------------------------------------

@pytest.mark.parametrize("kind,param,rule_id,slug,label", [
    ("fixed", 0.90, "fixed@0.90", "fixed_0_90", "≥ 0.90"),
    ("argmax", None, "argmax", "argmax", "argmax"),
    ("val_f1", None, "val_f1", "val_f1", "F1-opt (val)"),
    ("val_precision", 0.90, "val_precision@0.90", "val_precision_0_90", "P≥0.90 (val)"),
    ("deployed", None, "deployed", "deployed", "deployed thresholds"),
])
def test_test_rule_identity_round_trip(kind, param, rule_id, slug, label):
    from sed.train.eval import TestRule, parse_rule_id

    rule = TestRule(kind, param)
    assert rule.id == rule_id
    assert rule.slug == slug
    assert rule.label == label
    # id → TestRule → id survives, and equality holds (frozen dataclass).
    assert parse_rule_id(rule.id) == rule


def test_test_rule_rejects_bad_kind_and_out_of_range_param():
    from sed.train.eval import TestRule

    with pytest.raises(ValueError):
        TestRule("coin_flip")
    with pytest.raises(ValueError):
        TestRule("fixed", 1.5)
    # A param on a parameterless kind is simply dropped, not an error.
    assert TestRule("argmax", 0.5).param is None


def test_decision_metrics_report_macro_precision_and_recall():
    classes = ["_negative", "a", "b"]
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.1, 0.7, 0.2],    # true=b, pred=a → hurts a's precision and b's recall
    ])
    targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
    pred_idx = np.array([0, 1, 2, 1])
    cm = _confusion_matrix(probs, targets, classes, pred_idx=pred_idx)
    m = _decision_metrics(probs, targets, classes, cm,
                          thresholds={"_negative": 0.5, "a": 0.5, "b": 0.5})
    pc = m["per_class"]
    expected_p = np.mean([pc["a"]["precision"], pc["b"]["precision"]])
    expected_r = np.mean([pc["a"]["recall"], pc["b"]["recall"]])
    assert abs(m["precision_macro_targets"] - expected_p) < 1e-9
    assert abs(m["recall_macro_targets"] - expected_r) < 1e-9
    # a: TP=1 FP=1 → P=0.5;  b: TP=1 FN=1 → R=0.5
    assert abs(pc["a"]["precision"] - 0.5) < 1e-9
    assert abs(pc["b"]["recall"] - 0.5) < 1e-9


def test_read_frozen_test_evals_folds_in_legacy_blocks():
    from sed.train.eval import read_frozen_test_evals

    manifest = {
        "frozen_test": {
            "argmax": {"f1_macro_targets": 0.9, "ap_macro_targets": 0.95,
                       "test_set_hash": "h1"},
        },
        # legacy fixed-threshold benchmark
        "benchmark_metrics": {"f1_macro_targets": 0.8, "threshold": 0.9,
                              "test_set_hash": "h1"},
        # legacy train-auto test, scored on the run's OWN snapshot
        "test_metrics": {"f1_macro_targets": 0.7, "runtime_settings_applied": True,
                         "test_source": "model", "test_set_hash": "h0"},
    }
    evals = read_frozen_test_evals(manifest)
    assert set(evals) == {"argmax", "fixed@0.90", "deployed"}
    assert evals["argmax"]["origin"] == "rule"
    assert evals["fixed@0.90"]["origin"] == "legacy_benchmark"
    assert evals["fixed@0.90"]["f1_macro_targets"] == 0.8
    # test_source "model" ⇒ flagged as the not-comparable training-time score
    assert evals["deployed"]["origin"] == "train_auto"
    # every entry carries a rule descriptor for the UI
    assert evals["argmax"]["rule"]["label"] == "argmax"


def test_read_frozen_test_evals_recovers_macro_pr_and_false_fire_from_per_class():
    """Blocks written before macro P/R and the false-fire rate existed must fill
    those columns from per_class rather than showing blanks — most manifests on
    disk predate the keys, and re-scoring every old run is not an option."""
    from sed.train.test_rules import read_frozen_test_evals

    manifest = {
        "frozen_test": {
            "fixed@0.90": {
                "f1_macro_targets": 0.88, "ap_macro_targets": 0.96,
                "test_set_hash": "h1",
                "per_class": {
                    "cheers": {"precision": 0.8, "recall": 0.6, "f1": 0.69,
                               "tp": 30, "fp": 10, "fn": 20, "support": 50},
                    "attention": {"precision": 0.6, "recall": 0.8, "f1": 0.69,
                                  "tp": 40, "fp": 20, "fn": 10, "support": 50},
                    # 25 of 100 background clips wrongly fired a target.
                    "_negative": {"precision": 0.9, "recall": 0.75, "f1": 0.82,
                                  "tp": 75, "fp": 5, "fn": 25, "support": 100},
                },
            },
        },
    }
    block = read_frozen_test_evals(manifest)["fixed@0.90"]
    # Macros average the TARGET classes only; _negative never counts.
    assert block["precision_macro_targets"] == pytest.approx(0.7)
    assert block["recall_macro_targets"] == pytest.approx(0.7)
    assert block["false_fire_rate"] == pytest.approx(0.25)


def test_test_rules_stays_torch_free():
    """`sed.train.test_rules` is imported on the inspect-server page-load path,
    where pulling in torch costs seconds. The prior-safe metric helpers added
    beside `false_fire_rate` must stay pure dict/arithmetic. Checked in a fresh
    interpreter because this session has already imported torch via eval."""
    probe = "import sed.train.test_rules, sys; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False", "importing test_rules pulled in torch"


def test_read_frozen_test_evals_recovers_prior_safe_false_alarm_figures():
    """`false_per_1000_negatives` and the test prior must be recoverable from
    per_class, so old manifests gain the prior-safe columns without re-scoring."""
    from sed.train.test_rules import read_frozen_test_evals

    manifest = {
        "frozen_test": {
            "fixed@0.90": {
                "f1_macro_targets": 0.88, "n_samples": 200,
                "per_class": {
                    "cheers": {"precision": 0.8, "recall": 0.6, "support": 100},
                    # 25 of 100 background clips wrongly fired a target.
                    "_negative": {"precision": 0.9, "recall": 0.75, "fn": 25, "support": 100},
                },
            },
        },
    }
    block = read_frozen_test_evals(manifest)["fixed@0.90"]
    assert block["false_per_1000_negatives"] == pytest.approx(250.0)
    assert block["n_negatives"] == 100
    assert block["negative_prior"] == pytest.approx(0.5)


def test_decision_metrics_reports_prior_and_prior_safe_false_alarm_rate():
    """The same model scored on two differently-balanced test sets keeps its
    per-negative false-alarm rate while its precision moves — which is exactly
    why the rate, not precision, is the cross-test-set comparison."""
    classes = ["_negative", "cheers"]

    def score(n_neg: int) -> dict:
        """One true `cheers`, plus n_neg background clips of which 25% fire."""
        n_fire = n_neg // 4
        probs = np.array(
            [[0.1, 0.9]]                       # the true positive
            + [[0.1, 0.9]] * n_fire            # background that fires
            + [[0.9, 0.1]] * (n_neg - n_fire)  # background correctly rejected
        )
        targets = np.array([[0, 1]] + [[1, 0]] * n_neg)
        cm = _confusion_matrix(probs, targets, classes)
        return _decision_metrics(probs, targets, classes, cm, {c: 0.5 for c in classes})

    balanced, skewed = score(4), score(100)

    assert balanced["n_negatives"] == 4
    assert balanced["negative_prior"] == pytest.approx(4 / 5)
    assert skewed["negative_prior"] == pytest.approx(100 / 101)
    # The model's per-negative error rate is identical in both, and the
    # prior-safe figure says so.
    assert balanced["false_per_1000_negatives"] == pytest.approx(250.0)
    assert skewed["false_per_1000_negatives"] == pytest.approx(250.0)
    # Precision for the same model at the same threshold is prior-bound: 1 TP
    # against 1 FP at the balanced prior, against 25 FP at the skewed one.
    assert balanced["precision_macro_targets"] == pytest.approx(1 / 2)
    assert skewed["precision_macro_targets"] == pytest.approx(1 / 26)


def test_read_frozen_test_evals_keeps_stored_metrics_over_derived_ones():
    """Derivation is a fallback, not an override: a stored macro wins."""
    from sed.train.test_rules import read_frozen_test_evals

    manifest = {"frozen_test": {"argmax": {
        "f1_macro_targets": 0.9, "precision_macro_targets": 0.123,
        "per_class": {"cheers": {"precision": 0.999, "recall": 0.5}},
    }}}
    block = read_frozen_test_evals(manifest)["argmax"]
    assert block["precision_macro_targets"] == 0.123
    assert block["recall_macro_targets"] == pytest.approx(0.5)   # absent ⇒ derived


def test_read_deployed_test_evals_keys_by_precision_and_rule():
    """FP16 and INT8 (and one rule vs the next) must accumulate side by side —
    a single flat block meant scoring one variant erased the other."""
    from sed.train.test_rules import read_deployed_test_evals

    manifest = {
        "deployed_test": {
            "fp16": {"deployed": {"f1_macro_targets": 0.91, "ir_precision": "fp16",
                                  "model_file": "model_fp16.xml"}},
            "int8": {
                "deployed": {"f1_macro_targets": 0.86, "ir_precision": "int8",
                             "model_file": "model_int8.xml"},
                "fixed@0.90": {"f1_macro_targets": 0.84, "ir_precision": "int8",
                               "model_file": "model_int8.xml"},
            },
        },
    }
    evals = read_deployed_test_evals(manifest)
    assert set(evals) == {"fp16:deployed", "int8:deployed", "int8:fixed@0.90"}
    assert evals["fp16:deployed"]["f1_macro_targets"] == 0.91
    assert evals["int8:fixed@0.90"]["rule"]["label"] == "≥ 0.90"
    assert evals["int8:deployed"]["origin"] == "rule"


def test_read_deployed_test_evals_bridges_the_pre_section_flat_block():
    """Old manifests carry one `test_metrics_exported`; it must surface under the
    precision its model_file names so those runs keep their deployed columns."""
    from sed.train.test_rules import read_deployed_test_evals

    evals = read_deployed_test_evals({
        "test_metrics_exported": {
            "f1_macro_targets": 0.89, "model_file": "model_int8.xml",
            "per_class": {"cheers": {"precision": 0.9, "recall": 0.7}},
        },
    })
    assert set(evals) == {"int8:deployed"}
    block = evals["int8:deployed"]
    assert block["origin"] == "legacy_exported"
    assert block["ir_precision"] == "int8"
    # Deployed P/R were dropped by the old writer; recover them too.
    assert block["precision_macro_targets"] == pytest.approx(0.9)
    assert block["recall_macro_targets"] == pytest.approx(0.7)


def test_deployed_rules_exclude_the_val_fitted_ones():
    """val_f1 / val_precision fit thresholds on CHECKPOINT probabilities, so
    applying them to a quantized IR would compare two different decision
    layers — the API must refuse rather than quietly produce a number."""
    from sed.train.eval import EXPORTED_RULE_KINDS

    assert set(EXPORTED_RULE_KINDS) == {"fixed", "argmax", "deployed"}


def test_metrics_manifest_block_is_one_shape_for_every_scoring_path():
    """Checkpoint-under-a-rule and exported-IR blocks must carry the same keys,
    or a metric silently exists in one tier of the runs table and not the other."""
    from sed.train.eval import _metrics_manifest_block

    metrics = {
        "rule": {"id": "fixed@0.90"}, "threshold": 0.9, "thresholds": {"cheers": 0.9},
        "thresholds_source": "fixed_threshold:0.900", "runtime_settings_applied": False,
        "score_profile": "fixed_threshold_benchmark", "prediction_mode": "runtime_thresholds",
        "probability_mode": "sigmoid", "negative_max": 0.5, "margin_to_second_best": 0.08,
        "f1_macro_targets": 0.9, "ap_macro_targets": 0.95,
        "precision_macro_targets": 0.88, "recall_macro_targets": 0.92,
        "false_fire_rate": 0.01, "accuracy": 0.93, "n_samples": 197,
        "per_class": {}, "test_source": "current", "test_set_hash": "h1",
    }
    checkpoint = _metrics_manifest_block(metrics)
    exported = _metrics_manifest_block(
        {**metrics, "model_file": "model_int8.xml", "ir_precision": "int8"})

    # Every metric the runs table reads survives in BOTH.
    for key in ("f1_macro_targets", "ap_macro_targets", "precision_macro_targets",
                "recall_macro_targets", "false_fire_rate", "accuracy", "n_samples"):
        assert checkpoint[key] == exported[key] == metrics[key]
    # …and only the exported one is tagged with an engine.
    assert "ir_precision" not in checkpoint and "model_file" not in checkpoint
    assert exported["ir_precision"] == "int8"
    assert exported["model_file"] == "model_int8.xml"


def test_read_frozen_test_evals_prefers_real_entry_over_reconstructed():
    from sed.train.eval import read_frozen_test_evals

    manifest = {
        "frozen_test": {
            "fixed@0.90": {"f1_macro_targets": 0.91, "test_set_hash": "h1"},
        },
        # a legacy benchmark at the SAME id must not clobber the real one
        "benchmark_metrics": {"f1_macro_targets": 0.42, "threshold": 0.9},
    }
    evals = read_frozen_test_evals(manifest)
    assert evals["fixed@0.90"]["f1_macro_targets"] == 0.91
    assert evals["fixed@0.90"]["origin"] == "rule"


def test_evaluate_run_under_rule_dispatches_each_kind(monkeypatch):
    """The dispatcher maps every rule kind to the right evaluate_run call, and
    the val-fitted rules use the NON-persisting fit — never the calibrator that
    rewrites runtime_settings.json."""
    import sed.train.eval as eval_mod
    from sed.train.eval import TestRule, evaluate_run_under_rule

    captured: dict = {}

    def fake_evaluate_run(cfg, project_id, model_id, **kw):
        captured.update(kw)
        return {"f1_macro_targets": 0.5}

    def fake_fit(cfg, project_id, model_id, *, mode, target_precision):
        captured["fit"] = (mode, target_precision)
        return {"cheers": 0.61}, 100

    def forbidden_calibrate(*a, **k):
        raise AssertionError("calibrate_run_thresholds must NOT run when scoring a rule")

    monkeypatch.setattr(eval_mod, "evaluate_run", fake_evaluate_run)
    monkeypatch.setattr(eval_mod, "fit_run_thresholds", fake_fit)
    monkeypatch.setattr(eval_mod, "calibrate_run_thresholds", forbidden_calibrate)

    def dispatch(rule):
        captured.clear()
        evaluate_run_under_rule(Config(), "p", "m", rule)
        return captured

    c = dispatch(TestRule("fixed", 0.9))
    assert c["threshold"] == 0.9 and c["use_runtime_settings"] is False
    assert c["manifest_section"] == ("frozen_test", "fixed@0.90")
    assert "decision" not in c
    # Every rule scores the CURRENT project frozen test, never the run's own
    # snapshot — that is what makes two runs comparable and what stops the
    # result being tagged as a training-time score.
    for rule in (TestRule("fixed", 0.9), TestRule("argmax"), TestRule("deployed")):
        assert dispatch(rule)["use_model_dataset"] is False

    c = dispatch(TestRule("argmax"))
    assert c["decision"] == "argmax" and c["use_runtime_settings"] is False

    c = dispatch(TestRule("deployed"))
    assert c["use_runtime_settings"] is True

    c = dispatch(TestRule("val_precision", 0.8))
    assert c["fit"] == ("val_precision", 0.8)
    assert c["thresholds_override"] == {"cheers": 0.61}
    assert c["use_runtime_settings"] is False


def test_collect_predictions_reports_batch_sample_and_class_progress(monkeypatch, tmp_path):
    """Frozen evaluation progress is generated by the scoring loop itself,
    rather than an elapsed-time animation that can claim work is moving when it is not."""
    import torch
    import sed.train.eval as eval_mod

    class FakeDataset:
        def __init__(self, *_args, **_kwargs):
            self.rows = [
                {"path": "/audio/first.wav", "class_name": "_negative"},
                {"path": "/audio/second.wav", "class_name": "cheers"},
            ]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return torch.zeros(4), torch.tensor([float(index == 0), float(index == 1)])

    class FakeModel:
        @classmethod
        def load_from_checkpoint(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return self

        def __call__(self, x):
            return torch.zeros((x.shape[0], 2))

    monkeypatch.setattr(eval_mod, "SedDataset", FakeDataset)
    monkeypatch.setattr(eval_mod, "SedLitModule", FakeModel)
    cfg = Config()
    cfg.training.batch_size = 1
    events = []

    probs, targets, classes, rows = eval_mod._collect_predictions(
        cfg, "p", tmp_path / "model.ckpt", tmp_path, ["_negative", "cheers"],
        progress=events.append,
    )

    assert probs.shape == (2, 2)
    assert targets.shape == (2, 2)
    assert classes == ["_negative", "cheers"]
    assert rows == FakeDataset().rows
    scoring = [event for event in events if event["phase"] == "scoring frozen test"]
    assert scoring[-1]["completed"] == 2
    assert scoring[-1]["total"] == 2
    assert scoring[-1]["current_name"] == "second.wav"
    assert scoring[-1]["current_class"] == "cheers"


def test_fit_run_thresholds_does_not_touch_runtime_settings(tmp_path, monkeypatch):
    """fit_run_thresholds is the pure fit: it must never write the bundle, so a
    val-fitted rule can be scored without changing what the model deploys."""
    import numpy as np

    import sed.config as cfg_mod
    import sed.train.eval as eval_mod
    from sed.config import ProjectLayout, dump_runtime_settings

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    layout = ProjectLayout("p")
    model_dir = layout.model_dir("m")
    (model_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (model_dir / "dataset" / "val.csv").write_text(
        "path,class_name\nx.wav,cheers\n", encoding="utf-8")
    runtime_path = model_dir / "runtime_settings.json"
    dump_runtime_settings({"thresholds": {"cheers": 0.5}}, runtime_path)
    before = runtime_path.read_bytes()

    classes = ["_negative", "cheers"]
    monkeypatch.setattr(eval_mod, "_model_config", lambda md, cfg: cfg)
    monkeypatch.setattr(eval_mod, "_load_model_classes", lambda md: classes)
    monkeypatch.setattr(eval_mod, "_csv_row_count", lambda p: 1)
    monkeypatch.setattr(
        "sed.train.registry.get_checkpoint", lambda p, m: model_dir / "ckpt")
    probs = np.array([[0.2, 0.8], [0.7, 0.3]])
    targets = np.array([[0, 1], [1, 0]])
    monkeypatch.setattr(
        eval_mod, "_collect_predictions",
        lambda *a, **k: (probs, targets, classes, [{}, {}]))

    result = eval_mod.fit_run_thresholds(Config(), "p", "m", mode="val_f1")
    assert result is not None
    fitted, n_val = result
    assert n_val == 2 and "cheers" in fitted
    # The whole point: the bundle is byte-for-byte unchanged.
    assert runtime_path.read_bytes() == before


def test_class_mapping_materializes_a_remapped_test_set(tmp_path):
    """The execution half of class mapping: folding a refined taxonomy onto the
    coarser space a run was trained on is what lets an orphaned run be scored at
    all. Without this, `check_runs` can only *propose* the fix."""
    from sed.train.eval import _materialize_mapped_dataset

    src = tmp_path / "dataset"
    src.mkdir()
    (src / "test.csv").write_text(
        "path,class_name,session_id\n"
        "/a.wav,loud,s1\n/b.wav,quiet,s2\n/c.wav,_negative,s3\n",
        encoding="utf-8",
    )

    out_dir, applied = _materialize_mapped_dataset(
        src, {"loud": "cheers", "quiet": "cheers"}, tmp_path / "mapped"
    )

    assert applied == {"loud": "cheers", "quiet": "cheers"}
    rows = list(csv.DictReader((out_dir / "test.csv").open()))
    assert [r["class_name"] for r in rows] == ["cheers", "cheers", "_negative"]
    # Paths must survive so leakage checks and prediction tables still line up.
    assert [r["path"] for r in rows] == ["/a.wav", "/b.wav", "/c.wav"]
    # The source is never edited in place.
    assert "loud" in (src / "test.csv").read_text()


def test_class_mapping_is_a_no_op_without_a_mapping(tmp_path):
    from sed.train.eval import _materialize_mapped_dataset

    src = tmp_path / "dataset"
    src.mkdir()
    (src / "test.csv").write_text("path,class_name\n/a.wav,cheers\n", encoding="utf-8")

    out_dir, applied = _materialize_mapped_dataset(src, {}, tmp_path / "mapped")
    assert out_dir == src and applied == {}


def test_class_mapping_only_records_labels_it_actually_changed(tmp_path):
    """`class_mapping` on a metrics block says the score is coarser than the test
    set — so it must list real merges, not every key that was offered."""
    from sed.train.eval import _materialize_mapped_dataset

    src = tmp_path / "dataset"
    src.mkdir()
    (src / "test.csv").write_text(
        "path,class_name\n/a.wav,cheers\n/b.wav,loud\n", encoding="utf-8")

    _out, applied = _materialize_mapped_dataset(
        src, {"loud": "cheers", "whistle": "cheers"}, tmp_path / "mapped"
    )
    assert applied == {"loud": "cheers"}, "an unused mapping entry is not a merge"
