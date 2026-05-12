"""Tests for the test-set evaluation helpers."""

from __future__ import annotations

import numpy as np

from sed.config import Config, InferenceCfg
from sed.train.eval import (
    _average_precision,
    _closed_set_decisions,
    _configured_thresholds,
    _confusion_matrix,
    _decision_metrics,
    _per_class_metrics,
    _prediction_records,
    _runtime_decisions,
    _write_prediction_table,
)
from sed.train.objective import logits_to_probabilities, probability_mode


def test_average_precision_perfect_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert _average_precision(scores, labels) == 1.0


def test_average_precision_no_positives():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 0, 0])
    assert _average_precision(scores, labels) == 0.0


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


def test_per_class_metrics_shape_and_macro():
    classes = ["_negative", "cheers_loud", "cheers_quiet"]
    probs = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.2, 0.8],
        [0.0, 0.8, 0.2],     # misclassified — true=cheers_quiet, pred=cheers_loud
    ])
    targets = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])
    m = _per_class_metrics(
        probs,
        targets,
        classes,
        thresholds={"_negative": 0.5, "cheers_loud": 0.5, "cheers_quiet": 0.5},
    )
    assert m["n_samples"] == 4
    assert set(m["per_class"].keys()) == set(classes)
    # macro over targets only (excludes _negative)
    target_names = ["cheers_loud", "cheers_quiet"]
    expected_f1 = np.mean([m["per_class"][c]["f1"] for c in target_names])
    assert abs(m["f1_macro_targets"] - expected_f1) < 1e-9
    # _negative has perfect TP=1 here
    assert m["per_class"]["_negative"]["f1"] == 1.0


def test_per_class_metrics_handles_empty():
    classes = ["_negative", "cheers"]
    probs = np.zeros((0, 2))
    targets = np.zeros((0, 2))
    m = _per_class_metrics(probs, targets, classes)
    assert m["n_samples"] == 0
    for c in classes:
        assert m["per_class"][c]["f1"] == 0.0


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
