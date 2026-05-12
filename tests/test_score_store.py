"""Tests for the per-clip score store and the metrics re-derived from it.

The store exists so a frozen-test score can be re-thresholded without another
forward pass. That is only safe if the numbers recomputed from stored
probabilities are the numbers the evaluator wrote — so that round trip is what
these tests pin.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from sed.train import score_store
from sed.train.metrics import confusion_matrix, decision_metrics, runtime_pred_indices

CLASSES = ["_negative", "loud", "medium", "quiet"]


def _fake_run(tmp_path, *, n=200, seed=0):
    """A run directory with a plausible probability matrix and ground truth."""
    rng = np.random.default_rng(seed)
    true_idx = rng.integers(0, len(CLASSES), size=n).astype(np.int16)
    probs = rng.random((n, len(CLASSES))).astype(np.float32) * 0.4
    # Make the true class usually win, so the decision gates have something to do.
    probs[np.arange(n), true_idx] += 0.55
    run_dir = tmp_path / "2026-07-30_1200"
    (run_dir / "reports").mkdir(parents=True)
    return run_dir, probs, true_idx


def test_write_then_read_round_trips_exactly(tmp_path):
    run_dir, probs, true_idx = _fake_run(tmp_path)
    key = score_store.write_scores(
        run_dir,
        engine="checkpoint",
        test_set_hash="sha256:c042871754b0e982",
        classes=CLASSES,
        probs=probs,
        true_idx=true_idx,
        paths=[f"/clips/{i}.wav" for i in range(len(true_idx))],
        probability_mode="sigmoid",
    )
    assert key == "checkpoint__c0428717"

    loaded = score_store.read_scores(run_dir, key)
    assert loaded is not None
    got_probs, got_true, sidecar = loaded
    # float32 in, float32 out — the store must not quantize what the model produced.
    np.testing.assert_array_equal(got_probs, probs)
    np.testing.assert_array_equal(got_true, true_idx.astype(np.int64))
    assert sidecar["classes"] == CLASSES
    assert sidecar["n"] == len(true_idx)
    assert sidecar["probability_mode"] == "sigmoid"


def test_hash8_accepts_prefixed_and_bare_digests():
    assert score_store.hash8("sha256:C042871754B0") == "c0428717"
    assert score_store.hash8("c042871754b0") == "c0428717"
    assert score_store.hash8("") == "unknown"


def test_engines_do_not_share_a_key():
    """A quantized IR disagrees with the checkpoint by design, so its scores
    must never land in the same slot."""
    h = "sha256:c042871754b0"
    keys = {score_store.score_key(e, h) for e in ("checkpoint", "ov_fp16", "ov_int8")}
    assert len(keys) == 3


def test_one_hot_ignores_out_of_range_labels(tmp_path):
    out = score_store.one_hot(np.array([0, 2, -1, 99]), 3)
    assert out.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]


def test_metrics_recomputed_from_store_match_the_direct_computation(tmp_path):
    """The whole premise: stored probabilities reproduce the stored metrics.

    Anything that breaks this makes the Runs & compare numbers a second,
    silently-diverging opinion rather than the same measurement re-cut.
    """
    run_dir, probs, true_idx = _fake_run(tmp_path, n=500, seed=7)
    targets = score_store.one_hot(true_idx, len(CLASSES))
    thresholds = {"_negative": 0.5, "loud": 0.9, "medium": 0.9, "quiet": 0.9}

    def compute(p):
        pred = runtime_pred_indices(p, CLASSES, thresholds, 0.5, 0.08)
        cm = confusion_matrix(p, targets, CLASSES, pred_idx=pred)
        return decision_metrics(
            p, targets, CLASSES, cm, thresholds=thresholds, threshold=0.9,
            margin_to_second_best=0.08,
        )

    direct = compute(probs)
    key = score_store.write_scores(
        run_dir, engine="checkpoint", test_set_hash="sha256:abcdef0123456789",
        classes=CLASSES, probs=probs, true_idx=true_idx,
        paths=[""] * len(true_idx),
    )
    stored_probs, stored_true, _side = score_store.read_scores(run_dir, key)
    replayed = compute(stored_probs)

    np.testing.assert_array_equal(stored_true, true_idx.astype(np.int64))
    for metric in (
        "f1_macro_targets", "ap_macro_targets", "precision_macro_targets",
        "recall_macro_targets", "false_per_1000_negatives", "accuracy",
        "cross_target_confusion", "band_pauc",
    ):
        assert replayed[metric] == pytest.approx(direct[metric]), metric


def test_backfill_recovers_a_store_from_a_prediction_table(tmp_path):
    """Runs scored before the store existed keep their probabilities in the TSV;
    recovering them must not need a GPU."""
    run_dir, probs, true_idx = _fake_run(tmp_path, n=40, seed=3)
    reports = run_dir / "reports"
    header = [
        "index", "file", "true_class", "predicted_class", "path",
        *[f"p:{c}" for c in CLASSES],
    ]
    lines = ["\t".join(header)]
    for i in range(len(true_idx)):
        lines.append("\t".join([
            str(i), f"{i}.wav", CLASSES[true_idx[i]], CLASSES[int(np.argmax(probs[i]))],
            f"/clips/{i}.wav", *[f"{probs[i, c]:.6f}" for c in range(len(CLASSES))],
        ]))
    (reports / "fixed_0_90_predictions.tsv").write_text("\n".join(lines), encoding="utf-8")
    (reports / "fixed_0_90_metrics.json").write_text(
        json.dumps({"test_set_hash": "sha256:deadbeef12345678", "probability_mode": "sigmoid"}),
        encoding="utf-8",
    )

    result = score_store.backfill_run(run_dir)
    assert result["written"] == "checkpoint__deadbeef"

    loaded = score_store.read_scores(run_dir, result["written"])
    assert loaded is not None
    got_probs, got_true, sidecar = loaded
    np.testing.assert_array_equal(got_true, true_idx.astype(np.int64))
    # The TSV rounds to six decimals — close, but explicitly not exact, which is
    # why the sidecar records where the numbers came from.
    np.testing.assert_allclose(got_probs, probs, atol=1e-6)
    assert sidecar["source"] == "backfilled"
    assert sidecar["backfilled_from"] == "fixed_0_90_predictions.tsv"


def test_backfill_refuses_a_table_without_a_test_set_hash(tmp_path):
    """A score that cannot be tied to a test set cannot be compared to anything,
    so it is better absent than silently filed under the wrong benchmark."""
    run_dir, probs, true_idx = _fake_run(tmp_path, n=5)
    reports = run_dir / "reports"
    header = ["true_class", "path", *[f"p:{c}" for c in CLASSES]]
    lines = ["\t".join(header)]
    for i in range(len(true_idx)):
        lines.append("\t".join([
            CLASSES[true_idx[i]], f"/clips/{i}.wav",
            *[f"{probs[i, c]:.6f}" for c in range(len(CLASSES))],
        ]))
    (reports / "test_predictions.tsv").write_text("\n".join(lines), encoding="utf-8")

    result = score_store.backfill_run(run_dir)
    assert result["written"] is None
    assert "test_set_hash" in result["reason"]


def test_backfill_is_idempotent(tmp_path):
    run_dir, probs, true_idx = _fake_run(tmp_path, n=10)
    reports = run_dir / "reports"
    header = ["true_class", "path", *[f"p:{c}" for c in CLASSES]]
    lines = ["\t".join(header)] + [
        "\t".join([CLASSES[true_idx[i]], f"/c/{i}.wav",
                   *[f"{probs[i, c]:.6f}" for c in range(len(CLASSES))]])
        for i in range(len(true_idx))
    ]
    (reports / "test_predictions.tsv").write_text("\n".join(lines), encoding="utf-8")
    (reports / "test_metrics.json").write_text(
        json.dumps({"test_set_hash": "sha256:aaaabbbbcccc"}), encoding="utf-8")

    first = score_store.backfill_run(run_dir)
    second = score_store.backfill_run(run_dir)
    assert first["written"] == second["written"]
    assert second["reason"] == "already present"


def test_available_lists_only_complete_pairs(tmp_path):
    run_dir, probs, true_idx = _fake_run(tmp_path, n=8)
    key = score_store.write_scores(
        run_dir, engine="checkpoint", test_set_hash="sha256:1111222233334444",
        classes=CLASSES, probs=probs, true_idx=true_idx, paths=[""] * len(true_idx),
    )
    assert set(score_store.available(run_dir)) == {key}
    # A sidecar with no matrix beside it is not a usable score.
    (score_store.scores_dir(run_dir) / f"{key}.npz").unlink()
    assert score_store.available(run_dir) == {}


def test_score_store_and_metrics_stay_torch_free():
    """Both are reached on the inspect-server request path, where importing the
    training runtime costs seconds per page."""
    probe = (
        "import sed.train.score_store, sed.train.metrics, "
        "sed.train.operating_point, sys; print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False", "the metric core pulled in torch"
