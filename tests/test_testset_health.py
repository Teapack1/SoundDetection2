"""Tests for frozen-test-set health.

Both failures these guard against were live on this project: the `cheers` test
had no `_negative` clips at all, and the `pretraining` test ranked architectures
on 24 positives per class.
"""

from __future__ import annotations

import pytest

from sed.train.testset_health import (
    health_for_test_csv,
    read_class_counts,
    frozen_test_health,
    wilson_halfwidth,
)


def _codes(health: dict) -> set[str]:
    return {f["code"] for f in health["findings"]}


def test_no_negatives_is_an_error_because_false_positives_go_unmeasured():
    """The real `cheers` test set: 197 positives, zero background."""
    health = frozen_test_health({"cheers_loud": 67, "cheers_medium": 64, "cheers_quiet": 66})

    assert health["ok"] is False
    assert health["can_measure_false_positives"] is False
    assert "no_negatives" in _codes(health)
    assert health["n_negatives"] == 0
    assert health["negative_prior"] == pytest.approx(0.0)


def test_low_support_is_reported_with_the_resolution_it_actually_buys():
    """The real `pretraining` test set: 74 positives against 42k negatives."""
    health = frozen_test_health({"_negative": 42272, "loud": 26, "medium": 24, "quiet": 24})

    assert health["can_measure_false_positives"] is True
    assert "low_support" in _codes(health)
    assert health["negative_prior"] == pytest.approx(42272 / 42346)

    # 24 clips resolves recall to roughly ±12 points at the 90% operating point,
    # so 1-3 point differences between architectures are not measurable.
    halfwidth = health["per_class"]["quiet"]["recall_ci_halfwidth"]
    assert 0.10 < halfwidth < 0.15
    # A large class is precise; the interval must shrink with support.
    assert health["per_class"]["_negative"]["recall_ci_halfwidth"] < 0.01


def test_a_well_formed_test_set_raises_no_errors_or_warnings():
    health = frozen_test_health({"_negative": 4000, "cheers": 400, "clap": 350})

    assert health["ok"] is True
    assert _codes(health) == {"prior"}, "only the informational prior note"
    assert health["n_target_clips"] == 750
    assert health["n_target_classes"] == 2


def test_empty_and_target_only_sets_are_errors():
    assert "empty" in _codes(frozen_test_health({}))
    assert "no_targets" in _codes(frozen_test_health({"_negative": 100}))


def test_findings_are_ordered_worst_first():
    health = frozen_test_health({"cheers": 10})
    severities = [f["severity"] for f in health["findings"]]
    assert severities == sorted(severities, key={"error": 0, "warn": 1, "info": 2}.get)


def test_wilson_halfwidth_shrinks_with_support_and_handles_degenerate_n():
    assert wilson_halfwidth(0.9, 0) is None
    wide, narrow = wilson_halfwidth(0.9, 25), wilson_halfwidth(0.9, 2500)
    assert wide > narrow > 0
    # Wilson stays inside [0, 1] near the boundary, unlike the normal approximation.
    assert 0.9 + wilson_halfwidth(0.99, 30) <= 1.5


def test_read_class_counts_and_health_from_a_composed_csv(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "path,class_name,session_id\n"
        "a.wav,_negative,s1\n"
        "b.wav,_negative,s1\n"
        "c.wav,cheers,s2\n",
        encoding="utf-8",
    )
    assert read_class_counts(csv_path) == {"_negative": 2, "cheers": 1}

    health = health_for_test_csv(csv_path)
    assert health["n_samples"] == 3
    assert health["negative_prior"] == pytest.approx(2 / 3)


def test_missing_test_csv_reports_a_missing_finding_rather_than_raising(tmp_path):
    health = health_for_test_csv(tmp_path / "absent.csv")
    assert health["ok"] is False
    assert "missing" in _codes(health)


def test_test_health_stays_torch_free():
    """Read on the inspect-server page-load path, where torch costs seconds."""
    import subprocess
    import sys

    probe = "import sed.train.testset_health, sys; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
