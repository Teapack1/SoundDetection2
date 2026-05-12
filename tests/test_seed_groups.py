"""Tests for seed replication grouping.

Architectures were ranked on 1-3 point frozen-test gaps from single runs. These
cover the fingerprint that decides which runs are replicates of one experiment.
"""

from __future__ import annotations

import pytest

from sed.train.seed_groups import (
    experiment_key,
    group_runs,
    run_seed,
    summarize,
)


def _cfg(**over) -> dict:
    cfg = {
        "project": {"id": "p", "created_at": "2026-01-01T00:00:00Z"},
        "dataset": {"target_classes": ["_negative", "cheers"]},
        "features": {"n_mels": 128, "hop_length": 220},
        "model": {"backbone": "mn10_as", "pretrained": True},
        "training": {"seed": 42, "epochs": 60, "lr_head": 1e-4},
        "deploy": {"active_model": "run-a"},
    }
    for block, values in over.items():
        cfg[block] = {**cfg.get(block, {}), **values}
    return cfg


def test_runs_differing_only_by_seed_share_an_experiment_key():
    assert experiment_key(_cfg()) == experiment_key(_cfg(training={"seed": 7}))


def test_a_real_configuration_change_breaks_the_key():
    base = experiment_key(_cfg())
    assert experiment_key(_cfg(model={"backbone": "mn04_as"})) != base
    assert experiment_key(_cfg(training={"epochs": 80})) != base
    assert experiment_key(_cfg(features={"n_mels": 256})) != base
    assert experiment_key(_cfg(dataset={"target_classes": ["_negative", "wine"]})) != base


def test_deployment_and_ui_state_do_not_split_replicates():
    """Activating a model or changing a log level re-runs nothing."""
    base = experiment_key(_cfg())
    assert experiment_key(_cfg(deploy={"active_model": "other"})) == base
    assert experiment_key({**_cfg(), "logging": {"level": "DEBUG"}}) == base
    assert experiment_key({**_cfg(), "artnet": {"enabled": True}}) == base
    # Project creation time is provenance, not configuration.
    assert experiment_key(_cfg(project={"created_at": "2027-05-05T00:00:00Z"})) == base


def test_seed_derived_feature_stats_do_not_split_replicates():
    """Standardization stats are fitted from the seeded split, so two seed
    replicates disagree on them — they must not be part of the identity."""
    a = _cfg(features={"standardize_stats_split_seed": 42, "standardize_stats_clips": 1667})
    b = _cfg(features={"standardize_stats_split_seed": 7, "standardize_stats_clips": 1670})
    assert experiment_key(a) == experiment_key(b)


def test_run_seed_reads_the_seed_and_tolerates_junk():
    assert run_seed(_cfg()) == 42
    assert run_seed({}) is None
    assert run_seed({"training": {"seed": "not-a-number"}}) is None


def test_group_runs_marks_single_seed_groups_as_unreplicated():
    runs = [
        {"id": "a", "config": _cfg()},
        {"id": "b", "config": _cfg(training={"seed": 7})},
        {"id": "c", "config": _cfg(model={"backbone": "mn04_as"})},
    ]
    groups = {g["experiment_key"]: g for g in group_runs(runs)}
    assert len(groups) == 2

    replicated = next(g for g in groups.values() if g["n"] == 2)
    assert sorted(replicated["run_ids"]) == ["a", "b"]
    assert replicated["replicated"] is True
    assert replicated["n_distinct_seeds"] == 2

    lone = next(g for g in groups.values() if g["n"] == 1)
    assert lone["run_ids"] == ["c"]
    assert lone["replicated"] is False


def test_two_runs_at_the_same_seed_are_not_independent_replicates():
    """Re-running an identical config is not a second sample of the seed
    distribution — averaging them would understate the spread."""
    runs = [{"id": "a", "config": _cfg()}, {"id": "b", "config": _cfg()}]
    group = group_runs(runs)[0]
    assert group["n"] == 2
    assert group["n_distinct_seeds"] == 1
    assert group["replicated"] is False


def test_summarize_reports_no_spread_for_a_single_value():
    """sd of one run must be None, not 0.0 — zero would read as reproducible."""
    one = summarize([0.9])
    assert one["n"] == 1 and one["mean"] == pytest.approx(0.9) and one["sd"] is None

    many = summarize([0.90, 0.94, 0.92])
    assert many["n"] == 3
    assert many["mean"] == pytest.approx(0.92)
    assert many["sd"] == pytest.approx(0.02)          # sample sd (n-1)
    assert (many["min"], many["max"]) == (0.90, 0.94)


def test_summarize_ignores_missing_values():
    assert summarize([None, 0.8, None])["n"] == 1
    assert summarize([])["mean"] is None
    assert summarize([None, None])["sd"] is None


def test_seed_groups_stays_torch_free():
    import subprocess
    import sys

    probe = "import sed.train.seed_groups, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
