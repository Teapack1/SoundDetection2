from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_training_batch_sizes_distinguish_physical_accumulated_and_global_batch():
    from sed.train.batching import training_batch_sizes

    training = SimpleNamespace(batch_size=16, accumulate_grad_batches=4)

    assert training_batch_sizes(training) == {
        "physical_batch_size": 16,
        "accumulate_grad_batches": 4,
        "world_size": 1,
        "effective_batch_size": 64,
    }
    assert training_batch_sizes(training, world_size=2)["effective_batch_size"] == 128


def test_training_batch_sizes_default_legacy_configs_to_one_accumulation_step():
    from sed.train.batching import training_batch_sizes

    assert training_batch_sizes(SimpleNamespace(batch_size=32)) == {
        "physical_batch_size": 32,
        "accumulate_grad_batches": 1,
        "world_size": 1,
        "effective_batch_size": 32,
    }


def test_lightning_trainer_kwargs_receive_gradient_accumulation():
    from sed.train.train import _trainer_batching_kwargs

    training = SimpleNamespace(batch_size=8, accumulate_grad_batches=3)

    assert _trainer_batching_kwargs(training) == {"accumulate_grad_batches": 3}


def test_resource_schedule_reports_physical_batches_and_optimizer_steps():
    from sed.train.resource_estimate import batch_schedule_counts

    assert batch_schedule_counts(100, {
        "physical_batch_size": 16,
        "accumulate_grad_batches": 4,
    }) == {
        "physical_batches_per_epoch": 7,
        "optimizer_steps_per_epoch": 2,
    }


def test_clink_compute_budget_is_advisory_only():
    from sed.data.inspect_server import _clink_cnn_v2_compute_verdict

    assert _clink_cnn_v2_compute_verdict(0.25) is None
    verdict = _clink_cnn_v2_compute_verdict(0.251)
    assert verdict is not None
    assert verdict["level"] == "warn"
    assert verdict["kind"] == "compute_budget"
    assert "Training remains allowed" in verdict["text"]


@pytest.mark.parametrize(
    ("bands", "frames", "expected_gmac", "over_budget"),
    [
        (128, 81, 0.081, False),
        (256, 81, 0.163, False),
        (128, 161, 0.158, False),
        (463, 81, 0.295, True),
        (925, 81, 0.589, True),
    ],
)
def test_clink_profile_mac_estimates_match_benchmark_table(
    bands, frames, expected_gmac, over_budget,
):
    from sed.data.inspect_server import (
        _clink_cnn_v2_compute_verdict,
        _clink_cnn_v2_gmac,
    )

    measured = _clink_cnn_v2_gmac(1, bands, frames)
    assert measured == pytest.approx(expected_gmac, abs=0.001)
    assert (_clink_cnn_v2_compute_verdict(measured) is not None) is over_budget


def _standardized_linear_cfg(train_csv):  # noqa: ANN001
    from sed.config import Config
    from sed.util import sha256_file

    cfg = Config()
    cfg.features.feature_type = "linear"
    cfg.features.standardize = True
    cfg.features.standardize_mean = -3.25
    cfg.features.standardize_std = 1.75
    cfg.features.standardize_stats_resolved = True
    cfg.features.standardize_stats_dataset_sha256 = sha256_file(train_csv)
    cfg.features.standardize_stats_split_seed = cfg.training.seed
    cfg.features.standardize_stats_val_fraction = cfg.dataset.val_fraction
    cfg.features.standardize_stats_clips = 24
    return cfg


def test_standardized_linear_preflight_accepts_fresh_stats_and_records_provenance(
    tmp_path,
):
    from sed.train.train import (
        _feature_standardization_provenance,
        _validate_fitted_feature_scaling,
    )

    train_csv = tmp_path / "train.csv"
    train_csv.write_text("path,class_name\n/a.wav,glass\n", encoding="utf-8")
    cfg = _standardized_linear_cfg(train_csv)

    _validate_fitted_feature_scaling(cfg, train_csv)
    assert _feature_standardization_provenance(cfg) == {
        "enabled": True,
        "method": "zscore",
        "scope": "global",
        "feature_type": "linear",
        "resolved": True,
        "mean": -3.25,
        "std": 1.75,
        "bands": 0,
        "dataset_sha256": cfg.features.standardize_stats_dataset_sha256,
        "split_seed": cfg.training.seed,
        "val_fraction": cfg.dataset.val_fraction,
        "clips": 24,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("standardize_stats_resolved", False, "statistics are not resolved"),
        ("standardize_stats_dataset_sha256", "", "dataset SHA-256 is missing"),
        ("standardize_stats_clips", 0, "clip count must be positive"),
        ("standardize_stats_split_seed", -1, "does not match training seed"),
        (
            "standardize_stats_val_fraction",
            0.0,
            "does not match training validation fraction",
        ),
        (
            "standardize_stats_dataset_sha256",
            "0" * 64,
            "current train.csv SHA-256 does not match",
        ),
    ],
)
def test_standardized_linear_preflight_rejects_unresolved_or_stale_stats(
    tmp_path, field, value, message,
):
    from sed.train.train import _validate_fitted_feature_scaling

    train_csv = tmp_path / "train.csv"
    train_csv.write_text("path,class_name\n/a.wav,glass\n", encoding="utf-8")
    cfg = _standardized_linear_cfg(train_csv)
    setattr(cfg.features, field, value)

    with pytest.raises(RuntimeError, match=message):
        _validate_fitted_feature_scaling(cfg, train_csv)


def test_parent_fine_tune_keeps_parent_standardization_provenance(tmp_path):
    from sed.train.train import _validate_fitted_feature_scaling

    train_csv = tmp_path / "child-train.csv"
    train_csv.write_text("path,class_name\nchild.wav,glass\n", encoding="utf-8")
    cfg = _standardized_linear_cfg(train_csv)
    cfg.model.initialization = "parent"
    cfg.model.parent_project = "pretraining"
    cfg.model.parent_run = "parent-run"
    cfg.features.standardize_stats_dataset_sha256 = "a" * 64
    cfg.features.standardize_stats_split_seed = cfg.training.seed + 1

    # Reusing these values is required for encoder transfer; the later strict
    # parent-compatibility check validates the complete numerical signature.
    _validate_fitted_feature_scaling(cfg, train_csv)


def test_fitted_scaling_gate_covers_standardized_mel(tmp_path):
    """A scratch mel z-score run must be gated exactly like linear (fix A)."""
    from sed.config import Config
    from sed.train.train import _validate_fitted_feature_scaling
    from sed.util import sha256_file

    train_csv = tmp_path / "train.csv"
    train_csv.write_text("path,class_name\n/a.wav,glass\n", encoding="utf-8")
    cfg = Config()
    cfg.features.scaling_method = "zscore"
    cfg.features.scaling_stats = {"mean": [-4.0], "std": [2.0]}
    cfg.features.standardize_stats_resolved = False
    cfg = Config.model_validate(cfg.model_dump(mode="json"))

    with pytest.raises(RuntimeError, match="statistics are not resolved"):
        _validate_fitted_feature_scaling(cfg, train_csv)

    # Resolved manual/checkpoint constants (no dataset hash) pass for mel.
    cfg.features.standardize_stats_resolved = True
    _validate_fitted_feature_scaling(cfg, train_csv)

    # A fitted-but-stale hash is rejected for mel too.
    cfg.features.standardize_stats_dataset_sha256 = "0" * 64
    cfg.features.standardize_stats_clips = 24
    cfg.features.standardize_stats_split_seed = cfg.training.seed
    cfg.features.standardize_stats_val_fraction = cfg.dataset.val_fraction
    with pytest.raises(RuntimeError, match="does not match the statistics dataset"):
        _validate_fitted_feature_scaling(cfg, train_csv)

    # A fresh fit passes.
    cfg.features.standardize_stats_dataset_sha256 = sha256_file(train_csv)
    _validate_fitted_feature_scaling(cfg, train_csv)
