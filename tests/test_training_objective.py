from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from sed.config import Config, ProjectLayout
from sed.train.datamodule import SedDataset
from sed.train.train import (
    BestValCheckpoint,
    GracefulStopCallback,
    _semantic_checkpoint_path,
    _write_val_csv_snapshot,
    train_project,
)


def _write_dataset_csv(path: Path, wav: Path, class_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "path", "class_name", "session_id", "sample_rate", "duration_s",
            "is_target", "origin",
        ])
        w.writerow([str(wav), class_name, "s1", 32000, "1.0000", 1, "project"])


def _write_dataset_rows(path: Path, rows: list[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "path", "class_name", "session_id", "sample_rate", "duration_s",
            "is_target", "origin",
        ])
        for idx, (wav, class_name) in enumerate(rows):
            w.writerow([str(wav), class_name, f"s{idx}", 32000, "1.0000", 1, "project"])


def _write_empty_dataset_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "path", "class_name", "session_id", "sample_rate", "duration_s",
            "is_target", "origin",
        ])


def _write_minimal_dataset(layout: ProjectLayout, wav: Path, classes: list[str]) -> None:
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_dataset_csv(layout.dataset_dir / "train.csv", wav, classes[-1])
    _write_empty_dataset_csv(layout.dataset_dir / "val.csv")
    _write_empty_dataset_csv(layout.dataset_dir / "test.csv")
    classes_json = str(classes).replace("'", '"')
    index_json = ", ".join(f'"{c}": {i}' for i, c in enumerate(classes))
    (layout.dataset_dir / "classes.json").write_text(
        f'{{"classes": {classes_json}, "class_index": {{{index_json}}}}}\n',
        encoding="utf-8",
    )
    (layout.dataset_dir / "dataset_snapshot.json").write_text("{}", encoding="utf-8")


def _efficientat_cfg(classes: list[str]) -> Config:
    cfg = Config()
    cfg.model.classes = list(classes)
    cfg.dataset.target_classes = list(classes)
    cfg.dataset.classes = list(classes)
    cfg.features.frontend = "efficientat"
    cfg.features.fmax = 15000
    cfg.training.auto_export = False
    cfg.training.auto_test = False
    return cfg


def test_cross_entropy_dataset_returns_class_index(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    csv_path = tmp_path / "test.csv"
    _write_dataset_csv(csv_path, wav, "cheers")
    cfg = Config()
    cfg.training.loss = "cross_entropy"

    ds = SedDataset(csv_path, ["_negative", "cheers"], cfg)
    _x, y = ds[0]

    assert y.item() == 1


def test_legacy_categorical_crossentropy_alias_uses_softmax_objective(tmp_path):
    from sed.train.objective import probability_mode, training_metric_descriptor

    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    csv_path = tmp_path / "test.csv"
    _write_dataset_csv(csv_path, wav, "cheers")
    cfg = Config()
    cfg.training.loss = "categorical_crossentropy"
    loaded = Config.model_validate({"training": {"loss": "categorical_crossentropy"}})

    ds = SedDataset(csv_path, ["_negative", "cheers"], cfg)
    _x, y = ds[0]
    desc = training_metric_descriptor(cfg)

    assert y.item() == 1
    assert loaded.training.loss == "categorical_crossentropy"
    assert probability_mode(cfg) == "softmax"
    assert desc["effective_loss"] == "cross_entropy"
    assert desc["probability_mode"] == "softmax"


def test_default_bce_dataset_returns_one_hot_target(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    csv_path = tmp_path / "test.csv"
    _write_dataset_csv(csv_path, wav, "cheers")
    cfg = Config()

    ds = SedDataset(csv_path, ["_negative", "cheers"], cfg)
    _x, y = ds[0]

    assert cfg.training.loss == "bce_logits"
    assert y.tolist() == [0.0, 1.0]


def test_legacy_dataset_returns_one_hot_target(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    csv_path = tmp_path / "test.csv"
    _write_dataset_csv(csv_path, wav, "cheers")
    cfg = Config()
    cfg.training.loss = "asymmetric_focal"

    ds = SedDataset(csv_path, ["_negative", "cheers"], cfg)
    _x, y = ds[0]

    assert y.tolist() == [0.0, 1.0]


def test_training_metric_descriptor_matches_loss_modes():
    from sed.train.objective import training_metric_descriptor

    cfg = Config()
    bce = training_metric_descriptor(cfg)
    assert bce["effective_loss"] == "bce_logits"
    assert bce["probability_mode"] == "sigmoid"
    assert "Per-label" in bce["val_accuracy_description"]

    cfg.training.loss = "cross_entropy"
    ce = training_metric_descriptor(cfg)
    assert ce["effective_loss"] == "cross_entropy"
    assert ce["probability_mode"] == "softmax"
    assert "argmax" in ce["val_accuracy_description"]

    cfg.training.loss = "asymmetric_focal"
    focal = training_metric_descriptor(cfg)
    assert focal["effective_loss"] == "asymmetric_focal"
    assert focal["probability_mode"] == "sigmoid"


def test_training_metric_descriptor_reports_frame_and_byol_overrides():
    from sed.train.objective import training_metric_descriptor

    cfg = Config()
    cfg.model.task = "frame"
    cfg.training.loss = "cross_entropy"
    frame = training_metric_descriptor(cfg)
    assert frame["effective_loss"] == "frame_bce_logits"
    assert frame["probability_mode"] == "sigmoid"

    cfg.training.pretraining_objective = "byol"
    byol = training_metric_descriptor(cfg)
    assert byol["effective_loss"] == "byol_ssl"
    assert byol["show_val_accuracy"] is False
    assert byol["val_score_key"] == "val/byol_score"


def test_training_dataset_resamples_raw_48k_clip_to_model_rate(tmp_path):
    wav = tmp_path / "raw_48k.wav"
    sf.write(str(wav), np.linspace(-0.5, 0.5, 48000, dtype="float32"), 48000)
    csv_path = tmp_path / "train.csv"
    _write_dataset_csv(csv_path, wav, "cheers")
    cfg = Config()
    cfg.model.sample_rate = 32000
    cfg.features.window_seconds = 1.0

    ds = SedDataset(csv_path, ["_negative", "cheers"], cfg)
    wave, crop_offset_s = ds._load(str(wav))

    assert wave.shape == (32000,)
    assert wave.dtype == np.float32
    assert crop_offset_s == 0.0


def test_training_dataset_skips_unreadable_audio_rows(tmp_path, caplog):
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    sf.write(str(good), np.zeros(32000, dtype="float32"), 32000)
    bad.write_bytes(b"not really a wav")
    csv_path = tmp_path / "train.csv"
    _write_dataset_rows(csv_path, [(bad, "cheers"), (good, "cheers")])
    cfg = Config()

    caplog.set_level("WARNING", logger="sed.train.datamodule")
    ds = SedDataset(csv_path, ["_negative", "cheers"], cfg)

    assert len(ds) == 1
    assert ds.rows[0]["path"] == str(good)
    assert len(ds.skipped_audio_rows) == 1
    assert "skipped 1 unreadable/empty audio row" in caplog.text


def test_fixed_threshold_trigger_metrics_replay_runtime_gates():
    import torch

    from sed.train.lit_module import (
        fixed_threshold_trigger_counts,
        summarize_fixed_threshold_counts,
    )

    # classes: _negative, glass_a, glass_b
    probabilities = torch.tensor([
        [0.10, 0.96, 0.02],  # correct A trigger
        [0.70, 0.96, 0.02],  # correct A score, vetoed by _negative
        [0.02, 0.91, 0.94],  # correct B winner, abstains on 0.05 margin
        [0.10, 0.95, 0.01],  # false trigger on a truly negative clip
        [0.80, 0.96, 0.01],  # truly negative, vetoed rather than triggered
    ])
    targets = torch.tensor([1, 1, 2, 0, 0])

    counts = fixed_threshold_trigger_counts(
        probabilities,
        targets,
        negative_index=0,
        threshold=0.9,
        negative_max=0.5,
        margin_to_second_best=0.05,
    )
    recall, fp_per_1000 = summarize_fixed_threshold_counts(
        *counts,
        negative_index=0,
    )

    assert counts[0].tolist() == [0, 1, 0]
    assert counts[1].tolist() == [2, 2, 1]
    assert counts[2].item() == 1
    assert counts[3].item() == 2
    assert recall.item() == 0.25
    assert fp_per_1000.item() == 500.0


def test_fixed_threshold_metrics_without_negative_clips_are_finite():
    import torch

    from sed.train.lit_module import (
        fixed_threshold_trigger_counts,
        summarize_fixed_threshold_counts,
    )

    counts = fixed_threshold_trigger_counts(
        torch.tensor([[0.95, 0.05], [0.10, 0.96]]),
        torch.tensor([0, 1]),
        negative_index=None,
        threshold=0.9,
    )
    recall, fp_per_1000 = summarize_fixed_threshold_counts(
        *counts,
        negative_index=None,
    )

    assert recall.item() == 1.0
    assert counts[3].item() == 0
    assert fp_per_1000.item() == 0.0


def test_best_val_checkpoint_uses_false_positive_lexicographic_policy(tmp_path):
    from sed.train.lit_module import VAL_FIXED_FP_KEY, VAL_FIXED_RECALL_KEY

    def metrics(recall, fp, ap, loss):
        return {
            VAL_FIXED_RECALL_KEY: recall,
            VAL_FIXED_FP_KEY: fp,
            "val/ap_macro_targets": ap,
            "val/loss": loss,
        }

    cb = BestValCheckpoint(tmp_path)
    below_floor = metrics(0.94, 0.0, 0.99, 0.1)
    eligible = metrics(0.95, 20.0, 0.80, 0.4)
    assert cb._is_better_supervised(below_floor) is True
    cb.best_selection_metrics = below_floor
    # Reaching the recall floor always beats an ineligible epoch.
    assert cb._is_better_supervised(eligible) is True
    cb.best_selection_metrics = eligible
    # Once eligible, higher recall does not compensate for more false triggers.
    assert cb._is_better_supervised(metrics(0.99, 30.0, 0.99, 0.1)) is False
    assert cb._is_better_supervised(metrics(0.95, 10.0, 0.70, 0.8)) is True
    cb.best_selection_metrics = metrics(0.95, 10.0, 0.70, 0.8)
    # Equal false-trigger rates break on AP, then loss.
    assert cb._is_better_supervised(metrics(0.95, 10.0, 0.80, 0.9)) is True
    cb.best_selection_metrics = metrics(0.95, 10.0, 0.80, 0.9)
    assert cb._is_better_supervised(metrics(0.95, 10.0, 0.80, 0.7)) is True


def test_best_val_checkpoint_fallback_prioritizes_recall_before_false_positives(tmp_path):
    from sed.train.lit_module import VAL_FIXED_FP_KEY, VAL_FIXED_RECALL_KEY

    cb = BestValCheckpoint(tmp_path)
    cb.best_selection_metrics = {
        VAL_FIXED_RECALL_KEY: 0.90,
        VAL_FIXED_FP_KEY: 0.0,
        "val/ap_macro_targets": 0.99,
        "val/loss": 0.1,
    }
    candidate = {
        VAL_FIXED_RECALL_KEY: 0.94,
        VAL_FIXED_FP_KEY: 100.0,
        "val/ap_macro_targets": 0.50,
        "val/loss": 1.0,
    }

    assert cb._is_better_supervised(candidate) is True


def test_best_val_checkpoint_loss_mode_tie_breaks_on_accuracy(tmp_path):
    cb = BestValCheckpoint(
        tmp_path, monitor="val/loss", mode="min",
        tiebreak="val/accuracy", tiebreak_mode="max", min_delta=0.001,
    )

    assert cb._is_better(0.5, 0.8) is True
    cb.best_score = 0.5
    cb.best_tiebreak = 0.8
    assert cb._is_better(0.4995, 0.7) is False
    assert cb._is_better(0.4995, 0.9) is True
    assert cb._is_better(0.498, 0.1) is True


def test_graceful_stop_callback_stops_after_validation(tmp_path):
    stop_file = tmp_path / "stop"
    stop_file.write_text("stop", encoding="utf-8")
    trainer = SimpleNamespace(sanity_checking=False, should_stop=False)
    cb = GracefulStopCallback(stop_file, has_val=True)

    cb.on_train_epoch_end(trainer, None)
    assert trainer.should_stop is False

    cb.on_validation_end(trainer, None)
    assert trainer.should_stop is True
    assert cb.triggered is True


def test_graceful_stop_callback_stops_after_epoch_without_validation(tmp_path):
    stop_file = tmp_path / "stop"
    stop_file.write_text("stop", encoding="utf-8")
    trainer = SimpleNamespace(sanity_checking=False, should_stop=False)
    cb = GracefulStopCallback(stop_file, has_val=False)

    cb.on_train_epoch_end(trainer, None)
    assert trainer.should_stop is True
    assert cb.triggered is True


def test_mixup_batch_convex_combines_inputs_and_targets():
    import torch

    from sed.train.lit_module import mixup_batch

    torch.manual_seed(0)
    x = torch.stack([torch.zeros(1, 4, 4), torch.ones(1, 4, 4)])
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    mx, my = mixup_batch(x, y, alpha=0.3)

    assert mx.shape == x.shape and my.shape == y.shape
    # one-hot rows stay a distribution after mixing
    assert torch.allclose(my.sum(dim=1), torch.ones(2))
    assert ((my >= 0) & (my <= 1)).all()
    # inputs are convex combinations of the two originals
    assert ((mx >= 0) & (mx <= 1)).all()


def test_clip_training_step_applies_mixup_for_bce_only():
    import torch

    from sed.config import Config
    from sed.train.lit_module import SedLitModule

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "glass"]}})
    cfg.model.backbone = "kws-tcresnet"
    cfg.model.pretrained = False
    cfg.model.params = {"time_stride": 1}
    cfg.features.window_seconds = 0.4
    cfg.training.loss = "bce_logits"
    cfg.training.mixup_alpha = 0.3
    module = SedLitModule(cfg, ["_negative", "glass"])

    torch.manual_seed(0)
    batch = (torch.randn(4, 1, 128, 41), torch.eye(2).repeat(2, 1))
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)


def _tiny_module_cfg(loss: str = "bce_logits"):
    from sed.config import Config

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative", "glass"]}})
    cfg.model.backbone = "kws-tcresnet"
    cfg.model.pretrained = False
    cfg.model.params = {"time_stride": 1}
    cfg.features.window_seconds = 0.4
    cfg.training.loss = loss
    return cfg


def test_legacy_categorical_crossentropy_builds_cross_entropy_loss():
    import torch.nn as nn

    from sed.train.lit_module import SedLitModule

    cfg = _tiny_module_cfg("categorical_crossentropy")
    cfg.training.mixup_alpha = 0.3
    module = SedLitModule(cfg, ["_negative", "glass"])

    assert module.single_label is True
    assert module._mixup_active is False
    assert isinstance(module.loss_fn, nn.CrossEntropyLoss)


@pytest.mark.parametrize("loss_name", ["cross_entropy", "bce_logits", "asymmetric_focal"])
def test_supervised_losses_keep_all_one_class_weight_buffers(loss_name):
    """Class frequency is corrected once by the sampler, not again by loss."""
    import torch

    from sed.train.lit_module import SedLitModule

    module = SedLitModule(
        _tiny_module_cfg(loss_name),
        ["glass_a", "_negative", "glass_b"],
    )

    if loss_name == "cross_entropy":
        weights = module.loss_fn.weight
    elif loss_name == "bce_logits":
        weights = module.loss_fn.pos_weight
    else:
        weights = module.loss_fn.class_weights
    assert torch.equal(weights, torch.ones(3))


def test_mixup_is_not_applied_to_asymmetric_focal_loss(monkeypatch):
    """AsymmetricFocalLoss hard-thresholds targets at 0.5, so soft mixed labels
    would flip its focusing direction — mixup must stay gated to bce_logits."""
    import torch

    from sed.train import lit_module as lm

    calls: list[str] = []
    real_mixup = lm.mixup_batch
    monkeypatch.setattr(
        lm, "mixup_batch",
        lambda x, y, a: (calls.append("hit"), real_mixup(x, y, a))[1],
    )

    torch.manual_seed(0)
    batch = (torch.randn(4, 1, 128, 41), torch.eye(2).repeat(2, 1))

    cfg = _tiny_module_cfg("asymmetric_focal")
    cfg.training.mixup_alpha = 0.3
    focal = lm.SedLitModule(cfg, ["_negative", "glass"])
    assert not focal._mixup_active
    assert torch.isfinite(focal.training_step(batch, 0))
    assert calls == []

    cfg = _tiny_module_cfg("bce_logits")
    cfg.training.mixup_alpha = 0.3
    bce = lm.SedLitModule(cfg, ["_negative", "glass"])
    assert bce._mixup_active
    assert torch.isfinite(bce.training_step(batch, 0))
    assert calls == ["hit"]


def test_optimizer_excludes_norm_and_bias_from_weight_decay():
    from sed.train.lit_module import SedLitModule

    cfg = _tiny_module_cfg()
    cfg.training.weight_decay = 0.05
    module = SedLitModule(cfg, ["_negative", "glass"])

    optim = module.configure_optimizers()["optimizer"]
    seen = 0
    for group in optim.param_groups:
        for p in group["params"]:
            seen += 1
            if p.ndim <= 1:
                assert group["weight_decay"] == 0.0
            else:
                assert group["weight_decay"] == 0.05
    assert seen == len(list(module.model.parameters())) > 0


def test_focal_loss_config_default_matches_class_default():
    from sed.config import Config
    from sed.train.model import AsymmetricFocalLoss
    import inspect

    sig = inspect.signature(AsymmetricFocalLoss.__init__)
    cfg = Config.model_validate({"dataset": {"target_classes": ["a"]}})
    assert sig.parameters["gamma_neg"].default == cfg.training.focal_gamma_neg
    assert sig.parameters["gamma_pos"].default == cfg.training.focal_gamma_pos


def test_semantic_checkpoint_path_uses_best_epoch(tmp_path):
    best = {"epoch": 3, "val_loss": 0.12345, "val_accuracy": 0.98765}

    path = _semantic_checkpoint_path(
        tmp_path,
        tmp_path / "x.ckpt",
        best,
        fallback_epoch=0,
    )

    assert path.name == "checkpoint_best_epoch-003.ckpt"


def test_efficientat_loader_suppresses_model_stdout(tmp_path, monkeypatch, capsys):
    fake_root = tmp_path / "EfficientAT"
    (fake_root / "metadata").mkdir(parents=True)
    (fake_root / "metadata" / "class_labels_indices.csv").write_text("index,mid,display_name\n")
    model_py = fake_root / "models" / "mn" / "model.py"
    model_py.parent.mkdir(parents=True)
    model_py.write_text(
        "import torch.nn as nn\n"
        "def get_model(width_mult=1.0, pretrained_name=None):\n"
        "    print('LOUD MODEL SUMMARY')\n"
        "    return nn.Sequential(nn.Flatten(), nn.Linear(128 * 100, 8))\n",
        encoding="utf-8",
    )
    for pkg in [fake_root / "models", fake_root / "models" / "mn"]:
        (pkg / "__init__.py").write_text("", encoding="utf-8")
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    from sed.train import model as model_mod

    monkeypatch.setattr(model_mod, "EFFICIENTAT_PATH", fake_root)
    loaded = model_mod._load_efficientat("mn10_as", pretrained=False)
    sys.path = [p for p in sys.path if p != str(fake_root)]
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            sys.modules.pop(name, None)

    assert loaded is not None
    assert "LOUD MODEL SUMMARY" not in capsys.readouterr().out


def test_train_project_rejects_dataset_class_config_drift(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("p")
    layout.create()
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    _write_minimal_dataset(layout, wav, ["_negative", "cheers"])

    cfg = _efficientat_cfg(["_negative", "other"])

    try:
        train_project(cfg, "p", auto_export=False, auto_test=False)
    except RuntimeError as exc:
        assert "dataset/classes.json does not match" in str(exc)
    else:
        raise AssertionError("expected class drift error")


def test_train_project_rejects_unresolved_standardized_linear_stats_before_bundle(
    tmp_path, monkeypatch,
):
    from sed import config as cfg_mod
    from sed.train import train as train_mod
    from sed.util import sha256_file

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("p")
    layout.create()
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    _write_minimal_dataset(layout, wav, ["_negative", "cheers"])

    cfg = _efficientat_cfg(["_negative", "cheers"])
    cfg.features.feature_type = "linear"
    cfg.features.standardize = True
    cfg.features.standardize_stats_resolved = False
    cfg.features.standardize_stats_dataset_sha256 = sha256_file(
        layout.dataset_dir / "train.csv",
    )
    cfg.features.standardize_stats_split_seed = cfg.training.seed
    cfg.features.standardize_stats_clips = 1

    monkeypatch.setattr(
        train_mod.registry,
        "new_model_id",
        lambda *_args, **_kwargs: pytest.fail("bundle id allocated before preflight"),
    )
    with pytest.raises(RuntimeError, match="statistics are not resolved"):
        train_mod.train_project(cfg, "p", auto_export=False, auto_test=False)


def test_train_project_blocks_dirty_composed_dataset_before_bundle(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data.compose import dataset_dirty_path
    from sed.train import train as train_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("p")
    layout.create()
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    _write_minimal_dataset(layout, wav, ["_negative", "cheers"])
    dataset_dirty_path(layout).write_text(
        json.dumps({"schema_version": 1, "count": 1, "paths": [str(wav)]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_mod.registry,
        "new_model_id",
        lambda *_args, **_kwargs: pytest.fail("bundle id allocated before dirty gate"),
    )

    with pytest.raises(RuntimeError, match="deleted source clip"):
        train_mod.train_project(
            _efficientat_cfg(["_negative", "cheers"]), "p",
            auto_export=False, auto_test=False,
        )


def test_train_project_rejects_pretrained_efficientat_legacy_frontend(tmp_path, monkeypatch):
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path)
    layout = ProjectLayout("p")
    layout.create()
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 32000)
    _write_minimal_dataset(layout, wav, ["_negative", "cheers"])

    cfg = _efficientat_cfg(["_negative", "cheers"])
    cfg.features.frontend = "legacy_torchaudio_db"

    # The legacy frontend was removed entirely (not merely off-contract), so it
    # is still rejected — now at feature extraction rather than as a hard contract
    # block. Off-contract feature *shapes* only warn (warn-not-block); an
    # unsupported frontend cannot run at all.
    try:
        train_project(cfg, "p", auto_export=False, auto_test=False)
    except (RuntimeError, ValueError) as exc:
        assert "efficientat" in str(exc).lower()
    else:
        raise AssertionError("expected an unsupported-frontend error")


def test_manifest_environment_and_csv_hash_helpers(tmp_path):
    from sed.train.train import _environment_versions, _optional_file_sha256

    env = _environment_versions()
    assert env["python"]
    assert "torch" in env and "openvino" in env and "nncf" in env

    csv = tmp_path / "train.csv"
    csv.write_text("path,class_name\n")
    assert len(_optional_file_sha256(csv)) == 64
    assert _optional_file_sha256(tmp_path / "missing.csv") == ""


def test_write_val_csv_snapshot_freezes_the_live_split_for_the_bundle(tmp_path):
    """Val isn't composed ahead of time — this is the only record of exactly
    which clips a given run held out, read back by INT8 export calibration
    and hashed into the manifest as val_csv_sha256."""
    rows = [
        {"path": "/a/1.wav", "class_name": "cheers", "session_id": "s1",
         "sample_rate": "32000", "duration_s": "1.0000", "is_target": "1",
         "origin": "project", "origin_class": "cheers", "source_stem": "",
         "original_source": "/source/cheers.wav", "augmentation_family": "pitch",
         "source_split_group": "recording:cheers-1",
         "frozen_source": "", "__csv_line": "2"},
        {"path": "/a/2.wav", "class_name": "_negative", "session_id": "s2",
         "sample_rate": "32000", "duration_s": "1.0000", "is_target": "0",
         "origin": "project", "origin_class": "_negative", "source_stem": "",
         "frozen_source": "", "__csv_line": "3"},
    ]
    dst = tmp_path / "dataset" / "val.csv"
    _write_val_csv_snapshot(rows, dst)

    with open(dst, encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert [r["path"] for r in written] == ["/a/1.wav", "/a/2.wav"]
    assert [r["class_name"] for r in written] == ["cheers", "_negative"]
    assert written[0]["original_source"] == "/source/cheers.wav"
    assert written[0]["augmentation_family"] == "pitch"
    assert written[0]["source_split_group"] == "recording:cheers-1"
    # The in-memory __csv_line bookkeeping field never leaks into the file.
    assert "__csv_line" not in written[0]


def test_model_stats_block_records_counts_and_geometry():
    from sed.train.train import _model_stats_block

    cfg = Config()  # defaults: mn10_as, mel-128, single channel
    stats = _model_stats_block(cfg, total_params=10, trainable_params=5)
    assert stats["total_params"] == 10
    assert stats["trainable_params"] == 5
    assert stats["input_channels"] == 1
    assert stats["n_bands"] == 128
    assert stats["n_frames"] > 0
    assert stats["feature_type"] == "mel"
    assert stats["source"] == "train"

    legacy = Config()
    legacy.model.backbone = "legacy-smallervggnet"
    legacy.model.sample_rate = 44100
    legacy.features.feature_type = "linear"
    legacy.features.stft_engine = "scipy"
    legacy.features.n_fft = 190
    legacy.features.win_length = 190
    legacy.features.hop_length = 167
    legacy.features.fmax = 22050
    legacy.features.window_seconds = 0.4
    legacy.features.colormap = "jet"
    legacy.features.image_size = 96
    stats = _model_stats_block(legacy, total_params=1, trainable_params=1)
    # Colormap render: the model input is the fixed 3×96×96 jet image.
    assert stats["input_channels"] == 3
    assert stats["n_bands"] == 96
    assert stats["n_frames"] == 96
    assert stats["feature_type"] == "linear"
    assert stats["colormap"] == "jet"
