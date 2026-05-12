"""Tests for the post-training BatchNorm recalibration pass."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

from sed.config import Config, ProjectLayout  # noqa: E402
from sed.train.bn_recalibrate import recalibrate_batchnorm  # noqa: E402
from sed.train.lit_module import SedLitModule  # noqa: E402


def _write_train_csv(path: Path, rows: list[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "path", "class_name", "session_id", "sample_rate", "duration_s",
            "is_target", "origin",
        ])
        for idx, (wav, class_name) in enumerate(rows):
            w.writerow([str(wav), class_name, f"s{idx}", 32000, "1.0000", 1, "project"])


def _seed_run(tmp_path, monkeypatch, *, backbone: str = "kws-bcresnet") -> tuple[Config, Path]:
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    layout = ProjectLayout("p")
    layout.create()

    classes = ["_negative", "glass"]
    cfg = Config()
    cfg.model.classes = classes
    cfg.dataset.target_classes = classes
    cfg.model.backbone = backbone
    cfg.model.pretrained = False
    if backbone == "kws-bcresnet":
        cfg.model.params = {"width_mult": 1.0, "time_stride": 1, "depth": 2}
    cfg.training.loss = "bce_logits"
    cfg.training.batch_size = 2
    cfg.training.bn_recalibrate_batches = 2

    rng = np.random.default_rng(0)
    wavs = []
    for i, cls in enumerate(["_negative", "glass", "_negative", "glass"]):
        wav = tmp_path / f"clip{i}.wav"
        sf.write(str(wav), (0.2 * rng.standard_normal(32000)).astype("float32"), 32000)
        wavs.append((wav, cls))

    model_dir = layout.model_dir("run-1")
    dataset_dir = model_dir / "dataset"
    _write_train_csv(dataset_dir / "train.csv", wavs)
    (dataset_dir / "classes.json").write_text(
        json.dumps({"classes": classes, "class_index": {c: i for i, c in enumerate(classes)}}),
        encoding="utf-8",
    )

    module = SedLitModule(cfg, classes)
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": {},
            "pytorch-lightning_version": "2.0.0",
        },
        model_dir / "checkpoint_best_epoch-001.ckpt",
    )
    (model_dir / "training_config.json").write_text(
        json.dumps(cfg.model_dump(mode="json")), encoding="utf-8"
    )
    return cfg, model_dir


def _bn_stats(state_dict: dict) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if k.endswith(("running_mean", "running_var"))}


def _weights(state_dict: dict) -> dict[str, torch.Tensor]:
    return {
        k: v for k, v in state_dict.items()
        if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def test_recalibration_refreshes_stats_and_preserves_weights(tmp_path, monkeypatch):
    cfg, model_dir = _seed_run(tmp_path, monkeypatch)
    ckpt = model_dir / "checkpoint_best_epoch-001.ckpt"
    before = torch.load(str(ckpt), map_location="cpu", weights_only=False)["state_dict"]

    info = recalibrate_batchnorm(cfg, "p", checkpoint_path=ckpt, num_batches=2)

    assert info["bn_layers"] > 0
    assert info["batches"] == 2
    out = Path(info["checkpoint"])
    assert out.name == "checkpoint_best_epoch-001_bnrecal.ckpt"
    after = torch.load(str(out), map_location="cpu", weights_only=False)["state_dict"]

    # Learned weights are bit-identical; only BN running stats moved.
    for key, tensor in _weights(before).items():
        assert torch.equal(tensor, after[key]), f"weight drifted: {key}"
    stats_before = _bn_stats(before)
    stats_after = _bn_stats(after)
    assert stats_before.keys() == stats_after.keys()
    moved = sum(
        0 if torch.allclose(stats_before[k], stats_after[k]) else 1
        for k in stats_before
    )
    assert moved > 0, "no BN running statistic changed"


def test_recalibrated_checkpoint_round_trips_through_the_loader(tmp_path, monkeypatch):
    cfg, model_dir = _seed_run(tmp_path, monkeypatch)
    ckpt = model_dir / "checkpoint_best_epoch-001.ckpt"
    info = recalibrate_batchnorm(cfg, "p", checkpoint_path=ckpt, num_batches=1)

    lit = SedLitModule.load_from_checkpoint(
        str(info["checkpoint"]), cfg=cfg, classes=list(cfg.model.classes), map_location="cpu"
    )
    lit.eval()
    with torch.no_grad():
        logits = lit(torch.randn(2, 1, 128, 100))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_recalibration_is_noop_for_models_without_batchnorm(tmp_path, monkeypatch):
    pytest.importorskip("timm")
    cfg, model_dir = _seed_run(tmp_path, monkeypatch, backbone="convnext_nano")
    cfg.model.params = {}
    module = SedLitModule(cfg, list(cfg.model.classes))
    has_bn = any(
        isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in module.model.modules()
    )
    if has_bn:
        pytest.skip("ConvNeXt Nano unexpectedly contains BatchNorm; pick another BN-free backbone")
    ckpt = model_dir / "checkpoint_best_epoch-001.ckpt"
    torch.save(
        {"state_dict": module.state_dict(), "hyper_parameters": {},
         "pytorch-lightning_version": "2.0.0"},
        ckpt,
    )
    (model_dir / "training_config.json").write_text(
        json.dumps(cfg.model_dump(mode="json")), encoding="utf-8"
    )

    info = recalibrate_batchnorm(cfg, "p", checkpoint_path=ckpt, num_batches=1)

    assert info == {"checkpoint": None, "bn_layers": 0, "batches": 0, "samples": 0}
    assert not list(model_dir.glob("*_bnrecal.ckpt"))
