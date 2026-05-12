from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from sed.config import Config, ProjectLayout
from sed.train.datamodule import SedDataset
from sed.train.train import LossAccuracyCheckpoint, _semantic_checkpoint_path, train_project


def _write_dataset_csv(path: Path, wav: Path, class_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "path", "class_name", "session_id", "sample_rate", "duration_s",
            "is_target", "origin",
        ])
        w.writerow([str(wav), class_name, "s1", 32000, "1.0000", 1, "project"])


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
    (layout.dataset_dir / "classes.json").write_text(
        '{"classes": %s, "class_index": {%s}}\n' % (
            str(classes).replace("'", '"'),
            ", ".join(f'"{c}": {i}' for i, c in enumerate(classes)),
        ),
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


def test_training_dataset_resamples_raw_48k_clip_to_model_rate(tmp_path):
    wav = tmp_path / "raw_48k.wav"
    sf.write(str(wav), np.linspace(-0.5, 0.5, 48000, dtype="float32"), 48000)
    csv_path = tmp_path / "train.csv"
    _write_dataset_csv(csv_path, wav, "cheers")
    cfg = Config()
    cfg.model.sample_rate = 32000
    cfg.features.window_seconds = 1.0

    ds = SedDataset(csv_path, ["_negative", "cheers"], cfg)
    wave = ds._load(str(wav))

    assert wave.shape == (32000,)
    assert wave.dtype == np.float32


def test_loss_accuracy_checkpoint_tie_breaks_on_accuracy(tmp_path):
    cb = LossAccuracyCheckpoint(tmp_path, min_delta=0.001)

    assert cb._is_better(0.5, 0.8) is True
    cb.best_loss = 0.5
    cb.best_accuracy = 0.8
    assert cb._is_better(0.4995, 0.7) is False
    assert cb._is_better(0.4995, 0.9) is True
    assert cb._is_better(0.498, 0.1) is True


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
        assert False, "expected class drift error"


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

    try:
        train_project(cfg, "p", auto_export=False, auto_test=False)
    except RuntimeError as exc:
        assert "EfficientAT AudioSet-pretrained training requires" in str(exc)
        assert "features.frontend" in str(exc)
    else:
        assert False, "expected EfficientAT feature contract error"
