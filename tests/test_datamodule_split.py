"""SedDataModule's live train/val split: train.csv is the whole pool (no
val.csv is composed ahead of time); setup() carves val out of it at train
start, seeded by cfg.training.seed. See sed.data.splits.split_pool_rows."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler

from sed.config import Config, ProjectLayout
from sed.train.datamodule import (
    SedDataModule,
    _class_balance_weights,
    _sampling_group_key,
    _uses_class_balancing,
)

SR = 32000


def _write_pool_csv(layout: ProjectLayout, tmp_path: Path, classes: list[str],
                     n_sessions: int = 6, clips_per_session: int = 2) -> None:
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for class_name in classes:
        for s in range(n_sessions):
            for c in range(clips_per_session):
                wav = tmp_path / f"{class_name}_{s}_{c}.wav"
                sf.write(str(wav), np.zeros(SR, dtype="float32"), SR)
                rows.append([str(wav), class_name, f"{class_name}_sess{s}",
                             SR, "1.0000", 1, "project", class_name, "", ""])
    with open(layout.dataset_dir / "train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_name", "session_id", "sample_rate", "duration_s",
                    "is_target", "origin", "origin_class", "source_stem", "frozen_source"])
        w.writerows(rows)
    with open(layout.dataset_dir / "test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_name", "session_id", "sample_rate", "duration_s",
                     "is_target", "origin", "origin_class", "source_stem", "frozen_source"])
    class_index = {c: i for i, c in enumerate(classes)}
    import json
    (layout.dataset_dir / "classes.json").write_text(
        json.dumps({"classes": classes, "class_index": class_index}), encoding="utf-8",
    )


def _cfg(classes: list[str], val_fraction: float = 0.2, seed: int = 42) -> Config:
    cfg = Config.model_validate({"dataset": {"target_classes": classes, "val_fraction": val_fraction}})
    cfg.training.seed = seed
    return cfg


def test_setup_splits_train_val_live_without_a_val_csv(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    layout = ProjectLayout("p")
    layout.create()
    classes = ["_negative", "cheers"]
    _write_pool_csv(layout, tmp_path, classes)
    assert not (layout.dataset_dir / "val.csv").exists()

    cfg = _cfg(classes)
    dm = SedDataModule(cfg=cfg, project_id="p")
    dm.setup()

    assert len(dm.train_ds) > 0
    assert len(dm.val_ds) > 0
    assert len(dm.train_ds) + len(dm.val_ds) == 24  # 2 classes * 6 sessions * 2 clips

    train_paths = {row["path"] for row in dm.train_ds.rows}
    val_paths = {row["path"] for row in dm.val_ds.rows}
    assert not (train_paths & val_paths), "train and val must be disjoint"

    # Session-grouped: no session straddles train and val.
    train_sessions = {row["session_id"] for row in dm.train_ds.rows}
    val_sessions = {row["session_id"] for row in dm.val_ds.rows}
    assert not (train_sessions & val_sessions)


def test_setup_is_idempotent_within_one_run(tmp_path, monkeypatch):
    """Lightning may call setup() more than once per run; the split must not
    be re-randomized mid-run."""
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    layout = ProjectLayout("p")
    layout.create()
    classes = ["_negative", "cheers"]
    _write_pool_csv(layout, tmp_path, classes)

    cfg = _cfg(classes)
    dm = SedDataModule(cfg=cfg, project_id="p")
    dm.setup()
    first_val_paths = [row["path"] for row in dm.val_ds.rows]
    dm.setup()
    second_val_paths = [row["path"] for row in dm.val_ds.rows]

    assert first_val_paths == second_val_paths


def test_setup_same_seed_reproduces_same_split_across_instances(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    layout = ProjectLayout("p")
    layout.create()
    classes = ["_negative", "cheers"]
    _write_pool_csv(layout, tmp_path, classes)

    dm1 = SedDataModule(cfg=_cfg(classes, seed=7), project_id="p")
    dm1.setup()
    dm2 = SedDataModule(cfg=_cfg(classes, seed=7), project_id="p")
    dm2.setup()

    assert {r["path"] for r in dm1.val_ds.rows} == {r["path"] for r in dm2.val_ds.rows}


def test_val_fraction_changes_val_size(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    layout = ProjectLayout("p")
    layout.create()
    classes = ["_negative", "cheers"]
    _write_pool_csv(layout, tmp_path, classes, n_sessions=10, clips_per_session=2)

    dm_small = SedDataModule(cfg=_cfg(classes, val_fraction=0.1), project_id="p")
    dm_small.setup()
    dm_big = SedDataModule(cfg=_cfg(classes, val_fraction=0.4), project_id="p")
    dm_big.setup()

    assert len(dm_big.val_ds) > len(dm_small.val_ds)


class _RowsDataset:
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return index


def _sampler_datamodule(
    rows: list[dict[str, str]],
    *,
    seed: int = 42,
    mode: str = "balanced",
    backbone: str = "clink-cnn-v2",
    objective: str = "supervised",
) -> SedDataModule:
    classes = list(dict.fromkeys(row["class_name"] for row in rows))
    cfg = _cfg(classes, seed=seed)
    cfg.model.backbone = backbone
    cfg.training.class_balance = mode
    cfg.training.pretraining_objective = objective
    cfg.training.batch_size = 2
    cfg.training.num_workers = 0
    dm = SedDataModule(cfg=cfg, project_id="unused")
    dm.train_ds = _RowsDataset(rows)
    dm.val_ds = _RowsDataset(rows[:1])
    dm.test_ds = _RowsDataset(rows[:1])
    return dm


def test_balanced_weights_follow_hierarchical_probability_contract():
    rows = (
        [{"class_name": "_negative", "split_group": "neg-small"}]
        + [{"class_name": "_negative", "split_group": "neg-large"}] * 4
        + [{"class_name": "glass_a", "split_group": "a-small"}]
        + [{"class_name": "glass_a", "split_group": "a-large"}] * 9
        + [{"class_name": "glass_b", "split_group": "b"}] * 4
    )

    weights = _class_balance_weights(rows, "balanced")

    assert float(weights.sum()) == pytest.approx(1.0)
    assert float(weights[:5].sum()) == pytest.approx(0.5)
    assert float(weights[5:15].sum()) == pytest.approx(0.25)
    assert float(weights[15:].sum()) == pytest.approx(0.25)
    # Group mass follows sqrt(size): 1:2 for sizes 1:4 and 1:3 for 1:9.
    assert float(weights[1:5].sum() / weights[0]) == pytest.approx(2.0)
    assert float(weights[6:15].sum() / weights[5]) == pytest.approx(3.0)
    # Selection within each group is uniform.
    assert len(set(weights[1:5].tolist())) == 1
    assert len(set(weights[6:15].tolist())) == 1


def test_balanced_without_negative_balances_present_classes():
    rows = (
        [{"class_name": "glass_a", "split_group": "a"}] * 9
        + [{"class_name": "glass_b", "split_group": "b"}]
    )

    weights = _class_balance_weights(rows, "balanced")

    assert float(weights[:9].sum()) == pytest.approx(0.5)
    assert float(weights[9:].sum()) == pytest.approx(0.5)


def test_balanced_with_only_negative_assigns_all_mass_to_negative():
    rows = [{"class_name": "_negative", "split_group": "room"}] * 4

    weights = _class_balance_weights(rows, "balanced")

    assert float(weights.sum()) == pytest.approx(1.0)
    assert weights.tolist() == pytest.approx([0.25] * 4)


def test_sqrt_weights_temper_root_and_target_class_imbalance():
    rows = (
        [{"class_name": "_negative", "split_group": "neg"}] * 16
        + [{"class_name": "glass_a", "split_group": "a"}] * 9
        + [{"class_name": "glass_b", "split_group": "b"}]
    )

    weights = _class_balance_weights(rows, "sqrt")
    negative_mass = float(weights[:16].sum())
    glass_a_mass = float(weights[16:25].sum())
    glass_b_mass = float(weights[25:].sum())

    # Root mass is sqrt(negative rows):sqrt(all target rows), not the raw 16:10.
    assert negative_mass / (glass_a_mass + glass_b_mass) == pytest.approx(
        np.sqrt(16) / np.sqrt(10),
    )
    # Target classes are also tempered by their row counts: sqrt(9):sqrt(1).
    assert glass_a_mass / glass_b_mass == pytest.approx(3.0)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_sampling_group_key_uses_safe_provenance_fallbacks():
    original_key = _sampling_group_key(
        {
            "source_split_group": "original",
            "split_group": "derived",
            "session_id": "session",
            "path": "/a.wav",
        },
        0,
    )
    inherited_key = _sampling_group_key(
        {"split_group": "original", "session_id": "session", "path": "/a.wav"}, 0,
    )
    assert original_key == inherited_key == "group:original"
    assert _sampling_group_key({"session_id": "session", "path": "/a.wav"}, 0) == (
        "session:session"
    )
    assert _sampling_group_key({"path": "/one/a.wav"}, 0) == "path:/one/a.wav"
    assert _sampling_group_key({}, 7) == "row:7"


def test_balanced_loader_is_seeded_replacement_sampler_across_epochs():
    rows = [
        {"class_name": "_negative", "split_group": f"neg-{index // 3}"}
        for index in range(12)
    ] + [
        {"class_name": "glass", "split_group": f"glass-{index // 2}"}
        for index in range(8)
    ]

    dm1 = _sampler_datamodule(rows, seed=7)
    dm2 = _sampler_datamodule(rows, seed=7)
    dm3 = _sampler_datamodule(rows, seed=8)
    loader1 = dm1.train_dataloader()
    loader2 = dm2.train_dataloader()
    loader3 = dm3.train_dataloader()

    assert isinstance(loader1.sampler, WeightedRandomSampler)
    assert loader1.sampler.replacement is True
    assert loader1.sampler.num_samples == len(rows)
    assert loader1.drop_last is False
    sampled1 = list(loader1.sampler)
    assert sampled1 == list(loader2.sampler)
    assert sampled1 != list(loader3.sampler)
    sampled1_epoch2 = list(loader1.sampler)
    assert sampled1_epoch2 == list(loader2.sampler)
    assert sampled1_epoch2 != sampled1
    assert isinstance(dm1.val_dataloader().sampler, SequentialSampler)
    assert isinstance(dm1.test_dataloader().sampler, SequentialSampler)


def test_class_balance_off_remains_the_default():
    rows = [{"class_name": "glass", "split_group": "g"}]
    cfg = _cfg(["glass"])
    cfg.training.batch_size = 1
    cfg.training.num_workers = 0
    dm = SedDataModule(cfg=cfg, project_id="unused")
    dm.train_ds = _RowsDataset(rows)

    loader = dm.train_dataloader()

    assert cfg.training.class_balance == "off"
    assert isinstance(loader.sampler, RandomSampler)
    assert not isinstance(loader.sampler, WeightedRandomSampler)


@pytest.mark.parametrize(
    "backbone",
    ["clink-cnn", "clink-cnn-v2", "mn10_as", "timm-resnet18"],
)
@pytest.mark.parametrize("mode", ["sqrt", "balanced"])
def test_class_balancing_is_model_agnostic_for_supervised_training(backbone, mode):
    rows = [
        {"class_name": "_negative", "split_group": "negative"},
        {"class_name": "glass", "split_group": "glass"},
    ]
    dm = _sampler_datamodule(rows, backbone=backbone, mode=mode)

    assert _uses_class_balancing(dm.cfg) is True
    assert isinstance(dm.train_dataloader().sampler, WeightedRandomSampler)


@pytest.mark.parametrize("backbone", ["clink-cnn", "clink-cnn-v2", "mn10_as"])
def test_byol_ignores_class_balancing(backbone):
    rows = [
        {"class_name": "_negative", "split_group": "negative"},
        {"class_name": "glass", "split_group": "glass"},
    ]
    dm = _sampler_datamodule(
        rows,
        backbone=backbone,
        mode="balanced",
        objective="byol",
    )

    assert _uses_class_balancing(dm.cfg) is False
    loader = dm.train_dataloader()
    assert isinstance(loader.sampler, RandomSampler)
    assert not isinstance(loader.sampler, WeightedRandomSampler)
