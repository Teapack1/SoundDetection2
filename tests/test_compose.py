"""Tests for dataset composition: train/val splitting and external
auto-discovery (no [dataset.external] block needed). Frozen-test stability
is covered in test_frozen_test.py."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from sed.data.splits import train_val_split


@dataclass
class FakeClip:
    session_id: str
    class_name: str
    path: str = "x"


def _patch_library_paths(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")


def _library_clip(
    tmp_path,
    label: str,
    *,
    project_id: str = "_library",
    dataset_role: str = "train",
    density: str = "normal",
    review_status: str = "accepted",
    tags: list[str] | None = None,
    recorded_channels: list[int] | None = None,
    triggered_channels: list[int] | None = None,
    clip_kind: str | None = None,
    channel_index: int | None = None,
):
    from sed.data.library import Library

    src = tmp_path / f"{label}_{project_id}.wav"
    sf.write(str(src), np.zeros(32000, dtype=np.float32), 32000)
    lib = Library()
    clip = lib.import_clip(
        src,
        label=label,
        project_id=project_id,
        dataset_role=dataset_role,
        density=density,
        review_status=review_status,
    )
    if (
        tags
        or recorded_channels is not None
        or triggered_channels is not None
        or clip_kind is not None
        or channel_index is not None
    ):
        wav = clip.absolute_path(lib.root)
        meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        if tags:
            meta["tags"] = tags
        if recorded_channels is not None:
            meta["recorded_channels"] = recorded_channels
        if triggered_channels is not None:
            meta["triggered_channels"] = triggered_channels
        if clip_kind is not None:
            meta["clip_kind"] = clip_kind
        if channel_index is not None:
            meta["channel_index"] = channel_index
        wav.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lib.rebuild_manifest()
        clip = next(c for c in lib.all_clips() if c.uuid == clip.uuid)
    return clip


def test_train_val_split_returns_train_and_val_only():
    clips = []
    for s in range(10):
        for _ in range(20):
            clips.append(FakeClip(session_id=f"sess_{s}", class_name="x"))
    train, val, modes = train_val_split(clips, seed=0)
    assert "random" in modes["x"]
    assert len(train) > 0
    assert len(val) > 0
    assert len(train) + len(val) == 200


def test_train_val_split_mixed_classes_keep_val_coverage():
    clips = []
    for i in range(200):
        clips.append(FakeClip(session_id=f"lib-{i:04d}", class_name="lib_class"))
    for _ in range(100):
        clips.append(FakeClip(session_id="onsite_sess", class_name="onsite_class"))

    _train, val, modes = train_val_split(clips, seed=0)
    assert "random" in modes["lib_class"]
    assert "random" in modes["onsite_class"]
    val_classes = {c.class_name for c in val}
    assert "lib_class" in val_classes
    assert "onsite_class" in val_classes


def test_train_val_split_handles_tiny_class_gracefully():
    clips = [FakeClip(session_id="s1", class_name="t"),
             FakeClip(session_id="s2", class_name="t")]
    train, val, _ = train_val_split(clips, seed=0)
    assert len(train) + len(val) == 2


def test_ambience_category_is_rejected_in_project_config(tmp_path, monkeypatch):
    import pytest
    from sed import config as cfg_mod
    from sed.config import ProjectLayout, load_config

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(
        """
[dataset]
target_classes = ["_negative", "cling"]

[dataset.project]
"_negative" = [{ category = "ambience", labels = ["room"] }]
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="category='ambience' was removed"):
        load_config("p")


def test_project_positive_labels_map_to_target_class(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset

    layout = ProjectLayout("p")
    layout.create()
    mapped = _library_clip(tmp_path, "source_label", project_id="p")
    default = _library_clip(tmp_path, "cheers_loud", project_id="p")

    cfg = Config()
    cfg.model.classes = ["_negative", "cheers_loud"]
    cfg.dataset.project = {
        "cheers_loud": [{"category": "positives", "labels": ["source_label"]}]
    }

    summary = compose_dataset(cfg, "p")

    assert summary["by_origin"]["cheers_loud"]["sounds"] == 1
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")
    assert mapped.relpath in rows
    assert default.relpath not in rows


def test_dataset_sources_map_labels_to_classes(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset

    layout = ProjectLayout("p")
    layout.create()
    loud = _library_clip(tmp_path, "glass_loud", project_id="p")
    room = _library_clip(tmp_path, "room_tone", project_id="p")
    ignored = _library_clip(tmp_path, "cheers_loud", project_id="p")

    cfg = Config.model_validate({
        "dataset": {
            "target_classes": ["_negative", "cheers_loud"],
            "sources": {
                "cheers_loud": [{"labels": ["glass_loud"]}],
                "_negative": [{"labels": ["room_tone"]}],
            },
        },
    })

    summary = compose_dataset(cfg, "p")

    assert summary["by_origin"]["cheers_loud"]["sounds"] == 1
    assert summary["by_origin"]["_negative"]["sounds"] == 1
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")
    assert loud.relpath in rows
    assert room.relpath in rows
    assert ignored.relpath not in rows


def test_dataset_sources_project_ids_filter_library_clips(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset

    layout = ProjectLayout("p")
    layout.create()
    keep = _library_clip(tmp_path, "cling", project_id="p")
    drop = _library_clip(tmp_path, "cling", project_id="_library")

    cfg = Config.model_validate({
        "dataset": {
            "target_classes": ["cling"],
            "sources": {
                "cling": [{"labels": ["cling"], "project_ids": ["p"]}],
            },
        },
    })

    summary = compose_dataset(cfg, "p")

    assert summary["by_origin"]["cling"]["sounds"] == 1
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")
    assert keep.relpath in rows
    assert drop.relpath not in rows


def test_ui_rule_filters_by_triggered_microphone_set(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    keep = _library_clip(
        tmp_path,
        "glass",
        project_id="p",
        recorded_channels=[2],
        triggered_channels=[2],
    )
    drop = _library_clip(
        tmp_path,
        "glass",
        project_id="p",
        recorded_channels=[0],
        triggered_channels=[0],
    )

    cfg = Config.model_validate({"dataset": {"target_classes": ["glass"]}})
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "glass": [{
                "labels": ["glass"],
                "project_ids": ["p"],
                "triggered_channels": [2],
                "split": "train",
            }]
        },
    })

    summary = compose_dataset(cfg, "p")

    assert summary["by_class"]["glass"] == 1
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")
    assert keep.relpath in rows
    assert drop.relpath not in rows


def test_ui_rule_filters_legacy_split_rows_by_channel_index(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    keep = _library_clip(
        tmp_path,
        "glass",
        project_id="p",
        clip_kind="channel_split",
        channel_index=1,
    )
    drop = _library_clip(
        tmp_path,
        "glass",
        project_id="p",
        clip_kind="channel_split",
        channel_index=0,
    )

    cfg = Config.model_validate({"dataset": {"target_classes": ["glass"]}})
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "glass": [{
                "labels": ["glass"],
                "project_ids": ["p"],
                "clip_kind": "channel_split",
                "channel_index": 1,
                "split": "train",
            }]
        },
    })

    summary = compose_dataset(cfg, "p")

    assert summary["by_class"]["glass"] == 1
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")
    assert keep.relpath in rows
    assert drop.relpath not in rows


def test_ui_managed_selection_ignores_config_sources(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    ui_clip = _library_clip(tmp_path, "ui_label", project_id="p")
    cfg_clip = _library_clip(tmp_path, "config_label", project_id="p")

    cfg = Config.model_validate({
        "dataset": {
            "target_classes": ["cling"],
            "sources": {"cling": [{"labels": ["config_label"]}]},
        },
    })
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {"cling": [{"labels": ["ui_label"], "split": "train"}]},
    })

    summary = compose_dataset(cfg, "p")

    assert summary["by_class"]["cling"] == 1
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")
    assert ui_clip.relpath in rows
    assert cfg_clip.relpath not in rows


def test_ui_managed_empty_assignment_disables_default_label(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    _library_clip(tmp_path, "cling", project_id="p")
    cfg = Config.model_validate({"dataset": {"target_classes": ["cling"]}})
    write_dataset_selection(layout, {"ui_managed": True, "assignments": {}})

    summary = compose_dataset(cfg, "p")

    assert summary["by_class"]["cling"] == 0


def test_ui_rule_filters_metadata_and_tags(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    keep = _library_clip(
        tmp_path,
        "room",
        project_id="p",
        density="busy",
        tags=["wine-glass", "near-bar"],
    )
    _library_clip(tmp_path, "room", project_id="p", density="busy", tags=["other"])
    _library_clip(tmp_path, "room", project_id="_library", density="busy", tags=["wine-glass"])
    _library_clip(tmp_path, "room", project_id="p", density="quiet", tags=["wine-glass"])

    cfg = Config.model_validate({"dataset": {"target_classes": ["_negative"]}})
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "_negative": [{
                "labels": ["room"],
                "project_ids": ["p"],
                "density": "busy",
                "dataset_role": "train",
                "review_status": "accepted",
                "tags_all": ["wine-glass"],
                "source_bucket": "raw",
                "split": "train",
            }]
        },
    })

    summary = compose_dataset(cfg, "p")

    assert summary["by_class"]["_negative"] == 1
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")
    assert keep.relpath in rows


def test_ui_rule_without_status_still_uses_only_accepted_clips(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    keep = _library_clip(tmp_path, "glass", project_id="p", review_status="accepted")
    rejected = _library_clip(tmp_path, "glass", project_id="p", review_status="rejected")

    cfg = Config.model_validate({"dataset": {"target_classes": ["glass"]}})
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "glass": [{
                "labels": ["glass"],
                "project_ids": ["p"],
                "split": "train",
            }]
        },
    })

    summary = compose_dataset(cfg, "p")

    assert summary["by_class"]["glass"] == 1
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")
    assert keep.relpath in rows
    assert rejected.relpath not in rows


def test_ui_rules_include_accepted_derived_only_when_explicitly_assigned(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, LIBRARY_DIR, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    derived = LIBRARY_DIR / "derived" / "p" / "trainval" / "cling" / "aug.wav"
    derived.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(derived), np.zeros(32000, dtype=np.float32), 32000)
    derived.with_suffix(".json").write_text(
        json.dumps({
            "schema_version": 2,
            "category": "derived",
            "label": "cling",
            "class_name": "cling",
            "project_id": "p",
            "dataset_role": "train",
            "review_status": "accepted",
            "sample_rate": 32000,
            "duration_s": 1.0,
            "uuid": "derived-cling",
        }),
        encoding="utf-8",
    )
    Library().rebuild_manifest()

    cfg = Config.model_validate({"dataset": {"target_classes": ["cling"]}})
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {"cling": [{"labels": ["some_raw_source"], "split": "train"}]},
    })

    summary = compose_dataset(cfg, "p")

    assert summary["by_origin"]["cling"]["derived"] == 0

    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "cling": [{
                "labels": ["cling"],
                "project_ids": ["p"],
                "source_bucket": "derived",
                "split": "train",
            }]
        },
    })
    summary = compose_dataset(cfg, "p")
    assert summary["by_origin"]["cling"]["derived"] == 1


def test_compose_skips_soft_excluded_train_candidates(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_excluded_clip_paths

    layout = ProjectLayout("p")
    layout.create()
    keep = _library_clip(tmp_path, "cling", project_id="p")
    drop = _library_clip(tmp_path, "cling", project_id="p")
    from sed.config import LIBRARY_DIR
    drop_path = LIBRARY_DIR / drop.relpath
    write_excluded_clip_paths(layout, [drop_path])

    cfg = Config()
    cfg.model.classes = ["cling"]

    summary = compose_dataset(cfg, "p")

    assert summary["by_origin"]["cling"]["sounds"] == 1
    assert summary["excluded"]["total"] == 1
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")
    assert keep.relpath in rows
    assert drop.relpath not in rows


def test_project_and_external_negatives_auto_fold_into_negative(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset

    layout = ProjectLayout("p")
    layout.create()
    _library_clip(tmp_path, "cling", project_id="p")
    _library_clip(tmp_path, "room", project_id="p")
    _library_clip(tmp_path, "kitchen", project_id="_library")

    cfg = Config()
    cfg.model.classes = ["_negative", "cling"]
    cfg.dataset.project = {"_negative": [{"label": "room"}, {"label": "kitchen"}]}

    summary = compose_dataset(cfg, "p")

    assert summary["by_origin"]["_negative"]["sounds"] == 2
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")
    assert ",_negative," in rows


def test_compose_allows_target_only_model_without_negative(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset

    layout = ProjectLayout("p")
    layout.create()
    for cls in ("cling_a", "cling_b", "cling_c"):
        for _ in range(3):
            _library_clip(tmp_path, cls, project_id="p")

    cfg = Config.model_validate({
        "dataset": {"target_classes": ["cling_a", "cling_b", "cling_c"]},
    })

    summary = compose_dataset(cfg, "p")

    assert "_negative" not in summary["by_class"]
    assert summary["by_class"] == {"cling_a": 3, "cling_b": 3, "cling_c": 3}
    assert '"_negative"' not in (layout.dataset_dir / "classes.json").read_text(
        encoding="utf-8"
    )


def test_explicit_negative_mappings_filter_negatives(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset

    layout = ProjectLayout("p")
    layout.create()
    _library_clip(tmp_path, "cling", project_id="p")
    room = _library_clip(tmp_path, "room", project_id="p")
    street = _library_clip(tmp_path, "street", project_id="p")
    kitchen = _library_clip(tmp_path, "kitchen", project_id="_library")
    garden = _library_clip(tmp_path, "garden", project_id="_library")

    cfg = Config()
    cfg.model.classes = ["_negative", "cling"]
    # Explicit filters: only room and kitchen become _negative.
    cfg.dataset.project = {
        "_negative": [{"labels": ["room"]}]
    }
    cfg.dataset.external = {
        "_negative": [{"labels": ["kitchen"]}]
    }

    summary = compose_dataset(cfg, "p")

    assert summary["by_origin"]["_negative"]["sounds"] == 2
    
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")
    
    assert room.relpath in rows
    assert kitchen.relpath in rows
    assert street.relpath not in rows
    assert garden.relpath not in rows


def test_custom_positive_class_maps_from_negatives(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset
    from sed.data.inspect import _gather_clips

    layout = ProjectLayout("p")
    layout.create()
    room = _library_clip(tmp_path, "room", project_id="p")
    kitchen = _library_clip(tmp_path, "kitchen", project_id="_library")

    cfg = Config()
    cfg.model.classes = ["cling"]
    cfg.dataset.project = {
        "cling": [{"labels": ["room"]}]
    }
    cfg.dataset.external = {
        "cling": [{"labels": ["kitchen"]}]
    }

    summary = compose_dataset(cfg, "p")

    # Assert composer discovered them as 'cling'
    assert summary["by_class"]["cling"] == 2
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    rows += (layout.dataset_dir / "val.csv").read_text(encoding="utf-8")

    assert room.relpath in rows
    assert kitchen.relpath in rows

    # Assert inspect UI's _gather_clips also gathers them under 'cling'
    from pathlib import Path
    clips = _gather_clips(layout, cfg)
    clip_dict = {Path(p).name: cls for p, cls, _ in clips}
    assert len(clip_dict) == 2
    assert set(clip_dict.values()) == {"cling"}


def test_compose_warns_about_short_raw_clips_in_train(tmp_path, monkeypatch, caplog):
    import logging

    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    short_src = tmp_path / "legacy_clink.wav"
    sf.write(str(short_src), np.zeros(12800, dtype=np.float32), 32000)  # 0.4 s
    Library().import_clip(short_src, label="legacy_clink", project_id="p")
    _library_clip(tmp_path, "room_tone", project_id="p")  # full 1.0 s clip

    cfg = Config.model_validate({
        "dataset": {
            "target_classes": ["_negative", "cheers_loud"],
            "sources": {
                "cheers_loud": [{"labels": ["legacy_clink"]}],
                "_negative": [{"labels": ["room_tone"]}],
            },
        },
    })

    with caplog.at_level(logging.WARNING, logger="sed.data.compose"):
        compose_dataset(cfg, "p")

    warnings = [r.getMessage() for r in caplog.records if "shorter than" in str(r.msg)]
    assert len(warnings) == 1
    assert "cheers_loud" in warnings[0]
    assert "_negative" not in warnings[0]
