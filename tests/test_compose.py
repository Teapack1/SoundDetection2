"""Tests for dataset composition: train/val splitting and external
auto-discovery (no [dataset.external] block needed). Frozen-test stability
is covered in test_frozen_test.py."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
import soundfile as sf



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


def test_deleted_composed_path_sets_dirty_state_without_rewriting_csv(
    tmp_path, monkeypatch,
):
    _patch_library_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import (
        clear_dataset_dirty,
        dataset_compose_state,
        load_dataset_dirty,
        mark_composed_dataset_dirty_for_paths,
    )

    layout = ProjectLayout("p")
    layout.create()
    source = tmp_path / "data_library" / "sounds" / "glass" / "one.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    csv_text = f"path,class_name\n{source},glass\n"
    (layout.dataset_dir / "train.csv").write_text(csv_text, encoding="utf-8")

    result = mark_composed_dataset_dirty_for_paths(layout, [source])

    assert result["dirty"] is True
    assert (layout.dataset_dir / "train.csv").read_text(encoding="utf-8") == csv_text
    assert load_dataset_dirty(layout)["splits"][str(source.resolve())] == ["train"]
    state = dataset_compose_state(layout, {}, Config())
    assert state["dirty"] is True
    assert state["dirty_count"] == 1
    assert "Compose before training" in state["reason"]
    clear_dataset_dirty(layout)
    assert load_dataset_dirty(layout) == {}


def test_derived_session_rule_matches_generated_run_prefix(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config
    from sed.data.compose import _library_clips_for_class
    from sed.data.library import Library

    out = tmp_path / "data_library" / "derived" / "p" / "trainval" / "glass"
    out.mkdir(parents=True)
    wav = out / "aug_20260725_000350_s42_aug_glass_0001.wav"
    sf.write(str(wav), np.zeros(32000, dtype=np.float32), 32000)
    wav.with_suffix(".json").write_text(
        json.dumps({
            "category": "derived",
            "label": "glass",
            "class_name": "glass",
            "project_id": "p",
            "density": "empty",
            "review_status": "accepted",
            "tags": ["glass"],
            "augmentation": "overlay",
            "augmentation_run_prefix": "aug-20260725-000350-s42",
        }),
        encoding="utf-8",
    )
    Library().rebuild_manifest()

    cfg = Config.model_validate({"dataset": {"target_classes": ["glass"]}})
    selection = {
        "ui_managed": True,
        "assignments": {
            "glass": [{
                "source_bucket": "derived",
                "labels": ["glass"],
                "project_ids": ["p"],
                "density": "empty",
                "tags_all": ["glass"],
                "session": "aug-20260725-000350-s42",
                "split": "train",
            }],
        },
    }

    clips = _library_clips_for_class(
        cfg, "p", "glass", split="train", selection=selection,
    )

    assert len(clips) == 1
    assert clips[0].category == "derived"


def test_clip_to_source_preserves_origin_class_when_reassigned(tmp_path):
    from sed.data.compose import _clip_to_source
    from sed.data.library import LibraryClip

    clip = LibraryClip(
        uuid="clip-1",
        relpath="raw/original.wav",
        label="original",
        origin_class="original",
        session_id="session-1",
        sample_rate=32000,
        duration_s=1.0,
    )

    source = _clip_to_source(
        clip,
        "reassigned",
        origin="project",
        base=tmp_path,
    )

    assert source.class_name == "reassigned"
    assert source.origin_class == "original"
    assert source.split_bucket == "project:original"


def _library_clip(
    tmp_path,
    label: str,
    *,
    project_id: str = "_library",
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


def _mined_clip(
    tmp_path,
    label: str,
    *,
    project_id: str = "p",
    session_id: str = "inference-20260618T010203Z-abc123",
):
    from sed.config import LIBRARY_DIR
    from sed.data.library import Library

    out = LIBRARY_DIR / "mined" / label
    out.mkdir(parents=True, exist_ok=True)
    wav = out / f"{label}_{session_id}.wav"
    sf.write(str(wav), np.zeros(32000, dtype=np.float32), 32000)
    wav.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "category": "mined",
                "label": label,
                "class_name": label,
                "project_id": project_id,
                "dataset_role": "train",
                "density": "normal",
                "review_status": "accepted",
                "uuid": f"mined-{label}-{session_id}",
                "session_id": session_id,
                "recording_counter": 1,
                "origin": "runtime_mining",
                "created_at": "2026-06-18T01:02:03Z",
                "recorded_at": "2026-06-18T01:02:03Z",
                "predicted_class": label,
                "confidence": 0.91,
                "kind": "threshold",
                "recorded_channels": [0],
                "triggered_channels": [0],
                "tags": ["mined", "threshold", label],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lib = Library()
    lib.rebuild_manifest()
    return next(c for c in lib.all_clips() if c.uuid == f"mined-{label}-{session_id}")


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
    assert loud.relpath in rows
    assert room.relpath in rows
    assert ignored.relpath not in rows


def test_compose_auto_assigns_mined_buckets_to_negative(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import load_dataset_selection, compose_dataset

    layout = ProjectLayout("p")
    layout.create()
    mined = _mined_clip(tmp_path, "cheers_loud", project_id="p")

    cfg = Config()
    cfg.model.classes = ["_negative", "cheers_loud"]

    summary = compose_dataset(cfg, "p")

    assert summary["by_origin"]["_negative"]["mined"] == 1
    selection = load_dataset_selection(layout)
    rules = selection["assignments"]["_negative"]
    assert len(rules) == 1
    assert rules[0]["source_bucket"] == "mined"
    assert rules[0]["auto_source"] == "mined"
    assert rules[0]["labels"] == ["cheers_loud"]
    assert rules[0]["session"] == mined.session_id

    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    assert mined.relpath in rows
    assert "_negative" in rows


def test_removed_auto_mined_rule_stays_removed_on_compose(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import (
        compose_dataset,
        load_dataset_selection,
        mark_auto_mined_rules_dismissed,
        write_dataset_selection,
    )

    layout = ProjectLayout("p")
    layout.create()
    _mined_clip(tmp_path, "cheers_loud", project_id="p")

    cfg = Config()
    cfg.model.classes = ["_negative", "cheers_loud"]
    compose_dataset(cfg, "p")

    selection = load_dataset_selection(layout)
    auto_rule = selection["assignments"]["_negative"][0]
    assert mark_auto_mined_rules_dismissed(selection, [auto_rule]) is True
    selection["assignments"]["_negative"] = []
    write_dataset_selection(layout, selection)

    summary = compose_dataset(cfg, "p")
    refreshed = load_dataset_selection(layout)

    assert summary["by_origin"]["_negative"]["mined"] == 0
    assert refreshed["assignments"].get("_negative") in (None, [])
    assert auto_rule["auto_bucket_key"] in refreshed["auto_mined_dismissed_bucket_keys"]


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
            "original_source": "/recordings/glass-session-a.wav",
            "augmentation_family": "overlay",
            "source_split_group": "glass-session-a",
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
    import csv

    with open(layout.dataset_dir / "train.csv", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["original_source"] == "/recordings/glass-session-a.wav"
    assert row["augmentation_family"] == "overlay"
    assert row["source_split_group"] == "glass-session-a"
    assert row["split_group"] == "glass-session-a"


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
        # Pinned, not defaulted: this test is about the short-clip warning, so
        # the window must stay longer than the 0.4 s fixture clip regardless of
        # what the project default happens to be.
        "features": {"window_seconds": 1.0},
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


def _make_pretraining_corpus(classes: dict[str, int], *, nested: bool = False):
    """Write WAVs under data_library/pretraining/<label>/ and build the index."""
    from sed.config import LIBRARY_DIR
    from sed.data.pretraining import PretrainIndex

    for label, n in classes.items():
        for i in range(n):
            sub = "sessA" if nested else ""
            d = LIBRARY_DIR / "pretraining" / label / sub if sub else LIBRARY_DIR / "pretraining" / label
            d.mkdir(parents=True, exist_ok=True)
            sf.write(str(d / f"{label}_{i:03d}.wav"), np.zeros(32000, dtype=np.float32), 32000)
    return PretrainIndex().rebuild()


def _make_pretraining_session_corpus(label: str, n_sessions: int):
    """Write one pretraining clip per independent session."""
    from sed.config import LIBRARY_DIR
    from sed.data.pretraining import PretrainIndex

    for i in range(n_sessions):
        d = LIBRARY_DIR / "pretraining" / label / f"sess{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / f"{label}_{i:03d}.wav"), np.zeros(32000, dtype=np.float32), 32000)
    return PretrainIndex().rebuild()


def test_pretraining_index_rebuild_and_clip_synthesis(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)
    from sed.data.library import PRETRAIN_CATEGORY
    from sed.data.pretraining import PretrainIndex

    _make_pretraining_corpus({"klatsch": 3, "cheers": 2}, nested=True)
    index = PretrainIndex()

    assert index.labels_with_counts() == {"klatsch": 3, "cheers": 2}
    assert len(index.rows_for_label("klatsch")) == 3
    assert len(index.rows_for_label("klatsch", limit=2)) == 2

    clip = index.clip_for_row(index.rows_for_label("klatsch")[0])
    assert clip.label == "klatsch"
    assert clip.category == PRETRAIN_CATEGORY
    assert clip.relpath.startswith("pretraining/klatsch/")
    assert clip.session_id == "sessA"          # nested folder -> session
    assert clip.sample_rate == 32000           # probed lazily
    # The corpus must NOT be folded into the main library manifest.
    from sed.data.library import Library
    assert all(c.category != PRETRAIN_CATEGORY for c in Library().all_clips())


def test_pretraining_rule_includes_clips_in_compose(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    _make_pretraining_corpus({"klatsch": 6, "cheers": 4})

    cfg = Config.model_validate({"dataset": {"target_classes": ["cling"]}})
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "cling": [{
                "source_bucket": "pretraining",
                "labels": ["klatsch"],
                "split": "train",
            }],
        },
    })

    summary = compose_dataset(cfg, "p")

    assert summary["by_class"]["cling"] == 6
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    assert "pretraining/klatsch" in rows
    assert "pretraining/cheers" not in rows   # only the assigned folder is pulled


def test_pretraining_subbucket_splits_by_sample_rate(tmp_path, monkeypatch):
    """A folder with mixed sample rates splits into per-rate subbuckets, and a
    pretraining rule can pin one subbucket by sample_rate."""
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import LIBRARY_DIR, Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection
    from sed.data.pretraining import PretrainIndex

    folder = LIBRARY_DIR / "pretraining" / "clap"
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(str(folder / "a.wav"), np.zeros(32000, dtype=np.float32), 32000)
    sf.write(str(folder / "b.wav"), np.zeros(32000, dtype=np.float32), 32000)
    sf.write(str(folder / "c.wav"), np.zeros(16000, dtype=np.float32), 16000)
    PretrainIndex().rebuild()

    # The mixed-rate folder yields two subbuckets.
    stats = PretrainIndex().subbuckets_with_stats()
    keys = sorted(stats.keys())
    assert keys == ["clap @ 16000 Hz", "clap @ 32000 Hz"]
    assert stats["clap @ 32000 Hz"]["count"] == 2
    assert stats["clap @ 16000 Hz"]["count"] == 1
    assert stats["clap @ 32000 Hz"]["origin_class"] == "clap"

    layout = ProjectLayout("p")
    layout.create()
    cfg = Config.model_validate({"dataset": {"target_classes": ["clap"]}})
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "clap": [{
                "source_bucket": "pretraining",
                "labels": ["clap"],
                "sample_rate": 32000,   # pin the 32 kHz subbucket only
                "split": "train",
            }],
        },
    })

    summary = compose_dataset(cfg, "p")
    assert summary["by_class"]["clap"] == 2          # only the two 32 kHz clips
    rows = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8")
    assert "clap/a.wav" in rows and "clap/b.wav" in rows
    assert "clap/c.wav" not in rows                  # 16 kHz subbucket excluded
    # CSV carries the original class.
    assert "origin_class" in (layout.dataset_dir / "train.csv").read_text(encoding="utf-8").splitlines()[0]


def _pretraining_project_selection(class_name: str, label: str) -> dict:
    return {
        "ui_managed": True,
        "assignments": {
            class_name: [{"source_bucket": "pretraining", "labels": [label], "split": "train"}],
        },
    }


def test_compose_writes_whole_pool_no_val_split(tmp_path, monkeypatch):
    """compose_dataset no longer splits train/val — train.csv is the whole pool."""
    _patch_library_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    _make_pretraining_corpus({"cling": 20})
    write_dataset_selection(layout, _pretraining_project_selection("cling", "cling"))

    cfg = Config.model_validate({"dataset": {"target_classes": ["cling"]}})
    summary = compose_dataset(cfg, "p")

    assert "val" not in summary["splits"]
    assert summary["splits"]["train"] == 20
    assert "val_fraction" not in summary["split_config"]
    assert not (layout.dataset_dir / "val.csv").exists()


def test_compose_removes_stale_project_val_csv_and_writes_split_columns(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    _make_pretraining_corpus({"cling": 4})
    write_dataset_selection(layout, _pretraining_project_selection("cling", "cling"))
    stale_val = layout.dataset_dir / "val.csv"
    stale_val.write_text("legacy,val\n", encoding="utf-8")

    cfg = Config.model_validate({"dataset": {"target_classes": ["cling"]}})
    summary = compose_dataset(cfg, "p")

    assert not stale_val.exists()
    header = (layout.dataset_dir / "train.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "split_bucket" in header
    assert "split_group" in header
    assert "source_stem" in header
    assert "original_source" in header
    assert "augmentation_family" in header
    assert "source_split_group" in header
    assert summary["split_diagnostics"]["strategy"] == "bucket-session-grouped"


def test_val_fraction_drives_live_split_pool_rows(tmp_path, monkeypatch):
    """cfg.dataset.val_fraction drives the live train/val split at train time
    (sed.data.splits.split_pool_rows over train.csv), not dataset composition."""
    _patch_library_paths(tmp_path, monkeypatch)
    import csv as csvlib
    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection
    from sed.data.splits import split_pool_rows

    layout = ProjectLayout("p")
    layout.create()
    _make_pretraining_session_corpus("cling", 20)
    write_dataset_selection(layout, _pretraining_project_selection("cling", "cling"))

    cfg = Config.model_validate({"dataset": {"target_classes": ["cling"]}})
    compose_dataset(cfg, "p")
    with open(layout.dataset_dir / "train.csv", encoding="utf-8") as f:
        rows = list(csvlib.DictReader(f))

    _train_small, val_small, _modes = split_pool_rows(rows, val_frac=0.1, seed=42)
    _train_big, val_big, _modes = split_pool_rows(rows, val_frac=0.4, seed=42)

    assert len(val_small) == 2   # round(0.1 * 20)
    assert len(val_big) == 8     # round(0.4 * 20)


def test_pretraining_test_enabled_false_disables_auto_carve(tmp_path, monkeypatch):
    """The split-derived carve can be turned off without losing the configured %."""
    _patch_library_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    _make_pretraining_session_corpus("cling", 20)
    write_dataset_selection(layout, _pretraining_project_selection("cling", "cling"))

    cfg_off = Config.model_validate({
        "project": {"kind": "pretraining"},
        "dataset": {
            "target_classes": ["cling"],
            "pretraining_test_enabled": False,
            "pretraining_test_fraction": 0.2,
        },
    })
    off = compose_dataset(cfg_off, "p")
    assert off["frozen_test"]["present"] is False
    assert off["splits"]["test"] == 0
    assert not (layout.dataset_dir / "frozen_test_split.json").exists()

    cfg_on = Config.model_validate({
        "project": {"kind": "pretraining"},
        "dataset": {
            "target_classes": ["cling"],
            "pretraining_test_enabled": True,
            "pretraining_test_fraction": 0.2,
        },
    })
    on = compose_dataset(cfg_on, "p", build_split_test=True)
    assert on["frozen_test"]["present"] is True
    assert on["splits"]["test"] > 0
    assert (layout.dataset_dir / "frozen_test_split.json").exists()


def test_pretraining_test_disable_then_reenable_resumes_same_carve(tmp_path, monkeypatch):
    """Disabling doesn't discard the persisted carve; re-enabling reuses it unchanged."""
    _patch_library_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    _make_pretraining_session_corpus("cling", 20)
    write_dataset_selection(layout, _pretraining_project_selection("cling", "cling"))

    def _cfg(enabled: bool) -> "Config":
        from sed.config import Config
        return Config.model_validate({
            "project": {"kind": "pretraining"},
            "dataset": {
                "target_classes": ["cling"],
                "pretraining_test_enabled": enabled,
                "pretraining_test_fraction": 0.2,
            },
        })

    first = compose_dataset(_cfg(True), "p", build_split_test=True)
    first_test_rows = (layout.dataset_dir / "test.csv").read_text(encoding="utf-8")
    assert first["frozen_test"]["total"] > 0

    disabled_preview = compose_dataset(_cfg(False), "p")
    assert disabled_preview["needs_confirmation"] is True
    disabled = compose_dataset(_cfg(False), "p", confirm_frozen_test_change=True)
    assert disabled["frozen_test"]["present"] is False

    resumed = compose_dataset(_cfg(True), "p")
    resumed_test_rows = (layout.dataset_dir / "test.csv").read_text(encoding="utf-8")
    assert resumed["frozen_test"]["total"] == first["frozen_test"]["total"]
    assert resumed_test_rows == first_test_rows


def test_routine_compose_never_auto_carves_split_derived_frozen_test(tmp_path, monkeypatch):
    """Ordinary compose (source-rule/class/sample edits) must never conjure a
    split-derived frozen test on its own — only an explicit build does."""
    _patch_library_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    for _ in range(20):
        _library_clip(tmp_path, "glass", project_id="p")
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "glass": [{"labels": ["glass"], "project_ids": ["p"], "split": "train"}],
        },
    })

    cfg = Config.model_validate({
        "dataset": {
            "target_classes": ["glass"],
            "split_test_enabled": True,
            "split_test_fraction": 0.2,
        },
    })
    summary = compose_dataset(cfg, "p")

    assert summary["frozen_test"]["present"] is False
    assert summary["splits"]["test"] == 0
    assert not (layout.dataset_dir / "frozen_test_split.json").exists()


def test_venue_project_can_opt_into_split_derived_frozen_test(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    for _ in range(20):
        _library_clip(tmp_path, "glass", project_id="p")
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "glass": [{"labels": ["glass"], "project_ids": ["p"], "split": "train"}],
        },
    })

    cfg = Config.model_validate({
        "dataset": {
            "target_classes": ["glass"],
            "split_test_enabled": True,
            "split_test_fraction": 0.2,
        },
    })
    summary = compose_dataset(cfg, "p", build_split_test=True)

    assert summary["frozen_test"]["present"] is True
    assert summary["frozen_test"]["source"] == "split_derived"
    assert summary["frozen_test"]["source_counts"]["split_derived"] > 0
    assert (layout.dataset_dir / "frozen_test_split.json").exists()
    split_manifest = json.loads(
        (layout.dataset_dir / "frozen_test_split.json").read_text(encoding="utf-8")
    )
    assert split_manifest["carve"] == "bucket-session-grouped"
    assert split_manifest["split_diagnostics"]["strategy"] == "bucket-session-grouped"
    assert "split_derived" in (layout.dataset_dir / "test.csv").read_text(encoding="utf-8")


def test_singleton_source_buckets_represent_every_class_in_train_val_and_test(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)
    import csv as csvlib

    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection
    from sed.data.splits import split_pool_rows_detailed

    layout = ProjectLayout("p")
    layout.create()
    _make_pretraining_corpus({
        "noise-a": 4, "noise-b": 4, "noise-c": 4,
        "cheers-a": 4, "cheers-b": 4, "cheers-c": 4,
    })
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "_negative": [{
                "source_bucket": "pretraining",
                "labels": ["noise-a", "noise-b", "noise-c"],
                "split": "train",
            }],
            "cheers": [{
                "source_bucket": "pretraining",
                "labels": ["cheers-a", "cheers-b", "cheers-c"],
                "split": "train",
            }],
        },
    })
    cfg = Config.model_validate({
        "project": {"kind": "pretraining"},
        "dataset": {
            "target_classes": ["_negative", "cheers"],
            "val_fraction": 0.25,
            "split_test_enabled": True,
            "split_test_fraction": 0.20,
        },
    })

    summary = compose_dataset(cfg, "p", build_split_test=True)
    assert summary["frozen_test"]["by_class"]["_negative"] > 0
    assert summary["frozen_test"]["by_class"]["cheers"] > 0

    with open(layout.dataset_dir / "train.csv", encoding="utf-8") as f:
        pool_rows = list(csvlib.DictReader(f))
    train_rows, val_rows, diagnostics = split_pool_rows_detailed(
        pool_rows, val_frac=cfg.dataset.val_fraction, seed=cfg.training.seed,
    )
    for class_name in cfg.model.classes:
        class_train = [row for row in train_rows if row["class_name"] == class_name]
        class_val = [row for row in val_rows if row["class_name"] == class_name]
        assert class_train, f"{class_name} missing from train"
        assert class_val, f"{class_name} missing from validation"
        assert not (
            {row["split_group"] for row in class_train}
            & {row["split_group"] for row in class_val}
        )
    assert diagnostics["by_class"]["_negative"]["val"] > 0
    assert diagnostics["by_class"]["cheers"]["val"] > 0


def test_selected_frozen_buckets_add_to_split_derived_test(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    _make_pretraining_session_corpus("cling", 30)
    _library_clip(tmp_path, "heldout_glass", project_id="p")
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {
            "glass": [
                {"source_bucket": "pretraining", "labels": ["cling"], "split": "train"},
                {"labels": ["heldout_glass"], "project_ids": ["p"], "split": "test"},
            ],
        },
    })

    cfg = Config.model_validate({
        "project": {"kind": "pretraining"},
        "dataset": {
            "target_classes": ["glass"],
            "split_test_enabled": True,
            "split_test_fraction": 0.2,
        },
    })
    summary = compose_dataset(cfg, "p", build_split_test=True)

    counts = summary["frozen_test"]["source_counts"]
    assert summary["frozen_test"]["source"] == "combined"
    assert counts["selected_buckets"] == 1
    assert counts["split_derived"] > 0
    rows = (layout.dataset_dir / "test.csv").read_text(encoding="utf-8")
    assert "selected_buckets" in rows
    assert "split_derived" in rows


def test_split_derived_frozen_test_stays_fixed_when_dataset_grows(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    _make_pretraining_session_corpus("cling", 20)
    write_dataset_selection(layout, _pretraining_project_selection("glass", "cling"))
    cfg = Config.model_validate({
        "project": {"kind": "pretraining"},
        "dataset": {
            "target_classes": ["glass"],
            "split_test_enabled": True,
            "split_test_fraction": 0.2,
        },
    })

    first = compose_dataset(cfg, "p", build_split_test=True)
    first_rows = (layout.dataset_dir / "test.csv").read_text(encoding="utf-8")
    _make_pretraining_session_corpus("cling", 30)
    second = compose_dataset(cfg, "p")
    second_rows = (layout.dataset_dir / "test.csv").read_text(encoding="utf-8")

    assert second["frozen_test"]["hash"] == first["frozen_test"]["hash"]
    assert second_rows == first_rows


def test_changing_split_derived_frozen_test_requires_confirmation(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)
    from sed.config import Config, ProjectLayout
    from sed.data.compose import compose_dataset, write_dataset_selection

    layout = ProjectLayout("p")
    layout.create()
    _make_pretraining_session_corpus("cling", 30)
    write_dataset_selection(layout, _pretraining_project_selection("glass", "cling"))

    cfg_small = Config.model_validate({
        "project": {"kind": "pretraining"},
        "dataset": {
            "target_classes": ["glass"],
            "split_test_enabled": True,
            "split_test_fraction": 0.1,
        },
    })
    first = compose_dataset(cfg_small, "p", build_split_test=True)
    first_rows = (layout.dataset_dir / "test.csv").read_text(encoding="utf-8")

    cfg_large = Config.model_validate({
        "project": {"kind": "pretraining"},
        "dataset": {
            "target_classes": ["glass"],
            "split_test_enabled": True,
            "split_test_fraction": 0.3,
        },
    })
    # Routine compose (no explicit rebuild) must not resize the frozen test
    # just because the configured fraction changed.
    routine = compose_dataset(cfg_large, "p")
    assert routine.get("needs_confirmation") is not True
    assert routine["frozen_test"]["hash"] == first["frozen_test"]["hash"]
    assert (layout.dataset_dir / "test.csv").read_text(encoding="utf-8") == first_rows

    preview = compose_dataset(cfg_large, "p", build_split_test=True)
    assert preview["needs_confirmation"] is True
    assert (layout.dataset_dir / "test.csv").read_text(encoding="utf-8") == first_rows

    confirmed = compose_dataset(cfg_large, "p", confirm_frozen_test_change=True, build_split_test=True)
    assert confirmed.get("needs_confirmation") is not True
    assert confirmed["frozen_test"]["hash"] != first["frozen_test"]["hash"]


def test_selection_fingerprint_ignores_volatile_rule_fields():
    from sed.data.compose import selection_rules_fingerprint

    saved = {"assignments": {"cling": [
        {"labels": ["x"], "id": "aaa", "count": 3, "updated_at": "t1"},
    ]}}
    resaved = {"assignments": {"cling": [
        {"labels": ["x"], "id": "bbb", "count": 9, "updated_at": "t2"},
    ]}}
    changed = {"assignments": {"cling": [{"labels": ["y"]}]}}

    assert selection_rules_fingerprint(saved) == selection_rules_fingerprint(resaved)
    assert selection_rules_fingerprint(saved) != selection_rules_fingerprint(changed)


def test_compose_records_selection_fingerprint_for_staleness(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import Config, ProjectLayout
    from sed.data.compose import (
        compose_dataset,
        dataset_compose_state,
        load_dataset_selection,
        write_dataset_selection,
    )

    layout = ProjectLayout("p")
    layout.create()
    _library_clip(tmp_path, "ui_label", project_id="p")
    cfg = Config.model_validate({"dataset": {"target_classes": ["cling"]}})
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {"cling": [{"labels": ["ui_label"], "split": "train"}]},
    })

    # Rules exist but nothing composed yet -> stale.
    state = dataset_compose_state(layout, load_dataset_selection(layout), cfg)
    assert state["composed"] is False
    assert state["stale"] is True

    summary = compose_dataset(cfg, "p")
    assert summary["selection_fingerprint"]
    state = dataset_compose_state(layout, load_dataset_selection(layout), cfg)
    assert state["composed"] is True
    assert state["stale"] is False

    # Volatile bookkeeping churn (ids/counts/timestamps) must not flag stale.
    selection = load_dataset_selection(layout)
    selection["assignments"]["cling"][0].update({"count": 99, "updated_at": "2026-01-01T00:00:00Z"})
    write_dataset_selection(layout, selection)
    state = dataset_compose_state(layout, load_dataset_selection(layout), cfg)
    assert state["stale"] is False

    # A real rule change flags stale until the next compose.
    selection["assignments"]["cling"].append({"labels": ["other"], "split": "train"})
    write_dataset_selection(layout, selection)
    state = dataset_compose_state(layout, load_dataset_selection(layout), cfg)
    assert state["stale"] is True

    compose_dataset(cfg, "p")
    state = dataset_compose_state(layout, load_dataset_selection(layout), cfg)
    assert state["stale"] is False


def test_dataset_presets_save_apply_delete_roundtrip(tmp_path, monkeypatch):
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import ProjectLayout
    from sed.data.compose import (
        apply_dataset_preset,
        dataset_preset_state,
        delete_dataset_preset,
        load_dataset_selection,
        save_dataset_preset,
        write_dataset_selection,
    )

    layout = ProjectLayout("p")
    layout.create()
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {"cling": [{"labels": ["ui_label"], "split": "train", "id": "r1"}]},
        "augmentation_recipes": [{"id": "keepme"}],
    })

    selection = load_dataset_selection(layout)
    record = save_dataset_preset(layout, "dataset 01", selection)
    assert record["slug"] == "dataset-01"
    assert record["rule_count"] == 1

    state = dataset_preset_state(layout, selection)
    assert state["active"] == "dataset-01"
    assert state["modified"] is False

    # Wipe the rules -> active preset reads as modified.
    selection["assignments"] = {}
    write_dataset_selection(layout, selection)
    state = dataset_preset_state(layout, load_dataset_selection(layout))
    assert state["modified"] is True

    # Applying restores the rules but keeps non-compose selection state.
    restored, applied = apply_dataset_preset(layout, "dataset-01")
    assert applied["name"] == "dataset 01"
    assert restored["assignments"]["cling"][0]["labels"] == ["ui_label"]
    assert restored["augmentation_recipes"] == [{"id": "keepme"}]
    state = dataset_preset_state(layout, load_dataset_selection(layout))
    assert state["modified"] is False

    # Save under a second name, switch between the two.
    selection = load_dataset_selection(layout)
    selection["assignments"] = {"cling": [{"labels": ["other"], "split": "train"}]}
    write_dataset_selection(layout, selection)
    save_dataset_preset(layout, "pretraining 01", load_dataset_selection(layout))
    state = dataset_preset_state(layout, load_dataset_selection(layout))
    assert [p["slug"] for p in state["presets"]] == ["dataset-01", "pretraining-01"]
    assert state["active"] == "pretraining-01"

    restored, _ = apply_dataset_preset(layout, "dataset 01")
    assert restored["assignments"]["cling"][0]["labels"] == ["ui_label"]

    assert delete_dataset_preset(layout, "dataset 01") is True
    state = dataset_preset_state(layout, load_dataset_selection(layout))
    assert [p["slug"] for p in state["presets"]] == ["pretraining-01"]
    assert state["active"] == ""
    assert delete_dataset_preset(layout, "dataset 01") is False


def test_dataset_preset_v2_round_trips_classes_and_split(tmp_path, monkeypatch):
    """A preset is the whole dataset definition — rules AND classes — so
    switching between a pretraining corpus and a venue fine-tune is one action."""
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import ProjectLayout, load_config
    from sed.data.compose import (
        apply_dataset_preset,
        dataset_preset_state,
        load_dataset_selection,
        save_dataset_preset,
        write_dataset_selection,
    )

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(
        '[dataset]\ntarget_classes = ["_negative", "cheers", "clap"]\nval_fraction = 0.2\n',
        encoding="utf-8",
    )
    write_dataset_selection(layout, {"ui_managed": True, "assignments": {}})

    cfg = load_config("p")
    record = save_dataset_preset(layout, "domain corpus", load_dataset_selection(layout), cfg)

    assert record["schema_version"] == 2
    assert record["classes"] == ["_negative", "cheers", "clap"]
    assert record["split"]["val_fraction"] == pytest.approx(0.2)

    # Move the project to a different class space, then restore the preset.
    from sed.config import update_config_toml_values
    update_config_toml_values(
        layout.config_toml, {"dataset": {"target_classes": ["_negative", "wine"]}}
    )
    assert list(load_config("p").dataset.target_classes) == ["_negative", "wine"]

    apply_dataset_preset(layout, "domain-corpus")
    restored = load_config("p")
    assert list(restored.dataset.target_classes) == ["_negative", "cheers", "clap"]
    assert restored.dataset.val_fraction == pytest.approx(0.2)

    state = dataset_preset_state(layout, load_dataset_selection(layout), restored)
    assert state["modified"] is False
    entry = next(p for p in state["presets"] if p["slug"] == "domain-corpus")
    assert entry["has_classes"] is True
    assert entry["classes_match_current"] is True


def test_dataset_preset_state_flags_a_class_space_change_as_modified(tmp_path, monkeypatch):
    """Rule-identical but a different class space is still a different dataset;
    the preset must not read as clean."""
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import ProjectLayout, load_config, update_config_toml_values
    from sed.data.compose import (
        dataset_preset_state,
        load_dataset_selection,
        save_dataset_preset,
        write_dataset_selection,
    )

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(
        '[dataset]\ntarget_classes = ["_negative", "cheers"]\n', encoding="utf-8"
    )
    write_dataset_selection(layout, {"ui_managed": True, "assignments": {}})
    save_dataset_preset(layout, "d1", load_dataset_selection(layout), load_config("p"))

    update_config_toml_values(
        layout.config_toml, {"dataset": {"target_classes": ["_negative", "cheers", "clap"]}}
    )
    state = dataset_preset_state(layout, load_dataset_selection(layout), load_config("p"))
    assert state["modified"] is True


def test_dataset_preset_v1_without_classes_leaves_project_classes_alone(tmp_path, monkeypatch):
    """Presets saved before v2 carry no class list. Applying one must restore
    rules only — never clear the project's classes — and say so in the state."""
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import ProjectLayout, load_config
    from sed.data.compose import (
        apply_dataset_preset,
        dataset_preset_state,
        load_dataset_selection,
        save_dataset_preset,
        write_dataset_selection,
    )

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(
        '[dataset]\ntarget_classes = ["_negative", "cheers"]\n', encoding="utf-8"
    )
    write_dataset_selection(layout, {"ui_managed": True, "assignments": {}})

    # No cfg → v1-shaped record, exactly what is already on disk for old presets.
    record = save_dataset_preset(layout, "legacy", load_dataset_selection(layout))
    assert record["schema_version"] == 1
    assert "classes" not in record

    before = list(load_config("p").dataset.target_classes)
    apply_dataset_preset(layout, "legacy")
    assert list(load_config("p").dataset.target_classes) == before

    state = dataset_preset_state(layout, load_dataset_selection(layout), load_config("p"))
    entry = next(p for p in state["presets"] if p["slug"] == "legacy")
    assert entry["has_classes"] is False
    assert entry["classes_match_current"] is None


def test_dataset_preset_provenance_travels_with_a_run(tmp_path, monkeypatch):
    """A project moves between dataset definitions by switching presets, so a
    run's numbers are unattributable without the name that produced them."""
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import ProjectLayout, load_config
    from sed.data.compose import (
        load_dataset_selection,
        save_dataset_preset,
        write_dataset_selection,
    )
    from sed.train.train import _dataset_preset_provenance

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(
        '[dataset]\ntarget_classes = ["_negative", "cheers"]\n', encoding="utf-8")
    write_dataset_selection(layout, {
        "ui_managed": True,
        "assignments": {"cheers": [{"labels": ["glass"], "split": "train", "id": "r1"}]},
    })
    cfg = load_config("p")
    save_dataset_preset(layout, "domain corpus", load_dataset_selection(layout), cfg)

    prov = _dataset_preset_provenance(layout, cfg)
    assert prov["preset_name"] == "domain corpus"
    assert prov["preset_slug"] == "domain-corpus"
    assert prov["preset_classes"] == ["_negative", "cheers"]
    assert prov["preset_modified"] is False


def test_editing_after_loading_marks_the_run_as_not_matching_the_preset(tmp_path, monkeypatch):
    """A preset can be loaded and then changed — the run was NOT trained on the
    saved definition, and the runs table has to be able to say so."""
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import ProjectLayout, load_config
    from sed.data.compose import (
        load_dataset_selection,
        save_dataset_preset,
        write_dataset_selection,
    )
    from sed.train.train import _dataset_preset_provenance

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(
        '[dataset]\ntarget_classes = ["_negative", "cheers"]\n', encoding="utf-8")
    write_dataset_selection(layout, {"ui_managed": True, "assignments": {}})
    cfg = load_config("p")
    save_dataset_preset(layout, "d1", load_dataset_selection(layout), cfg)

    selection = load_dataset_selection(layout)
    selection["assignments"] = {"cheers": [{"labels": ["late-edit"], "split": "train"}]}
    write_dataset_selection(layout, selection)

    assert _dataset_preset_provenance(layout, cfg)["preset_modified"] is True


def test_preset_provenance_never_fails_a_run(tmp_path, monkeypatch):
    """Provenance is bookkeeping — a project without presets must still train."""
    _patch_library_paths(tmp_path, monkeypatch)

    from sed.config import ProjectLayout, load_config
    from sed.train.train import _dataset_preset_provenance

    layout = ProjectLayout("p")
    layout.create()
    layout.config_toml.write_text(
        '[dataset]\ntarget_classes = ["_negative"]\n', encoding="utf-8")

    prov = _dataset_preset_provenance(layout, load_config("p"))
    assert prov["preset_name"] == ""
    assert prov["preset_classes"] == ["_negative"]
