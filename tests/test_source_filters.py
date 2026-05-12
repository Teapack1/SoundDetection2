"""Shared recipe source-filter resolution (sed.data.source_filters).

Both the inspect UI counts and the augmentation materializer resolve recipe
rules through this module; these tests pin that shared behavior — especially
the pretraining bucket, which lives in its own index and used to silently
resolve to zero clips.
"""

import numpy as np
import pytest
import soundfile as sf


def _patch_library(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.data import library as lib_mod

    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    monkeypatch.setattr(lib_mod, "LIBRARY_DIR", tmp_path / "data_library")


def _seed_pretraining_corpus(classes: dict[str, int], sample_rate: int = 32000) -> None:
    from sed import config as cfg_mod
    from sed.data.pretraining import PretrainIndex

    for label, n in classes.items():
        d = cfg_mod.LIBRARY_DIR / "pretraining" / label
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            sf.write(
                str(d / f"{label}_{i:03d}.wav"),
                np.zeros(sample_rate // 2, dtype="float32"),
                sample_rate,
            )
    PretrainIndex().rebuild()


def _import_raw_clip(tmp_path, name: str, label: str, project_id: str = "p"):
    from sed.data.library import Library

    src = tmp_path / f"{name}.wav"
    sf.write(str(src), np.zeros(32000, dtype=np.float32), 32000)
    return Library().import_clip(src, label=label, project_id=project_id)


def test_bucket_aliases_map_to_categories():
    from sed.data.library import (
        DERIVED_CATEGORY,
        MINED_CATEGORY,
        PRETRAIN_CATEGORY,
        RAW_CATEGORY,
    )
    from sed.data.source_filters import category_for_bucket

    assert category_for_bucket({}) == RAW_CATEGORY
    assert category_for_bucket({"source_bucket": "sounds"}) == RAW_CATEGORY
    assert category_for_bucket({"source_bucket": "augmented"}) == DERIVED_CATEGORY
    assert category_for_bucket({"category": "mined"}) == MINED_CATEGORY
    assert category_for_bucket({"source_bucket": "pretraining"}) == PRETRAIN_CATEGORY
    assert category_for_bucket({"source_bucket": "bogus"}) is None


def test_unknown_bucket_resolves_to_no_clips(tmp_path, monkeypatch):
    _patch_library(tmp_path, monkeypatch)
    from sed.data.library import Library
    from sed.data.source_filters import library_paths_for_filters

    _import_raw_clip(tmp_path, "a", "glass")
    paths = library_paths_for_filters(
        Library(),
        "glass",
        project_id="p",
        frozen_test=False,
        filters=[{"source_bucket": "bogus", "labels": ["glass"]}],
    )
    assert paths == []


def test_pretraining_bucket_resolves_corpus_clips(tmp_path, monkeypatch):
    _patch_library(tmp_path, monkeypatch)
    from sed.data.library import Library
    from sed.data.source_filters import library_paths_for_filters

    _seed_pretraining_corpus({"klatsch": 3, "cheers": 2})
    paths = library_paths_for_filters(
        Library(),
        "klatsch",
        project_id="p",
        frozen_test=False,
        filters=[{"source_bucket": "pretraining", "labels": ["klatsch"]}],
    )
    assert len(paths) == 3
    assert all(p.exists() for p, _label in paths)
    assert {label for _p, label in paths} == {"klatsch"}


def test_pretraining_bucket_wildcard_and_sample_rate_pin(tmp_path, monkeypatch):
    _patch_library(tmp_path, monkeypatch)
    from sed.data.library import Library
    from sed.data.source_filters import library_paths_for_filters

    _seed_pretraining_corpus({"klatsch": 3}, sample_rate=32000)
    _seed_pretraining_corpus({"cheers": 2}, sample_rate=16000)

    def resolve(rule):
        return library_paths_for_filters(
            Library(),
            "klatsch",
            project_id="p",
            frozen_test=False,
            filters=[rule],
        )

    assert len(resolve({"source_bucket": "pretraining", "labels": ["*"]})) == 5
    pinned = resolve(
        {"source_bucket": "pretraining", "labels": ["*"], "sample_rate": 16000}
    )
    assert len(pinned) == 2
    assert {label for _p, label in pinned} == {"cheers"}


def test_pretraining_bucket_never_feeds_frozen_test(tmp_path, monkeypatch):
    _patch_library(tmp_path, monkeypatch)
    from sed.data.library import Library
    from sed.data.source_filters import library_paths_for_filters

    _seed_pretraining_corpus({"klatsch": 3})
    paths = library_paths_for_filters(
        Library(),
        "klatsch",
        project_id="p",
        frozen_test=True,
        filters=[{"source_bucket": "pretraining", "labels": ["klatsch"]}],
    )
    assert paths == []


def test_inspect_count_matches_generation_source_count(tmp_path, monkeypatch):
    """The UI recipe count and the generation source pool must agree."""
    _patch_library(tmp_path, monkeypatch)
    from sed.data.inspect_server import _count_library_filter
    from sed.data.library import Library
    from sed.data.source_filters import library_paths_for_filters

    _seed_pretraining_corpus({"klatsch": 4})
    _import_raw_clip(tmp_path, "a", "glass")
    _import_raw_clip(tmp_path, "b", "glass")

    lib = Library()
    for rule in (
        {"source_bucket": "pretraining", "labels": ["klatsch"]},
        {"labels": ["glass"], "project_ids": ["p"]},
    ):
        count = _count_library_filter(lib, rule, project_id="p")
        paths = library_paths_for_filters(
            lib,
            str(rule.get("labels", [""])[0]),
            project_id="p",
            frozen_test=False,
            filters=[rule],
        )
        assert count == len(paths) > 0


def test_settings_snapshot_covers_generator_knobs():
    from sed.config import Config
    from sed.data.source_filters import augmentation_settings_snapshot

    cfg = Config.model_validate({"dataset": {"target_classes": ["a"]}})
    snap = augmentation_settings_snapshot(cfg)
    assert snap["mix_map_enabled"] == cfg.augmentation.mix_map_enabled
    assert snap["samples_per_source"] == pytest.approx(cfg.augmentation.samples_per_source)
    assert snap["semitones_range"] == list(cfg.augmentation.semitones_range)
    assert snap["snr_db_range"] == list(cfg.augmentation.snr_db_range)
    assert augmentation_settings_snapshot(None) == {}
