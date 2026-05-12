"""Focused regression coverage for composition-only library rules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def test_library_target_class_chips_refresh_when_source_rules_are_empty():
    """Clearing the last rule must remove its stale chip sample total."""
    source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    start = source.index("function renderSelectionRules()")
    end = source.index("// The always-visible summary strip", start)
    function = source[start:end]
    empty_return = function.index("if (!filtered.length)")
    assert function.index("tcRenderChips();") < empty_return


def test_library_tag_cloud_starts_compact_and_can_expand():
    source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    assert "const previewLimit = 5;" in source
    assert "data-tag-more" in source
    # The toggle must name how many tags are still hidden, and flip the
    # expanded flag. The wording of the button is not the contract, so match
    # the interpolation rather than a copy string that reviewers may reword.
    assert "${hiddenCount}" in source
    assert 'cloud.dataset.expanded = expanded ? "false" : "true"' in source


def test_review_panel_origin_counts_cover_raw_derived_and_frozen_items():
    from pathlib import Path

    from sed.data.inspect import review_panel_origin_counts

    raw = [(Path("loud.wav"), "loud", {"origin_class": "ikea-wine"})]
    derived = [(
        Path("aug.wav"),
        "ikea-wine × noise",
        {"assigned_target_class": "loud", "origin_class": "ikea-wine"},
    )]
    frozen = [(Path("test.wav"), "loud", {"test_origin": "holdout-room"})]

    counts = review_panel_origin_counts(raw, derived, frozen)
    assert counts["dataset"] == {"loud": {"ikea-wine": 1}}
    # The Augmented tab renders one block per augmentation group, so its counts
    # must be keyed by that group — keying them by the assigned target class
    # ("loud") produced summary rows for a block the tab never renders, showing
    # "0 total" with nothing previewable under it.
    assert counts["augmented"] == {"ikea-wine × noise": {"ikea-wine": 1}}
    assert counts["frozen"] == {"loud": {"holdout-room": 1}}


def test_clip_tab_summary_checkboxes_filter_each_preview_tab():
    source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    for marker in (
        'data-summary-tab',
        'class="summary-subbucket-check"',
        "SUBBUCKET_EXCLUDED",
        "filterReviewClipsList",
        "renderClipTabSummaries",
    ):
        assert marker in source


def test_summary_rows_are_limited_to_previewable_buckets():
    """A row whose checkbox cannot change what the tab shows is noise.

    Blocks the tab does not render (no clips and no true total) and empty
    buckets were both listed with a "0 total" and nothing previewable under
    them; the summary drops them and says how many it hid.
    """
    source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    assert "if (!total && !clips.length) { hidden += Object.keys(origins).length; return; }" in source
    assert ".filter(origin => Number(origins[origin] || 0) > 0)" in source
    # The Augmented tab groups by augmentation recipe, so its column is named
    # for that rather than for a target class it has no block for.
    assert 'const SUMMARY_KEY_HEADER = { augmented: "augmentation group" };' in source
    assert "{{ 'augmentation group' if tab_type == 'augmented' else 'target class' }}" in source


def test_stale_compose_button_uses_ink_that_reads_on_solid_amber():
    """--amber-deep-* is "amber text on a tinted surface" and flips light in the
    dark theme, so using it on the solid amber fill gave light-on-light."""
    source = Path("src/sed/data/inspect.py").read_text(encoding="utf-8")
    assert ".compose-run.stale, .compose-run.stale:hover { background: var(--amber-solid);" in source
    assert "color: var(--amber-ink)" in source
    # Defined in BOTH palettes, dark in both, because --amber-solid is bright in both.
    assert source.count("--amber-ink:") == 2


def _paths(tmp_path, monkeypatch):
    from sed import config as config_mod
    from sed.data import library as library_mod

    monkeypatch.setattr(config_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(config_mod, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(library_mod, "LIBRARY_DIR", tmp_path / "library")


def test_legacy_rule_roles_migrate_but_recording_roles_do_not(tmp_path, monkeypatch):
    _paths(tmp_path, monkeypatch)
    from sed.config import ProjectLayout
    from sed.data.compose import load_dataset_selection, write_dataset_selection
    from sed.data.library import Library

    layout = ProjectLayout("p")
    layout.create()
    write_dataset_selection(layout, {
        "assignments": {"glass": [
            {"id": "holdout", "labels": ["glass"], "dataset_role": "test"},
            {"id": "ordinary", "labels": ["glass"], "dataset_role": "ignore"},
        ]},
    })
    selection = load_dataset_selection(layout)
    assert selection["assignments"]["glass"][0]["split"] == "test"
    assert all("dataset_role" not in row for row in selection["assignments"]["glass"])

    source = tmp_path / "glass.wav"
    sf.write(source, np.zeros(1600, dtype=np.float32), 16000)
    clip = Library().import_clip(source, label="glass", project_id="p", dataset_role="test")
    sidecar = clip.absolute_path(Library().root).with_suffix(".json")
    assert "dataset_role" not in json.loads(sidecar.read_text(encoding="utf-8"))
    assert not hasattr(clip, "dataset_role")


def test_mono_filter_uses_audio_channels_and_keeps_channel_split_mono(tmp_path, monkeypatch):
    _paths(tmp_path, monkeypatch)
    from sed.data.library import Library, clip_matches_mic_rule

    mono = tmp_path / "mono.wav"
    multi = tmp_path / "multi.wav"
    sf.write(mono, np.zeros((1600, 1), dtype=np.float32), 16000)
    sf.write(multi, np.zeros((1600, 2), dtype=np.float32), 16000)
    lib = Library()
    mono_clip = lib.import_clip(mono, label="glass", project_id="p")
    multi_clip = lib.import_clip(multi, label="glass", project_id="p")
    assert clip_matches_mic_rule(mono_clip, {"mic_filter": "mono"})
    assert not clip_matches_mic_rule(multi_clip, {"mic_filter": "mono"})
    assert clip_matches_mic_rule(multi_clip, {"mic_filter": "multichannel"})

    mono_clip.clip_kind = "channel_split"
    mono_clip.channel_count = 2
    mono_clip.channel_index = 1
    assert clip_matches_mic_rule(mono_clip, {"mic_filter": "mono"})
    assert not clip_matches_mic_rule(mono_clip, {"mic_filter": "multichannel"})


def test_blank_dataset_and_preset_rename_are_composition_state(tmp_path, monkeypatch):
    _paths(tmp_path, monkeypatch)
    from sed.config import ProjectLayout
    from sed.data.compose import (
        apply_dataset_preset,
        load_dataset_selection,
        new_dataset_selection,
        rename_dataset_preset,
        save_dataset_preset,
        write_dataset_selection,
    )

    layout = ProjectLayout("p")
    layout.create()
    selection = {
        "ui_managed": True,
        "assignments": {"glass": [{"labels": ["glass"], "split": "test"}]},
    }
    write_dataset_selection(layout, selection)
    record = save_dataset_preset(layout, "holdout set", selection)
    renamed = rename_dataset_preset(layout, record["slug"], "frozen glass")
    assert renamed["slug"] == "frozen-glass"
    loaded, _ = apply_dataset_preset(layout, renamed["slug"])
    assert loaded["assignments"]["glass"][0]["split"] == "test"
    blank = new_dataset_selection(layout)
    assert blank["assignments"] == {}
    assert blank["ui_managed"] is True
    assert load_dataset_selection(layout)["assignments"] == {}


def test_augmentation_run_start_is_stable_for_non_timestamp_prefix():
    from sed.data.augment import _augmentation_started_at

    first = _augmentation_started_at("manual-batch")
    second = _augmentation_started_at("manual-batch")
    assert first == second


def test_report_cards_report_unreadable_clips_instead_of_dropping_them(tmp_path, caplog):
    """AGENTS.md: never swallow an audio error.

    A static report inlines each thumbnail, so it must decode every clip. A
    corrupt WAV used to vanish from the report with no trace — the operator saw
    a smaller dataset than they had and no reason why. It is still skipped (one
    bad file must not kill the whole report) but now says so.
    """
    import logging

    from sed.data.inspect import _card_audio_preview

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"RIFF....not actually a wav")

    with caplog.at_level(logging.WARNING, logger="sed.data.inspect"):
        assert _card_audio_preview(
            corrupt, interactive=False, metadata_cache=None) is None
    assert "corrupt.wav" in caplog.text
    assert "unreadable" in caplog.text

    # A readable clip still returns geometry plus an inlined thumbnail. The
    # thumbnail's mel geometry is the project's 32 kHz one, so write that rate.
    good = tmp_path / "good.wav"
    sf.write(str(good), np.zeros(32000, dtype="float32"), 32000)
    preview = _card_audio_preview(good, interactive=False, metadata_cache=None)
    assert preview is not None
    sample_rate, duration, spec_b64 = preview
    assert sample_rate == 32000
    assert duration == pytest.approx(1.0)
    assert spec_b64

    # A caller carrying a known duration keeps it rather than recomputing.
    _, kept, _ = _card_audio_preview(
        good, interactive=False, metadata_cache=None, duration=0.25)
    assert kept == pytest.approx(0.25)
