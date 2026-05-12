"""Tests for named frozen-test sets and the leakage remedy."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sed.config import ProjectLayout
from sed.data.frozen_test_presets import (
    apply_frozen_test_preset,
    delete_frozen_test_preset,
    derive_preset_excluding,
    frozen_test_preset_state,
    preset_rows,
    read_test_csv,
    save_frozen_test_preset,
    frozen_test_hash,
)

HEADER = "path,class_name,session_id,sample_rate,duration_s,is_target,origin\n"


def _layout(tmp_path, monkeypatch) -> ProjectLayout:
    from sed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    layout = ProjectLayout("p")
    layout.create()
    return layout


def _write_test_csv(layout, rows: list[tuple[str, str, str]]) -> None:
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{p},{c},{s},44100,0.4,1,sounds\n" for p, c, s in rows)
    (layout.dataset_dir / "test.csv").write_text(HEADER + body, encoding="utf-8")


def test_hash_matches_the_composer_that_stamps_every_manifest():
    """`test_set_hash` on a manifest block is what tells two runs they were
    scored on the same ground truth. If these two drift, that breaks silently."""
    from sed.data.compose import _test_set_hash

    rows = [
        {"path": "/lib/b.wav", "class_name": "cheers"},
        {"path": "/lib/a.wav", "class_name": "_negative"},
    ]
    clips = [SimpleNamespace(path=r["path"], class_name=r["class_name"]) for r in rows]
    assert frozen_test_hash(rows) == _test_set_hash(clips)


def test_hash_ignores_row_order_but_not_membership():
    a = [{"path": "/x/1.wav", "class_name": "c"}, {"path": "/x/2.wav", "class_name": "c"}]
    assert frozen_test_hash(a) == frozen_test_hash(list(reversed(a)))
    assert frozen_test_hash(a) != frozen_test_hash(a[:1])


def test_save_and_apply_round_trips_a_named_test_set(tmp_path, monkeypatch):
    layout = _layout(tmp_path, monkeypatch)
    _write_test_csv(layout, [
        ("/lib/a.wav", "cheers", "s1"),
        ("/lib/n.wav", "_negative", "s2"),
    ])

    record = save_frozen_test_preset(layout, "benchmark v1")
    assert record["slug"] == "benchmark-v1"
    assert record["n_clips"] == 2
    assert record["by_class"] == {"cheers": 1, "_negative": 1}

    # Overwrite the live test set, then restore it from the preset.
    _write_test_csv(layout, [("/lib/other.wav", "cheers", "s9")])
    assert frozen_test_preset_state(layout)["presets"][0]["is_current"] is False

    apply_frozen_test_preset(layout, "benchmark-v1")
    _fields, rows = read_test_csv(layout.dataset_dir / "test.csv")
    assert sorted(r["path"] for r in rows) == ["/lib/a.wav", "/lib/n.wav"]

    state = frozen_test_preset_state(layout)
    assert state["presets"][0]["is_current"] is True
    assert state["current_unsaved"] is False


def test_applying_a_preset_updates_the_snapshot_hash_so_stale_scores_are_flagged(
    tmp_path, monkeypatch
):
    """A changed test set must invalidate comparability of earlier scores; the
    runs table detects that from dataset_snapshot.json's frozen_test hash."""
    layout = _layout(tmp_path, monkeypatch)
    _write_test_csv(layout, [("/lib/a.wav", "cheers", "s1")])
    (layout.dataset_dir / "dataset_snapshot.json").write_text(
        json.dumps({"frozen_test": {"hash": "sha256:old", "total": 99}}), encoding="utf-8"
    )
    record = save_frozen_test_preset(layout, "v1")

    apply_frozen_test_preset(layout, "v1")

    snap = json.loads((layout.dataset_dir / "dataset_snapshot.json").read_text())
    assert snap["frozen_test"]["hash"] == record["hash"]
    assert snap["frozen_test"]["total"] == 1
    assert snap["frozen_test"]["preset"] == "v1"


def test_derive_excluding_removes_leaked_clips_without_touching_the_source(
    tmp_path, monkeypatch
):
    """The remedy is additive: the original benchmark stays reproducible for the
    scores already computed against it."""
    layout = _layout(tmp_path, monkeypatch)
    _write_test_csv(layout, [
        ("/lib/leaked.wav", "cheers", "s1"),
        ("/lib/clean.wav", "cheers", "s2"),
        ("/lib/n.wav", "_negative", "s3"),
    ])
    original = save_frozen_test_preset(layout, "v1")

    derived = derive_preset_excluding(
        layout, "v1 without leaks", ["/lib/leaked.wav"], source="v1"
    )

    assert derived["n_clips"] == 2
    assert derived["excluded_count"] == 1
    assert derived["derived_from"] == "v1"
    assert derived["hash"] != original["hash"]

    # The source preset and the live test set are untouched.
    _f, source_rows = preset_rows(layout, "v1")
    assert len(source_rows) == 3
    _f, live = read_test_csv(layout.dataset_dir / "test.csv")
    assert len(live) == 3


def test_derive_excluding_normalizes_paths_before_matching(tmp_path, monkeypatch):
    layout = _layout(tmp_path, monkeypatch)
    _write_test_csv(layout, [
        (str(tmp_path / "lib" / "a.wav"), "cheers", "s1"),
        (str(tmp_path / "lib" / "b.wav"), "cheers", "s2"),
    ])
    # Same file, different spelling of the path.
    messy = str(tmp_path / "lib" / "sub" / ".." / "a.wav")
    derived = derive_preset_excluding(layout, "trimmed", [messy])
    assert derived["n_clips"] == 1


def test_derive_excluding_refuses_to_empty_the_test_set(tmp_path, monkeypatch):
    """Removing everything is not a benchmark — say so instead of writing an
    empty CSV that scores as a silent zero."""
    layout = _layout(tmp_path, monkeypatch)
    _write_test_csv(layout, [("/lib/a.wav", "cheers", "s1")])

    with pytest.raises(ValueError, match="would empty the test set"):
        derive_preset_excluding(layout, "empty", ["/lib/a.wav"])


def test_state_flags_a_live_test_set_that_has_no_name(tmp_path, monkeypatch):
    """An unnamed live test set cannot be restored after the next rebuild."""
    layout = _layout(tmp_path, monkeypatch)
    _write_test_csv(layout, [("/lib/a.wav", "cheers", "s1")])

    assert frozen_test_preset_state(layout)["current_unsaved"] is True
    save_frozen_test_preset(layout, "named")
    assert frozen_test_preset_state(layout)["current_unsaved"] is False


def test_saving_the_same_name_updates_in_place_and_keeps_created_at(tmp_path, monkeypatch):
    layout = _layout(tmp_path, monkeypatch)
    _write_test_csv(layout, [("/lib/a.wav", "cheers", "s1")])
    first = save_frozen_test_preset(layout, "v1")

    _write_test_csv(layout, [("/lib/a.wav", "cheers", "s1"), ("/lib/b.wav", "cheers", "s2")])
    second = save_frozen_test_preset(layout, "v1")

    assert second["created_at"] == first["created_at"]
    assert second["n_clips"] == 2
    assert len(frozen_test_preset_state(layout)["presets"]) == 1


def test_delete_removes_the_index_row_and_the_csv(tmp_path, monkeypatch):
    layout = _layout(tmp_path, monkeypatch)
    _write_test_csv(layout, [("/lib/a.wav", "cheers", "s1")])
    save_frozen_test_preset(layout, "v1")

    assert delete_frozen_test_preset(layout, "v1") is True
    assert frozen_test_preset_state(layout)["presets"] == []
    assert delete_frozen_test_preset(layout, "v1") is False


def test_unknown_preset_raises_keyerror(tmp_path, monkeypatch):
    layout = _layout(tmp_path, monkeypatch)
    with pytest.raises(KeyError):
        apply_frozen_test_preset(layout, "nope")


def test_frozen_test_presets_stay_torch_free():
    import subprocess
    import sys

    probe = "import sed.data.frozen_test_presets, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
