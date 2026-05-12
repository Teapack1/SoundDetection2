"""Tests for dataset composition: adaptive splitting and library
auto-discovery (no [dataset.library] block needed)."""

from __future__ import annotations

from dataclasses import dataclass

from sed.data.splits import session_split_adaptive


@dataclass
class FakeClip:
    session_id: str
    class_name: str
    path: str = "x"


def test_adaptive_splits_class_with_many_sessions_session_aware():
    """A class with ≥3 sessions should get session-aware splitting."""
    clips = []
    for s in range(10):
        for _ in range(20):
            clips.append(FakeClip(session_id=f"sess_{s}", class_name="x"))
    train, val, test, modes = session_split_adaptive(clips, seed=0)
    assert "session" in modes["x"]
    # No session in two splits
    train_sess = {c.session_id for c in train}
    val_sess = {c.session_id for c in val}
    test_sess = {c.session_id for c in test}
    assert not (train_sess & val_sess)
    assert not (train_sess & test_sess)
    assert not (val_sess & test_sess)


def test_adaptive_splits_class_with_one_session_fall_back_to_random():
    """The user's actual problem: a single-session class of many clips.
    Should fall back to random clip-level split so val/test aren't empty."""
    clips = [FakeClip(session_id="only", class_name="x") for _ in range(500)]
    train, val, test, modes = session_split_adaptive(clips, seed=0)
    assert "random" in modes["x"]
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == 500


def test_adaptive_splits_mixed_classes_per_class_strategy():
    """Each class picks its own strategy. Library-style class (every clip
    its own session) ends up session-aware; on-site class with 1 session
    ends up random — both produce non-empty val/test."""
    clips = []
    # Library-style class: 200 clips, each a distinct session
    for i in range(200):
        clips.append(FakeClip(session_id=f"lib-{i:04d}", class_name="lib_class"))
    # Single-session on-site class: 100 clips all share one session_id
    for i in range(100):
        clips.append(FakeClip(session_id="onsite_sess", class_name="onsite_class"))

    train, val, test, modes = session_split_adaptive(clips, seed=0)
    assert "session" in modes["lib_class"]
    assert "random" in modes["onsite_class"]
    # Both classes appear in val
    val_classes = {c.class_name for c in val}
    assert "lib_class" in val_classes
    assert "onsite_class" in val_classes


def test_adaptive_handles_tiny_class_gracefully():
    """Even with only 2 clips total, splitter should not crash."""
    clips = [FakeClip(session_id="s1", class_name="t"),
             FakeClip(session_id="s2", class_name="t")]
    train, val, test, modes = session_split_adaptive(clips, seed=0)
    assert len(train) + len(val) + len(test) == 2


def test_auto_discover_strips_clink_prefix(tmp_path, monkeypatch):
    """`clink_wine_thin` should auto-match library label `wine_thin`."""
    from sed.data import compose as compose_mod
    from sed.data.compose import _auto_discover_library

    class FakeLibrary:
        """Pretends to have `glasses_clean/wine_thin/...` only — no exact
        `clink_wine_thin` label."""
        def query(self, category=None, label=None, tags_all=None):
            if category == "glasses_clean" and label == "wine_thin":
                return [object()]
            if category == "glasses_clean" and label == "clink_wine_thin":
                return []
            if category == "negatives_external" and label is None:
                return [object(), object()]
            return []

    lib = FakeLibrary()
    rules = _auto_discover_library(
        lib,
        classes=["_negative", "clink_wine_thin"],
        explicit_rules={},
    )
    # `clink_wine_thin` should have matched via prefix-stripping to `wine_thin`
    assert "clink_wine_thin" in rules
    assert rules["clink_wine_thin"][0]["label"] == "wine_thin"
    # _negative pulled from negatives_external (no label = all)
    assert "_negative" in rules


def test_auto_discover_exact_match_wins(tmp_path, monkeypatch):
    """If an exact-name label exists in the library, prefer it over the
    prefix-stripped fallback."""
    from sed.data.compose import _auto_discover_library

    class FakeLibrary:
        def query(self, category=None, label=None, tags_all=None):
            # Both exist; exact-class-name should win
            if category == "glasses_clean" and label == "clink_wine_thin":
                return [object()]
            if category == "glasses_clean" and label == "wine_thin":
                return [object()]
            return []

    rules = _auto_discover_library(
        FakeLibrary(),
        classes=["_negative", "clink_wine_thin"],
        explicit_rules={},
    )
    assert rules["clink_wine_thin"][0]["label"] == "clink_wine_thin"


def test_explicit_overrides_replace_auto(tmp_path, monkeypatch):
    """[dataset.library] in project.toml fully replaces auto-discovery for
    that class (not append)."""
    from sed.data.compose import _auto_discover_library

    class FakeLibrary:
        def query(self, category=None, label=None, tags_all=None):
            return [object()]  # always matches

    rules = _auto_discover_library(
        FakeLibrary(),
        classes=["clink_wine_thin"],
        explicit_rules={"clink_wine_thin": {"category": "glasses_clean", "label": "custom"}},
    )
    # User's explicit choice should be the only one
    assert rules["clink_wine_thin"] == [
        {"category": "glasses_clean", "label": "custom"}
    ]
