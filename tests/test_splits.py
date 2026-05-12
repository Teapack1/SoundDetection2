"""Tests for session-aware splitting — guards against the #1 silent bug."""

from dataclasses import dataclass

from sed.data.splits import session_split


@dataclass
class FakeClip:
    session_id: str
    class_name: str


def test_no_session_appears_in_two_splits():
    clips = []
    for s in range(20):
        for _ in range(50):
            clips.append(FakeClip(session_id=f"session_{s}",
                                  class_name="clink_wine_thin" if s % 2 else "_negative"))
    train, val, test = session_split(clips, val_frac=0.2, test_frac=0.2)

    train_sessions = {c.session_id for c in train}
    val_sessions = {c.session_id for c in val}
    test_sessions = {c.session_id for c in test}
    assert not (train_sessions & val_sessions)
    assert not (train_sessions & test_sessions)
    assert not (val_sessions & test_sessions)


def test_split_proportions():
    clips = []
    for s in range(40):
        for _ in range(20):
            clips.append(FakeClip(session_id=f"s{s}", class_name="x"))
    train, val, test = session_split(clips, val_frac=0.2, test_frac=0.2)
    total = len(train) + len(val) + len(test)
    assert total == len(clips)
    # rough proportions
    assert 0.5 <= len(train) / total <= 0.7
    assert 0.10 <= len(val) / total <= 0.30
    assert 0.10 <= len(test) / total <= 0.30


# ---------------------------------------------------------------------------
# split_train_val_no_leakage — the split dataset composition actually uses
# ---------------------------------------------------------------------------

from sed.data.splits import split_train_val_no_leakage  # noqa: E402


@dataclass
class FakeSourceClip:
    path: str
    class_name: str
    session_id: str
    origin: str = "sounds"
    source_stem: str = ""


def _raw(stem: str, cls: str, session: str) -> FakeSourceClip:
    return FakeSourceClip(path=f"/lib/sounds/{stem}.wav", class_name=cls,
                          session_id=session)


def _derived(stem: str, cls: str, source_stem: str) -> FakeSourceClip:
    return FakeSourceClip(path=f"/lib/derived/{stem}.wav", class_name=cls,
                          session_id=f"aug-of-{source_stem}",
                          origin="derived", source_stem=source_stem)


def test_no_session_straddles_train_and_val():
    raw = [
        _raw(f"s{s}c{i}", "clink", f"session_{s}")
        for s in range(8)
        for i in range(10)
    ]
    train, val, modes = split_train_val_no_leakage(raw, [], val_frac=0.15, seed=0)
    train_sessions = {c.session_id for c in train}
    val_sessions = {c.session_id for c in val}
    assert not (train_sessions & val_sessions)
    assert val and train
    assert "session-grouped" in modes["clink"]


def test_derived_variants_never_enter_val_and_follow_their_source():
    raw = [
        _raw(f"s{s}c{i}", "clink", f"session_{s}")
        for s in range(6)
        for i in range(5)
    ]
    derived = [
        _derived(f"aug_{c.path.rsplit('/', 1)[-1][:-4]}_{k}", "clink",
                 c.path.rsplit("/", 1)[-1][:-4])
        for c in raw
        for k in range(2)
    ]
    train, val, _ = split_train_val_no_leakage(raw, derived, val_frac=0.2, seed=1)

    assert all(c.origin != "derived" for c in val), "val must stay raw-only"
    val_stems = {c.path.rsplit("/", 1)[-1][:-4] for c in val}
    for c in train:
        if c.origin == "derived":
            assert c.source_stem not in val_stems, (
                f"derived clip of val source {c.source_stem} leaked into train"
            )
    # variants of val sources are dropped, not reassigned
    n_dropped = sum(1 for d in derived if d.source_stem in val_stems)
    assert len(train) + len(val) == len(raw) + len(derived) - n_dropped
    assert n_dropped > 0


def test_few_sessions_falls_back_to_per_clip_split():
    raw = [_raw(f"c{i}", "clink", "only_session") for i in range(20)]
    train, val, modes = split_train_val_no_leakage(raw, [], val_frac=0.15, seed=2)
    assert val, "fallback must still produce a val split"
    assert len(train) + len(val) == len(raw)
    assert "per-clip fallback" in modes["clink"]


def test_unlinkable_derived_clips_stay_in_train():
    raw = [_raw(f"s0c{i}", "clink", f"session_{i}") for i in range(6)]
    derived = [_derived("aug_unknown_0", "clink", source_stem="")]
    train, val, _ = split_train_val_no_leakage(raw, derived, val_frac=0.2, seed=3)
    assert any(c.origin == "derived" for c in train)
    assert all(c.origin != "derived" for c in val)
