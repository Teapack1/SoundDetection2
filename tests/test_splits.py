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
