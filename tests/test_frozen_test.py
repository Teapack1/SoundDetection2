"""Tests for the frozen project test set + dataset split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf

from sed.data.compose import _scan_frozen_test, _test_set_hash, SourceClip
from sed.data.splits import train_val_split


@dataclass
class FakeClip:
    session_id: str
    class_name: str
    path: str = "x"


class FakeLayout:
    def __init__(self, root):
        self.test_dir = root / "recordings" / "test"

    def test_positives_dir(self, class_name: str):
        return self.test_dir / "positives" / class_name


def _write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(3200, dtype="float32"), 32000)


def test_train_val_split_no_test_partition():
    clips = []
    for _ in range(200):
        clips.append(FakeClip(session_id="s", class_name="a"))
    for _ in range(100):
        clips.append(FakeClip(session_id="s", class_name="b"))
    train, val, modes = train_val_split(clips, val_frac=0.15, seed=0)
    assert len(train) + len(val) == 300
    # Per-class coverage
    val_classes = {c.class_name for c in val}
    train_classes = {c.class_name for c in train}
    assert {"a", "b"} <= val_classes
    assert {"a", "b"} <= train_classes
    # No overlap between train and val for same path
    train_set = {id(c) for c in train}
    val_set = {id(c) for c in val}
    assert not (train_set & val_set)
    # Validation share roughly matches val_frac (±20%)
    assert 0.10 <= len(val) / 300 <= 0.20
    assert "random" in modes["a"]


def test_train_val_split_tiny_class():
    clips = [FakeClip(session_id="s", class_name="x") for _ in range(2)]
    train, val, _ = train_val_split(clips, val_frac=0.15, seed=0)
    # At most one in val, the other in train; never empty train.
    assert len(train) >= 1
    assert len(train) + len(val) == 2


def _src(path: str, cls: str) -> SourceClip:
    return SourceClip(
        path=path, class_name=cls, session_id=path,
        sample_rate=32000, duration_s=1.0,
        is_target=cls != "_negative", origin="project",
    )


def test_test_set_hash_is_order_independent():
    a = [
        _src("/a/1.wav", "cheers_loud"),
        _src("/a/2.wav", "cheers_quiet"),
        _src("/a/3.wav", "_negative"),
    ]
    b = list(reversed(a))
    assert _test_set_hash(a) == _test_set_hash(b)


def test_test_set_hash_changes_when_clip_added():
    a = [
        _src("/a/1.wav", "cheers_loud"),
        _src("/a/2.wav", "cheers_quiet"),
    ]
    h1 = _test_set_hash(a)
    a.append(_src("/a/3.wav", "_negative"))
    assert _test_set_hash(a) != h1


def test_test_set_hash_changes_when_class_changes():
    a = [_src("/a/1.wav", "cheers_loud")]
    b = [_src("/a/1.wav", "cheers_quiet")]
    assert _test_set_hash(a) != _test_set_hash(b)


def test_frozen_test_positive_library_label_maps_to_target_class(tmp_path):
    layout = FakeLayout(tmp_path)
    _write_wav(
        layout.test_positives_dir("cling_ikea_type1") / "source-label.wav"
    )
    _write_wav(layout.test_positives_dir("cheers_loud") / "target-class.wav")

    clips = _scan_frozen_test(
        layout,
        classes=["_negative", "cheers_loud"],
        positive_rules={
            "cheers_loud": [
                {"category": "positives", "label": "cling_ikea_type1"},
            ],
        },
    )

    assert [c.class_name for c in clips] == ["cheers_loud", "cheers_loud"]
    assert {c.path.rsplit("/", 1)[-1] for c in clips} == {
        "source-label.wav",
        "target-class.wav",
    }


def test_frozen_test_scans_test_augmented_folder(tmp_path):
    layout = FakeLayout(tmp_path)
    _write_wav(layout.test_dir / "augmented" / "cheers_loud" / "new_aug_cheers_loud_0001.wav")
    _write_wav(layout.test_dir / "augmented" / "_negative" / "new_aug_negative_0001.wav")

    clips = _scan_frozen_test(layout, classes=["_negative", "cheers_loud"])

    by_name = {c.path.rsplit("/", 1)[-1]: c for c in clips}
    assert by_name["new_aug_cheers_loud_0001.wav"].class_name == "cheers_loud"
    assert by_name["new_aug_cheers_loud_0001.wav"].is_target is True
    assert by_name["new_aug_cheers_loud_0001.wav"].origin == "augmented"
    assert by_name["new_aug_negative_0001.wav"].class_name == "_negative"
    assert by_name["new_aug_negative_0001.wav"].is_target is False
    assert by_name["new_aug_negative_0001.wav"].origin == "augmented"
