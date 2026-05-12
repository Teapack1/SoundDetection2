"""Tests for the frozen project test set + dataset split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf

from sed.data.compose import _scan_frozen_test, _test_set_hash, SourceClip


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


def _sess_src(path: str, cls: str, session: str) -> SourceClip:
    return SourceClip(
        path=path, class_name=cls, session_id=session,
        sample_rate=32000, duration_s=1.0,
        is_target=cls != "_negative", origin="pretraining",
    )


class FakeCarveLayout:
    def __init__(self, root):
        self.dataset_dir = root / "dataset"


def test_pretraining_frozen_test_carve_is_session_grouped(tmp_path):
    from sed.data.compose import _resolve_pretraining_frozen_test

    by_class = {"cheers": [
        _sess_src(f"/lib/s{s}/c{i}.wav", "cheers", f"session_{s}")
        for s in range(8) for i in range(10)
    ]}
    layout = FakeCarveLayout(tmp_path)

    test_clips, remaining = _resolve_pretraining_frozen_test(
        layout, "proj", by_class, ["cheers"], 0.15,
    )

    assert test_clips, "carve must hold out material"
    test_sessions = {c.session_id for c in test_clips}
    rest_sessions = {c.session_id for c in remaining["cheers"]}
    assert not (test_sessions & rest_sessions), (
        "test sessions must never feed train/val"
    )
    # frozen: the same clips come back on the next compose
    test2, _ = _resolve_pretraining_frozen_test(
        layout, "proj", by_class, ["cheers"], 0.15,
    )
    assert {c.path for c in test2} == {c.path for c in test_clips}


def test_legacy_random_carve_triggers_session_leak_warning(tmp_path, caplog):
    import json

    from sed.data.compose import _resolve_pretraining_frozen_test

    by_class = {"cheers": [
        _sess_src(str(tmp_path / f"s{s}/c{i}.wav"), "cheers", f"session_{s}")
        for s in range(4) for i in range(10)
    ]}
    layout = FakeCarveLayout(tmp_path)
    layout.dataset_dir.mkdir(parents=True)
    # Legacy schema-1 carve: one random clip out of each session.
    keys = [str(tmp_path / f"s{s}/c0.wav") for s in range(4)]
    (layout.dataset_dir / "pretraining_test.json").write_text(
        json.dumps({"schema_version": 1, "fraction": 0.1, "keys": keys,
                    "by_class": {"cheers": keys}}),
        encoding="utf-8",
    )

    caplog.set_level("WARNING", logger="sed.data.compose")
    test_clips, _ = _resolve_pretraining_frozen_test(
        layout, "proj", by_class, ["cheers"], 0.1,
    )

    assert len(test_clips) == 4
    assert "shares recording sessions" in caplog.text


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
