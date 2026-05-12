"""Recording-group inference: refine imported corpora without creating leakage."""

from __future__ import annotations

from sed.data.sessions import (
    condition_key,
    group_source,
    recording_base,
    refined_group_key,
)
from sed.data.splits import split_group_key


def test_trailing_counter_and_timestamp_are_recognized_conventions():
    assert recording_base("combUniCafe2_0001.wav") == "combUniCafe2"
    assert recording_base("juray_6918.wav") == "juray"
    assert recording_base("attention_smallRoom78_0860.wav") == "attention_smallRoom78"
    # An embedded epoch stamp becomes a coarse time block, so one continuous
    # capture yields several groups instead of one per clip.
    early = recording_base("listen_not1775601566.3468602.wav")
    late = recording_base("listen_not1775604398.2318141.wav")
    assert early is not None and early.startswith("listen_not@")
    assert late is not None and late != early


def test_opaque_filenames_are_not_refined():
    """No recognized convention must mean no subdivision.

    Splitting an opaque name per clip would put clips from one recording on both
    sides of a split, which is the leakage the grouping exists to prevent.
    """
    for name in ("s0c0.wav", "recording.wav", "clip.wav"):
        assert recording_base(name) is None
        assert group_source(f"/lib/sounds/{name}") == "declared"
    assert refined_group_key("/lib/sounds/s0c0.wav", "session_0") == "session_0"


def test_simultaneous_microphone_channels_share_one_group():
    """mic1/mic2/mic3 of the same moment are one recording, never two groups."""
    keys = {
        refined_group_key(f"/lib/venue1/various_02_mic{n}_0299.wav", "venue1")
        for n in (1, 2, 3)
    }
    assert len(keys) == 1


def test_one_import_folder_splits_into_several_groups():
    """A folder of wavs with no session metadata still yields >1 group."""
    paths = [
        "/lib/_cheers/cheers_smallRoom_0001.wav",
        "/lib/_cheers/cheers_smallRoom_0002.wav",
        "/lib/_cheers/cheers_yufa_0428.wav",
        "/lib/_cheers/cheers_bim256_0044.wav",
    ]
    groups = {refined_group_key(p, "_cheers") for p in paths}
    assert len(groups) == 3
    # Clips of the same recording stay together.
    assert refined_group_key(paths[0], "_cheers") == refined_group_key(paths[1], "_cheers")


def test_declared_scope_is_never_widened():
    """Identical filenames under different declared sessions stay separate."""
    a = refined_group_key("/lib/a/cheers_0001.wav", "session_a")
    b = refined_group_key("/lib/b/cheers_0001.wav", "session_b")
    assert a != b


def test_condition_is_coarser_than_and_distinct_from_the_split_group():
    """A condition is the acoustic domain, not the leakage unit.

    Two different recordings in one room share a condition but must land in
    different split groups.
    """
    one = "/lib/_smallRoom/combUniCafe2_0001.wav"
    two = "/lib/_smallRoom/combUniCafe9_0001.wav"
    assert condition_key(one, "_smallRoom") == condition_key(two, "_smallRoom")
    assert refined_group_key(one, "_smallRoom") != refined_group_key(two, "_smallRoom")


def test_split_group_key_refines_rows_and_respects_opaque_names():
    refined = split_group_key({
        "path": "/lib/_smallRoom/combUniCafe2_0001.wav",
        "session_id": "_smallRoom",
        "split_group": "_smallRoom",
    })
    assert refined == "_smallRoom/combUniCafe2"

    # Composition copies the coarse session into split_group for corpora that
    # never had sessions, so that field must not short-circuit refinement...
    assert refined != "_smallRoom"
    # ...but an opaque filename still falls back to the declared group.
    assert split_group_key({
        "path": "/lib/sounds/s0c0.wav",
        "session_id": "session_0",
    }) == "session_0"
