"""Browsable sub-groups inside a source subbucket.

The cases here are the two real shapes in the library — imported corpus folders
whose only signal is the filename, and recorder captures whose filename is a
timestamp plus a UUID — because an axis that works for one is useless for the
other, and the picker's whole job is to tell them apart.
"""

from pathlib import Path

from sed.data.subgroups import (
    MAX_BROWSABLE_GROUPS,
    build_subgroup_tree,
    default_axis,
    subgroup_key,
    subgroup_keys,
)

ROOT = "/srv/data_library"


def _origin(_wav, meta):
    return str(meta.get("origin_class") or "")


def _corpus(name, folder="_cheers", **meta):
    return (Path(f"{ROOT}/pretraining/{folder}/{name}"), "cheers",
            {"origin_class": "_cheers", **meta})


def test_corpus_clip_groups_by_the_name_prefix_its_folder_lost():
    """An imported corpus is "a directory of wavs" — the name is all there is."""
    keys = subgroup_keys(
        f"{ROOT}/pretraining/_cheers/_champaigne-19-PNY-299/champ_aug_01_0234.wav",
        {}, root=ROOT)
    assert keys["prefix"] == "champ_aug_01"
    assert keys["folder"] == "_cheers/_champaigne-19-PNY-299"
    # No sidecar and no epoch in the name: those axes have nothing to say, and
    # say nothing rather than inventing an "unknown" group.
    assert keys["session"] == ""
    assert keys["date"] == ""


def test_one_long_capture_is_split_into_time_blocks_not_one_group():
    """Epoch-stamped onset clips would otherwise collapse to a single group."""
    early = subgroup_key(f"{ROOT}/pretraining/-cheers/listen_champ1642541907.46.wav",
                         {}, "prefix", root=ROOT)
    later = subgroup_key(f"{ROOT}/pretraining/-cheers/listen_champ1642545907.11.wav",
                         {}, "prefix", root=ROOT)
    assert early.startswith("listen_champ@")
    assert early != later
    # ... but clips seconds apart stay together.
    same = subgroup_key(f"{ROOT}/pretraining/-cheers/listen_champ1642541912.02.wav",
                        {}, "prefix", root=ROOT)
    assert same == early


def test_recorder_clip_groups_by_session_and_folder_never_by_its_uuid_name():
    name = "20260605T141042.662Z_ikea_wine_ch0_b705e1f9_f99f34f4.wav"
    keys = subgroup_keys(
        f"{ROOT}/sounds/ikea_wine_ch0/{name}",
        {"session_id": "20260605T140649Z-63eef7",
         "source_split_group": "b705e1f91c0e4025b80785bb2ad3489f",
         "recorded_at": "20260605T141042.662Z"},
        root=ROOT)
    assert keys["session"] == "20260605T140649Z-63eef7"
    assert keys["folder"] == "ikea_wine_ch0"
    assert keys["date"] == "20260605"
    # The recorder's name carries no recording identity, so refusing to guess is
    # the point: a prefix here would be one group per clip.
    assert keys["prefix"] == ""


def test_folder_key_survives_clips_outside_the_library():
    assert subgroup_key("/elsewhere/take3/clip.wav", {}, "folder", root=ROOT) == "take3"


def test_date_falls_back_to_the_filename_epoch_and_ignores_plain_ids():
    epoch = subgroup_key("/x/listen_cheers1642541907.4671087.wav", {}, "date", root=ROOT)
    assert epoch == "20220118"
    # A long digit run outside any plausible capture window is an id, not a date.
    assert subgroup_key("/x/clip_999999999999.wav", {}, "date", root=ROOT) == ""


def test_tree_drops_axes_that_give_one_group_per_clip():
    """``source_split_group`` is a per-clip UUID for recorder captures.

    Listing 600 one-clip "groups" would be an identifier masquerading as a
    grouping, so that axis must not reach the dropdown at all.
    """
    items = [
        (Path(f"{ROOT}/sounds/ikea_wine_ch{i % 2}/20260605T1410{i:02d}.0Z_w_{i:04d}_ab.wav"),
         "cheers_loud",
         {"origin_class": "ikea_wine", "session_id": f"uuid-{i}"})
        for i in range(120)
    ]
    axes = build_subgroup_tree(items, origin_of=_origin, root=ROOT)["cheers_loud"]["ikea_wine"]
    assert set(axes) == {"folder"}
    assert axes["folder"] == {"ikea_wine_ch0": 60, "ikea_wine_ch1": 60}


def test_tree_keeps_many_small_groups_when_they_are_genuinely_shared():
    """321 time blocks over 1 800 clips is a real grouping, just a fine one."""
    items = [
        _corpus(f"listen_champ{1642541907 + (i // 6) * 2000}.{i}.wav", folder="-cheers")
        for i in range(1800)
    ]
    axes = build_subgroup_tree(items, origin_of=_origin, root=ROOT)["cheers"]["_cheers"]
    assert len(axes["prefix"]) == 300
    assert sum(axes["prefix"].values()) == 1800


def test_tree_omits_a_subbucket_that_nothing_subdivides():
    items = [_corpus(f"juray_{i:04d}.wav", folder="jurayNoise") for i in range(50)]
    assert build_subgroup_tree(items, origin_of=_origin, root=ROOT) == {}


def test_default_axis_prefers_the_finest_grouping_you_can_still_scan():
    # The real _cheers shape: 31 recording types beats 6 folders.
    assert default_axis({"prefix": 31, "folder": 6, "session": 6}) == "prefix"
    # The real ambience_rnd shape: a 661-group prefix is not scannable, so the
    # 4 folders win even though they are coarser.
    assert default_axis({"prefix": 661, "folder": 4, "session": 4}) == "folder"
    # Nothing browsable at all: fall back to the coarsest, which is all that is
    # left to read.
    assert default_axis({"prefix": 321, "session": 900}) == "prefix"
    assert default_axis({"folder": 1, "session": 1}) == ""
    assert default_axis({}) == ""
    assert default_axis({"session": MAX_BROWSABLE_GROUPS}) == "session"
