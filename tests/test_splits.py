"""Tests for split_train_val_no_leakage — the split dataset composition uses."""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# split_train_val_no_leakage — the split dataset composition actually uses
# ---------------------------------------------------------------------------

from sed.data.splits import split_train_val_no_leakage


@dataclass
class FakeSourceClip:
    path: str
    class_name: str
    session_id: str
    origin: str = "sounds"
    source_stem: str = ""
    origin_class: str = ""
    split_bucket: str = ""
    split_group: str = ""
    source_split_group: str = ""


def _raw(stem: str, cls: str, session: str) -> FakeSourceClip:
    return FakeSourceClip(path=f"/lib/sounds/{stem}.wav", class_name=cls,
                          session_id=session, origin_class=cls,
                          split_bucket=f"sounds:{cls}", split_group=session)


def _bucketed(stem: str, cls: str, bucket: str, session: str) -> FakeSourceClip:
    return FakeSourceClip(path=f"/lib/sounds/{bucket}/{stem}.wav", class_name=cls,
                          session_id=session, origin_class=bucket,
                          split_bucket=f"sounds:{bucket}", split_group=session)


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


def test_derived_variants_follow_source_group_when_filename_provenance_changed():
    raw = [
        _raw(f"s{s}c{i}", "clink", f"session_{s}")
        for s in range(6)
        for i in range(5)
    ]
    derived = [
        FakeSourceClip(
            path=f"/lib/derived/opaque-{s}-{i}.wav",
            class_name="clink",
            session_id=f"derived-{s}-{i}",
            origin="derived",
            source_stem="renamed-source",
            source_split_group=f"session_{s}",
        )
        for s in range(6)
        for i in range(2)
    ]

    train, val, _ = split_train_val_no_leakage(
        raw, derived, val_frac=0.2, seed=1,
    )
    val_groups = {c.source_split_group or c.split_group for c in val}

    assert val_groups
    assert all(
        c.source_split_group not in val_groups
        for c in train
        if c.origin == "derived"
    )
    assert len(train) + len(val) < len(raw) + len(derived)


def test_single_session_uses_contiguous_fallback_to_represent_class():
    raw = [_raw(f"c{i:03d}", "clink", "only_session") for i in range(20)]
    train, val, modes = split_train_val_no_leakage(raw, [], val_frac=0.15, seed=2)
    assert len(val) == 3
    assert train
    assert len(train) + len(val) == len(raw)
    assert {c.session_id for c in train} == {"only_session"}
    assert {c.session_id for c in val} == {"only_session"}
    assert "contiguous-session-fallback" in modes["clink"]


def test_singleton_buckets_hold_out_whole_group_without_session_leakage():
    raw = [
        _bucketed(f"source_{s}_c{i}", "_negative", f"source_{s}", f"session_{s}")
        for s in range(6)
        for i in range(4)
    ]
    train, val, modes = split_train_val_no_leakage(raw, [], val_frac=0.25, seed=3)

    assert train and val
    assert not ({c.session_id for c in train} & {c.session_id for c in val})
    assert {c.split_bucket for c in train}.isdisjoint({c.split_bucket for c in val})
    assert "class-representative" in modes["_negative"]


def test_sequential_test_then_val_carves_keep_single_group_class_in_all_splits():
    raw = [_raw(f"c{i:03d}", "attention", "only_session") for i in range(20)]

    remaining, test, test_modes = split_train_val_no_leakage(
        raw, [], val_frac=0.20, seed=7,
    )
    train, val, val_modes = split_train_val_no_leakage(
        remaining, [], val_frac=0.25, seed=42,
    )

    assert len(test) == 4
    assert len(val) == 4
    assert len(train) == 12
    assert "contiguous-session-fallback" in test_modes["attention"]
    assert "contiguous-session-fallback" in val_modes["attention"]


def test_bucketed_split_represents_every_sufficient_bucket():
    raw = [
        _bucketed(f"{bucket}_s{s}_c{i}", "cheers", bucket, f"{bucket}_session_{s}")
        for bucket in ("bar", "concert", "stadium")
        for s in range(4)
        for i in range(2)
    ]
    train, val, modes = split_train_val_no_leakage(raw, [], val_frac=0.25, seed=2)

    assert "bucket-session-grouped" in modes["cheers"]
    train_buckets = {c.split_bucket for c in train}
    val_buckets = {c.split_bucket for c in val}
    assert val_buckets == {
        "sounds:bar",
        "sounds:concert",
        "sounds:stadium",
    }
    assert val_buckets <= train_buckets


def test_sparse_bucket_warns_by_staying_train_only():
    raw = [
        _bucketed(f"wide_s{s}_c{i}", "cheers", "wide", f"wide_session_{s}")
        for s in range(4)
        for i in range(2)
    ]
    raw += [
        _bucketed(f"tiny_c{i}", "cheers", "tiny", "tiny_only_session")
        for i in range(5)
    ]
    train, val, _ = split_train_val_no_leakage(raw, [], val_frac=0.25, seed=4)

    assert {c.split_bucket for c in val} == {"sounds:wide"}
    assert sum(1 for c in train if c.split_bucket == "sounds:tiny") == 5


def test_giant_session_never_swallows_val():
    # One session holds 70% of the class; it must stay in train, with val
    # assembled from the smaller sessions near the 15% target.
    raw = [_raw(f"giant{i}", "clink", "giant_session") for i in range(700)]
    for s in range(6):
        raw += [_raw(f"s{s}c{i}", "clink", f"small_{s}") for i in range(50)]
    train, val, modes = split_train_val_no_leakage(raw, [], val_frac=0.15, seed=0)
    val_sessions = {c.session_id for c in val}
    assert "giant_session" not in val_sessions
    assert val, "val must not be empty"
    frac = len(val) / (len(train) + len(val))
    assert 0.05 <= frac <= 0.25, f"val fraction {frac:.2%} strayed from target"
    assert "session-grouped" in modes["clink"]


def test_all_sessions_oversized_takes_smallest_into_val():
    raw = [_raw(f"a{i}", "clink", "big_a") for i in range(100)]
    raw += [_raw(f"b{i}", "clink", "big_b") for i in range(90)]
    raw += [_raw(f"c{i}", "clink", "big_c") for i in range(80)]
    train, val, _ = split_train_val_no_leakage(raw, [], val_frac=0.15, seed=1)
    val_sessions = {c.session_id for c in val}
    assert val_sessions == {"big_c"}, "smallest oversized session should back-fill val"
    assert {c.session_id for c in train} == {"big_a", "big_b"}


def test_unlinkable_derived_clips_stay_in_train():
    raw = [_raw(f"s0c{i}", "clink", f"session_{i}") for i in range(6)]
    derived = [_derived("aug_unknown_0", "clink", source_stem="")]
    train, val, _ = split_train_val_no_leakage(raw, derived, val_frac=0.2, seed=3)
    assert any(c.origin == "derived" for c in train)
    assert all(c.origin != "derived" for c in val)
