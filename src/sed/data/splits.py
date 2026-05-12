"""Per-class train/val splitting.

``split_train_val_no_leakage`` is what the training pipeline uses: raw clips
are split per class grouped by recording session, and derived/augmented clips
never enter val. When a class has only one independent session, representing
it on both sides is mathematically incompatible with session isolation; that
case uses a deterministic contiguous holdout and reports the compromise in
the split diagnostics. A derived variant joins train only when its source clip
is not in val. This normally keeps val an honest generalisation check;
otherwise an overlay-augmented copy of a val clip in train (or a near-duplicate
clip from the same session) inflates val metrics, which drive checkpoint
selection and early stopping.

Unlike the frozen test set, val is not a project-level artifact: dataset
composition writes the whole train+val pool to ``train.csv`` (see
sed.data.compose), and ``split_pool_rows`` below re-derives train/val from it
at train time, seeded by ``cfg.training.seed`` — an ordinary, on-the-fly
train/val split like any other training pipeline, recomputed every run
instead of being pinned to whatever the dataset looked like at compose time.

There is no test split here: test.csv is built deterministically from library
source rules whose ``split`` is ``"test"`` (plus an explicitly-built
split-derived slice) by sed.data.compose. When no such clips are selected,
test.csv is empty and post-training evaluation is skipped.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sed.data.sessions import refined_group_key

log = logging.getLogger(__name__)


def _field(obj, name: str, default: str = "") -> str:
    if isinstance(obj, dict):
        return str(obj.get(name) or default)
    return str(getattr(obj, name, default) or default)


def _clip_path(obj) -> str:
    return _field(obj, "path")


def _clip_class(obj) -> str:
    return _field(obj, "class_name")


def _path_stem(obj) -> str:
    path = _clip_path(obj)
    return Path(path).stem if path else ""


def split_group_key(obj) -> str:
    """Leakage group for splitting.

    ``split_group`` lets composition pin a stronger group/cluster id (for
    example multichannel siblings), and an explicit pin always wins outright.

    Otherwise the declared scope (recorder ``session_id``, or the import folder
    for corpora that never had sessions) is *refined* by filename convention --
    see :mod:`sed.data.sessions`. Refinement can only subdivide a declared
    group, never merge two, so an imported folder of thousands of wavs stops
    being one indivisible group while deliberate pins stay intact.
    """
    declared = (
        _field(obj, "source_split_group")
        or _field(obj, "split_group")
        or _field(obj, "session_id")
    )
    path = _clip_path(obj)
    if not path:
        return declared or _path_stem(obj)
    # The declared value scopes the group; the filename refines within it.
    # Composition copies the coarse session into ``split_group``/
    # ``source_split_group`` for corpora that never had sessions, so these are
    # not deliberate pins and must still be refined. The genuine pin -- a
    # multichannel sibling cluster -- survives regardless, because
    # ``recording_base`` folds mic tags together.
    return refined_group_key(path, declared)


def split_bucket_key(obj) -> str:
    """Source bucket identity used for representative split diagnostics.

    Explicit ``split_bucket`` wins. Legacy rows fall back through origin class
    and source-ish labels before using origin/category. This keeps old CSVs
    readable while new composition can write a precise bucket key.
    """
    for name in ("split_bucket", "origin_class", "library_label", "source_label", "label", "origin"):
        value = _field(obj, name)
        if value:
            return value
    return "_unknown"


@dataclass
class _Group:
    key: str
    items: list
    bucket_counts: Counter

    @property
    def size(self) -> int:
        return len(self.items)


def _groups_for_items(items: Sequence) -> dict[str, _Group]:
    raw: dict[str, list] = defaultdict(list)
    for clip in items:
        raw[split_group_key(clip)].append(clip)
    groups: dict[str, _Group] = {}
    for key, members in raw.items():
        groups[key] = _Group(
            key=key,
            items=members,
            bucket_counts=Counter(split_bucket_key(c) for c in members),
        )
    return groups


def _choose_bucketed_val_paths(
    class_name: str,
    items: Sequence,
    *,
    val_frac: float,
    rng: random.Random,
) -> tuple[set[str], str, list[str]]:
    """Choose held-out raw paths while preferring whole independent groups.

    Whole recording groups are the normal unit. When all groups belong to
    different singleton source buckets, holding out a whole bucket is allowed:
    class representation in validation is more important than forcing every
    source bucket to remain in train, and session isolation is still intact.

    A class with only one group cannot be present in both splits without
    session leakage. If it has at least two clips, use one deterministic
    contiguous edge block and report the fallback loudly in diagnostics.
    """
    groups = _groups_for_items(items)
    n = len(items)
    warnings_out: list[str] = []
    if val_frac <= 0.0:
        return set(), f"all-train ({n} clip{'s' if n != 1 else ''})", warnings_out
    if n < 2:
        msg = (
            f"class '{class_name}' has only {n} clip(s); it cannot be represented "
            "in both train and holdout"
        )
        log.warning(msg)
        warnings_out.append(msg)
        return set(), f"all-train ({n} clip{'s' if n != 1 else ''})", warnings_out
    if len(groups) < 2:
        n_val = max(1, min(n - 1, int(round(val_frac * n))))
        ordered = sorted(items, key=lambda clip: (_clip_path(clip), _path_stem(clip)))
        # An edge block creates only one train/holdout boundary among sequential
        # windows. The seeded side choice avoids always reserving the oldest or
        # newest material while staying reproducible.
        picked = ordered[:n_val] if rng.randrange(2) == 0 else ordered[-n_val:]
        msg = (
            f"class '{class_name}' has only {len(groups)} split group(s); "
            f"used a contiguous {n_val}-clip holdout so the class is represented; "
            "the recording session is shared across train and holdout"
        )
        log.warning(msg)
        warnings_out.append(msg)
        return (
            {_clip_path(clip) for clip in picked},
            f"contiguous-session-fallback ({n} clips, groups={len(groups)}, val={n_val})",
            warnings_out,
        )

    bucket_to_groups: dict[str, list[str]] = defaultdict(list)
    bucket_totals: Counter = Counter()
    for key, group in groups.items():
        for bucket, count in group.bucket_counts.items():
            bucket_to_groups[bucket].append(key)
            bucket_totals[bucket] += count

    # First represent buckets that can contribute an independent group while
    # keeping another group from that same bucket in train.
    eligible_buckets = [
        bucket for bucket, keys in sorted(bucket_to_groups.items())
        if len(set(keys)) >= 2 and bucket_totals[bucket] >= 2
    ]
    if not eligible_buckets:
        msg = (
            f"class '{class_name}' has no source bucket with at least two "
            "independent split groups; holding out whole source bucket(s) at "
            "class level so the class is represented without session leakage"
        )
        log.warning(msg)
        warnings_out.append(msg)

    val_keys: set[str] = set()

    def bucket_val_count(bucket: str) -> int:
        return sum(groups[key].bucket_counts.get(bucket, 0) for key in val_keys)

    def can_add_group(key: str, *, preserve_bucket_train: bool = True) -> bool:
        if key in val_keys:
            return False
        if len(val_keys) + 1 >= len(groups):
            return False
        if not preserve_bucket_train:
            return True
        touched = groups[key].bucket_counts.keys()
        for bucket in touched:
            keys = set(bucket_to_groups[bucket])
            if keys and keys <= (val_keys | {key}):
                return False
        return True

    def choose_candidate(bucket: str, target: int) -> str | None:
        candidates = [key for key in set(bucket_to_groups[bucket]) if can_add_group(key)]
        if not candidates:
            return None
        rng.shuffle(candidates)
        current = bucket_val_count(bucket)
        candidates.sort(
            key=lambda key: (
                abs((current + groups[key].bucket_counts.get(bucket, 0)) - target),
                groups[key].size,
                key,
            )
        )
        return candidates[0]

    for bucket in eligible_buckets:
        target = max(1, int(round(val_frac * bucket_totals[bucket])))
        while bucket_val_count(bucket) < target:
            key = choose_candidate(bucket, target)
            if key is None:
                break
            val_keys.add(key)
            # One representative group is enough for tiny buckets; larger
            # buckets keep adding until near their target.
            if bucket_totals[bucket] < 4:
                break
        if bucket_val_count(bucket) == 0:
            msg = (
                f"class '{class_name}' bucket '{bucket}' could not contribute "
                "to val without emptying train for that bucket"
            )
            log.warning(msg)
            warnings_out.append(msg)

    n_val_target = max(1, int(round(val_frac * n)))
    # If bucket representation undershot the class target, top up with whole
    # independent groups even when that means holding out a singleton source
    # bucket. This retains session isolation and guarantees class coverage.
    while sum(groups[key].size for key in val_keys) < n_val_target:
        candidates = [
            key for key in groups
            if can_add_group(key, preserve_bucket_train=False)
        ]
        if not candidates:
            break
        rng.shuffle(candidates)
        current = sum(groups[key].size for key in val_keys)
        candidates.sort(
            key=lambda key: (
                abs((current + groups[key].size) - n_val_target),
                groups[key].size,
                key,
            )
        )
        val_keys.add(candidates[0])

    n_val = sum(groups[key].size for key in val_keys)
    val_paths = {
        _clip_path(clip)
        for key in val_keys
        for clip in groups[key].items
    }
    return (
        val_paths,
        f"bucket-session-grouped class-representative ({len(bucket_to_groups)} buckets, "
        f"{len(groups)} groups, {n} clips, val={n_val})",
        warnings_out,
    )


def _build_split_diagnostics(
    raw_clips: Sequence,
    derived_clips: Sequence,
    train: Sequence,
    val: Sequence,
    *,
    val_frac: float,
    seed: int,
    modes: dict[str, str],
    class_warnings: dict[str, list[str]],
    dropped_derived: int,
    unlinked_derived: int,
) -> dict:
    pool = list(raw_clips) + list(derived_clips)
    classes = sorted({_clip_class(c) for c in pool if _clip_class(c)})
    diag_by_class: dict[str, dict] = {}
    warnings_out: list[str] = []

    def _counts(clips: Sequence) -> dict[str, int]:
        return dict(Counter(split_bucket_key(c) for c in clips))

    for class_name in classes:
        p = [c for c in pool if _clip_class(c) == class_name]
        tr = [c for c in train if _clip_class(c) == class_name]
        va = [c for c in val if _clip_class(c) == class_name]
        buckets: dict[str, dict] = {}
        bucket_names = sorted(
            set(_counts(p)) | set(_counts(tr)) | set(_counts(va))
        )
        for bucket in bucket_names:
            p_b = [c for c in p if split_bucket_key(c) == bucket]
            tr_b = [c for c in tr if split_bucket_key(c) == bucket]
            va_b = [c for c in va if split_bucket_key(c) == bucket]
            row = {
                "pool": len(p_b),
                "train": len(tr_b),
                "val": len(va_b),
                "pool_groups": len({split_group_key(c) for c in p_b}),
                "train_groups": len({split_group_key(c) for c in tr_b}),
                "val_groups": len({split_group_key(c) for c in va_b}),
            }
            buckets[bucket] = row
            if row["pool_groups"] >= 2 and row["train"] > 0 and row["val"] == 0:
                msg = f"class '{class_name}' bucket '{bucket}' has no val samples"
                warnings_out.append(msg)
                class_warnings.setdefault(class_name, []).append(msg)
        train_groups = {split_group_key(c) for c in tr}
        val_groups = {split_group_key(c) for c in va}
        overlap = sorted(train_groups & val_groups)
        if overlap:
            msg = (
                f"class '{class_name}' has {len(overlap)} split group(s) in both train and val"
            )
            warnings_out.append(msg)
            class_warnings.setdefault(class_name, []).append(msg)
        diag_by_class[class_name] = {
            "mode": modes.get(class_name, ""),
            "pool": len(p),
            "train": len(tr),
            "val": len(va),
            "pool_groups": len({split_group_key(c) for c in p}),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "buckets": buckets,
            "warnings": sorted(set(class_warnings.get(class_name, []))),
        }
        warnings_out.extend(class_warnings.get(class_name, []))
    if unlinked_derived:
        warnings_out.append(
            f"{unlinked_derived} derived clip(s) carry no source provenance"
        )
    if dropped_derived:
        warnings_out.append(
            f"{dropped_derived} derived clip(s) dropped because their source was held out"
        )
    return {
        "schema_version": 1,
        "strategy": "bucket-session-grouped",
        "val_fraction": float(val_frac),
        "seed": int(seed),
        "modes": dict(modes),
        "by_class": diag_by_class,
        "dropped_derived": int(dropped_derived),
        "unlinked_derived": int(unlinked_derived),
        "warnings": sorted(set(warnings_out)),
    }


def split_train_val_detailed(
    raw_clips: Sequence,
    derived_clips: Sequence,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, dict]:
    rng = random.Random(seed)

    by_class: dict[str, list] = defaultdict(list)
    for c in raw_clips:
        by_class[_clip_class(c)].append(c)

    train: list = []
    val: list = []
    modes: dict[str, str] = {}
    class_warnings: dict[str, list[str]] = defaultdict(list)

    for class_name in sorted(by_class):
        items = list(by_class[class_name])
        val_paths, mode, warnings_for_class = _choose_bucketed_val_paths(
            class_name, items, val_frac=val_frac, rng=rng,
        )
        modes[class_name] = mode
        class_warnings[class_name].extend(warnings_for_class)
        if not val_paths:
            train.extend(items)
            continue
        for clip in items:
            (val if _clip_path(clip) in val_paths else train).append(clip)

    val_stems = {_path_stem(c) for c in val}
    val_source_groups = {split_group_key(c) for c in val}
    dropped = 0
    unlinked = 0
    for c in derived_clips:
        source_group = _field(c, "source_split_group")
        stem = _field(c, "source_stem")
        if (source_group and source_group in val_source_groups) or (
            stem and stem in val_stems
        ):
            dropped += 1
            continue
        if not source_group and not stem:
            unlinked += 1
        train.append(c)
    if dropped:
        log.info(
            "dropped %d derived clip(s) whose source landed in val "
            "(train/val leakage guard)", dropped,
        )
    if unlinked:
        log.warning(
            "%d derived clip(s) carry no source provenance; kept in train "
            "but their val-leakage risk cannot be verified.", unlinked,
        )

    diagnostics = _build_split_diagnostics(
        raw_clips,
        derived_clips,
        train,
        val,
        val_frac=val_frac,
        seed=seed,
        modes=modes,
        class_warnings=class_warnings,
        dropped_derived=dropped,
        unlinked_derived=unlinked,
    )
    return train, val, diagnostics


def split_train_val_no_leakage(
    raw_clips: Sequence,
    derived_clips: Sequence,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, dict[str, str]]:
    """Per-class train/val split with explicit leakage diagnostics.

    Raw clips are split with whole recording sessions kept on one side whenever
    at least two independent groups exist. Singleton source buckets may be held
    out whole so every class is represented without session leakage. A class
    with only one group and at least two clips uses a deterministic contiguous
    edge holdout, with an explicit warning that the session spans both sides.

    Derived/augmented clips never enter val. A variant joins train only when
    its source clip is NOT in val; variants of val sources are dropped
    entirely. Variants whose source is outside this split (e.g. a library
    base not selected directly) stay in train. Unlinkable variants (no
    ``source`` in the sidecar) also stay in train, with a warning, since
    their leakage risk cannot be checked.
    """
    train, val, diagnostics = split_train_val_detailed(
        raw_clips, derived_clips, val_frac=val_frac, seed=seed,
    )
    return train, val, diagnostics["modes"]


_DERIVED_ORIGINS = {"derived", "augmented"}


class _CsvRow:
    """Duck-typed adapter so ``split_train_val_no_leakage`` can consume a
    plain ``train.csv`` row dict (as read by ``csv.DictReader``) the same way
    it consumes a ``SourceClip``."""

    __slots__ = ("row",)

    def __init__(self, row: dict) -> None:
        self.row = row

    @property
    def class_name(self) -> str:
        return self.row.get("class_name", "")

    @property
    def session_id(self) -> str:
        return self.row.get("session_id", "")

    @property
    def origin(self) -> str:
        return self.row.get("origin", "")

    @property
    def source_stem(self) -> str:
        return self.row.get("source_stem", "")

    @property
    def origin_class(self) -> str:
        return self.row.get("origin_class", "")

    @property
    def split_bucket(self) -> str:
        return self.row.get("split_bucket", "")

    @property
    def split_group(self) -> str:
        return self.row.get("split_group", "")

    @property
    def source_split_group(self) -> str:
        return self.row.get("source_split_group", "")

    @property
    def path(self) -> str:
        return self.row.get("path", "")


def split_pool_rows_detailed(
    rows: Sequence[dict],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], dict]:
    """Split ``train.csv`` row dicts into train/val, live.

    Thin adapter around ``split_train_val_no_leakage`` for callers that only
    have plain CSV rows: the training datamodule (train time, not persisted)
    and pre-flight diagnostics (a preview, also not persisted).
    """
    raw = [_CsvRow(r) for r in rows if r.get("origin") not in _DERIVED_ORIGINS]
    derived = [_CsvRow(r) for r in rows if r.get("origin") in _DERIVED_ORIGINS]
    train, val, diagnostics = split_train_val_detailed(
        raw, derived, val_frac=val_frac, seed=seed,
    )
    return [c.row for c in train], [c.row for c in val], diagnostics


def split_pool_rows(
    rows: Sequence[dict],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    train, val, diagnostics = split_pool_rows_detailed(
        rows, val_frac=val_frac, seed=seed,
    )
    return train, val, diagnostics["modes"]
