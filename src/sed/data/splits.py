"""Per-class train/val splitting.

``split_train_val_no_leakage`` is what dataset composition uses: raw clips
are split per class grouped by recording session, and derived/augmented
clips never enter val — a variant joins train only when its source clip is
not in val. This keeps val an honest generalisation check; otherwise an
overlay-augmented copy of a val clip in train (or a near-duplicate clip from
the same session) inflates val metrics, which drive checkpoint selection and
early stopping.

``train_val_split`` is the older per-class random split kept for callers
that operate on a flat clip list without provenance.

There is no test split here: test.csv is built deterministically from library
sidecars marked ``dataset_role="test"`` by sed.data.compose. When no such
clips are selected, test.csv is empty and post-training evaluation is skipped.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)


def train_val_split(
    clips: Sequence,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, dict[str, str]]:
    """Per-class random split into train + val only."""
    rng = random.Random(seed)

    by_class: dict[str, list] = defaultdict(list)
    for c in clips:
        by_class[c.class_name].append(c)

    train: list = []
    val: list = []
    modes: dict[str, str] = {}

    for class_name, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, int(round(val_frac * n))) if n >= 2 else 0
        if n_val >= n:
            n_val = max(0, n - 1)
        val_part = items[:n_val]
        train_part = items[n_val:]
        train.extend(train_part)
        val.extend(val_part)
        modes[class_name] = f"random ({n} clip{'s' if n != 1 else ''})"

    return train, val, modes


def split_train_val_no_leakage(
    raw_clips: Sequence,
    derived_clips: Sequence,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, dict[str, str]]:
    """Leakage-free per-class train/val split.

    Raw clips are split with whole recording sessions kept on one side, so
    near-duplicate clips captured seconds apart never straddle train and val.
    Classes with fewer than 3 sessions fall back to per-clip groups (with a
    warning) so val still gets material.

    Derived/augmented clips never enter val. A variant joins train only when
    its source clip is NOT in val; variants of val sources are dropped
    entirely. Variants whose source is outside this split (e.g. a library
    base not selected directly) stay in train. Unlinkable variants (no
    ``source`` in the sidecar) also stay in train, with a warning, since
    their leakage risk cannot be checked.
    """
    rng = random.Random(seed)

    by_class: dict[str, list] = defaultdict(list)
    for c in raw_clips:
        by_class[c.class_name].append(c)

    train: list = []
    val: list = []
    modes: dict[str, str] = {}

    for class_name in sorted(by_class):
        items = list(by_class[class_name])
        n = len(items)
        if n < 2:
            train.extend(items)
            modes[class_name] = f"all-train ({n} clip{'s' if n != 1 else ''})"
            continue

        groups: dict[str, list] = defaultdict(list)
        for c in items:
            groups[c.session_id or Path(c.path).stem].append(c)
        grouped = len(groups) >= 3
        if not grouped:
            log.warning(
                "class '%s' has only %d recording session(s); falling back to "
                "a per-clip split — clips from one session may land in both "
                "train and val. Record more sessions for an honest split.",
                class_name, len(groups),
            )
            groups = {f"clip-{i}": [c] for i, c in enumerate(items)}

        group_keys = sorted(groups)
        rng.shuffle(group_keys)
        n_val_target = max(1, int(round(val_frac * n)))
        val_keys: list[str] = []
        n_val = 0
        for key in group_keys:
            if n_val >= n_val_target:
                break
            if len(val_keys) + 1 >= len(group_keys):
                break                       # always keep >= 1 group for train
            val_keys.append(key)
            n_val += len(groups[key])
        val_key_set = set(val_keys)
        for key in group_keys:
            (val if key in val_key_set else train).extend(groups[key])
        modes[class_name] = (
            f"session-grouped ({len(groups)} sessions, {n} clips, val={n_val})"
            if grouped
            else f"per-clip fallback ({n} clips, val={n_val})"
        )

    val_stems = {Path(c.path).stem for c in val}
    dropped = 0
    unlinked = 0
    for c in derived_clips:
        stem = getattr(c, "source_stem", "") or ""
        if stem and stem in val_stems:
            dropped += 1
            continue
        if not stem:
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
    return train, val, modes


def session_split(
    clips: Sequence,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Legacy session-aware train/val/test split helper.

    Runtime composition now gets test clips from library sidecars, but older
    callers/tests still import this helper. It keeps whole sessions in exactly
    one split to avoid leakage.
    """
    rng = random.Random(seed)
    by_session: dict[str, list] = defaultdict(list)
    for clip in clips:
        by_session[getattr(clip, "session_id", "")].append(clip)

    sessions = list(by_session)
    rng.shuffle(sessions)
    n_sessions = len(sessions)
    if n_sessions == 0:
        return [], [], []

    n_test = int(round(test_frac * n_sessions))
    n_val = int(round(val_frac * n_sessions))
    if test_frac > 0 and n_sessions >= 3:
        n_test = max(1, n_test)
    if val_frac > 0 and n_sessions >= 3:
        n_val = max(1, n_val)
    if n_test + n_val >= n_sessions:
        overflow = n_test + n_val - n_sessions + 1
        n_test = max(0, n_test - overflow)

    test_sessions = set(sessions[:n_test])
    val_sessions = set(sessions[n_test:n_test + n_val])

    train: list = []
    val: list = []
    test: list = []
    for session in sessions:
        target = (
            test if session in test_sessions
            else val if session in val_sessions
            else train
        )
        target.extend(by_session[session])
    return train, val, test
