"""Session-aware train/val/test splitting.

Random clip-level splitting LEAKS: the same recording session ends up in train
and val and inflates val accuracy by 10–20 % F1. We split by session_id
instead, ensuring a session is never in two splits.

If session ids are very few (e.g. all clips from one session), we fall back to
clip-level splitting and emit a warning — but the user should fix this by
recording multiple sessions per class.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Sequence, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


def session_split(
    clips: Sequence,                 # any object with .session_id and .class_name
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    rng = random.Random(seed)

    # Group by session_id; track classes that appear in each session
    session_to_clips: dict[str, list] = defaultdict(list)
    for c in clips:
        session_to_clips[c.session_id].append(c)

    sessions = list(session_to_clips.keys())
    rng.shuffle(sessions)

    # Use per-class session counts to avoid putting all of class X into one split
    # Strategy: assign sessions greedily to whichever split is most "deficient"
    # for any class it contains.
    target_train = 1.0 - val_frac - test_frac
    targets = {"train": target_train, "val": val_frac, "test": test_frac}
    class_counts_total: dict[str, int] = defaultdict(int)
    for c in clips:
        class_counts_total[c.class_name] += 1

    splits = {"train": [], "val": [], "test": []}
    split_class_counts: dict[str, dict[str, int]] = {
        "train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int),
    }

    def deficit(split: str) -> float:
        """How under-quota the split is across classes (lower = more wanted)."""
        d = 0.0
        for cls, total in class_counts_total.items():
            current = split_class_counts[split][cls]
            target = targets[split] * total
            d += max(0.0, target - current)
        return d

    for sess in sessions:
        sess_clips = session_to_clips[sess]
        # pick the split with the largest deficit
        best = max(splits.keys(), key=lambda s: deficit(s))
        splits[best].extend(sess_clips)
        for c in sess_clips:
            split_class_counts[best][c.class_name] += 1

    # warn if any class is missing in val or test
    for split_name in ("val", "test"):
        present = set(split_class_counts[split_name].keys())
        missing = set(class_counts_total.keys()) - present
        if missing:
            log.warning(
                "split %s missing classes %s — consider recording more sessions",
                split_name, sorted(missing),
            )

    return splits["train"], splits["val"], splits["test"]


def _random_clip_split(clips: list, val_frac: float, test_frac: float,
                       rng: random.Random) -> tuple[list, list, list]:
    """Fallback: random clip-level split (no session protection).

    Used for classes that have too few distinct sessions for session-aware
    splitting to produce balanced splits. The trade-off is that two clips
    from the same physical session might end up in train + val — but if you
    only have one session, you don't have that problem to begin with (no
    second session to be in train).
    """
    items = list(clips)
    rng.shuffle(items)
    n = len(items)
    n_test = max(1, int(round(test_frac * n))) if n > 1 else 0
    n_val = max(1, int(round(val_frac * n))) if n > 2 else 0
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    return train, val, test


def session_split_adaptive(
    clips: Sequence,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    min_sessions_for_session_split: int = 3,
    seed: int = 42,
) -> tuple[list, list, list, dict[str, str]]:
    """Per-class split that picks the right strategy automatically.

    For each class:
      - If the class has ≥ `min_sessions_for_session_split` distinct
        session_ids, do session-aware splitting (no leakage).
      - Otherwise, fall back to random clip-level splitting WITHIN that
        class (with a warning). This is the right move for library-heavy
        classes where every clip is independent anyway, and for projects
        with only one or two on-site recording sessions where strict
        session-splitting would put everything in train.

    Returns (train, val, test, modes) where modes is {class_name: "session"|"random"}.
    """
    rng = random.Random(seed)

    by_class: dict[str, list] = defaultdict(list)
    for c in clips:
        by_class[c.class_name].append(c)

    train: list = []
    val: list = []
    test: list = []
    modes: dict[str, str] = {}

    for class_name, class_clips in by_class.items():
        sessions = {c.session_id for c in class_clips}
        if len(sessions) >= min_sessions_for_session_split:
            t, v, te = session_split(class_clips, val_frac=val_frac,
                                     test_frac=test_frac, seed=seed)
            modes[class_name] = f"session ({len(sessions)} sessions)"
        else:
            t, v, te = _random_clip_split(class_clips, val_frac, test_frac, rng)
            modes[class_name] = (
                f"random ({len(sessions)} session{'s' if len(sessions)!=1 else ''} "
                f"— too few for session-aware splitting)"
            )
            if len(sessions) == 1 and len(class_clips) > 50:
                log.warning(
                    "class '%s' has %d clips but only 1 session_id. "
                    "Used random clip-level split as fallback. To improve "
                    "split robustness, record this class across 3+ separate "
                    "sessions (different days / contexts).",
                    class_name, len(class_clips),
                )
        train.extend(t); val.extend(v); test.extend(te)

    return train, val, test, modes
