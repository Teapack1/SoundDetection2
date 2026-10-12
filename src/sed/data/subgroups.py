"""Browsable sub-groups inside one source subbucket.

A source subbucket (``origin_class``) is often the whole of an imported corpus
folder: ``_cheers`` alone holds 13 106 clips. Reviewing it means reaching *part*
of it -- one recording session, one recording type -- and the review tabs cap
their card lists, so that has to be a real filter rather than a scroll.

There is no single field that carves every subbucket into useful pieces, because
the two kinds of clip in the library carry their provenance differently:

* **Imported corpora** were only ever "a directory of wavs". Their one real
  signal is the filename (``champ_aug_01_0001.wav``), so the *name prefix* axis
  is the useful one and ``session_id`` degenerates to the import folder.
* **Recorder captures** carry genuine per-clip metadata, so ``session_id`` and
  the on-disk folder are meaningful while the filename is a timestamp plus a
  UUID -- a name-prefix grouping there gives one group per clip.

So every axis is computed, useless ones are dropped (see :func:`build_subgroup_tree`),
and :func:`default_axis` picks the most informative survivor. The caller keeps
the choice visible and overridable rather than guessing silently.

Everything here is pure string work over metadata the caller already holds: no
audio decode, no ``stat``, and deliberately no :meth:`Path.resolve` -- one
resolve per clip costs seconds on a 75 k-clip corpus (see
``compose.py::_append_clip_once``).
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from pathlib import Path

from sed.data.sessions import recording_base

# Display order, and the tie-break order when two axes score the same.
AXES: tuple[str, ...] = ("prefix", "folder", "session", "date")

AXIS_LABELS: dict[str, str] = {
    "prefix": "by name prefix",
    "folder": "by folder",
    "session": "by recording session",
    "date": "by date",
}

# Above this many groups an axis stops being something you can browse in a
# dropdown, so it is no longer eligible as the default (it stays selectable).
MAX_BROWSABLE_GROUPS = 80
# Hard ceiling on what one (subbucket, axis) may contribute to the payload.
MAX_AXIS_GROUPS = 1000

# Category directories under ``data_library/``. Stripped from folder keys so the
# axis reads as a location inside the library rather than a storage detail.
_CATEGORY_DIRS = frozenset({"sounds", "mined", "derived", "pretraining", "streams"})

# Epoch-like stamp in a filename, matched even when glued onto letters -- the
# same convention ``sessions.recording_base`` recognises.
_EPOCH = re.compile(r"(\d{9,})")
# Plausible capture epochs (2001-09-09 .. 2096-09-08). Keeps a long digit run
# that happens to be an id from being read as a date.
_EPOCH_MIN, _EPOCH_MAX = 1_000_000_000, 4_000_000_000


def library_root() -> str:
    """Normalised ``data_library`` path, resolved once for folder keys."""
    from sed.config import LIBRARY_DIR

    return os.path.normpath(str(LIBRARY_DIR))


@lru_cache(maxsize=None)
def _folder_of(directory: str, root: str) -> str:
    """Cached per directory: a 75 k-clip corpus lives in tens of folders."""
    if not directory:
        return ""
    if root and (directory == root or directory.startswith(root + os.sep)):
        relative = directory[len(root):].strip(os.sep)
        parts = relative.split(os.sep) if relative else []
        if parts and parts[0] in _CATEGORY_DIRS:
            parts = parts[1:]
        return "/".join(parts)
    return os.path.basename(directory)


def _folder_key(path: str, root: str) -> str:
    """Directory of ``path`` relative to the library, minus the category dir.

    Falls back to the immediate parent's name when the clip lives outside the
    library, which keeps the axis meaningful for ad-hoc and test layouts.
    """
    return _folder_of(os.path.dirname(os.path.normpath(path)), root)


def _date_key(filename: str, meta: Mapping) -> str:
    """``YYYYMMDD`` from recorded metadata, else from an epoch in the name."""
    digits = re.sub(r"\D", "", str(meta.get("recorded_at") or "")[:12])
    if len(digits) >= 8:
        return digits[:8]
    match = _EPOCH.search(filename)
    if match:
        try:
            epoch = int(match.group(1))
        except ValueError:
            return ""
        if _EPOCH_MIN <= epoch <= _EPOCH_MAX:
            return time.strftime("%Y%m%d", time.gmtime(epoch))
    return ""


def _session_key(meta: Mapping) -> str:
    return str(
        meta.get("session_id")
        or meta.get("source_split_group")
        or meta.get("split_group")
        or meta.get("test_session_id")
        or ""
    ).strip()


def subgroup_key(path: str | Path, meta: Mapping, axis: str, *, root: str = "") -> str:
    """This clip's key on one axis; ``""`` means "no value on this axis".

    A clip with no value is left out of that axis entirely rather than pooled
    into a synthetic "unknown" group, so selecting a group never shows clips
    that only happen to share a missing field.
    """
    text = str(path)
    if axis == "prefix":
        return recording_base(os.path.basename(text)) or ""
    if axis == "folder":
        return _folder_key(text, root)
    if axis == "session":
        return _session_key(meta)
    if axis == "date":
        return _date_key(os.path.basename(text), meta)
    return ""


def subgroup_keys(path: str | Path, meta: Mapping, *, root: str = "") -> dict[str, str]:
    """This clip's key on every axis. See :func:`subgroup_key`.

    Splits the path once and shares it across the axes: this runs per clip over
    a 75 k-clip corpus, where re-deriving the basename four times is measurable.
    """
    text = str(path)
    filename = os.path.basename(text)
    return {
        "prefix": recording_base(filename) or "",
        "folder": _folder_key(text, root),
        "session": _session_key(meta),
        "date": _date_key(filename, meta),
    }


def _keep_axis(groups: int, clips: int) -> bool:
    """Whether an axis carves this subbucket into something worth listing.

    Rejects the two degenerate shapes: more groups than a payload should carry,
    and "one group per clip" -- an axis where almost every clip is its own group
    is an identifier, not a grouping (recorder ``source_split_group`` is a UUID
    per clip). Small subbuckets are exempt from the ratio rule, where a handful
    of one- or two-clip groups is still perfectly browsable.
    """
    if groups <= 1 or groups > MAX_AXIS_GROUPS:
        return False
    return groups <= MAX_BROWSABLE_GROUPS or groups * 2 <= clips


def build_subgroup_tree(
    items: Iterable[tuple[Path | str, str, Mapping]],
    *,
    origin_of: Callable[[Path, Mapping], str],
    root: str | None = None,
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """``panel_key -> subbucket -> axis -> {group: clip count}``.

    ``items`` are the ``(path, panel_key, meta)`` tuples the review tabs already
    gather, and ``origin_of`` is the caller's subbucket rule, so this stays the
    same inventory the summary panel counts. ``panel_key`` is whatever that tab
    renders class blocks under -- the target class everywhere except Augmented,
    which groups by augmentation recipe.
    """
    resolved_root = library_root() if root is None else root
    raw: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    clip_counts: dict[tuple[str, str], int] = {}
    for path, panel_key, meta in items:
        key = str(panel_key).strip()
        if not key:
            continue
        wav = path if isinstance(path, Path) else Path(str(path))
        origin = str(origin_of(wav, meta) or "").strip()
        if not origin:
            continue
        clip_counts[(key, origin)] = clip_counts.get((key, origin), 0) + 1
        by_axis = raw.setdefault(key, {}).setdefault(origin, {})
        for axis, value in subgroup_keys(wav, meta, root=resolved_root).items():
            if not value:
                continue
            groups = by_axis.setdefault(axis, {})
            groups[value] = groups.get(value, 0) + 1

    tree: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for key in sorted(raw):
        for origin in sorted(raw[key]):
            by_axis = raw[key][origin]
            clips = clip_counts[(key, origin)]
            kept = {
                axis: dict(sorted(groups.items()))
                for axis in AXES
                if (groups := by_axis.get(axis)) and _keep_axis(len(groups), clips)
            }
            if kept:
                tree.setdefault(key, {})[origin] = kept
    return tree


def default_axis(axis_counts: Mapping[str, int]) -> str:
    """The axis to preselect, or ``""`` when none subdivides the subbucket.

    Prefers the finest grouping that is still browsable in a dropdown; when
    every axis overshoots that, falls back to the coarsest one, which is the
    only thing left that a reader can actually scan.
    """
    usable = {
        axis: int(count)
        for axis in AXES
        if (count := int(axis_counts.get(axis, 0) or 0)) > 1
    }
    if not usable:
        return ""
    browsable = {a: n for a, n in usable.items() if n <= MAX_BROWSABLE_GROUPS}
    pool = browsable or usable
    best = max(pool.values()) if browsable else min(pool.values())
    return next(axis for axis in AXES if pool.get(axis) == best)


__all__ = [
    "AXES",
    "AXIS_LABELS",
    "MAX_AXIS_GROUPS",
    "MAX_BROWSABLE_GROUPS",
    "build_subgroup_tree",
    "default_axis",
    "library_root",
    "subgroup_key",
    "subgroup_keys",
]
