"""Recording-group and acoustic-condition inference.

The dataset splitter must never let two clips from the *same physical recording*
land on opposite sides of a train/val split, because the model would then be
scored on audio it effectively memorized. The unit that must not straddle a
split is called the **recording group**.

Historically that unit came from ``session_id``, but ``session_id`` carries two
different meanings depending on where a clip came from:

* Clips captured by :mod:`sed.data.recorder` get a genuine per-recording id.
* Clips imported from other corpora (``data_library/pretraining/...``) get the
  *import folder* name, because those datasets were only ever "a directory of
  wavs". A single folder can hold thousands of clips from many different
  recordings, so treating it as one indivisible group leaves a class with a
  handful of enormous groups and makes a representative split impossible.

This module recovers a finer group from filename conventions, which is the only
signal those imported corpora carry. Two rules matter:

* **Splitting** on a differing recording base or a distant timestamp is safe: it
  can only make groups smaller and the split more representative.
* **Merging** simultaneous microphone channels (``mic1``/``mic2``/``mic3`` of the
  same moment) is required: those are the same event, so they must share a group
  or the split leaks.

Inference only ever refines *within* an already-declared group, so an explicit
pin from composition is never widened.

**A recording group is not an acoustic condition.** Two sessions recorded in the
same room with the same microphone are acoustically identical but must still be
split apart, and one long session can wander across very different conditions.
:func:`condition_key` therefore reports the coarse domain (venue/room/import
folder) separately, for reporting and for checking that validation represents
every condition -- it is deliberately *not* the leakage unit.
"""

from __future__ import annotations

import os
import re

# Epoch-like stamps (``...not1775601566.3468602``). Deliberately matched even
# when glued straight onto letters, which is how the recorder names onset clips.
_TIMESTAMP = re.compile(r"(\d{9,})(?:\.(\d+))?")
# Simultaneous capture channels of one moment - merged, never split.
_MIC_TAG = re.compile(r"(?<![a-z])mic\s*[_-]?\d+", re.IGNORECASE)
# Trailing per-clip counter (``..._0292``).
_COUNTER = re.compile(r"[_-]\d{1,6}$")
_SEPARATORS = re.compile(r"[_-]{2,}")

# Clips from one continuous recording are bucketed into blocks of this many
# seconds. A whole night of onset-triggered capture is one folder and one base
# name, so without bucketing it collapses to a single group; bucketing keeps
# adjacent clips together (no leakage) while still yielding several independent
# groups from a long session.
DEFAULT_SESSION_BLOCK_SECONDS = 1800


def recording_base(filename: str,
                   *, block_seconds: int = DEFAULT_SESSION_BLOCK_SECONDS) -> str | None:
    """Recording identity inferred from one filename, or ``None``.

    Strips the per-clip counter, folds simultaneous microphone channels
    together, and replaces an embedded capture timestamp with a coarse time
    block so one long session yields several groups instead of one.

    Returns ``None`` when the filename follows none of these conventions.  That
    case must not be refined: an opaque name carries no evidence that two clips
    are different recordings, so subdividing would put clips from one recording
    on both sides of a split -- the exact leakage the grouping exists to stop.
    """
    stem = os.path.splitext(str(filename or ""))[0]
    block: int | None = None
    recognized = False

    match = _TIMESTAMP.search(stem)
    if match:
        recognized = True
        try:
            block = int(int(match.group(1)) // max(1, int(block_seconds)))
        except (TypeError, ValueError):
            block = None
        stem = _TIMESTAMP.sub("", stem)

    stripped = _COUNTER.sub("", stem)
    if stripped != stem:
        recognized = True
    stem = _MIC_TAG.sub("", stripped)

    if not recognized:
        return None
    stem = _SEPARATORS.sub("_", stem).strip("_-")
    base = stem or "clip"
    return f"{base}@{block}" if block is not None else base


def condition_key(path: str, declared: str = "") -> str:
    """Coarse acoustic domain (venue / room / import folder).

    Used for reporting and for checking that validation represents every
    condition. This is **not** the leakage unit -- see :func:`refined_group_key`.
    """
    declared = str(declared or "").strip()
    if declared:
        return declared
    directory = os.path.dirname(str(path or ""))
    return os.path.basename(directory) or "unknown"


def refined_group_key(path: str, declared: str = "",
                      *, block_seconds: int = DEFAULT_SESSION_BLOCK_SECONDS) -> str:
    """Leakage group: a declared scope refined by filename convention.

    The declared value (recorder ``session_id`` or import folder) always scopes
    the result, so inference can only subdivide an existing group -- never merge
    two declared groups together.  When the filename follows no recognized
    convention the declared group is returned unchanged.
    """
    scope = condition_key(path, declared)
    base = recording_base(os.path.basename(str(path or "")), block_seconds=block_seconds)
    return f"{scope}/{base}" if base else scope


def group_source(path: str, declared: str = "") -> str:
    """``"declared"`` when the filename adds nothing, else ``"inferred"``.

    Lets the UI show whether a class's group count is real metadata or was
    recovered from naming conventions, so the split diagnostics stay auditable.
    """
    _ = declared  # scope only affects the key, not where the evidence came from
    return "inferred" if recording_base(os.path.basename(str(path or ""))) else "declared"


__all__ = [
    "DEFAULT_SESSION_BLOCK_SECONDS",
    "condition_key",
    "group_source",
    "recording_base",
    "refined_group_key",
]
