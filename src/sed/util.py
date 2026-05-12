"""Small dependency-free helpers shared across the data/train/runtime layers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_path(path: Path, *, separator: str = "_") -> Path:
    """Return ``path``, or the first ``<stem><sep><n><suffix>`` that is free.

    Every writer of library audio needs this: recorded clips, imported clips,
    split channels, mined runtime detections, and generated augmentations all
    derive a filename from metadata that can legitimately repeat. ``separator``
    exists because augmentation stems already end in an index, so they use
    ``__2`` to stay unambiguous.
    """
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    n = 2
    while True:
        candidate = path.with_name(f"{stem}{separator}{n}{suffix}")
        if not candidate.exists():
            return candidate
        n += 1


class PhasedProgress:
    """Map per-phase completion onto one monotonic 0..1 bar.

    Long operations (compose, report build) run several phases of very
    different cost, so a naive "phase N of M" bar jumps. Callers declare a
    relative weight per phase; each phase then reports its own 0..1 fraction and
    this scales it into that phase's slice of the whole.

    The bar deliberately stops at 0.99 — completion is signalled by the caller
    finishing, not by a fraction, so a phase that overruns its estimate cannot
    make the UI claim it is done. Reporting never raises: a broken progress sink
    must not take down the operation it is only describing.
    """

    __slots__ = ("_sink", "_starts", "_spans")

    def __init__(
        self,
        weights: dict[str, float] | list[tuple[str, float]],
        sink: Callable[[dict], None] | None,
    ) -> None:
        items = list(weights.items() if isinstance(weights, dict) else weights)
        total = sum(weight for _, weight in items) or 1.0
        self._sink = sink
        self._starts: dict[str, float] = {}
        self._spans: dict[str, float] = {}
        acc = 0.0
        for phase, weight in items:
            span = weight / total
            self._starts[phase] = acc
            self._spans[phase] = span
            acc += span

    def __call__(
        self, phase: str, *, fraction: float = 0.0, detail: str = "", **extra
    ) -> None:
        if self._sink is None:
            return
        start = self._starts.get(phase, 0.0)
        span = self._spans.get(phase, 0.0)
        value = min(0.99, max(0.0, start + span * min(1.0, max(0.0, fraction))))
        try:
            self._sink({
                "phase": phase,
                "phase_detail": detail or phase,
                "progress": value,
                **extra,
            })
        except Exception:  # noqa: BLE001 — a progress sink must never break its caller
            pass
