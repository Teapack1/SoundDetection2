"""Reusable audio library, shared across projects.

Layout:

    data_library/
    ├── glasses_clean/<glass_type>/<uuid>.wav
    ├── ambience/<source_label>/<uuid>.wav
    ├── negatives_external/<class>/<uuid>.wav
    └── manifest.jsonl   # one line per clip with metadata

Every clip carries a JSON sidecar with provenance: where it was recorded,
which session, which device, mic position, and tags. Sidecars allow other
projects to filter (e.g. "only thin-walled glasses recorded after 2026").

This module owns:
  - `library_path()` — canonical filesystem path
  - `add_clip()` — register a recording with metadata
  - `query()` — filter clips by tags / class / glass_type / etc.
"""

from __future__ import annotations

import json
import shutil
import uuid as _uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from sed.config import LIBRARY_DIR


@dataclass
class LibraryClip:
    uuid: str
    relpath: str                     # relative to data_library/
    category: str                    # "glasses_clean" | "ambience" | "negatives_external"
    label: str                       # e.g. "wine_thin", "bar_busy", "cutlery"
    duration_s: float
    sample_rate: int
    channels: int
    recorded_at: str                 # ISO8601
    session_id: str = ""
    device: str = ""
    mic_position: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def absolute_path(self, base: Path = LIBRARY_DIR) -> Path:
        return base / self.relpath


class Library:
    def __init__(self, root: Path = LIBRARY_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.touch(exist_ok=True)

    # ---- read --------------------------------------------------------------

    def all_clips(self) -> list[LibraryClip]:
        out: list[LibraryClip] = []
        with open(self.manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(LibraryClip(**json.loads(line)))
        return out

    def query(
        self,
        category: str | None = None,
        label: str | None = None,
        tags_all: list[str] | None = None,
    ) -> list[LibraryClip]:
        clips = self.all_clips()
        if category is not None:
            clips = [c for c in clips if c.category == category]
        if label is not None:
            clips = [c for c in clips if c.label == label]
        if tags_all:
            req = set(tags_all)
            clips = [c for c in clips if req.issubset(set(c.tags))]
        return clips

    # ---- write -------------------------------------------------------------

    def add_clip(
        self,
        source_path: Path,
        category: str,
        label: str,
        sample_rate: int,
        channels: int,
        duration_s: float,
        session_id: str = "",
        device: str = "",
        mic_position: str = "",
        tags: list[str] | None = None,
        notes: str = "",
        move: bool = True,
    ) -> LibraryClip:
        cat_dir = self.root / category / label
        cat_dir.mkdir(parents=True, exist_ok=True)

        clip_uuid = str(_uuid.uuid4())
        target = cat_dir / f"{clip_uuid}.wav"
        if move:
            shutil.move(str(source_path), str(target))
        else:
            shutil.copy2(str(source_path), str(target))

        clip = LibraryClip(
            uuid=clip_uuid,
            relpath=str(target.relative_to(self.root)),
            category=category,
            label=label,
            duration_s=duration_s,
            sample_rate=sample_rate,
            channels=channels,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            session_id=session_id,
            device=device,
            mic_position=mic_position,
            tags=tags or [],
            notes=notes,
        )
        with open(self.manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(clip)) + "\n")
        return clip

    def remove_clip(self, clip_uuid: str) -> bool:
        clips = self.all_clips()
        keep = [c for c in clips if c.uuid != clip_uuid]
        if len(keep) == len(clips):
            return False
        # rewrite manifest
        with open(self.manifest, "w", encoding="utf-8") as f:
            for c in keep:
                f.write(json.dumps(asdict(c)) + "\n")
        # delete file
        for c in clips:
            if c.uuid == clip_uuid:
                p = self.root / c.relpath
                if p.exists():
                    p.unlink()
        return True

    def labels(self, category: str) -> list[str]:
        return sorted({c.label for c in self.all_clips() if c.category == category})

    def stats(self) -> dict:
        clips = self.all_clips()
        out: dict = {"total": len(clips), "categories": {}}
        for c in clips:
            cat = out["categories"].setdefault(c.category, {"count": 0, "labels": {}})
            cat["count"] += 1
            cat["labels"][c.label] = cat["labels"].get(c.label, 0) + 1
        return out
