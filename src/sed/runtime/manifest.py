"""Manifest IO for trained-model artefacts.

Kept openvino-free so non-runtime modules (calibration, doctor, tests) can
import it without dragging in the inference runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

from sed.config import ProjectLayout


def load_manifest(layout: ProjectLayout) -> dict:
    with open(layout.models_dir / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_path(layout: ProjectLayout, manifest: dict) -> Path:
    """Use 'current.openvino' if present, else manifest.model_int8."""
    cur = layout.models_dir / "current.openvino"
    if cur.exists() or cur.is_symlink():
        if cur.is_symlink():
            target = cur.resolve()
        else:
            # text file fallback (filesystems without symlink support)
            try:
                rel = cur.read_text().strip()
                target = (layout.models_dir / rel).resolve()
            except Exception:  # noqa: BLE001
                target = layout.models_dir / manifest["model_int8"]
        if target.exists():
            return target
    return layout.models_dir / manifest["model_int8"]
