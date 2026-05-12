"""Deployment artefact resolution for the runtime.

The training_config.json snapshot (written by `sed train` into the model
bundle at `models/<model_id>/`) is the single source of truth for feature
params, classes, and the inference baseline. There is no separate export manifest.

Active OpenVINO IR is resolved through a tiny convention within a model dir:

    active_openvino.xml   ← symlink/sentinel to model_int8.xml or model_fp16.xml
    model_int8.xml/.bin   ← INT8 IR (preferred)
    model_fp16.xml/.bin   ← FP16 IR (fallback)

Kept openvino-free so non-runtime modules (calibration, doctor, tests) can
import it without dragging in the inference runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sed.config import (
    Config,
    ProjectLayout,
    apply_runtime_settings,
    load_config_snapshot,
    load_runtime_settings,
    resolve_active_model_dir,
)

log = logging.getLogger(__name__)


def load_runtime_config(layout: ProjectLayout, active_model: str = "") -> Config:
    """Load the active model's deployment Config from its training_config.json.

    The runtime reads everything (classes, feature params, inference baseline)
    from this. Editing project config.toml after training does not change the
    behaviour of an already-deployed model.
    """
    model_dir = resolve_active_model_dir(layout, active_model)
    cfg = load_config_snapshot(model_dir / "training_config.json")
    return apply_runtime_settings(
        cfg, load_runtime_settings(model_dir / "runtime_settings.json")
    )


def resolve_model_path(model_dir: Path) -> Path:
    """Return the OpenVINO IR the runtime should load from a model dir.

    `model_dir` is one deployed model bundle (e.g. models/<run_id>/), resolved
    via sed.config.resolve_active_model_dir. Priority within it:
      1. `active_openvino.xml` (symlink or text-file sentinel)
      2. `model_int8.xml`      (preferred)
      3. `model_fp16.xml`      (fallback)
    """
    for active_name in ("active_openvino.xml", "active_model.xml"):
        active = model_dir / active_name
        if active.is_symlink():
            target = active.resolve()
            if target.exists():
                return target
        elif active.exists():
            # text-file sentinel on filesystems without symlinks
            try:
                rel = active.read_text().strip()
                target = (model_dir / rel).resolve()
                if target.exists():
                    return target
            except Exception:  # noqa: BLE001
                pass

    int8 = model_dir / "model_int8.xml"
    if int8.exists():
        return int8
    fp16 = model_dir / "model_fp16.xml"
    if fp16.exists():
        return fp16
    fp32 = model_dir / "model_fp32.xml"
    if fp32.exists():
        return fp32
    raise FileNotFoundError(
        f"No OpenVINO IR in {model_dir}. Train a model (`sed train`) or copy a "
        f"model bundle into models/, then `sed models select <id>`."
    )
