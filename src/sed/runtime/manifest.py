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

from pathlib import Path


def resolve_model_path(model_dir: Path, prefer: str | None = None) -> Path:
    """Return the OpenVINO IR the runtime should load from a model dir.

    `model_dir` is one deployed model bundle (e.g. models/<run_id>/), resolved
    via sed.config.resolve_active_model_dir. Priority within it:
      1. `active_openvino.xml` (symlink or text-file sentinel)
      2. `model_int8.xml`      (preferred)
      3. `model_fp16.xml`      (fallback)

    `prefer` overrides that priority for testing a specific exported variant
    (e.g. "fp16" to score the un-quantized IR even though INT8 is deployed).
    It bypasses the active-symlink resolution entirely — pass None/"auto" to
    keep the normal deployment behaviour.
    """
    if prefer and prefer != "auto":
        explicit = model_dir / f"model_{prefer}.xml"
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"No model_{prefer}.xml export in {model_dir}.")

    active = model_dir / "active_openvino.xml"
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
    raise FileNotFoundError(
        f"No OpenVINO IR in {model_dir}. Train a model (`sed train`) or copy a "
        f"model bundle into models/, then `sed models select <id>`."
    )


def newest_checkpoint(model_dir: Path) -> Path | None:
    """Return the most recent PyTorch checkpoint in a model bundle, if any."""
    ckpts = sorted(model_dir.glob("checkpoint_*.ckpt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def resolve_runtime_artifact(model_dir: Path, prefer: str | None = None) -> tuple[str, Path]:
    """Resolve the runnable artifact for a model bundle → ``(engine, path)``.

    ``engine`` is one of ``"openvino"``, ``"onnx"``, ``"checkpoint"``. The
    default chain prefers the deployed OpenVINO IR, then the portable ONNX
    export, then the raw PyTorch checkpoint — so every trained run is runnable
    even when it was never exported.

    ``prefer`` pins the choice: ``"int8"``/``"fp16"`` select that OpenVINO
    variant (via resolve_model_path), ``"onnx"``/``"checkpoint"`` select those
    engines, and an explicit prefer never silently falls through — a missing
    artifact raises FileNotFoundError so callers can surface why.
    """
    if prefer in ("int8", "fp16"):
        return "openvino", resolve_model_path(model_dir, prefer=prefer)
    if prefer == "onnx":
        onnx = model_dir / "model.onnx"
        if onnx.exists():
            return "onnx", onnx
        raise FileNotFoundError(f"No model.onnx export in {model_dir}.")
    if prefer == "checkpoint":
        ckpt = newest_checkpoint(model_dir)
        if ckpt is not None:
            return "checkpoint", ckpt
        raise FileNotFoundError(f"No checkpoint_*.ckpt in {model_dir}.")

    tried: list[str] = []
    try:
        return "openvino", resolve_model_path(model_dir)
    except FileNotFoundError:
        tried.append("OpenVINO IR (model_int8/model_fp16.xml)")
    onnx = model_dir / "model.onnx"
    if onnx.exists():
        return "onnx", onnx
    tried.append("ONNX (model.onnx)")
    ckpt = newest_checkpoint(model_dir)
    if ckpt is not None:
        return "checkpoint", ckpt
    tried.append("PyTorch checkpoint (checkpoint_*.ckpt)")
    raise FileNotFoundError(
        f"No runnable artifact in {model_dir}; tried " + ", ".join(tried) + ". "
        "Train a model (`sed train`) or export one (`sed models export`)."
    )
