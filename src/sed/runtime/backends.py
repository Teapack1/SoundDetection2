"""Inference engine backends for the runtime.

One trained run can carry three runnable artifacts, resolved in preference
order by sed.runtime.manifest.resolve_runtime_artifact:

    OpenVINO IR (model_int8/model_fp16.xml)  ← deployed default, fastest CPU
    ONNX (model.onnx)                        ← portable fallback
    PyTorch checkpoint (checkpoint_*.ckpt)   ← universal — every run has one

Each backend exposes the same contract: ``infer(batch) -> logits (B, C)``.
Callers (RealtimeInference, FP/h calibration) stay engine-agnostic; the
objective-aware probability conversion happens outside, exactly once.

Kept torch/openvino-free at import time so lightweight edge installs can
import the runtime package; heavy imports happen inside the backend that
needs them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sed.audio.features import (
    feature_band_count,
    feature_channels,
    feature_frame_count_for_cfg,
)
from sed.config import Config
from sed.runtime.manifest import resolve_runtime_artifact

log = logging.getLogger(__name__)

__all__ = [
    "CheckpointBackend",
    "OnnxBackend",
    "OpenVINOBackend",
    "build_inference_backend",
    "compile_openvino_model",
    "load_openvino_model",
    "normalize_openvino_logits",
    "openvino_input_shape",
]


# ---------------------------------------------------------------------------
# Shared shape helpers (moved from sed.runtime.infer; re-exported there)
# ---------------------------------------------------------------------------


def openvino_input_shape(cfg: Config, batch_size: int | None = None) -> list[int]:
    """Return the model input shape implied by the frozen inference config.

    The channel dim is the feature's channel count (3 for a colormap image, else
    1) — the same value the export traces — so the shape check matches the IR."""
    batch = -1 if batch_size is None else int(batch_size)
    return [
        batch,
        feature_channels(cfg.features),
        feature_band_count(cfg.features, cfg.model.sample_rate),
        feature_frame_count_for_cfg(cfg),
    ]


def load_openvino_model(core, model_path: Path, cfg: Config, *, batch_size: int | None = None):  # noqa: ANN001
    """Read an OpenVINO IR and optionally reshape it to a fixed runtime batch.

    Runtime inference always knows its batch size: one window per selected
    inference mic. Reshaping to that static batch also lets older EfficientAT
    exports with dynamic-rank Squeeze nodes compile on the CPU plugin.
    """
    model = core.read_model(str(model_path))
    _validate_openvino_input_shape(model.input(0).partial_shape, cfg)
    if batch_size is None:
        return model
    shape = openvino_input_shape(cfg, batch_size=batch_size)
    try:
        model.reshape({model.input(0): shape})
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not reshape OpenVINO model input to {shape}. "
            "Re-export this model bundle or select a model matching the frozen "
            "training_config.json."
        ) from exc
    _validate_openvino_input_shape(model.input(0).partial_shape, cfg)
    return model


def compile_openvino_model(
    core,  # noqa: ANN001
    model_path: Path,
    cfg: Config,
    *,
    batch_size: int | None,
    config: dict[str, str] | None = None,
):  # noqa: ANN001
    """Read, validate, reshape, and compile the deployed OpenVINO model."""
    model = load_openvino_model(core, model_path, cfg, batch_size=batch_size)
    try:
        compiled = core.compile_model(model, "CPU", config=config or {})
    except Exception as exc:  # noqa: BLE001
        shape = openvino_input_shape(cfg, batch_size=batch_size)
        raise RuntimeError(
            f"OpenVINO CPU compile failed for {model_path} with input shape {shape}: {exc}"
        ) from exc
    return model, compiled


def normalize_openvino_logits(
    logits: np.ndarray,
    *,
    batch_size: int,
    num_classes: int,
) -> np.ndarray:
    """Normalize model outputs to ``(batch, classes)``.

    Legacy EfficientAT exports can return ``(1, batch, classes)`` after a static
    reshape because the traced single-batch branch reinserted a singleton dim.
    Shared by every backend — ONNX and checkpoint outputs pass through the same
    check so a class-count drift fails loudly regardless of engine.
    """
    arr = np.asarray(logits)
    expected = (int(batch_size), int(num_classes))
    if arr.shape == expected:
        return arr
    if arr.shape == (1, *expected):
        return arr.reshape(expected)
    if arr.size == expected[0] * expected[1]:
        return arr.reshape(expected)
    raise RuntimeError(
        "Model output shape does not match the selected classes: "
        f"got {arr.shape}, expected {expected}."
    )


def _known_dim_value(dim) -> int | None:  # noqa: ANN001
    """Return an OpenVINO/static shape dimension, or None for dynamic."""
    if dim is None:
        return None
    if isinstance(dim, int):
        return dim
    if getattr(dim, "is_dynamic", False):
        return None
    for attr in ("get_length", "get_min_length"):
        fn = getattr(dim, attr, None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:  # noqa: BLE001
                pass
    try:
        return int(dim)
    except Exception:  # noqa: BLE001
        return None


def _validate_openvino_input_shape(partial_shape, cfg: Config) -> None:  # noqa: ANN001
    """Check the exported model input matches the frozen config snapshot."""
    dims = list(partial_shape)
    if len(dims) != 4:
        raise RuntimeError(
            "Selected model must accept input shape "
            "(batch, channels, feature_bands, n_frames); got "
            f"{partial_shape!s}."
        )
    expected = {
        1: ("channels", feature_channels(cfg.features)),
        2: ("feature bands", feature_band_count(cfg.features, cfg.model.sample_rate)),
        3: (
            "feature frames",
            feature_frame_count_for_cfg(cfg),
        ),
    }
    mismatches: list[str] = []
    for idx, (label, expected_value) in expected.items():
        actual = _known_dim_value(dims[idx])
        if actual is not None and actual != expected_value:
            mismatches.append(f"{label}={actual} (expected {expected_value})")
    if mismatches:
        raise RuntimeError(
            "Selected model input shape does not match the selected "
            "model snapshot: "
            + "; ".join(mismatches)
            + ". Re-export this model bundle or run `sed models select` with "
            "the matching model."
        )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class OpenVINOBackend:
    """Deployed default: compiled OpenVINO IR, latency-tuned for CPU."""

    engine = "openvino"

    def __init__(self, cfg: Config, model_path: Path, *, batch_size: int):
        import openvino as ov

        self.path = Path(model_path)
        self.batch_size = int(batch_size)
        self.num_classes = len(cfg.model.classes)
        self.core = ov.Core()
        self.model, self.compiled = compile_openvino_model(
            self.core,
            self.path,
            cfg,
            batch_size=self.batch_size,
            config={
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_NUM_THREADS": str(cfg.inference.inference_threads),
            },
        )
        self.infer_req = self.compiled.create_infer_request()

    def infer(self, batch: np.ndarray) -> np.ndarray:
        out = self.infer_req.infer({0: batch})
        return normalize_openvino_logits(
            list(out.values())[0],
            batch_size=batch.shape[0],
            num_classes=self.num_classes,
        )


class OnnxBackend:
    """Portable fallback: model.onnx via ONNX Runtime (CPU provider)."""

    engine = "onnx"

    def __init__(self, cfg: Config, model_path: Path, *, batch_size: int):
        import onnxruntime as ort

        self.path = Path(model_path)
        self.batch_size = int(batch_size)
        self.num_classes = len(cfg.model.classes)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(cfg.inference.inference_threads)
        self.session = ort.InferenceSession(
            str(self.path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Symbolic dims come back as strings; validate only the known ones.
        dims = [d if isinstance(d, int) else None for d in inp.shape]
        _validate_openvino_input_shape(dims, cfg)

    def infer(self, batch: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: batch.astype(np.float32)})
        return normalize_openvino_logits(
            out[0], batch_size=batch.shape[0], num_classes=self.num_classes
        )


class CheckpointBackend:
    """Universal fallback: the training checkpoint, eager PyTorch on CPU.

    Model construction params come from the bundle's frozen
    training_config.json (the same snapshot evaluation loads), so the rebuilt
    SedModel always matches the checkpoint — including backbones that cannot
    export (DyMN, transformer encoders).
    """

    engine = "checkpoint"

    def __init__(self, cfg: Config, model_path: Path, *, batch_size: int):
        import torch

        from sed.config import load_config_snapshot
        from sed.train.lit_module import SedLitModule

        self.path = Path(model_path)
        self.batch_size = int(batch_size)
        self.num_classes = len(cfg.model.classes)
        model_dir = self.path.parent
        snapshot = model_dir / "training_config.json"
        run_cfg = load_config_snapshot(snapshot) if snapshot.exists() else cfg
        expected = openvino_input_shape(cfg, batch_size=batch_size)
        got = openvino_input_shape(run_cfg, batch_size=batch_size)
        if got != expected:
            raise RuntimeError(
                "Checkpoint's frozen feature geometry does not match the "
                f"runtime config: checkpoint {got}, runtime {expected}. "
                "Select the matching model (`sed models select`)."
            )
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch.load` with `weights_only=False`.*",
            )
            lit = SedLitModule.load_from_checkpoint(
                str(self.path),
                cfg=run_cfg,
                classes=list(cfg.model.classes),
                map_location="cpu",
            )
        lit.eval()
        torch.set_num_threads(max(1, int(cfg.inference.inference_threads)))
        self._torch = torch
        self._lit = lit

    def infer(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            logits = self._lit(torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)))
        return normalize_openvino_logits(
            logits.cpu().numpy(), batch_size=batch.shape[0], num_classes=self.num_classes
        )


def build_inference_backend(
    cfg: Config,
    model_dir: Path,
    *,
    prefer: str | None = None,
    batch_size: int = 1,
):
    """Resolve a bundle's runnable artifact and construct its backend.

    ``prefer``: None/"auto" walks OpenVINO → ONNX → checkpoint; "int8"/"fp16"
    pin an OpenVINO variant; "onnx"/"checkpoint" pin those engines. An explicit
    prefer never silently falls through to another engine — missing artifacts
    or missing runtimes raise with an actionable message. In auto mode, an ONNX
    artifact is loaded through the OpenVINO ONNX frontend when onnxruntime is
    not installed before dropping to the checkpoint.
    """
    prefer = None if prefer in (None, "", "auto") else str(prefer)
    engine, path = resolve_runtime_artifact(model_dir, prefer=prefer)
    if engine == "openvino":
        return OpenVINOBackend(cfg, path, batch_size=batch_size)
    if engine == "onnx":
        try:
            return OnnxBackend(cfg, path, batch_size=batch_size)
        except ImportError as exc:
            try:
                backend = OpenVINOBackend(cfg, path, batch_size=batch_size)
            except Exception:  # noqa: BLE001
                if prefer == "onnx":
                    raise RuntimeError(
                        "ONNX engine requested but neither onnxruntime nor "
                        "OpenVINO can load model.onnx. Install one: "
                        "`pip install onnxruntime`."
                    ) from exc
                # auto mode: fall through to the checkpoint below
                backend = None
            if backend is not None:
                log.info("onnxruntime not installed — loading %s via OpenVINO", path.name)
                backend.engine = "onnx"  # reported engine reflects the artifact
                return backend
            engine, path = resolve_runtime_artifact(model_dir, prefer="checkpoint")
    try:
        return CheckpointBackend(cfg, path, batch_size=batch_size)
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch checkpoint engine requires torch (`pip install torch`), "
            "or export this run to OpenVINO/ONNX for the lightweight runtime."
        ) from exc
