"""Tests for runtime artifact resolution and the engine preference chain."""

from __future__ import annotations

import time

import pytest

from sed.runtime.manifest import (
    newest_checkpoint,
    resolve_model_path,
    resolve_runtime_artifact,
)


def _seed(model_dir, *, int8=False, fp16=False, onnx=False, ckpts=()):
    model_dir.mkdir(parents=True, exist_ok=True)
    if int8:
        (model_dir / "model_int8.xml").write_text("ir", encoding="utf-8")
        (model_dir / "model_int8.bin").write_bytes(b"b")
    if fp16:
        (model_dir / "model_fp16.xml").write_text("ir", encoding="utf-8")
        (model_dir / "model_fp16.bin").write_bytes(b"b")
    if onnx:
        (model_dir / "model.onnx").write_bytes(b"onnx")
    for name in ckpts:
        (model_dir / name).write_bytes(b"ckpt")
    return model_dir


def test_auto_prefers_openvino_then_onnx_then_checkpoint(tmp_path):
    full = _seed(tmp_path / "full", int8=True, onnx=True, ckpts=["checkpoint_best_epoch-001.ckpt"])
    engine, path = resolve_runtime_artifact(full)
    assert engine == "openvino"
    assert path == full / "model_int8.xml"

    no_ir = _seed(tmp_path / "no_ir", onnx=True, ckpts=["checkpoint_best_epoch-001.ckpt"])
    engine, path = resolve_runtime_artifact(no_ir)
    assert engine == "onnx"
    assert path == no_ir / "model.onnx"

    ckpt_only = _seed(tmp_path / "ckpt_only", ckpts=["checkpoint_best_epoch-001.ckpt"])
    engine, path = resolve_runtime_artifact(ckpt_only)
    assert engine == "checkpoint"
    assert path == ckpt_only / "checkpoint_best_epoch-001.ckpt"


def test_empty_bundle_lists_the_tried_chain(tmp_path):
    empty = _seed(tmp_path / "empty")
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_runtime_artifact(empty)
    msg = str(excinfo.value)
    assert "OpenVINO" in msg and "ONNX" in msg and "checkpoint" in msg


def test_explicit_prefer_never_falls_through(tmp_path):
    ir_only = _seed(tmp_path / "ir_only", fp16=True)
    with pytest.raises(FileNotFoundError):
        resolve_runtime_artifact(ir_only, prefer="onnx")
    with pytest.raises(FileNotFoundError):
        resolve_runtime_artifact(ir_only, prefer="checkpoint")
    with pytest.raises(FileNotFoundError):
        resolve_runtime_artifact(ir_only, prefer="int8")
    engine, path = resolve_runtime_artifact(ir_only, prefer="fp16")
    assert engine == "openvino"
    assert path == ir_only / "model_fp16.xml"


def test_prefer_pins_onnx_and_checkpoint_engines(tmp_path):
    full = _seed(tmp_path / "full", int8=True, onnx=True, ckpts=["checkpoint_best_epoch-003.ckpt"])
    assert resolve_runtime_artifact(full, prefer="onnx") == ("onnx", full / "model.onnx")
    engine, path = resolve_runtime_artifact(full, prefer="checkpoint")
    assert engine == "checkpoint"
    assert path.name == "checkpoint_best_epoch-003.ckpt"


def test_newest_checkpoint_picks_latest_mtime(tmp_path):
    d = _seed(tmp_path / "m", ckpts=["checkpoint_best_epoch-001.ckpt"])
    time.sleep(0.02)
    later = d / "checkpoint_best_epoch-002_bnrecal.ckpt"
    later.write_bytes(b"ckpt2")
    assert newest_checkpoint(d) == later
    empty = tmp_path / "empty"
    empty.mkdir()
    assert newest_checkpoint(empty) is None


def test_resolve_model_path_behaviour_unchanged(tmp_path):
    d = _seed(tmp_path / "m", int8=True, fp16=True)
    assert resolve_model_path(d) == d / "model_int8.xml"
    assert resolve_model_path(d, prefer="fp16") == d / "model_fp16.xml"
    with pytest.raises(FileNotFoundError):
        resolve_model_path(_seed(tmp_path / "none"))


def _tiny_run_bundle(tmp_path):
    """A minimal never-exported run bundle: checkpoint + frozen config snapshot."""
    import json

    import torch

    from sed.config import Config
    from sed.train.lit_module import SedLitModule

    cfg = Config()
    cfg.model.classes = ["_negative", "glass"]
    cfg.model.backbone = "kws-bcresnet"
    cfg.model.params = {"width_mult": 1.0, "time_stride": 1, "depth": 2}
    cfg.training.loss = "bce_logits"
    module = SedLitModule(cfg, list(cfg.model.classes))

    model_dir = tmp_path / "run-ckpt-only"
    model_dir.mkdir()
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": {},
            "pytorch-lightning_version": "2.0.0",
        },
        model_dir / "checkpoint_best_epoch-001.ckpt",
    )
    (model_dir / "training_config.json").write_text(
        json.dumps(cfg.model_dump(mode="json")), encoding="utf-8"
    )
    return cfg, model_dir


def test_checkpoint_backend_runs_a_never_exported_bundle(tmp_path):
    import numpy as np

    from sed.runtime.backends import CheckpointBackend, build_inference_backend, openvino_input_shape

    cfg, model_dir = _tiny_run_bundle(tmp_path)
    backend = build_inference_backend(cfg, model_dir, prefer=None, batch_size=2)
    assert isinstance(backend, CheckpointBackend)
    assert backend.engine == "checkpoint"

    shape = openvino_input_shape(cfg, batch_size=2)
    rng = np.random.default_rng(0)
    batch = rng.standard_normal(shape, dtype=np.float32)
    logits = backend.infer(batch)
    assert logits.shape == (2, 2)
    assert np.isfinite(logits).all()


def test_checkpoint_backend_rejects_geometry_drift(tmp_path):
    from sed.config import Config
    from sed.runtime.backends import CheckpointBackend

    cfg, model_dir = _tiny_run_bundle(tmp_path)
    drifted = Config.model_validate(cfg.model_dump(mode="json"))
    drifted.features.n_mels = 64
    with pytest.raises(RuntimeError, match="feature geometry"):
        CheckpointBackend(
            drifted, model_dir / "checkpoint_best_epoch-001.ckpt", batch_size=1
        )
