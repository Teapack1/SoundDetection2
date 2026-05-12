"""Export a trained checkpoint to ONNX, then OpenVINO IR (FP16 + INT8).

The INT8 IR is what runs on the CL210/CL260 — typically 3–4× faster than FP16
on Intel CPUs with negligible accuracy loss for MobileNet-class CNNs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from sed.audio.features import MelExtractor, feature_frame_count
from sed.audio.resample import resample_audio
from sed.config import Config, ProjectLayout
from sed.train.lit_module import SedLitModule, load_classes

log = logging.getLogger(__name__)


def convert_onnx_to_ir(onnx_path: Path, n_mels: int, n_frames: int):
    """ONNX → OpenVINO model with a dynamic batch dimension.

    Batch stays dynamic (-1): live inference batches one window per
    inference mic, so a static batch-1 IR would crash multi-mic setups.
    """
    import openvino as ov
    return ov.convert_model(str(onnx_path), input=[("mel", [-1, 1, n_mels, n_frames])])


def _export_onnx(model: torch.nn.Module, out_path: Path, n_mels: int, n_frames: int) -> Path:
    model.eval()
    dummy = torch.zeros(1, 1, n_mels, n_frames)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["mel"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
    )
    log.info("ONNX → %s", out_path)
    return out_path


def _build_calibration_set(
    cfg: Config,
    project_id: str,
    mel: MelExtractor,
    n_samples: int = 300,
    dataset_dir: Path | None = None,
) -> list[np.ndarray]:
    """Pull random clips from val.csv to feed NNCF for INT8 calibration."""
    import csv
    import random
    import soundfile as sf

    layout = ProjectLayout(project_id)
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else layout.dataset_dir
    rows = []
    with open(dataset_dir / "val.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    random.seed(0)
    random.shuffle(rows)
    rows = rows[:n_samples]

    target_n = int(cfg.features.window_seconds * cfg.model.sample_rate)
    out: list[np.ndarray] = []
    for row in rows:
        try:
            data, sr = sf.read(row["path"], dtype="float32", always_2d=True)
            wave = data[:, 0]
            if sr != cfg.model.sample_rate:
                wave = resample_audio(wave, orig_sr=sr, target_sr=cfg.model.sample_rate)
            if len(wave) >= target_n:
                wave = wave[:target_n]
            else:
                wave = np.pad(wave, (0, target_n - len(wave)))
            spec = mel(wave)
            if isinstance(spec, torch.Tensor):
                spec = spec.numpy()
            out.append(spec[0:1, None, :, :].astype(np.float32))
        except Exception:  # noqa: BLE001
            continue
    return out


def export_project(
    cfg: Config,
    project_id: str,
    calibration_n: int = 300,
    *,
    ckpt_path: Path | None = None,
    out_dir: Path | None = None,
    dataset_dir: Path | None = None,
) -> dict:
    """Export ONNX + OpenVINO IR (FP16 + INT8) and write a manifest.

    The trainer passes `ckpt_path`, `out_dir`, and `dataset_dir` so the export
    lands inside the model bundle being built (`models/<model_id>/`). When
    omitted, it falls back to the project's models/ dir for ad-hoc exports.
    """
    layout = ProjectLayout(project_id)
    if ckpt_path is None:
        ckpt_path = layout.models_dir / "current.ckpt"
    if out_dir is None:
        out_dir = layout.models_dir
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else layout.dataset_dir
    ckpt_path = Path(ckpt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.exists():
        raise RuntimeError(f"No checkpoint at {ckpt_path}; train first.")

    classes = load_classes(dataset_dir)
    lit = SedLitModule.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, classes=classes, map_location="cpu"
    )
    lit.eval()

    n_mels = cfg.features.n_mels
    n_frames = feature_frame_count(
        sample_rate=cfg.model.sample_rate,
        window_seconds=cfg.features.window_seconds,
        hop_length=cfg.features.hop_length,
        n_fft=cfg.features.n_fft,
        frontend=cfg.features.frontend,
    )

    onnx_path = out_dir / "model.onnx"
    _export_onnx(lit.model, onnx_path, n_mels, n_frames)

    # OpenVINO IR (FP16) ----------------------------------------------------
    import openvino as ov
    fp16 = convert_onnx_to_ir(onnx_path, n_mels, n_frames)
    fp16_path = out_dir / "model_fp16.xml"
    ov.save_model(fp16, str(fp16_path), compress_to_fp16=True)
    log.info("OpenVINO FP16 → %s", fp16_path)

    # OpenVINO IR (INT8) via NNCF ------------------------------------------
    int8_path = out_dir / "model_int8.xml"
    int8_succeeded = False
    try:
        import nncf
        mel = MelExtractor(
            sample_rate=cfg.model.sample_rate,
            n_fft=cfg.features.n_fft,
            win_length=cfg.features.win_length,
            hop_length=cfg.features.hop_length,
            n_mels=n_mels,
            fmin=cfg.features.fmin,
            fmax=cfg.features.fmax,
            frontend=cfg.features.frontend,
        )
        calib = _build_calibration_set(
            cfg,
            project_id,
            mel,
            calibration_n,
            dataset_dir=dataset_dir,
        )
        if not calib:
            raise RuntimeError("No calibration samples — skipping INT8")

        def transform(item):
            return item

        calibration_dataset = nncf.Dataset(calib, transform_func=transform)
        int8 = nncf.quantize(fp16, calibration_dataset, preset=nncf.QuantizationPreset.PERFORMANCE)
        ov.save_model(int8, str(int8_path))
        int8_succeeded = True
        log.info("OpenVINO INT8 → %s", int8_path)
    except Exception as e:  # noqa: BLE001
        log.error("INT8 quantization failed (%s) — keeping FP16", e)

    # `active_openvino.xml` is the canonical pointer the runtime loads. It's a
    # symlink (or text-file sentinel on filesystems without symlinks) to
    # model_int8.xml when INT8 quantization succeeded, else model_fp16.xml.
    # The runtime resolves it through sed.runtime.manifest.resolve_model_path.
    active = out_dir / "active_openvino.xml"
    target = int8_path if int8_succeeded else fp16_path
    try:
        if active.exists() or active.is_symlink():
            active.unlink()
        active.symlink_to(target.name)
    except OSError:
        active.write_text(target.name)

    # The runtime no longer reads a separate export manifest — every parameter
    # it needs lives in training_config.json (sample rate, mel/window params,
    # inference baseline) plus classes.json (class order). This returns a
    # small dict for human-facing CLI output only.
    return {
        "project_id": project_id,
        "classes": classes,
        "active_openvino": active.name,
        "active_target": target.name,
        "active_precision": "int8" if int8_succeeded else "fp16",
        "model_int8": int8_path.name if int8_succeeded else None,
        "model_fp16": fp16_path.name,
        "onnx": onnx_path.name,
        "feature_params": {
            "sample_rate": cfg.model.sample_rate,
            "frontend": cfg.features.frontend,
            "n_mels": n_mels,
            "window_seconds": cfg.features.window_seconds,
            "hop_length": cfg.features.hop_length,
            "n_fft": cfg.features.n_fft,
            "win_length": cfg.features.win_length,
            "fmin": cfg.features.fmin,
            "fmax": cfg.features.fmax,
        },
    }
