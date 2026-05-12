"""Export a trained checkpoint to ONNX, then OpenVINO IR (FP32 + INT8).

The INT8 IR is what runs on the CL210/CL260 — typically 3–4× faster than FP32
on Intel CPUs with negligible accuracy loss for MobileNet-class CNNs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from sed.audio.features import MelExtractor
from sed.audio.resample import resample_audio
from sed.config import Config, ProjectLayout
from sed.train.lit_module import SedLitModule, load_classes

log = logging.getLogger(__name__)


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
) -> list[np.ndarray]:
    """Pull random clips from val.csv to feed NNCF for INT8 calibration."""
    import csv
    import random
    import soundfile as sf

    layout = ProjectLayout(project_id)
    rows = []
    with open(layout.dataset_dir / "val.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    random.seed(0)
    random.shuffle(rows)
    rows = rows[:n_samples]

    target_n = int(cfg.features.window_seconds * cfg.audio.sample_rate)
    out: list[np.ndarray] = []
    for row in rows:
        try:
            data, sr = sf.read(row["path"], dtype="float32", always_2d=True)
            wave = data[:, 0]
            if sr != cfg.audio.sample_rate:
                wave = resample_audio(wave, orig_sr=sr, target_sr=cfg.audio.sample_rate)
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


def export_project(cfg: Config, project_id: str, calibration_n: int = 300) -> dict:
    layout = ProjectLayout(project_id)
    ckpt_path = layout.models_dir / "current.ckpt"
    if not ckpt_path.exists():
        raise RuntimeError(f"No checkpoint at {ckpt_path}; train first.")

    classes = load_classes(layout.dataset_dir)
    lit = SedLitModule.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, classes=classes, map_location="cpu"
    )
    lit.eval()

    n_mels = cfg.features.n_mels
    n_frames = int(round(
        cfg.features.window_seconds * cfg.audio.sample_rate / cfg.features.hop_length
    )) + 1

    onnx_path = layout.models_dir / "model.onnx"
    _export_onnx(lit.model, onnx_path, n_mels, n_frames)

    # OpenVINO IR (FP32) ----------------------------------------------------
    import openvino as ov
    core = ov.Core()
    fp32 = ov.convert_model(str(onnx_path), input=[1, 1, n_mels, n_frames])
    fp32_path = layout.models_dir / "model_fp32.xml"
    ov.save_model(fp32, str(fp32_path))
    log.info("OpenVINO FP32 → %s", fp32_path)

    # OpenVINO IR (INT8) via NNCF ------------------------------------------
    int8_path = layout.models_dir / "model_int8.xml"
    try:
        import nncf
        mel = MelExtractor(
            sample_rate=cfg.audio.sample_rate,
            n_fft=cfg.features.n_fft,
            win_length=cfg.features.win_length,
            hop_length=cfg.features.hop_length,
            n_mels=n_mels,
            fmin=cfg.features.fmin,
            fmax=cfg.features.fmax,
        )
        calib = _build_calibration_set(cfg, project_id, mel, calibration_n)
        if not calib:
            raise RuntimeError("No calibration samples — skipping INT8")

        def transform(item):
            return item

        calibration_dataset = nncf.Dataset(calib, transform_func=transform)
        int8 = nncf.quantize(fp32, calibration_dataset, preset=nncf.QuantizationPreset.PERFORMANCE)
        ov.save_model(int8, str(int8_path))
        log.info("OpenVINO INT8 → %s", int8_path)
    except Exception as e:  # noqa: BLE001
        log.error("INT8 quantization failed (%s) — keeping FP32", e)
        int8_path = fp32_path

    # Manifest used by the runtime to load the right artifacts -------------
    manifest = {
        "project_id": project_id,
        "classes": classes,
        "negative_index": classes.index("_negative"),

        # ---- Feature parameters: MUST match between train and inference ----
        # These define the input shape the model expects. Computing different
        # mel features at inference time would produce shape mismatches or
        # silent accuracy drops, so the runtime reads these from here, not
        # from the live config.
        "sample_rate": cfg.audio.sample_rate,
        "input_sample_rate": cfg.audio.input_sample_rate,
        "n_mels": n_mels,
        "window_seconds": cfg.features.window_seconds,
        "hop_length": cfg.features.hop_length,
        "n_fft": cfg.features.n_fft,
        "win_length": cfg.features.win_length,
        "fmin": cfg.features.fmin,
        "fmax": cfg.features.fmax,

        # ---- Inference baseline: written here as a record of what was
        # calibrated against. The runtime reads these from the manifest at
        # startup, but they're also still in the live config and can be
        # overridden per venue without retraining. Calibration writes its
        # own per-class threshold map under [thresholds] in project.toml.
        "inference_baseline": {
            "hop_ms": cfg.inference.hop_ms,
            "smoothing_median_window": cfg.inference.smoothing_median_window,
            "smoothing_ema_alpha": cfg.inference.smoothing_ema_alpha,
            "hysteresis_low": cfg.inference.hysteresis_low,
            "consecutive_frames": cfg.inference.consecutive_frames,
            "negative_max": cfg.inference.negative_max,
            "margin_to_second_best": cfg.inference.margin_to_second_best,
            "trigger_refractory_ms": cfg.inference.trigger_refractory_ms,
            "default_threshold": cfg.inference.default_threshold,
        },

        # ---- Model files ----
        "model_int8": str(int8_path.relative_to(layout.models_dir)),
        "model_fp32": str(fp32_path.relative_to(layout.models_dir)),
        "onnx":       str(onnx_path.relative_to(layout.models_dir)),
    }
    with open(layout.models_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    # 'current.openvino' alias points to the int8 by default
    current_link = layout.models_dir / "current.openvino"
    target = (int8_path if int8_path.exists() else fp32_path)
    try:
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to(target.name)
    except OSError:
        # On filesystems that don't support symlinks (Windows, some FAT mounts),
        # write the path into a sentinel file instead.
        current_link.write_text(target.name)

    return manifest
