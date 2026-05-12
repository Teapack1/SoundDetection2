"""Export a trained checkpoint to ONNX, then OpenVINO IR (FP16 + INT8).

The INT8 IR is what runs on the CL210/CL260 — typically 3–4× faster than FP16
on Intel CPUs with negligible accuracy loss for MobileNet-class CNNs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from sed.audio.features import (
    build_feature_extractor,
    feature_band_count,
    feature_channels,
    feature_frame_count_for_cfg,
    window_sample_count,
)
from sed.audio.resample import resample_audio
from sed.config import Config, ProjectLayout, validate_clink_cnn_v2_feature_contract
from sed.train.lit_module import SedLitModule, load_classes

log = logging.getLogger(__name__)


def convert_onnx_to_ir(onnx_path: Path, n_bands: int, n_frames: int, channels: int = 1):
    """ONNX → OpenVINO model with a dynamic batch dimension.

    Batch stays dynamic (-1): live inference batches one window per
    inference mic, so a static batch-1 IR would crash multi-mic setups.
    ``n_bands`` is the feature height (mel n_mels or linear bin count);
    ``channels`` is the feature channel count (3 for a colormap image, else 1).
    """
    import openvino as ov
    return ov.convert_model(str(onnx_path), input=[("mel", [-1, int(channels), n_bands, n_frames])])


def _export_onnx(model: torch.nn.Module, out_path: Path, n_bands: int, n_frames: int,
                 channels: int = 1) -> Path:
    model.eval()
    dummy = torch.zeros(1, int(channels), n_bands, n_frames)
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


def _calibration_csv(dataset_dir: Path) -> Path:
    """Which composed CSV feeds NNCF's INT8 calibration.

    A run bundle snapshots the live val split as ``val.csv``, which is the
    closest stand-in for what the deployed model will see. Project dataset
    dirs no longer carry one — ``train.csv`` is the whole undivided train+val
    pool and val is carved out at train time (see sed.data.splits) — so ad-hoc
    exports of an older bundle fall back to that pool rather than failing into
    the generic "INT8 quantization failed" branch and silently shipping FP16.
    """
    for name in ("val.csv", "train.csv"):
        candidate = Path(dataset_dir) / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no val.csv or train.csv under {dataset_dir} to calibrate INT8 from — "
        "compose the dataset (or export from a run bundle that snapshotted one)"
    )


def _build_calibration_set(
    cfg: Config,
    project_id: str,
    mel,
    n_samples: int = 300,
    dataset_dir: Path | None = None,
) -> list[np.ndarray]:
    """Pull random composed clips to feed NNCF for INT8 calibration."""
    import csv
    import random
    import soundfile as sf

    from sed.data.augment import _ensure_length

    layout = ProjectLayout(project_id)
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else layout.dataset_dir
    csv_path = _calibration_csv(dataset_dir)
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    random.seed(0)
    random.shuffle(rows)
    n_available = len(rows)
    rows = rows[:n_samples]
    log.info(
        "INT8 calibration source: %s (%d of %d composed rows)",
        csv_path.name, len(rows), n_available,
    )

    target_n = window_sample_count(cfg.model.sample_rate, cfg.features.window_seconds)
    out: list[np.ndarray] = []
    skipped: list[str] = []
    for row in rows:
        try:
            data, sr = sf.read(row["path"], dtype="float32", always_2d=True)
            wave = data[:, 0]
            if sr != cfg.model.sample_rate:
                wave = resample_audio(wave, orig_sr=sr, target_sr=cfg.model.sample_rate)
            # Same window fit the datamodule applies (centre-crop / zero-pad), so
            # the activation ranges NNCF measures match what training produced.
            wave = _ensure_length(wave, target_n)
            spec = mel(wave)
            if isinstance(spec, torch.Tensor):
                spec = spec.numpy()
            # (1, n_mels, T) spectrogram → (1, 1, n_mels, T); (1, C, H, W) colormap
            # image is already NCHW.
            sample = spec[0:1, None, :, :] if spec.ndim == 3 else spec[0:1]
            out.append(sample.astype(np.float32))
        except Exception as exc:  # noqa: BLE001 — one bad clip must not end the export
            skipped.append(f"{row.get('path', '?')} ({type(exc).__name__}: {exc})")
    if skipped:
        # A thin calibration set is the usual cause of an INT8 model that
        # quantized "successfully" and then scores badly, so say what was lost.
        log.warning(
            "INT8 calibration skipped %d/%d unreadable clip(s): %s%s",
            len(skipped), len(rows), "; ".join(skipped[:5]),
            f"; +{len(skipped) - 5} more" if len(skipped) > 5 else "",
        )
    return out


def set_active_precision(out_dir: Path, precision: str) -> tuple[Path, Path]:
    """Point `active_openvino.xml` at the given exported IR precision.

    The pointer is a symlink where the filesystem allows it, else a text-file
    sentinel with the target name (both resolved by
    sed.runtime.manifest.resolve_model_path). Returns (active_path, target_path).
    """
    precision = str(precision).lower()
    if precision not in ("fp16", "int8"):
        raise ValueError(f"precision must be fp16 or int8, got {precision!r}")
    target = Path(out_dir) / f"model_{precision}.xml"
    if not target.exists():
        raise FileNotFoundError(f"no exported IR at {target}")
    active = Path(out_dir) / "active_openvino.xml"
    try:
        if active.exists() or active.is_symlink():
            active.unlink()
        active.symlink_to(target.name)
    except OSError:
        active.write_text(target.name)
    return active, target


EXPORT_TARGETS: tuple[str, ...] = ("onnx", "fp16", "int8")


def resolve_export_targets(targets: set[str] | None) -> dict[str, bool]:
    """Normalize a requested target set into what actually gets built.

    ONNX is always written when any IR is requested (it is the IR source and a
    runtime fallback); INT8 quantizes the FP16 IR, so requesting INT8 implies
    FP16. Raises on unknown / empty selections.
    """
    wanted = {str(t).lower() for t in (targets if targets is not None else EXPORT_TARGETS)}
    unknown = wanted - set(EXPORT_TARGETS)
    if unknown:
        raise ValueError(f"unknown export target(s): {sorted(unknown)} (expected {EXPORT_TARGETS})")
    if not wanted:
        raise ValueError("no export targets selected")
    want_int8 = "int8" in wanted
    want_fp16 = "fp16" in wanted or want_int8
    return {
        "onnx": "onnx" in wanted or want_fp16,
        "fp16": want_fp16,
        "int8": want_int8,
    }


def export_project(
    cfg: Config,
    project_id: str,
    calibration_n: int = 300,
    *,
    ckpt_path: Path | None = None,
    out_dir: Path | None = None,
    dataset_dir: Path | None = None,
    targets: set[str] | None = None,
    active_precision: str | None = None,
) -> dict:
    """Export ONNX + OpenVINO IR (FP16 + INT8) and write a manifest.

    The trainer passes `ckpt_path`, `out_dir`, and `dataset_dir` so the export
    lands inside the model bundle being built (`models/<model_id>/`). When
    omitted, it falls back to the project's models/ dir for ad-hoc exports.

    ``targets`` selects which artifacts to build (subset of ``EXPORT_TARGETS``;
    default all). ONNX is always written when any IR is requested because it is
    the IR source and a valid runtime fallback; INT8 quantizes the FP16 IR, so
    requesting INT8 also builds FP16. A run with only ONNX (or nothing beyond
    the checkpoint) still runs live — the runtime resolves OpenVINO → ONNX →
    checkpoint. ``active_precision`` (``fp16``/``int8``) picks which IR the
    runtime loads by default; ``None`` auto-picks INT8 when built, else FP16.
    """
    build = resolve_export_targets(targets)   # validates first, before any work
    validate_clink_cnn_v2_feature_contract(cfg, phase="export")
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

    n_bands = feature_band_count(cfg.features, cfg.model.sample_rate)
    n_frames = feature_frame_count_for_cfg(cfg)
    channels = feature_channels(cfg.features)

    want_int8 = build["int8"]
    want_fp16 = build["fp16"]
    build_onnx = build["onnx"]

    onnx_path = out_dir / "model.onnx"
    if build_onnx:
        _export_onnx(lit.model, onnx_path, n_bands, n_frames, channels)

    fp16_path = out_dir / "model_fp16.xml"
    int8_path = out_dir / "model_int8.xml"
    int8_succeeded = False
    # How many clips NNCF actually saw. A thin calibration set is a common cause
    # of an INT8 model that quantized "successfully" and then scores badly, and
    # it is invisible unless it is recorded.
    calibration_samples = 0
    active = target = None

    if want_fp16:
        # OpenVINO IR (FP16) ------------------------------------------------
        import openvino as ov
        fp16 = convert_onnx_to_ir(onnx_path, n_bands, n_frames, channels)
        ov.save_model(fp16, str(fp16_path), compress_to_fp16=True)
        log.info("OpenVINO FP16 → %s", fp16_path)

        # OpenVINO IR (INT8) via NNCF --------------------------------------
        if want_int8:
            try:
                import nncf
                mel = build_feature_extractor(cfg)
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
                calibration_samples = len(calib)
                log.info(
                    "OpenVINO INT8 → %s (%d calibration clips)",
                    int8_path, calibration_samples,
                )
            except Exception as e:  # noqa: BLE001
                log.error("INT8 quantization failed (%s) — keeping FP16", e)

        # `active_openvino.xml` is the canonical pointer the runtime loads — a
        # symlink (or text-file sentinel) to the chosen IR. The runtime resolves
        # it through sed.runtime.manifest.resolve_model_path.
        auto = "int8" if int8_succeeded else "fp16"
        choice = str(active_precision or auto).lower()
        if choice == "int8" and not int8_succeeded:
            choice = "fp16"          # requested INT8 unavailable → deploy FP16
        active, target = set_active_precision(out_dir, choice)
    else:
        log.info("ONNX-only export (no OpenVINO IR) → %s", onnx_path)

    # Human-facing CLI summary only; the runtime reads training_config.json +
    # classes.json, never this dict.
    return {
        "project_id": project_id,
        "classes": classes,
        "active_openvino": active.name if active is not None else None,
        "active_target": target.name if target is not None else None,
        "active_precision": (target.stem.replace("model_", "") if target is not None else None),
        "model_int8": int8_path.name if int8_succeeded else None,
        "model_fp16": fp16_path.name if want_fp16 else None,
        "onnx": onnx_path.name if build_onnx else None,
        "int8_calibration_clips": calibration_samples,
        "feature_params": {
            "sample_rate": cfg.model.sample_rate,
            "frontend": cfg.features.frontend,
            "feature_type": cfg.features.feature_type,
            "pcen": cfg.features.pcen,
            "n_bands": n_bands,
            "n_mels": cfg.features.n_mels,
            "window_seconds": cfg.features.window_seconds,
            "hop_length": cfg.features.hop_length,
            "n_fft": cfg.features.n_fft,
            "win_length": cfg.features.win_length,
            "fmin": cfg.features.fmin,
            "fmax": cfg.features.fmax,
        },
    }
