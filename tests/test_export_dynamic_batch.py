"""Regression: the exported OpenVINO IR must keep a dynamic batch dimension.

Live inference batches one window per inference mic (multi-mic), so a static
batch-1 IR crashes the runtime the moment inference_channels lists 2+ mics.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ov = pytest.importorskip("openvino")

from sed.audio.features import feature_frame_count_for_cfg
from sed.config import Config
from sed.export.openvino import convert_onnx_to_ir
from sed.runtime.infer import compile_openvino_model, normalize_openvino_logits

N_MELS = 16
N_FRAMES = 21


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3)
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, x):
        h = self.conv(x).mean(dim=(2, 3))
        return self.fc(h)


class _LegacySqueezeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3)
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, x):
        h = torch.nn.functional.adaptive_avg_pool2d(self.conv(x), (1, 1)).squeeze()
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.fc(h)


def _export_tiny_onnx(path):
    model = _TinyModel().eval()
    dummy = torch.zeros(1, 1, N_MELS, N_FRAMES)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["mel"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
    )


def _export_legacy_squeeze_onnx(path):
    model = _LegacySqueezeModel().eval()
    dummy = torch.zeros(1, 1, N_MELS, N_FRAMES)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["mel"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
    )


def test_converted_ir_accepts_multi_mic_batches(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    _export_tiny_onnx(onnx_path)

    ir = convert_onnx_to_ir(onnx_path, N_MELS, N_FRAMES)
    dims = list(ir.input(0).partial_shape)
    assert dims[0].is_dynamic, f"batch dim must be dynamic, got {ir.input(0).partial_shape}"

    compiled = ov.Core().compile_model(ir, "CPU")
    req = compiled.create_infer_request()
    for batch in (1, 2, 3):
        x = np.zeros((batch, 1, N_MELS, N_FRAMES), dtype=np.float32)
        out = list(req.infer({0: x}).values())[0]
        assert out.shape == (batch, 3)


def test_runtime_shape_uses_feature_band_count_for_linear(tmp_path):
    """Linear-feature bundles must load: the runtime input height is the
    feature band count, not n_mels (regression: linear models were exported
    fine but failed runtime shape validation and calibration reshape)."""
    from sed.audio.features import feature_band_count, feature_frame_count
    from sed.runtime.infer import openvino_input_shape

    cfg = Config()
    cfg.model.sample_rate = 6401
    cfg.features.n_mels = N_MELS
    assert openvino_input_shape(cfg)[2] == N_MELS

    cfg.features.feature_type = "linear"
    bands = feature_band_count(cfg.features, cfg.model.sample_rate)
    assert bands != cfg.features.n_mels
    assert openvino_input_shape(cfg)[2] == bands

    frames = feature_frame_count(
        sample_rate=cfg.model.sample_rate,
        window_seconds=cfg.features.window_seconds,
        hop_length=cfg.features.hop_length,
        n_fft=cfg.features.n_fft,
        frontend=cfg.features.frontend,
    )
    onnx_path = tmp_path / "linear.onnx"
    _export_tiny_onnx(onnx_path)
    ir = convert_onnx_to_ir(onnx_path, bands, frames)
    ir_path = tmp_path / "linear.xml"
    ov.save_model(ir, str(ir_path))

    core = ov.Core()
    _model, compiled = compile_openvino_model(core, ir_path, cfg, batch_size=1)
    req = compiled.create_infer_request()
    x = np.zeros((1, 1, bands, frames), dtype=np.float32)
    logits = normalize_openvino_logits(
        list(req.infer({0: x}).values())[0], batch_size=1, num_classes=3,
    )
    assert logits.shape == (1, 3)


def test_runtime_compile_handles_legacy_dynamic_rank_squeeze(tmp_path):
    cfg = Config()
    cfg.model.sample_rate = 6401
    cfg.features.n_mels = N_MELS
    # Derived from the config so the project's default window can change
    # without invalidating this compile check.
    frames = feature_frame_count_for_cfg(cfg)

    onnx_path = tmp_path / "legacy.onnx"
    _export_legacy_squeeze_onnx(onnx_path)
    ir = convert_onnx_to_ir(onnx_path, N_MELS, frames)
    ir_path = tmp_path / "legacy.xml"
    ov.save_model(ir, str(ir_path))

    core = ov.Core()
    _model, compiled = compile_openvino_model(core, ir_path, cfg, batch_size=2)
    req = compiled.create_infer_request()

    x = np.zeros((2, 1, N_MELS, frames), dtype=np.float32)
    logits = normalize_openvino_logits(
        list(req.infer({0: x}).values())[0],
        batch_size=2,
        num_classes=3,
    )

    assert logits.shape == (2, 3)


def test_resolve_export_targets_dependencies():
    """INT8 implies FP16 (its quantization source); any IR implies ONNX."""
    from sed.export.openvino import resolve_export_targets

    # default = everything
    assert resolve_export_targets(None) == {"onnx": True, "fp16": True, "int8": True}
    # FP16-only keeps ONNX (the IR source) but no INT8
    assert resolve_export_targets({"fp16"}) == {"onnx": True, "fp16": True, "int8": False}
    # INT8 pulls in FP16 (+ ONNX)
    assert resolve_export_targets({"int8"}) == {"onnx": True, "fp16": True, "int8": True}
    # ONNX-only builds no IR — the run still deploys on ONNX/checkpoint
    assert resolve_export_targets({"onnx"}) == {"onnx": True, "fp16": False, "int8": False}


def test_resolve_export_targets_rejects_bad_input():
    from sed.export.openvino import resolve_export_targets

    with pytest.raises(ValueError):
        resolve_export_targets({"bogus"})
    with pytest.raises(ValueError):
        resolve_export_targets(set())


# ClinkCNN-v2's dynamic-batch export is covered by
# tests/test_clink_cnn_v2.py::test_exports_to_openvino_with_dynamic_batch.


def test_int8_calibration_falls_back_to_the_train_pool(tmp_path):
    """Project dataset dirs carry no val.csv — train.csv is the whole undivided
    train+val pool and val is carved out at train time. Without the fallback the
    missing file lands in the generic "INT8 quantization failed" branch and the
    bundle silently ships FP16.
    """
    from sed.export.openvino import _calibration_csv

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with pytest.raises(FileNotFoundError, match="no val.csv or train.csv"):
        _calibration_csv(dataset)

    (dataset / "train.csv").write_text("path,class_name\n", encoding="utf-8")
    assert _calibration_csv(dataset).name == "train.csv"

    # A run bundle snapshots its live val split; that is the better stand-in
    # for what the deployed model sees, so it wins when both exist.
    (dataset / "val.csv").write_text("path,class_name\n", encoding="utf-8")
    assert _calibration_csv(dataset).name == "val.csv"
