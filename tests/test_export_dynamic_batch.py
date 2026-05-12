"""Regression: the exported OpenVINO IR must keep a dynamic batch dimension.

Live inference batches one window per inference mic (multi-mic), so a static
batch-1 IR crashes the runtime the moment inference_channels lists 2+ mics.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ov = pytest.importorskip("openvino")

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


def test_runtime_compile_handles_legacy_dynamic_rank_squeeze(tmp_path):
    onnx_path = tmp_path / "legacy.onnx"
    _export_legacy_squeeze_onnx(onnx_path)
    ir = convert_onnx_to_ir(onnx_path, N_MELS, N_FRAMES)
    ir_path = tmp_path / "legacy.xml"
    ov.save_model(ir, str(ir_path))

    cfg = Config()
    cfg.model.sample_rate = 6401
    cfg.features.n_mels = N_MELS
    core = ov.Core()
    _model, compiled = compile_openvino_model(core, ir_path, cfg, batch_size=2)
    req = compiled.create_infer_request()

    x = np.zeros((2, 1, N_MELS, N_FRAMES), dtype=np.float32)
    logits = normalize_openvino_logits(
        list(req.infer({0: x}).values())[0],
        batch_size=2,
        num_classes=3,
    )

    assert logits.shape == (2, 3)
