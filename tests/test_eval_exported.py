"""Exported-model scoring: the frozen test set runs through the OpenVINO IR
with the same dataset/probability path as the torch-checkpoint evaluation."""

import csv

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")
ov = pytest.importorskip("openvino")

from sed.audio.features import feature_frame_count
from sed.config import Config
from sed.train.eval import _collect_predictions_openvino


def _tiny_ir(tmp_path, cfg, num_classes):
    n_frames = feature_frame_count(
        sample_rate=cfg.model.sample_rate,
        window_seconds=cfg.features.window_seconds,
        hop_length=cfg.features.hop_length,
        n_fft=cfg.features.n_fft,
        frontend=cfg.features.frontend,
    )

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(cfg.features.n_mels, num_classes)

        def forward(self, x):
            return self.fc(x.mean(dim=(1, 3)))

    onnx_path = tmp_path / "model.onnx"
    torch.onnx.export(
        Tiny().eval(),
        torch.zeros(1, 1, cfg.features.n_mels, n_frames),
        str(onnx_path),
        input_names=["mel"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
    )
    ir = ov.convert_model(
        str(onnx_path), input=[("mel", [-1, 1, cfg.features.n_mels, n_frames])]
    )
    ir_path = tmp_path / "model_fp16.xml"
    ov.save_model(ir, str(ir_path))
    return ir_path


def test_collect_predictions_openvino_matches_dataset_contract(tmp_path):
    cfg = Config()
    classes = ["_negative", "cheers"]
    rng = np.random.default_rng(0)
    rows = []
    for i, cls in enumerate(classes):
        wav = tmp_path / f"clip{i}.wav"
        sf.write(str(wav), (rng.standard_normal(32000) * 0.1).astype(np.float32), 32000)
        rows.append([str(wav), cls, f"s{i}", 32000, "1.0", int(cls != "_negative"), "sounds"])
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    with open(dataset_dir / "test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_name", "session_id", "sample_rate",
                    "duration_s", "is_target", "origin"])
        w.writerows(rows)

    ir_path = _tiny_ir(tmp_path, cfg, num_classes=len(classes))
    probs, targets, out_classes, out_rows = _collect_predictions_openvino(
        cfg, "any-project", ir_path, dataset_dir, classes
    )

    assert probs.shape == (2, 2)
    assert targets.shape == (2, 2)
    assert out_classes == classes
    assert len(out_rows) == 2
    # cross_entropy objective → softmax rows sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(targets, np.eye(2, dtype=np.float32))
