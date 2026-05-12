"""Exported-model scoring: the frozen test set runs through the OpenVINO IR
with the same dataset/probability path as the torch-checkpoint evaluation."""

import csv
import os

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
    cfg.training.loss = "cross_entropy"   # the config default is bce_logits/sigmoid
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


def test_set_active_precision_repoints_symlink(tmp_path):
    from sed.export.openvino import set_active_precision

    (tmp_path / "model_fp16.xml").write_text("fp16")
    (tmp_path / "model_int8.xml").write_text("int8")

    active, target = set_active_precision(tmp_path, "int8")
    assert active.name == "active_openvino.xml"
    assert target.name == "model_int8.xml"
    assert active.read_text() == "int8"

    active, target = set_active_precision(tmp_path, "fp16")
    assert target.name == "model_fp16.xml"
    assert active.read_text() == "fp16"

    with pytest.raises(FileNotFoundError):
        set_active_precision(tmp_path / "empty", "fp16")
    with pytest.raises(ValueError):
        set_active_precision(tmp_path, "int4")


# --- INT8 deploy decision -------------------------------------------------
# Quantization "succeeding" says nothing about whether the result is safe to
# deploy: INT8 moves class boundaries, and it moves them exactly where a
# confident model separates a target from the room. These pin the measurement
# that decides, and the fallback when it comes out badly.

def _pointer_target(model_dir):
    """What `active_openvino.xml` points at — a symlink where the filesystem
    allows one, else a text sentinel naming the target."""
    active = model_dir / "active_openvino.xml"
    return os.readlink(active) if active.is_symlink() else active.read_text()


def _decision(monkeypatch, tmp_path, *, fp16_recall, int8_recall, tolerance=0.02,
              has_int8=True):
    """Run `decide_ir_precision` against canned IR scores."""
    from sed.config import Config, ProjectLayout
    from sed.train import eval as ev

    model_dir = tmp_path / "models" / "run"
    model_dir.mkdir(parents=True)
    (model_dir / "model_fp16.xml").write_text("fp16")
    if has_int8:
        (model_dir / "model_int8.xml").write_text("int8")

    monkeypatch.setattr(ProjectLayout, "model_dir", lambda self, mid: model_dir)

    def fake_evaluate(cfg, project_id, model_id, *, precision, rule, **kw):
        recall = fp16_recall if precision == "fp16" else int8_recall
        if recall is None:
            raise RuntimeError(f"cannot score {precision}")
        # One curve point is enough: recall_at_budget just picks from them.
        return {"operating_curve": [
            {"threshold": 0.9, "recall_macro": recall, "false_per_1000": 0.5},
        ]}

    monkeypatch.setattr(ev, "evaluate_exported_run", fake_evaluate)
    cfg = Config()
    cfg.training.int8_max_regression = tolerance
    cfg.training.calibrate_per_1000 = 1.0
    return ev.decide_ir_precision(cfg, "p", "run"), model_dir


def test_int8_deploys_when_it_holds_its_recall(tmp_path, monkeypatch):
    decision, model_dir = _decision(monkeypatch, tmp_path, fp16_recall=0.90, int8_recall=0.895)
    assert decision["precision"] == "int8"
    assert "within tolerance" in decision["reason"]
    assert _pointer_target(model_dir) == "model_int8.xml"


def test_int8_falls_back_to_fp16_when_it_costs_too_much_recall(tmp_path, monkeypatch):
    """The documented behaviour that used to exist only in a comment: an INT8
    IR that quantized fine but lost real detections must not reach production."""
    decision, model_dir = _decision(monkeypatch, tmp_path, fp16_recall=0.94, int8_recall=0.90)
    assert decision["precision"] == "fp16"
    assert "over the" in decision["reason"]
    assert decision["fp16"] == pytest.approx(0.94)
    assert decision["int8"] == pytest.approx(0.90)
    assert _pointer_target(model_dir) == "model_fp16.xml"


def test_a_negative_tolerance_always_keeps_int8(tmp_path, monkeypatch):
    decision, _ = _decision(
        monkeypatch, tmp_path, fp16_recall=0.94, int8_recall=0.10, tolerance=-1.0)
    assert decision["precision"] == "int8"


def test_a_missing_or_unscorable_int8_deploys_fp16(tmp_path, monkeypatch):
    decision, _ = _decision(
        monkeypatch, tmp_path, fp16_recall=0.9, int8_recall=0.9, has_int8=False)
    assert decision["precision"] == "fp16"
    assert decision["int8"] is None

    decision, _ = _decision(monkeypatch, tmp_path / "b", fp16_recall=0.9, int8_recall=None)
    assert decision["precision"] == "fp16"


def test_the_decision_records_what_it_measured(tmp_path, monkeypatch):
    """A bare 'fp16' in the manifest is unauditable; the metric, both scores and
    the tolerance have to travel with it."""
    decision, _ = _decision(monkeypatch, tmp_path, fp16_recall=0.9, int8_recall=0.89)
    assert decision["metric"] == "recall_at_1_per_1000"
    assert decision["tolerance"] == pytest.approx(0.02)
    assert decision["rule"].startswith("fixed@")
    assert decision["decided_at"]
