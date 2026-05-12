from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


def test_class_activation_map_uses_classifier_weights_and_bias():
    from sed.train.explain import class_activation_map

    fmap = torch.tensor([[
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 1.0], [0.0, 1.0]],
    ]])
    weights = torch.tensor([
        [1.0, 0.5],
        [-1.0, 2.0],
    ])
    bias = torch.tensor([0.25, -0.5])

    cam = class_activation_map(fmap, weights, 0, classifier_bias=bias)

    expected = fmap[:, 0] + 0.5 * fmap[:, 1] + 0.25
    assert torch.allclose(cam, expected)


def test_build_mined_review_set_preserves_sidecar_probabilities(tmp_path, monkeypatch):
    from sed import config as cfg_mod
    from sed.config import Config, ProjectLayout
    from sed.train.explain import build_review_set

    monkeypatch.setattr(cfg_mod, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(cfg_mod, "LIBRARY_DIR", tmp_path / "data_library")
    layout = ProjectLayout("p")
    layout.create()

    mined = cfg_mod.LIBRARY_DIR / "mined"
    cheers_dir = mined / "cheers_loud" / "session-a"
    cheers_dir.mkdir(parents=True)
    cheers_wav = cheers_dir / "hit.wav"
    sf.write(str(cheers_wav), np.zeros(32000, dtype="float32"), 32000)
    cheers_wav.with_suffix(".json").write_text(json.dumps({
        "predicted_class": "cheers_loud",
        "confidence": 0.91,
        "kind": "runtime",
        "session_id": "s1",
        "probs": {"_negative": 0.02, "cheers_loud": 0.91},
    }), encoding="utf-8")

    unknown_dir = mined / "unknown_noise"
    unknown_dir.mkdir(parents=True)
    unknown_wav = unknown_dir / "noise.wav"
    sf.write(str(unknown_wav), np.zeros(32000, dtype="float32"), 32000)

    review = build_review_set(
        Config(),
        "p",
        "run-1",
        "mined",
        classes=["_negative", "cheers_loud"],
        work_dir=tmp_path / "work",
    )

    assert review.scope == "mined"
    assert review.csv_path.exists()
    rows = {Path(row["path"]).name: row for row in review.rows}
    assert rows["hit.wav"]["class_name"] == "cheers_loud"
    assert rows["hit.wav"]["sidecar_p:cheers_loud"] == "0.910000"
    assert rows["hit.wav"]["has_sidecar_probabilities"] == "true"
    assert rows["noise.wav"]["class_name"] == "_negative"
    assert rows["noise.wav"]["has_sidecar_probabilities"] == "false"


class _TinyLit(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2, 2, bias=False),
        )
        with torch.no_grad():
            self.model[0].weight.fill_(0.2)
            self.model[4].weight.copy_(torch.tensor([[1.0, 0.5], [0.25, 1.0]]))

    def forward(self, x):
        return self.model(x)


def test_grad_cam_and_occlusion_fallbacks_return_input_sized_maps():
    from sed.train.explain import _grad_cam, _occlusion_heatmap

    lit = _TinyLit()
    x = torch.ones(1, 1, 8, 8, dtype=torch.float32)

    grad = _grad_cam(lit, x.clone().requires_grad_(True), 0)
    occ = _occlusion_heatmap(lit, x, 0, max_steps_per_axis=4)

    assert grad is not None
    assert tuple(grad.shape) == (1, 8, 8)
    assert tuple(occ.shape) == (1, 8, 8)
    assert torch.isfinite(grad).all()
    assert torch.isfinite(occ).all()
    assert float(occ.max()) >= 0.0
