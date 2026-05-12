"""SpecAugment masking: applied on the train split only, with the
normalised-domain fill value matching EfficientAT's pre-normalisation 0.0."""

import csv

import numpy as np
import soundfile as sf
import torch

from sed.config import Config
from sed.train.datamodule import SedDataset


def _make_dataset(tmp_path, cfg, *, train):
    wav = tmp_path / "clip.wav"
    if not wav.exists():
        rng = np.random.default_rng(0)
        sf.write(str(wav), (rng.standard_normal(32000) * 0.1).astype(np.float32), 32000)
    csv_path = tmp_path / "rows.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "class_name", "session_id", "sample_rate",
                        "duration_s", "is_target", "origin"])
            w.writerow([str(wav), "_negative", "s0", 32000, "1.0", 0, "sounds"])
    return SedDataset(csv_path, ["_negative"], cfg, train=train)


def _cfg_with_big_masks() -> Config:
    cfg = Config()
    cfg.training.specaugment_enabled = True
    cfg.training.specaugment_freq_mask = 60
    cfg.training.specaugment_time_mask = 40
    return cfg


def test_train_split_gets_masked_with_normalised_fill_value(tmp_path):
    cfg = _cfg_with_big_masks()
    reference, _ = _make_dataset(tmp_path, cfg, train=False)[0]
    ds = _make_dataset(tmp_path, cfg, train=True)
    expected_fill = (0.0 + 4.5) / 5.0  # efficientat fast-norm equivalent of raw 0.0

    torch.manual_seed(7)
    masked_any = False
    for _ in range(10):
        spec, _ = ds[0]
        diff = spec != reference
        if diff.any():
            masked_any = True
            assert torch.allclose(
                spec[diff], torch.full_like(spec[diff], expected_fill)
            ), "masked cells must use the normalised-domain fill value"
    assert masked_any, "with mask widths 60/40 some draw must mask something"


def test_val_and_test_loading_is_unmasked_and_deterministic(tmp_path):
    cfg = _cfg_with_big_masks()
    ds = _make_dataset(tmp_path, cfg, train=False)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)


def test_specaugment_can_be_disabled(tmp_path):
    cfg = _cfg_with_big_masks()
    reference, _ = _make_dataset(tmp_path, cfg, train=False)[0]
    cfg.training.specaugment_enabled = False
    ds = _make_dataset(tmp_path, cfg, train=True)
    spec, _ = ds[0]
    assert torch.equal(spec, reference)
