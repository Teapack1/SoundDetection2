"""Tests for the feature/model advisory layer.

This is a research tool: advice must never block or remove a configuration, only
explain what it costs on this task.
"""

from __future__ import annotations


from sed.config import Config
from sed.train.advice import (
    config_advice,
    feature_advice,
    mel_band_width_hz,
    model_advice,
    training_advice,
)


def _codes(findings) -> set[str]:
    return {f["code"] for f in findings}


def _features(**over):
    cfg = Config()
    for k, v in over.items():
        setattr(cfg.features, k, v)
    return cfg.features


# ---- the physics the advice rests on ----------------------------------------


def test_mel_bands_widen_toward_the_top_of_the_range():
    """The whole 'mel smears the partial comb' argument depends on this."""
    low = mel_band_width_hz(500, n_mels=128, fmin=80, fmax=20000)
    high = mel_band_width_hz(8000, n_mels=128, fmin=80, fmax=20000)

    assert low < high
    # ~30 Hz at 500 Hz vs ~230 Hz at 8 kHz for this geometry.
    assert 20 < low < 45
    assert 180 < high < 300


def test_more_bands_narrow_the_smear():
    coarse = mel_band_width_hz(8000, n_mels=128, fmin=80, fmax=20000)
    dense = mel_band_width_hz(8000, n_mels=256, fmin=80, fmax=20000)
    assert dense < coarse / 1.8


def test_mel_band_width_is_none_outside_the_range():
    assert mel_band_width_hz(30000, n_mels=128, fmin=0, fmax=20000) is None


# ---- feature advice ----------------------------------------------------------


def test_a_low_fmax_is_flagged_because_it_discards_glass_identity():
    findings = feature_advice(_features(fmax=8000, n_mels=64), sample_rate=16000)
    codes = _codes(findings)
    assert "fmax_below_glass_band" in codes
    assert "sample_rate_caps_glass_band" in codes


def test_fmin_at_zero_is_flagged_as_bands_spent_on_noise():
    findings = feature_advice(_features(fmin=0, fmax=20000), sample_rate=44100)
    finding = next(f for f in findings if f["code"] == "fmin_includes_rumble")
    assert finding["severity"] == "info", "a suggestion, never a blocker"
    assert finding["suggest"]["fmin"] == 200


def test_a_well_chosen_feature_raises_no_warnings():
    findings = feature_advice(
        _features(fmin=250, fmax=20000, n_mels=128, n_fft=2048,
                  hop_length=220, win_length=512, window_seconds=0.4),
        sample_rate=44100,
    )
    assert not [f for f in findings if f["severity"] in ("warn", "error")]


def test_a_dense_bank_from_a_high_fmin_surfaces_the_fft_time_tradeoff():
    """256 bands starting at 250 Hz genuinely need more FFT resolution than
    n_fft 2048 provides — and buying it costs time resolution on a 0.2 s event.
    The advisor must name that tension rather than recommending an impossible
    combination."""
    findings = feature_advice(
        _features(fmin=250, fmax=20000, n_mels=256, n_fft=2048), sample_rate=44100
    )
    finding = next(f for f in findings if f["code"] == "empty_mel_filters")
    assert finding["suggest"]["n_fft"] > 2048
    assert "costs time resolution" in finding["message"]
    assert finding["detail"]["narrowest_band_hz"] < finding["detail"]["fft_bin_hz"]


def test_a_coarse_hop_is_flagged_for_short_transients():
    findings = feature_advice(
        _features(fmin=250, fmax=20000, n_mels=256, hop_length=640), sample_rate=32000
    )
    finding = next(f for f in findings if f["code"] == "hop_coarse_for_transients")
    assert finding["suggest"]["hop_length"] == 160     # 5 ms at 32 kHz


def test_empty_mel_filters_are_flagged():
    """Bands narrower than an FFT bin carry nothing and render as black rows."""
    findings = feature_advice(
        _features(fmin=0, fmax=20000, n_mels=256, n_fft=512), sample_rate=44100
    )
    assert "empty_mel_filters" in _codes(findings)


def test_a_window_shorter_than_the_ring_is_flagged():
    findings = feature_advice(
        _features(fmin=250, fmax=20000, window_seconds=0.1), sample_rate=44100
    )
    assert "window_shorter_than_decay" in _codes(findings)


def test_findings_are_ordered_worst_first():
    findings = feature_advice(_features(fmin=0, fmax=8000, n_mels=64), sample_rate=16000)
    order = [f["severity"] for f in findings]
    assert order == sorted(order, key={"error": 0, "warn": 1, "info": 2}.get)


# ---- model advice ------------------------------------------------------------


def test_a_band_limited_backbone_is_advised_against_but_never_removed():
    """CED stays fully supported; the advice only says what it cannot hear."""
    cfg = Config()
    cfg.model.backbone = "ced-tiny"
    cfg.features.fmax = 8000
    findings = model_advice(cfg.model, cfg.features, 16000)

    finding = next(f for f in findings if f["code"] == "backbone_frontend_caps_band")
    assert finding["severity"] == "warn"
    assert "remains fully supported" in finding["message"]

    # Still a legal, buildable configuration.
    from sed.train.backbones import all_specs
    assert any(str(name).startswith("ced") for name in all_specs())


def test_imagenet_init_is_flagged_on_timm_backbones():
    cfg = Config()
    cfg.model.backbone = "resnet18"
    cfg.model.pretrained = True
    finding = next(
        f for f in model_advice(cfg.model, cfg.features, 32000)
        if f["code"] == "imagenet_init_transfers_poorly"
    )
    assert finding["suggest"]["initialization"] == "random"


def test_scratch_timm_draws_no_init_advice():
    cfg = Config()
    cfg.model.backbone = "resnet18"
    cfg.model.pretrained = False
    cfg.model.initialization = "random"
    assert "imagenet_init_transfers_poorly" not in _codes(
        model_advice(cfg.model, cfg.features, 32000)
    )


def test_audioset_contract_drift_is_flagged_only_when_the_geometry_moves():
    cfg = Config()
    cfg.model.backbone = "mn10_as"
    cfg.model.pretrained = True

    cfg.features.fmax = 15000
    on_contract = model_advice(cfg.model, cfg.features, 32000)
    assert "upstream_contract_drift" not in _codes(on_contract)

    cfg.features.fmax = 20000
    drifted = model_advice(cfg.model, cfg.features, 44100)
    finding = next(f for f in drifted if f["code"] == "upstream_contract_drift")
    assert "44100 Hz vs 32 kHz" in finding["message"]


# ---- training schedule ------------------------------------------------------


def _finetune_cfg(*, classes=("_negative", "clink_wine", "clink_tumbler"), **training):
    cfg = Config()
    cfg.model.classes = list(classes)
    cfg.model.initialization = "parent"
    cfg.model.parent_project = "pa"
    cfg.model.parent_run = "run-a"
    cfg.model.pretrained = False
    cfg.training.fine_tune_policy = "last_two"
    for key, value in training.items():
        setattr(cfg.training, key, value)
    return cfg


def test_frozen_batchnorm_without_recalibration_is_flagged():
    """The exported model would normalize with the parent's constants under
    weights that moved — a silent deployment defect, not a training one."""
    cfg = _finetune_cfg(freeze_batchnorm=True, bn_recalibrate=False)
    finding = next(
        f for f in training_advice(cfg, {"rows": 800})
        if f["code"] == "frozen_bn_without_recalibration"
    )
    assert finding["severity"] == "warn"
    assert finding["suggest"] == {"bn_recalibrate": True}

    cleared = _finetune_cfg(freeze_batchnorm=True, bn_recalibrate=True)
    assert "frozen_bn_without_recalibration" not in _codes(
        training_advice(cleared, {"rows": 800})
    )


def test_thin_transfer_set_is_told_to_freeze_batchnorm():
    cfg = _finetune_cfg(freeze_batchnorm=False)
    assert "bn_relearned_from_a_thin_sample" in _codes(training_advice(cfg, {"rows": 400}))
    # A large set can estimate its own statistics honestly.
    assert "bn_relearned_from_a_thin_sample" not in _codes(
        training_advice(cfg, {"rows": 40000})
    )


def test_sibling_classes_get_separation_advice_but_a_single_target_does_not():
    """Detecting a clink and naming which glass made it are different problems;
    only the second one needs the decision layer to commit."""
    siblings = training_advice(_finetune_cfg(), {"rows": 800})
    assert "margin_too_small_for_siblings" in _codes(siblings)

    single = training_advice(
        _finetune_cfg(classes=("_negative", "clink")), {"rows": 800},
    )
    assert "margin_too_small_for_siblings" not in _codes(single)
    assert "mixup_blurs_sibling_classes" not in _codes(single)


def test_frozen_encoder_is_flagged_when_classes_are_variants_of_one_sound():
    cfg = _finetune_cfg(fine_tune_policy="head")
    finding = next(
        f for f in training_advice(cfg, {"rows": 600})
        if f["code"] == "frozen_encoder_cannot_separate_siblings"
    )
    assert finding["severity"] == "warn"


def test_wide_frequency_masking_is_flagged_against_the_partial_comb():
    cfg = _finetune_cfg(specaugment_enabled=True, specaugment_freq_mask=24)
    cfg.features.n_mels = 128
    finding = next(
        f for f in training_advice(cfg, {"rows": 4000})
        if f["code"] == "specaugment_masks_the_partials"
    )
    assert finding["suggest"]["specaugment_freq_mask"] < 24

    cfg.training.specaugment_freq_mask = 8
    assert "specaugment_masks_the_partials" not in _codes(training_advice(cfg, {"rows": 4000}))


def test_a_head_that_outruns_the_encoder_is_flagged():
    """Early stopping fires on a head that fitted the parent's frozen features,
    so the encoder never gets to adapt."""
    cfg = _finetune_cfg(lr_backbone=1e-6, lr_head=1e-3)
    assert "head_outruns_the_encoder" in _codes(training_advice(cfg, {"rows": 800}))


def test_lr_ramp_longer_than_patience_is_flagged():
    cfg = _finetune_cfg(warmup_epochs=10, patience=6)
    assert "lr_warmup_outlasts_patience" in _codes(training_advice(cfg, {"rows": 800}))


def test_training_advice_never_blocks_and_survives_a_default_config():
    for finding in training_advice(Config(), {}):
        assert finding["severity"] in {"error", "warn", "info"}
        assert finding["message"]


def test_config_advice_groups_every_area():
    advice = config_advice(Config())
    assert set(advice) == {"features", "model", "training"}
    assert isinstance(advice["features"], list)
    assert isinstance(advice["training"], list)


def test_advice_stays_torch_free():
    import subprocess
    import sys

    probe = "import sed.train.advice, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
