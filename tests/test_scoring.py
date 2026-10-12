"""Tests for the shared decision core.

Turning audio into a trigger is one rule, but it used to exist three times: the
live runtime, the FP sweep and the clip evaluator. These pin the single copy.
"""

from __future__ import annotations

import numpy as np
import pytest

from sed.config import Config
from sed.runtime.scoring import WindowScorer, build_arbiter, pool_mic_probabilities


class StubBackend:
    """Deterministic 'model': fires on window energy."""

    engine = "stub"
    path = "stub"

    def __init__(self, classes: list[str]):
        self.classes = classes
        self.calls = 0

    def infer(self, batch: np.ndarray) -> np.ndarray:
        self.calls += 1
        out = np.zeros((batch.shape[0], len(self.classes)), dtype=np.float32)
        for i in range(batch.shape[0]):
            hot = min(1.0, float(np.mean(np.abs(batch[i]))) * 4.0)
            out[i, 0] = 1.0 - hot
            out[i, 1] = hot
        return out


def _cfg(*, onset_gating=True, sr=32000) -> Config:
    cfg = Config()
    cfg.model.sample_rate = sr
    cfg.model.classes = ["_negative", "cheers"]
    cfg.dataset.target_classes = ["_negative", "cheers"]
    cfg.audio.record_sample_rate = sr
    cfg.features.window_seconds = 0.4
    cfg.inference.inference_channels = [0]
    cfg.inference.onset_gating_enabled = onset_gating
    return cfg


def test_infer_still_exposes_pooling_from_the_shared_core():
    """`sed.runtime.infer` is the runtime's public surface; the refactor must not
    move a name that tests and tooling import from there."""
    from sed.runtime.infer import pool_mic_probabilities as from_infer

    assert from_infer is pool_mic_probabilities


def test_pooling_best_keeps_one_mics_whole_vector():
    """Mixing one mic's target score with another's negative would let through a
    trigger neither mic supports."""
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert pool_mic_probabilities(probs, "best", {0}).tolist() == [0.2, 0.8]
    assert pool_mic_probabilities(probs, "mean", {0}).tolist() == pytest.approx([0.55, 0.45])
    assert pool_mic_probabilities(probs, "max", {0}).tolist() == [0.9, 0.8]


def test_pooling_best_ignores_suppressor_scores_when_choosing_the_mic():
    """"Best" means the mic that best heard something we would fire on.

    A mic that mainly heard cutlery is not evidence the event happened there, so
    it must not win the vote and impose its whole probability vector.
    """
    classes = ["_negative", "riedel_wine", "hardnegs"]
    # mic 0 heard the glass; mic 1 mostly heard the hard negative.
    probs = np.array([[0.10, 0.80, 0.05], [0.10, 0.30, 0.95]])
    target_idx = [classes.index("riedel_wine")]
    assert pool_mic_probabilities(probs, "best", {0}, target_idx).tolist() == [0.10, 0.80, 0.05]
    # Without the target restriction the suppressor picks the wrong mic.
    assert pool_mic_probabilities(probs, "best", {0}).tolist() == [0.10, 0.30, 0.95]


def test_build_arbiter_derives_suppressor_roles_from_targets_block():
    from sed.config import Config

    classes = ["_negative", "riedel_wine", "hardnegs"]
    cfg = Config.model_validate({
        "model": {"classes": classes}, "dataset": {"target_classes": classes},
    })
    arbiter = build_arbiter(cfg, classes)
    assert arbiter.trigger_classes == {"riedel_wine"}

    # An explicit [targets] entry overrides the name-based default in both
    # directions — nothing here is enforced by class name alone.
    cfg_enabled = Config.model_validate({
        "model": {"classes": classes}, "dataset": {"target_classes": classes},
        "targets": {"hardnegs": {"enabled": True}},
    })
    assert build_arbiter(cfg_enabled, classes).trigger_classes is None


def test_pooling_rejects_non_2d_input():
    """Preserved from the pre-refactor implementation — silently accepting a 1-D
    vector would hide a caller passing one mic where a matrix was expected."""
    with pytest.raises(ValueError):
        pool_mic_probabilities(np.array([0.5, 0.5]), "best", {0})


def test_build_arbiter_disables_temporal_consensus_only_when_onset_gated():
    """Onset-gated production scores one window per transient, so median/EMA and
    N-of-N do not apply — reproducing it without this measures a different system."""
    classes = ["_negative", "cheers"]

    gated = build_arbiter(_cfg(onset_gating=True), classes)
    assert gated.smoothers["cheers"].smoothing_enabled is False
    assert gated.smoothers["cheers"].consec == 1

    sliding = build_arbiter(_cfg(onset_gating=False), classes)
    assert sliding.smoothers["cheers"].smoothing_enabled is True
    assert sliding.smoothers["cheers"].consec == 3


def test_window_scorer_pads_and_trims_to_the_model_window():
    """A clip shorter or longer than the window must still produce exactly one
    decision, with the same geometry training used."""
    cfg = _cfg()
    classes = list(cfg.model.classes)
    scorer = WindowScorer(cfg, backend=StubBackend(classes), classes=classes)

    for n in (int(0.2 * 32000), int(0.4 * 32000), int(1.0 * 32000)):
        scored = scorer.score([np.zeros(n, dtype=np.float32)])
        assert set(scored.probs) == set(classes)
        assert scored.probs_per_mic.shape == (1, 2)


def test_window_scorer_batches_every_mic_into_one_inference_call():
    cfg = _cfg()
    cfg.inference.inference_channels = [0, 1, 2]
    classes = list(cfg.model.classes)
    backend = StubBackend(classes)
    scorer = WindowScorer(cfg, backend=backend, classes=classes, active_channels=[0, 1, 2])

    scored = scorer.score([np.zeros(12800, dtype=np.float32) for _ in range(3)])
    assert backend.calls == 1, "three mics must cost one batched call, not three"
    assert scored.probs_per_mic.shape == (3, 2)
