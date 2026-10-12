"""Generalizing `_negative` into an explicit, assignable class role.

`Config.class_roles()` used to hardwire role "negative" to the literal name
`_negative`. These tests cover the two things that change: (1) any class can
now be given `role = "negative"` explicitly and behave exactly like
`_negative` everywhere — loss, sampler, veto gate, eval — and (2) a
config with no `role =` key anywhere (the entire existing fleet) must keep
producing byte-identical behaviour to before this feature existed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sed.config import Config, TargetCfg


# --- Config layer: precedence and multi-negative-role support -------------

def test_explicit_role_makes_a_non_negative_class_behave_like_negative():
    cfg = Config(targets={"cutlery": TargetCfg(role="negative")})
    classes = ["_negative", "cutlery", "glass"]

    roles = cfg.class_roles(classes)
    assert roles["_negative"] == "negative"
    assert roles["cutlery"] == "negative"
    assert roles["glass"] == "target"

    # Both negative-role classes share inference.negative_max by default.
    assert cfg.class_threshold("cutlery") == pytest.approx(cfg.inference.negative_max)
    vetoes = cfg.class_vetoes(classes)
    assert vetoes["_negative"] == pytest.approx(cfg.inference.negative_max)
    assert vetoes["cutlery"] == pytest.approx(cfg.inference.negative_max)


def test_negative_role_class_can_override_its_own_veto_ceiling():
    cfg = Config(targets={"cutlery": TargetCfg(role="negative", veto_max=0.3)})
    vetoes = cfg.class_vetoes(["_negative", "cutlery"])
    assert vetoes["_negative"] == pytest.approx(cfg.inference.negative_max)
    assert vetoes["cutlery"] == pytest.approx(0.3)
    # class_threshold always resolves a negative-role class to negative_max —
    # veto_max only affects class_vetoes(), a deliberately separate axis.
    assert cfg.class_threshold("cutlery") == pytest.approx(cfg.inference.negative_max)


def test_negative_entry_with_only_mine_set_keeps_negative_role():
    """The literal-`_negative` tier must sit above the legacy enabled-bool
    tier: a [targets]._negative] block written only to set `mine = true`
    (leaving `enabled` at its default True) must not be reinterpreted as a
    firing target."""
    cfg = Config(targets={"_negative": TargetCfg(mine=True)})
    assert cfg.class_roles(["_negative"])["_negative"] == "negative"
    assert cfg.class_threshold("_negative") == pytest.approx(cfg.inference.negative_max)


def test_role_target_rejects_veto_max():
    with pytest.raises(Exception):
        TargetCfg(role="target", veto_max=0.5)


def test_mine_threshold_resolution_order():
    cfg = Config()
    cfg.inference.default_threshold = 0.61
    cfg.thresholds = {"cheers_quiet": 0.66}
    cfg.mining.per_class_min_confidence = {"cheers_loud": 0.88}
    cfg.targets = {"cheers_quiet": TargetCfg(mine_threshold=0.33)}

    assert cfg.mine_threshold("cheers_quiet") == pytest.approx(0.33)  # explicit wins
    assert cfg.mine_threshold("cheers_loud") == pytest.approx(0.88)   # legacy dict next
    assert cfg.mine_threshold("cheers_medium") == pytest.approx(0.61)  # class_threshold fallback


# --- Backward-compatibility guarantee --------------------------------------
# For a config with no `role =` key anywhere, every one of these must behave
# exactly as it did before `role` existed: `_negative` (by name) is the only
# negative-role class, everything else defaults via the suppressor-name
# heuristic or `enabled`.

def test_zero_role_config_matches_historical_role_resolution():
    cfg = Config(targets={"hardnegs": TargetCfg(enabled=False, veto_max=0.4)})
    classes = ["_negative", "hardnegs", "glass"]
    roles = cfg.class_roles(classes)
    assert roles == {"_negative": "negative", "hardnegs": "suppressor", "glass": "target"}
    assert cfg.class_threshold("_negative") == pytest.approx(cfg.inference.negative_max)
    assert cfg.class_vetoes(classes) == {
        "_negative": pytest.approx(cfg.inference.negative_max),
        "hardnegs": pytest.approx(0.4),
    }


def test_zero_role_config_no_negative_class_is_closed_set():
    cfg = Config()
    roles = cfg.class_roles(["glass_a", "glass_b"])
    assert roles == {"glass_a": "target", "glass_b": "target"}
    assert cfg.class_vetoes(["glass_a", "glass_b"]) == {}


# --- Training layer: multiple negative-role indices ------------------------

def test_asymmetric_focal_loss_applies_gamma_neg_to_every_negative_index():
    from sed.train.model import AsymmetricFocalLoss

    loss = AsymmetricFocalLoss(
        num_classes=4, negative_indices=[0, 2], gamma_pos=1.0, gamma_neg=3.0,
    )
    logits = torch.zeros(1, 4)
    targets = torch.zeros(1, 4)
    targets[0, 1] = 1.0  # the one true positive is a non-negative column

    # Directly probe the internal gamma buffer construction via forward: with
    # logits all zero, prob=0.5 everywhere, so the focal factor is
    # (1-0.5)**gamma = 0.5**gamma. Reconstruct it the same way the loss does.
    gammas = torch.full((4,), 1.0)
    gammas[[0, 2]] = 3.0
    expected_focal = (0.5) ** gammas
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    expected = (expected_focal.unsqueeze(0) * bce).mean()
    actual = loss(logits, targets)
    assert actual.item() == pytest.approx(expected.item(), rel=1e-5)


def test_class_balance_weights_pools_two_negative_classes_by_size():
    from sed.train.datamodule import _class_balance_weights

    rows = (
        [{"class_name": "_negative", "split_group": "neg-a"}] * 3
        + [{"class_name": "hardneg", "split_group": "neg-b"}] * 1
        + [{"class_name": "glass", "split_group": "pos-a"}] * 4
    )
    weights = _class_balance_weights(
        rows, "balanced", negative_classes={"_negative", "hardneg"},
    )
    neg_mass = weights[:3].sum().item() + weights[3:4].sum().item()
    target_mass = weights[4:].sum().item()
    assert neg_mass == pytest.approx(0.5, abs=1e-6)
    assert target_mass == pytest.approx(0.5, abs=1e-6)
    # Within the negative pool, mass is size-weighted (sqrt), so the larger
    # bucket (_negative, 3 rows) gets more than the smaller one (hardneg, 1 row).
    assert weights[:3].sum().item() > weights[3:4].sum().item()


def test_class_balance_weights_default_negative_set_is_just_negative():
    """No `negative_classes` argument (the un-updated-caller default) must
    keep today's behaviour: only the literal `_negative` is pooled."""
    from sed.train.datamodule import _class_balance_weights

    rows = (
        [{"class_name": "_negative", "split_group": "neg-a"}] * 2
        + [{"class_name": "hardneg", "split_group": "neg-b"}] * 2
        + [{"class_name": "glass", "split_group": "pos-a"}] * 2
    )
    weights = _class_balance_weights(rows, "balanced")
    # hardneg is NOT pooled as negative, so it is treated as a second target
    # class sharing the 50% target mass with glass.
    neg_mass = weights[:2].sum().item()
    assert neg_mass == pytest.approx(0.5, abs=1e-6)


# --- Eval/metrics layer: false_fire_rate / cross_target_confusion ---------

def test_false_fire_rate_sums_across_every_negative_role_class():
    from sed.train.test_rules import false_fire_rate

    per_class = {
        "_negative": {"support": 100, "fn": 5, "role": "negative"},
        "hardneg": {"support": 50, "fn": 10, "role": "negative"},
        "cutlery_suppressor": {"support": 20, "fn": 2, "role": "suppressor"},
        "glass": {"support": 30, "fn": 3, "role": "target"},
    }
    rate = false_fire_rate(per_class)
    # Every non-target row (negative + suppressor) contributes to the
    # denominator/numerator; the target row (glass) does not.
    expected = (5 + 10 + 2) / (100 + 50 + 20)
    assert rate == pytest.approx(expected)


def test_false_fire_rate_falls_back_to_literal_negative_when_role_absent():
    """Old, already-persisted manifests have no `role` key at all — every
    reader must fall back to `name == "_negative"` so historical runs keep
    scoring identically."""
    from sed.train.test_rules import false_fire_rate

    per_class = {
        "_negative": {"support": 100, "fn": 5},
        "hardnegs": {"support": 50, "fn": 10},
        "glass": {"support": 30, "fn": 3},
    }
    rate = false_fire_rate(per_class)
    assert rate == pytest.approx(5 / 100)


def test_cross_target_confusion_excludes_suppressor_from_target_set():
    from sed.train.metrics import cross_target_confusion

    classes = ["_negative", "hardneg", "glass_a", "glass_b"]
    roles = {"_negative": "negative", "hardneg": "suppressor", "glass_a": "target", "glass_b": "target"}
    # Rows/cols: _negative, hardneg, glass_a, glass_b
    cm = np.array([
        [90, 0, 0, 0],
        [0, 10, 0, 0],
        [0, 5, 15, 0],   # 5 glass_a clips predicted as hardneg (suppressor)
        [0, 0, 2, 18],   # 2 glass_b clips predicted as glass_a (real cross-target confusion)
    ])
    value = cross_target_confusion(cm, classes, roles=roles)
    # Only target-vs-target confusion counts: 2 wrong out of (15+2)+(18) target support.
    target_support = int(cm[2, :].sum() + cm[3, :].sum())
    assert value == pytest.approx(2 / target_support)


def test_decision_metrics_annotates_role_and_generalizes_negative_prior():
    from sed.train.metrics import decision_metrics

    classes = ["_negative", "hardneg", "glass"]
    roles = {"_negative": "negative", "hardneg": "negative", "glass": "target"}
    n = 10
    probs = np.zeros((n, 3), dtype=np.float64)
    targets = np.zeros((n, 3), dtype=np.float64)
    # 4 _negative clips, 3 hardneg clips, 3 glass clips.
    for i in range(4):
        targets[i, 0] = 1.0
        probs[i, 0] = 0.9
    for i in range(4, 7):
        targets[i, 1] = 1.0
        probs[i, 1] = 0.9
    for i in range(7, 10):
        targets[i, 2] = 1.0
        probs[i, 2] = 0.9
    pred_idx = np.argmax(probs, axis=1)
    from sed.train.metrics import confusion_matrix
    cm = confusion_matrix(probs, targets, classes, pred_idx=pred_idx)

    metrics = decision_metrics(
        probs, targets, classes, cm, thresholds={c: 0.5 for c in classes}, roles=roles,
    )
    assert metrics["per_class"]["_negative"]["role"] == "negative"
    assert metrics["per_class"]["hardneg"]["role"] == "negative"
    assert metrics["per_class"]["glass"]["role"] == "target"
    # n_negatives sums support across every non-target row, not just `_negative`.
    assert metrics["n_negatives"] == 7
    assert metrics["negative_prior"] == pytest.approx(0.7)
