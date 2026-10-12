"""Tests for the smoother + arbiter."""

from sed.runtime.smooth import ClassSmoother, TriggerArbiter


def test_class_smoother_basic_fire():
    s = ClassSmoother(median_window=3, ema_alpha=0.5, threshold=0.5,
                      hysteresis_low=0.2, consecutive_frames=2)
    fires = []
    for v in [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9]:
        fires.append(s.update(v)[0])
    assert any(fires), "Should fire at least once on a clean spike"


def test_class_smoother_rejects_single_spike():
    s = ClassSmoother(median_window=3, ema_alpha=0.5, threshold=0.6,
                      hysteresis_low=0.2, consecutive_frames=2)
    fires = []
    for v in [0.1, 0.1, 0.9, 0.1, 0.1, 0.1]:
        fires.append(s.update(v)[0])
    assert not any(fires), "single-frame spike should be filtered out"


def test_class_smoother_disabled_fires_on_single_window():
    # With smoothing off, a single window over threshold fires immediately —
    # no median/EMA/consecutive-frame state.
    s = ClassSmoother(median_window=3, ema_alpha=0.5, threshold=0.6,
                      hysteresis_low=0.2, consecutive_frames=3,
                      smoothing_enabled=False)
    ready, value = s.update(0.9)
    assert ready is True
    assert value == 0.9
    # And it does not fire below threshold.
    assert s.update(0.4)[0] is False


def test_arbiter_smoothing_disabled_matches_per_window():
    # smoothing_enabled=False makes the arbiter fire on the first window that
    # passes the gates, mirroring the frozen-test per-window decision.
    classes = ["_negative", "clink"]
    arbiter = TriggerArbiter(
        classes=classes,
        thresholds={"clink": 0.5},
        smoothers_cfg=dict(median_window=3, ema_alpha=0.4,
                           default_threshold=0.5, hysteresis_low=0.3,
                           consecutive_frames=3, smoothing_enabled=False),
        negative_classes={"_negative"},
        vetoes={"_negative": 0.5},
        margin_to_second_best=0.0,
        global_refractory_ms=0,
    )
    ev = arbiter.step({"_negative": 0.1, "clink": 0.9}, now=0.0)
    assert ev is not None
    assert ev.class_name == "clink"


def test_arbiter_negative_gates():
    classes = ["_negative", "clink"]
    arbiter = TriggerArbiter(
        classes=classes,
        thresholds={"clink": 0.5},
        smoothers_cfg=dict(median_window=1, ema_alpha=1.0,
                           default_threshold=0.5, hysteresis_low=0.1,
                           consecutive_frames=1),
        negative_classes={"_negative"},
        vetoes={"_negative": 0.5},
        margin_to_second_best=0.0,
        global_refractory_ms=0,
    )
    # Strong clink but ALSO strong negative → should NOT trigger
    ev = arbiter.step({"_negative": 0.8, "clink": 0.9}, now=0.0)
    assert ev is None


def test_arbiter_margin():
    classes = ["_negative", "clink_a", "clink_b"]
    arbiter = TriggerArbiter(
        classes=classes,
        thresholds={"clink_a": 0.5, "clink_b": 0.5},
        smoothers_cfg=dict(median_window=1, ema_alpha=1.0,
                           default_threshold=0.5, hysteresis_low=0.1,
                           consecutive_frames=1),
        negative_classes={"_negative"},
        vetoes={"_negative": 0.9},
        margin_to_second_best=0.20,
        global_refractory_ms=0,
    )
    # Both targets close → ambiguous, should not fire
    ev = arbiter.step({"_negative": 0.1, "clink_a": 0.7, "clink_b": 0.65}, now=0.0)
    assert ev is None
    # Clear winner → should fire
    ev = arbiter.step({"_negative": 0.1, "clink_a": 0.85, "clink_b": 0.55}, now=1.0)
    assert ev is not None
    assert ev.class_name == "clink_a"


def test_arbiter_restricts_active_trigger_classes():
    classes = ["_negative", "cheers_loud", "cheers_quiet"]
    arbiter = TriggerArbiter(
        classes=classes,
        thresholds={"cheers_loud": 0.5, "cheers_quiet": 0.5},
        smoothers_cfg=dict(median_window=1, ema_alpha=1.0,
                           default_threshold=0.5, hysteresis_low=0.1,
                           consecutive_frames=1),
        negative_classes={"_negative"},
        vetoes={"_negative": 0.9},
        margin_to_second_best=0.0,
        global_refractory_ms=0,
        trigger_classes={"cheers_loud"},
    )

    ev = arbiter.step(
        {"_negative": 0.1, "cheers_loud": 0.4, "cheers_quiet": 0.95},
        now=0.0,
    )
    assert ev is None

    ev = arbiter.step(
        {"_negative": 0.1, "cheers_loud": 0.95, "cheers_quiet": 0.4},
        now=1.0,
    )
    assert ev is not None
    assert ev.class_name == "cheers_loud"


def _suppressor_arbiter(**overrides):
    kwargs = dict(
        classes=["_negative", "riedel_wine", "ikea_wine", "hardnegs"],
        thresholds={"riedel_wine": 0.5, "ikea_wine": 0.5, "hardnegs": 0.5},
        smoothers_cfg=dict(median_window=1, ema_alpha=1.0, default_threshold=0.5,
                           hysteresis_low=0.1, consecutive_frames=1),
        negative_classes={"_negative"},
        vetoes={"_negative": 0.5},
        margin_to_second_best=0.05,
        global_refractory_ms=0,
        trigger_classes={"riedel_wine", "ikea_wine"},
    )
    kwargs.update(overrides)
    return TriggerArbiter(**kwargs)


def test_suppressor_winning_the_window_blocks_the_trigger():
    """A hard-negative class must be able to WIN, not merely exist.

    Excluding it from the comparison would hand the window to the strongest
    glass class while the model was saying "cutlery" — the exact false positive
    the class was trained to prevent.
    """
    ev = _suppressor_arbiter().step(
        {"_negative": 0.1, "riedel_wine": 0.80, "ikea_wine": 0.2, "hardnegs": 0.95},
        now=0.0,
    )
    assert ev is None


def test_target_still_fires_when_it_beats_the_suppressor():
    ev = _suppressor_arbiter().step(
        {"_negative": 0.1, "riedel_wine": 0.95, "ikea_wine": 0.2, "hardnegs": 0.1},
        now=0.0,
    )
    assert ev is not None and ev.class_name == "riedel_wine"


def test_suppressor_veto_blocks_even_when_a_target_wins():
    probs = {"_negative": 0.1, "riedel_wine": 0.95, "ikea_wine": 0.2, "hardnegs": 0.45}
    assert _suppressor_arbiter().step(probs, now=0.0) is not None
    assert _suppressor_arbiter(
        vetoes={"_negative": 0.5, "hardnegs": 0.4}
    ).step(probs, now=0.0) is None


def test_arbiter_matches_the_checkpoint_selection_decision():
    """Runtime, evaluation and checkpoint selection must gate identically.

    They diverged before: the arbiter dropped non-firing classes from the
    comparison while selection and evaluation compared every non-negative class.
    """
    import numpy as np
    import torch

    from sed.train.lit_module import fixed_threshold_trigger_counts
    from sed.train.metrics import runtime_decisions

    classes = ["_negative", "riedel_wine", "ikea_wine", "hardnegs"]
    firing = {"riedel_wine", "ikea_wine"}
    mask = torch.tensor([c in firing for c in classes])
    thresholds = {"riedel_wine": 0.5, "ikea_wine": 0.5, "hardnegs": 0.5}
    rows = [
        {"_negative": 0.1, "riedel_wine": 0.80, "ikea_wine": 0.20, "hardnegs": 0.95},
        {"_negative": 0.1, "riedel_wine": 0.95, "ikea_wine": 0.20, "hardnegs": 0.10},
        {"_negative": 0.9, "riedel_wine": 0.95, "ikea_wine": 0.20, "hardnegs": 0.10},
        {"_negative": 0.1, "riedel_wine": 0.62, "ikea_wine": 0.60, "hardnegs": 0.10},
    ]
    probs = np.array([[r[c] for c in classes] for r in rows], dtype=np.float32)

    live = [
        _suppressor_arbiter().step(r, now=float(i) * 10.0) is not None
        for i, r in enumerate(rows)
    ]
    _idx, decisions = runtime_decisions(
        probs, classes, thresholds, 0.5, 0.05, trigger_classes=firing,
    )
    offline = [bool(d["triggered"]) for d in decisions]
    # Label every row `_negative`, so the false-trigger count IS the selection
    # layer's own view of "did this window fire".
    targets = torch.zeros((len(rows), len(classes)))
    targets[:, 0] = 1.0
    counts = fixed_threshold_trigger_counts(
        torch.tensor(probs), targets,
        negative_indices=[0], threshold=0.5, negative_max=0.5,
        margin_to_second_best=0.05, firing_mask=mask,
    )
    assert live == offline == [False, True, False, False]
    assert int(counts[2]) == sum(live)
    assert int(counts[3]) == len(rows)


def test_arbiter_allows_no_negative_class():
    arbiter = TriggerArbiter(
        classes=["cling_a", "cling_b"],
        thresholds={"cling_a": 0.5, "cling_b": 0.5},
        smoothers_cfg=dict(
            median_window=1,
            ema_alpha=1.0,
            default_threshold=0.5,
            hysteresis_low=0.1,
            consecutive_frames=1,
        ),
        margin_to_second_best=0.05,
        global_refractory_ms=0,
    )

    ev = arbiter.step({"cling_a": 0.8, "cling_b": 0.2}, now=1.0)

    assert ev is not None
    assert ev.class_name == "cling_a"


def _split_negative_arbiter(**overrides):
    """Two negative-role classes (`_negative` and `cutlery`), one genuine
    suppressor (`hardnegs`) — exercises a project that split its background
    pool into a named bucket, alongside the pre-existing suppressor mechanism."""
    kwargs = dict(
        classes=["_negative", "cutlery", "hardnegs", "clink"],
        thresholds={"clink": 0.5, "hardnegs": 0.5},
        smoothers_cfg=dict(median_window=1, ema_alpha=1.0, default_threshold=0.5,
                           hysteresis_low=0.1, consecutive_frames=1),
        negative_classes={"_negative", "cutlery"},
        vetoes={"_negative": 0.5, "cutlery": 0.5},
        margin_to_second_best=0.05,
        global_refractory_ms=0,
        trigger_classes={"clink"},
    )
    kwargs.update(overrides)
    return TriggerArbiter(**kwargs)


def test_second_negative_class_vetoes_the_window_like_negative():
    """A negative-role class that is NOT literally named `_negative` must
    veto exactly like `_negative` does — this is the whole point of letting
    `role = "negative"` be assigned to any class."""
    arbiter = _split_negative_arbiter()
    probs = {"_negative": 0.1, "cutlery": 0.9, "hardnegs": 0.1, "clink": 0.95}
    assert arbiter.step(probs, now=0.0) is None


def test_negative_role_classes_never_enter_window_competition():
    """Neither `_negative` nor a second negative-role class may ever win the
    window — only `role == "suppressor"`/`"target"` classes compete."""
    arbiter = _split_negative_arbiter()
    # Both negative-role classes score highest; only hardnegs (suppressor)
    # and clink (target) compete, so hardnegs — not cutlery — should win and
    # block the trigger (suppressor-wins-blocks, matching the module contract).
    probs = {"_negative": 0.99, "cutlery": 0.98, "hardnegs": 0.6, "clink": 0.4}
    assert arbiter.step(probs, now=0.0) is None  # vetoed by _negative/cutlery anyway

    # Drop both negative-role scores below their ceiling so only the
    # competition matters: hardnegs beats clink and blocks (suppressor logic).
    probs2 = {"_negative": 0.1, "cutlery": 0.1, "hardnegs": 0.6, "clink": 0.4}
    assert arbiter.step(probs2, now=1.0) is None


def test_suppressor_still_competes_and_wins_unlike_negative_role_classes():
    arbiter = _split_negative_arbiter()
    probs = {"_negative": 0.1, "cutlery": 0.1, "hardnegs": 0.1, "clink": 0.95}
    ev = arbiter.step(probs, now=0.0)
    assert ev is not None
    assert ev.class_name == "clink"
