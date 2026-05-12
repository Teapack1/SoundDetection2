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
        negative_index=0,
        negative_max=0.5,
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
        negative_index=0,
        negative_max=0.5,
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
        negative_index=0,
        negative_max=0.9,
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
        negative_index=0,
        negative_max=0.9,
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
        negative_index=None,
        negative_max=0.5,
        margin_to_second_best=0.05,
        global_refractory_ms=0,
    )

    ev = arbiter.step({"cling_a": 0.8, "cling_b": 0.2}, now=1.0)

    assert ev is not None
    assert ev.class_name == "cling_a"
