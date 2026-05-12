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
