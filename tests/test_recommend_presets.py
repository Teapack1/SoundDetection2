"""Named training-strategy presets for the Train Lab."""

from __future__ import annotations

import pytest

from sed.train.recommend import (
    TRAINING_PRESETS,
    fine_tune_depth_label,
    preset_hparams,
    preset_is_applicable,
    recommend_hparams,
    training_presets,
)

_VALID_POLICIES = {"head", "last_stage", "last_two", "full"}


def _is_valid_policy(p: str) -> bool:
    if p in _VALID_POLICIES:
        return True
    return p.startswith("last_n:") and p[len("last_n:"):].isdigit()


SMALL = {"glass": 400, "_negative": 1600}
LARGE = {"glass": 30000, "_negative": 30000}


def test_catalog_matches_definitions():
    catalog = training_presets()
    ids = [p["id"] for p in catalog]
    assert ids == [p["id"] for p in TRAINING_PRESETS]
    assert ids == [
        "auto", "venue_finetune", "venue_probe",
        "domain_pretrain_scratch", "domain_pretrain", "scratch_full", "ssl_byol",
    ]
    assert ids[0] == "auto"  # default first
    # Every preset declares which initialization it is designed for.
    for p in catalog:
        assert p["requires_init"] in {"any", "transfer", "scratch"}
    by_id = {p["id"]: p for p in catalog}
    assert by_id["venue_finetune"]["requires_init"] == "transfer"
    assert by_id["domain_pretrain"]["requires_init"] == "transfer"
    assert by_id["scratch_full"]["requires_init"] == "scratch"
    assert by_id["ssl_byol"]["requires_pretraining_project"] is True


def test_preset_applicability_follows_initialization():
    by_id = {p["id"]: p for p in TRAINING_PRESETS}

    def ok(pid, has_weights, pretraining=True):
        return preset_is_applicable(
            by_id[pid], has_transfer_weights=has_weights,
            is_pretraining_project=pretraining,
        )

    # Transfer presets need real encoder weights; scratch needs the opposite.
    assert ok("venue_finetune", True) and not ok("venue_finetune", False)
    assert ok("domain_pretrain", True) and not ok("domain_pretrain", False)
    assert ok("scratch_full", False) and not ok("scratch_full", True)
    assert ok("auto", True) and ok("auto", False)
    # BYOL is limited to pretraining projects, either initialization.
    assert ok("ssl_byol", True) and ok("ssl_byol", False)
    assert not ok("ssl_byol", True, pretraining=False)


@pytest.mark.parametrize("by_class,n_train", [(SMALL, 2000), (LARGE, 60000)])
def test_every_preset_returns_sane_values(by_class, n_train):
    for preset in TRAINING_PRESETS:
        pretrained = preset["requires_init"] != "scratch"
        r = preset_hparams(
            preset["id"], backbone="mn10_as", pretrained=pretrained,
            by_class=by_class, n_train=n_train, classes=["glass", "_negative"],
        )
        v = r["values"]
        assert _is_valid_policy(v["fine_tune_policy"]), (preset["id"], v["fine_tune_policy"])
        assert 0 < float(v["lr_head"]) <= 1e-1
        assert 0 < float(v["lr_backbone"]) <= 1e-2
        assert int(v["epochs"]) > 0
        assert 0.0 < float(v["lr_layer_decay"]) <= 1.0
        assert v["loss"] == "bce_logits"
        assert v["class_balance"] in {"off", "sqrt", "balanced"}
        # Masking + mixup follow the *resolved depth and data*, not the preset
        # name: a full-depth run adapts every stage and memorizes
        # session-correlated corpora without them, while shallow transfer stays a
        # clean baseline. "auto" therefore regularizes once the dataset is large
        # enough that it resolves to full depth.
        if preset["id"] == "domain_pretrain":
            assert v["specaugment_enabled"] is True
            assert float(v["mixup_alpha"]) == pytest.approx(0.3)
        elif preset["id"] == "scratch_full":
            assert v["specaugment_enabled"] is True
            assert float(v["mixup_alpha"]) == pytest.approx(0.2)
        elif preset["id"] != "ssl_byol":
            full_depth = str(v["fine_tune_policy"]) == "full"
            assert v["specaugment_enabled"] is (full_depth and n_train >= 500)
            if not full_depth:
                assert float(v["mixup_alpha"]) == 0.0


def test_forced_policies_are_honored_on_pretrained():
    def policy(pid):
        return preset_hparams(
            pid, backbone="mn10_as", pretrained=True,
            by_class=SMALL, n_train=2000, classes=["glass", "_negative"],
        )["values"]["fine_tune_policy"]

    assert policy("venue_finetune") == "last_two"
    assert policy("domain_pretrain") == "full"


def test_forced_policies_are_honored_for_parent_transfer_without_builtin_weights():
    def resolved(pid):
        return preset_hparams(
            pid, backbone="mn01", pretrained=True,
            by_class=SMALL, n_train=2000, classes=["glass", "_negative"],
            n_stages=5,
        )["values"]

    assert resolved("venue_finetune")["fine_tune_policy"] == "last_two"
    assert resolved("domain_pretrain")["fine_tune_policy"] == "full"


def test_recommendations_clamp_depth_to_model_stage_count():
    # Partial depth scales with the model: ~30% of a 17-stage MN is 5 stages.
    medium_17 = recommend_hparams(
        backbone="mn10_as",
        pretrained=True,
        train_depth="full",
        fine_tune_policy="",
        by_class=SMALL,
        n_train=2000,
        classes=["glass", "_negative"],
        n_stages=17,
    )
    assert medium_17["values"]["fine_tune_policy"] == "last_n:5"
    assert "last 5 of 17 encoder stages" in medium_17["reasons"]["fine_tune_policy"]

    medium_2 = recommend_hparams(
        backbone="mn10_as",
        pretrained=True,
        train_depth="full",
        fine_tune_policy="",
        by_class=SMALL,
        n_train=2000,
        classes=["glass", "_negative"],
        n_stages=2,
    )
    assert medium_2["values"]["fine_tune_policy"] == "full"
    assert "full fine-tune (2 of 2 encoder stages)" in medium_2["reasons"]["fine_tune_policy"]

    # Tiny sets stay shallow but still scale: ~10% of 17 stages is 2.
    tiny_17 = recommend_hparams(
        backbone="mn10_as",
        pretrained=True,
        train_depth="full",
        fine_tune_policy="",
        by_class={"glass": 40, "_negative": 80},
        n_train=120,
        classes=["glass", "_negative"],
        n_stages=17,
    )
    assert tiny_17["values"]["fine_tune_policy"] == "last_two"
    assert "last 2 of 17 encoder stages" in tiny_17["reasons"]["fine_tune_policy"]

    # Without stage metadata the named fallbacks apply.
    medium_unknown = recommend_hparams(
        backbone="mn10_as",
        pretrained=True,
        train_depth="full",
        fine_tune_policy="",
        by_class=SMALL,
        n_train=2000,
        classes=["glass", "_negative"],
    )
    assert medium_unknown["values"]["fine_tune_policy"] == "last_two"

    tiny_unknown = recommend_hparams(
        backbone="mn10_as",
        pretrained=True,
        train_depth="full",
        fine_tune_policy="",
        by_class={"glass": 40, "_negative": 80},
        n_train=120,
        classes=["glass", "_negative"],
    )
    assert tiny_unknown["values"]["fine_tune_policy"] == "last_stage"

    large = recommend_hparams(
        backbone="mn10_as",
        pretrained=True,
        train_depth="last_two",
        fine_tune_policy="",
        by_class=LARGE,
        n_train=60000,
        classes=["glass", "_negative"],
        n_stages=17,
    )
    assert large["values"]["fine_tune_policy"] == "full"


def test_preset_policies_clamp_to_model_stage_count():
    venue = preset_hparams(
        "venue_finetune", backbone="mn10_as", pretrained=True,
        by_class=SMALL, n_train=2000, classes=["glass", "_negative"],
        n_stages=2,
    )
    assert venue["values"]["fine_tune_policy"] == "full"
    assert "full fine-tune (2 of 2 encoder stages)" in venue["reasons"]["fine_tune_policy"]


def test_venue_preset_depth_scales_with_model_stage_count():
    def venue_policy(n_stages):
        return preset_hparams(
            "venue_finetune", backbone="mn10_as", pretrained=True,
            by_class=SMALL, n_train=2000, classes=["glass", "_negative"],
            n_stages=n_stages,
        )["values"]["fine_tune_policy"]

    assert venue_policy(17) == "last_n:5"   # ~30% of a 17-stage MN
    assert venue_policy(5) == "last_two"    # ~30% of 5 folds to the named policy
    assert venue_policy(None) == "last_two"  # unknown stages → named fallback


def test_auto_recommendation_chooses_depth_instead_of_preserving_current_head():
    r = recommend_hparams(
        backbone="mn10_as",
        pretrained=True,
        train_depth="head",
        fine_tune_policy="head",
        by_class=SMALL,
        n_train=2000,
        classes=["glass", "_negative"],
        n_stages=5,
    )

    assert r["values"]["fine_tune_policy"] == "last_two"
    assert "last 2 of 5 encoder stages" in r["reasons"]["fine_tune_policy"]


def test_scratch_legacy_recommendation_uses_equal_moderate_lrs():
    r = recommend_hparams(
        backbone="legacy-smallervggnet",
        pretrained=False,
        train_depth="full",
        fine_tune_policy="full",
        by_class=SMALL,
        n_train=2000,
        classes=["glass", "_negative"],
        n_stages=3,
    )

    assert r["values"]["fine_tune_policy"] == "full"
    assert r["values"]["lr_backbone"] == pytest.approx(3e-4)
    assert r["values"]["lr_head"] == pytest.approx(3e-4)


@pytest.mark.parametrize("preset_id", ["auto", "scratch_full"])
def test_clink_v2_domain_pretraining_recipe(preset_id):
    kwargs = dict(
        backbone="clink-cnn-v2", pretrained=False,
        by_class=LARGE, n_train=60000, classes=["glass", "_negative"], n_stages=9,
    )
    rec = (
        recommend_hparams(train_depth="full", **kwargs)
        if preset_id == "auto"
        else preset_hparams(preset_id, **kwargs)
    )
    v = rec["values"]
    assert v["fine_tune_policy"] == "full"
    assert v["epochs"] == 40 and v["warmup_epochs"] == 3 and v["patience"] == 8
    # Precision is advised from general constraints (memory/stability), not
    # from the backbone name: mixed precision halves activation memory.
    assert v["batch_size"] == 64 and v["precision"] == "16-mixed"
    assert v["lr_backbone"] == pytest.approx(3e-4)
    assert v["lr_head"] == pytest.approx(3e-4)
    assert v["weight_decay"] == pytest.approx(1e-3)
    assert v["class_balance"] == "balanced"
    assert v["specaugment_enabled"] is True
    assert v["specaugment_freq_mask"] == 8 and v["specaugment_time_mask"] == 6
    assert v["mixup_alpha"] == pytest.approx(0.1)


@pytest.mark.parametrize("preset_id", ["auto", "venue_finetune"])
def test_clink_v2_small_parent_fine_tune_recipe(preset_id):
    kwargs = dict(
        backbone="clink-cnn-v2", pretrained=True,
        by_class=SMALL, n_train=2000, classes=["glass", "_negative"], n_stages=9,
    )
    rec = (
        recommend_hparams(train_depth="full", **kwargs)
        if preset_id == "auto"
        else preset_hparams(preset_id, **kwargs)
    )
    v = rec["values"]
    assert v["fine_tune_policy"] == "last_two"
    assert v["fine_tune_warmup_epochs"] == 2
    assert v["epochs"] == 40 and v["patience"] == 8
    assert v["batch_size"] == 32 and v["precision"] == "16-mixed"
    assert v["lr_backbone"] == pytest.approx(3e-5)
    assert v["lr_head"] == pytest.approx(1e-4)
    assert v["weight_decay"] == pytest.approx(1e-3)
    assert v["class_balance"] == "balanced"
    assert v["freeze_batchnorm"] is True
    assert v["specaugment_enabled"] is True
    assert v["specaugment_freq_mask"] == 4 and v["specaugment_time_mask"] == 4
    assert v["mixup_alpha"] == 0.0


def test_layer_decay_protects_the_same_share_on_any_stage_granularity():
    """A per-stage constant means different things on differently-grained models.

    A fixed 0.75 leaves an EfficientAT MN's stem at 0.75**16 ≈ 1% of the backbone
    LR — effectively frozen despite the run being labelled a full fine-tune —
    while a 5-stage ConvNeXt keeps 32%. The recommendation is therefore derived
    from a target end-to-end ratio and the number of stages the policy opens.
    """
    from sed.train.recommend import FULL_DEPTH_LR_RATIO, layer_decay_for_ratio

    for n_stages in (5, 9, 17):
        v = recommend_hparams(
            backbone="mn10_as", pretrained=True, train_depth="full",
            fine_tune_policy="full", force_policy=True,
            by_class=LARGE, n_train=60000, classes=["glass", "_negative"],
            n_stages=n_stages, n_groups=400,
        )["values"]
        decay = float(v["lr_layer_decay"])
        assert decay ** (n_stages - 1) == pytest.approx(FULL_DEPTH_LR_RATIO, rel=0.02)

    # A single trainable stage has nothing to decay across.
    assert layer_decay_for_ratio(1, 0.1) == 1.0
    assert layer_decay_for_ratio(None, 0.1) == 1.0


def test_small_transfer_freezes_batchnorm_on_stage_aware_depths():
    """`last_two` resolves to the depth string "last_n:2", so a membership test
    against ("head", "last_block") silently stopped recommending frozen BN for
    every multi-stage model once depths became stage-aware."""
    for n_stages in (5, 9, 17):
        v = recommend_hparams(
            backbone="mn10_as", pretrained=True, train_depth="",
            by_class=SMALL, n_train=800, classes=["glass", "_negative"],
            n_stages=n_stages, n_groups=15,
        )["values"]
        assert v["freeze_batchnorm"] is True, n_stages
        # Frozen BN ships the parent's running statistics under adapted weights;
        # the post-fit pass is what makes that deployable.
        assert v["bn_recalibrate"] is True, n_stages

    # Enough data and a full-depth adaptation: BN re-learns honestly.
    big = recommend_hparams(
        backbone="mn10_as", pretrained=True, train_depth="",
        by_class=LARGE, n_train=60000, classes=["glass", "_negative"],
        n_stages=17, n_groups=400,
    )["values"]
    assert big["freeze_batchnorm"] is False


def test_mixup_runs_imply_batchnorm_recalibration():
    """Mixup trains BatchNorm on convex blends that never occur at inference."""
    v = recommend_hparams(
        backbone="mn10_as", pretrained=False, train_depth="full",
        by_class=LARGE, n_train=60000, classes=["glass", "_negative"],
        n_stages=17, n_groups=30,
    )["values"]
    assert float(v["mixup_alpha"]) > 0
    assert v["freeze_batchnorm"] is False
    assert v["bn_recalibrate"] is True


def test_depth_labels_include_stage_counts_when_known():
    assert fine_tune_depth_label("head", 5) == "head only (0 of 5 encoder stages)"
    assert fine_tune_depth_label("last_n:3", 5) == "last 3 of 5 encoder stages"
    assert fine_tune_depth_label("full", 5) == "full fine-tune (5 of 5 encoder stages)"


def test_preset_signature_overrides_apply():
    venue = preset_hparams(
        "venue_finetune", backbone="mn10_as", pretrained=True,
        by_class=SMALL, n_train=2000, classes=["glass", "_negative"],
    )["values"]
    assert venue["freeze_batchnorm"] is True

    ssl = preset_hparams(
        "ssl_byol", backbone="mn10_as", pretrained=True,
        by_class=LARGE, n_train=60000, classes=["glass", "_negative"],
    )["values"]
    assert ssl["pretraining_objective"] == "byol"
    assert ssl["class_balance"] == "off"


@pytest.mark.parametrize(
    "by_class,classes,expected",
    [
        ({"a": 100, "b": 100}, ["a", "b"], "off"),
        ({"a": 100, "b": 80}, ["a", "b"], "off"),
        ({"a": 100, "b": 79}, ["a", "b"], "sqrt"),
        ({"a": 100, "b": 0}, ["a", "b"], "off"),
        ({"a": 100}, ["a"], "off"),
        # Background is compared with the complete target pool, then target
        # classes with one another. 400 negative + 4x100 targets is already the
        # full policy's 50/50 hierarchy despite a 4:1 per-label count ratio.
        (
            {"_negative": 400, "a": 100, "b": 100, "c": 100, "d": 100},
            ["_negative", "a", "b", "c", "d"],
            "off",
        ),
        (
            {"_negative": 100, "a": 100, "b": 100, "c": 100, "d": 100},
            ["_negative", "a", "b", "c", "d"],
            "sqrt",
        ),
    ],
)
def test_generic_class_balance_recommendation_uses_hierarchical_count_ratio(
    by_class, classes, expected,
):
    values = recommend_hparams(
        backbone="mn10_as",
        pretrained=True,
        train_depth="last_n:2",
        by_class=by_class,
        n_train=sum(by_class.values()),
        classes=classes,
        n_stages=5,
    )["values"]

    assert values["class_balance"] == expected
    assert "sampling_strategy" not in values
    assert "negative_weight" not in values


def test_preset_epoch_budgets_match_strategy_intent():
    vals = {
        pid: preset_hparams(
            pid, backbone="mn10_as", pretrained=(pid != "scratch_full"),
            by_class=SMALL, n_train=2000, classes=["glass", "_negative"],
            n_stages=5,
        )["values"]
        for pid in (
            "auto", "venue_finetune", "domain_pretrain", "scratch_full", "ssl_byol",
        )
    }

    assert vals["auto"]["epochs"] == 35
    assert vals["auto"]["fine_tune_policy"] == "last_two"
    assert vals["venue_finetune"]["epochs"] == 34
    assert vals["venue_finetune"]["lr_backbone"] == pytest.approx(5e-5)
    assert vals["venue_finetune"]["lr_head"] == pytest.approx(1e-4)
    assert vals["venue_finetune"]["freeze_batchnorm"] is True
    assert vals["domain_pretrain"]["epochs"] == 35
    assert vals["domain_pretrain"]["warmup_epochs"] == 3
    # Layer-wise decay is expressed as an end-to-end LR ratio, not a fixed
    # per-stage constant: over 5 trainable stages 0.5623**4 ≈ 0.10, the same
    # protection a 17-stage MobileNet gets from its own (larger) per-stage value.
    assert vals["domain_pretrain"]["lr_layer_decay"] == pytest.approx(0.5623, abs=1e-4)
    assert vals["domain_pretrain"]["specaugment_enabled"] is True
    assert vals["domain_pretrain"]["mixup_alpha"] == pytest.approx(0.3)
    assert vals["ssl_byol"]["epochs"] == 80
    assert vals["ssl_byol"]["pretraining_objective"] == "byol"

    # Scratch: full network, one moderate LR, long budget, regularization on.
    scratch = vals["scratch_full"]
    assert scratch["fine_tune_policy"] == "full"
    assert scratch["epochs"] == 80
    assert scratch["warmup_epochs"] == 8
    assert scratch["lr_backbone"] == pytest.approx(3e-4)
    assert scratch["lr_head"] == pytest.approx(3e-4)
    assert scratch["lr_layer_decay"] == pytest.approx(1.0)
    assert scratch["freeze_batchnorm"] is False
    assert scratch["specaugment_enabled"] is True
    assert scratch["mixup_alpha"] == pytest.approx(0.2)


def test_large_corpus_and_scratch_budgets_scale_with_strategy():
    large_domain = preset_hparams(
        "domain_pretrain", backbone="mn10_as", pretrained=True,
        by_class=LARGE, n_train=60000, classes=["glass", "_negative"],
    )["values"]
    large_ssl = preset_hparams(
        "ssl_byol", backbone="mn10_as", pretrained=True,
        by_class=LARGE, n_train=60000, classes=["glass", "_negative"],
    )["values"]
    large_scratch = preset_hparams(
        "scratch_full", backbone="mn10_as", pretrained=False,
        by_class=LARGE, n_train=60000, classes=["glass", "_negative"],
    )["values"]

    # 40 (not 60): early stopping on val AP typically fires around epoch 15-25
    # on a large corpus, and the cosine schedule must have actually decayed by
    # then instead of holding near-peak LR for the whole overfitting phase.
    assert large_domain["epochs"] == 40
    assert large_ssl["epochs"] == 100
    # A large corpus needs fewer repeated passes even from scratch.
    assert large_scratch["epochs"] == 60
    assert large_scratch["fine_tune_policy"] == "full"


def test_non_byol_presets_restore_supervised_learning_signal():
    # Every non-BYOL preset must explicitly carry the supervised signal so the UI
    # switches back from byol when a supervised preset is selected.
    for pid in ("auto", "venue_finetune", "domain_pretrain", "scratch_full"):
        v = preset_hparams(
            pid, backbone="mn10_as", pretrained=(pid != "scratch_full"),
            by_class=SMALL, n_train=2000, classes=["glass", "_negative"],
        )["values"]
        assert v["pretraining_objective"] == "supervised", pid


def test_scratch_backbone_falls_back_to_full():
    # A non-pretrained backbone can't do partial-depth transfer, even if a preset
    # forces a shallow policy (the UI additionally disables such presets).
    v = preset_hparams(
        "venue_finetune", backbone="mn01", pretrained=False,
        by_class=SMALL, n_train=2000, classes=["glass", "_negative"],
    )["values"]
    assert v["fine_tune_policy"] == "full"


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        preset_hparams(
            "nope", backbone="mn10_as", pretrained=True,
            by_class=SMALL, n_train=2000, classes=["glass", "_negative"],
        )


def test_presets_configure_after_training_evaluation():
    """Supervised presets opt into calibrated auto-test; BYOL opts everything out."""
    from sed.train.recommend import preset_hparams

    common = dict(
        backbone="mn10_as", by_class={"_negative": 500, "cheers": 500},
        n_train=1000, classes=["_negative", "cheers"],
    )
    venue = preset_hparams("venue_finetune", pretrained=True, **common)["values"]
    assert venue["auto_test"] is True
    assert venue["calibrate_thresholds"] is True
    assert venue["bn_recalibrate"] is True
    assert venue["calibrate_after"] is True

    domain = preset_hparams("domain_pretrain", pretrained=True, **common)["values"]
    assert domain["auto_test"] is True and domain["calibrate_thresholds"] is True

    scratch = preset_hparams("scratch_full", pretrained=False, **common)["values"]
    assert scratch["auto_test"] is True and scratch["calibrate_thresholds"] is True

    ssl = preset_hparams("ssl_byol", pretrained=True, **common)["values"]
    assert ssl["auto_test"] is False
    assert ssl["calibrate_thresholds"] is False
    assert ssl["bn_recalibrate"] is False
    assert ssl["calibrate_after"] is False


@pytest.mark.parametrize(
    ("n_groups", "expect_masking"),
    [(600, False), (20, True)],
)
def test_regularization_follows_recording_group_redundancy(n_groups, expect_masking):
    """Redundancy, not row count, drives the regularization advice.

    The same 60k rows spread over 600 independent recording groups is ~100 clips
    per group -- diverse material. Concentrated into 20 groups it is ~3000 clips
    per group, which a model memorizes regardless of how many rows there are.
    """
    rec = recommend_hparams(
        backbone="mn10_as", pretrained=True, train_depth="last_two",
        fine_tune_policy="last_two", force_policy=True,
        by_class={"glass": 30000, "_negative": 30000}, n_train=60000,
        classes=["glass", "_negative"], n_stages=9, n_groups=n_groups,
    )
    assert rec["values"]["specaugment_enabled"] is expect_masking
    if expect_masking:
        assert rec["values"]["weight_decay"] == pytest.approx(0.01)
        assert "recording group" in rec["reasons"]["specaugment_enabled"]


def test_scratch_encoder_is_regularized_for_any_backbone():
    """The advice must not depend on a per-backbone table."""
    for backbone in ("mn10_as", "transient-resnet", "clink-cnn-v2", "kws-bcresnet"):
        rec = recommend_hparams(
            backbone=backbone, pretrained=False, train_depth="full",
            by_class={"glass": 20000, "_negative": 20000}, n_train=40000,
            classes=["glass", "_negative"], n_stages=9, n_groups=30,
        )
        assert rec["values"]["specaugment_enabled"] is True, backbone
        assert float(rec["values"]["mixup_alpha"]) > 0.0, backbone


def test_venue_probe_freezes_the_encoder_via_policy_not_a_zero_lr():
    """The evening venue fit: ~500 clips from one room. Freezing must come from
    the depth policy — `apply_train_depth` marks encoder stages non-trainable —
    because a zero LR would leave BatchNorm and the optimizer state live."""
    r = preset_hparams(
        "venue_probe", backbone="mn10_as", pretrained=True,
        by_class={"wine": 500, "_negative": 1000}, n_train=1500,
        classes=["wine", "_negative"],
    )
    v = r["values"]
    assert v["fine_tune_policy"] == "head"
    assert float(v["lr_backbone"]) > 0, "freezing is the policy's job, not the LR's"
    assert v["freeze_batchnorm"] is True
    assert v["bn_recalibrate"] is False, "nothing in the encoder moved"
    assert v["calibrate_thresholds"] is True


def test_domain_pretrain_scratch_is_a_full_random_init_run():
    """The flagship pretraining run — random init on a large in-domain corpus,
    whose output becomes the parent for venue fine-tunes."""
    r = preset_hparams(
        "domain_pretrain_scratch", backbone="mn10_as", pretrained=False,
        by_class={"cheers": 30000, "_negative": 40000}, n_train=70000,
        classes=["cheers", "_negative"],
    )
    v = r["values"]
    assert v["fine_tune_policy"] == "full"
    assert v["freeze_batchnorm"] is False, "BatchNorm must learn its own statistics"
    assert float(v["lr_layer_decay"]) == 1.0, "nothing pretrained to preserve"
    assert int(v["epochs"]) > 0


def test_the_two_workflow_stages_are_gated_to_their_initialization():
    """A venue probe needs weights to freeze; a scratch pretrain must not be
    offered one — otherwise a transfer recipe silently becomes a scratch run."""
    by_id = {p["id"]: p for p in training_presets()}
    assert by_id["venue_probe"]["requires_init"] == "transfer"
    assert by_id["domain_pretrain_scratch"]["requires_init"] == "scratch"
