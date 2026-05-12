"""Parametric ``last_n:k`` fine-tune policy handling (model-aware depth control)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sed.config import TrainingCfg
from sed.train.fine_tuning import (
    available_fine_tune_policies,
    canonicalize_fine_tune_policy_for_stages,
    coerce_fine_tune_policy,
    fine_tune_policy_stage_count,
    last_n_stage_count,
    normalize_fine_tune_policy,
    resolve_fine_tune_plan,
)


def _cfg(policy: str, *, initialization: str = "upstream", pretrained: bool = True,
         epochs: int = 40, warmup: int = 0):
    return SimpleNamespace(
        model=SimpleNamespace(initialization=initialization, pretrained=pretrained),
        training=SimpleNamespace(
            fine_tune_policy=policy, train_depth="full",
            epochs=epochs, fine_tune_warmup_epochs=warmup,
        ),
    )


def _lit_cfg(policy: str, *, warmup: int = 1):
    from sed.config import Config

    cfg = Config()
    cfg.model.backbone = "mn01"
    cfg.model.pretrained = False
    cfg.model.initialization = "parent"
    cfg.model.parent_project = "seed"
    cfg.model.parent_run = "run"
    cfg.training.fine_tune_policy = policy
    cfg.training.fine_tune_warmup_epochs = warmup
    cfg.training.epochs = 8
    return cfg


def _trainable_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _optimizer_param_ids(optimizer) -> set[int]:
    return {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }


@pytest.mark.parametrize("value,expected", [
    ("last_n:0", "head"),
    ("last_n:1", "last_stage"),
    ("last_n:2", "last_two"),
    ("last_n:3", "last_n:3"),
    ("last_n=4", "last_n:4"),
    ("last_two", "last_two"),
    ("full", "full"),
])
def test_normalize_folds_small_last_n_into_named(value, expected):
    assert normalize_fine_tune_policy(value) == expected


def test_last_n_stage_count():
    assert last_n_stage_count("last_n:3") == 3
    assert last_n_stage_count("last_n=5") == 5
    assert last_n_stage_count("full") is None
    assert last_n_stage_count("last_two") is None


def test_resolve_plan_last_n_keeps_head_warmup():
    plan = resolve_fine_tune_plan(_cfg("last_n:3", epochs=40))
    assert plan.policy == "last_n:3"
    assert plan.initial_depth == "head"
    assert plan.target_depth == "last_n:3"
    assert plan.warmup_epochs >= 1  # bounded automatic head warm-up


def test_resolve_plan_clamps_warmup_so_target_depth_gets_an_epoch():
    one_epoch = resolve_fine_tune_plan(_cfg("full", epochs=1))
    assert one_epoch.initial_depth == one_epoch.target_depth == "full"
    assert one_epoch.warmup_epochs == 0

    too_long = resolve_fine_tune_plan(_cfg("last_n:3", epochs=3, warmup=99))
    assert too_long.initial_depth == "head"
    assert too_long.target_depth == "last_n:3"
    assert too_long.warmup_epochs == 2


def test_resolve_plan_scratch_forces_full_even_for_last_n():
    plan = resolve_fine_tune_plan(_cfg("last_n:3", initialization="random", pretrained=False))
    assert plan.target_depth == "full"


def test_coerce_preserves_valid_last_n_but_falls_back_when_too_deep():
    cfg = _cfg("last_n:3")
    assert coerce_fine_tune_policy(cfg, n_stages=5) == "last_n:3"
    # k reaches or exceeds the stage count → full fine-tune
    assert coerce_fine_tune_policy(cfg, n_stages=3) == "full"
    assert coerce_fine_tune_policy(cfg, n_stages=2) == "full"
    # scratch cannot do partial depth
    scratch = _cfg("last_n:3", initialization="random", pretrained=False)
    assert coerce_fine_tune_policy(scratch, n_stages=5) == "full"


def test_stage_canonicalization_counts_partial_depths_strictly_below_full():
    assert fine_tune_policy_stage_count("head", n_stages=17) == 0
    assert fine_tune_policy_stage_count("last_two", n_stages=17) == 2
    assert fine_tune_policy_stage_count("full", n_stages=17) == 17

    assert canonicalize_fine_tune_policy_for_stages("last_stage", n_stages=1) == "full"
    assert canonicalize_fine_tune_policy_for_stages("last_two", n_stages=2) == "full"
    assert canonicalize_fine_tune_policy_for_stages("last_n:17", n_stages=17) == "full"
    assert canonicalize_fine_tune_policy_for_stages("last_n:3", n_stages=17) == "last_n:3"


def test_available_policies_filter_partial_depths_by_stage_count():
    cfg = _cfg("last_two")
    assert [p.id for p in available_fine_tune_policies(cfg, n_stages=1)] == ["head", "full"]
    assert [p.id for p in available_fine_tune_policies(cfg, n_stages=2)] == ["head", "last_stage", "full"]
    assert [p.id for p in available_fine_tune_policies(cfg, n_stages=3)] == ["head", "last_stage", "last_two", "full"]

    parent = _cfg("last_two", initialization="parent", pretrained=False)
    assert [p.id for p in available_fine_tune_policies(parent, n_stages=2)] == ["head", "last_stage", "full"]

    scratch = _cfg("last_two", initialization="random", pretrained=False)
    assert [p.id for p in available_fine_tune_policies(scratch, n_stages=17)] == ["full"]


def test_training_cfg_accepts_and_canonicalizes_last_n():
    assert TrainingCfg(fine_tune_policy="last_n:3", epochs=40).fine_tune_policy == "last_n:3"
    assert TrainingCfg(fine_tune_policy="last_n=4", epochs=40).fine_tune_policy == "last_n:4"
    assert TrainingCfg(fine_tune_policy="last_n:2", epochs=40).fine_tune_policy == "last_two"
    with pytest.raises(ValueError):
        TrainingCfg(fine_tune_policy="last_n:banana", epochs=40)


def test_pretrained_flag_without_available_weights_is_not_transfer():
    """pretrained=True on a backbone with no published weights builds a random
    encoder (the adapters gate the actual load on pretrained_available), so it
    must train fully instead of warming up a head on frozen random features."""
    from sed.train.fine_tuning import available_fine_tune_policies, is_transfer_initialization

    cfg = _cfg("last_two")
    cfg.model.backbone = "mn01"  # registered, no AudioSet checkpoint
    assert not is_transfer_initialization(cfg)
    assert [p.id for p in available_fine_tune_policies(cfg)] == ["full"]
    plan = resolve_fine_tune_plan(cfg)
    assert plan.target_depth == "full" and plan.warmup_epochs == 0

    cfg.model.backbone = "mn10_as"  # AudioSet weights exist
    assert is_transfer_initialization(cfg)


def test_full_fine_tune_unfreezes_all_scheduled_params_after_warmup():
    from sed.train.lit_module import SedLitModule

    module = SedLitModule(_lit_cfg("full", warmup=1), ["_negative", "glass"])
    assert module.fine_tune_plan.initial_depth == "head"
    assert module.fine_tune_plan.target_depth == "full"
    assert module.freeze_info["train_depth"] == "head"
    assert module.target_freeze_info["train_depth"] == "full"
    assert (
        module.target_freeze_info["trainable_params"]
        == module.target_freeze_info["total_params"]
    )

    optimizer = module.configure_optimizers()["optimizer"]
    assert {id(p) for p in module.model.parameters()} <= _optimizer_param_ids(optimizer)
    assert module._maybe_unfreeze_for_fine_tune(0) is False
    assert module._maybe_unfreeze_for_fine_tune(1) is True
    assert module.freeze_info["train_depth"] == "full"
    assert _trainable_params(module.model) == module.target_freeze_info["trainable_params"]


def test_last_n_fine_tune_unfreezes_only_selected_tail_stages():
    from sed.train.backbones import describe_train_depth
    from sed.train.lit_module import SedLitModule

    module = SedLitModule(_lit_cfg("last_n:3", warmup=1), ["_negative", "glass"])
    head_params = module.freeze_info["trainable_params"]
    full_params = describe_train_depth(module.model, "full")["trainable_params"]
    target = module.target_freeze_info

    assert target["train_depth"] == "last_n:3"
    assert len(target["trainable_stages"]) == 3
    assert head_params < target["trainable_params"] < full_params

    assert module._maybe_unfreeze_for_fine_tune(1) is True
    assert module.freeze_info["train_depth"] == "last_n:3"
    assert len(module.freeze_info["trainable_stages"]) == 3
    assert _trainable_params(module.model) == target["trainable_params"]


def test_freeze_batchnorm_preserves_running_stats_in_trainable_stages():
    import torch
    from torch import nn

    from sed.train.lit_module import SedLitModule

    cfg = _lit_cfg("last_two", warmup=0)
    cfg.training.freeze_batchnorm = True
    module = SedLitModule(cfg, ["_negative", "glass"])
    batchnorms = [
        sub for sub in module.model.backbone.modules()
        if isinstance(sub, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]
    assert batchnorms
    assert any(
        bn.running_mean is not None
        for bn in batchnorms
    )

    # Reproduce Lightning's epoch transition: train() first, then the module's
    # hook must put all requested encoder BN layers back in eval mode.
    module.train()
    module.on_train_epoch_start()
    assert all(not bn.training for bn in batchnorms)
    assert all(not p.requires_grad for bn in batchnorms for p in bn.parameters())

    before = [
        bn.running_mean.detach().clone() if bn.running_mean is not None else None
        for bn in batchnorms
    ]
    with torch.no_grad():
        module.model(torch.randn(4, 1, 128, 100))
    after = [bn.running_mean for bn in batchnorms]
    for old, new in zip(before, after, strict=True):
        if old is not None:
            assert torch.equal(old, new)
