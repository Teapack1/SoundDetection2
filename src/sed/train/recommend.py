"""Heuristic hyperparameter recommendations for the Train Lab.

Given the composed dataset (size + class balance), the chosen feature, and the
backbone (pretrained vs from-scratch, size) + train depth, propose sensible
starting hyperparameters with a one-line rationale per field. These are
*suggestions* the operator can accept or override — not a tuner.
"""

from __future__ import annotations

from typing import Any

from sed.config import BACKBONES


def _backbone_is_large(backbone: str) -> bool:
    """Rough 'is this a heavy net' test, for batch-size sizing."""
    meta = BACKBONES.get(str(backbone))
    if meta is not None:
        return float(meta.get("feat_dim", 0)) >= 1500 or float(meta.get("width", 0)) >= 2.0
    # Non-mn (timm/custom): consult the registry feat_dim if available.
    try:
        from sed.train.backbones import all_specs
        spec = all_specs().get(str(backbone))
        if spec and spec.feat_dim:
            return int(spec.feat_dim) >= 1500
    except Exception:  # noqa: BLE001
        pass
    return False


def _policy_phrase(policy: str) -> str:
    """Human phrase for a policy id, safe for parametric ``last_n:k`` ids."""
    from sed.train.fine_tuning import last_n_stage_count

    k = last_n_stage_count(policy)
    if k is not None:
        return f"the last {k} stages"
    return {
        "head": "the head only",
        "last_stage": "the last stage",
        "last_two": "the last two stages",
        "full": "the full encoder",
    }.get(policy, policy.replace("_", " "))


def scaled_partial_policy(n_stages: int, fraction: float, min_stages: int = 1) -> str:
    """Model-aware partial depth: train a fixed share of the encoder stages.

    Stage granularity differs wildly between adapters (EfficientAT MN exposes 17
    per-block groups, ConvNeXt 5 coarse stages), so a fixed "last 2" means ~12%
    of one model and 40% of another. Scaling by a fraction of the probed stage
    count keeps the recommendation meaning the same thing on every model.
    """
    k = max(int(min_stages), int(round(int(n_stages) * float(fraction))))
    return f"last_n:{k}"


#: End-to-end LR attenuation from the deepest trainable stage down to the
#: shallowest one. Expressing layer-wise decay as a *ratio* instead of a
#: per-stage constant is what makes it mean the same thing on every model:
#: a fixed 0.75 costs an EfficientAT MN (17 stage groups) 0.75**16 ≈ 1% of the
#: backbone LR at the stem — effectively freezing it — while costing a 5-stage
#: ConvNeXt only 0.75**4 ≈ 32%. Same knob, two different recipes.
FULL_DEPTH_LR_RATIO = 0.10
PARTIAL_DEPTH_LR_RATIO = 0.50


def layer_decay_for_ratio(trainable_stages: int | None, ratio: float) -> float:
    """Per-stage LR decay that yields ``ratio`` from the top stage to the bottom.

    ``trainable_stages`` is how many encoder stages the fine-tune policy will
    actually open — frozen stages take no gradient, so their LR is irrelevant
    and must not enter the exponent. Returns 1.0 (uniform LR) when fewer than
    two stages train, and is clamped so a two-stage span cannot produce an
    extreme per-stage drop.
    """
    k = int(trainable_stages or 0)
    if k <= 1:
        return 1.0
    return round(min(0.95, max(0.55, float(ratio) ** (1.0 / (k - 1)))), 4)


def _policy_to_depth(policy: str, n_stages: int | None = None) -> str:
    """Map a curated/parametric fine-tune policy to a ``train_depth`` string."""
    from sed.train.fine_tuning import (
        canonicalize_fine_tune_policy_for_stages,
        last_n_stage_count,
    )

    p = canonicalize_fine_tune_policy_for_stages(policy or "", n_stages=n_stages)
    if p == "head":
        return "head"
    if p == "last_stage":
        return "last_block"
    if p == "last_two":
        return "last_n:2"
    if p == "full":
        return "full"
    k = last_n_stage_count(p)
    return f"last_n:{k}" if k is not None else "full"


def fine_tune_depth_label(policy: str, n_stages: int | None = None) -> str:
    """Human-readable fine-tune depth, including stage count when known."""
    from sed.train.fine_tuning import (
        canonicalize_fine_tune_policy_for_stages,
        last_n_stage_count,
    )

    p = canonicalize_fine_tune_policy_for_stages(policy or "", n_stages=n_stages)
    k = last_n_stage_count(p)
    if p == "head":
        return (
            f"head only (0 of {int(n_stages)} encoder stages)"
            if n_stages else
            "head only (freeze encoder)"
        )
    if p == "full":
        return (
            f"full fine-tune ({int(n_stages)} of {int(n_stages)} encoder stages)"
            if n_stages else
            "full fine-tune (all encoder stages)"
        )
    if p == "last_stage":
        k = 1
    elif p == "last_two":
        k = 2
    if k is not None:
        if n_stages:
            k = max(1, min(int(k), int(n_stages)))
            return f"last {k} of {int(n_stages)} encoder stages"
        return f"last {k} encoder stage{'s' if int(k) != 1 else ''}"
    return "full fine-tune (all encoder stages)"


def recommend_hparams(
    *,
    backbone: str,
    pretrained: bool,
    train_depth: str,
    fine_tune_policy: str = "",
    force_policy: bool = False,
    by_class: dict[str, int],
    n_train: int,
    classes: list[str],
    n_stages: int | None = None,
    n_groups: int | None = None,
) -> dict[str, Any]:
    """Return ``{"values": {...}, "reasons": {...}}`` of suggested training knobs.

    When ``force_policy`` is set and ``fine_tune_policy`` is a valid transfer
    policy, the requested policy is honored (used by the named presets); scratch
    encoders always train fully regardless. Otherwise the policy is chosen from
    the dataset size heuristics.

    ``n_groups`` is the smallest per-class count of independent recording groups
    (see :mod:`sed.data.sessions`). It matters more than the row count: a corpus
    with many clips but few groups is redundant, and the regularization advice
    keys off that redundancy rather than off the backbone name.
    """
    from sed.train.fine_tuning import (
        canonicalize_fine_tune_policy_for_stages,
        fine_tune_policy_stage_count,
        last_n_stage_count,
        normalize_fine_tune_policy,
    )

    # ``pretrained`` here means "the Train tab resolved this run to start from
    # non-random encoder weights" (AudioSet upstream or a parent encoder). Do not
    # re-check built-in AudioSet availability here: parent transfers can exist for
    # custom/backbone variants that have no bundled upstream weights.
    has_weights = bool(pretrained)
    large = _backbone_is_large(backbone)
    values: dict[str, Any] = {}
    reasons: dict[str, str] = {}

    stage_known = bool(n_stages and int(n_stages) > 0)
    if not has_weights:
        policy = "full"
        reasons["fine_tune_policy"] = "random/scratch encoders must train all stages"
    elif n_train and n_train < 300:
        policy = scaled_partial_policy(int(n_stages), 0.10, 1) if stage_known else "last_stage"
        reasons["fine_tune_policy"] = (
            "tiny transfer set → adapt only ~10% of the encoder stages after head warm-up"
            if stage_known else
            "tiny transfer set → train only the last encoder stage after head warm-up"
        )
    elif n_train and n_train >= 10000:
        policy = "full"
        reasons["fine_tune_policy"] = "large labelled set → full fine-tune can adapt the whole encoder"
    else:
        policy = scaled_partial_policy(int(n_stages), 0.30, 2) if stage_known else "last_two"
        reasons["fine_tune_policy"] = (
            "default supervised transfer: adapt roughly the last third of the pretrained encoder"
            if stage_known else
            "default supervised transfer recipe for pretrained audio encoders"
        )
    depth = _policy_to_depth(policy, n_stages=n_stages)
    if has_weights:
        requested_by_heuristic = policy
        policy = canonicalize_fine_tune_policy_for_stages(policy, n_stages=n_stages)
        depth = _policy_to_depth(policy, n_stages=n_stages)
        if policy != requested_by_heuristic and policy == "full":
            reasons["fine_tune_policy"] = (
                f"{_policy_phrase(requested_by_heuristic)} covers every stage "
                "on this model, so use full fine-tune"
            )
    requested = normalize_fine_tune_policy(fine_tune_policy or "")
    requested_valid = (
        requested in {"head", "last_stage", "last_two", "full"}
        or last_n_stage_count(requested) is not None
    )
    if force_policy and requested_valid and has_weights:
        policy = canonicalize_fine_tune_policy_for_stages(requested, n_stages=n_stages)
        depth = _policy_to_depth(policy, n_stages=n_stages)
        if policy != requested and policy == "full":
            reasons["fine_tune_policy"] = (
                f"preset requested {_policy_phrase(requested)}, which covers "
                "every stage on this model"
            )
        else:
            reasons["fine_tune_policy"] = f"forced by preset: train {_policy_phrase(policy)}"
    values["fine_tune_policy"] = policy
    reasons["fine_tune_policy"] = (
        f"{fine_tune_depth_label(policy, n_stages)} — {reasons['fine_tune_policy']}"
    )
    values["fine_tune_warmup_epochs"] = 0
    reasons["fine_tune_warmup_epochs"] = (
        "0 lets the trainer use the bounded automatic head warm-up"
        if has_weights and policy not in {"head"} else
        "no separate head warm-up needed"
    )

    # Batch size — smaller for heavy nets to fit memory.
    values["batch_size"] = 32 if large else 64
    reasons["batch_size"] = ("heavier backbone → smaller batch to fit memory"
                             if large else "default for compact backbones")

    # Epochs + warmup — this is a max-epoch budget, not a promise to run every
    # epoch when validation/early-stopping is available. Keep transfer presets
    # short enough to be usable, but give scratch and full-depth adaptation more
    # room.
    depth_k = last_n_stage_count(depth)
    if not has_weights:
        epochs, warmup = 80, 8
        reasons["epochs"] = "scratch encoder needs a larger max-epoch budget"
    elif depth in ("head", "head_only"):
        epochs, warmup = 18, 1
        reasons["epochs"] = "head-only transfer is a quick linear-probe budget"
    elif depth in ("last_block",):
        epochs, warmup = 28, 3
        reasons["epochs"] = "single-stage transfer usually converges quickly"
    elif depth == "full":
        epochs, warmup = 45, 5
        reasons["epochs"] = "full transfer gets more room to adapt safely"
    else:
        epochs, warmup = 35, 4
        reasons["epochs"] = "last-stage transfer budget with early stopping"
    if n_train and n_train < 300:
        epochs = int(round(epochs * 1.25))
        reasons["epochs"] += "; +25% for a tiny set (early stopping protects)"
    elif n_train and n_train < 2000:
        epochs = int(round(epochs * 1.15))
        reasons["epochs"] += "; +15% for a small set"
    elif has_weights and n_train and n_train >= 20000:
        epochs = max(25, int(round(epochs * 0.85)))
        reasons["epochs"] += "; large corpora need fewer repeated passes"
    values["epochs"] = epochs
    values["warmup_epochs"] = warmup
    reasons["warmup_epochs"] = "linear LR warmup before cosine decay"

    # Learning rates — neutral defaults keep the head close to the encoder
    # instead of letting a fresh classifier rewrite the run before the backbone
    # has adapted. Scratch/legacy-style models use one LR for the whole network.
    if not has_weights:
        values["lr_backbone"] = 3e-4
        values["lr_head"] = 3e-4
        reasons["lr_backbone"] = "scratch/legacy-style training uses one moderate whole-network LR"
        reasons["lr_head"] = "same as backbone for scratch/legacy-style training"
    elif depth in ("head", "head_only"):
        values["lr_backbone"] = 1e-5
        reasons["lr_backbone"] = "backbone frozen — LR is nearly irrelevant"
        values["lr_head"] = 3e-4
        reasons["lr_head"] = "moderate head LR for a quick frozen-encoder probe"
    elif depth == "full":
        values["lr_backbone"] = 3e-5
        values["lr_head"] = 1e-4
        reasons["lr_backbone"] = "low full-transfer LR preserves transferred features"
        reasons["lr_head"] = "modest head boost while the full encoder adapts"
    else:
        values["lr_backbone"] = 5e-5
        values["lr_head"] = 1e-4
        reasons["lr_backbone"] = "gentle partial-transfer LR for the unfrozen encoder stages"
        reasons["lr_head"] = "modest head boost, not a 10x faster head"
    # How many encoder stages this policy actually opens. Everything that has to
    # mean the same thing on a 5-stage ConvNeXt and a 17-stage MN — layer-wise LR
    # decay, the BatchNorm decision — is derived from this rather than from the
    # depth string, which carries no scale.
    trainable_stages = fine_tune_policy_stage_count(policy, n_stages=n_stages)
    depth_is_deep = depth in ("half", "full") or (depth_k is not None and depth_k >= 2)
    if has_weights and depth_is_deep:
        ratio = FULL_DEPTH_LR_RATIO if depth == "full" else PARTIAL_DEPTH_LR_RATIO
        decay = layer_decay_for_ratio(trainable_stages, ratio)
        if decay >= 1.0:
            # Stage count unknown (adapter could not report one): fall back to the
            # historical per-stage constants rather than silently dropping decay.
            decay = 0.75 if depth == "full" else 0.85
            values["lr_layer_decay"] = decay
            reasons["lr_layer_decay"] = "decay LR toward earlier stages to protect low-level features"
        else:
            values["lr_layer_decay"] = decay
            reasons["lr_layer_decay"] = (
                f"{decay:.2f} per stage — the shallowest of the {int(trainable_stages)} "
                f"trainable stages ends at {ratio:.0%} of the backbone LR, so early "
                "feature detectors are protected by the same amount on any model"
            )
    else:
        values["lr_layer_decay"] = 1.0
        reasons["lr_layer_decay"] = "uniform backbone LR"

    # Loss + class balance. BCE keeps train/eval/export on the same independent-
    # class sigmoid contract. Frequency correction belongs in one place only: the
    # shared group-aware sampler, never a second set of objective-specific weights.
    values["loss"] = "bce_logits"
    reasons["loss"] = "default multi-label BCE/sigmoid objective"
    present_by_class = {
        class_name: int(by_class.get(class_name, 0))
        for class_name in dict.fromkeys(classes)
        if int(by_class.get(class_name, 0)) > 0
    }
    present_counts = list(present_by_class.values())
    if len(present_counts) < 2:
        values["class_balance"] = "off"
        reasons["class_balance"] = "fewer than two represented classes; keep natural row frequency"
    else:
        # `_negative` is a background pool, not another material category.
        # Match the sampler's hierarchy: compare background with all targets,
        # then compare target classes with each other. This avoids reporting a
        # perfectly distributed 50/50 corpus as imbalanced merely because its
        # target half is split across several labels.
        target_counts = [
            count for name, count in present_by_class.items() if name != "_negative"
        ]
        ratios: list[float] = []
        negative_count = present_by_class.get("_negative", 0)
        if negative_count and target_counts:
            target_total = sum(target_counts)
            ratios.append(max(negative_count, target_total) / min(negative_count, target_total))
        if len(target_counts) >= 2:
            ratios.append(max(target_counts) / min(target_counts))
        if not negative_count:
            ratios.append(max(present_counts) / min(present_counts))
        count_ratio = max(ratios, default=1.0)
        if count_ratio <= 1.25:
            values["class_balance"] = "off"
            reasons["class_balance"] = (
                "background/target and target-class counts are within 1.25x; "
                "keep natural row frequency"
            )
        else:
            values["class_balance"] = "sqrt"
            reasons["class_balance"] = (
                f"temper the {count_ratio:.2f}x class-count range without fully equalizing it"
            )

    # SpecAugment + mixup + weight decay, chosen from the data rather than from
    # a per-backbone table, so the recommendation is meaningful for any model.
    #
    # The decisive quantity is how much *independent* material there is. A corpus
    # can hold 70k clips drawn from a few dozen recording groups; the model then
    # memorizes those conditions within a couple of epochs regardless of size.
    # ``n_groups`` is the smallest per-class independent-recording-group count
    # (see sed.data.sessions); clips per group measures the redundancy.
    rows = int(n_train or 0)
    groups = int(n_groups or 0)
    per_group = (rows / groups) if (rows and groups) else 0.0
    redundant = per_group >= 150.0
    scratch = not has_weights
    # A scratch encoder learns every feature from this corpus, so it has the most
    # to memorize; a shallow transfer barely adapts and needs the least.
    wants_regularization = rows >= 500 and (scratch or redundant or depth == "full")

    values["specaugment_enabled"] = bool(wants_regularization)
    if not wants_regularization:
        reasons["specaugment_enabled"] = (
            "small or shallow-transfer run; masking would remove signal the model barely sees"
            if rows < 500 else
            "shallow transfer over diverse data does not need extra masking"
        )
    elif scratch and redundant:
        reasons["specaugment_enabled"] = (
            f"scratch encoder over ~{per_group:.0f} clips per independent recording group — "
            "masking is the main defense against memorizing those conditions"
        )
    elif scratch:
        reasons["specaugment_enabled"] = "scratch encoder learns every feature here; masking keeps it honest"
    elif redundant:
        reasons["specaugment_enabled"] = (
            f"~{per_group:.0f} clips per independent recording group is highly redundant"
        )
    else:
        reasons["specaugment_enabled"] = "full-depth training adapts every stage; masking limits memorization"

    # Mixup mainly fixes overconfidence, and only the BCE objective consumes its
    # soft labels. It needs enough data to be worth the slower convergence.
    mixup = 0.2 if (wants_regularization and rows >= 2000) else 0.0
    values["mixup_alpha"] = mixup
    reasons["mixup_alpha"] = (
        "softens overconfident predictions (bce_logits only)" if mixup
        else "off: too little data to pay for slower convergence, or no memorization pressure"
    )

    # Weight decay is the always-on baseline; the 1e-4 default is effectively no
    # constraint for a small encoder on a redundant corpus.
    if wants_regularization and (scratch or redundant):
        values["weight_decay"] = 0.01
        reasons["weight_decay"] = "meaningful L2 constraint; 1e-4 barely regularizes a redundant corpus"
    else:
        values["weight_decay"] = 1e-4
        reasons["weight_decay"] = "light default decay"
    # BatchNorm on a transferred encoder. Running mean/var are estimates, and a
    # few hundred venue clips are a worse estimate of the deployment input
    # distribution than the upstream corpus they would replace — so a small,
    # shallow transfer keeps the inherited statistics and a large or full-depth
    # run re-learns them.
    #
    # Decide from the SHARE of the encoder being adapted, never from the depth
    # string: `last_two` and every scaled policy spell themselves "last_n:k", so
    # a membership test against the legacy names silently excuses every
    # multi-stage model from freezing BN on a small set.
    if not has_weights:
        adapting_share = 1.0
    elif trainable_stages is not None and n_stages and int(n_stages) > 0:
        adapting_share = float(trainable_stages) / float(int(n_stages))
    else:
        adapting_share = 1.0 if depth == "full" else 0.5
    values["freeze_batchnorm"] = bool(
        has_weights and adapting_share < 0.75 and (not n_train or n_train < 5000)
    )
    reasons["freeze_batchnorm"] = (
        f"only {adapting_share:.0%} of the encoder adapts on {int(n_train or 0)} clips — "
        "keep the inherited running statistics rather than re-estimating them from a "
        "small, single-condition sample"
        if values["freeze_batchnorm"]
        else "let BatchNorm adapt: enough data, or enough of the encoder is moving, "
             "to re-estimate its statistics honestly"
    )

    # Frozen BN means the shipped running statistics were measured on somebody
    # else's data while the weights above them moved; mixup means they were
    # measured on convex blends that never occur at inference. Both leave the
    # deployed model normalizing with the wrong constants, and both are repaired
    # by the same post-fit pass — so recommend it exactly when one of them holds.
    values["bn_recalibrate"] = bool(values["freeze_batchnorm"] or float(values["mixup_alpha"]) > 0)
    reasons["bn_recalibrate"] = (
        "BatchNorm was frozen, so its running statistics still describe the parent's "
        "data — re-measure them on this project before export"
        if values["freeze_batchnorm"] and float(values["mixup_alpha"]) <= 0 else
        "mixup trains on convex blends the deployed model never sees — re-measure "
        "BatchNorm on un-mixed clips before export"
        if float(values["mixup_alpha"]) > 0 else
        "BatchNorm learned its statistics on this data already"
    )

    values["patience"] = max(6, int(round(values["epochs"] * 0.22)))
    reasons["patience"] = "early-stopping patience ≈ 20–25% of the epoch budget"

    # Precision is a memory/stability trade, not a speed one: without Tensor
    # Cores fp16 buys little throughput, but it still roughly halves activation
    # memory, which is what actually limits batch size on a small card. Keep the
    # advice general — it does not depend on which backbone is selected — and let
    # "Compute estimate" measure the real peak instead of guessing.
    values["precision"] = "16-mixed"
    reasons["precision"] = (
        "mixed precision roughly halves activation memory, which is what limits "
        "batch size; switch to 32-true only if training produces NaNs"
    )

    if str(backbone) == "clink-cnn-v2" and not has_weights:
        # Domain pretraining is deliberately short and conservative: v1's best
        # checkpoint was its first epoch on the current low-group-diversity
        # corpus. Group-aware sampling supplies the important regularization;
        # masking stays light enough to preserve narrow glass resonances.
        values.update({
            "epochs": 40,
            "warmup_epochs": 3,
            "patience": 8,
            "batch_size": 64,
            "lr_backbone": 3e-4,
            "lr_head": 3e-4,
            "weight_decay": 1e-3,
            "class_balance": "balanced",
            "specaugment_enabled": True,
            "specaugment_freq_mask": 8,
            "specaugment_time_mask": 6,
            "mixup_alpha": 0.1,
            "bn_recalibrate": True,
        })
        reasons.update({
            "epochs": "40-epoch domain-pretraining ceiling; early stopping guards against v1-style overfit",
            "warmup_epochs": "three-epoch LR warmup for the compact scratch encoder",
            "patience": "stop eight checks after the fixed-threshold operating point stops improving",
            "batch_size": "compact encoder is designed for batch 64 without gradient checkpointing",
            "lr_backbone": "moderate whole-network AdamW LR for random initialization",
            "lr_head": "same LR as the scratch encoder",
            "weight_decay": "moderate AdamW decay; diversity-aware sampling carries most regularization",
            "class_balance": "equalize negative/target exposure while tempering recording groups",
            "specaugment_enabled": "light task-tailored masking regularizes session and venue signatures",
            "specaugment_freq_mask": "mask at most 8 of 128 bands so narrow resonances survive",
            "specaugment_time_mask": "mask at most 6 of 81 frames so attack/decay remains visible",
            "mixup_alpha": "light BCE-compatible mixup for domain-pretraining regularization",
            "bn_recalibrate": "re-measure BatchNorm on un-mixed clips so the exported parent normalizes correctly",
        })
    elif str(backbone) == "clink-cnn-v2" and has_weights and (not n_train or n_train < 10000):
        # A ClinkCNN-v2 transfer can only be parent-initialized. Keep the first
        # two epochs head-only, then adapt exactly the final two of its nine
        # balanced stages while preserving upstream BN statistics.
        values.update({
            "fine_tune_policy": "last_two",
            "fine_tune_warmup_epochs": 2,
            "epochs": 40,
            "warmup_epochs": 2,
            "patience": 8,
            "batch_size": 32,
            "lr_backbone": 3e-5,
            "lr_head": 1e-4,
            "weight_decay": 1e-3,
            "class_balance": "balanced",
            "freeze_batchnorm": True,
            "specaugment_enabled": True,
            "specaugment_freq_mask": 4,
            "specaugment_time_mask": 4,
            "mixup_alpha": 0.0,
            "bn_recalibrate": True,
        })
        reasons.update({
            "fine_tune_policy": "small parent transfer: train the last 2 of 9 encoder stages",
            "fine_tune_warmup_epochs": "train only the fresh classifier for the first two epochs",
            "epochs": "40-epoch small-dataset transfer ceiling with early stopping",
            "warmup_epochs": "brief optimizer warmup aligned with the head-only phase",
            "patience": "eight validation checks tolerate small-set variance without a long overfit tail",
            "batch_size": "batch 32 leaves room for venue/noise overlays during small-set transfer",
            "lr_backbone": "gentle LR for the final two transferred encoder stages",
            "lr_head": "modest boost for the fresh glass-category classifier",
            "weight_decay": "moderate AdamW decay for a small transfer set",
            "class_balance": "equalize negative/target exposure without repeating one recording group excessively",
            "freeze_batchnorm": "retain parent BatchNorm statistics on a small child dataset",
            "specaugment_enabled": "retain only light masks alongside dataset-level venue/noise overlays",
            "specaugment_freq_mask": "light four-band masking preserves material resonances",
            "specaugment_time_mask": "light four-frame masking preserves the transient envelope",
            "mixup_alpha": "off for small-set material discrimination",
            "bn_recalibrate": "parent BatchNorm statistics were kept frozen — re-measure them on this venue before export",
        })

    return {"values": values, "reasons": reasons}


def _put(rec: dict[str, Any], field: str, value: Any, reason: str) -> None:
    rec["values"][field] = value
    rec["reasons"][field] = reason


def _patience(epochs: int, *, minimum: int = 6, ratio: float = 0.22) -> int:
    return max(int(minimum), int(round(int(epochs) * float(ratio))))


def _apply_preset_profile(
    rec: dict[str, Any],
    preset_id: str,
    *,
    has_weights: bool,
    n_train: int,
) -> None:
    """Tune the generic recommendation into the named preset's actual intent."""
    if preset_id == "venue_finetune" and has_weights:
        epochs = 40 if n_train and n_train < 2000 else 34
        _put(rec, "epochs", epochs, "small venue transfer budget; early stopping is expected")
        _put(rec, "warmup_epochs", 3, "brief warmup before cosine decay")
        _put(rec, "patience", _patience(epochs, minimum=7), "moderate patience for venue adaptation")
        _put(rec, "lr_backbone", 5e-5, "gentle LR for venue adaptation without washing out transferred features")
        _put(rec, "lr_head", 1e-4, "modest head boost for stable transfer")
        # lr_layer_decay is left at the generic recommendation on purpose: it is
        # derived from the number of stages this policy actually opens, so it
        # already means "mild stage decay" on any backbone.
        _put(rec, "freeze_batchnorm", True, "small venue fine-tunes should keep BN stats stable")
        _put(rec, "auto_test", True, "score the frozen test so the run lands comparable in Runs & compare")
        _put(rec, "calibrate_thresholds", True,
             "score the calibrated decision layer, not a one-size threshold — small fine-tunes shift probability scales")
        _put(rec, "bn_recalibrate", True,
             "frozen-BN fine-tunes deploy better with running stats re-measured on venue data")
        _put(rec, "calibrate_after", True,
             "deployment step: sweep thresholds on venue negatives to the target FP/h")
        return

    if preset_id == "venue_probe" and has_weights:
        # Head-only: the encoder is frozen, so the schedule only has to fit a
        # linear classifier. Short, higher head LR, and nothing that would move
        # encoder weights or their BatchNorm statistics.
        epochs = 25 if n_train and n_train < 2000 else 20
        _put(rec, "epochs", epochs, "only a classifier head is being fitted — a short budget suffices")
        _put(rec, "warmup_epochs", 1, "almost no warmup needed when the encoder is frozen")
        _put(rec, "patience", _patience(epochs, minimum=5), "a head converges quickly")
        _put(rec, "lr_head", 3e-4, "the head is the only thing learning, so it can move faster")
        # Freezing comes from `policy: "head"` (apply_train_depth marks every
        # encoder stage non-trainable), not from a zero LR — the backbone value
        # is simply never applied, so it stays at the generic recommendation.
        _put(rec, "lr_layer_decay", 1.0, "no per-stage decay applies with a frozen encoder")
        _put(rec, "freeze_batchnorm", True, "a frozen encoder must keep its upstream BatchNorm statistics")
        _put(rec, "bn_recalibrate", False, "nothing in the encoder changed, so its running stats are still valid")
        _put(rec, "auto_test", True, "score the frozen test so the run lands comparable in Runs & compare")
        _put(rec, "calibrate_thresholds", True,
             "a head-only fit shifts probability scales — score the calibrated decision layer")
        _put(rec, "calibrate_after", True,
             "deployment step: sweep thresholds on venue negatives to the false-alarm budget")
        return

    if preset_id == "domain_pretrain_scratch":
        # The flagship pretraining run: a large corpus, random init, and the
        # result becomes the parent every venue fine-tune starts from. Numbers
        # follow the scratch recipe; the separate id exists because the label
        # "train from scratch (custom/legacy architectures)" steered people away
        # from the run that matters most.
        epochs = 60 if (n_train and n_train >= 20000) else 80
        _put(rec, "epochs", epochs, "domain pretraining needs a long budget; early stopping ends it sooner")
        _put(rec, "warmup_epochs", 8, "long LR warmup stabilizes early scratch training")
        _put(rec, "patience", _patience(epochs, minimum=10, ratio=0.2), "scratch convergence is slow")
        _put(rec, "lr_backbone", 3e-4, "one moderate whole-network LR — random weights have nothing to protect")
        _put(rec, "lr_head", 3e-4, "same as backbone: the head is not special when everything is random")
        _put(rec, "lr_layer_decay", 1.0, "uniform LR — no pretrained low-level features to preserve")
        _put(rec, "freeze_batchnorm", False, "BatchNorm must learn its statistics from scratch")
        _put(rec, "specaugment_enabled", True, "masking is the cheapest invariance on a large corpus")
        _put(rec, "mixup_alpha", 0.3, "mixup regularizes against near-duplicate corpus clips (BCE objective)")
        _put(rec, "bn_recalibrate", True,
             "this run becomes the parent for every fine-tune — ship it with BatchNorm "
             "statistics measured on un-mixed clips")
        _put(rec, "auto_test", True, "score the frozen test so the run lands comparable in Runs & compare")
        _put(rec, "calibrate_thresholds", True,
             "scratch probability scales are uncalibrated — score the deployed decision layer")
        return

    if preset_id == "domain_pretrain":
        if has_weights:
            epochs = 40 if n_train and n_train >= 10000 else 35
            _put(rec, "epochs", epochs,
                 "budget sized so cosine decay actually anneals before early stopping ends the run")
            _put(rec, "warmup_epochs", 3, "brief LR ramp; the head warm-up already precedes the encoder unfreeze")
            _put(rec, "patience", _patience(epochs, minimum=8, ratio=0.2), "patience on val AP for slower full-depth convergence")
            _put(rec, "lr_backbone", 3e-5, "low full-transfer LR for stable domain adaptation")
            _put(rec, "lr_head", 1e-4, "modest head boost while the encoder adapts")
            # Stage-count-aware full-depth decay comes from the generic path.
            _put(rec, "freeze_batchnorm", False, "large/full-domain runs should let BN adapt")
            _put(rec, "specaugment_enabled", True,
                 "a full fine-tune memorizes session/venue signatures without masking noise")
            _put(rec, "mixup_alpha", 0.3,
                 "mixup regularizes the full encoder against near-duplicate corpus clips")
            _put(rec, "bn_recalibrate", True,
                 "mixup-trained BatchNorm statistics describe convex blends — re-measure them on real clips")
            _put(rec, "auto_test", True, "score the frozen test so the run lands comparable in Runs & compare")
            _put(rec, "calibrate_thresholds", True,
                 "mixup-trained BCE probabilities are soft — score the calibrated decision layer")
        return

    if preset_id == "scratch_full":
        # Scratch training regardless of dataset size: nothing to preserve, so a
        # single moderate LR, a long budget, and strong regularization.
        epochs = 60 if (n_train and n_train >= 20000) else 80
        _put(rec, "epochs", epochs, "scratch training needs a long max-epoch budget; early stopping ends it sooner")
        _put(rec, "warmup_epochs", 8, "long LR warmup stabilizes early scratch training")
        _put(rec, "patience", _patience(epochs, minimum=10, ratio=0.2), "scratch convergence is slow — give val AP room to improve")
        _put(rec, "lr_backbone", 3e-4, "one moderate whole-network LR — random weights have nothing to protect")
        _put(rec, "lr_head", 3e-4, "same as backbone: the head is not special when everything is random")
        _put(rec, "lr_layer_decay", 1.0, "uniform LR — no pretrained low-level features to preserve")
        _put(rec, "freeze_batchnorm", False, "BatchNorm must learn its statistics from scratch")
        _put(rec, "specaugment_enabled", True, "scratch models overfit fastest; masking is the cheapest invariance")
        _put(rec, "mixup_alpha", 0.2, "mild mixup regularizes scratch training (applies with the BCE objective)")
        _put(rec, "bn_recalibrate", True,
             "re-measure BatchNorm on un-mixed clips so the exported model normalizes like it will at inference")
        _put(rec, "auto_test", True, "score the frozen test so the run lands comparable in Runs & compare")
        _put(rec, "calibrate_thresholds", True,
             "scratch probability scales are uncalibrated — score the deployed decision layer")
        return

    if preset_id == "ssl_byol":
        # Only knobs BYOL actually reads: its optimizer has no LR schedule and
        # no head/layer-decay groups (see ByolLitModule.configure_optimizers).
        epochs = 80 if not (n_train and n_train >= 50000) else 100
        _put(rec, "epochs", epochs, "BYOL needs a real representation-learning budget")
        _put(rec, "patience", 12, "validation BYOL loss can improve slowly")
        _put(rec, "lr_backbone", 1e-4 if has_weights else 3e-4,
             "conservative LR for pretrained BYOL; higher when learning from scratch")
        _put(rec, "class_balance", "off", "BYOL ignores labels and keeps natural row frequency")
        _put(rec, "freeze_batchnorm", False, "SSL should learn its own normalization statistics")
        _put(rec, "ssl_ema_decay", 0.996, "default BYOL teacher EMA decay")
        _put(rec, "auto_test", False, "BYOL has no classifier to test — the linear probe reports quality")
        _put(rec, "calibrate_thresholds", False, "no supervised decision layer to calibrate")
        _put(rec, "bn_recalibrate", False, "the SSL linear probe evaluates the raw representation")
        _put(rec, "calibrate_after", False, "no deployable classifier after a BYOL run")


# Named strategy presets shown in the Train Lab. Each one forces a fine-tune
# *strategy*; the concrete numbers still come from ``recommend_hparams`` so they
# stay adapted to the current dataset. ``overrides`` maps a field to
# ``(value, reason)`` applied on top of the recommendation.
#
# ``requires_init`` gates a preset to the model initialization it is designed
# for: "transfer" needs non-random encoder weights (upstream or parent),
# "scratch" needs random initialization, "any" always applies. The Train tab
# disables presets whose requirement the current model does not meet, so a
# transfer recipe is never silently downgraded to full scratch training.
TRAINING_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "auto",
        "label": "Recommended (from your data)",
        "policy": "",  # empty → data-driven policy selection
        "requires_init": "any",
        "when": "Default. Picks depth, LR, epochs and loss from dataset size, class "
                "balance and the model initialization (pretrained vs scratch).",
        "overrides": {},
    },
    {
        "id": "venue_finetune",
        "label": "Fine-tune pretrained — small dataset",
        "policy": "last_two",  # fallback when the model's stage count is unknown
        "policy_scale": {"fraction": 0.30, "min_stages": 2},
        "requires_init": "transfer",
        "when": "Pretrained/parent encoder + a few hundred to a few thousand clips. "
                "Gentle transfer: head warm-up, roughly the last third of the encoder "
                "stages (scaled to the selected model), BatchNorm frozen.",
        "overrides": {},
    },
    {
        "id": "venue_probe",
        "label": "Venue probe — frozen encoder, head only",
        "policy": "head",
        "requires_init": "transfer",
        "when": "A few hundred clips from ONE venue day. Freezes the encoder and "
                "fits only the classifier head, so the run learns glass timbre "
                "instead of the room and mic it was recorded with. Trains in "
                "minutes, and a new class can be added later without touching "
                "the encoder.",
        "overrides": {
            "freeze_batchnorm": (True, "a frozen encoder must keep its upstream BatchNorm statistics"),
            "calibrate_thresholds": (True, "a head-only fit needs its decision layer calibrated to be deployable"),
        },
    },
    {
        "id": "domain_pretrain_scratch",
        "label": "Domain pretrain from scratch — large corpus",
        "policy": "full",
        "requires_init": "scratch",
        "when": "A large in-domain corpus and no usable upstream weights — the "
                "flagship pretraining run. Published weights fitted at a "
                "different sample rate, hop and fmax carry a weak prior here, so "
                "training from random init on your own corpus is often stronger. "
                "The resulting run becomes the parent for venue fine-tunes.",
        "overrides": {},
    },
    {
        "id": "domain_pretrain",
        "label": "Full fine-tune pretrained — large corpus / domain shift",
        "policy": "full",
        "requires_init": "transfer",
        "when": "Pretrained/parent encoder + a large labelled corpus or strong domain "
                "shift. Adapts the whole encoder: low LR, layer-wise decay, SpecAugment + mixup.",
        "overrides": {},
    },
    {
        "id": "scratch_full",
        "label": "Train from scratch (random init)",
        "policy": "full",
        "requires_init": "scratch",
        "when": "Random initialization — no pretrained weights (custom features, legacy "
                "or custom architectures). One moderate LR for the whole network, long "
                "epoch budget, SpecAugment + mixup for regularization.",
        "overrides": {},
    },
    {
        "id": "ssl_byol",
        "label": "Self-supervised (BYOL)",
        "policy": "full",
        "requires_init": "any",
        "requires_pretraining_project": True,
        "when": "Learn an encoder from unlabelled audio with BYOL; follow up with a "
                "supervised run (init = from previous run) to train a classifier head.",
        "overrides": {
            "pretraining_objective": ("byol", "self-supervised EMA teacher/student, no labels"),
        },
    },
)

_PRESET_BY_ID = {p["id"]: p for p in TRAINING_PRESETS}


def training_presets() -> list[dict[str, Any]]:
    """Preset catalog (id/label/when/requirements) for the UI dropdown."""
    return [{
        "id": p["id"], "label": p["label"], "when": p["when"],
        "requires_init": p.get("requires_init", "any"),
        "requires_pretraining_project": bool(p.get("requires_pretraining_project", False)),
    } for p in TRAINING_PRESETS]


def preset_is_applicable(
    preset: dict[str, Any],
    *,
    has_transfer_weights: bool,
    is_pretraining_project: bool = True,
) -> bool:
    """Whether a preset matches the current initialization."""
    requires = str(preset.get("requires_init", "any"))
    if requires == "transfer" and not has_transfer_weights:
        return False
    if requires == "scratch" and has_transfer_weights:
        return False
    if bool(preset.get("requires_pretraining_project", False)) and not is_pretraining_project:
        return False
    return True


def preset_hparams(
    preset_id: str,
    *,
    backbone: str,
    pretrained: bool,
    by_class: dict[str, int],
    n_train: int,
    classes: list[str],
    n_stages: int | None = None,
    n_groups: int | None = None,
) -> dict[str, Any]:
    """Resolve a named preset to ``{"values": {...}, "reasons": {...}}``.

    Reuses ``recommend_hparams`` for dataset-aware numbers, forcing the preset's
    strategy, then layers the preset's fixed overrides on top.
    """
    preset = _PRESET_BY_ID.get(str(preset_id))
    if preset is None:
        raise KeyError(f"unknown training preset: {preset_id!r}")
    policy = str(preset["policy"] or "")
    # Partial-depth presets scale to the model's probed stage count so "gentle
    # transfer" means the same share of the encoder on a 17-stage MN as on a
    # 5-stage ConvNeXt; the named policy stays as the unknown-stage fallback.
    scale = preset.get("policy_scale")
    if policy and scale and n_stages and int(n_stages) > 0:
        policy = scaled_partial_policy(
            int(n_stages),
            float(scale.get("fraction", 0.30)),
            int(scale.get("min_stages", 1)),
        )
    rec = recommend_hparams(
        backbone=backbone,
        pretrained=pretrained,
        train_depth=policy,
        fine_tune_policy=policy,
        force_policy=bool(policy),
        n_stages=n_stages,
        n_groups=n_groups,
        by_class=by_class,
        n_train=n_train,
        classes=classes,
    )
    # Every preset carries an explicit learning signal so switching away from the
    # BYOL preset restores supervised (overrides below flip it back to byol).
    rec["values"]["pretraining_objective"] = "supervised"
    rec["reasons"]["pretraining_objective"] = "supervised training from labels"
    _apply_preset_profile(
        rec,
        str(preset["id"]),
        has_weights=bool(pretrained),
        n_train=int(n_train or 0),
    )
    if str(backbone) == "clink-cnn-v2" and str(preset["id"]) == "scratch_full":
        _put(rec, "epochs", 40, "fixed ClinkCNN-v2 domain-pretraining ceiling")
        _put(rec, "warmup_epochs", 3, "three-epoch scratch LR warmup")
        _put(rec, "patience", 8, "stop before a repeat of v1's long overfit tail")
        _put(rec, "batch_size", 64, "compact encoder fits batch 64 without checkpointing")
        _put(rec, "weight_decay", 1e-3, "moderate AdamW decay")
        _put(rec, "class_balance", "balanced", "balance classes and recording-group diversity")
        _put(rec, "specaugment_freq_mask", 8, "mask at most 8 of 128 mel bands")
        _put(rec, "specaugment_time_mask", 6, "mask at most 6 of 81 frames")
        _put(rec, "mixup_alpha", 0.1, "light domain-pretraining mixup")
    if str(backbone) == "clink-cnn-v2" and str(preset["id"]) == "venue_finetune" and pretrained:
        _put(rec, "fine_tune_policy", "last_two", "adapt exactly the last 2 of 9 encoder stages")
        _put(rec, "fine_tune_warmup_epochs", 2, "two head-only warm-up epochs")
        _put(rec, "epochs", 40, "small-dataset transfer ceiling")
        _put(rec, "warmup_epochs", 2, "brief optimizer warmup")
        _put(rec, "patience", 8, "small-set early stopping patience")
        _put(rec, "batch_size", 32, "small transfer batch")
        _put(rec, "lr_backbone", 3e-5, "gentle parent-encoder LR")
        _put(rec, "lr_head", 1e-4, "fresh classifier LR")
        _put(rec, "weight_decay", 1e-3, "moderate AdamW decay")
        _put(rec, "class_balance", "balanced", "balance classes and source groups")
        _put(rec, "freeze_batchnorm", True, "preserve parent BN statistics")
        _put(rec, "specaugment_enabled", True, "retain light spectrogram masking")
        _put(rec, "specaugment_freq_mask", 4, "light frequency masking")
        _put(rec, "specaugment_time_mask", 4, "light temporal masking")
        _put(rec, "mixup_alpha", 0.0, "mixup off for material fine-tuning")
    for field, (value, reason) in preset.get("overrides", {}).items():
        rec["values"][field] = value
        rec["reasons"][field] = reason
    return rec
