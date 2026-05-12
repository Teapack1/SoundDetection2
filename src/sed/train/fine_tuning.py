"""Safe, named fine-tuning plans used by the Train Lab."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

# Parametric "train the last <k> encoder stages" policy id, e.g. "last_n:3".
# The named policies above cover 0/1/2 stages and full; this covers 3..K for the
# model-aware "last N of K" control while keeping the same head warm-up schedule.
_LAST_N_RE = re.compile(r"^last_n[:=](\d+)$")


def last_n_stage_count(policy: str) -> int | None:
    """Return ``k`` for a ``last_n:<k>`` policy id, else ``None``."""
    m = _LAST_N_RE.match(str(policy or "").strip().lower())
    return int(m.group(1)) if m else None


def _known_stage_count(n_stages: int | None) -> int | None:
    if n_stages is None:
        return None
    try:
        k = int(n_stages)
    except (TypeError, ValueError):
        return None
    return k if k > 0 else None


def fine_tune_policy_stage_count(policy: str, n_stages: int | None = None) -> int | None:
    """Number of trainable encoder stages requested by ``policy``.

    ``full`` only has a numeric count when the model's stage count is known.
    """
    p = normalize_fine_tune_policy(policy or "")
    if p == "head":
        return 0
    if p == "last_stage":
        return 1
    if p == "last_two":
        return 2
    if p == "full":
        return _known_stage_count(n_stages)
    return last_n_stage_count(p)


def canonicalize_fine_tune_policy_for_stages(
    policy: str, n_stages: int | None = None,
) -> str:
    """Normalize ``policy`` and collapse over-deep partial depths to ``full``.

    Partial policies are only meaningful when they train fewer than all encoder
    stages. If a model has two stages, "last two" is therefore full fine-tuning.
    """
    p = normalize_fine_tune_policy(policy or "")
    stage_count = _known_stage_count(n_stages)
    if stage_count is None:
        return p
    k = fine_tune_policy_stage_count(p, n_stages=stage_count)
    if k is None:
        return p
    if k <= 0:
        return "head"
    if k >= stage_count:
        return "full"
    if k == 1:
        return "last_stage"
    if k == 2:
        return "last_two"
    return f"last_n:{k}"


@dataclass(frozen=True)
class FineTunePolicy:
    id: str
    label: str
    target_depth: str
    description: str
    transfer_only: bool = False
    min_stages: int = 0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FineTunePlan:
    policy: str
    initial_depth: str
    target_depth: str
    warmup_epochs: int

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


FINE_TUNE_POLICIES: tuple[FineTunePolicy, ...] = (
    FineTunePolicy(
        "head", "head only", "head",
        "Freeze the encoder and train only the new classifier head. Useful as a "
        "linear-probe baseline or with very little labelled data; usually not "
        "the final production setting when the venue differs from the source data.",
        transfer_only=True,
    ),
    FineTunePolicy(
        "last_stage", "last encoder stage", "last_block",
        "Warm up the head, then train the head plus the final encoder stage. A "
        "conservative transfer setting for tiny datasets.",
        transfer_only=True, min_stages=1,
    ),
    FineTunePolicy(
        "last_two", "last two encoder stages", "last_n:2",
        "Warm up the head, then train the head plus the last two encoder stages. "
        "This is the default supervised transfer setting for pretrained audio encoders.",
        transfer_only=True, min_stages=2,
    ),
    FineTunePolicy(
        "full", "full encoder", "full",
        "Train every encoder stage and the head. Required for random/scratch "
        "initialization; also useful for large datasets or strong domain shift.",
    ),
)

_POLICY_BY_ID = {policy.id: policy for policy in FINE_TUNE_POLICIES}
_ALIASES = {
    "head_only": "head",
    "last_block": "last_stage",
    "last": "last_stage",
    "last_n:2": "last_two",
    "last_n=2": "last_two",
    # Historical UI option. Keep old configs loadable, but do not expose it as
    # a curated policy because "half the network" is not a stable research recipe.
    "half": "last_two",
}


def normalize_fine_tune_policy(policy: str) -> str:
    """Return the curated policy id for user/config input.

    ``last_n:<k>`` is canonicalized so the small cases fold into the named
    policies (0→head, 1→last_stage, 2→last_two) and k>=3 stays parametric.
    """
    value = str(policy or "full").strip().lower()
    k = last_n_stage_count(value)
    if k is not None:
        if k <= 0:
            return "head"
        if k == 1:
            return "last_stage"
        if k == 2:
            return "last_two"
        return f"last_n:{k}"
    return _ALIASES.get(value, value)


def _pretrained_weights_available(cfg: Any) -> bool:
    """Whether ``cfg.model.pretrained`` can actually load real weights.

    The adapters silently build a random encoder when the spec has no
    published weights (see ``TimmAdapter.build``), so ``pretrained=True`` on
    such a backbone must not earn a frozen-head warm-up schedule.
    """
    try:
        from sed.train.backbones import get_spec

        return bool(get_spec(str(getattr(cfg.model, "backbone", "") or "")).pretrained_available)
    except Exception:  # noqa: BLE001 — registry unavailable: trust the flag
        return True


def is_transfer_initialization(cfg: Any) -> bool:
    """Whether the encoder starts from meaningful non-random weights."""
    initialization = str(getattr(cfg.model, "initialization", "upstream") or "upstream").lower()
    if initialization == "random":
        return False
    if initialization == "parent":
        return True
    return bool(getattr(cfg.model, "pretrained", False)) and _pretrained_weights_available(cfg)


def available_fine_tune_policies(cfg: Any, *, n_stages: int | None = None) -> list[FineTunePolicy]:
    """Policies that make sense for the current model initialization.

    Random encoders must train fully: a frozen random feature extractor makes
    head-only and partial-stage training meaningless. Transfer initializations
    expose conservative stage-based options, filtered by the stage count when
    the adapter can report it.
    """
    if not is_transfer_initialization(cfg):
        return [_POLICY_BY_ID["full"]]
    stage_count = _known_stage_count(n_stages)
    out: list[FineTunePolicy] = []
    for policy in FINE_TUNE_POLICIES:
        if stage_count is not None:
            requested = fine_tune_policy_stage_count(policy.id, n_stages=stage_count)
            if policy.id != "full" and requested is not None and 0 < requested >= stage_count:
                continue
            if policy.min_stages > stage_count:
                continue
        out.append(policy)
    return out


def coerce_fine_tune_policy(cfg: Any, *, n_stages: int | None = None) -> str:
    """Current policy if available, otherwise the nearest safe curated option."""
    if not is_transfer_initialization(cfg):
        return "full"
    current = canonicalize_fine_tune_policy_for_stages(
        getattr(cfg.training, "fine_tune_policy", "full"),
        n_stages=n_stages,
    )
    # Preserve a valid parametric last_n:k (from the model-aware depth control):
    # it is usable only on a transferred encoder and when k is shallower than
    # the full stage count.
    # Otherwise (scratch, or k > stage count) fall through to a named policy.
    k = last_n_stage_count(current)
    stage_count = _known_stage_count(n_stages)
    if k is not None and (stage_count is None or k < stage_count):
        return current
    available = available_fine_tune_policies(cfg, n_stages=n_stages)
    available_ids = {policy.id for policy in available}
    if current in available_ids:
        return current
    if "last_two" in available_ids:
        return "last_two"
    if "last_stage" in available_ids:
        return "last_stage"
    return "full"


def resolve_fine_tune_plan(cfg: Any) -> FineTunePlan:
    """Convert the stable UI policy into exact, stage-based behaviour.

    Scratch models train fully from epoch zero. Transfer policies use a short
    head warm-up before unfreezing their named stage range; this avoids the
    unreliable "pick arbitrary layers" interaction while retaining a useful
    adaptation schedule.
    """
    policy = normalize_fine_tune_policy(getattr(cfg.training, "fine_tune_policy", "manual"))
    if policy == "manual":
        depth = str(getattr(cfg.training, "train_depth", "full") or "full")
        return FineTunePlan("manual", depth, depth, 0)
    last_n = last_n_stage_count(policy)
    if last_n is None and policy not in _POLICY_BY_ID:
        policy = "full"
    if not is_transfer_initialization(cfg):
        return FineTunePlan("full", "full", "full", 0)
    if policy == "head":
        return FineTunePlan(policy, "head", "head", 0)
    target = f"last_n:{last_n}" if last_n is not None else _POLICY_BY_ID[policy].target_depth
    requested = int(getattr(cfg.training, "fine_tune_warmup_epochs", 0) or 0)
    epochs = max(1, int(cfg.training.epochs))
    auto_warmup = min(5, max(1, round(epochs * 0.1)))
    warmup = min(requested or auto_warmup, max(0, epochs - 1))
    initial = "head" if warmup > 0 else target
    return FineTunePlan(policy, initial, target, warmup)


__all__ = [
    "FineTunePlan",
    "FineTunePolicy",
    "FINE_TUNE_POLICIES",
    "available_fine_tune_policies",
    "canonicalize_fine_tune_policy_for_stages",
    "coerce_fine_tune_policy",
    "fine_tune_policy_stage_count",
    "is_transfer_initialization",
    "last_n_stage_count",
    "normalize_fine_tune_policy",
    "resolve_fine_tune_plan",
]
