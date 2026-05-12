"""Lightning module wrapping the SED model and training logic."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

from sed.audio.features import (
    feature_band_count,
    feature_channels,
    feature_frame_count_for_cfg,
)
from sed.config import Config
from sed.train.backbones import apply_train_depth, describe_train_depth
from sed.train.fine_tuning import resolve_fine_tune_plan
from sed.train.model import AsymmetricFocalLoss, SedModel
from sed.train.objective import (
    BCE_LOSSES,
    is_single_label_loss,
    logits_to_probabilities,
    objective_name,
    targets_to_indices,
)


log = logging.getLogger(__name__)


# Supervised checkpoint selection deliberately measures the production-style
# decision layer at one stable operating point.  Keep these names stable: they
# are persisted in metrics.csv, training_report.json, and manifest.json.
FIXED_SELECTION_THRESHOLD = 0.90
FIXED_SELECTION_RECALL_FLOOR = 0.95
VAL_FIXED_RECALL_KEY = "val/recall_macro_targets_at_0_9"
VAL_FIXED_FP_KEY = "val/positive_triggers_per_1000_negatives_at_0_9"
VAL_FIXED_NEGATIVE_COUNT_KEY = "val/negative_clip_count"
VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY = (
    "val/positive_trigger_count_on_negatives_at_0_9"
)


def fixed_threshold_trigger_counts(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    negative_index: int | None,
    threshold: float = FIXED_SELECTION_THRESHOLD,
    negative_max: float = 0.5,
    margin_to_second_best: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sufficient statistics for fixed-threshold trigger selection.

    This replays the single-window runtime decision contract: the strongest
    positive class is considered, then the negative veto, positive-class margin,
    and shared confidence threshold can abstain.  The returned tuple is
    ``(target_tp, target_support, negative_triggers, negative_clips)``.  Keeping
    counts instead of averaging batches makes the epoch metric independent of
    batch size and straightforward to reduce across distributed workers.
    """
    if probabilities.ndim != 2:
        raise ValueError(
            "fixed-threshold validation expects probabilities shaped (batch, classes)"
        )
    num_classes = int(probabilities.shape[1])
    if targets.ndim == 1:
        target_bin = F.one_hot(targets.long(), num_classes=num_classes).bool()
    elif targets.ndim == 2 and targets.shape == probabilities.shape:
        target_bin = targets >= 0.5
    else:
        raise ValueError(
            "fixed-threshold validation targets must be class indices or match probabilities"
        )

    positive_indices = [
        idx for idx in range(num_classes)
        if negative_index is None or idx != negative_index
    ]
    target_tp = torch.zeros(num_classes, dtype=torch.long, device=probabilities.device)
    target_support = target_bin.sum(dim=0, dtype=torch.long)
    zero = torch.zeros((), dtype=torch.long, device=probabilities.device)
    if not positive_indices or probabilities.shape[0] == 0:
        return target_tp, target_support, zero, zero

    positive_index = torch.tensor(
        positive_indices, dtype=torch.long, device=probabilities.device
    )
    positive_probabilities = probabilities.index_select(1, positive_index)
    winner_column = positive_probabilities.argmax(dim=1)
    row_index = torch.arange(probabilities.shape[0], device=probabilities.device)
    winner_index = positive_index[winner_column]
    winner_score = positive_probabilities[row_index, winner_column]
    triggered = winner_score >= float(threshold)

    if negative_index is not None:
        triggered &= probabilities[:, negative_index] <= float(negative_max)
    if len(positive_indices) > 1 and margin_to_second_best > 0:
        runner_up = positive_probabilities.topk(k=2, dim=1).values[:, 1]
        triggered &= (winner_score - runner_up) >= float(margin_to_second_best)

    predicted = F.one_hot(winner_index, num_classes=num_classes).bool()
    target_tp = (predicted & target_bin & triggered.unsqueeze(1)).sum(
        dim=0, dtype=torch.long
    )

    any_true_target = target_bin.index_select(1, positive_index).any(dim=1)
    if negative_index is None:
        truly_negative = ~any_true_target
    else:
        # A malformed multi-label row that marks both a target and `_negative`
        # is not a truly negative clip and must not inflate the false-trigger
        # denominator.
        truly_negative = target_bin[:, negative_index] & ~any_true_target
    negative_clips = truly_negative.sum(dtype=torch.long)
    negative_triggers = (triggered & truly_negative).sum(dtype=torch.long)
    return target_tp, target_support, negative_triggers, negative_clips


def summarize_fixed_threshold_counts(
    target_tp: torch.Tensor,
    target_support: torch.Tensor,
    negative_triggers: torch.Tensor,
    negative_clips: torch.Tensor,
    *,
    negative_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute target macro recall and negative triggers/1000 from counts.

    Unsupported target classes contribute zero recall, matching the existing
    explicit-class torchmetrics behavior.  A validation split with no truly
    negative clips reports a finite 0.0 rate; the separately logged negative
    count makes that absence explicit in reports instead of leaking a NaN into
    checkpoint selection.
    """
    target_indices = [
        idx for idx in range(int(target_support.numel()))
        if negative_index is None or idx != negative_index
    ]
    # Float64 keeps exact count ratios such as 19/20 aligned with the persisted
    # 0.95 eligibility gate instead of rounding just below it in float32.
    dtype = torch.float64
    if target_indices:
        index = torch.tensor(target_indices, dtype=torch.long, device=target_support.device)
        support = target_support.index_select(0, index).to(dtype)
        true_positives = target_tp.index_select(0, index).to(dtype)
        recall_by_class = torch.where(
            support > 0,
            true_positives / support.clamp_min(1.0),
            torch.zeros_like(support),
        )
        macro_recall = recall_by_class.mean()
    else:
        macro_recall = torch.zeros((), dtype=dtype, device=target_support.device)

    negative_total = negative_clips.to(dtype)
    fp_per_1000 = torch.where(
        negative_total > 0,
        1000.0 * negative_triggers.to(dtype) / negative_total.clamp_min(1.0),
        torch.zeros_like(negative_total),
    )
    return macro_recall, fp_per_1000


def _fmt_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch mixup: convex-combine spectrograms and their soft targets.

    Only valid for the multi-label BCE clip objective, which consumes the soft
    labels directly. Single-label CE and frame targets stay untouched, and the
    asymmetric focal loss hard-thresholds targets at 0.5 — soft mixed labels
    would flip its focusing direction — so mixup is gated to BCE.
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1.0 - lam) * x[perm], lam * y + (1.0 - lam) * y[perm]


def split_decay_param_groups(params: list, lr: float, weight_decay: float) -> list[dict]:
    """AdamW hygiene: biases and norm-affine (1-D) parameters skip weight decay."""
    decayed = [p for p in params if p.ndim > 1]
    undecayed = [p for p in params if p.ndim <= 1]
    groups: list[dict] = []
    if decayed:
        groups.append({"params": decayed, "lr": lr, "weight_decay": weight_decay})
    if undecayed:
        groups.append({"params": undecayed, "lr": lr, "weight_decay": 0.0})
    return groups


class SedLitModule(L.LightningModule):
    def __init__(self, cfg: Config, classes: list[str]):
        super().__init__()
        self.cfg = cfg
        self.classes = classes
        self.num_classes = len(classes)
        self.neg_idx: int | None = (
            classes.index("_negative") if "_negative" in classes else None
        )
        self.single_label = is_single_label_loss(cfg)
        self.frame_task = cfg.model.task == "frame"
        self.mixup_alpha = float(getattr(cfg.training, "mixup_alpha", 0.0) or 0.0)
        if self.frame_task:
            # Frame targets may overlap, so even projects that normally use
            # single-label clip classification train frame SED with BCE.
            self.single_label = False
        self.loss_name = objective_name(cfg)
        self._mixup_active = (
            self.mixup_alpha > 0
            and not self.frame_task
            and self.loss_name in BCE_LOSSES
        )
        if self.mixup_alpha > 0 and not self._mixup_active:
            log.warning(
                "training.mixup_alpha=%s is ignored for loss %r — mixup soft "
                "labels are only consumed by the bce_logits objective",
                self.mixup_alpha,
                self.loss_name,
            )

        self.model = SedModel(
            backbone_name=cfg.model.backbone,
            num_classes=self.num_classes,
            pretrained=cfg.model.pretrained,
            n_bands=feature_band_count(cfg.features, cfg.model.sample_rate),
            n_frames=feature_frame_count_for_cfg(cfg),
            params=dict(cfg.model.params),
            task=cfg.model.task,
            input_channels=feature_channels(cfg.features),
            head_dropout=cfg.training.head_dropout,
        )

        # Block-level freezing ("train depth"): freeze a leading prefix of the
        # backbone and (optionally) hold frozen-stage BatchNorm in eval so its
        # running stats don't drift. The head is always trainable.
        self.fine_tune_plan = resolve_fine_tune_plan(cfg)
        self._fine_tune_unfrozen = self.fine_tune_plan.warmup_epochs == 0
        self.freeze_info, self._frozen_modules = self._apply_train_depth(
            self.fine_tune_plan.initial_depth,
        )
        self.target_freeze_info = describe_train_depth(
            self.model,
            self.fine_tune_plan.target_depth,
            freeze_batchnorm=cfg.training.freeze_batchnorm,
        )
        self.target_freeze_info["fine_tune_plan"] = self.fine_tune_plan.to_payload()

        if self.frame_task:
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.single_label:
            # Class frequency is corrected once, in the model-agnostic
            # dataloader sampler. Keep an explicit all-one buffer so historical
            # checkpoints that stored loss weights remain load-compatible.
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.ones(self.num_classes))
        elif self.loss_name == "asymmetric_focal":
            self.loss_fn = AsymmetricFocalLoss(
                num_classes=self.num_classes,
                negative_index=self.neg_idx,
                gamma_pos=cfg.training.focal_gamma_pos,
                gamma_neg=cfg.training.focal_gamma_neg,
            )
        elif self.loss_name in BCE_LOSSES:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.ones(self.num_classes)
            )
        else:
            raise ValueError(
                "Unsupported training.loss. Use cross_entropy/categorical_crossentropy, "
                "asymmetric_focal, or bce_logits."
            )

        if self.single_label:
            self.val_ap = MulticlassAveragePrecision(num_classes=self.num_classes, average=None)
            self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average=None)
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average=None)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes, average=None)
        else:
            self.val_ap = MultilabelAveragePrecision(num_labels=self.num_classes, average=None)
            self.val_f1 = MultilabelF1Score(num_labels=self.num_classes, average=None,
                                            threshold=0.5)
            self.val_precision = MultilabelPrecision(
                num_labels=self.num_classes, average=None, threshold=0.5
            )
            self.val_recall = MultilabelRecall(
                num_labels=self.num_classes, average=None, threshold=0.5
            )

        # Fixed-threshold selection is count-based rather than a torchmetrics
        # average so it exactly represents all validation clips.  Non-persistent
        # buffers follow the module's device but do not alter checkpoint format.
        self.register_buffer(
            "_val_fixed_target_tp", torch.zeros(self.num_classes, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_val_fixed_target_support", torch.zeros(self.num_classes, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_val_fixed_negative_triggers", torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_val_fixed_negative_clips", torch.zeros((), dtype=torch.long),
            persistent=False,
        )

    # ---- forward / steps -------------------------------------------------

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _ = batch_idx
        x, y = batch
        if self.frame_task:
            logits = self.model.forward_frames(x)
            targets = F.adaptive_max_pool1d(y, logits.shape[1]).transpose(1, 2)
            loss = self.loss_fn(logits, targets)
            self.log("train/loss", loss, on_epoch=True, prog_bar=True)
            return loss
        if self._mixup_active and x.shape[0] > 1:
            x, y = mixup_batch(x, y, self.mixup_alpha)
        logits = self(x)
        loss = self.loss_fn(logits, targets_to_indices(y)) if self.single_label else self.loss_fn(logits, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _ = batch_idx
        x, y = batch
        if self.frame_task:
            frame_logits = self.model.forward_frames(x)
            frame_targets = F.adaptive_max_pool1d(y, frame_logits.shape[1]).transpose(1, 2)
            loss = self.loss_fn(frame_logits, frame_targets)
            logits = frame_logits.amax(dim=1)
            y = frame_targets.amax(dim=1)
        else:
            logits = self(x)
            loss = None
        y_idx = targets_to_indices(y)
        if loss is None:
            loss = self.loss_fn(logits, y_idx) if self.single_label else self.loss_fn(logits, y)
        prob = logits_to_probabilities(logits, self.cfg)
        if self.single_label:
            self.val_ap.update(prob, y_idx)
            self.val_f1.update(prob, y_idx)
            self.val_precision.update(prob, y_idx)
            self.val_recall.update(prob, y_idx)
        else:
            y_int = (y >= 0.5).int()
            self.val_ap.update(prob, y_int)
            self.val_f1.update(prob, y_int)
            self.val_precision.update(prob, y_int)
            self.val_recall.update(prob, y_int)
        fixed_counts = fixed_threshold_trigger_counts(
            prob.detach(),
            y_idx if self.single_label else y,
            negative_index=self.neg_idx,
            threshold=FIXED_SELECTION_THRESHOLD,
            negative_max=float(self.cfg.inference.negative_max),
            margin_to_second_best=float(self.cfg.inference.margin_to_second_best),
        )
        self._val_fixed_target_tp.add_(fixed_counts[0])
        self._val_fixed_target_support.add_(fixed_counts[1])
        self._val_fixed_negative_triggers.add_(fixed_counts[2])
        self._val_fixed_negative_clips.add_(fixed_counts[3])
        if self.single_label:
            acc = (torch.argmax(prob, dim=1) == y_idx).float().mean()
        else:
            # Multi-label: argmax accuracy is ill-defined; use per-label
            # correctness at 0.5 so the tie-break metric stays meaningful.
            acc = ((prob >= 0.5).int() == (y >= 0.5).int()).float().mean()
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/accuracy", acc, prog_bar=True, on_epoch=True, batch_size=x.shape[0])
        return loss

    def on_validation_epoch_end(self):
        ap = self.val_ap.compute()
        f1 = self.val_f1.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        for i, name in enumerate(self.classes):
            self.log(f"val/ap/{name}", ap[i])
            self.log(f"val/f1/{name}", f1[i])
            self.log(f"val/p/{name}", precision[i])
            self.log(f"val/r/{name}", recall[i])
        # Macro mean across all classes EXCEPT _negative — what we actually care
        # about (target-class detection quality)
        target_mask = torch.tensor(
            [self.neg_idx is None or i != self.neg_idx for i in range(self.num_classes)],
            device=ap.device,
        )
        self.log("val/ap_macro_targets", ap[target_mask].mean() if target_mask.any() else ap.mean(), prog_bar=True)
        self.log("val/f1_macro_targets", f1[target_mask].mean() if target_mask.any() else f1.mean(), prog_bar=True)
        fixed_target_tp = self._distributed_sum(self._val_fixed_target_tp)
        fixed_target_support = self._distributed_sum(self._val_fixed_target_support)
        fixed_negative_triggers = self._distributed_sum(self._val_fixed_negative_triggers)
        fixed_negative_clips = self._distributed_sum(self._val_fixed_negative_clips)
        fixed_recall, fixed_fp_per_1000 = summarize_fixed_threshold_counts(
            fixed_target_tp,
            fixed_target_support,
            fixed_negative_triggers,
            fixed_negative_clips,
            negative_index=self.neg_idx,
        )
        self.log(VAL_FIXED_RECALL_KEY, fixed_recall, prog_bar=True)
        self.log(VAL_FIXED_FP_KEY, fixed_fp_per_1000, prog_bar=True)
        self.log(VAL_FIXED_NEGATIVE_COUNT_KEY, fixed_negative_clips.float())
        self.log(
            VAL_FIXED_NEGATIVE_TRIGGER_COUNT_KEY,
            fixed_negative_triggers.float(),
        )
        self.val_ap.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self._val_fixed_target_tp.zero_()
        self._val_fixed_target_support.zero_()
        self._val_fixed_negative_triggers.zero_()
        self._val_fixed_negative_clips.zero_()

    @staticmethod
    def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
        """Return a detached all-rank sum without changing the local buffer."""
        result = value.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        return result

    # ---- optimiser -------------------------------------------------------

    def on_train_epoch_start(self):
        # The optimiser is deliberately created with all backbone parameters;
        # only ``requires_grad`` changes here. This makes scheduled unfreezing
        # deterministic and avoids rebuilding optimiser state mid-run.
        self._maybe_unfreeze_for_fine_tune(int(self.current_epoch))
        # Keep frozen stages, plus every encoder BatchNorm requested by
        # freeze_batchnorm, in eval mode. Lightning resets the whole module to
        # train() at each epoch, so this must be re-applied every time.
        for module in self._frozen_modules:
            module.eval()

    def _apply_train_depth(self, depth: str) -> tuple[dict, list[nn.Module]]:
        info, frozen_modules = apply_train_depth(
            self.model,
            depth,
            freeze_batchnorm=self.cfg.training.freeze_batchnorm,
        )
        info["fine_tune_plan"] = self.fine_tune_plan.to_payload()
        return info, frozen_modules

    def _maybe_unfreeze_for_fine_tune(self, epoch: int) -> bool:
        if self._fine_tune_unfrozen or epoch < self.fine_tune_plan.warmup_epochs:
            return False
        self.freeze_info, self._frozen_modules = self._apply_train_depth(
            self.fine_tune_plan.target_depth,
        )
        self._fine_tune_unfrozen = True
        log.info(
            "fine-tune unfreeze at epoch %d: train depth=%s, "
            "trainable=%s/%s params, %d/%d stages%s",
            epoch + 1,
            self.freeze_info.get("train_depth"),
            _fmt_param_count(int(self.freeze_info.get("trainable_params") or 0)),
            _fmt_param_count(int(self.freeze_info.get("total_params") or 0)),
            len(self.freeze_info.get("trainable_stages") or []),
            int(self.freeze_info.get("n_stages") or 0),
            " · BN frozen" if self.freeze_info.get("freeze_batchnorm") else "",
        )
        if getattr(self, "_trainer", None) is not None:
            self.log("train/unfreeze_epoch", float(epoch), prog_bar=False)
        return True

    def configure_optimizers(self):
        head_params = list(self.model.head.parameters())
        head_ids = {id(p) for p in head_params}
        lr_backbone = self.cfg.training.lr_backbone
        decay = self.cfg.training.lr_layer_decay

        # Per-stage LR groups: deeper (later) stages get the full lr_backbone;
        # earlier stages are scaled down by lr_layer_decay**depth_from_top. With
        # decay == 1.0 this reproduces the original single backbone group.
        weight_decay = self.cfg.training.weight_decay
        param_groups: list[dict] = []
        assigned: set[int] = set(head_ids)
        stages = self.model.backbone_stages()
        for depth_from_top, (name, params) in enumerate(reversed(stages)):
            # Include frozen-but-scheduled parameters now. They have no grads
            # until the policy opens them, then retain normal AdamW state.
            trainable = [p for p in params if id(p) not in assigned]
            for p in trainable:
                assigned.add(id(p))
            if trainable:
                lr = lr_backbone * (decay ** depth_from_top) if decay < 1.0 else lr_backbone
                param_groups.extend(split_decay_param_groups(trainable, lr, weight_decay))
        # Any remaining trainable backbone params (not under a stage) at lr_backbone.
        leftover = [p for p in self.model.parameters() if id(p) not in assigned]
        for p in leftover:
            assigned.add(id(p))
        if leftover:
            param_groups.extend(split_decay_param_groups(leftover, lr_backbone, weight_decay))
        param_groups.extend(
            split_decay_param_groups(head_params, self.cfg.training.lr_head, weight_decay)
        )
        optim = torch.optim.AdamW(param_groups)

        # Linear warmup + cosine
        from torch.optim.lr_scheduler import LambdaLR
        warmup_epochs = self.cfg.training.warmup_epochs
        total_epochs = self.cfg.training.epochs

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

        sched = LambdaLR(optim, lr_lambda)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def load_classes(dataset_dir: Path) -> list[str]:
    with open(dataset_dir / "classes.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(c) for c in data]
    return data["classes"]
