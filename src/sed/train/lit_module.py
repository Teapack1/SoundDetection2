"""Lightning module wrapping the SED model and training logic."""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
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

from sed.config import Config
from sed.audio.features import feature_frame_count
from sed.train.model import AsymmetricFocalLoss, SedModel
from sed.train.objective import (
    LEGACY_BCE_LOSSES,
    is_single_label_loss,
    logits_to_probabilities,
    objective_name,
    targets_to_indices,
)


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
        self.loss_name = objective_name(cfg)

        self.model = SedModel(
            backbone_name=cfg.model.backbone,
            num_classes=self.num_classes,
            pretrained=cfg.model.pretrained,
            n_mels=cfg.features.n_mels,
            n_frames=feature_frame_count(
                sample_rate=cfg.model.sample_rate,
                window_seconds=cfg.features.window_seconds,
                hop_length=cfg.features.hop_length,
                n_fft=cfg.features.n_fft,
                frontend=cfg.features.frontend,
            ),
        )

        if self.single_label:
            weights = torch.ones(self.num_classes)
            if self.neg_idx is not None:
                weights[self.neg_idx] = cfg.training.negative_weight
            self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        elif self.loss_name == "asymmetric_focal":
            self.loss_fn = AsymmetricFocalLoss(
                num_classes=self.num_classes,
                negative_index=self.neg_idx,
                gamma_pos=cfg.training.focal_gamma_pos,
                gamma_neg=cfg.training.focal_gamma_neg,
                negative_weight=cfg.training.negative_weight,
            )
        elif self.loss_name in LEGACY_BCE_LOSSES:
            pos_weight = torch.ones(self.num_classes)
            if self.neg_idx is not None:
                pos_weight[self.neg_idx] = cfg.training.negative_weight
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            raise ValueError(
                "Unsupported training.loss. Use cross_entropy, asymmetric_focal, or bce_logits."
            )

        if self.single_label:
            self.val_ap = MulticlassAveragePrecision(num_classes=self.num_classes, average=None)
            self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average=None)
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average=None)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes, average=None)
            self.test_ap = MulticlassAveragePrecision(num_classes=self.num_classes, average=None)
            self.test_f1 = MulticlassF1Score(num_classes=self.num_classes, average=None)
            self.test_precision = MulticlassPrecision(num_classes=self.num_classes, average=None)
            self.test_recall = MulticlassRecall(num_classes=self.num_classes, average=None)
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
            self.test_ap = MultilabelAveragePrecision(num_labels=self.num_classes, average=None)
            self.test_f1 = MultilabelF1Score(num_labels=self.num_classes, average=None,
                                             threshold=0.5)
            self.test_precision = MultilabelPrecision(
                num_labels=self.num_classes, average=None, threshold=0.5
            )
            self.test_recall = MultilabelRecall(
                num_labels=self.num_classes, average=None, threshold=0.5
            )

    # ---- forward / steps -------------------------------------------------

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _ = batch_idx
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, targets_to_indices(y)) if self.single_label else self.loss_fn(logits, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _ = batch_idx
        x, y = batch
        logits = self(x)
        y_idx = targets_to_indices(y)
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
        self.val_ap.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def test_step(self, batch, batch_idx):
        _ = batch_idx
        x, y = batch
        logits = self(x)
        y_idx = targets_to_indices(y)
        loss = self.loss_fn(logits, y_idx) if self.single_label else self.loss_fn(logits, y)
        prob = logits_to_probabilities(logits, self.cfg)
        if self.single_label:
            self.test_ap.update(prob, y_idx)
            self.test_f1.update(prob, y_idx)
            self.test_precision.update(prob, y_idx)
            self.test_recall.update(prob, y_idx)
        else:
            y_int = (y >= 0.5).int()
            self.test_ap.update(prob, y_int)
            self.test_f1.update(prob, y_int)
            self.test_precision.update(prob, y_int)
            self.test_recall.update(prob, y_int)
        if self.single_label:
            acc = (torch.argmax(prob, dim=1) == y_idx).float().mean()
        else:
            acc = ((prob >= 0.5).int() == (y >= 0.5).int()).float().mean()
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/accuracy", acc, on_epoch=True, batch_size=x.shape[0])
        return loss

    def on_test_epoch_end(self):
        ap = self.test_ap.compute()
        f1 = self.test_f1.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        for i, name in enumerate(self.classes):
            self.log(f"test/ap/{name}", ap[i])
            self.log(f"test/f1/{name}", f1[i])
            self.log(f"test/p/{name}", precision[i])
            self.log(f"test/r/{name}", recall[i])
        target_mask = torch.tensor(
            [self.neg_idx is None or i != self.neg_idx for i in range(self.num_classes)],
            device=ap.device,
        )
        self.log("test/ap_macro_targets", ap[target_mask].mean() if target_mask.any() else ap.mean())
        self.log("test/f1_macro_targets", f1[target_mask].mean() if target_mask.any() else f1.mean())
        self.test_ap.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()

    # ---- optimiser -------------------------------------------------------

    def configure_optimizers(self):
        head_params = list(self.model.head.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in self.model.parameters()
                           if id(p) not in head_ids and p.requires_grad]
        optim = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.cfg.training.lr_backbone},
                {"params": head_params, "lr": self.cfg.training.lr_head},
            ],
            weight_decay=self.cfg.training.weight_decay,
        )

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
