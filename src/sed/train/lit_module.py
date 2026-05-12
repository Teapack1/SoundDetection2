"""Lightning module wrapping the SED model and training logic."""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

from sed.config import Config
from sed.train.model import AsymmetricFocalLoss, SedModel


class SedLitModule(L.LightningModule):
    def __init__(self, cfg: Config, classes: list[str]):
        super().__init__()
        self.cfg = cfg
        self.classes = classes
        self.num_classes = len(classes)
        self.neg_idx = classes.index("_negative")
        self.save_hyperparameters({"classes": classes})

        self.model = SedModel(
            backbone_name=cfg.model.backbone,
            num_classes=self.num_classes,
            pretrained=cfg.model.pretrained,
        )

        if cfg.training.loss == "asymmetric_focal":
            self.loss_fn = AsymmetricFocalLoss(
                num_classes=self.num_classes,
                negative_index=self.neg_idx,
                gamma_pos=cfg.training.focal_gamma_pos,
                gamma_neg=cfg.training.focal_gamma_neg,
                negative_weight=cfg.training.negative_weight,
            )
        else:
            pos_weight = torch.ones(self.num_classes)
            pos_weight[self.neg_idx] = cfg.training.negative_weight
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.val_ap = MultilabelAveragePrecision(num_labels=self.num_classes, average=None)
        self.val_f1 = MultilabelF1Score(num_labels=self.num_classes, average=None,
                                        threshold=0.5)
        self.val_precision = MultilabelPrecision(
            num_labels=self.num_classes, average=None, threshold=0.5
        )
        self.val_recall = MultilabelRecall(
            num_labels=self.num_classes, average=None, threshold=0.5
        )

    # ---- forward / steps -------------------------------------------------

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        prob = torch.sigmoid(logits)
        # torchmetrics needs int targets
        y_int = (y >= 0.5).int()
        self.val_ap.update(prob, y_int)
        self.val_f1.update(prob, y_int)
        self.val_precision.update(prob, y_int)
        self.val_recall.update(prob, y_int)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
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
            [i != self.neg_idx for i in range(self.num_classes)],
            device=ap.device,
        )
        self.log("val/ap_macro_targets", ap[target_mask].mean(), prog_bar=True)
        self.log("val/f1_macro_targets", f1[target_mask].mean(), prog_bar=True)
        self.val_ap.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

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
        return json.load(f)["classes"]
