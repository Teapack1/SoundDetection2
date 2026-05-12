"""BYOL-style audio representation pretraining for clip-capable encoders."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from sed.audio.features import feature_band_count, feature_channels, feature_frame_count_for_cfg
from sed.config import Config
from sed.train.model import SedModel
from sed.train.objective import targets_to_indices


class ByolLitModule(L.LightningModule):
    """An event-preserving EMA teacher/student objective.

    It deliberately uses the same spectrogram encoder and checkpoint namespace
    as supervised runs. A later venue run can therefore copy only
    ``model.backbone`` without knowing which objective produced the parent.
    """

    def __init__(self, cfg: Config, classes: list[str]):
        super().__init__()
        self.cfg = cfg
        self.classes = classes
        if cfg.model.task != "clip":
            raise ValueError("BYOL pretraining currently supports clip encoders only")
        self.model = SedModel(
            backbone_name=cfg.model.backbone,
            num_classes=len(classes),
            pretrained=cfg.model.pretrained,
            n_bands=feature_band_count(cfg.features, cfg.model.sample_rate),
            n_frames=feature_frame_count_for_cfg(cfg),
            params=dict(cfg.model.params),
            input_channels=feature_channels(cfg.features),
        )
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                feature_channels(cfg.features),
                feature_band_count(cfg.features, cfg.model.sample_rate),
                feature_frame_count_for_cfg(cfg),
            )
            dim = int(self.model._extract_features(dummy).shape[-1])
        projection = max(128, min(1024, dim))
        self.projector = nn.Sequential(nn.Linear(dim, projection), nn.BatchNorm1d(projection), nn.ReLU(),
                                       nn.Linear(projection, projection))
        self.predictor = nn.Sequential(nn.Linear(projection, projection), nn.ReLU(), nn.Linear(projection, projection))
        self.target_backbone = copy.deepcopy(self.model.backbone).eval()
        self.target_projector = copy.deepcopy(self.projector).eval()
        for parameter in list(self.target_backbone.parameters()) + list(self.target_projector.parameters()):
            parameter.requires_grad = False

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Small feature-space views that preserve a brief event's identity."""
        out = x.clone()
        # Gain/noise survives log/PCEN feature pipelines without turning a
        # 0.4-second impact into a different acoustic event.
        out = out + torch.randn((out.shape[0], 1, 1, 1), device=out.device, dtype=out.dtype) * 0.025
        width = max(1, int(out.shape[-1] * 0.06))
        for i in range(out.shape[0]):
            shift = int(torch.randint(-width, width + 1, (1,), device=out.device))
            out[i] = torch.roll(out[i], shifts=shift, dims=-1)
        if out.shape[-1] >= 8:
            mask = max(1, int(out.shape[-1] * 0.08))
            start = torch.randint(0, out.shape[-1] - mask + 1, (1,), device=out.device).item()
            out[:, :, :, start:start + mask] = 0.0
        return out

    def _online(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.projector(self.model._extract_features(x)))

    def _target(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.model.adapter.forward_features(self.target_backbone, x)
            return self.target_projector(feat)

    @staticmethod
    def _loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 2.0 - 2.0 * F.cosine_similarity(F.normalize(left, dim=-1), F.normalize(right, dim=-1), dim=-1).mean()

    def training_step(self, batch, batch_idx):
        _ = batch_idx
        x, _labels = batch
        view_a, view_b = self._augment(x), self._augment(x)
        loss = (self._loss(self._online(view_a), self._target(view_b))
                + self._loss(self._online(view_b), self._target(view_a))) * 0.5
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _ = batch_idx
        x, _labels = batch
        loss = self._loss(self._online(x), self._target(x))
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        # Tie-break score for checkpointing. This is not classification accuracy.
        self.log("val/byol_score", -loss, on_epoch=True, batch_size=x.shape[0])
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:  # noqa: ANN001
        _ = outputs, batch, batch_idx
        decay = float(self.cfg.training.ssl_ema_decay)
        with torch.no_grad():
            for online, target in zip(self.model.backbone.parameters(), self.target_backbone.parameters()):
                target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)
            for online, target in zip(self.projector.parameters(), self.target_projector.parameters()):
                target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)

    def configure_optimizers(self):
        from sed.train.lit_module import split_decay_param_groups

        params = list(self.model.backbone.parameters()) + list(self.projector.parameters()) + list(self.predictor.parameters())
        return torch.optim.AdamW(split_decay_param_groups(
            params, self.cfg.training.lr_backbone, self.cfg.training.weight_decay,
        ))


def linear_probe_evaluation(
    cfg: Config,
    project_id: str,
    checkpoint: Path,
    classes: list[str],
    *,
    dataset_dir: Path,
) -> dict[str, Any]:
    """Score an SSL encoder with one fixed frozen-encoder probe protocol."""
    from sed.train.datamodule import SedDataModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SedModel(
        backbone_name=cfg.model.backbone, num_classes=len(classes), pretrained=False,
        n_bands=feature_band_count(cfg.features, cfg.model.sample_rate),
        n_frames=feature_frame_count_for_cfg(cfg),
        params=dict(cfg.model.params),
        input_channels=feature_channels(cfg.features),
    ).to(device)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    prefix = "model.backbone."
    encoder_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    encoder.backbone.load_state_dict(encoder_state, strict=True)
    encoder.backbone.eval()
    for parameter in encoder.backbone.parameters():
        parameter.requires_grad = False

    probe_cfg = cfg.model_copy(deep=True)
    probe_cfg.training.specaugment_enabled = False
    dm = SedDataModule(cfg=probe_cfg, project_id=project_id, dataset_dir=dataset_dir)
    dm.setup("fit")
    with torch.no_grad():
        first_x, _first_y = next(iter(dm.train_dataloader()))
        feature_dim = int(encoder._extract_features(first_x.to(device)).shape[-1])
    head = nn.Linear(feature_dim, len(classes)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = min(25, max(5, int(cfg.training.epochs) // 4))
    for _epoch in range(epochs):
        head.train()
        for x, target in dm.train_dataloader():
            x, target = x.to(device), target.to(device)
            with torch.no_grad():
                features = encoder._extract_features(x)
            logits = head(features)
            loss = F.cross_entropy(logits, targets_to_indices(target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loader = dm.test_dataloader() if len(dm.test_ds) else dm.val_dataloader()
    truth: list[int] = []
    predicted: list[int] = []
    head.eval()
    with torch.no_grad():
        for x, target in loader:
            features = encoder._extract_features(x.to(device))
            predicted.extend(torch.argmax(head(features), dim=1).cpu().tolist())
            truth.extend(targets_to_indices(target).cpu().tolist())
    total = len(truth)
    correct = sum(int(a == b) for a, b in zip(predicted, truth))
    f1s: dict[str, float] = {}
    for index, name in enumerate(classes):
        tp = sum(int(a == index and b == index) for a, b in zip(predicted, truth))
        fp = sum(int(a == index and b != index) for a, b in zip(predicted, truth))
        fn = sum(int(a != index and b == index) for a, b in zip(predicted, truth))
        f1s[name] = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    target_f1 = [score for name, score in f1s.items() if name != "_negative"] or list(f1s.values())
    return {
        "protocol": "fixed-frozen-encoder-linear-probe-v1",
        "epochs": epochs,
        "n_samples": total,
        "accuracy": correct / total if total else 0.0,
        "f1_macro_targets": sum(target_f1) / len(target_f1) if target_f1 else 0.0,
        "per_class_f1": f1s,
    }


__all__ = ["ByolLitModule", "linear_probe_evaluation"]
