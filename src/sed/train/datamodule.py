"""PyTorch Lightning DataModule for SED training.

Reads the CSVs produced by `clinkd dataset compose`, applies waveform-domain
augmentation on the fly (incl. domain mixing with ambience pool and external
negatives), then computes log-mel spectrograms with `MelExtractor`.

Per the project requirement, NO class-balanced batch sampling — the natural
distribution (heavy negatives) is preserved so the loss properly penalizes
false positives. We rely on asymmetric focal loss + the negative-class weight
(see model.AsymmetricFocalLoss) to make the optimiser care about the rarer
positive classes.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

import lightning as L
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from sed.audio.features import MelExtractor
from sed.audio.resample import resample_audio
from sed.config import Config, ProjectLayout
from sed.data.augment import (
    AudioPool,
    WaveformAugmenter,
    mixup_pair,
    resolve_pool_path,
    spec_augment,
    _ensure_length,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SedDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        classes: list[str],
        cfg: Config,
        augment: bool,
        layout: "ProjectLayout | None" = None,
    ):
        self.cfg = cfg
        self.augment = augment
        self.classes = classes
        self.class_index = {c: i for i, c in enumerate(classes)}
        self.layout = layout
        self.target_n = int(cfg.features.window_seconds * cfg.audio.sample_rate)

        with open(csv_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.mel = MelExtractor(
            sample_rate=cfg.audio.sample_rate,
            n_fft=cfg.features.n_fft,
            win_length=cfg.features.win_length,
            hop_length=cfg.features.hop_length,
            n_mels=cfg.features.n_mels,
            fmin=cfg.features.fmin,
            fmax=cfg.features.fmax,
            device="cpu",
        )

        if augment:
            # Pool ambience from BOTH the shared library AND this project's
            # on-site empty-venue recordings. The project ambience captures
            # the venue's specific reverb signature; the library provides
            # crowd noise variety. See DESIGN.md "domain mixing".
            ambience_sources: list[Path] = [
                resolve_pool_path(cfg.augmentation.ambience_pool),  # library
            ]
            if self.layout is not None:
                ambience_sources.append(self.layout.ambience_dir)   # project
            self.amb_pool = AudioPool(ambience_sources, cfg.audio.sample_rate)
            log.info(
                "ambience pool: %d clips across %d source(s)",
                len(self.amb_pool), len(ambience_sources),
            )
            self.ext_pool = AudioPool(
                resolve_pool_path(cfg.augmentation.external_negative_pool),
                cfg.audio.sample_rate,
            )
            self.aug = WaveformAugmenter(
                sample_rate=cfg.audio.sample_rate,
                ambience_pool=self.amb_pool,
                external_neg_pool=self.ext_pool,
                domain_mix_prob=cfg.augmentation.domain_mix_prob,
                snr_db_range=tuple(cfg.augmentation.snr_db_range),
                per_class_min_snr_db=dict(cfg.augmentation.per_class_min_snr_db),
                external_neg_mix_prob=cfg.augmentation.external_negative_mix_prob,
                masking_check=cfg.augmentation.masking_check,
                masking_min_db=cfg.augmentation.masking_min_db,
                time_shift_ms=cfg.augmentation.time_shift_ms,
                pitch_shift_prob=cfg.augmentation.pitch_shift_prob,
                pitch_semitones=tuple(cfg.augmentation.pitch_semitones),
                time_stretch_prob=cfg.augmentation.time_stretch_prob,
                time_stretch=tuple(cfg.augmentation.time_stretch),
                gaussian_noise_amp=tuple(cfg.augmentation.gaussian_noise_amp),
            )
        else:
            self.aug = None

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, path: str) -> np.ndarray:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # Use channel 0 by default (multi-channel recordings)
        wave = data[:, 0]
        if sr != self.cfg.audio.sample_rate:
            wave = resample_audio(wave, orig_sr=sr, target_sr=self.cfg.audio.sample_rate)
        return _ensure_length(wave, self.target_n)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        wave = self._load(row["path"])
        is_target = bool(int(row["is_target"]))

        if self.aug is not None:
            wave = self.aug(wave, is_target=is_target, class_name=row["class_name"])

        # log-mel
        spec = self.mel(wave)
        if isinstance(spec, torch.Tensor):
            spec = spec.numpy()
        spec = spec[0]                     # (n_mels, T)

        if self.augment:
            spec = spec_augment(
                spec,
                freq_masks=self.cfg.augmentation.specaug_freq_masks,
                freq_width=self.cfg.augmentation.specaug_freq_width,
                time_masks=self.cfg.augmentation.specaug_time_masks,
                time_width=self.cfg.augmentation.specaug_time_width,
            )

        target = np.zeros(len(self.classes), dtype=np.float32)
        target[self.class_index[row["class_name"]]] = 1.0

        return torch.from_numpy(spec).unsqueeze(0), torch.from_numpy(target)


# ---------------------------------------------------------------------------
# Mixup-on-batch collator
# ---------------------------------------------------------------------------


class MixupCollator:
    def __init__(self, alpha: float = 0.3, p: float = 0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, batch):
        specs = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        if self.alpha > 0 and random.random() < self.p:
            perm = torch.randperm(specs.size(0))
            lam = float(np.random.beta(self.alpha, self.alpha))
            specs = lam * specs + (1 - lam) * specs[perm]
            targets = lam * targets + (1 - lam) * targets[perm]
        return specs, targets


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class SedDataModule(L.LightningDataModule):
    def __init__(self, cfg: Config, project_id: str):
        super().__init__()
        self.cfg = cfg
        self.layout = ProjectLayout(project_id)

    def setup(self, stage: str | None = None) -> None:
        with open(self.layout.dataset_dir / "classes.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.classes: list[str] = meta["classes"]

        self.train_ds = SedDataset(
            self.layout.dataset_dir / "train.csv",
            self.classes, self.cfg, augment=True, layout=self.layout,
        )
        self.val_ds = SedDataset(
            self.layout.dataset_dir / "val.csv",
            self.classes, self.cfg, augment=False, layout=self.layout,
        )
        self.test_ds = SedDataset(
            self.layout.dataset_dir / "test.csv",
            self.classes, self.cfg, augment=False, layout=self.layout,
        )

    def train_dataloader(self) -> DataLoader:
        # drop_last=True would yield ZERO batches when len(dataset) <
        # batch_size. Lightning then aborts with "No training batches" which
        # is confusing. Only drop the partial tail when we have enough data
        # for it not to matter.
        drop_last = len(self.train_ds) >= 2 * self.cfg.training.batch_size
        if not drop_last:
            log.warning(
                "train set has %d rows; batch_size=%d. Keeping the partial "
                "last batch (drop_last=False) so we have at least one batch "
                "per epoch. Record more data for production training.",
                len(self.train_ds), self.cfg.training.batch_size,
            )
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,                                # natural distribution
            num_workers=self.cfg.training.num_workers,
            persistent_workers=self.cfg.training.num_workers > 0,
            collate_fn=MixupCollator(self.cfg.training.mixup_alpha, p=0.5),
            drop_last=drop_last,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            persistent_workers=self.cfg.training.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
        )
