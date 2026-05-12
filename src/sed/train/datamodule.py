"""PyTorch Lightning DataModule for SED training.

Reads the CSVs produced by ``sed dataset compose`` and computes log-mel
spectrograms with ``MelExtractor``. Augmentation is handled entirely offline
by ``sed dataset augment`` — the augmented WAVs are referenced as ordinary
rows in ``train.csv``, so there is no runtime augmentation here.

Per the project requirement, NO class-balanced batch sampling — the natural
distribution (heavy negatives) is preserved so the loss properly penalises
false positives.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import lightning as L
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from sed.audio.features import MelExtractor
from sed.audio.resample import resample_audio
from sed.config import Config, ProjectLayout
from sed.data.augment import _ensure_length
from sed.train.objective import is_single_label_loss

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
        layout: "ProjectLayout | None" = None,
        *,
        single_label: bool | None = None,
        train: bool = False,
    ):
        self.cfg = cfg
        self.classes = classes
        self.class_index = {c: i for i, c in enumerate(classes)}
        self.layout = layout
        self.single_label = is_single_label_loss(cfg) if single_label is None else single_label
        self.target_n = int(cfg.features.window_seconds * cfg.model.sample_rate)

        # SpecAugment masking on the TRAIN split only — val/test stay clean.
        # The pretrained EfficientAT backbone was trained with freq/time
        # masking, so fine-tuning keeps a (window-scaled) version of it.
        # EfficientAT masks the log-mel with 0.0 BEFORE the (x+4.5)/5
        # normalisation; masking after normalisation must therefore fill with
        # the equivalent normalised value, not 0.
        self._freq_mask = None
        self._time_mask = None
        self._mask_value = (0.0 + 4.5) / 5.0
        if (
            train
            and cfg.training.specaugment_enabled
            and (cfg.training.specaugment_freq_mask > 0
                 or cfg.training.specaugment_time_mask > 0)
        ):
            import torchaudio
            if cfg.training.specaugment_freq_mask > 0:
                self._freq_mask = torchaudio.transforms.FrequencyMasking(
                    cfg.training.specaugment_freq_mask
                )
            if cfg.training.specaugment_time_mask > 0:
                self._time_mask = torchaudio.transforms.TimeMasking(
                    cfg.training.specaugment_time_mask
                )

        with open(csv_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.mel = MelExtractor(
            sample_rate=cfg.model.sample_rate,
            n_fft=cfg.features.n_fft,
            win_length=cfg.features.win_length,
            hop_length=cfg.features.hop_length,
            n_mels=cfg.features.n_mels,
            fmin=cfg.features.fmin,
            fmax=cfg.features.fmax,
            frontend=cfg.features.frontend,
            device="cpu",
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, path: str) -> np.ndarray:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wave = data[:, 0]
        if sr != self.cfg.model.sample_rate:
            wave = resample_audio(wave, orig_sr=sr, target_sr=self.cfg.model.sample_rate)
        return _ensure_length(wave, self.target_n)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        wave = self._load(row["path"])
        spec = self.mel(wave)
        if isinstance(spec, torch.Tensor):
            spec = spec.numpy()
        spec = spec[0]                     # (n_mels, T)

        spec_t = torch.from_numpy(spec).unsqueeze(0)   # (1, n_mels, T)
        if self._freq_mask is not None:
            spec_t = self._freq_mask(spec_t, mask_value=self._mask_value)
        if self._time_mask is not None:
            spec_t = self._time_mask(spec_t, mask_value=self._mask_value)

        class_idx = self.class_index[row["class_name"]]
        if self.single_label:
            return spec_t, torch.tensor(class_idx, dtype=torch.long)

        target = np.zeros(len(self.classes), dtype=np.float32)
        target[class_idx] = 1.0
        return spec_t, torch.from_numpy(target)


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class SedDataModule(L.LightningDataModule):
    def __init__(self, cfg: Config, project_id: str, dataset_dir: Path | None = None):
        super().__init__()
        self.cfg = cfg
        self.layout = ProjectLayout(project_id)
        self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else self.layout.dataset_dir
        self.single_label = is_single_label_loss(cfg)

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        with open(self.dataset_dir / "classes.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.classes: list[str] = meta["classes"]

        self.train_ds = SedDataset(
            self.dataset_dir / "train.csv",
            self.classes, self.cfg, layout=self.layout,
            single_label=self.single_label,
            train=True,
        )
        self.val_ds = SedDataset(
            self.dataset_dir / "val.csv",
            self.classes, self.cfg, layout=self.layout,
            single_label=self.single_label,
        )
        self.test_ds = SedDataset(
            self.dataset_dir / "test.csv",
            self.classes, self.cfg, layout=self.layout,
            single_label=self.single_label,
        )

    def train_dataloader(self) -> DataLoader:
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
