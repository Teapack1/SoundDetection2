"""PyTorch Lightning DataModule for SED training.

Reads the CSVs produced by ``sed dataset compose`` and computes log-mel
spectrograms with ``MelExtractor``. Augmentation is handled entirely offline
by ``sed dataset augment`` — the augmented WAVs are referenced as ordinary
rows in ``train.csv``, so there is no runtime augmentation here.

``train.csv`` is the whole train+val pool (no val.csv is composed): ``setup``
splits it into train/val live, on the fly, seeded by ``cfg.training.seed`` —
an ordinary part of the training procedure, not a project-level artifact.
Only the frozen test set (``test.csv``) is built ahead of time, in the
Library tab.

Class balancing is model-agnostic and configured once through
``training.class_balance``. ``sqrt`` moderately tempers label imbalance;
``balanced`` assigns half of the probability mass to ``_negative`` (when
present) and splits the other half uniformly across target classes. Both modes
temper recording groups within each class by their square-root size. Losses are
left unweighted so the correction is applied exactly once.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import lightning as lightning
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sed.audio.features import build_feature_extractor, window_sample_count
from sed.audio.resample import resample_audio
from sed.config import Config, ProjectLayout
from sed.data.augment import _ensure_length
from sed.train.objective import is_single_label_loss

log = logging.getLogger(__name__)

# Distinguishes "not analysed yet" from "analysed, no transient found" in the
# per-clip onset cache — None is a real, cacheable answer.
_UNSET = object()

def _uses_class_balancing(cfg: Config) -> bool:
    """Whether this supervised run uses the unified balanced sampler."""
    return (
        str(cfg.training.pretraining_objective) == "supervised"
        and str(cfg.training.class_balance) != "off"
    )


def _sampling_group_key(row: dict, row_index: int) -> str:
    """Stable recording-group key for hierarchical training sampling.

    New composed rows carry ``split_group``. Older datasets fall back to their
    session, then full path; a row-local key is the last resort so missing
    provenance cannot accidentally collapse unrelated clips into one group.
    Prefixes keep values from different provenance fields unambiguous.
    """
    # New provenance pins derived variants to their original recording. It is
    # stronger than a legacy split_group and mirrors split_group_key(), so the
    # sampler cannot reward one source merely because it has many augmentation
    # families.
    source_split_group = str(row.get("source_split_group") or "").strip()
    if source_split_group:
        return f"group:{source_split_group}"
    split_group = str(row.get("split_group") or "").strip()
    if split_group:
        return f"group:{split_group}"
    session_id = str(row.get("session_id") or "").strip()
    if session_id:
        return f"session:{session_id}"
    path = str(row.get("path") or "").strip()
    if path:
        return f"path:{path}"
    return f"row:{row_index}"


def _class_balance_weights(rows: list[dict], mode: str) -> torch.Tensor:
    """Return normalized per-row probabilities for one balance mode.

    ``sqrt`` uses square-root-tempered mass at the background/target and target
    class levels. ``balanced`` gives background and targets 50% each, then
    divides target mass uniformly. In both modes, a recording group's mass is
    proportional to ``sqrt(size)`` and clips are uniform inside the group.
    """
    if not rows:
        return torch.empty(0, dtype=torch.double)
    if mode not in {"sqrt", "balanced"}:
        raise ValueError(f"unsupported class-balance mode: {mode!r}")

    by_class: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_class[str(row.get("class_name") or "")].append(index)

    class_names = sorted(by_class)
    target_classes = [name for name in class_names if name != "_negative"]
    if "_negative" in by_class and target_classes:
        if mode == "balanced":
            negative_mass = 0.5
            target_mass = 0.5
            target_weights = {name: 1.0 for name in target_classes}
        else:
            negative_size = len(by_class["_negative"])
            target_size = sum(len(by_class[name]) for name in target_classes)
            root_denominator = math.sqrt(negative_size) + math.sqrt(target_size)
            negative_mass = math.sqrt(negative_size) / root_denominator
            target_mass = math.sqrt(target_size) / root_denominator
            target_weights = {
                name: math.sqrt(len(by_class[name])) for name in target_classes
            }
        target_denominator = sum(target_weights.values())
        class_mass = {"_negative": negative_mass}
        class_mass.update({
            name: target_mass * target_weights[name] / target_denominator
            for name in target_classes
        })
    else:
        class_weights = {
            name: (math.sqrt(len(by_class[name])) if mode == "sqrt" else 1.0)
            for name in class_names
        }
        class_denominator = sum(class_weights.values())
        class_mass = {
            name: class_weights[name] / class_denominator for name in class_names
        }

    weights = torch.zeros(len(rows), dtype=torch.double)
    for class_name in class_names:
        groups: dict[str, list[int]] = defaultdict(list)
        for index in by_class[class_name]:
            groups[_sampling_group_key(rows[index], index)].append(index)
        group_denominator = sum(math.sqrt(len(indices)) for indices in groups.values())
        for indices in groups.values():
            # group mass is proportional to sqrt(n); division by n makes clip
            # selection uniform inside that group.
            per_clip = class_mass[class_name] / (
                group_denominator * math.sqrt(len(indices))
            )
            weights[indices] = per_clip
    return weights


def _specaugment_fill(cfg: Config) -> float | torch.Tensor:
    """Feature-domain fill for SpecAugment masks — scalar or ``(F, 1)`` tensor.

    The EfficientAT contract masked raw log-mel values with 0.0 before applying
    fast normalization, so the equivalent post-normalization fill is 0.9. A
    dataset-scaled feature carries that same raw 0.0 through its own scaling:
    z-score → ``(0 − mean)/std`` (a per-band vector when fitted per band),
    min-max → ``clip((0 − lo)/(hi − lo))``. Linear, raw-power, and PCEN
    features without an affine keep the plain zero fill.
    """
    from sed.audio.features import resolve_feature_scaling

    f = cfg.features
    spec = resolve_feature_scaling(f)
    if spec.declared:
        if not spec.active:
            return 0.0
        a = np.asarray(spec.a, dtype=np.float32)
        b = np.asarray(spec.b, dtype=np.float32)
        if spec.method == "zscore":
            fill = (0.0 - a) / b
        else:
            fill = np.clip((0.0 - a) / (b - a), 0.0, 1.0).astype(np.float32)
        if fill.size == 1:
            return float(fill[0])
        return torch.from_numpy(fill.reshape(-1, 1))
    if (
        str(getattr(f, "feature_type", "mel")) == "mel"
        and not bool(getattr(f, "pcen", False))
        and str(getattr(f, "frontend", "efficientat")) == "efficientat"
    ):
        if bool(getattr(f, "standardize", False)):
            # Legacy scalar z-score on a config mutated without re-validation.
            std = float(getattr(f, "standardize_std", 1.0)) or 1.0
            return (0.0 - float(getattr(f, "standardize_mean", 0.0))) / std
        return (0.0 + 4.5) / 5.0
    return 0.0


def _apply_spec_mask(
    spec_t: torch.Tensor, mask_param: int, axis: int, fill: float | torch.Tensor,
) -> torch.Tensor:
    """One SpecAugment mask along ``axis`` (-2 bands, -1 frames), in place.

    Mirrors torchaudio's masking draw — width ~ U[0, param), start ~
    U[0, size − width) — but supports a per-band ``(F, 1)`` fill tensor, which
    torchaudio's scalar ``mask_value`` cannot express.
    """
    size = int(spec_t.shape[axis])
    if size <= 0 or mask_param <= 0:
        return spec_t
    width = float(torch.rand(1).item()) * min(int(mask_param), size)
    start = float(torch.rand(1).item()) * (size - width)
    lo, hi = int(start), int(start + width)
    if hi <= lo:
        return spec_t
    band_axis = axis in (-2, spec_t.ndim - 2)
    fill_value: float | torch.Tensor = fill
    if isinstance(fill, torch.Tensor):
        fill_value = fill.to(dtype=spec_t.dtype)
        if band_axis:
            fill_value = fill_value[lo:hi]
    if band_axis:
        spec_t[..., lo:hi, :] = fill_value
    else:
        spec_t[..., :, lo:hi] = fill_value
    return spec_t


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _safe_exception_message(exc: BaseException) -> str:
    try:
        msg = str(exc)
    except Exception:
        msg = repr(exc)
    return msg or repr(exc)


def _audio_probe_error(path: str) -> str | None:
    if not path:
        return "missing path"
    wav_path = Path(path)
    if not wav_path.exists():
        return "file does not exist"
    if not wav_path.is_file():
        return "path is not a file"
    try:
        info = sf.info(str(wav_path))
    except Exception as exc:
        return f"{type(exc).__name__}: {_safe_exception_message(exc)}"
    if int(info.frames) <= 0:
        return "empty audio"
    if int(info.samplerate) <= 0:
        return f"invalid sample rate: {info.samplerate}"
    if int(info.channels) <= 0:
        return f"invalid channel count: {info.channels}"
    return None


def _read_csv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = []
        for lineno, row in enumerate(csv.DictReader(f), start=2):
            row["__csv_line"] = str(lineno)
            rows.append(row)
    return rows


class SedDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        classes: list[str],
        cfg: Config,
        layout: ProjectLayout | None = None,
        *,
        single_label: bool | None = None,
        train: bool = False,
        rows: list[dict] | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.cfg = cfg
        self.classes = classes
        self.class_index = {c: i for i, c in enumerate(classes)}
        self.layout = layout
        self.single_label = is_single_label_loss(cfg) if single_label is None else single_label
        self.target_n = window_sample_count(cfg.model.sample_rate, cfg.features.window_seconds)

        # Window placement for clips longer than the model window. The transient
        # is put `window_lead_in_s` into the window — the geometry the onset-gated
        # runtime scores — instead of centring the clip, which would crop the
        # event out of most recordings. Locating a transient costs ~30 ms, so
        # results are memoized per clip for the life of the (persistent) worker.
        self._lead_in_n = int(round(cfg.window_lead_in_s() * cfg.model.sample_rate))
        self._onset_cache: dict[str, int | None] = {}
        # Analysis geometry must follow this run's feature config, not the
        # EfficientAT defaults the augmenter module was last left on.
        from sed.data.augment import set_augment_features

        set_augment_features(cfg)

        # SpecAugment masking on the TRAIN split only — val/test stay clean. It
        # masks feature rows and time frames, so it works for mel bands and
        # linear-frequency bins. The fill value is feature-domain aware and may
        # be a per-band vector for dataset-scaled features.
        self._freq_mask_param = 0
        self._time_mask_param = 0
        self._mask_fill = _specaugment_fill(cfg)
        if train and cfg.training.specaugment_enabled:
            self._freq_mask_param = max(0, int(cfg.training.specaugment_freq_mask))
            self._time_mask_param = max(0, int(cfg.training.specaugment_time_mask))

        self.rows = list(rows) if rows is not None else _read_csv_rows(self.csv_path)

        # Strong frame training only consumes human-confirmed intervals. A
        # negative clip remains useful as an all-zero temporal target; positive
        # weak clips are skipped rather than silently being turned into false
        # negatives.
        if cfg.model.task == "frame":
            self.rows = [row for row in self.rows if self._has_frame_target(row)]
        self.skipped_audio_rows = self._filter_readable_audio_rows()

        # One feature definition drives the whole pipeline: mel+efficientat uses
        # the verified MelExtractor; linear/PCEN use FeatureExtractor.
        self.mel = build_feature_extractor(cfg)

    def _intervals(self, path: str) -> list[dict]:
        sidecar = Path(path).with_suffix(".json")
        if not sidecar.exists():
            return []
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [row for row in (raw.get("event_intervals") or [])
                if isinstance(row, dict) and bool(row.get("confirmed"))]

    def _has_frame_target(self, row: dict[str, str]) -> bool:
        if row.get("class_name") == "_negative":
            return True
        expected = row.get("class_name", "")
        return any(str(interval.get("label") or "") == expected
                   for interval in self._intervals(row["path"]))

    def __len__(self) -> int:
        return len(self.rows)

    def _row_location(self, row: dict[str, str]) -> str:
        line = row.get("__csv_line") or "?"
        class_name = row.get("class_name") or "?"
        path = row.get("path") or ""
        return f"{self.csv_path}:{line} class={class_name!r} path={path}"

    def _filter_readable_audio_rows(self) -> list[tuple[dict[str, str], str]]:
        kept: list[dict[str, str]] = []
        skipped: list[tuple[dict[str, str], str]] = []
        for row in self.rows:
            err = _audio_probe_error(row.get("path", ""))
            if err is None:
                kept.append(row)
            else:
                skipped.append((row, err))
        self.rows = kept
        if skipped:
            preview = "; ".join(
                f"{self._row_location(row)} ({err})"
                for row, err in skipped[:5]
            )
            if len(skipped) > 5:
                preview += f"; +{len(skipped) - 5} more"
            log.warning(
                "%s: skipped %d unreadable/empty audio row(s), kept %d/%d. %s",
                self.csv_path,
                len(skipped),
                len(kept),
                len(kept) + len(skipped),
                preview,
            )
        return skipped

    def _onset_sample(self, wave: np.ndarray, path: str) -> int | None:
        """Sample index of the clip's triggering transient, or None if unknown.

        Three sources, best first:

        1. A human-confirmed event interval. If someone has annotated where the
           event is, that beats anything inferred — and it keeps the window and
           the frame targets derived from the same annotation consistent.
        2. ``onset_offset_s`` from the sidecar. The recorder knows exactly where
           it placed the onset (it is the preroll it drew), so this is free and
           exact for anything recorded since that field was added.
        3. The spectral-flux detector the augmenter uses to place its SNR
           window. Costs ~30 ms per clip, which is why results are memoized:
           dataloader workers are persistent, so a clip is analysed at most once
           per run.
        """
        cached = self._onset_cache.get(path, _UNSET)
        if cached is not _UNSET:
            return cached

        offset_s = self._annotated_onset_offset(path)
        if offset_s is None:
            offset_s = self._sidecar_onset_offset(path)
        if offset_s is None:
            try:
                from sed.data.augment import _frame_to_sample, _locate_transient_frame

                sr = int(self.cfg.model.sample_rate)
                frame = _locate_transient_frame(wave, sr)
                onset = int(_frame_to_sample(frame, sr))
            except Exception as exc:  # noqa: BLE001 — fall back to the centre crop
                log.warning(
                    "could not locate the transient in %s (%s: %s); "
                    "falling back to a centre crop",
                    path, type(exc).__name__, _safe_exception_message(exc),
                )
                onset = None
        else:
            onset = int(round(offset_s * float(self.cfg.model.sample_rate)))

        self._onset_cache[path] = onset
        return onset

    def _annotated_onset_offset(self, path: str) -> float | None:
        """Onset of the earliest human-confirmed event interval, if any."""
        onsets = []
        for interval in self._intervals(path):
            try:
                onsets.append(float(interval["onset_s"]))
            except (KeyError, TypeError, ValueError):
                continue
        return min(onsets) if onsets else None

    def _sidecar_onset_offset(self, path: str) -> float | None:
        sidecar = Path(path).with_suffix(".json")
        if not sidecar.exists():
            return None
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — an unreadable sidecar just means "unknown"
            return None
        value = raw.get("onset_offset_s")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _crop_start(self, wave: np.ndarray, path: str) -> int:
        """First sample of the model window inside a (possibly longer) clip.

        The window is placed so the transient sits ``cfg.window_lead_in_s()``
        into it — the same geometry the onset-gated runtime scores, so the model
        never sees a framing in training that it will not see in the venue.

        This matters far more than it looks: clips are recorded with a randomized
        preroll spread across the whole clip, so a centre crop of a window
        shorter than the recording cuts the transient out entirely for most of
        them. On this project's own library, centre-cropping a 0.4 s window lost
        the transient in 52% of positive clips.
        """
        excess = len(wave) - self.target_n
        if excess <= 0:
            return 0
        onset_sample = self._onset_sample(wave, path)
        if onset_sample is None:
            return excess // 2                      # nothing better to go on
        return int(min(max(onset_sample - self._lead_in_n, 0), excess))

    def _load(self, path: str) -> tuple[np.ndarray, float]:
        """Return the window-fitted wave plus the crop offset in seconds.

        Interval annotations are in original-clip time, so the frame-target
        mapping must shift them left by this offset (0 for short/exact clips).
        """
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as exc:
            raise RuntimeError(
                f"failed to read audio {path!r}: "
                f"{type(exc).__name__}: {_safe_exception_message(exc)}"
            ) from exc
        if data.shape[0] == 0:
            raise RuntimeError(f"failed to read audio {path!r}: empty audio")
        wave = data[:, 0]
        if sr != self.cfg.model.sample_rate:
            wave = resample_audio(wave, orig_sr=sr, target_sr=self.cfg.model.sample_rate)
        start = self._crop_start(wave, path)
        crop_offset_s = start / float(self.cfg.model.sample_rate)
        if len(wave) > self.target_n:
            wave = wave[start:start + self.target_n]
        return _ensure_length(wave, self.target_n), crop_offset_s

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        wave, crop_offset_s = self._load(row["path"])
        spec = self.mel(wave)
        if isinstance(spec, torch.Tensor):
            spec = spec.numpy()
        spec = spec[0]                     # (n_mels, T) or (C, H, W) for image features

        # A spectrogram feature is 2-D (bands, T) → add the single channel; a
        # colormap image feature is already (C, H, W) → keep its channels.
        spec_t = torch.from_numpy(spec)
        if spec_t.ndim == 2:
            spec_t = spec_t.unsqueeze(0)   # (1, n_mels, T)
        if self._freq_mask_param > 0:
            spec_t = _apply_spec_mask(spec_t, self._freq_mask_param, -2, self._mask_fill)
        if self._time_mask_param > 0:
            spec_t = _apply_spec_mask(spec_t, self._time_mask_param, -1, self._mask_fill)

        if self.cfg.model.task == "frame":
            target = torch.zeros((len(self.classes), spec_t.shape[-1]), dtype=torch.float32)
            if row["class_name"] == "_negative" and "_negative" in self.class_index:
                target[self.class_index["_negative"], :] = 1.0
            duration = float(self.cfg.features.window_seconds)
            for interval in self._intervals(row["path"]):
                label = str(interval.get("label") or "")
                if label not in self.class_index:
                    continue
                # Interval times are original-clip coordinates; shift into the
                # cropped window and drop events that fall outside it.
                onset_s = float(interval["onset_s"]) - crop_offset_s
                offset_s = float(interval["offset_s"]) - crop_offset_s
                if offset_s <= 0.0 or onset_s >= duration:
                    continue
                start = max(
                    0,
                    min(spec_t.shape[-1], round(onset_s / duration * spec_t.shape[-1])),
                )
                end = max(
                    start + 1,
                    min(spec_t.shape[-1], round(offset_s / duration * spec_t.shape[-1])),
                )
                target[self.class_index[label], start:end] = 1.0
            return spec_t, target

        class_idx = self.class_index[row["class_name"]]
        if self.single_label:
            return spec_t, torch.tensor(class_idx, dtype=torch.long)

        target = np.zeros(len(self.classes), dtype=np.float32)
        target[class_idx] = 1.0
        return spec_t, torch.from_numpy(target)


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class SedDataModule(lightning.LightningDataModule):
    def __init__(self, cfg: Config, project_id: str, dataset_dir: Path | None = None):
        super().__init__()
        self.cfg = cfg
        self.layout = ProjectLayout(project_id)
        self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else self.layout.dataset_dir
        self.single_label = is_single_label_loss(cfg)

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        if getattr(self, "train_ds", None) is not None:
            return  # already split for this run — Lightning may call setup() twice
        with open(self.dataset_dir / "classes.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.classes: list[str] = meta["classes"]

        # Val is not a compose-time artifact: it's carved live out of the
        # train+val pool (train.csv) here, at train start, seeded by the run's
        # training seed — an ordinary training-pipeline split, recomputed
        # every run rather than pinned to whatever the dataset looked like
        # when it was last composed.
        from sed.data.splits import split_pool_rows_detailed
        pool_rows = _read_csv_rows(self.dataset_dir / "train.csv")
        val_fraction = float(getattr(self.cfg.dataset, "val_fraction", 0.15) or 0.15)
        train_rows, val_rows, self.split_diagnostics = split_pool_rows_detailed(
            pool_rows, val_frac=val_fraction, seed=self.cfg.training.seed,
        )
        self.split_modes = dict(self.split_diagnostics.get("modes") or {})

        self.train_ds = SedDataset(
            self.dataset_dir / "train.csv",
            self.classes, self.cfg, layout=self.layout,
            single_label=self.single_label,
            train=True, rows=train_rows,
        )
        self.val_ds = SedDataset(
            self.dataset_dir / "train.csv",
            self.classes, self.cfg, layout=self.layout,
            single_label=self.single_label,
            rows=val_rows,
        )
        self.test_ds = SedDataset(
            self.dataset_dir / "test.csv",
            self.classes, self.cfg, layout=self.layout,
            single_label=self.single_label,
        )
        if len(self.train_ds) == 0:
            raise RuntimeError(
                f"{self.dataset_dir / 'train.csv'} has no readable audio rows after filtering"
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
        sampler = None
        shuffle = True
        if _uses_class_balancing(self.cfg):
            generator = torch.Generator()
            generator.manual_seed(int(self.cfg.training.seed))
            sampler = WeightedRandomSampler(
                _class_balance_weights(
                    self.train_ds.rows,
                    str(self.cfg.training.class_balance),
                ),
                num_samples=len(self.train_ds),
                replacement=True,
                generator=generator,
            )
            shuffle = False
            # The hierarchical contract defines an epoch as exactly one draw
            # per original training row. Do not discard the final partial batch.
            drop_last = False
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
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
