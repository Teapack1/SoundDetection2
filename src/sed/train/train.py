"""Training entry point for `sed train`."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from sed.config import Config, ProjectLayout
from sed.train.datamodule import SedDataModule
from sed.train.lit_module import SedLitModule, load_classes
from sed.train.plot import plot_training_curves

log = logging.getLogger(__name__)


def _count_csv_rows(path: Path) -> int:
    """Return data rows in a CSV (excluding header). 0 if missing/empty."""
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f) - 1                          # exclude header
    return max(0, n)


def train_project(cfg: Config, project_id: str, resume: str | None = None) -> Path:
    layout = ProjectLayout(project_id)

    # ---- Pre-flight: report dataset state, but only HARD-FAIL on empty train ----
    csvs = {name: layout.dataset_dir / f"{name}.csv"
            for name in ("train", "val", "test")}
    if not csvs["train"].exists():
        raise RuntimeError(
            f"No dataset for project '{project_id}'. Run `clinkd dataset compose` first."
        )
    counts = {name: _count_csv_rows(p) for name, p in csvs.items()}
    log.info("dataset rows — train=%d val=%d test=%d",
             counts["train"], counts["val"], counts["test"])

    if counts["train"] == 0:
        raise RuntimeError(
            "train.csv is empty — can't train on zero rows. "
            f"Run `clinkd dataset compose --project {project_id}` first, "
            "and check the per-class counts in its summary output."
        )
    if counts["val"] == 0:
        log.warning(
            "val.csv is empty. Training will run without held-out validation — "
            "EarlyStopping and ModelCheckpoint's monitored metric won't be "
            "computed, so the trainer will run for the full `epochs` budget "
            "and save the last checkpoint. To get a proper val split, record "
            "more sessions OR re-run `dataset compose` (the adaptive splitter "
            "should usually create non-empty val/test now)."
        )
    if counts["test"] == 0:
        log.warning("test.csv is empty — final test-set evaluation will be skipped.")

    L.seed_everything(cfg.training.seed, workers=True)

    classes = load_classes(layout.dataset_dir)
    log.info("Training on %d classes: %s", len(classes), classes)

    # Snapshot the effective config so inference + future re-training are
    # reproducible against the same parameters this model was trained on.
    # See sed.config.load_config_for_inference for how this is consumed.
    from sed.config import dump_config_snapshot
    snapshot = dump_config_snapshot(
        cfg, layout.models_dir / "training_config.json"
    )
    log.info("config snapshot → %s", snapshot)

    dm = SedDataModule(cfg=cfg, project_id=project_id)
    model = SedLitModule(cfg=cfg, classes=classes)

    log_dir = layout.reports_dir / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(save_dir=str(log_dir), name="csv")

    has_val = counts["val"] > 0
    callbacks: list = [LearningRateMonitor(logging_interval="epoch")]
    if has_val:
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(layout.models_dir),
            filename="best-{epoch:02d}-{val/f1_macro_targets:.3f}",
            monitor="val/f1_macro_targets",
            mode="max",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_cb)
        callbacks.append(EarlyStopping(
            monitor="val/f1_macro_targets", mode="max",
            patience=cfg.training.patience, min_delta=0.001,
        ))
    else:
        # No val → just save the last checkpoint at end of training. There's
        # no monitored metric to pick a "best" by, so save_last is what we use.
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(layout.models_dir),
            filename="last-{epoch:02d}",
            save_top_k=0,                              # don't try to pick best
            save_last=True,
        )
        callbacks.append(checkpoint_cb)
        log.warning(
            "training without validation: no EarlyStopping; will run full "
            "%d epochs and save the last checkpoint.", cfg.training.epochs,
        )

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices="auto",
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        limit_val_batches=(1.0 if has_val else 0),    # skip val phase entirely
        num_sanity_val_steps=(2 if has_val else 0),
    )
    trainer.fit(model, datamodule=dm, ckpt_path=resume)

    # Save the "current" alias for downstream commands. Prefer the
    # validation-best checkpoint; fall back to the last-epoch checkpoint
    # when no val was available.
    best_path_str = checkpoint_cb.best_model_path or ""
    last_path_str = checkpoint_cb.last_model_path or ""
    chosen_str = best_path_str if (best_path_str and Path(best_path_str).is_file()) \
                                  else last_path_str
    if chosen_str and Path(chosen_str).is_file():
        chosen_ckpt = Path(chosen_str)
        current = layout.models_dir / "current.ckpt"
        if current.exists() or current.is_symlink():
            current.unlink()
        shutil.copy2(chosen_ckpt, current)
        log.info("checkpoint → %s (from %s)", current, chosen_ckpt.name)
    else:
        log.warning(
            "No checkpoint produced. Training likely aborted very early; "
            "check the logs above."
        )
        return layout.models_dir / "current.ckpt"

    # Test set evaluation — only if test split has data and we have a
    # val-tracked best checkpoint; otherwise just skip.
    if counts["test"] > 0 and has_val and best_path_str:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
        if test_results:
            with open(layout.reports_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(test_results, f, indent=2, default=float)
    elif counts["test"] == 0:
        log.info("test.csv empty — skipping test-set evaluation.")
    elif not has_val:
        log.info("no val phase ran — skipping test-set evaluation (no \"best\" to load).")

    # Plot training curves
    metrics_csv = next(log_dir.rglob("metrics.csv"), None)
    if metrics_csv is not None and metrics_csv.exists():
        plot_training_curves(metrics_csv, layout.reports_dir / "training_curves.png")

    return layout.models_dir / "current.ckpt"
