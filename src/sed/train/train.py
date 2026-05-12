"""Training entry point for `sed train`.

Each invocation produces a self-contained model bundle under
`projects/<id>/models/<model_id>/` capturing the effective config, the
checkpoint, the exported OpenVINO model (when auto_export is on), the
training curves, the test-set metrics + confusion matrix (when auto_test is on
and a test set is present), and the dataset CSVs it trained on.

The trained model is auto-selected as the active model (`[deploy] active_model`)
when it exports cleanly; switch later with `sed models select` — see
sed.train.registry.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from sed.config import (
    Config,
    ProjectLayout,
    dump_config_snapshot,
    dump_runtime_settings_from_config,
    validate_pretrained_feature_contract,
)
from sed.train import registry
from sed.train.datamodule import SedDataModule
from sed.train.lit_module import SedLitModule, load_classes
from sed.train.plot import plot_training_curves, summarize_training_metrics, write_training_report

log = logging.getLogger(__name__)


def _validate_dataset_classes_match_config(cfg: Config, classes: list[str]) -> None:
    configured = list(cfg.model.classes)
    if configured == classes:
        return
    raise RuntimeError(
        "dataset/classes.json does not match [dataset] target_classes in config.toml. "
        f"dataset={classes}; config={configured}. "
        "Run `sed dataset compose --project <id>` after editing target_classes, "
        "then train again."
    )

class LossAccuracyCheckpoint(Callback):
    """Save the validation-loss winner, using validation accuracy as tie-break."""

    def __init__(
        self,
        dirpath: Path,
        min_delta: float = 0.001,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.min_delta = float(min_delta)
        self.best_model_path = ""
        self.best_loss: float | None = None
        self.best_accuracy: float | None = None
        self.best_epoch: int | None = None

    def _is_better(self, val_loss: float, val_accuracy: float) -> bool:
        if self.best_loss is None:
            return True
        if val_loss < self.best_loss - self.min_delta:
            return True
        if abs(val_loss - self.best_loss) <= self.min_delta:
            return val_accuracy > (self.best_accuracy if self.best_accuracy is not None else float("-inf"))
        return False

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        _ = pl_module
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        loss_raw = metrics.get("val/loss")
        if loss_raw is None:
            return
        acc_raw = metrics.get("val/accuracy")
        val_loss = float(loss_raw.detach().cpu().item() if hasattr(loss_raw, "detach") else loss_raw)
        val_accuracy = float(
            acc_raw.detach().cpu().item() if hasattr(acc_raw, "detach") else (acc_raw or 0.0)
        )
        if not self._is_better(val_loss, val_accuracy):
            return
        previous = Path(self.best_model_path) if self.best_model_path else None
        self.best_loss = val_loss
        self.best_accuracy = val_accuracy
        self.best_epoch = int(trainer.current_epoch)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        path = self.dirpath / _best_checkpoint_name(self.best_epoch)
        trainer.save_checkpoint(path)
        self.best_model_path = str(path)
        if previous is not None and previous.exists() and previous != path:
            previous.unlink()


def _quiet_lightning_info_logs() -> None:
    lightning_logger = logging.getLogger("lightning")
    lightning_logger.handlers.clear()
    lightning_logger.setLevel(logging.WARNING)
    lightning_logger.propagate = True


def _fmt_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _train_log_every_n_steps(n_rows: int, batch_size: int) -> int:
    if batch_size <= 0:
        return 1
    if n_rows >= 2 * batch_size:
        n_batches = max(1, n_rows // batch_size)
    else:
        n_batches = max(1, math.ceil(n_rows / batch_size))
    return max(1, min(10, n_batches))


def _device_summary() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"cuda:0 ({name})"
        return "cpu"
    except Exception:
        return "auto"


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f) - 1
    return max(0, n)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out
    except Exception:  # noqa: BLE001
        return ""


def _load_dataset_snapshot(layout: ProjectLayout) -> dict:
    path = layout.dataset_dir / "dataset_snapshot.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _copy_dataset_artifacts(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "val.csv", "test.csv", "classes.json", "dataset_snapshot.json"):
        src = src_dir / name
        if not src.exists():
            raise RuntimeError(f"Missing dataset artifact for the model bundle: {src}")
        shutil.copy2(src, dst_dir / name)


def _best_checkpoint_name(epoch: int) -> str:
    return f"checkpoint_best_epoch-{epoch:03d}.ckpt"


def _semantic_checkpoint_path(
    model_dir: Path,
    chosen_ckpt: Path,
    best: dict | None,
    fallback_epoch: int,
) -> Path:
    if best:
        epoch = int(best["epoch"])
        return model_dir / _best_checkpoint_name(epoch)
    if chosen_ckpt.name.startswith("checkpoint_"):
        return model_dir / chosen_ckpt.name
    return model_dir / f"checkpoint_last_epoch-{fallback_epoch:03d}.ckpt"


def _move_checkpoint(src: Path, dst: Path) -> Path:
    if dst.exists() and src.samefile(dst):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dst)
    return dst


def train_project(
    cfg: Config,
    project_id: str,
    *,
    description: str | None = None,
    auto_test: bool | None = None,
    auto_export: bool | None = None,
) -> Path:
    _quiet_lightning_info_logs()
    layout = ProjectLayout(project_id)
    layout.create()

    # Pre-flight on the dataset CSVs.
    csvs = {name: layout.dataset_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if not csvs["train"].exists():
        raise RuntimeError(
            f"No dataset for project '{project_id}'. Run `sed dataset compose` first."
        )
    counts = {name: _count_csv_rows(p) for name, p in csvs.items()}
    log.info("dataset rows — train=%d val=%d test=%d",
             counts["train"], counts["val"], counts["test"])
    if counts["train"] == 0:
        raise RuntimeError(
            "train.csv is empty — can't train on zero rows. "
            f"Run `sed dataset compose --project {project_id}` first."
        )
    if counts["val"] == 0:
        log.warning(
            "val.csv is empty. Training will run without held-out validation — "
            "EarlyStopping and ModelCheckpoint's monitored metric won't be "
            "computed, so the trainer will run for the full `epochs` budget."
        )
    if counts["test"] == 0:
        log.warning("test.csv is empty — final test-set evaluation will be skipped.")

    do_test = (cfg.training.auto_test if auto_test is None else auto_test) and counts["test"] > 0
    do_export = cfg.training.auto_export if auto_export is None else auto_export

    L.seed_everything(cfg.training.seed, workers=True)

    classes = load_classes(layout.dataset_dir)
    _validate_dataset_classes_match_config(cfg, classes)
    validate_pretrained_feature_contract(cfg, phase="training")
    log.info("Training on %d classes: %s", len(classes), classes)

    # --- Allocate the model bundle dir and freeze the effective config there ---
    model_id = registry.new_model_id(project_id, description=description)
    model_dir = layout.model_dir(model_id)
    run_reports = model_dir / "reports"
    for d in (model_dir, run_reports):
        d.mkdir(parents=True, exist_ok=True)

    snapshot_path = dump_config_snapshot(cfg, model_dir / "training_config.json")
    log.info("config snapshot → %s", snapshot_path)
    dump_runtime_settings_from_config(cfg, model_dir / "runtime_settings.json")
    # Keep a copy of dataset/classes.json next to the model so the runtime can
    # load class order without the dataset/ dir present.
    classes_src = layout.dataset_dir / "classes.json"
    if classes_src.exists():
        shutil.copy2(classes_src, model_dir / "classes.json")

    run_dataset = model_dir / "dataset"
    _copy_dataset_artifacts(layout.dataset_dir, run_dataset)

    dm = SedDataModule(cfg=cfg, project_id=project_id)
    model = SedLitModule(cfg=cfg, classes=classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    csv_logger = CSVLogger(save_dir=str(run_reports), name="training")

    has_val = counts["val"] > 0
    callbacks: list = [LearningRateMonitor(logging_interval="epoch")]
    if has_val:
        checkpoint_cb = LossAccuracyCheckpoint(model_dir, min_delta=0.001)
        callbacks.append(checkpoint_cb)
        callbacks.append(EarlyStopping(
            monitor="val/loss", mode="min",
            patience=cfg.training.patience, min_delta=0.001,
        ))
    else:
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(model_dir),
            filename="checkpoint_last_epoch-{epoch:03d}",
            save_top_k=0,
            save_last=True,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_cb)
        log.warning(
            "training without validation: no EarlyStopping; will run full "
            "%d epochs and save the last checkpoint.", cfg.training.epochs,
        )

    log_every_n_steps = _train_log_every_n_steps(counts["train"], cfg.training.batch_size)
    log.info(
        "model params: trainable=%s total=%s",
        _fmt_param_count(trainable_params),
        _fmt_param_count(total_params),
    )
    log.info(
        "trainer: device=%s precision=%s epochs=%d batch_size=%d validation=%s",
        _device_summary(),
        cfg.training.precision,
        cfg.training.epochs,
        cfg.training.batch_size,
        "on" if has_val else "off",
    )

    trainer_kwargs = {
        "max_epochs": cfg.training.epochs,
        "accelerator": "auto",
        "devices": "auto",
        "precision": cfg.training.precision,
        "callbacks": callbacks,
        "logger": csv_logger,
        "log_every_n_steps": log_every_n_steps,
        "gradient_clip_val": 1.0,
        "num_sanity_val_steps": (2 if has_val else 0),
        "enable_model_summary": False,
    }
    if not has_val:
        trainer_kwargs["limit_val_batches"] = 0

    warnings.filterwarnings(
        "ignore",
        message=r"Checkpoint directory .* exists and is not empty\.",
        category=UserWarning,
        module=r"lightning\.pytorch\.callbacks\.model_checkpoint",
    )
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=dm)

    best_path_str = getattr(checkpoint_cb, "best_model_path", "") or ""
    last_path_str = getattr(checkpoint_cb, "last_model_path", "") or ""
    chosen_str = best_path_str if (best_path_str and Path(best_path_str).is_file()) \
                                  else last_path_str
    if not (chosen_str and Path(chosen_str).is_file()):
        log.warning(
            "No checkpoint produced. Training likely aborted very early; "
            "check the logs above."
        )
        return model_dir

    epochs_ran = trainer.current_epoch + 1
    stopped_early = epochs_ran < cfg.training.epochs

    # Training curves + compact metrics report.
    training_summary = {"best": None, "epochs": []}
    metrics_csv = next(run_reports.rglob("metrics.csv"), None)
    if metrics_csv is not None and metrics_csv.exists():
        training_summary = summarize_training_metrics(metrics_csv)
        if training_summary.get("epochs"):
            epochs_ran = len(training_summary["epochs"])
            stopped_early = epochs_ran < cfg.training.epochs
        plot_training_curves(
            metrics_csv,
            run_reports / "training_curves.png",
            summary=training_summary,
        )
        write_training_report(
            metrics_csv,
            run_reports / "training_report.json",
            epochs_ran=epochs_ran,
            epochs_max=cfg.training.epochs,
            patience=cfg.training.patience,
        )

    chosen_ckpt = Path(chosen_str)
    checkpoint_path = _move_checkpoint(
        chosen_ckpt,
        _semantic_checkpoint_path(
            model_dir,
            chosen_ckpt,
            training_summary.get("best"),
            max(0, epochs_ran - 1),
        ),
    )
    log.info("model checkpoint → %s", checkpoint_path)

    # --- Build the model manifest ---
    best = training_summary.get("best") or {}
    selected_val_loss = best.get("val_loss", best.get("score"))
    selected_val_accuracy = best.get("val_accuracy")
    best_val_f1 = best.get("val_f1_macro_targets")
    selected_epoch = best.get("epoch") if best else max(0, epochs_ran - 1)
    dataset_snap = _load_dataset_snapshot(layout)
    manifest = {
        "model_id": model_id,
        "description": description or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset": {
            "train": counts["train"], "val": counts["val"], "test": counts["test"],
            "by_class": dataset_snap.get("by_class", {}),
            "test_set_hash": dataset_snap.get("frozen_test", {}).get("hash", ""),
            "frozen_test_present": dataset_snap.get("frozen_test", {}).get("present", False),
        },
        "training": {
            "epochs_ran": epochs_ran,
            "epochs_max": cfg.training.epochs,
            "stopped_early": stopped_early,
            "selection_metric": "val/loss" if has_val else "last checkpoint",
            "selection_mode": "min" if has_val else "last",
            "selection_tiebreak": "val/accuracy" if has_val else "",
            "selected_epoch": selected_epoch,
            "selected_score": selected_val_loss,
            "selected_val_loss": selected_val_loss,
            "selected_val_accuracy": selected_val_accuracy,
            "selected_train_loss": best.get("train_loss"),
            "best_val_f1_macro_targets": best_val_f1,
            "best_val_accuracy": selected_val_accuracy,
            "best_checkpoint_name": checkpoint_path.name,
        },
        "files": {
            "checkpoint": str(checkpoint_path.relative_to(model_dir)),
            "training_config": "training_config.json",
            "runtime_settings": "runtime_settings.json",
            "dataset_snapshot": "dataset/dataset_snapshot.json",
            "training_report": "reports/training_report.json",
            "training_curves": "reports/training_curves.png",
        },
        "exported": False,
        "test_metrics": None,
    }

    # --- Optional auto-test ---
    if do_test:
        try:
            from sed.train.eval import evaluate_run
            metrics = evaluate_run(cfg, project_id, model_id, use_model_dataset=True)
            manifest["test_metrics"] = {
                "threshold": metrics["threshold"],
                "thresholds": metrics["thresholds"],
                "prediction_mode": metrics["prediction_mode"],
                "probability_mode": metrics["probability_mode"],
                "negative_max": metrics["negative_max"],
                "margin_to_second_best": metrics["margin_to_second_best"],
                "f1_macro_targets": metrics["f1_macro_targets"],
                "ap_macro_targets": metrics["ap_macro_targets"],
                "accuracy": metrics["accuracy"],
                "n_samples": metrics["n_samples"],
                "per_class": metrics["per_class"],
            }
            manifest["files"]["test_metrics"] = "reports/test_metrics.json"
            manifest["files"]["confusion_matrix"] = "reports/confusion_matrix.png"
            manifest["files"]["confusion_matrix_json"] = "reports/confusion_matrix.json"
            manifest["files"]["test_predictions"] = "reports/test_predictions.tsv"
        except Exception as exc:  # noqa: BLE001
            log.error("auto_test failed: %s", exc)

    # --- Optional auto-export (ONNX + OpenVINO into the model bundle dir) ---
    if do_export:
        try:
            from sed.export.openvino import export_project
            export_info = export_project(
                cfg, project_id,
                ckpt_path=checkpoint_path,
                out_dir=model_dir,
                dataset_dir=run_dataset,
            )
            manifest["exported"] = True
            manifest["export"] = {
                "source_checkpoint": str(checkpoint_path.relative_to(model_dir)),
                "active_precision": export_info.get("active_precision"),
                "active_target": export_info["active_target"],
            }
            manifest["files"]["active_openvino"] = export_info["active_openvino"]
            if export_info.get("model_int8"):
                manifest["files"]["openvino_int8"] = export_info["model_int8"]
            manifest["files"]["openvino_fp16"] = export_info["model_fp16"]
            manifest["files"]["onnx"] = export_info["onnx"]
        except Exception as exc:  # noqa: BLE001
            log.error("auto_export failed: %s", exc)

    # --- Score the exported (deployed) IR on the frozen test set ---
    # The torch-checkpoint metrics above describe training; production runs
    # the exported IR (typically INT8), so both views go into the manifest.
    if do_test and manifest.get("exported"):
        try:
            from sed.train.eval import evaluate_exported_run
            exported_metrics = evaluate_exported_run(
                cfg, project_id, model_id, use_model_dataset=True
            )
            manifest["test_metrics_exported"] = {
                "model_file": exported_metrics["model_file"],
                "thresholds": exported_metrics["thresholds"],
                "prediction_mode": exported_metrics["prediction_mode"],
                "f1_macro_targets": exported_metrics["f1_macro_targets"],
                "ap_macro_targets": exported_metrics["ap_macro_targets"],
                "accuracy": exported_metrics["accuracy"],
                "n_samples": exported_metrics["n_samples"],
                "per_class": exported_metrics["per_class"],
                "test_set_hash": exported_metrics["test_set_hash"],
            }
            manifest["files"]["test_metrics_exported"] = "reports/test_metrics_exported.json"
            torch_f1 = (manifest.get("test_metrics") or {}).get("f1_macro_targets")
            exported_f1 = exported_metrics["f1_macro_targets"]
            if torch_f1 is not None and torch_f1 - exported_f1 > 0.02:
                log.warning(
                    "exported %s scores F1(targets)=%.3f vs %.3f for the torch "
                    "checkpoint — quantization cost the model accuracy. Consider "
                    "deploying the FP16 IR instead (point active_openvino.xml at "
                    "model_fp16.xml) or recalibrating thresholds.",
                    exported_metrics["model_file"], exported_f1, torch_f1,
                )
        except Exception as exc:  # noqa: BLE001
            log.error("exported-model test scoring failed: %s", exc)

    registry.save_manifest(project_id, model_id, manifest)
    log.info("model manifest → %s", model_dir / "manifest.json")

    # --- Make the freshly trained model the active one (it deployed cleanly) ---
    if do_export and manifest.get("exported"):
        try:
            registry.set_active_model(project_id, model_id)
            log.info("active_model → %s", model_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("auto-select failed: %s", exc)

    return model_dir
