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

import csv
import json
import logging
import math
import os
import shutil
import subprocess
import time
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
    validate_clink_cnn_v2_feature_contract,
    validate_pretrained_feature_contract,
)
from sed.train import registry
from sed.train.batching import training_batch_sizes
from sed.train.datamodule import SedDataModule
from sed.train.lit_module import (
    VAL_FIXED_FP_KEY,
    VAL_FIXED_RECALL_KEY,
    SedLitModule,
    _fmt_param_count,
    load_classes,
)
from sed.train.plot import (
    SUPERVISED_SELECTION_POLICY_NAME,
    plot_training_curves,
    summarize_training_metrics,
    supervised_selection_rank,
    write_training_report,
)

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

class BestValCheckpoint(Callback):
    """Save the best-validation checkpoint for a configurable monitor.

    Supervised runs use the fixed-threshold false-positive policy persisted by
    :func:`sed.train.plot.supervised_selection_policy`.  Runs that never log
    classifier metrics (BYOL) retain scalar ``val/loss`` selection with
    ``val/byol_score`` as a tie-break.
    """

    def __init__(
        self,
        dirpath: Path,
        monitor: str = VAL_FIXED_RECALL_KEY,
        mode: str = "lexicographic",
        tiebreak: str = "val/loss",
        tiebreak_mode: str = "min",
        min_delta: float = 0.001,
        selection_policy: str | None = None,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.tiebreak = tiebreak
        self.tiebreak_mode = tiebreak_mode
        self.min_delta = float(min_delta)
        self.selection_policy = selection_policy or (
            SUPERVISED_SELECTION_POLICY_NAME
            if mode == "lexicographic"
            else "scalar"
        )
        self.best_model_path = ""
        self.best_score: float | None = None
        self.best_tiebreak: float | None = None
        self.best_epoch: int | None = None
        self.best_selection_metrics: dict[str, float] | None = None

    def _is_better(self, score: float, tiebreak: float | None) -> bool:
        if self.best_score is None:
            return True
        sign = 1.0 if self.mode == "max" else -1.0
        if sign * (score - self.best_score) > self.min_delta:
            return True
        if abs(score - self.best_score) <= self.min_delta and tiebreak is not None:
            if self.best_tiebreak is None:
                return True
            tb_sign = 1.0 if self.tiebreak_mode == "max" else -1.0
            return tb_sign * (tiebreak - self.best_tiebreak) > 0
        return False

    def _is_better_supervised(self, candidate: dict[str, float]) -> bool:
        candidate_rank = supervised_selection_rank(candidate)
        if candidate_rank is None:
            return False
        if self.best_selection_metrics is None:
            return True
        incumbent_rank = supervised_selection_rank(self.best_selection_metrics)
        return incumbent_rank is None or candidate_rank > incumbent_rank

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        _ = pl_module
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if self.selection_policy == SUPERVISED_SELECTION_POLICY_NAME:
            candidate: dict[str, float] = {}
            for key in (
                VAL_FIXED_RECALL_KEY,
                VAL_FIXED_FP_KEY,
                "val/ap_macro_targets",
                "val/loss",
            ):
                value = _metric_as_float(metrics.get(key))
                if value is None:
                    return
                candidate[key] = value
            if not self._is_better_supervised(candidate):
                return
            score = candidate[VAL_FIXED_RECALL_KEY]
            tiebreak = candidate[VAL_FIXED_FP_KEY]
            self.best_selection_metrics = candidate
        else:
            score = _metric_as_float(metrics.get(self.monitor))
            if score is None:
                return
            tiebreak = _metric_as_float(metrics.get(self.tiebreak))
            if not self._is_better(score, tiebreak):
                return
        previous = Path(self.best_model_path) if self.best_model_path else None
        self.best_score = score
        self.best_tiebreak = tiebreak
        self.best_epoch = int(trainer.current_epoch)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        path = self.dirpath / _best_checkpoint_name(self.best_epoch)
        trainer.save_checkpoint(path)
        self.best_model_path = str(path)
        if previous is not None and previous.exists() and previous != path:
            previous.unlink()


class GracefulStopCallback(Callback):
    """Stop training at a clean epoch boundary when Inspect writes a stop file."""

    def __init__(self, stop_file: Path, *, has_val: bool):
        super().__init__()
        self.stop_file = Path(stop_file)
        self.has_val = bool(has_val)
        self.triggered = False

    def _maybe_stop(self, trainer, where: str) -> None:  # noqa: ANN001
        if self.triggered or trainer.sanity_checking or not self.stop_file.exists():
            return
        self.triggered = True
        trainer.should_stop = True
        log.warning("graceful stop requested after %s; finalizing model bundle", where)

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        _ = pl_module
        if self.has_val:
            self._maybe_stop(trainer, "validation")

    def on_train_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        _ = pl_module
        if not self.has_val:
            self._maybe_stop(trainer, "train epoch")


def _quiet_lightning_info_logs() -> None:
    lightning_logger = logging.getLogger("lightning")
    lightning_logger.handlers.clear()
    lightning_logger.setLevel(logging.WARNING)
    lightning_logger.propagate = True


def _model_stats_block(
    cfg,
    total_params: int,
    trainable_params: int,
    fine_tune_info: dict | None = None,
) -> dict:
    """Manifest ``model_stats``: parameter counts + resolved input geometry.

    ``trainable_params`` is the count at the fine-tune target depth (what
    ultimately trains), not the warm-up subset. ``fine_tune_info`` (the model's
    target ``freeze_info``) supplies ``n_stages`` + trained-stage count so the
    Runs table can render depth as "trained/total stages"; both are optional and
    absent on backbones without a stage decomposition.
    """
    from sed.train.backbones import model_input_geometry

    block = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        **model_input_geometry(cfg),
        "source": "train",
    }
    if isinstance(fine_tune_info, dict):
        n_stages = fine_tune_info.get("n_stages")
        trained_stages = fine_tune_info.get("trainable_stages")
        if n_stages:
            block["n_stages"] = int(n_stages)
        if trained_stages is not None:
            block["trained_stages"] = len(trained_stages)
    return block


def _metric_as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_metric(metrics: dict, *keys: str) -> object | None:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


class EpochConsoleLogger(Callback):
    """Emit stable per-epoch text for CLI and Inspect live logs."""

    def __init__(self, max_epochs: int, *, has_val: bool):
        super().__init__()
        self.max_epochs = int(max_epochs)
        self.has_val = bool(has_val)
        self._emitted: set[int] = set()
        self._inline_active = False
        self._inline_width = 0
        self._epoch_started_at = 0.0

    @staticmethod
    def _fmt(value: float | None, digits: int = 4) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    def _line(self, trainer) -> str:  # noqa: ANN001
        metrics = trainer.callback_metrics
        train_loss = _metric_as_float(_first_metric(metrics, "train/loss_epoch", "train/loss"))
        val_loss = _metric_as_float(metrics.get("val/loss"))
        val_acc = _metric_as_float(metrics.get("val/accuracy"))
        val_f1 = _metric_as_float(metrics.get("val/f1_macro_targets"))
        val_recall_fixed = _metric_as_float(metrics.get(VAL_FIXED_RECALL_KEY))
        val_fp_fixed = _metric_as_float(metrics.get(VAL_FIXED_FP_KEY))
        lr = self._current_lr(metrics)
        if lr is None:
            lr = self._optimizer_lr(trainer)
        elapsed = max(0.0, time.monotonic() - self._epoch_started_at) if self._epoch_started_at else 0.0
        total = self._batch_total(trainer) or 0
        parts = [self._progress_line(total, total, elapsed, train_loss, lr, done=True)]
        if self.has_val:
            parts.extend([
                f"val_loss: {self._fmt(val_loss)}",
                f"val_accuracy: {self._fmt(val_acc)}",
                f"val_f1: {self._fmt(val_f1)}",
                f"val_recall@0.9: {self._fmt(val_recall_fixed)}",
                f"val_FP/1k@0.9: {self._fmt(val_fp_fixed, digits=1)}",
            ])
        return " - ".join(parts)

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = int(seconds // 60)
        return f"{minutes}m {int(seconds % 60)}s"

    @staticmethod
    def _bar(step: int, total: int, width: int = 30) -> str:
        if total <= 0:
            return "." * width
        done = min(width, max(0, int(width * step / total)))
        if step >= total:
            return "=" * width
        if done <= 0:
            return ">" + "." * (width - 1)
        return "=" * max(0, done - 1) + ">" + "." * max(0, width - done)

    @staticmethod
    def _current_lr(metrics: dict) -> float | None:
        for key, value in metrics.items():
            if str(key).startswith(("lr-", "lr/")):
                lr = _metric_as_float(value)
                if lr is not None:
                    return lr
        return None

    @staticmethod
    def _optimizer_lr(trainer) -> float | None:  # noqa: ANN001
        try:
            for optimizer in getattr(trainer, "optimizers", []) or []:
                for group in optimizer.param_groups:
                    lr = _metric_as_float(group.get("lr"))
                    if lr is not None:
                        return lr
        except Exception:  # noqa: BLE001
            return None
        return None

    def _progress_line(
        self,
        step: int,
        total: int,
        elapsed: float,
        loss: float | None,
        lr: float | None,
        *,
        done: bool = False,
    ) -> str:
        total = max(1, int(total or 1))
        step = min(max(0, int(step)), total)
        per_step = elapsed / max(1, step) if step else 0.0
        eta = "" if done else f"ETA: {self._duration(per_step * max(0, total - step))}"
        prefix = f"{step}/{total} [{self._bar(step, total)}]"
        timing = self._duration(elapsed) if done else eta
        parts = [prefix, timing, f"{per_step * 1000.0:.0f}ms/step"]
        if loss is not None:
            parts.append(f"loss: {self._fmt(loss)}")
        if lr is not None:
            parts.append(f"lr: {lr:.3g}")
        return " - ".join(part for part in parts if part)

    @staticmethod
    def _batch_total(trainer) -> int | None:  # noqa: ANN001
        try:
            total = trainer.num_training_batches
            if total is None or not math.isfinite(float(total)):
                return None
            return max(1, int(total))
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _batch_loss(outputs: object) -> float | None:
        if isinstance(outputs, dict):
            return _metric_as_float(outputs.get("loss"))
        return _metric_as_float(outputs)

    def _write_inline(self, line: str) -> None:
        padding = " " * max(0, self._inline_width - len(line))
        print("\r" + line + padding, end="", flush=True)
        self._inline_active = True
        self._inline_width = len(line)

    def _finish_inline(self) -> None:
        if self._inline_active:
            print("", flush=True)
            self._inline_active = False
            self._inline_width = 0

    def _emit_once(self, trainer) -> None:  # noqa: ANN001
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        if epoch in self._emitted:
            return
        self._emitted.add(epoch)
        self._finish_inline()
        print(self._line(trainer), flush=True)

    def on_train_start(self, trainer, pl_module) -> None:  # noqa: ANN001
        _ = trainer, pl_module
        print(f"[train] started | epochs={self.max_epochs}", flush=True)

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # noqa: ANN001
        _ = pl_module
        self._finish_inline()
        self._epoch_started_at = time.monotonic()
        print(f"Epoch {int(trainer.current_epoch) + 1}/{self.max_epochs}", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001
        _ = pl_module, batch
        total = self._batch_total(trainer)
        if total is None:
            return
        step = int(batch_idx) + 1
        loss = self._batch_loss(outputs)
        lr = self._current_lr(trainer.callback_metrics)
        if lr is None:
            lr = self._optimizer_lr(trainer)
        elapsed = max(0.0, time.monotonic() - self._epoch_started_at) if self._epoch_started_at else 0.0
        self._write_inline(
            self._progress_line(step, total, elapsed, loss, lr)
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        _ = pl_module
        if not self.has_val:
            self._emit_once(trainer)

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        _ = pl_module
        if self.has_val:
            self._emit_once(trainer)

    def on_train_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        _ = trainer, pl_module
        self._finish_inline()
        print("[train] finished", flush=True)


def _train_log_every_n_steps(n_rows: int, batch_size: int) -> int:
    if batch_size <= 0:
        return 1
    if n_rows >= 2 * batch_size:
        n_batches = max(1, n_rows // batch_size)
    else:
        n_batches = max(1, math.ceil(n_rows / batch_size))
    return max(1, min(10, n_batches))


def _trainer_batching_kwargs(training) -> dict[str, int]:  # noqa: ANN001
    """Lightning kwargs derived from the shared batch-size contract."""
    batching = training_batch_sizes(training)
    return {"accumulate_grad_batches": batching["accumulate_grad_batches"]}


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


def _environment_versions() -> dict[str, str]:
    """Library versions that materially affect training/export numerics.

    The repo SHA alone can't reproduce a run: torch/NNCF/OpenVINO releases
    change kernels and INT8 quantization behaviour.
    """
    import platform
    from importlib import metadata

    out = {"python": platform.python_version()}
    for pkg in ("torch", "lightning", "numpy", "openvino", "nncf", "onnx"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:  # noqa: BLE001 - optional packages may be absent
            out[pkg] = ""
    return out


def _dataset_preset_provenance(layout: ProjectLayout, cfg: Config) -> dict:
    """The named dataset definition this run was trained from.

    A project is worked by switching presets — a pretraining corpus, then a
    venue fine-tune — and each carries its own class space. Recording the name
    and classes with the run is what lets Runs & Compare say *which* dataset a
    number came from, and lets two runs be recognised as measuring different
    things rather than looking directly comparable.

    ``modified`` matters as much as the name: a preset can be loaded and then
    edited, in which case the run was not trained on the saved definition.
    """
    try:
        from sed.data.compose import dataset_preset_state, load_dataset_selection

        state = dataset_preset_state(layout, load_dataset_selection(layout), cfg)
    except Exception as exc:  # noqa: BLE001 — provenance must never fail a run
        log.debug("dataset preset provenance unavailable: %s", exc)
        return {}
    active = str(state.get("active") or "")
    entry = next(
        (p for p in state.get("presets", []) if p.get("slug") == active), {}
    ) if active else {}
    return {
        "preset_slug": active,
        "preset_name": str(entry.get("name") or state.get("active_name") or ""),
        "preset_modified": bool(state.get("modified")),
        "preset_classes": list(entry.get("classes") or cfg.dataset.target_classes),
    }


def _optional_file_sha256(path: Path) -> str:
    from sed.util import sha256_file

    try:
        return sha256_file(path) if path.exists() else ""
    except OSError:
        return ""


def _effective_scaling_method(features) -> str:
    """scaling_method with the legacy ``standardize`` bool honored.

    Config validation migrates ``standardize`` into ``scaling_method``;
    programmatic callers that mutated a validated Config in place may not have
    re-validated, so the legacy spelling still counts as a declared z-score.
    """
    method = str(getattr(features, "scaling_method", "none") or "none")
    if method == "none" and bool(getattr(features, "standardize", False)):
        return "zscore"
    return method


def _feature_standardization_provenance(cfg: Config) -> dict[str, object]:
    """Stable manifest record for dataset-fitted feature normalization."""
    features = cfg.features
    scaling_stats = dict(getattr(features, "scaling_stats", {}) or {})
    method = _effective_scaling_method(features)
    return {
        "enabled": method != "none",
        "method": method,
        "scope": str(getattr(features, "scaling_scope", "global") or "global"),
        "feature_type": str(getattr(features, "feature_type", "mel")),
        "resolved": bool(getattr(features, "standardize_stats_resolved", True)),
        "mean": float(getattr(features, "standardize_mean", 0.0)),
        "std": float(getattr(features, "standardize_std", 1.0)),
        # Fitted band count only — the full vectors already live in the frozen
        # [features] block of this same snapshot.
        "bands": max((len(v) for v in scaling_stats.values()), default=0),
        "dataset_sha256": str(
            getattr(features, "standardize_stats_dataset_sha256", "") or ""
        ),
        "split_seed": int(
            getattr(features, "standardize_stats_split_seed", 0) or 0
        ),
        "val_fraction": float(
            getattr(features, "standardize_stats_val_fraction", 0.0) or 0.0
        ),
        "clips": int(getattr(features, "standardize_stats_clips", 0) or 0),
    }


def _validate_fitted_feature_scaling(cfg: Config, train_csv: Path) -> None:
    """Reject unresolved or stale fitted scaling before allocating a bundle.

    Applies to every dataset-fitted scaling method (z-score / min-max, global
    or per-band, mel or linear). Scratch/domain-pretraining runs fit their own
    live grouped training split, so the dataset hash and split seed must still
    match. A parent fine-tune deliberately inherits its parent's exact
    normalization; refitting on the child data would invalidate encoder
    compatibility, so the parent's fitted provenance remains authoritative.
    Resolved statistics without a dataset hash are a manual/checkpoint contract
    (an external encoder's own constants) — legal for mel; the experimental
    linear presets must always be fitted on this dataset.
    """
    features = cfg.features
    if _effective_scaling_method(features) == "none":
        return

    provenance = _feature_standardization_provenance(cfg)
    is_linear = str(getattr(features, "feature_type", "mel")) == "linear"
    expected_sha = str(provenance["dataset_sha256"]).strip().lower()
    problems: list[str] = []
    if not provenance["resolved"]:
        problems.append("statistics are not resolved")
    if not expected_sha:
        if is_linear:
            problems.append("statistics dataset SHA-256 is missing")
        elif not problems:
            return  # resolved manual/checkpoint constants — no fit to check
    if expected_sha and int(provenance["clips"]) <= 0:
        problems.append("statistics clip count must be positive")
    inherits_parent_contract = str(cfg.model.initialization) == "parent"
    if expected_sha and not inherits_parent_contract:
        training_seed = int(cfg.training.seed)
        if int(provenance["split_seed"]) != training_seed:
            problems.append(
                "statistics split seed "
                f"{provenance['split_seed']} does not match training seed {training_seed}"
            )
        current_val_fraction = float(cfg.dataset.val_fraction)
        if abs(float(provenance["val_fraction"]) - current_val_fraction) > 1e-9:
            problems.append(
                "statistics validation fraction "
                f"{provenance['val_fraction']} does not match training validation fraction "
                f"{current_val_fraction}"
            )
        current_sha = _optional_file_sha256(train_csv).lower()
        if not current_sha:
            problems.append("current train.csv could not be hashed")
        elif current_sha != expected_sha:
            problems.append(
                "current train.csv SHA-256 does not match the statistics dataset"
            )
    if problems:
        raise RuntimeError(
            "Dataset-fitted feature scaling requires fresh train-only statistics: "
            + "; ".join(problems)
            + ". Re-estimate normalization statistics from the current grouped "
            "training split for scratch training, or adopt the exact resolved "
            "frontend from the selected parent run."
        )


def _load_dataset_snapshot(layout: ProjectLayout) -> dict:
    path = layout.dataset_dir / "dataset_snapshot.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _copy_dataset_artifacts(src_dir: Path, dst_dir: Path) -> None:
    """Copy the composed dataset into the model bundle for provenance.

    val.csv is not among these: it isn't composed ahead of time, so
    ``_write_val_csv_snapshot`` writes this run's actual live-split val rows
    into ``dst_dir`` separately, once the datamodule has split them.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "test.csv", "classes.json", "dataset_snapshot.json"):
        src = src_dir / name
        if not src.exists():
            raise RuntimeError(f"Missing dataset artifact for the model bundle: {src}")
        shutil.copy2(src, dst_dir / name)
    for name in ("frozen_test_manifest.json", "frozen_test_split.json"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


_TRAIN_CSV_COLUMNS = [
    "path", "class_name", "session_id", "sample_rate", "duration_s",
    "is_target", "origin", "origin_class", "source_stem",
    "split_bucket", "split_group", "original_source", "augmentation_family",
    "source_split_group", "frozen_source",
]


def _write_val_csv_snapshot(rows: list[dict], dst_path: Path) -> None:
    """Freeze this run's live train/val split into the model bundle.

    Val isn't composed ahead of time (sed.data.compose writes only
    train.csv, the undivided pool) — this is the only record of exactly
    which clips this run held out, for provenance (``val_csv_sha256`` in the
    manifest) and for INT8 export calibration to read.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_TRAIN_CSV_COLUMNS)
        for row in rows:
            w.writerow([row.get(col, "") for col in _TRAIN_CSV_COLUMNS])


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
    # One preflight owns both constructor-parameter strictness and automatic
    # classifier-dropout resolution.  The returned config is what the module
    # builds and what the run snapshot freezes.
    from sed.train.model_lab import materialize_training_config

    cfg = materialize_training_config(cfg)
    validate_clink_cnn_v2_feature_contract(cfg, phase="training")
    layout = ProjectLayout(project_id)
    layout.create()

    # Pre-flight on the dataset CSVs. train.csv is the whole train+val pool;
    # val is carved out live, at train time, once the datamodule is built.
    csvs = {name: layout.dataset_dir / f"{name}.csv" for name in ("train", "test")}
    if not csvs["train"].exists():
        raise RuntimeError(
            f"No dataset for project '{project_id}'. Run `sed dataset compose` first."
        )
    from sed.data.compose import load_dataset_dirty

    dirty = load_dataset_dirty(layout)
    if dirty:
        count = int(dirty.get("count") or len(dirty.get("paths") or []))
        raise RuntimeError(
            f"The composed dataset references {count} deleted source clip"
            f"{'s' if count != 1 else ''}. Run `sed dataset compose --project "
            f"{project_id}` before training."
        )
    counts = {name: _count_csv_rows(p) for name, p in csvs.items()}
    log.info("dataset rows — train+val pool=%d test=%d", counts["train"], counts["test"])
    if counts["train"] == 0:
        raise RuntimeError(
            "train.csv is empty — can't train on zero rows. "
            f"Run `sed dataset compose --project {project_id}` first."
        )
    _validate_fitted_feature_scaling(cfg, csvs["train"])
    if counts["test"] == 0:
        log.warning("test.csv is empty — final test-set evaluation will be skipped.")

    # Which frozen-test decision rules to score when training finishes. Same rule
    # ids Batch evaluate writes, so a fresh run lands in the same runs-table
    # columns as a re-scored one instead of in a parallel legacy block.
    test_rules = cfg.training.resolved_auto_test_rules()
    if auto_test is False:
        test_rules = []
    do_test = bool(test_rules) and counts["test"] > 0
    do_export = cfg.training.auto_export if auto_export is None else auto_export
    if cfg.training.pretraining_objective == "byol":
        if cfg.model.task != "clip":
            raise RuntimeError("BYOL pretraining currently supports clip task only")
        # A BYOL checkpoint has an encoder but no trained classifier head.
        # Linear-probe reporting is stored separately by the lab; it must never
        # masquerade as a production frozen-test model or be exported.
        do_test = False
        do_export = False
    if do_export:
        try:
            from sed.train.backbones import resolve_adapter

            export_supported = bool(resolve_adapter(cfg.model.backbone).spec.export_supported)
        except Exception:  # noqa: BLE001 - export will report the real issue later
            export_supported = True
        if not export_supported:
            log.warning(
                "auto_export disabled: %s is marked export-unsupported in the model catalog",
                cfg.model.backbone,
            )
            do_export = False

    L.seed_everything(cfg.training.seed, workers=True)

    classes = load_classes(layout.dataset_dir)
    _validate_dataset_classes_match_config(cfg, classes)
    if cfg.model.task == "frame":
        from sed.train.backbones import resolve_adapter
        adapter = resolve_adapter(cfg.model.backbone)
        if "frame" not in set(adapter.spec.task_capabilities):
            raise RuntimeError(
                f"{cfg.model.backbone} is not a declared frame-capable encoder. "
                "Choose a temporal KWS/EfficientAT frame recipe or use clip task."
            )
    # Never block on the feature design — warn only. A linear/PCEN/off-contract
    # feature trains from scratch (scratch/random initialization in the Model tab).
    validate_pretrained_feature_contract(cfg, phase="training")
    log.info("Training on %d classes: %s", len(classes), classes)

    # --- Allocate the model bundle dir and freeze the effective config there ---
    model_id = registry.new_model_id(project_id, description=description)
    model_dir = layout.model_dir(model_id)
    run_reports = model_dir / "reports"
    for d in (model_dir, run_reports):
        d.mkdir(parents=True, exist_ok=True)

    from sed.train.model_lab import (
        cached_weight_provenance,
        recipe_from_config,
        recipe_signature,
        snapshot_project_architecture,
        weight_by_id,
    )
    recipe = recipe_from_config(cfg)
    code_snapshot_hash = snapshot_project_architecture(project_id, recipe, model_dir / "architecture")
    weight = None
    weight_provenance = None
    # Only an upstream initialization actually loads the catalog checkpoint;
    # random/parent runs must not record provenance for weights never loaded.
    if cfg.model.initialization == "upstream" and cfg.model.pretrained:
        try:
            weight = weight_by_id(recipe.architecture_id, recipe.weight_id)
            weight_provenance = weight.to_payload() if weight else None
        except KeyError:
            # Legacy/informal timm identifiers retain their normal trainer path;
            # their exact source still appears in the frozen config snapshot.
            weight_provenance = None

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

    # Split train/val live, now, seeded by cfg.training.seed — setup() is
    # idempotent, so Lightning's own setup() call inside trainer.fit() below
    # reuses this same split rather than re-randomizing it mid-run.
    dm = SedDataModule(cfg=cfg, project_id=project_id)
    dm.setup()
    counts["val"] = len(dm.val_ds)
    log.info("train/val split (seed=%d) — train=%d val=%d",
             cfg.training.seed, len(dm.train_ds), counts["val"])
    if counts["val"] == 0:
        log.warning(
            "val split is empty. Training will run without held-out validation — "
            "EarlyStopping and ModelCheckpoint's monitored metric won't be "
            "computed, so the trainer will run for the full `epochs` budget."
        )
    _write_val_csv_snapshot(dm.val_ds.rows, run_dataset / "val.csv")

    if cfg.training.pretraining_objective == "byol":
        from sed.train.ssl import ByolLitModule
        model = ByolLitModule(cfg=cfg, classes=classes)
    else:
        model = SedLitModule(cfg=cfg, classes=classes)
    if weight is not None:
        weight_provenance = cached_weight_provenance(weight)
    parent_provenance = None
    if cfg.model.initialization == "parent":
        from sed.train.transfer import initialize_encoder_from_parent

        parent_provenance = initialize_encoder_from_parent(model, cfg)
        log.info("loaded encoder from %s/%s", cfg.model.parent_project, cfg.model.parent_run)
        head_transfer = parent_provenance.get("head_transfer") or {}
        warm = list(head_transfer.get("transferred_classes") or [])
        if warm:
            fresh = list(head_transfer.get("new_classes") or [])
            log.info(
                "classifier head warm-started from the parent for %d/%d class(es): %s%s",
                len(warm), len(classes), ", ".join(warm),
                f" — random for {', '.join(fresh)}" if fresh else "",
            )
        elif head_transfer.get("skipped_reason"):
            log.info(
                "classifier head starts random (%s)", head_transfer["skipped_reason"],
            )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    csv_logger = CSVLogger(save_dir=str(run_reports), name="training")

    has_val = counts["val"] > 0
    callbacks: list = [
        LearningRateMonitor(logging_interval="epoch"),
        EpochConsoleLogger(cfg.training.epochs, has_val=has_val),
    ]
    graceful_stop_cb = None
    stop_file_raw = os.environ.get("SED_TRAIN_STOP_FILE", "").strip()
    if stop_file_raw:
        graceful_stop_cb = GracefulStopCallback(Path(stop_file_raw), has_val=has_val)
        callbacks.append(graceful_stop_cb)
    if has_val:
        # BYOL logs only an SSL val/loss. Supervised checkpoint selection uses
        # the production-style fixed-threshold lexicographic policy; AP remains
        # the scalar early-stopping signal because Lightning's built-in callback
        # accepts one monitored value.
        if cfg.training.pretraining_objective == "byol":
            early_stop_monitor, early_stop_mode = "val/loss", "min"
            checkpoint_cb = BestValCheckpoint(
                model_dir, monitor=early_stop_monitor, mode=early_stop_mode,
                tiebreak="val/byol_score", tiebreak_mode="max", min_delta=0.001,
                selection_policy="scalar",
            )
        else:
            early_stop_monitor, early_stop_mode = "val/ap_macro_targets", "max"
            checkpoint_cb = BestValCheckpoint(
                model_dir,
                selection_policy=SUPERVISED_SELECTION_POLICY_NAME,
            )
        callbacks.append(checkpoint_cb)
        callbacks.append(EarlyStopping(
            monitor=early_stop_monitor, mode=early_stop_mode,
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

    configured_batching = training_batch_sizes(cfg.training)
    log_every_n_steps = _train_log_every_n_steps(
        counts["train"], configured_batching["physical_batch_size"],
    )
    fi = getattr(model, "freeze_info", None)
    target_fi = getattr(model, "target_freeze_info", None)
    plan = getattr(model, "fine_tune_plan", None)
    target_params = (
        int(target_fi.get("trainable_params"))
        if isinstance(target_fi, dict) and target_fi.get("trainable_params") is not None
        else trainable_params
    )
    if plan is not None and getattr(plan, "warmup_epochs", 0):
        log.info(
            "fine-tune policy=%s: warm-up depth=%s for %d epoch(s), then target depth=%s",
            getattr(plan, "policy", ""),
            getattr(plan, "initial_depth", ""),
            int(getattr(plan, "warmup_epochs", 0)),
            getattr(plan, "target_depth", ""),
        )
    log.info(
        "model params: trainable_now=%s target_trainable=%s total=%s",
        _fmt_param_count(trainable_params),
        _fmt_param_count(target_params),
        _fmt_param_count(total_params),
    )
    if fi:
        log.info(
            "train depth now=%s: %d/%d stages trainable (%.0f%% of params)%s",
            fi.get("train_depth"),
            len(fi.get("trainable_stages") or []),
            fi.get("n_stages", 0),
            100.0 * float(fi.get("trainable_fraction") or 1.0),
            " · BN frozen" if fi.get("freeze_batchnorm") else "",
        )
    target_differs = bool(
        target_fi and (
            not fi
            or target_fi.get("train_depth") != fi.get("train_depth")
            or int(target_fi.get("trainable_params") or 0) != int(fi.get("trainable_params") or 0)
        )
    )
    if target_differs:
        warmup_epochs = int(getattr(plan, "warmup_epochs", 0) or 0) if plan is not None else 0
        suffix = f" after {warmup_epochs} warm-up epoch(s)" if warmup_epochs else ""
        log.info(
            "train depth target%s: %s, %d/%d stages trainable (%.0f%% of params)%s",
            suffix,
            target_fi.get("train_depth"),
            len(target_fi.get("trainable_stages") or []),
            target_fi.get("n_stages", 0),
            100.0 * float(target_fi.get("trainable_fraction") or 1.0),
            " · BN frozen" if target_fi.get("freeze_batchnorm") else "",
        )
    log.info(
        "trainer: device=%s precision=%s epochs=%d physical_batch=%d "
        "accumulate=%d effective_batch=%d validation=%s",
        _device_summary(),
        cfg.training.precision,
        cfg.training.epochs,
        configured_batching["physical_batch_size"],
        configured_batching["accumulate_grad_batches"],
        configured_batching["effective_batch_size"],
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
        **_trainer_batching_kwargs(cfg.training),
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

    # --- Optional BatchNorm running-stats recalibration. Runs before the
    # manifest/calibration/auto-test/export chain so every downstream step
    # consumes the recalibrated checkpoint. BYOL runs skip it (the linear
    # probe evaluates the raw representation).
    bn_recal_info = None
    if cfg.training.bn_recalibrate and cfg.training.pretraining_objective != "byol":
        try:
            from sed.train.bn_recalibrate import recalibrate_batchnorm

            info = recalibrate_batchnorm(
                cfg, project_id,
                checkpoint_path=checkpoint_path,
                dataset_dir=run_dataset,
            )
            if info.get("checkpoint"):
                checkpoint_path = Path(info["checkpoint"])
                bn_recal_info = {
                    "bn_layers": info["bn_layers"],
                    "batches": info["batches"],
                    "samples": info["samples"],
                    "checkpoint": checkpoint_path.name,
                }
                log.info("bn recalibration → %s", checkpoint_path.name)
        except Exception as exc:  # noqa: BLE001
            log.warning("bn recalibration failed (continuing with fit checkpoint): %s", exc)

    ssl_linear_probe = None
    if cfg.training.pretraining_objective == "byol":
        try:
            from sed.train.ssl import linear_probe_evaluation

            ssl_linear_probe = linear_probe_evaluation(
                cfg, project_id, checkpoint_path, classes, dataset_dir=run_dataset,
            )
            with open(run_reports / "ssl_linear_probe.json", "w", encoding="utf-8") as f:
                json.dump(ssl_linear_probe, f, indent=2)
            log.info(
                "SSL linear probe: F1(targets)=%.3f accuracy=%.3f",
                ssl_linear_probe["f1_macro_targets"], ssl_linear_probe["accuracy"],
            )
        except Exception as exc:  # noqa: BLE001
            log.error("SSL linear-probe evaluation failed: %s", exc)

    # --- Build the model manifest ---
    best = training_summary.get("best") or {}
    selected_val_loss = best.get("val_loss", best.get("score"))
    selected_val_accuracy = best.get("val_accuracy")
    selected_val_byol_score = best.get("val_byol_score")
    best_val_f1 = best.get("val_f1_macro_targets")
    best_val_ap = best.get("val_ap_macro_targets")
    selected_val_recall_fixed = best.get("val_recall_macro_targets_at_0_9")
    selected_val_fp_fixed = best.get(
        "val_positive_triggers_per_1000_negatives_at_0_9"
    )
    selection_policy_details = (
        best.get("selection_policy")
        or training_summary.get("selection_policy")
    )
    selected_epoch = best.get("epoch") if best else max(0, epochs_ran - 1)
    dataset_snap = _load_dataset_snapshot(layout)
    actual_batching = training_batch_sizes(
        cfg.training,
        world_size=max(1, int(getattr(trainer, "world_size", 1) or 1)),
    )
    manifest = {
        "model_id": model_id,
        "description": description or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": _environment_versions(),
        "model_recipe": {
            **recipe.to_payload(),
            "signature": recipe_signature(recipe),
            "weight": weight_provenance,
            "custom_code_snapshot_sha256": code_snapshot_hash,
        },
        "initialization": parent_provenance or {
            "mode": cfg.model.initialization,
            "weight_id": cfg.model.weight_id,
        },
        "model_stats": _model_stats_block(
            cfg, total_params, target_params,
            fine_tune_info=target_fi if isinstance(target_fi, dict) else fi,
        ),
        "dataset": {
            "train": counts["train"], "val": counts["val"], "test": counts["test"],
            "by_class": dataset_snap.get("by_class", {}),
            "by_origin": dataset_snap.get("by_origin", {}),
            # Which named dataset definition produced this run. A project moves
            # between definitions (pretraining corpus -> venue fine-tune) by
            # switching presets, so without the name a run's numbers cannot be
            # attributed to the data that made them.
            **_dataset_preset_provenance(layout, cfg),
            "train_csv_sha256": _optional_file_sha256(run_dataset / "train.csv"),
            "val_csv_sha256": _optional_file_sha256(run_dataset / "val.csv"),
            "test_set_hash": dataset_snap.get("frozen_test", {}).get("hash", ""),
            "frozen_test_present": dataset_snap.get("frozen_test", {}).get("present", False),
            "frozen_test_source": dataset_snap.get("frozen_test", {}).get("source", ""),
            "frozen_test_source_counts": dataset_snap.get("frozen_test", {}).get("source_counts", {}),
            "frozen_test_manifest": dataset_snap.get("frozen_test", {}).get("manifest", ""),
            "feature_standardization": _feature_standardization_provenance(cfg),
            "split_diagnostics": getattr(dm, "split_diagnostics", {}),
        },
        "training": {
            **actual_batching,
            "epochs_ran": epochs_ran,
            "epochs_max": cfg.training.epochs,
            "stopped_early": stopped_early,
            "selection_metric": (best.get("monitor") or "val/loss") if has_val else "last checkpoint",
            "selection_mode": (best.get("mode") or "min") if has_val else "last",
            "selection_tiebreak": (best.get("selection_tiebreak") or "") if has_val else "",
            "selection_policy": selection_policy_details if has_val else None,
            "selected_epoch": selected_epoch,
            "selected_score": best.get("score", selected_val_loss),
            "selected_val_loss": selected_val_loss,
            "selected_val_accuracy": selected_val_accuracy,
            "selected_val_byol_score": selected_val_byol_score,
            "selected_val_recall_macro_targets_at_0_9": selected_val_recall_fixed,
            "selected_val_positive_triggers_per_1000_negatives_at_0_9": (
                selected_val_fp_fixed
            ),
            "selected_val_negative_clip_count": best.get("val_negative_clip_count"),
            "selected_val_positive_trigger_count_on_negatives_at_0_9": best.get(
                "val_positive_trigger_count_on_negatives_at_0_9"
            ),
            "selected_train_loss": best.get("train_loss"),
            "best_val_f1_macro_targets": best_val_f1,
            "best_val_ap_macro_targets": best_val_ap,
            "best_val_accuracy": selected_val_accuracy,
            "best_checkpoint_name": checkpoint_path.name,
            "stopped_by_request": bool(graceful_stop_cb and graceful_stop_cb.triggered),
            "fine_tune_plan": getattr(model, "fine_tune_plan", None).to_payload()
            if getattr(model, "fine_tune_plan", None) else {},
            "objective": cfg.training.pretraining_objective,
            "bn_recalibrated": bn_recal_info,
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
    if ssl_linear_probe is not None:
        manifest["ssl_linear_probe"] = ssl_linear_probe
        manifest["files"]["ssl_linear_probe"] = "reports/ssl_linear_probe.json"

    # After BN recalibration two checkpoint files coexist and the registry's
    # no-manifest glob would resolve the stale one — persist the manifest early
    # so calibration/auto-test resolve files.checkpoint (the recalibrated file).
    # The manifest is re-saved with test/export results at the end as usual.
    if bn_recal_info is not None:
        registry.save_manifest(project_id, model_id, manifest)

    # --- Optional val-based threshold calibration. Fits per-class thresholds on
    # this run's val split and SAVES them into the bundle, so a "deployed" rule
    # below scores the same decision layer production will deploy.
    if cfg.training.calibrate_thresholds and has_val:
        try:
            from sed.train.eval import calibrate_run_thresholds
            calibrate_run_thresholds(cfg, project_id, model_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("threshold calibration failed: %s", exc)

    # --- Optional false-alarm-budget calibration on negative recordings.
    # Both this and the val fit above shape the decision layer, so they belong
    # adjacent AND ahead of the scoring below: a "deployed" rule scored before
    # this ran would describe a decision layer the bundle no longer has.
    # `model_id` is explicit because this run is not the active model yet.
    if cfg.training.calibrate_after and cfg.training.pretraining_objective != "byol":
        try:
            from sed.runtime.calibrate import calibrate

            result = calibrate(
                cfg, project_id,
                target_per_1000=float(cfg.training.calibrate_per_1000),
                negative_source=str(cfg.training.calibrate_negsrc),
                model_id=model_id,
            )
            log.info(
                "false-alarm calibration: %.2f fires per 1000 clips over %d negative clip(s)",
                result.get("system_per_1000") or 0.0, result.get("clips_scored") or 0,
            )
        except Exception as exc:  # noqa: BLE001 — a failed sweep must not lose the run
            log.warning("false-alarm calibration failed: %s", exc)

    # --- Frozen-test scoring, one block per decision rule ---
    # The same call Batch evaluate makes, so results land in
    # manifest["frozen_test"][rule.id] against the CURRENT frozen test rather
    # than in a legacy block the runs table has to reconstruct and flag.
    if do_test:
        from sed.train.eval import (
            _metrics_manifest_block,
            evaluate_run_under_rule,
            parse_rule_id,
        )
        # evaluate_run_under_rule updates the manifest ON DISK, so persist this
        # process's copy first and reload it after, rather than mirroring blocks
        # and report-file pointers by hand.
        registry.save_manifest(project_id, model_id, manifest)
        first_metrics: dict | None = None
        first_rule = None
        for rule_id in test_rules:
            try:
                rule = parse_rule_id(rule_id)
                log.info("auto_test: scoring frozen test under %s", rule.label)
                metrics = evaluate_run_under_rule(cfg, project_id, model_id, rule)
                if first_metrics is None:
                    first_metrics, first_rule = metrics, rule
            except Exception as exc:  # noqa: BLE001
                log.error("auto_test failed for rule %s: %s", rule_id, exc)
        manifest = registry.load_manifest(project_id, model_id)
        manifest.setdefault("files", {})
        # The first rule also fills the legacy flat block that the inference
        # model picker and `sed models compare` still read.
        if first_metrics is not None:
            manifest["test_metrics"] = _metrics_manifest_block(first_metrics)
            manifest["files"]["test_metrics"] = f"reports/{first_rule.slug}_metrics.json"
            manifest["files"]["confusion_matrix"] = f"reports/{first_rule.slug}_confusion_matrix.png"
            manifest["files"]["confusion_matrix_json"] = f"reports/{first_rule.slug}_confusion_matrix.json"
            manifest["files"]["test_predictions"] = f"reports/{first_rule.slug}_predictions.tsv"


    # --- Optional auto-export (ONNX + OpenVINO into the model bundle dir) ---
    if do_export:
        try:
            from sed.export.openvino import export_project
            export_info = export_project(
                cfg, project_id,
                ckpt_path=checkpoint_path,
                out_dir=model_dir,
                dataset_dir=run_dataset,
                targets=set(cfg.training.export_targets or ()) or None,
            )
            manifest["exported"] = True
            # Merge, not replace: the block also carries `precision_decision`
            # from `sed models set-precision`, which a re-export must not wipe.
            manifest.setdefault("export", {}).update({
                "source_checkpoint": str(checkpoint_path.relative_to(model_dir)),
                "active_precision": export_info.get("active_precision"),
                "active_target": export_info["active_target"],
                "int8_calibration_clips": export_info.get("int8_calibration_clips"),
            })
            manifest["files"]["active_openvino"] = export_info["active_openvino"]
            if export_info.get("model_int8"):
                manifest["files"]["openvino_int8"] = export_info["model_int8"]
            manifest["files"]["openvino_fp16"] = export_info["model_fp16"]
            manifest["files"]["onnx"] = export_info["onnx"]

            # Quantization succeeding says nothing about whether it is safe to
            # deploy: INT8 moves class boundaries, and it moves them exactly
            # where a confident model separates a clink from the room. Score
            # both IRs and let the measurement decide.
            if export_info.get("model_int8") and cfg.training.export_verify_int8 and do_test:
                try:
                    from sed.train.eval import decide_ir_precision
                    decision = decide_ir_precision(cfg, project_id, model_id)
                    manifest = registry.load_manifest(project_id, model_id)
                    manifest.setdefault("files", {})
                    manifest.setdefault("export", {}).update({
                        "active_precision": decision["precision"],
                        "active_target": f"model_{decision['precision']}.xml",
                        "precision_decision": f"auto: {decision['reason']}",
                        "precision_scores": {
                            "metric": decision["metric"],
                            "fp16": decision["fp16"],
                            "int8": decision["int8"],
                            "tolerance": decision["tolerance"],
                        },
                    })
                except Exception as exc:  # noqa: BLE001
                    log.error("INT8 verification failed (%s) — keeping the exporter's choice", exc)
        except Exception as exc:  # noqa: BLE001
            log.error("auto_export failed: %s", exc)



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
