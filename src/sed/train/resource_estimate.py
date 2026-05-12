"""Pre-flight training resource estimate for the Train tab.

Builds the exact Lightning module a run would use (same config, same
freeze policy at its steady-state target depth) and measures a few real
optimizer steps on synthetic batches, reporting measured peak GPU memory,
step time and a schedule projection. Run as a subprocess — see
``sed.data.inspect_server`` — so the CUDA context and every allocation are
fully released afterwards and never compete with a real training run:

    python -m sed.train.resource_estimate config.json \
        --train-rows 1200 --val-rows 140

The single stdout line is a JSON payload; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from sed.config import Config


def batch_schedule_counts(train_rows: int, batching: dict[str, int]) -> dict[str, int]:
    physical = max(1, int(batching["physical_batch_size"]))
    accumulation = max(1, int(batching["accumulate_grad_batches"]))
    physical_batches = math.ceil(max(0, int(train_rows)) / physical)
    optimizer_steps = math.ceil(physical_batches / accumulation)
    return {
        "physical_batches_per_epoch": physical_batches,
        "optimizer_steps_per_epoch": optimizer_steps,
    }


def _build_module(cfg: Config, classes: list[str]):
    if cfg.training.pretraining_objective == "byol":
        from sed.train.ssl import ByolLitModule
        return ByolLitModule(cfg=cfg, classes=classes)
    from sed.train.lit_module import SedLitModule
    return SedLitModule(cfg=cfg, classes=classes)


def _optimizer_for(module) -> torch.optim.Optimizer:
    try:
        configured = module.configure_optimizers()
        if isinstance(configured, dict):
            return configured["optimizer"]
        if isinstance(configured, (list, tuple)):
            first = configured[0]
            return first[0] if isinstance(first, (list, tuple)) else first
        return configured
    except Exception:  # noqa: BLE001 — scheduler wiring may want a Trainer
        trainable = [p for p in module.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=1e-4)


def _synthetic_batch(cfg: Config, n_classes: int, batch: int,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    from sed.audio.features import (
        feature_band_count,
        feature_channels,
        feature_frame_count_for_cfg,
    )
    bands = feature_band_count(cfg.features, cfg.model.sample_rate)
    frames = feature_frame_count_for_cfg(cfg)
    channels = feature_channels(cfg.features)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(batch, channels, bands, frames, generator=generator).to(device)
    if cfg.model.task == "frame":
        y = (torch.rand(batch, n_classes, frames, generator=generator) > 0.9).float().to(device)
    else:
        y = torch.zeros(batch, n_classes)
        y[torch.arange(batch), torch.randint(n_classes, (batch,), generator=generator)] = 1.0
        y = y.to(device)
    return x, y


def _autocast_setup(precision: str, device: torch.device):
    """(context factory, GradScaler|None, note|None) matching Lightning."""
    if device.type != "cuda":
        # The estimator deliberately measures CPU fallbacks in FP32.  Returning
        # ``torch.autocast`` itself here used to fail because it still requires
        # a device_type argument, and the canonical Lightning spelling is now
        # ``32-true`` rather than the legacy ``32`` alias.
        note = None if precision == "32-true" else "no GPU — measured in fp32 on CPU"
        return None, None, note
    if precision == "16-mixed":
        return (lambda: torch.autocast("cuda", dtype=torch.float16)),\
            torch.amp.GradScaler("cuda"), None
    if precision == "bf16-mixed":
        # PyTorch can report BF16 as executable through software fallback on
        # old CUDA devices.  Native, useful GPU BF16 starts with Ampere (SM80);
        # an emulated estimate would be both slow and misleading.
        capability = torch.cuda.get_device_capability(device)
        if capability < (8, 0) or not torch.cuda.is_bf16_supported():
            return None, None, "GPU has no native bf16 support — measured in fp32"
        return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)), None, None
    return None, None, None


def _measure_steps(module, optimizer, cfg: Config, n_classes: int, batch: int,
                   device: torch.device, autocast_factory, scaler,
                   n_steps: int = 6, accumulation: int = 1) -> dict:
    """Run real physical batches with configured accumulation and measure them."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    x, y = _synthetic_batch(cfg, n_classes, batch, device)
    step_times: list[float] = []
    accumulation = max(1, int(accumulation))
    optimizer.zero_grad(set_to_none=True)
    for step in range(n_steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        if autocast_factory is not None:
            with autocast_factory():
                loss = module.training_step((x, y), 0) / accumulation
        else:
            loss = module.training_step((x, y), 0) / accumulation
        update = (step + 1) % accumulation == 0 or step + 1 == n_steps
        if scaler is not None:
            scaler.scale(loss).backward()
            if update:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if update:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        # First steps pay cuDNN autotune / lazy-init costs; keep them out of
        # the timing but inside the peak-memory window.
        if step >= 2:
            step_times.append(time.perf_counter() - started)

    forward_s = None
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            if autocast_factory is not None:
                with autocast_factory():
                    module(x)
            else:
                module(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_s = time.perf_counter() - started
    except Exception:  # noqa: BLE001 — BYOL modules have no plain forward
        pass

    out = {
        "step_s": statistics.median(step_times) if step_times else None,
        "forward_s": forward_s,
    }
    if device.type == "cuda":
        out["peak_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    del x, y
    return out


def estimate(cfg: Config, classes: list[str], train_rows: int, val_rows: int) -> dict:
    from sed.train.model_lab import materialize_training_config

    cfg = materialize_training_config(cfg)
    notes: list[str] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    module = _build_module(cfg, classes)
    # Measure at the policy's steady-state target depth: the head-only warm-up
    # phase needs less memory, so the target phase is what has to fit.
    plan = getattr(module, "fine_tune_plan", None)
    if plan is not None and plan.initial_depth != plan.target_depth:
        module.freeze_info, module._frozen_modules = module._apply_train_depth(plan.target_depth)
        notes.append(
            f"measured at target depth '{plan.target_depth}' "
            f"(head warm-up phase needs less)"
        )
    params_total = sum(p.numel() for p in module.parameters())
    params_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

    from sed.audio.features import (
        feature_band_count,
        feature_channels,
        feature_frame_count_for_cfg,
    )
    from sed.train.batching import training_batch_sizes

    batching = training_batch_sizes(cfg.training)
    payload: dict = {
        "ok": True,
        "device": {"kind": device.type},
        "model": {
            "backbone": cfg.model.backbone,
            "task": cfg.model.task,
            "objective": cfg.training.pretraining_objective,
            "classes": len(classes),
            "input_shape": [
                feature_channels(cfg.features),
                feature_band_count(cfg.features, cfg.model.sample_rate),
                feature_frame_count_for_cfg(cfg),
            ],
            "params_total": params_total,
            "params_trainable": params_trainable,
            # fp32 weights + AdamW exp_avg/exp_avg_sq + gradients
            "checkpoint_mib": params_total * 4 / 2**20,
            "optimizer_state_mib": params_trainable * 12 / 2**20,
        },
        "batching": batching,
        "notes": notes,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        payload["device"].update({
            "name": props.name,
            "total_mib": props.total_memory / 2**20,
            "free_mib": torch.cuda.mem_get_info(device)[0] / 2**20,
        })

    module.train().to(device)
    optimizer = _optimizer_for(module)
    autocast_factory, scaler, precision_note = _autocast_setup(cfg.training.precision, device)
    if precision_note:
        notes.append(precision_note)
        if device.type == "cuda":
            autocast_factory, scaler = None, None

    requested = batching["physical_batch_size"]
    batch = requested
    measured: dict | None = None
    while batch >= 1:
        try:
            measured = _measure_steps(
                module, optimizer, cfg, len(classes), batch, device,
                autocast_factory, scaler,
                accumulation=batching["accumulate_grad_batches"],
            )
            break
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            batch //= 2
    if measured is None:
        payload["measure"] = {
            "batch_size": requested,
            "physical_batch_size": requested,
            "accumulate_grad_batches": batching["accumulate_grad_batches"],
            "effective_batch_size": batching["effective_batch_size"],
            "precision": cfg.training.precision,
            "fits": False, "max_fit_batch": 0,
        }
        notes.append("even batch size 1 does not fit on this GPU")
        return payload

    payload["measure"] = {
        "batch_size": requested,
        "physical_batch_size": requested,
        "accumulate_grad_batches": batching["accumulate_grad_batches"],
        "effective_batch_size": batching["effective_batch_size"],
        "precision": cfg.training.precision,
        "fits": batch == requested,
        "measured_batch": batch,
        "max_fit_batch": None if batch == requested else batch,
        **measured,
    }
    if batch != requested:
        notes.append(
            f"batch {requested} ran out of memory — largest fitting batch is ~{batch}"
        )

    step_s = measured.get("step_s")
    if step_s and train_rows > 0:
        # Compute-only projection at the requested batch (scaled from the
        # measured one when they differ): dataloading/augmentation can add
        # overhead on top, and early stopping may finish sooner.
        per_row = step_s / batch
        schedule_counts = batch_schedule_counts(train_rows, batching)
        train_epoch_s = train_rows * per_row
        forward_s = measured.get("forward_s")
        val_epoch_s = val_rows * (forward_s / batch if forward_s else per_row * 0.4)
        epoch_s = train_epoch_s + val_epoch_s
        payload["schedule"] = {
            "train_rows": train_rows,
            "val_rows": val_rows,
            # Keep the historical field as optimizer steps: this is the LR
            # scheduler/global-step count operators care about.  The explicit
            # physical count makes accumulated runs unambiguous.
            "steps_per_epoch": schedule_counts["optimizer_steps_per_epoch"],
            **schedule_counts,
            "epochs": int(cfg.training.epochs),
            "epoch_s": epoch_s,
            "total_s": epoch_s * int(cfg.training.epochs),
        }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_json", type=Path,
                        help="Effective run config as a JSON dump of Config")
    parser.add_argument("--train-rows", type=int, default=0)
    parser.add_argument("--val-rows", type=int, default=0)
    parser.add_argument("--classes", type=int, default=0,
                        help="Class count fallback when the dataset has no classes.json")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    with open(args.config_json, encoding="utf-8") as f:
        cfg = Config.model_validate(json.load(f))
    classes: list[str] | None = None
    if args.dataset_dir is not None:
        try:
            from sed.train.lit_module import load_classes
            classes = load_classes(args.dataset_dir)
        except Exception:  # noqa: BLE001 — dataset may not be composed yet
            classes = None
    if not classes:
        classes = [f"class_{i}" for i in range(max(2, args.classes))]

    try:
        payload = estimate(cfg, classes, args.train_rows, args.val_rows)
    except Exception as exc:  # noqa: BLE001 — report, don't traceback to stdout
        import traceback
        traceback.print_exc()
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
