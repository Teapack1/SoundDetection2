"""Shared physical/accumulated batch-size semantics for training surfaces."""

from __future__ import annotations

from typing import Any


def training_batch_sizes(training: Any, *, world_size: int = 1) -> dict[str, int]:
    """Return the global optimizer-update batch represented by a config.

    ``batch_size`` is the physical micro-batch handled by each process/device.
    Lightning accumulates that many gradients before one optimizer update, and
    distributed workers contribute one micro-batch each.  Callers that do not
    yet know the Trainer world size use the single-device default.
    """
    physical = max(1, int(getattr(training, "batch_size", 1) or 1))
    accumulation = max(
        1,
        int(getattr(training, "accumulate_grad_batches", 1) or 1),
    )
    workers = max(1, int(world_size or 1))
    return {
        "physical_batch_size": physical,
        "accumulate_grad_batches": accumulation,
        "world_size": workers,
        "effective_batch_size": physical * accumulation * workers,
    }


__all__ = ["training_batch_sizes"]
